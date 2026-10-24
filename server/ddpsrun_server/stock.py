"""Shadeform stock: which vendor and region has a GPU free right now. HYPERUN-STOCK.

WHAT IT ANSWERS. The Stock screen shows, for every Shadeform vendor and every region it
sells in, whether a GPU machine is in stock, and lets an enabled account ask for a Slack
DM when a GPU it names comes back into stock. Nothing here rents, reserves or prices
anything: the person reading it rents on Shadeform's own site, with an account this
service does not hold.

WHERE THE STOCK COMES FROM. The operator already keeps it. Its CronJob
hyperun-shadeform-catalog reads `GET /v1/instances/types` every 10 minutes and writes the
answer, with the time it read it, into the ConfigMap of the same name in the system
namespace (PACSrun driver/shadeform/catalog_refresh.py). This module reads that ConfigMap,
so the board costs Shadeform no extra call and shows exactly the stock the operator places
jobs on. The price of that choice is the latency: a region that comes into stock shows up
within one refresh, ten minutes, plus the watcher's own schedule.

END-TO-END FLOW.
  1. `k8s.Cluster.shadeform_catalog()` reads the document out of the ConfigMap.
  2. `summarize()` folds it into vendor -> region -> the GPU shapes offered there, each with
     its availability. The board needs nothing else.
  3. The `/v1/stock` route serves that summary, cached for `CACHE_SECONDS`.
  4. `stockwatch` (a CronJob) runs `step()` over the stored alerts and the same summary,
     and DMs the owner of every alert whose GPU has just come into stock.

WHAT "IN STOCK" MEANS. An on-demand entry whose `available` is true at the moment the
catalog was read. Spot entries are left out: the operator never places a job on one, and a
board that counted them would promise a machine the platform does not use. CPU-only
instance types are left out too, because the board is about GPUs.

Grep anchor: HYPERUN-STOCK
"""

from __future__ import annotations

import datetime
import json
import re
import secrets
from typing import Any

# The ConfigMap the operator's catalog CronJob writes, and its one data key. Spelled here
# exactly as PACSrun's config/deploy/shadeform-catalog.yaml names them.
CATALOG_CONFIGMAP = "hyperun-shadeform-catalog"
CATALOG_KEY = "instance_types.json"

# The ConfigMap that holds the alerts and the namespaces allowed to set them. It is created
# once by an operator (config/deploy/hyperun-stock.yaml says how), so the gateway needs only
# `get` and `update` on this one name, never `create` on the namespace.
ALERTS_CONFIGMAP = "hyperun-stock-alerts"
ALERTS_KEY = "alerts.json"
ALLOWED_KEY = "allowed-namespaces"

# How long one read of the catalog is served before the ConfigMap is read again. The
# catalog changes every 600 s, so a minute keeps the board at most a minute behind it while
# a page full of people reloading reads the apiserver about once a minute.
CACHE_SECONDS = 60

# How old the catalog may be before the board says so. The operator's decider uses the same
# hour (LiveCatalogStaleAfter), so the board and the placement go stale together.
STALE_AFTER_SECONDS = 3600

# Per account. An alert is a standing question, and a person asks a handful of them; the
# cap keeps a script that loops on POST from filling the one ConfigMap everybody shares.
MAX_ALERTS_PER_NAMESPACE = 20

REFRESH_MINUTES = 10


class StockError(Exception):
    """The catalog document is not one this module can read."""


class AlertError(ValueError):
    """An alert request that cannot be stored, with the reason a person can act on."""


def _utc(text: str | None) -> datetime.datetime | None:
    """Parse the catalog's `2026-10-24T07:50:01Z`, or None for anything else."""
    if not text:
        return None
    try:
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def summarize(document: dict[str, Any], now: datetime.datetime | None = None) -> dict[str, Any]:
    """Fold the Shadeform catalog into the shape the board draws.

    Args:
        document: what the operator's CronJob wrote, `{"instance_types": [...],
            "hyperun_fetched_at_utc": "..."}`. Every row carries `cloud` (the vendor),
            `configuration.gpu_type` and `configuration.num_gpus`, and `availability`, a list
            of `{region, display_name, available, rental_type}`.
        now: the time to judge staleness against; defaults to the clock.

    Returns:
        `{source, fetched_at, refresh_minutes, stale, vendors: [{vendor, regions: [{region,
        name, in_stock, offers: [{gpu, gpus, available}]}]}]}`. Vendors and regions are
        sorted by name, offers by GPU and count. One offer per (GPU, count) per region: two
        instance types with the same shape in one region are one line on the board, in stock
        when either is.

    Raises:
        StockError: the document has no `instance_types` list. A malformed catalog is the
            operator's to fix, and an empty board would read as "nothing is in stock".
    """
    rows = document.get("instance_types") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise StockError("the Shadeform catalog has no instance_types list")

    # vendor -> region -> {"name": display name, "offers": {(gpu, gpus): available}}
    vendors: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        conf = row.get("configuration") or {}
        gpu = conf.get("gpu_type") or row.get("gpu_type")
        gpus = conf.get("num_gpus") or row.get("num_gpus")
        vendor = row.get("cloud")
        if not gpu or gpu == "CPU" or not vendor or not isinstance(gpus, int) or gpus < 1:
            continue
        for entry in row.get("availability") or []:
            if not isinstance(entry, dict) or not entry.get("region"):
                continue
            if entry.get("rental_type") not in (None, "on_demand"):
                continue
            region = vendors.setdefault(vendor, {}).setdefault(
                entry["region"], {"name": entry.get("display_name") or entry["region"], "offers": {}})
            key = (gpu, gpus)
            region["offers"][key] = region["offers"].get(key, False) or bool(entry.get("available"))

    fetched_at = document.get("hyperun_fetched_at_utc")
    read = _utc(fetched_at)
    clock = now or datetime.datetime.now(datetime.timezone.utc)
    stale = read is None or (clock - read).total_seconds() > STALE_AFTER_SECONDS
    return {
        "source": "shadeform",
        "fetched_at": fetched_at,
        "refresh_minutes": REFRESH_MINUTES,
        "stale": stale,
        "vendors": [
            {
                "vendor": vendor,
                "regions": [
                    {
                        "region": region,
                        "name": info["name"],
                        "in_stock": any(info["offers"].values()),
                        "offers": [{"gpu": g, "gpus": n, "available": a}
                                   for (g, n), a in sorted(info["offers"].items())],
                    }
                    for region, info in sorted(regions.items(), key=lambda kv: (kv[1]["name"], kv[0]))
                ],
            }
            for vendor, regions in sorted(vendors.items())
        ],
    }


# ------------------------------------------------------------------ alerts
#
# An alert is a standing question -- "tell me when an H100 x8 is in stock, anywhere or at
# this vendor" -- that its owner set on the Stock screen. It lives in the ConfigMap above as
# one JSON list, shared by every enabled namespace and read back filtered to the caller's.

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+-]{0,39}$")
_REGION = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


def new_alert(body: dict[str, Any], namespace: str, owner: str,
              now: datetime.datetime | None = None) -> dict[str, Any]:
    """Validate one alert request and give it an id.

    Args:
        body: `{gpu, gpus, vendor, region}` from the screen. `gpu` is required and spelled
            as the catalog spells it (`H100`, `A100_80G`); the other three may be null,
            which means "any".
        namespace: the caller's, from their credential. The security boundary.
        owner: the caller's owner label, the key `HYPERUN_SLACK_USER_MAP` is read with.

    Returns:
        The alert as it is stored, with `state` saying it has not been checked yet.

    Raises:
        AlertError: a field is missing or not something the catalog could ever match.
    """
    gpu = body.get("gpu")
    if not isinstance(gpu, str) or not _NAME.match(gpu):
        raise AlertError("gpu is required, spelled as the catalog spells it (for example H100)")
    gpus = body.get("gpus")
    if gpus is not None and (not isinstance(gpus, int) or isinstance(gpus, bool) or not 1 <= gpus <= 16):
        raise AlertError("gpus is the number of GPUs in one machine, 1 to 16, or null for any")
    vendor = body.get("vendor")
    if vendor is not None and (not isinstance(vendor, str) or not _REGION.match(vendor)):
        raise AlertError("vendor is a Shadeform cloud name such as hyperstack, or null for any")
    region = body.get("region")
    if region is not None and (not isinstance(region, str) or not _REGION.match(region)):
        raise AlertError("region is a Shadeform region such as montreal-canada-2, or null for any")
    clock = now or datetime.datetime.now(datetime.timezone.utc)
    return {
        "id": secrets.token_hex(6),
        "namespace": namespace,
        "owner": owner,
        "gpu": gpu,
        "gpus": gpus,
        "vendor": vendor,
        "region": region,
        "created_at": clock.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "state": {"in_stock": None, "since": None, "places": [], "checked_at": None,
                  "notified_at": None},
    }


def hits(alert: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, str]]:
    """Every (vendor, region) where this alert's GPU is in stock now.

    Returns:
        `[{vendor, region, name, counts}]`, sorted, one entry per region however many
        machine shapes match there; `counts` is the GPUs-per-machine in stock, ascending.
    """
    found = []
    for vendor in summary.get("vendors", []):
        if alert.get("vendor") and vendor["vendor"] != alert["vendor"]:
            continue
        for region in vendor["regions"]:
            if alert.get("region") and region["region"] != alert["region"]:
                continue
            counts = sorted({o["gpus"] for o in region["offers"]
                             if o["available"] and o["gpu"] == alert["gpu"]
                             and (alert.get("gpus") is None or o["gpus"] == alert["gpus"])})
            if counts:
                found.append({"vendor": vendor["vendor"], "region": region["region"],
                              "name": region["name"], "counts": counts})
    return sorted(found, key=lambda h: (h["vendor"], h["name"], h["region"]))


def step(alerts: list[dict[str, Any]], summary: dict[str, Any],
         now: datetime.datetime | None = None) -> tuple[list[dict[str, Any]], list[tuple[dict, list]]]:
    """One watcher pass over every alert: its new state, and which ones to message.

    THE STATE IS TWO FACTS, KEPT APART. `in_stock` is what the catalog says now, and `since`
    is when the current run of stock began. `notified_at` is when the owner was told about
    THAT run. Keeping them apart is what lets a failed DM be retried: the stock is recorded
    truthfully either way, and an alert in stock with no `notified_at` is one the next pass
    messages again. The screen reads the same two facts, so it never says "waiting" about a
    GPU that is in stock merely because Slack was unreachable.

    ONE MESSAGE PER RETURN TO STOCK, NOT ONE PER CHECK. While the stock lasts, the alert
    stays quiet; when it is gone, `since` and `notified_at` are cleared, and the next return
    fires again. A watcher that messaged on every pass would send six DMs an hour about the
    same machine, and the second one teaches a person to ignore the first.

    A NEW ALERT WHOSE GPU IS ALREADY IN STOCK FIRES ON ITS FIRST CHECK. The person asked to
    know; that it was true a minute before they asked does not make it less worth saying.

    Args:
        alerts: as stored.
        summary: `summarize()` of the catalog this pass read. A stale summary must not be
            passed in at all (see stockwatch): a transition judged on hour-old stock is a DM
            about a machine that may be long gone.

    Returns:
        (the alerts with their state updated, [(alert, hits)] for every alert to message).
        The caller sets `notified_at` on the ones whose message went.
    """
    clock = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated, fire = [], []
    for alert in alerts:
        found = hits(alert, summary)
        state = dict(alert.get("state") or {})
        if found:
            if not state.get("in_stock"):
                state["since"] = clock
                state["notified_at"] = None
            state["in_stock"] = True
        else:
            state.update(in_stock=False, since=None, notified_at=None)
        state["places"] = [f"{h['vendor']}/{h['region']}" for h in found]
        state["checked_at"] = clock
        current = {**alert, "state": state}
        if found and not state.get("notified_at"):
            fire.append((current, found))
        updated.append(current)
    return updated, fire


GPU_NAMES = {"A100_80G": "A100 80GB", "V100_32G": "V100 32GB", "H100_nvl": "H100 NVL",
             "RTXPro6000": "RTX PRO 6000", "RTX6000Ada": "RTX 6000 Ada", "RTX4090": "RTX 4090",
             "RTX5090": "RTX 5090", "RTX6000": "RTX 6000", "RTX4000Ada": "RTX 4000 Ada"}


def shape_word(alert: dict[str, Any]) -> str:
    """`H100 ×8`, `A100 80GB` -- the alert's GPU as the screen writes it."""
    name = GPU_NAMES.get(alert["gpu"], alert["gpu"])
    return f"{name} ×{alert['gpus']}" if alert.get("gpus") else name


VENDOR_NAMES = {"massedcompute": "Massed Compute", "lambdalabs": "Lambda",
                "hyperstack": "Hyperstack", "crusoe": "Crusoe", "paperspace": "Paperspace",
                "scaleway": "Scaleway", "vultr": "Vultr", "digitalocean": "DigitalOcean",
                "nebius": "Nebius", "voltagepark": "Voltage Park", "denvr": "Denvr",
                "latitude": "Latitude.sh", "excesssupply": "Excess Supply", "imwt": "IMWT",
                "verda": "Verda", "boostrun": "Boostrun", "horizon": "Horizon",
                "phyntec": "Phyntec"}


def vendor_word(vendor: str) -> str:
    """`Crusoe` for `crusoe` -- the vendor as the Stock screen writes it."""
    return VENDOR_NAMES.get(vendor, vendor)


def board_link(ui_url: str, alert: dict[str, Any]) -> str:
    """The Stock screen, filtered to this alert's GPU and count."""
    query = f"gpu={alert['gpu']}" + (f"&count={alert['gpus']}" if alert.get("gpus") else "")
    return f"{ui_url.rstrip('/')}/#/stock/{query}" if ui_url else ""


def headline(alert: dict[str, Any], found: list[dict[str, Any]]) -> str:
    """The DM's title: a noun phrase and its count, never a sentence.

    `A100 80GB ×4 재고 발생  |  region 1곳`. The reader's first question is "what, and how
    much of it", and both fit in the title; "재고가 생겼습니다" answered neither.
    """
    return f"{shape_word(alert)} 재고 발생  |  region {len(found)}곳"


def _when(fetched_at: str | None) -> str:
    """`10:50 UTC`, the minute the catalog was read, or `-` when nobody said."""
    stamp = (fetched_at or "")[11:16]
    return f"{stamp} UTC" if stamp else "-"


def blocks_for(alert: dict[str, Any], found: list[dict[str, Any]], fetched_at: str | None,
               ui_url: str, limit: int = 20) -> list[dict[str, Any]]:
    """The DM as Slack Block Kit, in the shape of cloud-usage's Main 4 and Main 5.

    ★ THE SAME SHAPE AS THE OTHER REPORTS THIS PERSON READS IN THE SAME SLACK: a header,
    sections under `[ ... ]` labels, a divider between them, and a context line at the
    bottom (monitor_v2/hyperun/report.py and vendor/report.py). Main 5 puts its table in the
    body because its body is almost all numbers whose columns must line up, and this one is
    the same: vendor, place, region and count are what a person scans to pick one.

    ★★ NOUNS AND COUNTS, NOT SENTENCES. Every label and value is a noun phrase with its
    quantity -- `region 1곳`, `구매 없음` -- so the eye lands on the fact.

    THE TABLE IS A `markdown` BLOCK, which Slack draws as a real table; Main 5 sends the
    same block. It stops at `limit` rows and the context line counts the rest.
    """
    vendors = sorted({h["vendor"] for h in found})
    rows = found[:limit]
    table = ("| vendor | 위치 | region | 장비당 GPU |\n| --- | --- | --- | --- |\n" +
             "\n".join(f"| {vendor_word(h['vendor'])} | {h['name']} | {h['region']} | "
                       + " ".join(f"×{n}" for n in h["counts"]) + " |" for h in rows))
    link = board_link(ui_url, alert)
    context = [f"Shadeform 기준 {_when(fetched_at)}", "10분 간격 확인",
               "구매 없음 · 대여 경로 shadeform.com"]
    if len(found) > limit:
        context.insert(0, f"표 밖 region {len(found) - limit}곳")
    if link:
        context.append(f"<{link}|재고 화면>")
    return [
        {"type": "header", "text": {"type": "plain_text", "text": headline(alert, found)[:150],
                                    "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "*[ 재고 발생 ]*"}},
        {"type": "section", "fields": [
            {"type": "mrkdwn", "text": f"*GPU*\n`{shape_word(alert)}`"},
            {"type": "mrkdwn", "text": "*조건*\n`" + (vendor_word(alert["vendor"])
                                                      if alert.get("vendor") else "모든 vendor") + "`"},
            {"type": "mrkdwn", "text": f"*vendor*\n`{len(vendors)}곳`"},
            {"type": "mrkdwn", "text": f"*region*\n`{len(found)}곳`"},
        ]},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn",
                                     "text": f"*[ region 별 재고  {_when(fetched_at)} ]*"}},
        {"type": "markdown", "text": table},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": "  ·  ".join(context)}]},
    ]


def message_for(alert: dict[str, Any], found: list[dict[str, Any]], fetched_at: str | None,
                ui_url: str, limit: int = 5) -> str:
    """The same DM as plain text: Slack's notification preview, and any client without blocks.

    The blocks are the message and this is the stand-in, but it carries the same facts in
    the same order, so a person who reads only the preview reads the same story.
    """
    lines = [f"[hyperun 재고] {headline(alert, found)}"]
    for h in found[:limit]:
        lines.append(f"• {vendor_word(h['vendor'])} · {h['name']} · "
                     + " ".join(f"×{n}" for n in h["counts"]))
    if len(found) > limit:
        lines.append(f"• 그 밖에 region {len(found) - limit}곳")
    link = board_link(ui_url, alert)
    lines.append(f"Shadeform 기준 {_when(fetched_at)} · 구매 없음" + (f" · <{link}|재고 화면>" if link else ""))
    return "\n".join(lines)


def read_alerts(data: dict[str, str] | None) -> tuple[list[dict[str, Any]], set[str]]:
    """The alerts and the allowed namespaces out of the ConfigMap's data.

    A missing key is an empty list, not an error: the ConfigMap is created with only the
    allowed namespaces in it, and the first alert is what writes `alerts.json`.
    """
    data = data or {}
    try:
        alerts = json.loads(data.get(ALERTS_KEY) or "[]")
    except ValueError as exc:
        raise StockError(f"{ALERTS_CONFIGMAP} holds {ALERTS_KEY} that is not JSON") from exc
    if not isinstance(alerts, list):
        raise StockError(f"{ALERTS_CONFIGMAP} holds {ALERTS_KEY} that is not a list")
    allowed = {ns.strip() for ns in (data.get(ALLOWED_KEY) or "").replace("\n", ",").split(",")
               if ns.strip()}
    return [a for a in alerts if isinstance(a, dict)], allowed


def write_alerts(alerts: list[dict[str, Any]]) -> str:
    """The `alerts.json` value, compact and in a stable order so a diff of it reads."""
    return json.dumps(sorted(alerts, key=lambda a: (a.get("namespace", ""), a.get("created_at", ""),
                                                    a.get("id", ""))),
                      separators=(",", ":"), ensure_ascii=False)
