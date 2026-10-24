"""Tell an alert's owner, by Slack DM, when its GPU comes back into stock. HYPERUN-STOCK-WATCH.

END-TO-END FLOW of one pass (CronJob hyperun-stock-watch, config/deploy/hyperun-stock.yaml,
every 10 minutes and two minutes after the catalog CronJob, so each pass reads a fresh
catalog rather than one that is about to be replaced):

  1. read the operator's Shadeform catalog and summarize it (stock.summarize). A catalog over
     an hour old ends the pass: a DM judged on stale stock names a machine that may be gone.
  2. read the alerts ConfigMap, and keep only the alerts of namespaces still listed in it.
  3. stock.step: the new state of every alert, and which ones have just come into stock.
  4. DM the owner of each of those, with the same Slack call and the same owner-to-Slack map
     the job monitor uses (monitor.notify, monitor.slack_id_for).
  5. write the state back, conditional on the version read in 2. On a conflict -- somebody
     added or removed an alert in the same second -- read again and lay the new state over
     the alerts that are there now, by id. Nothing is sent twice: the messages went in 4, and
     only their record is being retried.

WHAT A FAILED DM DOES. Nothing is recorded as told, so the next pass, ten minutes later,
tries again while the stock lasts. A person who is not in HYPERUN_SLACK_USER_MAP is never
messaged; each pass logs that, and the screen still shows the stock.

WHAT IT DOES NOT DO. It rents nothing and reserves nothing. Every alert is a question about
stock, answered by a message.

Grep anchor: HYPERUN-STOCK-WATCH
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Callable

from . import k8s, monitor, stock

logger = logging.getLogger(__name__)


def ui_url() -> str:
    """Where the Stock screen is: the first of the origins the gateway allows."""
    origins = os.environ.get("HYPERUN_UI_ORIGINS", "")
    return next((o.strip() for o in origins.split(",") if o.strip()), "")


def run_once(cluster: Any, slack_token: str, link_base: str,
             send: Callable[..., bool] = monitor.notify,
             now: datetime.datetime | None = None) -> int:
    """One pass. Returns how many DMs went.

    Args:
        cluster: a `k8s.Cluster`, or a test double with the same three stock methods.
        slack_token: the bot token; empty means nothing can be sent, which is logged.
        link_base: the screen's origin, for the link in the DM; empty leaves the link out.
        send: `monitor.notify`, replaced in tests.
    """
    summary = stock.summarize(cluster.shadeform_catalog(), now)
    if summary["stale"]:
        logger.warning("the Shadeform catalog was read at %s, over an hour ago; no alert is "
                       "judged on it", summary["fetched_at"])
        return 0
    try:
        data, version = cluster.stock_alerts()
    except k8s.NotFound:
        logger.info("%s does not exist, so no account has alerts", stock.ALERTS_CONFIGMAP)
        return 0
    alerts, allowed = stock.read_alerts(data)
    watched = [a for a in alerts if a.get("namespace") in allowed]
    updated, fire = stock.step(watched, summary, now)
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")

    told = set()
    for alert, found in fire:
        slack_id = monitor.slack_id_for(alert.get("owner", ""))
        if not slack_id:
            logger.warning("alert %s (%s) is in stock, but %s has no Slack id in "
                           "HYPERUN_SLACK_USER_MAP", alert["id"], stock.shape_word(alert),
                           alert.get("owner", ""))
            continue
        # Block Kit is the message and the text is its stand-in for the notification
        # preview, the same pair the monitor sends (monitor.notify sends both).
        if send(slack_id, stock.message_for(alert, found, summary["fetched_at"], link_base),
                slack_token,
                blocks=stock.blocks_for(alert, found, summary["fetched_at"], link_base)):
            told.add(alert["id"])
            logger.info("alert %s: %s is in stock in %d place(s); DM sent", alert["id"],
                        stock.shape_word(alert), len(found))
        else:
            logger.warning("alert %s: the DM did not go; the next pass tries again", alert["id"])

    states = {}
    for alert in updated:
        state = dict(alert["state"])
        if alert["id"] in told:
            state["notified_at"] = stamp
        states[alert["id"]] = state

    for _ in range(3):
        current, _allowed = stock.read_alerts(data)
        merged = [{**a, "state": states.get(a.get("id"), a.get("state"))} for a in current]
        try:
            cluster.put_stock_alerts(stock.write_alerts(merged), version)
            break
        except k8s.Conflict:
            data, version = cluster.stock_alerts()
    else:
        logger.warning("the alert state could not be written after three tries; the next "
                       "pass rechecks from what is stored")
    return len(told)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    token = os.environ.get("HYPERUN_SLACK_BOT_TOKEN", "").strip()
    if not token:
        logger.warning("HYPERUN_SLACK_BOT_TOKEN is not set: stock is checked, nobody is told")
    try:
        sent = run_once(k8s.Cluster.connect(), token, ui_url())
    except Exception:  # noqa: BLE001 - one log line with the reason is the whole report
        logger.exception("the stock watch pass failed")
        return 1
    logger.info("pass complete, %d DM(s) sent", sent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
