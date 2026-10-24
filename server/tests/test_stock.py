"""The Stock board and its alerts (HYPERUN-STOCK, HYPERUN-STOCK-WATCH).

WHAT THIS DEFENDS.
  1. The board's summary says what the catalog says: on-demand GPU shapes only, one line
     per (GPU, count) per region, in stock when any instance type of that shape is.
  2. An alert fires once per return to stock, fires again after the stock went away, and a
     DM that did not go is retried rather than recorded as sent.
  3. Only the namespaces the ConfigMap lists may set alerts, and each sees only its own.
  4. Every write to the shared ConfigMap is conditional on the version read, and a write
     that loses the race is retried without losing the other writer's change.

HOW THE CLUSTER IS STOOD IN FOR. The routes and the watcher run on the REAL `k8s.Cluster`,
over a stand-in for the kubernetes CoreV1Api that raises the real `ApiException` with the
real status codes and enforces `metadata.resourceVersion` the way the apiserver does. A
double written from the routes' side would agree with whatever the routes expect; this one
can only agree with the apiserver.
"""
import datetime
import json
import types

import pytest
from fastapi.testclient import TestClient
from kubernetes.client.rest import ApiException

from ddpsrun_server import auth, k8s, main, monitor, stock, stockwatch

NOW = datetime.datetime(2026, 10, 24, 8, 0, tzinfo=datetime.timezone.utc)
SYSTEM = "hyperun-system"


def row(cloud, gpu, gpus, *places, rental="on_demand"):
    """One instance type, available in the (region, available) places given."""
    return {"cloud": cloud, "configuration": {"gpu_type": gpu, "num_gpus": gpus},
            "availability": [{"region": r, "display_name": r.split("-")[0].title(),
                              "available": a, "rental_type": rental} for r, a in places]}


def catalog(*rows, fetched="2026-10-24T07:50:01Z"):
    return {"instance_types": list(rows), "hyperun_fetched_at_utc": fetched}


def h100_nowhere():
    return catalog(row("hyperstack", "H100", 8, ("montreal-canada-2", False)),
                   row("lambdalabs", "H100", 1, ("dulles-usa-1", True)))


def h100_in_montreal():
    return catalog(row("hyperstack", "H100", 8, ("montreal-canada-2", True)),
                   row("lambdalabs", "H100", 1, ("dulles-usa-1", True)))


# ------------------------------------------------------------------ the summary


def test_the_summary_keeps_on_demand_gpu_shapes_and_merges_duplicates():
    doc = catalog(
        row("massedcompute", "L40S", 1, ("desmoines-usa-1", False), ("wichita-usa-1", False)),
        # A second instance type with the same shape in the same region: one line, in stock.
        row("massedcompute", "L40S", 1, ("desmoines-usa-1", True)),
        row("massedcompute", "CPU", 0, ("desmoines-usa-1", True)),
        row("massedcompute", "A100_80G", 8, ("desmoines-usa-1", True), rental="spot"),
    )
    summary = stock.summarize(doc, NOW)
    [vendor] = summary["vendors"]
    regions = {r["region"]: r for r in vendor["regions"]}
    assert regions["desmoines-usa-1"]["offers"] == [{"gpu": "L40S", "gpus": 1, "available": True}]
    assert regions["desmoines-usa-1"]["in_stock"] is True
    # A region that only sells the shape out of stock is still on the board, as "none".
    assert regions["wichita-usa-1"]["in_stock"] is False
    assert summary["stale"] is False and summary["refresh_minutes"] == 10


def test_an_hour_old_catalog_is_stale_and_a_missing_time_is_stale_too():
    assert stock.summarize(h100_nowhere(), NOW + datetime.timedelta(hours=2))["stale"] is True
    assert stock.summarize(catalog(fetched=None), NOW)["stale"] is True


def test_a_document_without_instance_types_is_an_error_not_an_empty_board():
    with pytest.raises(stock.StockError):
        stock.summarize({"rows": []}, NOW)


# ------------------------------------------------------------------ alerts, pure


def test_an_alert_needs_a_gpu_and_sensible_filters():
    alert = stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-alice", "alice", NOW)
    assert alert["vendor"] is None and alert["state"]["in_stock"] is None
    for bad in ({}, {"gpu": ""}, {"gpu": "H100", "gpus": 0}, {"gpu": "H100", "gpus": True},
                {"gpu": "H100", "vendor": "Not A Cloud!"}, {"gpu": "<script>"}):
        with pytest.raises(stock.AlertError):
            stock.new_alert(bad, "lab-alice", "alice", NOW)


def test_hits_respect_the_vendor_region_and_count():
    summary = stock.summarize(h100_in_montreal(), NOW)
    any_h100 = stock.new_alert({"gpu": "H100"}, "lab-alice", "alice", NOW)
    assert [h["region"] for h in stock.hits(any_h100, summary)] == ["montreal-canada-2", "dulles-usa-1"]
    eight = stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-alice", "alice", NOW)
    assert [h["vendor"] for h in stock.hits(eight, summary)] == ["hyperstack"]
    lambda_only = stock.new_alert({"gpu": "H100", "gpus": 8, "vendor": "lambdalabs"},
                                  "lab-alice", "alice", NOW)
    assert stock.hits(lambda_only, summary) == []


def test_an_alert_fires_once_per_return_to_stock():
    alert = stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-alice", "alice", NOW)
    out = stock.summarize(h100_nowhere(), NOW)
    back = stock.summarize(h100_in_montreal(), NOW)

    alerts, fire = stock.step([alert], out, NOW)
    assert fire == [] and alerts[0]["state"]["in_stock"] is False

    alerts, fire = stock.step(alerts, back, NOW)
    assert len(fire) == 1 and alerts[0]["state"]["since"]
    # The caller records the DM; the next pass with the same stock says nothing.
    alerts[0]["state"]["notified_at"] = "2026-10-24T08:02:00Z"
    alerts, fire = stock.step(alerts, back, NOW)
    assert fire == []

    # Gone, then back: a second message.
    alerts, fire = stock.step(alerts, out, NOW)
    assert fire == [] and alerts[0]["state"]["notified_at"] is None
    alerts, fire = stock.step(alerts, back, NOW)
    assert len(fire) == 1


def test_a_dm_that_did_not_go_is_tried_again():
    alert = stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-alice", "alice", NOW)
    back = stock.summarize(h100_in_montreal(), NOW)
    alerts, fire = stock.step([alert], back, NOW)
    assert len(fire) == 1
    # Nobody set notified_at, because the DM failed: the next pass fires again.
    alerts, fire = stock.step(alerts, back, NOW)
    assert len(fire) == 1 and alerts[0]["state"]["in_stock"] is True


def test_the_message_names_the_shape_the_places_and_the_link():
    alert = stock.new_alert({"gpu": "A100_80G", "gpus": 8}, "lab-alice", "alice", NOW)
    found = [{"vendor": f"v{i}", "region": f"r{i}", "name": f"Place {i}"} for i in range(7)]
    text = stock.message_for(alert, found, "2026-10-24T07:50:01Z", "https://ui.example")
    assert text.startswith("[hyperun 재고] A100 80GB ×8 재고가 생겼습니다 (모든 vendor).")
    assert "• v0 · Place 0" in text and "• v5" not in text and "그 밖에 2곳" in text
    assert "07:50 UTC" in text
    assert "<https://ui.example/#/stock/gpu=A100_80G&count=8|재고 보기>" in text


def test_read_alerts_takes_a_list_separated_by_commas_or_lines():
    alerts, allowed = stock.read_alerts({stock.ALLOWED_KEY: "lab-alice,\n lab-bob\n"})
    assert alerts == [] and allowed == {"lab-alice", "lab-bob"}
    with pytest.raises(stock.StockError):
        stock.read_alerts({stock.ALERTS_KEY: "{not json"})


# ------------------------------------------------------------------ the apiserver stand-in


class FakeCore:
    """ConfigMaps by (namespace, name), with the apiserver's 404 and 409."""

    def __init__(self):
        self.maps: dict[tuple[str, str], dict] = {}
        self.reads = 0
        self.races = 0  # how many of the next patches another writer wins first

    def put(self, name, data):
        self.maps[(SYSTEM, name)] = {"data": data, "version": 1}

    def read_namespaced_config_map(self, name, namespace):
        self.reads += 1
        found = self.maps.get((namespace, name))
        if found is None:
            raise ApiException(status=404, reason="Not Found")
        return types.SimpleNamespace(
            data=dict(found["data"]) if found["data"] is not None else None,
            metadata=types.SimpleNamespace(resource_version=str(found["version"])))

    def patch_namespaced_config_map(self, name, namespace, body):
        found = self.maps.get((namespace, name))
        if found is None:
            raise ApiException(status=404, reason="Not Found")
        if self.races:
            self.races -= 1
            found["version"] += 1
            raise ApiException(status=409, reason="Conflict")
        if body["metadata"]["resourceVersion"] != str(found["version"]):
            raise ApiException(status=409, reason="Conflict")
        found["data"] = {**(found["data"] or {}), **body["data"]}
        found["version"] += 1

    def alerts(self):
        return json.loads(self.maps[(SYSTEM, stock.ALERTS_CONFIGMAP)]["data"].get(stock.ALERTS_KEY, "[]"))


@pytest.fixture
def core():
    fake = FakeCore()
    fake.put(stock.CATALOG_CONFIGMAP, {stock.CATALOG_KEY: json.dumps(h100_in_montreal())})
    fake.put(stock.ALERTS_CONFIGMAP, {stock.ALLOWED_KEY: "lab-alice"})
    return fake


def test_the_real_cluster_maps_the_apiserver_answers(core):
    cluster = k8s.Cluster(None, core)
    assert cluster.shadeform_catalog()["hyperun_fetched_at_utc"] == "2026-10-24T07:50:01Z"
    data, version = cluster.stock_alerts()
    with pytest.raises(k8s.Conflict):
        cluster.put_stock_alerts("[]", str(int(version) + 5))
    del core.maps[(SYSTEM, stock.CATALOG_CONFIGMAP)]
    with pytest.raises(k8s.NotFound):
        cluster.shadeform_catalog()
    core.put(stock.CATALOG_CONFIGMAP, {})
    with pytest.raises(k8s.ClusterError):
        cluster.shadeform_catalog()


# ------------------------------------------------------------------ the routes


@pytest.fixture
def client(tmp_path, monkeypatch, core):
    tokens = tmp_path / "tokens.json"
    tokens.write_text(json.dumps({"tokens": [
        {"sha256": auth.hash_token("alice-token"), "user": "alice", "namespace": "lab-alice"},
        {"sha256": auth.hash_token("bob-token"), "user": "bob", "namespace": "lab-bob"},
    ]}))
    monkeypatch.setenv("HYPERUN_RESULT_BUCKET", "<RESULT_BUCKET>")
    monkeypatch.setenv("HYPERUN_TOKENS_PATH", str(tokens))
    monkeypatch.setattr(main.Cluster, "connect", staticmethod(lambda: k8s.Cluster(None, core)))
    with TestClient(main.app) as test_client:
        yield test_client


def as_(who, client, method, path, **kwargs):
    return client.request(method, path, headers={"Authorization": f"Bearer {who}-token"}, **kwargs)


def test_the_board_needs_a_sign_in_and_is_read_once_a_minute(client, core):
    assert client.get("/v1/stock").status_code == 401
    first = as_("alice", client, "GET", "/v1/stock")
    assert first.status_code == 200
    vendors = {v["vendor"]: v for v in first.json()["vendors"]}
    assert vendors["hyperstack"]["regions"][0]["offers"] == [{"gpu": "H100", "gpus": 8, "available": True}]
    reads = core.reads
    assert as_("bob", client, "GET", "/v1/stock").status_code == 200
    assert core.reads == reads, "a second request within the minute read the ConfigMap again"


def test_no_catalog_is_a_503_that_names_the_cronjob(client, core):
    del core.maps[(SYSTEM, stock.CATALOG_CONFIGMAP)]
    answer = as_("alice", client, "GET", "/v1/stock")
    assert answer.status_code == 503 and stock.CATALOG_CONFIGMAP in answer.json()["detail"]


def test_alerts_are_only_for_listed_namespaces(client, core):
    assert as_("bob", client, "GET", "/v1/stock/alerts").status_code == 403
    assert as_("alice", client, "GET", "/v1/stock/alerts").json() == {"alerts": [], "limit": 20}
    del core.maps[(SYSTEM, stock.ALERTS_CONFIGMAP)]
    assert as_("alice", client, "GET", "/v1/stock/alerts").status_code == 403


def test_an_alert_is_stored_once_and_seen_only_by_its_owner(client, core):
    made = as_("alice", client, "POST", "/v1/stock/alerts", json={"gpu": "H100", "gpus": 8})
    assert made.status_code == 201
    [stored] = core.alerts()
    assert stored["namespace"] == "lab-alice" and stored["owner"] == "alice"
    again = as_("alice", client, "POST", "/v1/stock/alerts", json={"gpu": "H100", "gpus": 8})
    assert again.status_code == 409 and "already set" in again.json()["detail"]
    assert as_("alice", client, "POST", "/v1/stock/alerts", json={"gpu": "<b>"}).status_code == 422
    # Bob, once enabled, sees none of Alice's and cannot remove hers.
    core.maps[(SYSTEM, stock.ALERTS_CONFIGMAP)]["data"][stock.ALLOWED_KEY] = "lab-alice,lab-bob"
    assert as_("bob", client, "GET", "/v1/stock/alerts").json()["alerts"] == []
    assert as_("bob", client, "DELETE", f"/v1/stock/alerts/{stored['id']}").status_code == 404
    assert as_("alice", client, "DELETE", f"/v1/stock/alerts/{stored['id']}").status_code == 204
    assert core.alerts() == []


def test_a_write_that_loses_the_race_is_retried_on_a_fresh_read(client, core):
    core.races = 1
    made = as_("alice", client, "POST", "/v1/stock/alerts", json={"gpu": "L40S"})
    assert made.status_code == 201 and len(core.alerts()) == 1


def test_one_account_keeps_at_most_the_limit(client, core):
    for n in range(stock.MAX_ALERTS_PER_NAMESPACE):
        assert as_("alice", client, "POST", "/v1/stock/alerts",
                   json={"gpu": "H100", "gpus": (n % 16) + 1,
                         "vendor": f"vendor-{n}"}).status_code == 201
    over = as_("alice", client, "POST", "/v1/stock/alerts", json={"gpu": "B200"})
    assert over.status_code == 409 and "the most one account may keep" in over.json()["detail"]


# ------------------------------------------------------------------ the watcher


@pytest.fixture
def watcher(core, monkeypatch):
    monkeypatch.setenv("HYPERUN_SLACK_USER_MAP", json.dumps({"alice": "U0ALICE"}))
    sent = []

    def send(slack_id, text, token):
        sent.append((slack_id, text, token))
        return send.ok
    send.ok = True
    cluster = k8s.Cluster(None, core)

    def run(document=None, when=NOW):
        if document is not None:
            core.put(stock.CATALOG_CONFIGMAP, {stock.CATALOG_KEY: json.dumps(document)})
        return stockwatch.run_once(cluster, "xoxb-test", "https://ui.example", send=send, now=when)
    run.sent, run.send = sent, send
    return run


def store(core, *alerts):
    core.maps[(SYSTEM, stock.ALERTS_CONFIGMAP)]["data"][stock.ALERTS_KEY] = stock.write_alerts(list(alerts))


def test_the_watcher_messages_once_and_records_it(core, watcher):
    store(core, stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-alice", "alice", NOW))
    assert watcher(h100_nowhere()) == 0
    assert watcher(h100_in_montreal()) == 1
    slack_id, text, token = watcher.sent[0]
    assert slack_id == "U0ALICE" and token == "xoxb-test" and "hyperstack" in text
    assert core.alerts()[0]["state"]["notified_at"] == "2026-10-24T08:00:00Z"
    assert watcher() == 0 and len(watcher.sent) == 1


def test_a_failed_dm_is_not_recorded_and_is_sent_next_pass(core, watcher):
    store(core, stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-alice", "alice", NOW))
    watcher.send.ok = False
    assert watcher(h100_in_montreal()) == 0
    state = core.alerts()[0]["state"]
    assert state["in_stock"] is True and state["notified_at"] is None
    watcher.send.ok = True
    assert watcher() == 1


def test_an_owner_without_a_slack_id_is_never_messaged(core, watcher):
    store(core, stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-alice", "carol", NOW))
    assert watcher(h100_in_montreal()) == 0 and watcher.sent == []
    assert core.alerts()[0]["state"]["in_stock"] is True


def test_a_stale_catalog_is_not_judged(core, watcher):
    store(core, stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-alice", "alice", NOW))
    assert watcher(h100_in_montreal(), when=NOW + datetime.timedelta(hours=3)) == 0
    assert watcher.sent == [] and core.alerts()[0]["state"]["checked_at"] is None


def test_a_racing_write_keeps_the_other_writers_alert(core, watcher, monkeypatch):
    mine = stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-alice", "alice", NOW)
    store(core, mine)
    real_put = k8s.Cluster.put_stock_alerts
    added = stock.new_alert({"gpu": "L4"}, "lab-alice", "alice", NOW)

    def put_after_somebody_added_one(self, text, version):
        # The first write finds that the screen stored an alert in between.
        monkeypatch.setattr(k8s.Cluster, "put_stock_alerts", real_put)
        cm = core.maps[(SYSTEM, stock.ALERTS_CONFIGMAP)]
        cm["data"][stock.ALERTS_KEY] = stock.write_alerts([mine, added])
        cm["version"] += 1
        return real_put(self, text, version)
    monkeypatch.setattr(k8s.Cluster, "put_stock_alerts", put_after_somebody_added_one)

    assert watcher(h100_in_montreal()) == 1
    by_id = {a["id"]: a for a in core.alerts()}
    assert set(by_id) == {mine["id"], added["id"]}
    assert by_id[mine["id"]]["state"]["notified_at"] and by_id[added["id"]]["state"]["in_stock"] is None


def test_alerts_of_an_account_taken_off_the_list_are_kept_but_not_watched(core, watcher):
    store(core, stock.new_alert({"gpu": "H100", "gpus": 8}, "lab-bob", "alice", NOW))
    assert watcher(h100_in_montreal()) == 0
    [kept] = core.alerts()
    assert kept["namespace"] == "lab-bob" and kept["state"]["checked_at"] is None


def test_the_watcher_uses_the_monitors_slack_call():
    # One Slack client in this image, not two that drift apart.
    assert stockwatch.run_once.__defaults__[0] is monitor.notify
