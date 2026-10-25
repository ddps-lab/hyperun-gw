"""Creating, listing, reading and deleting a HyperServing through the gateway, and its API key.

HYPERUN-SERVING. WHAT IT PINS, against test_routes.FakeCluster (whose serving methods mirror
k8s.Cluster's branches; tests/test_k8s.py drives the real ones):
  1. POST /v1/servings writes the contract's object: owner label, spec.model with the preset's
     numbers or the fetched config, the two token defaults, the budget as a decimal string, the
     placement filled in, the job's ServiceAccount (HYPERUN-SERVING-SA), and hfTokenSecret as a
     secretKeyRef into hyperun-user-secrets.
  2. Every problem with a request is named at once and nothing is created; a name clash is 409.
  3. The view carries the ranked plans with score, the chosen plan with its stages, the catalog,
     the endpoint PATH (never the upstream address or the Secret's name) and a billing sentence.
  4. The list leaves the plans out and hides a namespace-mate's serving; get and delete read it
     as 404 (HYPERUN-OWNER-GATE).
  5. The API key comes back to the owner only, with Cache-Control: no-store, and a serving with
     no key yet is a 409 that says why.
"""

import pytest

from test_routes import FakeCluster, as_alice, as_root, client, cluster, main  # noqa: F401

servings = main.servings
modelconfig = main.modelconfig

PRESET_BODY = {"name": "chat-8b", "model": {"preset": "llama-3.1-8b"}, "hourly_budget": 3.0}

# A serving as the operator would write its status, built on the 2026-10-19 snapshot numbers the
# job tests use: rank 1 one A100 80 GB at $1.35/h scoring 12.558, a two-machine rank 2.
STATUS = {
    "phase": "Serving",
    "message": "",
    "servingCatalog": "Shadeform catalog fetched 2026-10-20T02:39:32Z (live ConfigMap "
                      "hyperun-shadeform-catalog)",
    "servingPlans": [
        {"rank": 1, "score": "12.558", "priceBasis": "on_demand", "costPerHour": "1.3500",
         "throughputRPS": "16.953", "latencySeconds": "1.760", "nodes": 1, "executable": True,
         "pipelines": [{"domain": "hyperstack/montreal-canada-2", "throughputRPS": "16.953",
                        "costPerHour": "1.3500", "stages": [
                            {"instanceType": "hyperstack_A100_80G", "zone": "montreal-canada-2",
                             "layers": 32, "tp": 1, "pricePerHour": "1.3500"}]}]},
        {"rank": 2, "score": "9.100", "priceBasis": "on_demand", "costPerHour": "2.2000",
         "throughputRPS": "20.020", "latencySeconds": "2.100", "nodes": 2, "executable": False,
         "pipelines": []},
    ],
    "servingPlanChosen": {"rank": 1, "score": "12.558", "costPerHour": "1.3500",
                          "throughputRPS": "16.953", "latencySeconds": "1.760", "attempt": 1,
                          "stages": [{"cloud": "hyperstack", "region": "montreal-canada-2",
                                      "instanceType": "A100_80G", "layers": 32, "tp": 1}]},
    "endpoint": {"path": "/v1/servings/lab-alice/chat-8b/v1", "apiKeySecret": "chat-8b-api-key",
                 "ready": True, "upstream": "http://10.0.42.7:8000"},
    "machines": [{"name": "chat-8b-0", "instanceType": "A100_80G"}],
}


def seed(cluster, name="chat-8b", owner="alice", namespace="lab-alice", status=None):
    """Put a serving on the fake cluster as the operator would have left it."""
    cluster.servings[(namespace, name)] = {
        "apiVersion": "hyperun.io/v1alpha1", "kind": "HyperServing",
        "metadata": {"name": name, "namespace": namespace,
                     "labels": {main.naming.OWNER_LABEL: owner},
                     "creationTimestamp": "2026-10-20T01:00:00Z"},
        "spec": {"model": {"name": "meta-llama/Llama-3.1-8B-Instruct", "preset": "llama-3.1-8b",
                           "config": dict(modelconfig.PRESETS["llama-3.1-8b"]["config"]),
                           "configSource": "preset:llama-3.1-8b"},
                 "workload": {"avgInputTokens": 763, "avgOutputTokens": 232},
                 "hourlyBudget": "3.0",
                 "placement": {"regions": ["shadeform"], "capacityType": "on-demand"}},
        "status": STATUS if status is None else status,
    }


# ------------------------------------------------------------------ 1. create


def test_a_preset_serving_becomes_the_contracts_object(client, cluster):
    response = as_alice(client, "POST", "/v1/servings", json=PRESET_BODY)
    assert response.status_code == 201, response.text
    obj = cluster.servings[("lab-alice", "chat-8b")]
    assert (obj["apiVersion"], obj["kind"]) == ("hyperun.io/v1alpha1", "HyperServing")
    assert obj["metadata"]["labels"] == {main.naming.OWNER_LABEL: "alice"}
    spec = obj["spec"]
    assert spec["model"] == {"name": "meta-llama/Llama-3.1-8B-Instruct",
                             "preset": "llama-3.1-8b",
                             "config": modelconfig.PRESETS["llama-3.1-8b"]["config"],
                             "configSource": "preset:llama-3.1-8b"}
    # ShuntServe's own workload when the caller leaves the averages out.
    assert spec["workload"] == {"avgInputTokens": 763, "avgOutputTokens": 232}
    # A decimal string, as PACSrun's API carries no floats.
    assert spec["hourlyBudget"] == "3.0"
    assert spec["placement"] == {"regions": ["shadeform"], "capacityType": "on-demand"}
    # HYPERUN-SERVING-SA. The ServiceAccount a job gets, which the EC2 role trusts in each tenant
    # namespace. Without it the driver pods run as `default` and every multi-VM plan fails at
    # sts:AssumeRoleWithWebIdentity (2026-10-25). The test app leaves HYPERUN_SERVICE_ACCOUNT
    # unset, so this is the setting's default.
    assert spec["serviceAccountName"] == client.app.state.settings.service_account
    assert spec["serviceAccountName"] == "hyperunjob-writer"
    assert "hfTokenSecret" not in spec and "maxSpendUSD" not in spec
    view = response.json()
    assert view["endpoint"] == {"path": "/v1/servings/lab-alice/chat-8b/v1", "ready": False}
    assert view["billing"] == ("No plan rented yet. When one is, it costs at most $3.00/h, "
                               "$72.00 a day. There is no total cap: it runs until you delete it.")


def test_an_hf_id_is_read_with_the_named_token_and_the_token_rides_as_a_secret_ref(
        client, cluster, monkeypatch):
    cluster.secrets["lab-alice"] = {"HF_TOKEN": "hf-alice-secret-value"}
    calls = []

    def fetch(url, token=None):
        import json
        calls.append((url, token))
        config = {"architectures": ["Qwen3ForCausalLM"], "hidden_size": 4096,
                  "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
                  "head_dim": 128, "intermediate_size": 12288, "vocab_size": 151936,
                  "torch_dtype": "bfloat16"}
        return modelconfig.Fetched(200, {}, json.dumps(config).encode())

    monkeypatch.setattr(modelconfig, "fetch_url", fetch)
    body = {"name": "qwen", "model": {"name": "Qwen/Qwen3-8B", "revision": "main"},
            "avg_input_tokens": 512, "avg_output_tokens": 128, "hourly_budget": 2.5,
            "max_spend_usd": 50, "image": "vllm/vllm-openai:v0.10.0",
            "hf_token_secret": "HF_TOKEN"}
    response = as_alice(client, "POST", "/v1/servings", json=body)
    assert response.status_code == 201, response.text
    assert calls == [("https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json",
                      "hf-alice-secret-value")]
    spec = cluster.servings[("lab-alice", "qwen")]["spec"]
    assert spec["model"]["name"] == "Qwen/Qwen3-8B" and spec["model"]["revision"] == "main"
    assert spec["model"]["config"]["numLayers"] == 36
    assert spec["model"]["configSource"].endswith("/Qwen/Qwen3-8B/resolve/main/config.json")
    assert spec["workload"] == {"avgInputTokens": 512, "avgOutputTokens": 128}
    assert (spec["maxSpendUSD"], spec["image"]) == ("50.0", "vllm/vllm-openai:v0.10.0")
    # The NAME of the registered value, never the value: a secretKeyRef into the namespace's one
    # store, the shape PACSrun's own CRDs use for a Secret.
    assert spec["hfTokenSecret"] == {"name": "hyperun-user-secrets", "key": "HF_TOKEN"}
    assert "hf-alice-secret-value" not in response.text
    assert "hf-alice-secret-value" not in str(cluster.servings)


def test_a_caller_filled_config_is_checked_and_needs_no_fetch(client, cluster, monkeypatch):
    def no_fetch(url, token=None):
        raise AssertionError("a config the caller filled must not be fetched again")
    monkeypatch.setattr(modelconfig, "fetch_url", no_fetch)
    config = servings.config_view(modelconfig.PRESETS["llama-3.1-8b"]["config"])
    response = as_alice(client, "POST", "/v1/servings", json={
        "name": "own", "model": {"name": "meta-llama/Llama-3.1-8B", "config": config},
        "hourly_budget": 1.5})
    assert response.status_code == 201, response.text
    assert cluster.servings[("lab-alice", "own")]["spec"]["model"]["configSource"] == "caller"


def test_every_problem_is_named_at_once_and_nothing_is_created(client, cluster):
    response = as_alice(client, "POST", "/v1/servings", json={
        "name": "Chat_8B", "model": {"preset": "llama-3.1-8b", "name": "x/y",
                                     "config_url": "https://models.example.com/c.json"},
        "avg_input_tokens": 0, "hf_token_secret": "HF_TOKEN"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    for phrase in ("name 'Chat_8B'", "hourly_budget is missing", "avg_input_tokens is 0",
                   "exactly one of preset", "HF_TOKEN is not registered"):
        assert phrase in detail
    assert cluster.servings == {}


def test_a_moe_model_is_refused_at_create_with_the_lookups_words(client, cluster, monkeypatch):
    import json

    moe = {"architectures": ["Qwen3MoeForCausalLM"], "num_experts": 128}
    monkeypatch.setattr(modelconfig, "fetch_url",
                        lambda url, token=None: modelconfig.Fetched(200, {},
                                                                    json.dumps(moe).encode()))
    response = as_alice(client, "POST", "/v1/servings", json={
        "name": "moe", "model": {"name": "Qwen/Qwen3-30B-A3B"}, "hourly_budget": 5})
    assert response.status_code == 422
    assert response.json()["detail"].startswith("[model-config-moe]")
    assert cluster.servings == {}


def test_a_name_that_exists_is_a_409(client, cluster):
    seed(cluster)
    response = as_alice(client, "POST", "/v1/servings", json=PRESET_BODY)
    assert response.status_code == 409
    assert "hyperun serving-delete chat-8b" in response.json()["detail"]


def test_creating_needs_a_login(client, cluster):
    assert client.post("/v1/servings", json=PRESET_BODY).status_code == 401
    assert cluster.servings == {}


# ------------------------------------------------------------------ 3 and 4. read


def test_the_view_carries_plans_score_chosen_stages_catalog_and_the_path_only(client, cluster):
    seed(cluster)
    response = as_alice(client, "GET", "/v1/servings/chat-8b")
    assert response.status_code == 200, response.text
    view = response.json()
    assert view["phase"] == "Serving" and view["namespace"] == "lab-alice"
    assert [(p["rank"], p["score"]) for p in view["serving_plans"]] == [(1, "12.558"),
                                                                        (2, "9.100")]
    chosen = view["serving_plan_chosen"]
    assert (chosen["rank"], chosen["score"], chosen["cost_per_hour"]) == (1, "12.558", "1.3500")
    assert chosen["stages"] == [{"cloud": "hyperstack", "region": "montreal-canada-2",
                                 "instance_type": "A100_80G", "layers": 32, "tp": 1}]
    assert view["serving_catalog"].startswith("Shadeform catalog fetched")
    assert view["endpoint"] == {"path": "/v1/servings/lab-alice/chat-8b/v1", "ready": True}
    assert view["model"]["config"]["num_layers"] == 32
    assert view["billing"] == ("Renting $1.35/h now (rank 1), $32.40 a day. There is no total "
                               "cap: it runs until you delete it.")
    # HYPERUN-NO-INTERNAL-NAMES: the vLLM address and the Secret's object name stay inside.
    assert "10.0.42.7" not in response.text and "chat-8b-api-key" not in response.text


def test_the_list_leaves_plans_out_and_hides_a_namespace_mates_serving(client, cluster):
    seed(cluster)
    seed(cluster, name="bobs", owner="bob")
    answer = as_alice(client, "GET", "/v1/servings").json()
    assert answer["total"] == 1
    row = answer["servings"][0]
    assert row["name"] == "chat-8b" and "serving_plans" not in row
    assert row["serving_plan_chosen"]["rank"] == 1
    # The operator sees both.
    assert as_root(client, "GET", "/v1/servings?namespace=lab-alice").json()["total"] == 2


def test_a_namespace_mates_serving_reads_as_absent_and_cannot_be_deleted(client, cluster):
    seed(cluster, name="bobs", owner="bob")
    assert as_alice(client, "GET", "/v1/servings/bobs").status_code == 404
    assert as_alice(client, "DELETE", "/v1/servings/bobs").status_code == 404
    assert as_alice(client, "GET", "/v1/servings/bobs/api-key").status_code == 404
    assert ("lab-alice", "bobs") in cluster.servings


def test_delete_removes_it_and_a_second_delete_is_404(client, cluster):
    seed(cluster)
    assert as_alice(client, "DELETE", "/v1/servings/chat-8b").status_code == 204
    assert cluster.servings == {}
    assert as_alice(client, "DELETE", "/v1/servings/chat-8b").status_code == 404


# ------------------------------------------------------------------ 5. the key


def test_the_owner_reads_the_key_once_and_nothing_caches_it(client, cluster):
    seed(cluster)
    cluster.serving_keys[("lab-alice", "chat-8b")] = "sk-serving-test-key"
    response = as_alice(client, "GET", "/v1/servings/chat-8b/api-key")
    assert response.status_code == 200
    assert response.json() == {"api_key": "sk-serving-test-key",
                               "endpoint_path": "/v1/servings/lab-alice/chat-8b/v1"}
    assert response.headers["cache-control"] == "no-store"


def test_a_serving_without_a_key_yet_is_a_409_naming_its_phase(client, cluster):
    seed(cluster, status={"phase": "Planning"})
    response = as_alice(client, "GET", "/v1/servings/chat-8b/api-key")
    assert response.status_code == 409
    assert "no API key yet" in response.json()["detail"] and "Planning" in response.json()["detail"]


@pytest.mark.parametrize("path", ["/v1/servings/Not_A_Name", "/v1/servings/x/api-key"])
def test_an_unknown_or_malformed_name_is_404(client, path):
    assert as_alice(client, "GET", path).status_code == 404
