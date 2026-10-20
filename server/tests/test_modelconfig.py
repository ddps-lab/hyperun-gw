"""The model-config lookup: config.json in, the contract's dense-decoder numbers out, or a refusal.

HYPERUN-MODEL-CONFIG. WHAT IT PINS.
  1. A Llama 3.1 and a Qwen3 config.json map to the contract's `config` block exactly, including
     the two derivations (head_dim absent -> hidden / heads; Qwen3's stated head_dim kept even
     where it is NOT hidden / heads).
  2. A mixture of experts is refused by the keys that make it one, before its architecture.
  3. An architecture outside the four dense families, and a missing number, are refused by name.
  4. An HF id and a URL both become the right URL to read; a huggingface.co URL gives back its
     id and revision; a URL aimed at a private address is refused before anything is sent.
  5. Through the route: a gated repository without a token is a clear 422, a named token is
     read from the namespace and handed to the fetch and never echoed, and an unregistered name
     is a 400.

THE FIXTURES ARE THE REAL FILES. Qwen3-8B and Qwen3-30B-A3B are the config.json the Hub served on
2026-10-20 (GET https://huggingface.co/<id>/resolve/main/config.json, copied whole). Llama 3.1 8B
is gated and the Hub answered that anonymous read with 401 `X-Error-Code: GatedRepo`, so its
fixture carries the keys this module reads with the numbers PACSrun's planner holds for the same
model (solver/shuntserve/plan.py, LLAMA_3_1_8B), which were read with transformers.AutoConfig.

The imports come through test_routes so this file adds no line naming the package (the
old-name ceiling in .github/scripts/old-name-ceiling.sh counts such lines).
"""

import pytest

from test_routes import as_alice, client, cluster, main  # noqa: F401  (fixtures)

modelconfig = main.modelconfig
Fetched = modelconfig.Fetched
ModelConfigError = modelconfig.ModelConfigError

QWEN3_8B = {"architectures": ["Qwen3ForCausalLM"], "attention_bias": False,
            "attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645,
            "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096,
            "initializer_range": 0.02, "intermediate_size": 12288,
            "max_position_embeddings": 40960, "max_window_layers": 36, "model_type": "qwen3",
            "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06, "rope_scaling": None, "rope_theta": 1000000,
            "sliding_window": None, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
            "transformers_version": "4.51.0", "use_cache": True, "use_sliding_window": False,
            "vocab_size": 151936}

QWEN3_30B_A3B = {"architectures": ["Qwen3MoeForCausalLM"], "attention_bias": False,
                 "attention_dropout": 0.0, "bos_token_id": 151643, "decoder_sparse_step": 1,
                 "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
                 "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 6144,
                 "max_position_embeddings": 40960, "max_window_layers": 48,
                 "mlp_only_layers": [], "model_type": "qwen3_moe",
                 "moe_intermediate_size": 768, "norm_topk_prob": True,
                 "num_attention_heads": 32, "num_experts": 128, "num_experts_per_tok": 8,
                 "num_hidden_layers": 48, "num_key_value_heads": 4,
                 "output_router_logits": False, "rms_norm_eps": 1e-06, "rope_scaling": None,
                 "rope_theta": 1000000.0, "router_aux_loss_coef": 0.001,
                 "sliding_window": None, "tie_word_embeddings": False,
                 "torch_dtype": "bfloat16", "transformers_version": "4.51.0",
                 "use_cache": True, "use_sliding_window": False, "vocab_size": 151936}

LLAMA_31_8B = {"architectures": ["LlamaForCausalLM"], "hidden_size": 4096,
               "intermediate_size": 14336, "model_type": "llama", "num_attention_heads": 32,
               "num_hidden_layers": 32, "num_key_value_heads": 8, "tie_word_embeddings": False,
               "torch_dtype": "bfloat16", "vocab_size": 128256}


def fetcher(answers):
    """A stand-in for fetch_url: url -> (status, headers, json-or-bytes). Records each call."""
    import json

    calls = []

    def fetch(url, token=None):
        calls.append((url, token))
        status, headers, body = answers[url]
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        return Fetched(status, {k.lower(): v for k, v in headers.items()}, raw)

    fetch.calls = calls
    return fetch


HF = "https://huggingface.co"


# ------------------------------------------------------------------ 1. mapping


def test_a_llama_config_maps_to_the_contract_and_says_what_it_derived():
    family, config, notes = modelconfig.dense_config(LLAMA_31_8B)
    assert family == "Llama 3.x"
    assert config == {"architecture": "LlamaForCausalLM", "hiddenSize": 4096, "numLayers": 32,
                      "numAttentionHeads": 32, "numKeyValueHeads": 8, "headDim": 128,
                      "intermediateSize": 14336, "vocabSize": 128256, "dtype": "bfloat16"}
    # Exactly the preset's numbers, so a Llama 3.1 8B read from the Hub and the preset agree.
    assert config == modelconfig.PRESETS["llama-3.1-8b"]["config"]
    assert notes == ["head_dim is absent, so it is hidden_size / num_attention_heads = "
                     "4096 / 32 = 128."]


def test_a_qwen3_config_maps_to_the_contract_with_its_own_head_dim():
    family, config, notes = modelconfig.dense_config(QWEN3_8B)
    assert family == "Qwen3 dense"
    assert config == {"architecture": "Qwen3ForCausalLM", "hiddenSize": 4096, "numLayers": 36,
                      "numAttentionHeads": 32, "numKeyValueHeads": 8, "headDim": 128,
                      "intermediateSize": 12288, "vocabSize": 151936, "dtype": "bfloat16"}
    assert notes == []


def test_a_stated_head_dim_wins_where_hidden_over_heads_would_be_wrong():
    """Qwen3-4B (Hub, 2026-10-20): hidden 2560, 32 heads, head_dim 128. 2560 / 32 is 80, which
    would undercount every KV-cache byte by a third. It also ties its embeddings, which the note
    turns into a parameter count: 151936 x 2560 = 388,956,160 (0.78 GB at 2 bytes)."""
    qwen3_4b = {**QWEN3_8B, "hidden_size": 2560, "intermediate_size": 9728,
                "tie_word_embeddings": True}
    _, config, notes = modelconfig.dense_config(qwen3_4b)
    assert config["headDim"] == 128
    assert any("388,956,160 parameters (0.78 GB" in n for n in notes)


def test_no_kv_heads_means_plain_multi_head_attention_and_a_newer_dtype_key_is_read():
    older = {k: v for k, v in LLAMA_31_8B.items()
             if k not in ("num_key_value_heads", "torch_dtype")}
    older["dtype"] = "float16"
    _, config, notes = modelconfig.dense_config(older)
    assert config["numKeyValueHeads"] == 32 and config["dtype"] == "float16"
    assert any("no grouped-query attention" in n for n in notes)


# ------------------------------------------------------------------ 2 and 3. refusals


def test_a_mixture_of_experts_is_refused_by_the_keys_that_make_it_one():
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.dense_config(QWEN3_30B_A3B)
    assert caught.value.code == "model-config-moe"
    assert "num_experts=128" in caught.value.message
    assert "num_experts_per_tok=8" in caught.value.message
    assert "moe_intermediate_size=768" in caught.value.message
    # Mixtral spells it num_local_experts (Hub, 2026-10-20), and its architecture would also
    # be refused; MoE is checked FIRST so the reason given is the real one.
    mixtral = {**LLAMA_31_8B, "architectures": ["MixtralForCausalLM"], "num_local_experts": 8,
               "num_experts_per_tok": 2}
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.dense_config(mixtral)
    assert caught.value.code == "model-config-moe" and "num_local_experts=8" in str(caught.value)


def test_a_dense_config_carrying_an_empty_moe_key_is_not_refused():
    """Some dense configs serialise a MoE key as null or 0; only a real value makes it MoE."""
    _, config, _ = modelconfig.dense_config({**QWEN3_8B, "num_experts": None,
                                             "num_experts_per_tok": 0})
    assert config["numLayers"] == 36


def test_an_unsupported_architecture_is_refused_with_the_supported_list():
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.dense_config({**LLAMA_31_8B, "architectures": ["GPT2LMHeadModel"]})
    assert caught.value.code == "model-config-architecture"
    for family in ("Llama 3.x", "Qwen2.5", "Qwen3 dense", "Mistral dense"):
        assert family in caught.value.message


def test_a_missing_number_is_refused_naming_every_one():
    broken = {k: v for k, v in QWEN3_8B.items() if k not in ("vocab_size", "hidden_size")}
    broken["num_hidden_layers"] = True      # a bool is not a size
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.dense_config(broken)
    assert caught.value.code == "model-config-missing"
    for key in ("hidden_size", "num_hidden_layers", "vocab_size"):
        assert key in caught.value.message


def test_a_caller_filled_config_gets_the_same_checks():
    family, clean = modelconfig.check_given_config(
        {**modelconfig.PRESETS["llama-3.1-8b"]["config"], "extra": 1})
    assert family == "Llama 3.x" and "extra" not in clean
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.check_given_config({"architecture": "LlamaForCausalLM", "hiddenSize": 0})
    assert caught.value.code == "model-config-missing" and "hiddenSize" in caught.value.message


# ------------------------------------------------------------------ 4. where to read


def test_an_hf_id_and_a_revision_become_the_resolve_url():
    source = modelconfig.source_for("Qwen/Qwen3-8B", None, "refs/pr/1")
    assert source.url == f"{HF}/Qwen/Qwen3-8B/resolve/refs%2Fpr%2F1/config.json"
    assert (source.repo, source.revision) == ("Qwen/Qwen3-8B", "refs/pr/1")
    assert modelconfig.source_for("Qwen/Qwen3-8B", None, None).url.endswith(
        "/resolve/main/config.json")


def test_a_huggingface_url_gives_back_its_id_and_a_blob_link_reads_the_file():
    source = modelconfig.source_for(None, f"{HF}/Qwen/Qwen3-8B/blob/main/config.json", None)
    assert source.url == f"{HF}/Qwen/Qwen3-8B/resolve/main/config.json"
    assert (source.repo, source.revision) == ("Qwen/Qwen3-8B", "main")
    other = modelconfig.source_for(None, "https://models.example.com/qwen/config.json", None)
    assert other.repo == "" and other.url == "https://models.example.com/qwen/config.json"


@pytest.mark.parametrize("repo, url, revision, code", [
    (None, None, None, "model-config-ask"),
    ("Qwen/Qwen3-8B", f"{HF}/x/y/resolve/main/config.json", None, "model-config-ask"),
    ("not an id", None, None, "model-config-ask"),
    ("Qwen/Qwen3-8B", None, "../../etc", "model-config-ask"),
    (None, "http://models.example.com/config.json", None, "model-config-url"),
    (None, "https://user:pw@models.example.com/config.json", None, "model-config-url"),
])
def test_a_malformed_ask_is_refused_before_anything_is_fetched(repo, url, revision, code):
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.source_for(repo, url, revision)
    assert caught.value.code == code


@pytest.mark.parametrize("address", ["10.100.0.1", "169.254.169.254", "169.254.170.23",
                                     "127.0.0.1", "192.168.1.5", "::1", "fd00::1"])
def test_a_host_that_resolves_to_a_private_address_is_refused(address):
    """This pod is inside the cluster: 169.254.170.23 is the EKS Pod Identity agent that hands
    out its AWS credentials, 10.100.0.1 the apiserver's ClusterIP range."""
    def resolve(host, port, type=0):
        return [(None, None, None, "", (address, port))]
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.check_host("config.example.com", resolve=resolve)
    assert caught.value.code == "model-config-url" and address in caught.value.message
    # One public and one private answer is still refused: the client may use either.
    def mixed(host, port, type=0):
        return [(None, None, None, "", ("93.184.216.34", port)),
                (None, None, None, "", (address, port))]
    with pytest.raises(ModelConfigError):
        modelconfig.check_host("config.example.com", resolve=mixed)


def test_lookup_reads_the_id_and_maps_it():
    fetch = fetcher({f"{HF}/Qwen/Qwen3-8B/resolve/main/config.json": (200, {}, QWEN3_8B)})
    found = modelconfig.lookup("Qwen/Qwen3-8B", None, None, fetch=fetch)
    assert (found.repo, found.family, found.config["numLayers"]) == ("Qwen/Qwen3-8B",
                                                                     "Qwen3 dense", 36)
    assert fetch.calls == [(f"{HF}/Qwen/Qwen3-8B/resolve/main/config.json", None)]


def test_lookup_reads_a_url_the_same_way():
    url = "https://models.example.com/llama/config.json"
    found = modelconfig.lookup(None, url, None, fetch=fetcher({url: (200, {}, LLAMA_31_8B)}))
    assert found.source == url and found.repo == "" and found.config["vocabSize"] == 128256


@pytest.mark.parametrize("status, headers, code, phrase", [
    (401, {"X-Error-Code": "GatedRepo"}, "model-config-gated", "is gated, and no Hugging Face "
                                                                "token was sent"),
    (401, {}, "model-config-gated", "gated, private or does not exist"),
    (404, {"X-Error-Code": "RepoNotFound"}, "model-config-not-found", "no Hugging Face repository"),
    (404, {"X-Error-Code": "EntryNotFound"}, "model-config-not-found", "has no config.json"),
    (503, {}, "model-config-unreachable", "answered 503"),
])
def test_each_refusal_from_the_hub_says_what_to_do(status, headers, code, phrase):
    url = f"{HF}/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json"
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.lookup("meta-llama/Llama-3.1-8B-Instruct", None, None,
                           fetch=fetcher({url: (status, headers, b"")}))
    assert caught.value.code == code and phrase in caught.value.message


def test_a_gated_repo_with_a_token_that_lacks_access_says_to_accept_the_licence():
    url = f"{HF}/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json"
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.lookup("meta-llama/Llama-3.1-8B-Instruct", None, None, token="hf-test",
                           fetch=fetcher({url: (403, {"X-Error-Code": "GatedRepo"}, b"")}))
    assert "cannot read" in caught.value.message and "licence" in caught.value.message
    assert "hf-test" not in str(caught.value)


def test_a_body_that_is_not_json_or_is_huge_is_refused():
    url = "https://models.example.com/config.json"
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.lookup(None, url, None, fetch=fetcher({url: (200, {}, b"<html>")}))
    assert caught.value.code == "model-config-missing"
    huge = b"{" + b" " * (modelconfig.FETCH_MAX_BYTES + 1)
    with pytest.raises(ModelConfigError) as caught:
        modelconfig.lookup(None, url, None, fetch=fetcher({url: (200, {}, huge)}))
    assert caught.value.code == "model-config-url"


# ------------------------------------------------------------------ 5. the route


def test_the_route_answers_the_snake_case_config_and_the_crd_block(client, monkeypatch):
    fetch = fetcher({f"{HF}/Qwen/Qwen3-8B/resolve/main/config.json": (200, {}, QWEN3_8B)})
    monkeypatch.setattr(modelconfig, "fetch_url", fetch)
    response = as_alice(client, "POST", "/v1/model-config", json={"model": "Qwen/Qwen3-8B"})
    assert response.status_code == 200, response.text
    answer = response.json()
    assert answer["name"] == "Qwen/Qwen3-8B" and answer["family"] == "Qwen3 dense"
    assert answer["config"]["num_layers"] == 36 and answer["config"]["head_dim"] == 128
    assert answer["spec_config"]["numLayers"] == 36
    # No token was named, so none was read and none was sent.
    assert fetch.calls[0][1] is None


def test_a_gated_repo_without_a_token_is_a_clear_422(client, monkeypatch):
    url = f"{HF}/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json"
    monkeypatch.setattr(modelconfig, "fetch_url", fetcher({url: (
        401, {"X-Error-Code": "GatedRepo"}, b"Access to model is restricted.")}))
    response = as_alice(client, "POST", "/v1/model-config",
                        json={"model": "meta-llama/Llama-3.1-8B-Instruct"})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail.startswith("[model-config-gated]")
    assert "hyperun secret-set HF_TOKEN" in detail and "hf_token_secret" in detail


def test_a_named_token_is_read_from_the_namespace_sent_to_the_hub_and_never_echoed(
        client, cluster, monkeypatch):
    cluster.secrets["lab-alice"] = {"HF_TOKEN": "hf-alice-secret-value"}
    url = f"{HF}/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json"
    fetch = fetcher({url: (200, {}, LLAMA_31_8B)})
    monkeypatch.setattr(modelconfig, "fetch_url", fetch)
    response = as_alice(client, "POST", "/v1/model-config",
                        json={"model": "meta-llama/Llama-3.1-8B-Instruct",
                              "hf_token_secret": "HF_TOKEN"})
    assert response.status_code == 200, response.text
    assert fetch.calls == [(url, "hf-alice-secret-value")]
    assert "hf-alice-secret-value" not in response.text


def test_a_token_name_that_is_not_registered_is_a_400(client, monkeypatch):
    monkeypatch.setattr(modelconfig, "fetch_url", fetcher({}))
    response = as_alice(client, "POST", "/v1/model-config",
                        json={"model": "Qwen/Qwen3-8B", "hf_token_secret": "HF_TOKEN"})
    assert response.status_code == 400
    assert "hyperun secret-set HF_TOKEN" in response.json()["detail"]


def test_a_moe_model_is_a_422_through_the_route(client, monkeypatch):
    url = f"{HF}/Qwen/Qwen3-30B-A3B/resolve/main/config.json"
    monkeypatch.setattr(modelconfig, "fetch_url", fetcher({url: (200, {}, QWEN3_30B_A3B)}))
    response = as_alice(client, "POST", "/v1/model-config",
                        json={"model": "Qwen/Qwen3-30B-A3B"})
    assert response.status_code == 422
    assert response.json()["detail"].startswith("[model-config-moe]")


def test_the_route_needs_a_login(client):
    assert client.post("/v1/model-config", json={"model": "Qwen/Qwen3-8B"}).status_code == 401


def test_the_supported_model_guidance_lists_presets_families_and_a_recommendation(client):
    answer = client.get("/v1/serving-models").json()
    assert [p["preset"] for p in answer["presets"]] == ["llama-3.1-8b", "llama-3.1-70b"]
    assert {f["family"] for f in answer["families"]} == {"Llama 3.x", "Qwen2.5", "Qwen3 dense",
                                                         "Mistral dense"}
    assert answer["recommended"] == "llama-3.1-8b"
    assert "Qwen/Qwen3-8B" in answer["recommendation"]
    assert "num_experts" in answer["refused"]
