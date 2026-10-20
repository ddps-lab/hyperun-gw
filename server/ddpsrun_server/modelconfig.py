"""Which models a serving can be planned for, and the numbers the planner needs about one.

HYPERUN-MODEL-CONFIG. A HyperunServing names its model in `spec.model` and carries the model's
structure in `spec.model.config`: the eight numbers ShuntServe's cost model reads (hidden size,
layer count, attention and KV heads, head size, MLP width, vocabulary, dtype). This module is
where those numbers come from when the caller did not pick a preset.

END-TO-END FLOW of one `POST /v1/model-config` (main.model_config_route):

  1. The caller sends a Hugging Face repository id ("Qwen/Qwen3-8B", optionally a revision) or a
     URL to a config.json, and optionally the NAME of a value they registered with
     `PUT /v1/secrets/{name}` (e.g. HF_TOKEN) for a gated repository.
  2. `source_for()` turns that into the one URL to read. A repository id becomes
     `https://huggingface.co/<id>/resolve/<revision>/config.json`; a huggingface.co URL is read
     back into an id and a revision so the serving can name the weights it loads; any other
     URL is used as it is and has no id.
  3. `fetch_config()` GETs it: https only, every address the host resolves to must be a public
     one, at most 3 redirects each checked the same way, 10 s, 1 MiB. The HF token goes only to
     huggingface.co and never follows a redirect elsewhere.
  4. `dense_config()` refuses a mixture-of-experts model by the keys that make it one, an
     architecture outside the four dense families below, and a config missing a number; and
     otherwise returns the contract's `config` block plus notes on what was derived.
  5. The route answers `ModelConfigResponse`; `servings.resolve_model` calls the same `lookup`
     when a serving is created from an id or a URL without a config.

WHY THE GATEWAY FETCHES IT AND NOT THE OPERATOR. The person choosing a model is at the screen or
the CLI, and a model the planner cannot handle should be refused while they are still choosing,
not after a HyperunServing exists and its status says so. A coding agent can do the same thing
itself (read config.json, fill `spec.model.config`); agent/skills/hyperun/SKILL.md says how.

WHAT THIS DOES NOT DO. It does not check that vLLM can load the weights, that the weights fit the
plan's GPUs (the planner does that), or that the HF token can download the weights rather than
only the config. A gated repository's config.json and its weights sit behind the same licence,
so a token that reads one reads the other, but nothing here proves it.

Grep anchor: HYPERUN-MODEL-CONFIG
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import quote, urlsplit

import httpx

# ---------------------------------------------------------------------------------------------
# What the planner is known to handle.
# ---------------------------------------------------------------------------------------------

# The two presets PACSrun's planner carries (solver/shuntserve/plan.py, MODEL_PRESETS, at PACSrun
# 01302ce). The numbers are COPIED from LLAMA_3_1_8B and LLAMA_3_1_70B there, where head_dim None
# means hidden_size / num_attention_heads: 4096 / 32 = 128 and 8192 / 64 = 128. The HF ids are
# the Instruct checkpoints: 8B is the model of the first real serving test (plan.py, "the user's
# choice, 2026-10-19"), and 70B is the same family at the size ShuntServe's README plans.
PRESETS: dict[str, dict[str, Any]] = {
    "llama-3.1-8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "gated": True,
        "config": {"architecture": "LlamaForCausalLM", "hiddenSize": 4096, "numLayers": 32,
                   "numAttentionHeads": 32, "numKeyValueHeads": 8, "headDim": 128,
                   "intermediateSize": 14336, "vocabSize": 128256, "dtype": "bfloat16"},
        "why": "the model of the first real serving test; 16 GB of FP16 weights, so one 24 GB "
               "or larger GPU can hold it",
    },
    "llama-3.1-70b": {
        "name": "meta-llama/Llama-3.1-70B-Instruct",
        "gated": True,
        "config": {"architecture": "LlamaForCausalLM", "hiddenSize": 8192, "numLayers": 80,
                   "numAttentionHeads": 64, "numKeyValueHeads": 8, "headDim": 128,
                   "intermediateSize": 28672, "vocabSize": 128256, "dtype": "bfloat16"},
        "why": "ShuntServe's own README model; about 141 GB of FP16 weights, so it needs several "
               "80 GB GPUs",
    },
}

# What is recommended when the caller names no model. The 8B preset: it is the one a real serving
# test has run, and it is the smallest, so its plans are the cheapest to try.
RECOMMENDED_PRESET = "llama-3.1-8b"

# ★ THE DENSE FAMILIES, BY THE `architectures[0]` STRING config.json CARRIES. These are the
# user's "known-good" list (2026-10-20): decoder-only transformers whose layers are all the same
# attention + MLP block, which is the structure ShuntServe's cost model assumes (it splits a model
# by LAYER RANGES and prices each range by the same per-layer arithmetic). A family absent here is
# refused rather than guessed at, because a plan priced on the wrong structure looks exactly as
# confident as a right one.
DENSE_FAMILIES: dict[str, dict[str, str]] = {
    "LlamaForCausalLM": {"family": "Llama 3.x", "example": "meta-llama/Llama-3.1-8B-Instruct"},
    "Qwen2ForCausalLM": {"family": "Qwen2.5", "example": "Qwen/Qwen2.5-7B-Instruct"},
    "Qwen3ForCausalLM": {"family": "Qwen3 dense", "example": "Qwen/Qwen3-8B"},
    "MistralForCausalLM": {"family": "Mistral dense", "example": "mistralai/Mistral-7B-Instruct-v0.3"},
}

# The keys that make a config.json a mixture of experts. Each is how one family spells "this layer
# has N expert MLPs and a router picks some": Qwen2-MoE and Qwen3-MoE write num_experts and
# moe_intermediate_size, Mixtral writes num_local_experts, DeepSeek writes n_routed_experts, and
# every one of them writes num_experts_per_tok. A MoE model's per-token compute and its memory are
# different quantities, and ShuntServe's cost model has one number for both, so it is refused.
MOE_KEYS: tuple[str, ...] = ("num_experts", "num_local_experts", "n_routed_experts",
                             "num_experts_per_tok", "moe_intermediate_size")

# config.json key -> the contract's camelCase key. Every one is required except head_dim and
# num_key_value_heads, which have well-defined defaults (see dense_config).
REQUIRED_KEYS: tuple[tuple[str, str], ...] = (
    ("hidden_size", "hiddenSize"),
    ("num_hidden_layers", "numLayers"),
    ("num_attention_heads", "numAttentionHeads"),
    ("intermediate_size", "intermediateSize"),
    ("vocab_size", "vocabSize"),
)

# spec.model.config's keys, in the contract's order.
CONTRACT_KEYS: tuple[str, ...] = ("architecture", "hiddenSize", "numLayers", "numAttentionHeads",
                                  "numKeyValueHeads", "headDim", "intermediateSize", "vocabSize",
                                  "dtype")

HF_HOST = "huggingface.co"
FETCH_TIMEOUT_SECONDS = 10.0
# A real config.json is 1 to 3 KB (Llama-3.1-8B-Instruct's is 0.9 KB). 1 MiB is far past any of
# them and still small enough that a URL pointing at something huge costs this pod nothing.
FETCH_MAX_BYTES = 1024 * 1024
MAX_REDIRECTS = 3

# An HF repository id: owner/name, each part what the Hub accepts (letters, digits, - _ .).
HF_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
# A revision: a branch, a tag or a commit. Slashes are allowed (refs/pr/1) and percent-encoded
# into the URL; ".." is refused so a revision cannot walk the path.
HF_REVISION = re.compile(r"^[A-Za-z0-9._/-]{1,100}$")
# https://huggingface.co/<owner>/<name>/(resolve|blob)/<revision>/config.json
HF_CONFIG_URL = re.compile(
    r"^/(?P<repo>[^/]+/[^/]+)/(?:resolve|blob)/(?P<revision>.+)/config\.json$")


class ModelConfigError(Exception):
    """A model this lookup cannot answer for. `code` is a stable word a caller can branch on.

    The routes turn it into HTTP 422 (the model) or 502 (the Hub could not be reached), with
    `[code] message` as the detail, the shape `/v1/jobs` refusals already use.
    """

    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


@dataclass
class Fetched:
    """What one GET of a config.json answered."""

    status: int
    headers: dict[str, str]
    body: bytes


@dataclass
class Source:
    """Where a config.json is read from, and what it says about the model's identity."""

    url: str
    repo: str = ""          # the HF id the serving loads, "" for a URL that is not the Hub
    revision: str = ""      # "" when none was named; the Hub then reads `main`


@dataclass
class ModelConfig:
    """What `lookup` found: the contract's `config` block and where it came from."""

    repo: str
    revision: str
    source: str
    family: str
    config: dict[str, Any]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# 2. Which URL to read.
# ---------------------------------------------------------------------------------------------

def source_for(repo: str | None, config_url: str | None, revision: str | None) -> Source:
    """Turn the caller's words into the one URL to read.

    Args:
        repo: an HF repository id, e.g. "Qwen/Qwen3-8B".
        config_url: a URL to a config.json. Exactly one of the two.
        revision: a branch, tag or commit for `repo`; ignored for a URL, which names its own.

    Returns:
        A `Source`. For a huggingface.co URL the id and revision are read back out of it, so a
        serving created from that URL still names the weights vLLM must load.

    Raises:
        ModelConfigError: neither or both given, or a malformed id, revision or URL.
    """
    repo = (repo or "").strip()
    config_url = (config_url or "").strip()
    revision = (revision or "").strip()
    if bool(repo) == bool(config_url):
        raise ModelConfigError(
            "model-config-ask",
            "send exactly one of `model` (a Hugging Face repository id such as Qwen/Qwen3-8B) "
            "and `config_url` (a URL to a config.json).")
    if revision and (not HF_REVISION.fullmatch(revision) or ".." in revision):
        raise ModelConfigError("model-config-ask",
                               f"revision {revision!r} is not a branch, tag or commit name.")
    if repo:
        if not HF_REPO_ID.fullmatch(repo):
            raise ModelConfigError(
                "model-config-ask",
                f"{repo!r} is not a Hugging Face repository id. It looks like owner/name, e.g. "
                "Qwen/Qwen3-8B.")
        rev = revision or "main"
        return Source(url=f"https://{HF_HOST}/{repo}/resolve/{quote(rev, safe='')}/config.json",
                      repo=repo, revision=revision)

    parts = urlsplit(config_url)
    if parts.scheme != "https" or not parts.hostname:
        raise ModelConfigError("model-config-url",
                               f"{config_url!r} is not an https URL. Only https is read.")
    if parts.username or parts.password:
        raise ModelConfigError(
            "model-config-url",
            "the URL carries a user name or password. Send a token by naming a registered "
            "secret in hf_token_secret instead, so it never travels in a URL.")
    if parts.hostname == HF_HOST:
        matched = HF_CONFIG_URL.fullmatch(parts.path)
        if matched and HF_REPO_ID.fullmatch(matched["repo"]):
            # A /blob/ link is the Hub's HTML page about the file; /resolve/ is the file.
            rev = matched["revision"]
            return Source(url=f"https://{HF_HOST}/{matched['repo']}/resolve/{rev}/config.json",
                          repo=matched["repo"], revision=rev)
    return Source(url=config_url)


# ---------------------------------------------------------------------------------------------
# 3. Reading it.
# ---------------------------------------------------------------------------------------------

def check_host(host: str, resolve: Callable[..., Any] = socket.getaddrinfo) -> None:
    """Refuse a host that resolves to anything but a public address.

    ★ WHY: THIS POD IS INSIDE THE CLUSTER. A URL the caller chooses is fetched from here, so
    without this check `https://10.100.0.1/...` reaches the apiserver's address, a Service's
    ClusterIP reaches anything in the VPC, and 169.254.169.254 or 169.254.170.23 reach the
    instance metadata service and the EKS Pod Identity agent, which hands out this pod's AWS
    credentials. EVERY address the name resolves to must be global, because a name can carry
    one public and one private address and the client may use either.

    WHAT IT DOES NOT STOP: a name that resolves to a public address here and a private one a
    moment later when httpx resolves it again (DNS rebinding). The route is authenticated, so
    the caller is a registered person, and the answer is parsed as JSON and never echoed raw.

    Raises:
        ModelConfigError: the host does not resolve, or resolves to a non-public address.
    """
    try:
        infos = resolve(host, 443, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as exc:
        raise ModelConfigError("model-config-unreachable",
                               f"{host} does not resolve ({exc}).", status=502) from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ModelConfigError(
                "model-config-url",
                f"{host} resolves to {address}, which is not a public address. This server "
                f"reads config.json only from the public internet.")


def fetch_url(url: str, token: str | None = None) -> Fetched:
    """GET one config.json under the rules in the module docstring.

    Args:
        url: an https URL `source_for` produced.
        token: the HF token to send, or None. Sent as `Authorization: Bearer` to huggingface.co
            ONLY, and dropped on any redirect that leaves it. Never logged.

    Returns:
        The final answer, its body cut at FETCH_MAX_BYTES + 1 (so an oversized body is
        detectable without being read whole).

    Raises:
        ModelConfigError: a refused address, too many redirects, or the network failed.
    """
    current = url
    with httpx.Client(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=False) as http:
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(current)
            if parts.scheme != "https" or not parts.hostname:
                raise ModelConfigError("model-config-url",
                                       f"a redirect led to {current!r}, which is not https.")
            check_host(parts.hostname)
            headers = {"Accept": "application/json"}
            if token and parts.hostname == HF_HOST:
                headers["Authorization"] = f"Bearer {token}"
            try:
                with http.stream("GET", current, headers=headers) as response:
                    if response.is_redirect and response.headers.get("location"):
                        current = str(response.url.join(response.headers["location"]))
                        continue
                    body = b""
                    for chunk in response.iter_bytes():
                        body += chunk
                        if len(body) > FETCH_MAX_BYTES:
                            break
                    return Fetched(response.status_code, dict(response.headers), body)
            except httpx.HTTPError as exc:
                # The exception names the URL and the failure, never the request headers.
                raise ModelConfigError(
                    "model-config-unreachable",
                    f"could not read {current}: {type(exc).__name__}", status=502) from exc
    raise ModelConfigError("model-config-url",
                           f"{url} redirected more than {MAX_REDIRECTS} times.")


def _hub_refusal(fetched: Fetched, source: Source, token_sent: bool) -> ModelConfigError:
    """Turn a non-200 from the Hub into a sentence that says what to do.

    The Hub names its reason in `X-Error-Code` (GatedRepo, RepoNotFound, RevisionNotFound,
    EntryNotFound) and a sentence in `X-Error-Message`. Measured 2026-10-20 without a token:
    meta-llama/Llama-3.1-8B-Instruct answered 401 with `X-Error-Code: GatedRepo`, and a
    repository that does not exist answered 401 with NO code and "Invalid username or password."
    So a 401 with no code is described as gated, private or absent, all three.
    """
    code = fetched.headers.get("x-error-code", "")
    where = source.repo or source.url
    if code == "GatedRepo" or (fetched.status in (401, 403) and source.repo):
        if not token_sent:
            what = ("is gated" if code == "GatedRepo" else
                    "is gated, private or does not exist (the Hub answers 401 for all three "
                    "without a token)")
            return ModelConfigError(
                "model-config-gated",
                f"{where} {what}, and no Hugging Face token was sent. Check the id, accept the "
                f"model's licence on https://{HF_HOST}/{source.repo or ''}, register a token "
                f"with `hyperun secret-set HF_TOKEN`, and name it in hf_token_secret.")
        return ModelConfigError(
            "model-config-gated",
            f"the Hugging Face token you named cannot read {where}. Accept the model's licence "
            f"on https://{HF_HOST}/{source.repo or ''} with the account that token belongs to.")
    if code in ("RepoNotFound",) or (fetched.status == 404 and not code and source.repo):
        return ModelConfigError("model-config-not-found",
                                f"there is no Hugging Face repository {where}.")
    if code in ("RevisionNotFound", "EntryNotFound") or fetched.status == 404:
        return ModelConfigError(
            "model-config-not-found",
            f"{where} has no config.json"
            + (f" at revision {source.revision}" if source.revision else "") + ".")
    # A 5xx, a 429 rate limit, or anything else: the Hub's problem, not the model's.
    return ModelConfigError("model-config-unreachable",
                            f"{source.url} answered {fetched.status}.", status=502)


# ---------------------------------------------------------------------------------------------
# 4. Mapping it.
# ---------------------------------------------------------------------------------------------

def _positive_int(value: Any) -> int | None:
    """A config number as a positive int, or None. bool is refused: True is not a size."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def dense_config(raw: Any) -> tuple[str, dict[str, Any], list[str]]:
    """Refuse or map one parsed config.json.

    Args:
        raw: the parsed JSON.

    Returns:
        (family name, the contract's `config` block in camelCase, notes on derived values).

    Raises:
        ModelConfigError: `model-config-moe` for a mixture of experts, naming the keys found;
            `model-config-architecture` for anything outside DENSE_FAMILIES; and
            `model-config-missing` naming every key that is absent or not a positive integer.
    """
    if not isinstance(raw, dict):
        raise ModelConfigError("model-config-missing", "config.json is not a JSON object.")
    # MoE first, so a Qwen3-MoE or a Mixtral is told WHY rather than "unsupported architecture".
    moe = [k for k in MOE_KEYS if raw.get(k) not in (None, 0, False, [])]
    if moe:
        raise ModelConfigError(
            "model-config-moe",
            f"this is a mixture-of-experts model ({', '.join(f'{k}={raw[k]!r}' for k in moe)}). "
            f"The planner's cost model has one number for a layer's compute and its memory, "
            f"and a MoE layer's are different, so its plans would be wrong. Choose a dense "
            f"model: {', '.join(v['family'] for v in DENSE_FAMILIES.values())}.")
    architectures = raw.get("architectures") or []
    architecture = architectures[0] if isinstance(architectures, list) and architectures else ""
    if architecture not in DENSE_FAMILIES:
        raise ModelConfigError(
            "model-config-architecture",
            f"architecture {architecture or 'missing'!r} is not one the planner is known to "
            f"handle. Supported dense families: "
            + ", ".join(f"{v['family']} ({k}, e.g. {v['example']})"
                        for k, v in DENSE_FAMILIES.items()) + ".")

    config: dict[str, Any] = {"architecture": architecture}
    missing = []
    for source_key, contract_key in REQUIRED_KEYS:
        number = _positive_int(raw.get(source_key))
        if number is None:
            missing.append(source_key)
        config[contract_key] = number
    if missing:
        raise ModelConfigError(
            "model-config-missing",
            f"config.json has no positive integer for {', '.join(missing)}, and the planner "
            f"needs every one of them.")

    notes: list[str] = []
    heads = config["numAttentionHeads"]
    kv = raw.get("num_key_value_heads")
    if kv is None:
        # transformers' own default for these four families: no key means every head has its
        # own K and V (plain multi-head attention).
        config["numKeyValueHeads"] = heads
        notes.append(f"num_key_value_heads is absent, so it equals num_attention_heads ({heads}): "
                     f"no grouped-query attention.")
    elif _positive_int(kv) is None:
        raise ModelConfigError("model-config-missing",
                               f"num_key_value_heads is {kv!r}, not a positive integer.")
    else:
        config["numKeyValueHeads"] = kv
    head_dim = raw.get("head_dim")
    if head_dim is None:
        config["headDim"] = config["hiddenSize"] // heads
        notes.append(f"head_dim is absent, so it is hidden_size / num_attention_heads = "
                     f"{config['hiddenSize']} / {heads} = {config['headDim']}.")
    elif _positive_int(head_dim) is None:
        raise ModelConfigError("model-config-missing",
                               f"head_dim is {head_dim!r}, not a positive integer.")
    else:
        # Qwen3 states it, and it is NOT always hidden/heads: Qwen3-4B has hidden 2560 and 32
        # heads (80) but head_dim 128. Deriving it there would undercount KV memory by a third.
        config["headDim"] = head_dim
    # transformers wrote `torch_dtype` until 4.56 and `dtype` after; read either.
    dtype = raw.get("torch_dtype") or raw.get("dtype") or ""
    config["dtype"] = str(dtype)
    if not dtype:
        notes.append("config.json names no dtype; vLLM will choose one when it loads the model.")
    if raw.get("tie_word_embeddings") is True:
        # The presets the planner was built on (Llama 3.1) have a separate lm_head; a model that
        # shares it with the embedding table holds vocab x hidden fewer parameters than counted.
        extra = config["vocabSize"] * config["hiddenSize"]
        notes.append(
            f"tie_word_embeddings is true: the output layer shares the embedding table, so a "
            f"cost model that counts a separate lm_head overestimates the weights by "
            f"{extra:,} parameters ({extra * 2 / 1e9:.2f} GB at 2 bytes each).")
    # The contract's own key order, so the block reads the way the CRD documents it.
    ordered = {k: config[k] for k in CONTRACT_KEYS}
    return DENSE_FAMILIES[architecture]["family"], ordered, notes


def check_given_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Check a `config` block a caller filled in themselves (e.g. an agent that read config.json).

    The same rules as `dense_config`, applied to the contract's own camelCase shape: every
    number a positive integer, the architecture one of the dense families.

    Returns:
        (family, the block with only the contract's keys).

    Raises:
        ModelConfigError: `model-config-architecture` or `model-config-missing`.
    """
    architecture = str(config.get("architecture") or "")
    if architecture not in DENSE_FAMILIES:
        raise ModelConfigError(
            "model-config-architecture",
            f"config.architecture {architecture or 'missing'!r} is not one of "
            f"{', '.join(DENSE_FAMILIES)}.")
    keys = CONTRACT_KEYS[1:-1]
    missing = [k for k in keys if _positive_int(config.get(k)) is None]
    if missing:
        raise ModelConfigError(
            "model-config-missing",
            f"config has no positive integer for {', '.join(missing)}. POST /v1/model-config "
            f"fills every one from the model's config.json.")
    clean = {"architecture": architecture, **{k: config[k] for k in keys},
             "dtype": str(config.get("dtype") or "")}
    return DENSE_FAMILIES[architecture]["family"], clean


def lookup(repo: str | None, config_url: str | None, revision: str | None,
           token: str | None = None, fetch: Callable[..., Fetched] | None = None) -> ModelConfig:
    """Steps 2 to 4 of the module docstring, for one model.

    Args:
        repo, config_url, revision: as `source_for` takes them.
        token: the HF token value, or None.
        fetch: replaces `fetch_url`, for tests. Looked up at call time so a test may also
            monkeypatch the module attribute.

    Returns:
        A `ModelConfig`.

    Raises:
        ModelConfigError: any refusal on the way.
    """
    source = source_for(repo, config_url, revision)
    fetched = (fetch or fetch_url)(source.url, token)
    if fetched.status != 200:
        raise _hub_refusal(fetched, source, token_sent=bool(token))
    if len(fetched.body) > FETCH_MAX_BYTES:
        raise ModelConfigError("model-config-url",
                               f"{source.url} is larger than {FETCH_MAX_BYTES} bytes; a "
                               f"config.json is a few kilobytes.")
    try:
        raw = json.loads(fetched.body)
    except ValueError as exc:
        raise ModelConfigError("model-config-missing",
                               f"{source.url} is not JSON, so it is not a config.json.") from exc
    family, config, notes = dense_config(raw)
    return ModelConfig(repo=source.repo, revision=source.revision, source=source.url,
                       family=family, config=config, notes=notes)


def guidance() -> dict[str, Any]:
    """The supported-model guidance `GET /v1/serving-models` returns. Static, no cluster call."""
    return {
        "presets": [
            {"preset": name, "name": p["name"], "gated": p["gated"], "why": p["why"],
             "config": p["config"]}
            for name, p in PRESETS.items()
        ],
        "families": [
            {"architecture": arch, "family": v["family"], "example": v["example"]}
            for arch, v in DENSE_FAMILIES.items()
        ],
        "recommended": RECOMMENDED_PRESET,
        "recommendation": (
            f"{RECOMMENDED_PRESET}, unless you already know which model you need: "
            f"{PRESETS[RECOMMENDED_PRESET]['why']}. It is gated, so its weights need a Hugging "
            f"Face token with the Llama 3.1 licence accepted (`hyperun secret-set HF_TOKEN`). "
            f"With no token, Qwen/Qwen3-8B is a dense model of the same size that is not gated."),
        "refused": (
            "mixture-of-experts models (config.json carries any of "
            + ", ".join(MOE_KEYS) + "), and any architecture outside the families above."),
    }
