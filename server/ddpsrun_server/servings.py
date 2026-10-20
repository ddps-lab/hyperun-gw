"""A model served behind an endpoint: what a caller sends, what they get back, and the object.

HYPERUN-SERVING. A HyperunServing (PACSrun's operator, contract of 2026-10-20) asks for one model
to be served within an hourly budget. The operator ranks serving plans with the same ShuntServe
planner a serving JOB uses, rents the chosen plan, starts vLLM on it, and writes where vLLM
listens into `status.endpoint.upstream`. This gateway then serves it at
`/v1/servings/<ns>/<name>/v1`, an OpenAI-compatible path guarded by the serving's own API key
(proxy.py, HYPERUN-SERVING-PROXY).

END-TO-END FLOW of one `POST /v1/servings` (main.create_serving):

  1. `ServingRequest` is the body: a name, the model (a preset, an HF id, a config.json URL, or
     a config block the caller filled), the two average token counts with ShuntServe's
     defaults, the hourly budget, and optionally a total spend cap, an image and the NAME of a
     registered HF token.
  2. `request_problems()` names every value the request model let through but the operator
     would refuse, so the caller hears all of them at once, before anything is rented.
  3. `resolve_model()` turns the model ask into the contract's `spec.model`: a preset's numbers
     from modelconfig.PRESETS, a caller's config block checked, or config.json fetched and
     mapped (modelconfig.lookup).
  4. `to_hyperunserving()` builds the object: name and namespace, the owner label the owner gate
     reads, and the spec with the placement a serving needs filled in (on-demand, [shadeform]).
  5. `ServingView.from_object()` goes the other way for every read: plans, the chosen plan with
     its stages, the catalog, the endpoint path, and the billing sentence.

WHAT NEVER CROSSES THE API: `status.endpoint.upstream` (an address inside our account or on a
rented machine) and `status.endpoint.apiKeySecret` (an object name). The endpoint PATH crosses,
and with it the namespace, which is in the URL a client has to call; that is the same exception
`JobView.result_path` makes, for the same reason.

Grep anchor: HYPERUN-SERVING
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from . import modelconfig, naming
from .auth import Principal
from .config import SERVING_GROUP, SERVING_KIND, SERVING_VERSION, USER_SECRET_NAME
from .models import (
    SECRET_NAME_PATTERN,
    SERVING_CAPACITY,
    SERVING_VENDOR,
    ServingPlanView,
    _text,
    _whole,
    decimal_text,
    serving_plans_from,
)

# ShuntServe's own workload (solver/shuntserve/plan.py, WORKLOAD: 763 tokens in, 232 out). Used
# when the caller leaves the two averages out, and shown as the default on the form and in help.
DEFAULT_AVG_INPUT_TOKENS = 763
DEFAULT_AVG_OUTPUT_TOKENS = 232

# A serving name is part of a URL path, of the Secret `<name>-api-key`, and of whatever objects
# the operator names after it. A DNS-1035 label (lowercase, starts with a letter) is what a
# Service name must be, and 40 characters leaves room for the operator's suffixes inside the
# 63-character label limit.
SERVING_NAME = re.compile(r"^[a-z]([-a-z0-9]{0,38}[a-z0-9])?$")

# What a registered secret's NAME may be: the rule PUT /v1/secrets/{name} applies, so a name that
# could never have been registered is refused before the cluster is asked.
VARIABLE_NAME = re.compile(SECRET_NAME_PATTERN)


def endpoint_path(namespace: str, name: str) -> str:
    """The OpenAI base path a client calls, e.g. /v1/servings/lab-alice/chat-8b/v1.

    The gateway serves this route itself (main.serving_proxy), so it knows the path before the
    operator writes `status.endpoint.path`; the two are the same by contract.
    """
    return f"/v1/servings/{namespace}/{name}/v1"


# ---------------------------------------------------------------------------------------------
# 1. The request.
# ---------------------------------------------------------------------------------------------

class ModelConfigBody(BaseModel):
    """The contract's `spec.model.config`, in this API's snake_case.

    POST /v1/model-config answers exactly this shape (`config`), so a caller can pass its answer
    straight back here.
    """

    architecture: str = Field(description="config.json's architectures[0], e.g. "
                              "LlamaForCausalLM. One of the dense families GET "
                              "/v1/serving-models lists.")
    hidden_size: int
    num_layers: int = Field(description="config.json's num_hidden_layers.")
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    dtype: str = Field(default="", description="config.json's torch_dtype, e.g. bfloat16.")

    def as_contract(self) -> dict[str, Any]:
        """The camelCase block the CRD carries."""
        return {"architecture": self.architecture, "hiddenSize": self.hidden_size,
                "numLayers": self.num_layers, "numAttentionHeads": self.num_attention_heads,
                "numKeyValueHeads": self.num_key_value_heads, "headDim": self.head_dim,
                "intermediateSize": self.intermediate_size, "vocabSize": self.vocab_size,
                "dtype": self.dtype}


def config_view(contract: dict[str, Any]) -> dict[str, Any]:
    """The contract's camelCase config back into this API's snake_case."""
    return {"architecture": contract.get("architecture", ""),
            "hidden_size": contract.get("hiddenSize"), "num_layers": contract.get("numLayers"),
            "num_attention_heads": contract.get("numAttentionHeads"),
            "num_key_value_heads": contract.get("numKeyValueHeads"),
            "head_dim": contract.get("headDim"),
            "intermediate_size": contract.get("intermediateSize"),
            "vocab_size": contract.get("vocabSize"), "dtype": contract.get("dtype", "")}


class ServingModelRequest(BaseModel):
    """Which model. One of: `preset`; `name` (an HF id); `config_url`; or `name` with `config`."""

    preset: str | None = Field(
        default=None,
        description=f"A preset the planner carries: {', '.join(modelconfig.PRESETS)}. Fills "
        "the HF id and the config itself.")
    name: str | None = Field(
        default=None,
        description="The Hugging Face repository id vLLM loads, e.g. Qwen/Qwen3-8B. Without "
        "`config`, the server reads its config.json (POST /v1/model-config shows what it "
        "would find).")
    revision: str | None = Field(default=None, description="A branch, tag or commit of `name`.")
    config_url: str | None = Field(
        default=None,
        description="A URL to a config.json, instead of reading it by `name`. A huggingface.co "
        "URL also gives the id and the revision; any other URL needs `name` as well, because "
        "vLLM loads weights by id.")
    config: ModelConfigBody | None = Field(
        default=None,
        description="The model's structure, filled by the caller (e.g. from POST "
        "/v1/model-config, or by reading config.json yourself). Needs `name`.")


class ServingRequest(BaseModel):
    """The body of `POST /v1/servings`."""

    # Unknown fields are dropped, as on POST /v1/jobs (models.SubmitRequest says why).
    model_config = ConfigDict(extra="ignore")

    name: str = Field(
        description="The serving's name: lowercase letters, digits and '-', starting with a "
        "letter, at most 40 characters. It is part of the endpoint URL "
        "/v1/servings/<namespace>/<name>/v1, so it must be unique in your namespace.")
    model: ServingModelRequest
    avg_input_tokens: int = Field(
        default=DEFAULT_AVG_INPUT_TOKENS,
        description=f"Average prompt length of the traffic, in tokens. Default "
        f"{DEFAULT_AVG_INPUT_TOKENS}, ShuntServe's own workload.")
    avg_output_tokens: int = Field(
        default=DEFAULT_AVG_OUTPUT_TOKENS,
        description=f"Average answer length of the traffic, in tokens. Default "
        f"{DEFAULT_AVG_OUTPUT_TOKENS}.")
    hourly_budget: float | None = Field(
        default=None,
        description="REQUIRED. The most the served plan may cost, US dollars PER HOUR for all "
        "its machines together, e.g. 3.00. The planner ranks plans by requests per second per "
        "dollar-hour among those within it. The serving bills by the hour until it is deleted.")
    max_spend_usd: float | None = Field(
        default=None,
        description="Optional. Total US dollars after which the operator stops the serving.")
    image: str | None = Field(default=None, description="The vLLM image. Leave it out for the "
                              "operator's default.")
    hf_token_secret: str | None = Field(
        default=None,
        description="The NAME of a value you registered with `hyperun secret-set` (e.g. "
        "HF_TOKEN) holding a Hugging Face token, for a gated model. The value never travels "
        "through this API: the server reads config.json with it, and the operator hands it to "
        "vLLM to download the weights.")


def request_problems(body: ServingRequest, registered: Callable[[], dict[str, Any]]) -> list[str]:
    """Every reason the operator would refuse this request, each naming the field.

    Args:
        body: the request.
        registered: returns the names this namespace registered (`Cluster.user_secrets`); only
            called when `hf_token_secret` is set, so a request without one makes no cluster call.

    Returns:
        Sentences, empty when the request may be created.
    """
    problems = []
    if not SERVING_NAME.fullmatch(body.name or ""):
        problems.append(
            f"name {body.name!r} must be lowercase letters, digits and '-', start with a letter, "
            f"end with a letter or digit, and be at most 40 characters: it is part of the "
            f"endpoint URL and of the Secret <name>-api-key.")
    budget = body.hourly_budget
    if budget is None or not math.isfinite(budget) or budget <= 0:
        problems.append(
            "hourly_budget is " + ("missing" if budget is None else f"{budget}")
            + ". It is required: the most the plan may cost in US dollars PER HOUR for all its "
            "machines together, e.g. 3.00.")
    spend = body.max_spend_usd
    if spend is not None and (not math.isfinite(spend) or spend <= 0):
        problems.append(f"max_spend_usd is {spend}; send a number above 0 or leave it out.")
    for field_name in ("avg_input_tokens", "avg_output_tokens"):
        if getattr(body, field_name) < 1:
            problems.append(f"{field_name} is {getattr(body, field_name)}; it is an average "
                            f"length in tokens and must be at least 1.")
    m = body.model
    asks = [k for k in ("preset", "config_url") if getattr(m, k)]
    if m.name and not m.config_url:
        asks.append("name")
    if len(asks) != 1:
        problems.append(
            "model needs exactly one of preset, name (an HF id, optionally with config) and "
            f"config_url; this request has {', '.join(asks) or 'none'}. GET /v1/serving-models "
            f"lists the presets and the supported families.")
    if m.preset and m.preset not in modelconfig.PRESETS:
        problems.append(f"model.preset {m.preset!r} is not one of "
                        f"{', '.join(modelconfig.PRESETS)}.")
    if m.config is not None and not m.name:
        problems.append("model.config needs model.name: vLLM loads weights by HF id.")
    secret = body.hf_token_secret
    if secret:
        if not VARIABLE_NAME.fullmatch(secret):
            problems.append(f"hf_token_secret {secret!r} is not a registered name; names look "
                            f"like HF_TOKEN.")
        elif secret not in registered():
            problems.append(f"hf_token_secret {secret} is not registered in your namespace. "
                            f"Register it with `hyperun secret-set {secret}`.")
    return problems


def resolve_model(body: ServingRequest, token: str | None,
                  fetch: Callable[..., Any] | None = None) -> dict[str, Any]:
    """The contract's `spec.model` for this request. Step 3 of the module docstring.

    Args:
        body: a request `request_problems` found nothing wrong with.
        token: the HF token VALUE for a fetch, or None.
        fetch: passed to modelconfig.lookup, for tests.

    Returns:
        {name, revision?, preset?, config, configSource}.

    Raises:
        modelconfig.ModelConfigError: the model was refused or config.json could not be read.
    """
    m = body.model
    if m.preset:
        preset = modelconfig.PRESETS[m.preset]
        spec = {"name": m.name or preset["name"], "preset": m.preset,
                "config": dict(preset["config"]), "configSource": f"preset:{m.preset}"}
    elif m.config is not None:
        _, checked = modelconfig.check_given_config(m.config.as_contract())
        spec = {"name": m.name, "config": checked,
                "configSource": m.config_url or "caller"}
    else:
        found = modelconfig.lookup(m.name if not m.config_url else None, m.config_url,
                                   m.revision, token=token, fetch=fetch)
        name = m.name or found.repo
        if not name:
            raise modelconfig.ModelConfigError(
                "model-config-ask",
                f"{found.source} is not a huggingface.co URL, so it does not say which weights "
                f"vLLM should load. Send model.name (the HF id) with it.")
        spec = {"name": name, "config": found.config, "configSource": found.source}
        if found.revision and not m.revision:
            spec["revision"] = found.revision
    if m.revision:
        spec["revision"] = m.revision
    return spec


def to_hyperunserving(body: ServingRequest, principal: Principal,
                      model_spec: dict[str, Any]) -> dict[str, Any]:
    """Step 4: the HyperunServing object to create in the caller's namespace.

    WHAT THE SERVER FILLS AND THE CALLER CANNOT: the namespace (from the token), the owner label
    (HYPERUN-OWNER-GATE reads it, so a namespace-mate cannot read this serving's API key), and
    the placement. A serving's plans are Shadeform on-demand machines, as a serving job's are
    (models.SERVING_VENDOR / SERVING_CAPACITY), so nothing about placement is offered.
    """
    spec: dict[str, Any] = {
        "model": model_spec,
        "workload": {"avgInputTokens": body.avg_input_tokens,
                     "avgOutputTokens": body.avg_output_tokens},
        # Decimal strings: PACSrun's API carries no floats (models.decimal_text says why).
        "hourlyBudget": decimal_text(body.hourly_budget),
        "placement": {"regions": [SERVING_VENDOR], "capacityType": SERVING_CAPACITY},
    }
    if body.max_spend_usd is not None:
        spec["maxSpendUSD"] = decimal_text(body.max_spend_usd)
    if body.image:
        spec["image"] = body.image
    if body.hf_token_secret:
        # ★ CONTRACT ASSUMPTION (2026-10-20): a secretKeyRef-shaped object, the way PACSrun's
        # own CRDs reference a Secret (api/v1alpha1/shared_types.go, env valueFrom.secretKeyRef).
        # The registered value lives in the namespace's one hyperun-user-secrets Secret under
        # its own name, so both halves are needed. If the operator's field is a bare string,
        # the apiserver refuses the create with a type error rather than accepting it wrongly.
        spec["hfTokenSecret"] = {"name": USER_SECRET_NAME, "key": body.hf_token_secret}
    labels = {naming.OWNER_LABEL: principal.user}
    return {
        "apiVersion": f"{SERVING_GROUP}/{SERVING_VERSION}",
        "kind": SERVING_KIND,
        "metadata": {"name": body.name, "namespace": principal.namespace,
                     "labels": {k: v for k, v in labels.items() if v}},
        "spec": spec,
    }


# ---------------------------------------------------------------------------------------------
# 5. The view.
# ---------------------------------------------------------------------------------------------

class ServingStageChosenView(BaseModel):
    """One machine of the chosen plan: where it is rented and which layers it runs."""

    cloud: str = Field(default="", description="The provider behind Shadeform, e.g. hyperstack.")
    region: str = Field(default="", description="The Shadeform region.")
    instance_type: str = Field(default="", description="The Shadeform instance type, e.g. "
                               "A100_80G.")
    layers: int = Field(default=0, description="How many consecutive model layers it holds.")
    tp: int = Field(default=0, description="Tensor-parallel size: the machine's GPU count.")


class ServingChosenView(BaseModel):
    """status.servingPlanChosen of a serving: the plan it rents, with every machine of it."""

    rank: int = Field(description="Its rank in `serving_plans`.")
    score: str | None = Field(default=None, description="Requests per second per "
                              "dollar-hour, why it was chosen over the later ranks.")
    cost_per_hour: str | None = Field(default=None, description="US dollars per hour, every "
                                      "machine together.")
    throughput_rps: str | None = Field(default=None, description="The planner's estimate.")
    latency_seconds: str | None = Field(default=None, description="The planner's estimate.")
    attempt: int = Field(default=0, description="How many plans this serving has chosen, this "
                         "one included.")
    stages: list[ServingStageChosenView] = Field(default_factory=list)


class ServingEndpointView(BaseModel):
    """Where to send OpenAI requests, and whether anything answers there yet."""

    path: str = Field(description="Append to the gateway's address: <server><path> is the "
                      "OpenAI base_url. Authenticate with the serving's API key as "
                      "`Authorization: Bearer`, read once from GET /v1/servings/{name}/api-key.")
    ready: bool = Field(default=False, description="True when the operator says vLLM answers. "
                        "Until then the path answers 503.")


class ServingView(BaseModel):
    """What `GET /v1/servings/{name}` returns."""

    name: str
    namespace: str = Field(description="Part of the endpoint path, which is why it crosses "
                           "the API here.")
    user: str = Field(default="", description="Who created it.")
    phase: str = Field(default="", description="The operator's phase word, verbatim. Empty "
                       "until it has looked at the serving once.")
    message: str = Field(default="")
    created_at: str | None = None
    model: dict[str, Any] = Field(default_factory=dict, description="name, revision, preset, "
                                  "config_source and config (snake_case), from spec.model.")
    avg_input_tokens: int | None = None
    avg_output_tokens: int | None = None
    hourly_budget: str | None = Field(default=None, description="US dollars per hour, as sent.")
    max_spend_usd: str | None = None
    image: str | None = None
    serving_plans: list[ServingPlanView] = Field(
        default_factory=list,
        description="The ranked plans, highest score (requests per second per dollar-hour) "
        "first, every one within the hourly budget. Left OUT of GET /v1/servings.")
    serving_plan_chosen: ServingChosenView | None = None
    serving_catalog: str = Field(default="", description="Which Shadeform catalog the plans "
                                 "were ranked on, and when it was fetched.")
    endpoint: ServingEndpointView
    machines: list[dict[str, Any]] = Field(
        default_factory=list,
        description="status.machines as the operator writes it, passed through unchanged.")
    billing: str = Field(default="", description="What this serving costs while it exists, in "
                         "one sentence.")

    @staticmethod
    def from_object(obj: dict[str, Any]) -> "ServingView":
        """Build the view from a HyperunServing as the apiserver returned it."""
        metadata = obj.get("metadata") or {}
        spec = obj.get("spec") or {}
        status = obj.get("status") or {}
        name = str(metadata.get("name") or "")
        namespace = str(metadata.get("namespace") or "")
        model = spec.get("model") or {}
        workload = spec.get("workload") or {}
        endpoint = status.get("endpoint") or {}
        chosen = _chosen_from(status.get("servingPlanChosen"))
        budget = _text(spec.get("hourlyBudget"))
        return ServingView(
            name=name, namespace=namespace,
            user=str((metadata.get("labels") or {}).get(naming.OWNER_LABEL, "")),
            phase=str(status.get("phase") or ""),
            message=str(status.get("message") or ""),
            created_at=metadata.get("creationTimestamp"),
            model={"name": model.get("name") or "", "revision": model.get("revision") or "",
                   "preset": model.get("preset") or "",
                   "config_source": model.get("configSource") or "",
                   "config": config_view(model.get("config") or {})},
            avg_input_tokens=workload.get("avgInputTokens"),
            avg_output_tokens=workload.get("avgOutputTokens"),
            hourly_budget=budget,
            max_spend_usd=_text(spec.get("maxSpendUSD")),
            image=spec.get("image"),
            serving_plans=serving_plans_from(status),
            serving_plan_chosen=chosen,
            serving_catalog=str(status.get("servingCatalog") or ""),
            # The gateway's own path, not status.endpoint.path, and never upstream or
            # apiKeySecret (the module docstring says why).
            endpoint=ServingEndpointView(path=endpoint_path(namespace, name),
                                         ready=endpoint.get("ready") is True),
            machines=[m for m in status.get("machines") or [] if isinstance(m, dict)],
            billing=billing_sentence(budget, chosen, _text(spec.get("maxSpendUSD"))),
        )


def _chosen_from(chosen: Any) -> ServingChosenView | None:
    """status.servingPlanChosen of a serving, or None before a plan is chosen."""
    if not isinstance(chosen, dict) or not chosen:
        return None
    return ServingChosenView(
        rank=_whole(chosen.get("rank")), score=_text(chosen.get("score")),
        cost_per_hour=_text(chosen.get("costPerHour")),
        throughput_rps=_text(chosen.get("throughputRPS")),
        latency_seconds=_text(chosen.get("latencySeconds")),
        attempt=_whole(chosen.get("attempt")),
        stages=[ServingStageChosenView(
            cloud=str(s.get("cloud") or ""), region=str(s.get("region") or ""),
            instance_type=str(s.get("instanceType") or ""),
            layers=_whole(s.get("layers")), tp=_whole(s.get("tp")))
            for s in chosen.get("stages") or [] if isinstance(s, dict)],
    )


def billing_sentence(budget: str | None, chosen: ServingChosenView | None,
                     max_spend: str | None) -> str:
    """What the serving costs while it exists. Every figure is a product of two numbers shown.

    The point of saying it on every read: a serving does not end by itself the way a job does.
    It bills by the hour until somebody deletes it, so the page that shows it must say how much
    an hour and how much a day, not leave the reader to remember.
    """
    cap = (f" The operator stops it at ${float(max_spend):,.2f} in total." if max_spend else
           " There is no total cap: it runs until you delete it.")
    try:
        if chosen and chosen.cost_per_hour:
            rate = float(chosen.cost_per_hour)
            return (f"Renting ${rate:,.2f}/h now (rank {chosen.rank}), ${rate * 24:,.2f} a day."
                    + cap)
        if budget:
            rate = float(budget)
            return (f"No plan rented yet. When one is, it costs at most ${rate:,.2f}/h, "
                    f"${rate * 24:,.2f} a day." + cap)
    except ValueError:
        pass
    return ""


class ServingListResponse(BaseModel):
    """What `GET /v1/servings` returns."""

    servings: list[ServingView]
    total: int


class ServingKeyResponse(BaseModel):
    """What `GET /v1/servings/{name}/api-key` returns. Shown once; never cached."""

    api_key: str = Field(description="Send as `Authorization: Bearer <api_key>` to the "
                         "endpoint path. Treat it like a password.")
    endpoint_path: str


class ModelConfigRequest(BaseModel):
    """The body of `POST /v1/model-config`."""

    model: str | None = Field(default=None, description="A Hugging Face repository id, e.g. "
                              "Qwen/Qwen3-8B.")
    revision: str | None = Field(default=None, description="A branch, tag or commit.")
    config_url: str | None = Field(default=None, description="A URL to a config.json, instead "
                                   "of `model`. https only, public addresses only.")
    hf_token_secret: str | None = Field(
        default=None,
        description="The NAME of a registered value holding a Hugging Face token (e.g. "
        "HF_TOKEN), for a gated repository. Sent to huggingface.co only.")


class ModelConfigResponse(BaseModel):
    """What `POST /v1/model-config` returns: the model's structure, or a 422 saying why not."""

    name: str = Field(description="The HF id, \"\" for a URL that is not huggingface.co.")
    revision: str = ""
    config_source: str = Field(description="The URL that was read.")
    family: str = Field(description="Which supported dense family it belongs to.")
    config: dict[str, Any] = Field(description="snake_case; pass it as model.config to POST "
                                   "/v1/servings.")
    spec_config: dict[str, Any] = Field(description="The same numbers in the camelCase shape "
                                        "HyperunServing's spec.model.config takes, for an "
                                        "agent writing the object itself.")
    notes: list[str] = Field(default_factory=list, description="What was derived rather than "
                             "read, and what the planner may get wrong for this model.")
