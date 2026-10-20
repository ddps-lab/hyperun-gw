"""The OpenAI-compatible endpoint of a serving: /v1/servings/{ns}/{name}/v1/* to its vLLM.

HYPERUN-SERVING-PROXY. A HyperunServing's operator starts vLLM on the rank 0 machine of the plan
it rented and writes where that vLLM listens into `status.endpoint.upstream`. This module is what
lets an OpenAI SDK reach it through the gateway's own address, with the serving's own API key and
without any hyperun login.

END-TO-END FLOW of one proxied request (main.serving_proxy calls `handle`):

  1. `bearer_of()` reads `Authorization: Bearer <key>`. None -> 401.
  2. `Lookups.serving()` fetches the HyperunServing (cached LOOKUP_TTL_SECONDS). Absent -> 404.
  3. `Lookups.key_digest()` reads the Secret `<name>-api-key`, key `api-key`, and keeps ONLY its
     SHA-256 (cached KEY_TTL_SECONDS). No Secret yet -> 503: the operator has not issued a key.
  4. `hmac.compare_digest` of the two digests. Different -> 401.
  5. `status.endpoint.ready` must be true and `upstream_for(status.endpoint.upstream)` must
     give an `Upstream`. Otherwise 503, naming the phase.
  6. The request body is read up to MAX_REQUEST_BYTES. More -> 413.
  7. `Upstream.open()` sends it to `<upstream>/v1/<path>` and hands back the status, the headers
     and an async iterator of bytes. The route returns them as a StreamingResponse, so an SSE
     stream (`"stream": true`) reaches the caller chunk by chunk as vLLM writes it. A response
     past MAX_RESPONSE_BYTES is cut there.
  8. One log line: namespace, name, method, path, status, bytes, seconds. NEVER the key, never a
     header, never a byte of the prompt or the answer.

THE ONE INTERFACE THE UPSTREAM KIND PLUGS INTO. `Upstream` has one method, `open`. How the
gateway reaches rank 0's vLLM is being decided on the operator side; whatever it is, it becomes
one class with `open` and one more branch in `upstream_for`. `HttpUpstream` covers anything
reachable by a plain HTTP URL from this pod: a Service DNS name, a ClusterIP, a machine's
address. The tests plug in a double through the same interface.

WHAT `status.endpoint.upstream` MUST BE for `HttpUpstream`: either a string "http://host:port"
or an object {"url": "http://host:port"}, the ROOT of the vLLM server (the gateway appends
/v1/<path>; a trailing /v1 is tolerated and not doubled). Anything else answers 503 "this gateway
does not know the upstream kind".

Grep anchor: HYPERUN-SERVING-PROXY
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Mapping, Protocol

import httpx
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .k8s import ClusterError, NotFound

logger = logging.getLogger("hyperun.proxy")

# A chat request with an 8192-token context is about 32 KB of text, and vLLM's own default
# context for the presets is 8192 (solver/shuntserve/plan.py, WORKLOAD max_model_len). 2 MiB is
# sixty times that and still small enough that a client cannot park megabytes in this pod's
# 512 MiB limit (config/deploy/hyperun-gw.yaml) per connection.
MAX_REQUEST_BYTES = 2 * 1024 * 1024
# An answer of 8192 tokens streamed as SSE is roughly 8192 events of 250 bytes, 2 MB. 32 MiB is
# sixteen times that; a response past it is cut, because a stream that never ends would
# otherwise hold this pod's connection and the ALB's for as long as the upstream keeps writing.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
# How long a looked-up serving and a key digest are reused. A request rate of 10 per second
# would otherwise be 20 apiserver reads per second for one serving. A deleted serving or a
# rotated key is seen at most this late.
LOOKUP_TTL_SECONDS = 5.0
KEY_TTL_SECONDS = 30.0
# vLLM can sit on a request until it is scheduled; a long answer can take minutes. The read
# timeout is the gap between two chunks, not the whole answer. The ALB's idle timeout is 300 s
# (config/deploy/hyperun-gateway.yaml), so a gap longer than this would be cut there anyway.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=290.0, write=60.0, pool=10.0)

# What goes to the upstream: the body's type, what the client accepts, and the bearer, which
# IS this serving's key (checked above) so a vLLM started with --api-key accepts it too.
# Nothing else: no cookie, no X-Forwarded-*, no Accept-Encoding (so the bytes come back as vLLM
# wrote them and can be passed on without decoding).
REQUEST_HEADERS = ("content-type", "accept", "authorization")
# What comes back. content-length is dropped because the response is re-chunked.
RESPONSE_HEADERS = ("content-type", "cache-control", "x-request-id")


# ---------------------------------------------------------------------------------------------
# The interface.
# ---------------------------------------------------------------------------------------------

@dataclass
class UpstreamRequest:
    """One request on its way to the upstream, already authenticated and size-checked."""

    method: str
    path: str                  # after /v1/, e.g. "chat/completions"
    query: str                 # the raw query string, "" for none
    headers: dict[str, str]    # only REQUEST_HEADERS
    body: bytes


class UpstreamResponse(Protocol):
    """What `Upstream.open` hands back: headers now, the body as it arrives."""

    status_code: int
    headers: Mapping[str, str]

    def aiter_bytes(self) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...


class Upstream(Protocol):
    """HYPERUN-SERVING-UPSTREAM. The one thing a new way of reaching vLLM has to implement."""

    async def open(self, request: UpstreamRequest) -> UpstreamResponse: ...


class _HttpxResponse:
    """An httpx streaming response, seen through `UpstreamResponse`."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.status_code = response.status_code
        self.headers = response.headers

    def aiter_bytes(self) -> AsyncIterator[bytes]:
        # aiter_raw and not aiter_bytes: no decoding and no re-chunking, so what vLLM flushed as
        # one SSE event leaves here as one chunk.
        return self._response.aiter_raw()

    async def aclose(self) -> None:
        await self._response.aclose()


_client: httpx.AsyncClient | None = None


def _shared_client() -> httpx.AsyncClient:
    """One connection pool for every upstream, made on first use inside the event loop."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT, follow_redirects=False)
    return _client


class HttpUpstream:
    """vLLM at a plain HTTP(S) URL this pod can reach."""

    def __init__(self, root: str, client: httpx.AsyncClient | None = None) -> None:
        """
        Args:
            root: the vLLM server's root URL.
            client: an httpx client to send with; None is the process-wide pool. Tests pass one
                built on httpx.MockTransport.
        """
        root = root.rstrip("/")
        # status.endpoint.upstream may name the server root or its /v1; both mean the same vLLM.
        self.root = root[:-3] if root.endswith("/v1") else root
        self._client = client

    async def open(self, request: UpstreamRequest) -> UpstreamResponse:
        url = f"{self.root}/v1/{request.path}" + (f"?{request.query}" if request.query else "")
        client = self._client or _shared_client()
        sent = client.build_request(request.method, url, headers=request.headers,
                                    content=request.body)
        return _HttpxResponse(await client.send(sent, stream=True))


def upstream_for(value: Any) -> Upstream | None:
    """status.endpoint.upstream -> the `Upstream` that reaches it, or None for a kind this
    gateway does not know. THE ONE PLACE a new upstream kind is added."""
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        return HttpUpstream(value)
    if isinstance(value, dict) and isinstance(value.get("url"), str) \
            and value["url"].startswith(("http://", "https://")):
        return HttpUpstream(value["url"])
    return None


# ---------------------------------------------------------------------------------------------
# Lookups, cached.
# ---------------------------------------------------------------------------------------------

def digest(key: str) -> bytes:
    """SHA-256 of a key. What is cached and compared; the key itself is not kept."""
    return hashlib.sha256(key.encode("utf-8")).digest()


class Lookups:
    """The serving object and its key digest, per (namespace, name), each with its own TTL.

    Built once per process in main.build_state. Keeping a digest and not the key means a heap
    dump of this pod holds no serving's key.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._servings: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}
        self._keys: dict[tuple[str, str], tuple[float, bytes]] = {}

    async def serving(self, cluster: Any, namespace: str, name: str) -> dict[str, Any]:
        """The HyperunServing. Raises NotFound or ClusterError from the cluster."""
        hit = self._servings.get((namespace, name))
        if hit and self._clock() - hit[0] < LOOKUP_TTL_SECONDS:
            return hit[1]
        # The kubernetes client is synchronous; in the event loop it would stall every other
        # request this pod is streaming.
        obj = await run_in_threadpool(cluster.get_serving, namespace, name)
        self._servings[(namespace, name)] = (self._clock(), obj)
        return obj

    async def key_digest(self, cluster: Any, namespace: str, name: str) -> bytes:
        """SHA-256 of the serving's API key. Raises NotFound before the operator issued one."""
        hit = self._keys.get((namespace, name))
        if hit and self._clock() - hit[0] < KEY_TTL_SECONDS:
            return hit[1]
        value = digest(await run_in_threadpool(cluster.serving_api_key, namespace, name))
        self._keys[(namespace, name)] = (self._clock(), value)
        return value


# ---------------------------------------------------------------------------------------------
# The request.
# ---------------------------------------------------------------------------------------------

def error(status: int, code: str, message: str, **extra: str) -> JSONResponse:
    """An error in the shape OpenAI clients read: {"error": {"message", "type", "code"}}.

    The callers of this path are OpenAI SDKs, which surface `error.message` and ignore FastAPI's
    usual `detail`; answering in their shape is what makes the reason reach the person.
    """
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"message": message, "type": code, "code": code,
                                           **extra}})


def bearer_of(request: Request) -> str:
    """The bearer the caller sent, or "" for none. Never logged."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else ""


async def read_capped(request: Request, limit: int | None = None) -> bytes | None:
    """The request body, or None when it is larger than `limit` (MAX_REQUEST_BYTES when None).

    Content-Length is checked first so a declared 100 MB body is refused before a byte is read;
    the stream is still counted, because chunked uploads declare nothing. The limit is read at
    call time, not as a default argument, which Python would freeze at import.
    """
    limit = MAX_REQUEST_BYTES if limit is None else limit
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        return None
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            return None
    return bytes(body)


async def handle(request: Request, cluster: Any, lookups: Lookups, namespace: str, name: str,
                 path: str) -> Response:
    """Steps 1 to 8 of the module docstring for one request."""
    started = time.monotonic()
    where = f"{namespace}/{name}"
    presented = bearer_of(request)
    if not presented:
        return error(401, "missing_api_key",
                     "Send this serving's API key as `Authorization: Bearer <key>`. Its owner "
                     "reads it with `hyperun serving-key` or the Servings page.")
    try:
        obj = await lookups.serving(cluster, namespace, name)
    except NotFound:
        return error(404, "serving_not_found", f"There is no serving {where}.")
    except ClusterError as exc:
        return error(502, "cluster_error", f"Could not read serving {where}: {exc}")
    status = obj.get("status") or {}
    phase = str(status.get("phase") or "not started")
    try:
        stored = await lookups.key_digest(cluster, namespace, name)
    except NotFound:
        return error(503, "serving_not_ready",
                     f"Serving {where} has no API key yet (phase {phase}). The operator "
                     f"issues one when it starts the serving; try again shortly.", phase=phase)
    except ClusterError as exc:
        return error(502, "cluster_error", f"Could not read the key of {where}: {exc}")
    if not hmac.compare_digest(digest(presented), stored):
        return error(401, "invalid_api_key", f"That is not the API key of serving {where}.")

    endpoint = status.get("endpoint") or {}
    upstream = upstream_for(endpoint.get("upstream")) if endpoint.get("ready") is True else None
    if upstream is None:
        why = (f"is not ready yet (phase {phase})" if endpoint.get("ready") is not True else
               "names an upstream this gateway does not know how to reach")
        message = str(status.get("message") or "")
        return error(503, "serving_not_ready",
                     f"Serving {where} {why}." + (f" {message}" if message else ""), phase=phase)

    body = await read_capped(request)
    if body is None:
        return error(413, "request_too_large",
                     f"The request body is over {MAX_REQUEST_BYTES} bytes.")
    sent = UpstreamRequest(
        method=request.method, path=path, query=request.url.query,
        headers={k: v for k, v in request.headers.items() if k.lower() in REQUEST_HEADERS},
        body=body)
    try:
        response = await upstream.open(sent)
    except httpx.HTTPError as exc:
        # The exception's text names the URL and the failure kind; the URL is an internal
        # address, so only the kind goes back.
        logger.warning("proxy %s %s %s -> upstream failed: %s", where, request.method, path,
                       type(exc).__name__)
        return error(502, "upstream_unreachable",
                     f"Serving {where} did not answer ({type(exc).__name__}).")

    headers = {k: v for k, v in response.headers.items() if k.lower() in RESPONSE_HEADERS}
    if "text/event-stream" in response.headers.get("content-type", ""):
        # Tell any proxy between here and the caller not to hold the stream back. The ALB does
        # not buffer; this is for whatever the caller's side has.
        headers["X-Accel-Buffering"] = "no"
        headers.setdefault("Cache-Control", "no-cache")

    async def relay() -> AsyncIterator[bytes]:
        sent_bytes = 0
        try:
            async for chunk in response.aiter_bytes():
                if sent_bytes + len(chunk) > MAX_RESPONSE_BYTES:
                    logger.warning("proxy %s %s %s cut at %d bytes", where, request.method,
                                   path, sent_bytes)
                    break
                sent_bytes += len(chunk)
                yield chunk
        finally:
            await response.aclose()
            logger.info("proxy %s %s %s -> %d, %d bytes, %.2f s", where, request.method, path,
                        response.status_code, sent_bytes, time.monotonic() - started)

    return StreamingResponse(relay(), status_code=response.status_code, headers=headers)
