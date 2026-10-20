"""The serving endpoint: /v1/servings/{ns}/{name}/v1/* through the gateway to the serving's vLLM.

HYPERUN-SERVING-PROXY. WHAT IT PINS, with a test double behind the one `Upstream` interface:
  1. The key is required (401), a wrong key is 401, a hyperun login is NOT a key (401), an
     unknown serving is 404, and a serving with no key yet or not ready is 503 naming its phase.
  2. A streamed answer reaches the caller as the SAME chunks the upstream produced, one ASGI body
     message each, checked below the test client (which would join them).
  3. Neither the key, a wrong key, nor any byte of the prompt or the answer is in any log record.
  4. A request body over the cap is 413 and never reaches the upstream; an answer over its cap
     is cut; a ".." path segment never leaves vLLM's /v1.
  5. The serving and the key digest are cached, so ten requests are not twenty apiserver reads.
  6. HttpUpstream composes <root>/v1/<path> and sends only the three allowed headers.
"""

import asyncio
import json
import logging

import httpx
import pytest

from test_routes import as_alice, client, cluster, main  # noqa: F401  (fixtures)

proxy = main.proxy

NS, NAME, KEY = "lab-alice", "chat-8b", "sk-serving-a1b2c3d4e5f6"
BASE = f"/v1/servings/{NS}/{NAME}/v1"
PROMPT_MARKER = "the-secret-prompt-7f3a"
ANSWER_MARKER = "the-private-answer-9c1d"
UPSTREAM = "http://upstream.test:8000"


def seed(cluster, ready=True, upstream=UPSTREAM, phase="Serving", key=KEY):
    cluster.servings[(NS, NAME)] = {
        "metadata": {"name": NAME, "namespace": NS},
        "status": {"phase": phase, "endpoint": {"ready": ready, "upstream": upstream}},
    }
    if key is not None:
        cluster.serving_keys[(NS, NAME)] = key


class FakeResponse:
    """What the double's `open` returns: the UpstreamResponse shape."""

    def __init__(self, chunks, status=200, content_type="text/event-stream"):
        self.status_code = status
        self.headers = {"content-type": content_type, "content-length": "999",
                        "x-request-id": "req-1", "set-cookie": "upstream=1"}
        self._chunks = chunks
        self.closed = False

    async def _gen(self):
        for chunk in self._chunks:
            await asyncio.sleep(0)
            yield chunk

    def aiter_bytes(self):
        return self._gen()

    async def aclose(self):
        self.closed = True


class FakeUpstream:
    """The test double behind `Upstream`: records what it was sent, answers with chunks."""

    def __init__(self, chunks, **kwargs):
        self.chunks = chunks
        self.kwargs = kwargs
        self.requests = []
        self.responses = []

    async def open(self, request):
        self.requests.append(request)
        response = FakeResponse(self.chunks, **self.kwargs)
        self.responses.append(response)
        return response


SSE = [b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
       ('data: {"choices":[{"delta":{"content":"' + ANSWER_MARKER + '"}}]}\n\n').encode(),
       b"data: [DONE]\n\n"]


@pytest.fixture
def upstream(monkeypatch):
    """Plug the double in where status.endpoint.upstream would be turned into an Upstream."""
    double = FakeUpstream(SSE)
    seen = []

    def upstream_for(value):
        seen.append(value)
        return double if value == UPSTREAM else None

    monkeypatch.setattr(proxy, "upstream_for", upstream_for)
    double.seen = seen
    return double


def chat(client, key=KEY, body=None, path="chat/completions", headers=None):
    payload = body if body is not None else {
        "model": "x", "stream": True,
        "messages": [{"role": "user", "content": PROMPT_MARKER}]}
    sent = {"Content-Type": "application/json", **(headers or {})}
    if key is not None:
        sent["Authorization"] = f"Bearer {key}"
    return client.post(f"{BASE}/{path}", content=json.dumps(payload), headers=sent)


# ------------------------------------------------------------------ 1. who may call


def test_no_key_is_401_in_the_shape_openai_clients_read(client, cluster, upstream):
    seed(cluster)
    response = chat(client, key=None)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "missing_api_key"
    assert upstream.requests == []


def test_a_wrong_key_is_401_and_reaches_nothing(client, cluster, upstream):
    seed(cluster)
    response = chat(client, key="sk-not-the-key")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert upstream.requests == []


def test_a_hyperun_login_is_not_the_serving_key(client, cluster, upstream):
    """The path takes the serving's key only; alice's own token opens her jobs, not this."""
    seed(cluster)
    assert chat(client, key="alice-token").status_code == 401


def test_an_unknown_serving_is_404(client, cluster, upstream):
    response = chat(client)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "serving_not_found"


def test_a_serving_with_no_key_yet_is_503_with_its_phase(client, cluster, upstream):
    seed(cluster, key=None, ready=False, phase="Planning")
    response = chat(client)
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "serving_not_ready" and error["phase"] == "Planning"
    assert "no API key yet" in error["message"]


def test_a_serving_that_is_not_ready_is_503_with_its_phase(client, cluster, upstream):
    seed(cluster, ready=False, phase="Renting")
    response = chat(client)
    assert response.status_code == 503
    assert response.json()["error"]["phase"] == "Renting"
    assert "not ready yet (phase Renting)" in response.json()["error"]["message"]
    assert upstream.requests == []


def test_an_upstream_kind_this_gateway_does_not_know_is_503(client, cluster, upstream):
    seed(cluster, upstream={"kind": "tailnet", "host": "rank0"})
    response = chat(client)
    assert response.status_code == 503
    assert "does not know how to reach" in response.json()["error"]["message"]


# ------------------------------------------------------------------ 2. streaming


def asgi_messages(method, path, headers, body=b""):
    """Run one request through the app at the ASGI level and return every message it sent.

    Below the test client on purpose: TestClient joins the body before handing it back, so it
    cannot tell one 300-byte chunk from three 100-byte ones. The app's state is the `client`
    fixture's (its lifespan built it), so this must run inside that fixture.
    """
    sent = []

    async def run():
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()     # never disconnect; cancelled when the app is done

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                 "method": method, "scheme": "http", "path": path,
                 "raw_path": path.encode(), "query_string": b"", "root_path": "",
                 "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                 "client": ("test", 1), "server": ("test", 80)}
        await main.app(scope, receive, send)

    asyncio.run(run())
    return sent


def test_a_streamed_answer_passes_through_as_the_same_chunks(client, cluster, upstream):
    seed(cluster)
    body = json.dumps({"stream": True, "messages": [{"content": PROMPT_MARKER}]}).encode()
    messages = asgi_messages("POST", f"{BASE}/chat/completions",
                             {"Authorization": f"Bearer {KEY}",
                              "Content-Type": "application/json"}, body)
    start = next(m for m in messages if m["type"] == "http.response.start")
    assert start["status"] == 200
    headers = {k.decode(): v.decode() for k, v in start["headers"]}
    assert headers["content-type"] == "text/event-stream"
    assert headers["x-accel-buffering"] == "no"
    # What the upstream sent, chunk for chunk: no joining, no splitting, no re-encoding.
    bodies = [m["body"] for m in messages if m["type"] == "http.response.body" and m["body"]]
    assert bodies == SSE
    # The upstream got the prompt as sent and the path after /v1/.
    request = upstream.requests[0]
    assert (request.method, request.path, request.body) == ("POST", "chat/completions", body)
    assert upstream.responses[0].closed


def test_only_the_allowed_headers_cross_in_either_direction(client, cluster, upstream):
    seed(cluster)
    response = chat(client, headers={"Cookie": "session=1", "X-Forwarded-For": "1.2.3.4",
                                     "Accept": "text/event-stream"})
    assert response.status_code == 200
    assert set(upstream.requests[0].headers) == {"content-type", "accept", "authorization"}
    # The bearer forwarded IS the serving's key, so a vLLM started with --api-key accepts it.
    assert upstream.requests[0].headers["authorization"] == f"Bearer {KEY}"
    assert "set-cookie" not in response.headers
    assert response.headers["x-request-id"] == "req-1"
    assert response.content == b"".join(SSE)


def test_a_non_streamed_error_from_vllm_keeps_its_status_and_body(client, cluster, monkeypatch):
    seed(cluster)
    double = FakeUpstream([b'{"error":{"message":"bad model"}}'], status=400,
                          content_type="application/json")
    monkeypatch.setattr(proxy, "upstream_for", lambda value: double)
    response = chat(client, body={"model": "nope"})
    assert response.status_code == 400
    assert response.json() == {"error": {"message": "bad model"}}


# ------------------------------------------------------------------ 3. nothing in the logs


def test_neither_key_nor_prompt_nor_answer_is_ever_logged(client, cluster, upstream, caplog):
    caplog.set_level(logging.DEBUG)
    seed(cluster)
    assert chat(client).status_code == 200
    assert chat(client, key="sk-wrong-key-55aa").status_code == 401
    cluster.servings.clear()
    main.app.state.serving_lookups = proxy.Lookups()
    assert chat(client).status_code == 404
    text = caplog.text + " ".join(str(r.args) for r in caplog.records)
    for secret in (KEY, "sk-wrong-key-55aa", PROMPT_MARKER, ANSWER_MARKER):
        assert secret not in text
    # And the line that IS written says what happened, so the log is still useful.
    assert any("proxy lab-alice/chat-8b POST chat/completions -> 200" in r.getMessage()
               for r in caplog.records)


# ------------------------------------------------------------------ 4. limits


def test_a_body_over_the_cap_is_413_and_never_reaches_the_upstream(client, cluster, upstream,
                                                                   monkeypatch):
    seed(cluster)
    monkeypatch.setattr(proxy, "MAX_REQUEST_BYTES", 1000)
    response = chat(client, body={"messages": [{"content": "x" * 2000}]})
    assert response.status_code == 413
    assert upstream.requests == []


def test_a_chunked_body_that_declares_no_length_is_counted_and_refused(client, cluster,
                                                                       upstream, monkeypatch):
    """Content-Length is only a declaration, and a chunked upload makes none; the body is
    counted as it arrives, so an undeclared 2000 bytes still meets the 1000-byte cap."""
    seed(cluster)
    monkeypatch.setattr(proxy, "MAX_REQUEST_BYTES", 1000)
    response = client.post(f"{BASE}/chat/completions",
                           content=(b"x" * 500 for _ in range(4)),
                           headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 413
    assert upstream.requests == []


def test_an_answer_over_the_cap_is_cut(client, cluster, monkeypatch):
    seed(cluster)
    double = FakeUpstream([b"a" * 400, b"b" * 400, b"c" * 400])
    monkeypatch.setattr(proxy, "upstream_for", lambda value: double)
    monkeypatch.setattr(proxy, "MAX_RESPONSE_BYTES", 1000)
    response = chat(client)
    assert response.content == b"a" * 400 + b"b" * 400
    assert double.responses[0].closed


@pytest.mark.parametrize("path", ["../metrics", "models/../../health", "chat//completions"])
def test_a_path_cannot_leave_vllms_v1(client, cluster, upstream, path):
    seed(cluster)
    response = client.get(f"{BASE}/{path}", headers={"Authorization": f"Bearer {KEY}"})
    assert response.status_code == 404
    assert upstream.requests == []


# ------------------------------------------------------------------ 5. the cache


def test_ten_requests_read_the_serving_and_the_key_once(client, cluster, upstream):
    seed(cluster)
    for _ in range(10):
        assert client.get(f"{BASE}/models",
                          headers={"Authorization": f"Bearer {KEY}"}).status_code == 200
    assert (cluster.serving_reads, cluster.key_reads) == (1, 1)
    assert [r.path for r in upstream.requests] == ["models"] * 10


def test_the_cache_keeps_a_digest_and_not_the_key(client, cluster, upstream):
    seed(cluster)
    chat(client)
    cached = main.app.state.serving_lookups._keys[(NS, NAME)][1]
    assert cached == proxy.digest(KEY) and KEY.encode() not in cached


# ------------------------------------------------------------------ 6. the real upstream


class Chunks(httpx.AsyncByteStream):
    """A response body that arrives in pieces, as a network one does. httpx reads a `content=`
    body eagerly, which a real response never is, so the mock answers with this instead."""

    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def test_http_upstream_appends_v1_and_the_path_once():
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url), dict(request.headers), request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=Chunks([b"data: 1\n\n", b"data: 2\n\n"]))

    async def run(root):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            response = await proxy.HttpUpstream(root, client=http).open(proxy.UpstreamRequest(
                method="POST", path="chat/completions", query="a=1",
                headers={"authorization": f"Bearer {KEY}"}, body=b"{}"))
            chunks = [c async for c in response.aiter_bytes()]
            await response.aclose()
            return response.status_code, chunks

    # Two pieces in, the same two pieces out: aiter_raw does not re-chunk.
    assert asyncio.run(run(UPSTREAM)) == (200, [b"data: 1\n\n", b"data: 2\n\n"])
    assert asyncio.run(run(UPSTREAM + "/v1/"))[0] == 200
    assert [s[1] for s in seen] == [f"{UPSTREAM}/v1/chat/completions?a=1"] * 2


@pytest.mark.parametrize("value, root", [
    (UPSTREAM, UPSTREAM), ({"url": UPSTREAM + "/v1"}, UPSTREAM),
    ({"kind": "tailnet"}, None), ("10.0.0.1:8000", None), (None, None),
])
def test_upstream_for_knows_a_url_string_and_a_url_object_and_nothing_else(value, root):
    found = proxy.upstream_for(value)
    assert (found.root if found else None) == root
