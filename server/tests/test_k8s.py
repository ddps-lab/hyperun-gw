"""The log relay's line filter, and the serving methods' answers to each apiserver branch.

`redact` is the only pure function in `k8s.py`; the rest needs a cluster. What
it protects: the driver writes its own bookkeeping onto the same stdout as the
user's training output, and those names are ours, not theirs.

HYPERUN-SERVING. The serving methods are driven here through stand-ins for the
two kubernetes API objects that raise the real `ApiException` with the real
status codes, so what is tested is `Cluster`'s own branching (404 -> NotFound,
409 -> Conflict, a Secret's base64) and not a double written from the routes'
side, which would agree with whatever the routes expect.
"""

import base64
import types

import pytest
from kubernetes.client.rest import ApiException

from ddpsrun_server.k8s import Cluster, ClusterError, Conflict, NotFound, redact


def test_the_keepalive_line_is_dropped_entirely():
    # The driver prints one of these every 30 seconds for the whole life of the
    # job, purely to hold the log stream open. A user reading a training curve
    # does not want one between every progress line.
    assert redact("PACSRUN_KEEPALIVE") is None
    assert redact("  PACSRUN_KEEPALIVE  ") is None


def test_an_internal_name_is_masked_but_the_line_survives():
    # The line may be the user's own output with our token appended to it, so
    # dropping the whole line would throw away something they need.
    assert redact("done, PACSRUN_EXIT=0") == "done, <internal>=0"
    assert redact("PACSRUN_ARTIFACT=/workspace/out.tar") == "<internal>=/workspace/out.tar"


def test_ordinary_output_passes_through_unchanged():
    line = "{'loss': 0.42, 'epoch': 1.0}"
    assert redact(line) == line
    progress = " 25%|##        | 25/99 [03:45<08:37,  6.99s/it]"
    assert redact(progress) == progress


def test_a_word_that_merely_starts_with_the_prefix_in_prose_is_left_alone():
    # The pattern requires the uppercase form with a word boundary, so a user's
    # own lowercase variable is not touched.
    assert redact("hyperun_exit is not ours") == "hyperun_exit is not ours"


def test_the_gpu_telemetry_line_is_dropped_from_the_user_facing_log():
    # It goes to /v1/jobs/{id}/metrics, which reads the log unredacted. Leaving
    # it here masked put "<internal>=97,38380,45440,72,304.0" between every
    # couple of training lines, every 30 seconds, for the life of the job.
    assert redact("PACSRUN_GPU=94,38200,45440,71,298.5") is None
    assert redact("  PACSRUN_GPU=1,2,3,4,5.0") is None


def test_a_training_line_that_merely_mentions_gpu_survives():
    line = "using GPU 0: NVIDIA L40S"
    assert redact(line) == line


def test_the_new_spelling_of_the_runner_lines_is_handled_like_the_old():
    """The hyperun cluster's driver prints HYPERUN_ where it printed PACSRUN_; both are its own."""
    assert redact("HYPERUN_KEEPALIVE") is None
    assert redact("HYPERUN_GPU=94,38200,45440,71,298.5") is None
    assert redact("HYPERUN_GPU_CARD=0,94,38200,81920,71,298.5") is None
    assert redact("HYPERUN_ARTIFACT=/workspace/out.tar") == "<internal>=/workspace/out.tar"


# ------------------------------------------------------------ HYPERUN-SERVING


class Answers:
    """Stands in for CustomObjectsApi and CoreV1Api: each call answers from `script`, either a
    value to return or an HTTP status to raise as the client's own ApiException."""

    def __init__(self, **script):
        self.script = script
        self.calls = []

    def __getattr__(self, method):
        def call(**kwargs):
            self.calls.append((method, kwargs))
            answer = self.script[method]
            if isinstance(answer, int):
                raise ApiException(status=answer, reason="test")
            return answer
        return call


def secret(data):
    """A V1Secret-shaped object with base64 data, as read_namespaced_secret returns it."""
    return types.SimpleNamespace(data={k: base64.b64encode(v.encode()).decode()
                                       for k, v in data.items()})


def test_create_serving_turns_409_into_conflict_and_404_into_crd_not_installed():
    body = {"metadata": {"name": "chat"}}
    with pytest.raises(Conflict):
        Cluster(Answers(create_namespaced_custom_object=409), None).create_serving("ns", body)
    with pytest.raises(ClusterError) as caught:
        Cluster(Answers(create_namespaced_custom_object=404), None).create_serving("ns", body)
    assert "CRD is not installed" in str(caught.value)
    custom = Answers(create_namespaced_custom_object={"ok": 1})
    assert Cluster(custom, None).create_serving("ns", body) == {"ok": 1}
    assert custom.calls[0][1]["plural"] == "hyperunservings"


def test_get_list_delete_serving_read_a_404_as_absence():
    with pytest.raises(NotFound):
        Cluster(Answers(get_namespaced_custom_object=404), None).get_serving("ns", "x")
    with pytest.raises(NotFound):
        Cluster(Answers(delete_namespaced_custom_object=404), None).delete_serving("ns", "x")
    # A list on a cluster without the CRD is an empty list, so the screen says "none".
    assert Cluster(Answers(list_namespaced_custom_object=404), None).list_servings("ns") == []
    with pytest.raises(ClusterError):
        Cluster(Answers(list_namespaced_custom_object=403), None).list_servings("ns")


def test_the_api_key_is_read_from_name_dash_api_key_and_decoded():
    core = Answers(read_namespaced_secret=secret({"api-key": "sk-test"}))
    assert Cluster(None, core).serving_api_key("lab-alice", "chat") == "sk-test"
    # The Secret name is built from the serving name; a caller cannot aim it elsewhere.
    assert core.calls == [("read_namespaced_secret",
                           {"name": "chat-api-key", "namespace": "lab-alice"})]


def test_no_key_secret_or_no_api_key_entry_is_not_found_and_a_403_is_a_cluster_error():
    with pytest.raises(NotFound):
        Cluster(None, Answers(read_namespaced_secret=404)).serving_api_key("ns", "chat")
    with pytest.raises(NotFound):
        Cluster(None, Answers(read_namespaced_secret=secret({"other": "x"}))).serving_api_key(
            "ns", "chat")
    with pytest.raises(ClusterError):
        Cluster(None, Answers(read_namespaced_secret=403)).serving_api_key("ns", "chat")


def test_one_registered_value_is_read_from_the_namespaces_store_and_nothing_else():
    core = Answers(read_namespaced_secret=secret({"HF_TOKEN": "hf-x", "GITHUB_PAT": "gh-x"}))
    assert Cluster(None, core).user_secret_value("lab-alice", "HF_TOKEN") == "hf-x"
    assert core.calls[0][1]["name"] == "hyperun-user-secrets"
    with pytest.raises(NotFound):
        Cluster(None, core).user_secret_value("lab-alice", "MISSING")
    with pytest.raises(NotFound):
        Cluster(None, Answers(read_namespaced_secret=404)).user_secret_value("ns", "HF_TOKEN")
