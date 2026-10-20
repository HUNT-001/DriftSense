from __future__ import annotations

import io
import json
import signal
import threading
import time
import urllib.error

import pytest

from tests.service._helpers import job_status, random_pair, require_service_deps

require_service_deps()

UPLOAD = "uploads/" + "b" * 32


def _status(client, job_id):
    return job_status(client.get(f"/v1/jobs/{job_id}"), 200)


def _submit(client, payload):
    return job_status(client.post("/v1/jobs", json=payload), 202)["job_id"]


def _upload_pair(services, template: bytes, search: bytes):
    services.artifacts.put_bytes(f"{UPLOAD}/template.png", template, "image/png")
    services.artifacts.put_bytes(f"{UPLOAD}/search.png", search, "image/png")
    return {"template_key": f"{UPLOAD}/template.png", "search_key": f"{UPLOAD}/search.png"}


def _poll(worker, tries: int = 50) -> list[str]:
    """poll_once until a message arrives (moto's zero-visibility release is clock-granular on Windows)."""
    for _ in range(tries):
        out = worker.poll_once(wait_seconds=0)
        if out:
            return out
        time.sleep(0.02)
    return []


def test_worker_success_fake_engine(client, services, make_worker, queue_depth, worker_info):
    job_id = _submit(client, {"demo_id": "demo_finfet_easy"})
    stream = io.StringIO()
    assert make_worker(emf_stream=stream).poll_once(wait_seconds=0) == ["succeeded"]
    body = _status(client, job_id)
    assert body["status"] == "succeeded" and body["engine"] == "fake"
    assert body["result"]["margin"]["decision"] == "commit"
    assert body["worker"] == worker_info.model_dump(mode="json")
    from driftsense.contracts import LocalizeResult

    stored = LocalizeResult.model_validate_json(services.artifacts.get_bytes(f"jobs/{job_id}/result.json"))
    assert stored.model_dump(mode="json") == body["result"]
    assert services.jobs.get_item(job_id)["result_key"] == f"jobs/{job_id}/result.json"
    assert queue_depth() == (0, 0)
    lines = [json.loads(x) for x in stream.getvalue().splitlines()]
    assert lines[0]["JobsSucceeded"] == 1 and lines[0]["JobsFailed"] == 0
    assert {x["Stage"] for x in lines[1:]} == {"ingest", "score"}


def test_worker_uses_uploaded_pair_and_finds_offset(client, services, make_worker):
    job_id = _submit(client, _upload_pair(services, *random_pair()))
    assert make_worker().poll_once(wait_seconds=0) == ["succeeded"]
    off = _status(client, job_id)["result"]["offset"]
    assert (off["x_px"], off["y_px"]) == (80.0, 64.0)


def test_demo_prefers_s3_objects(client, services, make_worker):
    t, s = random_pair(3)
    services.artifacts.put_bytes("demos/demo_grid/template.png", t, "image/png")
    services.artifacts.put_bytes("demos/demo_grid/search.png", s, "image/png")
    job_id = _submit(client, {"demo_id": "demo_grid"})
    make_worker().poll_once(wait_seconds=0)
    off = _status(client, job_id)["result"]["offset"]
    assert (off["x_px"], off["y_px"]) == (80.0, 64.0)


def test_synthetic_demos_are_deterministic():
    from service.demos import DEMOS, synthetic_pair

    for demo_id in DEMOS:
        assert synthetic_pair(demo_id) == synthetic_pair(demo_id)


def test_escalate_with_agent_awaits_approval(client, make_worker):
    job_id = _submit(client, {"demo_id": "demo_grid", "agent": True, "budget": {"tau": 5.0}})
    assert make_worker().poll_once(wait_seconds=0) == ["awaiting_approval"]
    body = _status(client, job_id)
    assert body["status"] == "awaiting_approval"
    assert body["approval"]["options"] == body["result"]["candidates"][:3]
    assert body["approval"]["margin"] == body["result"]["margin"]


def test_escalate_without_agent_still_succeeds(client, make_worker):
    job_id = _submit(client, {"demo_id": "demo_grid", "budget": {"tau": 5.0}})
    assert make_worker().poll_once(wait_seconds=0) == ["succeeded"]
    assert _status(client, job_id)["result"]["margin"]["decision"] == "escalate"


def test_bad_input_fails_without_retry(client, services, make_worker, queue_depth):
    job_id = _submit(client, _upload_pair(services, b"not an image", b"also not"))
    stream = io.StringIO()
    assert make_worker(emf_stream=stream).poll_once(wait_seconds=0) == ["failed_bad_input"]
    body = _status(client, job_id)
    assert body["status"] == "failed" and body["error"]["code"] == "bad_input"
    assert queue_depth() == (0, 0)  # deleted: no retry
    assert json.loads(stream.getvalue().splitlines()[0])["JobsFailed"] == 1


def test_missing_upload_is_bad_input(client, make_worker):
    job_id = _submit(client, {"template_key": f"{UPLOAD}/template.png", "search_key": f"{UPLOAD}/search.png"})
    assert make_worker().poll_once(wait_seconds=0) == ["failed_bad_input"]
    assert "upload not found" in _status(client, job_id)["error"]["message"]


def test_transient_error_retries_then_fails_and_goes_to_dlq(client, services, make_worker, queue_depth):
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("engine exploded")

    worker = make_worker(localize_fn=flaky)
    job_id = _submit(client, {"demo_id": "demo_grid"})
    assert _poll(worker) == ["retry"]
    assert _status(client, job_id)["status"] == "queued"  # visible to the UI as waiting again
    assert _poll(worker) == ["retry"]  # backoff 0 in tests -> visible again
    assert _poll(worker) == ["failed_engine_error"]
    body = _status(client, job_id)
    assert body["status"] == "failed" and body["error"]["code"] == "engine_error"
    assert "engine exploded" in body["error"]["message"]
    assert len(calls) == 3
    # not deleted on the final receive but released: the redrive policy moves it to the DLQ on the next receive
    dlq_url = services.queue.url + "-dlq"
    dlq = []
    for _ in range(50):
        assert worker.poll_once(wait_seconds=0) == []
        dlq = services.queue.client.receive_message(
            QueueUrl=dlq_url, WaitTimeSeconds=0, MaxNumberOfMessages=10
        ).get("Messages", [])
        if dlq:
            break
        time.sleep(0.02)
    assert queue_depth() == (0, 0)
    assert [json.loads(m["Body"]) for m in dlq] == [{"job_id": job_id}]


def test_transient_error_then_success(client, make_worker):
    state = {"n": 0}
    from driftsense.api import localize

    def flaky_once(*args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise TimeoutError("transient")
        return localize(*args, **kwargs)

    job_id = _submit(client, {"demo_id": "demo_grid"})
    worker = make_worker(localize_fn=flaky_once)
    assert _poll(worker) == ["retry"]
    assert _poll(worker) == ["succeeded"]
    assert _status(client, job_id)["error"] is None


def test_terminal_job_is_skipped_idempotently(client, services, make_worker, queue_depth):
    job_id = _submit(client, {"demo_id": "demo_grid"})
    worker = make_worker()
    assert worker.poll_once(wait_seconds=0) == ["succeeded"]
    first = _status(client, job_id)
    services.queue.send(job_id)  # duplicate delivery
    assert worker.poll_once(wait_seconds=0) == ["skipped"]
    assert _status(client, job_id) == first
    assert queue_depth() == (0, 0)


def test_malformed_and_unknown_messages_are_dropped(services, make_worker, queue_depth):
    services.queue.client.send_message(QueueUrl=services.queue.url, MessageBody="{not json")
    services.queue.send("c" * 32)
    worker = make_worker()
    assert sorted(worker.poll_once(wait_seconds=0) + worker.poll_once(wait_seconds=0)) == [
        "dropped",
        "dropped",
    ]
    assert queue_depth() == (0, 0)


def test_heartbeat_extends_visibility(client, services, make_worker):
    seen = []
    real = services.queue.change_visibility

    def spy(handle, timeout):
        seen.append(timeout)
        return real(handle, timeout)

    def slow(*args, **kwargs):
        time.sleep(0.35)
        from driftsense.api import localize

        return localize(*args, **kwargs)

    services.queue.change_visibility = spy
    _submit(client, {"demo_id": "demo_grid"})
    worker = make_worker(localize_fn=slow, heartbeat_s=0.1, visibility_timeout=123)
    assert worker.poll_once(wait_seconds=0) == ["succeeded"]
    assert len(seen) >= 2 and set(seen) == {123}


def test_sigterm_sets_stop_flag_and_finishes_current_message(client, make_worker, queue_depth):
    from driftsense.api import localize

    ids = [_submit(client, {"demo_id": "demo_grid"}) for _ in range(2)]
    worker = make_worker()

    def stop_midway(*args, **kwargs):
        worker.request_stop(signal.SIGTERM, None)  # what the SIGTERM handler does
        return localize(*args, **kwargs)

    worker.localize_fn = stop_midway
    worker.run_forever(wait_seconds=0)  # returns after finishing the in-flight message
    statuses = sorted(_status(client, i)["status"] for i in ids)
    assert statuses == ["queued", "succeeded"]
    assert worker.stop_event.is_set()
    assert queue_depth() == (1, 0)


def test_install_signal_handlers_routes_sigterm(make_worker, monkeypatch):
    installed = {}
    monkeypatch.setattr(signal, "signal", lambda sig, fn: installed.setdefault(sig, fn))
    worker = make_worker()
    worker.install_signal_handlers()
    installed[signal.SIGTERM](signal.SIGTERM, None)
    assert worker.stop_event.is_set()


def test_run_forever_exits_immediately_when_stopped(make_worker):
    worker = make_worker()
    worker.stop_event.set()
    t = threading.Thread(target=worker.run_forever, kwargs={"wait_seconds": 0})
    t.start()
    t.join(2)
    assert not t.is_alive()


def test_imds_absent_returns_none(monkeypatch):
    from service import worker as w

    monkeypatch.delenv("AWS_EC2_METADATA_DISABLED", raising=False)

    def unreachable(req, timeout):
        assert timeout == 1.0
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(w, "_urlopen", unreachable)
    assert w.imds_instance_type() is None
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    assert w.imds_instance_type() is None


def test_imds_v2_token_flow(monkeypatch):
    from service import worker as w

    monkeypatch.delenv("AWS_EC2_METADATA_DISABLED", raising=False)
    seen = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake(req, timeout):
        seen.append((req.get_method(), req.full_url, dict(req.header_items())))
        return Resp(b"tok123" if req.full_url.endswith("/api/token") else b"c8g.large")

    monkeypatch.setattr(w, "_urlopen", fake)
    assert w.imds_instance_type() == "c8g.large"
    assert seen[0][0] == "PUT" and seen[1][0] == "GET"
    assert seen[1][2].get("X-aws-ec2-metadata-token") == "tok123"


def test_worker_info_and_startup_log(caplog, monkeypatch):
    import cv2

    from service import worker as w

    monkeypatch.setattr(w, "imds_instance_type", lambda: None)
    info = w.collect_worker_info()
    assert info.opencv_version == cv2.__version__ and info.instance_type is None
    assert info.cool is ("/opt/cool" in cv2.__file__)
    assert w.detect_arch("AMD64") == "x86_64" and w.detect_arch("aarch64") == "aarch64"
    assert w.detect_arch("arm64") == "arm64" and w.detect_arch("sparc") == "unknown"
    report = w.opencv_report()
    assert report["kleidicv"] == ("kleidicv" in cv2.getBuildInformation().lower())


def test_log_startup_mentions_cv2_and_kleidicv(make_worker, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="driftsense.worker")
    make_worker().log_startup()
    text = caplog.text
    assert "cv2.__version__=" in text and "cv2.__file__=" in text and "KleidiCV=" in text


@pytest.mark.parametrize("engine", ["fake", "v1"])
def test_worker_with_engine_env(engine, client, settings, make_worker, monkeypatch):
    from dataclasses import replace

    from driftsense.api import available_engines, localize

    if engine not in available_engines():
        pytest.skip(f"engine {engine} not importable in this checkout")
    try:
        localize(*random_pair(), engine=engine)
    except NotImplementedError:
        pytest.skip(f"{engine} is still a stub")
    monkeypatch.setenv("DRIFTSENSE_ENGINE", engine)
    job_id = _submit(client, {"demo_id": "demo_finfet_easy"})
    worker = make_worker(settings=replace(settings, engine=engine))
    assert worker.poll_once(wait_seconds=0) == ["succeeded"]
    body = _status(client, job_id)
    assert body["engine"] == engine and body["result"]["engine"] == engine
