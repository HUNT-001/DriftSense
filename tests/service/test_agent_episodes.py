"""Agent episodes: POST /v1/agent/episodes -> worker runs driftsense.agent.runner.run_episode (real v2 on a
VirtualSEM) -> trace streamed to S3 -> human approval through DynamoDB -> final state. All under moto."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest

from tests.service._helpers import error_info, job_status, require_service_deps

require_service_deps()
pytest.importorskip("driftsense.agent.runner")
pytest.importorskip("driftsense.sim")

FAST = {"approval_poll_s": 0.05, "trace_upload_s": 0.1}


def _episode(client, family="dram", seed=7, token=None, code=202):
    body = {"family": family, "seed": seed}
    if token:
        body["client_token"] = token
    return job_status(client.post("/v1/agent/episodes", json=body), code)


def _trace(client, job_id):
    from driftsense.contracts import TraceEvent

    r = client.get(f"/v1/jobs/{job_id}/trace")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert r.content == b"" or r.content.endswith(b"\n")
    return [TraceEvent.model_validate_json(line) for line in r.content.decode().splitlines()]


def _wait_status(client, job_id, wanted, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = job_status(client.get(f"/v1/jobs/{job_id}"), 200)
        if body["status"] == wanted:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {wanted}; last={body['status']}")


def _start_worker(worker):
    out: list = []
    t = threading.Thread(target=lambda: out.extend(worker.poll_once(wait_seconds=0)), daemon=True)
    t.start()
    return t, out


def _episode_json(services, job_id):
    return json.loads(services.artifacts.get_bytes(f"jobs/{job_id}/episode.json"))


# ---------- API ----------
def test_demos_list_agent_episodes(client):
    from service.models import DemoInfo

    demos = [DemoInfo.model_validate(d) for d in client.get("/v1/demos").json()]
    agent = {d.demo_id: d for d in demos if d.agent}
    assert {"vsem-dram-7", "vsem-finfet-3"} <= set(agent)
    assert all(
        d.family and d.seed is not None and d.demo_id == f"vsem-{d.family}-{d.seed}" for d in agent.values()
    )
    assert any(not d.agent for d in demos)  # classic localize demos are still there


def test_episode_endpoint_is_an_agent_job_and_idempotent(client, services, queue_depth):
    job = _episode(client, "finfet", 3, token="ep-1")
    assert job["status"] == "queued" and job["trace_key"] is None
    req = json.loads(services.jobs.get_item(job["job_id"])["request"])
    assert req["demo_id"] == "vsem-finfet-3" and req["agent"] is True and req["client_token"] == "ep-1"
    # same token, same episode -> same job (200), through either endpoint
    assert _episode(client, "finfet", 3, token="ep-1", code=200)["job_id"] == job["job_id"]
    same = client.post("/v1/jobs", json={"demo_id": "vsem-finfet-3", "agent": True, "client_token": "ep-1"})
    assert job_status(same, 200)["job_id"] == job["job_id"]
    # same token, different episode -> conflict
    error_info(
        client.post("/v1/agent/episodes", json={"family": "dram", "seed": 3, "client_token": "ep-1"}),
        409,
        "idempotency_conflict",
    )
    assert queue_depth() == (1, 0)


@pytest.mark.parametrize(
    "body",
    [
        {"family": "sram", "seed": 1},
        {"family": "dram", "seed": -1},
        {"family": "dram", "seed": 2**31},
        {"family": "dram"},
        {"family": "dram", "seed": 1, "extra": True},
        {"family": "dram", "seed": 1, "client_token": "x" * 129},
    ],
)
def test_episode_body_validation(client, body):
    error_info(client.post("/v1/agent/episodes", json=body), 422, "validation_error")


@pytest.mark.parametrize(
    "body",
    [
        {"demo_id": "vsem-dram-7"},  # episodes need agent=true
        {"demo_id": "vsem-dram-x", "agent": True},
        {"demo_id": "vsem-dram-07", "agent": True},
        {"demo_id": "vsem-dram-2147483648", "agent": True},
    ],
)
def test_jobs_endpoint_vsem_validation(client, body):
    error_info(client.post("/v1/jobs", json=body), 422, "validation_error")


def test_approval_409_for_episode_not_awaiting(client):
    job = _episode(client)
    error_info(
        client.post(f"/v1/jobs/{job['job_id']}/approval", json={"chosen_rank": 1, "approver": "op"}),
        409,
        "invalid_state",
    )


# ---------- worker: real episodes ----------
def test_episode_commit_end_to_end(client, services, make_worker, queue_depth):
    """vsem-dram-7: the bootstrap frame clears tau -> commit + image-only verify -> succeeded."""
    from driftsense.contracts import LocalizeResult

    job_id = _episode(client, "dram", 7)["job_id"]
    assert make_worker().poll_once(wait_seconds=0) == ["succeeded"]
    body = job_status(client.get(f"/v1/jobs/{job_id}"), 200)
    assert body["engine"] == "v2" and body["trace_key"] == f"traces/{job_id}.jsonl"
    assert body["error"] is None and body["approval"] is None
    result = LocalizeResult.model_validate(body["result"])
    assert (
        result.engine == "v2"
        and result.margin.decision == "commit"
        and result.margin.tau == pytest.approx(0.10)
    )

    events = _trace(client, job_id)
    assert {e.run_id for e in events} == {job_id}
    assert [e.step for e in events] == list(range(len(events)))
    kinds = [e.kind for e in events]
    assert kinds[-1] == "episode_end" and "commit_move" in kinds and "escalate" not in kinds
    commit = next(e for e in events if e.kind == "commit_move")
    assert commit.because and ">= tau" in commit.because and commit.stage_move_nm is not None

    ep = _episode_json(services, job_id)
    assert ep["outcome"] == "commit" and ep["run_id"] == job_id and ep["n_events"] == len(events)
    assert ep["committed_move_nm"] == list(commit.stage_move_nm)
    assert ep["post_commit_verify"]["post_commit_verify"] is True
    assert ep["operating_point"]["source"] == "eval/agent_eval.py"
    for name in (
        "episode.json",
        "result.json",
        "overlay.png",
        "top5_gallery.png",
        "search.png",
        "template.png",
    ):
        assert client.get(f"/v1/jobs/{job_id}/artifacts/{name}").status_code == 200, name
    uris = {a.uri for a in result.artifacts}
    assert {f"jobs/{job_id}/episode.json", f"jobs/{job_id}/result.json"} <= uris
    assert (
        LocalizeResult.model_validate_json(client.get(f"/v1/jobs/{job_id}/artifacts/result.json").content)
        == result
    )
    assert queue_depth() == (0, 0)
    # a duplicate delivery of a finished episode is a no-op
    services.queue.send(job_id)
    assert make_worker().poll_once(wait_seconds=0) == ["skipped"]


def test_episode_escalation_approval_round_trip(client, services, settings, make_worker):
    """vsem-dram-17 escalates on its own (periodic ambiguity): park in awaiting_approval, stream the
    trace, operator picks rank 1, the worker resumes the runner and the job succeeds."""
    job_id = _episode(client, "dram", 17)["job_id"]
    worker = make_worker(settings=replace(settings, **FAST))
    thread, out = _start_worker(worker)
    waiting = _wait_status(client, job_id, "awaiting_approval")
    assert waiting["trace_key"] == f"traces/{job_id}.jsonl"
    req = waiting["approval"]
    assert 1 <= len(req["options"]) <= 3 and req["margin"]["decision"] == "escalate"
    # the trace was uploaded at awaiting_approval and already shows the request
    events = _trace(client, job_id)
    assert events[-1].kind == "tool_call" and events[-1].tool == "request_human_approval"
    assert all(e.kind != "episode_end" for e in events)

    resumed = job_status(
        client.post(
            f"/v1/jobs/{job_id}/approval", json={"chosen_rank": 1, "approver": "alice", "note": "ok"}
        ),
        200,
    )
    assert resumed["status"] == "running"
    thread.join(timeout=60)
    assert out == ["succeeded"]
    body = job_status(client.get(f"/v1/jobs/{job_id}"), 200)
    assert body["status"] == "succeeded" and body["error"] is None
    opt = req["options"][0]
    assert (body["result"]["offset"]["x_px"], body["result"]["offset"]["y_px"]) == (opt["x_px"], opt["y_px"])
    assert any('"operator_choice"' in w for w in body["result"]["warnings"])
    events = _trace(client, job_id)
    assert events[-1].kind == "episode_end"
    esc = next(e for e in events if e.kind == "escalate")
    assert esc.result_summary == {"chosen_rank": 1, "approver": "alice"}
    ep = _episode_json(services, job_id)
    assert ep["outcome"] == "escalate" and ep["approval"]["chosen_rank"] == 1
    # decided: a second approval is refused
    error_info(
        client.post(f"/v1/jobs/{job_id}/approval", json={"chosen_rank": 2, "approver": "bob"}),
        409,
        "invalid_state",
    )


def test_episode_rejection_fails_rejected_by_operator(client, services, settings, make_worker):
    """max_steps=1 forces escalation (budget exhausted on step 0); the operator rejects every option."""
    job_id = _episode(client, "dram", 1)["job_id"]
    worker = make_worker(settings=replace(settings, agent_max_steps=1, **FAST))
    thread, out = _start_worker(worker)
    waiting = _wait_status(client, job_id, "awaiting_approval")
    error_info(
        client.post(f"/v1/jobs/{job_id}/approval", json={"chosen_rank": 3 + 1, "approver": "x"}),
        422,
        "validation_error",
    )
    n = len(waiting["approval"]["options"])
    if n < 3:
        error_info(
            client.post(f"/v1/jobs/{job_id}/approval", json={"chosen_rank": n + 1, "approver": "x"}),
            422,
            "validation_error",
        )
    job_status(
        client.post(f"/v1/jobs/{job_id}/approval", json={"chosen_rank": None, "approver": "carol"}), 200
    )
    thread.join(timeout=60)
    assert out == ["failed"]
    body = job_status(client.get(f"/v1/jobs/{job_id}"), 200)
    assert body["status"] == "failed" and body["error"]["code"] == "rejected_by_operator"
    assert "carol" in body["error"]["message"]
    assert body["result"] is not None and body["result"]["margin"]["decision"] == "escalate"
    assert _episode_json(services, job_id)["approval"] == {
        "chosen_rank": None,
        "approver": "carol",
        "note": None,
    }


def test_episode_approval_timeout(client, services, settings, make_worker):
    job_id = _episode(client, "dram", 1)["job_id"]
    worker = make_worker(settings=replace(settings, agent_max_steps=1, approval_timeout_s=0.3, **FAST))
    assert worker.poll_once(wait_seconds=0) == ["failed"]
    body = job_status(client.get(f"/v1/jobs/{job_id}"), 200)
    assert body["status"] == "failed" and body["error"]["code"] == "approval_timeout"
    assert body["approval"] is not None  # what the operator was asked stays on the job
    decision = json.loads(services.jobs.get_item(job_id)["decision"])
    assert decision["chosen_rank"] is None and decision["approver"] == "timeout"
    events = _trace(client, job_id)
    assert events[-1].kind == "episode_end"
    assert any(e.kind == "escalate" and e.result_summary["approver"] == "timeout" for e in events)


def test_redelivered_episode_reuses_stored_decision(client, services, settings, make_worker):
    """A worker that died while awaiting approval: the redelivered episode reuses the stored decision."""
    from driftsense.contracts import ApprovalDecision

    job_id = _episode(client, "dram", 1)["job_id"]
    decision = ApprovalDecision(chosen_rank=2, approver="dave")
    services.jobs.transition(
        job_id, "awaiting_approval", allowed_from=("queued",), set_fields={"decision": decision}
    )
    worker = make_worker(settings=replace(settings, agent_max_steps=1, approval_timeout_s=0.3, **FAST))
    assert worker.poll_once(wait_seconds=0) == ["succeeded"]
    body = job_status(client.get(f"/v1/jobs/{job_id}"), 200)
    assert any('"rank":2' in w and "dave" in w for w in body["result"]["warnings"])


# ---------- units ----------
def test_trace_uploader_publishes_complete_lines_only(services, tmp_path):
    from service.agent import TraceUploader

    path = tmp_path / "t.jsonl"
    up = TraceUploader(services.artifacts, "traces/x.jsonl", path, interval_s=60)
    assert up.flush() is True and services.artifacts.get_bytes("traces/x.jsonl") == b""
    path.write_bytes(b'{"a":1}\n{"b":')
    assert up.flush() is True and services.artifacts.get_bytes("traces/x.jsonl") == b'{"a":1}\n'
    assert up.flush() is False  # unchanged complete prefix -> no re-upload
    path.write_bytes(b'{"a":1}\n{"b":2}\n')
    assert up.flush() is True and services.artifacts.get_bytes("traces/x.jsonl").count(b"\n") == 2


def _fake_ep(outcome, approval=None, events=()):
    from types import SimpleNamespace

    return SimpleNamespace(outcome=outcome, approval=approval, events=list(events))


def test_final_state_mapping():
    from datetime import UTC, datetime

    from driftsense.contracts import ApprovalDecision, TraceEvent
    from service.agent import final_state

    err = TraceEvent(
        run_id="r", step=3, ts=datetime.now(UTC), kind="error", because="tool timeout > 15s: safe abort"
    )
    fs = final_state(_fake_ep("abort", events=[err]), None, timed_out=False)
    assert (fs.status, fs.error.code, fs.error.message) == ("failed", "agent_abort", err.because)
    fs = final_state(_fake_ep("escalate", None, [err]), None, timed_out=False)
    assert (fs.status, fs.error.code) == ("failed", "budget_exhausted")
    fs = final_state(
        _fake_ep("escalate", ApprovalDecision(chosen_rank=None, approver="timeout")), None, timed_out=True
    )
    assert fs.error.code == "approval_timeout"
    fs = final_state(
        _fake_ep("escalate", ApprovalDecision(chosen_rank=None, approver="op")), None, timed_out=False
    )
    assert fs.error.code == "rejected_by_operator"
    fs = final_state(
        _fake_ep("escalate", ApprovalDecision(chosen_rank=1, approver="op")), None, timed_out=False
    )
    assert fs.status == "succeeded" and fs.error is None
    assert final_state(_fake_ep("commit"), None, timed_out=False).status == "succeeded"


def test_parse_vsem():
    from service.demos import parse_vsem

    assert parse_vsem("vsem-dram-7") == ("dram", 7)
    assert parse_vsem("vsem-random-0") == ("random", 0)
    assert parse_vsem(f"vsem-finfet-{2**31 - 1}") == ("finfet", 2**31 - 1)
    for bad in (None, "", "vsem-dram-", "vsem-dram-01", "vsem-sram-1", "demo_grid", f"vsem-dram-{2**31}"):
        assert parse_vsem(bad) is None


def test_effective_tool_timeout_outlasts_approval_window():
    from service.settings import Settings

    s = Settings.from_env({"APPROVAL_TIMEOUT_S": "900"})
    assert s.approval_timeout_s == 900 and s.effective_tool_timeout_s >= 900 + 60
    assert Settings.from_env({}).agent_tau == pytest.approx(0.10)


def test_trace_uploader_thread_mirrors_periodically(services, tmp_path):
    from service.agent import TraceUploader

    path = tmp_path / "t.jsonl"
    with TraceUploader(services.artifacts, "traces/y.jsonl", path, interval_s=0.05) as up:
        path.write_bytes(b'{"step":0}\n')
        deadline = time.monotonic() + 5
        while (
            services.artifacts.get_bytes("traces/y.jsonl") != b'{"step":0}\n' and time.monotonic() < deadline
        ):
            time.sleep(0.02)
    assert services.artifacts.get_bytes("traces/y.jsonl") == b'{"step":0}\n' and up.uploads >= 2


def test_episode_timeouts_follow_runner_signature():
    from types import SimpleNamespace

    from service.agent import episode_timeouts

    st = SimpleNamespace(agent_tool_timeout_s=15.0, approval_timeout_s=900.0, effective_tool_timeout_s=960.0)

    def new_runner(sem, *, tool_timeout_s=15.0, approval_timeout_s=None): ...

    def old_runner(sem, *, tool_timeout_s=15.0): ...

    assert episode_timeouts(new_runner, st) == {"tool_timeout_s": 15.0, "approval_timeout_s": 960.0}
    assert episode_timeouts(old_runner, st) == {"tool_timeout_s": 960.0}
