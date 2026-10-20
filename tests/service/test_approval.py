from __future__ import annotations

import json

import pytest

from tests.service._helpers import error_info, job_status, require_service_deps

require_service_deps()

ESCALATE = {"demo_id": "demo_grid", "agent": True, "budget": {"tau": 5.0}}  # margin < 5 always -> escalate


@pytest.fixture
def awaiting_job(client, make_worker):
    job = job_status(client.post("/v1/jobs", json=ESCALATE), 202)
    assert make_worker().poll_once(wait_seconds=0) == ["awaiting_approval"]
    body = job_status(client.get(f"/v1/jobs/{job['job_id']}"), 200)
    assert body["status"] == "awaiting_approval"
    return body


def _drain(services) -> list[dict]:
    resp = services.queue.client.receive_message(
        QueueUrl=services.queue.url, MaxNumberOfMessages=10, WaitTimeSeconds=0
    )
    return [json.loads(m["Body"]) for m in resp.get("Messages", [])]


def test_approval_409_when_not_awaiting(client):
    job = job_status(client.post("/v1/jobs", json={"demo_id": "demo_grid"}), 202)
    error_info(
        client.post(f"/v1/jobs/{job['job_id']}/approval", json={"chosen_rank": 1, "approver": "op"}),
        409,
        "invalid_state",
    )


def test_approval_404_unknown(client):
    error_info(
        client.post("/v1/jobs/" + "f" * 32 + "/approval", json={"chosen_rank": 1, "approver": "op"}),
        404,
        "not_found",
    )


def test_approve_rank_succeeds_and_enqueues_resume(client, services, awaiting_job):
    job_id = awaiting_job["job_id"]
    options = awaiting_job["approval"]["options"]
    assert 1 <= len(options) <= 3 and [o["rank"] for o in options] == list(range(1, len(options) + 1))
    _drain(services)  # nothing pending except what approval enqueues
    body = job_status(
        client.post(
            f"/v1/jobs/{job_id}/approval",
            json={"chosen_rank": 2, "approver": "alice", "note": "rank 2 matches overlay"},
        ),
        200,
    )
    assert body["status"] == "succeeded"
    assert (body["result"]["offset"]["x_px"], body["result"]["offset"]["y_px"]) == (
        options[1]["x_px"],
        options[1]["y_px"],
    )
    assert any('"operator_choice"' in w for w in body["result"]["warnings"])
    decision = json.loads(services.jobs.get_item(job_id)["decision"])
    assert decision == {"chosen_rank": 2, "approver": "alice", "note": "rank 2 matches overlay"}
    assert _drain(services) == [{"job_id": job_id, "resume": True}]
    # second decision: state machine refuses
    error_info(
        client.post(f"/v1/jobs/{job_id}/approval", json={"chosen_rank": 1, "approver": "bob"}),
        409,
        "invalid_state",
    )


def test_reject_all_fails_with_rejected_by_operator(client, awaiting_job):
    body = job_status(
        client.post(
            f"/v1/jobs/{awaiting_job['job_id']}/approval", json={"chosen_rank": None, "approver": "alice"}
        ),
        200,
    )
    assert body["status"] == "failed"
    assert body["error"]["code"] == "rejected_by_operator"
    assert body["job_id"] == awaiting_job["job_id"] and body["approval"] == awaiting_job["approval"]
    # the response IS the updated JobStatus: a fresh GET agrees with it
    assert job_status(client.get(f"/v1/jobs/{awaiting_job['job_id']}"), 200) == body


def test_resume_message_for_decided_job_is_skipped(client, make_worker, awaiting_job, queue_depth):
    client.post(f"/v1/jobs/{awaiting_job['job_id']}/approval", json={"chosen_rank": 1, "approver": "a"})
    assert make_worker().poll_once(wait_seconds=0) == ["skipped"]
    assert queue_depth() == (0, 0)
    assert client.get(f"/v1/jobs/{awaiting_job['job_id']}").json()["status"] == "succeeded"


@pytest.mark.parametrize(
    "payload",
    [
        {"chosen_rank": 4, "approver": "a"},  # contract: le=3
        {"chosen_rank": 0, "approver": "a"},
        {"chosen_rank": 1},  # approver required
    ],
)
def test_approval_body_validation(client, awaiting_job, payload):
    error_info(
        client.post(f"/v1/jobs/{awaiting_job['job_id']}/approval", json=payload), 422, "validation_error"
    )
