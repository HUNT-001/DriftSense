"""End-to-end flow in-process under moto: ``python -m service.e2e_local [--demo ID] [--engine fake|v1|v2] [--agent]``.

Creates table/queue(+DLQ)/bucket, POSTs a job through the FastAPI TestClient, runs one worker
iteration, GETs the job and prints the final JobStatus JSON. Exit 0 on succeeded/awaiting_approval.
With ``--agent`` it POSTs /v1/agent/episodes (default ``vsem-dram-17``, which escalates), runs the real
``driftsense.agent.runner`` episode, approves rank 1 from an auto-approver thread when the episode asks,
validates every trace line and exits 0 only if the job ends ``succeeded``.
Dev-only: needs moto + httpx (never imported by the API or the worker)."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import warnings
from dataclasses import replace

LOCAL_ENV = {
    "AWS_REGION": "ap-south-1",
    "AWS_DEFAULT_REGION": "ap-south-1",
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SECURITY_TOKEN": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_EC2_METADATA_DISABLED": "true",
    "JOBS_TABLE": "driftsense-jobs-local",
    "DATA_BUCKET": "driftsense-data-local",
    "DRIFTSENSE_ENGINE": "fake",
    "ENGINE_GIT_SHA": "local",
}


def provision(
    region: str,
    table: str,
    bucket: str,
    queue_name: str = "driftsense-jobs-local",
    max_receive_count: int = 3,
) -> str:
    """Create the AWS resources the service expects (inside an active moto mock). Returns the queue URL."""
    import boto3

    ddb = boto3.client("dynamodb", region_name=region)
    ddb.create_table(
        TableName=table,
        BillingMode="PAY_PER_REQUEST",
        AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
        KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
    )
    ddb.update_time_to_live(
        TableName=table, TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"}
    )
    s3 = boto3.client("s3", region_name=region)
    s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
    sqs = boto3.client("sqs", region_name=region)
    dlq_url = sqs.create_queue(QueueName=queue_name + "-dlq")["QueueUrl"]
    dlq_arn = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["QueueArn"])["Attributes"][
        "QueueArn"
    ]
    redrive = json.dumps({"deadLetterTargetArn": dlq_arn, "maxReceiveCount": str(max_receive_count)})
    return sqs.create_queue(
        QueueName=queue_name, Attributes={"VisibilityTimeout": "180", "RedrivePolicy": redrive}
    )["QueueUrl"]


DEFAULT_AGENT_DEMO = "vsem-dram-17"  # escalates on its own: exercises the human-approval path


def _auto_approver(client, job_id: str, stop, log: list, rank: int = 1) -> None:
    """Stand-in operator: approve ``rank`` as soon as the episode asks for a decision."""
    while not stop.is_set():
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body.get("status") == "awaiting_approval":
            r = client.post(f"/v1/jobs/{job_id}/approval", json={"chosen_rank": rank, "approver": "e2e-auto"})
            log.append(
                {"approved_rank": rank, "status_code": r.status_code, "status": r.json().get("status")}
            )
            return
        if body.get("status") in ("succeeded", "failed"):
            return
        stop.wait(0.1)


def run(demo_id: str | None = None, engine: str | None = None, agent: bool = False) -> tuple[int, dict]:
    for key, value in LOCAL_ENV.items():
        os.environ.setdefault(key, value)
    if engine:
        os.environ["DRIFTSENSE_ENGINE"] = engine
    demo_id = demo_id or (DEFAULT_AGENT_DEMO if agent else "demo_finfet_easy")

    with warnings.catch_warnings():  # starlette's "use httpx2" notice is noise for a dev script
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    from moto import mock_aws

    from service.api import Services, create_app
    from service.demos import parse_vsem
    from service.models import JobStatus
    from service.settings import Settings
    from service.worker import Worker

    episode = parse_vsem(demo_id) if agent else None
    with mock_aws():
        env = os.environ
        queue_url = provision(env["AWS_REGION"], env["JOBS_TABLE"], env["DATA_BUCKET"])
        os.environ["JOBS_QUEUE_URL"] = queue_url
        settings = Settings.from_env()
        if episode is not None:
            settings = replace(settings, approval_poll_s=0.2, trace_upload_s=0.5)
        services = Services.from_settings(settings)
        client = TestClient(create_app(services=services))
        if episode is not None:
            family, seed = episode
            resp = client.post(
                "/v1/agent/episodes", json={"family": family, "seed": seed, "client_token": "e2e-local"}
            )
        else:
            resp = client.post(
                "/v1/jobs", json={"demo_id": demo_id, "agent": agent, "client_token": "e2e-local"}
            )
        if resp.status_code != 202:
            return 1, {"stage": "create", "status_code": resp.status_code, "body": resp.json()}
        job_id = resp.json()["job_id"]
        emf_buf = io.StringIO()
        worker = Worker(
            settings,
            jobs=services.jobs,
            artifacts=services.artifacts,
            queue=services.queue,
            emf_stream=emf_buf,
        )
        stop, approvals = threading.Event(), []
        approver = None
        if episode is not None:
            approver = threading.Thread(
                target=_auto_approver,
                args=(client, job_id, stop, approvals),
                name="e2e-approver",
                daemon=True,
            )
            approver.start()
        try:
            outcomes = worker.poll_once(wait_seconds=0)
        finally:
            stop.set()
            if approver is not None:
                approver.join(timeout=10)
        final = client.get(f"/v1/jobs/{job_id}")
        body = final.json()
        JobStatus.model_validate(body)
        if episode is not None:
            from driftsense.contracts import TraceEvent

            trace = client.get(f"/v1/jobs/{job_id}/trace")
            events = [TraceEvent.model_validate_json(x) for x in trace.text.splitlines()]
            ep = client.get(f"/v1/jobs/{job_id}/artifacts/episode.json").json()
            summary = {
                "trace_events": len(events),
                "trace_kinds": sorted({e.kind for e in events}),
                "approvals": approvals,
                "episode": {k: ep.get(k) for k in ("outcome", "steps", "dose", "final_margin", "approval")},
            }
            ok = (
                final.status_code == 200
                and body["status"] == "succeeded"
                and events[-1].kind == "episode_end"
            )
            return (0 if ok and outcomes else 1), {"job": body, "agent": summary}
        ok = final.status_code == 200 and body["status"] in ("succeeded", "awaiting_approval")
        if not outcomes:
            ok = False
        return (0 if ok else 1), body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m service.e2e_local", description=__doc__)
    parser.add_argument(
        "--demo",
        default=None,
        help=f"demo id (default demo_finfet_easy; with --agent {DEFAULT_AGENT_DEMO}, any vsem-<family>-<seed>)",
    )
    parser.add_argument("--engine", default=None, help="fake|v1|v2 (default: $DRIFTSENSE_ENGINE or fake)")
    parser.add_argument(
        "--agent",
        action="store_true",
        help="run an agent episode (POST /v1/agent/episodes) with an auto-approver that picks rank 1",
    )
    args = parser.parse_args(argv)
    code, body = run(args.demo, args.engine, args.agent)
    print(json.dumps(body, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
