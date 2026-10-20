"""End-to-end flow in-process under moto: ``python -m service.e2e_local [--demo ID] [--engine fake|v1|v2]``.

Creates table/queue(+DLQ)/bucket, POSTs a demo job through the FastAPI TestClient, runs one worker
iteration, GETs the job and prints the final JobStatus JSON. Exit 0 on succeeded/awaiting_approval.
Dev-only: needs moto + httpx (never imported by the API or the worker).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import warnings

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


def run(
    demo_id: str = "demo_finfet_easy", engine: str | None = None, agent: bool = False
) -> tuple[int, dict]:
    for key, value in LOCAL_ENV.items():
        os.environ.setdefault(key, value)
    if engine:
        os.environ["DRIFTSENSE_ENGINE"] = engine

    with warnings.catch_warnings():  # starlette's "use httpx2" notice is noise for a dev script
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient
    from moto import mock_aws

    from service.api import Services, create_app
    from service.models import JobStatus
    from service.settings import Settings
    from service.worker import Worker

    with mock_aws():
        env = os.environ
        queue_url = provision(env["AWS_REGION"], env["JOBS_TABLE"], env["DATA_BUCKET"])
        os.environ["JOBS_QUEUE_URL"] = queue_url
        settings = Settings.from_env()
        services = Services.from_settings(settings)
        client = TestClient(create_app(services=services))
        resp = client.post("/v1/jobs", json={"demo_id": demo_id, "agent": agent, "client_token": "e2e-local"})
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
        outcomes = worker.poll_once(wait_seconds=0)
        final = client.get(f"/v1/jobs/{job_id}")
        body = final.json()
        JobStatus.model_validate(body)
        ok = final.status_code == 200 and body["status"] in ("succeeded", "awaiting_approval")
        if not outcomes:
            ok = False
        return (0 if ok else 1), body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m service.e2e_local", description=__doc__)
    parser.add_argument("--demo", default="demo_finfet_easy")
    parser.add_argument("--engine", default=None, help="fake|v1|v2 (default: $DRIFTSENSE_ENGINE or fake)")
    parser.add_argument("--agent", action="store_true", help="agent=true: escalations wait for approval")
    args = parser.parse_args(argv)
    code, body = run(args.demo, args.engine, args.agent)
    print(json.dumps(body, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
