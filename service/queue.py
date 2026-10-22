"""Thin SQS wrapper. Message body: ``{"job_id": "<id>"}`` or ``{"job_id": "<id>", "resume": true}``."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Message:
    job_id: str | None
    resume: bool
    receipt_handle: str
    receive_count: int
    message_id: str
    raw_body: str


class JobQueue:
    def __init__(self, queue_url: str, *, client=None, region: str | None = None):
        self.url = queue_url
        if client is None:
            import boto3

            client = boto3.client("sqs", region_name=region)
        self.client = client

    def send(self, job_id: str, *, resume: bool = False) -> str:
        body: dict = {"job_id": job_id}
        if resume:
            body["resume"] = True
        resp = self.client.send_message(QueueUrl=self.url, MessageBody=json.dumps(body))
        return resp["MessageId"]

    def receive(
        self, *, wait_seconds: int = 20, max_messages: int = 1, visibility_timeout: int | None = None
    ) -> list[Message]:
        kwargs: dict = {
            "QueueUrl": self.url,
            "MaxNumberOfMessages": max_messages,
            "WaitTimeSeconds": wait_seconds,
            "MessageSystemAttributeNames": ["ApproximateReceiveCount"],
        }
        if visibility_timeout is not None:
            kwargs["VisibilityTimeout"] = visibility_timeout
        resp = self.client.receive_message(**kwargs)
        out = []
        for m in resp.get("Messages", []):
            try:
                body = json.loads(m["Body"])
                job_id = body.get("job_id") if isinstance(body, dict) else None
                resume = bool(body.get("resume", False)) if isinstance(body, dict) else False
            except (ValueError, TypeError):
                job_id, resume = None, False
            count = int(m.get("Attributes", {}).get("ApproximateReceiveCount", "1"))
            out.append(
                Message(
                    job_id=job_id if isinstance(job_id, str) else None,
                    resume=resume,
                    receipt_handle=m["ReceiptHandle"],
                    receive_count=count,
                    message_id=m["MessageId"],
                    raw_body=m["Body"],
                )
            )
        return out

    def delete(self, receipt_handle: str) -> None:
        self.client.delete_message(QueueUrl=self.url, ReceiptHandle=receipt_handle)

    def change_visibility(self, receipt_handle: str, timeout_s: int) -> None:
        self.client.change_message_visibility(
            QueueUrl=self.url, ReceiptHandle=receipt_handle, VisibilityTimeout=timeout_s
        )
