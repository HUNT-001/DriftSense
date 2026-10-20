"""JobStore (DynamoDB) and ArtifactStore (S3). boto3 clients are injectable for tests.

DynamoDB item layout (table PK ``job_id`` (S), TTL attribute ``expires_at`` (N, epoch seconds)):

  job item:    job_id, status, created_at, updated_at, expires_at, contract_version, request (JobCreate JSON),
               engine?, result? (LocalizeResult JSON), approval? (ApprovalRequest JSON), error? (ErrorInfo JSON),
               worker? (WorkerInfo JSON), trace_key?, result_key?, decision? (ApprovalDecision JSON)
  token item:  job_id = "token#<client_token>", target_job_id, request, created_at, expires_at
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from botocore.config import Config
from botocore.exceptions import ClientError

from service.models import CONTRACT_VERSION, JobCreate, JobStatus

TOKEN_PREFIX = "token#"
_JSON_FIELDS = ("result", "approval", "error", "worker")
_SETTABLE = frozenset(
    {"engine", "result", "approval", "error", "worker", "trace_key", "result_key", "decision"}
)


class InvalidTransition(Exception):
    def __init__(self, job_id: str, current: str | None):
        super().__init__(f"job {job_id}: invalid transition from {current!r}")
        self.job_id = job_id
        self.current = current


class IdempotencyConflict(Exception):
    """Same client_token re-used with a different request body."""


def _boto_client(service: str, region: str | None, **cfg: Any):
    import boto3

    return boto3.client(service, region_name=region, config=Config(**cfg) if cfg else None)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _to_attr(value: Any) -> dict:
    if isinstance(value, bool):
        return {"BOOL": value}
    if isinstance(value, (int, float)):
        return {"N": str(value)}
    if hasattr(value, "model_dump_json"):
        return {"S": value.model_dump_json()}
    if isinstance(value, (dict, list)):
        return {"S": json.dumps(value, separators=(",", ":"), sort_keys=True)}
    return {"S": str(value)}


def _from_attr(attr: dict) -> Any:
    if "S" in attr:
        return attr["S"]
    if "N" in attr:
        n = attr["N"]
        return int(n) if n.lstrip("-").isdigit() else float(n)
    if "BOOL" in attr:
        return attr["BOOL"]
    if "NULL" in attr:
        return None
    raise ValueError(f"unsupported attribute {attr!r}")


def _plain(item: dict) -> dict[str, Any]:
    return {k: _from_attr(v) for k, v in item.items()}


def item_to_status(item: dict[str, Any]) -> JobStatus:
    """Build a contract-valid JobStatus from a plain (decoded) DynamoDB item."""
    data: dict[str, Any] = {
        "job_id": item["job_id"],
        "status": item["status"],
        "created_at": item["created_at"],
        "updated_at": item["updated_at"],
    }
    if item.get("engine"):
        data["engine"] = item["engine"]
    if item.get("trace_key"):
        data["trace_key"] = item["trace_key"]
    for name in _JSON_FIELDS:
        if item.get(name):
            data[name] = json.loads(item[name])
    return JobStatus.model_validate(data)


class JobStore:
    def __init__(
        self,
        table_name: str,
        *,
        client=None,
        region: str | None = None,
        ttl_days: int = 7,
        clock: Callable[[], datetime] = _utcnow,
    ):
        self.table = table_name
        self.client = client or _boto_client("dynamodb", region)
        self.ttl_s = ttl_days * 86400
        self.clock = clock

    # ---------- reads ----------
    def get_item(self, job_id: str) -> dict[str, Any] | None:
        resp = self.client.get_item(TableName=self.table, Key={"job_id": {"S": job_id}}, ConsistentRead=True)
        item = resp.get("Item")
        return _plain(item) if item else None

    def get(self, job_id: str) -> JobStatus | None:
        if job_id.startswith(TOKEN_PREFIX):
            return None
        item = self.get_item(job_id)
        if not item or "status" not in item:
            return None
        return item_to_status(item)

    def get_request(self, item: dict[str, Any]) -> JobCreate:
        return JobCreate.model_validate_json(item["request"])

    # ---------- writes ----------
    def create(self, req: JobCreate, *, job_id: str | None = None) -> tuple[JobStatus, bool]:
        """Create a queued job. Returns (status, created). Idempotent on ``req.client_token``."""
        job_id = job_id or uuid.uuid4().hex
        now = self.clock()
        expires = int(now.timestamp()) + self.ttl_s
        request_json = req.model_dump_json()
        job_item = {
            "job_id": {"S": job_id},
            "status": {"S": "queued"},
            "created_at": {"S": _iso(now)},
            "updated_at": {"S": _iso(now)},
            "expires_at": {"N": str(expires)},
            "contract_version": {"S": CONTRACT_VERSION},
            "request": {"S": request_json},
        }
        put_job = {
            "Put": {
                "TableName": self.table,
                "Item": job_item,
                "ConditionExpression": "attribute_not_exists(job_id)",
            }
        }
        if not req.client_token:
            self.client.transact_write_items(TransactItems=[put_job])
            return item_to_status(_plain(job_item)), True

        token_key = TOKEN_PREFIX + req.client_token
        token_item = {
            "job_id": {"S": token_key},
            "target_job_id": {"S": job_id},
            "request": {"S": request_json},
            "created_at": {"S": _iso(now)},
            "expires_at": {"N": str(expires)},
        }
        put_token = {
            "Put": {
                "TableName": self.table,
                "Item": token_item,
                "ConditionExpression": "attribute_not_exists(job_id)",
            }
        }
        try:
            self.client.transact_write_items(TransactItems=[put_token, put_job])
            return item_to_status(_plain(job_item)), True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in (
                "TransactionCanceledException",
                "ConditionalCheckFailedException",
            ):
                raise
        existing_token = self.get_item(token_key)
        if existing_token is None:  # pragma: no cover - token expired between the two calls
            raise RuntimeError("idempotency token vanished; retry")
        stored = json.loads(existing_token["request"])
        if stored != json.loads(request_json):
            raise IdempotencyConflict(req.client_token)
        status = self.get(existing_token["target_job_id"])
        if status is None:  # pragma: no cover - job item expired before its token
            raise RuntimeError("idempotent job not found; retry")
        return status, False

    def transition(
        self,
        job_id: str,
        to: str,
        *,
        allowed_from: Iterable[str],
        set_fields: dict[str, Any] | None = None,
        remove_fields: Iterable[str] = (),
    ) -> JobStatus:
        """Conditionally move ``job_id`` to ``to``. Raises InvalidTransition if the current status is not allowed."""
        set_fields = dict(set_fields or {})
        remove_fields = list(remove_fields)
        unknown = (set(set_fields) | set(remove_fields)) - _SETTABLE
        if unknown:
            raise ValueError(f"cannot set {sorted(unknown)}")
        names = {"#s": "status", "#u": "updated_at", "#k": "job_id"}
        values: dict[str, dict] = {":to": {"S": to}, ":now": {"S": _iso(self.clock())}}
        sets = ["#s = :to", "#u = :now"]
        removes = []
        for i, (name, value) in enumerate(set_fields.items()):
            names[f"#f{i}"] = name
            if value is None:
                removes.append(f"#f{i}")
            else:
                values[f":v{i}"] = _to_attr(value)
                sets.append(f"#f{i} = :v{i}")
        for j, name in enumerate(remove_fields):
            names[f"#r{j}"] = name
            removes.append(f"#r{j}")
        allowed = list(allowed_from)
        for i, st in enumerate(allowed):
            values[f":a{i}"] = {"S": st}
        expr = "SET " + ", ".join(sets) + (" REMOVE " + ", ".join(removes) if removes else "")
        cond = "attribute_exists(#k) AND #s IN (" + ", ".join(f":a{i}" for i in range(len(allowed))) + ")"
        try:
            resp = self.client.update_item(
                TableName=self.table,
                Key={"job_id": {"S": job_id}},
                UpdateExpression=expr,
                ConditionExpression=cond,
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=values,
                ReturnValues="ALL_NEW",
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                current = self.get_item(job_id)
                raise InvalidTransition(job_id, current.get("status") if current else None) from exc
            raise
        return item_to_status(_plain(resp["Attributes"]))


class ArtifactStore:
    def __init__(self, bucket: str, *, client=None, region: str | None = None):
        self.bucket = bucket
        # SigV4 so presigned PUTs bind Content-Type/Content-Length in X-Amz-SignedHeaders; virtual-hosted
        # regional URLs so browsers are not redirected (a 307 would break the CORS preflight).
        self.client = client or _boto_client(
            "s3", region, signature_version="s3v4", s3={"addressing_style": "virtual"}
        )

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        return key

    def put_json(self, key: str, payload: Any) -> str:
        if hasattr(payload, "model_dump_json"):
            body = payload.model_dump_json(indent=2)
        else:
            body = json.dumps(payload, indent=2, sort_keys=True)
        return self.put_bytes(key, body.encode("utf-8"), "application/json")

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def head(self, key: str) -> tuple[int, str | None] | None:
        """(size_bytes, stored content type) or None if the object does not exist."""
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        return int(resp["ContentLength"]), resp.get("ContentType")

    def presign_get(self, key: str, expires_in: int, content_type: str | None = None) -> str:
        params = {"Bucket": self.bucket, "Key": key}
        if content_type:
            params["ResponseContentType"] = content_type
        return self.client.generate_presigned_url(
            "get_object", Params=params, ExpiresIn=expires_in, HttpMethod="GET"
        )

    def presign_put(self, key: str, content_type: str, size_bytes: int, expires_in: int) -> str:
        return self.client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "ContentType": content_type,
                "ContentLength": size_bytes,
            },
            ExpiresIn=expires_in,
            HttpMethod="PUT",
        )
