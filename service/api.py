"""DriftSense job API: FastAPI app, wrapped by Mangum for Lambda (``service.api.handler``).

The API never imports cv2/numpy at module import (Lambda may not ship them); /v1/version probes lazily.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import sys
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from mangum import Mangum
from starlette.exceptions import HTTPException as StarletteHTTPException

from service import SERVICE_VERSION
from service.demos import get_demo, list_demos
from service.models import (
    ALLOWED_CONTENT_TYPES,
    CONTRACT_VERSION,
    JOB_ID_PATTERN,
    PRESIGN_EXPIRES_S,
    UPLOAD_KEY_PATTERN,
    ApprovalDecision,
    DemoInfo,
    ErrorInfo,
    Health,
    JobCreate,
    JobStatus,
    LocalizeResult,
    UploadRequest,
    UploadResponse,
    VersionInfo,
)
from service.queue import JobQueue
from service.settings import Settings
from service.storage import (
    ArtifactStore,
    IdempotencyConflict,
    InvalidTransition,
    JobStore,
)

log = logging.getLogger("driftsense.api")

_JOB_ID_RE = re.compile(JOB_ID_PATTERN)
_UPLOAD_KEY_RE = re.compile(UPLOAD_KEY_PATTERN)
_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
MAX_CLIENT_TOKEN_LEN = 128
# Lambda's synchronous response limit is 6 MB (base64 inflates binary bodies): above this, redirect to S3.
MAX_INLINE_BYTES = 5 * 1024 * 1024
DOWNLOAD_PRESIGN_S = 300
ARTIFACT_CACHE_CONTROL = "private, max-age=300"
TRACE_CACHE_CONTROL = "private, no-cache"  # traces grow while an agent episode runs
_CONTENT_TYPES = {
    "png": "image/png",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "svg": "image/svg+xml",
    "csv": "text/csv",
    "txt": "text/plain; charset=utf-8",
}


def content_type_for(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message


@dataclass
class Services:
    settings: Settings
    jobs: JobStore
    artifacts: ArtifactStore
    queue: JobQueue

    @classmethod
    def from_settings(cls, settings: Settings) -> Services:
        settings.require_aws()
        region = settings.aws_region
        return cls(
            settings,
            JobStore(settings.jobs_table, region=region, ttl_days=settings.job_ttl_days),
            ArtifactStore(settings.data_bucket, region=region),
            JobQueue(settings.jobs_queue_url, region=region),
        )


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=ErrorInfo(code=code, message=message).model_dump())


def _job_response(status: JobStatus, status_code: int = 200) -> JSONResponse:
    # Round-trip through the contract model so every JobStatus response is guaranteed valid.
    body = JobStatus.model_validate(status.model_dump()).model_dump(mode="json")
    return JSONResponse(status_code=status_code, content=body)


def _s3_download(
    artifacts: ArtifactStore, key: str, content_type: str, cache_control: str, missing: ApiError
) -> Response:
    """Inline body for small objects; 307 to a short-lived presigned GET above MAX_INLINE_BYTES."""
    head = artifacts.head(key)
    if head is None:
        raise missing
    size, _stored_type = head
    if size > MAX_INLINE_BYTES:
        url = artifacts.presign_get(key, DOWNLOAD_PRESIGN_S, content_type)
        return RedirectResponse(url, status_code=307, headers={"Cache-Control": "no-store"})
    body = artifacts.get_bytes(key)
    if body is None:
        raise missing
    return Response(content=body, media_type=content_type, headers={"Cache-Control": cache_control})


def _arch() -> str:
    m = platform.machine().lower()
    return {"amd64": "x86_64", "x64": "x86_64"}.get(m, m or "unknown")


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or (services.settings if services else Settings.from_env())
    app = FastAPI(title="DriftSense job API", version=SERVICE_VERSION)
    app.state.settings = settings
    app.state.services = services

    if settings.ui_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[settings.ui_origin],
            allow_methods=["GET", "POST"],
            allow_headers=["content-type"],
            max_age=600,
        )

    def svc() -> Services:
        if app.state.services is None:  # lazily, so cold starts and /healthz never touch AWS
            app.state.services = Services.from_settings(app.state.settings)
        return app.state.services

    # ----- error shapes: always ErrorInfo -----
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return _error(422, "validation_error", "; ".join(parts) or "invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail))

    # ----- endpoints -----
    @app.get("/healthz", response_model=Health)
    def healthz() -> Health:
        return Health(ok=True)

    @app.get("/v1/version", response_model=VersionInfo)
    def version() -> VersionInfo:
        cv_ver = cv_file = np_ver = None
        try:
            import cv2

            cv_ver, cv_file = cv2.__version__, cv2.__file__
        except (ImportError, OSError) as exc:  # Lambda may not ship cv2
            log.debug("cv2 unavailable: %r", exc)
        try:
            import numpy

            np_ver = numpy.__version__
        except (ImportError, OSError) as exc:
            log.debug("numpy unavailable: %r", exc)
        return VersionInfo(
            service_version=SERVICE_VERSION,
            git_sha=settings.engine_git_sha,
            contract_version=CONTRACT_VERSION,
            engine=settings.engine,
            opencv_version=cv_ver,
            opencv_file=cv_file,
            numpy_version=np_ver,
            python_version=sys.version.split()[0],
            arch=_arch(),
            platform=platform.platform(),
        )

    @app.get("/v1/demos", response_model=list[DemoInfo])
    def demos() -> list[DemoInfo]:
        return list_demos()

    @app.post("/v1/uploads", response_model=UploadResponse)
    def create_upload(body: UploadRequest) -> UploadResponse:
        s = svc()
        ext = ALLOWED_CONTENT_TYPES.get(body.content_type)
        if ext is None:
            raise ApiError(
                415, "unsupported_media_type", f"content_type must be one of {sorted(ALLOWED_CONTENT_TYPES)}"
            )
        if body.size_bytes > s.settings.max_upload_bytes:
            raise ApiError(413, "too_large", f"size_bytes exceeds MAX_UPLOAD_MB={s.settings.max_upload_mb}")
        stem, _, suffix = body.filename.rpartition(".")
        stem = (stem or suffix).rsplit("/", 1)[-1].lower()
        role = body.role or (stem if stem in ("template", "search") else None)
        if role is None:
            raise ApiError(
                422, "validation_error", "role must be 'template' or 'search' (or use that filename)"
            )
        if ext == "tif" and suffix.lower() in ("tif", "tiff"):
            ext = suffix.lower()
        upload_id = body.upload_id or uuid.uuid4().hex
        key = f"uploads/{upload_id}/{role}.{ext}"
        url = s.artifacts.presign_put(key, body.content_type, body.size_bytes, PRESIGN_EXPIRES_S)
        return UploadResponse(
            upload_id=upload_id,
            key=key,
            url=url,
            headers={"Content-Type": body.content_type, "Content-Length": str(body.size_bytes)},
            expires_in=PRESIGN_EXPIRES_S,
            max_bytes=s.settings.max_upload_bytes,
        )

    @app.post("/v1/jobs", status_code=202, responses={200: {"description": "idempotent replay"}})
    def create_job(body: JobCreate) -> JSONResponse:
        s = svc()
        if body.demo_id and get_demo(body.demo_id) is None:
            raise ApiError(422, "validation_error", f"unknown demo_id {body.demo_id!r}")
        if not body.demo_id:
            for name, key in (("template_key", body.template_key), ("search_key", body.search_key)):
                if not _UPLOAD_KEY_RE.match(key or ""):
                    raise ApiError(
                        422, "validation_error", f"{name} must be an uploads/<id>/<role>.<ext> key"
                    )
        if body.client_token is not None and not (0 < len(body.client_token) <= MAX_CLIENT_TOKEN_LEN):
            raise ApiError(422, "validation_error", f"client_token must be 1..{MAX_CLIENT_TOKEN_LEN} chars")
        try:
            status, created = s.jobs.create(body)
        except IdempotencyConflict:
            raise ApiError(
                409, "idempotency_conflict", "client_token was already used with a different request"
            ) from None
        if not created:
            return _job_response(status, 200)
        try:
            s.queue.send(status.job_id)
        except Exception as exc:
            log.exception("enqueue failed for %s", status.job_id)
            try:
                s.jobs.transition(
                    status.job_id,
                    "failed",
                    allowed_from=("queued",),
                    set_fields={"error": ErrorInfo(code="enqueue_failed", message=str(exc)[:500])},
                )
            except InvalidTransition:
                pass
            raise ApiError(503, "enqueue_failed", "could not enqueue the job; retry with a new client_token") from exc
        return _job_response(status, 202)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> JSONResponse:
        status = svc().jobs.get(job_id) if _JOB_ID_RE.match(job_id) else None
        if status is None:
            raise ApiError(404, "not_found", f"job {job_id} not found")
        return _job_response(status)

    @app.get("/v1/jobs/{job_id}/artifacts/{name}", response_model=None)
    def get_artifact(job_id: str, name: str) -> Response:
        """Stream jobs/<id>/<name> (artifact URIs resolve here by basename)."""
        if "/" in name or "\\" in name or ".." in name or not _ARTIFACT_NAME_RE.match(name):
            raise ApiError(
                400, "bad_request", "artifact name must match ^[A-Za-z0-9._-]{1,128}$ without '..'"
            )
        s = svc()
        status = s.jobs.get(job_id) if _JOB_ID_RE.match(job_id) else None
        if status is None:
            raise ApiError(404, "not_found", f"job {job_id} not found")
        return _s3_download(
            s.artifacts,
            f"jobs/{job_id}/{name}",
            content_type_for(name),
            ARTIFACT_CACHE_CONTROL,
            ApiError(404, "not_found", f"artifact {name} not found for job {job_id}"),
        )

    @app.get("/v1/jobs/{job_id}/trace", response_model=None)
    def get_trace(job_id: str) -> Response:
        """The agent trace (TraceEvent JSONL) once the job has a trace_key."""
        s = svc()
        status = s.jobs.get(job_id) if _JOB_ID_RE.match(job_id) else None
        if status is None:
            raise ApiError(404, "not_found", f"job {job_id} not found")
        not_ready = ApiError(404, "not_ready", f"no trace for job {job_id} yet")
        key = status.trace_key
        if not key or not key.startswith("traces/") or ".." in key:
            raise not_ready
        return _s3_download(s.artifacts, key, "application/x-ndjson", TRACE_CACHE_CONTROL, not_ready)

    @app.post("/v1/jobs/{job_id}/approval")
    def approve(job_id: str, decision: ApprovalDecision) -> JSONResponse:
        s = svc()
        status = s.jobs.get(job_id) if _JOB_ID_RE.match(job_id) else None
        if status is None:
            raise ApiError(404, "not_found", f"job {job_id} not found")
        if status.status != "awaiting_approval" or status.approval is None:
            raise ApiError(409, "invalid_state", f"job is {status.status}, not awaiting_approval")
        fields: dict[str, Any] = {"decision": decision}
        if decision.chosen_rank is None:
            to = "failed"
            note = f": {decision.note}" if decision.note else ""
            fields["error"] = ErrorInfo(
                code="rejected_by_operator",
                message=f"all candidates rejected by {decision.approver}{note}"[:1000],
            )
        else:
            options = {c.rank: c for c in status.approval.options}
            if decision.chosen_rank not in options:
                raise ApiError(
                    422, "validation_error", f"chosen_rank must be one of {sorted(options)} or null"
                )
            to = "succeeded"
            if status.result is not None:
                fields["result"] = _apply_choice(status.result, options[decision.chosen_rank], decision)
        try:
            new_status = s.jobs.transition(job_id, to, allowed_from=("awaiting_approval",), set_fields=fields)
        except InvalidTransition as exc:
            raise ApiError(409, "invalid_state", f"job is {exc.current}, not awaiting_approval") from None
        try:
            s.queue.send(job_id, resume=True)
        except Exception:
            log.exception("resume enqueue failed for %s", job_id)
        return _job_response(new_status)

    return app


def _apply_choice(result: LocalizeResult, chosen, decision: ApprovalDecision) -> LocalizeResult:
    """Record the operator's choice: offset := chosen candidate, plus an audit warning."""
    offset = result.offset.model_copy(
        update={"x_px": chosen.x_px, "y_px": chosen.y_px, "x_nm": None, "y_nm": None}
    )
    if result.offset.pixel_size_nm:
        px = result.offset.pixel_size_nm
        offset = offset.model_copy(update={"x_nm": chosen.x_px * px, "y_nm": chosen.y_px * px})
    audit = json.dumps(
        {"operator_choice": {"rank": chosen.rank, "approver": decision.approver}}, separators=(",", ":")
    )
    return result.model_copy(update={"offset": offset, "warnings": [*result.warnings, audit]})


app = create_app()
handler = Mangum(app, lifespan="off")
