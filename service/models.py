"""Re-exports of the frozen contract models plus API-only request/response models.

Contract models are NEVER redefined here: they come from ``driftsense.contracts`` (owned by Person A).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from driftsense.contracts import (  # noqa: F401  (re-exports)
    CONTRACT_VERSION,
    ApprovalDecision,
    ApprovalRequest,
    Artifact,
    Budget,
    Candidate,
    DriftSenseInputError,
    ErrorInfo,
    JobCreate,
    JobStatus,
    LocalizeResult,
    Margin,
    Mode,
    Offset,
    WorkerInfo,
)

ALLOWED_CONTENT_TYPES = {"image/png": "png", "image/tiff": "tif"}
UPLOAD_ROLES = ("template", "search")
PRESIGN_EXPIRES_S = 15 * 60

JOB_ID_PATTERN = r"^[0-9a-f]{32}$"
UPLOAD_KEY_PATTERN = r"^uploads/[0-9a-f]{32}/(template|search)\.(png|tif|tiff)$"
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})


class _ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UploadRequest(_ApiModel):
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Client file name; its stem selects the role if `role` is omitted",
    )
    content_type: str = Field(..., description="image/png or image/tiff")
    size_bytes: int = Field(..., gt=0, description="Exact byte size of the file that will be PUT")
    role: Literal["template", "search"] | None = Field(
        None, description="template|search; defaults to the filename stem when it is one of those"
    )
    upload_id: str | None = Field(
        None, pattern=JOB_ID_PATTERN, description="Reuse an upload_id so template and search share a prefix"
    )


class UploadResponse(_ApiModel):
    upload_id: str
    key: str
    url: str
    method: Literal["PUT"] = "PUT"
    headers: dict[str, str] = Field(..., description="Headers the PUT must send exactly (signed)")
    expires_in: int
    max_bytes: int


class DemoInfo(_ApiModel):
    demo_id: str
    title: str
    lattice: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str = ""


class VersionInfo(_ApiModel):
    service_version: str
    git_sha: str
    contract_version: str
    engine: str | None = None
    opencv_version: str | None = None
    opencv_file: str | None = None
    numpy_version: str | None = None
    python_version: str
    arch: str
    platform: str


class Health(_ApiModel):
    ok: bool = True
