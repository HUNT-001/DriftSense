"""DriftSense public contracts. CHANGING THIS FILE REQUIRES A shared/* BRANCH AND BOTH APPROVALS.
Bump CONTRACT_VERSION (semver) on any change; minor = additive optional fields only."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "1.0.0"


class DriftSenseInputError(ValueError):
    """Raised for undecodable images, template larger than search, empty input."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Mode(str, Enum):
    FAST = "fast"  # lattice + roughness, no cascade
    ACCURATE = "accurate"  # full pipeline incl. subpixel
    DENSE = "dense"  # dense candidate scan (recall@20 path)
    AUTO = "auto"  # engine decides (cascade if margin low)


class Budget(_Model):
    max_ms: int = Field(2000, ge=1, description="Soft wall-clock budget for one localize call")
    max_candidates: int = Field(20, ge=1, le=200)
    tau: float | None = Field(
        None, description="Override calibrated uniqueness-margin threshold; None = calibrated"
    )


class LatticeType(str, Enum):
    FINFET = "finfet"
    DRAM = "dram"
    GRID = "grid"
    STAGGERED = "staggered"
    UNKNOWN = "unknown"


class Offset(_Model):
    """Template top-left location in search-image coordinates (x right, y down), subpixel."""

    x_px: float
    y_px: float
    x_nm: float | None = None
    y_nm: float | None = None
    pixel_size_nm: float | None = None


class Candidate(_Model):
    rank: int = Field(..., ge=1)
    x_px: float
    y_px: float
    score: float = Field(..., description="Final fused score, higher is better")
    lattice_score: float | None = None
    roughness_score: float | None = None


class Margin(_Model):
    margin: float = Field(..., description="Calibrated uniqueness margin between rank-1 and rank-2")
    tau: float
    calibrated: bool
    p_correct: float | None = Field(None, ge=0, le=1)
    decision: Literal["commit", "escalate"]

    @model_validator(mode="after")
    def _decision_matches(self) -> Margin:
        if (self.decision == "commit") != (self.margin >= self.tau):
            raise ValueError("decision must be 'commit' iff margin >= tau")
        return self


class StageTiming(_Model):
    stage: str = Field(
        ...,
        description="e.g. ingest, lattice, rectify, roughness, score, cascade, contour, confidence, subpixel, explain",
    )
    ms: float = Field(..., ge=0)


class Artifact(_Model):
    kind: Literal["heatmap", "overlay", "contour", "candidates", "json"]
    uri: str = Field(..., description="Relative path locally, s3:// key in service")
    sha256: str | None = None


class LocalizeResult(_Model):
    contract_version: str = CONTRACT_VERSION
    engine: Literal["fake", "v1", "v2"]
    engine_version: str
    mode: Mode
    lattice_type: LatticeType = LatticeType.UNKNOWN
    offset: Offset
    candidates: list[Candidate] = Field(..., min_length=1)
    margin: Margin
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = Field(..., ge=0)
    artifacts: list[Artifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ranks(self) -> LocalizeResult:
        if [c.rank for c in self.candidates] != list(range(1, len(self.candidates) + 1)):
            raise ValueError("candidate ranks must be 1..n in order")
        return self


# ---------- Agent trace (one JSON object per line, file: traces/<run_id>.jsonl) ----------
class TraceEvent(_Model):
    v: Literal["1"] = "1"
    run_id: str
    step: int = Field(..., ge=0)
    ts: datetime
    kind: Literal[
        "observe",
        "tool_call",
        "tool_result",
        "decision",
        "commit_move",
        "escalate",
        "human_approval",
        "error",
        "episode_end",
    ]
    tool: str | None = None
    args: dict = Field(default_factory=dict)
    result_summary: dict = Field(default_factory=dict)
    margin: float | None = None
    tau: float | None = None
    because: str | None = Field(None, description="Human-readable reason, e.g. 'margin 0.31 >= tau 0.22'")
    stage_move_nm: tuple[float, float] | None = None
    latency_ms: float | None = None
    policy: Literal["rules", "llm"] | None = None


# ---------- Service job API ----------
class JobCreate(_Model):
    demo_id: str | None = None
    template_key: str | None = None
    search_key: str | None = None
    mode: Mode = Mode.AUTO
    budget: Budget = Budget()
    agent: bool = False
    client_token: str | None = Field(None, description="Idempotency key")

    @model_validator(mode="after")
    def _source(self) -> JobCreate:
        if not self.demo_id and not (self.template_key and self.search_key):
            raise ValueError("provide demo_id or both template_key and search_key")
        return self


class WorkerInfo(_Model):
    instance_type: str | None = None
    arch: Literal["x86_64", "aarch64", "arm64", "unknown"] = "unknown"
    opencv_version: str
    numpy_version: str
    cool: bool = False


class ApprovalRequest(_Model):
    options: list[Candidate] = Field(..., min_length=1, max_length=3)
    margin: Margin


class ApprovalDecision(_Model):
    chosen_rank: int | None = Field(None, ge=1, le=3, description="None = reject all")
    approver: str
    note: str | None = None


class ErrorInfo(_Model):
    code: str
    message: str


class JobStatus(_Model):
    job_id: str
    status: Literal["queued", "running", "awaiting_approval", "succeeded", "failed"]
    created_at: datetime
    updated_at: datetime
    engine: Literal["fake", "v1", "v2"] | None = None
    result: LocalizeResult | None = None
    trace_key: str | None = None
    approval: ApprovalRequest | None = None
    error: ErrorInfo | None = None
    worker: WorkerInfo | None = None


# ---------- results/*.json ----------
class EnvInfo(_Model):
    python: str
    opencv: str
    numpy: str
    platform: str
    arch: str
    instance_type: str | None = None
    cool: bool = False


class Metric(_Model):
    value: float
    unit: str
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None


class ResultsFile(_Model):
    schema_id: Literal["driftsense.results/1"] = "driftsense.results/1"
    name: str
    produced_by: str = Field(..., description="Script path that generated this file")
    git_sha: str
    created_at: datetime
    dataset: Literal["dev", "sealed", "synthetic", "bench"]
    env: EnvInfo
    metrics: dict[str, Metric]
    protocol_sha256: str | None = None


SCHEMAS = {
    "localize_result": LocalizeResult,
    "trace_event": TraceEvent,
    "job_create": JobCreate,
    "job_status": JobStatus,
    "approval_decision": ApprovalDecision,
    "results_file": ResultsFile,
}


def export_schemas(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        (out_dir / f"{name}.schema.json").write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        )
