"""Environment parsing for the API and the worker. No credentials live here: boto3 resolves them."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

ENGINES = ("fake", "v1", "v2")
METRICS_SINKS = ("stdout", "cloudwatch", "both")

# The service runs v2 unless DRIFTSENSE_ENGINE says otherwise. driftsense.api.DEFAULT_ENGINE is still
# "v1" until A flips it, so the service must not rely on it. Tests set DRIFTSENSE_ENGINE=fake.
SERVICE_DEFAULT_ENGINE = "v2"

# ---------- Agent episode operating point ----------
# Copied verbatim from eval/agent_eval.py (TAU, SITE_NM, PIXEL_NM, REF_FOV_NM, REF_DWELL_US, BASE_FOV_NM,
# BASE_DWELL_US, TEMPLATE_FRAC, MAX_STEPS; the main G5 evaluation runs with no dose cap, max_dose=None).
# eval/ is not an installed package, so it is not imported. Keep these in sync with that file: the
# service then runs exactly the episodes behind results/agent/g5_episode.json.
AGENT_TAU = 0.10
AGENT_SITE_NM = 2000.0
AGENT_PIXEL_NM = 1.0
AGENT_REF_FOV_NM = 240.0
AGENT_REF_DWELL_US = 12.0
AGENT_BASE_FOV_NM = 500.0
AGENT_BASE_DWELL_US = 1.0
AGENT_TEMPLATE_FRAC = 0.40
AGENT_MAX_STEPS = 6
AGENT_MAX_DOSE: float | None = None
# driftsense.agent.runner default per-tool timeout (acquire/localize/refine).
AGENT_TOOL_TIMEOUT_S = 15.0

APPROVAL_TIMEOUT_S = 900  # 15 min operator window
APPROVAL_POLL_S = 2.0
TRACE_UPLOAD_S = 2.0


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _float(env: Mapping[str, str], name: str, default: float | None) -> float | None:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not value > 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _opt(env: Mapping[str, str], name: str) -> str | None:
    raw = env.get(name, "").strip()
    return raw or None


@dataclass(frozen=True)
class Settings:
    jobs_table: str = ""
    jobs_queue_url: str = ""
    data_bucket: str = ""
    ui_origin: str | None = None
    max_upload_mb: int = 20
    engine: str = SERVICE_DEFAULT_ENGINE
    engine_git_sha: str = "unknown"
    aws_region: str | None = None
    job_ttl_days: int = 7
    metrics_sink: str = "stdout"  # stdout (EMF) | cloudwatch (PutMetricData) | both
    # agent episodes
    approval_timeout_s: float = APPROVAL_TIMEOUT_S
    approval_poll_s: float = APPROVAL_POLL_S
    trace_upload_s: float = TRACE_UPLOAD_S
    agent_tau: float = AGENT_TAU
    agent_max_steps: int = AGENT_MAX_STEPS
    agent_max_dose: float | None = AGENT_MAX_DOSE
    agent_tool_timeout_s: float = AGENT_TOOL_TIMEOUT_S

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        engine = _opt(env, "DRIFTSENSE_ENGINE") or SERVICE_DEFAULT_ENGINE
        if engine not in ENGINES:
            raise ValueError(f"DRIFTSENSE_ENGINE must be one of {ENGINES}, got {engine!r}")
        sink = (_opt(env, "METRICS_SINK") or "stdout").lower()
        if sink not in METRICS_SINKS:
            raise ValueError(f"METRICS_SINK must be one of {METRICS_SINKS}, got {sink!r}")
        return cls(
            jobs_table=env.get("JOBS_TABLE", "").strip(),
            jobs_queue_url=env.get("JOBS_QUEUE_URL", "").strip(),
            data_bucket=env.get("DATA_BUCKET", "").strip(),
            ui_origin=_opt(env, "UI_ORIGIN"),
            max_upload_mb=_int(env, "MAX_UPLOAD_MB", 20),
            engine=engine,
            engine_git_sha=_opt(env, "ENGINE_GIT_SHA") or "unknown",
            aws_region=_opt(env, "AWS_REGION") or _opt(env, "AWS_DEFAULT_REGION"),
            job_ttl_days=_int(env, "JOB_TTL_DAYS", 7),
            metrics_sink=sink,
            approval_timeout_s=_float(env, "APPROVAL_TIMEOUT_S", APPROVAL_TIMEOUT_S),
            approval_poll_s=_float(env, "APPROVAL_POLL_S", APPROVAL_POLL_S),
            trace_upload_s=_float(env, "TRACE_UPLOAD_S", TRACE_UPLOAD_S),
            agent_tau=_float(env, "AGENT_TAU", AGENT_TAU),
            agent_max_steps=_int(env, "AGENT_MAX_STEPS", AGENT_MAX_STEPS),
            agent_max_dose=_float(env, "AGENT_MAX_DOSE", AGENT_MAX_DOSE),
            agent_tool_timeout_s=_float(env, "AGENT_TOOL_TIMEOUT_S", AGENT_TOOL_TIMEOUT_S),
        )

    @property
    def effective_tool_timeout_s(self) -> float:
        """Per-tool timeout handed to run_episode.

        The runner applies ONE timeout to every tool, including request_human_approval, which blocks on
        the operator. It must therefore outlast the approval window, or a slow operator aborts the episode.
        """
        return max(self.agent_tool_timeout_s, self.approval_timeout_s + 60.0)

    def require_aws(self) -> None:
        missing = [
            n
            for n, v in (
                ("JOBS_TABLE", self.jobs_table),
                ("JOBS_QUEUE_URL", self.jobs_queue_url),
                ("DATA_BUCKET", self.data_bucket),
            )
            if not v
        ]
        if missing:
            raise RuntimeError(f"missing required env vars: {', '.join(missing)}")
