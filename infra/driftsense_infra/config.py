"""Context-key parsing shared by every stack.

Context values passed on the CLI (``-c key=value``) arrive as strings, values
from ``cdk.json`` may be real booleans/numbers, so everything goes through the
helpers below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from constructs import Node

REGION = "ap-south-1"
PREFIX = "DriftSense"
METRIC_NAMESPACE = "DriftSense"

# v2-freeze engine SHA, pinned by the protocol-v1 tag ("Engine pinned: v2-freeze (...)").
V2_FREEZE_SHA = "f9395b84dd654ff1d537f6817c258a0f2b9373de"
ENGINES = ("v1", "v2", "fake")
API_ASSETS = ("auto", "service", "placeholder")
# Agent episodes block on a human approval for up to this long (service default, settings.py).
DEFAULT_APPROVAL_TIMEOUT_S = 900

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _raw(node: Node, key: str) -> Any:
    return node.try_get_context(key)


def ctx_str(node: Node, key: str, default: Optional[str] = None) -> Optional[str]:
    value = _raw(node, key)
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def ctx_bool(node: Node, key: str, default: bool = False) -> bool:
    value = _raw(node, key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"context '{key}' must be a boolean (true/false), got {value!r}")


def ctx_number(node: Node, key: str, default: float) -> float:
    value = _raw(node, key)
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"context '{key}' must be a number, got {value!r}") from exc


_AMI_RE = re.compile(r"^ami-[0-9a-f]{8,17}$")


def _ami(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if not _AMI_RE.match(value):
        raise ValueError(f"context 'cool_ami_id' must look like ami-0123456789abcdef0, got {value!r}")
    return value


def _azs(value: str) -> tuple[str, ...]:
    azs = tuple(a.strip() for a in value.split(",") if a.strip())
    if not azs or not all(a.startswith(REGION) for a in azs):
        raise ValueError(f"context 'azs' must be a comma-separated list of {REGION}x zones, got {value!r}")
    return azs


@dataclass(frozen=True)
class Settings:
    """All context keys understood by the app (see infra/README.md)."""

    interface_endpoints: bool
    azs: tuple[str, ...]
    bench_az: Optional[str]
    approval_timeout_s: int
    bedrock_inference_profile: Optional[str]
    alert_email: Optional[str]
    engine: str
    api_asset: str
    budget_usd: float
    cool_ami_id: Optional[str]
    engine_git_sha: str
    bedrock: bool
    bedrock_model_id: str
    bench: bool
    bench_large: bool
    bench_run_id: Optional[str]
    bench_max_minutes: int
    root_device_name: str
    github_repo: str
    github_oidc_provider_arn: Optional[str]
    job_ttl_days: int
    log_level: str

    @classmethod
    def from_node(cls, node: Node) -> "Settings":
        api_asset = ctx_str(node, "api_asset", "auto")
        if api_asset not in API_ASSETS:
            raise ValueError(f"context 'api_asset' must be one of {API_ASSETS}, got {api_asset!r}")
        engine = ctx_str(node, "engine", "v2")
        if engine not in ENGINES:
            raise ValueError(f"context 'engine' must be one of {ENGINES}, got {engine!r}")
        approval_timeout_s = int(ctx_number(node, "approval_timeout_s", DEFAULT_APPROVAL_TIMEOUT_S))
        if approval_timeout_s <= 0:
            raise ValueError("context 'approval_timeout_s' must be > 0")
        return cls(
            interface_endpoints=ctx_bool(node, "interface_endpoints", False),
            azs=_azs(ctx_str(node, "azs", "ap-south-1a,ap-south-1b,ap-south-1c")),
            bench_az=ctx_str(node, "bench_az"),
            approval_timeout_s=approval_timeout_s,
            bedrock_inference_profile=ctx_str(node, "bedrock_inference_profile"),
            # alarm_email is accepted as an alias of alert_email.
            alert_email=ctx_str(node, "alert_email") or ctx_str(node, "alarm_email"),
            engine=engine,
            api_asset=api_asset,
            budget_usd=ctx_number(node, "budget_usd", 100.0),
            cool_ami_id=_ami(ctx_str(node, "cool_ami_id")),
            engine_git_sha=ctx_str(node, "engine_git_sha", V2_FREEZE_SHA),
            bedrock=ctx_bool(node, "bedrock", False),
            bedrock_model_id=ctx_str(node, "bedrock_model_id", "amazon.nova-lite-v1:0"),
            bench=ctx_bool(node, "bench", False),
            bench_large=ctx_bool(node, "bench_large", False),
            bench_run_id=ctx_str(node, "bench_run_id"),
            bench_max_minutes=int(ctx_number(node, "bench_max_minutes", 360)),
            root_device_name=ctx_str(node, "root_device_name", "/dev/sda1"),
            github_repo=ctx_str(node, "github_repo", "HUNT-001/DriftSense"),
            github_oidc_provider_arn=ctx_str(node, "github_oidc_provider_arn"),
            job_ttl_days=int(ctx_number(node, "job_ttl_days", 7)),
            log_level=ctx_str(node, "log_level", "INFO"),
        )
