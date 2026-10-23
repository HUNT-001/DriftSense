"""FastMCP server exposing the agent tools over stdio (optional).

The FastMCP import is GUARDED: this module imports and the tool registry works
with ``mcp`` not installed -- only :func:`create_mcp_server` / :func:`main`
require it. The G5 smoke test therefore exercises the real tool registry and one
``localize`` directly (:func:`smoke`), without needing a live stdio transport.

Enabling a real stdio server (and CI-level MCP Inspector testing) needs ``mcp``
added to ``requirements.in`` on a ``shared/*`` branch (``make lock`` + both
approvals) -- out of scope for this A-owned PR; see the PR handoff and
``driftsense/agent/README.md``.

Run (once ``mcp`` is available)::

    python -m driftsense.agent.mcp_server            # stdio, default twin
    npx @modelcontextprotocol/inspector python -m driftsense.agent.mcp_server
"""

from __future__ import annotations

from typing import Any

# The arg models must be module-level: with `from __future__ import annotations`
# FastMCP resolves each tool's string annotations against this module's globals.
from driftsense.agent.tools import (
    TOOL_SCHEMAS,
    AcquireArgs,
    ApprovalArgs,
    CommitStageMoveArgs,
    LocalizeArgs,
    RefineArgs,
    ToolRegistry,
    tool_json_schemas,
)
from driftsense.agent.trace import TraceWriter
from driftsense.sim import VirtualSEM

TOOL_NAMES: tuple[str, ...] = tuple(TOOL_SCHEMAS.keys())


def build_registry(
    *,
    seed: int = 7,
    family: str = "dram",
    site_size_nm: float = 2000.0,
    pixel_nm: float = 1.0,
    tau: float = 0.10,
    template_frac: float = 0.40,
    trace_path: str | None = None,
    max_dose: float | None = None,
    max_acquisitions: int | None = None,
) -> ToolRegistry:
    """Fresh single-episode registry over a VirtualSEM twin (no ``mcp`` needed)."""
    sem = VirtualSEM(
        seed=seed,
        family=family,
        site_size_nm=site_size_nm,
        pixel_nm=pixel_nm,
        max_dose=max_dose,
        max_acquisitions=max_acquisitions,
    )
    trace = TraceWriter(run_id=f"mcp-{family}-{seed}", path=trace_path)
    return ToolRegistry(sem, trace, tau=tau, template_frac=template_frac)


def list_tools() -> dict[str, dict]:
    """Tool name -> JSON schema (what FastMCP would advertise)."""
    return tool_json_schemas()


def smoke(seed: int = 7, family: str = "dram") -> dict[str, Any]:
    """Offline smoke test: list tools, then acquire a reference + search + localize.

    Returns a small summary used by the G5 acceptance test; requires no ``mcp``.
    """
    reg = build_registry(seed=seed, family=family)
    reg.acquire(center_nm=(0.0, 0.0), fov_nm=240.0, dwell_us=12.0)  # reference template
    reg.acquire(center_nm=(0.0, 0.0), fov_nm=500.0, dwell_us=1.0)  # search frame
    result = reg.localize(mode="standard", rect_hypotheses=2, top_k=5)
    return {
        "tools": list(TOOL_NAMES),
        "n_tools": len(TOOL_NAMES),
        "engine": result.engine,
        "margin": result.margin.margin,
        "decision": result.margin.decision,
        "n_candidates": len(result.candidates),
    }


def create_mcp_server(server_name: str = "driftsense-agent", **registry_kwargs: Any) -> Any:
    """Build a FastMCP server exposing the six agent tools over stdio.

    Imports FastMCP lazily; raises a clear error if ``mcp`` is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only when mcp present
        raise RuntimeError(
            "FastMCP unavailable: `mcp` is not installed. Add it to requirements.in "
            "on a shared/* branch (make lock + both approvals) to enable the stdio server."
        ) from exc

    reg = build_registry(**registry_kwargs)
    mcp = FastMCP(server_name)

    @mcp.tool()
    def acquire(args: AcquireArgs) -> dict:
        return reg.acquire(**args.model_dump())

    @mcp.tool()
    def localize(args: LocalizeArgs) -> dict:
        return reg.localize(**args.model_dump()).model_dump(mode="json")

    @mcp.tool()
    def refine(args: RefineArgs) -> dict:
        return reg.refine(**args.model_dump()).model_dump(mode="json")

    @mcp.tool()
    def commit_stage_move(args: CommitStageMoveArgs) -> dict:
        return reg.commit_stage_move(**args.model_dump())

    @mcp.tool()
    def request_human_approval(args: ApprovalArgs) -> dict:
        return reg.request_human_approval(**args.model_dump()).model_dump(mode="json")

    @mcp.tool()
    def get_trace() -> list[dict]:
        return reg.get_trace()

    return mcp


def main() -> None:  # pragma: no cover - entry point, needs mcp + a live transport
    create_mcp_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
