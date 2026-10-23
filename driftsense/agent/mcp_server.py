"""FastMCP server exposing the agent tools over stdio or streamable HTTP.

The FastMCP import is GUARDED: this module imports and the tool registry works
with ``mcp`` not installed -- only :func:`create_mcp_server` / :func:`main`
require it. The G5 smoke test therefore exercises the real tool registry and one
``localize`` directly (:func:`smoke`), without needing a live transport.

The runner's hard safety gate (never commit below tau) is enforced here too: an
MCP client calling ``commit_stage_move`` without a prior ``localize`` whose
decision was ``commit`` is refused and the refusal is traced (:func:`gated_commit`).

Run::

    python -m driftsense.agent.mcp_server                                   # stdio
    python -m driftsense.agent.mcp_server --transport streamable-http --port 8001   # http://127.0.0.1:8001/mcp
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


def gated_commit(reg: ToolRegistry, **kwargs: Any) -> dict[str, Any]:
    """``commit_stage_move`` behind the runner's hard gate: refuse unless the last
    ``localize`` decided ``commit`` (margin >= tau). A refusal moves nothing."""
    last = reg.last_result
    if last is None or last.margin.decision != "commit":
        margin = None if last is None else last.margin.margin
        because = (
            "SAFETY: commit refused, no localize result yet"
            if last is None
            else f"SAFETY: commit refused at margin {margin:.3f} < tau {last.margin.tau:.3f}"
        )
        reg.trace.emit(
            "escalate",
            tool="commit_stage_move",
            args=CommitStageMoveArgs(**kwargs).model_dump(),
            result_summary={"refused": True},
            margin=margin,
            tau=reg.tau,
            because=because,
        )
        return {"refused": True, "reason": because, "next": "request_human_approval"}
    return reg.commit_stage_move(**kwargs)


def create_mcp_server(
    server_name: str = "driftsense-agent",
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    **registry_kwargs: Any,
) -> Any:
    """Build a FastMCP server exposing the six agent tools (stdio or streamable HTTP).

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
    mcp = FastMCP(server_name, host=host, port=port)

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
        """Move the stage; refused unless the last localize decided commit (margin >= tau)."""
        return gated_commit(reg, **args.model_dump())

    @mcp.tool()
    def request_human_approval(args: ApprovalArgs) -> dict:
        return reg.request_human_approval(**args.model_dump()).model_dump(mode="json")

    @mcp.tool()
    def get_trace() -> list[dict]:
        return reg.get_trace()

    return mcp


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs mcp + a live transport
    import argparse

    ap = argparse.ArgumentParser(description="DriftSense agent tools over MCP")
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--family", default="dram")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    create_mcp_server(host=a.host, port=a.port, family=a.family, seed=a.seed).run(transport=a.transport)


if __name__ == "__main__":  # pragma: no cover
    main()
