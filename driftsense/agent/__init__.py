"""DriftSense active-perception agent (G5).

An agent that drives a single :class:`driftsense.sim.VirtualSEM` and the
``driftsense.v2`` localizer under a closed perception loop: it *acquires*,
*localizes*, reads the uniqueness margin produced by the OpenCV pipeline, and
lets that margin choose the next physical action (dwell up, FOV up, more
rectify hypotheses, contour mode) until it can either commit a corrective
stage move at margin >= tau or escalate to a human.

Public surface:
  * :class:`~driftsense.agent.tools.ToolRegistry` -- the schema-validated tools.
  * :func:`~driftsense.agent.runner.run_episode` -- the safe orchestration loop
    (the service entry point).
  * :func:`~driftsense.agent.policy_rules.decide` -- the offline default policy.
  * :class:`~driftsense.agent.trace.TraceWriter` -- contract TraceEvent sink.

The rules policy and the runner have no network or LLM dependency; the optional
LLM planner lives behind a flag in :mod:`driftsense.agent.policy_llm` and the
FastMCP server behind a guarded import in :mod:`driftsense.agent.mcp_server`.
"""

from __future__ import annotations

from driftsense.agent.policy_rules import Decision, decide
from driftsense.agent.runner import EpisodeResult, run_episode
from driftsense.agent.tools import TOOL_SCHEMAS, ToolRegistry
from driftsense.agent.trace import TraceWriter

__all__ = [
    "Decision",
    "decide",
    "EpisodeResult",
    "run_episode",
    "ToolRegistry",
    "TOOL_SCHEMAS",
    "TraceWriter",
]
