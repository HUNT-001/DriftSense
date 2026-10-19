"""Optional LLM planner (behind the ``--planner`` flag).

This is NOT the default. The offline rules policy
(:func:`driftsense.agent.policy_rules.decide`) is the default and the fallback,
and the runner is the safety gate. The planner may only *propose* a
schema-validated **perception** tool (``acquire`` / ``localize`` / ``refine``);
the terminal commit / escalate decisions are left to the rules + the runner's
hard safety gate, so a hallucinated plan can never drive an unsafe stage move.

The planner takes an injected ``complete`` callable (``str -> str`` returning a
JSON tool call) so there is no hard dependency on any SDK or network: with no
``complete`` provided it proposes nothing and the runner falls back to rules.
``make check`` and the G5 episode therefore pass with no LLM available.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from driftsense.agent.policy_rules import Decision, Observation
from driftsense.agent.tools import TOOL_SCHEMAS, ToolRegistry

# Tools the planner is allowed to pick. Terminal actions are deliberately absent.
_PLANNER_TOOLS = ("acquire", "localize", "refine")

_SYSTEM = """You drive an SEM alignment agent. Given the current observation you
may choose ONE perception action to improve the localization margin before the
rule-based controller decides whether to commit. Reply with a single JSON object
{"tool": <name>, "args": {...}} where tool is one of: acquire, localize, refine.
Do not commit or escalate; the controller does that. Prefer the cheapest action
that is likely to raise the margin above tau."""


class LLMPlanner:
    """Best-effort planner that validates every proposal against the tool schema."""

    def __init__(self, complete: Callable[[str], str] | None = None) -> None:
        self._complete = complete

    def _prompt(self, obs: Observation) -> str:
        return (
            f"{_SYSTEM}\n\nobservation = "
            + json.dumps(
                {
                    "margin": round(obs.margin, 4),
                    "tau": obs.tau,
                    "snr": round(obs.snr, 3),
                    "fov_nm": obs.fov_nm,
                    "dwell_us": obs.dwell_us,
                    "lattice_uncertain": obs.lattice_uncertain,
                    "period_ambiguous": obs.period_ambiguous,
                    "at_fov_edge": obs.at_fov_edge,
                }
            )
            + "\n\nallowed tools = "
            + json.dumps({t: TOOL_SCHEMAS[t].model_json_schema() for t in _PLANNER_TOOLS})
        )

    def propose(self, obs: Observation, reg: ToolRegistry) -> Decision | None:
        """Return a schema-valid perception Decision, or None to defer to rules."""
        if self._complete is None:
            return None
        try:
            raw = self._complete(self._prompt(obs))
            call = json.loads(raw)
            tool = call.get("tool")
            args = call.get("args", {})
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None
        if tool not in _PLANNER_TOOLS:
            return None
        try:
            validated = TOOL_SCHEMAS[tool](**args)  # pydantic schema validation
        except ValidationError:
            return None
        return Decision(
            tool=tool,
            kwargs=validated.model_dump(),
            reason=f"llm planner: {tool}",
            terminal=False,
        )


def echo_planner(tool: str = "localize", **args: Any) -> LLMPlanner:
    """A deterministic, offline stand-in planner for tests/demos (no network).

    It always proposes the same schema-valid perception tool, exercising the
    planner path and the runner's rules/safety fallback without any LLM.
    """
    payload = json.dumps({"tool": tool, "args": args})
    return LLMPlanner(complete=lambda _prompt: payload)
