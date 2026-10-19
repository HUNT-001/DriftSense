"""Offline default policy: margin-driven active perception.

``decide`` is a pure function of an :class:`Observation` (assembled by the runner
from the last localize + instrument/budget state) returning the next
:class:`Decision`. It has NO network or LLM dependency and must pass with
``mcp`` not installed -- it is both the default policy and the safety-net
fallback when the optional LLM planner is enabled.

The ladder (first match wins), straight from the G5 brief:

  1. margin >= tau .......................... commit_stage_move  (+ verify)
  2. budget exhausted ....................... request_human_approval (escalate)
  3. low SNR ................................ acquire(dwell x2)
  4. lattice / rectify uncertain ............ localize(rect_hypotheses=4)
  5. top-2 one period apart at FOV edge ..... acquire(fov x2)
  6. contact / low margin ................... localize(mode="contour")
  7. otherwise .............................. refine

Every branch yields a human-readable ``reason`` that the runner copies into the
TraceEvent ``because`` field -- the Agentic-Vision evidence that the OpenCV
margin changed the next action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Tunables (image-derived SNR scale is set by the Immerkaer estimate in tools).
SNR_FLOOR = 7.0  # below this a frame is "low SNR" and worth more dwell
DWELL_MAX_US = 64.0  # hard ceiling the policy will not propose beyond
FOV_MAX_NM = 1200.0  # hard ceiling on the proposed FOV


@dataclass(frozen=True)
class Observation:
    """Everything the policy may look at -- derived from images + budget only."""

    margin: float
    tau: float
    snr: float
    fov_nm: float
    dwell_us: float
    center_nm: tuple[float, float]
    estimate_nm: tuple[float, float]
    lattice_uncertain: bool
    period_ambiguous: bool
    at_fov_edge: bool
    budget_exhausted: bool


@dataclass(frozen=True)
class Decision:
    tool: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    terminal: bool = False  # commit or escalate ends the episode


def decide(obs: Observation) -> Decision:
    tau, m = obs.tau, obs.margin

    # 1. Confident enough: commit the corrective move (runner verifies after).
    if m >= tau:
        dx, dy = -obs.estimate_nm[0], -obs.estimate_nm[1]
        return Decision(
            tool="commit_stage_move",
            kwargs={"dx_nm": dx, "dy_nm": dy},
            reason=f"margin {m:.3f} >= tau {tau:.3f}",
            terminal=True,
        )

    # 2. Out of budget before reaching tau: hand to a human.
    if obs.budget_exhausted:
        return Decision(
            tool="request_human_approval",
            kwargs={"evidence": {"margin": m, "tau": tau, "snr": round(obs.snr, 2)}},
            reason=f"budget exhausted at margin {m:.3f} < tau {tau:.3f}",
            terminal=True,
        )

    # 3. Noise-limited: double the dwell to raise SNR (noise ~ 1/sqrt(dwell)).
    if obs.snr < SNR_FLOOR and obs.dwell_us * 2.0 <= DWELL_MAX_US:
        return Decision(
            tool="acquire",
            kwargs={
                "center_nm": obs.center_nm,
                "fov_nm": obs.fov_nm,
                "dwell_us": obs.dwell_us * 2.0,
            },
            reason=f"low SNR {obs.snr:.2f} < {SNR_FLOOR:.1f}: dwell x2",
        )

    # 4. Lattice / rectify ambiguous: re-localize with more rectify hypotheses.
    if obs.lattice_uncertain:
        return Decision(
            tool="localize",
            kwargs={"mode": "standard", "rect_hypotheses": 4, "top_k": 8},
            reason="lattice/rectify uncertain: rect_hypotheses=4",
        )

    # 5. Periodic tie at the frame edge: widen the FOV to break the ambiguity.
    if obs.period_ambiguous and obs.at_fov_edge and obs.fov_nm * 2.0 <= FOV_MAX_NM:
        return Decision(
            tool="acquire",
            kwargs={
                "center_nm": obs.center_nm,
                "fov_nm": obs.fov_nm * 2.0,
                "dwell_us": obs.dwell_us,
            },
            reason="top-2 one period apart at FOV edge: fov x2",
        )

    # 6. Low-margin contact case: contour mode (S5) for a sharper fingerprint.
    return Decision(
        tool="localize",
        kwargs={"mode": "contour", "rect_hypotheses": 4, "top_k": 8},
        reason=f"low margin {m:.3f} < tau {tau:.3f}: contour re-localize",
    )
