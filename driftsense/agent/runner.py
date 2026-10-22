"""Episode runner: the safe orchestration loop and the service entry point.

The runner enforces the HARD safety rules that NO policy -- rules or LLM -- may
override:

  * never commit a stage move when margin < tau (a policy that tries is coerced
    to an escalation);
  * a <= ``max_steps`` decision-loop cap AND a dose budget (a projected or
    instrument-raised dose overrun becomes an escalation);
  * a per-tool wall-clock timeout (a tool that overruns aborts the episode
    safely);
  * any exception -> safe abort + escalate, never a silent commit.

Every step appends a :class:`driftsense.contracts.TraceEvent`; the ``commit_move``
event records ``because="margin X >= tau Y"`` -- the evidence that the OpenCV
margin drove the physical action.

Calling convention for the service (see PR handoff): ``run_episode`` is
synchronous and CPU/IO-bound (OpenCV); the service should call it in a worker
thread / executor, not on the event loop. Escalation surfaces as
``outcome="escalate"`` with the approval payload being a
:class:`driftsense.contracts.ApprovalDecision`.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Literal

from driftsense.agent.policy_rules import Decision, Observation, decide
from driftsense.agent.tools import ToolRegistry
from driftsense.agent.trace import TraceWriter
from driftsense.contracts import ApprovalDecision, LocalizeResult, TraceEvent
from driftsense.sim import VirtualSEM
from driftsense.sim.instrument import AcquisitionLimitError, DoseError

Outcome = Literal["commit", "escalate", "abort"]


@dataclass
class EpisodeResult:
    run_id: str
    outcome: Outcome
    steps: int
    dose: float
    wall_ms: float
    tau: float
    final_margin: float | None
    estimate_nm: tuple[float, float] | None
    decision_acq_index: int | None
    committed_move_nm: tuple[float, float] | None
    approval: ApprovalDecision | None
    events: list[TraceEvent]
    # LocalizeResult the terminal decision was taken on (None if no localize ran)
    last_result: LocalizeResult | None = None
    # PNG inputs of that localize: the reference template and the search frame it ran on
    template_png: bytes | None = None
    search_png: bytes | None = None


class _SafeExec:
    """Run each tool call with a hard per-tool wall-clock timeout."""

    def __init__(self, timeout_s: float) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._timeout = timeout_s

    def call(
        self, fn: Callable[..., Any], *, _timeout_s: float | None | str = "default", **kwargs: Any
    ) -> Any:
        fut = self._pool.submit(fn, **kwargs)
        return fut.result(timeout=self._timeout if _timeout_s == "default" else _timeout_s)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def run_episode(
    sem: VirtualSEM,
    *,
    tau: float,
    run_id: str,
    max_steps: int = 6,
    max_dose: float | None = None,
    ref_fov_nm: float = 260.0,
    ref_dwell_us: float = 24.0,
    base_fov_nm: float = 360.0,
    base_dwell_us: float = 3.0,
    template_frac: float = 0.55,
    approval_cb: Callable[..., ApprovalDecision] | None = None,
    planner: Any | None = None,
    trace_path: str | None = None,
    tool_timeout_s: float = 15.0,
    approval_timeout_s: float | None = None,
    on_reference: Callable[[int], None] | None = None,
    on_decision: Callable[[int, tuple[float, float]], None] | None = None,
) -> EpisodeResult:
    """Run one active-perception episode over ``sem`` and return its outcome.

    Parameters mirror the physical knobs: a high-dwell reference acquire seeds the
    template, then low-dwell/wide-FOV search acquires drive the margin loop.
    ``planner`` (optional, from :mod:`driftsense.agent.policy_llm`) may only
    *pick* a schema-validated tool; the rules policy remains the fallback and the
    runner remains the safety gate. ``on_reference`` / ``on_decision`` are
    evaluator-only hooks fired just after the reference acquire and once at the
    terminal decision respectively (they may read ``sem.verify`` to grade the
    drift *change* since the reference); agent code never calls them.

    ``tool_timeout_s`` bounds every instrument/localizer tool call. The human
    approval wait is bounded separately by ``approval_timeout_s`` (None = the
    ``approval_cb`` enforces its own deadline), so an operator taking longer than
    a tool timeout does not abort the episode. ``EpisodeResult.last_result`` is
    the LocalizeResult the terminal decision was taken on; ``template_png`` and
    ``search_png`` are the PNG inputs it was computed from.
    """
    import time

    t0 = time.perf_counter()
    trace = TraceWriter(run_id, trace_path)
    reg = ToolRegistry(sem, trace, tau=tau, template_frac=template_frac, approval_cb=approval_cb)
    ex = _SafeExec(tool_timeout_s)
    trace.emit(
        "observe",
        result_summary={
            "episode_start": True,
            "tau": tau,
            "max_steps": max_steps,
            "policy": _policy_name(planner),
        },
    )

    outcome: Outcome = "abort"
    approval: ApprovalDecision | None = None
    committed_move: tuple[float, float] | None = None
    final_margin: float | None = None

    def _budget_exhausted(step: int) -> bool:
        if step >= max_steps - 1:
            return True
        if max_dose is not None and sem.total_dose >= 0.9 * max_dose:
            return True
        return False

    decision_result: LocalizeResult | None = None
    decision_search: bytes | None = None

    def _fire_decision() -> None:
        nonlocal decision_result, decision_search
        decision_result = reg.last_result
        decision_search = reg.last_search_bytes
        if on_decision is not None and reg.last_search_meta is not None and reg.last_estimate_nm is not None:
            on_decision(int(reg.last_search_meta["acq_index"]), reg.last_estimate_nm)

    try:
        # -- bootstrap: reference template + first search frame + localize ----
        ex.call(reg.acquire, center_nm=(0.0, 0.0), fov_nm=ref_fov_nm, dwell_us=ref_dwell_us)
        if on_reference is not None and reg.last_search_meta is not None:
            on_reference(int(reg.last_search_meta["acq_index"]))
        ex.call(reg.acquire, center_nm=(0.0, 0.0), fov_nm=base_fov_nm, dwell_us=base_dwell_us)
        ex.call(reg.localize, mode="standard", rect_hypotheses=2, top_k=5)

        # -- decision loop ----------------------------------------------------
        for step in range(max_steps):
            obs = _observe(reg, budget_exhausted=_budget_exhausted(step))
            dec = _choose(obs, reg, planner)
            final_margin = obs.margin

            if dec.tool == "commit_stage_move":
                # HARD GATE: never commit below tau, whatever the policy said.
                if obs.margin < tau:
                    dec = Decision(
                        tool="request_human_approval",
                        kwargs={"evidence": {"blocked_commit": True, "margin": obs.margin, "tau": tau}},
                        reason=f"SAFETY: refused commit at margin {obs.margin:.3f} < tau {tau:.3f}",
                        terminal=True,
                    )
                else:
                    _fire_decision()
                    ex.call(reg.commit_stage_move, **dec.kwargs)
                    committed_move = (dec.kwargs["dx_nm"], dec.kwargs["dy_nm"])
                    trace.emit(
                        "commit_move",
                        tool="commit_stage_move",
                        margin=obs.margin,
                        tau=tau,
                        because=dec.reason,
                        stage_move_nm=committed_move,
                        policy=_policy_name(planner),
                    )
                    _verify_after_commit(reg, ex, base_fov_nm, base_dwell_us)
                    outcome = "commit"
                    break

            if dec.tool == "request_human_approval":
                _fire_decision()
                approval = ex.call(
                    reg.request_human_approval,
                    _timeout_s=approval_timeout_s,
                    top3=reg.top3(),
                    evidence=dec.kwargs.get("evidence", {}),
                )
                trace.emit(
                    "escalate",
                    margin=obs.margin,
                    tau=tau,
                    because=dec.reason,
                    result_summary={"chosen_rank": approval.chosen_rank, "approver": approval.approver},
                    policy=_policy_name(planner),
                )
                outcome = "escalate"
                break

            # non-terminal: acquire / localize / refine
            trace.emit("decision", tool=dec.tool, because=dec.reason, policy=_policy_name(planner))
            ex.call(getattr(reg, dec.tool), **dec.kwargs)
            if dec.tool == "acquire":
                # a fresh frame needs a localize before the next observation
                ex.call(reg.localize, mode="standard", rect_hypotheses=2, top_k=5)
        else:
            # loop exhausted without a terminal decision -> escalate
            _fire_decision()
            approval = ex.call(
                reg.request_human_approval,
                _timeout_s=approval_timeout_s,
                top3=reg.top3(),
                evidence={"reason": "max_steps reached", "margin": final_margin, "tau": tau},
            )
            trace.emit(
                "escalate",
                margin=final_margin,
                tau=tau,
                because="max_steps reached without margin >= tau",
                result_summary={"chosen_rank": approval.chosen_rank, "approver": approval.approver},
            )
            outcome = "escalate"

    except (DoseError, AcquisitionLimitError) as exc:
        # Budget overrun raised by the instrument -> escalate, never commit.
        trace.emit("error", because=f"instrument budget: {exc}")
        outcome = "escalate"
    except FuturesTimeout:
        trace.emit("error", because=f"tool timeout > {tool_timeout_s}s: safe abort")
        outcome = "abort"
    except Exception as exc:  # noqa: BLE001 -- any failure must abort safely, not commit
        trace.emit("error", because=f"unhandled {type(exc).__name__}: {exc}")
        outcome = "abort"
    finally:
        ex.close()

    wall_ms = (time.perf_counter() - t0) * 1e3
    decision_acq = int(reg.last_search_meta["acq_index"]) if reg.last_search_meta else None
    trace.emit(
        "episode_end",
        result_summary={"outcome": outcome, "dose": sem.total_dose, "n_acq": sem.n_acquisitions},
    )
    return EpisodeResult(
        run_id=run_id,
        outcome=outcome,
        steps=sem.n_acquisitions,
        dose=sem.total_dose,
        wall_ms=wall_ms,
        tau=tau,
        final_margin=final_margin,
        estimate_nm=reg.last_estimate_nm,
        decision_acq_index=decision_acq,
        committed_move_nm=committed_move,
        approval=approval,
        events=trace.events,
        last_result=decision_result if decision_result is not None else reg.last_result,
        template_png=reg.template_bytes,
        search_png=decision_search if decision_result is not None else reg.last_search_bytes,
    )


def _policy_name(planner: Any | None) -> str:
    return "llm" if planner is not None else "rules"


def _observe(reg: ToolRegistry, *, budget_exhausted: bool) -> Observation:
    assert reg.last_result is not None and reg.last_search_meta is not None
    assert reg.last_snr is not None and reg.last_estimate_nm is not None
    sig = reg.derived_signals()
    m = reg.last_result.margin
    return Observation(
        margin=m.margin,
        tau=m.tau,
        snr=reg.last_snr,
        fov_nm=reg.last_search_meta["fov_nm"],
        dwell_us=reg.last_search_meta["dwell_us"],
        center_nm=reg.last_center_nm,
        estimate_nm=reg.last_estimate_nm,
        lattice_uncertain=sig["lattice_uncertain"],
        period_ambiguous=sig["period_ambiguous"],
        at_fov_edge=sig["at_fov_edge"],
        budget_exhausted=budget_exhausted,
    )


def _choose(obs: Observation, reg: ToolRegistry, planner: Any | None) -> Decision:
    """LLM planner may propose; rules decide on any failure or invalid pick."""
    if planner is not None:
        try:
            proposal = planner.propose(obs, reg)
            if proposal is not None:
                return proposal
        except Exception:  # noqa: BLE001 -- planner is best-effort; rules are the floor
            pass
    return decide(obs)


def _verify_after_commit(reg: ToolRegistry, ex: _SafeExec, fov_nm: float, dwell_us: float) -> None:
    """Image-only verification: re-acquire + re-localize to confirm centring.

    This uses ONLY images (never ``sem.verify``); it records the residual margin
    so the trace shows the commit was confirmed by a fresh observation.
    """
    try:
        ex.call(reg.acquire, center_nm=reg.last_center_nm, fov_nm=fov_nm, dwell_us=dwell_us)
        res = ex.call(reg.localize, mode="standard", rect_hypotheses=2, top_k=5)
        reg.trace.emit(
            "observe",
            result_summary={
                "post_commit_verify": True,
                "residual_estimate_nm": [round(reg.last_estimate_nm[0], 2), round(reg.last_estimate_nm[1], 2)]
                if reg.last_estimate_nm
                else None,
            },
            margin=res.margin.margin,
            tau=res.margin.tau,
        )
    except (DoseError, AcquisitionLimitError, FuturesTimeout):
        # Verification is best-effort; the commit itself already stands.
        reg.trace.emit("observe", result_summary={"post_commit_verify": "skipped (budget/timeout)"})
