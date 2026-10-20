"""Agent episodes in the worker: ``driftsense.agent.runner.run_episode`` over a ``VirtualSEM``.

An episode job is ``JobCreate(demo_id="vsem-<family>-<seed>", agent=True)``. The worker:

* builds ``VirtualSEM(seed, family)`` with the G5 operating point (``service.settings.AGENT_*``);
* runs the real runner with ``run_id = job_id`` and ``trace_path`` in a temp dir, while a
  :class:`TraceUploader` thread mirrors the growing JSONL to ``traces/<job_id>.jsonl`` every
  ``TRACE_UPLOAD_S`` (only complete lines, so every uploaded line is a valid ``TraceEvent``);
* answers the runner's ``approval_cb`` from DynamoDB (:class:`DynamoApprover`): the job goes to
  ``awaiting_approval`` with the runner's ``ApprovalRequest``, the API stores the operator's
  ``ApprovalDecision`` on the item and the callback returns it (or a timeout decision);
* maps the :class:`EpisodeResult` onto the final job state (:func:`final_state`).

Nothing here reads ground truth: ``on_reference`` / ``on_decision`` (evaluator-only hooks that call
``VirtualSEM.verify``) are never passed.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from service.models import (
    ApprovalDecision,
    ApprovalRequest,
    ErrorInfo,
    LocalizeResult,
    apply_operator_choice,
)
from service.storage import ArtifactStore, InvalidTransition, JobStore

log = logging.getLogger("driftsense.worker.agent")

TIMEOUT_APPROVER = "timeout"


def trace_key(job_id: str) -> str:
    return f"traces/{job_id}.jsonl"


# ---------- trace mirroring ----------
class TraceUploader:
    """Mirror a growing local JSONL trace to S3: periodically from a thread, and on demand (``flush``)."""

    def __init__(self, artifacts: ArtifactStore, key: str, path: Path, interval_s: float):
        self.artifacts, self.key, self.path, self.interval_s = artifacts, key, path, interval_s
        self.uploads = 0
        self._sent = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="trace-upload", daemon=True)

    def flush(self) -> bool:
        """Upload the complete lines written so far if they changed. Returns True if it uploaded."""
        with self._lock:
            try:
                data = self.path.read_bytes()
            except FileNotFoundError:
                data = b""
            data = data[: data.rfind(b"\n") + 1]  # never publish a half-written line
            if len(data) == self._sent:
                return False
            self.artifacts.put_bytes(self.key, data, "application/x-ndjson")
            self._sent = len(data)
            self.uploads += 1
            return True

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.flush()
            except Exception:  # noqa: BLE001 - mirroring is best-effort; the final flush is not
                log.warning("trace upload failed", exc_info=True)

    def __enter__(self) -> TraceUploader:
        self.flush()  # the key exists (possibly empty) as soon as the job is running
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)


# ---------- human approval through DynamoDB ----------
class DynamoApprover:
    """The runner's ``approval_cb``: park the job in ``awaiting_approval`` and wait for the API.

    ``POST /v1/jobs/{id}/approval`` stores ``decision`` on the item and sets the job back to
    ``running``; this callback polls for it every ``poll_s`` up to ``timeout_s``. On timeout it stores
    and returns ``ApprovalDecision(chosen_rank=None, approver="timeout")``. A decision already on the
    item (left by an earlier delivery of the same deterministic episode) is reused, so an operator never
    has to approve twice after a worker restart.
    """

    def __init__(
        self,
        jobs: JobStore,
        job_id: str,
        *,
        timeout_s: float,
        poll_s: float,
        on_wait: Callable[[], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.jobs, self.job_id = jobs, job_id
        self.timeout_s, self.poll_s = timeout_s, poll_s
        self.on_wait = on_wait
        self.sleep = sleep
        self.requests: list[ApprovalRequest] = []
        self.decision: ApprovalDecision | None = None
        self.timed_out = False

    def _stored(self) -> ApprovalDecision | None:
        item = self.jobs.get_item(self.job_id) or {}
        raw = item.get("decision")
        return ApprovalDecision.model_validate_json(raw) if raw else None

    def __call__(self, req: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(req)
        prior = self._stored()
        if prior is not None:
            log.info("job %s: reusing stored approval decision %s", self.job_id, prior.model_dump())
            self.decision = prior
            return prior
        self.jobs.transition(
            self.job_id, "awaiting_approval", allowed_from=("running",), set_fields={"approval": req}
        )
        log.info(
            "job %s awaiting approval (margin %.3f < tau %.3f)",
            self.job_id,
            req.margin.margin,
            req.margin.tau,
        )
        if self.on_wait is not None:
            try:
                self.on_wait()
            except Exception:  # noqa: BLE001
                log.warning("on_wait hook failed", exc_info=True)
        deadline = time.monotonic() + self.timeout_s
        while True:
            decision = self._stored()
            if decision is not None:
                break
            if time.monotonic() >= deadline:
                decision = ApprovalDecision(
                    chosen_rank=None,
                    approver=TIMEOUT_APPROVER,
                    note=f"no operator decision within {self.timeout_s:g} s",
                )
                try:
                    self.jobs.transition(
                        self.job_id,
                        "running",
                        allowed_from=("awaiting_approval",),
                        set_fields={"decision": decision},
                    )
                    self.timed_out = True
                except InvalidTransition:  # the operator answered at the last moment: theirs wins
                    decision = self._stored() or decision
                    self.timed_out = decision.approver == TIMEOUT_APPROVER
                break
            self.sleep(self.poll_s)
        self.decision = decision
        return decision


# ---------- capturing the episode's LocalizeResults ----------
@dataclass
class LocalizeCall:
    template: bytes
    search: bytes
    result: LocalizeResult


@dataclass
class Capture:
    calls: list[LocalizeCall] = field(default_factory=list)
    available: bool = True


_CAPTURE_LOCK = threading.Lock()


@contextlib.contextmanager
def capture_localize() -> Iterator[Capture]:
    """Record every LocalizeResult the agent's tools compute during one episode.

    ``EpisodeResult`` does not expose the LocalizeResult(s) and the trace carries only summaries, so
    the service wraps ``driftsense.agent.tools._api_localize`` (the name the tools call) for the
    duration of the episode. Read-only: arguments and results pass through unchanged. Serialized by a
    lock (the worker runs one episode at a time).
    """
    cap = Capture()
    with _CAPTURE_LOCK:
        from driftsense.agent import tools

        orig = getattr(tools, "_api_localize", None)
        if orig is None:
            log.warning("driftsense.agent.tools._api_localize not found: episode result will be empty")
            cap.available = False
            yield cap
            return

        def recording(
            template_bytes: bytes, search_bytes: bytes, *args: Any, **kwargs: Any
        ) -> LocalizeResult:
            result = orig(template_bytes, search_bytes, *args, **kwargs)
            cap.calls.append(LocalizeCall(template_bytes, search_bytes, result))
            return result

        tools._api_localize = recording
        try:
            yield cap
        finally:
            tools._api_localize = orig


def decision_call(events: list[Any], cap: Capture) -> LocalizeCall | None:
    """The localize the terminal decision was taken on: the last one before commit/escalate.

    (After a commit the runner re-acquires and re-localizes to verify centring; that verification
    result is reported in episode.json, not as the job result.)
    """
    if not cap.calls:
        return None
    seen, at_decision = 0, None
    for ev in events:
        terminal_call = ev.kind == "tool_call" and ev.tool in ("commit_stage_move", "request_human_approval")
        if terminal_call or ev.kind in ("commit_move", "escalate"):
            at_decision = seen
            break
        if ev.kind == "tool_result" and ev.tool == "localize":
            seen += 1
    if not at_decision:
        return cap.calls[-1]
    return cap.calls[min(at_decision, len(cap.calls)) - 1]


# ---------- outcome -> job state ----------
@dataclass(frozen=True)
class FinalState:
    status: str  # succeeded | failed
    result: LocalizeResult | None
    error: ErrorInfo | None


def final_state(ep: Any, result: LocalizeResult | None, *, timed_out: bool) -> FinalState:
    """Map an EpisodeResult onto succeeded/failed.

    succeeded: outcome == commit, or the operator chose a rank (result.offset := that candidate).
    failed:    rejected_by_operator | approval_timeout | budget_exhausted (escalated with no approval
               request, e.g. the instrument refused an acquisition) | agent_abort.
    """
    if ep.outcome == "commit":
        return FinalState("succeeded", result, None)
    if ep.outcome == "escalate":
        decision: ApprovalDecision | None = ep.approval
        if decision is None:
            reason = _last_error(ep) or "episode escalated before an approval request"
            return FinalState("failed", result, ErrorInfo(code="budget_exhausted", message=reason[:1000]))
        if decision.chosen_rank is not None:
            chosen = next(
                (c for c in (result.candidates if result else []) if c.rank == decision.chosen_rank), None
            )
            if result is not None and chosen is not None:
                result = apply_operator_choice(result, chosen, decision)
            return FinalState("succeeded", result, None)
        if timed_out or decision.approver == TIMEOUT_APPROVER:
            return FinalState(
                "failed", result, ErrorInfo(code="approval_timeout", message=decision.note or "timeout")
            )
        note = f": {decision.note}" if decision.note else ""
        return FinalState(
            "failed",
            result,
            ErrorInfo(
                code="rejected_by_operator",
                message=f"all candidates rejected by {decision.approver}{note}"[:1000],
            ),
        )
    reason = _last_error(ep) or f"episode outcome {ep.outcome}"
    return FinalState("failed", result, ErrorInfo(code="agent_abort", message=reason[:1000]))


def _last_error(ep: Any) -> str | None:
    for ev in reversed(ep.events):
        if ev.kind == "error" and ev.because:
            return ev.because
    return None


def episode_summary(ep: Any, *, family: str, seed: int, settings: Any, post_commit: dict | None) -> dict:
    """The jobs/<id>/episode.json document."""

    def pair(v: Any) -> list[float] | None:
        return [float(v[0]), float(v[1])] if v is not None else None

    return {
        "run_id": ep.run_id,
        "family": family,
        "seed": seed,
        "outcome": ep.outcome,
        "steps": ep.steps,
        "dose": ep.dose,
        "wall_ms": ep.wall_ms,
        "tau": ep.tau,
        "final_margin": ep.final_margin,
        "estimate_nm": pair(ep.estimate_nm),
        "decision_acq_index": ep.decision_acq_index,
        "committed_move_nm": pair(ep.committed_move_nm),
        "approval": ep.approval.model_dump(mode="json") if ep.approval is not None else None,
        "n_events": len(ep.events),
        "post_commit_verify": post_commit,
        "policy": "rules",
        "operating_point": {
            "tau": settings.agent_tau,
            "max_steps": settings.agent_max_steps,
            "max_dose": settings.agent_max_dose,
            "site_nm": _s("AGENT_SITE_NM"),
            "pixel_nm": _s("AGENT_PIXEL_NM"),
            "ref_fov_nm": _s("AGENT_REF_FOV_NM"),
            "ref_dwell_us": _s("AGENT_REF_DWELL_US"),
            "base_fov_nm": _s("AGENT_BASE_FOV_NM"),
            "base_dwell_us": _s("AGENT_BASE_DWELL_US"),
            "template_frac": _s("AGENT_TEMPLATE_FRAC"),
            "source": "eval/agent_eval.py",
        },
    }


def _s(name: str) -> Any:
    from service import settings as mod

    return getattr(mod, name)


def post_commit_verify(ep: Any) -> dict | None:
    """The runner's image-only post-commit verification observation, if any."""
    for ev in ep.events:
        if ev.kind == "observe" and ev.result_summary.get("post_commit_verify") is not None:
            return {"margin": ev.margin, "tau": ev.tau, **ev.result_summary}
    return None


def dumps(doc: dict) -> bytes:
    return json.dumps(doc, indent=2, sort_keys=True).encode("utf-8")
