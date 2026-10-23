"""Agent tools: schema-validated actions over a VirtualSEM + the v2 localizer.

Every tool has a pydantic argument model (so the LLM planner and the FastMCP
server can both validate inputs against the same schema) and is executed through
:class:`ToolRegistry`, which owns the one instrument, the reference template and
the trace. The registry calls ``driftsense.v2`` and ``driftsense.sim`` strictly
read-only and NEVER touches :meth:`VirtualSEM.verify` -- the hidden ground truth
is the evaluator's alone.

Tool-level ``mode`` names are an agent abstraction mapped onto the contract
:class:`Mode` (there is deliberately no new contract field):

    standard -> AUTO       contour  -> ACCURATE (runs S5 contour + explain)
    fast     -> FAST       dense    -> DENSE
    accurate -> ACCURATE

``rect_hypotheses`` is an advisory the agent records in the trace (v2's rectify
enumerates hypotheses internally); ``top_k`` maps onto ``Budget.max_candidates``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from driftsense.agent.trace import TraceWriter
from driftsense.api import localize as _api_localize
from driftsense.contracts import (
    ApprovalDecision,
    ApprovalRequest,
    Budget,
    Candidate,
    LocalizeResult,
    Margin,
    Mode,
)
from driftsense.sim import VirtualSEM

# Tool mode -> contract Mode. Keys are the agent vocabulary used by the policies.
_MODE_MAP: dict[str, Mode] = {
    "standard": Mode.AUTO,
    "fast": Mode.FAST,
    "accurate": Mode.ACCURATE,
    "contour": Mode.ACCURATE,
    "dense": Mode.DENSE,
}


# --------------------------------------------------------------------------- #
# Argument schemas                                                            #
# --------------------------------------------------------------------------- #
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AcquireArgs(_Args):
    center_nm: tuple[float, float] = Field((0.0, 0.0), description="Commanded stage centre (nm).")
    fov_nm: float = Field(..., gt=0, description="Field of view (nm); larger disambiguates periodic ties.")
    dwell_us: float = Field(..., gt=0, description="Dwell per pixel (us); higher raises SNR as sqrt(dwell).")
    scan_rotation_deg: float = Field(0.0, description="Scan rotation (deg).")


class LocalizeArgs(_Args):
    mode: Literal["standard", "fast", "accurate", "contour", "dense"] = "standard"
    rect_hypotheses: int = Field(2, ge=1, le=8, description="Advisory rectify-hypothesis count.")
    top_k: int = Field(5, ge=1, le=50, description="Candidates to keep (-> Budget.max_candidates).")


class RefineArgs(_Args):
    top_k: int = Field(5, ge=1, le=50, description="Candidates to keep for the accurate re-localize.")


class CommitStageMoveArgs(_Args):
    dx_nm: float = Field(..., description="Relative stage move, x (nm).")
    dy_nm: float = Field(..., description="Relative stage move, y (nm).")


class ApprovalArgs(_Args):
    top3: list[Candidate] = Field(..., min_length=1, max_length=3)
    evidence: dict[str, Any] = Field(default_factory=dict)


class GetTraceArgs(_Args):
    pass


#: Public tool name -> argument model. Consumed by the LLM planner and MCP server.
TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "acquire": AcquireArgs,
    "localize": LocalizeArgs,
    "refine": RefineArgs,
    "commit_stage_move": CommitStageMoveArgs,
    "request_human_approval": ApprovalArgs,
    "get_trace": GetTraceArgs,
}


def tool_json_schemas() -> dict[str, dict]:
    """JSON Schema per tool (for FastMCP and the LLM planner)."""
    return {name: model.model_json_schema() for name, model in TOOL_SCHEMAS.items()}


# --------------------------------------------------------------------------- #
# Image helpers                                                               #
# --------------------------------------------------------------------------- #
def _decode(png: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError("acquire returned an undecodable PNG")
    return arr


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _estimate_snr(img: np.ndarray) -> float:
    """Immerkaer fast noise estimate -> structure/noise ratio (image-derived).

    Uses the Laplacian-of-Laplacian mask, whose response to smooth structure is
    ~0, so it isolates per-pixel noise. Monotone in dwell by construction of the
    twin but computed purely from the returned image (the agent never sees dwell
    as ground truth).
    """
    f = img.astype(np.float64)
    mask = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    conv = cv2.filter2D(f, -1, mask, borderType=cv2.BORDER_REPLICATE)
    sigma = float(np.abs(conv).mean()) * np.sqrt(0.5 * np.pi) / 6.0
    signal = float(f.std())
    return signal / (sigma + 1e-9)


# --------------------------------------------------------------------------- #
# Tool registry                                                               #
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """Owns the instrument, the reference template and the trace for one episode.

    The first :meth:`acquire` (the runner's high-dwell bootstrap) is adopted as
    the reference; its central crop becomes the template every later search frame
    is localized against. The agent reads the OpenCV ``margin`` and never the
    hidden stage error.
    """

    def __init__(
        self,
        sem: VirtualSEM,
        trace: TraceWriter,
        *,
        tau: float,
        template_frac: float = 0.55,
        approval_cb: Callable[[ApprovalRequest], ApprovalDecision] | None = None,
    ) -> None:
        self.sem = sem
        self.trace = trace
        self.tau = float(tau)
        self.template_frac = float(template_frac)
        self.approval_cb = approval_cb or _default_reject

        self.template_bytes: bytes | None = None
        self.template_shape: tuple[int, int] | None = None  # (h, w)
        self.last_search_bytes: bytes | None = None
        self.last_search_meta: dict[str, Any] | None = None
        self.last_center_nm: tuple[float, float] = (0.0, 0.0)
        self.last_snr: float | None = None
        self.last_result: LocalizeResult | None = None
        self.last_estimate_nm: tuple[float, float] | None = None

    # -- tool: acquire ----------------------------------------------------- #
    def acquire(self, **kwargs: Any) -> dict[str, Any]:
        args = AcquireArgs(**kwargs)
        self.trace.emit("tool_call", tool="acquire", args=args.model_dump())
        t0 = time.perf_counter()
        png, meta = self.sem.acquire(
            tuple(args.center_nm), args.fov_nm, args.dwell_us, args.scan_rotation_deg
        )
        img = _decode(png)
        snr = _estimate_snr(img)
        self.last_search_bytes = png
        self.last_search_meta = meta
        self.last_center_nm = tuple(args.center_nm)
        self.last_snr = snr
        if self.template_bytes is None:
            self._adopt_reference(img)
        summary = {
            "acq_index": meta["acq_index"],
            "fov_nm": meta["fov_nm"],
            "dwell_us": meta["dwell_us"],
            "width_px": meta["width_px"],
            "snr": round(snr, 3),
            "dose_this_acq": meta["dose_this_acq"],
            "cumulative_dose": meta["cumulative_dose"],
        }
        latency_ms = (time.perf_counter() - t0) * 1000.0  # wall time of the acquisition, not the dwell
        self.trace.emit("tool_result", tool="acquire", result_summary=summary, latency_ms=latency_ms)
        return summary

    def _adopt_reference(self, img: np.ndarray) -> None:
        """Crop the centre of the (high-dwell) reference frame as the template."""
        h, w = img.shape[:2]
        th, tw = max(16, int(h * self.template_frac)), max(16, int(w * self.template_frac))
        y0, x0 = (h - th) // 2, (w - tw) // 2
        crop = img[y0 : y0 + th, x0 : x0 + tw]
        self.template_bytes = _encode(crop)
        self.template_shape = (th, tw)
        self.trace.emit(
            "observe",
            result_summary={"reference_template": True, "template_px": [tw, th]},
        )

    # -- tool: localize ---------------------------------------------------- #
    def localize(self, **kwargs: Any) -> LocalizeResult:
        args = LocalizeArgs(**kwargs)
        if self.template_bytes is None or self.last_search_bytes is None:
            raise RuntimeError("localize requires a reference template and a search acquisition")
        self.trace.emit("tool_call", tool="localize", args=args.model_dump())
        mode = _MODE_MAP[args.mode]
        budget = Budget(max_candidates=args.top_k, tau=self.tau)
        result = _api_localize(self.template_bytes, self.last_search_bytes, mode, budget, engine="v2")
        self._record_result(result, mode=args.mode, rect_hypotheses=args.rect_hypotheses)
        return result

    # -- tool: refine ------------------------------------------------------ #
    def refine(self, **kwargs: Any) -> LocalizeResult:
        """Second pass in ACCURATE mode (subpixel + contour) on the current frame."""
        args = RefineArgs(**kwargs)
        self.trace.emit("tool_call", tool="refine", args=args.model_dump())
        if self.template_bytes is None or self.last_search_bytes is None:
            raise RuntimeError("refine requires a reference template and a search acquisition")
        budget = Budget(max_candidates=args.top_k, tau=self.tau)
        result = _api_localize(
            self.template_bytes, self.last_search_bytes, Mode.ACCURATE, budget, engine="v2"
        )
        self._record_result(result, mode="accurate", rect_hypotheses=4)
        return result

    def _record_result(self, result: LocalizeResult, *, mode: str, rect_hypotheses: int) -> None:
        self.last_result = result
        est = self._estimate_from_result(result)
        self.last_estimate_nm = est
        m = result.margin
        summary = {
            "mode": mode,
            "rect_hypotheses": rect_hypotheses,
            "lattice_type": result.lattice_type.value,
            "n_candidates": len(result.candidates),
            "rank1_score": round(result.candidates[0].score, 4),
            "estimate_nm": [round(est[0], 2), round(est[1], 2)],
            "warnings": result.warnings,
        }
        self.trace.emit(
            "tool_result",
            tool="localize",
            result_summary=summary,
            margin=m.margin,
            tau=m.tau,
            latency_ms=result.total_ms,
        )

    def _estimate_from_result(self, result: LocalizeResult) -> tuple[float, float]:
        """Drift estimate (nm) = (frame centre - matched feature centre) * pixel.

        Positive components mean the tracked feature has drifted toward -x/-y in
        the frame, i.e. the stage error grew by this much since the reference.
        """
        assert self.last_search_meta is not None and self.template_shape is not None
        th, tw = self.template_shape
        w = self.last_search_meta["width_px"]
        h = self.last_search_meta["height_px"]
        pix = self.last_search_meta["pixel_nm"]
        feat_cx = result.offset.x_px + tw / 2.0
        feat_cy = result.offset.y_px + th / 2.0
        return ((w / 2.0 - feat_cx) * pix, (h / 2.0 - feat_cy) * pix)

    # -- tool: commit_stage_move ------------------------------------------- #
    def commit_stage_move(self, **kwargs: Any) -> dict[str, Any]:
        args = CommitStageMoveArgs(**kwargs)
        self.trace.emit("tool_call", tool="commit_stage_move", args=args.model_dump())
        self.sem.commit_stage_move(args.dx_nm, args.dy_nm)
        summary = {"dx_nm": args.dx_nm, "dy_nm": args.dy_nm}
        self.trace.emit("tool_result", tool="commit_stage_move", result_summary=summary)
        return summary

    # -- tool: request_human_approval -------------------------------------- #
    def request_human_approval(self, **kwargs: Any) -> ApprovalDecision:
        args = ApprovalArgs(**kwargs)
        if self.last_result is None:
            raise RuntimeError("request_human_approval requires a prior localize")
        req = ApprovalRequest(options=args.top3, margin=self.last_result.margin)
        self.trace.emit(
            "tool_call",
            tool="request_human_approval",
            args={"n_options": len(args.top3), "evidence": args.evidence},
            margin=req.margin.margin,
            tau=req.margin.tau,
        )
        decision = self.approval_cb(req)
        self.trace.emit(
            "tool_result",
            tool="request_human_approval",
            result_summary={"chosen_rank": decision.chosen_rank, "approver": decision.approver},
        )
        return decision

    # -- tool: get_trace --------------------------------------------------- #
    def get_trace(self, **_: Any) -> list[dict[str, Any]]:
        return [ev.model_dump(mode="json") for ev in self.trace.events]

    # -- derived signals (read-only; no ground truth) ---------------------- #
    def top3(self) -> list[Candidate]:
        assert self.last_result is not None
        return list(self.last_result.candidates[:3])

    def derived_signals(self) -> dict[str, bool]:
        """Policy-facing booleans computed from the last result + frame geometry.

        All image-derived; no stage ground truth is consulted.
        """
        assert self.last_result is not None and self.last_search_meta is not None
        assert self.template_shape is not None
        res = self.last_result
        cands = res.candidates
        th, tw = self.template_shape
        w, h = self.last_search_meta["width_px"], self.last_search_meta["height_px"]
        below_tau = res.margin.margin < res.margin.tau

        lattice_uncertain = res.lattice_type.value == "unknown" or any(
            "rectify" in wmsg for wmsg in res.warnings
        )

        period_ambiguous = False
        if below_tau and len(cands) >= 2:
            dist = float(np.hypot(cands[0].x_px - cands[1].x_px, cands[0].y_px - cands[1].y_px))
            # A discrete alternative placement (not a sub-pixel neighbour) -> a
            # classic periodic-lattice tie rather than a noisy single peak.
            period_ambiguous = 8.0 <= dist <= 0.9 * max(tw, th)

        edge = 0.5 * tw
        x, y = cands[0].x_px, cands[0].y_px
        at_fov_edge = x < edge or y < edge or x > (w - tw - edge) or y > (h - th - edge)

        return {
            "lattice_uncertain": bool(lattice_uncertain),
            "period_ambiguous": bool(period_ambiguous),
            "at_fov_edge": bool(at_fov_edge),
        }


def _default_reject(req: ApprovalRequest) -> ApprovalDecision:
    """Offline default: no human present -> reject all (escalation stands)."""
    return ApprovalDecision(chosen_rank=None, approver="auto:none", note="no human available")


def margin_of(result: LocalizeResult) -> Margin:
    return result.margin
