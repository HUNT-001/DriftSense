"""S6 — cascade re-ranking.

Pure orchestration. The S4 dense path returns up to top-K lattice-consistent
candidates ranked by residual correlation. When the top two are close (margin <
tau), residual correlation alone is not decisive, so we escalate: re-rank the
top-K by the frozen v1 LER fingerprint, which keys on per-line roughness shape
and is the sharper (but slower) discriminator between a true copy and a replica.

Modes:
    dense   — keep the S4 residual ranking as-is (fast path).
    v1      — always re-rank the whole top-K by the v1 fingerprint.
    cascade — dense, but escalate to a v1 re-rank only when margin < tau.

Re-ranking reorders a fixed candidate set; it never adds or drops members, so it
can change top-1 but leaves recall@K unchanged (surfaced explicitly in the eval).

No cv2 hot-path ops live here; the v1 fingerprint is imported read-only and lazily
(same pattern as driftsense.engines.v1_adapter) so the import graph stays clean
and v1 is never modified. The per-candidate loop is O(K), not O(pixels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from driftsense.contracts import Candidate

CascadeMode = Literal["dense", "cascade", "v1"]

# Calibrated uniqueness-margin default (same floor the v1 adapter uses when a
# budget does not override tau).
_DEFAULT_TAU = 0.10


@dataclass
class CascadeConfig:
    top_k: int = 20  # how many dense candidates to re-rank
    tau: float = _DEFAULT_TAU  # margin below which cascade escalates to v1
    recall_target: float = 0.98  # DEV recall@20 at/above which dense alone suffices (G2)
    detrend_order: int = 1  # passed through to the v1 fingerprint extractor
    # Guard: on escalation, v1 may overturn the dense rank-1 only if it prefers a
    # DIFFERENT candidate by at least this much fingerprint similarity. Calibrated
    # on DEV (plateau 0.20-0.25); a blind re-rank (0.0) is net-negative on DEV
    # because it demotes already-correct dense picks when margins are tight.
    accept_margin: float = 0.20


@dataclass
class CascadeResult:
    candidates: list[Candidate]  # re-ranked (or passed-through) candidates
    mode_used: CascadeMode  # the path actually taken
    escalated: bool  # True iff a v1 re-rank was performed
    margin: float  # dense rank-1 minus rank-2 score (inf if < 2 candidates)


def _dense_margin(dense: list[Candidate]) -> float:
    if len(dense) < 2:
        return float("inf")
    return float(dense[0].score - dense[1].score)


def _as_float_gray(img: np.ndarray) -> np.ndarray:
    """v1 fingerprint expects a float frame; map 8/16-bit gray to [0, 1]."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return (a.astype(np.float32)) / 255.0
    if a.dtype == np.uint16:
        return (a.astype(np.float32)) / 65535.0
    return a.astype(np.float32)


def _v1_rerank(
    dense: list[Candidate],
    template: np.ndarray,
    search: np.ndarray,
    cfg: CascadeConfig,
    *,
    guard: bool,
) -> list[Candidate]:
    """Re-rank the top-K by v1 fingerprint similarity; keep residual as fallback.

    Candidates whose fingerprint is undefined (out of bounds / degenerate patch)
    fall back below every scored candidate so they never jump ahead of a copy.

    With `guard` (the cascade path), v1 is allowed to overturn the dense rank-1
    only when it prefers a different candidate by at least cfg.accept_margin;
    otherwise the dense rank-1 is kept and only the tail is reordered by v1. The
    pure-v1 path passes guard=False for an unconditional v1 ordering.
    """
    from localization.ler_fingerprint import (  # lazy, read-only (frozen v1)
        extract_fingerprint,
        score_candidate,
    )

    th, tw = template.shape[:2]
    ref_fp = extract_fingerprint(_as_float_gray(template), detrend_order=cfg.detrend_order)
    sf = _as_float_gray(search)

    k = dense[: cfg.top_k]
    worst = min((c.score for c in k), default=0.0)
    v1: list[float] = []
    for c in k:
        fp = score_candidate(ref_fp, sf, c.x_px + tw / 2.0, c.y_px + th / 2.0, tw, th)
        v1.append(float(fp) if np.isfinite(fp) else worst - 1.0)

    order = sorted(range(len(k)), key=lambda i: (-v1[i], -k[i].score))
    if guard and order and order[0] != 0 and (v1[order[0]] - v1[0]) < cfg.accept_margin:
        # v1 is not confident enough to overturn dense's rank-1: keep it at the top.
        order = [0] + [i for i in order if i != 0]

    out: list[Candidate] = []
    for rank, i in enumerate(order, start=1):
        c = k[i]
        out.append(
            Candidate(
                rank=rank,
                x_px=c.x_px,
                y_px=c.y_px,
                score=v1[i],  # fused score is now the v1 fingerprint similarity
                lattice_score=c.lattice_score,
                roughness_score=c.score,  # preserve the S4 residual score
            )
        )
    return out


def rerank(
    dense: list[Candidate],
    template: np.ndarray,
    search: np.ndarray,
    *,
    mode: CascadeMode = "cascade",
    tau: float | None = None,
    cfg: CascadeConfig | None = None,
) -> CascadeResult:
    """Re-rank S4 dense candidates per `mode`. `tau` overrides cfg.tau if given."""
    cfg = cfg or CascadeConfig()
    tau = cfg.tau if tau is None else float(tau)
    margin = _dense_margin(dense)

    if not dense or mode == "dense":
        return CascadeResult(candidates=list(dense), mode_used="dense", escalated=False, margin=margin)

    if mode == "v1":
        return CascadeResult(
            candidates=_v1_rerank(dense, template, search, cfg, guard=False),
            mode_used="v1",
            escalated=True,
            margin=margin,
        )

    # cascade: escalate only when the dense decision is not confident.
    if margin < tau:
        return CascadeResult(
            candidates=_v1_rerank(dense, template, search, cfg, guard=True),
            mode_used="cascade",
            escalated=True,
            margin=margin,
        )
    return CascadeResult(candidates=list(dense), mode_used="cascade", escalated=False, margin=margin)
