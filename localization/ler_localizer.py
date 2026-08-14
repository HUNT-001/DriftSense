"""
Two-stage LER-aware localizer (Stage 2).

    1000x1000 search
           |
           v
    Stage A: coarse candidate generation      (NCC score map + NMS -> top-K)
           |
           v
    Stage B: LER fingerprint re-ranking       (the disambiguator)
           |
           v
    Stage C: sub-pixel refinement             (parabolic peak fit)
           |
           v
         (x, y)

WHY THIS ORDER
--------------
Stage A is cheap and has high RECALL but poor PRECISION on periodic
structures: the true location is almost always among its top-K peaks, but the
top-1 peak is frequently a period alias.  Stage B has no way to search a
million positions, but it discriminates between a handful of candidates
extremely well (measured d' ~ 3.9-5.0 versus ~1.2 for raw NCC).

So the two stages are complementary rather than redundant: A supplies recall,
B supplies precision.

HONEST SCOPE NOTE
-----------------
Stage A is currently NCC-based.  Frequency-domain candidate generation
(pitch/orientation estimation to place candidates on the lattice directly) is
the natural replacement and is Stage 2A; it would cut cost and make candidate
placement structure-aware.  It is not required for the disambiguation claim,
which is what Stage B tests.

The `recall_at_k` field returned by `localize` is the key diagnostic: if the
truth is not in the candidate set, re-ranking cannot possibly recover it, and
the failure belongs to Stage A, not Stage B.  Reporting the two separately
keeps the attribution honest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from localization.ler_fingerprint import (extract_fingerprint,
                                           fingerprint_similarity,
                                           LERFingerprint)


# ===========================================================================

@dataclass
class Candidate:
    x: float             # centre-x in search image
    y: float             # centre-y in search image
    ncc: float           # Stage-A score
    fp: float = float("nan")   # Stage-B fingerprint similarity
    combined: float = float("nan")


@dataclass
class LocalizationResult:
    method: str
    pred_x: float
    pred_y: float
    score: float
    runtime_ms: float
    candidates: list[Candidate] = field(default_factory=list)
    # Diagnostics
    n_candidates: int = 0
    rank_of_truth: int = -1     # filled in by the evaluator when GT is known
    stage_a_pred: tuple[float, float] = (float("nan"), float("nan"))

    def error(self, gt_x: float, gt_y: float) -> float:
        return float(np.hypot(self.pred_x - gt_x, self.pred_y - gt_y))


# ===========================================================================
# Stage A — coarse candidate generation
# ===========================================================================

def ncc_score_map(ref: np.ndarray, search: np.ndarray) -> np.ndarray:
    return cv2.matchTemplate(search.astype(np.float32),
                              ref.astype(np.float32),
                              cv2.TM_CCOEFF_NORMED)


def top_k_peaks(score_map: np.ndarray, k: int,
                 min_distance: int = 5) -> list[tuple[int, int, float]]:
    """
    Greedy non-maximum suppression over the score map.

    `min_distance` must be kept SMALLER than the structural period, otherwise
    suppression would delete the very alias candidates we need to rank — and
    could delete the true peak when it sits next to a stronger alias.
    """
    sm = score_map.copy()
    peaks = []
    for _ in range(k):
        _, val, _, loc = cv2.minMaxLoc(sm)
        if not np.isfinite(val) or val < -1.0:
            break
        peaks.append((loc[0], loc[1], float(val)))
        x0 = max(0, loc[0] - min_distance)
        x1 = min(sm.shape[1], loc[0] + min_distance + 1)
        y0 = max(0, loc[1] - min_distance)
        y1 = min(sm.shape[0], loc[1] + min_distance + 1)
        sm[y0:y1, x0:x1] = -np.inf
    return peaks


# ===========================================================================
# Stage C — sub-pixel refinement
# ===========================================================================

def subpixel_peak(score_map: np.ndarray, px: int, py: int) -> tuple[float, float]:
    """Parabolic fit on the 3x3 neighbourhood of a score-map peak."""
    h, w = score_map.shape
    dx = dy = 0.0
    if 0 < px < w - 1:
        a, b, c = score_map[py, px - 1], score_map[py, px], score_map[py, px + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            dx = float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    if 0 < py < h - 1:
        a, b, c = score_map[py - 1, px], score_map[py, px], score_map[py + 1, px]
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            dy = float(np.clip(0.5 * (a - c) / den, -1.0, 1.0))
    return dx, dy


# ===========================================================================
# Full pipeline
# ===========================================================================

def localize(ref: np.ndarray,
             search: np.ndarray,
             top_k: int = 40,
             nms_distance: int = 5,
             fp_weight: float = 1.0,
             ncc_weight: float = 0.0,
             return_candidates: bool = True,
             candidate_source: str = "ncc",
             snap_radius: int = 5,
             max_lattice_candidates: int = 2500,
             prefilter_keep: int = 0) -> LocalizationResult:
    """
    Localize `ref` inside `search` using coarse NCC + LER fingerprint ranking.

    Parameters
    ----------
    top_k        : number of Stage-A candidates to re-rank
    nms_distance : suppression radius in the NCC score map (px)
    fp_weight    : weight on fingerprint similarity in the final score
    ncc_weight   : weight on NCC in the final score.  0.0 gives pure
                   fingerprint re-ranking, which is the cleanest test of the
                   Stage-2 claim; a small positive value acts as a tie-break
                   when the fingerprint is undefined.
    candidate_source :
        'ncc'     — top-K NCC peaks (Stage-1 behaviour).
        'lattice' — ALL spectrally-consistent lattice placements, snapped to
                    the local NCC max but deliberately NOT trimmed by NCC.

    WHY 'lattice' DOES NOT TRIM BY NCC
    ----------------------------------
    Measured: the untrimmed lattice set contains the truth ~96% of the time on
    the hard tier, versus 77.5% for NCC top-150.  But snapping those
    candidates to local NCC maxima and then RANKING them by NCC collapses the
    result back to exactly 77.5% — the ranking re-imposes the photometric
    criterion and discards the structural constraint that generation just
    established.

    The lattice's value is that it reduces ~800,000 possible placements to
    ~1500 structurally valid ones (a ~500x reduction) while keeping the truth.
    Ranking that set is the fingerprint's job, not NCC's.
    """
    t0 = time.perf_counter()
    rh, rw = ref.shape[:2]

    # ---- Stage A ----
    smap = ncc_score_map(ref, search)
    peaks = top_k_peaks(smap, top_k, nms_distance)
    if not peaks:
        return LocalizationResult("LER2Stage", float("nan"), float("nan"),
                                   float("nan"),
                                   (time.perf_counter() - t0) * 1000)

    stage_a = (peaks[0][0] + rw / 2.0, peaks[0][1] + rh / 2.0)

    if candidate_source == "lattice":
        from localization.spectral import lattice_candidates, snap_to_local_ncc
        lat, basis = lattice_candidates(search, ref,
                                         max_candidates=max_lattice_candidates)
        if lat:
            lat = snap_to_local_ncc(lat, smap, (rh, rw), snap_radius)
            # Optional structural prefilter: shrink the candidate set using the
            # low-frequency envelope (CD/contrast field), which is neither LER
            # nor NCC.  Recall-oriented: keeps the most reference-like `keep`.
            if prefilter_keep and len(lat) > prefilter_keep:
                from localization.structural import prefilter_candidates
                pitch = (min(basis.periods) if basis.ok else 12.0)
                lat = prefilter_candidates(ref, search, lat, pitch,
                                            keep=prefilter_keep)
            peaks = []
            for (cx, cy) in lat:
                x = int(round(cx - rw / 2.0))
                y = int(round(cy - rh / 2.0))
                if 0 <= x < smap.shape[1] and 0 <= y < smap.shape[0]:
                    peaks.append((x, y, float(smap[y, x])))
        # If the spectral basis failed, silently fall back to the NCC peaks.

    # ---- Stage B ----
    ref_fp = extract_fingerprint(ref)
    cands: list[Candidate] = []

    for (tlx, tly, ncc) in peaks:
        cx = tlx + rw / 2.0
        cy = tly + rh / 2.0
        patch = search[tly:tly + rh, tlx:tlx + rw]
        if patch.shape[0] != rh or patch.shape[1] != rw:
            continue
        if ref_fp.is_empty:
            fp = float("nan")
        else:
            fp = fingerprint_similarity(ref_fp, extract_fingerprint(patch))
        cands.append(Candidate(x=cx, y=cy, ncc=ncc, fp=fp))

    if not cands:
        return LocalizationResult("LER2Stage", float("nan"), float("nan"),
                                   float("nan"),
                                   (time.perf_counter() - t0) * 1000)

    # Combine.  Candidates whose fingerprint is undefined fall back to NCC
    # alone so they can still be selected if nothing else is measurable.
    for c in cands:
        fp = c.fp if np.isfinite(c.fp) else -1.0
        c.combined = fp_weight * fp + ncc_weight * c.ncc

    best = max(cands, key=lambda c: c.combined)

    # ---- Stage C ----
    tlx = int(round(best.x - rw / 2.0))
    tly = int(round(best.y - rh / 2.0))
    tlx = int(np.clip(tlx, 0, smap.shape[1] - 1))
    tly = int(np.clip(tly, 0, smap.shape[0] - 1))
    dx, dy = subpixel_peak(smap, tlx, tly)

    dt = (time.perf_counter() - t0) * 1000
    return LocalizationResult(
        method="LER2Stage",
        pred_x=best.x + dx,
        pred_y=best.y + dy,
        score=best.combined,
        runtime_ms=dt,
        candidates=cands if return_candidates else [],
        n_candidates=len(cands),
        stage_a_pred=stage_a,
    )


# ===========================================================================
# Diagnostics
# ===========================================================================

def candidate_recall(cands: list[Candidate],
                      gt_x: float, gt_y: float,
                      tol: float = 5.0) -> tuple[bool, int]:
    """
    Was the ground truth present in the candidate set, and at what NCC rank?

    Returns (found, rank) where rank is the index in NCC-sorted order, or -1.
    """
    order = sorted(range(len(cands)), key=lambda i: -cands[i].ncc)
    for rank, i in enumerate(order):
        c = cands[i]
        if np.hypot(c.x - gt_x, c.y - gt_y) <= tol:
            return True, rank
    return False, -1
