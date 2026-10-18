"""S4 — residual-on-residual scoring at lattice-consistent offsets.

Pure functions only. One normalised cross-correlation of the template roughness
residual against the search roughness residual (TM_CCOEFF_NORMED), then keep only
candidate offsets that are consistent with the S1 lattice: the true copy and its
periodic replicas sit on the real-space lattice through the global peak, so we
restrict the candidate set to that lattice and suppress everything else. Ranking
over that set is what separates the true copy (high roughness correlation) from a
replica (low), which plain NCC cannot do.

Hot-path exceptions (as established in S0–S2): cv2.matchTemplate and
cv2.minMaxLoc live *only* here. NMS is grayscale dilation (KleidiCV-listed). No
Python per-pixel loops; the small loop over lattice (i, j) indices is O(#replicas),
not O(pixels).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from driftsense.contracts import Candidate
from driftsense.v2.lattice import Lattice
from driftsense.v2.roughness import Residual


@dataclass
class ScoreConfig:
    nms_ksize: int = 5  # dilation window for local-max non-max suppression
    lattice_tol_px: float = 1.5  # how close a peak must sit to a lattice node
    max_lattice_steps: int = 6  # search |i|,|j| <= this many lattice vectors
    min_score: float = 0.05  # floor below which a peak is not a candidate


@dataclass
class ScoreResult:
    candidates: list[Candidate]  # ranked, lattice-consistent (top-left coords)
    score_map: np.ndarray  # full TM_CCOEFF_NORMED map (search-template coords)
    anchor_xy: tuple[int, int]  # global-peak top-left offset (lattice origin)


def _real_space_vectors(lattice: Lattice) -> np.ndarray | None:
    """Real-space lattice translations (rows a1, a2) from reciprocal k1, k2.

    a_i . k_j = delta_ij, so A = inv(K)^T with K = [k1; k2]. A translation by a1
    (or a2) maps the ideal structure onto itself — these are the only shifts a
    periodic replica can occur at. Returns None if the lattice is degenerate.
    """
    k = np.array([lattice.k1, lattice.k2], dtype=np.float64)
    if abs(float(np.linalg.det(k))) < 1e-9:
        return None
    return np.linalg.inv(k).T  # rows a1, a2 in px


def _nms_peaks(score_map: np.ndarray, ksize: int) -> np.ndarray:
    """Local maxima of `score_map` via grayscale dilation. Returns (N,) bool-at."""
    dil = cv2.dilate(score_map, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize)))
    return score_map >= dil  # ties resolved later by score ordering


def score_residual(
    template: Residual,
    search: Residual,
    lattice: Lattice,
    max_candidates: int = 20,
    cfg: ScoreConfig | None = None,
) -> ScoreResult:
    """Score the template residual across the search residual; rank lattice nodes."""
    cfg = cfg or ScoreConfig()
    t = template.residual
    s = search.residual
    if t.shape[0] > s.shape[0] or t.shape[1] > s.shape[1]:
        raise ValueError("template residual larger than search residual")

    score_map = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
    _, _, _, (ax, ay) = cv2.minMaxLoc(score_map)  # anchor = global peak (top-left)
    h, w = score_map.shape

    # Candidate offsets = the real-space lattice through the anchor, clipped to
    # the valid score-map region. These are the true copy + periodic replicas.
    peak_mask = _nms_peaks(score_map, cfg.nms_ksize)
    nodes: list[tuple[int, int]] = [(ax, ay)]
    A = _real_space_vectors(lattice)
    if A is not None:
        a1, a2 = A[0], A[1]
        rng = range(-cfg.max_lattice_steps, cfg.max_lattice_steps + 1)
        for i in rng:
            for j in rng:
                if i == 0 and j == 0:
                    continue
                px = ax + i * a1[0] + j * a2[0]
                py = ay + i * a1[1] + j * a2[1]
                xi, yi = int(round(px)), int(round(py))
                if 0 <= xi < w and 0 <= yi < h:
                    nodes.append((xi, yi))

    # Snap each node to the nearest NMS local-max within the tolerance window and
    # read its score; dedupe by snapped location, keep the best per location.
    tol = max(1, int(round(cfg.lattice_tol_px)))
    best: dict[tuple[int, int], float] = {}
    for xi, yi in nodes:
        x0, x1 = max(0, xi - tol), min(w, xi + tol + 1)
        y0, y1 = max(0, yi - tol), min(h, yi + tol + 1)
        win = score_map[y0:y1, x0:x1]
        wlm = peak_mask[y0:y1, x0:x1]
        if not wlm.any():
            continue
        local = np.where(wlm, win, -np.inf)
        off = int(np.argmax(local))
        ly, lx = divmod(off, local.shape[1])
        sx, sy = x0 + lx, y0 + ly
        sc = float(score_map[sy, sx])
        if sc >= cfg.min_score and sc > best.get((sx, sy), -np.inf):
            best[(sx, sy)] = sc

    ranked = sorted(best.items(), key=lambda kv: -kv[1])[:max_candidates]
    candidates = [
        Candidate(rank=i + 1, x_px=float(x), y_px=float(y), score=sc, roughness_score=sc)
        for i, ((x, y), sc) in enumerate(ranked)
    ]
    return ScoreResult(candidates=candidates, score_map=score_map, anchor_xy=(ax, ay))
