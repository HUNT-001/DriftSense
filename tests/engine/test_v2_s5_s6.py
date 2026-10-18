"""v2 S5-S6 tests: cascade re-ranking (mode routing, the confidence guard, set
invariance, and a slow cascade-vs-dense check on DEV), plus the S5 contour
scaffold and the hot-path op guard for the new modules.

v1 (localization.*) is imported read-only by cascade for the fingerprint re-rank;
it is never modified here.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from driftsense.v2.cascade import CascadeConfig, rerank
from driftsense.v2.contour import ContourConfig, extract_contours
from driftsense.v2.lattice import detect_lattice
from driftsense.v2.roughness import residual
from driftsense.v2.score import score_residual

V2_DIR = Path(__file__).resolve().parents[2] / "driftsense" / "v2"
DATASET_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "dataset"
TIERS = ("clean", "nominal", "hard", "ambiguous")
THR_PX = 3.0


def rough_bars(*, pitch: float = 16.0, n: int = 300, rough_px: float = 0.15, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    _, x = np.mgrid[0:n, 0:n].astype(np.float64)
    jitter = np.cumsum(rng.normal(0, rough_px, n))
    xr = x + jitter[:, None]
    img = 128 + 70 * np.cos(2 * np.pi * xr / pitch) + rng.normal(0, 3, (n, n))
    return np.clip(img, 0, 255).astype(np.uint8)


def _dense_pair(seed: int = 3):
    search = rough_bars(seed=seed)
    tx, ty, th, tw = 112, 96, 100, 96
    tmpl = search[ty : ty + th, tx : tx + tw].copy()
    lat = detect_lattice(search)
    dense = score_residual(residual(tmpl, lat), residual(search, lat), lat, max_candidates=20).candidates
    return dense, tmpl, search, (tx, ty)


def _coords(cands) -> set[tuple[int, int]]:
    return {(round(c.x_px), round(c.y_px)) for c in cands}


# --------------------------------------------------------------------------- #
# S6 cascade — mode routing
# --------------------------------------------------------------------------- #
def test_dense_mode_passes_through_unchanged() -> None:
    dense, tmpl, search, _ = _dense_pair()
    res = rerank(dense, tmpl, search, mode="dense")
    assert res.mode_used == "dense" and not res.escalated
    assert [(round(c.x_px), round(c.y_px)) for c in res.candidates] == [
        (round(c.x_px), round(c.y_px)) for c in dense
    ]


def test_v1_mode_always_escalates() -> None:
    dense, tmpl, search, _ = _dense_pair()
    res = rerank(dense, tmpl, search, mode="v1")
    assert res.mode_used == "v1" and res.escalated


def test_cascade_escalates_only_when_margin_below_tau() -> None:
    dense, tmpl, search, _ = _dense_pair()
    margin = dense[0].score - dense[1].score
    assert margin > 0.1  # the synthetic true copy is unambiguous
    assert not rerank(dense, tmpl, search, mode="cascade", tau=0.1).escalated  # confident -> dense
    assert rerank(dense, tmpl, search, mode="cascade", tau=margin + 0.5).escalated  # forced


def test_empty_candidates_route_to_dense() -> None:
    _dense, tmpl, search, _ = _dense_pair()
    res = rerank([], tmpl, search, mode="cascade", tau=1.0)
    assert res.candidates == [] and not res.escalated and res.mode_used == "dense"


# --------------------------------------------------------------------------- #
# S6 cascade — invariants and the guard
# --------------------------------------------------------------------------- #
def test_rerank_preserves_the_candidate_set() -> None:
    """Re-ranking reorders a fixed set; it must not add or drop members (so it
    cannot change recall@K)."""
    dense, tmpl, search, _ = _dense_pair()
    for mode in ("v1", "cascade"):
        res = rerank(dense, tmpl, search, mode=mode, tau=10.0)  # force escalation
        assert _coords(res.candidates) == _coords(dense)
        assert [c.rank for c in res.candidates] == list(range(1, len(res.candidates) + 1))


def test_guard_keeps_correct_dense_top1_on_escalation() -> None:
    """Even when forced to escalate, the guard keeps the dense rank-1 unless v1 is
    confident a different candidate is better — here v1 agrees, so it stays."""
    dense, tmpl, search, (tx, ty) = _dense_pair()
    res = rerank(dense, tmpl, search, mode="cascade", tau=10.0)
    assert res.escalated
    top = res.candidates[0]
    assert abs(top.x_px - tx) <= 1.5 and abs(top.y_px - ty) <= 1.5
    assert top.roughness_score is not None  # S4 residual score preserved on re-rank


def test_v1_rerank_scores_are_fingerprint_similarities() -> None:
    dense, tmpl, search, _ = _dense_pair()
    res = rerank(dense, tmpl, search, mode="v1")
    assert res.candidates[0].score <= 1.0 + 1e-6  # similarity, not an NCC residual
    assert res.candidates[0].roughness_score == pytest.approx(dense[0].score)


# --------------------------------------------------------------------------- #
# S5 contour scaffold
# --------------------------------------------------------------------------- #
def test_contour_entry_point_is_scaffolded_for_a5() -> None:
    img = rough_bars(seed=1)
    with pytest.raises(NotImplementedError, match="DRAM contour: A5"):
        extract_contours(img, detect_lattice(img), ContourConfig())


# --------------------------------------------------------------------------- #
# hot-path op guard (S5-S6 modules)
# --------------------------------------------------------------------------- #
def test_no_hot_ops_or_pixel_loops_in_s5_s6_modules() -> None:
    pixel_loop = re.compile(r"for\s+\w+\s+in\s+range\([^)]*\b(shape|size|rows|cols|height|width)\b", re.I)
    for name in ("cascade.py", "contour.py"):
        src = (V2_DIR / name).read_text(encoding="utf-8")
        code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert "cv2.matchTemplate" not in code, f"{name}: matchTemplate (belongs in score.py)"
        assert "cv2.minMaxLoc" not in code, f"{name}: minMaxLoc (belongs in score.py)"
        assert "cv2.warpAffine" not in code, f"{name}: warpAffine (use warpPerspective)"
        assert not pixel_loop.search(code), f"{name}: Python per-pixel loop in hot path"


# --------------------------------------------------------------------------- #
# cascade vs dense on DEV (slow)
# --------------------------------------------------------------------------- #
def _dev_pairs(tier: str, n: int):
    d = DATASET_ROOT / tier
    for p in json.loads((d / "manifest.json").read_text())["pairs"][:n]:
        pid = p["pair_id"]
        t = cv2.imdecode(np.frombuffer((d / f"pair_{pid:04d}_ref.png").read_bytes(), np.uint8), 0)
        s = cv2.imdecode(np.frombuffer((d / f"pair_{pid:04d}_search.png").read_bytes(), np.uint8), 0)
        yield t, s, p["gt_x"], p["gt_y"], p["ref_w"], p["ref_h"]


@pytest.mark.slow
def test_cascade_matches_or_improves_dense_top1_on_dev() -> None:
    """Guarded cascade must not lose top-1 to the dense path, and must leave
    recall@20 unchanged (it only reorders a fixed set)."""

    def hit(c, gx, gy, rw, rh) -> bool:
        return math.hypot(c.x_px + rw / 2.0 - gx, c.y_px + rh / 2.0 - gy) <= THR_PX

    dense_top1 = casc_top1 = dense_rec = casc_rec = 0
    cfg = CascadeConfig()
    for tier in TIERS:
        for t, s, gx, gy, rw, rh in _dev_pairs(tier, 8):
            lat = detect_lattice(s)
            dense = score_residual(residual(t, lat), residual(s, lat), lat, max_candidates=20).candidates
            casc = rerank(dense, t, s, mode="cascade", cfg=cfg).candidates
            dense_top1 += int(bool(dense) and hit(dense[0], gx, gy, rw, rh))
            casc_top1 += int(bool(casc) and hit(casc[0], gx, gy, rw, rh))
            dense_rec += int(any(hit(c, gx, gy, rw, rh) for c in dense))
            casc_rec += int(any(hit(c, gx, gy, rw, rh) for c in casc))

    assert casc_rec == dense_rec, f"re-rank changed recall@20 ({dense_rec}->{casc_rec})"
    assert casc_top1 >= dense_top1, f"cascade top-1 {casc_top1} < dense {dense_top1}"
