"""v2 S3–S4 tests: roughness residual extraction, residual-on-residual scoring at
lattice-consistent offsets, and the hot-path op guard for the new modules.

The central S3–S4 claim is that the roughness residual separates the TRUE copy
from a one-period periodic replica, which plain NCC cannot. We test that directly
on synthetic periodic arrays with injected per-line roughness, and (slow) check
that the rate at which it picks the true copy is at least the frozen v1
fingerprint's rate on the same arrays.

v1 (localization.*) is imported read-only for the agreement check — never modified.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from driftsense.v2.lattice import detect_lattice
from driftsense.v2.roughness import RoughnessConfig, residual
from driftsense.v2.score import ScoreConfig, score_residual

V2_DIR = Path(__file__).resolve().parents[2] / "driftsense" / "v2"

# Hot-path op policy for the S3–S4 modules. matchTemplate / minMaxLoc are the
# established exceptions and may appear ONLY in score.py; roughness.py must be
# free of them. Neither may use warpAffine or a Python per-pixel loop.
_MATCH_EXEMPT = {"score.py"}


# --------------------------------------------------------------------------- #
# synthetic periodic arrays with injected per-line roughness
# --------------------------------------------------------------------------- #
def rough_bars(
    *, pitch: float = 16.0, n: int = 300, rough_px: float = 0.15, noise: float = 3.0, seed: int = 0
) -> np.ndarray:
    """Vertical cosine bars whose edges carry a slowly-varying per-row wiggle.

    The cumulative per-row displacement makes each stretch of lines carry its own
    roughness signature, which is what the residual fingerprint keys on.
    """
    rng = np.random.default_rng(seed)
    _, x = np.mgrid[0:n, 0:n].astype(np.float64)
    jitter = np.cumsum(rng.normal(0, rough_px, n))
    xr = x + jitter[:, None]
    img = 128 + 70 * np.cos(2 * np.pi * xr / pitch) + rng.normal(0, noise, (n, n))
    return np.clip(img, 0, 255).astype(np.uint8)


def flat_bars(*, pitch: float = 16.0, n: int = 256, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """Ideal roughness-free cosine bars (optionally with additive noise)."""
    rng = np.random.default_rng(seed)
    _, x = np.mgrid[0:n, 0:n].astype(np.float64)
    img = 128 + 70 * np.cos(2 * np.pi * x / pitch)
    if noise:
        img = img + rng.normal(0, noise, (n, n))
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# S3 — roughness residual
# --------------------------------------------------------------------------- #
def test_snap_odd_kernel_logs_and_snaps_to_supported_size() -> None:
    img = rough_bars(seed=1)
    lat = detect_lattice(img)
    res = residual(img, lat, RoughnessConfig(smooth_len=32))  # even -> snaps to 31
    assert res.smooth_len % 2 == 1
    assert res.smooth_len in (31, 33) or res.smooth_len >= 3
    assert any("snapped" in w for w in res.warnings)


def test_roughness_picks_the_along_line_axis() -> None:
    img = rough_bars(seed=2)  # vertical bars, period horizontal -> smooth columns
    res = residual(img, detect_lattice(img))
    assert res.along_axis == 1


def test_residual_is_zero_mean_in_band() -> None:
    img = rough_bars(seed=3)
    res = residual(img, detect_lattice(img))
    band = res.mask > 0
    assert band.any()
    assert abs(float(res.residual[band].mean())) < 0.05


def test_residual_larger_with_roughness_than_without() -> None:
    """The ideal periodic pattern is removed; only roughness survives."""
    lat_flat = detect_lattice(flat_bars())
    flat = residual(flat_bars(), lat_flat)
    r_flat = float(flat.residual[flat.mask > 0].std()) if (flat.mask > 0).any() else 0.0

    img = rough_bars(rough_px=0.3, noise=0.0, seed=5)
    rough = residual(img, detect_lattice(img))
    r_rough = float(rough.residual[rough.mask > 0].std())
    # locally normalised, so compare the fraction of band energy that is non-trivial
    assert r_rough > 0.3
    assert r_flat < r_rough


# --------------------------------------------------------------------------- #
# S4 — residual-on-residual scoring
# --------------------------------------------------------------------------- #
def _pair(seed: int, pitch: float = 16.0, n: int = 300):
    search = rough_bars(pitch=pitch, n=n, seed=seed)
    tx, ty, th, tw = 112, 96, 100, 96
    tmpl = search[ty : ty + th, tx : tx + tw].copy()
    return search, tmpl, (tx, ty, tw, th)


def test_score_top_candidate_is_the_true_offset() -> None:
    search, tmpl, (tx, ty, _tw, _th) = _pair(seed=3)
    lat = detect_lattice(search)
    res = score_residual(residual(tmpl, lat), residual(search, lat), lat, max_candidates=20)
    assert res.anchor_xy == (tx, ty)
    top = res.candidates[0]
    assert abs(top.x_px - tx) <= 1.5 and abs(top.y_px - ty) <= 1.5
    assert top.score > 0.9


def test_score_candidates_are_lattice_consistent() -> None:
    search, tmpl, (tx, ty, _tw, _th) = _pair(seed=3)
    lat = detect_lattice(search)
    res = score_residual(residual(tmpl, lat), residual(search, lat), lat, max_candidates=20)
    pitch = lat.period_px
    # every replica shares the true row and sits on an integer multiple of the pitch
    for c in res.candidates[1:]:
        if abs(c.y_px - ty) <= 1.5:
            k = (c.x_px - tx) / pitch
            assert abs(k - round(k)) < 0.2, f"candidate x={c.x_px} off-lattice (k={k:.2f})"


def test_score_map_has_full_correlation_shape() -> None:
    search, tmpl, (_tx, _ty, tw, th) = _pair(seed=7)
    lat = detect_lattice(search)
    res = score_residual(residual(tmpl, lat), residual(search, lat), lat)
    assert res.score_map.shape == (search.shape[0] - th + 1, search.shape[1] - tw + 1)


def test_score_rejects_template_larger_than_search() -> None:
    small = rough_bars(n=120, seed=1)
    big = rough_bars(n=300, seed=1)
    lat = detect_lattice(big)
    with pytest.raises(ValueError, match="larger than"):
        score_residual(residual(big, lat), residual(small, lat), lat)


def test_min_score_floor_drops_weak_peaks() -> None:
    search, tmpl, _ = _pair(seed=4)
    lat = detect_lattice(search)
    strict = score_residual(residual(tmpl, lat), residual(search, lat), lat, cfg=ScoreConfig(min_score=0.95))
    assert all(c.score >= 0.95 for c in strict.candidates)
    assert strict.candidates[0].score > 0.95  # the true copy survives the floor


# --------------------------------------------------------------------------- #
# hot-path op guard (S3–S4 modules)
# --------------------------------------------------------------------------- #
def test_no_disallowed_ops_or_pixel_loops_in_s3_s4_hot_paths() -> None:
    pixel_loop = re.compile(r"for\s+\w+\s+in\s+range\([^)]*\b(shape|size|rows|cols|height|width)\b", re.I)
    for name in ("roughness.py", "score.py"):
        src = (V2_DIR / name).read_text(encoding="utf-8")
        code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert "cv2.warpAffine" not in code, f"{name}: warpAffine (use warpPerspective)"
        assert not pixel_loop.search(code), f"{name}: Python per-pixel loop in hot path"
        if name not in _MATCH_EXEMPT:
            assert "cv2.matchTemplate" not in code, f"{name}: matchTemplate outside score.py"
            assert "cv2.minMaxLoc" not in code, f"{name}: minMaxLoc outside score.py"


# --------------------------------------------------------------------------- #
# the S3–S4 claim vs the frozen v1 fingerprint (slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_roughness_picks_true_copy_at_least_v1_rate() -> None:
    """On periodic arrays, residual scoring must beat a one-period replica at
    least as often as the frozen v1 LER fingerprint does."""
    from localization.ler_fingerprint import extract_fingerprint, score_candidate

    pitch = 16.0
    mine = v1 = trials = 0
    for seed in range(8):
        search, tmpl, (tx, ty, tw, th) = _pair(seed=seed, pitch=pitch)
        lat = detect_lattice(search)
        res = score_residual(residual(tmpl, lat), residual(search, lat), lat)
        sm = res.score_map
        rx = tx + int(round(pitch))
        if rx + 1 > sm.shape[1]:
            continue
        trials += 1
        mine += int(sm[ty, tx] > sm[ty, rx])  # true top-left beats replica top-left

        ref_fp = extract_fingerprint((tmpl / 255.0).astype(np.float32))
        sf = (search / 255.0).astype(np.float32)
        s_true = score_candidate(ref_fp, sf, tx + tw / 2.0, ty + th / 2.0, tw, th)
        s_rep = score_candidate(ref_fp, sf, rx + tw / 2.0, ty + th / 2.0, tw, th)
        v1 += int(np.isfinite(s_true) and np.isfinite(s_rep) and s_true > s_rep)

    assert trials >= 6, f"only {trials} usable trials"
    assert mine >= v1, f"residual scoring ({mine}/{trials}) below v1 ({v1}/{trials})"
    assert mine >= trials - 1, f"residual scoring missed too often ({mine}/{trials})"
