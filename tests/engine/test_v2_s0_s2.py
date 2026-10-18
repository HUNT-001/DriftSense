"""v2 S0–S2 tests: ingest QC, lattice period/angle, rectification, and the
hot-path op guard. Synthetic FinFET / DRAM / grid / staggered lattices with
known geometry; agreement vs the frozen v1 orientation estimator on golden pairs.

v1 (localization.*) is imported read-only for the agreement check — never modified.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from driftsense.v2.engine import run_s0_s2
from driftsense.v2.ingest import ingest
from driftsense.v2.lattice import detect_lattice
from driftsense.v2.rectify import rectify

V2_DIR = Path(__file__).resolve().parents[2] / "driftsense" / "v2"
HOT_PATH_MODULES = ["ingest.py", "lattice.py", "rectify.py", "engine.py"]


# --------------------------------------------------------------------------- #
# synthetic lattices (ground truth in pixels / degrees)
# --------------------------------------------------------------------------- #
def _coords(n: int, theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    t = np.radians(theta_deg)
    xr = x * np.cos(t) + y * np.sin(t)
    yr = -x * np.sin(t) + y * np.cos(t)
    return xr, yr


def grid(px: float, py: float, *, theta: float = 0.0, n: int = 256, seed: int = 0) -> np.ndarray:
    """Square 2-D cosine lattice (dominant = px axis)."""
    rng = np.random.default_rng(seed)
    xr, yr = _coords(n, theta)
    img = 128 + 60 * np.cos(2 * np.pi * xr / px) + 40 * np.cos(2 * np.pi * yr / py)
    img = img + rng.normal(0, 3, (n, n))
    return np.clip(img, 0, 255).astype(np.uint8)


def finfet(*, theta: float = 0.0, n: int = 256, seed: int = 1) -> np.ndarray:
    """Narrow fin pitch (dominant) + wider gate modulation ⟂ to it."""
    return grid(10.0, 28.0, theta=theta, n=n, seed=seed)


def dram(*, theta: float = 0.0, n: int = 256, seed: int = 2) -> np.ndarray:
    """Dot array: product of two cosines → strong peaks on both axes."""
    rng = np.random.default_rng(seed)
    xr, yr = _coords(n, theta)
    dots = (0.5 + 0.5 * np.cos(2 * np.pi * xr / 14.0)) * (0.5 + 0.5 * np.cos(2 * np.pi * yr / 18.0))
    img = 40 + 180 * dots + rng.normal(0, 3, (n, n))
    return np.clip(img, 0, 255).astype(np.uint8)


def staggered(*, theta: float = 0.0, n: int = 256, seed: int = 3) -> np.ndarray:
    """Two non-orthogonal reciprocal vectors (brick/hex-like)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    a = np.radians(theta)
    b = np.radians(theta + 68.0)
    ua = x * np.cos(a) + y * np.sin(a)
    ub = x * np.cos(b) + y * np.sin(b)
    img = 128 + 60 * np.cos(2 * np.pi * ua / 16.0) + 40 * np.cos(2 * np.pi * ub / 22.0)
    img = img + rng.normal(0, 3, (n, n))
    return np.clip(img, 0, 255).astype(np.uint8)


def _periods(lat) -> tuple[float, float]:
    return 1.0 / float(np.linalg.norm(lat.k1)), 1.0 / float(np.linalg.norm(lat.k2))


def _reduce45(deg: float) -> float:
    a = (deg + 45.0) % 90.0 - 45.0
    return 45.0 if a == -45.0 else a


# --------------------------------------------------------------------------- #
# S0 ingest / QC
# --------------------------------------------------------------------------- #
def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_ingest_accepts_8_and_16_bit_gray() -> None:
    img8 = grid(16, 20)
    r8 = ingest(_png(img8))
    assert r8.image.dtype == np.uint8 and r8.qc.ok

    img16 = img8.astype(np.uint16) * 257
    r16 = ingest(_png(img16))
    assert r16.image.dtype == np.uint16 and r16.qc.ok


def test_ingest_qc_rejects_degenerate_frames() -> None:
    flat = ingest(_png(np.full((64, 64), 128, np.uint8)))
    assert not flat.qc.ok and any("flat" in r for r in flat.qc.reasons)

    white = ingest(_png(np.full((64, 64), 255, np.uint8)))
    assert not white.qc.ok and any("clipped" in r for r in white.qc.reasons)


# --------------------------------------------------------------------------- #
# S1 lattice
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "img,p_dom,p_sub,dom_clear",
    [
        (grid(16.0, 24.0), 16.0, 24.0, True),
        (finfet(), 10.0, 28.0, True),
        (dram(), 14.0, 18.0, False),  # equal-amplitude axes: which is k1 is arbitrary
        (staggered(), 16.0, 22.0, True),
    ],
    ids=["grid", "finfet", "dram", "staggered"],
)
def test_lattice_period_error_below_half_px(img, p_dom, p_sub, dom_clear) -> None:
    lat = detect_lattice(img)
    p1, p2 = _periods(lat)
    # when one axis is clearly stronger, k1 must be that (dominant) axis
    if dom_clear:
        assert abs(p1 - p_dom) < 0.5, (p1, p_dom)
    # the two recovered periods must cover both true axes (order-independent)
    got = sorted([p1, p2])
    want = sorted([p_dom, p_sub])
    assert abs(got[0] - want[0]) < 0.5 and abs(got[1] - want[1]) < 0.5, (got, want)
    # k1, k2 must be non-collinear (explicit 2-D determinant, NumPy-2 safe)
    det = lat.k1[0] * lat.k2[1] - lat.k1[1] * lat.k2[0]
    assert abs(det) > 1e-6


@pytest.mark.parametrize("theta", [0.0, 7.0, -12.0, 20.0, -30.0])
def test_lattice_angle_tracks_rotation(theta) -> None:
    lat = detect_lattice(grid(16.0, 24.0, theta=theta))
    assert abs(_reduce45(lat.angle_deg - theta)) < 1.0


def test_lattice_confidence_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    assert detect_lattice(grid(16, 24)).confidence > 0.5
    assert detect_lattice(noise).confidence < 0.3


# --------------------------------------------------------------------------- #
# S2 rectify
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.0, 12.0, -7.0, 20.0])
def test_rectify_residual_below_point_two_px(theta) -> None:
    img = grid(16.0, 24.0, theta=theta)
    recs = rectify(img, detect_lattice(img))
    assert 1 <= len(recs) <= 4
    assert all(r.residual_px < 0.2 for r in recs), [r.residual_px for r in recs]


@pytest.mark.parametrize("theta", [0.0, 12.0, -7.0, 20.0])
def test_rectify_warp_roundtrip_below_point_one_px(theta) -> None:
    img = grid(16.0, 24.0, theta=theta)
    recs = rectify(img, detect_lattice(img))
    pts = np.array([[[10, 10], [120, 20], [30, 110], [64, 64]]], dtype=np.float64)
    for r in recs:
        back = cv2.perspectiveTransform(cv2.perspectiveTransform(pts, r.H), np.linalg.inv(r.H))
        assert float(np.abs(back - pts).max()) < 0.1


# --------------------------------------------------------------------------- #
# agreement with the frozen v1 orientation estimator (golden pairs)
# --------------------------------------------------------------------------- #
def test_lattice_agreement_with_v1_ge_98pct() -> None:
    from localization.lattice_fit import estimate_orientation  # v1, read-only

    golden = [
        grid(16.0, 24.0, theta=0.0),
        grid(16.0, 24.0, theta=7.0),
        grid(18.0, 22.0, theta=-12.0),
        grid(14.0, 20.0, theta=20.0),
        grid(20.0, 28.0, theta=-30.0),
        finfet(theta=0.0),
        finfet(theta=15.0),
        dram(theta=0.0),
        dram(theta=-9.0),
        grid(16.0, 24.0, theta=33.0),
    ]
    agree = considered = 0
    for img in golden:
        v1_deg, ok = estimate_orientation(img)
        if not ok:
            continue
        considered += 1
        mine = detect_lattice(img).angle_deg
        # orientation is a line: sign convention between the two differs, so the
        # agreement metric is sign-agnostic (compare mod 90, both signs).
        diff = min(abs(_reduce45(mine - v1_deg)), abs(_reduce45(mine + v1_deg)))
        if diff < 1.5:
            agree += 1
    assert considered >= 6, f"too few confident v1 pairs ({considered})"
    assert agree / considered >= 0.98, f"{agree}/{considered} agreed"


# --------------------------------------------------------------------------- #
# engine partial mode + hot-path op guard
# --------------------------------------------------------------------------- #
def test_run_s0_s2_end_to_end_on_a_pair() -> None:
    search = grid(16.0, 24.0, theta=9.0)
    template = search[40:140, 50:150]  # a crop of the scene
    res = run_s0_s2(_png(template), _png(search))
    assert res.search.qc.ok
    assert abs(_periods(res.lattice)[0] - 16.0) < 0.5
    assert res.rectified and res.rectified[0].residual_px < 0.2


def test_localize_is_now_fully_assembled() -> None:
    """localize was a stub through S0-S2; as of the G2 assembly it runs S0->S9 and
    returns a complete v2 LocalizeResult (full-pipeline coverage lives in
    test_v2_g2.py; this just guards that the former NotImplementedError is gone)."""
    from driftsense.contracts import Budget, LocalizeResult, Mode
    from driftsense.v2.engine import localize

    t = _png(grid(16, 24)[30:130, 30:130])
    s = _png(grid(16, 24))
    r = localize(t, s, Mode.AUTO, Budget())
    assert isinstance(r, LocalizeResult) and r.engine == "v2"
    assert len(r.candidates) >= 1 and r.timings


def test_no_matchtemplate_or_pixel_loops_in_hot_paths() -> None:
    pixel_loop = re.compile(r"for\s+\w+\s+in\s+range\([^)]*\b(shape|size|rows|cols|height|width)\b", re.I)
    for name in HOT_PATH_MODULES:
        src = (V2_DIR / name).read_text(encoding="utf-8")
        code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert "cv2.matchTemplate" not in code, f"{name}: matchTemplate in hot path"
        assert "cv2.warpAffine" not in code, f"{name}: warpAffine (use warpPerspective)"
        assert not pixel_loop.search(code), f"{name}: Python per-pixel loop in hot path"
