"""v2 G2 tests: S8 subpixel accuracy, S9 explain artifacts, the full S0->S9 engine
on FinFET + DRAM fixtures through the public contract, mode routing, and the
hot-path op guard for the G2 modules.

v1 (localization.*) is imported read-only by the engine's DENSE/cascade paths.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from driftsense.contracts import Budget, LocalizeResult, Mode
from driftsense.v2.engine import localize
from driftsense.v2.explain import build_all
from driftsense.v2.subpixel import SubpixelConfig, refine

V2_DIR = Path(__file__).resolve().parents[2] / "driftsense" / "v2"


# --------------------------------------------------------------------------- #
# synthetic fixtures (band-limited; analytic shifts, no resampling artifacts)
# --------------------------------------------------------------------------- #
def _finfet(n: int = 220, theta: float = 0.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    t = np.radians(theta)
    xr = x * np.cos(t) + y * np.sin(t)
    yr = -x * np.sin(t) + y * np.cos(t)
    img = 128 + 60 * np.cos(2 * np.pi * xr / 10.0) + 30 * np.cos(2 * np.pi * yr / 28.0)
    img = img + rng.normal(0, 3, (n, n))
    return np.clip(img, 0, 255).astype(np.uint8)


def _dram(n: int = 220, seed: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    dots = (0.5 + 0.5 * np.cos(2 * np.pi * x / 14.0)) * (0.5 + 0.5 * np.cos(2 * np.pi * y / 18.0))
    img = 40 + 180 * dots + rng.normal(0, 3, (n, n))
    return np.clip(img, 0, 255).astype(np.uint8)


def _png(a: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", a)
    assert ok
    return buf.tobytes()


def _bandlimited(n: int, dx: float, dy: float, seed: int = 7) -> np.ndarray:
    """A band-limited field sampled at coordinates shifted by (dx, dy) -- an exact
    sub-pixel shift with no interpolation."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    xs, ys = x - dx, y - dy
    img = np.zeros((n, n))
    for _ in range(12):
        fx, fy = rng.uniform(-0.25, 0.25, 2)
        img += np.cos(2 * np.pi * (fx * xs + fy * ys) + rng.uniform(0, 2 * np.pi))
    return img.astype(np.float32)


# --------------------------------------------------------------------------- #
# S8 subpixel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dx,dy", [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.3, -0.4), (-0.5, 0.5)])
def test_subpixel_recovers_half_pixel_shifts(dx: float, dy: float) -> None:
    t = _bandlimited(128, 0.0, 0.0)
    p = _bandlimited(128, dx, dy)
    r = refine(t, p)
    assert abs(r.dx_px - dx) < 0.05 and abs(r.dy_px - dy) < 0.05
    assert r.method in ("phase+ecc", "phase")


def test_subpixel_px_to_nm_and_zero_shift() -> None:
    t = _bandlimited(96, 0.0, 0.0)
    r = refine(t, t, pixel_size_nm=2.5)
    assert abs(r.dx_px) < 0.05 and abs(r.dy_px) < 0.05
    assert r.dx_nm == pytest.approx(r.dx_px * 2.5)


def test_subpixel_requires_matching_shapes() -> None:
    with pytest.raises(ValueError, match="match in shape"):
        refine(_bandlimited(64, 0, 0), _bandlimited(48, 0, 0), cfg=SubpixelConfig())


# --------------------------------------------------------------------------- #
# S9 explain
# --------------------------------------------------------------------------- #
def test_explain_builds_png_artifacts_with_alt_text() -> None:
    from driftsense.contracts import Candidate

    search = _dram()
    cands = [Candidate(rank=i + 1, x_px=20 + 14 * i, y_px=30, score=0.9 - 0.1 * i) for i in range(6)]
    ncc = np.random.default_rng(0).random((80, 90)).astype(np.float32)
    arts = build_all(search, cands, ncc, ncc.copy(), ref_w=80, ref_h=70)
    kinds = {a.kind for a in arts}
    assert {"heatmap", "candidates", "overlay"} <= kinds
    for a in arts:
        assert a.png[:8] == b"\x89PNG\r\n\x1a\n" and a.alt  # real PNG + non-empty alt


# --------------------------------------------------------------------------- #
# full S0->S9 engine via the contract, on FinFET + DRAM
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make", [_finfet, _dram])
def test_engine_s0_s9_returns_complete_result(make) -> None:
    search = make()
    tmpl = search[40:140, 50:150].copy()
    r = localize(_png(tmpl), _png(search), Mode.AUTO, Budget(max_candidates=20))
    LocalizeResult.model_validate_json(r.model_dump_json())  # full contract round-trip
    assert r.engine == "v2" and len(r.candidates) >= 1
    stages = [t.stage for t in r.timings]
    assert stages == ["ingest", "lattice", "rectify", "roughness", "score", "cascade", "subpixel", "explain"]
    assert r.total_ms >= 0 and r.artifacts  # AUTO produces explain artifacts


def test_fast_mode_is_lean_no_artifacts() -> None:
    search = _finfet()
    tmpl = search[40:140, 50:150].copy()
    r = localize(_png(tmpl), _png(search), Mode.FAST, Budget(max_candidates=20))
    assert r.engine == "v2" and len(r.candidates) >= 1
    assert r.artifacts == []  # FAST skips the S9 artifact build (production default)


def test_engine_guarantees_at_least_one_candidate_on_noise() -> None:
    rng = np.random.default_rng(0)
    s = (rng.random((200, 200)) * 255).astype(np.uint8)
    r = localize(_png(s[60:140, 60:140].copy()), _png(s), Mode.AUTO, Budget())
    assert len(r.candidates) >= 1  # never emits an empty candidate list


# --------------------------------------------------------------------------- #
# hot-path op guard (G2 modules)
# --------------------------------------------------------------------------- #
def test_no_disallowed_ops_or_pixel_loops_in_g2_modules() -> None:
    pixel_loop = re.compile(r"for\s+\w+\s+in\s+range\([^)]*\b(shape|size|rows|cols|height|width)\b", re.I)
    for name in ("engine.py", "subpixel.py", "explain.py"):
        src = (V2_DIR / name).read_text(encoding="utf-8")
        code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
        assert "cv2.matchTemplate" not in code, f"{name}: matchTemplate belongs in score.py"
        assert "cv2.minMaxLoc" not in code, f"{name}: minMaxLoc belongs in score.py"
        assert "cv2.warpAffine" not in code, f"{name}: use warpPerspective, not warpAffine"
        assert not pixel_loop.search(code), f"{name}: Python per-pixel loop in hot path"
