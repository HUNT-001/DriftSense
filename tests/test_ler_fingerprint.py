"""
Tests for LER fingerprint extraction (Stage 2B).

    python -m tests.test_ler_fingerprint

The central claim of Stage 2 is that the extracted displacement field IS the
injected line-edge roughness.  These tests verify that claim directly against
the generator's ground-truth LER profiles, rather than only checking that the
code runs.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cv2_warp_row(row: np.ndarray, M: np.ndarray, W: int) -> np.ndarray:
    """Shift a single image row by an affine matrix (test helper)."""
    return cv2.warpAffine(row.reshape(1, -1), M, (W, 1),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)[0]

from config import AperiodicityConfig, get_tier
from data_gen.structures import _render_bar, make_ler_profile
from data_gen.sem_effects import AcquisitionParams, render_sem_image
from data_gen.scene_composer import SceneComposer
from localization.ler_fingerprint import (estimate_lattice, estimate_pitch,
                                           estimate_phase, extract_fingerprint,
                                           fingerprint_similarity,
                                           crop_centered, score_candidate)

_tests = []


def test(fn):
    _tests.append(fn)
    return fn


# ---------------------------------------------------------------------------

def _build_known_ler(seed: int, H: int = 200, W: int = 200,
                      pitch: float = 24.0, phase: float = 6.0,
                      sigma: float = 0.6):
    """Render vertical bars with KNOWN per-line LER; return canvas + truth."""
    rng = np.random.default_rng(seed)
    canvas = np.full((H, W), 0.05, dtype=np.float32)
    truth = []
    x = phase
    while x < W:
        ler = make_ler_profile(H, sigma, 18.0, rng)
        _render_bar(canvas, x, axis=0, half_width=3.0,
                    brightness=0.95, offset=ler)
        truth.append((x, ler))
        x += pitch
    return canvas, truth


def _match_truth(af, truth, pitch=24.0, phase=6.0, detrend_order=1):
    """
    Correlate each extracted line against its corresponding true profile.

    The truth is detrended with the SAME polynomial order the extractor used.
    Comparing a linearly-detrended estimate against a merely mean-removed
    truth would be an apples-to-oranges comparison and would understate the
    recovery quality.
    """
    cs = []
    for li, pos in enumerate(af.line_positions):
        ti = int(round((pos - phase) / pitch))
        if ti < 0 or ti >= len(truth):
            continue
        m = af.mask[li]
        if m.sum() < 20:
            continue
        d = af.displacements[li][m]
        t = truth[ti][1][m].astype(np.float64)
        idx = np.arange(len(af.mask[li]), dtype=np.float64)[m]
        if detrend_order <= 0:
            t = t - t.mean()
        else:
            t = t - np.polyval(np.polyfit(idx, t, detrend_order), idx)
        if d.std() < 1e-6 or t.std() < 1e-6:
            continue
        cs.append(float(np.corrcoef(d, t)[0, 1]))
    return cs


# ===========================================================================
# Lattice estimation
# ===========================================================================

@test
def test_pitch_estimation_accurate():
    for true_pitch in (11.0, 18.0, 24.0, 37.0):
        canvas, _ = _build_known_ler(0, pitch=true_pitch, W=300, sigma=0.0)
        pitch, strength = estimate_pitch(canvas.mean(axis=0))
        assert abs(pitch - true_pitch) < 0.5, \
            f"pitch {pitch:.2f} != {true_pitch}"
        assert strength > 0.3, f"weak lattice strength {strength:.2f}"


@test
def test_phase_estimation_subpixel():
    """DFT-based phase must be accurate to well under a pixel."""
    for true_phase in (3.0, 6.5, 11.25, 19.75):
        canvas, _ = _build_known_ler(0, pitch=24.0, phase=true_phase, sigma=0.0)
        lat = estimate_lattice(canvas, axis=0)
        err = (lat.phase - true_phase) % 24.0
        err = min(err, 24.0 - err)
        assert err < 0.6, f"phase err {err:.3f}px for true {true_phase}"


# ===========================================================================
# The central claim: extraction recovers the injected LER
# ===========================================================================

@test
def test_extraction_recovers_injected_ler():
    """Extracted displacement must correlate ~1.0 with the true LER profile."""
    canvas, truth = _build_known_ler(1)
    fp = extract_fingerprint(canvas, axes=(0,))
    assert 0 in fp.axes, "no lattice detected"
    cs = _match_truth(fp.axes[0], truth)
    assert len(cs) >= 5, f"only {len(cs)} lines matched"
    assert np.mean(cs) > 0.95, f"mean corr {np.mean(cs):.3f} too low"


@test
def test_extraction_amplitude_is_unbiased():
    """Recovered RMS displacement must match the injected sigma."""
    for sigma in (0.3, 0.6, 1.0):
        canvas, _ = _build_known_ler(2, sigma=sigma)
        fp = extract_fingerprint(canvas, axes=(0,))
        af = fp.axes[0]
        rms = float(np.std(af.displacements[af.mask]))
        assert abs(rms - sigma) / sigma < 0.25, \
            f"recovered rms {rms:.3f} vs injected {sigma}"


@test
def test_extraction_survives_sem_noise():
    """LER must remain recoverable across the full electron-dose range."""
    canvas, truth = _build_known_ler(3)
    for dose, floor in ((700, 0.90), (150, 0.85), (25, 0.60)):
        acq = AcquisitionParams(
            blur_sigma=0.8, edge_strength=0.3, electrons_per_pixel=dose,
            drift_rate=0.0, read_noise_sigma=0.01, scanline_jitter_sigma=0.0,
            charging_amplitude=0.05, vignette_strength=0.05,
            vignette_cx=0.5, vignette_cy=0.5, brightness_offset=0.0,
            contrast_gain=1.0, gamma=1.0)
        img = render_sem_image(canvas, acq, np.random.default_rng(42))
        fp = extract_fingerprint(img, axes=(0,))
        cs = _match_truth(fp.axes[0], truth)
        assert np.mean(cs) > floor, \
            f"dose {dose}: corr {np.mean(cs):.3f} below {floor}"


@test
def test_flat_when_ler_disabled():
    """With no LER the displacement field must be near zero."""
    canvas, _ = _build_known_ler(4, sigma=0.0)
    fp = extract_fingerprint(canvas, axes=(0,))
    af = fp.axes[0]
    rms = float(np.std(af.displacements[af.mask]))
    assert rms < 0.08, f"spurious displacement rms {rms:.4f} with LER off"


@test
def test_invariant_to_photometric_change():
    """
    Brightness/contrast/gamma differences between acquisitions must not
    change the fingerprint — it encodes geometry, not intensity.
    """
    canvas, truth = _build_known_ler(5)
    a = extract_fingerprint(canvas, axes=(0,))
    bright = np.clip(canvas * 0.7 + 0.15, 0, 1).astype(np.float32)
    b = extract_fingerprint(bright, axes=(0,))
    sim = fingerprint_similarity(a, b)
    assert sim > 0.95, f"photometric change altered fingerprint (sim={sim:.3f})"


# ===========================================================================
# Discrimination — the property the whole method rests on
# ===========================================================================

@test
def test_linear_detrend_removes_rotation_ramp():
    """
    REGRESSION: relative rotation induces a LINEAR RAMP along each line that
    is ~15x larger than the LER amplitude.  Mean-only detrending leaves that
    ramp in the fingerprint, which destroyed discrimination above ~2.5 deg
    (measured: 13% vs NCC's 20% on the hard tier).  Linear detrending must
    suppress it.
    """
    canvas, truth = _build_known_ler(9)
    # Simulate relative rotation as a lateral ramp along the lines
    H, W = canvas.shape
    ramp_px = 8.0                      # ~5 deg over a 100 px patch
    yy = np.arange(H, dtype=np.float32)
    shift = (yy / H) * ramp_px
    rotated = np.empty_like(canvas)
    for y in range(H):
        M = np.float32([[1, 0, shift[y]], [0, 1, 0]])
        rotated[y] = cv2_warp_row(canvas[y], M, W)

    fp0 = extract_fingerprint(rotated, axes=(0,), detrend_order=0)
    fp1 = extract_fingerprint(rotated, axes=(0,), detrend_order=1)
    rms0 = float(np.std(fp0.axes[0].displacements[fp0.axes[0].mask]))
    rms1 = float(np.std(fp1.axes[0].displacements[fp1.axes[0].mask]))
    assert rms0 > 1.5, f"test setup failed to inject a ramp (rms0={rms0:.2f})"
    assert rms1 < rms0 * 0.5, \
        f"linear detrend did not suppress ramp: {rms0:.2f} -> {rms1:.2f}"

    cs = _match_truth(fp1.axes[0], truth, detrend_order=1)
    assert np.mean(cs) > 0.75, \
        f"LER not recovered under rotation ramp (corr={np.mean(cs):.3f})"


@test
def test_adjacent_lines_are_uncorrelated():
    """
    Neighbouring lines must carry independent roughness.  This is what makes
    a one-period shift detectable.
    """
    canvas, truth = _build_known_ler(6)
    fp = extract_fingerprint(canvas, axes=(0,))
    af = fp.axes[0]
    cross = []
    for i in range(af.n_lines - 1):
        m = af.mask[i] & af.mask[i + 1]
        if m.sum() < 40:
            continue
        cross.append(abs(np.corrcoef(af.displacements[i][m],
                                      af.displacements[i + 1][m])[0, 1]))
    assert np.mean(cross) < 0.35, \
        f"adjacent lines correlated ({np.mean(cross):.3f}) — no discrimination"


@test
def test_self_similarity_is_high():
    """A patch compared against itself must score ~1."""
    canvas, _ = _build_known_ler(7)
    fp = extract_fingerprint(canvas, axes=(0,))
    assert fingerprint_similarity(fp, fp) > 0.99


@test
def test_true_location_beats_periodic_replica():
    """
    The core Stage-2 claim, end to end on generated data: fingerprint
    similarity at the TRUE location must exceed similarity at a one-period
    replica.
    """
    tier = get_tier("clean")
    comp = SceneComposer(seed=4242)
    wins = 0
    trials = 0
    for i in range(6):
        ref, search, meta = comp.compose(i, tier, structure_type="dram")
        ref_fp = extract_fingerprint(ref)
        s_true = score_candidate(ref_fp, search, meta.gt_x, meta.gt_y,
                                  meta.ref_w, meta.ref_h)
        s_rep = score_candidate(ref_fp, search,
                                 meta.gt_x + meta.period_x, meta.gt_y,
                                 meta.ref_w, meta.ref_h)
        if not (np.isfinite(s_true) and np.isfinite(s_rep)):
            continue
        trials += 1
        wins += int(s_true > s_rep)
    assert trials >= 4, f"only {trials} usable trials"
    assert wins == trials, f"true location won only {wins}/{trials}"


@test
def test_comparator_does_not_realign_replicas():
    """
    REGRESSION GUARD.  If the comparator ever searches over line-index shifts
    to maximize similarity, it will re-align a periodic replica back onto the
    reference and the method will silently stop working while still looking
    healthy.  A one-period replica must score clearly LOWER than the truth.
    """
    canvas, _ = _build_known_ler(8, H=260, W=260)
    pitch = 24.0
    ref = canvas[40:160, 40:160]
    ref_fp = extract_fingerprint(ref, axes=(0,))

    same = fingerprint_similarity(ref_fp,
                                   extract_fingerprint(canvas[40:160, 40:160],
                                                        axes=(0,)))
    shifted = fingerprint_similarity(
        ref_fp,
        extract_fingerprint(canvas[40:160,
                                    40 + int(pitch):160 + int(pitch)],
                             axes=(0,)))
    assert same > 0.9, f"self-similarity {same:.3f} unexpectedly low"
    assert shifted < 0.5, \
        f"replica scored {shifted:.3f} — comparator is re-aligning!"
    assert same - shifted > 0.4, "insufficient separation"


# ===========================================================================

def main() -> int:
    passed = failed = 0
    print(f"Running {len(_tests)} tests\n" + "=" * 64)
    for fn in _tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}\n          {e}")
            failed += 1
        except Exception:
            print(f"  ERROR {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print("=" * 64)
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
