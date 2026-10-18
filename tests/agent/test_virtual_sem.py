"""
Tests for :class:`driftsense.sim.VirtualSEM`.

These assert the *twin* properties an alignment agent depends on — persistent
per-feature roughness, dwell-driven noise, honest dose accounting, a hidden
ground truth and the budget caps — not merely shapes and dtypes.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from driftsense.sim import (
    AcquisitionLimitError,
    DoseError,
    DriftModel,
    VirtualSEM,
)

# A stage with no positioning error — used to ISOLATE the imaging properties
# (feature persistence, noise scaling) from the drift the agent must solve.
NODRIFT = DriftModel(bias_sigma=0.0, walk_sigma=0.0, move_base=0.0, move_frac=0.0)


def _decode(png_bytes: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert arr is not None, "PNG failed to decode"
    return arr.astype(np.float32)


def _grab(sem: VirtualSEM, center, fov_nm=220.0, dwell_us=8.0, rot=0.0) -> np.ndarray:
    png, _ = sem.acquire(center, fov_nm, dwell_us, rot)
    return _decode(png)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


# ===========================================================================
# Determinism
# ===========================================================================


def test_same_seed_identical_images():
    """Two instruments with the same seed reproduce acquisitions bit-for-bit."""
    a = VirtualSEM(seed=42, family="finfet", site_size_nm=1400.0, pixel_nm=1.0)
    b = VirtualSEM(seed=42, family="finfet", site_size_nm=1400.0, pixel_nm=1.0)
    pa, ma = a.acquire((50.0, -30.0), 220.0, 6.0, 0.8)
    pb, mb = b.acquire((50.0, -30.0), 220.0, 6.0, 0.8)
    assert pa == pb, "identical seed/args must give identical PNG bytes"
    assert ma == mb, "meta must be reproducible"
    assert a.verify() == b.verify(), "ground-truth error must be reproducible"


def test_different_seed_differs():
    """A different seed yields a different wafer / acquisition."""
    a = _grab(VirtualSEM(seed=1, family="dram", site_size_nm=1200.0), (0.0, 0.0))
    b = _grab(VirtualSEM(seed=2, family="dram", site_size_nm=1200.0), (0.0, 0.0))
    assert not np.array_equal(a, b)


# ===========================================================================
# The twin property: persistent features
# ===========================================================================


def test_features_persist_across_acquisitions():
    """
    Re-imaging the SAME site returns the SAME structure (incl. LER), differing
    only by independent scan noise — while a DIFFERENT site decorrelates.
    Drift is disabled here so the test isolates feature persistence.
    """
    sem = VirtualSEM(seed=11, family="dram", site_size_nm=1500.0, drift=NODRIFT)
    i1 = _grab(sem, (0.0, 0.0), dwell_us=30.0)
    i2 = _grab(sem, (0.0, 0.0), dwell_us=30.0)
    i3 = _grab(sem, (400.0, 300.0), dwell_us=30.0)
    same = _ncc(i1, i2)
    diff = _ncc(i1, i3)
    assert same > 0.9, f"same-site acquisitions not persistent (ncc={same:.3f})"
    assert same - diff > 0.5, f"site not positionally distinct (same={same:.3f}, diff={diff:.3f})"


def test_independent_scan_noise():
    """Two acquisitions of one site are correlated but NOT identical (fresh noise)."""
    sem = VirtualSEM(seed=11, family="dram", site_size_nm=1500.0, drift=NODRIFT)
    i1 = _grab(sem, (0.0, 0.0), dwell_us=10.0)
    i2 = _grab(sem, (0.0, 0.0), dwell_us=10.0)
    assert not np.array_equal(i1, i2), "scan noise must be independent per acquisition"


# ===========================================================================
# Dwell-driven shot noise
# ===========================================================================


def test_noise_scales_inversely_with_sqrt_dwell():
    """
    Shot noise RMS must scale as 1/sqrt(dwell).  The residual between two
    same-config acquisitions isolates noise (structure cancels); drift is off so
    the two frames are perfectly registered.
    """

    def resid_std(dwell: float) -> float:
        sem = VirtualSEM(seed=5, family="dram", site_size_nm=1500.0, drift=NODRIFT)
        a = _grab(sem, (0.0, 0.0), dwell_us=dwell)
        b = _grab(sem, (0.0, 0.0), dwell_us=dwell)
        return float((a - b).std())

    lo = resid_std(2.0)  # low dwell  -> high noise
    hi = resid_std(8.0)  # high dwell -> low noise
    ratio = lo / hi
    expected = np.sqrt(8.0 / 2.0)  # = 2.0
    assert abs(ratio - expected) / expected < 0.25, f"noise ratio {ratio:.2f}, expected ~{expected:.2f}"


# ===========================================================================
# Geometry
# ===========================================================================


def test_scan_rotation_changes_image():
    """A scan rotation materially changes the image but not its size."""
    a = VirtualSEM(seed=9, family="finfet", site_size_nm=1400.0, drift=NODRIFT)
    b = VirtualSEM(seed=9, family="finfet", site_size_nm=1400.0, drift=NODRIFT)
    flat = _grab(a, (0.0, 0.0), dwell_us=20.0, rot=0.0)
    rot = _grab(b, (0.0, 0.0), dwell_us=20.0, rot=5.0)  # same noise stream
    assert flat.shape == rot.shape
    assert _ncc(flat, rot) < 0.97, "rotation had no visible effect"


def test_output_shape_matches_fov_and_pixel():
    """Image dimensions follow fov_nm / pixel_nm."""
    sem = VirtualSEM(seed=3, family="dram", site_size_nm=1200.0, pixel_nm=2.0)
    _, meta = sem.acquire((0.0, 0.0), fov_nm=200.0, dwell_us=5.0)
    assert meta["width_px"] == 100 and meta["height_px"] == 100
    img = _decode(sem.acquire((0.0, 0.0), 200.0, 5.0)[0])
    assert img.shape == (100, 100)


# ===========================================================================
# Dose accounting
# ===========================================================================


def test_dose_accounting_is_exact():
    """Cumulative dose equals sum(electrons_per_pixel * pixels)."""
    sem = VirtualSEM(seed=1, family="dram", site_size_nm=1200.0, dose_per_us=50.0)
    _, m1 = sem.acquire((0.0, 0.0), fov_nm=100.0, dwell_us=2.0)
    _, m2 = sem.acquire((0.0, 0.0), fov_nm=100.0, dwell_us=4.0)
    expected = 50.0 * 2.0 * (100 * 100) + 50.0 * 4.0 * (100 * 100)
    assert m1["dose_this_acq"] == pytest.approx(50.0 * 2.0 * 100 * 100)
    assert sem.total_dose == pytest.approx(expected)
    assert m2["cumulative_dose"] == pytest.approx(expected)
    assert sem.n_acquisitions == 2


# ===========================================================================
# Ground truth is evaluator-only
# ===========================================================================


def test_ground_truth_not_reachable_from_acquire_path():
    """
    The true stage error is returned ONLY by verify(); it must not leak through
    the (png_bytes, meta) an agent receives.
    """
    sem = VirtualSEM(seed=77, family="dram", site_size_nm=1300.0)
    _, meta = sem.acquire((120.0, -80.0), 200.0, 8.0)
    truth = sem.verify()

    # No key names the hidden state.
    forbidden = ("error", "true", "actual", "stage", "bias", "walk", "drift")
    assert not any(any(w in k.lower() for w in forbidden) for k in meta)

    # No meta value equals the hidden error components / true centre.
    true_cx = meta["requested_center_nm"][0] + truth["error_x_nm"]
    true_cy = meta["requested_center_nm"][1] + truth["error_y_nm"]
    leaked = {truth["error_x_nm"], truth["error_y_nm"], truth["error_mag_nm"], true_cx, true_cy}

    def _nums(obj):
        if isinstance(obj, (int, float)):
            yield float(obj)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _nums(v)

    for v in _nums(list(meta.values())):
        assert not any(abs(v - t) < 1e-6 for t in leaked), f"meta leaks ground truth: {v}"


def test_verify_before_acquire_raises():
    sem = VirtualSEM(seed=5, family="dram", site_size_nm=1000.0)
    with pytest.raises(RuntimeError):
        sem.verify()


def test_verify_tracks_last_acquisition_and_moves():
    """verify() reflects the most recent acquisition; a stage move shifts error."""
    sem = VirtualSEM(seed=8, family="dram", site_size_nm=1300.0)
    sem.acquire((0.0, 0.0), 200.0, 8.0)
    before = sem.verify()
    sem.commit_stage_move(500.0, 500.0)  # large move -> larger positioning error
    sem.acquire((0.0, 0.0), 200.0, 8.0)
    after = sem.verify()
    assert before != after, "stage move must change the true error"


# ===========================================================================
# Budget caps
# ===========================================================================


def test_acquisition_cap():
    sem = VirtualSEM(seed=2, family="dram", site_size_nm=1000.0, max_acquisitions=2)
    sem.acquire((0.0, 0.0), 120.0, 4.0)
    sem.acquire((0.0, 0.0), 120.0, 4.0)
    with pytest.raises(AcquisitionLimitError):
        sem.acquire((0.0, 0.0), 120.0, 4.0)
    assert sem.n_acquisitions == 2, "a rejected acquisition must not advance state"


def test_dose_cap():
    sem = VirtualSEM(seed=2, family="dram", site_size_nm=1000.0, max_dose=1000.0)
    with pytest.raises(DoseError):
        sem.acquire((0.0, 0.0), 120.0, 4.0)  # ~ 50*4*120*120 >> 1000
    assert sem.total_dose == 0.0, "a rejected acquisition must not spend dose"


# ===========================================================================
# Validation
# ===========================================================================


def test_invalid_construction_args():
    with pytest.raises(ValueError):
        VirtualSEM(seed=1, family="nanowire", site_size_nm=1000.0)
    with pytest.raises(ValueError):
        VirtualSEM(seed=1, family="dram", site_size_nm=-5.0)


def test_invalid_acquire_args():
    sem = VirtualSEM(seed=1, family="dram", site_size_nm=1000.0)
    with pytest.raises(ValueError):
        sem.acquire((0.0, 0.0), fov_nm=0.0, dwell_us=4.0)
    with pytest.raises(ValueError):
        sem.acquire((0.0, 0.0), fov_nm=100.0, dwell_us=0.0)


def test_png_bytes_are_valid_png():
    sem = VirtualSEM(seed=1, family="dram", site_size_nm=1000.0)
    png, _ = sem.acquire((0.0, 0.0), 150.0, 6.0)
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG stream"


# ===========================================================================
# End-to-end (heavier)
# ===========================================================================


@pytest.mark.slow
def test_long_sequence_is_reproducible_and_accounted():
    """
    A realistic multi-step agent session (acquire / move / acquire ...) on a
    larger site is fully reproducible and its dose accounting stays exact.
    """

    def run() -> tuple[bytes, float, dict]:
        sem = VirtualSEM(seed=2024, family="random", site_size_nm=2600.0, pixel_nm=1.0)
        total = 0.0
        last_png = b""
        for k in range(8):
            last_png, m = sem.acquire((40.0 * k, -25.0 * k), 240.0, 5.0, scan_rotation_deg=0.3 * k)
            total += m["dose_this_acq"]
            sem.commit_stage_move(60.0, -40.0)
        assert sem.total_dose == pytest.approx(total)
        return last_png, sem.total_dose, sem.verify()

    png_a, dose_a, v_a = run()
    png_b, dose_b, v_b = run()
    assert png_a == png_b, "long session not reproducible"
    assert dose_a == pytest.approx(dose_b)
    assert v_a == v_b
