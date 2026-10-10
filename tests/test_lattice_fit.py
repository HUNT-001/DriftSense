"""
Regression tests for the RANSAC / Fourier lattice-fit orientation tools and the
multi-hypothesis de-warp path added on top of Stage 2.

These test STRUCTURAL correctness (the estimators run, are self-consistent, and
recover a known synthetic rotation), not the end-to-end accuracy numbers — those
live in the benchmark. The orientation ablation itself concluded these estimators
do not beat the existing consensus vote on real hard-tier discrimination, so the
tests here deliberately assert only what is guaranteed.
"""
import numpy as np
import cv2
import pytest

from localization.lattice_fit import (detect_peaks, fit_lattice_ransac,
                                       _gauss_reduce, estimate_orientation_ransac,
                                       angular_signature,
                                       estimate_relative_rotation_spectral)
from localization.spectral import reduce_lattice_angle


def _grating(n=512, px=16, py=24, rot_deg=0.0):
    """A clean rectangular lattice: strong, unambiguous FFT peaks."""
    y, x = np.mgrid[0:n, 0:n]
    img = (0.5 + 0.5 * np.cos(2 * np.pi * x / px)) * \
          (0.5 + 0.5 * np.cos(2 * np.pi * y / py))
    img = img.astype(np.float32)
    if abs(rot_deg) > 1e-6:
        M = cv2.getRotationMatrix2D((n / 2, n / 2), rot_deg, 1.0)
        img = cv2.warpAffine(img, M, (n, n), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REFLECT_101)
    return img


def test_detect_peaks_finds_lattice():
    peaks = detect_peaks(_grating(), n_peaks=12)
    assert len(peaks) >= 2
    # peaks are (kx, ky, power), power strictly positive and sorted descending
    powers = [p for _, _, p in peaks]
    assert all(p > 0 for p in powers)
    assert powers == sorted(powers, reverse=True)


def test_gauss_reduce_returns_short_noncollinear():
    a = np.array([0.06, 0.0])
    b = np.array([0.06, 0.04])          # oblique basis of the same lattice
    v1, v2 = _gauss_reduce(a, b)
    # reduced vectors are no longer than the inputs and stay non-collinear
    assert np.linalg.norm(v1) <= np.linalg.norm(a) + 1e-9
    assert abs(v1[0] * v2[1] - v1[1] * v2[0]) > 1e-6


def test_ransac_fits_clean_lattice():
    fit = fit_lattice_ransac(detect_peaks(_grating(px=16, py=24)))
    assert fit is not None
    assert fit.ok
    assert fit.n_inliers >= 4
    assert 0.0 <= fit.inlier_power_frac <= 1.0001


def test_ransac_recovers_known_rotation():
    # The estimator is designed to DECLINE (ok=False) rather than emit a wrong
    # angle, so we assert two guaranteed properties: (1) when it is confident it
    # is accurate, and (2) it is confident on at least one clean case.
    confident = 0
    for applied in (-3.0, 3.0, 4.0):
        theta, ok = estimate_orientation_ransac(_grating(rot_deg=applied))
        if ok:
            confident += 1
            err = abs(reduce_lattice_angle(theta - applied))
            assert err < 1.5, f"applied {applied}, got {theta}, err {err}"
    assert confident >= 1


def test_angular_signature_normalised():
    g = angular_signature(_grating())
    assert g.shape[0] == 720
    assert g.min() >= 0.0
    assert abs(g.sum() - 1.0) < 1e-6


def test_spectral_relative_rotation_recovers_shift():
    ref = _grating(rot_deg=0.0)
    search = _grating(rot_deg=3.0)
    theta, ok = estimate_relative_rotation_spectral(ref, search)
    assert ok
    assert abs(reduce_lattice_angle(theta - 3.0)) < 1.5


def test_estimators_decline_gracefully_on_noise():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((512, 512)).astype(np.float32)
    # Must not raise, and must not claim confidence on structureless input.
    theta, ok = estimate_orientation_ransac(noise)
    assert np.isfinite(theta)
    assert ok is False
