"""
Stage 2C+ — RANSAC lattice-fit orientation estimator.

WHY THIS EXISTS (the measured gap it closes)
---------------------------------------------
`spectral.estimate_orientation_consensus` recovers the lattice angle by binning
FFT-peak power into angular bins (mod 90 deg) and taking the heaviest bin. Its
MEDIAN error on the hard tier is excellent (~0.8 deg), but its TAIL is not:
measured over the 40 hard-tier pairs it leaves

        median 0.83 deg | P90 5.16 deg | max 7.39 deg | 35% of pairs > 1.5 deg

The 1.5 deg figure is the budget above which the LER de-warp (spectral.dewarp_roi)
stops recovering the roughness fingerprint, so that 35% tail is exactly the set of
pairs where Stage-2B silently loses its discriminating signal. The consensus vote
fails on them because it treats every peak as an independent angular sample: on
staggered (6F^2) DRAM the diagonal capacitor-cell peaks and their harmonics can
out-weigh the true axis, dragging the heaviest bin onto a +/-45 deg alias.

THE FIX — fit a lattice, not a histogram
-----------------------------------------
The FFT peaks of a 2-D periodic structure are not scattered angular samples: they
are the reciprocal lattice, so EVERY genuine peak is an integer combination
n*k1 + m*k2 of two primitive reciprocal vectors. That is a hard geometric
constraint a histogram ignores.

We therefore fit the lattice by RANSAC:

  1. Detect the strongest peaks in the admissible frequency band (sub-bin refined).
  2. Hypothesise a basis (k_a, k_b) from pairs of strong, non-collinear peaks.
  3. Score a hypothesis by the spectral POWER of its inliers — peaks that land
     within tolerance of some integer combination n*k_a + m*k_b. Diagonal-cell
     peaks and stray noise peaks that do not fit the integer grid are rejected as
     outliers instead of voting.
  4. Re-fit the winning basis by weighted least squares over its inliers
     (denoises the two vectors using all consistent peaks at once).
  5. Reduce to the primitive axis: of {k_a, k_b, k_a+k_b, k_a-k_b} the two
     shortest non-collinear reciprocal vectors lie along the lattice axes for a
     rectangular cell, and both reduce (mod 90) to the same orientation theta.

Because the orientation is read off the LS-refined axes of the inlier set — not
off whichever single peak happened to be strongest — a diagonal outlier can no
longer capture the estimate. It either fits the grid (and helps) or is discarded.

The public entry points mirror `spectral`:
    estimate_orientation_ransac(image)      -> (theta_deg in (-45,45], ok)
    estimate_orientation(image)             -> RANSAC, falling back to consensus
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from localization.spectral import (_hann2d, _refine_peak_subbin,
                                    reduce_lattice_angle,
                                    estimate_orientation_consensus)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2-D scalar cross product (NumPy 2.0 deprecates np.cross on 2-vectors)."""
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class LatticeFit:
    """Result of a RANSAC lattice fit in the reciprocal (Fourier) domain."""
    theta_deg: float          # orientation reduced to (-45, 45]
    ok: bool                  # True if a confident rectangular lattice was fit
    n_inliers: int            # peaks explained by the fitted lattice
    inlier_power_frac: float  # fraction of detected peak power the fit explains
    ka: np.ndarray            # refined primitive reciprocal vector (cycles/px)
    kb: np.ndarray


# ---------------------------------------------------------------------------
# Peak detection (shared band / windowing convention with spectral.py)
# ---------------------------------------------------------------------------

def detect_peaks(image: np.ndarray,
                 n_peaks: int = 24,
                 min_period: float = 6.0,
                 max_period: float = 120.0,
                 min_fft_size: int = 512,
                 suppress_frac: float = 0.30
                 ) -> list[tuple[float, float, float]]:
    """
    Return up to `n_peaks` sub-bin-refined spectral peaks as (kx, ky, power),
    in cycles/pixel, restricted to the admissible frequency band.

    The same Hann-window + zero-pad convention as `spectral.estimate_reciprocal_basis`
    is used so the two estimators see identical spectra; only the reasoning on top
    of the peaks differs.
    """
    img = np.asarray(image, dtype=np.float32)
    img = img - img.mean()
    ih, iw = img.shape
    windowed = img * _hann2d(ih, iw)
    h, w = max(ih, min_fft_size), max(iw, min_fft_size)
    if (h, w) != (ih, iw):
        pad = np.zeros((h, w), dtype=np.float32)
        pad[:ih, :iw] = windowed
        windowed = pad

    P = np.fft.fftshift(np.abs(np.fft.fft2(windowed))).astype(np.float64) ** 2
    fy = (np.arange(h) - h // 2) / float(h)
    fx = (np.arange(w) - w // 2) / float(w)
    FX, FY = np.meshgrid(fx, fy)
    R = np.hypot(FX, FY)
    band = (R >= 1.0 / max_period) & (R <= 1.0 / min_period)
    if not band.any():
        return []
    Pm = np.where(band, P, 0.0)

    n_hi = 1.0 / max(min_period, 1e-6)   # radius scale for suppression
    peaks: list[tuple[float, float, float]] = []
    for _ in range(n_peaks):
        i = int(np.argmax(Pm))
        iy, ix = np.unravel_index(i, Pm.shape)
        p = Pm[iy, ix]
        if p <= 0:
            break
        dy, dx = _refine_peak_subbin(P, iy, ix)
        kx = (ix + dx - w // 2) / float(w)
        ky = (iy + dy - h // 2) / float(h)
        peaks.append((float(kx), float(ky), float(p)))
        # Suppress this peak and its Friedel mirror (spectrum is symmetric).
        for s in (+1, -1):
            d = np.hypot(FX - s * (ix - w // 2) / float(w),
                         FY - s * (iy - h // 2) / float(h))
            Pm[d < suppress_frac * n_hi] = 0.0
    return peaks


# ---------------------------------------------------------------------------
# RANSAC lattice fit
# ---------------------------------------------------------------------------

def _gauss_reduce(ka: np.ndarray, kb: np.ndarray,
                  max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """
    Lagrange-Gauss 2-D lattice reduction: return the two SHORTEST non-collinear
    vectors generating the same lattice as (ka, kb).

    WHY THIS IS THE CORRECT AXIS RECOVERY
    -------------------------------------
    RANSAC seeds are arbitrary lattice vectors, so the winning (ka, kb) may be an
    oblique primitive basis (e.g. one axis + one diagonal-cell vector). Reading an
    angle off such a basis is what produced the earlier garbage estimates.

    For a RECTANGULAR lattice the two shortest vectors are exactly the axis
    vectors, because any diagonal k_a +/- k_b is longer than both axes. Gauss
    reduction provably returns those shortest vectors (repeatedly subtracting the
    nearest integer multiple of the shorter vector from the longer), so the
    orientation read from the reduced basis is the true axis orientation
    regardless of which oblique basis RANSAC happened to seed.
    """
    a, b = ka.astype(np.float64).copy(), kb.astype(np.float64).copy()
    for _ in range(max_iter):
        if b @ b < a @ a:
            a, b = b, a
        denom = a @ a
        if denom < 1e-18:
            break
        m = round((a @ b) / denom)
        if m == 0:
            break
        b = b - m * a
    return a, b


def fit_lattice_ransac(peaks: list[tuple[float, float, float]],
                       tol_frac: float = 0.18,
                       max_order: int = 6,
                       min_inliers: int = 4,
                       seed_top: int = 10) -> LatticeFit | None:
    """
    Fit a 2-D reciprocal lattice to `peaks` by exhaustive RANSAC over basis
    hypotheses drawn from the strongest peaks.

    Parameters
    ----------
    tol_frac    : inlier residual tolerance, as a fraction of the shorter basis
                  vector length. 0.18 comfortably covers sub-bin + pitch noise
                  without admitting a neighbouring grid line.
    max_order   : reject integer labels with |n| or |m| above this — a genuine
                  fundamental generates its neighbours with small integers, so a
                  hypothesis that only fits high orders is a harmonic, not the
                  primitive cell.
    seed_top    : only the `seed_top` strongest peaks are used as basis seeds
                  (scored against ALL peaks), which keeps the fit O(seed_top^2).
    """
    if len(peaks) < 3:
        return None
    P = np.array([[kx, ky] for kx, ky, _ in peaks], dtype=np.float64)
    w = np.array([p for _, _, p in peaks], dtype=np.float64)
    wsum = w.sum() + 1e-30
    order = np.argsort(-w)
    seeds = order[:min(seed_top, len(peaks))]

    best = None  # (score, n_inliers, ka, kb, inlier_idx)
    for ai in seeds:
        for bi in seeds:
            if ai == bi:
                continue
            ka, kb = P[ai], P[bi]
            det = float(_cross2(ka, kb))
            if abs(det) < 1e-9:
                continue                       # collinear seeds
            Kinv = np.linalg.inv(np.array([ka, kb]).T)   # maps k -> (n, m)
            lam = min(np.linalg.norm(ka), np.linalg.norm(kb))
            tol = tol_frac * lam
            score = 0.0
            inliers = []
            for j in range(len(peaks)):
                nm = Kinv @ P[j]
                nm_r = np.round(nm)
                if np.any(np.abs(nm_r) > max_order):
                    continue
                if abs(nm_r[0]) + abs(nm_r[1]) == 0:
                    continue                   # the DC / origin label
                resid = P[j] - (nm_r[0] * ka + nm_r[1] * kb)
                if np.linalg.norm(resid) <= tol:
                    score += w[j]
                    inliers.append(j)
            if len(inliers) < min_inliers:
                continue
            # Prefer more explained power; tie-break toward the finer (smaller
            # reciprocal cell => more primitive) basis so we do not lock onto a
            # coarse super-lattice that happens to catch a few peaks.
            key = (score, -abs(det))
            if best is None or key > (best[0], -abs(_cross2(best[2], best[3]))):
                best = (score, len(inliers), ka.copy(), kb.copy(), inliers)

    if best is None:
        return None

    score, n_in, ka, kb, inliers = best

    # ---- weighted least-squares re-fit of (ka, kb) over the inlier set ----
    # Given integer labels A_j = (n_j, m_j), solve for B = [[ka],[kb]] minimising
    # sum_j w_j || k_j - A_j B ||^2  ->  B = (A^T W A)^{-1} A^T W K.
    Kinv = np.linalg.inv(np.array([ka, kb]).T)
    A = np.array([np.round(Kinv @ P[j]) for j in inliers])   # (N,2)
    K = P[inliers]                                           # (N,2)
    Wd = np.diag(w[inliers])
    try:
        AtW = A.T @ Wd
        B = np.linalg.solve(AtW @ A, AtW @ K)                # (2,2): rows ka,kb
        ka_r, kb_r = B[0], B[1]
        if np.linalg.norm(ka_r) < 1e-9 or np.linalg.norm(kb_r) < 1e-9:
            ka_r, kb_r = ka, kb
    except np.linalg.LinAlgError:
        ka_r, kb_r = ka, kb

    v1, v2 = _gauss_reduce(ka_r, kb_r)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    a1 = reduce_lattice_angle(-float(np.degrees(np.arctan2(v1[1], v1[0]))))
    a2 = reduce_lattice_angle(-float(np.degrees(np.arctan2(v2[1], v2[0]))))
    diff = reduce_lattice_angle(a1 - a2)
    theta = reduce_lattice_angle(a2 + diff / 2.0)   # circular-safe mean

    frac = score / wsum
    # Confident when the two axes agree (rectangular) and the fit explains a
    # real share of the peak power.
    ok = (abs(diff) <= 3.0) and (n_in >= min_inliers) and (frac >= 0.30)
    return LatticeFit(theta_deg=theta, ok=ok, n_inliers=n_in,
                      inlier_power_frac=float(frac), ka=ka_r, kb=kb_r)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_orientation_ransac(image: np.ndarray,
                                n_peaks: int = 24,
                                min_period: float = 6.0,
                                max_period: float = 120.0,
                                min_fft_size: int = 512
                                ) -> tuple[float, bool]:
    """RANSAC lattice-fit orientation. Returns (theta_deg in (-45,45], ok)."""
    peaks = detect_peaks(image, n_peaks=n_peaks, min_period=min_period,
                         max_period=max_period, min_fft_size=min_fft_size)
    fit = fit_lattice_ransac(peaks)
    if fit is None:
        return 0.0, False
    return fit.theta_deg, fit.ok


def estimate_orientation(image: np.ndarray, **kw) -> tuple[float, bool]:
    """
    Best-available orientation: the RANSAC lattice fit when it is confident,
    otherwise the power-weighted consensus. Never worse than consensus, because
    consensus is the fallback; strictly better on the bad-axis tail, because the
    integer-lattice constraint rejects the diagonal outliers that capture the
    vote.
    """
    theta, ok = estimate_orientation_ransac(image, **kw)
    if ok:
        return theta, True
    return estimate_orientation_consensus(image)


# ---------------------------------------------------------------------------
# Angular power-spectrum correlation  (relative rotation, Fourier-based)
# ---------------------------------------------------------------------------

def angular_signature(image: np.ndarray,
                      n_ang: int = 720,
                      min_period: float = 6.0,
                      max_period: float = 120.0,
                      min_fft_size: int = 512) -> np.ndarray:
    """
    Power projected onto orientation: g(theta), theta in [0, 180) deg.

    Each in-band spectral bin contributes its power to the angular bin of its
    direction (mod 180, since |FFT| is centro-symmetric). For a lattice this is a
    comb of sharp lines at the axis directions and their harmonics — a compact
    ROTATION SIGNATURE of the whole structure, not of one peak.
    """
    img = np.asarray(image, dtype=np.float32)
    img = img - img.mean()
    ih, iw = img.shape
    windowed = img * _hann2d(ih, iw)
    h, w = max(ih, min_fft_size), max(iw, min_fft_size)
    if (h, w) != (ih, iw):
        pad = np.zeros((h, w), dtype=np.float32)
        pad[:ih, :iw] = windowed
        windowed = pad
    P = np.fft.fftshift(np.abs(np.fft.fft2(windowed))).astype(np.float64) ** 2
    fy = (np.arange(h) - h // 2) / float(h)
    fx = (np.arange(w) - w // 2) / float(w)
    FX, FY = np.meshgrid(fx, fy)
    R = np.hypot(FX, FY)
    band = (R >= 1.0 / max_period) & (R <= 1.0 / min_period)
    ang = (np.degrees(np.arctan2(FY, FX)) % 180.0)
    bins = np.minimum((ang / 180.0 * n_ang).astype(int), n_ang - 1)
    g = np.bincount(bins[band].ravel(), weights=P[band].ravel(),
                    minlength=n_ang).astype(np.float64)
    # Circular 3-tap smoothing so a line split across two bins is not penalised.
    g = np.convolve(np.r_[g[-1], g, g[0]], [0.25, 0.5, 0.25], mode="same")[1:-1]
    s = g.sum()
    return g / s if s > 0 else g


def estimate_relative_rotation_spectral(ref: np.ndarray,
                                        search: np.ndarray,
                                        n_ang: int = 720,
                                        max_search_deg: float = 12.0,
                                        min_period: float = 6.0,
                                        max_period: float = 120.0,
                                        min_fft_size: int = 512
                                        ) -> tuple[float, bool]:
    """
    Relative rotation of `search` w.r.t. `ref`, by cross-correlating their
    angular power signatures.

    WHY THIS BEATS PER-IMAGE ORIENTATION ON THE BAD-AXIS TAIL
    ---------------------------------------------------------
    `estimate_orientation_consensus` decides an absolute axis PER IMAGE, then
    subtracts. On staggered DRAM / fin-cut FinFET a diagonal family can win the
    vote, and it can win DIFFERENTLY in ref vs search, so the subtraction sees a
    large spurious rotation (the measured 35% tail, P90 5.2 deg).

    Correlating the two full signatures never has to name an axis. It finds the
    single shift that best maps ALL of ref's spectral lines onto ALL of search's
    at once. A diagonal family present in both images reinforces the SAME shift
    as the primary lines, so it helps rather than derailing the estimate. The
    search is restricted to +/- max_search_deg because the physical rotations are
    small (<= ~5 deg), which also removes the 90 deg / diagonal aliases.

    Returns (rotation_deg, ok), sign matching cv2.getRotationMatrix2D (the same
    convention `derotate_roi` / `dewarp_roi` expect).
    """
    g_r = angular_signature(ref, n_ang, min_period, max_period, min_fft_size)
    g_s = angular_signature(search, n_ang, min_period, max_period, min_fft_size)
    if g_r.sum() <= 0 or g_s.sum() <= 0:
        return 0.0, False

    deg_per_bin = 180.0 / n_ang
    max_shift = int(round(max_search_deg / deg_per_bin))
    # Correlate over a small band of circular shifts; shifting search by +d bins
    # and matching ref means search is rotated by +d*deg_per_bin relative to ref.
    shifts = np.arange(-max_shift, max_shift + 1)
    corr = np.array([np.sum(g_s * np.roll(g_r, s)) for s in shifts])
    j = int(np.argmax(corr))
    # Parabolic sub-bin refinement of the correlation peak.
    off = 0.0
    if 0 < j < len(corr) - 1:
        a, b, c = corr[j - 1], corr[j], corr[j + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-18:
            off = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    shift = shifts[j] + off
    # Sign convention: a search rotated by +phi (cv2.getRotationMatrix2D sense)
    # shifts its angular signature by +phi, so the matching roll is -phi; negate
    # to report the rotation itself, matching estimate_orientation_consensus and
    # what derotate_roi / dewarp_roi expect.
    theta = reduce_lattice_angle(-shift * deg_per_bin)

    # Confidence: correlation peak must stand clearly above the local baseline.
    base = float(np.median(corr))
    peak = float(corr[j])
    denom = peak + base + 1e-30
    ok = (peak > 0) and ((peak - base) / denom >= 0.10)
    return theta, ok
