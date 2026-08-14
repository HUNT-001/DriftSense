"""
Stage 2A — frequency-domain lattice estimation and candidate generation.

THE MEASURED PROBLEM THIS SOLVES
--------------------------------
Stage-2 evaluation decomposed the task:

        FULL SEARCH
             |
      +------+------+
      |             |
   RECALL      DISCRIMINATION
 "is truth      "which candidate
  in top-K?"     is truth?"
      |             |
  22.5% lost    LER works (d' ~ 4-5)

On the hard tier, NCC-peak candidate generation misses the true location
entirely 22.5% of the time.  When that happens, re-ranking cannot recover it —
the loss is structural, not a discrimination failure.

WHY NCC PEAKS MISS
------------------
NCC peaks are chosen by score.  Under rotation, scale error and drift the true
location's score degrades, so it can fall below 150 alias peaks scattered
across the image.  Peak ranking is a *photometric* criterion applied to a
*geometric* problem.

THE FIX
-------
The reference cannot sit at an arbitrary position.  Its content is part of a
periodic lattice, so it can only align with the search image at positions
congruent to the lattice — a structurally constrained set.  Enumerating that
set gives candidates whose recall depends on the accuracy of the lattice
estimate, NOT on whether the true peak happened to outrank its aliases.

METHOD
------
For a 2-D lattice with real-space basis vectors v1, v2, the Fourier transform
has peaks at reciprocal vectors k1, k2 satisfying  k_i . v_j = delta_ij.
Estimating the two dominant non-DC Fourier peaks therefore yields the basis
directly, and — importantly — this is rotation-covariant: a rotated lattice
produces rotated Fourier peaks, so no separate orientation estimator is needed.

Given the reciprocal basis K (rows k1, k2), the lattice phase of an image is

        a_i = k_i . phi = -angle( sum_r I(r) exp(-2*pi*i*k_i.r) ) / (2*pi)

For the reference to align when placed at translation t, we need
phi_search == t + phi_reference, i.e.

        k_i . t  ==  a_search,i - a_reference,i   (mod 1)

whose solutions are

        t = V (delta_a + [n, m]),     V = K^-1,   n, m integer

Evaluating the reference's phase at the SEARCH image's reciprocal vectors
(rather than at its own) makes the relation exact under shared geometry and
keeps small relative rotation implicit in the shared basis.

References
----------
[F1] Kuglin & Hines (1975), "The phase correlation image alignment method,"
     Proc. IEEE Int. Conf. Cybernetics and Society. — Fourier phase encodes
     translation.
[F2] Ashcroft & Mermin, "Solid State Physics" (1976), ch. 5 — reciprocal
     lattice relation k_i . v_j = delta_ij.
[F3] Foroosh et al. (2002), IEEE TIP 11(3) — sub-pixel phase estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cv2 import (getRotationMatrix2D as cv2_getRotationMatrix2D,
                 warpAffine as _cv2_warpAffine,
                 INTER_LINEAR as _INTER_LINEAR,
                 BORDER_REFLECT_101 as _BORDER_REFLECT_101)


from cv2 import INTER_CUBIC as _INTER_CUBIC


def cv2_warpAffine(src, M, dsize):
    return _cv2_warpAffine(src, M, dsize, flags=_INTER_CUBIC,
                            borderMode=_BORDER_REFLECT_101)


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class ReciprocalBasis:
    """Estimated 2-D lattice, expressed in the reciprocal (Fourier) domain."""
    K: np.ndarray          # (2,2) rows are k1, k2 in cycles/pixel
    strength: np.ndarray   # (2,) relative spectral power of each peak
    ok: bool

    @property
    def V(self) -> np.ndarray:
        """Real-space basis; columns are v1, v2 (pixels)."""
        return np.linalg.inv(self.K)

    @property
    def periods(self) -> tuple[float, float]:
        V = self.V
        return (float(np.hypot(*V[:, 0])), float(np.hypot(*V[:, 1])))

    @property
    def orientation_deg(self) -> float:
        """Angle of the first basis vector, in degrees."""
        V = self.V
        return float(np.degrees(np.arctan2(V[1, 0], V[0, 0])))


# ===========================================================================
# Reciprocal basis estimation
# ===========================================================================

def _hann2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)


def _refine_peak_subbin(P: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    """
    Refine an FFT peak to sub-bin precision by parabolic interpolation.

    WHY THIS IS ESSENTIAL, NOT COSMETIC
    -----------------------------------
    A global lattice model only stays in phase across the image if the
    accumulated pitch error is smaller than half a period:

        rel_pitch_err * span  <  pitch / 2

    For a 1000 px image the FFT bin spacing is 1/1000 cycles/px, so at the
    FinFET fin frequency (k = 1/11 = 0.0909) the *quantization alone* is
    0.001 / 0.0909 = 1.1%.  Measured basis error was 0.6% — i.e. bin-limited.

        DRAM   (pitch 24): budget 1.20%, actual 0.9%  -> survives
        FinFET (pitch 11): budget 0.55%, actual 0.6%  -> FAILS

    That single factor explains why lattice candidate generation worked on
    DRAM (100%/92% recall) and collapsed on FinFET (41% on the nominal tier),
    despite the basis ORIENTATION and period being visually correct.

    Parabolic interpolation on log-power recovers roughly an order of
    magnitude of precision, bringing FinFET inside its budget.

    Returns (dy, dx) sub-bin offsets, each clamped to +/-0.5 bins.
    """
    h, w = P.shape
    L = np.log(P + 1e-30)

    def _parab(a: float, b: float, c: float) -> float:
        den = a - 2.0 * b + c
        if abs(den) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))

    dx = _parab(L[iy, ix - 1], L[iy, ix], L[iy, ix + 1]) \
        if 0 < ix < w - 1 else 0.0
    dy = _parab(L[iy - 1, ix], L[iy, ix], L[iy + 1, ix]) \
        if 0 < iy < h - 1 else 0.0
    return dy, dx


def estimate_reciprocal_basis(image: np.ndarray,
                               min_period: float = 6.0,
                               max_period: float = 120.0,
                               suppress_radius_frac: float = 0.35,
                               min_sin_angle: float = 0.25,
                               min_fft_size: int = 512
                               ) -> ReciprocalBasis:
    """
    Estimate the two fundamental reciprocal lattice vectors from a 2-D FFT.

    Robustness measures:
      * A Hann window suppresses spectral leakage from the finite aperture.
      * A DC disk is masked out, since the mean carries no lattice information.
      * After picking k1, its HARMONICS are suppressed as well as its
        neighbourhood.  Without this the second peak is almost always 2*k1
        (line profiles are square-ish and harmonic-rich), which would yield a
        degenerate, collinear basis.
      * k2 is required to be sufficiently non-collinear with k1
        (|sin(angle)| >= min_sin_angle), otherwise the basis is ill-conditioned.

    Returns a basis with ok=False if no usable lattice is found.
    """
    img = np.asarray(image, dtype=np.float32)
    img = img - img.mean()
    ih, iw = img.shape

    # Window first, then ZERO-PAD to at least `min_fft_size`.
    #
    # Angular resolution of an FFT peak scales as ~1/r, where r is the peak
    # radius in bins (r = size / pitch).  A ~100 px reference patch at pitch
    # 24 puts the peak at only ~4 bins, which is far too coarse to localize —
    # this is why reference-side orientation estimates were failing their
    # self-consistency check on 21-24 of 25 pairs while the 1000 px search
    # image estimated fine.
    #
    # Zero-padding interpolates the spectrum (it adds no information, but it
    # removes the peak-localization penalty of coarse sampling), taking the
    # reference peak from ~4 bins to ~21 bins.
    windowed = img * _hann2d(ih, iw)
    h = max(ih, min_fft_size)
    w = max(iw, min_fft_size)
    if (h, w) != (ih, iw):
        padded = np.zeros((h, w), dtype=np.float32)
        padded[:ih, :iw] = windowed
        windowed = padded

    F = np.fft.fftshift(np.abs(np.fft.fft2(windowed)))
    P = F.astype(np.float64) ** 2

    fy = (np.arange(h) - h // 2) / float(h)
    fx = (np.arange(w) - w // 2) / float(w)
    FX, FY = np.meshgrid(fx, fy)
    R = np.hypot(FX, FY)

    # Admissible frequency band
    band = (R >= 1.0 / max_period) & (R <= 1.0 / min_period)
    if not band.any():
        return ReciprocalBasis(np.eye(2), np.zeros(2), False)

    Pm = np.where(band, P, 0.0)
    total = Pm.sum() + 1e-30

    def _peak(arr):
        i = int(np.argmax(arr))
        iy, ix = np.unravel_index(i, arr.shape)
        return iy, ix, float(arr[iy, ix])

    def _freq(iy: int, ix: int) -> np.ndarray:
        """Sub-bin refined frequency vector at integer peak (iy, ix)."""
        dy, dx = _refine_peak_subbin(P, iy, ix)
        return np.array([(ix + dx - w // 2) / float(w),
                         (iy + dy - h // 2) / float(h)], dtype=np.float64)

    # ---- first fundamental ----
    iy1, ix1, p1 = _peak(Pm)
    if p1 <= 0:
        return ReciprocalBasis(np.eye(2), np.zeros(2), False)
    k1 = _freq(iy1, ix1)
    n1 = np.linalg.norm(k1)
    if n1 < 1e-9:
        return ReciprocalBasis(np.eye(2), np.zeros(2), False)

    # ---- suppress k1, its mirror, and its harmonics ----
    supp = suppress_radius_frac * n1
    Pm2 = Pm.copy()
    max_harm = int(np.floor((1.0 / min_period) / n1)) + 1
    for s in (+1, -1):
        for m in range(1, max_harm + 1):
            c = s * m * k1
            d = np.hypot(FX - c[0], FY - c[1])
            Pm2[d < supp] = 0.0

    # ---- second fundamental, constrained to be non-collinear ----
    k2 = None
    p2 = 0.0
    work = Pm2
    for _ in range(12):
        iy2, ix2, p2 = _peak(work)
        if p2 <= 0:
            break
        cand = _freq(iy2, ix2)
        n2 = np.linalg.norm(cand)
        if n2 < 1e-9:
            work = work.copy()
            work[iy2, ix2] = 0.0
            continue
        sin_ang = abs(np.cross(k1, cand)) / (n1 * n2)
        if sin_ang >= min_sin_angle:
            k2 = cand
            break
        d = np.hypot(FX - cand[0], FY - cand[1])
        work = work.copy()
        work[d < supp] = 0.0
        d = np.hypot(FX + cand[0], FY + cand[1])
        work[d < supp] = 0.0

    if k2 is None:
        # Only one lattice direction is detectable (e.g. pure line grating).
        # Complete the basis with the perpendicular direction at the same
        # spatial frequency; candidate generation then samples a square-ish
        # lattice, which still constrains one axis correctly.
        k2 = np.array([-k1[1], k1[0]], dtype=np.float64)
        p2 = 0.0

    K = np.vstack([k1, k2])
    if abs(np.linalg.det(K)) < 1e-12:
        return ReciprocalBasis(np.eye(2), np.zeros(2), False)

    return ReciprocalBasis(K=K,
                            strength=np.array([p1 / total, p2 / total]),
                            ok=True)


# ===========================================================================
# Lattice phase
# ===========================================================================

def dft_at(image: np.ndarray, k: np.ndarray) -> complex:
    """
    Evaluate the 2-D DFT of `image` at an arbitrary (non-grid) frequency k,
    in cycles per pixel.

    Separable evaluation: sum_y sum_x I(y,x) e^{-2*pi*i (kx*x + ky*y)}
                        = (e^{-2*pi*i*ky*y})^T  I  (e^{-2*pi*i*kx*x})

    No windowing is applied here.  A window is useful for PEAK DETECTION but
    would bias PHASE estimation on small patches, and phase is what this
    function is for.
    """
    img = np.asarray(image, dtype=np.float64)
    img = img - img.mean()
    h, w = img.shape
    ex = np.exp(-2j * np.pi * k[0] * np.arange(w))
    ey = np.exp(-2j * np.pi * k[1] * np.arange(h))
    return complex(ey @ img @ ex)


def lattice_phase(image: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Lattice phase offsets a = (a1, a2), each in [0, 1).

    a_i = k_i . phi, recovered from the argument of the DFT at k_i.
    """
    a = np.empty(2, dtype=np.float64)
    for i in range(2):
        z = dft_at(image, K[i])
        a[i] = (-np.angle(z) / (2.0 * np.pi)) % 1.0
    return a


# ===========================================================================
# Candidate generation
# ===========================================================================

def lattice_candidates(search: np.ndarray,
                        ref: np.ndarray,
                        basis: ReciprocalBasis | None = None,
                        max_candidates: int = 4000
                        ) -> tuple[list[tuple[float, float]], ReciprocalBasis]:
    """
    Enumerate all lattice-consistent placements of `ref` inside `search`.

    Returns candidate CENTRES (cx, cy) and the basis used.

    The returned set is the full ambiguity set: every position where the
    periodic structure could plausibly align.  It is deliberately large — its
    job is RECALL.  Stage 2B (LER fingerprint) supplies the precision, and an
    optional NCC pre-rank can trim the set before re-ranking.
    """
    if basis is None:
        basis = estimate_reciprocal_basis(search)
    if not basis.ok:
        return [], basis

    K = basis.K
    V = basis.V

    a_s = lattice_phase(search, K)
    a_r = lattice_phase(ref, K)
    delta = a_s - a_r

    sh, sw = search.shape[:2]
    rh, rw = ref.shape[:2]
    max_x0, max_y0 = sw - rw, sh - rh
    if max_x0 < 0 or max_y0 < 0:
        return [], basis

    # Bound the integer search range from the basis geometry
    v1, v2 = V[:, 0], V[:, 1]
    span = float(np.hypot(sw, sh))
    n1 = int(np.ceil(span / max(np.linalg.norm(v1), 1e-6))) + 2
    n2 = int(np.ceil(span / max(np.linalg.norm(v2), 1e-6))) + 2
    n1 = min(n1, 200)
    n2 = min(n2, 200)

    out: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()

    for n in range(-n1, n1 + 1):
        for m in range(-n2, n2 + 1):
            t = V @ (delta + np.array([n, m], dtype=np.float64))
            x0, y0 = t[0], t[1]
            if not (0.0 <= x0 <= max_x0 and 0.0 <= y0 <= max_y0):
                continue
            key = (int(round(x0)), int(round(y0)))
            if key in seen:
                continue
            seen.add(key)
            out.append((x0 + rw / 2.0, y0 + rh / 2.0))
            if len(out) >= max_candidates:
                return out, basis

    return out, basis


def reduce_lattice_angle(deg: float) -> float:
    """
    Reduce an angle into (-45, 45].

    A rectangular lattice is invariant under 90-degree rotation and under
    basis-vector swap/negation, so lattice orientation is only defined modulo
    90 degrees.  Because the physical rotations we must correct are small
    (<= ~5 degrees), reducing into (-45, 45] resolves the ambiguity uniquely.
    """
    return ((deg + 45.0) % 90.0) - 45.0


def principal_angle(basis: ReciprocalBasis) -> float:
    """
    Orientation of the dominant reciprocal vector, reduced to (-45, 45],
    expressed in the SAME SIGN CONVENTION as cv2.getRotationMatrix2D.

    SIGN CONVENTION (verified experimentally, do not "simplify")
    -----------------------------------------------------------
    cv2.getRotationMatrix2D takes a counter-clockwise angle in a y-UP frame,
    but image rows run y-DOWN, so a positive OpenCV angle rotates image
    content clockwise on screen.  The frequency grid here is built from row/
    column indices and therefore also runs y-down, so arctan2(ky, kx) comes
    out NEGATED relative to the OpenCV angle.

    Controlled check (synthetic DRAM, rotation applied with
    cv2.getRotationMatrix2D, angle recovered from the FFT):

        applied   raw arctan2   ratio
          -4.00          3.99   -1.00
          -2.00          2.00   -1.00
           2.00         -2.00   -1.00
           4.00         -3.99   -1.00

    The leading minus below converts to the OpenCV convention, so the value
    returned can be passed straight to `derotate_roi`.  Before this fix the
    median orientation error was ~2.3 degrees against a 0.79 degree budget —
    the estimator looked fundamentally too coarse when it was simply inverted.
    """
    k = basis.K[0] if basis.strength[0] >= basis.strength[1] else basis.K[1]
    return reduce_lattice_angle(-float(np.degrees(np.arctan2(k[1], k[0]))))


def principal_angle_robust(basis: ReciprocalBasis,
                            max_disagreement_deg: float = 1.5
                            ) -> tuple[float, bool]:
    """
    Orientation estimate with a self-consistency check.

    For a genuinely RECTANGULAR lattice the two reciprocal vectors are
    perpendicular, so after reduction modulo 90 degrees they must report the
    SAME orientation.  Averaging the two halves the noise.

    When they disagree the basis is not rectangular — the dominant peak is a
    diagonal vector, which is exactly what happens on staggered (6F^2) DRAM,
    where the capacitor cells form a honeycomb.  Those cases produced the
    large outliers that kept P95 orientation error at ~5 degrees even after
    the sign fix was applied (median was already 0.04 degrees).

    Rather than emit a confidently wrong angle, this returns ok=False so the
    caller can decline to de-rotate.  Declining is safe: it leaves the pair
    exactly as it is today rather than actively corrupting it.

    Returns
    -------
    (angle_deg, ok)
    """
    a1 = reduce_lattice_angle(
        -float(np.degrees(np.arctan2(basis.K[0][1], basis.K[0][0]))))
    a2 = reduce_lattice_angle(
        -float(np.degrees(np.arctan2(basis.K[1][1], basis.K[1][0]))))

    diff = reduce_lattice_angle(a1 - a2)
    if abs(diff) > max_disagreement_deg:
        # Not a rectangular lattice, or one peak is spurious.
        return principal_angle(basis), False

    # Circular-safe mean within the reduced window
    return reduce_lattice_angle(a2 + diff / 2.0), True


def estimate_orientation_consensus(image: np.ndarray,
                                    n_peaks: int = 16,
                                    min_period: float = 6.0,
                                    max_period: float = 120.0,
                                    min_fft_size: int = 512,
                                    bin_deg: float = 0.5
                                    ) -> tuple[float, bool]:
    """
    Robust lattice orientation by POWER-WEIGHTED angular consensus.

    WHY THE SINGLE DOMINANT VECTOR FAILS
    ------------------------------------
    `principal_angle` takes the single strongest FFT peak.  Its median error is
    excellent (0.03 deg) but ~30% of pairs have a bad-axis outlier (P90 5.8
    deg): on staggered (6F^2) DRAM the diagonal capacitor-cell peaks can be
    stronger than the line-grid peaks, and on FinFET the fin axis dominates the
    gate axis.  One wrong angle corrupts the entire de-warp for that pair,
    which end to end dragged estimated rot+scale (27%) below no correction
    (33%).

    THE CONSENSUS (why this is robust where the median was not)
    -----------------------------------------------------------
    For a rectangular lattice rotated by theta, BOTH line families and ALL
    their harmonics sit at reduced angle theta (mod 90).  Only the diagonal
    combination peaks sit at theta +/- 45.  So the line structure contributes
    MANY peaks carrying most of the spectral POWER at reduced angle theta,
    while the diagonal peaks are few and weaker.

    Summing peak power into angular bins (mod 90) and taking the heaviest bin
    therefore recovers theta even when the single strongest peak is a diagonal
    outlier — the collective weight of the true axis wins.  A plain median
    failed earlier because it counts peaks equally; this counts them by power.

    Returns
    -------
    (theta_deg in (-45, 45], ok)
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
        return 0.0, False
    Pm = np.where(band, P, 0.0)

    n1 = 1.0 / max(min_period, 1e-6)
    peaks = []   # (reduced_angle, power)
    for _ in range(n_peaks):
        i = int(np.argmax(Pm))
        iy, ix = np.unravel_index(i, Pm.shape)
        p = Pm[iy, ix]
        if p <= 0:
            break
        dy, dx = _refine_peak_subbin(P, iy, ix)
        kx = (ix + dx - w // 2) / float(w)
        ky = (iy + dy - h // 2) / float(h)
        ang = reduce_lattice_angle(-np.degrees(np.arctan2(ky, kx)))
        peaks.append((ang, float(p)))
        for s in (+1, -1):
            d = np.hypot(FX - s * (ix - w // 2) / float(w),
                         FY - s * (iy - h // 2) / float(h))
            Pm[d < 0.30 * n1] = 0.0

    if len(peaks) < 2:
        return (peaks[0][0] if peaks else 0.0), False

    # Power-weighted angular histogram over (-45, 45], wrapped.
    nb = int(round(90.0 / bin_deg))
    hist = np.zeros(nb, dtype=np.float64)
    for ang, p in peaks:
        b = int(((ang + 45.0) % 90.0) / bin_deg) % nb
        hist[b] += p
    # Smooth circularly so a peak split across two bins is not penalized.
    k = np.array([0.25, 0.5, 0.25])
    hist = np.convolve(np.r_[hist[-1], hist, hist[0]], k, mode="same")[1:-1]

    best = int(np.argmax(hist))
    # Sub-bin refine the histogram peak (parabolic, circular).
    l, r = hist[(best - 1) % nb], hist[(best + 1) % nb]
    den = l - 2 * hist[best] + r
    off = 0.5 * (l - r) / den if abs(den) > 1e-12 else 0.0
    theta = reduce_lattice_angle((best + off) * bin_deg - 45.0)

    # Confidence: fraction of total peak power within +/-2 deg of theta.
    tot = sum(p for _, p in peaks)
    near = sum(p for ang, p in peaks
               if abs(reduce_lattice_angle(ang - theta)) < 2.0)
    ok = tot > 0 and (near / tot) >= 0.5
    return theta, ok


def estimate_relative_rotation(ref: np.ndarray,
                                search: np.ndarray,
                                search_basis: ReciprocalBasis | None = None
                                ) -> tuple[float, bool]:
    """
    Estimate the rotation of `search` relative to `ref`, in degrees.

    DESIGN CONSTRAINT — why this is estimated ONCE, globally
    --------------------------------------------------------
    Relative rotation is a property of the ACQUISITION PAIR, not of any
    individual candidate.  Estimating it once and applying the identical
    correction to every candidate makes leakage structurally impossible: a
    uniform transform cannot preferentially favour one periodic replica over
    another.

    Fitting rotation per-candidate (especially by maximizing a fingerprint
    score) would risk the same class of bug already caught in
    `ler_fingerprint.compare_axis`, where re-alignment silently re-aligns a
    replica onto the reference and destroys the discriminating signal while
    still looking healthy.

    ACCURACY LIMIT
    --------------
    Angular resolution of an FFT peak is roughly 1/r radians, where r is the
    peak radius in bins (r = patch_size / pitch).  For a 100 px reference at
    pitch 24 that is only ~4 bins, i.e. ~14 degrees before sub-bin refinement —
    far coarser than the ~0.8 degree budget that LER extraction requires.
    The search image (1000 px, r ~ 42 bins) is an order of magnitude better.

    Returns
    -------
    (rotation_deg, ok)
    """
    b_r = estimate_reciprocal_basis(ref)
    b_s = search_basis if search_basis is not None \
        else estimate_reciprocal_basis(search)
    if not (b_r.ok and b_s.ok):
        return 0.0, False

    a_s, ok_s = principal_angle_robust(b_s)
    a_r, ok_r = principal_angle_robust(b_r)
    # Decline rather than emit a confidently wrong angle (see
    # principal_angle_robust); declining leaves the pair uncorrected, which is
    # strictly safer than corrupting it.
    return reduce_lattice_angle(a_s - a_r), (ok_s and ok_r)


def derotate_roi(search: np.ndarray,
                  cx: float, cy: float,
                  rw: int, rh: int,
                  theta_deg: float) -> np.ndarray | None:
    """
    Extract an ROI centred at (cx, cy) with `theta_deg` of rotation removed.

    A window LARGER than the target is cropped first, rotated about its
    centre, then centre-cropped to (rh, rw).  Working from the search image
    (which has real surrounding context) avoids the border-extrapolation
    artifacts that rotating the isolated reference patch would introduce.
    """
    if abs(theta_deg) < 1e-6:
        x0 = int(round(cx - rw / 2.0))
        y0 = int(round(cy - rh / 2.0))
        if x0 < 0 or y0 < 0 or x0 + rw > search.shape[1] or y0 + rh > search.shape[0]:
            return None
        return search[y0:y0 + rh, x0:x0 + rw]

    # Enough margin that rotation cannot pull in undefined pixels
    pad = int(np.ceil(0.5 * np.hypot(rw, rh) - min(rw, rh) / 2.0)) + 4
    bw, bh = rw + 2 * pad, rh + 2 * pad
    x0 = int(round(cx - bw / 2.0))
    y0 = int(round(cy - bh / 2.0))
    if x0 < 0 or y0 < 0 or x0 + bw > search.shape[1] or y0 + bh > search.shape[0]:
        return None

    big = search[y0:y0 + bh, x0:x0 + bw]
    M = cv2_getRotationMatrix2D((bw / 2.0, bh / 2.0), theta_deg, 1.0)
    # CUBIC interpolation is essential here.  Measured: de-rotating a 5-degree
    # candidate recovers the LER descriptor similarity from 0.37 to 0.67 with
    # cubic, but resampling with bilinear (the earlier default) blurs the
    # sub-pixel roughness and gives back much less.  Rotation resampling is
    # safe for LER precisely because it displaces the smooth envelope while
    # cubic reconstruction preserves the fine lateral detail; SCALE resampling
    # is a separate question handled elsewhere.
    rot = cv2_warpAffine(big, M, (bw, bh))
    ox, oy = (bw - rw) // 2, (bh - rh) // 2
    return rot[oy:oy + rh, ox:ox + rw]


def dewarp_roi(search: np.ndarray,
               cx: float, cy: float,
               rw: int, rh: int,
               theta_deg: float,
               scale: float) -> np.ndarray | None:
    """
    Extract an ROI with BOTH rotation (`theta_deg`) and scale removed.

    Generalizes `derotate_roi` to also undo a scale factor (candidate content
    is `scale` times the reference geometry, so we resize by 1/scale).  Uses
    cubic interpolation, which preserves sub-pixel LER far better than bilinear
    (see `derotate_roi`).  Scale correction is only worthwhile in combination
    with rotation: the two distortions compound and neither alone clears the
    success threshold (README, Stage 2E).
    """
    if abs(theta_deg) < 1e-6 and abs(scale - 1.0) < 1e-3:
        x0 = int(round(cx - rw / 2.0))
        y0 = int(round(cy - rh / 2.0))
        if x0 < 0 or y0 < 0 or x0 + rw > search.shape[1] or y0 + rh > search.shape[0]:
            return None
        return search[y0:y0 + rh, x0:x0 + rw]

    pad = int(np.ceil(0.5 * np.hypot(rw, rh) * max(scale, 1.0))) + 8
    bw, bh = rw + 2 * pad, rh + 2 * pad
    x0 = int(round(cx - bw / 2.0))
    y0 = int(round(cy - bh / 2.0))
    if x0 < 0 or y0 < 0 or x0 + bw > search.shape[1] or y0 + bh > search.shape[0]:
        return None
    big = search[y0:y0 + bh, x0:x0 + bw]
    M = cv2_getRotationMatrix2D((bw / 2.0, bh / 2.0), theta_deg, 1.0 / scale)
    rot = cv2_warpAffine(big, M, (bw, bh))
    ox, oy = (bw - rw) // 2, (bh - rh) // 2
    return rot[oy:oy + rh, ox:ox + rw]


def snap_to_local_ncc(candidates: list[tuple[float, float]],
                       score_map: np.ndarray,
                       ref_shape: tuple[int, int],
                       radius: int = 5) -> list[tuple[float, float]]:
    """
    Snap each lattice candidate to the local NCC maximum within `radius` px.

    WHY THIS IS NECESSARY
    ---------------------
    A global lattice model assumes a perfectly constant pitch.  Real pitch
    estimates carry a small relative error, and that error ACCUMULATES with
    distance from the phase origin: across 1000 px at pitch ~24 there are ~42
    periods, so a 0.5% pitch error becomes a ~5 px positional drift at the far
    edge.  Scale error (+/-7% on the hard tier) and drift shear add to this.

    Measured on the hard tier before this step: the nearest lattice candidate
    sat a median of 2.6 px from ground truth (p75 = 4.9, max = 7.8), which
    straddles the 5 px success tolerance — the lattice was identifying the
    correct CELL but not the exact offset within it.

    Snapping keeps the structural constraint (which cell) while letting a
    cheap photometric criterion fix the sub-cell offset.  The search radius is
    deliberately much smaller than the pitch, so snapping cannot hop to a
    neighbouring alias.
    """
    rh, rw = ref_shape
    H, W = score_map.shape
    out: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()

    for (cx, cy) in candidates:
        x = int(round(cx - rw / 2.0))
        y = int(round(cy - rh / 2.0))
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        win = score_map[y0:y1, x0:x1]
        iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
        bx, by = x0 + ix, y0 + iy
        key = (bx, by)
        if key in seen:
            continue
        seen.add(key)
        out.append((bx + rw / 2.0, by + rh / 2.0))
    return out


def rank_candidates_by_ncc(candidates: list[tuple[float, float]],
                            score_map: np.ndarray,
                            ref_shape: tuple[int, int],
                            top_k: int) -> list[tuple[float, float]]:
    """
    Trim a lattice candidate set to the top-K by NCC score.

    This is a cheap photometric pre-filter on a geometrically-constrained set —
    the opposite order from pure NCC peak-picking, and it keeps the structural
    guarantee while cutting the number of fingerprint extractions.
    """
    rh, rw = ref_shape
    scored = []
    H, W = score_map.shape
    for (cx, cy) in candidates:
        x = int(round(cx - rw / 2.0))
        y = int(round(cy - rh / 2.0))
        if 0 <= x < W and 0 <= y < H:
            scored.append((float(score_map[y, x]), cx, cy))
    scored.sort(key=lambda s: -s[0])
    return [(cx, cy) for (_, cx, cy) in scored[:top_k]]
