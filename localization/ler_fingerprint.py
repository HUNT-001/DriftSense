"""
LER fingerprint extraction — the core of Stage 2B.

MOTIVATION
----------
Stage-1 ablation established that a perfectly periodic scene is
information-theoretically unsolvable, and that enabling wafer physics moves
NCC from 0% to 100%.  The physical quantity responsible is Line-Edge
Roughness (LER): the stochastic, spatially-fixed waviness of each printed
line edge.

The property that makes LER usable is:

        LER_reference  ==  LER_search        (same physical wafer)
        noise_reference !=  noise_search      (independent acquisitions)

So LER is a *positional fingerprint*, while noise is not.

WHY NOT RAW PIXEL CORRELATION
------------------------------
Raw NCC is dominated by the periodic carrier — the bright/dark line pattern
that is IDENTICAL at every alias location.  The LER signal is a sub-pixel
perturbation riding on top of that carrier, contributing a tiny fraction of
the total pixel energy.  Correlating raw pixels therefore buries the
discriminating signal under the ambiguous one.

This module instead *demodulates*: it removes the periodic carrier and keeps
only the deviation from the ideal lattice.  Concretely, for every line it
measures the sub-pixel line-centre displacement as a function of position
along the line:

    line 0:  +0.8  -0.2  +1.1  -0.5  ...
    line 1:  -0.3  +0.7  +0.1  -0.9  ...
    line 2:  +0.4  +0.2  -0.8  +0.6  ...

That displacement field IS the fingerprint.

CRITICAL DESIGN CONSTRAINT — no per-candidate re-alignment
-----------------------------------------------------------
Adjacent lines carry statistically INDEPENDENT roughness.  Therefore
comparing the reference against a candidate one period away compares line i
with line i+1, which decorrelates — that is exactly the discrimination we
want.

It follows that the comparator must NEVER search over line-index shifts to
maximize similarity.  Doing so would re-align a periodic replica back onto
the reference and silently destroy the discriminating signal, producing an
experiment that looks fine and proves nothing.  Line correspondence here is
fixed by patch geometry (index within the patch) and is never re-estimated
per candidate.  See `fingerprint_similarity`.

METROLOGY NOTE
--------------
Real CD-SEM tools extract edge position per scan line by threshold crossing
or gradient centroid on the intensity profile; see Bunday et al., Proc. SPIE
6152 (2006) and Constantoudis et al., J. Micro/Nanolith. MEMS MOEMS 3(3)
(2004).  We use an intensity-weighted centroid, which is the same family of
estimator and is robust at low dose.

Our generator models *line-position* roughness (both edges of a bar move
together).  Real lines exhibit partially-independent left/right edge
roughness; the centroid estimator recovers the common-mode component, which
is the component our generator injects.  Extending to per-edge extraction is
a natural refinement and is noted as future work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ===========================================================================
# Lattice estimation
# ===========================================================================

@dataclass
class Lattice:
    """Estimated periodic carrier along one axis of a patch."""
    pitch: float        # spatial period (px)
    phase: float        # position of a line maximum (px), in [0, pitch)
    axis: int           # 0 -> lines vertical (positions vary along x)
                        # 1 -> lines horizontal (positions vary along y)
    strength: float     # normalized autocorrelation peak, in [0, 1]

    @property
    def valid(self) -> bool:
        return np.isfinite(self.pitch) and self.pitch > 0


def _axis_profile(patch: np.ndarray, axis: int) -> np.ndarray:
    """
    Collapse the patch to a 1-D profile in which the lattice is visible.

    axis=0 (vertical lines): average over rows -> profile varies along x.
    axis=1 (horizontal lines): average over columns -> profile varies along y.
    """
    return patch.mean(axis=0) if axis == 0 else patch.mean(axis=1)


def estimate_pitch(profile: np.ndarray,
                    min_pitch: float = 6.0,
                    max_pitch: float = 90.0) -> tuple[float, float]:
    """
    Estimate the dominant period of a 1-D profile via autocorrelation.

    Autocorrelation is preferred over raw FFT peak-picking because a square-ish
    line profile has strong harmonics; the autocorrelation peak lands on the
    fundamental.

    Returns
    -------
    (pitch_px, strength)  — strength is the normalized autocorrelation peak
    height in [0, 1]; nan pitch if no periodicity is detectable.
    """
    p = np.asarray(profile, dtype=np.float64)
    p = p - p.mean()
    if p.size < 8 or p.std() < 1e-9:
        return float("nan"), 0.0

    ac = np.correlate(p, p, mode="full")[p.size - 1:]
    ac = ac / (ac[0] + 1e-12)

    lo = int(np.floor(min_pitch))
    hi = int(min(np.ceil(max_pitch), ac.size - 2))
    if hi <= lo + 1:
        return float("nan"), 0.0

    seg = ac[lo:hi]
    idx = int(np.argmax(seg))
    strength = float(seg[idx])
    if strength < 0.05:
        return float("nan"), strength

    lag = float(lo + idx)
    # Parabolic refinement for sub-sample pitch
    if 0 < idx < seg.size - 1:
        a, b, c = seg[idx - 1], seg[idx], seg[idx + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            lag += 0.5 * (a - c) / denom
    return lag, strength


def estimate_phase(profile: np.ndarray, pitch: float) -> float:
    """
    Sub-pixel lattice phase via a single-bin DFT at the fundamental frequency.

    For profile(n) ~ A*cos(2*pi*(n - phi)/pitch) + c,

        z = sum_n profile(n) * exp(-2*pi*i*n/pitch)  ~  (A*N/2) * exp(-2*pi*i*phi/pitch)

    so  phi = -angle(z) * pitch / (2*pi).

    This is exact to sub-pixel precision and needs no interpolation, which is
    why we use it rather than argmax of the profile.
    """
    p = np.asarray(profile, dtype=np.float64)
    p = p - p.mean()
    n = np.arange(p.size)
    z = np.sum(p * np.exp(-2j * np.pi * n / pitch))
    phi = -np.angle(z) * pitch / (2 * np.pi)
    return float(phi % pitch)


def estimate_lattice(patch: np.ndarray, axis: int,
                      min_pitch: float = 6.0,
                      max_pitch: float = 90.0) -> Lattice:
    """Estimate pitch and phase of the periodic carrier along one axis."""
    prof = _axis_profile(patch, axis)
    pitch, strength = estimate_pitch(prof, min_pitch, max_pitch)
    if not np.isfinite(pitch):
        return Lattice(float("nan"), 0.0, axis, strength)
    phase = estimate_phase(prof, pitch)
    return Lattice(pitch, phase, axis, strength)


# ===========================================================================
# Displacement field extraction
# ===========================================================================

@dataclass
class AxisFingerprint:
    """Per-line displacement field for one axis."""
    displacements: np.ndarray   # (n_lines, n_samples) float32, detrended
    mask: np.ndarray            # (n_lines, n_samples) bool, valid samples
    line_positions: np.ndarray  # (n_lines,) line centres within the patch
    lattice: Lattice

    @property
    def n_lines(self) -> int:
        return self.displacements.shape[0]

    @property
    def n_valid(self) -> int:
        return int(self.mask.sum())


@dataclass
class LERFingerprint:
    """Fingerprint of a patch: displacement fields along both axes."""
    axes: dict[int, AxisFingerprint]
    patch_shape: tuple[int, int]

    @property
    def n_valid(self) -> int:
        return sum(a.n_valid for a in self.axes.values())

    @property
    def is_empty(self) -> bool:
        return len(self.axes) == 0


def extract_axis_fingerprint(patch: np.ndarray,
                              lattice: Lattice,
                              background_pct: float = 20.0,
                              min_weight_frac: float = 0.02,
                              detrend_order: int = 1
                              ) -> AxisFingerprint | None:
    """
    Measure sub-pixel line-centre displacement for every line along one axis.

    For each line at lattice position xc, and for each sample position along
    the line, we take a window of width `pitch` centred on xc, subtract a
    per-sample background (a low percentile of the window, which removes the
    slowly-varying illumination that charging and vignetting introduce), and
    compute the intensity-weighted centroid.  The deviation of that centroid
    from xc is the line displacement.

    Each line is then DETRENDED with a low-order polynomial fit
    (`detrend_order`, default 1 = linear).  This is not cosmetic — it is what
    makes the fingerprint robust to relative rotation:

      order 0 (mean removal) absorbs lattice-phase error and uniform sub-pixel
        misalignment between the two acquisitions.
      order 1 (linear) additionally absorbs the LINEAR RAMP that relative
        rotation induces along every line.  At 5 degrees over a 100 px patch a
        line drifts laterally by 100*tan(5deg) ~ 8.7 px, which is more than
        15x the ~0.55 px LER amplitude.  Without linear detrending that ramp
        completely swamps the roughness signal, and measured discrimination
        collapses above ~2.5 degrees.

    Roughness itself is barely affected by the fit, because LER has a short
    correlation length (~18 px) relative to the patch extent (~100 px), so it
    has little projection onto a linear basis.

    Returns None if no usable lines are found.
    """
    if not lattice.valid:
        return None

    # Orient so that lines are always vertical: rows index position ALONG the
    # line, columns index position ACROSS it.
    img = patch if lattice.axis == 0 else patch.T
    n_along, n_across = img.shape

    pitch = lattice.pitch
    half = pitch / 2.0

    # Enumerate line centres fully inside the patch
    positions = []
    x = lattice.phase % pitch
    while x < n_across:
        if x - half >= 0.0 and x + half <= n_across - 1:
            positions.append(x)
        x += pitch
    if not positions:
        return None

    n_lines = len(positions)
    D = np.zeros((n_lines, n_along), dtype=np.float32)
    M = np.zeros((n_lines, n_along), dtype=bool)

    for i, xc in enumerate(positions):
        lo = int(np.floor(xc - half))
        hi = int(np.ceil(xc + half)) + 1
        lo = max(lo, 0)
        hi = min(hi, n_across)
        if hi - lo < 3:
            continue

        slab = img[:, lo:hi].astype(np.float64)          # (n_along, w)
        coords = np.arange(lo, hi, dtype=np.float64)     # (w,)

        # Per-sample background removal -> robust to charging / vignetting
        bg = np.percentile(slab, background_pct, axis=1, keepdims=True)
        w = np.clip(slab - bg, 0.0, None)

        wsum = w.sum(axis=1)
        # A line is measurable only if it carries enough signal here
        thresh = min_weight_frac * (hi - lo)
        good = wsum > thresh

        cent = np.where(good,
                        (w * coords[None, :]).sum(axis=1) / np.maximum(wsum, 1e-9),
                        0.0)
        D[i] = (cent - xc).astype(np.float32)
        M[i] = good

    # Detrend each line over its valid samples (see docstring)
    idx = np.arange(n_along, dtype=np.float64)
    min_samples = max(2 * (detrend_order + 1), 8)
    for i in range(n_lines):
        m = M[i]
        if m.sum() < min_samples:
            M[i] = False
            continue
        if detrend_order <= 0:
            D[i, m] -= D[i, m].mean()
        else:
            coef = np.polyfit(idx[m], D[i, m].astype(np.float64),
                              detrend_order)
            D[i, m] -= np.polyval(coef, idx[m]).astype(np.float32)

    if not M.any():
        return None

    return AxisFingerprint(displacements=D, mask=M,
                            line_positions=np.asarray(positions, dtype=np.float64),
                            lattice=lattice)


def extract_fingerprint(patch: np.ndarray,
                         axes: Sequence[int] = (0, 1),
                         min_pitch: float = 6.0,
                         max_pitch: float = 90.0,
                         min_strength: float = 0.05,
                         detrend_order: int = 1) -> LERFingerprint:
    """
    Extract the LER fingerprint of a patch along the requested axes.

    Both axes are used when both carry a detectable lattice (DRAM grids), which
    roughly doubles the available signal.  FinFET patches typically yield a
    strong fin axis and a weaker gate axis; both are kept when detectable.
    """
    patch = np.asarray(patch, dtype=np.float32)
    out: dict[int, AxisFingerprint] = {}
    for ax in axes:
        lat = estimate_lattice(patch, ax, min_pitch, max_pitch)
        if not lat.valid or lat.strength < min_strength:
            continue
        af = extract_axis_fingerprint(patch, lat,
                                       detrend_order=detrend_order)
        if af is not None:
            out[ax] = af
    return LERFingerprint(axes=out, patch_shape=patch.shape)


# ===========================================================================
# Comparison
# ===========================================================================

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two flat vectors."""
    if a.size < 8:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def compare_axis(f1: AxisFingerprint,
                  f2: AxisFingerprint) -> tuple[float, int]:
    """
    Correlate two per-line displacement fields.

    Line correspondence is by INDEX WITHIN THE PATCH, fixed by geometry.  No
    search over line-index offsets is performed — see the module docstring for
    why that would invalidate the whole method.

    Returns
    -------
    (correlation, n_samples_used)
    """
    n_lines = min(f1.n_lines, f2.n_lines)
    n_samp = min(f1.displacements.shape[1], f2.displacements.shape[1])
    if n_lines < 1 or n_samp < 8:
        return float("nan"), 0

    D1 = f1.displacements[:n_lines, :n_samp]
    D2 = f2.displacements[:n_lines, :n_samp]
    M = f1.mask[:n_lines, :n_samp] & f2.mask[:n_lines, :n_samp]
    if M.sum() < 16:
        return float("nan"), int(M.sum())

    return _corr(D1[M], D2[M]), int(M.sum())


def fingerprint_similarity(f1: LERFingerprint,
                            f2: LERFingerprint) -> float:
    """
    Overall similarity of two fingerprints, in [-1, 1].

    Per-axis correlations are combined as a weighted mean, weighted by the
    number of valid samples contributing to each axis.  Returns nan when
    neither axis is comparable.
    """
    if f1.is_empty or f2.is_empty:
        return float("nan")

    num, den = 0.0, 0.0
    for ax in (0, 1):
        if ax in f1.axes and ax in f2.axes:
            c, n = compare_axis(f1.axes[ax], f2.axes[ax])
            if np.isfinite(c) and n > 0:
                num += c * n
                den += n
    if den <= 0:
        return float("nan")
    return float(num / den)


# ===========================================================================
# Convenience: score a candidate location in a search image
# ===========================================================================

def crop_centered(image: np.ndarray, cx: float, cy: float,
                   w: int, h: int) -> np.ndarray | None:
    """
    Crop a (h, w) window centred at (cx, cy).  Returns None if out of bounds.

    The crop position is rounded to integer pixels.  This is safe here because
    per-line mean removal in the fingerprint absorbs uniform sub-pixel shifts,
    while the discriminating roughness shape is unaffected.
    """
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    if x0 < 0 or y0 < 0 or x0 + w > image.shape[1] or y0 + h > image.shape[0]:
        return None
    return image[y0:y0 + h, x0:x0 + w]


def score_candidate(ref_fp: LERFingerprint,
                     search: np.ndarray,
                     cx: float, cy: float,
                     ref_w: int, ref_h: int) -> float:
    """Fingerprint similarity between a reference and a candidate location."""
    patch = crop_centered(search, cx, cy, ref_w, ref_h)
    if patch is None:
        return float("nan")
    return fingerprint_similarity(ref_fp, extract_fingerprint(patch))
