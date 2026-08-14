"""
Baseline localization algorithms for PS2 (Drift-Sense).

Three methods implemented:
  1. NCC     — Normalized Cross-Correlation (classic template matching)
  2. PhaseCC — Phase Correlation in the Fourier domain
  3. MultiNCC— Multi-scale (image pyramid) NCC

All methods accept float32 images in [0,1] and return a predicted (x, y)
centre coordinate inside the search image.

References
----------
[B1] Lewis, J.P. (1995). "Fast template matching." Vision Interface 95.
     — NCC algorithm and normalized score definition.

[B2] Kuglin, C.D. & Hines, D.C. (1975). "The phase correlation image alignment
     method." Proc. IEEE Int. Conf. Cybernetics and Society.
     — Phase correlation for shift estimation.

[B3] Lowe, D.G. (2004). "Distinctive image features from scale-invariant
     keypoints." IJCV 60(2).
     — Gaussian pyramid as multi-scale search basis.

[B4] Foroosh, H. et al. (2002). "Extension of phase correlation to subpixel
     registration." IEEE Trans. Image Process. 11(3).
     — Sub-pixel refinement via phase correlation peak fitting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LocalizationResult:
    method: str
    pred_x: float       # predicted centre-x in search image
    pred_y: float       # predicted centre-y in search image
    score: float        # best match score (NCC ∈ [−1,1]; phase ∈ [0,1])
    runtime_ms: float   # wall time in milliseconds
    score_map: np.ndarray | None = None  # full score map (optional)

    def error(self, gt_x: float, gt_y: float) -> float:
        """Euclidean distance to ground truth (px)."""
        return float(np.sqrt((self.pred_x - gt_x)**2 + (self.pred_y - gt_y)**2))


# ---------------------------------------------------------------------------
# 1. Normalized Cross-Correlation  (NCC)  [B1]
# ---------------------------------------------------------------------------

def ncc_localize(ref: np.ndarray, search: np.ndarray,
                 return_score_map: bool = False) -> LocalizationResult:
    """
    Template matching via Normalized Cross-Correlation.

    Uses OpenCV's TM_CCOEFF_NORMED which is equivalent to the zero-mean NCC
    (Lewis 1995 [B1]).  Score ∈ [−1, 1]; higher is better.

    Failure mode (relevant to PS2):
    --------------------------------
    On periodic structures (DRAM, FinFET) the NCC map has many near-equal
    peaks separated by the structural period.  The algorithm has no way to
    distinguish the true peak from alias peaks — producing errors that are
    exact multiples of the structural period.

    Parameters
    ----------
    ref    : float32 (rh, rw) template
    search : float32 (sh, sw) search image  (sh >= rh, sw >= rw)

    Returns
    -------
    LocalizationResult with pred_x, pred_y = centre of best NCC match
    """
    t0 = time.perf_counter()

    # cv2.matchTemplate expects uint8 or float32; use float32 directly
    score_map = cv2.matchTemplate(search.astype(np.float32),
                                   ref.astype(np.float32),
                                   cv2.TM_CCOEFF_NORMED)

    # Best peak
    _, max_val, _, max_loc = cv2.minMaxLoc(score_map)
    top_left_x, top_left_y = max_loc  # (col, row) of top-left of best match

    # Convert to centre coordinate
    pred_x = top_left_x + ref.shape[1] / 2.0
    pred_y = top_left_y + ref.shape[0] / 2.0

    dt = (time.perf_counter() - t0) * 1000

    return LocalizationResult(
        method     = "NCC",
        pred_x     = pred_x,
        pred_y     = pred_y,
        score      = float(max_val),
        runtime_ms = dt,
        score_map  = score_map if return_score_map else None,
    )


# ---------------------------------------------------------------------------
# 2. Phase Correlation  [B2, B4]
# ---------------------------------------------------------------------------

def phase_corr_localize(ref: np.ndarray, search: np.ndarray,
                         return_score_map: bool = False) -> LocalizationResult:
    """
    Localize via phase correlation in the Fourier domain.

    Phase correlation is robust to uniform illumination changes and has
    O(N log N) complexity.  However it assumes the reference is a sub-window
    of the search (no rotation/scale), and the cross-power spectrum peak
    can be aliased when the structure is periodic — same failure mode as NCC
    but in frequency space.

    Algorithm (Kuglin & Hines 1975 [B2])
    --------------------------------------
    1. Zero-pad reference to search size
    2. Compute FFTs of both
    3. Normalised cross-power spectrum  R = F_ref* · F_search / |F_ref* · F_search|
    4. IFFT of R → correlation surface with impulse at true displacement
    5. Sub-pixel peak fit (Foroosh et al. 2002 [B4])
    """
    t0 = time.perf_counter()

    sh, sw = search.shape[:2]
    rh, rw = ref.shape[:2]

    # Zero-pad reference to search size
    ref_padded = np.zeros((sh, sw), dtype=np.float32)
    ref_padded[:rh, :rw] = ref

    # Apply Hanning window to reduce spectral leakage
    win_h = np.hanning(sh).reshape(-1, 1)
    win_w = np.hanning(sw).reshape(1, -1)
    window = (win_h * win_w).astype(np.float32)
    ref_w   = ref_padded * window
    search_w = search.astype(np.float32) * window

    # FFTs
    F_ref    = np.fft.fft2(ref_w)
    F_search = np.fft.fft2(search_w)

    # Normalised cross-power spectrum
    R = F_ref.conj() * F_search
    denom = np.abs(R) + 1e-8
    R_norm = R / denom

    # Correlation surface
    corr = np.fft.ifft2(R_norm).real
    corr_shifted = np.fft.fftshift(corr)   # shift so zero-shift is at centre

    # Find peak in the correlation surface
    peak_idx = np.argmax(corr_shifted)
    peak_y, peak_x = np.unravel_index(peak_idx, corr_shifted.shape)

    # The shift is relative to centre of the (shifted) correlation map
    shift_y = peak_y - sh // 2
    shift_x = peak_x - sw // 2

    # Sub-pixel refinement via 3×3 neighbourhood parabolic fit [B4]
    shift_x, shift_y = _subpixel_parabolic(corr_shifted, peak_x, peak_y,
                                             sw, sh)

    # The reference centre is at (rw/2, rh/2) in the padded frame.
    # A displacement (shift_x, shift_y) in the correlation space means the
    # template matches at search position:
    pred_x = rw / 2.0 + shift_x
    pred_y = rh / 2.0 + shift_y

    # Clamp to valid range
    pred_x = float(np.clip(pred_x, 0, sw))
    pred_y = float(np.clip(pred_y, 0, sh))

    peak_score = float(corr_shifted[int(round(peak_y)), int(round(peak_x))])

    dt = (time.perf_counter() - t0) * 1000

    return LocalizationResult(
        method     = "PhaseCC",
        pred_x     = pred_x,
        pred_y     = pred_y,
        score      = peak_score,
        runtime_ms = dt,
        score_map  = corr_shifted if return_score_map else None,
    )


def _subpixel_parabolic(surface: np.ndarray,
                          px: int, py: int,
                          sw: int, sh: int) -> Tuple[float, float]:
    """
    3-point parabolic interpolation around peak for sub-pixel shift estimation.
    Falls back to integer peak if boundary.
    """
    shift_x_int = px - sw // 2
    shift_y_int = py - sh // 2

    if px <= 0 or px >= sw - 1 or py <= 0 or py >= sh - 1:
        return float(shift_x_int), float(shift_y_int)

    # X direction
    f_l, f_c, f_r = surface[py, px-1], surface[py, px], surface[py, px+1]
    denom_x = f_l - 2*f_c + f_r
    sub_x = 0.5 * (f_l - f_r) / denom_x if abs(denom_x) > 1e-10 else 0.0

    # Y direction
    f_u, f_c2, f_d = surface[py-1, px], surface[py, px], surface[py+1, px]
    denom_y = f_u - 2*f_c2 + f_d
    sub_y = 0.5 * (f_u - f_d) / denom_y if abs(denom_y) > 1e-10 else 0.0

    return float(shift_x_int + sub_x), float(shift_y_int + sub_y)


# ---------------------------------------------------------------------------
# 3. Multi-scale pyramid NCC  [B3]
# ---------------------------------------------------------------------------

def multiscale_ncc_localize(ref: np.ndarray, search: np.ndarray,
                              n_levels: int = 3,
                              top_k: int = 5,
                              return_score_map: bool = False) -> LocalizationResult:
    """
    Multi-scale template matching using a Gaussian pyramid.

    Strategy:
      1. Build n_levels Gaussian pyramids for both ref and search.
      2. NCC at the coarsest level → top-K candidate locations.
      3. Refine each candidate at successively finer levels (beam search).
      4. Return best candidate at full resolution.

    This reduces compute and — importantly — can escape some local periodicity
    traps by first seeing the global structure at low resolution.

    Parameters
    ----------
    ref      : float32 template
    search   : float32 search image
    n_levels : pyramid depth (3 = 8× downscale at coarsest)
    top_k    : candidate count to propagate between levels
    """
    t0 = time.perf_counter()

    # Build pyramids
    ref_pyr    = _build_pyramid(ref, n_levels)
    search_pyr = _build_pyramid(search, n_levels)

    # Coarsest level NCC — full search
    coarse_ref    = ref_pyr[-1]
    coarse_search = search_pyr[-1]

    score_map_coarse = cv2.matchTemplate(
        coarse_search.astype(np.float32),
        coarse_ref.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )

    # Extract top-K peaks from coarse score map
    candidates = _top_k_peaks(score_map_coarse, top_k, min_distance=5)
    scale = 2 ** (n_levels - 1)   # coarse → full scale factor

    # Refine through pyramid levels (coarse → fine)
    for level in range(n_levels - 2, -1, -1):
        level_scale = 2 ** level
        ref_l    = ref_pyr[level]
        search_l = search_pyr[level]
        rh_l, rw_l = ref_l.shape[:2]
        sh_l, sw_l = search_l.shape[:2]

        new_candidates = []
        for (cx, cy, score) in candidates:
            # Up-scale candidate to this level
            x_l = int(round(cx * 2))
            y_l = int(round(cy * 2))

            # Local search window around candidate (±8 px at this level)
            win = 8
            x0 = max(0, x_l - win)
            y0 = max(0, y_l - win)
            x1 = min(sw_l - rw_l, x_l + win)
            y1 = min(sh_l - rh_l, y_l + win)
            if x1 <= x0 or y1 <= y0:
                new_candidates.append((x_l, y_l, score))
                continue

            local_search = search_l[y0:y0 + (y1 - y0) + rh_l,
                                     x0:x0 + (x1 - x0) + rw_l]
            if local_search.shape[0] < rh_l or local_search.shape[1] < rw_l:
                new_candidates.append((x_l, y_l, score))
                continue

            local_map = cv2.matchTemplate(
                local_search.astype(np.float32),
                ref_l.astype(np.float32),
                cv2.TM_CCOEFF_NORMED,
            )
            _, lmax, _, lmax_loc = cv2.minMaxLoc(local_map)
            new_candidates.append((x0 + lmax_loc[0], y0 + lmax_loc[1], lmax))

        # Keep top-K at this level
        new_candidates.sort(key=lambda c: c[2], reverse=True)
        candidates = new_candidates[:top_k]

    # Best candidate at full resolution
    best_x, best_y, best_score = candidates[0]
    pred_x = best_x + ref.shape[1] / 2.0
    pred_y = best_y + ref.shape[0] / 2.0

    dt = (time.perf_counter() - t0) * 1000

    return LocalizationResult(
        method     = "MultiNCC",
        pred_x     = pred_x,
        pred_y     = pred_y,
        score      = float(best_score),
        runtime_ms = dt,
        score_map  = score_map_coarse if return_score_map else None,
    )


def _build_pyramid(image: np.ndarray, n_levels: int):
    """Return list of images from finest (index 0) to coarsest (index -1)."""
    pyr = [image.astype(np.float32)]
    for _ in range(n_levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _top_k_peaks(score_map: np.ndarray, k: int,
                  min_distance: int = 5) -> list:
    """
    Extract top-K local maxima from a score map with minimum separation.
    Returns list of (x, y, score) tuples (top-left corner of template).
    """
    sm = score_map.copy()
    peaks = []
    for _ in range(k):
        _, val, _, loc = cv2.minMaxLoc(sm)
        if val < -1.0:
            break
        peaks.append((loc[0], loc[1], float(val)))
        # Suppress neighbourhood
        y0 = max(0, loc[1] - min_distance)
        y1 = min(sm.shape[0], loc[1] + min_distance + 1)
        x0 = max(0, loc[0] - min_distance)
        x1 = min(sm.shape[1], loc[0] + min_distance + 1)
        sm[y0:y1, x0:x1] = -999.0
    return peaks


# ---------------------------------------------------------------------------
# Unified runner
# ---------------------------------------------------------------------------

def localize_all(ref: np.ndarray,
                  search: np.ndarray,
                  return_score_maps: bool = False,
                  ) -> dict[str, LocalizationResult]:
    """Run all three baseline methods and return a dict of results."""
    return {
        "NCC":      ncc_localize(ref, search, return_score_maps),
        "PhaseCC":  phase_corr_localize(ref, search, return_score_maps),
        "MultiNCC": multiscale_ncc_localize(ref, search,
                                             return_score_map=return_score_maps),
    }
