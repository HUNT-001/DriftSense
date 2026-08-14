"""
SEM (Scanning Electron Microscope) physical imaging effects — v2 (hardened).

All effects operate on normalized float32 images in [0, 1].

CHANGES FROM v1
---------------
  * Drift distortion vectorized with cv2.remap (was one warpAffine per scan
    line — O(H) OpenCV calls, dominating generation time).
  * Added detector read noise, scan-line gain jitter, specimen charging,
    vignetting, and brightness/contrast/gamma variation between acquisitions.
  * Poisson sampling made numerically safe for zero-intensity pixels.
  * All effects now driven by an explicit, serializable AcquisitionParams
    record so every image is exactly reproducible from its manifest entry.

Physical justifications / citations
------------------------------------
[E1] Edge brightening (secondary-electron yield peak):
     Goldstein et al., "Scanning Electron Microscopy and X-Ray Microanalysis,"
     4th ed., Springer 2018, section 2.4.  SE yield rises with local surface
     tilt (the "edge effect"), producing bright edges. Modelled as weighted
     gradient magnitude added to intensity, as in Sato et al., "SEM image
     simulator for CD metrology," Proc. SPIE 8681, 2013.

[E2] Shot noise (Poisson):
     SEM imaging is an electron-counting process; for mean N counts the
     variance is N.  Postek & Vladar, "Does your SEM really tell the truth?",
     Scanning 33, 2011.  Typical IC-inspection dose 25-700 e-/px.

[E3] Gaussian blur (beam spot / defocus):
     Finite probe size convolves the specimen with an approximately Gaussian
     PSF.  Bunday et al., Proc. SPIE 6152, 2006.

[E4] Scan drift distortion:
     Thermal/mechanical stage drift during raster acquisition produces a
     cumulative displacement along the slow-scan axis (shear-like warp).
     Vladar & Postek, Microscopy Today 13(4), 2005.

[E5] Independent noise realizations:
     Reference and search images come from separate acquisitions, so their
     stochastic components must be statistically independent.  Enforced by
     giving each acquisition its own Generator.

[E6] Detector / video-amplifier read noise:
     Additive, dose-independent Gaussian noise from the SE detector chain.
     Reimer, "Scanning Electron Microscopy," 2nd ed., Springer 1998, ch. 4.

[E7] Scan-line gain jitter:
     Line-to-line variation in amplifier gain and beam current produces the
     characteristic horizontal streaking of SEM images.
     Reimer 1998, ch. 4.

[E8] Specimen charging:
     Insulating layers accumulate charge, locally deflecting the beam and
     modulating SE collection -> slow, low-spatial-frequency brightness
     undulation and local distortion.
     Cazaux, "Correlations between ionization radiation damage and charging
     effects in TEM/SEM," Ultramicroscopy 60, 1995.

[E9] Vignetting / collection-efficiency falloff:
     SE collection efficiency depends on position relative to the detector,
     giving a smooth radial/linear intensity gradient across the field.
     Goldstein et al. 2018, ch. 4.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple

import cv2
import numpy as np

from config import SEMConfig


# ===========================================================================
# Acquisition parameter record
# ===========================================================================

@dataclass
class AcquisitionParams:
    """
    Full, serializable description of ONE SEM acquisition.

    Sampling this record and storing it in the manifest makes every generated
    image bit-reproducible and makes ablations trivial.
    """
    blur_sigma: float
    edge_strength: float
    electrons_per_pixel: float
    drift_rate: float
    read_noise_sigma: float
    scanline_jitter_sigma: float
    charging_amplitude: float
    vignette_strength: float
    vignette_cx: float
    vignette_cy: float
    brightness_offset: float
    contrast_gain: float
    gamma: float

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def sample_acquisition(cfg: SEMConfig,
                        noise_level: str,
                        rng: np.random.Generator) -> AcquisitionParams:
    """
    Draw one acquisition's parameters.

    `noise_level` in {'low','medium','high'} selects the electron dose range.
    NOTE the inverse relationship: LOW noise == HIGH dose.
    """
    dose_ranges = {
        "low":    cfg.dose_low,
        "medium": cfg.dose_medium,
        "high":   cfg.dose_high,
    }
    if noise_level not in dose_ranges:
        raise ValueError(f"noise_level must be one of {list(dose_ranges)}")
    d_lo, d_hi = dose_ranges[noise_level]

    def u(rng_range: Tuple[float, float]) -> float:
        return float(rng.uniform(rng_range[0], rng_range[1]))

    return AcquisitionParams(
        blur_sigma            = u(cfg.blur_sigma_range),
        edge_strength         = u(cfg.edge_strength_range),
        electrons_per_pixel   = float(rng.uniform(d_lo, d_hi)),
        drift_rate            = u(cfg.drift_rate_range),
        read_noise_sigma      = u(cfg.read_noise_sigma_range),
        scanline_jitter_sigma = u(cfg.scanline_jitter_sigma_range),
        charging_amplitude    = u(cfg.charging_amplitude_range),
        vignette_strength     = u(cfg.vignette_strength_range),
        vignette_cx           = float(rng.uniform(0.25, 0.75)),
        vignette_cy           = float(rng.uniform(0.25, 0.75)),
        brightness_offset     = u(cfg.brightness_offset_range),
        contrast_gain         = u(cfg.contrast_gain_range),
        gamma                 = u(cfg.gamma_range),
    )


# ===========================================================================
# [E3] Beam blur
# ===========================================================================

def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve with a Gaussian PSF approximating the electron probe [E3]."""
    if sigma <= 1e-3:
        return image.astype(np.float32, copy=True)
    ksize = max(3, int(6 * sigma) | 1)
    return cv2.GaussianBlur(image.astype(np.float32), (ksize, ksize), sigma)


# ===========================================================================
# [E1] Edge brightening
# ===========================================================================

def apply_edge_brightening(image: np.ndarray,
                            strength: float,
                            blur_sigma: float = 0.8) -> np.ndarray:
    """
    Add secondary-electron edge enhancement proportional to |grad I| [E1].

    The gradient map is lightly smoothed so that pixel noise is not amplified,
    and normalized by a robust percentile (not the max) so a single bright
    defect cannot suppress the whole edge signal.
    """
    img = image.astype(np.float32)
    if strength <= 0:
        return img.copy()
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    if blur_sigma > 0:
        k = max(3, int(6 * blur_sigma) | 1)
        edge = cv2.GaussianBlur(edge, (k, k), blur_sigma)
    # Robust normalization (99th percentile) — resistant to outliers
    norm = np.percentile(edge, 99.0)
    if norm > 1e-6:
        edge = np.clip(edge / norm, 0.0, 1.5)
    return np.clip(img + strength * edge, 0.0, 1.0)


# ===========================================================================
# [E4] Scan drift  (vectorized)
# ===========================================================================

def apply_drift_distortion(image: np.ndarray,
                             drift_per_line: float,
                             axis: int = 0) -> np.ndarray:
    """
    Apply cumulative raster drift as a single vectorized remap [E4].

    axis=0 : each row y is displaced horizontally by drift_per_line * y
             (the standard slow-scan drift geometry).

    Uses cv2.remap with BORDER_REPLICATE.  Callers should crop away the
    affected border, or apply drift to an oversized scene before cropping.
    """
    if abs(drift_per_line) < 1e-8:
        return image.astype(np.float32, copy=True)

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    if axis == 0:
        map_x = xx - drift_per_line * yy
        map_y = yy
    else:
        map_x = xx
        map_y = yy - drift_per_line * xx

    return cv2.remap(image.astype(np.float32), map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ===========================================================================
# [E8] Specimen charging
# ===========================================================================

def apply_charging(image: np.ndarray,
                    amplitude: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Low-spatial-frequency multiplicative brightness undulation from charge
    accumulation on insulating layers [E8].
    """
    if amplitude <= 1e-6:
        return image.astype(np.float32, copy=True)
    h, w = image.shape[:2]
    coarse = rng.standard_normal((5, 5)).astype(np.float32)
    field_ = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    std = field_.std()
    if std > 1e-8:
        field_ /= std
    return np.clip(image * (1.0 + amplitude * field_), 0.0, 1.0)


# ===========================================================================
# [E9] Vignetting
# ===========================================================================

def apply_vignetting(image: np.ndarray,
                      strength: float,
                      cx_frac: float = 0.5,
                      cy_frac: float = 0.5) -> np.ndarray:
    """
    Smooth radial falloff modelling SE collection-efficiency variation [E9].
    The detector is off-axis, so the centre is parameterized.
    """
    if strength <= 1e-6:
        return image.astype(np.float32, copy=True)
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = cx_frac * w, cy_frac * h
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r /= (r.max() + 1e-8)
    return np.clip(image * (1.0 - strength * r ** 2), 0.0, 1.0)


# ===========================================================================
# [E7] Scan-line gain jitter
# ===========================================================================

def apply_scanline_jitter(image: np.ndarray,
                           sigma: float,
                           rng: np.random.Generator) -> np.ndarray:
    """Row-wise multiplicative gain variation -> horizontal streaking [E7]."""
    if sigma <= 1e-8:
        return image.astype(np.float32, copy=True)
    h = image.shape[0]
    gains = 1.0 + rng.standard_normal(h).astype(np.float32) * sigma
    return np.clip(image * gains[:, None], 0.0, 1.0)


# ===========================================================================
# [E2] Shot noise  +  [E6] read noise
# ===========================================================================

def apply_shot_noise(image: np.ndarray,
                     electrons_per_pixel: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Poisson electron-counting noise [E2].

    Intensity is interpreted as a normalized count rate; counts are Poisson-
    sampled and renormalized.  A small dark-count floor avoids the degenerate
    zero-mean Poisson at fully black pixels.
    """
    img = np.clip(image, 0.0, 1.0).astype(np.float64)
    dark_floor = 0.5
    counts = img * electrons_per_pixel + dark_floor
    noisy = rng.poisson(counts).astype(np.float64)
    out = (noisy - dark_floor) / electrons_per_pixel
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_read_noise(image: np.ndarray,
                      sigma: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Additive, signal-independent detector/amplifier noise [E6]."""
    if sigma <= 1e-8:
        return image.astype(np.float32, copy=True)
    noise = rng.standard_normal(image.shape).astype(np.float32) * sigma
    return np.clip(image + noise, 0.0, 1.0)


# ===========================================================================
# Photometric response
# ===========================================================================

def apply_photometric(image: np.ndarray,
                       brightness_offset: float,
                       contrast_gain: float,
                       gamma: float) -> np.ndarray:
    """
    Apply brightness/contrast/gamma differences between two acquisitions.

    This is what breaks naive sum-of-squared-difference matching and is the
    reason NORMALIZED correlation is the correct classical baseline.
    """
    img = image.astype(np.float32)
    img = (img - 0.5) * contrast_gain + 0.5 + brightness_offset
    img = np.clip(img, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        img = np.power(img, gamma, dtype=np.float32)
    return np.clip(img, 0.0, 1.0)


# ===========================================================================
# Geometric transforms
# ===========================================================================

def build_affine(center: Tuple[float, float],
                  angle_deg: float,
                  scale: float) -> np.ndarray:
    """
    Return the 2x3 affine matrix for rotation+scale about `center`.

    Exposed separately (rather than hidden inside a warp helper) so the scene
    composer can map ground-truth coordinates through the EXACT same matrix
    used to warp pixels — eliminating any GT/pixel mismatch.
    """
    return cv2.getRotationMatrix2D(center, angle_deg, scale)


def warp_affine(image: np.ndarray, M: np.ndarray,
                 out_size: Tuple[int, int]) -> np.ndarray:
    """Apply a 2x3 affine matrix. out_size is (width, height)."""
    return cv2.warpAffine(image.astype(np.float32), M, out_size,
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0.0)


def transform_point(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Map a single point through a 2x3 affine matrix."""
    px = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    py = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return float(px), float(py)


def invert_affine(M: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix."""
    return cv2.invertAffineTransform(M)


# ===========================================================================
# Full acquisition pipeline
# ===========================================================================

def render_sem_image(clean_image: np.ndarray,
                     params: AcquisitionParams,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Apply the full SEM acquisition chain in physically-correct order.

      1. Beam blur                    [E3]  — optics, before detection
      2. Edge brightening             [E1]  — SE yield, a property of signal
      3. Charging                     [E8]  — specimen-level modulation
      4. Vignetting                   [E9]  — collection efficiency
      5. Drift warp                   [E4]  — raster geometry
      6. Photometric response               — amplifier gain/offset/gamma
      7. Scan-line gain jitter        [E7]  — per-line amplifier variation
      8. Shot noise                   [E2]  — counting statistics (LAST signal-
                                              dependent stochastic step)
      9. Read noise                   [E6]  — additive detector noise

    `rng` MUST be unique per acquisition to guarantee independence [E5].
    """
    img = clean_image.astype(np.float32)
    img = apply_gaussian_blur(img, params.blur_sigma)
    img = apply_edge_brightening(img, params.edge_strength)
    img = apply_charging(img, params.charging_amplitude, rng)
    img = apply_vignetting(img, params.vignette_strength,
                            params.vignette_cx, params.vignette_cy)
    img = apply_drift_distortion(img, params.drift_rate, axis=0)
    img = apply_photometric(img, params.brightness_offset,
                             params.contrast_gain, params.gamma)
    img = apply_scanline_jitter(img, params.scanline_jitter_sigma, rng)
    img = apply_shot_noise(img, params.electrons_per_pixel, rng)
    img = apply_read_noise(img, params.read_noise_sigma, rng)
    return img
