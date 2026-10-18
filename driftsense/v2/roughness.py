"""S3 — roughness residual extraction.

Pure functions only. Remove the ideal periodic pattern of a lattice frame and
keep each edge's stochastic line-edge roughness (LER), which is what makes a
periodic replica distinguishable from the true copy.

The method: smooth *along* the lattice lines (the direction the ideal structure
is constant in), so the smoothed frame is the ideal, roughness-free pattern;
the residual R = I - smooth_along is then the per-edge roughness. We keep R only
in a dilated band around the edges (where roughness is defined) and normalise it
locally so amplitude is comparable across the frame.

Hot-path ops are KleidiCV-listed only (sepFilter2D, GaussianBlur, Sobel, erode,
dilate). No matchTemplate, no Python per-pixel loops. The smoothing axis is chosen
from the S1 lattice orientation; sepFilter2D is axis-aligned, so this assumes the
dominant lattice axis is roughly image-aligned (true after S2 rectification, and
for near-upright frames). Rotated-frame handling is wired in a later stage (A4/A5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from driftsense.v2.lattice import Lattice

_log = logging.getLogger("driftsense.v2.roughness")

# KleidiCV accelerates separable filters at odd kernel lengths; anything else
# falls back to a scalar path. We snap a requested length to the nearest odd
# value in this set and log when we had to.
_KLEIDI_SEP_SIZES = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31, 41, 51, 61)


@dataclass
class RoughnessConfig:
    smooth_len: int = 0  # length of the along-line blur; 0 => auto from period
    smooth_auto_periods: float = 2.0  # auto smooth_len ~= this many dominant periods
    sobel_ksize: int = 3  # aperture for the edge-band gradient
    edge_thresh_frac: float = 0.18  # edge band = |grad| > frac * robust(|grad|)
    erode_ksize: int = 3  # structuring element to erode the background floor
    dilate_ksize: int = 3  # structuring element to widen edges into a band
    dilate_iter: int = 2  # how far the band spreads around each edge
    norm_sigma: float = 0.0  # local-normalisation Gaussian sigma; 0 => from tile
    norm_tile: int = 24  # per-tile window for local amplitude normalisation
    eps: float = 1e-6


@dataclass
class Residual:
    residual: np.ndarray  # float32, masked, locally normalised roughness field
    mask: np.ndarray  # uint8 {0,1} edge band the residual is defined on
    smooth_len: int  # along-line blur length actually used
    along_axis: int  # 0 => smoothed along rows (x), 1 => along columns (y)
    warnings: list[str] = field(default_factory=list)


def _snap_odd_kernel(k: int) -> tuple[int, bool]:
    """Snap `k` to the nearest KleidiCV-supported odd separable length."""
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    if k in _KLEIDI_SEP_SIZES:
        return k, False
    nearest = min(_KLEIDI_SEP_SIZES, key=lambda s: (abs(s - k), s))
    return nearest, True


def _along_axis(lattice: Lattice) -> int:
    """Which image axis the lattice *lines* run along (the smoothing direction).

    The lattice period is along k1 (angle_deg). Lines of constant structure run
    perpendicular to it. If the period is mostly horizontal (|angle| < 45) the
    lines are vertical, so we smooth down the columns (axis 1); otherwise along
    the rows (axis 0).
    """
    return 1 if abs(lattice.angle_deg) < 45.0 else 0


def _auto_smooth_len(lattice: Lattice, cfg: RoughnessConfig) -> int:
    if cfg.smooth_len > 0:
        return cfg.smooth_len
    period = lattice.period_px if np.isfinite(lattice.period_px) else cfg.norm_tile
    return int(round(cfg.smooth_auto_periods * period))


def residual(image: np.ndarray, lattice: Lattice, cfg: RoughnessConfig | None = None) -> Residual:
    """Extract the masked, locally normalised roughness residual of one frame."""
    cfg = cfg or RoughnessConfig()
    warnings: list[str] = []

    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("roughness.residual expects a single-channel frame")

    along = _along_axis(lattice)
    k, snapped = _snap_odd_kernel(_auto_smooth_len(lattice, cfg))
    if snapped:
        msg = f"smooth_len snapped to KleidiCV-supported {k} (fallback path otherwise)"
        warnings.append(msg)
        _log.info(msg)

    # Smooth ALONG the lines with a 1xk separable Gaussian (the cross-line axis
    # keeps a 1-tap identity, so only the along-line direction is blurred).
    ident = np.array([1.0], dtype=np.float32)
    gauss = cv2.getGaussianKernel(k, -1).astype(np.float32).ravel()
    if along == 1:  # lines vertical -> blur down columns (kernelY)
        smooth = cv2.sepFilter2D(img, cv2.CV_32F, ident, gauss)
    else:  # lines horizontal -> blur across rows (kernelX)
        smooth = cv2.sepFilter2D(img, cv2.CV_32F, gauss, ident)
    r = img - smooth  # ideal pattern removed; roughness remains

    # Edge band: threshold the gradient magnitude, erode the background floor to
    # kill isolated speckle, then dilate the surviving edges into a band.
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=cfg.sobel_ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=cfg.sobel_ksize)
    grad = cv2.magnitude(gx, gy)
    thr = cfg.edge_thresh_frac * float(np.percentile(grad, 99.0))
    edge = (grad > thr).astype(np.uint8)
    er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.erode_ksize, cfg.erode_ksize))
    dl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.dilate_ksize, cfg.dilate_ksize))
    edge = cv2.erode(edge, er)
    mask = cv2.dilate(edge, dl, iterations=cfg.dilate_iter)

    r = r * mask

    # Per-tile amplitude normalisation via Gaussian local statistics (box-free,
    # KleidiCV-listed). local_std = sqrt(E[r^2] - E[r]^2) over a ~tile window.
    sigma = cfg.norm_sigma if cfg.norm_sigma > 0 else cfg.norm_tile / 3.0
    ksz = _snap_odd_kernel(cfg.norm_tile)[0]
    local_mean = cv2.GaussianBlur(r, (ksz, ksz), sigma)
    local_energy = cv2.GaussianBlur(r * r, (ksz, ksz), sigma)
    local_std = np.sqrt(np.maximum(local_energy - local_mean * local_mean, 0.0))
    r = (r / (local_std + cfg.eps)) * mask

    return Residual(
        residual=np.ascontiguousarray(r, dtype=np.float32),
        mask=np.ascontiguousarray(mask, dtype=np.uint8),
        smooth_len=k,
        along_axis=along,
        warnings=warnings,
    )
