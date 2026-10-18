"""S8 — sub-pixel translation refinement.

Pure functions. Given the template and the matched search patch (same size),
estimate the residual sub-pixel shift between them: a coarse `phaseCorrelate`
(Hanning-windowed) seeds a `findTransformECC(MOTION_TRANSLATION)` refinement, and
if ECC fails to converge we fall back to the phase-correlation shift alone. The
shift is reported in pixels and, when a pixel size is known, in nanometres.

Hot-path ops are KleidiCV-listed / established exceptions only: createHanningWindow
(elementwise), phaseCorrelate and findTransformECC (the subpixel exceptions, same
class as matchTemplate/minMaxLoc). No Python per-pixel loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_ECC_EPS = 1e-5
_ECC_ITERS = 100


@dataclass
class SubpixelConfig:
    ecc_eps: float = _ECC_EPS
    ecc_iters: int = _ECC_ITERS
    use_ecc: bool = True
    gauss_filt_size: int = 5  # ECC input smoothing aperture (odd)


@dataclass
class SubpixelResult:
    dx_px: float  # sub-pixel shift of the search patch relative to the template
    dy_px: float
    dx_nm: float | None
    dy_nm: float | None
    method: str  # "phase+ecc" | "phase" | "none"
    response: float  # phaseCorrelate response (peak sharpness, 0..1)


def _f32(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return a.astype(np.float32)


def refine(
    template: np.ndarray,
    patch: np.ndarray,
    pixel_size_nm: float | None = None,
    cfg: SubpixelConfig | None = None,
) -> SubpixelResult:
    """Estimate the sub-pixel shift that aligns `patch` onto `template`.

    `template` and `patch` must be the same shape (the patch is the search window
    cropped at the integer-pixel candidate location).
    """
    cfg = cfg or SubpixelConfig()
    t = _f32(template)
    p = _f32(patch)
    if t.shape != p.shape:
        raise ValueError("subpixel.refine: template and patch must match in shape")

    h, w = t.shape[:2]
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    # phaseCorrelate(a, b) returns the shift of `b` relative to `a`; we want how
    # far `patch` sits from `template`, so a=template, b=patch.
    (sx, sy), response = cv2.phaseCorrelate(t, p, win)
    dx, dy, method = float(sx), float(sy), "phase"

    if cfg.use_ecc:
        warp = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(cfg.ecc_iters),
            float(cfg.ecc_eps),
        )
        try:
            # Align `patch` (input) onto `template`; the translation in `warp`
            # is then the sub-pixel offset of patch from template.
            _cc, warp = cv2.findTransformECC(
                t, p, warp, cv2.MOTION_TRANSLATION, criteria, None, cfg.gauss_filt_size
            )
            dx, dy = float(warp[0, 2]), float(warp[1, 2])
            method = "phase+ecc"
        except cv2.error:
            method = "phase"  # keep the phase-correlation estimate

    dx_nm = dx * pixel_size_nm if pixel_size_nm is not None else None
    dy_nm = dy * pixel_size_nm if pixel_size_nm is not None else None
    return SubpixelResult(
        dx_px=dx, dy_px=dy, dx_nm=dx_nm, dy_nm=dy_nm, method=method, response=float(response)
    )
