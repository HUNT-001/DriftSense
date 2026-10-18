"""S2 — coarse axis-alignment (rectification) hypotheses.

Pure function. Given a frame and its S1 lattice, propose up to four 3x3
de-rotation homographies that bring the dominant lattice axis onto an image
axis, apply each on a coarse (pyrDown) copy, and return them ranked by how
closely the rectified lattice lands on the axes.

Hot-path ops are KleidiCV-listed only (pyrDown, warpPerspective). The homography
is built from getRotationMatrix2D; rectification uses warpPerspective (3x3), never
warpAffine.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from driftsense.v2.lattice import Lattice, detect_lattice

_MAX_HYPOTHESES = 4


@dataclass
class Rectified:
    image: np.ndarray  # coarse (pyrDown) rectified frame
    H: np.ndarray  # 3x3 homography applied to the coarse frame
    angle_deg: float  # rotation applied (degrees), reduced to (-180, 180]
    residual_px: float  # dominant-axis deviation from an image axis, in px


def _axis_residual_deg(deg: float) -> float:
    """Signed distance of an orientation from the nearest image axis, (-45, 45]."""
    a = (deg + 45.0) % 90.0 - 45.0
    return 45.0 if a == -45.0 else a


def _homography(center: tuple[float, float], angle_deg: float) -> np.ndarray:
    """3x3 rotation homography about `center` (embeds the 2x3 affine)."""
    m = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    h = np.eye(3, dtype=np.float64)
    h[:2, :] = m
    return h


def rectify(image: np.ndarray, lattice: Lattice) -> list[Rectified]:
    """Up to four coarse de-rotation hypotheses, best (lowest residual) first."""
    coarse = cv2.pyrDown(image)
    ch, cw = coarse.shape[:2]
    center = (cw / 2.0, ch / 2.0)

    # getRotationMatrix2D(angle) applied to the frame shifts the detected lattice
    # orientation by -angle, so rotating by +angle_deg brings the dominant axis
    # onto x. The 90° multiples cover which lattice axis maps horizontal and the
    # flip ambiguity. residual_px is measured — not assumed — by re-detecting the
    # lattice on a central crop of each rectified frame.
    seen: set[int] = set()
    out: list[Rectified] = []
    for k in range(4):
        rot = lattice.angle_deg + 90.0 * k
        rot = (rot + 180.0) % 360.0 - 180.0
        key = int(round(rot * 100))
        if key in seen:
            continue
        seen.add(key)
        H = _homography(center, rot)
        warped = cv2.warpPerspective(coarse, H, (cw, ch), flags=cv2.INTER_LINEAR)
        residual_px = _residual_px(warped, lattice.period_px)
        out.append(Rectified(image=warped, H=H, angle_deg=rot, residual_px=residual_px))
        if len(out) >= _MAX_HYPOTHESES:
            break

    out.sort(key=lambda r: r.residual_px)
    return out


def _residual_px(rectified: np.ndarray, period_px: float) -> float:
    """Axis-deviation of the rectified lattice, in px over one period."""
    h, w = rectified.shape[:2]
    crop = rectified[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    resid_ang = _axis_residual_deg(detect_lattice(crop).angle_deg)
    return abs(float(np.sin(np.radians(resid_ang)))) * period_px
