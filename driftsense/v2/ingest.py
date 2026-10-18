"""S0 — ingest + quality control.

Pure functions only. Decode an 8- or 16-bit grayscale SEM frame and compute a
small quality report (mean/std, min/max, saturation %, SNR proxy) that later
stages use to decide whether a frame is worth processing. No scoring here.

Hot-path ops are KleidiCV-listed only (imdecode, cvtColor, meanStdDev,
minMaxLoc); the saturation count is a vectorised NumPy mask, not a pixel loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from driftsense.contracts import DriftSenseInputError

# QC thresholds (loose by design — this gate only rejects degenerate frames).
_MIN_STD = 1.0  # std below this => effectively flat
_MIN_DYNRANGE_FRAC = 0.02  # (max-min)/rail below this => no usable contrast
_MAX_SATURATION = 0.20  # >20% of pixels pinned at a rail => clipped
_MIN_SNR = 0.5  # mean/std proxy floor


@dataclass
class QCReport:
    ok: bool
    reasons: list[str]
    mean: float
    std: float
    vmin: float
    vmax: float
    saturation_pct: float
    snr: float


@dataclass
class IngestResult:
    image: np.ndarray  # single-channel, native depth (uint8 or uint16)
    qc: QCReport


def _rail(img: np.ndarray) -> float:
    return 255.0 if img.dtype == np.uint8 else 65535.0


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _qc(img: np.ndarray) -> QCReport:
    rail = _rail(img)
    mean_arr, std_arr = cv2.meanStdDev(img)
    mean = float(mean_arr[0, 0])
    std = float(std_arr[0, 0])
    vmin, vmax, _, _ = cv2.minMaxLoc(img)
    vmin, vmax = float(vmin), float(vmax)
    # saturation: fraction of pixels pinned at either rail (vectorised mask).
    pinned = int(np.count_nonzero((img <= 0) | (img >= rail)))
    saturation_pct = 100.0 * pinned / float(img.size)
    snr = mean / std if std > 1e-6 else 0.0

    reasons: list[str] = []
    if std < _MIN_STD:
        reasons.append(f"flat frame (std {std:.3f} < {_MIN_STD})")
    if (vmax - vmin) < _MIN_DYNRANGE_FRAC * rail:
        reasons.append(f"low dynamic range ({vmax - vmin:.0f} of {rail:.0f})")
    if saturation_pct > 100.0 * _MAX_SATURATION:
        reasons.append(f"clipped ({saturation_pct:.1f}% pinned at a rail)")
    if snr < _MIN_SNR:
        reasons.append(f"low SNR proxy ({snr:.3f} < {_MIN_SNR})")

    return QCReport(
        ok=not reasons,
        reasons=reasons,
        mean=mean,
        std=std,
        vmin=vmin,
        vmax=vmax,
        saturation_pct=saturation_pct,
        snr=snr,
    )


def ingest(image_bytes: bytes) -> IngestResult:
    """Decode an 8/16-bit gray frame and attach a QC report."""
    if not image_bytes:
        raise DriftSenseInputError("ingest: empty input")
    buf = np.frombuffer(image_bytes, np.uint8)
    # IMREAD_UNCHANGED keeps native bit depth (16-bit gray stays uint16).
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise DriftSenseInputError("ingest: not a decodable image")
    img = _to_gray(img)
    if img.dtype not in (np.uint8, np.uint16):
        img = img.astype(np.uint16) if img.itemsize > 1 else img.astype(np.uint8)
    return IngestResult(image=np.ascontiguousarray(img), qc=_qc(img))
