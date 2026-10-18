"""S5 — DRAM contour extraction (scaffold).

This is the entry point and result type the DRAM contour path will fill in A5.
The real logic (dot/blob contour tracing, sub-pixel edge fitting, pitch-aware
grouping) is deliberately NOT implemented here — building it would blow this PR's
scope. The scaffold fixes the signature and dataclass so S6/A5 can import against
a stable shape; the body raises a typed NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from driftsense.v2.lattice import Lattice


@dataclass
class ContourConfig:
    min_blob_px: int = 4  # smallest contour area to keep, in px
    subpixel: bool = True  # fit sub-pixel contour edges
    pitch_group: bool = True  # group contours onto the detected lattice


@dataclass
class ContourResult:
    contours: list[np.ndarray]  # (N, 2) float arrays of contour points, search coords
    centroids: np.ndarray  # (M, 2) float contour centroids
    pixel_size_nm: float | None = None
    warnings: list[str] = field(default_factory=list)


def extract_contours(image: np.ndarray, lattice: Lattice, cfg: ContourConfig | None = None) -> ContourResult:
    """Extract DRAM dot/line contours from a frame. (Full path lands in A5.)"""
    raise NotImplementedError("DRAM contour: A5")
