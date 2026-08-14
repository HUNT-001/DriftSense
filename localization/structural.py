"""
Stage 2D — structural prefilter.

ROLE IN THE PIPELINE
--------------------
    spectral lattice  ->  ~1500 structurally valid candidates
            |
            v
    STRUCTURAL PREFILTER  ->  <= ~150 candidates      (this module)
            |
            v
    LER fingerprint  ->  final periodic-replica disambiguation

The candidate-budget curve showed LER top-1 on the hard tier rising from 29%
(1500 candidates) to 55% (25 candidates): competitor count is a first-order
factor.  A prefilter that shrinks the set while keeping the truth therefore
directly buys accuracy, not just speed.

THE TWO-QUESTION DECOMPOSITION (why this is not just "more matching")
--------------------------------------------------------------------
    prefilter answers:  "does this location have the same COARSE STRUCTURE
                         and field-position signature as the reference?"
    LER stage answers:  "is this the same PHYSICAL location?"

To keep that separation clean and defensible, the prefilter must use signals
that are:

  * NOT LER            — LER is the fine discriminator; using it here would
                         collapse the two stages into one and destroy the
                         experimental separation.
  * NOT NCC ranking    — already shown to re-impose a photometric criterion
                         that discards the lattice's structural advantage.

THE SIGNAL IT USES: the low-frequency envelope
----------------------------------------------
On realistic wafers the critical dimension (linewidth) and the SE contrast
drift smoothly across the exposure field (dose/focus variation; Mack 2007).
This produces a slowly-varying INTENSITY ENVELOPE that is a function of
absolute field position and is independent of the periodic carrier.

Low-pass filtering an ROI with sigma comparable to the pitch removes the
periodic carrier (wavelength = pitch) AND the sub-pixel LER, leaving only that
envelope.  Downsampling yields a compact descriptor that encodes coarse field
position — exactly the information needed to reject candidates in the wrong
part of the field, without touching LER.

References
----------
[P1] Mack, "Fundamental Principles of Optical Lithography," Wiley 2007 — CD
     variation with dose/focus across the field.
[P2] Constantoudis et al., J. Micro/Nanolith. MEMS MOEMS 3(3), 2004 — LER is a
     distinct, high-spatial-frequency phenomenon from CD variation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# ===========================================================================

@dataclass
class StructuralDescriptor:
    """Compact, position-encoding descriptor of one ROI."""
    envelope: np.ndarray     # (g*g,) z-normalized low-frequency envelope
    grid: int

    def distance(self, other: "StructuralDescriptor") -> float:
        """
        Correlation distance in [0, 2]; 0 = identical envelope shape.

        Correlation (not raw L2) so that a global brightness/contrast offset
        between the two acquisitions does not by itself create distance — only
        the SHAPE of the envelope, i.e. the field-position signature, matters.
        """
        a, b = self.envelope, other.envelope
        if a.size != b.size or a.size == 0:
            return 2.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 2.0
        return float(1.0 - np.dot(a, b) / (na * nb))


def describe(patch: np.ndarray,
             pitch: float,
             grid: int = 12) -> StructuralDescriptor:
    """
    Build the low-frequency envelope descriptor of a patch.

    Steps:
      1. Gaussian low-pass with sigma ~ 0.75*pitch.  This suppresses the
         periodic carrier (wavelength = pitch) and the sub-pixel LER, keeping
         only the slow CD/contrast envelope.
      2. Resize to grid x grid (area interpolation = further anti-aliased
         downsampling).
      3. Remove the mean and z-normalize.  Removing the mean discards the
         DC brightness (which drifts between acquisitions and is not a
         reliable position cue); the remaining pattern is the envelope SHAPE.
    """
    img = np.asarray(patch, dtype=np.float32)
    sigma = max(0.75 * pitch, 1.5)
    k = int(6 * sigma) | 1
    low = cv2.GaussianBlur(img, (k, k), sigma)
    small = cv2.resize(low, (grid, grid), interpolation=cv2.INTER_AREA)
    v = small.astype(np.float64).ravel()
    v -= v.mean()
    s = v.std()
    if s > 1e-9:
        v /= s
    return StructuralDescriptor(envelope=v.astype(np.float32), grid=grid)


# ===========================================================================
# Prefilter
# ===========================================================================

def prefilter_candidates(ref: np.ndarray,
                          search: np.ndarray,
                          candidates: list[tuple[float, float]],
                          pitch: float,
                          keep: int = 150,
                          grid: int = 12) -> list[tuple[float, float]]:
    """
    Rank candidates by structural-envelope similarity to the reference and
    keep the closest `keep`.

    This is a COARSE gate: its job is recall (retain the truth), not precision
    (identify it).  It must keep the truth in its output for the downstream
    LER stage to have any chance, so `keep` should be chosen from a measured
    recall@K curve, not guessed.

    Returns the kept candidate centres, ordered by increasing structural
    distance (most reference-like first).
    """
    rh, rw = ref.shape[:2]
    ref_desc = describe(ref, pitch, grid)

    scored: list[tuple[float, float, float]] = []
    for (cx, cy) in candidates:
        x0 = int(round(cx - rw / 2.0))
        y0 = int(round(cy - rh / 2.0))
        if x0 < 0 or y0 < 0 or x0 + rw > search.shape[1] or y0 + rh > search.shape[0]:
            continue
        patch = search[y0:y0 + rh, x0:x0 + rw]
        d = ref_desc.distance(describe(patch, pitch, grid))
        scored.append((d, cx, cy))

    scored.sort(key=lambda s: s[0])
    return [(cx, cy) for (_, cx, cy) in scored[:keep]]
