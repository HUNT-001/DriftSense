"""
Synthetic semiconductor structure generators (v2 — physically corrected).

WHAT CHANGED FROM v1
--------------------
v1 contained a critical rendering bug: the line-width intensity profile was
applied ALONG the line direction instead of ACROSS it, so every "line" was a
single-pixel column/row containing a perpendicular intensity ramp.  No actual
periodic structure was produced.  v2 renders lines correctly as 2-D
anti-aliased bars perpendicular to their axis, and adds the aperiodic physics
(LER, CD variation, defects) that makes localization solvable at all.

Structure families
------------------
  DRAM   — dense 2-D array.  Two sub-variants:
             'grid'      : orthogonal bit-line / word-line crossings
             'staggered' : 6F^2-style offset (honeycomb) active-area layout,
                           which is what modern DRAM actually uses.
  FinFET — high-frequency fins crossed by low-frequency gates, plus fin-cut
           (diffusion-break) regions.  Strongly anisotropic spectrum.

References
----------
[S1] Bunday et al. (2006), "Determination of optimal parameters for CD-SEM
     measurement of line edge roughness," Proc. SPIE 6152.
[S2] Constantoudis et al. (2004), "Line edge roughness and critical dimension
     variation," J. Micro/Nanolith. MEMS MOEMS 3(3).
[S3] Mack, C. (2007), "Fundamental Principles of Optical Lithography," Wiley.
     — CD variation with dose/focus across the exposure field.
[S4] Kinoshita et al. (2016), "6F2 DRAM cell architecture," IEEE IEDM.
[S5] Applied Materials PS2 problem statement — DRAM arrays and FinFET
     structures as canonical template-matching failure cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple, List

import numpy as np
from scipy.ndimage import gaussian_filter1d

from config import AperiodicityConfig


# ===========================================================================
# Aperiodic field helpers
# ===========================================================================

def make_ler_profile(length: int,
                     sigma_px: float,
                     correlation_px: float,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Generate a spatially-correlated Line-Edge Roughness displacement profile.

    LER is modelled as white noise low-pass filtered to a finite correlation
    length, which reproduces the measured LER power spectral density shape
    reported in [S1] and [S2].  The result is normalized so its standard
    deviation equals `sigma_px`.

    Parameters
    ----------
    length         : number of samples along the line (px)
    sigma_px       : target RMS edge displacement (px)
    correlation_px : correlation length (px)

    Returns
    -------
    float32 array of length `length`, displacement in pixels.

    NOTE: This profile is generated ONCE PER SCENE and shared between the
    reference and search acquisitions, because roughness is a permanent
    physical property of the etched wafer — not per-scan noise.  This is the
    signal that makes periodic-structure localization solvable.
    """
    if sigma_px <= 0:
        return np.zeros(length, dtype=np.float32)
    white = rng.standard_normal(length).astype(np.float32)
    smooth = gaussian_filter1d(white, sigma=max(correlation_px / 3.0, 0.5),
                                mode="wrap")
    std = smooth.std()
    if std > 1e-8:
        smooth = smooth / std * sigma_px
    return smooth.astype(np.float32)


def make_smooth_field(height: int, width: int,
                       scale_px: float,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Generate a smooth zero-mean 2-D random field with unit standard deviation.

    Used for CD (linewidth) variation and contrast drift across the exposure
    field.  Implemented as low-pass filtered white noise on a coarse grid,
    then bilinearly upsampled — cheap and O(N) in output size.
    """
    import cv2
    coarse_h = max(4, int(np.ceil(height / max(scale_px, 8.0))) + 2)
    coarse_w = max(4, int(np.ceil(width  / max(scale_px, 8.0))) + 2)
    coarse = rng.standard_normal((coarse_h, coarse_w)).astype(np.float32)
    field_ = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    field_ -= field_.mean()
    std = field_.std()
    if std > 1e-8:
        field_ /= std
    return field_


# ===========================================================================
# Core line renderer  (THE v1 BUG FIX)
# ===========================================================================

def _render_bar(canvas: np.ndarray,
                 position: float,
                 axis: int,
                 half_width: np.ndarray | float,
                 brightness: np.ndarray | float,
                 offset: np.ndarray | float = 0.0,
                 edge_soft: float = 0.9) -> None:
    """
    Render one anti-aliased bar (line) into `canvas`, IN-PLACE, using max-blend.

    The intensity profile varies ACROSS the bar (perpendicular to its axis) —
    this is the correct geometry, and the specific bug that v1 got wrong.

    Parameters
    ----------
    canvas      : (H, W) float32 target
    position    : centre coordinate of the bar along the perpendicular axis
    axis        : 0 -> vertical bar   (constant x, spans all y)
                  1 -> horizontal bar (constant y, spans all x)
    half_width  : scalar, or per-along-axis array (enables CD variation)
    brightness  : scalar, or per-along-axis array
    offset      : scalar, or per-along-axis array of lateral displacement
                  (this is where LER enters)
    edge_soft   : width of the intensity roll-off at the bar edge (px).
                  ~0.9 px approximates the beam-limited edge slope in SEM.

    Blending uses element-wise maximum rather than addition so that
    intersections of two bars do not saturate to a flat clipped plateau;
    intersection brightening is instead handled physically by the
    secondary-electron edge-enhancement stage in sem_effects.py.
    """
    H, W = canvas.shape
    along_len = H if axis == 0 else W          # length along the bar
    perp_len  = W if axis == 0 else H          # extent perpendicular to bar

    # Broadcast parameters to along-axis vectors
    hw  = np.broadcast_to(np.asarray(half_width, dtype=np.float32),
                          (along_len,))
    br  = np.broadcast_to(np.asarray(brightness, dtype=np.float32),
                          (along_len,))
    off = np.broadcast_to(np.asarray(offset, dtype=np.float32),
                          (along_len,))

    centre = position + off                     # (along_len,)
    max_hw = float(hw.max()) + edge_soft + 2.0
    lo = int(np.floor(centre.min() - max_hw))
    hi = int(np.ceil (centre.max() + max_hw)) + 1
    lo_c, hi_c = max(0, lo), min(perp_len, hi)
    if hi_c <= lo_c:
        return                                   # bar lies fully outside canvas

    perp = np.arange(lo_c, hi_c, dtype=np.float32)          # (P,)
    dist = np.abs(perp[None, :] - centre[:, None])          # (along, P)

    # Smooth top-hat: 1 inside the bar, ramping to 0 over `edge_soft` px.
    profile = np.clip((hw[:, None] - dist) / edge_soft + 0.5, 0.0, 1.0)
    profile = profile * br[:, None]                          # (along, P)

    if axis == 0:                                # vertical bar
        region = canvas[:, lo_c:hi_c]
        np.maximum(region, profile, out=region)
    else:                                        # horizontal bar
        region = canvas[lo_c:hi_c, :]
        np.maximum(region, profile.T, out=region)


# ===========================================================================
# Defect injection
# ===========================================================================

def _inject_defects(canvas: np.ndarray,
                     n_defects: int,
                     defect_types: Tuple[str, ...],
                     rng: np.random.Generator) -> List[dict]:
    """
    Inject randomly-placed manufacturing defects; returns their descriptors.

    particle : bright Gaussian blob (foreign material / residue)
    bridge   : short bright bar connecting adjacent lines (electrical short)
    open     : dark rectangle erasing part of a line (electrical open)
    """
    H, W = canvas.shape
    records = []
    for _ in range(n_defects):
        dtype = str(rng.choice(defect_types))
        cy = int(rng.integers(0, H))
        cx = int(rng.integers(0, W))

        if dtype == "particle":
            radius = float(rng.uniform(2.5, 8.0))
            amp    = float(rng.uniform(0.45, 1.0))
            r = int(np.ceil(radius * 3))
            y0, y1 = max(0, cy - r), min(H, cy + r + 1)
            x0, x1 = max(0, cx - r), min(W, cx + r + 1)
            if y1 <= y0 or x1 <= x0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            blob = amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)
                                   / (2 * radius ** 2)))
            region = canvas[y0:y1, x0:x1]
            np.maximum(region, blob.astype(np.float32), out=region)

        elif dtype == "bridge":
            length = int(rng.integers(6, 26))
            thick  = int(rng.integers(2, 5))
            amp    = float(rng.uniform(0.6, 1.0))
            horiz  = bool(rng.integers(0, 2))
            if horiz:
                y0, y1 = max(0, cy), min(H, cy + thick)
                x0, x1 = max(0, cx), min(W, cx + length)
            else:
                y0, y1 = max(0, cy), min(H, cy + length)
                x0, x1 = max(0, cx), min(W, cx + thick)
            if y1 <= y0 or x1 <= x0:
                continue
            region = canvas[y0:y1, x0:x1]
            np.maximum(region, np.float32(amp), out=region)

        elif dtype == "open":
            length = int(rng.integers(5, 20))
            thick  = int(rng.integers(3, 8))
            horiz  = bool(rng.integers(0, 2))
            if horiz:
                y0, y1 = max(0, cy), min(H, cy + thick)
                x0, x1 = max(0, cx), min(W, cx + length)
            else:
                y0, y1 = max(0, cy), min(H, cy + length)
                x0, x1 = max(0, cx), min(W, cx + thick)
            if y1 <= y0 or x1 <= x0:
                continue
            canvas[y0:y1, x0:x1] *= 0.15
        else:
            continue

        records.append({"type": dtype, "x": cx, "y": cy})
    return records


# ===========================================================================
# DRAM
# ===========================================================================

@dataclass
class DRAMParams:
    variant: Literal["grid", "staggered"] = "grid"
    period_x: float = 24.0
    period_y: float = 28.0
    line_width_x: float = 6.0
    line_width_y: float = 6.0
    brightness_x: float = 0.85
    brightness_y: float = 1.0
    phase_x: float = 0.0
    phase_y: float = 0.0
    background: float = 0.06
    # staggered variant only:
    cell_radius: float = 4.5
    row_offset_frac: float = 0.5      # 0.5 -> honeycomb / 6F^2 style


def sample_dram_params(rng: np.random.Generator) -> DRAMParams:
    return DRAMParams(
        variant       = str(rng.choice(["grid", "staggered"])),
        period_x      = float(rng.uniform(18, 34)),
        period_y      = float(rng.uniform(20, 38)),
        line_width_x  = float(rng.uniform(4.0, 9.0)),
        line_width_y  = float(rng.uniform(4.0, 9.0)),
        brightness_x  = float(rng.uniform(0.70, 0.95)),
        brightness_y  = float(rng.uniform(0.85, 1.00)),
        phase_x       = float(rng.uniform(0, 40)),
        phase_y       = float(rng.uniform(0, 40)),
        background    = float(rng.uniform(0.03, 0.12)),
        cell_radius   = float(rng.uniform(3.0, 6.5)),
        row_offset_frac = float(rng.choice([0.0, 0.5])),
    )


def generate_dram(height: int, width: int,
                  params: DRAMParams,
                  aper: AperiodicityConfig,
                  rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """
    Render a DRAM-style array with correct 2-D bars plus aperiodic physics.

    Returns (image, info) where info records the true structural periods and
    defect list — used later by the periodicity failure analysis.
    """
    canvas = np.full((height, width), params.background, dtype=np.float32)

    # --- Aperiodic modulation fields ---
    if aper.enable_cd_variation:
        cd_field = make_smooth_field(height, width,
                                      aper.cd_variation_scale_px, rng)
    else:
        cd_field = None
    if aper.enable_contrast_drift:
        br_field = make_smooth_field(height, width,
                                      aper.cd_variation_scale_px * 1.7, rng)
    else:
        br_field = None

    def _cd_scale_along(axis: int, pos: float) -> np.ndarray:
        """Per-along-axis linewidth multiplier sampled from the CD field."""
        if cd_field is None:
            return np.float32(1.0)
        p = int(np.clip(round(pos), 0, (width - 1) if axis == 0 else (height - 1)))
        line = cd_field[:, p] if axis == 0 else cd_field[p, :]
        return (1.0 + aper.cd_variation_amplitude * line).astype(np.float32)

    def _br_scale_along(axis: int, pos: float) -> np.ndarray:
        if br_field is None:
            return np.float32(1.0)
        p = int(np.clip(round(pos), 0, (width - 1) if axis == 0 else (height - 1)))
        line = br_field[:, p] if axis == 0 else br_field[p, :]
        return (1.0 + aper.contrast_drift_amplitude * line).astype(np.float32)

    # --- Vertical bars (bit lines) ---
    x = params.phase_x % params.period_x
    while x < width:
        ler = (make_ler_profile(height, aper.ler_sigma_px,
                                 aper.ler_correlation_px, rng)
               if aper.enable_ler else 0.0)
        hw = (params.line_width_x / 2.0) * _cd_scale_along(0, x)
        br = params.brightness_x * _br_scale_along(0, x)
        _render_bar(canvas, x, axis=0, half_width=hw,
                    brightness=br, offset=ler)
        x += params.period_x

    # --- Horizontal bars (word lines) ---
    y = params.phase_y % params.period_y
    while y < height:
        ler = (make_ler_profile(width, aper.ler_sigma_px,
                                 aper.ler_correlation_px, rng)
               if aper.enable_ler else 0.0)
        hw = (params.line_width_y / 2.0) * _cd_scale_along(1, y)
        br = params.brightness_y * _br_scale_along(1, y)
        _render_bar(canvas, y, axis=1, half_width=hw,
                    brightness=br, offset=ler)
        y += params.period_y

    # --- Staggered capacitor cells (6F^2-style, [S4]) ---
    if params.variant == "staggered":
        r = params.cell_radius
        rr = int(np.ceil(r * 3))
        row_i = 0
        yy_c = params.phase_y % params.period_y + params.period_y / 2.0
        while yy_c < height:
            x_off = (params.row_offset_frac * params.period_x) * (row_i % 2)
            xx_c = params.phase_x % params.period_x + params.period_x / 2.0 + x_off
            while xx_c < width:
                cy, cx = int(round(yy_c)), int(round(xx_c))
                y0, y1 = max(0, cy - rr), min(height, cy + rr + 1)
                x0, x1 = max(0, cx - rr), min(width, cx + rr + 1)
                if y1 > y0 and x1 > x0:
                    gy, gx = np.mgrid[y0:y1, x0:x1]
                    cell = np.exp(-(((gy - yy_c) ** 2 + (gx - xx_c) ** 2)
                                     / (2 * r ** 2))).astype(np.float32)
                    region = canvas[y0:y1, x0:x1]
                    np.maximum(region, cell * 0.9, out=region)
                xx_c += params.period_x
            yy_c += params.period_y
            row_i += 1

    # --- Defects ---
    defects = []
    if aper.enable_defects:
        n_def = int(aper.defect_density * height * width)
        defects = _inject_defects(canvas, n_def, aper.defect_types, rng)

    info = {
        "family": "dram",
        "variant": params.variant,
        "period_x": params.period_x,
        "period_y": params.period_y,
        "n_defects": len(defects),
    }
    return np.clip(canvas, 0.0, 1.0), info


# ===========================================================================
# FinFET
# ===========================================================================

@dataclass
class FinFETParams:
    fin_period: float = 11.0
    gate_period: float = 42.0
    fin_width: float = 4.0
    gate_width: float = 8.0
    fin_brightness: float = 0.72
    gate_brightness: float = 1.0
    fin_phase: float = 0.0
    gate_phase: float = 0.0
    fin_axis: int = 0                 # 0 -> vertical fins
    background: float = 0.06
    fin_cut_period: float = 260.0     # diffusion-break spacing
    fin_cut_width: float = 16.0
    enable_fin_cut: bool = True


def sample_finfet_params(rng: np.random.Generator) -> FinFETParams:
    return FinFETParams(
        fin_period      = float(rng.uniform(9, 17)),
        gate_period     = float(rng.uniform(34, 62)),
        fin_width       = float(rng.uniform(3.0, 6.0)),
        gate_width      = float(rng.uniform(6.0, 12.0)),
        fin_brightness  = float(rng.uniform(0.60, 0.85)),
        gate_brightness = float(rng.uniform(0.85, 1.00)),
        fin_phase       = float(rng.uniform(0, 20)),
        gate_phase      = float(rng.uniform(0, 60)),
        fin_axis        = int(rng.integers(0, 2)),
        background      = float(rng.uniform(0.03, 0.12)),
        fin_cut_period  = float(rng.uniform(180, 380)),
        fin_cut_width   = float(rng.uniform(10, 24)),
        enable_fin_cut  = bool(rng.random() < 0.7),
    )


def generate_finfet(height: int, width: int,
                    params: FinFETParams,
                    aper: AperiodicityConfig,
                    rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """
    Render a FinFET-style structure: dense fins + sparse perpendicular gates,
    with optional fin-cut (diffusion break) regions.

    The two very different pitches make the 2-D spectrum strongly anisotropic,
    which is precisely what defeats isotropic correlation matchers [S5].
    """
    canvas = np.full((height, width), params.background, dtype=np.float32)
    gate_axis = 1 - params.fin_axis

    if aper.enable_cd_variation:
        cd_field = make_smooth_field(height, width,
                                      aper.cd_variation_scale_px, rng)
    else:
        cd_field = None
    if aper.enable_contrast_drift:
        br_field = make_smooth_field(height, width,
                                      aper.cd_variation_scale_px * 1.7, rng)
    else:
        br_field = None

    def _mod_along(f, axis, pos, amp):
        if f is None:
            return np.float32(1.0)
        limit = (width - 1) if axis == 0 else (height - 1)
        p = int(np.clip(round(pos), 0, limit))
        line = f[:, p] if axis == 0 else f[p, :]
        return (1.0 + amp * line).astype(np.float32)

    # --- Fins (high frequency) ---
    fin_extent = width if params.fin_axis == 0 else height
    fin_along  = height if params.fin_axis == 0 else width
    f = params.fin_phase % params.fin_period
    while f < fin_extent:
        ler = (make_ler_profile(fin_along, aper.ler_sigma_px,
                                 aper.ler_correlation_px, rng)
               if aper.enable_ler else 0.0)
        hw = (params.fin_width / 2.0) * _mod_along(
            cd_field, params.fin_axis, f, aper.cd_variation_amplitude)
        br = params.fin_brightness * _mod_along(
            br_field, params.fin_axis, f, aper.contrast_drift_amplitude)
        _render_bar(canvas, f, axis=params.fin_axis,
                    half_width=hw, brightness=br, offset=ler)
        f += params.fin_period

    # --- Fin cuts (diffusion breaks) erase fins in narrow bands ---
    if params.enable_fin_cut:
        cut_extent = height if params.fin_axis == 0 else width
        c = params.fin_cut_period * 0.5
        while c < cut_extent:
            lo = int(max(0, c - params.fin_cut_width / 2))
            hi = int(min(cut_extent, c + params.fin_cut_width / 2))
            if hi > lo:
                if params.fin_axis == 0:
                    canvas[lo:hi, :] *= 0.25
                else:
                    canvas[:, lo:hi] *= 0.25
            c += params.fin_cut_period

    # --- Gates (low frequency, perpendicular, drawn on top) ---
    gate_extent = height if params.fin_axis == 0 else width
    gate_along  = width if params.fin_axis == 0 else height
    g = params.gate_phase % params.gate_period
    while g < gate_extent:
        ler = (make_ler_profile(gate_along, aper.ler_sigma_px,
                                 aper.ler_correlation_px, rng)
               if aper.enable_ler else 0.0)
        hw = (params.gate_width / 2.0) * _mod_along(
            cd_field, gate_axis, g, aper.cd_variation_amplitude)
        br = params.gate_brightness * _mod_along(
            br_field, gate_axis, g, aper.contrast_drift_amplitude)
        _render_bar(canvas, g, axis=gate_axis,
                    half_width=hw, brightness=br, offset=ler)
        g += params.gate_period

    defects = []
    if aper.enable_defects:
        n_def = int(aper.defect_density * height * width)
        defects = _inject_defects(canvas, n_def, aper.defect_types, rng)

    info = {
        "family": "finfet",
        "variant": "fin_cut" if params.enable_fin_cut else "plain",
        "period_x": (params.fin_period if params.fin_axis == 0
                     else params.gate_period),
        "period_y": (params.gate_period if params.fin_axis == 0
                     else params.fin_period),
        "n_defects": len(defects),
    }
    return np.clip(canvas, 0.0, 1.0), info


# ===========================================================================
# Factory
# ===========================================================================

def generate_structure(structure_type: str,
                       height: int, width: int,
                       aper: AperiodicityConfig,
                       rng: np.random.Generator) -> Tuple[np.ndarray, dict]:
    """
    structure_type in {'dram', 'finfet', 'random'}.
    Returns (float32 image in [0,1], info dict with true periods).
    """
    if structure_type == "random":
        structure_type = str(rng.choice(["dram", "finfet"]))
    if structure_type == "dram":
        return generate_dram(height, width, sample_dram_params(rng), aper, rng)
    if structure_type == "finfet":
        return generate_finfet(height, width, sample_finfet_params(rng),
                                aper, rng)
    raise ValueError(f"Unknown structure_type: {structure_type!r}")
