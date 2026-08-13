"""
Central configuration for the DriftSense synthetic data engine.

Everything that controls dataset realism lives here so that experiments are
reproducible and ablations are one-line changes.

IMPORTANT DESIGN NOTE — solvability of the task
------------------------------------------------
A *perfectly* periodic scene is information-theoretically ambiguous: if the
structure repeats with period P, then the reference patch matches equally well
at every location offset by k·P.  No algorithm — classical or learned — can
resolve this, because the observation contains no signal that distinguishes
the candidates.

Real wafers are NOT perfectly periodic.  The aperiodic content that makes
localization solvable in practice is:

  1. Line-Edge Roughness (LER) — stochastic, spatially-fixed roughness of each
     printed edge caused by photoresist polymer/photon shot statistics.
     LER is a *stable physical property of the wafer*, so two separate SEM
     scans of the same region observe the SAME roughness (plus independent
     detector noise).  This makes LER a genuine positional fingerprint.
     Ref: Bunday et al., Proc. SPIE 6152 (2006); Constantoudis et al.,
          J. Micro/Nanolith. MEMS MOEMS 3(3), 2004.

  2. Critical-Dimension (CD) variation across the field — lithographic dose
     and focus vary slowly across the exposure field, so line width drifts
     smoothly across the image.  This gives a low-frequency positional cue.
     Ref: Mack, "Fundamental Principles of Optical Lithography," Wiley 2007.

  3. Defects and array-boundary features — particles, bridges, opens.
     Ref: Applied Materials PS2 problem statement.

DriftSense models all three, and each can be switched OFF via the
`AperiodicityConfig` flags below to *demonstrate the ambiguity limit* — a key
experiment showing that our Stage-2 gains come from exploiting real physics,
not from overfitting.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Tuple


# ---------------------------------------------------------------------------
# Image geometry (fixed by the challenge specification)
# ---------------------------------------------------------------------------

SEARCH_H = 1000
SEARCH_W = 1000

REF_MIN_SIZE = 80          # reference is ~10x smaller than search
REF_MAX_SIZE = 130

# The scene must be large enough that a rotated + scaled 1000x1000 crop is
# fully covered by real rendered content (no border padding artifacts).
# For rotation up to ~6 deg and scale down to 0.90:
#   required >= 1000 * sqrt(2) / 0.90  ~= 1572 -> round up with margin.
SCENE_H = 1650
SCENE_W = 1650

# Padding used when rendering the reference so blur/edge effects at the crop
# border match those in the search image.
REF_RENDER_PAD = 24


# ---------------------------------------------------------------------------
# Aperiodicity (the signal that makes the problem solvable)
# ---------------------------------------------------------------------------

@dataclass
class AperiodicityConfig:
    """Controls the aperiodic content that breaks translational ambiguity."""

    # --- Line-Edge Roughness ---
    enable_ler: bool = True
    ler_sigma_px: float = 0.55       # RMS edge displacement (px). Real 3-sigma
                                     # LER is ~2-5 nm; at ~1 nm/px this is ~0.5 px.
    ler_correlation_px: float = 18.0 # correlation length along the line (px).
                                     # Real LER correlation length ~20-50 nm.

    # --- Critical-Dimension variation across the field ---
    enable_cd_variation: bool = True
    cd_variation_amplitude: float = 0.22  # fractional linewidth swing (+/-22%)
    cd_variation_scale_px: float = 420.0  # spatial wavelength of CD drift

    # --- Brightness / material contrast drift across field ---
    enable_contrast_drift: bool = True
    contrast_drift_amplitude: float = 0.18

    # --- Defects ---
    enable_defects: bool = True
    defect_density: float = 1.6e-5   # defects per pixel of scene area
                                     # 1.6e-5 * 1650^2 ~= 44 defects/scene
    defect_types: Tuple[str, ...] = ("particle", "bridge", "open")


# ---------------------------------------------------------------------------
# SEM acquisition parameters
# ---------------------------------------------------------------------------

@dataclass
class SEMConfig:
    """Ranges for SEM imaging effects. Sampled independently per acquisition."""

    blur_sigma_range: Tuple[float, float] = (0.6, 1.9)
    edge_strength_range: Tuple[float, float] = (0.20, 0.45)
    drift_rate_range: Tuple[float, float] = (0.0, 0.05)

    # Electron dose (mean e- per pixel at full brightness) -> controls shot noise
    dose_low: Tuple[float, float] = (300.0, 700.0)     # LOW noise  = HIGH dose
    dose_medium: Tuple[float, float] = (90.0, 250.0)
    dose_high: Tuple[float, float] = (25.0, 70.0)      # HIGH noise = LOW dose

    # Detector / amplifier read noise (additive Gaussian, dose-independent)
    read_noise_sigma_range: Tuple[float, float] = (0.005, 0.030)

    # Horizontal scan-line gain jitter (line-to-line amplifier variation)
    scanline_jitter_sigma_range: Tuple[float, float] = (0.0, 0.025)

    # Specimen charging (slow low-frequency brightness undulation)
    charging_amplitude_range: Tuple[float, float] = (0.0, 0.10)

    # Vignetting / detector collection-efficiency falloff
    vignette_strength_range: Tuple[float, float] = (0.0, 0.16)

    # Global brightness / contrast / gamma variation between acquisitions
    brightness_offset_range: Tuple[float, float] = (-0.06, 0.06)
    contrast_gain_range: Tuple[float, float] = (0.88, 1.14)
    gamma_range: Tuple[float, float] = (0.85, 1.18)


# ---------------------------------------------------------------------------
# Geometric relationship between reference and search acquisitions
# ---------------------------------------------------------------------------

@dataclass
class GeometryConfig:
    rotation_deg_range: Tuple[float, float] = (-3.0, 3.0)
    scale_range: Tuple[float, float] = (0.96, 1.04)
    # Sub-pixel offset applied to the reference crop so ground truth is not
    # quantized to integers (essential for evaluating sub-pixel refinement).
    enable_subpixel_offset: bool = True


# ---------------------------------------------------------------------------
# Difficulty tiers
# ---------------------------------------------------------------------------

@dataclass
class TierConfig:
    """A named difficulty preset combining geometry + SEM + aperiodicity."""
    name: str
    geometry: GeometryConfig
    sem: SEMConfig
    aperiodicity: AperiodicityConfig
    noise_levels: Tuple[str, ...] = ("low", "medium", "high")


def _clean_tier() -> TierConfig:
    """
    CLEAN: no rotation, no scale, low noise.

    Purpose: isolates PERIODICITY as the sole difficulty.  If a classical
    matcher fails here, the failure is unambiguously caused by structural
    self-similarity, not by geometric distortion or noise.  This is the
    controlled experiment that justifies the whole Stage-2 design.
    """
    return TierConfig(
        name="clean",
        geometry=GeometryConfig(
            rotation_deg_range=(0.0, 0.0),
            scale_range=(1.0, 1.0),
            enable_subpixel_offset=True,
        ),
        sem=SEMConfig(
            blur_sigma_range=(0.6, 1.0),
            drift_rate_range=(0.0, 0.0),
            charging_amplitude_range=(0.0, 0.02),
            vignette_strength_range=(0.0, 0.03),
            scanline_jitter_sigma_range=(0.0, 0.005),
        ),
        aperiodicity=AperiodicityConfig(),
        noise_levels=("low",),
    )


def _nominal_tier() -> TierConfig:
    """NOMINAL: realistic tool conditions — small rotation/scale, mixed noise."""
    return TierConfig(
        name="nominal",
        geometry=GeometryConfig(
            rotation_deg_range=(-1.5, 1.5),
            scale_range=(0.985, 1.015),
        ),
        sem=SEMConfig(),
        aperiodicity=AperiodicityConfig(),
        noise_levels=("low", "medium"),
    )


def _hard_tier() -> TierConfig:
    """HARD: stresses every axis — large misalignment, low dose, heavy drift."""
    return TierConfig(
        name="hard",
        geometry=GeometryConfig(
            rotation_deg_range=(-5.0, 5.0),
            scale_range=(0.93, 1.07),
        ),
        sem=SEMConfig(
            blur_sigma_range=(0.8, 2.6),
            drift_rate_range=(0.0, 0.12),
            charging_amplitude_range=(0.02, 0.18),
            vignette_strength_range=(0.04, 0.26),
            scanline_jitter_sigma_range=(0.01, 0.05),
            read_noise_sigma_range=(0.015, 0.055),
        ),
        aperiodicity=AperiodicityConfig(),
        noise_levels=("medium", "high"),
    )


def _ambiguous_tier() -> TierConfig:
    """
    AMBIGUOUS (ablation): all aperiodic content DISABLED.

    This produces a perfectly periodic scene.  It is the theoretical lower
    bound — no method can exceed chance-among-aliases here.  Reporting this
    proves our Stage-2 gains come from exploiting LER/CD physics rather than
    from any dataset leakage.
    """
    return TierConfig(
        name="ambiguous",
        geometry=GeometryConfig(
            rotation_deg_range=(0.0, 0.0),
            scale_range=(1.0, 1.0),
        ),
        sem=SEMConfig(
            blur_sigma_range=(0.6, 1.0),
            drift_rate_range=(0.0, 0.0),
            charging_amplitude_range=(0.0, 0.0),
            vignette_strength_range=(0.0, 0.0),
            scanline_jitter_sigma_range=(0.0, 0.0),
        ),
        aperiodicity=AperiodicityConfig(
            enable_ler=False,
            enable_cd_variation=False,
            enable_contrast_drift=False,
            enable_defects=False,
        ),
        noise_levels=("low",),
    )


TIERS = {
    "clean":     _clean_tier,
    "nominal":   _nominal_tier,
    "hard":      _hard_tier,
    "ambiguous": _ambiguous_tier,
}


def get_tier(name: str) -> TierConfig:
    if name not in TIERS:
        raise ValueError(f"Unknown tier {name!r}. Options: {list(TIERS)}")
    return TIERS[name]()


def tier_to_dict(tier: TierConfig) -> dict:
    """Serializable snapshot of a tier config (written into the manifest)."""
    return {
        "name": tier.name,
        "geometry": asdict(tier.geometry),
        "sem": asdict(tier.sem),
        "aperiodicity": asdict(tier.aperiodicity),
        "noise_levels": list(tier.noise_levels),
    }
