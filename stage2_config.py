"""
FROZEN Stage-2B configuration.

WHY THIS FILE EXISTS
--------------------
Stage-2B hyperparameters (detrend order, top_k, scoring weights) were selected
by inspecting results on the `clean`/`nominal`/`hard`/`ambiguous` datasets.
Continuing to tune on those same sets and then quoting the resulting numbers
would be selection on the test set — the numbers would not survive contact
with Applied Materials' held-out data.

So Stage 2B is frozen here, as of the run that produced:

    clean      100.0% -> 100.0%   (+0.0)
    nominal     82.5% ->  87.5%   (+5.0)
    hard        17.5% ->  37.5%   (+20.0)
    ambiguous    0.0% ->   0.0%   (+0.0)

From this point on:

  * DEV  set  = outputs/dataset       (seed 42 family)  — used for development
  * TEST set  = outputs/dataset_test  (seed 9999 family) — touched ONCE, at the end

Stage 2A (spectral candidate generation) is developed against the DEV set and
measured on RECALL, not on final accuracy, so it cannot leak through the
frozen 2B parameters.

Do not edit the FROZEN block to chase a better number.  If a parameter must
change, record why, and re-run the held-out evaluation from scratch.
"""

from dataclasses import dataclass


# ===========================================================================
# FROZEN — do not tune further
# ===========================================================================

@dataclass(frozen=True)
class Stage2BParams:
    """Locked LER fingerprint + re-ranking parameters."""

    # --- Fingerprint extraction ---
    detrend_order: int = 1
    """1 = linear.  Removes the rotation-induced ramp along each line, which
    is ~15x the LER amplitude at 5 degrees.  Measured effect on the hard tier:
    +12.5% -> +20.0% gain over NCC."""

    background_pct: float = 20.0
    """Percentile used for per-sample background removal inside each line
    window; makes extraction robust to charging and vignetting."""

    min_weight_frac: float = 0.02
    min_pitch: float = 6.0
    max_pitch: float = 90.0
    min_lattice_strength: float = 0.05

    # --- Candidate re-ranking ---
    top_k: int = 150
    """Raising 40 -> 150 lifted hard-tier recall 47.5% -> 77.5%."""

    nms_distance: int = 5
    """Must stay BELOW the structural period, or suppression deletes the alias
    candidates the re-ranker exists to discriminate."""

    fp_weight: float = 1.0
    ncc_weight: float = 0.0
    """Deliberately 0.  All candidates carry near-identical NCC (~0.9), so a
    raw NCC term is a near-constant offset that never changes the ranking —
    verified inert across 0.0 / 0.3 / 1.0.  Making it useful would require
    z-scoring NCC across candidates first; that is a change, not a tune."""

    # --- Evaluation ---
    success_threshold_px: float = 5.0


FROZEN = Stage2BParams()


# ===========================================================================
# Dataset splits
# ===========================================================================

DEV_ROOT = "outputs/dataset"
DEV_SEED = 42

TEST_ROOT = "outputs/dataset_test"
TEST_SEED = 9999

TIERS = ("clean", "nominal", "hard", "ambiguous")


def describe() -> str:
    p = FROZEN
    return (f"Stage2B FROZEN: detrend_order={p.detrend_order} "
            f"top_k={p.top_k} nms={p.nms_distance} "
            f"fp_w={p.fp_weight} ncc_w={p.ncc_weight}")
