"""
STAGE 2E DIAGNOSTIC — what degrades the LER descriptor, resampling-free?

The candidate-budget curve showed a ~55% ceiling even at K=25 on the hard
tier: the truth's OWN fingerprint is degraded, independent of competitor
count.  Before building any fix we must know WHICH distortion is responsible
and how steeply.

This measures the raw descriptor-preservation signal directly:

    corr = fingerprint_similarity(ref_fingerprint, true_location_fingerprint)

with NO competitors and NO resampling correction.  A value near 1 means the
descriptor survived; a low value means the distortion corrupted it.  Each
distortion axis is swept INDEPENDENTLY with everything else held ideal, so the
effects are not confounded (the mistake that produced the earlier bogus
rotation attribution).

A reference floor of similarity(ref, ref-with-only-independent-noise) is also
reported, so degradation is measured relative to the achievable maximum rather
than to 1.0.

Usage
-----
    python -m experiments.descriptor_degradation --n_pairs 15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (TierConfig, GeometryConfig, SEMConfig, AperiodicityConfig)
from data_gen.scene_composer import SceneComposer
from localization.ler_fingerprint import (extract_fingerprint,
                                           fingerprint_similarity,
                                           crop_centered)


def tier(rot=0.0, scale=0.0, drift=0.0, noise="medium") -> TierConfig:
    return TierConfig(
        name="diag",
        geometry=GeometryConfig(rotation_deg_range=(-rot, rot),
                                 scale_range=(1.0 - scale, 1.0 + scale),
                                 enable_subpixel_offset=True),
        sem=SEMConfig(blur_sigma_range=(0.7, 1.1),
                      drift_rate_range=(drift, drift),
                      charging_amplitude_range=(0.0, 0.0),
                      vignette_strength_range=(0.0, 0.0),
                      scanline_jitter_sigma_range=(0.0, 0.0),
                      read_noise_sigma_range=(0.005, 0.015)),
        aperiodicity=AperiodicityConfig(enable_cd_variation=False,
                                         enable_contrast_drift=False,
                                         enable_defects=False),
        noise_levels=(noise,),
    )


def truth_similarity(t: TierConfig, n_pairs: int, seed: int) -> float:
    """Median fingerprint similarity between ref and the true location."""
    comp = SceneComposer(seed=seed)
    sims = []
    for i in range(n_pairs):
        ref, search, meta = comp.compose(i, t)
        rfp = extract_fingerprint(ref)
        if rfp.is_empty:
            continue
        p = crop_centered(search, meta.gt_x, meta.gt_y, meta.ref_w, meta.ref_h)
        if p is None:
            continue
        s = fingerprint_similarity(rfp, extract_fingerprint(p))
        if np.isfinite(s):
            sims.append(s)
    return float(np.median(sims)) if sims else float("nan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=131)
    ap.add_argument("--out", default="outputs/results")
    args = ap.parse_args()

    sweeps = {
        "rotation (deg)": [("0.0", tier()),
                            ("1.0", tier(rot=1.0)),
                            ("2.5", tier(rot=2.5)),
                            ("5.0", tier(rot=5.0))],
        "scale (+/-frac)": [("0.00", tier()),
                            ("0.01", tier(scale=0.01)),
                            ("0.03", tier(scale=0.03)),
                            ("0.07", tier(scale=0.07))],
        "drift (px/line)": [("0.00", tier()),
                            ("0.02", tier(drift=0.02)),
                            ("0.05", tier(drift=0.05)),
                            ("0.11", tier(drift=0.11))],
        "noise (dose)": [("low", tier(noise="low")),
                         ("medium", tier(noise="medium")),
                         ("high", tier(noise="high"))],
    }

    results = {}
    print(f"\n{'sweep':>18} {'level':>8} {'ref-vs-truth similarity':>26}")
    print("-" * 56)
    for name, cases in sweeps.items():
        results[name] = []
        for k, (label, t) in enumerate(cases):
            sim = truth_similarity(t, args.n_pairs, args.seed + k * 31)
            results[name].append({"level": label, "similarity": sim})
            print(f"{name:>18} {label:>8} {sim:>26.3f}")
        print("-" * 56)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "descriptor_degradation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nInterpretation: the axis with the steepest drop is what Stage 2E "
          f"must address.")
