"""
STAGE 2C — rotation compensation.

Two experiments, deliberately kept separate.

EXPERIMENT 2 (run first): what is the orientation ERROR BUDGET, and can we meet it?
-----------------------------------------------------------------------------------
Linear detrending already removes the rotation-induced ramp along each line.
The residual failure mode is different: a tilted line DRIFTS OUT of the
half-pitch centroid window used to measure its displacement.

    ramp across patch = H * tan(theta);   line escapes when ramp > half_pitch

    DRAM   (pitch 24): escapes at 6.8 deg -> residual budget 1.72 deg
    FinFET (pitch 11): escapes at 3.1 deg -> residual budget 0.79 deg

The FinFET break at 3.1 deg matches the observed degradation onset (~2.5 deg),
so the model is trusted.  Stage 2C must therefore estimate relative rotation
to better than ~0.79 deg.

EXPERIMENT 1: does de-rotation remove the rotation penalty?
-----------------------------------------------------------
Discrimination (truth vs periodic replicas) bucketed by applied rotation,
with and without de-rotation.  Target: 2.5-5 deg discrimination well above
the 27% currently measured.

DESIGN CONSTRAINT
-----------------
Rotation is estimated ONCE PER PAIR and the identical correction is applied to
every candidate.  Rotation is a property of the acquisition pair, not of a
candidate, so a uniform correction cannot preferentially favour one periodic
replica over another.  Fitting rotation per-candidate against the fingerprint
would risk the same re-alignment leakage already guarded against in
`ler_fingerprint.compare_axis`.

Usage
-----
    python -m experiments.rotation_study --n_pairs 12
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
from localization.spectral import (estimate_reciprocal_basis,
                                    principal_angle_robust,
                                    principal_angle,
                                    reduce_lattice_angle,
                                    estimate_relative_rotation,
                                    derotate_roi)
from experiments.ler_hypothesis import periodic_replicas


BUDGET_DEG = 0.79


def make_tier(rot: float, noise: str = "medium") -> TierConfig:
    """Fixed-magnitude rotation, everything else ideal."""
    return TierConfig(
        name=f"rot{rot}",
        geometry=GeometryConfig(rotation_deg_range=(-rot, rot),
                                 scale_range=(1.0, 1.0),
                                 enable_subpixel_offset=True),
        sem=SEMConfig(blur_sigma_range=(0.7, 1.1),
                      drift_rate_range=(0.0, 0.0),
                      charging_amplitude_range=(0.0, 0.0),
                      vignette_strength_range=(0.0, 0.0),
                      scanline_jitter_sigma_range=(0.0, 0.0),
                      read_noise_sigma_range=(0.005, 0.015)),
        aperiodicity=AperiodicityConfig(enable_cd_variation=False,
                                         enable_contrast_drift=False,
                                         enable_defects=False),
        noise_levels=(noise,),
    )


# ---------------------------------------------------------------------------
# Orientation estimators
# ---------------------------------------------------------------------------

def theta_search_only(ref, search, basis):
    """
    Orientation from the SEARCH image alone, assuming the reference sits at the
    nominal (un-rotated) tool orientation.

    HONEST CAVEAT: in this generator the reference is cropped from the
    UNWARPED scene, so `theta_ref == 0` exactly.  That makes this estimator
    partly a generator artifact and it must NOT be quoted as a general result.
    It is reported because it upper-bounds what perfect reference-orientation
    knowledge would buy, which isolates how much of the remaining error comes
    from the reference side.
    """
    a, ok = principal_angle_robust(basis)
    return a, ok


def theta_ref_vs_search(ref, search, basis):
    """General estimator: both orientations measured. No assumption."""
    return estimate_relative_rotation(ref, search, basis)


ESTIMATORS = {
    "search_only(assumes ref@0)": theta_search_only,
    "ref_vs_search(general)": theta_ref_vs_search,
}


# ---------------------------------------------------------------------------

def run(n_pairs: int, seed: int, rotations: list[float]) -> dict:
    out = {"orientation": [], "discrimination": []}

    for rot in rotations:
        tier = make_tier(rot)
        comp = SceneComposer(seed=seed + int(rot * 100))

        errs = {k: [] for k in ESTIMATORS}
        cover = {k: 0 for k in ESTIMATORS}
        n_ok = 0

        top1_plain, top1_derot = [], []

        for i in range(n_pairs):
            ref, search, meta = comp.compose(i, tier)
            basis = estimate_reciprocal_basis(search)
            if not basis.ok:
                continue
            n_ok += 1
            true_rot = meta.rotation_deg

            # ---------- Experiment 2: orientation error ----------
            thetas = {}
            for name, fn in ESTIMATORS.items():
                th, ok = fn(ref, search, basis)
                thetas[name] = (th, ok)
                if ok:
                    cover[name] += 1
                    errs[name].append(abs(reduce_lattice_angle(th - true_rot)))

            # ---------- Experiment 1: discrimination ----------
            cands = periodic_replicas(meta.gt_x, meta.gt_y,
                                       meta.period_x, meta.period_y,
                                       meta.ref_w, meta.ref_h)
            if not cands:
                continue
            ref_fp = extract_fingerprint(ref)
            if ref_fp.is_empty:
                continue

            # (a) no compensation
            def score_plain(cx, cy):
                p = crop_centered(search, cx, cy, meta.ref_w, meta.ref_h)
                return (fingerprint_similarity(ref_fp, extract_fingerprint(p))
                        if p is not None else float("nan"))

            st = score_plain(meta.gt_x, meta.gt_y)
            sr = [score_plain(cx, cy) for cx, cy in cands]
            sr = [s for s in sr if np.isfinite(s)]
            if np.isfinite(st) and sr:
                top1_plain.append(int(st > max(sr)))

            # (b) de-rotated by the GENERAL estimator, one angle for all
            th, ok = thetas["ref_vs_search(general)"]
            use_th = th if ok else 0.0

            def score_derot(cx, cy):
                p = derotate_roi(search, cx, cy, meta.ref_w, meta.ref_h,
                                  -use_th)
                return (fingerprint_similarity(ref_fp, extract_fingerprint(p))
                        if p is not None else float("nan"))

            st2 = score_derot(meta.gt_x, meta.gt_y)
            sr2 = [score_derot(cx, cy) for cx, cy in cands]
            sr2 = [s for s in sr2 if np.isfinite(s)]
            if np.isfinite(st2) and sr2:
                top1_derot.append(int(st2 > max(sr2)))

        for name in ESTIMATORS:
            v = errs[name]
            out["orientation"].append({
                "rotation": rot, "estimator": name, "n": n_ok,
                "coverage": (cover[name] / n_ok) if n_ok else 0.0,
                "median_err": float(np.median(v)) if v else None,
                "p95_err": float(np.percentile(v, 95)) if v else None,
                "within_budget": (sum(1 for x in v if x < BUDGET_DEG) / len(v))
                                  if v else None,
            })
        out["discrimination"].append({
            "rotation": rot,
            "n": len(top1_plain),
            "top1_plain": float(np.mean(top1_plain)) if top1_plain else None,
            "top1_derot": float(np.mean(top1_derot)) if top1_derot else None,
        })

    return out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--out", default="outputs/results")
    args = ap.parse_args()

    rotations = [0.5, 2.0, 4.0]
    res = run(args.n_pairs, args.seed, rotations)

    print("\n" + "=" * 84)
    print(f"EXPERIMENT 2 — orientation error   (budget {BUDGET_DEG} deg)")
    print("=" * 84)
    print(f"{'rot':>5} {'estimator':>28} {'cover':>7} {'median':>8} "
          f"{'P95':>8} {'<budget':>8}")
    print("-" * 84)
    for r in res["orientation"]:
        med = "n/a" if r["median_err"] is None else f"{r['median_err']:.2f}d"
        p95 = "n/a" if r["p95_err"] is None else f"{r['p95_err']:.2f}d"
        wb = "n/a" if r["within_budget"] is None else f"{r['within_budget']:.0%}"
        print(f"{r['rotation']:>5.1f} {r['estimator']:>28} "
              f"{r['coverage']:>7.0%} {med:>8} {p95:>8} {wb:>8}")

    print("\n" + "=" * 84)
    print("EXPERIMENT 1 — discrimination (truth vs periodic replicas)")
    print("=" * 84)
    print(f"{'rotation':>9} {'n':>4} {'no comp':>10} {'de-rotated':>12} {'gain':>8}")
    print("-" * 84)
    for d in res["discrimination"]:
        a = d["top1_plain"]; b = d["top1_derot"]
        if a is None or b is None:
            continue
        print(f"{d['rotation']:>9.1f} {d['n']:>4} {a:>10.0%} {b:>12.0%} "
              f"{b - a:>+8.0%}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "rotation_study.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nresults -> {out / 'rotation_study.json'}")
