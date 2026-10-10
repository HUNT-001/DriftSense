"""
THE STAGE-2 HYPOTHESIS EXPERIMENT.

CLAIM
-----
Line-Edge Roughness is the physical signal that resolves periodic ambiguity.
Specifically, for a reference patch and its periodic replicas in the search
image:

    fingerprint_similarity(ref, TRUE location)      is HIGH
    fingerprint_similarity(ref, PERIODIC REPLICA)   is LOW

whereas raw NCC scores both locations almost identically — which is precisely
why classical template matching aliases.

EXPERIMENTAL DESIGN
-------------------
This is a controlled factorial experiment, not a demo.

  Factor A — wafer physics (3 levels):
      none      : LER off, CD off, contrast off, defects off   [control]
      ler_only  : LER ON, everything else off                  [isolates LER]
      full      : LER + CD + contrast drift + defects          [realistic]

  Factor B — electron dose / noise (3 levels): low, medium, high

Everything else is held FIXED at zero: no rotation, no scale, no drift, no
charging, no vignetting, no scan-line jitter.  Blur is held in a narrow band.
This isolates the variable of interest.  Geometry robustness is a separate
concern handled by later stages; mixing it in here would confound the result.

The `ler_only` condition is the scientifically decisive one.  The `full`
condition additionally contains CD variation and defects, which are themselves
aperiodic and would independently break ambiguity — so `full` alone could not
attribute the effect to LER.

MEASUREMENTS
------------
For each pair we score the true location and up to 24 periodic replicas at
gt + (n*period_x, m*period_y) for n,m in [-2..2]\\{(0,0)}, using BOTH:

    - LER fingerprint similarity
    - raw NCC (the classical baseline), on identical crops

and report:

    separation : mean(true) - mean(replica)
    d'         : separation / pooled std        (discriminability)
    AUC        : P(score_true > score_replica)  (ranking quality)
    top1       : fraction of pairs where true beats EVERY replica

A method that resolves ambiguity must achieve top1 ~ 1.0.  Chance is
1/(1+n_replicas) ~ 0.04.

Usage
-----
    python -m experiments.ler_hypothesis --n_pairs 20
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (TierConfig, GeometryConfig, SEMConfig,
                    AperiodicityConfig, SEARCH_H, SEARCH_W)
from data_gen.scene_composer import SceneComposer
from localization.ler_fingerprint import (extract_fingerprint,
                                           fingerprint_similarity,
                                           crop_centered)


# ---------------------------------------------------------------------------
# Experimental conditions
# ---------------------------------------------------------------------------

PHYSICS_LEVELS = {
    #  name        ler    cd     contrast  defects
    "none":       (False, False, False,    False),
    "ler_only":   (True,  False, False,    False),
    "full":       (True,  True,  True,     True),
}

NOISE_LEVELS = ("low", "medium", "high")


def make_condition(physics: str, noise: str) -> TierConfig:
    """
    Build a tier that varies ONLY the factor under test.

    All geometric distortion is disabled so that any difference in the results
    is attributable to wafer physics and dose, not to misalignment.
    """
    ler, cd, contrast, defects = PHYSICS_LEVELS[physics]
    return TierConfig(
        name=f"{physics}__{noise}",
        geometry=GeometryConfig(
            rotation_deg_range=(0.0, 0.0),
            scale_range=(1.0, 1.0),
            enable_subpixel_offset=True,
        ),
        sem=SEMConfig(
            blur_sigma_range=(0.7, 1.1),
            drift_rate_range=(0.0, 0.0),
            charging_amplitude_range=(0.0, 0.0),
            vignette_strength_range=(0.0, 0.0),
            scanline_jitter_sigma_range=(0.0, 0.0),
            read_noise_sigma_range=(0.005, 0.015),
        ),
        aperiodicity=AperiodicityConfig(
            enable_ler=ler,
            enable_cd_variation=cd,
            enable_contrast_drift=contrast,
            enable_defects=defects,
        ),
        noise_levels=(noise,),
    )


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def periodic_replicas(gt_x: float, gt_y: float,
                       period_x: float, period_y: float,
                       ref_w: int, ref_h: int,
                       n_max: int = 2) -> list[tuple[float, float]]:
    """
    Enumerate periodic replica locations around the ground truth.

    These are the locations a classical matcher confuses with the truth: the
    underlying structure there is, by construction, near-identical.
    """
    out = []
    margin_x = ref_w / 2.0 + 2
    margin_y = ref_h / 2.0 + 2
    for n in range(-n_max, n_max + 1):
        for m in range(-n_max, n_max + 1):
            if n == 0 and m == 0:
                continue
            cx = gt_x + n * period_x
            cy = gt_y + m * period_y
            if (margin_x <= cx <= SEARCH_W - margin_x and
                    margin_y <= cy <= SEARCH_H - margin_y):
                out.append((cx, cy))
    return out


def ncc_at(ref: np.ndarray, search: np.ndarray,
            cx: float, cy: float) -> float:
    """Raw NCC between the reference and the crop centred at (cx, cy)."""
    patch = crop_centered(search, cx, cy, ref.shape[1], ref.shape[0])
    if patch is None:
        return float("nan")
    a = ref.astype(np.float32) - ref.mean()
    b = patch.astype(np.float32) - patch.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float((a * b).sum() / (na * nb))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class ScoreStats:
    n_pairs: int
    mean_true: float
    mean_replica: float
    separation: float
    d_prime: float
    auc: float
    top1: float
    chance_top1: float


def summarize(true_scores: list[float],
               replica_scores: list[list[float]]) -> ScoreStats:
    t = np.array([s for s in true_scores if np.isfinite(s)], dtype=float)
    flat = np.array([s for row in replica_scores for s in row
                     if np.isfinite(s)], dtype=float)

    if t.size == 0 or flat.size == 0:
        nan = float("nan")
        return ScoreStats(0, nan, nan, nan, nan, nan, nan, nan)

    sep = float(t.mean() - flat.mean())
    pooled = float(np.sqrt(0.5 * (t.var() + flat.var()))) + 1e-12
    dprime = sep / pooled

    # AUC over all (true, replica) comparisons, ties counted as 0.5
    wins = ties = total = 0
    top1_hits = top1_total = 0
    chances = []
    for ts, rs in zip(true_scores, replica_scores):
        rs_valid = [r for r in rs if np.isfinite(r)]
        if not np.isfinite(ts) or not rs_valid:
            continue
        for r in rs_valid:
            total += 1
            if ts > r:
                wins += 1
            elif ts == r:
                ties += 1
        top1_total += 1
        top1_hits += int(ts > max(rs_valid))
        chances.append(1.0 / (1 + len(rs_valid)))

    auc = (wins + 0.5 * ties) / total if total else float("nan")
    top1 = top1_hits / top1_total if top1_total else float("nan")

    return ScoreStats(
        n_pairs=top1_total,
        mean_true=float(t.mean()),
        mean_replica=float(flat.mean()),
        separation=sep,
        d_prime=dprime,
        auc=float(auc),
        top1=float(top1),
        chance_top1=float(np.mean(chances)) if chances else float("nan"),
    )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_cell(physics: str, noise: str, n_pairs: int, seed: int,
              structure: str = "random") -> dict:
    """Run one (physics x noise) cell of the factorial design."""
    tier = make_condition(physics, noise)
    comp = SceneComposer(seed=seed)

    ler_true, ler_rep = [], []
    ncc_true, ncc_rep = [], []

    for i in range(n_pairs):
        ref, search, meta = comp.compose(i, tier, structure_type=structure)

        ref_fp = extract_fingerprint(ref)
        if ref_fp.is_empty:
            continue

        cands = periodic_replicas(meta.gt_x, meta.gt_y,
                                   meta.period_x, meta.period_y,
                                   meta.ref_w, meta.ref_h)
        if not cands:
            continue

        # --- LER fingerprint ---
        p_true = crop_centered(search, meta.gt_x, meta.gt_y,
                                meta.ref_w, meta.ref_h)
        s_true = (fingerprint_similarity(ref_fp, extract_fingerprint(p_true))
                  if p_true is not None else float("nan"))
        s_reps = []
        for (cx, cy) in cands:
            p = crop_centered(search, cx, cy, meta.ref_w, meta.ref_h)
            s_reps.append(fingerprint_similarity(ref_fp, extract_fingerprint(p))
                          if p is not None else float("nan"))
        ler_true.append(s_true)
        ler_rep.append(s_reps)

        # --- Raw NCC on identical crops ---
        ncc_true.append(ncc_at(ref, search, meta.gt_x, meta.gt_y))
        ncc_rep.append([ncc_at(ref, search, cx, cy) for (cx, cy) in cands])

    return {
        "physics": physics,
        "noise": noise,
        "ler": summarize(ler_true, ler_rep),
        "ncc": summarize(ncc_true, ncc_rep),
        "_raw": {
            "ler_true": ler_true, "ler_rep": ler_rep,
            "ncc_true": ncc_true, "ncc_rep": ncc_rep,
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(cells: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("LER FINGERPRINT — true location vs periodic replicas")
    print("=" * 100)
    hdr = (f"{'physics':>10} {'noise':>7} | {'mean_true':>9} {'mean_rep':>9} "
           f"{'sep':>7} {'d-prime':>8} {'AUC':>7} {'top-1':>7} {'chance':>7}")
    print(hdr)
    print("-" * 100)
    for c in cells:
        s = c["ler"]
        print(f"{c['physics']:>10} {c['noise']:>7} | {s.mean_true:>9.3f} "
              f"{s.mean_replica:>9.3f} {s.separation:>7.3f} "
              f"{s.d_prime:>8.2f} {s.auc:>7.1%} {s.top1:>7.1%} "
              f"{s.chance_top1:>7.1%}")

    print("\n" + "=" * 100)
    print("RAW NCC (classical baseline) — same locations, same crops")
    print("=" * 100)
    print(hdr)
    print("-" * 100)
    for c in cells:
        s = c["ncc"]
        print(f"{c['physics']:>10} {c['noise']:>7} | {s.mean_true:>9.3f} "
              f"{s.mean_replica:>9.3f} {s.separation:>7.3f} "
              f"{s.d_prime:>8.2f} {s.auc:>7.1%} {s.top1:>7.1%} "
              f"{s.chance_top1:>7.1%}")

    print("\n" + "=" * 100)
    print("HEAD-TO-HEAD  top-1 accuracy (fraction where truth beats every replica)")
    print("=" * 100)
    print(f"{'physics':>10} {'noise':>7} | {'LER':>9} {'NCC':>9} {'gain':>9}")
    print("-" * 100)
    for c in cells:
        l, n = c["ler"].top1, c["ncc"].top1
        print(f"{c['physics']:>10} {c['noise']:>7} | {l:>9.1%} {n:>9.1%} "
              f"{l - n:>+9.1%}")


def make_figure(cells: list[dict], out_dir: str) -> Path:
    """Score distributions: LER vs NCC, under LER-off vs LER-on."""
    show = [("none", "medium"), ("ler_only", "medium"), ("full", "medium")]
    picked = [next((c for c in cells
                    if c["physics"] == p and c["noise"] == n), None)
              for (p, n) in show]
    picked = [c for c in picked if c]

    fig, axes = plt.subplots(2, len(picked), figsize=(6 * len(picked), 9))
    if len(picked) == 1:
        axes = axes.reshape(2, 1)

    bins_l = np.linspace(-1, 1, 41)
    for j, c in enumerate(picked):
        raw = c["_raw"]
        for row, (key_t, key_r, label) in enumerate(
                [("ler_true", "ler_rep", "LER fingerprint"),
                 ("ncc_true", "ncc_rep", "Raw NCC")]):
            t = [s for s in raw[key_t] if np.isfinite(s)]
            r = [s for row_ in raw[key_r] for s in row_ if np.isfinite(s)]
            ax = axes[row, j]
            ax.hist(r, bins=bins_l, alpha=0.65, color="tab:red",
                    label="periodic replica", density=True)
            ax.hist(t, bins=bins_l, alpha=0.75, color="tab:green",
                    label="TRUE location", density=True)
            st = c["ler"] if row == 0 else c["ncc"]
            ax.set_title(f"{label}\nphysics={c['physics']}, "
                         f"noise={c['noise']}\n"
                         f"top-1={st.top1:.0%}  AUC={st.auc:.0%}  "
                         f"d'={st.d_prime:.2f}", fontsize=10)
            ax.set_xlabel("similarity score")
            ax.set_ylabel("density")
            ax.legend(fontsize=8)
            ax.set_xlim(-1, 1)

    fig.suptitle("LER separates the true location from periodic replicas; "
                 "raw NCC does not\n"
                 "(left: no wafer physics — the control; "
                 "middle: LER only; right: full physics)",
                 fontsize=13)
    fig.tight_layout()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "ler_hypothesis.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--structure", default="random",
                    choices=["random", "dram", "finfet"])
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--figures", default="outputs/figures")
    args = ap.parse_args()

    cells = []
    k = 0
    for physics in PHYSICS_LEVELS:
        for noise in NOISE_LEVELS:
            print(f"  running physics={physics:>9}  noise={noise:>6} ...",
                  flush=True)
            cells.append(run_cell(physics, noise, args.n_pairs,
                                   args.seed + k * 101, args.structure))
            k += 1

    print_table(cells)
    fig = make_figure(cells, args.figures)
    print(f"\nfigure -> {fig}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    payload = [{
        "physics": c["physics"], "noise": c["noise"],
        "ler": c["ler"].__dict__, "ncc": c["ncc"].__dict__,
    } for c in cells]
    with open(out / "ler_hypothesis.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"results -> {out / 'ler_hypothesis.json'}")
