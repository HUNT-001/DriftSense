"""
CANDIDATE-BUDGET CURVE — how does LER accuracy scale with competitor count?

THE QUESTION
------------
Stage 2C established that the hard-tier bottleneck is NOT geometric distortion
(rotation/scale/drift give 83% even combined) but the SIZE of the candidate
set: 24 replicas -> 83%, ~1500 lattice candidates -> 40%.

This experiment quantifies that scaling law directly, isolating set size from
every other factor.

CLEAN ISOLATION VIA THE HYPERGEOMETRIC TRICK
--------------------------------------------
For each pair we LER-score the ENTIRE candidate set exactly once.  Let the
truth's score be s_t, and let `n` be the number of competitor candidates, of
which `n_below` score strictly below s_t.

If we were to keep the truth plus (K-1) competitors drawn uniformly at random,
the truth wins (is top-1) iff none of the sampled competitors outscore it:

    P(top-1 | K) = C(n_below, K-1) / C(n, K-1)

This is exact and needs no re-sampling.  It measures the pure effect of
COMPETITOR COUNT, holding the descriptor and every candidate's score fixed —
so it cannot be confounded by prefilter quality (a real prefilter would change
*which* competitors remain; here they are random).

A real structural prefilter can only do BETTER than this random baseline, by
preferentially removing high-scoring competitors.  So this curve is a lower
bound on what a prefilter buys, and its shape answers the design question:
"is it worth building a prefilter at all, and to what target K?"

Only pairs whose truth is actually in the candidate set are used (recall is a
separate, already-solved concern).

Usage
-----
    python -m experiments.candidate_budget --tier hard --n_pairs 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TIERS
from localization.ler_localizer import ncc_score_map
from localization.ler_fingerprint import (extract_fingerprint,
                                           fingerprint_similarity,
                                           crop_centered)
from localization.spectral import lattice_candidates, snap_to_local_ncc


K_GRID = (25, 50, 100, 150, 200, 300, 500, 1000, 1500)
TOL_PX = 5.0


def p_top1_given_k(n_below: int, n_total: int, k: int) -> float:
    """
    P(truth beats K-1 uniformly-random competitors) = C(n_below,K-1)/C(n,K-1).

    k is the total kept set size (truth + K-1 competitors).  If k-1 exceeds the
    number of competitors, the whole set is kept and the result is 1 iff the
    truth outscores all competitors (n_below == n_total).
    """
    m = k - 1
    if m <= 0:
        return 1.0
    if m >= n_total:
        return 1.0 if n_below == n_total else 0.0
    # log-comb for numerical safety
    lc = (math.lgamma(n_below + 1) - math.lgamma(m + 1) - math.lgamma(n_below - m + 1)) \
        if n_below >= m else None
    if lc is None:
        return 0.0
    ld = math.lgamma(n_total + 1) - math.lgamma(m + 1) - math.lgamma(n_total - m + 1)
    return float(math.exp(lc - ld))


def run(tier: str, dataset_root: str, n_pairs: int,
        max_candidates: int) -> dict:
    ds = Path(dataset_root) / tier
    pairs = json.load(open(ds / "manifest.json"))["pairs"][:n_pairs]

    per_pair = []
    used = 0
    for meta in pairs:
        pid = meta["pair_id"]
        rp = ds / f"pair_{pid:04d}_ref.png"
        sp = ds / f"pair_{pid:04d}_search.png"
        if not rp.exists():
            continue
        ref = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        search = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        rh, rw = ref.shape
        gx, gy = meta["gt_x"], meta["gt_y"]

        smap = ncc_score_map(ref, search)
        lat, basis = lattice_candidates(search, ref, max_candidates=max_candidates)
        if not lat:
            continue
        lat = snap_to_local_ncc(lat, smap, (rh, rw), radius=5)

        # Identify the truth candidate (nearest to GT); require it present
        d = [np.hypot(cx - gx, cy - gy) for (cx, cy) in lat]
        j = int(np.argmin(d))
        if d[j] > TOL_PX:
            continue  # recall miss — out of scope here

        ref_fp = extract_fingerprint(ref)
        if ref_fp.is_empty:
            continue

        scores = np.full(len(lat), -2.0, dtype=np.float64)
        for i, (cx, cy) in enumerate(lat):
            p = crop_centered(search, cx, cy, rw, rh)
            if p is None:
                continue
            s = fingerprint_similarity(ref_fp, extract_fingerprint(p))
            scores[i] = s if np.isfinite(s) else -2.0

        s_t = scores[j]
        comp = np.delete(scores, j)
        n_total = comp.size
        n_below = int((comp < s_t).sum())
        per_pair.append((n_below, n_total))
        used += 1

    # Aggregate expected accuracy at each K
    curve = {}
    for k in K_GRID:
        vals = [p_top1_given_k(nb, nt, k) for (nb, nt) in per_pair]
        curve[k] = float(np.mean(vals)) if vals else None

    return {
        "tier": tier,
        "n_pairs_used": used,
        "mean_candidates": float(np.mean([nt + 1 for (_, nt) in per_pair]))
                            if per_pair else 0.0,
        "curve": curve,
    }


def make_figure(results: list[dict], out_dir: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results:
        ks = [k for k in K_GRID if r["curve"][k] is not None]
        ys = [r["curve"][k] for k in ks]
        ax.plot(ks, ys, "o-", lw=2,
                label=f"{r['tier']} (n={r['n_pairs_used']}, "
                      f"~{r['mean_candidates']:.0f} candidates)")
    ax.set_xscale("log")
    ax.set_xlabel("candidate budget K (truth + K-1 random competitors)")
    ax.set_ylabel("expected LER top-1 accuracy")
    ax.set_title("Candidate-budget curve: LER accuracy vs competitor count\n"
                 "(descriptor and scores fixed; only the number of competitors "
                 "varies)")
    ax.axhline(0.95, color="gray", ls="--", alpha=0.6, label="95% target")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.02)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "candidate_budget.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="hard", choices=list(TIERS))
    ap.add_argument("--tiers", nargs="*", default=None)
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--max_candidates", type=int, default=2000)
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--out", default="outputs/results")
    args = ap.parse_args()

    tiers = args.tiers if args.tiers else [args.tier]
    results = []
    for t in tiers:
        print(f"  scoring tier '{t}' ...", flush=True)
        results.append(run(t, args.dataset, args.n_pairs, args.max_candidates))

    print("\n" + "=" * 78)
    print("CANDIDATE-BUDGET CURVE — expected LER top-1 vs kept set size K")
    print("=" * 78)
    hdr = f"{'tier':>10} {'npairs':>6} {'cands':>6} | " + \
          " ".join(f"{k:>6}" for k in K_GRID)
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        row = " ".join(("  n/a" if r["curve"][k] is None else f"{r['curve'][k]:>5.0%}")
                       for k in K_GRID)
        print(f"{r['tier']:>10} {r['n_pairs_used']:>6} "
              f"{r['mean_candidates']:>6.0f} | {row}")

    fig = make_figure(results, args.out)
    print(f"\nfigure -> {fig}")
    with open(Path(args.out) / "candidate_budget.json", "w") as f:
        json.dump(results, f, indent=2)
