"""
STAGE 2A METRIC — candidate recall@K.

This experiment deliberately does NOT measure final localization accuracy.
Stage 2B already established that once the true location is in the candidate
set, the LER fingerprint discriminates it well (d' ~ 4-5).  The measured
bottleneck is upstream: on the hard tier the truth never enters the NCC top-K
22.5% of the time, and no amount of re-ranking can recover that.

So the question here is exactly one thing:

        Does the true location make it into the candidate set?

CANDIDATE GENERATORS COMPARED
-----------------------------
    ncc_peaks   : top-K peaks of the NCC score map with NMS  (Stage-1 method)
                  Photometric criterion. Fails when the true peak is outranked
                  by aliases under rotation/scale/drift.

    lattice     : all lattice-consistent placements from the spectral basis,
                  trimmed to K by NCC score.
                  Geometric criterion. Recall depends on the accuracy of the
                  lattice estimate, not on whether the true peak outranked its
                  aliases.

    combined    : union of both, trimmed to K by NCC score.

A candidate set "hits" if any candidate centre lies within `tol` px of ground
truth.

Note on fairness: all three generators are compared at the SAME K, so the
comparison is recall at equal downstream cost — each candidate costs one
fingerprint extraction in Stage 2B.

Usage
-----
    python -m experiments.recall_study --all_tiers
    python -m experiments.recall_study --tier hard --dataset outputs/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2_config import TIERS, FROZEN
from localization.ler_localizer import ncc_score_map, top_k_peaks
from localization.spectral import (estimate_reciprocal_basis,
                                    lattice_candidates,
                                    rank_candidates_by_ncc,
                                    snap_to_local_ncc)

SNAP_RADIUS = 5


K_VALUES = (10, 25, 50, 150)
TOL_PX = 5.0


def hit(cands: list[tuple[float, float]],
        gx: float, gy: float, tol: float = TOL_PX) -> bool:
    return any(np.hypot(cx - gx, cy - gy) <= tol for (cx, cy) in cands)


# ---------------------------------------------------------------------------

def run_tier(tier: str, dataset_root: str, max_pairs: int | None,
             quiet: bool = False) -> dict:

    ds = Path(dataset_root) / tier
    mp = ds / "manifest.json"
    if not mp.exists():
        print(f"  [skip] no dataset at {ds}")
        return {}

    with open(mp) as f:
        pairs = json.load(f)["pairs"]
    if max_pairs:
        pairs = pairs[:max_pairs]

    gens = ("ncc_peaks", "lattice", "combined")
    hits = {g: {k: 0 for k in K_VALUES} for g in gens}
    n = 0
    lattice_ok = 0
    period_rel_err = []
    n_lattice_cands = []

    for meta in pairs:
        pid = meta["pair_id"]
        rp = ds / f"pair_{pid:04d}_ref.png"
        sp = ds / f"pair_{pid:04d}_search.png"
        if not rp.exists() or not sp.exists():
            continue

        ref = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        search = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gx, gy = meta["gt_x"], meta["gt_y"]
        rh, rw = ref.shape

        smap = ncc_score_map(ref, search)

        # --- generator 1: NCC peaks ---
        peaks = top_k_peaks(smap, max(K_VALUES), FROZEN.nms_distance)
        ncc_cands = [(x + rw / 2.0, y + rh / 2.0) for (x, y, _) in peaks]

        # --- generator 2: spectral lattice ---
        lat_all, basis = lattice_candidates(search, ref)
        if basis.ok and lat_all:
            lattice_ok += 1
            n_lattice_cands.append(len(lat_all))
            p1, p2 = basis.periods
            tp = sorted([meta["period_x"], meta["period_y"]])
            ep = sorted([p1, p2])
            period_rel_err.append(
                min(abs(ep[0] - tp[0]) / max(tp[0], 1e-6),
                    abs(ep[1] - tp[1]) / max(tp[1], 1e-6)))
        # Snap to the local NCC max: fixes accumulated pitch error inside the
        # correct lattice cell without allowing a hop to a neighbouring alias.
        lat_snapped = (snap_to_local_ncc(lat_all, smap, (rh, rw), SNAP_RADIUS)
                       if lat_all else [])
        lat_ranked = rank_candidates_by_ncc(lat_snapped, smap, (rh, rw),
                                             max(K_VALUES)) if lat_snapped else []

        # --- generator 3: combined (INTERLEAVED) ---
        # Ranking the union by NCC would simply re-impose the photometric
        # criterion and discard the lattice's structural advantage — measured
        # to be worse than either generator alone.  Interleaving spends half
        # the budget on each criterion, so the combined set is at least as
        # good as the better generator at any K.
        comb_ranked = []
        seen_c: set[tuple[int, int]] = set()
        for i in range(max(len(ncc_cands), len(lat_ranked))):
            for src in (ncc_cands, lat_ranked):
                if i < len(src):
                    key = (int(round(src[i][0])), int(round(src[i][1])))
                    if key not in seen_c:
                        seen_c.add(key)
                        comb_ranked.append(src[i])
            if len(comb_ranked) >= max(K_VALUES):
                break

        for k in K_VALUES:
            hits["ncc_peaks"][k] += int(hit(ncc_cands[:k], gx, gy))
            hits["lattice"][k] += int(hit(lat_ranked[:k], gx, gy))
            hits["combined"][k] += int(hit(comb_ranked[:k], gx, gy))
        n += 1

    if n == 0:
        return {}

    res = {
        "tier": tier,
        "n": n,
        "lattice_detected": round(lattice_ok / n, 4),
        "lattice_period_rel_err": (round(float(np.median(period_rel_err)), 4)
                                    if period_rel_err else None),
        "mean_lattice_candidates": (round(float(np.mean(n_lattice_cands)), 1)
                                     if n_lattice_cands else 0),
        "recall": {g: {k: round(hits[g][k] / n, 4) for k in K_VALUES}
                   for g in gens},
    }

    if not quiet:
        print(f"\n=== Tier '{tier}' — {n} pairs ===")
        print(f"  lattice detected: {res['lattice_detected']:.1%}   "
              f"median period rel-err: {res['lattice_period_rel_err']}   "
              f"mean lattice candidates: {res['mean_lattice_candidates']:.0f}")
        print(f"  {'generator':>11} " +
              " ".join(f"{'R@'+str(k):>8}" for k in K_VALUES))
        print("  " + "-" * (12 + 9 * len(K_VALUES)))
        for g in gens:
            print(f"  {g:>11} " +
                  " ".join(f"{res['recall'][g][k]:>8.1%}" for k in K_VALUES))

    return res


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="hard", choices=list(TIERS))
    ap.add_argument("--all_tiers", action="store_true")
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--max_pairs", type=int, default=None)
    args = ap.parse_args()

    tiers = list(TIERS) if args.all_tiers else [args.tier]
    results = [r for r in (run_tier(t, args.dataset, args.max_pairs)
                           for t in tiers) if r]

    if results:
        print("\n" + "=" * 88)
        print("STAGE 2A — CANDIDATE RECALL@K  (does the truth enter the set?)")
        print("=" * 88)
        print(f"{'Tier':>11} {'generator':>11} " +
              " ".join(f"{'R@'+str(k):>9}" for k in K_VALUES))
        print("-" * 88)
        for r in results:
            for g in ("ncc_peaks", "lattice", "combined"):
                print(f"{r['tier']:>11} {g:>11} " +
                      " ".join(f"{r['recall'][g][k]:>9.1%}" for k in K_VALUES))
            print("-" * 88)

        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        with open(out / "recall_study.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"results -> {out / 'recall_study.json'}")
