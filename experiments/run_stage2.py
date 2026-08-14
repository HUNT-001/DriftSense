"""
STAGE 2 END-TO-END EVALUATION — the decisive experiment.

Unlike `ler_hypothesis.py` (which compared the truth against ~24 known
periodic replicas), this evaluates the OPERATIONAL task: search the full
1000x1000 image with no prior knowledge of where the truth is.

That distinction matters.  In the 25-way comparison both raw NCC and the LER
fingerprint saturate at ~100%, so that experiment cannot separate them.  In
full search a method must beat on the order of 10^6 competing positions, and
extreme-value statistics make the required margin far larger.  Stage 1 showed
NCC dropping to 17.5% Acc@5 on the 'hard' tier under exactly that pressure.

METHODS COMPARED
----------------
    NCC          : classical template matching, argmax of the score map
    LER-2stage   : NCC top-K candidates, re-ranked by LER fingerprint

ATTRIBUTION DIAGNOSTICS
-----------------------
    recall@K     : was the truth among the K candidates at all?
                   If not, re-ranking cannot recover it and the failure
                   belongs to Stage A (candidate generation), not Stage B.
    ncc_rank     : the truth's rank in NCC order among the candidates.
                   rank 0 means NCC already had it right.
    rescued      : NCC top-1 was wrong AND the re-ranker fixed it.
    broken       : NCC top-1 was right AND the re-ranker broke it.

Reporting `rescued` and `broken` separately prevents a net improvement from
hiding a method that is simply trading one set of errors for another.

Usage
-----
    python -m experiments.run_stage2 --all_tiers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TIERS
from localization.baseline import ncc_localize
from localization.ler_localizer import localize, candidate_recall


THRESHOLDS = (1.0, 5.0, 20.0, 50.0)


def acc_at(errs, t):
    return sum(1 for e in errs if e <= t) / len(errs) if errs else 0.0


# ---------------------------------------------------------------------------

def run_tier(tier: str,
             dataset_root: str,
             results_root: str,
             top_k: int,
             max_pairs: int | None,
             quiet: bool = False,
             ncc_weight: float = 0.0,
             nms_distance: int = 5,
             write: bool = True,
             candidate_source: str = "ncc") -> dict:

    ds_dir = Path(dataset_root) / tier
    res_dir = Path(results_root) / tier
    res_dir.mkdir(parents=True, exist_ok=True)

    mpath = ds_dir / "manifest.json"
    if not mpath.exists():
        print(f"  [skip] no dataset for tier '{tier}'")
        return {}

    with open(mpath) as f:
        pairs = json.load(f)["pairs"]
    if max_pairs:
        pairs = pairs[:max_pairs]

    rows = []
    err_ncc, err_ler = [], []
    recall_hits = 0
    rescued = broken = 0
    ranks = []

    if not quiet:
        print(f"\n=== Tier '{tier}' — {len(pairs)} pairs, top_k={top_k} ===")
        print(f"{'ID':>4} {'Family':>7} {'NCC err':>9} {'LER err':>9} "
              f"{'inK':>4} {'rank':>5}  note")
        print("-" * 62)

    for meta in pairs:
        pid = meta["pair_id"]
        rp = ds_dir / f"pair_{pid:04d}_ref.png"
        sp = ds_dir / f"pair_{pid:04d}_search.png"
        if not rp.exists() or not sp.exists():
            continue

        ref = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        search = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gx, gy = meta["gt_x"], meta["gt_y"]

        base = ncc_localize(ref, search)
        e_ncc = base.error(gx, gy)

        res = localize(ref, search, top_k=top_k,
                        nms_distance=nms_distance, ncc_weight=ncc_weight,
                        candidate_source=candidate_source)
        e_ler = res.error(gx, gy)

        found, rank = candidate_recall(res.candidates, gx, gy, tol=5.0)
        recall_hits += int(found)
        if found:
            ranks.append(rank)

        note = ""
        if e_ncc > 5.0 and e_ler <= 5.0:
            rescued += 1
            note = "RESCUED"
        elif e_ncc <= 5.0 and e_ler > 5.0:
            broken += 1
            note = "broken"

        err_ncc.append(e_ncc)
        err_ler.append(e_ler)

        rows.append({
            "pair_id": pid,
            "family": meta["structure_family"],
            "noise": meta["noise_level"],
            "gt_x": gx, "gt_y": gy,
            "ncc_error": round(e_ncc, 3),
            "ler_error": round(e_ler, 3),
            "in_topk": bool(found),
            "ncc_rank_of_truth": rank,
            "n_candidates": res.n_candidates,
            "ms": round(res.runtime_ms, 1),
            "note": note,
        })

        if not quiet:
            print(f"{pid:>4} {meta['structure_family']:>7} "
                  f"{e_ncc:>9.1f} {e_ler:>9.1f} "
                  f"{'Y' if found else 'n':>4} {rank:>5}  {note}")

    n = len(rows)
    if n == 0:
        return {}

    summary = {
        "tier": tier,
        "n": n,
        "top_k": top_k,
        "recall_at_k": round(recall_hits / n, 4),
        "rescued": rescued,
        "broken": broken,
        "median_rank_of_truth": (float(np.median(ranks)) if ranks else None),
        "mean_ms": round(float(np.mean([r["ms"] for r in rows])), 1),
    }
    for name, errs in (("ncc", err_ncc), ("ler", err_ler)):
        summary[f"{name}_mean"] = round(float(np.mean(errs)), 2)
        summary[f"{name}_median"] = round(float(np.median(errs)), 2)
        for t in THRESHOLDS:
            summary[f"{name}_acc@{t:g}"] = round(acc_at(errs, t), 4)

    if not quiet:
        print("-" * 62)
        print(f"  recall@{top_k}          : {summary['recall_at_k']:.1%}")
        print(f"  median rank of truth : {summary['median_rank_of_truth']}")
        print(f"  rescued / broken     : {rescued} / {broken}")
        print(f"  {'method':>10} {'mean':>9} {'median':>9} "
              f"{'Acc@1':>7} {'Acc@5':>7} {'Acc@20':>7}")
        for name, label in (("ncc", "NCC"), ("ler", "LER-2stage")):
            print(f"  {label:>10} {summary[f'{name}_mean']:>9.1f} "
                  f"{summary[f'{name}_median']:>9.1f} "
                  f"{summary[f'{name}_acc@1']:>7.1%} "
                  f"{summary[f'{name}_acc@5']:>7.1%} "
                  f"{summary[f'{name}_acc@20']:>7.1%}")

    if write:
        with open(res_dir / "stage2_metrics.json", "w") as f:
            json.dump({"summary": summary, "pairs": rows}, f, indent=2)

    return summary


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="hard", choices=list(TIERS))
    ap.add_argument("--all_tiers", action="store_true")
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--max_pairs", type=int, default=None)
    args = ap.parse_args()

    tiers = list(TIERS) if args.all_tiers else [args.tier]
    summaries = [s for s in
                 (run_tier(t, args.dataset, args.out, args.top_k,
                            args.max_pairs) for t in tiers) if s]

    if summaries:
        print("\n" + "=" * 96)
        print("STAGE 2 vs BASELINE — full 1000x1000 search")
        print("=" * 96)
        print(f"{'Tier':>11} {'recall@K':>9} {'NCC Acc@5':>10} "
              f"{'LER Acc@5':>10} {'gain':>8} {'resc':>5} {'brok':>5} "
              f"{'NCC med':>9} {'LER med':>9}")
        print("-" * 96)
        for s in summaries:
            gain = s["ler_acc@5"] - s["ncc_acc@5"]
            print(f"{s['tier']:>11} {s['recall_at_k']:>9.1%} "
                  f"{s['ncc_acc@5']:>10.1%} {s['ler_acc@5']:>10.1%} "
                  f"{gain:>+8.1%} {s['rescued']:>5} {s['broken']:>5} "
                  f"{s['ncc_median']:>9.1f} {s['ler_median']:>9.1f}")

        with open(Path(args.out) / "stage2_summary.json", "w") as f:
            json.dump(summaries, f, indent=2)
