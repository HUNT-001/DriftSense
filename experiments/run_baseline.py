"""
Baseline evaluation across difficulty tiers — v2.

Usage
-----
    python -m experiments.run_baseline --tier clean
    python -m experiments.run_baseline --all_tiers

Outputs (outputs/results/<tier>/)
----------------------------------
    baseline_metrics.json  — per-pair predictions and errors
    baseline_summary.txt   — aggregate accuracy table, split by structure

Metric definition
-----------------
Error is the Euclidean distance in pixels between the predicted centre and
the ground-truth centre.  We report Acc@k = fraction of pairs with error <= k
for k in {1, 5, 20, 50}.  Acc@1 is the sub-pixel-grade metric that matters for
overlay/navigation recovery; Acc@50 measures "found roughly the right cell".
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
from localization.baseline import localize_all


THRESHOLDS = (1.0, 5.0, 20.0, 50.0)


def acc_at(errors, t):
    return sum(1 for e in errors if e <= t) / len(errors) if errors else 0.0


# ---------------------------------------------------------------------------

def run_tier(tier: str,
             dataset_root: str = "outputs/dataset",
             results_root: str = "outputs/results",
             max_pairs: int | None = None,
             quiet: bool = False) -> dict:

    ds_dir = Path(dataset_root) / tier
    res_dir = Path(results_root) / tier
    res_dir.mkdir(parents=True, exist_ok=True)

    mpath = ds_dir / "manifest.json"
    if not mpath.exists():
        print(f"  [skip] no dataset for tier '{tier}' at {ds_dir}")
        return {}

    with open(mpath) as f:
        manifest = json.load(f)
    pairs = manifest["pairs"][:max_pairs] if max_pairs else manifest["pairs"]

    methods = ["NCC", "PhaseCC", "MultiNCC"]
    rows = []
    errs = {m: [] for m in methods}
    errs_by_family = {m: {"dram": [], "finfet": []} for m in methods}

    if not quiet:
        print(f"\n=== Tier '{tier}' — {len(pairs)} pairs ===")
        print(f"{'ID':>4} {'Family':>7} {'Noise':>7} "
              f"{'NCC':>9} {'PhaseCC':>9} {'MultiNCC':>9}")
        print("-" * 52)

    for meta in pairs:
        pid = meta["pair_id"]
        ref_p = ds_dir / f"pair_{pid:04d}_ref.png"
        sea_p = ds_dir / f"pair_{pid:04d}_search.png"
        if not ref_p.exists() or not sea_p.exists():
            continue

        ref = cv2.imread(str(ref_p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        search = cv2.imread(str(sea_p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        results = localize_all(ref, search)

        row = {k: meta[k] for k in
               ("pair_id", "structure_family", "structure_variant",
                "noise_level", "gt_x", "gt_y", "period_x", "period_y",
                "rotation_deg", "scale")}

        for m, r in results.items():
            e = r.error(meta["gt_x"], meta["gt_y"])
            errs[m].append(e)
            errs_by_family[m][meta["structure_family"]].append(e)
            row[f"{m}_pred_x"] = round(r.pred_x, 3)
            row[f"{m}_pred_y"] = round(r.pred_y, 3)
            row[f"{m}_error"]  = round(e, 3)
            row[f"{m}_score"]  = round(r.score, 4)
            row[f"{m}_ms"]     = round(r.runtime_ms, 1)
        rows.append(row)

        if not quiet:
            print(f"{pid:>4} {meta['structure_family']:>7} "
                  f"{meta['noise_level']:>7} "
                  f"{row['NCC_error']:>9.1f} {row['PhaseCC_error']:>9.1f} "
                  f"{row['MultiNCC_error']:>9.1f}")

    # ---- aggregate ----
    summary = []
    for m in methods:
        e = errs[m]
        if not e:
            continue
        entry = {
            "method": m,
            "n": len(e),
            "mean_err": round(float(np.mean(e)), 3),
            "median_err": round(float(np.median(e)), 3),
            "mean_ms": round(float(np.mean([r[f"{m}_ms"] for r in rows])), 1),
        }
        for t in THRESHOLDS:
            entry[f"acc@{t:g}"] = round(acc_at(e, t), 4)
        for fam in ("dram", "finfet"):
            fe = errs_by_family[m][fam]
            entry[f"median_err_{fam}"] = (round(float(np.median(fe)), 3)
                                           if fe else None)
        summary.append(entry)

    if not quiet:
        print("\n" + "-" * 78)
        print(f"{'Method':>10} {'Mean':>9} {'Median':>9} "
              f"{'Acc@1':>7} {'Acc@5':>7} {'Acc@20':>7} {'Acc@50':>7} {'ms':>7}")
        print("-" * 78)
        for s in summary:
            print(f"{s['method']:>10} {s['mean_err']:>9.1f} "
                  f"{s['median_err']:>9.1f} "
                  f"{s['acc@1']:>7.1%} {s['acc@5']:>7.1%} "
                  f"{s['acc@20']:>7.1%} {s['acc@50']:>7.1%} {s['mean_ms']:>7.0f}")

    with open(res_dir / "baseline_metrics.json", "w") as f:
        json.dump({"tier": tier, "pairs": rows, "summary": summary}, f, indent=2)

    with open(res_dir / "baseline_summary.txt", "w") as f:
        f.write(f"DriftSense baseline — tier '{tier}'\n")
        f.write("=" * 78 + "\n")
        f.write(f"pairs: {len(rows)}\n\n")
        f.write(f"{'Method':>10} {'Mean':>9} {'Median':>9} "
                f"{'Acc@1':>7} {'Acc@5':>7} {'Acc@20':>7} {'Acc@50':>7}\n")
        f.write("-" * 78 + "\n")
        for s in summary:
            f.write(f"{s['method']:>10} {s['mean_err']:>9.1f} "
                    f"{s['median_err']:>9.1f} {s['acc@1']:>7.1%} "
                    f"{s['acc@5']:>7.1%} {s['acc@20']:>7.1%} "
                    f"{s['acc@50']:>7.1%}\n")
        f.write("\nMedian error by structure family:\n")
        for s in summary:
            f.write(f"  {s['method']:>10}  DRAM={s['median_err_dram']}  "
                    f"FinFET={s['median_err_finfet']}\n")

    return {"tier": tier, "summary": summary, "n": len(rows)}


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="clean", choices=list(TIERS))
    ap.add_argument("--all_tiers", action="store_true")
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--out", default="outputs/results")
    ap.add_argument("--max_pairs", type=int, default=None)
    args = ap.parse_args()

    tiers = list(TIERS) if args.all_tiers else [args.tier]
    all_res = [run_tier(t, args.dataset, args.out, args.max_pairs)
               for t in tiers]
    all_res = [r for r in all_res if r]

    if len(all_res) > 1:
        print("\n" + "=" * 78)
        print("CROSS-TIER COMPARISON  (median error px / Acc@5)")
        print("=" * 78)
        print(f"{'Tier':>12} {'NCC':>18} {'PhaseCC':>18} {'MultiNCC':>18}")
        print("-" * 78)
        for r in all_res:
            cells = []
            for m in ("NCC", "PhaseCC", "MultiNCC"):
                s = next((x for x in r["summary"] if x["method"] == m), None)
                cells.append(f"{s['median_err']:>8.1f} /{s['acc@5']:>7.1%}"
                             if s else " " * 18)
            print(f"{r['tier']:>12} " + " ".join(cells))
