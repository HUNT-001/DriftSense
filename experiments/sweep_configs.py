"""Config sweep on a tier: compares candidate sources and de-rotation.

Measures recall@K (Stage-A quality) and Acc@5 (end-to-end) so we can see
whether the bottleneck is candidate generation or discrimination.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localization.baseline import ncc_localize
from localization.ler_localizer import localize, candidate_recall


def acc_at(errs, t):
    return sum(1 for e in errs if e <= t) / len(errs) if errs else 0.0


def load_pairs(ds_dir, max_pairs):
    pairs = json.load(open(ds_dir / "manifest.json"))["pairs"]
    return pairs[:max_pairs] if max_pairs else pairs


def run(tier, ds_root, max_pairs, top_k):
    ds = Path(ds_root) / tier
    pairs = load_pairs(ds, max_pairs)
    configs = {
        "ncc":               dict(candidate_source="ncc"),
        "lattice":           dict(candidate_source="lattice"),
        "lattice+derotate":  dict(candidate_source="lattice", derotate=True),
        "lattice+multiwarp": dict(candidate_source="lattice", derotate=True,
                                  orientation="multi"),
    }
    agg = {name: {"err": [], "recall": 0, "ms": []} for name in configs}
    for meta in pairs:
        pid = meta["pair_id"]
        rp = ds / f"pair_{pid:04d}_ref.png"; sp = ds / f"pair_{pid:04d}_search.png"
        if not rp.exists() or not sp.exists():
            continue
        ref = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        search = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gx, gy = meta["gt_x"], meta["gt_y"]
        for name, kw in configs.items():
            t0 = time.perf_counter()
            res = localize(ref, search, top_k=top_k, nms_distance=5,
                           ncc_weight=0.0, **kw)
            agg[name]["ms"].append((time.perf_counter() - t0) * 1000)
            agg[name]["err"].append(res.error(gx, gy))
            found, _ = candidate_recall(res.candidates, gx, gy, tol=5.0)
            agg[name]["recall"] += int(found)
    n = len(agg["ncc"]["err"])
    print(f"\n=== tier '{tier}'  ({n} pairs, top_k={top_k}) ===")
    print(f"{'config':>18} {'recall@K':>9} {'Acc@1':>7} {'Acc@5':>7} "
          f"{'Acc@20':>7} {'median':>8} {'ms':>7}")
    print("-" * 70)
    rows = {}
    for name, a in agg.items():
        errs = a["err"]
        row = dict(recall=a["recall"]/n, acc1=acc_at(errs,1), acc5=acc_at(errs,5),
                   acc20=acc_at(errs,20), median=float(np.median(errs)),
                   ms=float(np.mean(a["ms"])))
        rows[name] = row
        print(f"{name:>18} {row['recall']:>9.1%} {row['acc1']:>7.1%} "
              f"{row['acc5']:>7.1%} {row['acc20']:>7.1%} {row['median']:>8.1f} "
              f"{row['ms']:>7.0f}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="hard")
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--max_pairs", type=int, default=None)
    ap.add_argument("--top_k", type=int, default=60)
    a = ap.parse_args()
    run(a.tier, a.dataset, a.max_pairs, a.top_k)
