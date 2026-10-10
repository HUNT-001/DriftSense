"""Measure orientation-estimator error against ground-truth rotation_deg.

Compares estimators on their angular error (median / P90 / outlier rate),
which is the quantity that determines de-warp quality and hence Stage-2B
discrimination.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localization.spectral import (estimate_orientation_consensus,
                                    estimate_reciprocal_basis, principal_angle,
                                    reduce_lattice_angle)
try:
    from localization.lattice_fit import estimate_orientation_ransac
    HAVE_RANSAC = True
except ImportError:
    HAVE_RANSAC = False


def rel(a):  # reduce to (-45,45]
    return reduce_lattice_angle(a)


def run(tier, ds_root, max_pairs):
    ds = Path(ds_root) / tier
    pairs = json.load(open(ds / "manifest.json"))["pairs"]
    if max_pairs:
        pairs = pairs[:max_pairs]
    ests = ["consensus", "single-peak"]
    if HAVE_RANSAC:
        ests.append("ransac")
    errs = {e: [] for e in ests}
    for meta in pairs:
        pid = meta["pair_id"]
        rp = ds / f"pair_{pid:04d}_ref.png"; sp = ds / f"pair_{pid:04d}_search.png"
        if not rp.exists():
            continue
        ref = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        search = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        gt = rel(meta["rotation_deg"])   # relative search-vs-ref rotation

        # consensus (relative = search - ref)
        ts, _ = estimate_orientation_consensus(search)
        tr, _ = estimate_orientation_consensus(ref)
        errs["consensus"].append(abs(rel(rel(ts - tr) - gt)))

        # single dominant peak (relative)
        bs, br = estimate_reciprocal_basis(search), estimate_reciprocal_basis(ref)
        if bs.ok and br.ok:
            errs["single-peak"].append(
                abs(rel(rel(principal_angle(bs) - principal_angle(br)) - gt)))

        if HAVE_RANSAC:
            rs, _ = estimate_orientation_ransac(search)
            rr, _ = estimate_orientation_ransac(ref)
            errs["ransac"].append(abs(rel(rel(rs - rr) - gt)))

    print(f"\n=== orientation error, tier '{tier}' ({len(pairs)} pairs) ===")
    print(f"{'estimator':>14} {'median':>8} {'P90':>8} {'max':>8} "
          f"{'>1.5deg':>8}")
    print("-" * 52)
    for e in ests:
        v = np.array(errs[e])
        if len(v) == 0:
            continue
        print(f"{e:>14} {np.median(v):>8.2f} {np.percentile(v,90):>8.2f} "
              f"{v.max():>8.2f} {(v>1.5).mean():>8.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="hard")
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--max_pairs", type=int, default=None)
    a = ap.parse_args()
    run(a.tier, a.dataset, a.max_pairs)
