"""
Periodicity failure analysis — the core evidence for the Stage-2 design.

THE HYPOTHESIS
--------------
Classical template matching on periodic semiconductor structures does not fail
randomly.  It fails by locking onto the WRONG PERIOD — an alias of the true
location.  If that is true, then the localization error vector should be very
close to an INTEGER MULTIPLE of the structural period:

        error_x  ~=  n * period_x        for integer n
        error_y  ~=  m * period_y        for integer m

If instead the errors were caused by noise or blur, the residual
`error mod period` would be uniformly distributed.

WHAT THIS SCRIPT PRODUCES
-------------------------
  1. Period estimated independently from the reference image via FFT, and
     validated against the manifest's ground-truth period.
  2. The "alias residual": distance from each error to the nearest lattice
     point n*period.  Small residual => alias failure.
  3. The fraction of failures classified as alias failures.
  4. Figures: error-vs-period scatter, residual histogram, alias lattice plot.

Usage
-----
    python -m experiments.analyze_failures --tier clean
    python -m experiments.analyze_failures --all_tiers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TIERS


# ---------------------------------------------------------------------------
# Period estimation from an image (independent of the manifest)
# ---------------------------------------------------------------------------

def estimate_period_1d(profile: np.ndarray,
                        min_period: float = 6.0,
                        max_period: float = 90.0) -> float:
    """
    Estimate the dominant spatial period of a 1-D profile via its
    autocorrelation, which is more robust to harmonics than raw FFT peak
    picking.

    Returns the period in samples, or nan if no clear periodicity.
    """
    p = profile.astype(np.float64)
    p = p - p.mean()
    if p.std() < 1e-8:
        return float("nan")
    ac = np.correlate(p, p, mode="full")[len(p) - 1:]
    ac /= (ac[0] + 1e-12)

    lo = int(np.floor(min_period))
    hi = int(min(np.ceil(max_period), len(ac) - 2))
    if hi <= lo + 1:
        return float("nan")

    seg = ac[lo:hi]
    # Find the highest local maximum in the valid lag range
    idx = int(np.argmax(seg))
    lag = lo + idx
    if seg[idx] < 0.06:
        return float("nan")

    # Parabolic sub-sample refinement
    if 0 < idx < len(seg) - 1:
        a, b, c = seg[idx - 1], seg[idx], seg[idx + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-12:
            lag += 0.5 * (a - c) / denom
    return float(lag)


def estimate_periods_from_image(img: np.ndarray) -> tuple[float, float]:
    """
    Estimate (period_x, period_y) from a 2-D image by collapsing along each
    axis and running 1-D autocorrelation period estimation.
    """
    px = estimate_period_1d(img.mean(axis=0))   # variation along x
    py = estimate_period_1d(img.mean(axis=1))   # variation along y
    return px, py


# ---------------------------------------------------------------------------
# Alias analysis
# ---------------------------------------------------------------------------

def alias_residual(error: float, period: float) -> tuple[float, int]:
    """
    Distance from `error` to the nearest integer multiple of `period`.

    Returns (residual_px, n) where n is the nearest multiple index.
    A residual near 0 with |n| >= 1 is the signature of an alias failure.
    """
    if not np.isfinite(period) or period <= 1e-6:
        return float("nan"), 0
    n = int(round(error / period))
    return float(abs(error - n * period)), n


def classify_failure(err_x: float, err_y: float,
                      period_x: float, period_y: float,
                      residual_tol: float = 0.30,
                      min_error_px: float = 5.0) -> str:
    """
    Classify a localization failure.

      'correct'    — error below min_error_px
      'alias'      — error is close to an integer multiple of the period in at
                     least one axis, and the residual is a small fraction of
                     the period
      'gross'      — large error that is NOT explained by periodicity

    `residual_tol` is expressed as a FRACTION of the period, so it scales with
    structure size.
    """
    total = float(np.hypot(err_x, err_y))
    if total < min_error_px:
        return "correct"

    rx, nx = alias_residual(abs(err_x), period_x)
    ry, ny = alias_residual(abs(err_y), period_y)

    ok_x = np.isfinite(rx) and rx <= residual_tol * period_x
    ok_y = np.isfinite(ry) and ry <= residual_tol * period_y

    if ok_x and ok_y and (abs(nx) >= 1 or abs(ny) >= 1):
        return "alias"
    return "gross"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_tier(tier: str,
                  dataset_root: str = "outputs/dataset",
                  results_root: str = "outputs/results",
                  fig_root: str = "outputs/figures",
                  method: str = "NCC") -> dict:

    ds_dir = Path(dataset_root) / tier
    res_dir = Path(results_root) / tier
    metrics_path = res_dir / "baseline_metrics.json"
    if not metrics_path.exists():
        print(f"  [skip] no baseline metrics for tier '{tier}'")
        return {}

    with open(metrics_path) as f:
        metrics = json.load(f)

    records = []
    for row in metrics["pairs"]:
        pid = row["pair_id"]
        ref_p = ds_dir / f"pair_{pid:04d}_ref.png"
        if not ref_p.exists():
            continue
        ref = cv2.imread(str(ref_p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # Period estimated from pixels only — no manifest peeking
        est_px, est_py = estimate_periods_from_image(ref)
        true_px, true_py = row["period_x"], row["period_y"]

        err_x = row[f"{method}_pred_x"] - row["gt_x"]
        err_y = row[f"{method}_pred_y"] - row["gt_y"]
        err   = row[f"{method}_error"]

        # Use the TRUE period for classification (estimation quality is
        # reported separately so the two questions stay decoupled).
        cls = classify_failure(err_x, err_y, true_px, true_py)
        rx, nx = alias_residual(abs(err_x), true_px)
        ry, ny = alias_residual(abs(err_y), true_py)

        records.append({
            "pair_id": pid,
            "family": row["structure_family"],
            "noise": row["noise_level"],
            "err": err, "err_x": err_x, "err_y": err_y,
            "true_px": true_px, "true_py": true_py,
            "est_px": est_px, "est_py": est_py,
            "resid_x": rx, "resid_y": ry, "n_x": nx, "n_y": ny,
            "class": cls,
        })

    if not records:
        return {}

    n = len(records)
    n_correct = sum(1 for r in records if r["class"] == "correct")
    n_alias   = sum(1 for r in records if r["class"] == "alias")
    n_gross   = sum(1 for r in records if r["class"] == "gross")
    n_failed  = n_alias + n_gross

    # Period estimation quality
    valid = [r for r in records
             if np.isfinite(r["est_px"]) and np.isfinite(r["true_px"])]
    if valid:
        rel_err_x = np.median([abs(r["est_px"] - r["true_px"]) / r["true_px"]
                                for r in valid])
    else:
        rel_err_x = float("nan")

    print(f"\n=== Periodicity failure analysis — tier '{tier}', "
          f"method {method} ===")
    print(f"  pairs analysed        : {n}")
    print(f"  correct (<5px)        : {n_correct:>3}  ({n_correct/n:.1%})")
    print(f"  ALIAS failures        : {n_alias:>3}  ({n_alias/n:.1%})")
    print(f"  gross failures        : {n_gross:>3}  ({n_gross/n:.1%})")
    if n_failed:
        print(f"  -> of all failures, {n_alias/n_failed:.1%} are period aliases")
    print(f"  period est. median rel. error (x): {rel_err_x:.1%}")

    # ---------------- figures ----------------
    fig_dir = Path(fig_root); fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    # (a) error_x vs period_x with alias lattice lines
    ax = axes[0]
    for r in records:
        ax.scatter(r["true_px"], abs(r["err_x"]), s=32,
                   c=("tab:green" if r["class"] == "correct"
                      else "tab:orange" if r["class"] == "alias"
                      else "tab:red"),
                   alpha=0.8, edgecolors="k", linewidths=0.4)
    pmin = min(r["true_px"] for r in records)
    pmax = max(r["true_px"] for r in records)
    pp = np.linspace(pmin * 0.9, pmax * 1.1, 50)
    for k in range(1, 26):
        ax.plot(pp, k * pp, color="gray", lw=0.6, alpha=0.45)
    ax.set_xlabel("true structural period_x (px)")
    ax.set_ylabel("|error_x| (px)")
    ax.set_title("(a) Errors fall on the alias lattice\n"
                 "grey lines = n x period")
    ax.set_ylim(0, max(abs(r["err_x"]) for r in records) * 1.1 + 1)

    # (b) residual histogram, normalized by period
    ax = axes[1]
    fails = [r for r in records if r["class"] != "correct"]
    if fails:
        frac = [r["resid_x"] / r["true_px"] for r in fails
                if np.isfinite(r["resid_x"])]
        ax.hist(frac, bins=np.linspace(0, 0.5, 21),
                color="tab:orange", edgecolor="k")
        ax.axvline(0.30, color="red", ls="--",
                   label="alias tolerance (0.30)")
        ax.legend(fontsize=9)
    ax.set_xlabel("|error_x mod period_x| / period_x")
    ax.set_ylabel("count")
    ax.set_title("(b) Alias residual\nclustered near 0 => period locking")

    # (c) failure taxonomy
    ax = axes[2]
    labels = ["correct\n(<5px)", "alias\nfailure", "gross\nfailure"]
    vals = [n_correct, n_alias, n_gross]
    ax.bar(labels, vals, color=["tab:green", "tab:orange", "tab:red"],
           edgecolor="k")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v}\n{v/n:.0%}",
                ha="center", fontsize=10)
    ax.set_ylabel("pairs")
    ax.set_title(f"(c) Failure taxonomy — {method}")
    ax.set_ylim(0, max(vals) * 1.25 + 1)

    fig.suptitle(f"Why classical matching fails on periodic structures "
                 f"(tier '{tier}', {method})", fontsize=14)
    fig.tight_layout()
    path = fig_dir / f"failure_analysis_{tier}_{method}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {path}")

    out = {
        "tier": tier, "method": method, "n": n,
        "n_correct": n_correct, "n_alias": n_alias, "n_gross": n_gross,
        "alias_share_of_failures": (n_alias / n_failed) if n_failed else None,
        "period_est_rel_err_x": (None if not np.isfinite(rel_err_x)
                                  else round(float(rel_err_x), 4)),
        "records": records,
    }
    with open(res_dir / f"failure_analysis_{method}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="clean", choices=list(TIERS))
    ap.add_argument("--all_tiers", action="store_true")
    ap.add_argument("--method", default="NCC",
                    choices=["NCC", "PhaseCC", "MultiNCC"])
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--results", default="outputs/results")
    ap.add_argument("--figures", default="outputs/figures")
    args = ap.parse_args()

    tiers = list(TIERS) if args.all_tiers else [args.tier]
    summaries = []
    for t in tiers:
        r = analyze_tier(t, args.dataset, args.results, args.figures,
                          args.method)
        if r:
            summaries.append(r)

    if len(summaries) > 1:
        print("\n" + "=" * 70)
        print(f"{'Tier':>12} {'correct':>9} {'alias':>9} {'gross':>9} "
              f"{'alias/fail':>11}")
        print("-" * 70)
        for s in summaries:
            share = ("n/a" if s["alias_share_of_failures"] is None
                     else f"{s['alias_share_of_failures']:.0%}")
            print(f"{s['tier']:>12} {s['n_correct']:>9} {s['n_alias']:>9} "
                  f"{s['n_gross']:>9} {share:>11}")
