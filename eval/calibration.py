"""A3 — fit and audit the v2 confidence calibration on a held-out CALIB split.

The engine's commit/abstain decision rests on a calibrated probability, so the
map margin_z -> P(correct) and the commit threshold tau_commit must be fit on
data the engine never trains or tunes on. This script owns that split:

  * CALIB is its OWN tiered dataset under ``outputs/dataset_calib/`` with its own
    seed family (:data:`CALIB_SEED`), deliberately disjoint from the DEV seed
    family (42...) and from the sealed seed (9999). Calibration therefore never
    touches DEV or the sealed set -- a leak would make the reliability numbers a
    fiction. It is generated here if absent.
  * For each pair we run the production v2 engine (AUTO / cascade path), read the
    ranked candidate scores and compute margin_z directly (independent of any
    existing calibration, so re-runs are stable), and label rank-1 correct within
    THR_PX of the ground truth.
  * We fit the isotonic calibrator (driftsense.v2.confidence), then report a
    reliability diagram, the Expected Calibration Error WITH a bootstrap CI, the
    Brier score, and a risk-coverage curve, plus the commit operating point
    (coverage and false-confident rate at tau_commit).

Emits (run ONCE; read the JSON back for docs/PR -- never retype numbers):
    results/accuracy/calibration.json            (ResultsFile; embeds the model)
    results/accuracy/calibration_reliability.png
    results/accuracy/calibration_risk_coverage.png

    python -m eval.calibration            # full CALIB (40 pairs/tier)
    python -m eval.calibration --fast     # small subset, wiring checks only
"""

from __future__ import annotations

import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from driftsense.api import localize  # noqa: E402
from driftsense.contracts import Budget, EnvInfo, Metric, Mode, ResultsFile  # noqa: E402
from driftsense.v2.confidence import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_MAD_FLOOR,
    Calibrator,
    fit_calibrator,
    margin_z,
    wilson_upper,
)

ROOT = Path(__file__).resolve().parent.parent
DEV_ROOT = ROOT / "outputs" / "dataset"  # the DEV split -- NEVER read here
CALIB_ROOT = ROOT / "outputs" / "dataset_calib"  # this script's own split
OUT_DIR = ROOT / "results" / "accuracy"
OUT_JSON = OUT_DIR / "calibration.json"

TIERS = ("clean", "nominal", "hard", "ambiguous")
THR_PX = 3.0
N_BINS = 10
N_BOOT = 2000
BOOT_SEED = 0

# Seed isolation. CALIB has its own family; DEV uses 42 (+ i*1000 per tier) and
# the sealed run uses 9999. These three sets must stay disjoint.
CALIB_SEED = 7331
DEV_SEED_BASE = 42
SEALED_SEED = 9999


def calib_tier_seeds() -> list[int]:
    """Per-tier generation seeds for CALIB (mirrors dataset_gen's +i*1000 rule)."""
    return [CALIB_SEED + i * 1000 for i in range(len(TIERS))]


def dev_tier_seeds() -> list[int]:
    return [DEV_SEED_BASE + i * 1000 for i in range(len(TIERS))]


# --------------------------------------------------------------------------- #
# CALIB dataset                                                               #
# --------------------------------------------------------------------------- #
def ensure_calib_dataset(n_pairs: int) -> None:
    """Generate the CALIB tiers under CALIB_ROOT if their manifests are missing."""
    from data_gen.dataset_gen import generate_dataset

    for tier, seed in zip(TIERS, calib_tier_seeds(), strict=True):
        if (CALIB_ROOT / tier / "manifest.json").exists():
            continue
        generate_dataset(
            tier_name=tier, n_pairs=n_pairs, seed=seed, out_root=str(CALIB_ROOT), audit=True, quiet=True
        )


def _pairs(tier: str, limit: int | None):
    import json

    d = CALIB_ROOT / tier
    pairs = json.loads((d / "manifest.json").read_text())["pairs"]
    if limit is not None:
        pairs = pairs[:limit]
    for p in pairs:
        pid = p["pair_id"]
        yield (
            (d / f"pair_{pid:04d}_ref.png").read_bytes(),
            (d / f"pair_{pid:04d}_search.png").read_bytes(),
            p["gt_x"],
            p["gt_y"],
            p["ref_w"],
            p["ref_h"],
        )


def _hit(c, gx, gy, rw, rh) -> bool:
    return math.hypot(c.x_px + rw / 2.0 - gx, c.y_px + rh / 2.0 - gy) <= THR_PX


def collect(limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Run v2 over CALIB; return (margin_z per pair, rank-1 correct 0/1).

    Mode.FAST is the G2-chosen production default (the roughness path), so the
    margins calibrated here are the ones the deployed single-shot engine reports.
    """
    zs: list[float] = []
    correct: list[int] = []
    budget = Budget(max_candidates=20)
    for tier in TIERS:
        for tb, sb, gx, gy, rw, rh in _pairs(tier, limit):
            r = localize(tb, sb, Mode.FAST, budget, engine="v2")  # production default
            zs.append(margin_z([c.score for c in r.candidates]))
            correct.append(int(_hit(r.candidates[0], gx, gy, rw, rh)))
    return np.asarray(zs, dtype=np.float64), np.asarray(correct, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Calibration metrics                                                         #
# --------------------------------------------------------------------------- #
def oof_predictions(zs: np.ndarray, correct: np.ndarray, k: int = 5) -> np.ndarray:
    """Out-of-fold P(correct): for each fold, fit the isotonic map on the OTHER
    folds and predict the held-out one. In-sample isotonic is trivially perfect
    (ECE ~ 0 by construction), so reliability/ECE/Brier must be read off these
    held-out predictions to mean generalisation. The SHIPPED model is still fit
    on all of CALIB; only the reported calibration quality uses cross-fitting.
    """
    n = zs.size
    if n < k:
        return np.array([fit_calibrator(zs, correct).p_of(float(z)) for z in zs])
    rng = np.random.default_rng(BOOT_SEED)
    fold = rng.integers(0, k, n)
    p = np.empty(n)
    for f in range(k):
        te = fold == f
        tr = ~te
        cal_f = fit_calibrator(zs[tr], correct[tr])
        p[te] = [cal_f.p_of(float(z)) for z in zs[te]]
    return p


def reliability_bins(p: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    """Equal-width confidence bins; returns (conf, acc, count) arrays per bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    acc = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        sel = idx == b
        cnt[b] = int(sel.sum())
        if cnt[b]:
            conf[b] = float(p[sel].mean())
            acc[b] = float(correct[sel].mean())
    return conf, acc, cnt


def ece_of(p: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    conf, acc, cnt = reliability_bins(p, correct, n_bins)
    n = int(cnt.sum())
    if n == 0:
        return 0.0
    gap = np.abs(acc - conf)
    return float(np.nansum((cnt / n) * np.where(cnt > 0, gap, 0.0)))


def ece_with_ci(p: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    """ECE point estimate with a bootstrap 95% CI over resampled pairs."""
    point = ece_of(p, correct, n_bins)
    n = p.size
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        s = rng.integers(0, n, n)
        boot[i] = ece_of(p[s], correct[s], n_bins)
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def risk_coverage(zs: np.ndarray, correct: np.ndarray, n_points: int = 20):
    """Commit by margin_z, high first; return (coverage, risk=wrong-fraction)."""
    order = np.argsort(-zs, kind="mergesort")
    wrong = (correct[order] == 0.0).astype(np.float64)
    n = zs.size
    cov, risk = [], []
    for k in range(1, n + 1):
        cov.append(k / n)
        risk.append(float(wrong[:k].mean()))
    cov = np.asarray(cov)
    risk = np.asarray(risk)
    if n > n_points:  # thin for the JSON / plot
        pick = np.linspace(0, n - 1, n_points).round().astype(int)
        cov, risk = cov[pick], risk[pick]
    return cov, risk


def operating_point(zs: np.ndarray, correct: np.ndarray, tau: float):
    """Coverage and false-confident rate (+ Wilson UB) at the commit threshold."""
    commit = zs >= tau
    n = int(commit.sum())
    wrong = int(((correct == 0.0) & commit).sum())
    coverage = n / zs.size if zs.size else 0.0
    fc = wrong / n if n else 0.0
    return coverage, fc, wilson_upper(wrong, n), n


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def plot_reliability(p: np.ndarray, correct: np.ndarray, out: Path) -> None:
    conf, acc, cnt = reliability_bins(p, correct)
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfect")
    sel = cnt > 0
    ax.plot(conf[sel], acc[sel], "o-", color="#1f77b4", label="v2 (CALIB)")
    ax.set_xlabel("predicted P(correct)")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig)


def plot_risk_coverage(cov: np.ndarray, risk: np.ndarray, tau_cov: float, alpha: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.plot(cov, risk, "o-", color="#d62728", label="risk vs coverage")
    ax.axhline(alpha, ls="--", color="gray", lw=1, label=f"target {alpha:.0%}")
    ax.axvline(tau_cov, ls=":", color="#1f77b4", lw=1, label=f"commit coverage {tau_cov:.0%}")
    ax.set_xlabel("coverage (fraction committed)")
    ax.set_ylabel("risk (false-confident fraction)")
    ax.set_title("Risk-coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.05, float(risk.max()) * 1.1 if risk.size else 0.05))
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out, bbox_inches="tight", facecolor="white", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Results file                                                                 #
# --------------------------------------------------------------------------- #
def _env() -> EnvInfo:
    return EnvInfo(
        python=sys.version.split()[0],
        opencv=cv2.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
        arch=platform.machine(),
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def build_metrics(calib: Calibrator, zs: np.ndarray, correct: np.ndarray, p: np.ndarray) -> dict[str, Metric]:
    """``p`` are the out-of-fold predictions used for reliability/ECE/Brier."""
    ece, ece_lo, ece_hi = ece_with_ci(p, correct)
    brier = float(np.mean((p - correct) ** 2)) if zs.size else 0.0
    cov_tau, fc_tau, fc_ub, n_commit = operating_point(zs, correct, calib.tau_commit)
    conf, acc, cnt = reliability_bins(p, correct)
    # A commit region exists only if some finite margin meets the target; if not,
    # tau_commit sits above every observed margin (safe abstain-everywhere).
    region_exists = bool(zs.size and calib.tau_commit <= float(zs.max()))

    metrics: dict[str, Metric] = {
        "tau_commit": Metric(value=float(calib.tau_commit), unit="margin_z", n=int(zs.size)),
        "commit_region_exists": Metric(value=1.0 if region_exists else 0.0, unit="bool"),
        "ece": Metric(value=ece, unit="fraction", n=int(zs.size), ci_low=ece_lo, ci_high=ece_hi),
        "brier": Metric(value=brier, unit="score", n=int(zs.size)),
        "false_confident_at_tau": Metric(value=fc_tau, unit="fraction", n=n_commit, ci_high=fc_ub),
        "false_confident_target": Metric(value=float(calib.alpha), unit="fraction"),
        "coverage_at_tau": Metric(value=cov_tau, unit="fraction", n=int(zs.size)),
        "accuracy_overall": Metric(
            value=float(correct.mean()) if zs.size else 0.0, unit="fraction", n=int(zs.size)
        ),
    }
    for b in range(N_BINS):
        if cnt[b]:
            metrics[f"reliability.bin{b}.conf"] = Metric(value=float(conf[b]), unit="fraction", n=int(cnt[b]))
            metrics[f"reliability.bin{b}.acc"] = Metric(value=float(acc[b]), unit="fraction", n=int(cnt[b]))
    cov, risk = risk_coverage(zs, correct)
    for i, (cv, rk) in enumerate(zip(cov, risk, strict=True)):
        metrics[f"rc.{i}.coverage"] = Metric(value=float(cv), unit="fraction")
        metrics[f"rc.{i}.risk"] = Metric(value=float(rk), unit="fraction")
    # Embed the fitted model so the engine and UI load tau + the map from here.
    for k, v in calib.to_model_dict().items():
        metrics[k] = Metric(value=float(v), unit="model")
    return metrics


def write_results(metrics: dict[str, Metric]) -> None:
    rf = ResultsFile(
        name="calibration",
        produced_by="eval/calibration.py",
        git_sha=_git_sha(),
        created_at=datetime.now(UTC),
        dataset="synthetic",  # CALIB split: NOT dev, NOT sealed
        env=_env(),
        metrics=metrics,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(rf.model_dump_json(indent=2) + "\n")


def main() -> int:
    fast = "--fast" in sys.argv
    n_pairs = 8 if fast else 40
    limit = 4 if fast else None

    ensure_calib_dataset(n_pairs)
    zs, correct = collect(limit)
    calib = fit_calibrator(zs, correct, alpha=DEFAULT_ALPHA, mad_floor=DEFAULT_MAD_FLOOR)
    p = oof_predictions(zs, correct)  # held-out predictions for honest reliability/ECE
    metrics = build_metrics(calib, zs, correct, p)
    write_results(metrics)

    plot_reliability(p, correct, OUT_DIR / "calibration_reliability.png")
    cov_tau = metrics["coverage_at_tau"].value
    cov, risk = risk_coverage(zs, correct)
    plot_risk_coverage(cov, risk, cov_tau, calib.alpha, OUT_DIR / "calibration_risk_coverage.png")

    region = metrics["commit_region_exists"].value >= 0.5
    print(f"wrote {OUT_JSON.relative_to(ROOT)}  (n={zs.size})")
    print(f"  overall top-1 accuracy    = {metrics['accuracy_overall'].value:.3f}")
    if region:
        print(f"  tau_commit (margin_z)     = {calib.tau_commit:.4f}")
    else:
        print(
            f"  tau_commit (margin_z)     = {calib.tau_commit:.4f}  (ABSTAIN-ALL: no margin "
            f"certifies <= {calib.alpha:.0%} false-confident on single-shot CALIB)"
        )
    print(
        f"  ECE                       = {metrics['ece'].value:.4f} "
        f"[{metrics['ece'].ci_low:.4f}, {metrics['ece'].ci_high:.4f}]"
    )
    print(f"  Brier                     = {metrics['brier'].value:.4f}")
    print(f"  coverage @ tau            = {metrics['coverage_at_tau'].value:.3f}")
    print(
        f"  false-confident @ tau     = {metrics['false_confident_at_tau'].value:.4f} "
        f"(Wilson UB {metrics['false_confident_at_tau'].ci_high:.4f}, target {calib.alpha:.2%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
