"""Profile the frozen v1 engine and emit the golden DEV baseline used by G2.

Produces two ResultsFile JSONs (validated by tests/engine/test_profile_outputs.py):
    results/accuracy/baseline_v1.json  — headline accuracy + latency
    results/accuracy/profile_v1.json   — cProfile hot spots + stage micro-timings

All numbers flow through the public contract: every pair is localized via
driftsense.api.localize(engine="v1"). Run ONCE; read the JSON back for docs/PR.

    python -m eval.profile_v1

DEV dataset = union of all four tiers under outputs/dataset/ (clean, nominal,
hard, ambiguous), per Person A's decision on 2026-10-18.
"""

from __future__ import annotations

import cProfile
import json
import math
import platform
import pstats
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from driftsense.api import localize
from driftsense.contracts import Budget, EnvInfo, Metric, Mode, ResultsFile
from driftsense.engines.v1_adapter import V1_VERSION as V1_LABEL

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "outputs" / "dataset"
TIERS = ("clean", "nominal", "hard", "ambiguous")
OUT_DIR = ROOT / "results" / "accuracy"

THR_PX = 3.0  # "correct" radius; matches dataset self-audit pass@3px
N_BOOT = 2000
BOOT_SEED = 0
PROFILE_PER_TIER = 4  # pairs/tier fed to cProfile for the hot-spot pass

# Prior figures we are asked to check against (source: P0 hand-off notes).
CLAIMED = {
    "latency_per_pair_ms": 11600.0,
    "fingerprint_ms": 9900.0,
    "matchtemplate_ms": 23.0,
}


def _pairs(tier: str):
    """Yield (pair_id, template_bytes, search_bytes, gt_x, gt_y, ref_w, ref_h)."""
    d = DATASET_ROOT / tier
    manifest = json.loads((d / "manifest.json").read_text())
    for p in manifest["pairs"]:
        pid = p["pair_id"]
        tb = (d / f"pair_{pid:04d}_ref.png").read_bytes()
        sb = (d / f"pair_{pid:04d}_search.png").read_bytes()
        yield pid, tb, sb, p["gt_x"], p["gt_y"], p["ref_w"], p["ref_h"]


def _centre(cand, ref_w: float, ref_h: float) -> tuple[float, float]:
    """Adapter stores template top-left; recover the centre for GT comparison."""
    return cand.x_px + ref_w / 2.0, cand.y_px + ref_h / 2.0


def _hit(px: float, py: float, gx: float, gy: float) -> bool:
    return math.hypot(px - gx, py - gy) <= THR_PX


def measure() -> dict:
    """One pass over the full DEV set; returns per-pair hits and latencies."""
    budget = Budget(max_candidates=20)
    top1, recall20, wall_ms, total_ms = [], [], [], []
    n = 0
    for tier in TIERS:
        for _pid, tb, sb, gx, gy, rw, rh in _pairs(tier):
            t0 = time.perf_counter()
            res = localize(tb, sb, Mode.ACCURATE, budget, engine="v1")
            wall_ms.append((time.perf_counter() - t0) * 1e3)
            total_ms.append(res.total_ms)
            cx, cy = _centre(res.candidates[0], rw, rh)
            top1.append(1 if _hit(cx, cy, gx, gy) else 0)
            recall20.append(1 if any(_hit(*_centre(c, rw, rh), gx, gy) for c in res.candidates) else 0)
            n += 1
    return {
        "n": n,
        "top1": top1,
        "recall20": recall20,
        "wall_ms": wall_ms,
        "total_ms": total_ms,
    }


def bootstrap_ci(hits: list[int]) -> tuple[float, float, float]:
    """Point estimate + 95% percentile bootstrap CI of a 0/1 mean."""
    arr = np.asarray(hits, dtype=float)
    point = float(arr.mean())
    rng = np.random.default_rng(BOOT_SEED)
    n = len(arr)
    means = arr[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def profile_hotspots() -> tuple[dict[str, tuple[float, int]], float]:
    """cProfile a fixed subset; return {func: (cum_ms, ncalls)} and wall ms."""
    budget = Budget(max_candidates=20)
    subset = []
    for tier in TIERS:
        for i, rec in enumerate(_pairs(tier)):
            if i >= PROFILE_PER_TIER:
                break
            subset.append(rec)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    for _pid, tb, sb, *_ in subset:
        localize(tb, sb, Mode.ACCURATE, budget, engine="v1")
    pr.disable()
    wall = (time.perf_counter() - t0) * 1e3

    st = pstats.Stats(pr)
    hot: dict[str, tuple[float, int]] = {}
    for (fn, _lineno, name), (_cc, ncalls, _tt, ct, _callers) in st.stats.items():
        label = name if fn in ("~", "") else f"{Path(fn).name}:{name}"
        prev = hot.get(label, (0.0, 0))
        hot[label] = (prev[0] + ct * 1e3, prev[1] + ncalls)
    return hot, wall


def _hot_ms(hot: dict[str, tuple[float, int]], needle: str) -> float:
    """Cumulative ms of the dominant hot-spot matching needle.

    Uses the max (outermost) match, not a sum: cProfile cumulative time of a
    parent already includes its children (e.g. extract_fingerprint includes
    extract_axis_fingerprint), so summing matches would double-count.
    """
    matches = [ms for label, (ms, _n) in hot.items() if needle.lower() in label.lower()]
    return max(matches) if matches else 0.0


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


def _write(name: str, produced_by: str, metrics: dict[str, Metric]) -> Path:
    rf = ResultsFile(
        name=name,
        produced_by=produced_by,
        git_sha=_git_sha(),
        created_at=datetime.now(UTC),
        dataset="dev",
        env=_env(),
        metrics=metrics,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(rf.model_dump_json(indent=2) + "\n")
    return path


def main() -> int:
    m = measure()
    n = m["n"]
    t1, t1_lo, t1_hi = bootstrap_ci(m["top1"])
    r20, r20_lo, r20_hi = bootstrap_ci(m["recall20"])
    wall = np.asarray(m["wall_ms"])
    total = np.asarray(m["total_ms"])
    lat_ms = float(total.mean())
    lat_lo, lat_hi = (float(x) for x in np.percentile(total, [2.5, 97.5]))

    baseline = {
        "top1_accuracy": Metric(value=t1, unit="fraction", n=n, ci_low=t1_lo, ci_high=t1_hi),
        "recall_at_20": Metric(value=r20, unit="fraction", n=n, ci_low=r20_lo, ci_high=r20_hi),
        "latency_per_pair_ms": Metric(value=lat_ms, unit="ms", n=n, ci_low=lat_lo, ci_high=lat_hi),
        "wall_per_pair_ms": Metric(value=float(wall.mean()), unit="ms", n=n),
    }
    base_path = _write("baseline_v1", "eval/profile_v1.py", baseline)

    hot, prof_wall = profile_hotspots()
    n_prof = PROFILE_PER_TIER * len(TIERS)
    mt_ms = _hot_ms(hot, "matchTemplate") / n_prof
    fp_ms = _hot_ms(hot, "extract_fingerprint") / n_prof
    top = sorted(hot.items(), key=lambda kv: kv[1][0], reverse=True)[:12]

    prof_metrics: dict[str, Metric] = {
        "matchtemplate_ms": Metric(value=mt_ms, unit="ms", n=n_prof),
        "fingerprint_ms": Metric(value=fp_ms, unit="ms", n=n_prof),
        "profiled_pairs_wall_ms": Metric(value=prof_wall, unit="ms", n=n_prof),
        "latency_per_pair_ms": Metric(value=lat_ms, unit="ms", n=n),
    }
    for i, (label, (ms, ncalls)) in enumerate(top):
        prof_metrics[f"hot.{i:02d}.{label}"] = Metric(value=ms / n_prof, unit="ms", n=ncalls)
    # Record the prior claims as data so a measured-vs-claimed table can be
    # rendered later without re-typing numbers into docs.
    for k, v in CLAIMED.items():
        prof_metrics[f"claimed.{k}"] = Metric(value=v, unit="ms")
    prof_path = _write("profile_v1", "eval/profile_v1.py", prof_metrics)

    # ---- console report (paste into PR body; never retyped) ----
    print(f"\nDEV baseline — v1 ({V1_LABEL}) — n={n} pairs across {', '.join(TIERS)}")
    print("-" * 68)
    print(f"  top-1 accuracy : {t1:.3f}  (95% CI {t1_lo:.3f}–{t1_hi:.3f}, n={n})")
    print(f"  recall@20      : {r20:.3f}  (95% CI {r20_lo:.3f}–{r20_hi:.3f}, n={n})")
    print(f"  latency/pair   : {lat_ms:.1f} ms  (contract total_ms; 95% CI {lat_lo:.1f}–{lat_hi:.1f})")
    print(f"  wall/pair      : {wall.mean():.1f} ms")
    print("\nMeasured vs claimed (prior P0 figures)")
    print("-" * 68)
    print(f"  {'metric':<22}{'measured':>14}{'claimed':>14}{'ratio':>10}")
    rows = [
        ("latency_per_pair_ms", lat_ms, CLAIMED["latency_per_pair_ms"]),
        ("fingerprint_ms", fp_ms, CLAIMED["fingerprint_ms"]),
        ("matchtemplate_ms", mt_ms, CLAIMED["matchtemplate_ms"]),
    ]
    for label, meas, claim in rows:
        ratio = meas / claim if claim else float("nan")
        flag = "  <-- DIFFERS" if not (0.5 <= ratio <= 2.0) else ""
        print(f"  {label:<22}{meas:>14.1f}{claim:>14.1f}{ratio:>10.3f}{flag}")
    print("\ncProfile hot spots (mean ms/pair over profiled subset):")
    for label, (ms, ncalls) in top:
        print(f"  {ms / n_prof:>9.2f} ms  {ncalls:>7} calls  {label}")
    print(f"\nwrote {base_path}")
    print(f"wrote {prof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
