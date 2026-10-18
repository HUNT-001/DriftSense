"""Interim S5-S6 (cascade re-rank) accuracy on the DEV set.

Runs the v2 pipeline through S4 to get the dense candidate set, then applies S6
cascade re-ranking, and reports the DENSE vs CASCADE paths side by side so we can
see whether the dense path alone can reach G2 (recall@20 >= 98%) or whether the
cascade is doing the work.

Key fact surfaced by this script: re-ranking reorders a fixed top-K set, so
dense and cascade recall@20 are identical by construction — the cascade can only
move the true copy *within* the set, which shows up in TOP-1, not recall@20.
Both are reported anyway, exactly so that equality is visible.

S5 contour is a scaffold in this PR (NotImplementedError until A5) and is not
exercised here.

Emits one ResultsFile JSON (schema driftsense.results/1):
    results/accuracy/v2_s5_s6.json

All numbers come from this one pass; run ONCE and read the JSON back for docs/PR.

    python -m eval.v2_s5_s6_eval

DEV dataset = union of all four tiers under outputs/dataset/, matching the other
A evals.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from driftsense.contracts import EnvInfo, Metric, ResultsFile
from driftsense.v2.cascade import CascadeConfig, rerank
from driftsense.v2.lattice import detect_lattice
from driftsense.v2.roughness import residual
from driftsense.v2.score import score_residual

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "outputs" / "dataset"
TIERS = ("clean", "nominal", "hard", "ambiguous")
OUT_DIR = ROOT / "results" / "accuracy"
BASELINE = OUT_DIR / "baseline_v1.json"

THR_PX = 3.0
N_BOOT = 2000
BOOT_SEED = 0
CFG = CascadeConfig()  # top_k=20, tau=0.10, recall_target=0.98


def _pairs(tier: str):
    d = DATASET_ROOT / tier
    manifest = json.loads((d / "manifest.json").read_text())
    for p in manifest["pairs"]:
        pid = p["pair_id"]
        tb = (d / f"pair_{pid:04d}_ref.png").read_bytes()
        sb = (d / f"pair_{pid:04d}_search.png").read_bytes()
        yield tb, sb, p["gt_x"], p["gt_y"], p["ref_w"], p["ref_h"]


def _decode(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)


def _hit(c, gx, gy, rw, rh) -> bool:
    return math.hypot(c.x_px + rw / 2.0 - gx, c.y_px + rh / 2.0 - gy) <= THR_PX


def _top1(cands, gx, gy, rw, rh) -> int:
    return int(bool(cands) and _hit(cands[0], gx, gy, rw, rh))


def _recall(cands, gx, gy, rw, rh) -> int:
    return int(any(_hit(c, gx, gy, rw, rh) for c in cands))


def measure() -> dict:
    out = {k: [] for k in ("dense_top1", "cascade_top1", "v1_top1", "dense_rec", "cascade_rec")}
    escalated = 0
    for tier in TIERS:
        for tb, sb, gx, gy, rw, rh in _pairs(tier):
            tmpl, search = _decode(tb), _decode(sb)
            lat = detect_lattice(search)
            dense = score_residual(
                residual(tmpl, lat), residual(search, lat), lat, max_candidates=CFG.top_k
            ).candidates
            casc = rerank(dense, tmpl, search, mode="cascade", cfg=CFG)
            v1 = rerank(dense, tmpl, search, mode="v1", cfg=CFG)
            escalated += int(casc.escalated)
            out["dense_top1"].append(_top1(dense, gx, gy, rw, rh))
            out["cascade_top1"].append(_top1(casc.candidates, gx, gy, rw, rh))
            out["v1_top1"].append(_top1(v1.candidates, gx, gy, rw, rh))
            out["dense_rec"].append(_recall(dense, gx, gy, rw, rh))
            out["cascade_rec"].append(_recall(casc.candidates, gx, gy, rw, rh))
    out["n"] = len(out["dense_top1"])
    out["escalation_rate"] = escalated / max(out["n"], 1)
    return out


def bootstrap_ci(hits: list[int]) -> tuple[float, float, float]:
    a = np.asarray(hits, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(BOOT_SEED)
    means = a[rng.integers(0, a.size, size=(N_BOOT, a.size))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


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


def _v1_ref() -> dict[str, float]:
    if not BASELINE.exists():
        return {}
    d = json.loads(BASELINE.read_text())
    return {k: float(v["value"]) for k, v in d.get("metrics", {}).items()}


def _metric(hits: list[int], n: int) -> Metric:
    val, lo, hi = bootstrap_ci(hits)
    return Metric(value=val, unit="fraction", n=n, ci_low=lo, ci_high=hi)


def main() -> int:
    m = measure()
    n = m["n"]
    v1 = _v1_ref()
    metrics: dict[str, Metric] = {
        "dense_recall_at_20": _metric(m["dense_rec"], n),
        "cascade_recall_at_20": _metric(m["cascade_rec"], n),
        "dense_top1": _metric(m["dense_top1"], n),
        "cascade_top1": _metric(m["cascade_top1"], n),
        "v1_rerank_top1": _metric(m["v1_top1"], n),
        "cascade_escalation_rate": Metric(value=m["escalation_rate"], unit="fraction", n=n),
        "tau": Metric(value=CFG.tau, unit="score"),
        "recall_target_g2": Metric(value=CFG.recall_target, unit="fraction"),
    }
    if v1:
        metrics["v1_golden_recall_at_20"] = Metric(value=v1.get("recall_at_20", 0.0), unit="fraction")
        metrics["v1_golden_top1"] = Metric(value=v1.get("top1_accuracy", 0.0), unit="fraction")

    rf = ResultsFile(
        name="v2_s5_s6",
        produced_by="eval/v2_s5_s6_eval.py",
        git_sha=_git_sha(),
        created_at=datetime.now(UTC),
        dataset="dev",
        env=_env(),
        metrics=metrics,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "v2_s5_s6.json"
    out.write_text(rf.model_dump_json(indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}  (n={n})")
    print(
        f"  dense   recall@20={metrics['dense_recall_at_20'].value:.3f}  top1={metrics['dense_top1'].value:.3f}"
    )
    print(
        f"  cascade recall@20={metrics['cascade_recall_at_20'].value:.3f}  top1={metrics['cascade_top1'].value:.3f}"
        f"  (escalated {m['escalation_rate']:.0%})"
    )
    print(f"  v1-rerank top1={metrics['v1_rerank_top1'].value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
