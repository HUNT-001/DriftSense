"""Interim S3–S4 (roughness + residual scoring) accuracy/speed on the DEV set.

Measures the v2 roughness-residual pipeline through S4 only — ingest-decode,
detect_lattice, roughness.residual on both frames, score_residual — with NO
cascade, subpixel, or rectification wiring yet (those land in later A stages).
These are deliberately *interim* numbers for the PR body; they are expected to
sit below the full v1 engine until cascade/subpixel are added.

Emits one ResultsFile JSON (schema driftsense.results/1):
    results/accuracy/v2_s3_s4.json

All numbers come from this one pass; run ONCE and read the JSON back for docs/PR.
Compares against the recorded golden v1 baseline (results/accuracy/baseline_v1.json)
rather than re-running the slow v1 engine.

    python -m eval.v2_s3_s4_eval

DEV dataset = union of all four tiers under outputs/dataset/ (clean, nominal,
hard, ambiguous), matching eval/profile_v1.py.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from driftsense.contracts import EnvInfo, Metric, ResultsFile
from driftsense.v2.lattice import detect_lattice
from driftsense.v2.roughness import residual
from driftsense.v2.score import score_residual

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "outputs" / "dataset"
TIERS = ("clean", "nominal", "hard", "ambiguous")
OUT_DIR = ROOT / "results" / "accuracy"
BASELINE = OUT_DIR / "baseline_v1.json"

THR_PX = 3.0  # "correct" radius; matches the dataset self-audit / profile_v1
N_BOOT = 2000
BOOT_SEED = 0
MAX_CANDIDATES = 20


def _pairs(tier: str):
    """Yield (template_bytes, search_bytes, gt_x, gt_y, ref_w, ref_h)."""
    d = DATASET_ROOT / tier
    manifest = json.loads((d / "manifest.json").read_text())
    for p in manifest["pairs"]:
        pid = p["pair_id"]
        tb = (d / f"pair_{pid:04d}_ref.png").read_bytes()
        sb = (d / f"pair_{pid:04d}_search.png").read_bytes()
        yield tb, sb, p["gt_x"], p["gt_y"], p["ref_w"], p["ref_h"]


def _decode(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)


def _hit(px: float, py: float, gx: float, gy: float) -> bool:
    return math.hypot(px - gx, py - gy) <= THR_PX


def measure() -> dict:
    """One S3–S4 pass over the full DEV set; per-pair hits and per-pair latency."""
    top1: list[int] = []
    recall20: list[int] = []
    stage_ms: list[float] = []
    for tier in TIERS:
        for tb, sb, gx, gy, rw, rh in _pairs(tier):
            tmpl = _decode(tb)
            search = _decode(sb)
            t0 = time.perf_counter()
            lat = detect_lattice(search)
            rt = residual(tmpl, lat)
            rs = residual(search, lat)
            res = score_residual(rt, rs, lat, max_candidates=MAX_CANDIDATES)
            stage_ms.append((time.perf_counter() - t0) * 1e3)

            # score_residual returns template top-left offsets; GT is a centre.
            hit1 = 0
            hitk = 0
            for c in res.candidates:
                cx, cy = c.x_px + rw / 2.0, c.y_px + rh / 2.0
                if _hit(cx, cy, gx, gy):
                    hitk = 1
                    if c.rank == 1:
                        hit1 = 1
            top1.append(hit1)
            recall20.append(hitk)
    return {"top1": top1, "recall20": recall20, "stage_ms": stage_ms, "n": len(top1)}


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


def main() -> int:
    m = measure()
    n = m["n"]
    t1, t1_lo, t1_hi = bootstrap_ci(m["top1"])
    r20, r20_lo, r20_hi = bootstrap_ci(m["recall20"])
    stage = np.asarray(m["stage_ms"])
    v1 = _v1_ref()

    metrics: dict[str, Metric] = {
        "top1_accuracy": Metric(value=t1, unit="fraction", n=n, ci_low=t1_lo, ci_high=t1_hi),
        "recall_at_20": Metric(value=r20, unit="fraction", n=n, ci_low=r20_lo, ci_high=r20_hi),
        "stage_ms_per_pair": Metric(
            value=float(stage.mean()),
            unit="ms",
            n=n,
            ci_low=float(np.percentile(stage, 2.5)),
            ci_high=float(np.percentile(stage, 97.5)),
        ),
    }
    if v1:
        metrics["v1_top1_accuracy"] = Metric(value=v1.get("top1_accuracy", 0.0), unit="fraction")
        metrics["v1_recall_at_20"] = Metric(value=v1.get("recall_at_20", 0.0), unit="fraction")
        metrics["v1_latency_per_pair_ms"] = Metric(value=v1.get("latency_per_pair_ms", 0.0), unit="ms")

    rf = ResultsFile(
        name="v2_s3_s4",
        produced_by="eval/v2_s3_s4_eval.py",
        git_sha=_git_sha(),
        created_at=datetime.now(UTC),
        dataset="dev",
        env=_env(),
        metrics=metrics,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "v2_s3_s4.json"
    out.write_text(rf.model_dump_json(indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}  (n={n})")
    print(f"  v2 S3-S4 top1={t1:.3f}  recall@20={r20:.3f}  stage={stage.mean():.1f} ms/pair")
    if v1:
        print(
            f"  v1 golden   top1={v1.get('top1_accuracy', 0):.3f}  recall@20={v1.get('recall_at_20', 0):.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
