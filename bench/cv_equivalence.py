"""OpenCV-version equivalence of the frozen v2 engine (4.12 vs 5.0).

Every DEV/sealed results file was produced with opencv-python-headless 4.12.0 (see each
file's ``env.opencv``), while the competition stack and the deployed COOL worker run
OpenCV 5. This script shows, on DEV data only, whether the frozen engine and agent give
the same answers on both versions, so the sealed numbers can be read as OpenCV 5 numbers
(or not). It never touches sealed data.

Two steps, each in a venv with the OpenCV version under test:

    python -m bench.cv_equivalence dump --out <dir>/cv412.jsonl     # OpenCV 4.12 venv
    python -m bench.cv_equivalence dump --out <dir>/cv5.jsonl       # OpenCV 5 venv
    python -m bench.cv_equivalence compare <dir>/cv412.jsonl <dir>/cv5.jsonl \
        --out results/bench/cv5_equivalence.json

Pair loading and the top-1 hit rule mirror eval/gate_g2.py (THR_PX = 3.0, the template
centre within 3 px of ground truth); episode knobs mirror eval/agent_eval.py.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "outputs" / "dataset"
TIERS = ("clean", "nominal", "hard", "ambiguous")  # eval/gate_g2.py
THR_PX = 3.0  # eval/gate_g2.py

# eval/agent_eval.py episode constants
TAU = 0.10
SITE_NM = 2000.0
PIXEL_NM = 1.0
REF_FOV_NM = 240.0
REF_DWELL_US = 12.0
BASE_FOV_NM = 500.0
BASE_DWELL_US = 1.0
TEMPLATE_FRAC = 0.40
MAX_STEPS = 6
EPISODES = [(f, s) for f in ("dram", "finfet") for s in range(4)]

OFFSET_TOL_PX = 1e-3  # same equality bar as the COOL pip-vs-COOL check


def _pairs(limit: int | None):
    for tier in TIERS:
        d = DATASET_ROOT / tier
        pairs = json.loads((d / "manifest.json").read_text())["pairs"]
        for p in pairs[:limit] if limit else pairs:
            pid = p["pair_id"]
            yield (
                f"{tier}/{pid:04d}",
                (d / f"pair_{pid:04d}_ref.png").read_bytes(),
                (d / f"pair_{pid:04d}_search.png").read_bytes(),
                p,
            )


def _hit(x: float, y: float, p: dict) -> bool:
    return math.hypot(x + p["ref_w"] / 2.0 - p["gt_x"], y + p["ref_h"] / 2.0 - p["gt_y"]) <= THR_PX


def dump(out: Path, limit: int | None) -> int:
    import cv2
    import numpy as np

    from driftsense.agent.runner import run_episode
    from driftsense.api import localize
    from driftsense.contracts import Budget, Mode
    from driftsense.sim import VirtualSEM

    rows = []
    budget = Budget(max_candidates=20)
    for pid, tb, sb, p in _pairs(limit):
        for mode in (Mode.FAST, Mode.AUTO):
            r = localize(tb, sb, mode, budget, engine="v2")
            c0 = r.candidates[0]
            rows.append(
                {
                    "kind": "pair",
                    "id": f"{pid}:{mode.value}",
                    "x": r.offset.x_px,
                    "y": r.offset.y_px,
                    "c0": [c0.x_px, c0.y_px],
                    "margin": r.margin.margin,
                    "decision": r.margin.decision,
                    "hit": _hit(c0.x_px, c0.y_px, p),
                }
            )
    for family, seed in EPISODES:
        sem = VirtualSEM(seed=seed, family=family, site_size_nm=SITE_NM, pixel_nm=PIXEL_NM)
        r = run_episode(
            sem,
            tau=TAU,
            run_id=f"{family}-{seed}",
            max_steps=MAX_STEPS,
            ref_fov_nm=REF_FOV_NM,
            ref_dwell_us=REF_DWELL_US,
            base_fov_nm=BASE_FOV_NM,
            base_dwell_us=BASE_DWELL_US,
            template_frac=TEMPLATE_FRAC,
        )
        rows.append(
            {
                "kind": "episode",
                "id": f"{family}-{seed}",
                "outcome": r.outcome,
                "steps": r.steps,
                "final_margin": r.final_margin,
                "estimate_nm": list(r.estimate_nm) if r.estimate_nm else None,
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "kind": "env",
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "arch": platform.machine(),
        "cv2_file": cv2.__file__,
        "created_at": datetime.now(UTC).isoformat(),
    }
    out.write_text("\n".join(json.dumps(r) for r in [header, *rows]) + "\n")
    print(f"wrote {len(rows)} rows ({header['opencv']}) -> {out}")
    return 0


def _load(path: Path) -> tuple[dict, dict[str, dict]]:
    lines = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    env = next(r for r in lines if r["kind"] == "env")
    return env, {r["id"]: r for r in lines if r["kind"] != "env"}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def compare(a_path: Path, b_path: Path, out: Path) -> int:
    from driftsense.contracts import EnvInfo, Metric, ResultsFile

    env_a, a = _load(a_path)
    env_b, b = _load(b_path)
    if a.keys() != b.keys():
        raise SystemExit("dumps cover different pairs/episodes; rerun both with the same --limit")
    pairs = [k for k in a if a[k]["kind"] == "pair"]
    eps = [k for k in a if a[k]["kind"] == "episode"]

    def offset_d(k: str) -> float:
        return math.hypot(a[k]["x"] - b[k]["x"], a[k]["y"] - b[k]["y"])

    d_off = [offset_d(k) for k in pairs]
    d_margin = [abs(a[k]["margin"] - b[k]["margin"]) for k in pairs]
    same_off = sum(d <= OFFSET_TOL_PX for d in d_off)
    same_dec = sum(a[k]["decision"] == b[k]["decision"] for k in pairs)
    same_hit = sum(a[k]["hit"] == b[k]["hit"] for k in pairs)
    hits_a = sum(a[k]["hit"] for k in pairs)
    hits_b = sum(b[k]["hit"] for k in pairs)
    same_outcome = sum(a[k]["outcome"] == b[k]["outcome"] for k in eps)
    same_steps = sum(a[k]["steps"] == b[k]["steps"] for k in eps)

    n, ne = len(pairs), len(eps)
    metrics = {
        "pairs.offset_equal_rate": Metric(value=same_off / n, unit="rate", n=n),
        "pairs.offset_max_abs_diff_px": Metric(value=max(d_off), unit="px", n=n),
        "pairs.offset_within_0p1px_rate": Metric(value=sum(d <= 0.1 for d in d_off) / n, unit="rate", n=n),
        "pairs.offset_diff_gt_1px.n": Metric(value=sum(d > 1.0 for d in d_off), unit="count", n=n),
        "pairs.offset_diff_gt_1px_both_wrong.n": Metric(
            value=sum(
                d > 1.0 and not a[k]["hit"] and not b[k]["hit"] for d, k in zip(d_off, pairs, strict=True)
            ),
            unit="count",
            n=n,
        ),
        "pairs.margin_median_abs_diff": Metric(value=statistics.median(d_margin), unit="margin", n=n),
        "pairs.margin_max_abs_diff": Metric(value=max(d_margin), unit="margin", n=n),
        "pairs.decision_agreement": Metric(value=same_dec / n, unit="rate", n=n),
        "pairs.top1_hit_agreement": Metric(value=same_hit / n, unit="rate", n=n),
        "pairs.top1_hit_rate.base": Metric(value=hits_a / n, unit="rate", n=n),
        "pairs.top1_hit_rate.cand": Metric(value=hits_b / n, unit="rate", n=n),
        "episodes.outcome_agreement": Metric(value=same_outcome / ne, unit="rate", n=ne),
        "episodes.steps_agreement": Metric(value=same_steps / ne, unit="rate", n=ne),
        "offset_tol_px": Metric(value=OFFSET_TOL_PX, unit="px"),
    }
    rf = ResultsFile(
        name=f"cv_equivalence_{env_a['opencv']}_vs_{env_b['opencv']}",
        produced_by="bench/cv_equivalence.py",
        git_sha=_git_sha(),
        created_at=datetime.now(UTC),
        dataset="dev",
        env=EnvInfo(
            python=env_b["python"],
            opencv=env_b["opencv"],
            numpy=env_b["numpy"],
            platform=env_b["platform"],
            arch=env_b["arch"],
            cool="/opt/cool" in env_b["cv2_file"],
        ),
        metrics=metrics,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rf.model_dump_json(indent=2) + "\n")
    side = out.with_suffix(".envs.json")
    side.write_text(json.dumps({"base": env_a, "cand": env_b}, indent=2) + "\n")
    print(json.dumps({k: m.value for k, m in metrics.items()}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--limit", type=int, default=None, help="pairs per tier (default: all)")
    c = sub.add_parser("compare")
    c.add_argument("base", type=Path)
    c.add_argument("cand", type=Path)
    c.add_argument("--out", type=Path, default=ROOT / "results" / "bench" / "cv5_equivalence.json")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    rc = dump(args.out, args.limit) if args.cmd == "dump" else compare(args.base, args.cand, args.out)
    print(f"done in {time.perf_counter() - t0:.1f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
