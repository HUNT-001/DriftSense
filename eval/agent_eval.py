"""G5 agent evaluation: four arms over VirtualSEM episodes, written to JSON.

Arms
----
  * ``v1_single``  -- one acquire + one v1 localize, commit iff margin >= tau.
  * ``v2_single``  -- one acquire + one v2 localize, commit iff margin >= tau.
  * ``agent_rules``-- full active-perception runner, offline rules policy.
  * ``agent_llm``  -- same runner with the LLM planner path (``--planner``;
                      uses an offline deterministic stand-in so it is
                      reproducible without a network).

Each episode shares an IDENTICAL first search frame between the single-shot arms
and the agent's bootstrap (VirtualSEM is deterministic from its seed), so the
single-shot arms see exactly the observation the agent started from. Grading uses
the hidden ``verify()`` truth (evaluator only): a drift *change* estimate within
``TOL_NM`` of the true change since the reference is correct.

Metrics (with Wilson 95% CIs on the rates): success, false-commit rate,
escalation rate, mean steps (acquisitions), mean dose, mean wall time. The C2
recovery count is episodes the agent commits correctly where v2 single-shot fell
below tau (would escalate).

Run ONCE; the JSON is the source of truth and is read back by docs/tests:

    python eval/agent_eval.py --n 24 --out results/agent/g5_episode.json --demos
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from driftsense.agent.policy_llm import echo_planner
from driftsense.agent.runner import run_episode
from driftsense.api import localize
from driftsense.contracts import (
    Budget,
    EnvInfo,
    Metric,
    Mode,
    ResultsFile,
)
from driftsense.sim import VirtualSEM

# -- Operating point (periodic-ambiguity regime; see module docstring / README) --
TAU = 0.10
TOL_NM = 4.0
FAMILIES = ("dram", "finfet")
SITE_NM = 2000.0
PIXEL_NM = 1.0
REF_FOV_NM = 240.0
REF_DWELL_US = 12.0
BASE_FOV_NM = 500.0
BASE_DWELL_US = 1.0
TEMPLATE_FRAC = 0.40
MAX_STEPS = 6

ARMS = ("v1_single", "v2_single", "agent_rules")


@dataclass
class ArmRecord:
    committed: bool
    escalated: bool
    success: bool
    false_commit: bool
    steps: int
    dose: float
    wall_ms: float
    margin: float | None


def _sem(seed: int, family: str, **kw) -> VirtualSEM:
    return VirtualSEM(seed=seed, family=family, site_size_nm=SITE_NM, pixel_nm=PIXEL_NM, **kw)


def _crop_template(ref_png: bytes) -> tuple[bytes, int, int]:
    ref = cv2.imdecode(np.frombuffer(ref_png, np.uint8), cv2.IMREAD_GRAYSCALE)
    h, w = ref.shape[:2]
    th, tw = max(16, int(h * TEMPLATE_FRAC)), max(16, int(w * TEMPLATE_FRAC))
    crop = ref[(h - th) // 2 : (h - th) // 2 + th, (w - tw) // 2 : (w - tw) // 2 + tw]
    _, buf = cv2.imencode(".png", crop)
    return buf.tobytes(), th, tw


def _single_shot(engine: str, seed: int, family: str) -> tuple[ArmRecord, float]:
    """One-shot arm on the identical first search frame; returns (record, margin)."""
    sem = _sem(seed, family)
    ref_png, _ = sem.acquire((0.0, 0.0), REF_FOV_NM, REF_DWELL_US)
    v = sem.verify()
    e0 = np.array([v["error_x_nm"], v["error_y_nm"]])
    tmpl, th, tw = _crop_template(ref_png)
    search_png, meta = sem.acquire((0.0, 0.0), BASE_FOV_NM, BASE_DWELL_US)
    v = sem.verify()
    change = np.array([v["error_x_nm"], v["error_y_nm"]]) - e0

    t0 = time.perf_counter()
    res = localize(tmpl, search_png, Mode.AUTO, Budget(max_candidates=5, tau=TAU), engine=engine)
    wall_ms = (time.perf_counter() - t0) * 1e3

    w, h, pix = meta["width_px"], meta["height_px"], meta["pixel_nm"]
    feat = np.array([res.offset.x_px + tw / 2.0, res.offset.y_px + th / 2.0])
    est = (np.array([w / 2.0, h / 2.0]) - feat) * pix
    err = float(np.linalg.norm(est - change))
    committed = res.margin.margin >= TAU
    ok = committed and err <= TOL_NM
    return (
        ArmRecord(
            committed=committed,
            escalated=not committed,
            success=ok,
            false_commit=committed and err > TOL_NM,
            steps=sem.n_acquisitions,
            dose=sem.total_dose,
            wall_ms=wall_ms,
            margin=res.margin.margin,
        ),
        res.margin.margin,
    )


def _agent(
    seed: int,
    family: str,
    *,
    planner=None,
    trace_path: str | None = None,
    max_steps: int = MAX_STEPS,
    **kw,
) -> ArmRecord:
    sem = _sem(seed, family)
    st: dict[str, np.ndarray] = {}

    def on_ref(_a: int) -> None:
        v = sem.verify()
        st["e0"] = np.array([v["error_x_nm"], v["error_y_nm"]])

    def on_dec(_a: int, est: tuple[float, float]) -> None:
        v = sem.verify()
        st["change"] = np.array([v["error_x_nm"], v["error_y_nm"]]) - st["e0"]
        st["est"] = np.array(est)

    r = run_episode(
        sem,
        tau=TAU,
        run_id=f"{family}-{seed}",
        max_steps=max_steps,
        ref_fov_nm=REF_FOV_NM,
        ref_dwell_us=REF_DWELL_US,
        base_fov_nm=BASE_FOV_NM,
        base_dwell_us=BASE_DWELL_US,
        template_frac=TEMPLATE_FRAC,
        planner=planner,
        trace_path=trace_path,
        on_reference=on_ref,
        on_decision=on_dec,
        **kw,
    )
    committed = r.outcome == "commit"
    err = float(np.linalg.norm(st["est"] - st["change"])) if "est" in st else None
    ok = committed and err is not None and err <= TOL_NM
    return ArmRecord(
        committed=committed,
        escalated=r.outcome in ("escalate", "abort"),
        success=ok,
        false_commit=committed and (err is None or err > TOL_NM),
        steps=r.steps,
        dose=r.dose,
        wall_ms=r.wall_ms,
        margin=r.final_margin,
    )


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate_metric(k: int, n: int) -> Metric:
    lo, hi = _wilson(k, n)
    return Metric(value=k / n if n else 0.0, unit="rate", ci_low=lo, ci_high=hi, n=n)


def _mean_metric(values: list[float], unit: str) -> Metric:
    n = len(values)
    return Metric(value=float(np.mean(values)) if n else 0.0, unit=unit, n=n)


def run(n_seeds: int, with_llm: bool) -> dict:
    records: dict[str, list[ArmRecord]] = {a: [] for a in ARMS}
    if with_llm:
        records["agent_llm"] = []
    v2_margins: list[float] = []
    agent_ok: list[bool] = []

    for family in FAMILIES:
        for seed in range(n_seeds):
            r1, _ = _single_shot("v1", seed, family)
            r2, m2 = _single_shot("v2", seed, family)
            ra = _agent(seed, family)
            records["v1_single"].append(r1)
            records["v2_single"].append(r2)
            records["agent_rules"].append(ra)
            v2_margins.append(m2)
            agent_ok.append(ra.success)
            if with_llm:
                records["agent_llm"].append(_agent(seed, family, planner=echo_planner()))

    metrics: dict[str, Metric] = {}
    n = len(records["agent_rules"])
    metrics["n_episodes"] = Metric(value=float(n), unit="episodes", n=n)
    for arm, recs in records.items():
        metrics[f"{arm}.success"] = _rate_metric(sum(r.success for r in recs), n)
        metrics[f"{arm}.false_commit"] = _rate_metric(sum(r.false_commit for r in recs), n)
        metrics[f"{arm}.escalation"] = _rate_metric(sum(r.escalated for r in recs), n)
        metrics[f"{arm}.mean_steps"] = _mean_metric([float(r.steps) for r in recs], "acquisitions")
        metrics[f"{arm}.mean_dose"] = _mean_metric([r.dose for r in recs], "electrons")
        metrics[f"{arm}.mean_wall_ms"] = _mean_metric([r.wall_ms for r in recs], "ms")

    # C2 recovery: v2 single-shot below tau (would escalate) but agent commits correctly.
    v2_below = [i for i, m in enumerate(v2_margins) if m < TAU]
    recovered = sum(1 for i in v2_below if agent_ok[i])
    metrics["c2_recovery.count"] = Metric(value=float(recovered), unit="episodes", n=len(v2_below))
    return {"metrics": metrics, "records": records}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _env() -> EnvInfo:
    return EnvInfo(
        python=platform.python_version(),
        opencv=cv2.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
        arch=platform.machine(),
    )


def _protocol_sha() -> str:
    cfg = {
        "tau": TAU,
        "tol_nm": TOL_NM,
        "families": FAMILIES,
        "site_nm": SITE_NM,
        "ref_fov_nm": REF_FOV_NM,
        "ref_dwell_us": REF_DWELL_US,
        "base_fov_nm": BASE_FOV_NM,
        "base_dwell_us": BASE_DWELL_US,
        "template_frac": TEMPLATE_FRAC,
        "max_steps": MAX_STEPS,
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def write_results(metrics: dict[str, Metric], out: Path) -> None:
    rf = ResultsFile(
        name="g5_episode",
        produced_by="eval/agent_eval.py",
        git_sha=_git_sha(),
        created_at=dt.datetime.now(dt.UTC),
        dataset="synthetic",
        env=_env(),
        metrics=metrics,
        protocol_sha256=_protocol_sha(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rf.model_dump_json(indent=2) + "\n", encoding="utf-8")


def print_table(path: str | Path) -> None:
    """Print the four-arm table straight from the results JSON (no recompute)."""
    rf = ResultsFile.model_validate_json(Path(path).read_text(encoding="utf-8"))
    m = rf.metrics
    arms = [a for a in ("v1_single", "v2_single", "agent_rules", "agent_llm") if f"{a}.success" in m]
    n = int(m["n_episodes"].value)
    print(f"\nG5 four-arm evaluation  (N={n} episodes, tau={TAU}, tol={TOL_NM} nm)")
    print(f"{'arm':<13}{'success':>18}{'false-commit':>18}{'escalation':>18}{'steps':>8}{'dose(M)':>10}")
    for a in arms:
        s, f, e = m[f"{a}.success"], m[f"{a}.false_commit"], m[f"{a}.escalation"]

        def ci(x):
            return f"{x.value:.2f} [{x.ci_low:.2f},{x.ci_high:.2f}]"

        print(
            f"{a:<13}{ci(s):>18}{ci(f):>18}{ci(e):>18}"
            f"{m[f'{a}.mean_steps'].value:>8.1f}{m[f'{a}.mean_dose'].value / 1e6:>10.1f}"
        )
    rec = m["c2_recovery.count"]
    print(
        f"\nC2 recovery: agent committed correctly on {int(rec.value)}/{int(rec.n)} pairs "
        f"where v2 single-shot fell below tau (would escalate)."
    )


def write_demos(n_seeds: int, trace_dir: Path) -> dict:
    """Write one committing and one escalating demo trace (G5 acceptance evidence)."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    commit_seed = esc_seed = None
    for family in FAMILIES:
        for seed in range(n_seeds):
            _, m2 = _single_shot("v2", seed, family)
            if commit_seed is None and m2 >= TAU:
                commit_seed = (seed, family)
            if esc_seed is None and m2 < TAU:
                esc_seed = (seed, family)
        if commit_seed and esc_seed:
            break

    info: dict[str, object] = {}
    if commit_seed:
        seed, family = commit_seed
        r = _agent(seed, family, trace_path=str(trace_dir / "demo_commit.jsonl"))
        info["demo_commit"] = {"seed": seed, "family": family, "outcome": r.committed}
    if esc_seed:
        seed, family = esc_seed
        # Cap the loop to one step so the sub-tau pair escalates to a human.
        r = _agent(seed, family, max_steps=1, trace_path=str(trace_dir / "demo_escalate.jsonl"))
        info["demo_escalate"] = {"seed": seed, "family": family, "escalated": r.escalated}
    else:  # no natural sub-tau pair in range -> force the budget-exhaustion path
        seed, family = (0, "dram")
        r = _agent(seed, family, max_dose=48e6, trace_path=str(trace_dir / "demo_escalate.jsonl"))
        info["demo_escalate"] = {
            "seed": seed,
            "family": family,
            "escalated": r.escalated,
            "forced": "max_dose",
        }
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="G5 agent four-arm evaluation")
    ap.add_argument("--n", type=int, default=24, help="seeds per family")
    ap.add_argument("--planner", action="store_true", help="also run the LLM-planner arm (offline stand-in)")
    ap.add_argument("--out", default="results/agent/g5_episode.json")
    ap.add_argument("--demos", action="store_true", help="also write traces/demo_*.jsonl")
    ap.add_argument("--no-write", action="store_true", help="print only; do not write results JSON")
    args = ap.parse_args()

    out = Path(args.out)
    res = run(args.n, with_llm=args.planner)
    if not args.no_write:
        write_results(res["metrics"], out)
    if args.demos:
        print("demos:", write_demos(args.n, Path("traces")))
    if out.exists():
        print_table(out)


if __name__ == "__main__":
    main()
