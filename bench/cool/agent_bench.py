"""Agent workload: active-perception episodes per hour on the VirtualSEM digital twin.

One episode is ``driftsense.agent.runner.run_episode`` over a fresh
``driftsense.sim.VirtualSEM(seed, family)``. It covers a reference acquire, low-dwell
search acquires, and v2 localize calls (``engine="v2"``, auto then accurate) until the
runner commits or escalates. That is the unit of work the service bills.

The operating point is copied, not imported, from ``eval/agent_eval.py`` lines 57-67 at
commit 46ff2f5 (the G5 evaluation), so the bench never imports ``eval/``. Episodes are
deterministic from ``(seed, family)``. Every arm runs the same episodes.

Phase 1, latency: sequential episodes at ``--latency-threads`` (default all vCPUs).
Each episode's final estimate goes to ``agent_offsets.csv`` for the cross-arm
equality check. Phase 2, throughput: one process per vCPU with cv2 threads=1, draining
``--jobs`` episodes per trial.

Keys: ``agent.latency.<tag>.{median_ms,p50_ms,p95_ms}``, ``agent.throughput.episodes_per_hour``,
``agent.steps_mean``, ``check.agent.success_rate`` and ``check.agent.commit_rate``.

Usage::

    python -m bench.cool.agent_bench --out-dir out/run/arm-cool
    python -m bench.cool.agent_bench --quick --out-dir /tmp/agent
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench.cool import common as C

# ---- copied from eval/agent_eval.py:57-67 (commit 46ff2f5); keep in sync by hand ----
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


def episodes(n_seeds: int) -> list[tuple[int, str]]:
    """Deterministic (seed, family) list: seeds 0..n-1 for each family, family-interleaved."""
    return [(s, f) for s in range(n_seeds) for f in FAMILIES]


def run_one(seed: int, family: str) -> dict:
    """One graded episode (grading as in eval/agent_eval.py ``_agent``, via the evaluator hooks)."""
    from driftsense.agent.runner import run_episode
    from driftsense.sim import VirtualSEM

    sem = VirtualSEM(seed=seed, family=family, site_size_nm=SITE_NM, pixel_nm=PIXEL_NM)
    st: dict[str, np.ndarray] = {}

    def on_ref(_a: int) -> None:
        v = sem.verify()
        st["e0"] = np.array([v["error_x_nm"], v["error_y_nm"]])

    def on_dec(_a: int, est: tuple[float, float]) -> None:
        v = sem.verify()
        st["change"] = np.array([v["error_x_nm"], v["error_y_nm"]]) - st["e0"]
        st["est"] = np.array(est)

    t0 = time.perf_counter_ns()
    r = run_episode(
        sem,
        tau=TAU,
        run_id=f"bench-{family}-{seed}",
        max_steps=MAX_STEPS,
        ref_fov_nm=REF_FOV_NM,
        ref_dwell_us=REF_DWELL_US,
        base_fov_nm=BASE_FOV_NM,
        base_dwell_us=BASE_DWELL_US,
        template_frac=TEMPLATE_FRAC,
        on_reference=on_ref,
        on_decision=on_dec,
    )
    wall = (time.perf_counter_ns() - t0) / 1e6
    committed = r.outcome == "commit"
    err = float(np.linalg.norm(st["est"] - st["change"])) if "est" in st else None
    est = r.estimate_nm
    return {
        "seed": seed,
        "family": family,
        "outcome": r.outcome,
        "steps": r.steps,
        "wall_ms": wall,
        "success": bool(committed and err is not None and err <= TOL_NM),
        "est_x_nm": None if est is None else float(est[0]),
        "est_y_nm": None if est is None else float(est[1]),
        "final_margin": r.final_margin,
    }


# ---------------------------------------------------------------- worker side
_W: dict = {}


def _init_worker(n_seeds: int) -> None:
    import cv2

    cv2.setNumThreads(1)
    _W["episodes"] = episodes(n_seeds)
    run_one(*_W["episodes"][0])  # warm-up


def _warm(_: int) -> int:
    time.sleep(0.05)
    return os.getpid()


def _job(i: int) -> float:
    s, f = _W["episodes"][i % len(_W["episodes"])]
    return run_one(s, f)["wall_ms"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--seeds", type=int, default=None, help="seeds per family (default 8; quick 1)")
    ap.add_argument("--latency-threads", default="all")
    ap.add_argument(
        "--workers", type=int, default=None, help="process-pool size (default os.cpu_count(); quick <=2)"
    )
    ap.add_argument(
        "--jobs", type=int, default=None, help="episodes per throughput trial (default 4*workers)"
    )
    ap.add_argument("--trials", type=int, default=None, help="throughput trials (default 3; quick 1)")
    ap.add_argument("--no-throughput", action="store_true")
    a = ap.parse_args(argv)

    warmup, _runs = C.resolve_counts(a)
    n_seeds = a.seeds or (1 if a.quick else 8)
    ncpu = os.cpu_count() or 1
    workers = a.workers or (min(ncpu, 2) if a.quick else ncpu)
    jobs = a.jobs or (workers if a.quick else 4 * workers)
    trials = a.trials or (1 if a.quick else 3)
    C.banner("agent")
    eps = episodes(n_seeds)
    lt = C.resolve_threads(a.latency_threads)
    tag = C.thread_tag(lt)
    C.set_cv_threads(lt)
    for _ in range(min(warmup, 1)):  # an episode is several localize calls: one warm-up is enough
        run_one(*eps[0])
    print(
        f"[agent] latency: {len(eps)} episodes (seeds 0..{n_seeds - 1} x {FAMILIES}), threads={lt}",
        flush=True,
    )
    recs = [run_one(s, f) for s, f in eps]
    lat = [r["wall_ms"] for r in recs]
    metrics: dict = {
        "meta.threads_N": C.metric(ncpu, "threads"),
        "meta.episodes": C.metric(len(eps), "episodes"),
        "meta.quick": C.metric(1.0 if a.quick else 0.0, "bool"),
    }
    metrics.update(C.timing_metrics(f"agent.latency.{tag}", lat))
    s = C.summarize(lat)
    metrics[f"agent.latency.{tag}.p50_ms"] = C.metric(s["p50"], "ms", n=s["n"])
    metrics["agent.steps_mean"] = C.metric(
        float(np.mean([r["steps"] for r in recs])), "acquisitions", n=len(recs)
    )
    n = len(recs)
    metrics["check.agent.success_rate"] = C.metric(sum(r["success"] for r in recs) / n, "fraction", n=n)
    metrics["check.agent.commit_rate"] = C.metric(
        sum(r["outcome"] == "commit" for r in recs) / n, "fraction", n=n
    )
    print(
        f"  agent.latency.{tag}: p50 {s['p50']:.0f} ms p95 {s['p95']:.0f} ms; success "
        f"{sum(r['success'] for r in recs)}/{n}"
    )
    rows = [["latency", tag, lt, f"{r['family']}-{r['seed']}", 0, f"{r['wall_ms']:.6f}"] for r in recs]
    trial_rows: list[list] = []
    if not a.no_throughput:
        mod = importlib.import_module("bench.cool.agent_bench")
        print(
            f"[agent] throughput: {workers} workers x cv2 threads=1, {jobs} episodes x {trials} trials",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=workers, initializer=mod._init_worker, initargs=(n_seeds,)
        ) as ex:
            list(ex.map(mod._warm, range(workers * 2)))
            for t in range(trials):
                with C.CpuMonitor() as mon:
                    t0 = time.perf_counter()
                    res = list(ex.map(mod._job, range(jobs), chunksize=1))
                    wall = time.perf_counter() - t0
                eph = jobs / wall * 3600.0
                trial_rows.append(
                    [
                        t,
                        workers,
                        jobs,
                        f"{wall:.6f}",
                        f"{eph:.3f}",
                        "" if mon.system_pct is None else f"{mon.system_pct:.2f}",
                    ]
                )
                rows += [["loaded", "t1", 1, j, t, f"{w:.6f}"] for j, w in enumerate(res)]
                print(f"  trial {t}: {jobs} episodes / {wall:.2f} s = {eph:,.0f} episodes/h", flush=True)
        eph_all = [float(r[4]) for r in trial_rows]
        med, lo, hi = C.bootstrap_median_ci(eph_all)
        metrics["agent.throughput.episodes_per_hour"] = C.metric(med, "episodes/h", lo, hi, len(eph_all))
        metrics["agent.throughput.workers"] = C.metric(workers, "processes")
        cpu = [float(r[5]) for r in trial_rows if r[5] != ""]
        if cpu:
            metrics["agent.throughput.cpu_pct"] = C.metric(
                sorted(cpu)[len(cpu) // 2], "%", min(cpu), max(cpu), len(cpu)
            )
    out: Path = a.out_dir
    C.write_csv(out / "agent_samples.csv", ["phase", "tag", "threads", "episode_or_job", "trial", "ms"], rows)
    C.write_csv(
        out / "agent_offsets.csv",
        ["episode", "outcome", "steps", "est_x_nm", "est_y_nm", "success"],
        [
            [
                f"{r['family']}-{r['seed']}",
                r["outcome"],
                r["steps"],
                "" if r["est_x_nm"] is None else repr(r["est_x_nm"]),
                "" if r["est_y_nm"] is None else repr(r["est_y_nm"]),
                int(r["success"]),
            ]
            for r in recs
        ],
    )
    if trial_rows:
        C.write_csv(
            out / "agent_throughput.csv",
            ["trial", "workers", "jobs", "wall_s", "episodes_per_hour", "cpu_pct"],
            trial_rows,
        )
    C.write_results(out / "agent.json", "agent.v2", metrics, C.produced_by(__file__))
    print(f"[agent] wrote {out / 'agent.json'} (+ agent_samples.csv, agent_offsets.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
