"""End-to-end latency (p50/p95 per localize call) and throughput (localizations/hour) of v2.

Phase 1, latency: sequential ``driftsense.api.localize(engine="v2")`` calls over K
committed DEV pairs at ``--latency-threads`` (default: all vCPUs, the single-job latency a
user sees). The per-pair offsets go to ``e2e_offsets.csv`` for the cross-arm equality check.

Phase 2, throughput: a ``ProcessPoolExecutor`` with one worker per vCPU. Each worker pins
``cv2.setNumThreads(1)`` (the cost-optimal batch configuration), loads the same DEV pairs,
and warms up. The pool then drains ``--jobs`` jobs, repeated over ``--trials`` trials.
psutil records system CPU utilisation during each trial.

Both phases run for each ``--modes`` value (default ``fast`` and ``auto``; ``auto`` is the
service default, ``JobCreate.mode``). Keys: ``e2e.<mode>.latency.<tag>.{median_ms,p50_ms,
p95_ms}``, ``e2e.<mode>.throughput.jobs_per_hour`` (CI over trials), ``e2e.<mode>.throughput.
cpu_pct``, ``e2e.<mode>.loaded_latency.t1.*`` and ``check.<mode>.top1_rate``.

Usage::

    python -m bench.cool.e2e --out-dir out/run/arm-cool
    python -m bench.cool.e2e --quick --out-dir /tmp/e2e
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from bench.cool import common as C
from bench.cool.pairs import load_dev_pairs

# ---------------------------------------------------------------- worker side (module-level for spawn)
_W: dict = {}


def _init_worker(engine: str, mode: str, k: int) -> None:
    import cv2

    cv2.setNumThreads(1)
    from driftsense.api import localize
    from driftsense.contracts import Budget, Mode

    _W["localize"] = localize
    _W["mode"] = Mode(mode)
    _W["engine"] = engine
    _W["budget"] = Budget()
    _W["pairs"] = [(p.template_png, p.search_png) for p in load_dev_pairs(k)]
    tb, sb = _W["pairs"][0]
    localize(tb, sb, _W["mode"], _W["budget"], engine=engine)  # per-worker warm-up


def _warm(_: int) -> int:
    time.sleep(0.05)  # keep the pool busy so every worker process is spawned and initialised
    return os.getpid()


def _job(i: int) -> tuple[float, float]:
    """One localize job; returns (wall_ms, process_cpu_ms)."""
    tb, sb = _W["pairs"][i % len(_W["pairs"])]
    c0 = time.process_time()
    t0 = time.perf_counter_ns()
    _W["localize"](tb, sb, _W["mode"], _W["budget"], engine=_W["engine"])
    return (time.perf_counter_ns() - t0) / 1e6, (time.process_time() - c0) * 1e3


# ---------------------------------------------------------------- driver
def run_latency(localize, mode, budget, engine: str, pairs, warmup: int, repeats: int):
    for _ in range(warmup):
        localize(pairs[0].template_png, pairs[0].search_png, mode, budget, engine=engine)
    samples, offsets = [], []
    for p in pairs:
        for r in range(repeats):
            t0 = time.perf_counter_ns()
            res = localize(p.template_png, p.search_png, mode, budget, engine=engine)
            samples.append(((time.perf_counter_ns() - t0) / 1e6, p.key, r))
            if r == 0:
                offsets.append(
                    [
                        mode.value,
                        p.key,
                        repr(float(res.offset.x_px)),
                        repr(float(res.offset.y_px)),
                        int(p.hit(res.offset.x_px, res.offset.y_px)),
                        res.engine,
                        res.engine_version,
                    ]
                )
    return samples, offsets


def run_throughput(engine: str, mode_value: str, k: int, workers: int, jobs: int, trials: int):
    # resolve worker functions by their importable module name (not __main__) so spawn-started
    # children (Windows/macOS, or forced spawn) can unpickle them
    mod = importlib.import_module("bench.cool.e2e")
    trial_rows, loaded = [], []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=mod._init_worker, initargs=(engine, mode_value, k)
    ) as ex:
        pids = set(ex.map(mod._warm, range(workers * 2)))
        for t in range(trials):
            with C.CpuMonitor() as mon:
                t0 = time.perf_counter()
                res = list(ex.map(mod._job, range(jobs), chunksize=1))
                wall = time.perf_counter() - t0
            jph = jobs / wall * 3600.0
            trial_rows.append(
                [
                    mode_value,
                    t,
                    workers,
                    jobs,
                    f"{wall:.6f}",
                    f"{jph:.3f}",
                    "" if mon.system_pct is None else f"{mon.system_pct:.2f}",
                ]
            )
            loaded.extend((w, c, t, j) for j, (w, c) in enumerate(res))
            print(
                f"  [{mode_value}] throughput trial {t}: {jobs} jobs / {wall:.2f} s = {jph:,.0f} jobs/h; "
                f"cpu {mon.system_pct}% ({len(pids)} worker pids seen)",
                flush=True,
            )
    return trial_rows, loaded, len(pids)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--engine", default="v2")
    ap.add_argument("--modes", nargs="+", default=["fast", "auto"])
    ap.add_argument("--pairs", type=int, default=None, help="K DEV pairs (default = --runs; quick 4)")
    ap.add_argument("--repeats", type=int, default=1, help="latency calls per pair")
    ap.add_argument(
        "--latency-threads", default="all", help="cv2 threads for the latency phase (default all)"
    )
    ap.add_argument(
        "--workers", type=int, default=None, help="process-pool size (default os.cpu_count(); quick <=4)"
    )
    ap.add_argument(
        "--jobs", type=int, default=None, help="jobs per throughput trial (default max(240, 15*workers))"
    )
    ap.add_argument("--trials", type=int, default=None, help="throughput trials (default 3; quick 1)")
    ap.add_argument("--no-throughput", action="store_true", help="skip the process-pool phase")
    a = ap.parse_args(argv)

    from driftsense.api import localize
    from driftsense.contracts import Budget, Mode

    warmup, runs = C.resolve_counts(a)
    k = a.pairs if a.pairs is not None else (4 if a.quick else runs)
    ncpu = os.cpu_count() or 1
    workers = a.workers or (min(ncpu, 4) if a.quick else ncpu)
    jobs = a.jobs or (workers * 2 if a.quick else max(240, 15 * workers))
    trials = a.trials or (1 if a.quick else 3)
    C.banner("e2e")
    pairs = load_dev_pairs(k)
    lt = C.resolve_threads(a.latency_threads)
    tag = C.thread_tag(lt)
    metrics: dict = {
        "meta.threads_N": C.metric(ncpu, "threads"),
        "meta.pairs": C.metric(k, "pairs"),
        "meta.quick": C.metric(1.0 if a.quick else 0.0, "bool"),
    }
    sample_rows: list[list] = []
    offsets_all: list[list] = []
    trial_all: list[list] = []
    budget = Budget()
    for mode_name in a.modes:
        mode = Mode(mode_name)
        C.set_cv_threads(lt)
        print(
            f"[e2e] {mode_name} latency: engine={a.engine} pairs={k} repeats={a.repeats} threads={lt}",
            flush=True,
        )
        with C.CpuMonitor() as mon:
            lat, offsets = run_latency(localize, mode, budget, a.engine, pairs, warmup, a.repeats)
        lat_ms = [s[0] for s in lat]
        metrics.update(C.timing_metrics(f"e2e.{mode_name}.latency.{tag}", lat_ms))
        s = C.summarize(lat_ms)
        metrics[f"e2e.{mode_name}.latency.{tag}.p50_ms"] = C.metric(s["p50"], "ms", n=s["n"])
        if mon.system_pct is not None:
            metrics[f"e2e.{mode_name}.latency.{tag}.cpu_pct"] = C.metric(mon.system_pct, "%")
        hits = sum(o[4] for o in offsets)
        metrics[f"check.{mode_name}.top1_rate"] = C.metric(
            hits / max(1, len(offsets)), "fraction", n=len(offsets)
        )
        print(
            f"  e2e.{mode_name}.latency.{tag}: p50 {s['p50']:.2f} ms [{s['ci_low']:.2f}, {s['ci_high']:.2f}] "
            f"p95 {s['p95']:.2f} ms; top-1 {hits}/{len(offsets)}"
        )
        sample_rows += [[mode_name, "latency", tag, lt, key, r, f"{ms:.6f}", ""] for ms, key, r in lat]
        offsets_all += offsets
        if a.no_throughput:
            continue
        print(
            f"[e2e] {mode_name} throughput: {workers} workers x cv2 threads=1, {jobs} jobs x {trials} trials",
            flush=True,
        )
        trial_rows, loaded, npids = run_throughput(a.engine, mode_name, k, workers, jobs, trials)
        jph = [float(r[5]) for r in trial_rows]
        med, lo, hi = C.bootstrap_median_ci(jph)
        pre = f"e2e.{mode_name}.throughput"
        metrics[f"{pre}.jobs_per_hour"] = C.metric(med, "jobs/h", lo, hi, len(jph))
        metrics[f"{pre}.workers"] = C.metric(workers, "processes")
        metrics[f"{pre}.jobs_per_trial"] = C.metric(jobs, "jobs")
        metrics[f"{pre}.worker_pids_seen"] = C.metric(npids, "processes")
        cpu = [float(r[6]) for r in trial_rows if r[6] != ""]
        if cpu:
            metrics[f"{pre}.cpu_pct"] = C.metric(
                sorted(cpu)[len(cpu) // 2], "%", min(cpu), max(cpu), len(cpu)
            )
        metrics.update(C.timing_metrics(f"e2e.{mode_name}.loaded_latency.t1", [x[0] for x in loaded]))
        metrics.update(C.timing_metrics(f"e2e.{mode_name}.loaded_cpu.t1", [x[1] for x in loaded]))
        sample_rows += [[mode_name, "loaded", "t1", 1, j, t, f"{w:.6f}", f"{c:.6f}"] for w, c, t, j in loaded]
        trial_all += trial_rows
        print(f"  {pre}.jobs_per_hour = {med:,.0f} [{lo:,.0f}, {hi:,.0f}] (n={len(jph)} trials)")

    out: Path = a.out_dir
    C.write_csv(
        out / "e2e_samples.csv",
        ["mode", "phase", "tag", "threads", "pair_or_job", "repeat_or_trial", "ms", "cpu_ms"],
        sample_rows,
    )
    C.write_csv(
        out / "e2e_offsets.csv",
        ["mode", "pair", "x_px", "y_px", "top1_hit", "engine", "engine_version"],
        offsets_all,
    )
    if trial_all:
        C.write_csv(
            out / "e2e_throughput.csv",
            ["mode", "trial", "workers", "jobs", "wall_s", "jobs_per_hour", "cpu_pct"],
            trial_all,
        )
    engines = sorted({o[5] for o in offsets_all})
    C.write_results(
        out / "e2e.json", "e2e." + ("+".join(engines) or "unknown"), metrics, C.produced_by(__file__)
    )
    print(f"[e2e] wrote {out / 'e2e.json'} (+ e2e_samples.csv, e2e_offsets.csv, e2e_throughput.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
