"""Per-stage timings of the v2 engine, from ``LocalizeResult.timings``.

Runs ``driftsense.api.localize(template, search, mode, engine="v2")`` over K committed DEV
pairs (``bench.cool.pairs``, round-robin over the four tiers) for each mode and
``cv2.setNumThreads`` value. It aggregates each stage's ``ms`` (median with a bootstrap
95% CI, and p95), plus ``total_ms`` and the externally measured wall time.

The stage names are the ones the engine emits (``driftsense/v2/engine.py``, ``_Clock.mark``):
ingest, lattice, rectify, roughness, score, cascade, subpixel, explain. Any other name
that shows up is kept and reported, never dropped.

Keys: ``stage.<mode>.<stage>.<t1|tN>.{median_ms,p95_ms}`` (``stage`` also takes ``total`` and
``wall``), ``check.<mode>.top1_rate.<tag>`` (the template centre within 3 px of the DEV
truth, the threshold in eval/gate_g2.py) and ``check.<mode>.escalated_rate.<tag>``.

Usage::

    python -m bench.cool.stagebench --out-dir out/run/arm-cool          # 30 pairs, fast + auto
    python -m bench.cool.stagebench --quick --threads 1 --out-dir /tmp/sb
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from bench.cool import common as C
from bench.cool.pairs import load_dev_pairs

CSV_HEADER = ["mode", "tag", "threads", "pair", "repeat", "stage", "ms"]
ENGINE_STAGES = ("ingest", "lattice", "rectify", "roughness", "score", "cascade", "subpixel", "explain")


def stage_key(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()) or "unnamed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--engine", default="v2", help="engine for driftsense.api.localize (default v2)")
    ap.add_argument(
        "--modes", nargs="+", default=["fast", "auto"], help="driftsense Mode values (default fast auto)"
    )
    ap.add_argument(
        "--pairs", type=int, default=None, help="K DEV pairs (default = --runs, i.e. 30; quick 4)"
    )
    ap.add_argument("--repeats", type=int, default=1, help="localize calls per pair (default 1)")
    ap.add_argument(
        "--threads", nargs="+", default=["1", "all"], help="cv2.setNumThreads values (default: 1 all)"
    )
    a = ap.parse_args(argv)

    from driftsense.api import localize
    from driftsense.contracts import Budget, Mode

    warmup, runs = C.resolve_counts(a)
    k = a.pairs if a.pairs is not None else (4 if a.quick else runs)
    threads = sorted({C.resolve_threads(t) for t in a.threads})
    C.banner("stagebench")
    pairs = load_dev_pairs(k)
    print(
        f"[stagebench] engine={a.engine} modes={a.modes} pairs={k} repeats={a.repeats} warmup {warmup} "
        f"threads {threads}",
        flush=True,
    )

    metrics: dict = {
        "meta.threads_N": C.metric(C.resolve_threads("all"), "threads"),
        "meta.pairs": C.metric(k, "pairs"),
        "meta.repeats": C.metric(a.repeats, "calls"),
        "meta.warmup": C.metric(warmup, "calls"),
        "meta.quick": C.metric(1.0 if a.quick else 0.0, "bool"),
    }
    rows: list[list] = []
    engines: set[str] = set()
    warnings: set[str] = set()
    stages_seen: dict[str, list[str]] = {}
    budget = Budget()
    for mode_name in a.modes:
        mode = Mode(mode_name)
        for n in threads:
            C.set_cv_threads(n)
            tag = C.thread_tag(n)
            for _ in range(warmup):
                localize(pairs[0].template_png, pairs[0].search_png, mode, budget, engine=a.engine)
            per_stage: dict[str, list[float]] = defaultdict(list)
            hits = calls = escalated = 0
            with C.CpuMonitor() as mon:
                for p in pairs:
                    for r in range(a.repeats):
                        t0 = time.perf_counter_ns()
                        res = localize(p.template_png, p.search_png, mode, budget, engine=a.engine)
                        wall = (time.perf_counter_ns() - t0) / 1e6
                        calls += 1
                        engines.add(f"{res.engine}:{res.engine_version}")
                        warnings.update(res.warnings)
                        escalated += any(w.startswith("cascade: escalated") for w in res.warnings)
                        names = []
                        for st in res.timings:
                            sk = stage_key(st.stage)
                            names.append(sk)
                            per_stage[sk].append(st.ms)
                            rows.append([mode_name, tag, n, p.key, r, sk, f"{st.ms:.6f}"])
                        stages_seen.setdefault(mode_name, names)
                        per_stage["total"].append(res.total_ms)
                        per_stage["wall"].append(wall)
                        rows.append([mode_name, tag, n, p.key, r, "total", f"{res.total_ms:.6f}"])
                        rows.append([mode_name, tag, n, p.key, r, "wall", f"{wall:.6f}"])
                        hits += p.hit(res.offset.x_px, res.offset.y_px)
            for st, vals in per_stage.items():
                metrics.update(C.timing_metrics(f"stage.{mode_name}.{st}.{tag}", vals))
                s = C.summarize(vals)
                print(
                    f"  stage.{mode_name}.{st}.{tag:<4} {s['median']:9.3f} ms [{s['ci_low']:.3f}, "
                    f"{s['ci_high']:.3f}] p95 {s['p95']:.3f} n={s['n']}"
                )
            metrics[f"check.{mode_name}.top1_rate.{tag}"] = C.metric(
                hits / max(1, calls), "fraction", n=calls
            )
            metrics[f"check.{mode_name}.escalated_rate.{tag}"] = C.metric(
                escalated / max(1, calls), "fraction", n=calls
            )
            if mon.system_pct is not None:
                metrics[f"meta.cpu_pct.{mode_name}.{tag}"] = C.metric(mon.system_pct, "%")
            print(f"  check.{mode_name}.top1_rate.{tag} = {hits}/{calls} (centre within 3 px of DEV truth)")

    eng = sorted(engines)
    name = "stagebench." + ("+".join(sorted({e.split(":")[0] for e in eng})) or "unknown")
    unknown = sorted({s for v in stages_seen.values() for s in v} - set(ENGINE_STAGES))
    if unknown:
        print(f"[stagebench] NOTE: stages not in the documented list: {unknown}", file=sys.stderr)
    out: Path = a.out_dir
    C.write_csv(out / "stagebench_samples.csv", CSV_HEADER, rows)
    C.write_json(
        out / "stagebench_meta.json",
        {
            "engines": eng,
            "warnings": sorted(warnings),
            "modes": a.modes,
            "threads": threads,
            "pairs": [p.key for p in pairs],
            "stages": stages_seen,
            "unknown_stages": unknown,
        },
    )
    C.write_results(out / "stagebench.json", name, metrics, C.produced_by(__file__))
    print(f"[stagebench] wrote {out / 'stagebench.json'} ({name}; engines {eng})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
