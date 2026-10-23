"""Per-op microbenchmark of every OpenCV call the v2 engine makes, at production parameters.

The op table is ``bench.cool.ops`` (one entry per engine call site, with its KleidiCV
group). Arguments are not synthesised. ``bench.cool.census`` runs
``driftsense.api.localize(engine="v2", mode=auto)`` once on the DEV capture pair
(``bench.cool.pairs.CAPTURE_PAIR``) and copies the arguments of the largest call at each
site. This benchmark replays exactly those calls: same dtype, shape, kernel, border and
flags as production.

Protocol: one untimed first call, ``--warmup`` warm-ups (default 5), then ``--runs`` timed
samples (default 30) at each ``cv2.setNumThreads`` value (default ``1`` and all vCPUs,
keys ``t1``/``tN``). Ops faster than ``--min-sample-ms`` are batched so one sample lasts at
least that long, and the per-call time is reported. Median with a bootstrap 95% CI, and p95.

Usage::

    python -m bench.cool.microbench --out-dir out/run/arm-cool
    python -m bench.cool.microbench --quick --threads 1 --out-dir /tmp/mb   # plumbing only
    python -m bench.cool.microbench --list
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from bench.cool import common as C
from bench.cool import ops as O


def _engine_replay(spec: O.OpSpec, cen) -> tuple[Callable[[], Any], dict]:
    import cv2

    st = cen.sites.get(spec.site)
    if st is None or st.capture is None:
        raise LookupError(f"site {spec.site} was not reached while capturing (mode auto, pair {cen_pair()})")
    args, kwargs = st.capture
    fn = getattr(cv2, spec.fn)
    if hasattr(fn, "__wrapped__"):  # never time through the census wrapper
        fn = fn.__wrapped__
    info = {"signature": st.capture_signature, "lines": sorted(st.lines), "capture_calls": st.calls}
    if spec.fn in CATCHES_CV2_ERROR:
        # The engine wraps this call in ``except cv2.error`` (subpixel.py, ECC non-convergence falls
        # back to the phase estimate), so a raising call is a production outcome: time it the same way.
        state = {"raised": 0}

        def guarded() -> Any:
            try:
                return fn(*args, **kwargs)
            except cv2.error:
                state["raised"] += 1
                return None

        info["raises_cv2_error"] = state
        return guarded, info
    return (lambda: fn(*args, **kwargs)), info


CATCHES_CV2_ERROR = frozenset({"findTransformECC"})


def cen_pair() -> str:
    from bench.cool.pairs import CAPTURE_PAIR

    return f"{CAPTURE_PAIR[0]}/{CAPTURE_PAIR[1]:04d}"


def _cap(cen, file: str, func: str, fn: str) -> tuple[tuple, dict]:
    st = cen.sites.get((file, func, fn))
    if st is None or st.capture is None:
        raise LookupError(f"capture missing for {fn} in {file}:{func}")
    return st.capture


def _numpy_rerank() -> tuple[Callable[[], Any], dict]:
    """S6 v1 fingerprint re-rank (pure NumPy) over the engine's 64-candidate pool.

    Uses the clean/0000 DEV pair: it is not de-rotated, so the engine's rectified frame is
    the native frame and the candidate coordinates are valid on it.
    """
    from bench.cool.pairs import load_pair
    from driftsense.api import localize
    from driftsense.contracts import Budget, Mode
    from driftsense.v2.cascade import CascadeConfig, rerank
    from driftsense.v2.engine import _CASCADE_POOL
    from driftsense.v2.ingest import ingest

    p = load_pair("clean", 0)
    res = localize(p.template_png, p.search_png, Mode.FAST, Budget(max_candidates=_CASCADE_POOL), engine="v2")
    pool = list(res.candidates)
    t_img, s_img = ingest(p.template_png).image, ingest(p.search_png).image
    cfg = CascadeConfig(top_k=len(pool))
    return (lambda: rerank(pool, t_img, s_img, mode="v1", cfg=cfg)), {
        "signature": f"pool={len(pool)} template={list(t_img.shape)} search={list(s_img.shape)} pair=clean/0000",
        "lines": [],
    }


def _numpy_residual(cen) -> tuple[Callable[[], Any], dict]:
    """The NumPy element-wise part of roughness.residual on the captured search frame."""
    import cv2

    from driftsense.v2.roughness import RoughnessConfig

    G = "driftsense/v2/roughness.py"
    sa, sk = _cap(cen, G, "residual", "sepFilter2D")
    img = sa[0]
    smooth = cv2.sepFilter2D(*sa, **sk)
    ga, gk = _cap(cen, G, "residual", "Sobel")
    gx = cv2.Sobel(*ga, **gk)
    grad = cv2.magnitude(gx, gx)
    da, dk = _cap(cen, G, "residual", "dilate")
    mask = cv2.dilate(*da, **dk)
    ba, bk = _cap(cen, G, "residual", "GaussianBlur")
    cfg = RoughnessConfig()
    r0 = (img - smooth) * mask  # the two GaussianBlur calls belong to the kleidicv op: done once here
    lm = cv2.GaussianBlur(r0, *ba[1:], **bk)
    le = cv2.GaussianBlur(r0 * r0, *ba[1:], **bk)

    def run() -> np.ndarray:
        thr = cfg.edge_thresh_frac * float(np.percentile(grad, 99.0))
        _edge = (grad > thr).astype(np.uint8)
        r = img - smooth
        r = r * mask
        _rr = r * r
        local_std = np.sqrt(np.maximum(le - lm * lm, 0.0))
        return (r / (local_std + cfg.eps)) * mask

    return run, {"signature": f"float32{list(img.shape)}", "lines": [114, 121, 122, 128, 135, 136, 137]}


def _counterfactual(spec: O.OpSpec, cen) -> tuple[Callable[[], Any], dict]:
    import cv2

    G = "driftsense/v2/roughness.py"
    pa, _ = _cap(cen, "driftsense/v2/rectify.py", "rectify", "pyrDown")
    u8 = pa[0]  # the native uint8 search frame
    if u8.dtype != np.uint8:
        raise TypeError(f"expected a uint8 frame, got {u8.dtype}")
    if spec.fn == "GaussianBlur":
        ba, bk = _cap(cen, G, "residual", "GaussianBlur")
        return (lambda: cv2.GaussianBlur(u8, *ba[1:], **bk)), {
            "signature": f"uint8{list(u8.shape)}, {ba[1:]}"
        }
    if spec.fn == "sepFilter2D":
        sa, _ = _cap(cen, G, "residual", "sepFilter2D")
        kx, ky = sa[2], sa[3]
        return (lambda: cv2.sepFilter2D(u8, -1, kx, ky)), {
            "signature": f"uint8{list(u8.shape)}, -1, {kx.shape}, {ky.shape}"
        }
    if spec.fn == "Sobel":
        return (lambda: cv2.Sobel(u8, cv2.CV_16S, 1, 0, ksize=3)), {
            "signature": f"uint8{list(u8.shape)} -> int16"
        }
    raise KeyError(spec.key)


def build(ops: list[O.OpSpec], cen) -> tuple[dict[str, Callable[[], Any]], dict[str, dict]]:
    """Make a zero-arg callable per op from the capture. Errors are recorded, not raised."""
    fns: dict[str, Callable[[], Any]] = {}
    meta: dict[str, dict] = {}
    for o in ops:
        m = {
            "op": o.fn,
            "variant": o.variant,
            "group": o.group,
            "stage": o.stage,
            "file": o.file,
            "func": o.func,
            "note": o.note,
            "modes": list(o.modes),
            "calls": o.calls,
            "engine_call": o.is_engine_call,
            "errors": {},
            "inner": {},
        }
        try:
            if o.is_engine_call:
                fn, info = _engine_replay(o, cen)
            elif o.variant == "s6_v1_fingerprint_rerank":
                fn, info = _numpy_rerank()
            elif o.variant == "s3_residual_elementwise":
                fn, info = _numpy_residual(cen)
            else:
                fn, info = _counterfactual(o, cen)
        except Exception as e:  # noqa: BLE001 - keep the campaign going; recorded per op
            m["errors"]["build"] = f"{type(e).__name__}: {e}"
            meta[o.key] = m
            continue
        m.update(info)
        fns[o.key] = fn
        meta[o.key] = m
    return fns, meta


def run(
    ops: list[O.OpSpec],
    fns: dict,
    meta: dict,
    threads: list[int],
    warmup: int,
    runs: int,
    min_sample_ms: float = 0.0,
    log: Callable[[str], None] = print,
) -> tuple[dict, list[list]]:
    """Time every buildable op at every thread count. Returns (metrics, csv_rows)."""
    import cv2

    metrics: dict = {"meta.threads_N": C.metric(C.resolve_threads("all"), "threads")}
    rows: list[list] = []
    for n in threads:
        actual = C.set_cv_threads(n)
        tag = C.thread_tag(n)
        log(f"[microbench] threads={n} (cv2.getNumThreads()={actual}) tag={tag}")
        with C.CpuMonitor() as mon:
            for o in ops:
                fn = fns.get(o.key)
                if fn is None:
                    continue
                try:
                    fn()  # first call outside timing (lazy init); warm-ups follow
                    inner = C.calibrate_inner(fn, min_sample_ms)
                    meta[o.key]["inner"][tag] = inner
                    ms = C.time_callable(fn, warmup, runs, inner=inner)
                except (cv2.error, AttributeError, TypeError, ValueError, RuntimeError, MemoryError) as e:
                    meta[o.key]["errors"][tag] = f"{type(e).__name__}: {e}"
                    log(f"  ! {o.key} {tag}: {type(e).__name__}: {e}")
                    continue
                s = C.summarize(ms)
                metrics.update(C.timing_metrics(f"{o.key}.{tag}", ms))
                rows.extend(
                    [o.key, o.fn, o.variant, o.group, o.stage, n, tag, inner, r, f"{v:.6f}"]
                    for r, v in enumerate(ms)
                )
                log(
                    f"  {o.key + '.' + tag:<52} {s['median']:9.4f} ms [{s['ci_low']:.4f}, {s['ci_high']:.4f}] "
                    f"p95 {s['p95']:.4f}  {o.group}"
                )
        if mon.system_pct is not None:
            metrics[f"meta.cpu_pct.{tag}"] = C.metric(mon.system_pct, "%")
        if mon.cores_used is not None:
            metrics[f"meta.cores_used.{tag}"] = C.metric(mon.cores_used, "cores")
        metrics[f"meta.wall_s.{tag}"] = C.metric(mon.wall_s, "s")
    return metrics, rows


CSV_HEADER = ["key", "op", "variant", "group", "stage", "threads", "tag", "inner", "run", "ms"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument(
        "--threads",
        nargs="+",
        default=["1", "all"],
        help="cv2.setNumThreads values; 'all' = os.cpu_count() (default: 1 all)",
    )
    ap.add_argument("--ops", default=None, help="regex filter on op keys, e.g. 'GaussianBlur|matchTemplate'")
    ap.add_argument(
        "--min-sample-ms",
        type=float,
        default=1.0,
        help="batch fast ops so each timed sample lasts >= this (default 1.0; 0 disables)",
    )
    ap.add_argument("--no-counterfactual", action="store_true", help="skip the counterfactual uint8 ops")
    ap.add_argument("--list", action="store_true", help="list ops (with engine file:line) and exit")
    a = ap.parse_args(argv)

    ops = list(O.ALL_OPS)
    if a.no_counterfactual:
        ops = [o for o in ops if o.group != O.COUNTERFACTUAL]
    if a.ops:
        rx = re.compile(a.ops)
        ops = [o for o in ops if rx.search(o.key)]
    if a.list:
        for o in ops:
            lines = ",".join(str(x) for x in O.lines_for(o)) or "-"
            print(f"{o.key:<48} {o.group:<14} {o.stage:<3} {o.file}:{lines} ({o.func})  {o.note}")
        return 0
    if not ops:
        print("no ops selected", file=sys.stderr)
        return 2
    warmup, runs = C.resolve_counts(a)
    threads = sorted({C.resolve_threads(t) for t in a.threads})
    C.banner("microbench")
    from bench.cool.census import capture_engine_args, uncovered_sites

    C.set_cv_threads(1)
    cen = capture_engine_args("auto")
    unc = uncovered_sites(cen.sites)
    if unc:
        print(f"[microbench] WARNING: engine cv2 sites not in the op table: {unc}", file=sys.stderr)
    fns, meta = build(ops, cen)
    print(
        f"[microbench] {len(fns)} ops built from the capture of pair {cen_pair()} (mode auto); "
        f"warmup {warmup}, runs {runs}, threads {threads}",
        flush=True,
    )
    metrics, rows = run(ops, fns, meta, threads, warmup, runs, min_sample_ms=a.min_sample_ms)
    metrics["meta.warmup"] = C.metric(warmup, "calls")
    metrics["meta.runs"] = C.metric(runs, "calls")
    metrics["meta.quick"] = C.metric(1.0 if a.quick else 0.0, "bool")
    metrics["meta.ops_built"] = C.metric(len(fns), "ops")
    metrics["meta.uncovered_sites"] = C.metric(len(unc), "sites")

    out: Path = a.out_dir
    C.write_csv(out / "microbench_samples.csv", CSV_HEADER, rows)
    for v in meta.values():
        if isinstance(v.get("raises_cv2_error"), dict):
            v["raises_cv2_error"] = bool(v["raises_cv2_error"]["raised"])
    C.write_json(
        out / "microbench_ops.json",
        {
            "groups": O.GROUP_DOC,
            "ops": meta,
            "threads": threads,
            "capture_pair": cen_pair(),
            "uncovered_sites": unc,
            "kleidicv_listed": sorted(O.KLEIDICV_LISTED),
            "listed_but_unused": list(O.LISTED_BUT_UNUSED),
            "quick": a.quick,
        },
    )
    C.write_results(out / "microbench.json", "microbench", metrics, C.produced_by(__file__))
    print(f"[microbench] wrote {out / 'microbench.json'} (+ microbench_samples.csv, microbench_ops.json)")
    errs = {k: v["errors"] for k, v in meta.items() if v["errors"]}
    if errs:
        print(f"[microbench] {len(errs)} ops raised errors: {errs}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
