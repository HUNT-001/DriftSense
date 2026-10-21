"""Call census of the real v2 engine: which cv2 calls run, how often, with what, for how long.

``Census`` temporarily wraps every cv2 function the v2 package calls (found by an AST scan
of ``driftsense/v2``, see ``bench.cool.ops.engine_cv2_calls``). The wrapper only records a
call whose *direct caller* is a file in ``driftsense/v2``. Any other caller passes straight
through. For each site ``(file, enclosing function, cv2 function)`` it records the call
count, the time spent inside cv2, the argument signature and, with ``capture=True``, a copy
of the arguments of the largest call. The microbench replays those copies, so every timed
op runs with the engine's real dtype, size, kernel and flags.

As a CLI it runs ``localize(engine="v2")`` over K DEV pairs per mode with
``cv2.setNumThreads(1)`` and writes ``census.json`` (ResultsFile) plus ``census_sites.json``:

* ``census.<mode>.<site>.calls_per_localize`` and ``.ms_per_localize`` (median over pairs);
* ``census.<mode>.share.kleidicv`` / ``.control`` / ``.non_cv2``: fractions of
  ``LocalizeResult.total_ms`` spent in KleidiCV-listed cv2 calls, other cv2 calls and
  everything else (NumPy, Python). The KleidiCV share bounds what COOL can do for the
  pipeline (Amdahl), whatever the per-op speed-up.

The wrapper costs about a microsecond per call. Stage and e2e timings are never taken
under the census.

Usage::

    python -m bench.cool.census --out-dir out/run/arm-cool
    python -m bench.cool.census --quick --out-dir /tmp/census
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench.cool import common as C
from bench.cool import ops as O

_ENGINE: list[tuple[str, Path]] = []
_REL_CACHE: dict[str, str | None] = {}


def _engine_location() -> tuple[str, Path]:
    """(lower-case posix prefix of the imported driftsense/v2 dir, its repo root)."""
    if not _ENGINE:
        import driftsense.v2 as v2

        d = Path(v2.__file__).resolve().parent
        _ENGINE.append((d.as_posix().lower() + "/", d.parents[1]))
    return _ENGINE[0]


def _rel(path: str) -> str | None:
    """Repo-relative path if ``path`` is a file of the imported v2 engine, else None."""
    if path in _REL_CACHE:
        return _REL_CACHE[path]
    prefix, root = _engine_location()
    p = Path(path).resolve()
    rel = p.relative_to(root).as_posix() if p.as_posix().lower().startswith(prefix) else None
    _REL_CACHE[path] = rel
    return rel


def _copy(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.copy()
    if isinstance(x, tuple):
        return tuple(_copy(v) for v in x)
    if isinstance(x, list):
        return [_copy(v) for v in x]
    return x


def _nbytes(x: Any) -> int:
    if isinstance(x, np.ndarray):
        return int(x.nbytes)
    if isinstance(x, tuple | list):
        return sum(_nbytes(v) for v in x)
    return 0


def describe(x: Any) -> str:
    """Compact, deterministic signature of an argument (arrays: dtype + shape)."""
    if isinstance(x, np.ndarray):
        return f"{x.dtype}{list(x.shape)}"
    if isinstance(x, tuple | list):
        inner = ",".join(describe(v) for v in x)
        return f"({inner})" if isinstance(x, tuple) else f"[{inner}]"
    if isinstance(x, float):
        return f"{x:.6g}"
    if x is None or isinstance(x, bool | int | str):
        return repr(x)
    return type(x).__name__


@dataclass
class SiteStats:
    site: tuple[str, str, str]
    lines: set[int] = field(default_factory=set)
    calls: int = 0
    ns: int = 0
    signatures: dict[str, int] = field(default_factory=dict)
    capture_bytes: int = -1
    capture: tuple[tuple, dict] | None = None
    capture_signature: str = ""


class Census:
    """Context manager that wraps engine-called cv2 functions (see module docstring)."""

    def __init__(self, capture: bool = False) -> None:
        self.capture = capture
        self.sites: dict[tuple[str, str, str], SiteStats] = {}
        self._orig: dict[str, Any] = {}

    def reset_counts(self) -> None:
        for s in self.sites.values():
            s.calls = 0
            s.ns = 0

    def _wrap(self, name: str, orig: Any) -> Any:
        pc = time.perf_counter_ns
        census = self

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            f = sys._getframe(1)
            rel = _rel(f.f_code.co_filename)
            if rel is None:
                return orig(*args, **kwargs)
            key = (rel, f.f_code.co_name, name)
            st = census.sites.get(key)
            if st is None:
                st = census.sites[key] = SiteStats(site=key)
            st.lines.add(f.f_lineno)
            sig = ", ".join([describe(a) for a in args] + [f"{k}={describe(v)}" for k, v in kwargs.items()])
            st.signatures[sig] = st.signatures.get(sig, 0) + 1
            if census.capture:
                nb = _nbytes(args) + _nbytes(tuple(kwargs.values()))
                if nb > st.capture_bytes:
                    st.capture_bytes = nb
                    st.capture = (_copy(args), {k: _copy(v) for k, v in kwargs.items()})
                    st.capture_signature = sig
            t0 = pc()
            try:
                return orig(*args, **kwargs)
            finally:
                st.ns += pc() - t0
                st.calls += 1

        wrapper.__name__ = name
        wrapper.__wrapped__ = orig  # type: ignore[attr-defined]
        return wrapper

    def __enter__(self) -> Census:
        import cv2

        for name in O.engine_cv2_functions():
            orig = getattr(cv2, name)
            self._orig[name] = orig
            setattr(cv2, name, self._wrap(name, orig))
        return self

    def __exit__(self, *exc: object) -> None:
        import cv2

        for name, orig in self._orig.items():
            setattr(cv2, name, orig)
        self._orig.clear()


def site_label(site: tuple[str, str, str]) -> str:
    """Metric-safe site label, e.g. ``GaussianBlur@roughness.residual``."""
    file, func, fn = site
    return f"{fn}@{Path(file).stem}.{func.strip('<>')}"


def capture_engine_args(mode: str = "auto", pair: tuple[str, int] | None = None) -> Census:
    """Run one ``localize(engine='v2')`` on a DEV pair under a capturing census."""
    from bench.cool.pairs import CAPTURE_PAIR, load_pair
    from driftsense.api import localize
    from driftsense.contracts import Budget, Mode

    p = load_pair(*(pair or CAPTURE_PAIR))
    with Census(capture=True) as cen:
        localize(p.template_png, p.search_png, Mode(mode), Budget(), engine="v2")
    return cen


def run_census(pairs: list, modes: list[str], warmup: int = 1) -> tuple[dict, dict]:
    """Per mode: per-site calls/ms per localize and the kleidicv/control/non-cv2 shares."""
    from driftsense.api import localize
    from driftsense.contracts import Budget, Mode

    metrics: dict = {}
    sites_meta: dict = {}
    for mode in modes:
        m = Mode(mode)
        for _ in range(warmup):
            localize(pairs[0].template_png, pairs[0].search_png, m, Budget(), engine="v2")
        per_site_calls: dict[tuple, list[int]] = defaultdict(list)
        per_site_ms: dict[tuple, list[float]] = defaultdict(list)
        shares: dict[str, list[float]] = defaultdict(list)
        totals: list[float] = []
        with Census(capture=False) as cen:
            for p in pairs:
                cen.reset_counts()
                res = localize(p.template_png, p.search_png, m, Budget(), engine="v2")
                total = float(res.total_ms)
                totals.append(total)
                by_group = {O.KLEIDICV: 0.0, O.CONTROL: 0.0}
                seen = set()
                for key, st in cen.sites.items():
                    per_site_calls[key].append(st.calls)
                    per_site_ms[key].append(st.ns / 1e6)
                    by_group[O.group_of(key[2])] += st.ns / 1e6
                    seen.add(key)
                for key in per_site_calls:  # sites not hit on this pair count as 0
                    if key not in seen:
                        per_site_calls[key].append(0)
                        per_site_ms[key].append(0.0)
                if total > 0:
                    shares[O.KLEIDICV].append(by_group[O.KLEIDICV] / total)
                    shares[O.CONTROL].append(by_group[O.CONTROL] / total)
                    shares["non_cv2"].append(
                        max(0.0, 1.0 - (by_group[O.KLEIDICV] + by_group[O.CONTROL]) / total)
                    )
            sites = dict(cen.sites)
        n = len(pairs)
        for key, calls in per_site_calls.items():
            lab = site_label(key)
            ms = per_site_ms[key]
            metrics[f"census.{mode}.{lab}.calls_per_localize"] = C.metric(float(np.mean(calls)), "calls", n=n)
            med, lo, hi = C.bootstrap_median_ci(ms)
            metrics[f"census.{mode}.{lab}.ms_per_localize"] = C.metric(med, "ms", lo, hi, n)
            st = sites[key]
            sites_meta.setdefault(mode, {})[lab] = {
                "file": key[0],
                "func": key[1],
                "fn": key[2],
                "lines": sorted(st.lines),
                "group": O.group_of(key[2]),
                "signatures": dict(sorted(st.signatures.items(), key=lambda kv: -kv[1])[:8]),
                "mean_calls_per_localize": float(np.mean(calls)),
                "median_ms_per_localize": med,
            }
        for g, vals in shares.items():
            if vals:
                med, lo, hi = C.bootstrap_median_ci(vals)
                metrics[f"census.{mode}.share.{g}"] = C.metric(med, "fraction", lo, hi, len(vals))
        med, lo, hi = C.bootstrap_median_ci(totals)
        metrics[f"census.{mode}.total_ms"] = C.metric(med, "ms", lo, hi, len(totals))
    return metrics, sites_meta


def uncovered_sites(sites: dict) -> list[str]:
    """Census sites that are neither benchmarked nor listed in NOT_BENCHMARKED."""
    known = {o.site for o in O.ENGINE_OPS} | set(O.NOT_BENCHMARKED)
    return sorted(site_label(k) for k in sites if k not in known)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_common_args(ap)
    ap.add_argument("--pairs", type=int, default=None, help="DEV pairs per mode (default 12; quick 4)")
    ap.add_argument("--modes", nargs="+", default=["fast", "auto"])
    a = ap.parse_args(argv)
    from bench.cool.pairs import load_dev_pairs

    k = a.pairs or (4 if a.quick else 12)
    C.set_cv_threads(1)
    C.banner("census")
    pairs = load_dev_pairs(k)
    metrics, sites = run_census(pairs, a.modes, warmup=1)
    metrics["meta.pairs"] = C.metric(k, "pairs")
    metrics["meta.quick"] = C.metric(1.0 if a.quick else 0.0, "bool")
    unc = sorted(
        {
            u
            for mode in sites
            for u in uncovered_sites({(v["file"], v["func"], v["fn"]): 0 for v in sites[mode].values()})
        }
    )
    out: Path = a.out_dir
    C.write_json(
        out / "census_sites.json", {"modes": sites, "uncovered": unc, "pairs": [p.key for p in pairs]}
    )
    C.write_results(out / "census.json", "census", metrics, C.produced_by(__file__))
    for mode in a.modes:
        sh = {g: metrics.get(f"census.{mode}.share.{g}") for g in (O.KLEIDICV, O.CONTROL, "non_cv2")}
        print(
            f"[census] {mode}: total {metrics[f'census.{mode}.total_ms'].value:.1f} ms; share "
            + ", ".join(f"{g} {v.value:.1%}" for g, v in sh.items() if v is not None)
        )
    if unc:
        print(f"[census] WARNING: engine cv2 sites not in the op table: {unc}", file=sys.stderr)
    print(f"[census] wrote {out / 'census.json'} (+ census_sites.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
