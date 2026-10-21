"""Shared timing / statistics / environment / output helpers for the COOL benchmark.

Methodology (Overall_plan.md section 7):
  * ``warmup`` untimed calls (default 5), then ``runs`` timed calls (default 30);
    ``--quick`` drops this to 3 warm-ups / 5 runs and is for plumbing checks only.
  * every call is timed with ``time.perf_counter_ns``;
  * we report the median with a percentile-bootstrap 95% CI (2000 resamples,
    seeded numpy RNG, so the CI is reproducible from the raw CSV), plus p95;
  * ``cv2.setNumThreads`` is pinned to 1 and to N = ``os.cpu_count()``;
  * psutil samples CPU utilisation over each measured block.

Raw per-run samples always go to a sibling CSV; the ResultsFile JSON only holds
aggregates (``driftsense.contracts.Metric``).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_WARMUP = 5
DEFAULT_RUNS = 30
QUICK_WARMUP = 3
QUICK_RUNS = 5
N_BOOT = 2000
BOOT_SEED = 20261011
CONF = 0.95

IMDS_BASE = "http://169.254.169.254/latest"


# --------------------------------------------------------------------------- timing
def time_callable(fn: Callable[[], Any], warmup: int, runs: int, inner: int = 1) -> np.ndarray:
    """Call ``fn`` ``warmup`` times untimed, then take ``runs`` timed samples.

    Each sample times ``inner`` back-to-back calls and is divided by ``inner`` so
    that very fast ops (a few microseconds) are not dominated by timer overhead.
    Returns per-call milliseconds (float64), one value per sample.
    """
    for _ in range(max(0, warmup)):
        fn()
    inner = max(1, int(inner))
    out = np.empty(max(1, runs), dtype=np.float64)
    pc = time.perf_counter_ns
    rng = range(inner)
    for i in range(out.size):
        t0 = pc()
        for _ in rng:
            fn()
        out[i] = (pc() - t0) / 1e6 / inner
    return out


def calibrate_inner(fn: Callable[[], Any], min_sample_ms: float, cap: int = 1000) -> int:
    """Smallest inner-loop count so one timed sample lasts >= ``min_sample_ms`` (1 if 0)."""
    if min_sample_ms <= 0:
        return 1
    t0 = time.perf_counter_ns()
    fn()
    one = max((time.perf_counter_ns() - t0) / 1e6, 1e-4)
    return int(min(cap, max(1, np.ceil(min_sample_ms / one))))


# --------------------------------------------------------------------------- statistics
def _alpha_bounds(conf: float) -> tuple[float, float]:
    a = (1.0 - conf) / 2.0
    return 100.0 * a, 100.0 * (1.0 - a)


def bootstrap_median_ci(
    samples: Iterable[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED, conf: float = CONF
) -> tuple[float, float, float]:
    """Median and percentile-bootstrap CI of the median. Deterministic for a given seed."""
    x = np.asarray(list(samples) if not isinstance(samples, np.ndarray) else samples, dtype=np.float64)
    if x.size == 0:
        raise ValueError("no samples")
    med = float(np.median(x))
    if x.size == 1:
        return med, med, med
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, _alpha_bounds(conf))
    return med, float(min(lo, med)), float(max(hi, med))


def bootstrap_ratio_ci(
    baseline: Iterable[float],
    candidate: Iterable[float],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    conf: float = CONF,
) -> tuple[float, float, float]:
    """Speed-up = median(baseline) / median(candidate) with a bootstrap CI.

    > 1 means the candidate is faster. The two arms are resampled independently
    (they are independent machines), and the ratio of resampled medians is taken.
    """
    ratio, draws = bootstrap_ratio_draws(baseline, candidate, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(draws, _alpha_bounds(conf))
    return ratio, float(min(lo, ratio)), float(max(hi, ratio))


def bootstrap_ratio_draws(
    baseline: Iterable[float], candidate: Iterable[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> tuple[float, np.ndarray]:
    """Point ratio median(baseline)/median(candidate) and its ``n_boot`` bootstrap draws."""
    b = np.asarray(list(baseline) if not isinstance(baseline, np.ndarray) else baseline, dtype=np.float64)
    c = np.asarray(list(candidate) if not isinstance(candidate, np.ndarray) else candidate, dtype=np.float64)
    if b.size == 0 or c.size == 0:
        raise ValueError("no samples")
    ratio = float(np.median(b) / np.median(c))
    rng = np.random.default_rng(seed)
    mb = np.median(b[rng.integers(0, b.size, size=(n_boot, b.size))], axis=1)
    mc = np.median(c[rng.integers(0, c.size, size=(n_boot, c.size))], axis=1)
    return ratio, mb / mc


def percentile_ci(draws: np.ndarray, point: float, conf: float = CONF) -> tuple[float, float]:
    lo, hi = np.percentile(np.asarray(draws, dtype=np.float64), _alpha_bounds(conf))
    return float(min(lo, point)), float(max(hi, point))


def ci_includes(lo: float | None, hi: float | None, value: float = 1.0) -> bool:
    return lo is not None and hi is not None and lo <= value <= hi


def summarize(samples: Iterable[float], seed: int = BOOT_SEED) -> dict[str, float]:
    x = np.asarray(list(samples) if not isinstance(samples, np.ndarray) else samples, dtype=np.float64)
    med, lo, hi = bootstrap_median_ci(x, seed=seed)
    return {
        "median": med,
        "ci_low": lo,
        "ci_high": hi,
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "mean": float(x.mean()),
        "min": float(x.min()),
        "max": float(x.max()),
        "n": int(x.size),
    }


# --------------------------------------------------------------------------- threads / CPU
def resolve_threads(spec: str | int) -> int:
    """'1' -> 1; 'all' / 'N' / '0' -> os.cpu_count(); any other int as given."""
    s = str(spec).strip().lower()
    if s in ("all", "n", "0", "max"):
        return os.cpu_count() or 1
    n = int(s)
    if n < 1:
        raise ValueError(f"bad thread count {spec!r}")
    return n


def thread_tag(n: int) -> str:
    """Stable metric-key tag: 't1' for single-thread, 'tN' for all vCPUs (N recorded separately)."""
    if n == 1:
        return "t1"
    return "tN" if n == (os.cpu_count() or 1) else f"t{n}"


def set_cv_threads(n: int) -> int:
    import cv2

    cv2.setNumThreads(int(n))
    return int(cv2.getNumThreads())


@dataclass
class CpuMonitor:
    """System-wide CPU % (psutil) and process(+children) CPU seconds over a block."""

    system_pct: float | None = None
    cores_used: float | None = None
    wall_s: float = 0.0
    _t0: float = 0.0
    _cpu0: float = 0.0
    _ok: bool = field(default=False, repr=False)

    @staticmethod
    def _proc_cpu() -> float:
        import psutil

        t = psutil.Process().cpu_times()
        return t.user + t.system + getattr(t, "children_user", 0.0) + getattr(t, "children_system", 0.0)

    def start(self) -> None:
        try:
            import psutil
        except ImportError:  # psutil missing: timings still work, utilisation is None
            self._ok = False
        else:
            try:
                psutil.cpu_percent(interval=None)
                self._cpu0 = self._proc_cpu()
                self._ok = True
            except (psutil.Error, OSError, NotImplementedError):  # unsupported platform
                self._ok = False
        self._t0 = time.perf_counter()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.wall_s = time.perf_counter() - self._t0
        if self._ok:
            import psutil

            self.system_pct = float(psutil.cpu_percent(interval=None))
            if self.wall_s > 0:
                self.cores_used = (self._proc_cpu() - self._cpu0) / self.wall_s


# --------------------------------------------------------------------------- environment
_IMDS_CACHE: dict[str, str | None] = {}
_IMDS_TOKEN: list[str | None] = []


def _imds_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def imds_get(path: str, timeout: float = 1.0) -> str | None:
    """Read an EC2 instance-metadata path via IMDSv2. Returns None off-EC2 (1 s timeout, cached).

    Set ``BENCH_NO_IMDS=1`` to skip the network probe entirely.
    """
    if os.environ.get("BENCH_NO_IMDS") == "1":
        return None
    if path in _IMDS_CACHE:
        return _IMDS_CACHE[path]
    opener = _imds_opener()
    if not _IMDS_TOKEN:
        try:
            req = urllib.request.Request(
                f"{IMDS_BASE}/api/token",
                method="PUT",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"},
            )
            with opener.open(req, timeout=timeout) as r:
                _IMDS_TOKEN.append(r.read().decode().strip())
        except (urllib.error.URLError, OSError, ValueError):
            _IMDS_TOKEN.append(None)
    token = _IMDS_TOKEN[0]
    val: str | None = None
    if token:
        try:
            req = urllib.request.Request(
                f"{IMDS_BASE}/{path.lstrip('/')}", headers={"X-aws-ec2-metadata-token": token}
            )
            with opener.open(req, timeout=timeout) as r:
                val = r.read().decode().strip() or None
        except (urllib.error.URLError, OSError, ValueError):
            val = None
    _IMDS_CACHE[path] = val
    return val


def normalized_arch() -> str:
    m = platform.machine().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(m, m or "unknown")


def cv2_is_cool(cv2_file: str | None = None) -> bool:
    if cv2_file is None:
        import cv2

        cv2_file = cv2.__file__ or ""
    return "/opt/cool" in str(cv2_file).replace("\\", "/")


def capture_env():
    """EnvInfo for this host (driftsense.contracts.EnvInfo)."""
    import cv2

    from driftsense.contracts import EnvInfo

    return EnvInfo(
        python=platform.python_version(),
        opencv=cv2.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
        arch=normalized_arch(),
        instance_type=imds_get("meta-data/instance-type"),
        cool=cv2_is_cool(cv2.__file__),
    )


def git_sha() -> str:
    env = os.environ.get("ENGINE_GIT_SHA")
    if env:
        return env.strip()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def produced_by(module_file: str) -> str:
    p = Path(module_file).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# --------------------------------------------------------------------------- outputs
def metric(
    value: float, unit: str, ci_low: float | None = None, ci_high: float | None = None, n: int | None = None
):
    from driftsense.contracts import Metric

    return Metric(
        value=float(value),
        unit=unit,
        ci_low=None if ci_low is None else float(ci_low),
        ci_high=None if ci_high is None else float(ci_high),
        n=n,
    )


def timing_metrics(prefix: str, samples_ms: Iterable[float]) -> dict:
    """``<prefix>.median_ms`` (with bootstrap CI and n) and ``<prefix>.p95_ms``."""
    s = summarize(samples_ms)
    return {
        f"{prefix}.median_ms": metric(s["median"], "ms", s["ci_low"], s["ci_high"], s["n"]),
        f"{prefix}.p95_ms": metric(s["p95"], "ms", n=s["n"]),
    }


def write_results(
    path: Path,
    name: str,
    metrics: dict,
    produced: str,
    env=None,
    dataset: str = "bench",
    sha: str | None = None,
):
    """Validate against driftsense.contracts.ResultsFile and write JSON. Returns the model."""
    from driftsense.contracts import ResultsFile

    rf = ResultsFile(
        name=name,
        produced_by=produced,
        git_sha=sha or git_sha(),
        created_at=datetime.now(UTC),
        dataset=dataset,
        env=env if env is not None else capture_env(),
        metrics=dict(sorted(metrics.items())),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rf.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return rf


def load_results(path: Path):
    from driftsense.contracts import ResultsFile

    return ResultsFile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, header: list[str], rows: Iterable[Iterable[Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- CLI helpers
def add_common_args(p) -> None:
    p.add_argument(
        "--quick",
        action="store_true",
        help=f"plumbing check: {QUICK_WARMUP} warm-ups / {QUICK_RUNS} runs, small inputs",
    )
    p.add_argument("--warmup", type=int, default=None, help=f"warm-up calls (default {DEFAULT_WARMUP})")
    p.add_argument("--runs", type=int, default=None, help=f"timed runs (default {DEFAULT_RUNS})")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out") / "local",
        help="directory for the JSON/CSV outputs (default out/local)",
    )
    p.add_argument("--seed", type=int, default=0, help="input-generation seed (default 0)")


def resolve_counts(args) -> tuple[int, int]:
    w = args.warmup if args.warmup is not None else (QUICK_WARMUP if args.quick else DEFAULT_WARMUP)
    r = args.runs if args.runs is not None else (QUICK_RUNS if args.quick else DEFAULT_RUNS)
    if not args.quick and r < DEFAULT_RUNS:
        print(f"[bench] WARNING: runs={r} < {DEFAULT_RUNS}; below the section-7 protocol", file=sys.stderr)
    return w, r


def banner(tool: str) -> None:
    import cv2

    print(
        f"[{tool}] cv2 {cv2.__version__} at {cv2.__file__} (cool={cv2_is_cool(cv2.__file__)}); "
        f"numpy {np.__version__}; python {platform.python_version()}; arch {normalized_arch()}; "
        f"cpus {os.cpu_count()}",
        flush=True,
    )
