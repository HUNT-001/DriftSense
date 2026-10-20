"""Statistics for the sealed-set evaluation (eval/protocol.json).

Pure functions, no I/O and no dataset access: everything here takes arrays or
counts and returns numbers, so it is unit-testable without touching the sealed
set. ``eval/sealed_run.py`` calls these to turn per-pair / per-episode records
into the pre-registered metrics; ``eval/charts.py`` and the docs read the
resulting JSON, never these functions directly.

Methods are fixed by the protocol:
  * binomial rates -> Wilson score interval, 95%, two-sided;
  * continuous summaries (subpixel nm error, agent mean steps/dose) -> median /
    p95 / mean with a 95% CI by BCa bootstrap, 10000 resamples, seed 9999;
  * ECE -> EQUAL-COUNT (quantile) binning. This is deliberately a DIFFERENT
    estimator from the A6 calibration report (eval/calibration.py), which uses
    equal-WIDTH bins. The two ECE numbers are not the same metric and must never
    be compared directly; see ``ece_equal_count`` and ECE_BINNING below.
  * McNemar -> paired arm-vs-arm comparison on per-pair correctness.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

WILSON_Z = 1.959963984540054  # two-sided 95%
BOOT_RESAMPLES = 10000
BOOT_SEED = 9999  # fixed by the protocol (dataset.seed)

# Tag carried into the results JSON so the sealed ECE is never confused with the
# A6 equal-width ECE in results/accuracy/calibration.json.
ECE_BINNING = "equal-count"


# --------------------------------------------------------------------------- #
# Binomial rates                                                              #
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion k/n.

    Returns ``(point, lower, upper)``. ``n == 0`` -> ``(0.0, 0.0, 1.0)`` (no
    information: the widest honest interval).
    """
    if n <= 0:
        return 0.0, 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return float(p), float(max(0.0, center - half)), float(min(1.0, center + half))


def wilson_lower(k: int, n: int, z: float = WILSON_Z) -> float:
    return wilson(k, n, z)[1]


def wilson_upper(k: int, n: int, z: float = WILSON_Z) -> float:
    return wilson(k, n, z)[2]


# --------------------------------------------------------------------------- #
# Paired comparison (arm vs arm on the same pairs)                            #
# --------------------------------------------------------------------------- #
def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, float]:
    """McNemar's test on two paired 0/1 correctness vectors (same pairs).

    ``b`` = count where A right & B wrong; ``c`` = A wrong & B right. Uses the
    continuity-corrected chi-square for ``b + c >= 25`` and the exact binomial
    (two-sided) otherwise. Returns discordant counts, the statistic, and p.
    """
    a = np.asarray(a_correct, dtype=bool).ravel()
    bb = np.asarray(b_correct, dtype=bool).ravel()
    if a.size != bb.size:
        raise ValueError("paired vectors must have equal length")
    b = int(np.sum(a & ~bb))
    c = int(np.sum(~a & bb))
    n = b + c
    if n == 0:
        return {"b": 0.0, "c": 0.0, "statistic": 0.0, "p_value": 1.0, "exact": 1.0}
    if n >= 25:
        stat = (abs(b - c) - 1.0) ** 2 / n
        p = math.erfc(math.sqrt(stat / 2.0))  # chi-square(1) survival = erfc(sqrt(x/2))
        exact = 0.0
    else:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
        p = min(1.0, 2.0 * tail)
        stat = float(min(b, c))
        exact = 1.0
    return {"b": float(b), "c": float(c), "statistic": float(stat), "p_value": float(p), "exact": exact}


# --------------------------------------------------------------------------- #
# Calibration: ECE with EQUAL-COUNT bins                                      #
# --------------------------------------------------------------------------- #
def ece_equal_count(p: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error using EQUAL-COUNT (quantile) bins.

    Each bin holds ~the same number of samples (quantile edges on ``p``), unlike
    the A6 equal-width estimator. ECE = sum_b (n_b / N) * |acc_b - conf_b|.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return 0.0
    order = np.argsort(p, kind="mergesort")
    p, correct = p[order], correct[order]
    # Split into n_bins near-equal contiguous chunks of the sorted array.
    edges = np.linspace(0, n, min(n_bins, n) + 1).round().astype(int)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        conf = float(p[lo:hi].mean())
        acc = float(correct[lo:hi].mean())
        ece += (hi - lo) / n * abs(acc - conf)
    return float(ece)


# --------------------------------------------------------------------------- #
# Risk-coverage                                                               #
# --------------------------------------------------------------------------- #
def risk_coverage(confidence: np.ndarray, correct: np.ndarray, n_points: int = 20):
    """Commit most-confident-first; return (coverage, risk=wrong-fraction) arrays."""
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    n = confidence.size
    if n == 0:
        return np.array([]), np.array([])
    order = np.argsort(-confidence, kind="mergesort")
    wrong = (correct[order] == 0.0).astype(np.float64)
    cov = np.arange(1, n + 1, dtype=np.float64) / n
    risk = np.cumsum(wrong) / np.arange(1, n + 1)
    if n > n_points:
        pick = np.linspace(0, n - 1, n_points).round().astype(int)
        cov, risk = cov[pick], risk[pick]
    return cov, risk


# --------------------------------------------------------------------------- #
# BCa bootstrap for continuous summaries                                      #
# --------------------------------------------------------------------------- #
def _percentile_of(boot: np.ndarray, theta_hat: float) -> float:
    below = float(np.mean(boot < theta_hat))
    below = min(max(below, 1e-6), 1 - 1e-6)  # keep the normal quantile finite
    return below


def _norm_ppf(q: float) -> float:
    # Acklam's rational approximation to the standard-normal inverse CDF.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        qq = math.sqrt(-2 * math.log(q))
        return (((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5]) / (
            (((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1
        )
    if q > phigh:
        qq = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0] * qq + c[1]) * qq + c[2]) * qq + c[3]) * qq + c[4]) * qq + c[5]) / (
            (((d[0] * qq + d[1]) * qq + d[2]) * qq + d[3]) * qq + 1
        )
    qq = q - 0.5
    r = qq * qq
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * qq
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bca_bootstrap(
    x: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = BOOT_RESAMPLES,
    seed: int = BOOT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """BCa (bias-corrected and accelerated) bootstrap CI for ``stat(x)``.

    Returns ``(point, lo, hi)`` at the two-sided ``1 - alpha`` level. Falls back
    to the plain percentile interval if the acceleration is undefined (n <= 1).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    theta_hat = float(stat(x))
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return theta_hat, theta_hat, theta_hat
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.array([stat(x[row]) for row in idx], dtype=np.float64)

    # bias correction z0
    z0 = _norm_ppf(_percentile_of(boot, theta_hat))
    # acceleration via jackknife
    jack = np.array([stat(np.delete(x, i)) for i in range(n)], dtype=np.float64)
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    acc = float(num / den) if den > 0 else 0.0

    def _adj(a: float) -> float:
        z = _norm_ppf(a)
        val = z0 + (z0 + z) / (1 - acc * (z0 + z))
        return _norm_cdf(val)

    lo_q, hi_q = _adj(alpha / 2.0), _adj(1 - alpha / 2.0)
    lo = float(np.quantile(boot, min(max(lo_q, 0.0), 1.0)))
    hi = float(np.quantile(boot, min(max(hi_q, 0.0), 1.0)))
    return theta_hat, lo, hi


def median(x: np.ndarray) -> float:
    return float(np.median(x)) if np.asarray(x).size else 0.0


def p95(x: np.ndarray) -> float:
    return float(np.percentile(x, 95)) if np.asarray(x).size else 0.0


def mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if np.asarray(x).size else 0.0


# --------------------------------------------------------------------------- #
# Subpixel error                                                              #
# --------------------------------------------------------------------------- #
def nm_error(px_error: np.ndarray, pixel_size_nm: float) -> np.ndarray:
    """Convert per-pair pixel errors to nm (px * nm/px)."""
    return np.asarray(px_error, dtype=np.float64).ravel() * float(pixel_size_nm)
