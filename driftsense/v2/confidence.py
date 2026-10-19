"""Calibrated confidence for v2 (A3).

The engine ranks lattice-consistent placements by a roughness-residual score. The
RAW gap between rank-1 and rank-2 is not comparable across images (its scale moves
with contrast, dose and structure), so it is a poor thing to threshold directly.
This module turns that gap into two calibrated, image-independent quantities:

  * ``margin_z`` -- a ROBUST standardised margin,

        margin_z = (s1 - s2) / (1.4826 * MAD(non-top scores)),

    where MAD is the median absolute deviation of the rank>=2 scores. 1.4826 is
    the Gaussian-consistency constant, so for roughly normal background scores
    margin_z reads as "how many sigma clear rank-1 is of the field". MAD (not the
    std) keeps it from being dragged around by the near-ties that periodic
    replicas produce.

  * ``p_correct`` -- a calibrated probability the rank-1 placement is correct,
    from a NONPARAMETRIC isotonic map margin_z -> P(correct) fit by pool-adjacent-
    violators on a held-out CALIB split (its own seed, never DEV or the sealed
    seed). Isotonic is the natural choice: more margin can only mean more
    confidence, and PAV imposes exactly that monotonicity without a shape
    assumption.

``tau_commit`` is the smallest margin_z at which the engine may COMMIT: the
smallest threshold whose false-confident rate (committed but rank-1 wrong) has a
Wilson 95% UPPER bound <= ``alpha`` (2%) on CALIB. Using the upper bound, not the
point estimate, is what makes the 2% a guarantee the finite CALIB set actually
supports rather than a lucky sample.

Pure NumPy; no cv2, no pydantic, no I/O except :func:`load_calibrator`, which
reads the fitted model back from the one results file the eval writes
(``results/accuracy/calibration.json``) so the engine and the UI share a single
source of truth for the calibrated tau.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from driftsense.contracts import Margin

# Raw-margin commit threshold used ONLY on the uncalibrated fallback, matching the
# engine's historical default so behaviour is unchanged until calibration is run.
LEGACY_RAW_TAU = 0.10

# Robustness floor on the MAD denominator: scores live on a bounded correlation
# scale, so a (near-)degenerate field of replicas must not blow margin_z up to
# +/-inf. Stored in the model so fit-time and run-time standardisation match.
DEFAULT_MAD_FLOOR = 1e-3
DEFAULT_ALPHA = 0.02  # false-confident target (Wilson UPPER bound)
WILSON_Z = 1.96  # 95% one-model two-sided z; used for the one-sided upper bound
_MODEL_PREFIX = "model."

Decision = Literal["commit", "abstain"]

# Canonical location of the fitted model, written by eval/calibration.py. The
# engine reads tau from here; Person B's UI renders the same file.
CALIBRATION_JSON = Path(__file__).resolve().parents[2] / "results" / "accuracy" / "calibration.json"


@dataclass(frozen=True)
class Confidence:
    """One assessment of a ranked candidate list."""

    margin_z: float
    p_correct: float
    decision: Decision


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation about the median (no consistency scaling)."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def margin_z(scores: list[float] | np.ndarray, mad_floor: float = DEFAULT_MAD_FLOOR) -> float:
    """Robust standardised margin between the two best candidate scores.

    Scores are the engine's candidate scores. They are sorted DESCENDING here so
    the result is the top-1/top-2 gap regardless of how a given mode orders the
    list (the AUTO cascade re-ranks by a fingerprint similarity whose numeric
    ``score`` is not the roughness ordering), making margin_z non-negative and
    mode-consistent. The MAD is taken over the non-top (rank>=2) scores -- the
    background field the winner must stand clear of. Fewer than two scores ->
    margin_z 0 (not unique enough to assess); a (near-)zero MAD is floored.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())[::-1]
    if s.size < 2:
        return 0.0
    non_top = s[1:]
    denom = 1.4826 * _mad(non_top)
    denom = max(denom, float(mad_floor))
    return float((s[0] - s[1]) / denom)


def wilson_upper(k: int, n: int, z: float = WILSON_Z) -> float:
    """Wilson score-interval UPPER bound for a binomial rate k/n (n==0 -> 1.0)."""
    if n <= 0:
        return 1.0
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    half = z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    return float(min(1.0, (center + half) / denom))


def pav(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pool-adjacent-violators isotonic regression (nondecreasing fit of ``y``).

    ``y`` is assumed already ordered by the covariate (here ascending margin_z).
    Returns the fitted value at each position. Standard O(n) block-merging PAV
    with optional weights.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if n == 0:
        return y.copy()
    w = np.ones(n) if w is None else np.asarray(w, dtype=np.float64).ravel()
    # Each block: [sum, weight, count]; merge left while monotonicity is violated.
    vals = list(y)
    wts = list(w)
    levels: list[float] = []
    lw: list[float] = []
    counts: list[int] = []
    for v, wi in zip(vals, wts, strict=True):
        levels.append(v)
        lw.append(wi)
        counts.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            tw = lw[-2] + lw[-1]
            merged = (levels[-2] * lw[-2] + levels[-1] * lw[-1]) / tw
            levels[-2:] = [merged]
            lw[-2:] = [tw]
            counts[-2:] = [counts[-2] + counts[-1]]
    out = np.empty(n, dtype=np.float64)
    i = 0
    for lvl, c in zip(levels, counts, strict=True):
        out[i : i + c] = lvl
        i += c
    return out


@dataclass(frozen=True)
class Calibrator:
    """Fitted isotonic margin_z -> P(correct) map plus the commit threshold.

    ``z_knots`` are block START margins (ascending) and ``p_knots`` their fitted
    probability levels; :meth:`p_of` is the resulting nondecreasing step function.
    ``calibrated`` is False for the fallback used when no model file is present.
    """

    z_knots: np.ndarray
    p_knots: np.ndarray
    tau_commit: float
    mad_floor: float = DEFAULT_MAD_FLOOR
    alpha: float = DEFAULT_ALPHA
    n_calib: int = 0
    calibrated: bool = True

    def p_of(self, z: float) -> float:
        """P(correct) for a margin_z, read off the isotonic step function."""
        if self.z_knots.size == 0:
            return float("nan")
        idx = int(np.searchsorted(self.z_knots, z, side="right") - 1)
        idx = max(0, min(idx, self.p_knots.size - 1))
        return float(self.p_knots[idx])

    def assess(self, scores: list[float] | np.ndarray) -> Confidence:
        z = margin_z(scores, self.mad_floor)
        decision: Decision = "commit" if z >= self.tau_commit else "abstain"
        return Confidence(margin_z=z, p_correct=self.p_of(z), decision=decision)

    def to_model_dict(self) -> dict[str, float]:
        """Flat ``model.*`` scalars so the model round-trips through the ResultsFile."""
        out: dict[str, float] = {
            f"{_MODEL_PREFIX}tau_commit": float(self.tau_commit),
            f"{_MODEL_PREFIX}mad_floor": float(self.mad_floor),
            f"{_MODEL_PREFIX}alpha": float(self.alpha),
            f"{_MODEL_PREFIX}n_calib": float(self.n_calib),
            f"{_MODEL_PREFIX}n_knots": float(self.z_knots.size),
            f"{_MODEL_PREFIX}calibrated": 1.0 if self.calibrated else 0.0,
        }
        for i, (zk, pk) in enumerate(zip(self.z_knots, self.p_knots, strict=True)):
            out[f"{_MODEL_PREFIX}z{i}"] = float(zk)
            out[f"{_MODEL_PREFIX}p{i}"] = float(pk)
        return out

    @classmethod
    def from_model_dict(cls, m: dict[str, float]) -> Calibrator:
        n = int(round(m[f"{_MODEL_PREFIX}n_knots"]))
        z = np.array([m[f"{_MODEL_PREFIX}z{i}"] for i in range(n)], dtype=np.float64)
        p = np.array([m[f"{_MODEL_PREFIX}p{i}"] for i in range(n)], dtype=np.float64)
        return cls(
            z_knots=z,
            p_knots=p,
            tau_commit=float(m[f"{_MODEL_PREFIX}tau_commit"]),
            mad_floor=float(m.get(f"{_MODEL_PREFIX}mad_floor", DEFAULT_MAD_FLOOR)),
            alpha=float(m.get(f"{_MODEL_PREFIX}alpha", DEFAULT_ALPHA)),
            n_calib=int(round(m.get(f"{_MODEL_PREFIX}n_calib", 0.0))),
            calibrated=bool(round(m.get(f"{_MODEL_PREFIX}calibrated", 1.0))),
        )


def _knots_from_fit(z_sorted: np.ndarray, ghat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the per-sample isotonic fit into (block-start z, level) knots."""
    if z_sorted.size == 0:
        return np.empty(0), np.empty(0)
    zk: list[float] = [float(z_sorted[0])]
    pk: list[float] = [float(ghat[0])]
    for zi, pi in zip(z_sorted[1:], ghat[1:], strict=True):
        if float(pi) != pk[-1]:  # a new isotonic level begins at this margin
            zk.append(float(zi))
            pk.append(float(pi))
    return np.asarray(zk, dtype=np.float64), np.asarray(pk, dtype=np.float64)


def fit_calibrator(
    zs: np.ndarray | list[float],
    correct: np.ndarray | list[int],
    *,
    alpha: float = DEFAULT_ALPHA,
    mad_floor: float = DEFAULT_MAD_FLOOR,
    z: float = WILSON_Z,
) -> Calibrator:
    """Fit the isotonic P(correct) map and the commit threshold on a CALIB split.

    ``zs`` are margin_z values, ``correct`` the 0/1 rank-1 correctness. tau_commit
    is the SMALLEST margin whose false-confident Wilson upper bound is <= alpha
    (largest coverage that still honours the 2% guarantee); if even the single
    most-confident sample cannot, tau_commit is set above every margin so nothing
    commits.
    """
    zs = np.asarray(zs, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=np.float64).ravel()
    if zs.size == 0:
        return fallback_calibrator()
    order = np.argsort(zs, kind="mergesort")
    z_sorted = zs[order]
    ghat = pav(y[order])
    z_knots, p_knots = _knots_from_fit(z_sorted, ghat)

    tau = float(z_sorted[-1]) + 1.0  # default: commit nothing
    for t in np.unique(z_sorted):  # ascending -> first valid t is the smallest
        commit = zs >= t
        n = int(commit.sum())
        wrong = int(((y == 0.0) & commit).sum())
        if n > 0 and wilson_upper(wrong, n, z) <= alpha:
            tau = float(t)
            break

    return Calibrator(
        z_knots=z_knots,
        p_knots=p_knots,
        tau_commit=tau,
        mad_floor=float(mad_floor),
        alpha=float(alpha),
        n_calib=int(zs.size),
    )


def fallback_calibrator() -> Calibrator:
    """Uncalibrated stand-in used when no fitted model is available.

    It carries ``calibrated=False`` so :func:`build_margin` keeps the engine's
    legacy raw-margin behaviour (and never claims a probability it cannot back).
    """
    return Calibrator(
        z_knots=np.empty(0),
        p_knots=np.empty(0),
        tau_commit=float("inf"),
        calibrated=False,
        n_calib=0,
    )


_CACHE: dict[str, tuple[float, Calibrator]] = {}


def load_calibrator(path: Path | str = CALIBRATION_JSON) -> Calibrator:
    """Load the fitted calibrator from the results file (mtime-cached).

    Any absence or malformation yields :func:`fallback_calibrator` -- the engine
    must never fail to localise just because calibration has not been run yet.
    """
    p = Path(path)
    try:
        mtime = p.stat().st_mtime
    except OSError:
        return fallback_calibrator()
    key = str(p)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        doc = json.loads(p.read_text())
        model = {
            k: float(v["value"]) for k, v in doc.get("metrics", {}).items() if k.startswith(_MODEL_PREFIX)
        }
        calib = Calibrator.from_model_dict(model) if model else fallback_calibrator()
    except (OSError, ValueError, KeyError, TypeError):
        calib = fallback_calibrator()
    _CACHE[key] = (mtime, calib)
    return calib


def clear_cache() -> None:
    """Drop the memoised calibrator (tests that rewrite the model file)."""
    _CACHE.clear()


def build_margin(scores: list[float] | np.ndarray, calib: Calibrator) -> Margin:
    """Build the contract :class:`Margin` for the CALIBRATED path (budget.tau is None).

    With a fitted model the reported margin is the robust ``margin_z`` and the
    threshold is ``tau_commit``, with ``p_correct`` populated and ``calibrated``
    True. Without one (fallback) it reproduces the engine's legacy raw rank-1/
    rank-2 gap against :data:`LEGACY_RAW_TAU`, uncalibrated and with no probability.
    The internal commit/abstain decision maps onto the contract's commit/escalate.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if calib.calibrated:
        c = calib.assess(s)
        decision = "commit" if c.decision == "commit" else "escalate"
        return Margin(
            margin=c.margin_z,
            tau=float(calib.tau_commit),
            calibrated=True,
            p_correct=max(0.0, min(1.0, c.p_correct)),
            decision=decision,
        )
    raw = float(s[0] - s[1]) if s.size >= 2 else 1.0
    return Margin(
        margin=raw,
        tau=LEGACY_RAW_TAU,
        calibrated=False,
        p_correct=None,
        decision="commit" if raw >= LEGACY_RAW_TAU else "escalate",
    )
