"""A3 — calibrated confidence: the robust margin, the isotonic map, the Wilson
commit threshold, the engine's Margin wiring, and the CALIB seed isolation.

All fast and dataset-free: the calibration logic is exercised on synthetic score
lists and label arrays, and the seed-isolation checks read only module constants
and source. The heavy end-to-end fit lives in eval/calibration.py (run once).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from driftsense.v2 import confidence as C

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# margin_z                                                                     #
# --------------------------------------------------------------------------- #
def test_margin_z_is_robust_standardised_gap():
    # non-top scores {0.5,0.5,0.4,0.4} -> median 0.45, MAD 0.05, denom 1.4826*0.05
    scores = [0.9, 0.5, 0.5, 0.4, 0.4]
    z = C.margin_z(scores)
    assert z == pytest.approx((0.9 - 0.5) / (1.4826 * 0.05), rel=1e-6)


def test_margin_z_sorts_so_it_is_nonnegative_and_order_free():
    # AUTO's cascade can store a non-monotonic .score ordering; margin_z sorts, so
    # the top-1/top-2 gap is the same whatever order the list arrives in, and >= 0.
    assert C.margin_z([0.5, 0.9, 0.4]) == pytest.approx(C.margin_z([0.9, 0.5, 0.4]))
    assert C.margin_z([0.2, 0.9, 0.3, 0.25]) >= 0.0


def test_margin_z_needs_two_candidates():
    assert C.margin_z([0.9]) == 0.0
    assert C.margin_z([]) == 0.0


def test_margin_z_floors_degenerate_mad():
    # All non-top identical -> MAD 0; floor keeps it finite and large-but-bounded.
    z = C.margin_z([1.0, 0.2, 0.2, 0.2], mad_floor=1e-3)
    assert np.isfinite(z) and z == pytest.approx(0.8 / 1e-3)


# --------------------------------------------------------------------------- #
# isotonic PAV + Wilson                                                        #
# --------------------------------------------------------------------------- #
def test_pav_is_nondecreasing_and_mean_preserving():
    y = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    g = C.pav(y)
    assert np.all(np.diff(g) >= -1e-12)
    assert g.mean() == pytest.approx(y.mean())


def test_pav_leaves_monotone_input_untouched():
    y = np.array([0.1, 0.1, 0.4, 0.7, 1.0])
    assert C.pav(y) == pytest.approx(y)


def test_wilson_upper_bounds_and_edges():
    assert C.wilson_upper(0, 0) == 1.0
    assert C.wilson_upper(0, 50) < 0.1  # 0/50 wrong -> tight upper bound
    assert 0.0 <= C.wilson_upper(1, 100) <= 1.0
    # upper bound always exceeds the point estimate
    assert C.wilson_upper(5, 100) > 0.05


# --------------------------------------------------------------------------- #
# fit_calibrator                                                               #
# --------------------------------------------------------------------------- #
def _synthetic(n: int = 400, seed: int = 0):
    """Monotone, with a genuinely clean high-margin tail (z>=3 always correct) so a
    non-empty 2% commit region exists -- below it, correctness is margin-graded."""
    rng = np.random.default_rng(seed)
    zs = rng.uniform(0.0, 6.0, n)
    graded = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(zs - 2.0)))).astype(float)
    correct = np.where(zs >= 3.0, 1.0, graded)
    return zs, correct


def test_fit_map_is_monotone_and_in_unit_interval():
    zs, correct = _synthetic()
    calib = C.fit_calibrator(zs, correct)
    assert np.all(np.diff(calib.p_knots) >= -1e-12)
    grid = np.linspace(0, 6, 50)
    ps = np.array([calib.p_of(z) for z in grid])
    assert np.all((ps >= 0.0) & (ps <= 1.0))
    assert np.all(np.diff(ps) >= -1e-12)  # nondecreasing in margin


def test_tau_commit_meets_false_confident_target_on_its_fit():
    zs, correct = _synthetic()
    alpha = 0.02
    calib = C.fit_calibrator(zs, correct, alpha=alpha)
    commit = zs >= calib.tau_commit
    n = int(commit.sum())
    wrong = int(((correct == 0.0) & commit).sum())
    assert n > 0, "threshold should admit a non-empty commit region on separable data"
    assert C.wilson_upper(wrong, n) <= alpha  # the guarantee the threshold is chosen for


def test_unsatisfiable_target_commits_nothing():
    # Pure noise: no margin can hit a 2% false-confident upper bound.
    rng = np.random.default_rng(1)
    zs = rng.uniform(0, 6, 300)
    correct = (rng.uniform(size=300) < 0.5).astype(float)
    calib = C.fit_calibrator(zs, correct, alpha=0.02)
    assert calib.tau_commit > zs.max()  # abstains everywhere


# --------------------------------------------------------------------------- #
# model round-trip + loading                                                   #
# --------------------------------------------------------------------------- #
def test_model_dict_roundtrip_preserves_map_and_tau():
    zs, correct = _synthetic()
    calib = C.fit_calibrator(zs, correct)
    back = C.Calibrator.from_model_dict(calib.to_model_dict())
    assert back.tau_commit == pytest.approx(calib.tau_commit)
    assert back.mad_floor == pytest.approx(calib.mad_floor)
    for z in np.linspace(0, 6, 25):
        assert back.p_of(z) == pytest.approx(calib.p_of(z))


def test_load_calibrator_missing_file_falls_back(tmp_path):
    C.clear_cache()
    calib = C.load_calibrator(tmp_path / "nope.json")
    assert not calib.calibrated and calib.tau_commit == float("inf")


def test_load_calibrator_reads_embedded_model(tmp_path):
    zs, correct = _synthetic()
    calib = C.fit_calibrator(zs, correct)
    metrics = {k: {"value": v, "unit": "model"} for k, v in calib.to_model_dict().items()}
    doc = {"schema_id": "driftsense.results/1", "metrics": metrics}
    f = tmp_path / "calibration.json"
    f.write_text(json.dumps(doc))
    C.clear_cache()
    loaded = C.load_calibrator(f)
    assert loaded.calibrated
    assert loaded.tau_commit == pytest.approx(calib.tau_commit)


# --------------------------------------------------------------------------- #
# build_margin: the engine wiring                                              #
# --------------------------------------------------------------------------- #
def test_build_margin_calibrated_is_consistent_and_probabilistic():
    zs, correct = _synthetic()
    calib = C.fit_calibrator(zs, correct)
    confident = [0.95, 0.10, 0.10, 0.08]  # large z
    m = C.build_margin(confident, calib)
    assert m.calibrated and m.p_correct is not None
    assert 0.0 <= m.p_correct <= 1.0
    # contract invariant: commit iff margin >= tau (validated by the model too)
    assert (m.decision == "commit") == (m.margin >= m.tau)


def test_build_margin_fallback_reproduces_legacy_raw():
    fb = C.fallback_calibrator()
    m = C.build_margin([0.9, 0.3, 0.2], fb)
    assert not m.calibrated and m.p_correct is None
    assert m.margin == pytest.approx(0.6) and m.tau == C.LEGACY_RAW_TAU
    assert m.decision == "commit"  # 0.6 >= 0.10
    m2 = C.build_margin([0.3, 0.28], fb)
    assert m2.decision == "escalate"  # 0.02 < 0.10


# --------------------------------------------------------------------------- #
# seed isolation: calibration never touches DEV or the sealed seed             #
# --------------------------------------------------------------------------- #
def test_calib_seeds_are_disjoint_from_dev_and_sealed():
    from eval import calibration as cal

    calib_seeds = set(cal.calib_tier_seeds())
    dev_seeds = set(cal.dev_tier_seeds())
    assert cal.SEALED_SEED == 9999
    assert calib_seeds.isdisjoint(dev_seeds)
    assert cal.SEALED_SEED not in calib_seeds
    # also disjoint from the per-pair composer seeds (dataset_gen uses seed+991)
    assert {s + 991 for s in calib_seeds}.isdisjoint({s + 991 for s in dev_seeds})


def test_calibration_reads_its_own_split_not_dev_or_sealed():
    from eval import calibration as cal

    # The split the collector reads is CALIB, a distinct directory from DEV.
    assert cal.CALIB_ROOT != cal.DEV_ROOT
    assert cal.CALIB_ROOT.name == "dataset_calib"
    assert cal.DEV_ROOT.name == "dataset"
    # 9999 is only ever the named sealed seed, never a CALIB generation seed.
    assert cal.SEALED_SEED not in cal.calib_tier_seeds()
