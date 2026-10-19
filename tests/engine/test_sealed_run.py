"""A9 — tests for the sealed-set runner and its statistics.

These run WITHOUT the sealed set: the guard must block by default, the stats are
pure, and the end-to-end wiring is exercised on a tiny DEV slice (outputs/dataset,
NOT sealed) through the same code path the sealed run uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from driftsense.contracts import Budget, ResultsFile
from eval import sealed_run as sr
from eval import stats


# --------------------------------------------------------------------------- #
# Guard: refuses by default                                                   #
# --------------------------------------------------------------------------- #
def test_guard_blocks_without_env(monkeypatch):
    monkeypatch.delenv("DRIFTSENSE_ALLOW_SEALED", raising=False)
    assert sr.sealed_unlocked() is False
    with pytest.raises(sr.SealedLocked):
        sr.require_unlock()


def test_guard_blocks_with_env_but_no_sentinel(monkeypatch):
    monkeypatch.setenv("DRIFTSENSE_ALLOW_SEALED", "1")
    # The unlock sentinel must not exist in a clean checkout.
    assert not sr.UNLOCK_SENTINEL.exists(), "eval/SEALED_UNLOCK must be absent in the repo"
    assert sr.sealed_unlocked() is False
    with pytest.raises(sr.SealedLocked):
        sr.require_unlock()


def test_main_sealed_arm_refuses_by_default(monkeypatch):
    monkeypatch.delenv("DRIFTSENSE_ALLOW_SEALED", raising=False)
    with pytest.raises(sr.SealedLocked):
        sr.main(["--arm", "v2_single"])


# --------------------------------------------------------------------------- #
# Protocol provenance                                                         #
# --------------------------------------------------------------------------- #
def test_live_protocol_hash_matches_pinned():
    protocol = sr.load_protocol()
    assert sr.protocol_sha256() == protocol["hash"]["value"]


def test_check_engine_sha_rejects_mismatch(monkeypatch):
    protocol = sr.load_protocol()
    monkeypatch.setattr(sr, "git_sha", lambda: "deadbeef" * 5)
    with pytest.raises(sr.SealedLocked):
        sr.check_engine_sha(protocol)


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def test_wilson_known_interval():
    pt, lo, hi = stats.wilson(50, 100)
    assert pt == 0.5
    assert lo == pytest.approx(0.404, abs=0.005)
    assert hi == pytest.approx(0.596, abs=0.005)


def test_wilson_zero_n_is_widest():
    assert stats.wilson(0, 0) == (0.0, 0.0, 1.0)


def test_mcnemar_exact_small_and_symmetric():
    a = np.array([1] * 10 + [0] * 10)
    b = np.array([0] * 10 + [0] * 10)  # A right & B wrong on first 10; no reverse
    out = stats.mcnemar(a, b)
    assert out["b"] == 10 and out["c"] == 0
    assert out["exact"] == 1.0 and out["p_value"] < 0.05
    sym = stats.mcnemar(a, a)
    assert sym["p_value"] == 1.0


def test_ece_equal_count_perfect_is_zero():
    p = np.array([0.0, 0.0, 1.0, 1.0])
    correct = np.array([0.0, 0.0, 1.0, 1.0])
    assert stats.ece_equal_count(p, correct, n_bins=2) == pytest.approx(0.0, abs=1e-9)


def test_ece_binning_tag_is_equal_count():
    assert stats.ECE_BINNING == "equal-count"


def test_bca_bootstrap_contains_point():
    x = np.arange(1.0, 10.0)
    pt, lo, hi = stats.bca_bootstrap(x, stats.median, n_boot=500)
    assert pt == 5.0
    assert lo <= pt <= hi


def test_risk_coverage_monotone_coverage():
    conf = np.array([0.9, 0.8, 0.1, 0.2])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    cov, risk = stats.risk_coverage(conf, correct)
    assert cov[0] < cov[-1]
    assert risk[0] == 0.0  # most-confident commit is correct


# --------------------------------------------------------------------------- #
# End-to-end wiring on DEV (no sealed access)                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_selftest_arms_validate_as_resultsfile(tmp_path):
    protocol = sr.load_protocol()
    budget = Budget(max_candidates=5)
    for arm in ("ncc", "v1", "v2_single"):
        rf, recs = sr.run_pair_arm(
            sr.DEV_ROOT, arm, None, budget, 1, "dev", protocol, sealed=False, out_dir=tmp_path
        )
        # round-trips through the frozen contract schema
        ResultsFile.model_validate_json(rf.model_dump_json())
        assert rf.dataset == "dev"
        assert rf.metrics["top1.pooled"].n == len(recs)


@pytest.mark.slow
def test_ablation_full_equals_v2_single_on_dev():
    budget = Budget(max_candidates=5)
    pair = next(sr.iter_pairs(sr.DEV_ROOT, "clean", 1))
    v2 = sr.run_localize("v2_single", None, pair.template, pair.search, budget)
    full = sr.run_localize("ablation", "full", pair.template, pair.search, budget)
    assert full.offset.x_px == pytest.approx(v2.offset.x_px)
    assert full.offset.y_px == pytest.approx(v2.offset.y_px)
    assert full.candidates[0].score == pytest.approx(v2.candidates[0].score)


@pytest.mark.slow
def test_reconstructed_rungs_run_and_are_flagged(tmp_path):
    protocol = sr.load_protocol()
    budget = Budget(max_candidates=5)
    for rung in ("no_roughness_fusion", "lattice_only"):
        rf, _ = sr.run_pair_arm(
            sr.DEV_ROOT, "ablation", rung, budget, 1, "dev", protocol, sealed=False, out_dir=tmp_path
        )
        ResultsFile.model_validate_json(rf.model_dump_json())
        assert rf.metrics[f"{rung}.reconstructed"].value == 1.0


# --------------------------------------------------------------------------- #
# Decision battery                                                            #
# --------------------------------------------------------------------------- #
def test_grade_marks_missing_inputs_na():
    protocol = sr.load_protocol()
    verdict = sr.grade({}, protocol)
    assert verdict["overall_pass"] is False
    assert verdict["n_decided"] == 0
    assert all(r["pass"] is None for r in verdict["rules"].values())


def test_grade_passes_on_synthetic_metrics():
    from driftsense.contracts import Metric

    def rate(v, lo, hi, n=1000):
        return Metric(value=v, unit="rate", ci_low=lo, ci_high=hi, n=n)

    v2 = {
        "top1.pooled": rate(0.97, 0.955, 0.98),
        "top1.clean": rate(0.99, 0.95, 1.0),
        "top1.nominal": rate(0.97, 0.90, 0.99),
        "top1.hard": rate(0.95, 0.88, 0.98),
        "top1.family.finfet": rate(0.96, 0.90, 0.99),
        "top1.family.dram": rate(0.95, 0.88, 0.98),
        "set_recall.pooled_structured": rate(0.995, 0.98, 1.0),
        "false_confident.pooled": rate(0.01, 0.0, 0.03),
        "ece": Metric(value=0.04, unit="fraction"),
        "subpixel_nm_error.median": Metric(value=0.3, unit="nm"),
        "subpixel_nm_error.p95": Metric(value=0.8, unit="nm"),
    }
    ag = {
        "agent_rules.false_commit": rate(0.0, 0.0, 0.02),
        "agent_rules.success": rate(0.98, 0.95, 0.99),
        "v2_single.success": rate(0.93, 0.88, 0.96),
        "agent_rules.mean_steps": Metric(value=1.8, unit="acquisitions"),
        "agent_rules.mean_dose": Metric(value=10e6, unit="electrons"),
        "dose_budget": Metric(value=48e6, unit="electrons"),
    }
    verdict = sr.grade(
        {
            "v2_single": v2,
            "v1": {"top1.pooled": rate(0.96, 0.94, 0.975)},
            "ncc": {"top1.pooled": rate(0.80, 0.77, 0.83)},
            "agent_rules": ag,
        },
        protocol=sr.load_protocol(),
    )
    assert verdict["overall_pass"] is True
    assert verdict["n_decided"] == verdict["n_rules"] == 12
