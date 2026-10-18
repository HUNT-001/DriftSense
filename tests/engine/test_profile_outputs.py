"""Validate the P1 golden-baseline JSONs against the ResultsFile contract.

These files are produced by `python -m eval.profile_v1`. They are committed
golden artifacts; this test guards their shape so downstream (G2, the service
latency assumption) can rely on them. It does NOT re-run the profiler.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftsense.contracts import ResultsFile

ROOT = Path(__file__).resolve().parent.parent.parent
ACC = ROOT / "results" / "accuracy"

BASELINE = ACC / "baseline_v1.json"
PROFILE = ACC / "profile_v1.json"

BASELINE_KEYS = {"top1_accuracy", "recall_at_20", "latency_per_pair_ms"}
PROFILE_KEYS = {"matchtemplate_ms", "fingerprint_ms", "latency_per_pair_ms"}


@pytest.mark.parametrize("path", [BASELINE, PROFILE], ids=["baseline", "profile"])
def test_results_file_validates(path: Path) -> None:
    assert path.exists(), f"{path} missing — run `python -m eval.profile_v1`"
    rf = ResultsFile.model_validate_json(path.read_text())
    assert rf.schema_id == "driftsense.results/1"
    assert rf.dataset == "dev"
    assert rf.produced_by == "eval/profile_v1.py"
    assert rf.git_sha
    assert rf.metrics, "metrics must not be empty"


def test_baseline_metrics_present_and_bounded() -> None:
    rf = ResultsFile.model_validate_json(BASELINE.read_text())
    assert BASELINE_KEYS <= set(rf.metrics)
    for key in ("top1_accuracy", "recall_at_20"):
        m = rf.metrics[key]
        assert m.unit == "fraction"
        assert 0.0 <= m.value <= 1.0
        assert m.n and m.n > 0
        assert m.ci_low is not None and m.ci_high is not None
        assert m.ci_low <= m.value <= m.ci_high
    lat = rf.metrics["latency_per_pair_ms"]
    assert lat.unit == "ms" and lat.value > 0


def test_profile_metrics_present() -> None:
    rf = ResultsFile.model_validate_json(PROFILE.read_text())
    assert PROFILE_KEYS <= set(rf.metrics)
    assert any(k.startswith("hot.") for k in rf.metrics), "expected cProfile hot spots"
    assert any(k.startswith("claimed.") for k in rf.metrics), "expected claimed figures"
    for key in PROFILE_KEYS:
        assert rf.metrics[key].unit == "ms"


def test_both_files_are_json() -> None:
    for path in (BASELINE, PROFILE):
        json.loads(path.read_text())  # fail loudly on hand-editing that breaks JSON
