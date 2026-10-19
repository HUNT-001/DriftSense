"""A7 — the pre-registered sealed-eval protocol is frozen.

These tests pin eval/protocol.json: its canonical sha256 (the value a human copies
into the protocol-v1 tag message) and the handful of fields a later sealed-run
session must not silently drift -- engine SHA, dataset seed, tiers, families, CI
method, arms, and the decision battery. They read ONLY eval/protocol.json; no sealed
data is touched.

The freeze hash is the sha256 of the canonical JSON (sorted keys, tight separators,
ASCII) of the object WITH the top-level ``hash`` key removed -- so it is independent
of line endings and formatting (the repo uses core.autocrlf=true) and does not depend
on the stored value itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROTOCOL_JSON = REPO / "eval" / "protocol.json"

# The frozen pre-registration hash. Changing protocol.json in any way changes this and
# breaks the test on purpose: re-registration is a deliberate, reviewed act.
EXPECTED_SHA256 = "776ed76e7434087905e0a309ddc64a5b79689e8331ecd43936345a3851f6d6c0"


def _canonical_sha256(obj: dict) -> str:
    """sha256 of the canonical JSON of ``obj`` with the top-level 'hash' key removed."""
    body = {k: v for k, v in obj.items() if k != "hash"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def protocol() -> dict:
    return json.loads(PROTOCOL_JSON.read_text(encoding="utf-8"))


def test_protocol_json_exists_and_parses(protocol: dict) -> None:
    assert protocol["protocol_id"] == "driftsense.sealed-eval"
    assert protocol["protocol_version"] == "1.0.0"
    assert protocol["status"] == "pre-registered"


def test_canonical_sha256_matches_frozen_value(protocol: dict) -> None:
    assert _canonical_sha256(protocol) == EXPECTED_SHA256


def test_stored_hash_matches_recomputed(protocol: dict) -> None:
    # The value the file carries must equal what the canonicalization recomputes.
    assert protocol["hash"]["algorithm"] == "sha256"
    assert protocol["hash"]["value"] == _canonical_sha256(protocol)
    assert protocol["hash"]["value"] == EXPECTED_SHA256


def test_engine_pinned_to_v2_freeze(protocol: dict) -> None:
    eng = protocol["engine"]
    assert eng["tag"] == "v2-freeze"
    # 40-hex commit SHA that v2-freeze resolves to.
    assert eng["git_sha"] == "f9395b84dd654ff1d537f6817c258a0f2b9373de"
    assert len(eng["git_sha"]) == 40
    assert eng["selector_env"] == "DRIFTSENSE_ENGINE=v2"


def test_dataset_seed_and_strata(protocol: dict) -> None:
    ds = protocol["dataset"]
    assert ds["name"] == "sealed"
    assert ds["seed"] == 9999
    assert ds["tiers"] == ["clean", "nominal", "hard", "ambiguous"]
    assert ds["structured_tiers"] == ["clean", "nominal", "hard"]
    assert ds["families"] == ["finfet", "dram", "grid", "staggered"]


def test_ci_method_is_wilson_95(protocol: dict) -> None:
    ci = protocol["ci"]
    assert ci["method"] == "wilson"
    assert ci["level"] == 0.95
    assert ci["sided"] == "two-sided"


def test_arms_are_the_pre_registered_set(protocol: dict) -> None:
    ids = [arm["id"] for arm in protocol["arms"]]
    assert ids == ["ncc", "v1", "v2_single", "agent_rules", "ablation_ladder"]
    ladder = next(a for a in protocol["arms"] if a["id"] == "ablation_ladder")
    rungs = [r["id"] for r in ladder["rungs"]]
    assert rungs == [
        "full",
        "no_cascade",
        "no_contour_subpixel",
        "no_calibration",
        "no_roughness_fusion",
        "lattice_only",
    ]


def test_metrics_cover_the_required_quantities(protocol: dict) -> None:
    keys = {m["key"] for m in protocol["metrics"]}
    required = {
        "top1.<tier>",
        "top1.family.<family>",
        "top1.pooled",
        "set_recall.<tier>",
        "false_confident.pooled",
        "ece",
        "subpixel_nm_error",
        "<arm>.success",
        "<arm>.false_commit",
        "<arm>.escalation",
        "<arm>.mean_steps",
        "<arm>.mean_dose",
    }
    assert required <= keys


def test_decision_battery_is_complete_and_frozen(protocol: dict) -> None:
    ids = [r["id"] for r in protocol["decision"]["rules"]]
    assert ids == [
        f"D{i}_{s}"
        for i, s in enumerate(
            [
                "accuracy_floor",
                "noninferior_to_v1",
                "beats_ncc",
                "per_tier_structured",
                "per_family",
                "set_recall",
                "false_confident",
                "calibration",
                "subpixel",
                "agent_false_commit",
                "agent_not_worse",
                "agent_budget",
            ],
            start=1,
        )
    ]
    assert "ALL of D1..D12" in protocol["decision"]["overall_pass"]


def test_protocol_does_not_embed_sealed_data(protocol: dict) -> None:
    # Guardrail: pre-registration must stay metadata-only. No results/metric values.
    blob = json.dumps(protocol)
    assert '"metrics": {' not in blob  # that shape belongs to a ResultsFile, not here
    assert "9999" in blob  # the seed is named, not the data
