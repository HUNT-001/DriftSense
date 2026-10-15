import json
import pathlib

import cv2
import numpy as np
import pytest

from driftsense.api import available_engines, localize
from driftsense.contracts import (
    CONTRACT_VERSION,
    Budget,
    DriftSenseInputError,
    JobCreate,
    JobStatus,
    LocalizeResult,
    Margin,
    Mode,
    TraceEvent,
    export_schemas,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _png(a):
    ok, buf = cv2.imencode(".png", a)
    assert ok
    return buf.tobytes()


@pytest.fixture
def pair():
    rng = np.random.default_rng(0)
    s = (rng.random((256, 256)) * 255).astype(np.uint8)
    return _png(s[64:128, 80:144].copy()), _png(s)


def test_committed_schemas_match_models(tmp_path):
    export_schemas(tmp_path)
    for f in sorted(tmp_path.glob("*.schema.json")):
        committed = ROOT / "contracts" / f.name
        assert committed.exists(), f"missing {committed}: run `make contracts`"
        assert json.loads(committed.read_text()) == json.loads(f.read_text()), (
            f"{f.name} drifted: run `make contracts` on a shared/* branch and get both approvals"
        )


@pytest.mark.parametrize("engine", ["fake", "v1", "v2"])
def test_engine_honours_contract(engine, pair, monkeypatch):
    if engine not in available_engines():
        pytest.skip(f"{engine} not importable yet")
    monkeypatch.setenv("DRIFTSENSE_ENGINE", engine)
    try:
        r = localize(*pair, mode=Mode.AUTO, budget=Budget())
    except NotImplementedError:
        pytest.skip(f"{engine} still a stub")
    assert isinstance(r, LocalizeResult) and r.contract_version == CONTRACT_VERSION
    LocalizeResult.model_validate_json(r.model_dump_json())
    assert 1 <= len(r.candidates) <= Budget().max_candidates
    assert r.total_ms >= 0


def test_fake_is_deterministic_and_correct(pair):
    a = localize(*pair, engine="fake")
    b = localize(*pair, engine="fake")
    assert a.offset == b.offset and (a.offset.x_px, a.offset.y_px) == (80.0, 64.0)


def test_bad_input_raises():
    with pytest.raises(DriftSenseInputError):
        localize(b"not an image", b"also not", engine="fake")


def test_margin_decision_invariant():
    with pytest.raises(ValueError):
        Margin(margin=0.1, tau=0.2, calibrated=True, decision="commit")


def test_trace_and_job_roundtrip(pair):
    r = localize(*pair, engine="fake")
    ev = TraceEvent(
        run_id="r1",
        step=0,
        ts="2026-10-11T00:00:00Z",
        kind="decision",
        margin=r.margin.margin,
        tau=r.margin.tau,
        because="margin vs tau",
    )
    assert TraceEvent.model_validate_json(ev.model_dump_json()) == ev
    js = JobStatus(
        job_id="j1",
        status="succeeded",
        created_at="2026-10-11T00:00:00Z",
        updated_at="2026-10-11T00:00:01Z",
        engine="fake",
        result=r,
    )
    assert JobStatus.model_validate_json(js.model_dump_json()).result == r
    with pytest.raises(ValueError):
        JobCreate()
