from __future__ import annotations

import io
import json

from tests.service._helpers import require_service_deps

require_service_deps()


def test_emf_line_format():
    from service import emf

    line = emf.metric_line(
        {"JobLatencyMs": (12.5, "Milliseconds"), "JobsSucceeded": (1, "Count")},
        {"Engine": "fake"},
        {"job_id": "j1"},
        timestamp_ms=1700000000000,
    )
    doc = json.loads(line)
    assert "\n" not in line
    meta = doc["_aws"]
    assert meta["Timestamp"] == 1700000000000
    (directive,) = meta["CloudWatchMetrics"]
    assert directive["Namespace"] == "DriftSense"
    assert directive["Dimensions"] == [["Engine"]]
    assert directive["Metrics"] == [
        {"Name": "JobLatencyMs", "Unit": "Milliseconds"},
        {"Name": "JobsSucceeded", "Unit": "Count"},
    ]
    # every metric and dimension referenced must be a top-level member
    assert doc["JobLatencyMs"] == 12.5 and doc["JobsSucceeded"] == 1 and doc["Engine"] == "fake"
    assert doc["job_id"] == "j1"


def test_emf_job_lines_and_stage_dimension():
    from service import emf

    buf = io.StringIO()
    emf.emit(
        emf.job_lines(
            job_id="j",
            engine="v2",
            outcome="failed",
            latency_ms=5.0,
            timings=[("lattice", 1.0), ("score", 2.0)],
        ),
        buf,
    )
    docs = [json.loads(x) for x in buf.getvalue().splitlines()]
    assert docs[0]["JobsFailed"] == 1 and docs[0]["JobsSucceeded"] == 0 and docs[0]["JobLatencyMs"] == 5.0
    stage_docs = docs[1:]
    assert [d["Stage"] for d in stage_docs] == ["lattice", "score"]
    for d in stage_docs:
        directive = d["_aws"]["CloudWatchMetrics"][0]
        assert directive["Dimensions"] == [["Stage"]]
        assert directive["Metrics"] == [{"Name": "StageMs", "Unit": "Milliseconds"}]
        assert isinstance(d["StageMs"], float)


def test_settings_parsing():
    import pytest

    from service.settings import Settings

    s = Settings.from_env({"JOBS_TABLE": "t", "JOBS_QUEUE_URL": "q", "DATA_BUCKET": "b"})
    assert (s.max_upload_mb, s.job_ttl_days, s.engine, s.ui_origin) == (
        20,
        7,
        "v2",
        None,
    )  # service default engine is v2
    assert s.max_upload_bytes == 20 * 1024 * 1024
    s.require_aws()
    s2 = Settings.from_env(
        {
            "MAX_UPLOAD_MB": "5",
            "JOB_TTL_DAYS": "2",
            "DRIFTSENSE_ENGINE": "v2",
            "UI_ORIGIN": "https://ui",
            "ENGINE_GIT_SHA": "abc",
        }
    )
    assert (s2.max_upload_mb, s2.job_ttl_days, s2.engine, s2.ui_origin, s2.engine_git_sha) == (
        5,
        2,
        "v2",
        "https://ui",
        "abc",
    )
    with pytest.raises(RuntimeError):
        s2.require_aws()
    with pytest.raises(ValueError):
        Settings.from_env({"DRIFTSENSE_ENGINE": "v9"})
    with pytest.raises(ValueError):
        Settings.from_env({"MAX_UPLOAD_MB": "lots"})


def test_e2e_local_main(service_env, monkeypatch, capsys):
    from driftsense.contracts import JobStatus
    from service import e2e_local

    monkeypatch.delenv("JOBS_QUEUE_URL", raising=False)
    assert e2e_local.main([]) == 0
    body = json.loads(capsys.readouterr().out)
    assert JobStatus.model_validate(body).status == "succeeded"


def test_e2e_local_hard_demo_agent_awaits_approval(service_env, monkeypatch, capsys):
    from service import e2e_local

    monkeypatch.delenv("JOBS_QUEUE_URL", raising=False)
    assert e2e_local.main(["--demo", "demo_dram_hard", "--agent"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "awaiting_approval"
