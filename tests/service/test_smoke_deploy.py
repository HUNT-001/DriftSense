"""scripts/smoke_deploy.py against a scripted fake API (no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("smoke_deploy", ROOT / "scripts" / "smoke_deploy.py")
smoke = importlib.util.module_from_spec(_spec)
sys.modules["smoke_deploy"] = smoke  # dataclasses resolve their module
_spec.loader.exec_module(smoke)

API = "https://api.example"
JOB = "a" * 32
WORKER_COOL = {"arch": "aarch64", "cool": True, "instance_type": "c8g.large", "opencv_version": "5.0.0"}


def fake_api(worker: dict, job_status: str = "succeeded"):
    def http(method, url, body=None, timeout=30.0):
        path = url.removeprefix(API)
        routes = {
            ("GET", "/healthz"): {"ok": True},
            ("GET", "/v1/version"): {"git_sha": "abc", "engine": "v2", "arch": "aarch64"},
            ("GET", "/v1/demos"): [{"demo_id": "demo_finfet_easy", "agent": False}],
            ("POST", "/v1/jobs"): {"job_id": JOB},
            ("GET", f"/v1/jobs/{JOB}"): {
                "status": job_status,
                "engine": "v2",
                "worker": worker,
                "result": {"artifacts": [{"uri": f"s3://b/jobs/{JOB}/heat.png"}], "offset": {}},
            },
        }
        if (method, path) in routes:
            return (202 if method == "POST" else 200), json.dumps(routes[method, path]).encode()
        if path == f"/v1/jobs/{JOB}/artifacts/heat.png":
            return 200, b"\x89PNG"
        return 404, b"{}"

    return http


@pytest.mark.parametrize(
    ("worker", "expect_cool", "rc"),
    [(WORKER_COOL, True, 0), ({**WORKER_COOL, "cool": False}, True, 1), ({"arch": "x86_64"}, False, 0)],
)
def test_worker_gate(monkeypatch, tmp_path, worker, expect_cool, rc):
    monkeypatch.setattr(smoke, "http", fake_api(worker))
    out = tmp_path / "ev.json"
    args = ["--api", API + "/", "--out", str(out)] + (["--expect-cool"] if expect_cool else [])
    assert smoke.main(args) == rc
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["passed"] is (rc == 0)
    assert [c["name"] for c in doc["checks"]][:3] == ["healthz", "version", "demos"]


def test_failed_job_fails_the_run(monkeypatch):
    monkeypatch.setattr(smoke, "http", fake_api(WORKER_COOL, job_status="failed"))
    assert smoke.main(["--api", API]) == 1


def test_outputs_file_supplies_urls(tmp_path):
    p = tmp_path / "cdk-outputs.json"
    p.write_text(
        json.dumps({"DriftSense-Api": {"ApiUrl": API + "/"}, "DriftSense-Storage": {"UiUrl": "https://ui"}}),
        encoding="utf-8",
    )
    assert smoke.from_outputs(p) == (API + "/", "https://ui")
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        smoke.from_outputs(p)
