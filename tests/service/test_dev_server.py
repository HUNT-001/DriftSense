"""service.dev_server: the app it builds serves the API, the SPA and a same-origin config.json."""

import os

import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from service.e2e_local import LOCAL_ENV


@pytest.fixture
def ui_dir(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>DriftSense</title>")
    return tmp_path


def test_dev_server_app_serves_api_ui_and_config(ui_dir, monkeypatch):
    for key, value in LOCAL_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("DRIFTSENSE_ENGINE", "fake")
    from service.dev_server import build

    with mock_aws():
        app, worker = build(ui_dir)
        client = TestClient(app, base_url="http://localhost:8000")
        assert client.get("/healthz").json() == {"ok": True}
        assert client.get("/config.json").json() == {"apiBase": "http://localhost:8000"}
        assert "DriftSense" in client.get("/").text
        r = client.post("/v1/jobs", json={"demo_id": "demo_finfet_easy"})
        assert r.status_code == 202
        assert worker.poll_once(wait_seconds=0)
        assert client.get(f"/v1/jobs/{r.json()['job_id']}").json()["status"] == "succeeded"
    os.environ.pop("JOBS_QUEUE_URL", None)
