"""Real v2 engine through the service (moto): demo pair -> worker -> result + downloadable artifacts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.service._helpers import job_status, require_service_deps

require_service_deps()
pytest.importorskip("driftsense.v2.engine")


def test_v2_demo_job_end_to_end(client, settings, make_worker):
    from driftsense.contracts import LocalizeResult

    job_id = job_status(client.post("/v1/jobs", json={"demo_id": "demo_finfet_easy"}), 202)["job_id"]
    worker = make_worker(settings=replace(settings, engine="v2"))
    assert worker.poll_once(wait_seconds=0) == ["succeeded"]
    body = job_status(client.get(f"/v1/jobs/{job_id}"), 200)
    assert body["engine"] == "v2" and body["result"]["engine"] == "v2"
    result = LocalizeResult.model_validate(body["result"])
    # every listed artifact is a jobs/<id>/<name> key that the download route serves
    names = []
    for art in result.artifacts:
        assert art.uri.startswith(f"jobs/{job_id}/"), art.uri
        name = art.uri.rsplit("/", 1)[-1]
        r = client.get(f"/v1/jobs/{job_id}/artifacts/{name}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
        names.append(name)
    assert {"overlay.png", "top5_gallery.png"} <= set(names)
    # v2 heatmaps carry no bytes: dropped with an explicit warning instead of a dangling uri
    assert any(w.startswith("service: artifacts not materialized") for w in result.warnings)
    r = client.get(f"/v1/jobs/{job_id}/artifacts/result.json")
    assert r.status_code == 200
    assert LocalizeResult.model_validate_json(r.content) == result


def test_settings_default_engine_is_v2():
    from service.settings import Settings

    assert Settings.from_env({}).engine == "v2"
    assert Settings.from_env({"DRIFTSENSE_ENGINE": "fake"}).engine == "fake"
