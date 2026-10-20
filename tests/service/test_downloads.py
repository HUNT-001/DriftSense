from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest

from tests.service._helpers import (
    error_info,
    job_status,
    random_pair,
    require_service_deps,
)

require_service_deps()

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


@pytest.fixture
def job_id(client):
    return job_status(client.post("/v1/jobs", json={"demo_id": "demo_grid"}), 202)["job_id"]


def test_artifact_png_streams_with_type_and_cache(client, services, job_id):
    services.artifacts.put_bytes(f"jobs/{job_id}/heatmap.png", PNG, "application/octet-stream")
    r = client.get(f"/v1/jobs/{job_id}/artifacts/heatmap.png")
    assert r.status_code == 200 and r.content == PNG
    assert r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == "private, max-age=300"


def test_result_json_artifact_after_worker(client, make_worker, job_id):
    assert make_worker().poll_once(wait_seconds=0) == ["succeeded"]
    r = client.get(f"/v1/jobs/{job_id}/artifacts/result.json")
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    from driftsense.contracts import LocalizeResult

    LocalizeResult.model_validate_json(r.content)


def test_artifact_404s(client, job_id):
    error_info(client.get(f"/v1/jobs/{job_id}/artifacts/missing.png"), 404, "not_found")
    error_info(client.get("/v1/jobs/" + "e" * 32 + "/artifacts/heatmap.png"), 404, "not_found")
    error_info(client.get("/v1/jobs/nope/artifacts/heatmap.png"), 404, "not_found")


@pytest.mark.parametrize(
    "name", ["..png", "a..png", "...", "bad name.png", "x" * 129, "a\\b.png", "h%C3%A9.png", "$.png"]
)
def test_artifact_bad_names_400(client, job_id, name):
    error_info(client.get(f"/v1/jobs/{job_id}/artifacts/{name}"), 400, "bad_request")


def test_artifact_slash_never_reaches_s3(client, services, job_id):
    services.artifacts.put_bytes(f"jobs/{job_id}/sub/x.png", PNG, "image/png")
    r = client.get(f"/v1/jobs/{job_id}/artifacts/sub%2Fx.png")
    assert r.status_code in (400, 404)
    error_info(r, r.status_code)


def test_large_artifact_redirects_to_presigned_get(client, services, job_id, monkeypatch):
    from service import api

    monkeypatch.setattr(api, "MAX_INLINE_BYTES", 16)
    services.artifacts.put_bytes(f"jobs/{job_id}/overlay.png", PNG, "image/png")
    r = client.get(f"/v1/jobs/{job_id}/artifacts/overlay.png", follow_redirects=False)
    assert r.status_code == 307
    loc = urlparse(r.headers["location"])
    q = parse_qs(loc.query)
    assert loc.path.endswith(f"jobs/{job_id}/overlay.png")
    assert q["X-Amz-Expires"] == ["300"] and q["response-content-type"] == ["image/png"]


def test_real_5mb_threshold(client, services, job_id):
    big = b"0" * (5 * 1024 * 1024 + 1)
    services.artifacts.put_bytes(f"jobs/{job_id}/big.json", big, "application/json")
    services.artifacts.put_bytes(f"jobs/{job_id}/edge.json", big[:-1], "application/json")
    assert client.get(f"/v1/jobs/{job_id}/artifacts/big.json", follow_redirects=False).status_code == 307
    assert client.get(f"/v1/jobs/{job_id}/artifacts/edge.json", follow_redirects=False).status_code == 200


def test_trace_not_ready_then_served(client, services, job_id, monkeypatch):
    error_info(client.get(f"/v1/jobs/{job_id}/trace"), 404, "not_ready")
    key = "traces/run-1.jsonl"
    services.jobs.transition(job_id, "queued", allowed_from=("queued",), set_fields={"trace_key": key})
    assert job_status(client.get(f"/v1/jobs/{job_id}"), 200)["trace_key"] == key
    error_info(client.get(f"/v1/jobs/{job_id}/trace"), 404, "not_ready")  # key set, object not written yet
    lines = [
        {"v": "1", "run_id": "run-1", "step": i, "ts": "2026-10-11T00:00:00Z", "kind": "observe"}
        for i in range(3)
    ]
    body = "\n".join(json.dumps(x) for x in lines) + "\n"
    services.artifacts.put_bytes(key, body.encode(), "application/octet-stream")
    r = client.get(f"/v1/jobs/{job_id}/trace")
    assert r.status_code == 200 and r.headers["content-type"] == "application/x-ndjson"
    from driftsense.contracts import TraceEvent

    assert [TraceEvent.model_validate_json(x).step for x in r.text.splitlines()] == [0, 1, 2]
    from service import api

    monkeypatch.setattr(api, "MAX_INLINE_BYTES", 8)
    red = client.get(f"/v1/jobs/{job_id}/trace", follow_redirects=False)
    assert red.status_code == 307 and "traces/run-1.jsonl" in red.headers["location"]


def test_trace_unknown_job_404(client):
    error_info(client.get("/v1/jobs/" + "d" * 32 + "/trace"), 404, "not_found")


def test_worker_artifact_uri_basename_is_route_name(client, services, make_worker, tmp_path):
    from driftsense.api import localize
    from driftsense.contracts import Artifact

    local = tmp_path / "heat map.png"
    local.write_bytes(PNG)

    def with_artifacts(*args, **kwargs):
        r = localize(*args, **kwargs)
        return r.model_copy(update={"artifacts": [Artifact(kind="heatmap", uri=str(local))]})

    upload = "uploads/" + "c" * 32
    t, s = random_pair()
    services.artifacts.put_bytes(f"{upload}/template.png", t, "image/png")
    services.artifacts.put_bytes(f"{upload}/search.png", s, "image/png")
    jid = job_status(
        client.post(
            "/v1/jobs", json={"template_key": f"{upload}/template.png", "search_key": f"{upload}/search.png"}
        ),
        202,
    )["job_id"]
    assert make_worker(localize_fn=with_artifacts).poll_once(wait_seconds=0) == ["succeeded"]
    (art,) = job_status(client.get(f"/v1/jobs/{jid}"), 200)["result"]["artifacts"]
    assert art["uri"] == f"jobs/{jid}/heat_map.png"
    name = art["uri"].rsplit("/", 1)[-1]
    r = client.get(f"/v1/jobs/{jid}/artifacts/{name}")
    assert r.status_code == 200 and r.content == PNG and r.headers["content-type"] == "image/png"


def test_artifact_object_name_sanitizer():
    from service.worker import artifact_object_name

    assert artifact_object_name("heatmap.png") == "heatmap.png"
    assert artifact_object_name("a b/..c.png") == "a_b__c.png"
    long = artifact_object_name("x" * 300 + ".png")
    assert len(long) <= 128 and long.endswith(".png")
