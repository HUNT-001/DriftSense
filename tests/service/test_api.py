from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest

from tests.service._helpers import error_info, job_status, require_service_deps

require_service_deps()

UPLOAD_KEY = "uploads/" + "a" * 32 + "/template.png"
SEARCH_KEY = "uploads/" + "a" * 32 + "/search.png"


def _queue_bodies(services) -> list[dict]:
    out = []
    while True:
        resp = services.queue.client.receive_message(
            QueueUrl=services.queue.url, MaxNumberOfMessages=10, WaitTimeSeconds=0, VisibilityTimeout=60
        )
        msgs = resp.get("Messages", [])
        if not msgs:
            return out
        out.extend(json.loads(m["Body"]) for m in msgs)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_version_reports_cv2_and_contract(client):
    from driftsense.contracts import CONTRACT_VERSION

    body = client.get("/v1/version").json()
    assert body["git_sha"] == "testsha"
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["opencv_version"] and body["opencv_file"].endswith(".py")
    assert body["python_version"] and body["arch"]


def test_version_without_cv2(client, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "cv2":
            raise ImportError("no cv2 on lambda")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    body = client.get("/v1/version").json()
    assert body["opencv_version"] is None and body["opencv_file"] is None


def test_demos_list(client):
    demos = client.get("/v1/demos").json()
    ids = {d["demo_id"] for d in demos}
    assert {"demo_finfet_easy", "demo_dram_hard", "demo_grid", "demo_staggered"} <= ids
    assert all(d["title"] for d in demos)


def test_create_job_demo_queued(client, services):
    from driftsense.contracts import CONTRACT_VERSION

    body = job_status(client.post("/v1/jobs", json={"demo_id": "demo_grid"}), 202)
    assert body["status"] == "queued" and len(body["job_id"]) == 32
    item = services.jobs.get_item(body["job_id"])
    assert item["contract_version"] == CONTRACT_VERSION
    assert item["expires_at"] > 0
    assert _queue_bodies(services) == [{"job_id": body["job_id"]}]
    got = job_status(client.get(f"/v1/jobs/{body['job_id']}"), 200)
    assert got["status"] == "queued" and got["job_id"] == body["job_id"]


def test_create_job_upload_keys(client):
    body = job_status(
        client.post(
            "/v1/jobs",
            json={
                "template_key": UPLOAD_KEY,
                "search_key": SEARCH_KEY,
                "mode": "fast",
                "budget": {"max_candidates": 7},
            },
        ),
        202,
    )
    assert body["status"] == "queued"


@pytest.mark.parametrize(
    "payload",
    [
        {},  # JobCreate needs a source
        {"template_key": UPLOAD_KEY},  # only one key
        {"demo_id": "demo_nope"},  # unknown demo
        {"template_key": "jobs/x/result.json", "search_key": SEARCH_KEY},  # not an uploads/ key
        {"demo_id": "demo_grid", "mode": "warp"},  # bad enum
        {"demo_id": "demo_grid", "extra": 1},  # extra=forbid
        {"demo_id": "demo_grid", "client_token": "x" * 129},
    ],
)
def test_create_job_validation_422(client, services, payload):
    error_info(client.post("/v1/jobs", json=payload), 422, "validation_error")
    assert _queue_bodies(services) == []


def test_idempotent_client_token(client, services):
    payload = {"demo_id": "demo_grid", "client_token": "tok-1"}
    first = job_status(client.post("/v1/jobs", json=payload), 202)
    second = job_status(client.post("/v1/jobs", json=payload), 200)
    assert first["job_id"] == second["job_id"]
    assert len(_queue_bodies(services)) == 1
    token_item = services.jobs.get_item("token#tok-1")
    assert token_item["target_job_id"] == first["job_id"] and token_item["expires_at"] > 0
    # a different body with the same token is a conflict, not a silent replay
    error_info(
        client.post("/v1/jobs", json={"demo_id": "demo_dram_hard", "client_token": "tok-1"}),
        409,
        "idempotency_conflict",
    )
    # the token item is not reachable as a job
    error_info(client.get("/v1/jobs/token%23tok-1"), 404, "not_found")


def test_unknown_job_404_errorinfo(client):
    error_info(client.get("/v1/jobs/" + "0" * 32), 404, "not_found")
    error_info(client.get("/v1/jobs/not-a-job"), 404, "not_found")


def test_upload_presign_binds_content_type_and_length(client):
    r = client.post(
        "/v1/uploads", json={"filename": "template.png", "content_type": "image/png", "size_bytes": 12345}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["key"] == f"uploads/{body['upload_id']}/template.png"
    assert body["headers"] == {"Content-Type": "image/png", "Content-Length": "12345"}
    assert body["expires_in"] == 900 and body["method"] == "PUT"
    q = parse_qs(urlparse(body["url"]).query)
    signed = q["X-Amz-SignedHeaders"][0].split(";")
    assert "content-type" in signed and "content-length" in signed
    assert q["X-Amz-Expires"] == ["900"]
    assert urlparse(body["url"]).path.endswith(body["key"])
    # second file of the pair shares the upload_id; role may be explicit, tiff keeps its suffix
    r2 = client.post(
        "/v1/uploads",
        json={
            "filename": "scan_01.tiff",
            "content_type": "image/tiff",
            "size_bytes": 10,
            "role": "search",
            "upload_id": body["upload_id"],
        },
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["key"] == f"uploads/{body['upload_id']}/search.tiff"


def test_upload_presign_put_roundtrip(client, services):
    """The presigned URL actually accepts the declared upload (moto validates the request)."""
    data = b"\x89PNG\r\n\x1a\n" + b"0" * 100
    body = client.post(
        "/v1/uploads", json={"filename": "search.png", "content_type": "image/png", "size_bytes": len(data)}
    ).json()
    requests = pytest.importorskip("requests")
    resp = requests.put(body["url"], data=data, headers=body["headers"], timeout=5)
    assert resp.status_code == 200, resp.text
    assert services.artifacts.get_bytes(body["key"]) == data


@pytest.mark.parametrize(
    "payload,status,code",
    [
        (
            {"filename": "template.png", "content_type": "image/png", "size_bytes": 20 * 1024 * 1024 + 1},
            413,
            "too_large",
        ),
        (
            {"filename": "template.jpg", "content_type": "image/jpeg", "size_bytes": 10},
            415,
            "unsupported_media_type",
        ),
        ({"filename": "template.png", "content_type": "image/png", "size_bytes": 0}, 422, "validation_error"),
        ({"filename": "photo.png", "content_type": "image/png", "size_bytes": 10}, 422, "validation_error"),
        (
            {"filename": "template.png", "content_type": "image/png", "size_bytes": 10, "upload_id": "../x"},
            422,
            "validation_error",
        ),
    ],
)
def test_upload_rejections(client, payload, status, code):
    error_info(client.post("/v1/uploads", json=payload), status, code)


def test_upload_limit_follows_env(make_client, settings):
    from dataclasses import replace

    c = make_client(replace(settings, max_upload_mb=1))
    error_info(
        c.post(
            "/v1/uploads",
            json={"filename": "template.png", "content_type": "image/png", "size_bytes": 1024 * 1024 + 1},
        ),
        413,
        "too_large",
    )


def test_cors_unset_allows_no_origin(client):
    r = client.get("/healthz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    pre = client.options(
        "/v1/jobs", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert "access-control-allow-origin" not in pre.headers


def test_cors_allows_only_ui_origin(make_client, settings):
    from dataclasses import replace

    c = make_client(replace(settings, ui_origin="https://ui.example"))
    ok = c.options(
        "/v1/jobs",
        headers={
            "Origin": "https://ui.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "https://ui.example"
    bad = c.options(
        "/v1/jobs", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert bad.headers.get("access-control-allow-origin") != "https://evil.example"
    simple = c.get("/healthz", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in simple.headers


def test_enqueue_failure_marks_job_failed(client, services, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("sqs down")

    monkeypatch.setattr(services.queue, "send", boom)
    error_info(client.post("/v1/jobs", json={"demo_id": "demo_grid"}), 503, "enqueue_failed")


def test_mangum_handler_http_api_v2_event(service_env):
    from service.api import handler

    event = {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": "/healthz",
        "rawQueryString": "",
        "headers": {"host": "api.example", "accept": "application/json"},
        "requestContext": {
            "http": {
                "method": "GET",
                "path": "/healthz",
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "pytest",
            },
            "stage": "$default",
            "requestId": "r1",
            "accountId": "123",
            "apiId": "a",
            "domainName": "api.example",
            "time": "11/Oct/2026:00:00:00 +0000",
            "timeEpoch": 0,
        },
        "isBase64Encoded": False,
    }

    class Ctx:
        function_name = "api"

    resp = handler(event, Ctx())
    assert resp["statusCode"] == 200 and json.loads(resp["body"]) == {"ok": True}


def test_api_import_does_not_need_cv2_numpy_or_engine(service_env):
    """Lambda may ship without cv2/numpy: importing the API must not pull them (or the engine) in."""
    import os
    import subprocess
    import sys

    code = (
        "import sys, warnings; warnings.simplefilter('ignore'); import service.api; "
        "bad = [m for m in ('cv2', 'numpy', 'driftsense.api', 'moto') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
