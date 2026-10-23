from __future__ import annotations

import pytest

REQUIRED = ("fastapi", "httpx", "mangum", "moto", "boto3", "driftsense", "driftsense.api")


def require_service_deps() -> None:
    for mod in REQUIRED:
        pytest.importorskip(mod)


def job_status(resp, expected_code: int | None = None) -> dict:
    """Assert the response body is a contract-valid JobStatus and return it."""
    from driftsense.contracts import JobStatus

    if expected_code is not None:
        assert resp.status_code == expected_code, resp.text
    body = resp.json()
    JobStatus.model_validate(body)
    return body


def error_info(resp, expected_code: int, code: str | None = None) -> dict:
    from driftsense.contracts import ErrorInfo

    assert resp.status_code == expected_code, resp.text
    body = resp.json()
    ErrorInfo.model_validate(body)
    if code is not None:
        assert body["code"] == code, body
    return body


def png_bytes(arr) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def random_pair(seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    s = (rng.random((256, 256)) * 255).astype(np.uint8)
    return png_bytes(s[64:128, 80:144].copy()), png_bytes(s)
