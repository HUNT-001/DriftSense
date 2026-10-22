"""Placeholder API Lambda (HTTP API payload v2.0).

Used with ``-c api_asset=placeholder`` (tests, or a checkout without service/); the real bundle is ``service.api.handler``.
"""

from __future__ import annotations

import json
import os


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event, context):  # noqa: ARG001 - Lambda signature
    http = (event or {}).get("requestContext", {}).get("http", {})
    method = http.get("method", "GET").upper()
    path = (event or {}).get("rawPath") or http.get("path") or "/"

    if method == "GET" and path == "/healthz":
        return _response(200, {"ok": True})
    if method == "GET" and path == "/v1/version":
        return _response(
            200,
            {"placeholder": True, "engine": os.environ.get("DRIFTSENSE_ENGINE", "unknown")},
        )
    return _response(404, {"error": "not_found", "path": path})
