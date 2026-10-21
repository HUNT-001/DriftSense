"""DriftSense API canary: GET /healthz and /v1/version, publish CanaryOk (1/0)."""

from __future__ import annotations

import json
import os
import urllib.request

TIMEOUT_S = 10


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"user-agent": "driftsense-canary/1"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310 - fixed https API URL
        if resp.status != 200:
            raise RuntimeError(f"{url} -> HTTP {resp.status}")
        return json.loads(resp.read().decode("utf-8"))


def check(base_url: str) -> tuple[bool, dict]:
    base = base_url.rstrip("/")
    detail: dict = {}
    try:
        health = _get(f"{base}/healthz")
        detail["healthz"] = health
        version = _get(f"{base}/v1/version")
        detail["version"] = version
        ok = health.get("ok") is True
    except Exception as exc:  # noqa: BLE001 - any failure means not OK
        detail["error"] = repr(exc)
        ok = False
    return ok, detail


def handler(event, context):  # noqa: ARG001 - Lambda signature
    import boto3  # provided by the Lambda runtime; imported lazily so check() is testable without it

    ok, detail = check(os.environ["API_URL"])
    boto3.client("cloudwatch").put_metric_data(
        Namespace=os.environ.get("METRIC_NAMESPACE", "DriftSense"),
        MetricData=[{"MetricName": "CanaryOk", "Value": 1.0 if ok else 0.0, "Unit": "Count"}],
    )
    print(json.dumps({"canary_ok": ok, **detail}))
    return {"ok": ok}
