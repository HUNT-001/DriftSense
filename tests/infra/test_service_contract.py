"""Infra <-> service contract (service/README.md on b/p4-service): env, IAM, queue timing."""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("aws_cdk")

from aws_cdk.assertions import Template  # noqa: E402

from .test_worker_stacks import _actions, _flatten_user_data, _resources, _statements  # noqa: E402

PREFIX = "DriftSense-"


@pytest.fixture(scope="module")
def worker_ud(default_app) -> str:
    (lt,) = Template.from_stack(default_app.worker).find_resources("AWS::EC2::LaunchTemplate").values()
    return _flatten_user_data(lt["Properties"]["LaunchTemplateData"]["UserData"])


def _api_fn(templates) -> dict:
    fns = templates[f"{PREFIX}Api"].find_resources("AWS::Lambda::Function")
    (fn,) = [f for f in fns.values() if f["Properties"].get("Runtime") == "python3.12"]
    return fn["Properties"]


def _by_action(statements: list[dict]) -> dict[str, list]:
    out: dict[str, list] = {}
    for st in statements:
        for action in _actions(st):
            out.setdefault(action, []).extend(_resources(st))
    return out


def test_worker_env_contract(worker_ud):
    for line in (
        "Environment=DRIFTSENSE_ENGINE=v2",
        "Environment=APPROVAL_TIMEOUT_S=900",
        "Environment=APPROVAL_POLL_S=2",
        "Environment=TRACE_UPLOAD_S=2",
        "Environment=METRICS_SINK=both",
    ):
        assert line in worker_ud, line
    for key in ("JOBS_TABLE", "JOBS_QUEUE_URL", "DATA_BUCKET", "AWS_REGION", "ENGINE_GIT_SHA"):
        assert f"Environment={key}=" in worker_ud, key


def test_worker_runs_from_app_root(worker_ud, default_app):
    # driftsense.sim imports the top-level config / data_gen modules
    assert "WorkingDirectory=/opt/driftsense/app" in worker_ud
    assert "Environment=PYTHONPATH=/opt/driftsense/app" in worker_ud
    from driftsense_infra.app_bundle import WORKER_PATHS

    for path in ("config.py", "data_gen", "localization", "driftsense", "service", "bench"):
        assert path in WORKER_PATHS, path
    stop = int(re.search(r"TimeoutStopSec=(\d+)", worker_ud).group(1))
    assert stop >= 1020


def test_worker_iam_contract(default_app):
    acts = _by_action(_statements(Template.from_stack(default_app.worker)))
    for a in (
        "dynamodb:GetItem",
        "dynamodb:UpdateItem",
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:ChangeMessageVisibility",
        "cloudwatch:PutMetricData",
        "s3:ListBucket",
    ):
        assert a in acts, a
    get = json.dumps(acts["s3:GetObject"])
    assert "/demos/*" in get and "/uploads/*" in get
    put = json.dumps(acts["s3:PutObject"])
    assert "/jobs/*" in put and "/traces/*" in put and "uploads" not in put
    assert not any(json.dumps(r).endswith('*"') for r in acts["s3:ListBucket"])  # the bucket ARN itself


def test_api_env_and_timeout_contract(templates):
    props = _api_fn(templates)
    env = props["Environment"]["Variables"]
    for key in (
        "JOBS_TABLE",
        "JOBS_QUEUE_URL",
        "DATA_BUCKET",
        "UI_ORIGIN",
        "ENGINE_GIT_SHA",
        "MAX_UPLOAD_MB",
        "JOB_TTL_DAYS",
    ):
        assert key in env, key
    assert props["Timeout"] >= 10


def test_api_iam_contract(templates):
    acts = _by_action(_statements(templates[f"{PREFIX}Api"]))
    for a in (
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:ConditionCheckItem",
        "sqs:SendMessage",
        "s3:ListBucket",
    ):
        assert a in acts, a
    get = json.dumps(acts["s3:GetObject"])
    assert "/jobs/*" in get and "/traces/*" in get
    assert "/uploads/*" in json.dumps(acts["s3:PutObject"])


def test_queue_timing_contract(templates):
    queues = templates[f"{PREFIX}Queue"].find_resources("AWS::SQS::Queue")
    (jobs,) = [q["Properties"] for q in queues.values() if "RedrivePolicy" in q["Properties"]]
    assert jobs["VisibilityTimeout"] >= 180
    assert jobs["RedrivePolicy"]["maxReceiveCount"] == 3
    assert jobs.get("MessageRetentionPeriod", 4 * 24 * 3600) >= 3600
