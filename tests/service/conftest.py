"""Fixtures for service tests. Nothing here is autouse and nothing heavy is imported at module level,
so other tests in this directory (e.g. the docs pipeline) are unaffected and CI without
fastapi/moto/boto3/driftsense simply skips the service modules (each one calls pytest.importorskip)."""

from __future__ import annotations

import io

import pytest

REGION = "ap-south-1"
TABLE = "driftsense-jobs-test"
BUCKET = "driftsense-data-test"


@pytest.fixture
def service_env(monkeypatch):
    env = {
        "AWS_REGION": REGION,
        "AWS_DEFAULT_REGION": REGION,
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SECURITY_TOKEN": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_EC2_METADATA_DISABLED": "true",
        "JOBS_TABLE": TABLE,
        "DATA_BUCKET": BUCKET,
        "JOBS_QUEUE_URL": "",
        "DRIFTSENSE_ENGINE": "fake",
        "ENGINE_GIT_SHA": "testsha",
        "MAX_UPLOAD_MB": "20",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("UI_ORIGIN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    return env


@pytest.fixture
def aws(service_env, monkeypatch):
    """Active moto mock with table, bucket and queue(+DLQ, maxReceiveCount 3) provisioned."""
    from moto import mock_aws

    from service.e2e_local import provision

    with mock_aws():
        url = provision(REGION, TABLE, BUCKET, queue_name="driftsense-jobs-test")
        monkeypatch.setenv("JOBS_QUEUE_URL", url)
        yield url


@pytest.fixture
def settings(aws):
    from service.settings import Settings

    return Settings.from_env()


@pytest.fixture
def services(settings):
    from service.api import Services

    return Services.from_settings(settings)


@pytest.fixture
def make_client(services):
    from fastapi.testclient import TestClient

    from service.api import create_app

    def _make(settings_override=None):
        svc = services
        if settings_override is not None:
            from service.api import Services

            svc = Services(settings_override, services.jobs, services.artifacts, services.queue)
        return TestClient(create_app(services=svc))

    return _make


@pytest.fixture
def client(make_client):
    return make_client()


@pytest.fixture
def worker_info():
    from service.models import WorkerInfo

    return WorkerInfo(
        instance_type=None, arch="x86_64", opencv_version="test", numpy_version="test", cool=False
    )


@pytest.fixture
def make_worker(settings, services, worker_info):
    from service.worker import Worker

    def _make(**kwargs):
        kwargs.setdefault("worker_info", worker_info)
        kwargs.setdefault("retry_backoff_s", 0)
        kwargs.setdefault("emf_stream", io.StringIO())
        return Worker(
            kwargs.pop("settings", settings),
            jobs=services.jobs,
            artifacts=services.artifacts,
            queue=services.queue,
            **kwargs,
        )

    return _make


@pytest.fixture
def queue_depth(services):
    def _depth() -> tuple[int, int]:
        attrs = services.queue.client.get_queue_attributes(
            QueueUrl=services.queue.url,
            AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
        )["Attributes"]
        return int(attrs["ApproximateNumberOfMessages"]), int(attrs["ApproximateNumberOfMessagesNotVisible"])

    return _depth
