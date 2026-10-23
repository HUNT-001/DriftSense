from __future__ import annotations

import io
import json
import logging
from dataclasses import replace

import pytest

from tests.service._helpers import job_status, require_service_deps

require_service_deps()

JOB = {
    "job_id": "j1",
    "engine": "fake",
    "outcome": "succeeded",
    "latency_ms": 12.5,
    "timings": [("ingest", 1.0), ("score", 2.0)],
}


class Recorder:
    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    def put_metric_data(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("throttled")
        return {}


class Untouchable:
    def __getattr__(self, name):
        raise AssertionError(f"cloudwatch client must not be used in stdout mode ({name})")


def test_datums_match_emf_names_units_dimensions():
    from service import emf

    emf_pairs = set()
    for line in emf.job_lines(**JOB):
        doc = json.loads(line)
        directive = doc["_aws"]["CloudWatchMetrics"][0]
        dims = tuple((k, doc[k]) for k in directive["Dimensions"][0])
        for m in directive["Metrics"]:
            emf_pairs.add((m["Name"], m["Unit"], dims, float(doc[m["Name"]])))
    datums = emf.job_datums(**JOB)
    cw_pairs = {
        (d["MetricName"], d["Unit"], tuple((x["Name"], x["Value"]) for x in d["Dimensions"]), d["Value"])
        for d in datums
    }
    assert cw_pairs == emf_pairs
    assert {d["MetricName"] for d in datums} == {"JobLatencyMs", "JobsSucceeded", "JobsFailed", "StageMs"}
    assert [d["Dimensions"] for d in datums if d["MetricName"] == "StageMs"] == [
        [{"Name": "Stage", "Value": "ingest"}],
        [{"Name": "Stage", "Value": "score"}],
    ]


def test_stdout_mode_is_default_and_never_calls_cloudwatch():
    from service.emf import MetricsSink

    buf = io.StringIO()
    sink = MetricsSink(stream=buf, cloudwatch=Untouchable())
    sink.job(**JOB)
    assert sink.mode == "stdout" and len(buf.getvalue().splitlines()) == 3


def test_cloudwatch_mode_with_botocore_stubber(service_env):
    import boto3
    from botocore.stub import ANY, Stubber

    from service.emf import MetricsSink

    client = boto3.client("cloudwatch", region_name="ap-south-1")
    buf = io.StringIO()
    with Stubber(client) as stub:
        stub.add_response("put_metric_data", {}, {"Namespace": "DriftSense", "MetricData": ANY})
        MetricsSink("cloudwatch", stream=buf, cloudwatch=client).job(**JOB)
        stub.assert_no_pending_responses()
    assert buf.getvalue() == ""  # no EMF in cloudwatch-only mode


def test_both_mode_writes_emf_and_puts(service_env):
    from service.emf import MetricsSink

    buf, rec = io.StringIO(), Recorder()
    MetricsSink("both", stream=buf, cloudwatch=rec).job(**JOB)
    assert len(buf.getvalue().splitlines()) == 3
    (call,) = rec.calls
    assert call["Namespace"] == "DriftSense" and len(call["MetricData"]) == 5  # 3 job + 2 stage datums


def test_batches_at_most_1000_datums_per_call():
    from service.emf import MetricsSink

    rec = Recorder()
    timings = [(f"s{i}", 1.0) for i in range(2497)]  # 3 job datums + 2497 stage datums = 2500
    MetricsSink("cloudwatch", cloudwatch=rec).job(**{**JOB, "timings": timings})
    assert [len(c["MetricData"]) for c in rec.calls] == [1000, 1000, 500]


def test_cloudwatch_errors_are_swallowed(service_env, caplog):
    import boto3
    from botocore.stub import Stubber

    from service.emf import MetricsSink

    caplog.set_level(logging.WARNING, logger="driftsense.metrics")
    client = boto3.client("cloudwatch", region_name="ap-south-1")
    buf = io.StringIO()
    with Stubber(client) as stub:
        stub.add_client_error("put_metric_data", service_error_code="AccessDenied", http_status_code=403)
        MetricsSink("both", stream=buf, cloudwatch=client).job(**JOB)  # must not raise
    assert len(buf.getvalue().splitlines()) == 3  # stdout half still delivered
    assert "PutMetricData failed" in caplog.text
    # a non-botocore failure on a later batch doesn't stop the others
    rec = Recorder(fail=True)
    assert MetricsSink("cloudwatch", cloudwatch=rec).put([{"MetricName": "x"}] * 1500) == 0
    assert len(rec.calls) == 2


def test_worker_job_survives_failing_cloudwatch(client, make_worker, settings):
    from service.emf import MetricsSink

    job_id = job_status(client.post("/v1/jobs", json={"demo_id": "demo_grid"}), 202)["job_id"]
    rec = Recorder(fail=True)
    worker = make_worker(
        settings=replace(settings, metrics_sink="cloudwatch"),
        metrics=MetricsSink("cloudwatch", cloudwatch=rec),
    )
    assert worker.poll_once(wait_seconds=0) == ["succeeded"]
    assert rec.calls and job_status(client.get(f"/v1/jobs/{job_id}"), 200)["status"] == "succeeded"


def test_worker_publishes_to_cloudwatch_under_moto(client, make_worker, settings, services):
    import boto3

    job_status(client.post("/v1/jobs", json={"demo_id": "demo_grid"}), 202)
    buf = io.StringIO()
    worker = make_worker(settings=replace(settings, metrics_sink="cloudwatch"), emf_stream=buf)
    assert worker.metrics.mode == "cloudwatch"
    assert worker.poll_once(wait_seconds=0) == ["succeeded"]
    assert buf.getvalue() == ""
    cw = boto3.client("cloudwatch", region_name=settings.aws_region)
    names = {m["MetricName"] for m in cw.list_metrics(Namespace="DriftSense")["Metrics"]}
    assert names == {"JobLatencyMs", "JobsSucceeded", "JobsFailed", "StageMs"}


def test_metrics_sink_setting():
    from service.emf import MetricsSink
    from service.settings import Settings

    assert Settings.from_env({}).metrics_sink == "stdout"
    assert Settings.from_env({"METRICS_SINK": "Both"}).metrics_sink == "both"
    assert Settings.from_env({"METRICS_SINK": "cloudwatch"}).metrics_sink == "cloudwatch"
    with pytest.raises(ValueError):
        Settings.from_env({"METRICS_SINK": "statsd"})
    with pytest.raises(ValueError):
        MetricsSink("statsd")
