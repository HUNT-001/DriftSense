"""Job metrics: CloudWatch Embedded Metric Format (EMF) lines on stdout and/or PutMetricData.

``METRICS_SINK`` = stdout (EMF, default) | cloudwatch (boto3 put_metric_data) | both. Metric names,
units and dimensions are identical in both sinks.

EMF spec: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Embedded_Metric_Format_Specification.html
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import IO, Any

NAMESPACE = "DriftSense"
MAX_DATUMS_PER_CALL = 1000
SINKS = ("stdout", "cloudwatch", "both")

log = logging.getLogger("driftsense.metrics")


def metric_line(
    metrics: Mapping[str, tuple[float, str]],
    dimensions: Mapping[str, str],
    properties: Mapping[str, object] | None = None,
    *,
    namespace: str = NAMESPACE,
    timestamp_ms: int | None = None,
) -> str:
    """One EMF document. ``metrics`` maps name -> (value, unit); all dimension keys form one dimension set."""
    doc: dict = {
        "_aws": {
            "Timestamp": int(timestamp_ms if timestamp_ms is not None else time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [list(dimensions.keys())],
                    "Metrics": [{"Name": name, "Unit": unit} for name, (_, unit) in metrics.items()],
                }
            ],
        },
    }
    for key, value in (properties or {}).items():
        doc[key] = value
    for key, value in dimensions.items():
        doc[key] = value
    for name, (value, _) in metrics.items():
        doc[name] = value
    return json.dumps(doc, separators=(",", ":"), default=str)


MetricSet = tuple[dict[str, tuple[float, str]], dict[str, str], dict[str, object]]


def job_metric_sets(
    *, job_id: str, engine: str, outcome: str, latency_ms: float, timings: Iterable[tuple[str, float]] = ()
) -> list[MetricSet]:
    """(metrics, dimensions, properties) for one finished job attempt; shared by both sinks."""
    succeeded = outcome in ("succeeded", "awaiting_approval")
    sets: list[MetricSet] = [
        (
            {
                "JobLatencyMs": (round(latency_ms, 3), "Milliseconds"),
                "JobsSucceeded": (1 if succeeded else 0, "Count"),
                "JobsFailed": (0 if succeeded else 1, "Count"),
            },
            {"Engine": engine},
            {"job_id": job_id, "outcome": outcome},
        )
    ]
    for stage, ms in timings:
        sets.append(
            (
                {"StageMs": (round(ms, 3), "Milliseconds")},
                {"Stage": stage},
                {"job_id": job_id, "engine": engine},
            )
        )
    return sets


def job_lines(
    *, job_id: str, engine: str, outcome: str, latency_ms: float, timings: Iterable[tuple[str, float]] = ()
) -> list[str]:
    """EMF lines for one finished job attempt: job-level counters + one StageMs line per stage."""
    return [
        metric_line(m, d, p)
        for m, d, p in job_metric_sets(
            job_id=job_id, engine=engine, outcome=outcome, latency_ms=latency_ms, timings=timings
        )
    ]


def job_datums(
    *,
    job_id: str,
    engine: str,
    outcome: str,
    latency_ms: float,
    timings: Iterable[tuple[str, float]] = (),
    timestamp: datetime | None = None,
) -> list[dict]:
    """The same metrics as ``job_lines``, as PutMetricData MetricDatum dicts."""
    ts = timestamp or datetime.now(UTC)
    datums = []
    for metrics, dims, _ in job_metric_sets(
        job_id=job_id, engine=engine, outcome=outcome, latency_ms=latency_ms, timings=timings
    ):
        dimensions = [{"Name": k, "Value": v} for k, v in dims.items()]
        for name, (value, unit) in metrics.items():
            datums.append(
                {
                    "MetricName": name,
                    "Dimensions": dimensions,
                    "Timestamp": ts,
                    "Value": float(value),
                    "Unit": unit,
                }
            )
    return datums


def emit(lines: Iterable[str], stream: IO[str] | None = None) -> None:
    out = stream or sys.stdout
    for line in lines:
        out.write(line + "\n")
    out.flush()


class MetricsSink:
    """Publishes per-job metrics to stdout (EMF), CloudWatch (PutMetricData) or both.

    Never raises: metric failures are logged and the job path continues.
    """

    def __init__(
        self,
        mode: str = "stdout",
        *,
        stream: IO[str] | None = None,
        cloudwatch: Any = None,
        region: str | None = None,
        namespace: str = NAMESPACE,
    ):
        if mode not in SINKS:
            raise ValueError(f"METRICS_SINK must be one of {SINKS}, got {mode!r}")
        self.mode, self.stream, self.namespace = mode, stream, namespace
        self._cw, self._region = cloudwatch, region

    @property
    def cloudwatch(self) -> Any:
        if self._cw is None:
            import boto3

            self._cw = boto3.client("cloudwatch", region_name=self._region)
        return self._cw

    def job(
        self,
        *,
        job_id: str,
        engine: str,
        outcome: str,
        latency_ms: float,
        timings: Iterable[tuple[str, float]] = (),
    ) -> None:
        timings = list(timings)
        kw = {
            "job_id": job_id,
            "engine": engine,
            "outcome": outcome,
            "latency_ms": latency_ms,
            "timings": timings,
        }
        if self.mode in ("stdout", "both"):
            try:
                emit(job_lines(**kw), self.stream)
            except Exception:
                log.warning("EMF emit failed for job %s", job_id, exc_info=True)
        if self.mode in ("cloudwatch", "both"):
            self.put(job_datums(**kw), job_id)

    def put(self, datums: list[dict], job_id: str = "") -> int:
        """PutMetricData in batches of <= 1000 datums. Returns the number of successful calls."""
        ok = 0
        for i in range(0, len(datums), MAX_DATUMS_PER_CALL):
            try:
                self.cloudwatch.put_metric_data(
                    Namespace=self.namespace, MetricData=datums[i : i + MAX_DATUMS_PER_CALL]
                )
                ok += 1
            except Exception:
                log.warning("PutMetricData failed for job %s", job_id, exc_info=True)
        return ok
