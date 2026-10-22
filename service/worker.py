"""SQS worker: ``python -m service.worker`` (systemd on the EC2 COOL instance).

Per message: load job -> skip if terminal -> running -> resolve inputs -> driftsense.api.localize ->
jobs/<id>/result.json -> succeeded | awaiting_approval | failed. Idempotent, heartbeat-extended,
SIGTERM-graceful, EMF metrics on stdout.

Agent episodes (``demo_id="vsem-<family>-<seed>"``, ``agent=true``) instead run
``driftsense.agent.runner.run_episode`` on a ``VirtualSEM`` with the trace mirrored to
``traces/<job_id>.jsonl`` and human approval through DynamoDB (see ``service/agent.py``)."""

from __future__ import annotations

import argparse
import hashlib
import logging
import mimetypes
import os
import platform
import re
import signal
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Self

from service import agent, emf
from service.demos import demo_keys, get_demo, parse_vsem, synthetic_pair
from service.models import (
    TERMINAL_STATUSES,
    ApprovalRequest,
    Artifact,
    DriftSenseInputError,
    ErrorInfo,
    JobCreate,
    LocalizeResult,
    WorkerInfo,
)
from service.queue import JobQueue, Message
from service.settings import (
    AGENT_BASE_DWELL_US,
    AGENT_BASE_FOV_NM,
    AGENT_PIXEL_NM,
    AGENT_REF_DWELL_US,
    AGENT_REF_FOV_NM,
    AGENT_SITE_NM,
    AGENT_TEMPLATE_FRAC,
    ENGINES,
    Settings,
)
from service.storage import ArtifactStore, InvalidTransition, JobStore

log = logging.getLogger("driftsense.worker")

IMDS_ENDPOINT = "http://169.254.169.254"
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------- environment introspection ----------
def detect_arch(machine: str | None = None) -> str:
    m = (machine if machine is not None else platform.machine()).lower()
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if m == "aarch64":
        return "aarch64"
    if m in ("arm64", "armv8", "armv8l"):
        return "arm64"
    return "unknown"


def _urlopen(req: urllib.request.Request, timeout: float):
    return _NO_PROXY.open(req, timeout=timeout)


def imds_instance_type(timeout: float = 1.0, endpoint: str = IMDS_ENDPOINT) -> str | None:
    """EC2 instance type via IMDSv2 (token PUT, then GET). None off-EC2 or on any error."""
    if os.environ.get("AWS_EC2_METADATA_DISABLED", "").strip().lower() == "true":
        return None
    try:
        token_req = urllib.request.Request(
            f"{endpoint}/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with _urlopen(token_req, timeout) as resp:
            token = resp.read().decode("ascii").strip()
        meta_req = urllib.request.Request(
            f"{endpoint}/latest/meta-data/instance-type", headers={"X-aws-ec2-metadata-token": token}
        )
        with _urlopen(meta_req, timeout) as resp:
            value = resp.read().decode("ascii").strip()
        return value or None
    except Exception:  # noqa: BLE001 - any failure means "not on EC2"
        return None


def opencv_report() -> dict[str, Any]:
    """cv2 version/path and KleidiCV presence (COOL evidence)."""
    import cv2

    build = cv2.getBuildInformation()
    return {
        "cv2_version": cv2.__version__,
        "cv2_file": cv2.__file__,
        "kleidicv": "kleidicv" in build.lower(),
        "cool": "/opt/cool" in (cv2.__file__ or ""),
    }


def collect_worker_info(instance_type: str | None = None, *, probe_imds: bool = True) -> WorkerInfo:
    import cv2
    import numpy

    if instance_type is None and probe_imds:
        instance_type = imds_instance_type()
    return WorkerInfo(
        instance_type=instance_type,
        arch=detect_arch(),
        opencv_version=cv2.__version__,
        numpy_version=numpy.__version__,
        cool="/opt/cool" in (cv2.__file__ or ""),
    )


def artifact_object_name(filename: str) -> str:
    """Map a local artifact file name onto the API's allowed name set ^[A-Za-z0-9._-]{1,128}$ (no '..')."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", filename).replace("..", "_")
    if len(name) > 128:
        stem, dot, ext = name.rpartition(".")
        name = (stem[:120] + dot + ext[:7]) if dot and len(ext) <= 7 else name[:128]
    return name or "artifact"


def render_explain_artifact(
    name: str, result: LocalizeResult, template: bytes | None, search: bytes | None
) -> bytes | None:
    """Re-render a v2 explain artifact from public inputs, or None if that is not possible."""
    if name not in ("overlay", "top5_gallery") or not template or not search or not result.candidates:
        return None
    try:
        import cv2
        import numpy as np

        from driftsense.v2 import explain

        t = cv2.imdecode(np.frombuffer(template, np.uint8), cv2.IMREAD_GRAYSCALE)
        s = cv2.imdecode(np.frombuffer(search, np.uint8), cv2.IMREAD_GRAYSCALE)
        if t is None or s is None:
            return None
        th, tw = t.shape[:2]
        if name == "overlay":
            return explain.overlay(s, result.candidates[0], tw, th).png
        return explain.gallery(s, list(result.candidates), tw, th).png
    except Exception:  # noqa: BLE001 - an explain artifact must never fail the job
        log.warning("could not render artifact %s", name, exc_info=True)
        return None


# ---------- heartbeat ----------
class Heartbeat:
    """Extends the SQS visibility timeout every ``interval_s`` while a message is being processed."""

    def __init__(self, queue: JobQueue, receipt_handle: str, interval_s: float, timeout_s: int):
        self.queue, self.handle = queue, receipt_handle
        self.interval_s, self.timeout_s = interval_s, timeout_s
        self.beats = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sqs-heartbeat", daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.queue.change_visibility(self.handle, self.timeout_s)
                self.beats += 1
            except Exception:
                log.warning("heartbeat: ChangeMessageVisibility failed", exc_info=True)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# ---------- worker ----------
class Worker:
    def __init__(
        self,
        settings: Settings,
        *,
        jobs: JobStore,
        artifacts: ArtifactStore,
        queue: JobQueue,
        localize_fn: Callable[..., LocalizeResult] | None = None,
        worker_info: WorkerInfo | None = None,
        max_receives: int = 3,
        heartbeat_s: float = 60.0,
        visibility_timeout: int = 180,
        retry_backoff_s: int = 30,
        emf_stream: IO[str] | None = None,
        metrics: emf.MetricsSink | None = None,
    ):
        if localize_fn is None:
            from driftsense.api import localize as localize_fn
        self.settings = settings
        self.engine = settings.engine
        self.jobs, self.artifacts, self.queue = jobs, artifacts, queue
        self.localize_fn = localize_fn
        self.info = worker_info or collect_worker_info()
        self.max_receives = max_receives
        self.heartbeat_s = heartbeat_s
        self.visibility_timeout = visibility_timeout
        self.retry_backoff_s = retry_backoff_s
        self.metrics = metrics or emf.MetricsSink(
            settings.metrics_sink, stream=emf_stream, region=settings.aws_region
        )
        self.stop_event = threading.Event()

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: Any) -> Worker:
        settings.require_aws()
        region = settings.aws_region
        return cls(
            settings,
            jobs=JobStore(settings.jobs_table, region=region, ttl_days=settings.job_ttl_days),
            artifacts=ArtifactStore(settings.data_bucket, region=region),
            queue=JobQueue(settings.jobs_queue_url, region=region),
            **kwargs,
        )

    # ----- lifecycle -----
    def request_stop(self, signum: int | None = None, _frame: Any = None) -> None:
        if not self.stop_event.is_set():
            log.info("stop requested (signal %s): finishing current message, then exiting", signum)
        self.stop_event.set()

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.request_stop)

    def log_startup(self) -> dict[str, Any]:
        report = opencv_report()
        report.update(
            engine=self.engine or "default",
            instance_type=self.info.instance_type,
            arch=self.info.arch,
            numpy=self.info.numpy_version,
            git_sha=self.settings.engine_git_sha,
            metrics_sink=self.metrics.mode,
        )
        log.info(
            "worker startup cv2.__version__=%s cv2.__file__=%s KleidiCV=%s cool=%s engine=%s instance=%s "
            "metrics_sink=%s",
            report["cv2_version"],
            report["cv2_file"],
            report["kleidicv"],
            report["cool"],
            report["engine"],
            report["instance_type"],
            report["metrics_sink"],
        )
        return report

    def run_forever(self, wait_seconds: int = 20) -> None:
        while not self.stop_event.is_set():
            try:
                self.poll_once(wait_seconds=wait_seconds)
            except Exception:
                log.exception("poll failed")
                self.stop_event.wait(5)
        log.info("worker stopped")

    def poll_once(self, wait_seconds: int = 20) -> list[str]:
        msgs = self.queue.receive(
            wait_seconds=wait_seconds, max_messages=1, visibility_timeout=self.visibility_timeout
        )
        return [self.handle(m) for m in msgs]

    # ----- one message -----
    def handle(self, msg: Message) -> str:
        if msg.job_id is None:
            log.warning("dropping malformed message %s: %r", msg.message_id, msg.raw_body[:200])
            self.queue.delete(msg.receipt_handle)
            return "dropped"
        job_id = msg.job_id
        item = self.jobs.get_item(job_id)
        if item is None or "status" not in item:
            log.warning("dropping message for unknown job %s", job_id)
            self.queue.delete(msg.receipt_handle)
            return "dropped"
        status = item["status"]
        req = JobCreate.model_validate_json(item["request"])
        episode = parse_vsem(req.demo_id) if req.agent else None
        if episode is not None:
            if status in TERMINAL_STATUSES:
                log.info("episode job %s is %s: skipping duplicate delivery", job_id, status)
                self.queue.delete(msg.receipt_handle)
                return "skipped"
            return self._handle_episode(msg, job_id, *episode)
        if status in TERMINAL_STATUSES or status == "awaiting_approval":
            # Duplicate delivery or an approval resume for a non-episode job: nothing to run.
            log.info("job %s is %s (resume=%s): skipping", job_id, status, msg.resume)
            self.queue.delete(msg.receipt_handle)
            return "skipped"

        engine_label = self.engine if self.engine in ENGINES else None
        try:
            self.jobs.transition(
                job_id,
                "running",
                allowed_from=("queued", "running"),
                set_fields={"worker": self.info, "engine": engine_label},
                remove_fields=("error",),
            )
        except InvalidTransition as exc:
            log.info("job %s moved to %s concurrently: skipping", job_id, exc.current)
            self.queue.delete(msg.receipt_handle)
            return "skipped"

        started = time.perf_counter()
        with Heartbeat(self.queue, msg.receipt_handle, self.heartbeat_s, self.visibility_timeout):
            try:
                template, search = self._load_inputs(req)
                result = self.localize_fn(template, search, req.mode, req.budget, engine=self.engine)
                result = self._store_artifacts(job_id, result, template, search)
                result_key = self.artifacts.put_json(f"jobs/{job_id}/result.json", result)
            except DriftSenseInputError as exc:
                self._fail(job_id, "bad_input", str(exc))
                self.queue.delete(msg.receipt_handle)
                self._metrics(job_id, engine_label or "unknown", "failed", started, None)
                return "failed_bad_input"
            except Exception as exc:  # noqa: BLE001
                return self._transient_failure(msg, job_id, exc, engine_label, started)

        fields = {"result": result, "engine": result.engine, "result_key": result_key}
        if result.margin.decision == "escalate" and req.agent:
            fields["approval"] = ApprovalRequest(options=result.candidates[:3], margin=result.margin)
            outcome = "awaiting_approval"
        else:
            outcome = "succeeded"
        self.jobs.transition(job_id, outcome, allowed_from=("running",), set_fields=fields)
        self.queue.delete(msg.receipt_handle)
        self._metrics(job_id, result.engine, outcome, started, result)
        return outcome

    # ----- agent episode -----
    def _handle_episode(self, msg: Message, job_id: str, family: str, seed: int) -> str:
        """Run driftsense.agent.runner.run_episode on VirtualSEM(seed, family); see service/agent.py."""
        st = self.settings
        key = agent.trace_key(job_id)
        try:
            self.jobs.transition(
                job_id,
                "running",
                allowed_from=("queued", "running", "awaiting_approval"),
                set_fields={"worker": self.info, "engine": "v2", "trace_key": key},
                remove_fields=("error",),
            )
        except InvalidTransition as exc:
            log.info("episode job %s moved to %s concurrently: skipping", job_id, exc.current)
            self.queue.delete(msg.receipt_handle)
            return "skipped"

        started = time.perf_counter()
        with (
            Heartbeat(self.queue, msg.receipt_handle, self.heartbeat_s, self.visibility_timeout),
            tempfile.TemporaryDirectory(prefix="ds-episode-") as tmp,
        ):
            trace_path = Path(tmp) / f"{job_id}.jsonl"
            try:
                from driftsense.agent.runner import run_episode
                from driftsense.sim import VirtualSEM

                sem = VirtualSEM(
                    seed=seed, family=family, site_size_nm=AGENT_SITE_NM, pixel_nm=AGENT_PIXEL_NM
                )
                with agent.TraceUploader(self.artifacts, key, trace_path, st.trace_upload_s) as uploader:
                    approver = agent.DynamoApprover(
                        self.jobs,
                        job_id,
                        timeout_s=st.approval_timeout_s,
                        poll_s=st.approval_poll_s,
                        on_wait=uploader.flush,
                    )
                    ep = run_episode(
                        sem,
                        tau=st.agent_tau,
                        run_id=job_id,
                        max_steps=st.agent_max_steps,
                        max_dose=st.agent_max_dose,
                        ref_fov_nm=AGENT_REF_FOV_NM,
                        ref_dwell_us=AGENT_REF_DWELL_US,
                        base_fov_nm=AGENT_BASE_FOV_NM,
                        base_dwell_us=AGENT_BASE_DWELL_US,
                        template_frac=AGENT_TEMPLATE_FRAC,
                        approval_cb=approver,
                        trace_path=str(trace_path),
                        **agent.episode_timeouts(run_episode, st),
                    )
                uploader.flush()  # the complete trace, episode_end included
                fields = self._finish_episode(job_id, family, seed, ep, approver)
            except Exception as exc:  # noqa: BLE001
                return self._transient_failure(msg, job_id, exc, "v2", started)

        status = fields.pop("_status")
        self.jobs.transition(job_id, status, allowed_from=("running", "awaiting_approval"), set_fields=fields)
        self.queue.delete(msg.receipt_handle)
        self._metrics(job_id, "v2", status, started, fields.get("result"))
        log.info(
            "episode %s (%s-%d): outcome=%s acquisitions=%d dose=%.3g -> %s",
            job_id,
            family,
            seed,
            ep.outcome,
            ep.steps,
            ep.dose,
            status,
        )
        return status

    def _finish_episode(
        self, job_id: str, family: str, seed: int, ep: Any, approver: agent.DynamoApprover
    ) -> dict[str, Any]:
        """Store result.json / episode.json / frames and build the final transition's fields."""
        result = None
        if ep.last_result is not None and ep.template_png is not None and ep.search_png is not None:
            result = self._store_artifacts(job_id, ep.last_result, ep.template_png, ep.search_png)
            self.artifacts.put_bytes(f"jobs/{job_id}/template.png", ep.template_png, "image/png")
            self.artifacts.put_bytes(f"jobs/{job_id}/search.png", ep.search_png, "image/png")
        final = agent.final_state(ep, result, timed_out=approver.timed_out)
        summary = agent.episode_summary(
            ep, family=family, seed=seed, settings=self.settings, post_commit=agent.post_commit_verify(ep)
        )
        self.artifacts.put_bytes(f"jobs/{job_id}/episode.json", agent.dumps(summary), "application/json")
        fields: dict[str, Any] = {"_status": final.status, "engine": "v2"}
        if final.result is not None:
            res = final.result.model_copy(
                update={
                    "artifacts": [
                        *final.result.artifacts,
                        Artifact(kind="json", uri=f"jobs/{job_id}/episode.json"),
                        Artifact(kind="json", uri=f"jobs/{job_id}/result.json"),
                    ]
                }
            )
            fields["result"] = res
            fields["result_key"] = self.artifacts.put_json(f"jobs/{job_id}/result.json", res)
        if final.error is not None:
            fields["error"] = final.error
        return fields

    def _transient_failure(
        self, msg: Message, job_id: str, exc: Exception, engine_label: str | None, started: float
    ) -> str:
        if msg.receive_count >= self.max_receives:
            log.error("job %s failed on final receive %d: %r", job_id, msg.receive_count, exc)
            self._fail(job_id, "engine_error", f"{type(exc).__name__}: {exc}")
            # Not deleted on purpose: release it now so the queue's redrive policy (maxReceiveCount=3)
            # moves it to the DLQ on the next receive, for inspection.
            try:
                self.queue.change_visibility(msg.receipt_handle, 0)
            except Exception:
                log.warning("could not release message for DLQ redrive", exc_info=True)
            self._metrics(job_id, engine_label or "unknown", "failed", started, None)
            return "failed_engine_error"
        log.warning(
            "job %s attempt %d failed, will retry: %r",
            job_id,
            msg.receive_count,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        try:
            self.jobs.transition(job_id, "queued", allowed_from=("running", "awaiting_approval"))
        except InvalidTransition:
            pass
        try:
            self.queue.change_visibility(msg.receipt_handle, self.retry_backoff_s * msg.receive_count)
        except Exception:
            log.warning("could not shorten visibility for retry", exc_info=True)
        return "retry"

    def _fail(self, job_id: str, code: str, message: str) -> None:
        self.jobs.transition(
            job_id,
            "failed",
            allowed_from=("running", "queued", "awaiting_approval"),
            set_fields={"error": ErrorInfo(code=code, message=message[:1000])},
        )

    def _load_inputs(self, req: JobCreate) -> tuple[bytes, bytes]:
        if req.demo_id:
            if get_demo(req.demo_id) is None:
                raise DriftSenseInputError(f"unknown demo_id {req.demo_id!r}")
            t_key, s_key = demo_keys(req.demo_id)
            t, s = self.artifacts.get_bytes(t_key), self.artifacts.get_bytes(s_key)
            if t is None or s is None:
                log.info("demo %s not in S3: using deterministic synthetic pair", req.demo_id)
                return synthetic_pair(req.demo_id)
            return t, s
        t, s = self.artifacts.get_bytes(req.template_key), self.artifacts.get_bytes(req.search_key)
        missing = [k for k, v in ((req.template_key, t), (req.search_key, s)) if v is None]
        if missing:
            raise DriftSenseInputError(f"upload not found: {', '.join(missing)}")
        return t, s

    def _store_artifacts(
        self, job_id: str, result: LocalizeResult, template: bytes | None = None, search: bytes | None = None
    ) -> LocalizeResult:
        """Make every engine artifact downloadable at jobs/<id>/<name> and set its uri to that key.

        * a uri that is a local file is uploaded as-is;
        * v2 returns artifacts as bare names (e.g. ``overlay.png``) with a sha256 but no bytes. The
          service re-renders the ones it can from public inputs with ``driftsense.v2.explain``
          (``overlay``, ``top5_gallery``) on the original search frame;
        * anything else (v2's heatmaps need engine-internal score maps) is dropped from the list with a
          warning, so the UI never gets a uri that 404s.
        """
        if not result.artifacts:
            return result
        out: list[Artifact] = []
        dropped: list[str] = []
        for art in result.artifacts:
            if art.uri.startswith("s3://") or art.uri.startswith(f"jobs/{job_id}/"):
                out.append(art)
                continue
            path = Path(art.uri)
            if path.is_file():
                data, name = path.read_bytes(), artifact_object_name(path.name)
                sha = art.sha256
            else:
                data = render_explain_artifact(path.stem, result, template, search)
                if data is None:
                    dropped.append(art.uri)
                    continue
                name = artifact_object_name(path.stem + ".png")
                sha = hashlib.sha256(data).hexdigest()
            key = f"jobs/{job_id}/{name}"
            ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
            self.artifacts.put_bytes(key, data, ctype)
            # uri = the S3 key; its basename is what GET /v1/jobs/{id}/artifacts/{name} serves
            out.append(Artifact(kind=art.kind, uri=key, sha256=sha))
        warnings = list(result.warnings)
        if dropped:
            warnings.append(
                f"service: artifacts not materialized (engine returned no bytes): {', '.join(dropped)}"
            )
        return result.model_copy(update={"artifacts": out, "warnings": warnings})

    def _metrics(
        self, job_id: str, engine: str, outcome: str, started: float, result: LocalizeResult | None
    ) -> None:
        timings = [(t.stage, t.ms) for t in result.timings] if result else []
        self.metrics.job(
            job_id=job_id,
            engine=engine,
            outcome=outcome,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            timings=timings,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m service.worker", description=__doc__)
    parser.add_argument("--once", action="store_true", help="process at most one receive, then exit")
    parser.add_argument("--wait", type=int, default=20, help="SQS long-poll seconds (default 20)")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    worker = Worker.from_settings(Settings.from_env())
    worker.log_startup()
    worker.install_signal_handlers()
    log.info("polling %s at %s", worker.queue.url, datetime.now(UTC).isoformat())
    if args.once:
        worker.poll_once(wait_seconds=args.wait)
    else:
        worker.run_forever(wait_seconds=args.wait)
    return 0


if __name__ == "__main__":
    sys.exit(main())
