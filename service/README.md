# DriftSense job service

FastAPI job API on Lambda (arm64, via Mangum) plus an SQS worker on the EC2 Graviton COOL instance.
It is built against the frozen contract (`driftsense.contracts`, `CONTRACT_VERSION`). It imports only
`driftsense.api.localize` / `available_engines` and the contract models. `service/models.py`
re-exports them and adds API-only models (`UploadRequest`, `UploadResponse`, `DemoInfo`, `VersionInfo`).

| Entry point | What |
|---|---|
| `service.api.handler` | Lambda handler (Mangum around `service.api.app`, `lifespan="off"`) |
| `python -m service.worker [--once] [--wait 20]` | SQS worker (systemd on EC2) |
| `python -m service.e2e_local [--demo ID] [--engine fake\|v1\|v2] [--agent]` | Full flow in-process under moto. Exit 0 on `succeeded`/`awaiting_approval` |

## Endpoints

All errors are `ErrorInfo`-shaped: `{"code": "...", "message": "..."}`. Every job response passes `JobStatus.model_validate`.

| Method + path | Body | Response |
|---|---|---|
| `GET /healthz` | | `{"ok": true}` |
| `GET /v1/version` | | `VersionInfo`: `git_sha` (`ENGINE_GIT_SHA`), `contract_version`, `service_version`, `engine`, `opencv_version` + `opencv_file` (null if cv2 is not importable, e.g. on Lambda), `numpy_version`, `python_version`, `arch`, `platform` |
| `GET /v1/demos` | | `[DemoInfo]`: `demo_id`, `title`, `lattice`, `difficulty`, `description` |
| `POST /v1/uploads` | `{"filename", "content_type", "size_bytes", "role"?, "upload_id"?}` | `UploadResponse`: `upload_id`, `key` = `uploads/<upload_id>/<template\|search>.<png\|tif\|tiff>`, presigned `url`, `method: "PUT"`, `headers` (the PUT must send exactly these: `Content-Type` and `Content-Length` are signed), `expires_in: 900`. Errors: 413 `too_large` (> `MAX_UPLOAD_MB`), 415 `unsupported_media_type` (only `image/png`, `image/tiff`), 422 when the role can't be inferred. `role` defaults to the filename stem (`template.png` → template). Pass the first upload's `upload_id` to put both files under one prefix |
| `POST /v1/jobs` | `JobCreate` | 202 + `JobStatus(status="queued")`. The same `client_token` with the same body returns 200 + the same job. The same token with a different body returns 409 `idempotency_conflict`. 422 `validation_error` covers no source, an unknown `demo_id`, keys that are not `uploads/<32hex>/<role>.<ext>`, or a token longer than 128 chars. 503 `enqueue_failed` means the job is marked failed |
| `GET /v1/jobs/{id}` | | `JobStatus`, or 404 `not_found` |
| `POST /v1/jobs/{id}/approval` | `ApprovalDecision` | Allowed only when the job is `awaiting_approval` (otherwise 409 `invalid_state`). Setting `chosen_rank` gives `succeeded`: `result.offset` becomes the chosen candidate and a JSON audit warning `{"operator_choice": {...}}` is appended. `null` gives `failed` with `rejected_by_operator`. The decision is stored as `decision` on the item, and `{"job_id", "resume": true}` is enqueued |

| `GET /v1/jobs/{id}/artifacts/{name}` | | Returns the bytes of `s3://$DATA_BUCKET/jobs/<id>/<name>` with a content type taken from the extension (`png` → `image/png`, `json` → `application/json`, `jsonl` → `application/x-ndjson`, tif, jpg, svg, csv, txt; otherwise `application/octet-stream`) and `Cache-Control: private, max-age=300`. A `name` that doesn't match `^[A-Za-z0-9._-]{1,128}$` or contains `..` returns 400 `bad_request`. An unknown job or object returns 404 `not_found`. Objects over 5 MB return a 307 redirect to a 5-minute presigned S3 GET (Lambda's response limit is 6 MB) |
| `GET /v1/jobs/{id}/trace` | | When the job has a `trace_key` (`traces/<run_id>.jsonl`), returns that object as `application/x-ndjson` with `Cache-Control: private, no-cache`, because the trace grows while the agent runs. With no trace yet (key or object missing) it returns 404 `not_ready`. An unknown job returns 404 `not_found`. The same 5 MB redirect rule applies |

**Artifact URI convention.** Every artifact the worker stores lives at `jobs/<id>/<name>`. Its `Artifact.uri` is
exactly that S3 key (for example `jobs/<id>/heatmap.png`). The UI resolves a URI that isn't http(s) by its basename:
`GET /v1/jobs/{id}/artifacts/<basename>`. The worker sanitizes local file names into the allowed name set, so
`heat map.png` is stored as `heat_map.png`. `result.json` is always available through the same route.

CORS: only `UI_ORIGIN` is allowed (methods GET/POST, header `content-type`). If it is unset, no CORS headers are sent.

## Job state machine

```text
queued -> running -> succeeded
                  -> awaiting_approval -> succeeded (operator picked rank k) | failed (rejected_by_operator)
                  -> failed (bad_input | engine_error)
running -> queued (transient error, receive < 3; SQS redelivers)
```

The worker ends in `awaiting_approval` only when `result.margin.decision == "escalate"` **and**
`JobCreate.agent` is true. Non-agent escalations end in `succeeded`, and the decision is visible in `result.margin`.

## Worker semantics

- **Polling:** long-polls 20 s with batch size 1, `VisibilityTimeout=180`, and reads `ApproximateReceiveCount`.
- **Heartbeat:** a heartbeat thread calls `ChangeMessageVisibility(180)` every 60 s while a job runs.
- **Duplicates:** if the job is already `succeeded`/`failed`/`awaiting_approval`, the message is deleted and skipped. This covers duplicate deliveries and resume messages for non-agent jobs.
- **Inputs:** a demo job reads `s3://$DATA_BUCKET/demos/<id>/{template,search}.png`. If they are missing, it uses a deterministic synthetic pair (`service/demos.py`, numpy + cv2). An upload job reads its `uploads/...` keys. A missing key gives `bad_input`.
- **Outputs:** `jobs/<id>/result.json` (the `LocalizeResult`). Engine artifacts that are local files are uploaded to `jobs/<id>/<name>`, and their `uri` is set to that key (see the artifact URI convention above).
- **Bad input:** `DriftSenseInputError` gives `failed` with `bad_input`, and the message is deleted (no retry).
- **Other exceptions:** the job goes back to `queued` and SQS retries with a backoff of 30 s × receive count. On the 3rd receive, the job becomes `failed` with `engine_error`. The message is *not* deleted; it is released (visibility 0) so the redrive policy (maxReceiveCount 3) moves it to the DLQ.
- **SIGTERM/SIGINT:** the worker finishes the current message, then stops polling.
- **`WorkerInfo`:** taken from cv2, numpy and `platform`. `instance_type` comes from IMDSv2 (token PUT then GET, 1 s timeout, no proxy). It is `None` off EC2 or when `AWS_EC2_METADATA_DISABLED=true`. `cool = "/opt/cool" in cv2.__file__`.
- **Startup log:** records `cv2.__version__`, `cv2.__file__`, whether `KleidiCV` appears in `cv2.getBuildInformation()`, `cool`, the engine and the instance type.
- **Metrics:** published under namespace `DriftSense`:
  - `JobLatencyMs`, `JobsSucceeded` and `JobsFailed`, with dimension `Engine`
  - `StageMs`, with dimension `Stage`, one value per `result.timings` entry

  `METRICS_SINK` chooses where they go:
  - `stdout` (default): EMF JSON lines. These need a CloudWatch agent or Lambda-style EMF ingestion.
  - `cloudwatch`: the same names, units and dimensions through `cloudwatch.put_metric_data`, batched per job (at most 1000 datums per call). Use this on EC2 without a CloudWatch agent. It needs `cloudwatch:PutMetricData`, conditioned on namespace `DriftSense`.
  - `both`: EMF lines and PutMetricData.

  A metrics failure (throttling, AccessDenied and so on) is logged as a warning and never fails or retries the job.

## Env vars

| Var | Used by | Default |
|---|---|---|
| `JOBS_TABLE` | API, worker | required |
| `JOBS_QUEUE_URL` | API, worker | required |
| `DATA_BUCKET` | API, worker | required |
| `UI_ORIGIN` | API (CORS) | unset = no CORS |
| `MAX_UPLOAD_MB` | API | 20 |
| `DRIFTSENSE_ENGINE` | worker (`fake\|v1\|v2`), echoed by `/v1/version` | unset → `driftsense.api.DEFAULT_ENGINE` |
| `ENGINE_GIT_SHA` | `/v1/version`, worker startup log | `unknown` |
| `AWS_REGION` | boto3 clients | boto3 default chain |
| `JOB_TTL_DAYS` | `expires_at` on job + token items | 7 |
| `AWS_EC2_METADATA_DISABLED` | worker (skip IMDS; standard botocore var) | unset |
| `METRICS_SINK` | worker: `stdout` (EMF) \| `cloudwatch` (PutMetricData) \| `both` | `stdout` |
| `LOG_LEVEL` | worker | INFO |

## Storage layout

**DynamoDB** (PK `job_id` (S), TTL `expires_at`):

- Job items have `status`, `created_at`, `updated_at`, `contract_version`, `request` (the JobCreate JSON) and the optional `engine`, `result`, `approval`, `error`, `worker`, `trace_key`, `result_key` and `decision`.
- Idempotency token items are `job_id = "token#<client_token>"` → `target_job_id`. They are written in the same `TransactWriteItems` as the job, with `attribute_not_exists`.

**S3:**

- `uploads/<upload_id>/<template|search>.<ext>`
- `jobs/<job_id>/result.json` and artifacts
- `traces/<run_id>.jsonl` (agent, I-G5)
- `demos/<demo_id>/{template,search}.png`

## Tests and local e2e

Until `driftsense/` is on `main`, put a copy of the contract package on `PYTHONPATH`. Never commit it here.

```bash
PYTHONPATH="<dir containing driftsense/>;." python -m pytest -q tests/service      # ':' on Linux/macOS
PYTHONPATH="<dir containing driftsense/>;." python -m service.e2e_local
python -m ruff check service tests/service
```

The tests use moto only (no network, no real AWS) and run in about 20 s. Every test module calls
`pytest.importorskip` for fastapi/httpx/mangum/moto/boto3/driftsense, so CI without them skips the module.
`test_worker_with_engine_env[v1]` runs only when `v1` is in `available_engines()`.

## Dependencies (tested versions)

**Runtime:**

- API (Lambda): `fastapi 0.143.0` (with `starlette 1.7.0`), `mangum 0.22.0`, `pydantic 2.14.0`, `boto3/botocore 1.43.111`. It needs no cv2 or numpy.
- Worker: the same plus `numpy 2.5.3` and the cv2 from the COOL venv (`opencv-python-headless 5.0.0.93` was used in dev; never pip-install opencv on COOL).

**Dev/test:** `moto[all] 5.2.3` (s3, sqs and dynamodb are enough), `httpx 0.28.1`, `requests` (one presigned-PUT round-trip test, skipped if absent), `pytest 9.1.1`, `ruff 0.17.0`.

## Future: agent episodes (I-G5)

`JobCreate.agent`, `JobStatus.trace_key`, the `awaiting_approval` state and the `{"job_id", "resume": true}`
message are already wired. To add agent episodes:

- Add `POST /v1/agent/episodes` that creates an `agent=true` job.
- Run `driftsense.agent.runner` in the worker for those jobs. It streams `traces/<run_id>.jsonl` and sets `trace_key`.
- For agent jobs, make the approval endpoint move the job to `queued` instead of a terminal state, so the worker resumes the episode on the resume message.
