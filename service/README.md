# DriftSense job service

FastAPI job API on Lambda (arm64, via Mangum) plus an SQS worker on the EC2 Graviton COOL instance.
It is built against the frozen contract (`driftsense.contracts`, `CONTRACT_VERSION` 1.0.0). It imports
`driftsense.api.localize`, the contract models and, in the worker only for agent episodes,
`driftsense.agent.runner.run_episode` and `driftsense.sim.VirtualSEM`. `service/models.py` re-exports the
contract models and adds API-only models (`UploadRequest`, `UploadResponse`, `DemoInfo`, `EpisodeCreate`,
`VersionInfo`).

| Entry point | What |
|---|---|
| `service.api.handler` | Lambda handler (Mangum around `service.api.app`, `lifespan="off"`) |
| `python -m service.worker [--once] [--wait 20]` | SQS worker (systemd on EC2) |
| `python -m service.e2e_local [--demo ID] [--engine fake\|v1\|v2] [--agent]` | Full flow in-process under moto. Exits 0 on `succeeded`/`awaiting_approval`. `--agent` runs a real agent episode (default `vsem-dram-17`, which escalates) with an auto-approver thread that picks rank 1, and exits 0 only on `succeeded` |

## Endpoints

All errors are `ErrorInfo`-shaped: `{"code": "...", "message": "..."}`. Every job response passes `JobStatus.model_validate`.

| Method + path | Body | Response |
|---|---|---|
| `GET /healthz` | | `{"ok": true}` |
| `GET /v1/version` | | `VersionInfo`: `git_sha` (`ENGINE_GIT_SHA`), `contract_version`, `service_version`, `engine`, `opencv_version` + `opencv_file` (null if cv2 is not importable, e.g. on Lambda), `numpy_version`, `python_version`, `arch`, `platform` |
| `GET /v1/demos` | | `[DemoInfo]`: `demo_id`, `title`, `lattice`, `difficulty`, `description`, `agent` (bool), `family`, `seed`. Agent demos (`agent: true`): `vsem-dram-7` and `vsem-finfet-3` commit on the first look, `vsem-dram-1` re-images and then commits, `vsem-dram-17` escalates to the operator |
| `POST /v1/uploads` | `{"filename", "content_type", "size_bytes", "role"?, "upload_id"?}` | `UploadResponse`: `upload_id`, `key` = `uploads/<upload_id>/<template\|search>.<png\|tif\|tiff>`, presigned `url`, `method: "PUT"`, `headers` (the PUT must send exactly these: `Content-Type` and `Content-Length` are signed), `expires_in: 900`. Errors: 413 `too_large` (> `MAX_UPLOAD_MB`), 415 `unsupported_media_type` (only `image/png`, `image/tiff`), 422 when the role can't be inferred. `role` defaults to the filename stem (`template.png` → template). Pass the first upload's `upload_id` to put both files under one prefix |
| `POST /v1/jobs` | `JobCreate` | 202 + `JobStatus(status="queued")`. The same `client_token` with the same body returns 200 + the same job. The same token with a different body returns 409 `idempotency_conflict`. 422 `validation_error` covers no source, an unknown `demo_id`, a `vsem-*` demo without `agent: true`, keys that are not `uploads/<32hex>/<role>.<ext>`, or a token longer than 128 chars. 503 `enqueue_failed` means the job is marked failed |
| `POST /v1/agent/episodes` | `{"family": "dram"\|"finfet"\|"random", "seed": 0..2147483647, "client_token"?: str}` | 202 + `JobStatus(status="queued")`. Stored as `JobCreate(demo_id="vsem-<family>-<seed>", agent=true, client_token)`, so `POST /v1/jobs` with that body is the same request: the same token returns 200 + the same job, and a different episode under the same token returns 409 `idempotency_conflict`. Unknown fields, a bad family or an out-of-range seed return 422 `validation_error` |
| `GET /v1/jobs/{id}` | | `JobStatus`, or 404 `not_found` |
| `POST /v1/jobs/{id}/approval` | `ApprovalDecision` | Allowed only when the job is `awaiting_approval` (otherwise 409 `invalid_state`). `chosen_rank` must be one of `approval.options[].rank` or `null` (otherwise 422). **Agent episode:** the decision is stored as `decision` on the item and the job goes back to `running` (the response has `status: "running"`). The worker is blocked in the runner's approval callback; it picks the decision up within `APPROVAL_POLL_S` and finishes the episode (see [Agent episodes](#agent-episodes)). **Other `agent: true` jobs:** a `chosen_rank` gives `succeeded` (`result.offset` becomes the chosen candidate and a JSON audit warning `{"operator_choice": {...}}` is appended). `null` gives `failed` with `rejected_by_operator`, and `{"job_id", "resume": true}` is enqueued |
| `GET /v1/jobs/{id}/artifacts/{name}` | | Returns the bytes of `s3://$DATA_BUCKET/jobs/<id>/<name>` with a content type taken from the extension (`png` → `image/png`, `json` → `application/json`, `jsonl` → `application/x-ndjson`, tif, jpg, svg, csv, txt; otherwise `application/octet-stream`) and `Cache-Control: private, max-age=300`. A `name` that doesn't match `^[A-Za-z0-9._-]{1,128}$` or contains `..` returns 400 `bad_request`. An unknown job or object returns 404 `not_found`. Objects over 5 MB return a 307 redirect to a 5-minute presigned S3 GET (Lambda's response limit is 6 MB) |
| `GET /v1/jobs/{id}/trace` | | When the job has a `trace_key` (`traces/<job_id>.jsonl`), returns that object as `application/x-ndjson` with `Cache-Control: private, no-cache`, because the trace grows while the agent runs. Each line is a `TraceEvent`. With no trace yet (key or object missing) it returns 404 `not_ready`. An unknown job returns 404 `not_found`. The same 5 MB redirect rule applies |

**Artifact URI convention.** Every artifact the worker stores lives at `jobs/<id>/<name>`. Its `Artifact.uri` is
exactly that S3 key (for example `jobs/<id>/overlay.png`). The UI resolves a URI that isn't http(s) by its basename:
`GET /v1/jobs/{id}/artifacts/<basename>`. The worker sanitizes local file names into the allowed name set, so
`heat map.png` is stored as `heat_map.png`. `result.json` is always available through the same route.

**v2 artifacts.** v2 returns `Artifact(uri="<name>.png", sha256=...)` with no bytes and no file on disk. The worker
re-renders `overlay` and `top5_gallery` from the inputs with `driftsense.v2.explain`, on the original search frame,
and sets `sha256` to the hash of the uploaded bytes. It drops the heatmaps, which need engine-internal score maps,
and appends the warning `service: artifacts not materialized (engine returned no bytes): ...`. Every artifact left
in the list can be downloaded.

CORS: only `UI_ORIGIN` is allowed (methods GET/POST, header `content-type`). If it is unset, no CORS headers are sent.

## Job state machine

```text
queued -> running -> succeeded
                  -> awaiting_approval -> succeeded (operator picked rank k) | failed (rejected_by_operator)
                  -> failed (bad_input | engine_error)
running -> queued (transient error, receive < 3; SQS redelivers)
```

For a plain (non-episode) job, the worker ends in `awaiting_approval` only when `result.margin.decision == "escalate"`
**and** `JobCreate.agent` is true. Non-agent escalations end in `succeeded`, and the decision is visible in `result.margin`.

Agent episodes:

```text
queued -> running -> succeeded                      (outcome commit)
                  -> awaiting_approval -> running   (operator decision via the API, or timeout)
                                       -> succeeded (operator chose a rank)
                                       -> failed    rejected_by_operator | approval_timeout
                  -> failed agent_abort       (runner outcome abort: tool timeout or exception)
                  -> failed budget_exhausted  (escalated without an approval request, e.g. the instrument refused)
```

## Agent episodes

The worker runs A's real runner, `driftsense.agent.runner.run_episode`, synchronously on
`VirtualSEM(seed, family, site_size_nm=2000, pixel_nm=1)`. It uses the G5 operating point, copied from
`eval/agent_eval.py` into `service/settings.py` as the `AGENT_*` constants: tau 0.10, max_steps 6, reference FOV
240 nm at 12 us, search FOV 500 nm at 1 us, template_frac 0.40 and no dose cap. `run_id` is the job id. The
evaluator-only hooks `on_reference` and `on_decision` read ground truth, so they are never passed. Episodes always
localize with v2, because the agent's tools call `engine="v2"` whatever `DRIFTSENSE_ENGINE` says. A real episode
takes about 1 to 3 s, plus any operator wait.

- **Trace:** `trace_key = traces/<job_id>.jsonl` is set when the job goes `running`. The runner writes the JSONL to
  a temp dir. A background thread uploads it to S3 every `TRACE_UPLOAD_S` when it has changed, and it is also
  uploaded at `awaiting_approval` and at the end. Only complete lines are uploaded, so every line always validates as
  a `TraceEvent`. Poll `GET /v1/jobs/{id}/trace` to follow progress.
- **Approval:** the runner's `approval_cb` sets the job to `awaiting_approval`, with `approval` set to the runner's
  `ApprovalRequest` (the top-3 candidates and the margin). It uploads the trace, then polls the item for `decision`
  every `APPROVAL_POLL_S`, for up to `APPROVAL_TIMEOUT_S`. The SQS heartbeat keeps running during the wait.
  - On timeout it stores and returns `ApprovalDecision(chosen_rank=null, approver="timeout")` and sets the job back to
    `running`.
  - If a decision is already stored, it is reused. That happens on a redelivery after the worker died mid-wait (the
    sim is deterministic, so the operator is asked the same question).
  - The runner applies one per-tool timeout to every tool, including the blocking approval. The service therefore
    passes `tool_timeout_s = max(AGENT_TOOL_TIMEOUT_S, APPROVAL_TIMEOUT_S + 60)`.
- **Result:** `JobStatus.result` is the `LocalizeResult` the terminal decision was taken on, which is the last
  localize before commit or escalate. The post-commit verification frame is reported in `episode.json`. If the
  operator chose rank k, `result.offset` becomes that candidate and the `operator_choice` audit warning is appended.
  `EpisodeResult` doesn't expose any LocalizeResult, so the service captures them by wrapping
  `driftsense.agent.tools._api_localize` (a read-only pass-through) for the duration of the episode.
- **Artifacts** (served at `GET /v1/jobs/{id}/artifacts/<name>`):
  - `result.json` and `episode.json`, both also listed in `result.artifacts` with `kind: "json"`
  - `overlay.png` and `top5_gallery.png`
  - `template.png` (the reference crop) and `search.png` (the decision frame)

  `episode.json` holds `run_id`, `family`, `seed`, `outcome`, `steps` (acquisitions), `dose`, `wall_ms`, `tau`,
  `final_margin`, `estimate_nm`, `decision_acq_index`, `committed_move_nm`, `approval`, `n_events`,
  `post_commit_verify`, `policy` and `operating_point`.

## Worker semantics

- **Polling:** long-polls 20 s with batch size 1, `VisibilityTimeout=180`, and reads `ApproximateReceiveCount`.
- **Heartbeat:** a heartbeat thread calls `ChangeMessageVisibility(180)` every 60 s while a job runs. This includes an agent episode's approval wait.
- **Duplicates:** if a plain job is already `succeeded`/`failed`/`awaiting_approval`, the message is deleted and skipped. This covers duplicate deliveries and resume messages. An agent episode is skipped only when it is terminal. Otherwise it reruns, reusing a stored decision.
- **Inputs:** a demo job reads `s3://$DATA_BUCKET/demos/<id>/{template,search}.png`. If they are missing, it uses a deterministic synthetic pair (`service/demos.py`, numpy + cv2). An upload job reads its `uploads/...` keys. A missing key gives `bad_input`.
- **Outputs:** `jobs/<id>/result.json` (the `LocalizeResult`) and the artifacts (see the artifact notes above).
- **Bad input:** `DriftSenseInputError` gives `failed` with `bad_input`, and the message is deleted (no retry).
- **Other exceptions:** the job goes back to `queued` and SQS retries with a backoff of 30 s × receive count. On the 3rd receive, the job becomes `failed` with `engine_error`. The message is *not* deleted; it is released (visibility 0) so the redrive policy (maxReceiveCount 3) moves it to the DLQ. Exceptions inside an episode are handled by the runner itself, which ends the episode with `agent_abort`.
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
| `DRIFTSENSE_ENGINE` | worker (`fake\|v1\|v2`) for plain localize jobs; echoed by `/v1/version` | unset → **`v2`**, the service default (`driftsense.api.DEFAULT_ENGINE` is still `v1`). Tests set `fake`. Agent episodes always use v2 |
| `ENGINE_GIT_SHA` | `/v1/version`, worker startup log (set it to the deployed commit) | `unknown` |
| `AWS_REGION` | boto3 clients | boto3 default chain |
| `JOB_TTL_DAYS` | `expires_at` on job + token items | 7 |
| `AWS_EC2_METADATA_DISABLED` | worker (skip IMDS; standard botocore var) | unset |
| `METRICS_SINK` | worker: `stdout` (EMF) \| `cloudwatch` (PutMetricData) \| `both` | `stdout` |
| `LOG_LEVEL` | worker | INFO |
| `APPROVAL_TIMEOUT_S` | worker: how long an episode waits for the operator | 900 |
| `APPROVAL_POLL_S` | worker: DynamoDB poll interval while waiting | 2 |
| `TRACE_UPLOAD_S` | worker: trace mirror interval | 2 |
| `AGENT_TAU`, `AGENT_MAX_STEPS`, `AGENT_MAX_DOSE`, `AGENT_TOOL_TIMEOUT_S` | worker: override the G5 operating point (demo/debug only) | 0.10, 6, none, 15 |

## Storage layout

**DynamoDB** (PK `job_id` (S), TTL `expires_at`):

- Job items have `status`, `created_at`, `updated_at`, `contract_version`, `request` (the JobCreate JSON) and the optional `engine`, `result`, `approval`, `error`, `worker`, `trace_key`, `result_key` and `decision`.
- Idempotency token items are `job_id = "token#<client_token>"` → `target_job_id`. They are written in the same `TransactWriteItems` as the job, with `attribute_not_exists`.

**S3:**

- `uploads/<upload_id>/<template|search>.<ext>`
- `jobs/<job_id>/result.json` and artifacts (`episode.json`, `overlay.png`, `top5_gallery.png`, `template.png`, `search.png` for episodes)
- `traces/<job_id>.jsonl` (agent episodes; run_id = job_id)
- `demos/<demo_id>/{template,search}.png`

## Infra checklist

**Worker (EC2, systemd):**

- Env: `JOBS_TABLE`, `JOBS_QUEUE_URL`, `DATA_BUCKET`, `AWS_REGION`, `DRIFTSENSE_ENGINE=v2` (or unset),
  `ENGINE_GIT_SHA=<deployed sha>`, `METRICS_SINK`, and optionally `APPROVAL_TIMEOUT_S` (900) and `LOG_LEVEL`.
- **Run from the repo root (or set `PYTHONPATH=<repo root>`).** `driftsense.sim` imports the top-level `config` and
  `data_gen` modules, and the `driftsense*` package install does not include them.
- IAM:
  - DynamoDB on the jobs table: `GetItem`, `UpdateItem`.
  - S3 on `$DATA_BUCKET`: `GetObject` on `demos/*` and `uploads/*`, `PutObject` on `jobs/*` and `traces/*`, and
    `ListBucket` on the bucket. Without `ListBucket`, a missing demo object returns 403 instead of 404 and the job
    fails instead of falling back to the synthetic pair.
  - SQS on the jobs queue: `ReceiveMessage`, `DeleteMessage`, `ChangeMessageVisibility`.
  - `cloudwatch:PutMetricData` (namespace `DriftSense`), only when `METRICS_SINK` is `cloudwatch` or `both`.
- Timeouts:
  - Queue `VisibilityTimeout` 180 s; the worker also sets 180 on receive.
  - The heartbeat re-extends visibility to 180 s every 60 s for as long as the message is held, including the
    approval wait. A 900 s wait needs about 15 heartbeats, far below SQS's 12 h visibility ceiling.
  - Redrive `maxReceiveCount` 3 → DLQ. Message retention ≥ 1 h.
  - systemd `TimeoutStopSec` ≥ `APPROVAL_TIMEOUT_S` + 120 (about 1020 s) if a SIGTERM must let a waiting episode
    finish. If the worker is killed sooner, SQS redelivers and the episode reruns, reusing any stored decision.

**API (Lambda):**

- Env: `JOBS_TABLE`, `JOBS_QUEUE_URL`, `DATA_BUCKET`, `UI_ORIGIN`, `ENGINE_GIT_SHA`, `DRIFTSENSE_ENGINE` (only
  echoed), `MAX_UPLOAD_MB`, `JOB_TTL_DAYS`.
- IAM:
  - DynamoDB: `GetItem`, `PutItem` and `ConditionCheckItem` (used by TransactWriteItems), `UpdateItem`.
  - S3: `GetObject` on `jobs/*` and `traces/*`, `PutObject` on `uploads/*` (for presigned PUTs), `ListBucket`.
  - SQS: `SendMessage`.
- A 10 s timeout is enough, because no request blocks on the worker.

## Tests and local e2e

```bash
python -m pytest -q tests/service          # moto only, no network; ~35 s incl. real v2 + real agent episodes
python -m service.e2e_local --engine fake
python -m service.e2e_local --engine v2
python -m service.e2e_local --engine v2 --agent
python -m ruff check service tests/service && python -m ruff format --check service tests/service
```

Every test module calls `pytest.importorskip` for fastapi/httpx/mangum/moto/boto3/driftsense, so CI without them
skips the module. The real-episode tests (`tests/service/test_agent_episodes.py`) take about 1 to 3 s each and run in
the default (non-slow) suite. They cover:

- commit
- natural escalation and the approval round trip (`vsem-dram-17`)
- rejection and timeout (forced with `max_steps=1`)
- redelivery that reuses a stored decision
- trace streaming
- validation and idempotency

## Dependencies (tested versions)

**Runtime:**

- API (Lambda): `fastapi`, `mangum`, `pydantic`, `boto3/botocore` (pinned in `requirements.lock`). It needs no cv2 or numpy.
- Worker: the same plus `numpy` and the cv2 from the COOL venv (`opencv 5.0.0` was used in dev; never pip-install opencv on COOL).

**Dev/test:** `moto` (s3, sqs and dynamodb are enough), `httpx`, `requests` (one presigned-PUT round-trip test, skipped if absent), `pytest`, `ruff`.
