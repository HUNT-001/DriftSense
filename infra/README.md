# DriftSense Pilot: AWS infrastructure (CDK v2, Python, ap-south-1)

Owner: Person B. One CDK app, all stacks prefixed `DriftSense-`, tagged `project=driftsense`, `owner=B`.
Region is fixed to **ap-south-1**. Worker: Graviton4 **c8g.large** on the **Cloud Optimized OpenCV (COOL) 3.1** AMI.

## Architecture

```text
Browser ──HTTPS──> CloudFront ──OAC──> S3 UI bucket (private; index.html + config.json {"apiBase": <ApiUrl>})
   │
   └──HTTPS──> API Gateway HTTP API ($default, 20 rps / burst 40, CORS = CloudFront origin only)
                   └─> Lambda service.api.handler (Python 3.12, arm64, Mangum/FastAPI, no cv2)
                          ──> DynamoDB jobs (TTL expires_at) ──> SQS jobs ──(3 receives)──> DLQ ──> alarm ──> SNS
                          ──> S3 data bucket: presigned PUT uploads/ (7-day expiry); GET jobs/ traces/
   SQS jobs ──long poll──> EC2 ASG 1..2 × c8g.large, COOL 3.1 AMI, public subnet, SG with NO inbound, SSM only
                              /opt/driftsense/venv (--system-site-packages on /opt/cool/venvs/python_3.12)
                              systemd driftsense-worker: python -m service.worker (cwd + PYTHONPATH = app root)
                              ──> S3 jobs/ traces/ ; DynamoDB status ; CloudWatch PutMetricData (ns DriftSense)
                              (optional, off) Bedrock InvokeModel on ONE model / inference profile
EventBridge rate(30 min) ──> canary Lambda ──GET /healthz, /v1/version──> CanaryOk metric ──> alarm ──> SNS
Bench (only -c bench=true): c7i.4xlarge pip | c8g.4xlarge pip | c8g.4xlarge COOL [| c8g.large COOL]
                              ──> bash bench/cool/run_all.sh <arm> s3://<data>/bench/<run>/<arm>/ ──> self-terminate
GitHub Actions (main) ──OIDC──> DriftSenseGithubDeploy ──sts:AssumeRole──> cdk-* bootstrap roles ──> cdk deploy
```

| Stack | What |
| --- | --- |
| `DriftSense-Network` | VPC, **public subnets only** in ap-south-1a/b/c (explicit, no AZ lookup), `nat_gateways=0`, free S3 + DynamoDB **gateway** endpoints, flow logs (7 days), worker SG (**no inbound**, all egress). Optional interface endpoints. |
| `DriftSense-Storage` | UI bucket + CloudFront (OAC, SPA 403/404 → `/index.html`, HTTPS redirect); data bucket (SSE-S3, TLS-only, BPA, 7-day expiry on `uploads/`, `jobs/`, `traces/`; `demos/` kept; CORS PUT/GET from CloudFront only); DynamoDB `jobs` (`job_id`, on-demand, TTL `expires_at`, PITR); access-log bucket. Outputs `UiBucketName`, `UiDistributionId`, `UiUrl`, `DataBucketName`, `JobsTableName`. |
| `DriftSense-Queue` | SQS jobs (SSE-SQS, TLS-only, visibility 900 s, wait 20 s, 4-day retention) + DLQ (`maxReceiveCount=3`, 14 days); SNS alarm topic; DLQ-not-empty alarm. |
| `DriftSense-Api` | HTTP API (`$default`, auto-deploy, throttle 20/40, JSON access logs) → Lambda (Python 3.12 **arm64**, 512 MB, 29 s). Routes `GET /healthz`, `ANY /{proxy+}`. Output `ApiUrl`. |
| `DriftSense-Budget` | Monthly COST budget (`budget_usd`, default 100), ACTUAL **$40 / $80** and FORECASTED 100 % e-mail alerts. |
| `DriftSense-Worker` | Launch template (COOL AMI from context, c8g.large, **IMDSv2 required**, hop limit 2, **no key pair**, public IP, encrypted gp3 30 GB) + ASG min 1 / max 2 (rolling update). Output `WorkerAsgName`. Only if `cool_ami_id` is set. |
| `DriftSense-Observability` | Dashboard `DriftSense`; canary Lambda every 30 min; alarms CanaryOk < 1 (2 periods, missing = breaching) and worker InService < 1 → SNS. |
| `DriftSense-GithubOidc` | GitHub OIDC provider + role `DriftSenseGithubDeploy`, trusted only for `aud=sts.amazonaws.com` and `sub=repo:HUNT-001/DriftSense:ref:refs/heads/main`; may only `sts:AssumeRole` into `role/cdk-*`. Output `DeployRoleArn`. |
| `DriftSense-Bench` | Only with `-c bench=true` (and `cool_ami_id`): self-terminating benchmark instances (below). |

## Service contract (matches `service/README.md`)

**API Lambda** (`service.api.handler`) env: `JOBS_TABLE`, `JOBS_QUEUE_URL`, `DATA_BUCKET`, `UI_ORIGIN`, `MAX_UPLOAD_MB=20`,
`DRIFTSENSE_ENGINE`, `ENGINE_GIT_SHA`, `JOB_TTL_DAYS`, `APPROVAL_TIMEOUT_S`, `LOG_LEVEL` (`AWS_REGION` is injected by Lambda).
Role: `dynamodb:GetItem/PutItem/UpdateItem/ConditionCheckItem` (TransactWriteItems), `sqs:SendMessage`, `s3:PutObject` on
`uploads/*` (presigning), `s3:GetObject` on `jobs/*`, `traces/*`, `demos/*`, `s3:ListBucket` on the data bucket (404, not 403,
for missing objects).

**Worker** systemd env: `JOBS_TABLE`, `JOBS_QUEUE_URL`, `DATA_BUCKET`, `AWS_REGION`, `AWS_DEFAULT_REGION`,
`DRIFTSENSE_ENGINE=v2`, `ENGINE_GIT_SHA=f9395b84dd654ff1d537f6817c258a0f2b9373de` (v2-freeze, from the `protocol-v1` tag),
`APPROVAL_TIMEOUT_S=900`, `APPROVAL_POLL_S=2`, `TRACE_UPLOAD_S=2`, `JOB_TTL_DAYS`, `LOG_LEVEL`, `METRICS_SINK=both`,
`WORKER_LOG_GROUP`, `PYTHONPATH=/opt/driftsense/app` (+ `BEDROCK_MODEL_ID` only with `-c bedrock=true`).
`WorkingDirectory=/opt/driftsense/app` (repo-root layout: `driftsense/`, `service/`, `bench/`, `config.py`, `data_gen/`,
`localization/`; `driftsense.sim` imports the top-level `config` and `data_gen`).
Role: SSM core; `sqs:ReceiveMessage/DeleteMessage/ChangeMessageVisibility/GetQueueAttributes`; `dynamodb:GetItem/PutItem/UpdateItem`;
`s3:GetObject+PutObject` on `jobs/*`, `traces/*`; `s3:GetObject` on `uploads/*`, `demos/*`; `s3:ListBucket` on the bucket;
`cloudwatch:PutMetricData` (condition `cloudwatch:namespace=DriftSense`); logs on `/driftsense/worker`.

**Rollback to v1:** `-c engine=v1` (one value; changes API + worker env and re-rolls the ASG). Override the SHA with
`-c engine_git_sha=<sha>`.

**Long agent episodes (15-min human approval).** The worker receives with `VisibilityTimeout=180` and a heartbeat thread
re-extends it to 180 s every 60 s while a job runs, including an episode blocked on `request_human_approval`. The queue's
default visibility is 900 s (≥ 180 ≥ 60, asserted in `tests/infra`), so even a receive without an explicit timeout or a
worker that stops heartbeating for <15 min does not cause a duplicate delivery. SQS caps a message's total visibility at
12 h, far above 15 min. systemd `TimeoutStopSec = APPROVAL_TIMEOUT_S + 120 = 1020` so a SIGTERM (rolling update, scale-in)
lets an in-flight episode finish (`effective_tool_timeout_s = approval + 60`).

**CORS:** API Gateway `cors_preflight` is the single source of truth (origin = `https://<cloudfront domain>`).

**Custom metrics** (namespace `DriftSense`): `JobLatencyMs`, `JobsSucceeded`, `JobsFailed` (dim `Engine`), `StageMs`
(dim `Stage`), `CanaryOk`. `METRICS_SINK=both` makes the worker call `PutMetricData` (EMF on stdout only reaches journald).

### Worker bootstrap (user-data, `set -euxo pipefail`; any failure stops the boot before the service starts)

1. Ensure the SSM agent runs; install AWS CLI if missing. 2. Download the app asset (built at synth from the repo paths
listed above + `requirements-cool.lock`) to `/opt/driftsense/app`. 3. Fail unless `/opt/cool/venvs/python_3.12/bin/python`
exists; create `/opt/driftsense/venv --system-site-packages` on it plus a `.pth` bridge to COOL's site-packages.
Step 4: **fail if `requirements-cool.lock` pins any `opencv*`.** Drop the lock's `numpy` pin when COOL ships numpy (COOL's cv2
is built against it). `pip install --no-deps -r` the lock (a closed pinned closure, so nothing transitive, in particular no
opencv wheel, can be pulled in); `pip check` is informational. 5. **Fail if `pip list --local` shows opencv** or if
`python -c "import cv2; print(cv2.__file__)"` is not under `/opt/cool/`. 6. Fail unless `driftsense.api.available_engines()`
contains the configured engine. 7. Write and start `driftsense-worker.service` (`python -m service.worker`, user `driftsense`).

### API Lambda bundle (no Docker)

`-c api_asset=auto` (default) bundles the real service when `service/api.py` exists, else uses the placeholder handler with a
synth warning; `service` forces the real bundle (CI uses it); `placeholder` is the stub (tests). The bundle is
`service/` (minus `worker.py`, `e2e_local.py`), `driftsense/__init__.py`, `driftsense/contracts.py` and
`pip install --target --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 --implementation cp
--python-version 3.12 --only-binary=:all: -c requirements.lock fastapi==… mangum==… pydantic==…` (versions from the lock;
boto3 comes from the runtime). Synth fails if `cv2`, `numpy`, `boto3` or `botocore` end up in it. It is cached in
`infra/.build/api_bundle` by an input hash, so only the first synth (or a change) needs network.

## Context keys

| Key | Default | Meaning |
| --- | --- | --- |
| `cool_ami_id` | — (**required** for worker/bench) | COOL 3.1 AMI ID in ap-south-1. Without it Worker and Bench are skipped (warning). |
| `alert_email` (alias `alarm_email`) | — | SNS alarm subscription + budget notifications. |
| `budget_usd` | `100` | Monthly budget limit (alerts at $40 / $80 actual regardless). |
| `engine` | `v2` | `DRIFTSENSE_ENGINE` (`v1` = rollback, `fake` = plumbing). |
| `engine_git_sha` | `f9395b84…373de` | `ENGINE_GIT_SHA` (v2-freeze). |
| `approval_timeout_s` | `900` | `APPROVAL_TIMEOUT_S`; also sets `TimeoutStopSec` (+120). |
| `job_ttl_days` / `log_level` | `7` / `INFO` | Passed to API and worker. |
| `api_asset` | `auto` | `auto` \| `service` \| `placeholder` (see above). |
| `azs` | `ap-south-1a,ap-south-1b,ap-south-1c` | VPC zones (public subnet per zone). |
| `interface_endpoints` | `false` | SQS/SSM/SSMMessages/EC2Messages interface endpoints (~$60/month, off). |
| `bedrock` | `false` | Add `bedrock:InvokeModel` for ONE model to the worker role (+ `BEDROCK_MODEL_ID` env). |
| `bedrock_model_id` | `amazon.nova-lite-v1:0` | Foundation model. |
| `bedrock_inference_profile` | — | e.g. `apac.amazon.nova-lite-v1:0`: grants the profile ARN + that one model in any region, conditioned on the profile. |
| `root_device_name` | `/dev/sda1` | Root device of the AMIs (Ubuntu). |
| `bench` / `bench_run_id` / `bench_max_minutes` | `false` / `adhoc` / `360` | Bench stack; S3 prefix `bench/<run_id>/`; hard self-terminate cap. |
| `bench_large` | `false` | Add the `arm-cool-large` c8g.large COOL arm (production cost claim). |
| `bench_az` | — | Pin bench instances to one AZ (if c8g/c7i capacity is missing in the first). |
| `github_repo` / `github_oidc_provider_arn` | `HUNT-001/DriftSense` / — | OIDC trust; reuse an existing provider. |

## cdk-nag

cdk-nag **3.x** (`AwsSolutionsChecks`) runs as a CDK policy-validation plugin over the whole app; **synth fails on any
unacknowledged finding** (asserted by `test_cdk_nag_blocks_unjustified_findings`). `driftsense_infra/nag.py` wraps it so
v2-style scoped suppressions (`applies_to` strings / regexes, limited to a construct subtree) still work; every
suppression carries a reason:

| Rule | Resource | Why |
| --- | --- | --- |
| S1 | access-log bucket | It is the access-log destination. |
| CFR1 / CFR2 | UI distribution | Global judges, no geo restriction; WAF cost (API is throttled, UI static). |
| CFR3 / CFR4 | UI distribution | Standard logging needs an ACL bucket; default `*.cloudfront.net` certificate. |
| IAM5 | API, worker, canary, bench, GitHub roles | Object keys under fixed prefixes, own log groups, the CDK asset-bucket read grant, `PutMetricData` (namespace condition), `role/cdk-*`, the one Bedrock model via profile. |
| IAM4 | worker, bench roles | `AmazonSSMManagedInstanceCore` (Session Manager baseline). |
| L1 | API, canary Lambdas | Python 3.12 pinned to match COOL `python_3.12`. |
| APIG4 | API routes | Public demo API without accounts; throttled, CORS-restricted. |
| AS3 | worker ASG | Scaling notifications replaced by the InService alarm. |
| EC28 / EC29 | bench instances | Short-lived; termination protection would defeat self-termination. |
| CFN W9010 / W3010 | bench instances | COOL AMI ID is a per-deploy context value; AZs are explicit by design. |

## Local checks

```bash
# from the repo root, any venv with infra/requirements.txt installed
python -m pytest -q tests/infra            # ~1-3 min, no network, no AWS
ruff check infra tests/infra && ruff format --check infra tests/infra
cd infra && npm ci && CDK_DEFAULT_ACCOUNT=123456789012 npx cdk synth -q -c cool_ami_id=ami-0123456789abcdef0
```

---

## Deploy runbook (human with AWS SSO admin; nobody else deploys)

Costs while running (ap-south-1, on-demand, estimates from `bench/cool/prices.json`, verify in the Pricing Calculator):
worker c8g.large ≈ $0.054/h + COOL fee ≈ $0.01/h ≈ **$47/month** if left on 24×7 (≈ $31 for ~500 h), public IPv4 ≈ $3.6/month,
everything else (Lambda, HTTP API, DynamoDB on-demand, SQS, S3, CloudFront, logs, canary) is cents at demo traffic.
Bench run: c8g.4xlarge ≈ $0.43/h (+$0.04/h COOL), c7i.4xlarge ≈ $0.7–0.8/h (look up; pass `--price-c7i`), ~1–2 h each,
hard cap 6 h. AWS credits do **not** cover the Marketplace COOL fee. Budget alerts at $40 and $80.

### 0. Tools and login

```bash
aws configure sso --profile driftsense      # once; then:
aws sso login --profile driftsense
export AWS_PROFILE=driftsense AWS_REGION=ap-south-1
export ACCT=$(aws sts get-caller-identity --query Account --output text)
cd infra && npm ci && python -m pip install -r requirements.txt   # Python 3.12 venv; Node 22
```

Deploy from a checkout of `main` that contains `service/`, `driftsense/`, `bench/` (the bundles are built from the working
tree; `ENGINE_GIT_SHA` is only a label, so the engine code in the checkout must be the v2-freeze engine).

### 1. Bootstrap (once per account/region)

```bash
npx cdk bootstrap aws://$ACCT/ap-south-1
```

### 2. Subscribe to COOL and find the AMI ID

1. AWS Marketplace console → search **"Cloud Optimized OpenCV"** (Graviton4, COOL 3.1) → *Continue to Subscribe* → accept
   terms (wait until the subscription is *Active*) → *Continue to Configuration*, region **Asia Pacific (Mumbai)**, version
   3.1: the page shows the **AMI ID**. Record it (evidence).
2. Or from the CLI:

```bash
aws ec2 describe-images --region ap-south-1 --owners aws-marketplace \
  --filters "Name=name,Values=*Cloud*Optimized*OpenCV*" "Name=architecture,Values=arm64" \
  --query 'sort_by(Images,&CreationDate)[].[ImageId,Name,CreationDate,ProductCodes[0].ProductCodeId]' --output table
# product-code approach (exact): take ProductCodeId from the row above or the Marketplace "Usage instructions"
aws ec2 describe-images --region ap-south-1 --owners aws-marketplace \
  --filters "Name=product-code,Values=<product-code>" --query 'sort_by(Images,&CreationDate)[-1].[ImageId,Name]'
export COOL_AMI=ami-xxxxxxxxxxxxxxxxx
```

Check the root device (`aws ec2 describe-images --image-ids $COOL_AMI --query 'Images[0].RootDeviceName'`); if it is not
`/dev/sda1`, add `-c root_device_name=<value>` to every command below.

(Optional Bedrock: enable model access for Amazon Nova Lite in ap-south-1; if only the APAC cross-region profile is offered
add `-c bedrock=true -c bedrock_inference_profile=apac.amazon.nova-lite-v1:0`, else `-c bedrock=true`. Off by default.)

### 3. Deploy

```bash
npx cdk deploy --all --require-approval broadening \
  -c cool_ami_id=$COOL_AMI -c alert_email=you@example.com -c api_asset=service \
  --outputs-file cdk-outputs.json
```

Confirm the SNS subscription e-mail. Read outputs:

```bash
API=$(jq -r '."DriftSense-Api".ApiUrl' cdk-outputs.json); API=${API%/}
UI_BUCKET=$(jq -r '."DriftSense-Storage".UiBucketName' cdk-outputs.json)
DIST=$(jq -r '."DriftSense-Storage".UiDistributionId' cdk-outputs.json)
UI_URL=$(jq -r '."DriftSense-Storage".UiUrl' cdk-outputs.json)
DATA=$(jq -r '."DriftSense-Storage".DataBucketName' cdk-outputs.json)
```

For CI: set GitHub repo variables `AWS_DEPLOY_ROLE_ARN` (output `DriftSense-GithubOidc.DeployRoleArn`), `COOL_AMI_ID`,
`ALERT_EMAIL`. Pushes to `main` touching `infra/`, `service/`, `driftsense/` or the locks then redeploy
(`.github/workflows/deploy.yml`). If a GitHub OIDC provider already exists: `-c github_oidc_provider_arn=<arn>`.

### 4. Upload the UI

```bash
(cd ../ui && npm ci && npm run build)
printf '{"apiBase": "%s"}\n' "$API" > ../ui/dist/config.json
aws s3 sync ../ui/dist "s3://$UI_BUCKET/" --delete --exclude config.json --exclude index.html \
  --cache-control "public,max-age=31536000,immutable"
aws s3 cp ../ui/dist/index.html "s3://$UI_BUCKET/index.html" --cache-control "no-cache"
aws s3 cp ../ui/dist/config.json "s3://$UI_BUCKET/config.json" --cache-control "no-cache" --content-type application/json
aws cloudfront create-invalidation --distribution-id "$DIST" --paths /index.html /config.json
echo "$UI_URL"
```

### 5. Verify G4 (COOL worker on Graviton4)

```bash
curl -s "$API/healthz"                    # {"ok":true}
curl -s "$API/v1/version" | jq            # git_sha = v2-freeze, engine v2, arch aarch64 (Lambda; opencv_file null there)
# worker: the boot log must show cv2 under /opt/cool, and the unit must be active
IID=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names \
  "$(jq -r '."DriftSense-Worker".WorkerAsgName' cdk-outputs.json)" --query 'AutoScalingGroups[0].Instances[0].InstanceId' --output text)
aws ssm start-session --target "$IID"     # then: sudo journalctl -u driftsense-worker | grep cv2.__file__
                                          #       sudo grep 'cv2:' /var/log/cloud-init-output.log
# submit a demo job and poll it
DEMO=$(curl -s "$API/v1/demos" | jq -r '.[0].demo_id')
JOB=$(curl -s -X POST "$API/v1/jobs" -H 'content-type: application/json' \
  -d "{\"demo_id\":\"$DEMO\",\"client_token\":\"g4-$(date +%s)\"}" | jq -r .job_id)
watch -n 2 "curl -s $API/v1/jobs/$JOB | jq '{status, worker}'"
```

G4 passes when `JobStatus.status == "succeeded"` and `worker.cool == true`, `worker.arch == "aarch64"`,
`worker.instance_type == "c8g.large"` (and the worker log shows `cv2.__file__` under `/opt/cool`). Keep the JSON and the
log line as evidence. `/v1/version` runs on Lambda, so its `opencv_file` is `null`; the COOL proof is the worker's
`JobStatus.worker` and its startup log.

### 6. Bench campaign (G6)

```bash
RUN=r$(date +%Y%m%d-%H%M)
npx cdk deploy DriftSense-Bench -c bench=true -c bench_run_id=$RUN -c bench_large=true \
  -c cool_ami_id=$COOL_AMI -c alert_email=you@example.com -c api_asset=service --require-approval never
# instances run bash bench/cool/run_all.sh <arm> s3://$DATA/bench/$RUN/<arm>/ and power off (= terminate)
aws ec2 describe-instances --filters "Name=tag:aws:cloudformation:stack-name,Values=DriftSense-Bench" \
  --query 'Reservations[].Instances[].[Tags[?Key==`aws:cloudformation:logical-id`]|[0].Value,State.Name]' --output table
mkdir -p ../out && aws s3 sync "s3://$DATA/bench/$RUN/" "../out/bench-$RUN/"
for a in x86-pip arm-pip arm-cool arm-cool-large; do ls "../out/bench-$RUN/$a"; done
cd .. && python -m bench.cool.report --arm out/bench-$RUN/x86-pip --arm out/bench-$RUN/arm-pip \
  --arm out/bench-$RUN/arm-cool --arm out/bench-$RUN/arm-cool-large \
  --price-c7i <usd_per_hour_ap-south-1> --out results/bench/ && cd infra
npx cdk destroy DriftSense-Bench -c bench=true -c bench_run_id=$RUN -c cool_ami_id=$COOL_AMI --force
```

Each arm also uploads `cloud-init-output.log` next to its results (on failure too). If an instance fails with
"Unsupported" capacity in its AZ, redeploy with `-c bench_az=ap-south-1b` (or `c`). A new `bench_run_id` replaces the
instances (new run).

### 7. Teardown (human only)

```bash
npx cdk destroy DriftSense-Bench -c bench=true -c cool_ami_id=$COOL_AMI --force   # if still present
npx cdk destroy --all -c cool_ami_id=$COOL_AMI
# RETAINed on purpose (empty and delete by hand when done): UI bucket, data bucket, access-log bucket, jobs table
aws s3 rm "s3://$DATA" --recursive && aws s3 rb "s3://$DATA"      # likewise the UI and access-log buckets
aws dynamodb delete-table --table-name "$(jq -r '."DriftSense-Storage".JobsTableName' cdk-outputs.json)"
```

Then cancel the COOL Marketplace subscription (Marketplace → Manage subscriptions) to stop any future fees.
