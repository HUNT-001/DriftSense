# DriftSense Pilot
# ---- shared targets ----
.PHONY: setup test-fast test lint contracts contracts-check claims lock check

setup:
	python -m pip install -U pip
	pip install -r requirements.lock
	pip install -e . --no-deps

test-fast:
	python -m pytest -q -m "not slow" -x

test:
	python -m pytest -q

lint:
	ruff check .
	ruff format --check .

contracts:
	python -c "from pathlib import Path; from driftsense.contracts import export_schemas; export_schemas(Path('contracts'))"

contracts-check: contracts
	git diff --exit-code contracts/

claims:
	python scripts/check_claims.py || test ! -d docs/templates

lock:
	pip-compile --strip-extras -q -o requirements.lock requirements.in
	grep -v '^opencv-python-headless' requirements.lock > requirements-cool.lock

check: lint test contracts-check claims

# ---- A targets ----
.PHONY: gate-g2 gate-g5 eval-dev calibration eval-sealed-grade eval

gate-g2:  # full DEV speed/accuracy gate; rewrites results/accuracy/{v2_dev,g2}.json
	python -m eval.gate_g2

gate-g5:  # agent arms on DEV episodes; prints the G5 verdict, writes nothing
	python -m eval.agent_eval --no-write

eval-dev: gate-g5  # read-only DEV check; `make gate-g2` re-measures and rewrites results

calibration:  # CALIB split only; rewrites results/accuracy/calibration.json
	python -m eval.calibration

eval-sealed-grade:  # re-grade committed results/sealed/*.json against D1..D12
	python -m eval.sealed_run --grade

eval: eval-dev eval-sealed-grade

# ---- B targets ----
.PHONY: docs docs-check report-pdf dev-server e2e-local infra-setup synth deploy destroy publish-ui gate-g4 smoke-local bench bench-local ui-setup ui-test ui-build

# Deploy inputs: COOL_AMI_ID (required for the worker) and ALERT_EMAIL (alarm/budget e-mails).
COOL_AMI_ID ?=
ALERT_EMAIL ?=
CDK_CTX = -c cool_ami_id=$(COOL_AMI_ID) -c api_asset=service $(if $(ALERT_EMAIL),-c alert_email=$(ALERT_EMAIL))

docs:
	python scripts/render_docs.py

docs-check:
	python scripts/render_docs.py --check

report-pdf: docs  # needs pandoc (+ a PDF engine such as xelatex or wkhtmltopdf)
	cd docs && pandoc REPORT.md -o REPORT.pdf --toc -V geometry:margin=2cm

dev-server: ui-build
	python -m service.dev_server --ui ui/dist

e2e-local:
	python -m service.e2e_local --engine fake
	python -m service.e2e_local --engine v2
	python -m service.e2e_local --engine v2 --agent

infra-setup:
	pip install -r infra/requirements.txt

synth:
	cd infra && npx --yes aws-cdk@2 synth --quiet

deploy:
	@test -n "$(COOL_AMI_ID)" || (echo "set COOL_AMI_ID=ami-... (infra/README.md step 2)"; exit 2)
	cd infra && npx --yes aws-cdk@2 deploy --all --require-approval never $(CDK_CTX) --outputs-file cdk-outputs.json

destroy:  # RETAINed buckets and the jobs table stay; see infra/README.md step 7
	cd infra && npx --yes aws-cdk@2 destroy --all --force $(CDK_CTX)

publish-ui: ui-build
	python scripts/publish_ui.py --outputs infra/cdk-outputs.json

gate-g4:  # live stack: health, version, demo job on the COOL worker, agent episode, UI config
	python scripts/smoke_deploy.py --outputs infra/cdk-outputs.json --agent --expect-cool --out smoke-evidence.json

smoke-local:  # same checks against `make dev-server` (or python -m service.dev_server)
	python scripts/smoke_deploy.py --api http://127.0.0.1:8000 --agent

bench: bench-local

bench-local:
	python -m bench.cool.microbench --quick
	python -m bench.cool.stagebench --quick

ui-setup:
	cd ui && npm ci

ui-test:
	cd ui && npm test

ui-build:
	cd ui && npm run build
