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
#   (gate-g2, gate-g5, eval-* land here in phases A4/A8)

# ---- B targets ----
.PHONY: docs docs-check dev-server e2e-local infra-setup synth deploy bench-local ui-setup ui-test ui-build

docs:
	python scripts/render_docs.py

docs-check:
	python scripts/render_docs.py --check

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
	cd infra && npx --yes aws-cdk@2 deploy --all --require-approval never

bench-local:
	python -m bench.cool.microbench --quick
	python -m bench.cool.stagebench --quick

ui-setup:
	cd ui && npm ci

ui-test:
	cd ui && npm test

ui-build:
	cd ui && npm run build
