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
#   (deploy, bench, docs land here in B's phases)
