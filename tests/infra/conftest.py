"""Fixtures for the CDK assertion tests.

The shared CI does not install aws-cdk-lib, so every test module here starts with
``pytest.importorskip("aws_cdk")`` and is reported as skipped there; run locally with
the infra venv (see infra/README.md).

Note: ``pytest.importorskip`` is deliberately NOT called at conftest module level:
pytest turns a conftest-level skip into a collection error, which would break CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

INFRA_DIR = Path(__file__).resolve().parents[2] / "infra"
if str(INFRA_DIR) not in sys.path:
    sys.path.insert(0, str(INFRA_DIR))

TEST_AMI = "ami-0123456789abcdef0"


def _feature_flags() -> dict:
    """Same feature flags as `cdk synth` (infra/cdk.json) so tests see the real templates."""
    return json.loads((INFRA_DIR / "cdk.json").read_text(encoding="utf-8"))["context"]


@pytest.fixture(scope="session")
def make_app(tmp_path_factory):
    pytest.importorskip("aws_cdk")
    pytest.importorskip("cdk_nag")
    from driftsense_infra.build import build_app

    def _make(**context):
        ctx = _feature_flags()
        # Never pip-download in tests: the API Lambda gets the stub handler unless a test overrides it.
        ctx["api_asset"] = "placeholder"
        ctx.update({k: v for k, v in context.items() if v is not None})
        outdir = tmp_path_factory.mktemp("cdkout")
        return build_app(context=ctx, outdir=str(outdir))

    return _make


@pytest.fixture(scope="session")
def default_app(make_app):
    """The app as `cdk synth -c cool_ami_id=<test ami>` builds it."""
    return make_app(cool_ami_id=TEST_AMI)


@pytest.fixture(scope="session")
def templates(default_app):
    from aws_cdk.assertions import Template

    return {stack.stack_name: Template.from_stack(stack) for stack in default_app.stacks()}
