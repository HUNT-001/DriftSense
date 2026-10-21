"""Assertions on the B3 stacks: COOL worker, observability/canary, GitHub OIDC, bench."""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("aws_cdk")

from aws_cdk.assertions import Annotations, Match, Template  # noqa: E402

from .conftest import TEST_AMI  # noqa: E402

PREFIX = "DriftSense-"
V2_FREEZE_SHA = "f9395b84dd654ff1d537f6817c258a0f2b9373de"


def _flatten_user_data(user_data) -> str:
    """Render Fn::Base64(Fn::Join) user-data to text (tokens as JSON) for substring checks."""
    if isinstance(user_data, dict) and "Fn::Base64" in user_data:
        user_data = user_data["Fn::Base64"]
    if isinstance(user_data, str):
        return user_data
    sep, parts = user_data["Fn::Join"]
    return sep.join(p if isinstance(p, str) else json.dumps(p) for p in parts)


def _statements(template: Template) -> list[dict]:
    out = []
    for policy in template.find_resources("AWS::IAM::Policy").values():
        out.extend(policy["Properties"]["PolicyDocument"]["Statement"])
    return out


def _actions(st: dict) -> list[str]:
    return [st["Action"]] if isinstance(st["Action"], str) else list(st["Action"])


def _resources(st: dict) -> list:
    return st["Resource"] if isinstance(st["Resource"], list) else [st["Resource"]]


@pytest.fixture(scope="session")
def worker_template(default_app):
    assert default_app.worker is not None
    return Template.from_stack(default_app.worker)


@pytest.fixture(scope="session")
def worker_user_data(worker_template) -> str:
    (lt,) = worker_template.find_resources("AWS::EC2::LaunchTemplate").values()
    return _flatten_user_data(lt["Properties"]["LaunchTemplateData"]["UserData"])


@pytest.fixture(scope="session")
def bench_app(make_app):
    return make_app(cool_ami_id=TEST_AMI, bench="true", bench_run_id="test-run")


# ---------------------------------------------------------------- worker


def test_worker_launch_template(worker_template):
    worker_template.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {
            "LaunchTemplateData": Match.object_like(
                {
                    "ImageId": TEST_AMI,
                    "InstanceType": "c8g.large",
                    "MetadataOptions": Match.object_like(
                        {"HttpTokens": "required", "HttpPutResponseHopLimit": 2}
                    ),
                    "Monitoring": {"Enabled": False},
                    "BlockDeviceMappings": [
                        Match.object_like(
                            {
                                "Ebs": {
                                    "Encrypted": True,
                                    "VolumeSize": 30,
                                    "VolumeType": "gp3",
                                    "DeleteOnTermination": True,
                                }
                            }
                        )
                    ],
                    "NetworkInterfaces": [Match.object_like({"AssociatePublicIpAddress": True})],
                }
            )
        },
    )
    (lt,) = worker_template.find_resources("AWS::EC2::LaunchTemplate").values()
    assert "KeyName" not in lt["Properties"]["LaunchTemplateData"]


def test_worker_ami_comes_from_context(make_app):
    built = make_app(cool_ami_id="ami-0fedcba9876543210")
    (lt,) = Template.from_stack(built.worker).find_resources("AWS::EC2::LaunchTemplate").values()
    assert lt["Properties"]["LaunchTemplateData"]["ImageId"] == "ami-0fedcba9876543210"


def test_worker_asg_min1_max2(worker_template):
    worker_template.has_resource_properties(
        "AWS::AutoScaling::AutoScalingGroup", {"MinSize": "1", "MaxSize": "2"}
    )


def test_worker_user_data_cool_guards(worker_user_data):
    ud = worker_user_data
    assert ud.startswith("#!/bin/bash")
    assert "set -euxo pipefail" in ud
    assert "/opt/cool/venvs/python_3.12" in ud
    assert "-m venv --system-site-packages /opt/driftsense/venv" in ud
    assert "pip list --local --format=freeze | grep -i opencv" in ud
    assert "exit 1" in ud
    assert "import cv2; print(cv2.__file__)" in ud
    assert "/opt/cool/*) ;;" in ud and "FATAL: cv2 is not loaded from /opt/cool" in ud
    # the lock is installed as a closed closure (no transitive opencv) and must not pin opencv itself
    assert "-m pip install --disable-pip-version-check --no-input --no-deps -r" in ud
    assert "requirements-cool.lock" in ud
    assert "grep -inE '^[[:space:]]*opencv' requirements-cool.lock" in ud
    _assert_never_pip_installs_opencv(ud)


def _assert_never_pip_installs_opencv(ud: str) -> None:
    commands = [ln for ln in ud.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in commands if re.search(r"pip\S*\s+install\b.*opencv", ln, re.I)]
    # COOL hosts install only requirements-cool.lock, never the PyPI-opencv requirements.lock
    assert not [ln for ln in commands if re.search(r"pip\S*\s+install\b.*requirements\.lock", ln)]


def test_worker_user_data_order_guards_before_service(worker_user_data):
    ud = worker_user_data
    assert (
        ud.index("--no-deps -r")
        < ud.index("FATAL: cv2 is not loaded")
        < ud.index("systemctl enable --now driftsense")
    )
    assert 'assert "v2" in e' in ud  # engine import smoke test before the unit starts


def test_worker_systemd_unit(worker_user_data):
    ud = worker_user_data
    assert "driftsense-worker.service" in ud
    assert "ExecStart=/opt/driftsense/venv/bin/python -m service.worker" in ud
    for key in (
        "JOBS_TABLE",
        "JOBS_QUEUE_URL",
        "DATA_BUCKET",
        "AWS_REGION",
        "DRIFTSENSE_ENGINE",
        "ENGINE_GIT_SHA",
        "JOB_TTL_DAYS",
    ):
        assert f"Environment={key}=" in ud, key
    assert "Environment=DRIFTSENSE_ENGINE=v2" in ud
    assert f"Environment=ENGINE_GIT_SHA={V2_FREEZE_SHA}" in ud
    assert "Environment=APPROVAL_TIMEOUT_S=900" in ud
    assert "Environment=METRICS_SINK=both" in ud
    assert "Environment=PYTHONPATH=/opt/driftsense/app" in ud
    assert "BEDROCK_MODEL_ID" not in ud  # Bedrock is off by default
    # SIGTERM must outlast an agent episode blocked on approval (900 s + tool slack)
    assert "Restart=always" in ud and "KillSignal=SIGTERM" in ud and "TimeoutStopSec=1020" in ud
    assert "systemctl enable --now driftsense-worker.service" in ud
    assert "amazon-ssm-agent" in ud


def test_worker_role_no_wildcard_on_data_services(worker_template):
    for st in _statements(worker_template):
        acts = _actions(st)
        if any(a.split(":")[0] in ("s3", "dynamodb", "sqs") for a in acts):
            assert "*" not in _resources(st), st
            assert not any(a.endswith(":*") for a in acts), st


def test_worker_role_permissions(worker_template):
    sts = {st.get("Sid"): st for st in _statements(worker_template)}
    assert sorted(_actions(sts["JobsTable"])) == [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
    ]
    assert {"sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:ChangeMessageVisibility"} <= set(
        _actions(sts["JobsQueueConsume"])
    )
    rw = json.dumps(sts["S3ResultsReadWrite"]["Resource"])
    assert "/jobs/*" in rw and "/traces/*" in rw and "uploads" not in rw
    ro = json.dumps(sts["S3InputsRead"]["Resource"])
    assert "/uploads/*" in ro and "/demos/*" in ro
    assert _actions(sts["S3InputsRead"]) == ["s3:GetObject"]
    metrics = sts["MetricsDriftSenseNamespaceOnly"]
    assert metrics["Condition"] == {"StringEquals": {"cloudwatch:namespace": "DriftSense"}}


def test_worker_role_ssm_managed_policy(worker_template):
    (role,) = worker_template.find_resources("AWS::IAM::Role").values()
    assert "AmazonSSMManagedInstanceCore" in json.dumps(role["Properties"]["ManagedPolicyArns"])


def _bedrock_statements(template: Template) -> list[dict]:
    return [s for s in _statements(template) if any(a.startswith("bedrock:") for a in _actions(s))]


def test_bedrock_off_by_default(worker_template):
    assert _bedrock_statements(worker_template) == []


def test_bedrock_restricted_to_one_model(make_app):
    built = make_app(cool_ami_id=TEST_AMI, bedrock="true")
    template = Template.from_stack(built.worker)
    (st,) = _bedrock_statements(template)
    assert _actions(st) == ["bedrock:InvokeModel"]
    resources = _resources(st)
    assert len(resources) == 1
    arn = json.dumps(resources[0])
    assert "foundation-model/amazon.nova-lite-v1:0" in arn and "*" not in arn
    (lt,) = template.find_resources("AWS::EC2::LaunchTemplate").values()
    ud = _flatten_user_data(lt["Properties"]["LaunchTemplateData"]["UserData"])
    assert "Environment=BEDROCK_MODEL_ID=amazon.nova-lite-v1:0" in ud


def test_bedrock_model_id_from_context(make_app):
    built = make_app(cool_ami_id=TEST_AMI, bedrock="true", bedrock_model_id="amazon.nova-micro-v1:0")
    text = json.dumps(_statements(Template.from_stack(built.worker)))
    assert "foundation-model/amazon.nova-micro-v1:0" in text and "nova-lite" not in text


def test_bedrock_inference_profile(make_app):
    built = make_app(
        cool_ami_id=TEST_AMI, bedrock="true", bedrock_inference_profile="apac.amazon.nova-lite-v1:0"
    )
    sts = {s["Sid"]: s for s in _bedrock_statements(Template.from_stack(built.worker))}
    assert set(sts) == {"BedrockOneProfile", "BedrockOneModelViaProfile"}
    assert "inference-profile/apac.amazon.nova-lite-v1:0" in json.dumps(sts["BedrockOneProfile"]["Resource"])
    via = sts["BedrockOneModelViaProfile"]
    assert "foundation-model/amazon.nova-lite-v1:0" in json.dumps(via["Resource"])
    assert "bedrock:InferenceProfileArn" in json.dumps(via["Condition"])


def test_engine_rollback_to_v1_is_one_context_value(make_app):
    built = make_app(cool_ami_id=TEST_AMI, engine="v1")
    (lt,) = Template.from_stack(built.worker).find_resources("AWS::EC2::LaunchTemplate").values()
    ud = _flatten_user_data(lt["Properties"]["LaunchTemplateData"]["UserData"])
    assert "Environment=DRIFTSENSE_ENGINE=v1" in ud and 'assert "v1" in e' in ud
    Template.from_stack(built.api).has_resource_properties(
        "AWS::Lambda::Function",
        {"Environment": {"Variables": Match.object_like({"DRIFTSENSE_ENGINE": "v1"})}},
    )


def test_engine_git_sha_overridable(make_app):
    built = make_app(cool_ami_id=TEST_AMI, engine_git_sha="abc123")
    (lt,) = Template.from_stack(built.worker).find_resources("AWS::EC2::LaunchTemplate").values()
    ud = _flatten_user_data(lt["Properties"]["LaunchTemplateData"]["UserData"])
    assert "Environment=ENGINE_GIT_SHA=abc123" in ud


def test_invalid_engine_rejected(make_app):
    with pytest.raises(ValueError, match="engine"):
        make_app(cool_ami_id=TEST_AMI, engine="v3")


def test_queue_visibility_tolerates_worker_heartbeat(default_app):
    from driftsense_infra.queue import QUEUE_VISIBILITY_S
    from driftsense_infra.worker import WORKER_HEARTBEAT_S, WORKER_RECEIVE_VISIBILITY_S

    assert WORKER_HEARTBEAT_S < WORKER_RECEIVE_VISIBILITY_S <= QUEUE_VISIBILITY_S
    assert QUEUE_VISIBILITY_S >= 900  # >= the 15-min approval window even without heartbeats
    Template.from_stack(default_app.queue).has_resource_properties(
        "AWS::SQS::Queue", {"VisibilityTimeout": QUEUE_VISIBILITY_S, "RedrivePolicy": Match.any_value()}
    )


def test_worker_and_bench_skipped_without_ami(make_app):
    built = make_app()
    assert built.worker is None and built.bench is None
    names = {s.stack_name for s in built.stacks()}
    assert f"{PREFIX}Worker" not in names and f"{PREFIX}Bench" not in names
    Annotations.from_stack(built.network).has_warning("*", Match.string_like_regexp("cool_ami_id"))


def test_invalid_ami_rejected(make_app):
    with pytest.raises(ValueError, match="cool_ami_id"):
        make_app(cool_ami_id="not-an-ami")


# ---------------------------------------------------------------- observability


def test_canary_every_30_minutes(default_app):
    obs = Template.from_stack(default_app.observability)
    obs.has_resource_properties(
        "AWS::Events::Rule", {"ScheduleExpression": "rate(30 minutes)", "State": "ENABLED"}
    )
    obs.has_resource_properties(
        "AWS::Lambda::Function",
        {"Architectures": ["arm64"], "Runtime": "python3.12", "Handler": "canary.handler"},
    )


def test_canary_alarm(default_app):
    Template.from_stack(default_app.observability).has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "MetricName": "CanaryOk",
            "Namespace": "DriftSense",
            "Threshold": 1,
            "ComparisonOperator": "LessThanThreshold",
            "EvaluationPeriods": 2,
            "TreatMissingData": "breaching",
            "AlarmActions": [Match.any_value()],
        },
    )


def test_worker_in_service_alarm(default_app):
    Template.from_stack(default_app.observability).has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "MetricName": "GroupInServiceInstances",
            "Namespace": "AWS/AutoScaling",
            "Threshold": 1,
            "ComparisonOperator": "LessThanThreshold",
        },
    )


def test_dashboard_has_custom_metrics(default_app):
    (dash,) = (
        Template.from_stack(default_app.observability).find_resources("AWS::CloudWatch::Dashboard").values()
    )
    body = json.dumps(dash["Properties"]["DashboardBody"])
    for name in (
        "JobLatencyMs",
        "JobsSucceeded",
        "JobsFailed",
        "StageMs",
        "ApproximateAgeOfOldestMessage",
        "GroupInServiceInstances",
        "5xx",
        "4xx",
        "Latency",
    ):
        assert name in body, name


def test_canary_handler_logic(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "infra" / "assets" / "canary" / "canary.py"
    spec = importlib.util.spec_from_file_location("ds_canary", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []
    monkeypatch.setattr(
        mod, "_get", lambda url: calls.append(url) or ({"ok": True} if url.endswith("healthz") else {})
    )
    ok, _ = mod.check("https://example.invalid/")
    assert ok and calls == ["https://example.invalid/healthz", "https://example.invalid/v1/version"]

    def boom(url):
        raise RuntimeError("down")

    monkeypatch.setattr(mod, "_get", boom)
    assert mod.check("https://example.invalid")[0] is False


# ---------------------------------------------------------------- OIDC


def test_github_oidc_trust_is_exact(default_app):
    oidc = Template.from_stack(default_app.oidc)
    oidc.has_resource_properties(
        "AWS::IAM::OIDCProvider",
        {"Url": "https://token.actions.githubusercontent.com", "ClientIdList": ["sts.amazonaws.com"]},
    )
    (role,) = oidc.find_resources("AWS::IAM::Role").values()
    assert role["Properties"]["RoleName"] == "DriftSenseGithubDeploy"
    (st,) = role["Properties"]["AssumeRolePolicyDocument"]["Statement"]
    assert st["Action"] == "sts:AssumeRoleWithWebIdentity"
    assert st["Condition"] == {
        "StringEquals": {
            "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
            "token.actions.githubusercontent.com:sub": "repo:HUNT-001/DriftSense:ref:refs/heads/main",
        }
    }
    assert "StringLike" not in st["Condition"]


def test_github_deploy_role_can_only_assume_cdk_roles(default_app):
    sts = _statements(Template.from_stack(default_app.oidc))
    assert len(sts) == 1
    assert _actions(sts[0]) == ["sts:AssumeRole"]
    assert "role/cdk-*" in json.dumps(sts[0]["Resource"])
    role = Template.from_stack(default_app.oidc).find_resources("AWS::IAM::Role")
    assert not any(r["Properties"].get("ManagedPolicyArns") for r in role.values())


def test_github_oidc_can_reuse_existing_provider(make_app):
    arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
    built = make_app(github_oidc_provider_arn=arn)
    oidc = Template.from_stack(built.oidc)
    oidc.resource_count_is("AWS::IAM::OIDCProvider", 0)
    assert arn in json.dumps(oidc.to_json())


# ---------------------------------------------------------------- bench


def test_bench_absent_by_default(default_app):
    assert default_app.bench is None
    assert f"{PREFIX}Bench" not in {s.stack_name for s in default_app.stacks()}


def test_bench_large_arm_optional(make_app):
    built = make_app(cool_ami_id=TEST_AMI, bench="true", bench_run_id="r2", bench_large="true")
    instances = Template.from_stack(built.bench).find_resources("AWS::EC2::Instance")
    assert sorted(i["Properties"]["InstanceType"] for i in instances.values()) == [
        "c7i.4xlarge",
        "c8g.4xlarge",
        "c8g.4xlarge",
        "c8g.large",
    ]
    uds = [_flatten_user_data(i["Properties"]["UserData"]) for i in instances.values()]
    (large,) = [u for u in uds if 'ARM_LABEL="graviton-cool-large"' in u]
    assert "/bench/r2/graviton-cool-large/" in large and "/opt/cool/venvs/python_3.12" in large


def test_bench_instances(bench_app):
    bench = Template.from_stack(bench_app.bench)
    instances = bench.find_resources("AWS::EC2::Instance")
    assert len(instances) == 3
    types = sorted(i["Properties"]["InstanceType"] for i in instances.values())
    assert types == ["c7i.4xlarge", "c8g.4xlarge", "c8g.4xlarge"]
    for inst in instances.values():
        props = inst["Properties"]
        assert props["InstanceInitiatedShutdownBehavior"] == "terminate"
        assert "KeyName" not in props
    images = [json.dumps(i["Properties"]["ImageId"]) for i in instances.values()]
    assert sum(TEST_AMI in img for img in images) == 1
    assert sum("canonical/ubuntu/server/24.04" in json.dumps(bench.to_json()["Parameters"]) for _ in [0]) == 1


def test_bench_user_data(bench_app):
    bench = Template.from_stack(bench_app.bench)
    uds = {
        json.dumps(i["Properties"]["InstanceType"]) + _flatten_user_data(i["Properties"]["UserData"]): None
        for i in bench.find_resources("AWS::EC2::Instance").values()
    }
    texts = list(uds)
    for label in ("x86-pip", "graviton-pip", "graviton-cool"):
        (ud,) = [t for t in texts if f'ARM_LABEL="{label}"' in t]
        assert 'bash bench/cool/run_all.sh "$ARM_LABEL" "$DEST"' in ud
        assert f"/bench/test-run/{label}/" in ud
        assert "export PYTHON=/opt/driftsense/venv/bin/python" in ud
        assert f"export DRIFTSENSE_ENGINE=v2 ENGINE_GIT_SHA={V2_FREEZE_SHA}" in ud
        assert ud.index("run_all.sh") < ud.rindex("shutdown -h now")
        assert "shutdown -h now" in ud and "shutdown -h +360" in ud
        if label == "graviton-cool":
            assert "/opt/cool/venvs/python_3.12" in ud and "grep -i opencv" in ud
            assert "requirements-cool.lock" in ud
            _assert_never_pip_installs_opencv(ud)
        else:
            assert "requirements.lock" in ud and "/opt/cool" not in ud


def test_bench_role_writes_only_bench_prefix(bench_app):
    for st in _statements(Template.from_stack(bench_app.bench)):
        if "s3:PutObject" in _actions(st):
            assert "/bench/*" in json.dumps(st["Resource"])


def test_bench_requires_imdsv2_and_nag_clean(bench_app):
    for stack in bench_app.stacks():
        for name, lt in Template.from_stack(stack).find_resources("AWS::EC2::LaunchTemplate").items():
            assert lt["Properties"]["LaunchTemplateData"]["MetadataOptions"]["HttpTokens"] == "required", name
        errors = Annotations.from_stack(stack).find_error("*", Match.string_like_regexp("AwsSolutions-.*"))
        assert errors == [], [e.entry.data for e in errors]
    # 3 bench instances each get an IMDSv2 launch template
    assert len(Template.from_stack(bench_app.bench).find_resources("AWS::EC2::LaunchTemplate")) == 3
