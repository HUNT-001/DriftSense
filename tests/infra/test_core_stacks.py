"""Assertions on the core (B1) stacks: network, storage, queue, api, budget."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("aws_cdk")

from aws_cdk.assertions import Annotations, Match, Template  # noqa: E402

PREFIX = "DriftSense-"
V2_FREEZE_SHA = "f9395b84dd654ff1d537f6817c258a0f2b9373de"


def _resources(template: Template, cfn_type: str) -> dict:
    return template.find_resources(cfn_type)


def _all_resources(templates: dict, cfn_type: str) -> dict:
    found = {}
    for name, template in templates.items():
        for logical_id, res in _resources(template, cfn_type).items():
            found[f"{name}/{logical_id}"] = res
    return found


# ---------------------------------------------------------------- network


def test_no_nat_gateways_anywhere(templates):
    assert _all_resources(templates, "AWS::EC2::NatGateway") == {}


def test_vpc_has_only_public_subnets_and_gateway_endpoints(templates):
    net = templates[f"{PREFIX}Network"]
    subnets = _resources(net, "AWS::EC2::Subnet")
    assert len(subnets) == 3  # ap-south-1a/b/c, explicit (no AZ lookup)
    assert all(s["Properties"]["MapPublicIpOnLaunch"] is True for s in subnets.values())
    endpoints = _resources(net, "AWS::EC2::VPCEndpoint")
    types = sorted(e["Properties"].get("VpcEndpointType", "Gateway") for e in endpoints.values())
    assert types == ["Gateway", "Gateway"]


def test_flow_logs_to_cloudwatch_with_7_day_retention(templates):
    net = templates[f"{PREFIX}Network"]
    net.has_resource_properties("AWS::EC2::FlowLog", {"LogDestinationType": "cloud-watch-logs"})
    net.has_resource_properties("AWS::Logs::LogGroup", {"RetentionInDays": 7})


def test_interface_endpoints_absent_by_default(templates):
    endpoints = _resources(templates[f"{PREFIX}Network"], "AWS::EC2::VPCEndpoint")
    assert not [e for e in endpoints.values() if e["Properties"].get("VpcEndpointType") == "Interface"]


def test_interface_endpoints_present_with_flag(make_app):
    built = make_app(interface_endpoints="true")
    net = Template.from_stack(built.network)
    endpoints = _resources(net, "AWS::EC2::VPCEndpoint")
    interface = [e for e in endpoints.values() if e["Properties"].get("VpcEndpointType") == "Interface"]
    assert len(interface) == 4
    services = " ".join(str(e["Properties"]["ServiceName"]) for e in interface)
    for svc in ("sqs", "ssm", "ssmmessages", "ec2messages"):
        assert f".{svc}" in services


def test_worker_sg_has_no_ingress(default_app, templates):
    net = templates[f"{PREFIX}Network"]
    sg_id = default_app.network.resolve(default_app.network.worker_sg.security_group_id)["Fn::GetAtt"][0]
    sg = net.to_json()["Resources"][sg_id]
    assert not sg["Properties"].get("SecurityGroupIngress")
    ingress = _resources(net, "AWS::EC2::SecurityGroupIngress")
    for rule in ingress.values():
        assert rule["Properties"]["GroupId"] != {"Fn::GetAtt": [sg_id, "GroupId"]}
    egress = sg["Properties"]["SecurityGroupEgress"]
    assert any(r.get("CidrIp") == "0.0.0.0/0" and r.get("IpProtocol") == "-1" for r in egress)


# ---------------------------------------------------------------- storage

FULL_BPA = {
    "BlockPublicAcls": True,
    "BlockPublicPolicy": True,
    "IgnorePublicAcls": True,
    "RestrictPublicBuckets": True,
}


def test_every_bucket_is_private_and_encrypted(templates):
    buckets = _all_resources(templates, "AWS::S3::Bucket")
    assert len(buckets) >= 2
    for name, bucket in buckets.items():
        props = bucket["Properties"]
        assert props["PublicAccessBlockConfiguration"] == FULL_BPA, name
        sse = props["BucketEncryption"]["ServerSideEncryptionConfiguration"][0]
        assert sse["ServerSideEncryptionByDefault"]["SSEAlgorithm"] == "AES256", name
        assert props["OwnershipControls"]["Rules"][0]["ObjectOwnership"] == "BucketOwnerEnforced", name


def test_every_bucket_enforces_tls(templates):
    for template in templates.values():
        for policy in _resources(template, "AWS::S3::BucketPolicy").values():
            statements = policy["Properties"]["PolicyDocument"]["Statement"]
            assert any(
                s["Effect"] == "Deny"
                and s.get("Condition", {}).get("Bool", {}).get("aws:SecureTransport") == "false"
                for s in statements
            )


def _data_bucket(default_app, templates) -> dict:
    stack = default_app.storage
    logical_id = stack.get_logical_id(stack.data_bucket.node.default_child)
    return templates[f"{PREFIX}Storage"].to_json()["Resources"][logical_id]


def test_data_bucket_lifecycle_7_days_not_on_demos(default_app, templates):
    rules = _data_bucket(default_app, templates)["Properties"]["LifecycleConfiguration"]["Rules"]
    prefixes = {r["Prefix"]: r for r in rules}
    assert set(prefixes) == {"uploads/", "jobs/", "traces/"}
    assert all(r["ExpirationInDays"] == 7 and r["Status"] == "Enabled" for r in rules)
    assert not any(r.get("Prefix", "").startswith("demos") for r in rules)
    assert all("Prefix" in r for r in rules), "a prefix-less rule would also expire demos/"


def test_data_bucket_cors_only_from_cloudfront(default_app, templates):
    cors = _data_bucket(default_app, templates)["Properties"]["CorsConfiguration"]["CorsRules"]
    assert len(cors) == 1
    assert sorted(cors[0]["AllowedMethods"]) == ["GET", "PUT"]
    assert {"content-type", "content-length"} <= set(cors[0]["AllowedHeaders"])
    origins = cors[0]["AllowedOrigins"]
    assert len(origins) == 1
    assert "DomainName" in str(origins[0]) and "https://" in str(origins[0])


def test_cloudfront_uses_oac_and_spa_fallback(templates):
    storage = templates[f"{PREFIX}Storage"]
    storage.resource_count_is("AWS::CloudFront::OriginAccessControl", 1)
    storage.has_resource_properties(
        "AWS::CloudFront::Distribution",
        {
            "DistributionConfig": Match.object_like(
                {
                    "DefaultRootObject": "index.html",
                    "DefaultCacheBehavior": Match.object_like({"ViewerProtocolPolicy": "redirect-to-https"}),
                    "CustomErrorResponses": Match.array_with(
                        [
                            Match.object_like(
                                {"ErrorCode": 403, "ResponseCode": 200, "ResponsePagePath": "/index.html"}
                            ),
                            Match.object_like(
                                {"ErrorCode": 404, "ResponseCode": 200, "ResponsePagePath": "/index.html"}
                            ),
                        ]
                    ),
                    "Origins": [Match.object_like({"OriginAccessControlId": Match.any_value()})],
                }
            )
        },
    )


def test_jobs_table_on_demand_ttl_pitr(templates):
    templates[f"{PREFIX}Storage"].has_resource_properties(
        "AWS::DynamoDB::Table",
        {
            "BillingMode": "PAY_PER_REQUEST",
            "KeySchema": [{"AttributeName": "job_id", "KeyType": "HASH"}],
            "TimeToLiveSpecification": {"AttributeName": "expires_at", "Enabled": True},
            "PointInTimeRecoverySpecification": {"PointInTimeRecoveryEnabled": True},
        },
    )


# ---------------------------------------------------------------- queue


def test_jobs_queue_redrive_and_settings(templates):
    queue = templates[f"{PREFIX}Queue"]
    queue.has_resource_properties(
        "AWS::SQS::Queue",
        {
            "RedrivePolicy": Match.object_like({"maxReceiveCount": 3}),
            "VisibilityTimeout": 900,
            "ReceiveMessageWaitTimeSeconds": 20,
            "SqsManagedSseEnabled": True,
        },
    )
    queue.has_resource_properties("AWS::SQS::Queue", {"MessageRetentionPeriod": 14 * 24 * 3600})


def test_dlq_alarm_goes_to_topic(templates):
    templates[f"{PREFIX}Queue"].has_resource_properties(
        "AWS::CloudWatch::Alarm",
        {
            "MetricName": "ApproximateNumberOfMessagesVisible",
            "Threshold": 0,
            "ComparisonOperator": "GreaterThanThreshold",
            "AlarmActions": [Match.any_value()],
        },
    )


# ---------------------------------------------------------------- api


def test_every_lambda_is_arm64(templates):
    functions = _all_resources(templates, "AWS::Lambda::Function")
    assert functions
    # Includes CDK-internal custom-resource handlers (forced to arm64 by the Arm64Lambdas aspect).
    for name, fn in functions.items():
        assert fn["Properties"].get("Architectures") == ["arm64"], name


def test_api_lambda_contract(templates):
    templates[f"{PREFIX}Api"].has_resource_properties(
        "AWS::Lambda::Function",
        {
            "Runtime": "python3.12",
            "Architectures": ["arm64"],
            "MemorySize": 512,
            "Timeout": 29,
            "Handler": "handler.handler",
            "Environment": {
                "Variables": {
                    "JOBS_TABLE": Match.any_value(),
                    "JOBS_QUEUE_URL": Match.any_value(),
                    "DATA_BUCKET": Match.any_value(),
                    "UI_ORIGIN": Match.any_value(),
                    "MAX_UPLOAD_MB": "20",
                    "DRIFTSENSE_ENGINE": "v2",
                    "ENGINE_GIT_SHA": V2_FREEZE_SHA,
                    "JOB_TTL_DAYS": "7",
                    "APPROVAL_TIMEOUT_S": "900",
                    "LOG_LEVEL": "INFO",
                }
            },
        },
    )
    templates[f"{PREFIX}Api"].has_resource_properties("AWS::Logs::LogGroup", {"RetentionInDays": 14})


def test_api_stage_throttled_default_stage(templates):
    api = templates[f"{PREFIX}Api"]
    api.resource_count_is("AWS::ApiGatewayV2::Stage", 1)
    api.has_resource_properties(
        "AWS::ApiGatewayV2::Stage",
        {
            "StageName": "$default",
            "AutoDeploy": True,
            "DefaultRouteSettings": Match.object_like(
                {"ThrottlingRateLimit": 20, "ThrottlingBurstLimit": 40}
            ),
            "AccessLogSettings": Match.object_like({"DestinationArn": Match.any_value()}),
        },
    )


def test_api_cors_restricted_to_cloudfront(templates):
    api = templates[f"{PREFIX}Api"]
    apis = _resources(api, "AWS::ApiGatewayV2::Api")
    (props,) = [a["Properties"] for a in apis.values()]
    cors = props["CorsConfiguration"]
    assert len(cors["AllowOrigins"]) == 1 and "*" not in str(cors["AllowOrigins"])
    assert sorted(cors["AllowMethods"]) == ["GET", "OPTIONS", "POST"]
    assert cors["AllowHeaders"] == ["content-type"]


def test_api_routes(templates):
    routes = _resources(templates[f"{PREFIX}Api"], "AWS::ApiGatewayV2::Route")
    keys = sorted(r["Properties"]["RouteKey"] for r in routes.values())
    assert keys == ["ANY /{proxy+}", "GET /healthz"]


def _api_policy_statements(templates) -> list[dict]:
    statements = []
    for policy in _resources(templates[f"{PREFIX}Api"], "AWS::IAM::Policy").values():
        statements.extend(policy["Properties"]["PolicyDocument"]["Statement"])
    return statements


def test_api_role_least_privilege(templates):
    statements = _api_policy_statements(templates)
    for st in statements:
        assert st["Resource"] != "*", st
    s3 = {
        (a if isinstance(a, str) else a[0]): st["Resource"]
        for st in statements
        for a in ([st["Action"]] if isinstance(st["Action"], str) else [st["Action"]])
        if str(a).startswith("s3:") or (isinstance(a, list) and str(a[0]).startswith("s3:"))
    }
    assert set(s3) == {"s3:PutObject", "s3:GetObject", "s3:ListBucket"}
    assert not str(s3["s3:ListBucket"]).endswith("*")
    assert "/uploads/*" in str(s3["s3:PutObject"])
    get = str(s3["s3:GetObject"])
    for prefix in ("/jobs/*", "/traces/*", "/demos/*"):
        assert prefix in get
    assert "/uploads/*" not in get
    assert not any("logs:" in str(st["Action"]) and st["Resource"] == "*" for st in statements)


def test_api_role_table_and_queue_actions(templates):
    by_sid = {st.get("Sid"): st for st in _api_policy_statements(templates)}
    assert sorted(by_sid["JobsTable"]["Action"]) == [
        "dynamodb:ConditionCheckItem",
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
    ]
    assert by_sid["JobsQueueSend"]["Action"] == "sqs:SendMessage"


def test_api_role_has_no_managed_policies(templates):
    for role in _resources(templates[f"{PREFIX}Api"], "AWS::IAM::Role").values():
        assert not role["Properties"].get("ManagedPolicyArns")


def test_api_asset_service_uses_bundle_and_service_handler(make_app, monkeypatch, tmp_path):
    from driftsense_infra import app_bundle

    (tmp_path / "service").mkdir()
    (tmp_path / "service" / "api.py").write_text("handler = None\n")
    calls = []
    monkeypatch.setattr(app_bundle, "build_api_bundle", lambda **kw: calls.append(kw) or tmp_path)
    built = make_app(api_asset="service")
    assert calls, "api_asset=service must build the real bundle"
    Template.from_stack(built.api).has_resource_properties(
        "AWS::Lambda::Function", {"Handler": "service.api.handler", "Architectures": ["arm64"]}
    )


def test_api_asset_auto_falls_back_to_placeholder_with_warning(make_app, monkeypatch):
    from driftsense_infra import app_bundle

    monkeypatch.setattr(app_bundle, "service_available", lambda repo_root=None: False)
    built = make_app(api_asset="auto")
    Template.from_stack(built.api).has_resource_properties(
        "AWS::Lambda::Function", {"Handler": "handler.handler"}
    )
    Annotations.from_stack(built.api).has_warning("*", Match.string_like_regexp("PLACEHOLDER"))


def test_invalid_api_asset_rejected(make_app):
    with pytest.raises(ValueError, match="api_asset"):
        make_app(api_asset="docker")


# ---------------------------------------------------------------- budget


def test_budget_without_email_has_no_notifications_and_warns(default_app, templates):
    budget = templates[f"{PREFIX}Budget"]
    (res,) = _resources(budget, "AWS::Budgets::Budget").values()
    assert res["Properties"]["Budget"]["BudgetLimit"] == {"Amount": 100, "Unit": "USD"}
    assert res["Properties"]["Budget"]["BudgetType"] == "COST"
    assert "NotificationsWithSubscribers" not in res["Properties"]
    Annotations.from_stack(default_app.budget).has_warning("*", Match.string_like_regexp("alert_email"))


def test_budget_with_email_has_three_notifications(make_app):
    built = make_app(alert_email="ops@example.com", budget_usd="90")
    (res,) = Template.from_stack(built.budget).find_resources("AWS::Budgets::Budget").values()
    assert res["Properties"]["Budget"]["BudgetLimit"]["Amount"] == 90
    notes = [n["Notification"] for n in res["Properties"]["NotificationsWithSubscribers"]]
    got = sorted((n["NotificationType"], n["ThresholdType"], n["Threshold"]) for n in notes)
    assert got == [
        ("ACTUAL", "ABSOLUTE_VALUE", 40),
        ("ACTUAL", "ABSOLUTE_VALUE", 80),
        ("FORECASTED", "PERCENTAGE", 100),
    ]
    subscribers = [s for n in res["Properties"]["NotificationsWithSubscribers"] for s in n["Subscribers"]]
    assert all(s == {"Address": "ops@example.com", "SubscriptionType": "EMAIL"} for s in subscribers)
    # alert email also subscribes to the alarm topic
    Template.from_stack(built.queue).has_resource_properties(
        "AWS::SNS::Subscription", {"Protocol": "email", "Endpoint": "ops@example.com"}
    )


# ---------------------------------------------------------------- cross-cutting


def test_cdk_nag_clean_synth(default_app):
    """cdk-nag 3 runs as a validation plugin: synth raises on any unacknowledged finding."""
    assembly = default_app.app.synth()
    assert {s.stack_name for s in assembly.stacks} >= {
        f"{PREFIX}Api",
        f"{PREFIX}Worker",
        f"{PREFIX}GithubOidc",
    }


def test_cdk_nag_clean_synth_with_interface_endpoints(make_app):
    make_app(interface_endpoints="true").app.synth()


def test_cdk_nag_blocks_unjustified_findings(make_app):
    """Guard against a no-op nag setup: an unsuppressed non-compliant resource must fail synth."""
    import aws_cdk as cdk
    from aws_cdk import aws_sqs as sqs

    built = make_app()
    sqs.Queue(built.queue, "NagCanaryUnencrypted", encryption=sqs.QueueEncryption.UNENCRYPTED)
    with pytest.raises(Exception, match="(?i)validation"):
        built.app.synth()
    assert isinstance(built.queue, cdk.Stack)


def test_every_launch_template_requires_imdsv2(default_app):
    """Guard: every EC2 launch template in every stack must require IMDSv2 tokens."""
    for stack in default_app.stacks():
        for name, lt in Template.from_stack(stack).find_resources("AWS::EC2::LaunchTemplate").items():
            md = lt["Properties"]["LaunchTemplateData"].get("MetadataOptions", {})
            assert md.get("HttpTokens") == "required", f"{stack.stack_name}/{name}"


def test_everything_tagged(templates):
    for name, template in templates.items():
        for logical_id, res in _resources(template, "AWS::SQS::Queue").items():
            tags = {t["Key"]: t["Value"] for t in res["Properties"].get("Tags", [])}
            assert tags.get("project") == "driftsense" and tags.get("owner") == "B", f"{name}/{logical_id}"
        for logical_id, res in _resources(template, "AWS::S3::Bucket").items():
            tags = {t["Key"]: t["Value"] for t in res["Properties"].get("Tags", [])}
            assert tags.get("project") == "driftsense" and tags.get("owner") == "B", f"{name}/{logical_id}"


def test_synth_with_concrete_account_needs_no_lookups(make_app, monkeypatch):
    """`cdk synth` with CDK_DEFAULT_ACCOUNT set must not need AWS calls (no AZ / AMI lookups)."""
    monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "123456789012")
    built = make_app(cool_ami_id="ami-0123456789abcdef0", bench="true", bench_run_id="acct")
    assembly = built.app.synth()
    manifest = json.loads((Path(assembly.directory) / "manifest.json").read_text(encoding="utf-8"))
    assert not manifest.get("missing"), manifest.get("missing")
    assert built.network.account == "123456789012"


def test_stacks_target_ap_south_1(default_app):
    for stack in default_app.stacks():
        assert stack.region == "ap-south-1"
        assert stack.stack_name.startswith(PREFIX)
