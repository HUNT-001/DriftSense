"""Api stack: HTTP API ($default stage, throttled) -> arm64 Python Lambda."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsii
from aws_cdk import Annotations, CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigateway as apigw_v1
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from aws_cdk.aws_apigatewayv2_integrations import HttpLambdaIntegration
from constructs import Construct

from .app_bundle import api_bundle_dir
from .nag import suppress

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MAX_UPLOAD_MB = "20"
# HTTP API ($context variables valid for API Gateway v2 access logs).
HTTP_API_ACCESS_LOG_FORMAT = json.dumps(
    {
        "requestId": "$context.requestId",
        "ip": "$context.identity.sourceIp",
        "requestTime": "$context.requestTime",
        "routeKey": "$context.routeKey",
        "status": "$context.status",
        "protocol": "$context.protocol",
        "responseLength": "$context.responseLength",
        "integrationLatency": "$context.integrationLatency",
        "integrationError": "$context.integrationErrorMessage",
    }
)


def _api_code(api_asset: str) -> tuple[lambda_.Code, str, str | None]:
    """Lambda code + handler for ``api_asset`` (auto | service | placeholder), plus an optional warning."""
    directory, handler, warning = api_bundle_dir(api_asset, ASSETS_DIR / "api_placeholder")
    return lambda_.Code.from_asset(str(directory)), handler, warning


@jsii.implements(apigwv2.IAccessLogSettings)
class _AccessLogs:
    """Python implementation of the IAccessLogSettings interface."""

    def __init__(self, destination: apigwv2.IAccessLogDestination, fmt: apigw_v1.AccessLogFormat) -> None:
        self._destination = destination
        self._format = fmt

    @property
    def destination(self) -> apigwv2.IAccessLogDestination:
        return self._destination

    @property
    def format(self) -> apigw_v1.AccessLogFormat:
        return self._format


class ApiStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        table: dynamodb.ITable,
        jobs_queue: sqs.IQueue,
        data_bucket: s3.IBucket,
        ui_origin: str,
        engine: str = "v2",
        engine_git_sha: str = "unknown",
        approval_timeout_s: int = 900,
        job_ttl_days: int = 7,
        log_level: str = "INFO",
        api_asset: str = "placeholder",
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        code, handler, warning = _api_code(api_asset)
        self.handler = handler
        if warning:
            Annotations.of(self).add_warning_v2("driftsense:api-placeholder", warning)
            print(f"WARNING: {warning}", file=sys.stderr)

        fn_logs = logs.LogGroup(
            self, "ApiFnLogs", retention=logs.RetentionDays.TWO_WEEKS, removal_policy=RemovalPolicy.DESTROY
        )

        # Hand-rolled role: avoids the AWSLambdaBasicExecutionRole managed policy (logs:* on every group).
        role = iam.Role(
            self,
            "ApiFnRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="DriftSense API Lambda (least privilege)",
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[fn_logs.log_group_arn],
            )
        )
        # Exactly what service/api.py uses (TransactWriteItems = PutItem + ConditionCheckItem per item).
        role.add_to_policy(
            iam.PolicyStatement(
                sid="JobsTable",
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:UpdateItem",
                    "dynamodb:ConditionCheckItem",
                ],
                resources=[table.table_arn],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="JobsQueueSend", actions=["sqs:SendMessage"], resources=[jobs_queue.queue_arn]
            )
        )
        # PutObject is what the presigned upload URL is signed with.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:PutObject"], resources=[data_bucket.arn_for_objects("uploads/*")]
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[data_bucket.arn_for_objects(p) for p in ("jobs/*", "traces/*", "demos/*")],
            )
        )
        # 404 (not 403) for a missing artifact/trace object.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ListDataBucket", actions=["s3:ListBucket"], resources=[data_bucket.bucket_arn]
            )
        )

        self.function = lambda_.Function(
            self,
            "ApiFn",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler=handler,
            code=code,
            memory_size=512,
            timeout=Duration.seconds(29),
            role=role,
            log_group=fn_logs,
            environment={
                "JOBS_TABLE": table.table_name,
                "JOBS_QUEUE_URL": jobs_queue.queue_url,
                "DATA_BUCKET": data_bucket.bucket_name,
                "UI_ORIGIN": ui_origin,
                "MAX_UPLOAD_MB": MAX_UPLOAD_MB,
                "DRIFTSENSE_ENGINE": engine,
                "ENGINE_GIT_SHA": engine_git_sha,
                "JOB_TTL_DAYS": str(job_ttl_days),
                "APPROVAL_TIMEOUT_S": str(approval_timeout_s),
                "LOG_LEVEL": log_level,
                # AWS_REGION is reserved and injected by the Lambda runtime.
            },
        )

        self.http_api = apigwv2.HttpApi(
            self,
            "HttpApi",
            api_name="driftsense-api",
            create_default_stage=False,
            cors_preflight=apigwv2.CorsPreflightOptions(
                allow_origins=[ui_origin],
                allow_methods=[
                    apigwv2.CorsHttpMethod.GET,
                    apigwv2.CorsHttpMethod.POST,
                    apigwv2.CorsHttpMethod.OPTIONS,
                ],
                allow_headers=["content-type"],
                max_age=Duration.hours(1),
            ),
        )

        access_logs = logs.LogGroup(
            self,
            "ApiAccessLogs",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )
        self.stage = apigwv2.HttpStage(
            self,
            "DefaultStage",
            http_api=self.http_api,
            stage_name="$default",
            auto_deploy=True,
            throttle=apigwv2.ThrottleSettings(rate_limit=20, burst_limit=40),
            access_log_settings=_AccessLogs(
                apigwv2.LogGroupLogDestination(access_logs),
                apigw_v1.AccessLogFormat.custom(HTTP_API_ACCESS_LOG_FORMAT),
            ),
        )

        integration = HttpLambdaIntegration("ApiFnIntegration", self.function)
        routes = [
            *self.http_api.add_routes(
                path="/healthz", methods=[apigwv2.HttpMethod.GET], integration=integration
            ),
            *self.http_api.add_routes(
                path="/{proxy+}", methods=[apigwv2.HttpMethod.ANY], integration=integration
            ),
        ]

        self.api_url = self.stage.url
        CfnOutput(self, "ApiUrl", value=self.api_url)

        suppress(
            role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Object-level S3 access is scoped to fixed key prefixes (uploads/ write; jobs/, traces/, "
                    "demos/ read) and logs to this function's own log group; the trailing * is the object key.",
                    "applies_to": [
                        {"regex": "/^Resource::.*/(uploads|jobs|traces|demos)/\\*$/"},
                        {"regex": "/^Resource::.*ApiFnLogs.*:\\*$/"},
                    ],
                }
            ],
            apply_to_children=True,
        )
        suppress(
            self.function,
            [
                {
                    "id": "AwsSolutions-L1",
                    "reason": "Pinned to Python 3.12 on purpose: the worker runs the COOL 3.1 python_3.12 venv and the "
                    "API shares the same service/ code and contracts, so both sides stay on one interpreter.",
                }
            ],
        )
        for route in routes:
            suppress(
                route,
                [
                    {
                        "id": "AwsSolutions-APIG4",
                        "reason": "Public demo API for judges: no user accounts by design. Abuse is bounded by stage "
                        "throttling (20 rps / burst 40), 20 MB presigned uploads and CORS restricted to the UI origin.",
                    }
                ],
            )
