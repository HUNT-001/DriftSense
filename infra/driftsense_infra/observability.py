"""Observability stack: dashboard, 30-minute canary, alarms -> SNS."""

from __future__ import annotations

from pathlib import Path

from aws_cdk import Duration, RemovalPolicy, Stack
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sqs as sqs
from constructs import Construct

from .config import METRIC_NAMESPACE
from .nag import suppress

CANARY_DIR = Path(__file__).resolve().parent.parent / "assets" / "canary"
CANARY_RATE = Duration.minutes(30)
FIVE_MINUTES = Duration.minutes(5)


def _search(
    expression: str, label: str, stat: str = "Average", period: Duration = FIVE_MINUTES
) -> cw.MathExpression:
    seconds = int(period.to_seconds())
    return cw.MathExpression(
        expression=f"SEARCH('{expression}', '{stat}', {seconds})",
        label=label,
        period=period,
        using_metrics={},
    )


class ObservabilityStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        http_api: apigwv2.HttpApi,
        api_url: str,
        api_function: lambda_.IFunction,
        jobs_queue: sqs.IQueue,
        dlq: sqs.IQueue,
        alarm_topic: sns.ITopic,
        worker_asg: autoscaling.AutoScalingGroup | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        alarm_action = cw_actions.SnsAction(alarm_topic)

        # ------------------------------------------------------------ canary
        canary_logs = logs.LogGroup(
            self, "CanaryLogs", retention=logs.RetentionDays.TWO_WEEKS, removal_policy=RemovalPolicy.DESTROY
        )
        canary_role = iam.Role(
            self,
            "CanaryRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="DriftSense canary (logs + one metric namespace)",
        )
        canary_role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogStream", "logs:PutLogEvents"], resources=[canary_logs.log_group_arn]
            )
        )
        canary_role.add_to_policy(
            iam.PolicyStatement(
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={"StringEquals": {"cloudwatch:namespace": METRIC_NAMESPACE}},
            )
        )
        self.canary = lambda_.Function(
            self,
            "CanaryFn",
            runtime=lambda_.Runtime.PYTHON_3_12,
            architecture=lambda_.Architecture.ARM_64,
            handler="canary.handler",
            code=lambda_.Code.from_asset(str(CANARY_DIR)),
            memory_size=128,
            timeout=Duration.seconds(30),
            role=canary_role,
            log_group=canary_logs,
            environment={"API_URL": api_url, "METRIC_NAMESPACE": METRIC_NAMESPACE},
        )
        self.canary_rule = events.Rule(
            self,
            "CanarySchedule",
            description="DriftSense canary: GET /healthz and /v1/version every 30 minutes",
            schedule=events.Schedule.rate(CANARY_RATE),
            targets=[targets.LambdaFunction(self.canary, retry_attempts=0)],
        )

        canary_ok = cw.Metric(
            namespace=METRIC_NAMESPACE, metric_name="CanaryOk", statistic="Minimum", period=CANARY_RATE
        )
        self.canary_alarm = cw.Alarm(
            self,
            "CanaryFailing",
            alarm_description="DriftSense API canary failed (or did not report) for 2 consecutive runs.",
            metric=canary_ok,
            threshold=1,
            comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=2,
            treat_missing_data=cw.TreatMissingData.BREACHING,
        )
        self.canary_alarm.add_alarm_action(alarm_action)

        # ------------------------------------------------------------ worker alarm
        in_service = None
        if worker_asg is not None:
            in_service = cw.Metric(
                namespace="AWS/AutoScaling",
                metric_name="GroupInServiceInstances",
                dimensions_map={"AutoScalingGroupName": worker_asg.auto_scaling_group_name},
                statistic="Minimum",
                period=Duration.minutes(5),
            )
            self.worker_alarm = cw.Alarm(
                self,
                "WorkerNotInService",
                alarm_description="DriftSense worker ASG has no InService instance.",
                metric=in_service,
                threshold=1,
                comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
                evaluation_periods=2,
                treat_missing_data=cw.TreatMissingData.BREACHING,
            )
            self.worker_alarm.add_alarm_action(alarm_action)

        # ------------------------------------------------------------ dashboard
        five = Duration.minutes(5)
        dash = cw.Dashboard(
            self, "Dashboard", dashboard_name="DriftSense", default_interval=Duration.hours(24)
        )
        dash.add_widgets(
            cw.GraphWidget(
                title="API 4xx / 5xx",
                left=[
                    http_api.metric_client_error(period=five, statistic="Sum"),
                    http_api.metric_server_error(period=five, statistic="Sum"),
                ],
                width=8,
            ),
            cw.GraphWidget(
                title="API latency (ms)",
                left=[
                    http_api.metric_latency(period=five, statistic="p50"),
                    http_api.metric_latency(period=five, statistic="p90"),
                ],
                width=8,
            ),
            cw.GraphWidget(
                title="API Lambda errors",
                left=[api_function.metric_errors(period=five, statistic="Sum")],
                right=[api_function.metric_invocations(period=five, statistic="Sum")],
                width=8,
            ),
        )
        dash.add_widgets(
            cw.GraphWidget(
                title="Jobs queue depth / age",
                left=[
                    jobs_queue.metric_approximate_number_of_messages_visible(period=five, statistic="Maximum")
                ],
                right=[jobs_queue.metric_approximate_age_of_oldest_message(period=five, statistic="Maximum")],
                width=8,
            ),
            cw.GraphWidget(
                title="DLQ depth",
                left=[dlq.metric_approximate_number_of_messages_visible(period=five, statistic="Maximum")],
                width=8,
            ),
            cw.GraphWidget(
                title="Worker ASG InService" if in_service else "Worker ASG (not deployed: no cool_ami_id)",
                left=[in_service] if in_service else [],
                width=8,
            ),
        )
        dash.add_widgets(
            cw.GraphWidget(
                title="Job latency (ms)",
                left=[
                    _search(
                        f'Namespace="{METRIC_NAMESPACE}" MetricName="JobLatencyMs"', "JobLatencyMs p50", "p50"
                    ),
                    _search(
                        f'Namespace="{METRIC_NAMESPACE}" MetricName="JobLatencyMs"', "JobLatencyMs p90", "p90"
                    ),
                ],
                width=8,
            ),
            cw.GraphWidget(
                title="Jobs succeeded / failed",
                left=[
                    _search(
                        f'Namespace="{METRIC_NAMESPACE}" MetricName="JobsSucceeded"', "JobsSucceeded", "Sum"
                    ),
                    _search(f'Namespace="{METRIC_NAMESPACE}" MetricName="JobsFailed"', "JobsFailed", "Sum"),
                ],
                width=8,
            ),
            cw.GraphWidget(
                title="Stage time (ms) by Stage",
                left=[_search(f'{{{METRIC_NAMESPACE},Stage}} MetricName="StageMs"', "StageMs", "Average")],
                width=8,
            ),
        )
        dash.add_widgets(
            cw.AlarmStatusWidget(
                title="Alarms",
                alarms=[self.canary_alarm] + ([self.worker_alarm] if worker_asg is not None else []),
                width=24,
            )
        )

        suppress(
            canary_role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "cloudwatch:PutMetricData has no resource-level permissions (pinned to the DriftSense "
                    "namespace by condition); log access is limited to the canary's own log group streams.",
                    "applies_to": ["Resource::*", {"regex": "/^Resource::.*CanaryLogs.*:\\*$/"}],
                }
            ],
            apply_to_children=True,
        )
        suppress(
            self.canary,
            [
                {
                    "id": "AwsSolutions-L1",
                    "reason": "Python 3.12 kept identical to the API Lambda and the COOL python_3.12 worker runtime.",
                }
            ],
        )
