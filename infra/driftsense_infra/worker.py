"""Worker stack: ASG (1-2) of c8g.large on the COOL 3.1 AMI, SSM-only, IMDSv2."""

from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from constructs import Construct

from . import userdata
from .app_bundle import WORKER_PATHS, app_asset
from .config import METRIC_NAMESPACE, Settings
from .nag import suppress

WORKER_INSTANCE_TYPE = "c8g.large"

# service/worker.py: receive with VisibilityTimeout=180 and a heartbeat thread that re-extends it to
# 180 s every 60 s while a job (including an agent episode blocked on human approval) runs. The queue
# default visibility (QueueStack, 900 s) only matters if a receive does not set one; both must exceed
# the heartbeat interval, which tests/infra asserts.
WORKER_HEARTBEAT_S = 60
WORKER_RECEIVE_VISIBILITY_S = 180
# systemd stop timeout: an agent episode can block on approval for approval_timeout_s (+60 s tool slack,
# Settings.effective_tool_timeout_s); give SIGTERM that plus 60 s before SIGKILL.
STOP_SLACK_S = 120


def bedrock_model_arn(stack: Stack, model_id: str, region: str | None = None) -> str:
    return f"arn:{stack.partition}:bedrock:{region or stack.region}::foundation-model/{model_id}"


def bedrock_profile_arn(stack: Stack, profile_id: str) -> str:
    return f"arn:{stack.partition}:bedrock:{stack.region}:{stack.account}:inference-profile/{profile_id}"


def worker_env(settings: Settings, *, table: str, queue_url: str, bucket: str, region: str) -> dict[str, str]:
    """The worker's systemd environment (also the service contract; see infra/README.md)."""
    env = {
        "JOBS_TABLE": table,
        "JOBS_QUEUE_URL": queue_url,
        "DATA_BUCKET": bucket,
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
        "DRIFTSENSE_ENGINE": settings.engine,
        "ENGINE_GIT_SHA": settings.engine_git_sha,
        "APPROVAL_TIMEOUT_S": str(settings.approval_timeout_s),
        "APPROVAL_POLL_S": "2",
        "TRACE_UPLOAD_S": "2",
        "JOB_TTL_DAYS": str(settings.job_ttl_days),
        "LOG_LEVEL": settings.log_level,
        # EMF on stdout only reaches journald (no CloudWatch agent); publish with PutMetricData as well.
        "METRICS_SINK": "both",
        "WORKER_LOG_GROUP": worker_log_group_name(),
    }
    if settings.bedrock:
        env["BEDROCK_MODEL_ID"] = settings.bedrock_inference_profile or settings.bedrock_model_id
    return env


def worker_log_group_name() -> str:
    return "/driftsense/worker"


class WorkerStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        settings: Settings,
        vpc: ec2.IVpc,
        worker_sg: ec2.ISecurityGroup,
        table: dynamodb.ITable,
        jobs_queue: sqs.IQueue,
        data_bucket: s3.IBucket,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        if not settings.cool_ami_id:
            raise ValueError("WorkerStack requires context cool_ami_id")

        self.log_group = logs.LogGroup(
            self,
            "WorkerLogs",
            log_group_name=worker_log_group_name(),
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ---------------------------------------------------------------- IAM
        self.role = iam.Role(
            self,
            "WorkerRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="DriftSense COOL worker (least privilege)",
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")],
        )
        # Exactly what service/worker.py uses (heartbeats via ChangeMessageVisibility).
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="JobsQueueConsume",
                actions=[
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                    "sqs:ChangeMessageVisibility",
                    "sqs:GetQueueAttributes",
                ],
                resources=[jobs_queue.queue_arn],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="JobsTable",
                actions=["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"],
                resources=[table.table_arn],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ResultsReadWrite",
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[data_bucket.arn_for_objects(p) for p in ("jobs/*", "traces/*")],
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="S3InputsRead",
                actions=["s3:GetObject"],
                resources=[data_bucket.arn_for_objects(p) for p in ("uploads/*", "demos/*")],
            )
        )
        # ListBucket lets a GetObject on a missing key return 404 (not 403): the worker falls back to the
        # synthetic demo pair when demos/<id>/ is absent.
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="S3ListDataBucket", actions=["s3:ListBucket"], resources=[data_bucket.bucket_arn]
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="MetricsDriftSenseNamespaceOnly",
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={"StringEquals": {"cloudwatch:namespace": METRIC_NAMESPACE}},
            )
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="OwnLogGroup",
                actions=["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
                resources=[self.log_group.log_group_arn],
            )
        )
        self.bedrock_model_arn = None
        if settings.bedrock:
            if settings.bedrock_inference_profile:
                # Cross-region inference profile (e.g. apac.amazon.nova-lite-v1:0): the caller needs the
                # profile ARN plus the same foundation model in whichever APAC region serves the call.
                self.bedrock_model_arn = bedrock_profile_arn(self, settings.bedrock_inference_profile)
                resources = [
                    self.bedrock_model_arn,
                    bedrock_model_arn(self, settings.bedrock_model_id, region="*"),
                ]
                conditions = {"StringLike": {"bedrock:InferenceProfileArn": self.bedrock_model_arn}}
                self.role.add_to_policy(
                    iam.PolicyStatement(
                        sid="BedrockOneProfile",
                        actions=["bedrock:InvokeModel"],
                        resources=[self.bedrock_model_arn],
                    )
                )
                self.role.add_to_policy(
                    iam.PolicyStatement(
                        sid="BedrockOneModelViaProfile",
                        actions=["bedrock:InvokeModel"],
                        resources=resources[1:],
                        conditions=conditions,
                    )
                )
            else:
                self.bedrock_model_arn = bedrock_model_arn(self, settings.bedrock_model_id)
                self.role.add_to_policy(
                    iam.PolicyStatement(
                        sid="BedrockOneModel",
                        actions=["bedrock:InvokeModel"],
                        resources=[self.bedrock_model_arn],
                    )
                )

        # ---------------------------------------------------------------- code + user-data
        self.app_asset, self.bundled_paths = app_asset(self, "WorkerApp", "worker_app", WORKER_PATHS)
        self.app_asset.grant_read(self.role)

        user_data = ec2.UserData.for_linux(shebang="#!/bin/bash")
        user_data.add_commands(
            userdata.PREAMBLE,
            userdata.ENSURE_SSM_AGENT,
            userdata.ENSURE_AWS_CLI,
            userdata.fetch_app(self.app_asset, self.region),
            userdata.COOL_VENV,
            userdata.ASSERT_COOL_OPENCV,
            userdata.assert_engine(settings.engine) if settings.engine != "fake" else "",
            userdata.worker_unit(
                worker_env(
                    settings,
                    table=table.table_name,
                    queue_url=jobs_queue.queue_url,
                    bucket=data_bucket.bucket_name,
                    region=self.region,
                ),
                stop_timeout_s=settings.approval_timeout_s + STOP_SLACK_S,
            ),
        )

        # ---------------------------------------------------------------- launch template + ASG
        self.launch_template = ec2.LaunchTemplate(
            self,
            "WorkerLaunchTemplate",
            machine_image=ec2.MachineImage.generic_linux({self.region: settings.cool_ami_id}),
            instance_type=ec2.InstanceType(WORKER_INSTANCE_TYPE),
            role=self.role,
            security_group=worker_sg,
            associate_public_ip_address=True,
            user_data=user_data,
            require_imdsv2=True,
            http_tokens=ec2.LaunchTemplateHttpTokens.REQUIRED,
            http_endpoint=True,
            http_put_response_hop_limit=2,
            detailed_monitoring=False,
            block_devices=[
                ec2.BlockDevice(
                    device_name=settings.root_device_name,
                    volume=ec2.BlockDeviceVolume.ebs(
                        30,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                        encrypted=True,
                        delete_on_termination=True,
                    ),
                )
            ],
        )

        self.asg = autoscaling.AutoScalingGroup(
            self,
            "WorkerAsg",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            launch_template=self.launch_template,
            min_capacity=1,
            max_capacity=2,
            group_metrics=[
                autoscaling.GroupMetrics(
                    autoscaling.GroupMetric.IN_SERVICE_INSTANCES, autoscaling.GroupMetric.DESIRED_CAPACITY
                )
            ],
            update_policy=autoscaling.UpdatePolicy.rolling_update(
                max_batch_size=1, min_instances_in_service=1
            ),
        )

        CfnOutput(self, "WorkerAsgName", value=self.asg.auto_scaling_group_name)
        self._suppressions()

    def _suppressions(self) -> None:
        suppress(
            self.role,
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "AmazonSSMManagedInstanceCore is the AWS-maintained minimum for Session Manager, the "
                    "only access path to the worker (no SSH, no inbound rules).",
                    "applies_to": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSSMManagedInstanceCore",
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Wildcards are limited to: object keys under fixed data-bucket prefixes (jobs/, traces/, "
                    "uploads/, demos/); the CDK asset bucket read grant (GetObject*/GetBucket*/List*) for the app "
                    "bundle; the worker's own log group streams; and cloudwatch:PutMetricData, which has no "
                    "resource-level permissions and is pinned to the DriftSense namespace by condition. Only with "
                    "-c bedrock_inference_profile: the ONE configured foundation model in any region (a "
                    "cross-region profile routes to several APAC regions), conditioned on that profile ARN.",
                    "applies_to": [
                        {"regex": "/^Resource::.*/(jobs|traces|uploads|demos)/\\*$/"},
                        {
                            "regex": "/^Resource::arn:(aws|<AWS::Partition>):s3:::cdk-[a-z0-9]+-assets-.*\\/\\*$/"
                        },
                        {"regex": "/^Resource::.*WorkerLogs.*:\\*$/"},
                        {
                            "regex": "/^Resource::arn:(aws|<AWS::Partition>):bedrock:\\*::foundation-model\\/[^*]+$/"
                        },
                        "Action::s3:GetObject*",
                        "Action::s3:GetBucket*",
                        "Action::s3:List*",
                        "Resource::*",
                    ],
                },
            ],
            apply_to_children=True,
        )
        suppress(
            self.asg,
            [
                {
                    "id": "AwsSolutions-AS3",
                    "reason": "Scaling-event notifications are not needed; the InService<1 CloudWatch alarm in "
                    "DriftSense-Observability pages the team via SNS instead.",
                },
            ],
            apply_to_children=True,
        )
