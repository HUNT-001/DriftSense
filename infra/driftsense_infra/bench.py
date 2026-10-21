"""Ephemeral COOL benchmark stack (only with -c bench=true).

Self-terminating instances run, from the app bundle root with the arm's venv active,
``bash bench/cool/run_all.sh <arm> s3://<data>/bench/<run_id>/<arm>/`` (the bench/ contract:
``run_all.sh`` uploads its own outputs to that URI; user-data adds the cloud-init log):

* ``x86-pip``             c7i.4xlarge, Ubuntu 24.04 amd64, OpenCV from PyPI (requirements.lock)
* ``graviton-pip``        c8g.4xlarge, Ubuntu 24.04 arm64, OpenCV from PyPI (requirements.lock)
* ``graviton-cool``       c8g.4xlarge, COOL 3.1 AMI, OpenCV from /opt/cool (requirements-cool.lock, never pip opencv)
* ``graviton-cool-large`` c8g.large COOL (production size, cost claim), only with ``-c bench_large=true``
"""

from __future__ import annotations

from dataclasses import dataclass

from aws_cdk import Acknowledgment, Annotations, Stack, Validations
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct

from . import userdata
from .app_bundle import BENCH_PATHS, app_asset
from .config import Settings
from .nag import suppress

UBUNTU_2404_SSM = "/aws/service/canonical/ubuntu/server/24.04/stable/current/{arch}/hvm/ebs-gp3/ami-id"


@dataclass(frozen=True)
class BenchArm:
    label: str
    instance_type: str
    cool: bool
    ubuntu_arch: str | None = None  # amd64 / arm64 for the canonical Ubuntu image


ARMS: tuple[BenchArm, ...] = (
    BenchArm("x86-pip", "c7i.4xlarge", cool=False, ubuntu_arch="amd64"),
    BenchArm("graviton-pip", "c8g.4xlarge", cool=False, ubuntu_arch="arm64"),
    BenchArm("graviton-cool", "c8g.4xlarge", cool=True),
)
LARGE_ARM = BenchArm("graviton-cool-large", "c8g.large", cool=True)
BENCH_OUT_ROOT = "/opt/driftsense/bench-out"


def bench_arms(include_large: bool) -> tuple[BenchArm, ...]:
    return ARMS + ((LARGE_ARM,) if include_large else ())


class BenchStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        settings: Settings,
        vpc: ec2.IVpc,
        worker_sg: ec2.ISecurityGroup,
        data_bucket: s3.IBucket,
        run_id: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        if not settings.cool_ami_id:
            raise ValueError("BenchStack requires context cool_ami_id")

        self.role = iam.Role(
            self,
            "BenchRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="DriftSense bench instances: SSM + write bench/ results",
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")],
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="BenchResultsWrite",
                actions=["s3:PutObject"],
                resources=[data_bucket.arn_for_objects("bench/*")],
            )
        )
        self.app_asset, self.bundled_paths = app_asset(self, "BenchApp", "bench_app", BENCH_PATHS)
        if "bench/" not in self.bundled_paths:
            Annotations.of(self).add_warning_v2(
                "driftsense:bench-missing",
                "bench/ is not in this checkout: bench instances will fail at run_all.sh (and still self-terminate).",
            )
        self.app_asset.grant_read(self.role)

        self.instances: dict[str, ec2.Instance] = {}
        for arm in bench_arms(settings.bench_large):
            if arm.cool:
                image = ec2.MachineImage.generic_linux({self.region: settings.cool_ami_id})
            else:
                image = ec2.MachineImage.from_ssm_parameter(
                    UBUNTU_2404_SSM.format(arch=arm.ubuntu_arch), os=ec2.OperatingSystemType.LINUX
                )
            dest = f"s3://{data_bucket.bucket_name}/bench/{run_id}/{arm.label}/"
            instance = ec2.Instance(
                self,
                f"Bench-{arm.label}",
                vpc=vpc,
                vpc_subnets=ec2.SubnetSelection(
                    subnet_type=ec2.SubnetType.PUBLIC,
                    availability_zones=[settings.bench_az] if settings.bench_az else None,
                ),
                associate_public_ip_address=True,
                security_group=worker_sg,
                instance_type=ec2.InstanceType(arm.instance_type),
                machine_image=image,
                role=self.role,
                require_imdsv2=True,
                instance_initiated_shutdown_behavior=ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
                detailed_monitoring=False,
                user_data=self._user_data(arm, dest, settings),
                user_data_causes_replacement=True,
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
            self.instances[arm.label] = instance
            if arm.cool:
                Validations.of(instance).acknowledge(
                    Acknowledgment(
                        id="CloudFormation-Validate::W9010",
                        reason="The COOL 3.1 AMI ID is a Marketplace image ID supplied per deploy via -c cool_ami_id (recorded "
                        "in the evidence); it is intentionally not resolved from an SSM parameter.",
                    )
                )
            Validations.of(instance).acknowledge(
                Acknowledgment(
                    id="CloudFormation-Validate::W3010",
                    reason="The VPC lists its AZs explicitly (context 'azs') so synth needs no AZ lookup; a bench "
                    "instance lands in one of them (context 'bench_az' picks one with c8g/c7i capacity).",
                )
            )
            suppress(
                instance,
                [
                    {
                        "id": "AwsSolutions-EC28",
                        "reason": "Short-lived benchmark instance; timings are measured in-process and uploaded "
                        "to S3, so 1-minute detailed monitoring adds cost without value.",
                    },
                    {
                        "id": "AwsSolutions-EC29",
                        "reason": "Bench instances must self-terminate (shutdown behaviour TERMINATE) when the run "
                        "ends; termination protection would defeat the cost guard.",
                    },
                ],
                apply_to_children=True,
            )

        suppress(
            self.role,
            [
                {
                    "id": "AwsSolutions-IAM4",
                    "reason": "AmazonSSMManagedInstanceCore is the AWS-maintained minimum for Session Manager "
                    "(no SSH keys, no inbound rules).",
                    "applies_to": [
                        "Policy::arn:<AWS::Partition>:iam::aws:policy/AmazonSSMManagedInstanceCore"
                    ],
                },
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "Writes are limited to object keys under bench/ in the data bucket; reads are the CDK "
                    "asset-bucket grant (GetObject*/GetBucket*/List*) for the app bundle.",
                    "applies_to": [
                        {"regex": "/^Resource::.*/bench/\\*$/"},
                        {
                            "regex": "/^Resource::arn:(aws|<AWS::Partition>):s3:::cdk-[a-z0-9]+-assets-.*\\/\\*$/"
                        },
                        "Action::s3:GetObject*",
                        "Action::s3:GetBucket*",
                        "Action::s3:List*",
                    ],
                },
            ],
            apply_to_children=True,
        )

    def _user_data(self, arm: BenchArm, dest: str, settings: Settings) -> ec2.UserData:
        ud = ec2.UserData.for_linux(shebang="#!/bin/bash")
        ud.add_commands(
            userdata.PREAMBLE,
            f'ARM_LABEL="{arm.label}"',
            f'DEST="{dest}"',
            "# Hard cost cap: power off (=terminate) after bench_max_minutes no matter what.",
            f"shutdown -h +{settings.bench_max_minutes}",
            "# Always upload the boot log and self-terminate, success or failure.",
            'trap \'aws s3 cp /var/log/cloud-init-output.log "${DEST}cloud-init-output.log" || true; '
            "shutdown -h now' EXIT",
            userdata.ENSURE_SSM_AGENT,
            userdata.ENSURE_AWS_CLI,
            userdata.fetch_app(self.app_asset, self.region),
        )
        if arm.cool:
            ud.add_commands(userdata.COOL_VENV, userdata.ASSERT_COOL_OPENCV)
        else:
            ud.add_commands(userdata.PIP_VENV)
        ud.add_commands(
            f"cd {userdata.APP_DIR}",
            f'export VIRTUAL_ENV={userdata.VENV_DIR} PATH="{userdata.VENV_DIR}/bin:$PATH"',
            f"export PYTHON={userdata.VENV_PY} PYTHONPATH={userdata.APP_DIR}",
            f"export DRIFTSENSE_ENGINE={settings.engine} ENGINE_GIT_SHA={settings.engine_git_sha}",
            f"export BENCH_OUT_ROOT={BENCH_OUT_ROOT} AWS_DEFAULT_REGION={self.region}",
            # run_all.sh uploads <out>/<arm>/ and <arm>.tar.gz to $DEST itself; a failed step still uploads.
            'bash bench/cool/run_all.sh "$ARM_LABEL" "$DEST"',
            "shutdown -h now",
        )
        return ud
