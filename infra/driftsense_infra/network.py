"""Network stack: public-only VPC, no NAT, free gateway endpoints."""

from __future__ import annotations

from aws_cdk import RemovalPolicy, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_logs as logs
from constructs import Construct


class NetworkStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        interface_endpoints: bool = False,
        availability_zones: tuple[str, ...] = ("ap-south-1a", "ap-south-1b", "ap-south-1c"),
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        flow_log_group = logs.LogGroup(
            self,
            "VpcFlowLogs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # Public subnets only: no NAT gateway (~$33/month each) and no ALB.
        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            ip_addresses=ec2.IpAddresses.cidr("10.40.0.0/16"),
            # Explicit AZs: no AZ lookup at synth (works without credentials) and the ASG can fall back
            # to another AZ if c8g capacity is short in one.
            availability_zones=list(availability_zones),
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
            ],
            gateway_endpoints={
                "S3": ec2.GatewayVpcEndpointOptions(service=ec2.GatewayVpcEndpointAwsService.S3),
                "DynamoDB": ec2.GatewayVpcEndpointOptions(service=ec2.GatewayVpcEndpointAwsService.DYNAMODB),
            },
            flow_logs={
                "all": ec2.FlowLogOptions(
                    destination=ec2.FlowLogDestination.to_cloud_watch_logs(flow_log_group),
                    traffic_type=ec2.FlowLogTrafficType.ALL,
                )
            },
        )

        # Worker SG: NO inbound rules at all; the worker is reached only via SSM.
        self.worker_sg = ec2.SecurityGroup(
            self,
            "WorkerSg",
            vpc=self.vpc,
            description="DriftSense worker: no inbound, all egress (SSM/SQS/S3 over HTTPS)",
            allow_all_outbound=True,
        )

        self.interface_endpoints: list[ec2.InterfaceVpcEndpoint] = []
        if interface_endpoints:
            # Opt-in only: each interface endpoint costs ~$8/month/AZ.
            endpoint_sg = ec2.SecurityGroup(
                self,
                "EndpointSg",
                vpc=self.vpc,
                description="DriftSense interface endpoints: HTTPS from the worker SG only",
                allow_all_outbound=False,
            )
            endpoint_sg.add_ingress_rule(self.worker_sg, ec2.Port.tcp(443), "HTTPS from DriftSense worker")
            for name, service in (
                ("Sqs", ec2.InterfaceVpcEndpointAwsService.SQS),
                ("Ssm", ec2.InterfaceVpcEndpointAwsService.SSM),
                ("SsmMessages", ec2.InterfaceVpcEndpointAwsService.SSM_MESSAGES),
                ("Ec2Messages", ec2.InterfaceVpcEndpointAwsService.EC2_MESSAGES),
            ):
                self.interface_endpoints.append(
                    self.vpc.add_interface_endpoint(
                        f"{name}Endpoint",
                        service=service,
                        subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
                        private_dns_enabled=True,
                        security_groups=[endpoint_sg],
                        open=False,
                    )
                )
