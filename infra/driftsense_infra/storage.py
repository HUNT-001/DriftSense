"""Storage stack: UI bucket + CloudFront (OAC), data bucket, DynamoDB jobs table."""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_cloudfront as cloudfront
from aws_cdk import aws_cloudfront_origins as origins
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_s3 as s3
from constructs import Construct

from .nag import suppress

# Prefixes on the data bucket that hold short-lived user data (7-day expiry).
EPHEMERAL_PREFIXES = ("uploads/", "jobs/", "traces/")


def _private_bucket_kwargs() -> dict:
    return dict(
        block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        encryption=s3.BucketEncryption.S3_MANAGED,
        enforce_ssl=True,
        object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
        versioned=False,
    )


class StorageStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Server access logs for the two content buckets (cheap; 30-day expiry).
        self.access_log_bucket = s3.Bucket(
            self,
            "AccessLogBucket",
            **_private_bucket_kwargs(),
            lifecycle_rules=[s3.LifecycleRule(id="expire-logs", expiration=Duration.days(30))],
            removal_policy=RemovalPolicy.RETAIN,
        )
        suppress(
            self.access_log_bucket,
            [
                {
                    "id": "AwsSolutions-S1",
                    "reason": "This is the server-access-log destination bucket; logging it into itself would loop.",
                }
            ],
        )

        # (a) UI bucket, private, served only through CloudFront OAC.
        self.ui_bucket = s3.Bucket(
            self,
            "UiBucket",
            **_private_bucket_kwargs(),
            server_access_logs_bucket=self.access_log_bucket,
            server_access_logs_prefix="ui/",
            removal_policy=RemovalPolicy.RETAIN,
        )

        spa_fallback = [
            cloudfront.ErrorResponse(
                http_status=status,
                response_http_status=200,
                response_page_path="/index.html",
                ttl=Duration.seconds(10),
            )
            for status in (403, 404)
        ]
        self.distribution = cloudfront.Distribution(
            self,
            "UiDistribution",
            comment="DriftSense Pilot UI",
            default_root_object="index.html",
            price_class=cloudfront.PriceClass.PRICE_CLASS_200,
            default_behavior=cloudfront.BehaviorOptions(
                origin=origins.S3BucketOrigin.with_origin_access_control(self.ui_bucket),
                viewer_protocol_policy=cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
                cache_policy=cloudfront.CachePolicy.CACHING_OPTIMIZED,
            ),
            error_responses=spa_fallback,
        )
        self.ui_origin = f"https://{self.distribution.distribution_domain_name}"
        suppress(
            self.distribution,
            [
                {
                    "id": "AwsSolutions-CFR1",
                    "reason": "Public competition demo for a global judging panel; no geo restriction requirement.",
                },
                {
                    "id": "AwsSolutions-CFR2",
                    "reason": "AWS WAF web ACL costs >= $5/month + per-rule fees, out of the $50-90 budget; the API "
                    "has stage throttling (20 rps / burst 40) and the UI is static.",
                },
                {
                    "id": "AwsSolutions-CFR3",
                    "reason": "CloudFront standard logging needs an ACL-enabled log bucket; the UI is static and the "
                    "origin bucket already has S3 server access logging.",
                },
                {
                    "id": "AwsSolutions-CFR4",
                    "reason": "Uses the default *.cloudfront.net certificate (no custom domain), for which the minimum "
                    "TLS version cannot be pinned; viewers are redirected to HTTPS.",
                },
            ],
        )

        # (b) Data bucket: uploads/, jobs/, traces/ expire after 7 days; demos/ is kept.
        self.data_bucket = s3.Bucket(
            self,
            "DataBucket",
            **_private_bucket_kwargs(),
            server_access_logs_bucket=self.access_log_bucket,
            server_access_logs_prefix="data/",
            lifecycle_rules=[
                s3.LifecycleRule(
                    id=f"expire-{prefix.rstrip('/')}",
                    prefix=prefix,
                    expiration=Duration.days(7),
                    abort_incomplete_multipart_upload_after=Duration.days(1),
                )
                for prefix in EPHEMERAL_PREFIXES
            ],
            cors=[
                s3.CorsRule(
                    allowed_methods=[s3.HttpMethods.PUT, s3.HttpMethods.GET],
                    allowed_origins=[self.ui_origin],
                    # Presigned PUTs sign Content-Type and Content-Length (+ any x-amz-* the SDK adds).
                    allowed_headers=["content-type", "content-length", "x-amz-*"],
                    exposed_headers=["ETag"],
                    max_age=3000,
                )
            ],
            removal_policy=RemovalPolicy.RETAIN,
        )

        # (c) Jobs table.
        self.table = dynamodb.Table(
            self,
            "JobsTable",
            partition_key=dynamodb.Attribute(name="job_id", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="expires_at",
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Outputs used by the deploy runbook (infra/README.md).
        CfnOutput(self, "UiBucketName", value=self.ui_bucket.bucket_name)
        CfnOutput(self, "UiDistributionId", value=self.distribution.distribution_id)
        CfnOutput(self, "UiUrl", value=self.ui_origin)
        CfnOutput(self, "DataBucketName", value=self.data_bucket.bucket_name)
        CfnOutput(self, "JobsTableName", value=self.table.table_name)
