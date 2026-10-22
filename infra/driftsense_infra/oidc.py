"""GitHub Actions OIDC provider + deploy role (main branch of one repo only)."""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_iam as iam
from constructs import Construct

from .nag import suppress

GITHUB_OIDC_HOST = "token.actions.githubusercontent.com"
AUDIENCE = "sts.amazonaws.com"
DEPLOY_ROLE_NAME = "DriftSenseGithubDeploy"


class GithubOidcStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        github_repo: str = "HUNT-001/DriftSense",
        existing_provider_arn: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        if existing_provider_arn:
            # Only one provider per URL may exist in an account; reuse it if already there.
            provider_arn = existing_provider_arn
        else:
            # Native AWS::IAM::OIDCProvider (no custom-resource Lambda); IAM fetches the thumbprint itself.
            provider = iam.OidcProviderNative(
                self, "GithubOidcProvider", url=f"https://{GITHUB_OIDC_HOST}", client_ids=[AUDIENCE]
            )
            provider_arn = provider.oidc_provider_arn

        self.subject = f"repo:{github_repo}:ref:refs/heads/main"
        self.role = iam.Role(
            self,
            "GithubDeployRole",
            role_name=DEPLOY_ROLE_NAME,
            description=f"GitHub Actions deploy for {github_repo}@main: may only assume the CDK bootstrap roles",
            max_session_duration=Duration.hours(1),
            assumed_by=iam.WebIdentityPrincipal(
                provider_arn,
                conditions={
                    "StringEquals": {
                        f"{GITHUB_OIDC_HOST}:aud": AUDIENCE,
                        f"{GITHUB_OIDC_HOST}:sub": self.subject,
                    }
                },
            ),
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="AssumeCdkBootstrapRolesOnly",
                actions=["sts:AssumeRole"],
                resources=[f"arn:{self.partition}:iam::{self.account}:role/cdk-*"],
            )
        )
        CfnOutput(
            self,
            "DeployRoleArn",
            value=self.role.role_arn,
            description="Set as GitHub variable AWS_DEPLOY_ROLE_ARN",
        )

        suppress(
            self.role,
            [
                {
                    "id": "AwsSolutions-IAM5",
                    "reason": "cdk deploy needs the CDK bootstrap roles (cdk-<qualifier>-deploy/file-publishing/"
                    "lookup-role-<acct>-<region>); the role-name prefix cdk-* in this account is the narrowest "
                    "stable pattern. No other permissions are granted.",
                    "applies_to": [
                        {
                            "regex": "/^Resource::arn:(aws|<AWS::Partition>):iam::(<AWS::AccountId>|\\d{12}):role\\/cdk-\\*$/"
                        }
                    ],
                }
            ],
            apply_to_children=True,
        )
