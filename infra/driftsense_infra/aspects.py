"""App-wide aspects."""

from __future__ import annotations

import jsii
from aws_cdk import CfnResource, IAspect
from aws_cdk import aws_lambda as lambda_
from constructs import IConstruct


@jsii.implements(IAspect)
class Arm64Lambdas:
    """Force every Lambda (including CDK-internal custom-resource handlers) onto Graviton.

    Our own functions already set ``Architecture.ARM_64``; CDK singleton providers such as
    ``Custom::VpcRestrictDefaultSG`` are plain Node.js handlers and run unchanged on arm64
    (cheaper per GB-second).
    """

    def visit(self, node: IConstruct) -> None:
        if isinstance(node, lambda_.CfnFunction):
            if not node.architectures:
                node.architectures = ["arm64"]
        elif isinstance(node, CfnResource) and node.cfn_resource_type == "AWS::Lambda::Function":
            node.add_property_override("Architectures", ["arm64"])
