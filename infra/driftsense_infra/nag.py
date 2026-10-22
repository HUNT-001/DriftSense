"""cdk-nag v3 integration with scoped, justified suppressions.

cdk-nag 3.x runs as a CDK *policy-validation plugin* (``Validations.of(app).add_plugins``)
and suppressions are CDK ``Validations.acknowledge()`` calls. Two gaps vs. v2 are closed here:

* v3 matches granular findings (``AwsSolutions-IAM5[Resource::...]``) only by their exact
  rendered text, which embeds cross-stack tokens (``Fn::GetStackOutput`` output names). We keep
  v2-style ``applies_to`` entries instead: plain strings are acknowledged exactly, and
  ``{"regex": "/.../"}`` entries are applied by :class:`ScopedNagPlugin`, which wraps the
  real ``AwsSolutionsChecks`` plugin and drops only findings whose rule, finding text and
  construct path all match a registered suppression.
* ``apply_to_children`` scopes a suppression to the construct's subtree (path prefix).

Synth therefore fails on any finding that is not explicitly justified here or in a stack.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import jsii
from aws_cdk import (
    Acknowledgment,
    IPolicyValidationContext,
    IPolicyValidationPlugin,
    PolicyValidationPluginReport,
    PolicyViolatingResource,
    PolicyViolation,
    Validations,
)
from constructs import IConstruct

AppliesTo = str | dict

_FINDING_RE = re.compile(r"^(?P<rule>[A-Za-z0-9-]+)\[(?P<finding>.*)\]$", re.S)


@dataclass(frozen=True)
class _Scoped:
    path: str
    subtree: bool
    rule_id: str
    pattern: re.Pattern
    reason: str

    def matches(self, rule_id: str, finding: str, construct_path: str | None) -> bool:
        if rule_id != self.rule_id or not construct_path or not self.pattern.search(finding):
            return False
        if construct_path == self.path or construct_path == f"{self.path}/Resource":
            return True
        return self.subtree and construct_path.startswith(self.path + "/")


# Registry keyed by the root (App) construct path ("" for an App), per app instance id.
_REGISTRY: dict[str, list[_Scoped]] = {}


def _root_key(construct: IConstruct) -> str:
    return construct.node.root.node.addr


def _regex(spec: str) -> re.Pattern:
    body = spec[1:-1] if len(spec) >= 2 and spec.startswith("/") and spec.endswith("/") else spec
    return re.compile(body)


def suppress(construct: IConstruct, rules: Iterable[dict], apply_to_children: bool = False) -> None:
    """Justified suppression(s) on ``construct`` (v2 dict format: id, reason, optional applies_to)."""
    for rule in rules:
        rule_id, reason = rule["id"], rule["reason"]
        if len(reason) < 20:
            raise ValueError(f"suppression {rule_id} needs a real reason")
        applies_to: list[AppliesTo] = list(rule.get("applies_to") or [])
        if not applies_to:
            Validations.of(construct).acknowledge(Acknowledgment(id=rule_id, reason=reason))
            continue
        for entry in applies_to:
            if isinstance(entry, str):
                pattern = re.compile("^" + re.escape(entry) + "$")
            else:
                pattern = _regex(entry["regex"])
            _REGISTRY.setdefault(_root_key(construct), []).append(
                _Scoped(construct.node.path, apply_to_children, rule_id, pattern, reason)
            )


def scoped_suppressions(root: IConstruct) -> list[_Scoped]:
    return list(_REGISTRY.get(_root_key(root), []))


@jsii.implements(IPolicyValidationPlugin)
class ScopedNagPlugin:
    """Wraps a cdk-nag NagPack plugin and removes findings covered by scoped regex suppressions."""

    def __init__(self, inner: IPolicyValidationPlugin, root: IConstruct) -> None:
        self._inner = inner
        self._root_key = _root_key(root)

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def version(self) -> str | None:
        return self._inner.version

    @property
    def rule_ids(self) -> list[str] | None:
        return self._inner.rule_ids

    def validate(self, context: IPolicyValidationContext) -> PolicyValidationPluginReport:
        report = self._inner.validate(context)
        scoped = _REGISTRY.get(self._root_key, [])
        kept: list[PolicyViolation] = []
        for violation in report.violations:
            match = _FINDING_RE.match(violation.rule_name)
            resources = list(violation.violating_resources)
            if match and scoped:
                rule_id, finding = match.group("rule"), match.group("finding")
                resources = [
                    r
                    for r in resources
                    if not any(s.matches(rule_id, finding, r.construct_path) for s in scoped)
                ]
            if not resources:
                continue
            kept.append(
                PolicyViolation(
                    description=violation.description,
                    rule_name=violation.rule_name,
                    violating_resources=[
                        PolicyViolatingResource(
                            locations=list(r.locations),
                            resource_logical_id=r.resource_logical_id,
                            template_path=r.template_path,
                            construct_path=r.construct_path,
                        )
                        for r in resources
                    ],
                    fix=violation.fix,
                    rule_metadata=violation.rule_metadata,
                    severity=violation.severity,
                )
            )
        return PolicyValidationPluginReport(
            success=not kept,
            violations=kept,
            metadata=report.metadata,
            plugin_version=report.plugin_version,
        )
