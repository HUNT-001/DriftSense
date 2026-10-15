"""v2 engine stub. Real implementation lands in P2 (prompts A2–A4)."""

from __future__ import annotations

from driftsense.contracts import Budget, LocalizeResult, Mode


def localize(template_bytes: bytes, search_bytes: bytes, mode: Mode, budget: Budget) -> LocalizeResult:
    raise NotImplementedError("v2 lands in P2")
