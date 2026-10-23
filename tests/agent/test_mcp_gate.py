"""MCP commit_stage_move enforces the runner's hard gate (never commit below tau)."""

from __future__ import annotations

from types import SimpleNamespace

from driftsense.agent.mcp_server import build_registry, gated_commit


def _kinds(reg):
    return [e["kind"] for e in reg.get_trace()]


def test_commit_refused_without_localize():
    reg = build_registry(seed=3, family="finfet")
    out = gated_commit(reg, dx_nm=5.0, dy_nm=0.0)
    assert out["refused"] is True and out["next"] == "request_human_approval"
    assert _kinds(reg)[-1] == "escalate"


def test_commit_refused_below_tau_and_allowed_at_commit(monkeypatch):
    reg = build_registry(seed=3, family="finfet")
    reg.last_result = SimpleNamespace(margin=SimpleNamespace(decision="escalate", margin=0.02, tau=0.1))
    moved = []
    monkeypatch.setattr(reg.sem, "commit_stage_move", lambda dx, dy: moved.append((dx, dy)))
    assert gated_commit(reg, dx_nm=1.0, dy_nm=2.0)["refused"] is True
    assert moved == []
    reg.last_result = SimpleNamespace(margin=SimpleNamespace(decision="commit", margin=0.3, tau=0.1))
    assert gated_commit(reg, dx_nm=1.0, dy_nm=2.0) == {"dx_nm": 1.0, "dy_nm": 2.0}
    assert moved == [(1.0, 2.0)]
