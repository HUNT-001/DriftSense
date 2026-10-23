"""Tests for the active-perception agent (G5).

Fast and deterministic; they must pass with ``mcp`` NOT installed. They assert
the behaviours the brief cares about -- the margin drives the next action, the
runner's hard safety rules hold whatever the policy says, the trace is complete
contract evidence, and the ground truth never leaks into agent-visible data --
plus the committed G5 evidence (demo traces + the four-arm results JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from driftsense.agent import run_episode
from driftsense.agent.mcp_server import list_tools, smoke
from driftsense.agent.policy_rules import (
    FOV_MAX_NM,
    SNR_FLOOR,
    Observation,
    decide,
)
from driftsense.agent.tools import AcquireArgs, LocalizeArgs
from driftsense.agent.trace import TraceWriter, read_trace
from driftsense.contracts import ResultsFile, TraceEvent
from driftsense.sim import VirtualSEM

REPO = Path(__file__).resolve().parents[2]

# Small site keeps every SEM-driven test well under a second.
SMALL = dict(family="dram", site_size_nm=1200.0, pixel_nm=1.0)
EP = dict(ref_fov_nm=200.0, ref_dwell_us=12.0, base_fov_nm=360.0, base_dwell_us=1.5, template_frac=0.45)


def _obs(**over) -> Observation:
    base = dict(
        margin=0.0,
        tau=0.1,
        snr=20.0,
        fov_nm=300.0,
        dwell_us=2.0,
        center_nm=(0.0, 0.0),
        estimate_nm=(1.0, -2.0),
        lattice_uncertain=False,
        period_ambiguous=False,
        at_fov_edge=False,
        budget_exhausted=False,
    )
    base.update(over)
    return Observation(**base)


# ---------------------------------------------------------------------------
# Rules policy -- pure, no SEM
# ---------------------------------------------------------------------------
def test_commit_when_margin_ge_tau():
    d = decide(_obs(margin=0.3, tau=0.1, estimate_nm=(2.0, -3.0)))
    assert d.tool == "commit_stage_move" and d.terminal
    assert d.kwargs == {"dx_nm": -2.0, "dy_nm": 3.0}  # corrective = -estimate
    assert d.reason == "margin 0.300 >= tau 0.100"


def test_budget_exhausted_escalates():
    d = decide(_obs(margin=0.0, budget_exhausted=True))
    assert d.tool == "request_human_approval" and d.terminal


def test_low_snr_doubles_dwell():
    d = decide(_obs(margin=0.0, snr=SNR_FLOOR - 1.0, dwell_us=2.0, fov_nm=300.0))
    assert d.tool == "acquire" and d.kwargs["dwell_us"] == 4.0 and d.kwargs["fov_nm"] == 300.0


def test_lattice_uncertain_adds_rect_hypotheses():
    d = decide(_obs(margin=0.0, snr=50.0, lattice_uncertain=True))
    assert d.tool == "localize" and d.kwargs["rect_hypotheses"] == 4


def test_period_tie_at_edge_doubles_fov():
    d = decide(_obs(margin=0.0, snr=50.0, period_ambiguous=True, at_fov_edge=True, fov_nm=300.0))
    assert d.tool == "acquire" and d.kwargs["fov_nm"] == 600.0


def test_fov_doubling_respects_ceiling():
    d = decide(_obs(margin=0.0, snr=50.0, period_ambiguous=True, at_fov_edge=True, fov_nm=FOV_MAX_NM))
    assert d.tool == "localize" and d.kwargs["mode"] == "contour"  # falls through to contour


def test_low_margin_falls_through_to_contour():
    d = decide(_obs(margin=0.05, snr=50.0))
    assert d.tool == "localize" and d.kwargs["mode"] == "contour"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
def test_tool_schema_rejects_bad_args():
    with pytest.raises(ValidationError):
        AcquireArgs(fov_nm=0.0, dwell_us=1.0)  # fov must be > 0
    with pytest.raises(ValidationError):
        LocalizeArgs(mode="nonsense")
    with pytest.raises(ValidationError):
        AcquireArgs(fov_nm=10.0, dwell_us=1.0, bogus=1)  # extra field forbidden


def test_list_tools_has_six():
    tools = list_tools()
    assert set(tools) == {
        "acquire",
        "localize",
        "refine",
        "commit_stage_move",
        "request_human_approval",
        "get_trace",
    }


# ---------------------------------------------------------------------------
# Trace writer
# ---------------------------------------------------------------------------
def test_trace_roundtrips(tmp_path):
    tw = TraceWriter("unit", tmp_path / "unit.jsonl")
    tw.emit("observe", result_summary={"hello": 1})
    tw.emit("commit_move", margin=0.3, tau=0.1, because="margin 0.300 >= tau 0.100")
    back = read_trace(tmp_path / "unit.jsonl")
    assert [e.kind for e in back] == ["observe", "commit_move"]
    assert [e.step for e in back] == [0, 1]
    assert back[1].because == "margin 0.300 >= tau 0.100"
    assert all(isinstance(e, TraceEvent) for e in back)


# ---------------------------------------------------------------------------
# Runner: behaviour + hard safety
# ---------------------------------------------------------------------------
def test_episode_commits_and_writes_full_trace(tmp_path):
    sem = VirtualSEM(seed=0, **SMALL)
    r = run_episode(sem, tau=0.10, run_id="t-commit", trace_path=str(tmp_path / "t.jsonl"), **EP)
    assert r.outcome == "commit"
    kinds = [e.kind for e in r.events]
    assert "commit_move" in kinds and kinds[-1] == "episode_end"
    # the commit event carries the margin>=tau justification (Agentic-Vision evidence)
    commit = next(e for e in r.events if e.kind == "commit_move")
    assert commit.because.startswith("margin ") and ">= tau" in commit.because
    assert commit.margin >= r.tau and commit.stage_move_nm is not None
    # every step is numbered and persisted
    assert read_trace(tmp_path / "t.jsonl")


def test_safety_never_commits_below_tau():
    # An unreachable tau forces the ladder to exhaust -> escalate, never commit.
    sem = VirtualSEM(seed=0, **SMALL)
    r = run_episode(sem, tau=5.0, run_id="t-safe", max_steps=3, **EP)
    assert r.outcome in ("escalate", "abort")
    assert not any(e.kind == "commit_move" for e in r.events)
    assert r.committed_move_nm is None


def test_step_cap_bounds_acquisitions():
    sem = VirtualSEM(seed=1, **SMALL)
    r = run_episode(sem, tau=5.0, run_id="t-cap", max_steps=2, **EP)
    assert r.outcome in ("escalate", "abort")
    assert sem.n_acquisitions <= 6  # bootstrap(2) + <=2 loop acquires, hard-bounded


def test_instrument_dose_cap_escalates_not_commits():
    # A dose cap below the bootstrap budget must escalate safely, never commit.
    sem = VirtualSEM(seed=0, max_dose=1.0, **SMALL)
    r = run_episode(sem, tau=0.10, run_id="t-dose", **EP)
    assert r.outcome == "escalate"
    assert not any(e.kind == "commit_move" for e in r.events)


def test_ground_truth_never_leaks_into_agent_data():
    # Capture the hidden truth via the evaluator hook, then assert it appears in
    # NO agent-visible trace value (args / result_summary).
    sem = VirtualSEM(seed=3, **SMALL)
    truth: dict[str, float] = {}

    def on_dec(_a, _est):
        truth.update(sem.verify())

    r = run_episode(sem, tau=0.10, run_id="t-leak", on_decision=on_dec, **EP)
    assert truth
    hidden = set(truth.values())

    def _nums(obj):
        if isinstance(obj, (int, float)):
            yield float(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _nums(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _nums(v)

    for ev in r.events:
        for v in list(_nums(ev.args)) + list(_nums(ev.result_summary)):
            assert not any(abs(v - t) < 1e-6 for t in hidden), f"trace leaks ground truth {v}"


def test_registry_does_not_call_verify():
    # Static guarantee: the agent modules never reference VirtualSEM.verify.
    for mod in ("tools", "runner", "policy_rules", "policy_llm"):
        src = (REPO / "driftsense" / "agent" / f"{mod}.py").read_text(encoding="utf-8")
        assert ".verify(" not in src, f"{mod}.py must not read the hidden ground truth"


# ---------------------------------------------------------------------------
# MCP smoke (offline) + committed G5 evidence
# ---------------------------------------------------------------------------
def test_mcp_smoke_runs_one_localize_offline():
    out = smoke(seed=0)
    assert out["n_tools"] == 6 and out["engine"] == "v2"
    assert out["n_candidates"] >= 1 and out["decision"] in ("commit", "escalate")


def test_demo_traces_have_commit_and_escalation():
    commit = read_trace(REPO / "traces" / "demo_commit.jsonl")
    esc = read_trace(REPO / "traces" / "demo_escalate.jsonl")
    assert any(e.kind == "commit_move" for e in commit)
    assert any(e.kind == "escalate" for e in esc)
    # escalation went through the human-approval tool
    assert any(e.tool == "request_human_approval" for e in esc)


def test_g5_results_readback_and_invariants():
    rf = ResultsFile.model_validate_json(
        (REPO / "results" / "agent" / "g5_episode.json").read_text(encoding="utf-8")
    )
    assert rf.schema_id == "driftsense.results/1" and rf.dataset == "synthetic"
    m = rf.metrics
    assert int(m["n_episodes"].value) > 0
    for met in m.values():
        if met.unit == "rate":
            assert 0.0 <= met.value <= 1.0
            assert met.ci_low is not None
            assert met.ci_low - 1e-9 <= met.value <= met.ci_high + 1e-9
    # By construction the agent sees the identical bootstrap frame as v2 single-shot,
    # so it can only turn v2's escalations into commits -- never the reverse.
    assert m["agent_rules.escalation"].value <= m["v2_single.escalation"].value + 1e-9
    rec = m["c2_recovery.count"]
    assert 0 <= rec.value <= rec.n


def test_results_json_is_valid_json():
    json.loads((REPO / "results" / "agent" / "g5_episode.json").read_text(encoding="utf-8"))


def test_mcp_server_registers_all_tools_when_mcp_installed():
    """With mcp installed, the real FastMCP server must build and expose the six tools."""
    import asyncio

    pytest.importorskip("mcp.server.fastmcp")
    from driftsense.agent.mcp_server import TOOL_NAMES, create_mcp_server

    server = create_mcp_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == set(TOOL_NAMES)


def test_slow_human_approval_is_not_a_tool_timeout():
    # The operator may take far longer than one tool call; only approval_timeout_s bounds that wait.
    import time

    from driftsense.contracts import ApprovalDecision

    def slow_operator(req):
        time.sleep(3.5)
        return ApprovalDecision(chosen_rank=1, approver="op")

    sem = VirtualSEM(seed=0, **SMALL)
    r = run_episode(
        sem, tau=5.0, run_id="t-slow", max_steps=3, tool_timeout_s=3.0, approval_cb=slow_operator, **EP
    )
    assert r.outcome == "escalate"
    assert r.approval is not None and r.approval.chosen_rank == 1


def test_episode_exposes_decision_result():
    sem = VirtualSEM(seed=0, **SMALL)
    r = run_episode(sem, tau=0.10, run_id="t-res", **EP)
    assert r.outcome == "commit" and r.last_result is not None
    # the result the commit was taken on clears tau (not the post-commit verify frame)
    assert r.last_result.margin.margin >= r.tau
    assert abs(r.last_result.margin.margin - r.final_margin) < 1e-9
    # and the PNG inputs it was computed on: the reference crop and the decision frame
    assert r.template_png is not None and r.search_png is not None
    assert r.template_png.startswith(b"\x89PNG") and r.search_png.startswith(b"\x89PNG")
    tpl = cv2.imdecode(np.frombuffer(r.template_png, np.uint8), cv2.IMREAD_UNCHANGED)
    srch = cv2.imdecode(np.frombuffer(r.search_png, np.uint8), cv2.IMREAD_UNCHANGED)
    assert tpl.shape[0] < srch.shape[0] and tpl.shape[1] < srch.shape[1]
