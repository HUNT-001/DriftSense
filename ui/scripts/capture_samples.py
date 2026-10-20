"""Capture the real engine outputs behind the committed demo traces (offline replay data).

The committed traces (``traces/demo_commit.jsonl``, ``traces/demo_escalate.jsonl``) record only
summaries of each ``localize`` call. Offline replay mode in the UI also wants the full
``LocalizeResult`` (top candidates, margin, timings), the approval request offered to the
operator and the S9 explain PNGs (NCC / roughness heatmaps, overlay, top-5 gallery).

This script re-runs exactly the two episodes ``eval/agent_eval.py`` used to write those traces
(same seed, family and knobs), records every ``localize`` result and the explain PNGs, and keeps
only the results whose margin reproduces the committed trace line to 1e-9. Nothing is invented:
an entry that does not reproduce is dropped and reported.

Run from the repo root with the project venv (OpenCV + numpy required):

    python ui/scripts/capture_samples.py            # writes ui/public/samples/*.capture.json + PNGs

It never writes outside ``ui/public/samples/`` (the episode trace goes to a temp dir).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "ui" / "public" / "samples"
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from driftsense.agent import tools as agent_tools  # noqa: E402
from driftsense.contracts import ApprovalDecision  # noqa: E402
from driftsense.v2 import engine as v2_engine  # noqa: E402
from eval import agent_eval as ae  # noqa: E402

# (sample name, seed, family, extra run_episode kwargs) -- mirrors eval/agent_eval.py demo_traces().
EPISODES = [
    ("demo_commit", 0, "dram", {}),
    ("demo_escalate", 1, "dram", {"max_steps": 1}),
]


def _read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    captured: list[dict] = []
    pngs: list[list] = []

    orig_record = agent_tools.ToolRegistry._record_result
    orig_build_all = v2_engine.build_all

    def record(self, result, **kw):  # type: ignore[no-untyped-def]
        captured.append({"result": result, "pngs": pngs[-1] if pngs else []})
        return orig_record(self, result, **kw)

    def build_all(*a, **kw):  # type: ignore[no-untyped-def]
        arts = list(orig_build_all(*a, **kw))
        pngs.append(arts)
        return arts

    agent_tools.ToolRegistry._record_result = record  # type: ignore[method-assign]
    v2_engine.build_all = build_all  # type: ignore[assignment]

    failures = 0
    for name, seed, family, kw in EPISODES:
        captured.clear()
        pngs.clear()
        requests: list = []

        def approval_cb(req, _requests=requests):  # type: ignore[no-untyped-def]
            _requests.append(req)
            return ApprovalDecision(chosen_rank=None, approver="auto:none")

        committed = _read_trace(REPO / "traces" / f"{name}.jsonl")
        loc_lines = [e for e in committed if e["kind"] == "tool_result" and e["tool"] == "localize"]
        with tempfile.TemporaryDirectory() as td:
            ae._agent(seed, family, trace_path=str(Path(td) / "t.jsonl"), approval_cb=approval_cb, **kw)

        sample_dir = OUT / name
        sample_dir.mkdir(exist_ok=True)
        entries = []
        for i, cap in enumerate(captured):
            res = cap["result"]
            line = loc_lines[i] if i < len(loc_lines) else None
            if line is None or abs(line["margin"] - res.margin.margin) > 1e-9:
                print(f"{name}: localize #{i} does not reproduce the committed trace; dropped", file=sys.stderr)
                continue
            alts: dict[str, str] = {}
            for art in cap["pngs"]:
                fname = f"{art.name}.png"
                sha = hashlib.sha256(art.png).hexdigest()
                listed = {a.uri: a.sha256 for a in res.artifacts}
                if listed.get(fname) != sha:
                    print(f"{name}: {fname} sha mismatch; dropped", file=sys.stderr)
                    continue
                # Re-encode as WebP (about 10x smaller) for the static bundle; the original PNG's
                # sha256 stays in result.artifacts.
                img = cv2.imdecode(np.frombuffer(art.png, np.uint8), cv2.IMREAD_COLOR)
                ok, webp = cv2.imencode(".webp", img, [cv2.IMWRITE_WEBP_QUALITY, 88])
                if not ok:
                    raise RuntimeError(f"WebP encode failed for {fname}")
                (sample_dir / f"step{line['step']}_{art.name}.webp").write_bytes(webp.tobytes())
                alts[fname] = art.alt
            entries.append(
                {
                    "trace_step": line["step"],
                    "result": json.loads(res.model_dump_json()),
                    "artifact_files": {
                        u: f"{name}/step{line['step']}_{u.removesuffix('.png')}.webp" for u in alts
                    },
                    "artifact_alt": alts,
                }
            )
        if not entries:
            failures += 1
        approval = json.loads(requests[0].model_dump_json()) if requests else None
        doc = {
            "_about": (
                "Real engine outputs re-captured by ui/scripts/capture_samples.py for the committed "
                f"trace traces/{name}.jsonl (seed {seed}, family {family}). Only localize results whose "
                "margin reproduces the committed trace are kept."
            ),
            "trace": f"{name}.jsonl",
            "seed": seed,
            "family": family,
            "localize": entries,
            "approval_request": approval,
        }
        (OUT / f"{name}.capture.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: kept {len(entries)}/{len(captured)} localize results, approval={'yes' if approval else 'no'}")

    agent_tools.ToolRegistry._record_result = orig_record  # type: ignore[method-assign]
    v2_engine.build_all = orig_build_all  # type: ignore[assignment]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
