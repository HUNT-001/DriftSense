"""bench/cv_equivalence.py: compare logic and the committed OpenCV 4.12 vs 5.0 result."""

import json
from pathlib import Path

from bench.cv_equivalence import compare
from driftsense.contracts import ResultsFile

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "bench"


def _dump(path: Path, opencv: str, rows: list[dict]) -> Path:
    env = {"kind": "env", "opencv": opencv, "numpy": "2", "python": "3.12", "platform": "x", "arch": "x"}
    env["cv2_file"] = "/site-packages/cv2/__init__.py"
    path.write_text("\n".join(json.dumps(r) for r in [env, *rows]) + "\n")
    return path


def _pair(i: int, x: float, hit: bool, decision: str = "escalate") -> dict:
    return {
        "kind": "pair",
        "id": f"p{i}",
        "x": x,
        "y": 0.0,
        "c0": [x, 0.0],
        "margin": 1.0,
        "decision": decision,
        "hit": hit,
    }


def test_compare_counts_disagreements(tmp_path):
    ep = {
        "kind": "episode",
        "id": "e",
        "outcome": "commit",
        "steps": 2,
        "final_margin": 0.3,
        "estimate_nm": None,
    }
    a = _dump(tmp_path / "a.jsonl", "4.12.0", [_pair(0, 1.0, True), _pair(1, 5.0, False), ep])
    b = _dump(tmp_path / "b.jsonl", "5.0.0", [_pair(0, 1.0, True), _pair(1, 50.0, False), ep])
    out = tmp_path / "r.json"
    compare(a, b, out)
    m = ResultsFile.model_validate_json(out.read_text()).metrics
    assert m["pairs.offset_equal_rate"].value == 0.5
    assert m["pairs.offset_diff_gt_1px_both_wrong.n"].value == 1
    assert m["pairs.decision_agreement"].value == 1.0 and m["episodes.outcome_agreement"].value == 1.0


def test_committed_equivalence_result_is_valid_and_reproducible(tmp_path):
    committed = ResultsFile.model_validate_json((RES / "cv5_equivalence.json").read_text())
    assert committed.dataset == "dev" and committed.env.opencv.startswith("5.")
    out = tmp_path / "r.json"
    compare(RES / "cv5_equivalence.base_cv412.jsonl", RES / "cv5_equivalence.cand_cv5.jsonl", out)
    again = ResultsFile.model_validate_json(out.read_text())
    assert {k: v.value for k, v in again.metrics.items()} == {
        k: v.value for k, v in committed.metrics.items()
    }
