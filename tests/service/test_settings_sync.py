"""The service's agent operating point (service/settings.py AGENT_*) is a manual copy of eval/agent_eval.py.
eval/ is not importable, so the constants are read from its source with ast and compared here."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from service import settings

AGENT_EVAL = Path(__file__).resolve().parents[2] / "eval" / "agent_eval.py"

# eval/agent_eval.py name -> service/settings.py name
SYNCED = {
    "TAU": "AGENT_TAU",
    "SITE_NM": "AGENT_SITE_NM",
    "PIXEL_NM": "AGENT_PIXEL_NM",
    "REF_FOV_NM": "AGENT_REF_FOV_NM",
    "REF_DWELL_US": "AGENT_REF_DWELL_US",
    "BASE_FOV_NM": "AGENT_BASE_FOV_NM",
    "BASE_DWELL_US": "AGENT_BASE_DWELL_US",
    "TEMPLATE_FRAC": "AGENT_TEMPLATE_FRAC",
    "MAX_STEPS": "AGENT_MAX_STEPS",
}


def _module_constants(path: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


@pytest.mark.skipif(not AGENT_EVAL.exists(), reason="eval/agent_eval.py not in this checkout")
@pytest.mark.parametrize(("eval_name", "service_name"), sorted(SYNCED.items()))
def test_agent_operating_point_matches_eval(eval_name, service_name):
    constants = _module_constants(AGENT_EVAL)
    assert eval_name in constants, f"{eval_name} no longer a literal constant in eval/agent_eval.py"
    assert getattr(settings, service_name) == constants[eval_name]
