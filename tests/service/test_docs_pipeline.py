"""Docs pipeline: scripts/render_docs.py + scripts/check_claims.py (Person B, B4)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("driftsense.contracts")

ROOT = Path(__file__).resolve().parents[2]
FIX = Path(__file__).resolve().parent / "fixtures" / "docs"
sys.path.insert(0, str(ROOT / "scripts"))

import check_claims  # noqa: E402
import render_docs  # noqa: E402
from _docs_common import GENERATED_PREFIX, HEADER_TEMPLATE, load_results  # noqa: E402

HEADER = HEADER_TEMPLATE.format(src="docs/templates/T.md.j2")
G2_SPEEDUP = "12.3× [DEV]<!--src:results/accuracy/g2.json#speedup fmt=x-->"


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """A throwaway repo: results/ and docs/templates/ copied from the fixtures."""
    shutil.copytree(FIX / "results", tmp_path / "results")
    shutil.copytree(FIX / "templates", tmp_path / "docs" / "templates")
    return tmp_path


def render(ws: Path, *extra: str, results: Path | None = None) -> int:
    return render_docs.main(
        [
            "--results-dir",
            str(results or ws / "results"),
            "--templates",
            str(ws / "docs" / "templates"),
            "--out",
            str(ws / "docs"),
            "--readme",
            str(ws / "README.md"),
            *extra,
        ]
    )


def claims(ws: Path, capsys, *extra: str) -> tuple[int, dict]:
    capsys.readouterr()
    rc = check_claims.main(["--root", str(ws), "--json", "--extra", *extra])
    return rc, json.loads(capsys.readouterr().out)


def codes(report: dict, severity: str = "error") -> set[str]:
    return {f["code"] for f in report["findings"] if f["severity"] == severity}


def write_template(ws: Path, name: str, body: str) -> None:
    (ws / "docs" / "templates" / name).write_text(body, encoding="utf-8")


def write_doc(ws: Path, name: str, body: str, generated: bool = True) -> Path:
    p = ws / "docs" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text((HEADER + "\n" if generated else "") + body, encoding="utf-8", newline="\n")
    return p


def only_templates(ws: Path, *keep: str) -> None:
    for p in (ws / "docs" / "templates").glob("*.j2"):
        if p.name not in keep:
            p.unlink()


# --------------------------------------------------------------------------- render


EXPECTED = {
    "default": G2_SPEEDUP,
    "raw": "12.3456 [DEV]<!--src:results/accuracy/g2.json#speedup fmt=raw-->",
    "int": "12 [DEV]<!--src:results/accuracy/g2.json#speedup fmt=int-->",
    "f1": "12.3 [DEV]<!--src:results/accuracy/g2.json#speedup fmt=f1-->",
    "f2": "12.35 [DEV]<!--src:results/accuracy/g2.json#speedup fmt=f2-->",
    "f3": "12.346 [DEV]<!--src:results/accuracy/g2.json#speedup fmt=f3-->",
    "pct": "47.5% [DEV]<!--src:results/accuracy/g2.json#top1_success fmt=pct-->",
    "pct-ci": "47.5% (95% CI 40.1–54.9%) [DEV]<!--src:results/accuracy/g2.json#top1_success fmt=pct ci-->",
    "ms": "23.5 ms (95% CI 22.9–24.1 ms) [DEV]<!--src:results/accuracy/g2.json#e2e_p50 fmt=ms ci-->",
    "ms-from-s": "42.1 ms [DEV]<!--src:results/accuracy/g2.json#latency_s fmt=ms-->",
    "x-ci": "12.3× (95% CI 11.0–13.5×) [DEV]<!--src:results/accuracy/g2.json#speedup fmt=x ci-->",
    "usd": "$0.0123 [DEV]<!--src:results/accuracy/g2.json#cost_per_1000 fmt=usd-->",
    "count": "1,234 [DEV]<!--src:results/accuracy/g2.json#pairs fmt=int-->",
    "n": "400 [DEV]<!--src:results/accuracy/g2.json#speedup field=n fmt=int-->",
    "nolabel": "12.3×<!--src:results/accuracy/g2.json#speedup fmt=x nolabel-->",
    "sealed": "47.5% (95% CI 40.1–54.9%) [SEALED]"
    "<!--src:results/sealed/finfet.json#top1_success fmt=pct ci-->",
    "bench": "1.045× (95% CI 1.006–1.079×)<!--src:results/bench/cool_bench.json"
    "#speedup.graviton-cool_vs_graviton-pip.kleidicv.geomean.tN fmt=x ci-->",
    "usd-small": "$0.000119<!--src:results/bench/cool_bench.json#cost.graviton-cool.usd_per_1000 fmt=usd-->",
    "sha": "0123abc<!--src:results/accuracy/g2.json#@git_sha-->",
    "created": "2026-10-14T09:30:00+00:00<!--src:results/accuracy/g2.json#@created_at-->",
    "date": "2026-10-14<!--src:results/accuracy/g2.json#@created_at fmt=date-->",
    "dataset": "dev<!--src:results/accuracy/g2.json#@dataset-->",
    "instance": "c8g.4xlarge<!--src:results/bench/cool_bench.json#@env.instance_type-->",
    "cool": "true<!--src:results/bench/cool_bench.json#@env.cool-->",
}


def test_every_fmt_renders_exactly(ws):
    assert render(ws) == 0
    lines = (ws / "docs" / "FORMATS.md").read_text(encoding="utf-8").splitlines()
    got = {ln[2:].split(": ", 1)[0]: ln.split(": ", 1)[1] for ln in lines if ln.startswith("- ")}
    for k, v in EXPECTED.items():
        assert got[k] == v, k


def test_generated_header_and_single_trailing_newline(ws):
    assert render(ws) == 0
    raw = (ws / "docs" / "FORMATS.md").read_bytes().decode("utf-8")
    first = raw.split("\n", 1)[0]
    assert first == HEADER_TEMPLATE.format(src="docs/templates/FORMATS.md.j2")
    assert first.startswith(GENERATED_PREFIX)
    assert raw.endswith("\n") and not raw.endswith("\n\n") and "\r" not in raw


def test_double_render_is_idempotent_and_deterministic(ws, tmp_path_factory, capsys):
    assert render(ws) == 0
    first = (ws / "docs" / "FORMATS.md").read_bytes()
    assert render(ws) == 0
    assert "unchanged" in capsys.readouterr().out
    assert (ws / "docs" / "FORMATS.md").read_bytes() == first
    other = tmp_path_factory.mktemp("other")
    shutil.copytree(FIX / "results", other / "results")
    shutil.copytree(FIX / "templates", other / "docs" / "templates")
    assert render(other) == 0
    assert (other / "docs" / "FORMATS.md").read_bytes() == first


def test_check_detects_missing_and_stale_output(ws, capsys):
    assert render(ws, "--check") == 1  # nothing rendered yet
    assert render(ws) == 0
    assert render(ws, "--check") == 0
    out = ws / "docs" / "FORMATS.md"
    out.write_text(
        out.read_text(encoding="utf-8").replace("# Formats", "# Formatz"),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert render(ws, "--check") == 1
    assert "stale" in capsys.readouterr().err
    assert render(ws) == 0
    g2 = ws / "results" / "accuracy" / "g2.json"
    g2.write_text(g2.read_text(encoding="utf-8").replace("12.3456", "13.3456"), encoding="utf-8")
    assert render(ws, "--check") == 1


def test_check_flags_orphaned_generated_file(ws, capsys):
    assert render(ws) == 0
    write_doc(ws, "ORPHAN.md", "text\n")
    capsys.readouterr()
    assert render(ws, "--check") == 1
    assert "orphan" in capsys.readouterr().err


def test_invalid_results_file_is_rejected_with_path(ws, capsys):
    assert render(ws, results=FIX / "results_invalid") == 1
    err = capsys.readouterr().err
    assert "bad.json" in err and "invalid ResultsFile" in err


def test_strict_undefined_names_template_and_line(ws, capsys):
    only_templates(ws)
    write_template(ws, "BAD.md.j2", "# t\n\nfine line\n{{ not_defined }}\n")
    assert render(ws) == 1
    err = capsys.readouterr().err
    assert "docs/templates/BAD.md.j2:4" in err and "undefined" in err


@pytest.mark.parametrize(
    "call, needle",
    [
        ('m("accuracy/nope", "speedup")', "results/accuracy/nope.json not found"),
        (
            'm("accuracy/g2", "speedupp")',
            "metric 'speedupp' not in results/accuracy/g2.json",
        ),
        ('m("accuracy/g2", "latency_s", ci=true)', "no ci_low/ci_high"),
        ('m("accuracy/g2", "speedup", fmt="pctt")', "unknown fmt='pctt'"),
        ('m("accuracy/g2", "speedup", fmt="ms")', "fmt='ms' needs unit"),
        ('n("accuracy/g2", "latency_s")', "has no n"),
        ('meta("accuracy/g2", "protocol_sha256")', "is null"),
        ('meta("accuracy/g2", "env.gpu")', "unknown meta field"),
    ],
)
def test_render_errors_name_template_and_line(ws, capsys, call, needle):
    only_templates(ws)
    write_template(ws, "BAD.md.j2", "# t\n\n{{ " + call + " }}\n")
    assert render(ws) == 1
    err = capsys.readouterr().err
    assert "docs/templates/BAD.md.j2:3" in err and needle in err


@pytest.mark.parametrize("label", ["", ", label=false"])
def test_dev_number_in_sealed_sentence_fails_render(ws, capsys, label):
    only_templates(ws)
    write_template(
        ws,
        "BAD.md.j2",
        '# t\n\nOn the sealed set we got {{ m("accuracy/g2", "speedup"' + label + ") }}.\n",
    )
    assert render(ws) == 1
    assert "sealed" in capsys.readouterr().err
    assert not (ws / "docs" / "BAD.md").exists()


def test_dev_and_sealed_in_separate_sentences_and_cells_render(ws):
    only_templates(ws)
    write_template(
        ws,
        "OK.md.j2",
        '# t\n\nSealed: {{ m("sealed/finfet", "top1_success") }}. Dev: {{ m("accuracy/g2", "speedup") }}.\n\n'
        '| Split | Value |\n|---|---|\n| sealed | {{ m("sealed/finfet", "top1_success") }} |\n'
        '| dev | {{ m("accuracy/g2", "top1_success") }} |\n',
    )
    assert render(ws) == 0


def test_allow_missing_renders_placeholder_that_check_claims_rejects(ws, capsys):
    only_templates(ws)
    write_template(
        ws,
        "DRAFT.md.j2",
        '# t\n\nValue {{ m("accuracy/nope", "speedup", fmt="pct") }}.\n',
    )
    assert render(ws) == 1
    assert render(ws, "--allow-missing") == 0
    text = (ws / "docs" / "DRAFT.md").read_text(encoding="utf-8")
    assert "⟨missing: results/accuracy/nope.json#speedup⟩" in text
    rc, rep = claims(ws, capsys, "--no-render-check")
    assert rc == 1 and "CL006" in codes(rep)


def test_requires_makes_template_pending_not_failing(ws, capsys):
    write_template(
        ws,
        "LATER.md.j2",
        '{#- requires: agent/arms -#}\n# t\n{{ m("agent/arms", "x") }}\n',
    )
    assert render(ws) == 0
    assert (
        "PENDING docs/templates/LATER.md.j2: waiting for results/agent/arms.json" in capsys.readouterr().err
    )
    assert not (ws / "docs" / "LATER.md").exists()
    assert render(ws, "--strict") == 1


def test_readme_rendered_only_with_flag(ws):
    write_template(
        ws,
        "README.md.j2",
        "# DriftSense\n\nSpeed-up {{ m('accuracy/g2', 'speedup') }}.\n",
    )
    assert render(ws) == 0
    assert not (ws / "README.md").exists()
    assert render(ws, "--include-readme") == 0
    assert G2_SPEEDUP in (ws / "README.md").read_text(encoding="utf-8")
    assert render(ws, "--check") == 0  # a generated README is picked up automatically


def test_partials_are_not_rendered_but_can_be_included(ws):
    write_template(ws, "_bits.md.j2", "Speed-up {{ m('accuracy/g2', 'speedup') }}.")
    write_template(ws, "USES.md.j2", "# t\n\n{% include '_bits.md.j2' %}\n")
    assert render(ws) == 0
    assert not (ws / "docs" / "_bits.md").exists()
    assert G2_SPEEDUP in (ws / "docs" / "USES.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- check_claims


def test_clean_render_passes_claims(ws, capsys):
    assert render(ws) == 0
    rc, rep = claims(ws, capsys)
    assert rc == 0, rep
    assert rep["summary"]["generated_docs"] == 1 and rep["summary"]["errors"] == 0


def _tamper(ws: Path, old: str, new: str) -> None:
    p = ws / "docs" / "FORMATS.md"
    text = p.read_text(encoding="utf-8")
    assert old in text
    p.write_text(text.replace(old, new, 1), encoding="utf-8", newline="\n")


@pytest.mark.parametrize(
    "old, new, code",
    [
        ("# Formats", "# Formats\n\nTop-1 accuracy was 55.5% overall.", "CL001"),
        ("# Formats", "# Formats\n\nIt runs in 12 ms on Graviton.", "CL001"),
        ("# Formats", "# Formats\n\nCost is $0.05 per job.", "CL001"),
        (
            "47.5% (95% CI 40.1–54.9%) [SEALED]",
            "48.5% (95% CI 40.1–54.9%) [SEALED]",
            "CL002",
        ),
        ("- int: 12 [DEV]", "- int: 13 [DEV]", "CL002"),
        ("sha: 0123abc", "sha: 0123abd", "CL002"),
        ("results/accuracy/g2.json#pairs", "results/accuracy/g9.json#pairs", "CL003"),
        ("results/accuracy/g2.json#pairs", "results/accuracy/g2.json#pairz", "CL003"),
        ("# Formats", "# Formats\n\nrecall@60 was high.", "CL004"),
        (
            "# Formats",
            "# Formats\n\nOn the sealed set the speed-up was " + G2_SPEEDUP + ".",
            "CL005",
        ),
        ("# Formats", "# Formats\n\nValue ⟨missing: results/x.json#y⟩.", "CL006"),
        ("# Formats", "# Formats\n\nTODO: add the table.", "CL007"),
        ("# Formats", "# Formats\n\nAccuracy is XX% on DRAM.", "CL007"),
        ("# Formats", "# Formats\n\n```\n" + G2_SPEEDUP + "\n```", "CL010"),
    ],
)
def test_claims_catches(ws, capsys, old, new, code):
    assert render(ws) == 0
    _tamper(ws, old, new)
    rc, rep = claims(ws, capsys, "--no-render-check")
    assert rc == 1
    assert code in codes(rep), rep["findings"]


def test_claims_detects_stale_render(ws, capsys):
    assert render(ws) == 0
    _tamper(ws, "# Formats", "# Formats (edited by hand)")
    rc, rep = claims(ws, capsys)
    assert rc == 1 and codes(rep) == {"CL008"}


def test_claims_detects_results_changed_after_render(ws, capsys):
    assert render(ws) == 0
    f = ws / "results" / "sealed" / "finfet.json"
    f.write_text(
        f.read_text(encoding="utf-8").replace('"value": 0.475', '"value": 0.485'),
        encoding="utf-8",
    )
    rc, rep = claims(ws, capsys)
    assert rc == 1 and {"CL002", "CL008"} <= codes(rep)


def test_claims_reports_invalid_results_file(ws, capsys):
    shutil.copy(
        FIX / "results_invalid" / "accuracy" / "bad.json",
        ws / "results" / "accuracy" / "bad.json",
    )
    rc, rep = claims(ws, capsys)
    assert rc == 1 and "CL009" in codes(rep)


ALLOWED = """# 10. Report

## 3.2 Agent loop

1. First step, released 2026-10-26 at 21:00 IST (2026-10-26T15:30Z).
2. Deadline Oct 26, 2026; schedule Oct 11–26; also 26 Oct.
3. OpenCV 5 and opencv-python 5.0.0.93 with Python 3.12, Python 3.10+, COOL 3.1, v2.1, v1.
4. Instances c8g.large, m8g.4xlarge and c7i.4xlarge in ap-south-1; stages S0–S9; top-1; p95; x86 and arm64.
5. See Prompt 15, Section 12, §6, issue #1776, citation \\[12\\]\\[38\\], [ref](https://example.com/v/2026/12.5).
6. Three diagrams, 3 arms, at most 6 steps.

Inline `cv2.matchTemplate` takes `23 ms` and `47.5%` in code spans.

```python
speedup = 12.3  # fenced code is not prose
cost = 0.0042
```

<img src="fig_2026.png" width="640">
"""


def test_allow_list_does_not_trigger(ws, capsys):
    only_templates(ws)
    write_doc(ws, "ALLOWED.md", ALLOWED)
    rc, rep = claims(ws, capsys, "--no-render-check")
    assert rep["findings"] == [] and rc == 0


@pytest.mark.parametrize(
    "line, n",
    [
        ("Accuracy 47.5% here", 1),
        ("speed-up of 12.3× or 3x", 2),
        ("runs in 23 ms", 1),
        ("costs $0.05", 1),
        ("error 0.42 nm", 1),
        ("about 200 pairs and 1,000 jobs", 2),
        ("tolerance 1e-3 px", 1),
        ("grid of 1000×1000 at 10⁶ placements", 2),
        ("kept for 7 days", 1),
        ("one 3 arms OpenCV 5 S9 c8g.large", 0),
        ("Wilson 95% CI on S0–S9, D1–D12, G2, v1-munichtech, recall@K, arm64 amd64", 0),
        ("a 95% interval", 1),
    ],
)
def test_claim_token_rules(line, n):
    assert len(check_claims.claim_tokens(line)) == n, check_claims.claim_tokens(line)


def test_legacy_docs_only_warn_unless_strict(ws, capsys):
    only_templates(ws)
    write_doc(
        ws,
        "BENCHMARK.md",
        "# Old\n\nrecall@60 is 97.5% on every tier. TODO\n",
        generated=False,
    )
    rc, rep = claims(ws, capsys)
    assert rc == 0
    assert {"CL001", "CL004", "CL007"} <= codes(rep, "warning") and codes(rep) == set()
    assert rep["summary"]["note"] == "no generated docs yet"
    capsys.readouterr()
    assert check_claims.main(["--root", str(ws), "--strict-legacy", "--extra"]) == 1


def test_legacy_skip_opt_out_cannot_cover_generated(ws, capsys):
    only_templates(ws)
    write_doc(
        ws,
        "REF.md",
        "<!-- check_claims: skip (examples) -->\nrecall@60 47.5%\n",
        generated=False,
    )
    rc, rep = claims(ws, capsys)
    assert rc == 0 and rep["findings"] == [] and rep["summary"]["skipped_docs"] == 1
    write_doc(ws, "GEN.md", "<!-- check_claims: skip -->\nAccuracy 47.5%.\n")
    rc, rep = claims(ws, capsys, "--no-render-check")
    assert rc == 1 and "CL001" in codes(rep)


def test_empty_repo_exits_zero(tmp_path, capsys):
    assert check_claims.main(["--root", str(tmp_path), "--extra"]) == 0
    assert "no generated docs yet" in capsys.readouterr().out


def test_cli_entry_point_runs(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "check_claims.py"),
            "--root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "no generated docs yet" in proc.stdout


# --------------------------------------------------------------------------- starter templates


REAL_TEMPLATES = ROOT / "docs" / "templates"


def test_starter_templates_parse_and_render(tmp_path, capsys):
    names = render_docs.list_templates(REAL_TEMPLATES)
    assert {
        "ARCHITECTURE.md.j2",
        "COOL_EVIDENCE.md.j2",
        "SUBMISSION.md.j2",
        "RESPONSIBLE_USE.md.j2",
        "README.md.j2",
    } <= set(names)
    args = [
        "--results-dir",
        str(tmp_path / "results"),
        "--templates",
        str(REAL_TEMPLATES),
        "--out",
        str(tmp_path / "docs"),
        "--readme",
        str(tmp_path / "README.md"),
    ]
    # No results yet: number-free pages render, the others are pending.
    assert render_docs.main(args) == 0
    assert (tmp_path / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8").count("```mermaid") == 3
    assert (tmp_path / "docs" / "RESPONSIBLE_USE.md").exists()
    cool = (tmp_path / "docs" / "COOL_EVIDENCE.md").read_text(encoding="utf-8")
    assert "campaign pending" in cool and "⟨missing" not in cool
    # root templates are never written over a hand-written root file without the flag
    assert not (tmp_path / "SUBMISSION.md").exists()
    assert not (tmp_path / "docs" / "SUBMISSION.md").exists()
    capsys.readouterr()
    rc = check_claims.main(
        [
            "--root",
            str(tmp_path),
            "--templates",
            str(REAL_TEMPLATES),
            "--json",
            "--extra",
        ]
    )
    rep = json.loads(capsys.readouterr().out)
    assert rc == 0 and codes(rep) == set(), rep["findings"]
    # Drafting mode exercises every m()/n()/meta() call without results.
    assert render_docs.main(args + ["--allow-missing", "--out", str(tmp_path / "draft")]) == 0
    draft = (tmp_path / "draft" / "COOL_EVIDENCE.md").read_text(encoding="utf-8")
    assert "campaign pending" in draft


# --------------------------------------------------------------------------- bench report


CMP = "speedup.graviton-cool_vs_graviton-pip"


def _real_ws(tmp_path: Path) -> Path:
    """Fixture results (incl. the trimmed bench report) + the REAL starter templates."""
    shutil.copytree(FIX / "results", tmp_path / "results")
    shutil.copytree(REAL_TEMPLATES, tmp_path / "docs" / "templates")
    return tmp_path


def _edit_metric(ws: Path, key: str, metric: str, **fields) -> None:
    p = ws / "results" / f"{key}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["metrics"].setdefault(metric, {"value": 0.0, "unit": "x"}).update(fields)
    p.write_text(json.dumps(d), encoding="utf-8")


def _cool(ws: Path) -> str:
    assert render(ws) == 0
    return (ws / "docs" / "COOL_EVIDENCE.md").read_text(encoding="utf-8")


def test_cool_evidence_renders_from_bench_report_and_passes_claims(tmp_path, capsys):
    ws = _real_ws(tmp_path)
    text = _cool(ws)
    # verdicts come from the CIs in the JSON, never from the template
    assert "the candidate is\nfaster, 1.045× (95% CI 1.006–1.079×)" in text
    assert "there is no\nmeasurable difference (1.030× (95% CI 0.998–1.071×)" in text
    assert "single measurement, no interval" in text
    assert text.count("Equality of results: **passed**.") == 3
    assert "`absdiff.u8` | 1.275× (95% CI 1.196–1.308×)" in text
    assert "| `graviton-cool` (Graviton, COOL) | $0.432" in text
    assert "graviton-cool-large" not in text.split("## Per-arm")[1]
    assert "every arm ran the same commit" in text
    rc, rep = claims(ws, capsys)
    assert rc == 0 and codes(rep) == set(), rep["findings"]
    assert render(ws, "--check") == 0


@pytest.mark.parametrize(
    "lo, hi, needle",
    [
        (0.98, 1.10, "there is no\nmeasurable difference"),
        (0.80, 0.95, "the candidate is\nslower"),
        (1.01, 1.10, "the candidate is\nfaster"),
    ],
)
def test_cool_headline_follows_the_interval(tmp_path, lo, hi, needle):
    ws = _real_ws(tmp_path)
    _edit_metric(ws, "bench/cool_bench", f"{CMP}.kleidicv.geomean.tN", ci_low=lo, ci_high=hi)
    head = _cool(ws).split("## COOL gain on the same Graviton hardware")[1].split("|")[0]
    assert needle in head


def test_cool_flags_failed_equality_moved_control_and_mixed_commits(tmp_path):
    ws = _real_ws(tmp_path)
    eq = "equality.graviton-cool_vs_graviton-pip.offset"
    _edit_metric(ws, "bench/cool_bench", f"{eq}.pass", value=0.0)
    _edit_metric(ws, "bench/cool_bench", f"{CMP}.numpy.geomean.tN", ci_low=1.1, ci_high=1.3)
    _edit_metric(ws, "bench/cool_bench", "meta.git_sha_consistent", value=0.0)
    text = _cool(ws)
    assert "Equality of results: **FAILED**." in text
    assert "**Caution:** the NumPy negative control moved" in text
    assert "did not all run the same commit" in text


def test_cool_optional_production_arm_appears_when_present(tmp_path):
    ws = _real_ws(tmp_path)
    for metric, unit, v in [
        ("cost.graviton-cool-large.usd_per_1000", "USD/1000 localizations", 0.0002),
        ("cost.graviton-cool-large.usd_per_hour", "USD/h", 0.1),
    ]:
        _edit_metric(ws, "bench/cool_bench", metric, value=v, unit=unit)
    assert "| `graviton-cool-large` (Graviton, COOL, production worker size) | $0.100" in _cool(ws)


# --------------------------------------------------------------------------- real results


REAL_RESULTS = ROOT / "results"
needs_real = pytest.mark.skipif(
    not (REAL_RESULTS / "sealed" / "decisions.json").is_file(),
    reason="real sealed results not present in this checkout",
)


def _real_results_ws(tmp_path: Path) -> Path:
    """The REAL results/ + the REAL templates in a throwaway root."""
    shutil.copytree(REAL_RESULTS, tmp_path / "results")
    shutil.copytree(REAL_TEMPLATES, tmp_path / "docs" / "templates")
    return tmp_path


@needs_real
def test_real_results_load_including_plain_json_documents():
    results, errors = load_results(REAL_RESULTS)
    assert errors == []
    # plain JSON documents are readable by q() but are not ResultsFiles
    assert "sealed/decisions" in results.raw and "sealed/decisions" not in results
    assert "sealed/disclosures" in results.raw and "sealed/disclosures" not in results
    assert "sealed/v2_single" in results


@needs_real
def test_all_b_docs_render_from_real_results_and_pass_claims(tmp_path, capsys):
    ws = _real_results_ws(tmp_path)
    assert render(ws, "--include-readme") == 0
    for p in (
        "README.md",
        "SUBMISSION.md",
        "docs/ARCHITECTURE.md",
        "docs/COOL_EVIDENCE.md",
        "docs/RESPONSIBLE_USE.md",
    ):
        text = (ws / p).read_text(encoding="utf-8")
        assert text.startswith(GENERATED_PREFIX), p
        assert "⟨missing" not in text and "TODO" not in text and "TBD" not in text
    sub = (ws / "SUBMISSION.md").read_text(encoding="utf-8")
    # honest negative findings are rendered from decisions.json, not typed
    assert "**Overall pass: false<!--src:results/sealed/decisions.json#overall_pass q-->.**" in sub
    assert "The sealed gate did not pass." in sub
    assert "v2 single-shot top-1 is below both baselines" in sub
    assert "campaign pending" in sub  # no COOL numbers before the campaign
    cool = (ws / "docs" / "COOL_EVIDENCE.md").read_text(encoding="utf-8")
    assert "campaign pending" in cool and "<!--src:" not in cool
    rc, rep = claims(ws, capsys, "SUBMISSION.md")
    assert rc == 0 and codes(rep) == set(), rep["findings"]
    assert rep["summary"]["generated_docs"] == 5
    # once generated, root docs are re-rendered without the flag; --check is clean
    assert render(ws, "--check") == 0


@needs_real
def test_committed_docs_are_up_to_date(capsys):
    """The rendered docs in the repo match a fresh render of the real results."""
    assert (
        render_docs.main(
            [
                "--results-dir",
                str(REAL_RESULTS),
                "--templates",
                str(REAL_TEMPLATES),
                "--out",
                str(ROOT / "docs"),
                "--readme",
                str(ROOT / "README.md"),
                "--check",
            ]
        )
        == 0
    ), capsys.readouterr().err


# --------------------------------------------------------------------------- A-grammar


A_FIX = FIX / "a_report"
A_REF = "origin/a/p9-report"
A_PLACEHOLDER_RE = render_docs.A_PLACEHOLDER


def _a_sources(source: str) -> dict[str, str]:
    """A's REPORT.md / PRIOR_WORK.md: from git (A's open PR branch) or the fixture copy."""
    out = {}
    for name in ("REPORT.md", "PRIOR_WORK.md"):
        if source == "fixture":
            out[name] = (A_FIX / name).read_text(encoding="utf-8")
            continue
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{A_REF}:docs/{name}"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if proc.returncode != 0:
            pytest.skip(f"{A_REF} not available: {proc.stderr.strip()}")
        out[name] = proc.stdout
    return out


@needs_real
@pytest.mark.parametrize("source", ["fixture", "git"])
def test_person_a_report_resolves_every_placeholder(tmp_path, capsys, source):
    """A's REPORT.md + PRIOR_WORK.md, used verbatim as templates (as they will be on the
    int/ branch), render against the REAL results/ and pass check_claims."""
    srcs = _a_sources(source)
    tdir = tmp_path / "docs" / "templates"
    tdir.mkdir(parents=True)
    shutil.copytree(REAL_RESULTS, tmp_path / "results")
    expected = 0
    for name, text in srcs.items():
        (tdir / (name + ".j2")).write_text(text, encoding="utf-8")
        body = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        expected += len(A_PLACEHOLDER_RE.findall(body))
    assert expected == 68  # A's count at 584c0ab
    capsys.readouterr()
    assert render(tmp_path) == 0, capsys.readouterr().err
    markers = 0
    for name in srcs:
        out = (tmp_path / "docs" / name).read_text(encoding="utf-8")
        visible = re.sub(r"<!--.*?-->", "", out, flags=re.DOTALL)
        assert "{{" not in visible and "}}" not in visible, name
        markers += out.count("<!--src:")
    assert markers == expected
    rc, rep = claims(tmp_path, capsys, "--no-render-check")
    assert rc == 0 and rep["findings"] == [], rep["findings"]
    rc, rep = claims(tmp_path, capsys)
    assert rc == 0, rep["findings"]


def _q_ws(tmp_path: Path, body: str) -> tuple[int, str]:
    shutil.copytree(FIX / "results", tmp_path / "results")
    (tmp_path / "results" / "sealed" / "decisions.json").write_text(
        json.dumps(
            {
                "rules": {"D1": {"pass": False, "detail": "lower=0.387"}, "D9": {"pass": None}},
                "overall_pass": False,
                "n_rules": 12,
                "items": [{"id": "first"}, {"id": "second"}],
                "dose": 65810000.0,
                "ratio": 10.776487,
                "frac": 0.0062,
            }
        ),
        encoding="utf-8",
    )
    tdir = tmp_path / "docs" / "templates"
    tdir.mkdir(parents=True)
    (tdir / "T.md.j2").write_text(body, encoding="utf-8")
    rc = render(tmp_path)
    out = tmp_path / "docs" / "T.md"
    return rc, out.read_text(encoding="utf-8") if out.exists() else ""


Q_CASES = [
    ("{{ sealed/finfet.json | metrics['top1_success'].value | pct }}%", "47.5%"),
    ("{{ sealed/finfet.json | metrics.top1_success.n | int }}", "200"),
    ("{{ sealed/finfet.json | metrics['top1_success'].ci_high | pct0 }}%", "55%"),
    ("{{ sealed/decisions.json | frac | pct2 }}%", "0.62%"),
    ("{{ sealed/decisions.json | ratio | x1 }}×", "10.8×"),
    ("{{ sealed/decisions.json | dose | g }}", "65,810,000"),
    ("{{ sealed/decisions.json | dose }}", "65810000"),
    ("{{ sealed/decisions.json | ratio }}", "10.776487"),
    ("{{ sealed/decisions.json | n_rules | int }}", "12"),
    ("{{ sealed/decisions.json | items.1.id }}", "second"),
    ("{{ sealed/decisions.json | items[0].id }}", "first"),
    ("{{ sealed/decisions.json | rules.D1.detail }}", "lower=0.387"),
    ("`{{ sealed/decisions.json | overall_pass }}`", "`false`"),
    ("{{ results/sealed/decisions.json | rules['D1'].pass }}", "false"),
    ("{{ accuracy/g2.json | metrics['e2e_p50'].value | ms0 }} ms", "23 ms"),
    ("{{ accuracy/g2.json | metrics['latency_s'].value | ms0 }} ms", "42 ms"),
    ("{{ sealed/finfet.json | protocol_sha256 }}", None),
]


@pytest.mark.parametrize("src, shown", Q_CASES)
def test_a_grammar_placeholders(tmp_path, capsys, src, shown):
    rc, out = _q_ws(tmp_path, "# t\n\nValue " + src + " here.\n")
    assert rc == 0, capsys.readouterr().err
    line = out.splitlines()[3]
    visible = re.sub(r"<!--src:[^>]*-->", "", line)
    if shown is not None:
        assert visible == f"Value {shown} here.", line
    assert "<!--src:results/" in line and " q" in line
    rc, rep = claims(tmp_path, capsys, "--no-render-check")
    assert rc == 0 and rep["findings"] == [], rep["findings"]


def test_a_grammar_marker_detects_tampering_and_missing_paths(tmp_path, capsys):
    rc, _ = _q_ws(tmp_path, "# t\n\nA {{ sealed/decisions.json | frac | pct2 }}%.\n")
    assert rc == 0
    p = tmp_path / "docs" / "T.md"
    good = p.read_text(encoding="utf-8")
    p.write_text(good.replace("0.62<!--", "0.72<!--"), encoding="utf-8")
    rc, rep = claims(tmp_path, capsys, "--no-render-check")
    assert rc == 1 and "CL002" in codes(rep)
    p.write_text(good.replace("#frac q", "#fracc q"), encoding="utf-8")
    rc, rep = claims(tmp_path, capsys, "--no-render-check")
    assert rc == 1 and "CL003" in codes(rep)


@pytest.mark.parametrize(
    "src, needle",
    [
        ("{{ sealed/decisions.json | nope }}", "key 'nope' not found"),
        ("{{ sealed/decisions.json | items.7.id }}", "out of range"),
        ("{{ sealed/decisions.json | rules }}", "point the path at a scalar"),
        ("{{ sealed/decisions.json | rules.D9.pass }}", "null"),
        ("{{ sealed/decisions.json | dose | pctt }}", "unknown placeholder fmt"),
        ("{{ sealed/decisions.json | overall_pass | pct }}", "needs a number"),
        ("{{ sealed/missing.json | x }}", "not found"),
        ("`see {{ sealed/decisions.json | dose }}`", "inside inline code"),
    ],
)
def test_a_grammar_errors_are_located(tmp_path, capsys, src, needle):
    rc, _ = _q_ws(tmp_path, "# t\n\n" + src + "\n")
    assert rc == 1
    err = capsys.readouterr().err
    assert "docs/templates/T.md.j2:3" in err and needle in err, err


def test_a_grammar_mixes_with_m_and_keeps_html_comments_verbatim(tmp_path, capsys):
    body = (
        "<!--\n  grammar: {{ <relpath> | <jsonpath> | <fmt?> }}\n-->\n\n"
        "# t\n\nA {{ sealed/decisions.json | n_rules | int }} and "
        "{{ m('accuracy/g2', 'speedup') }}.\n"
    )
    rc, out = _q_ws(tmp_path, body)
    assert rc == 0, capsys.readouterr().err
    assert "<!--\n  grammar: {{ <relpath> | <jsonpath> | <fmt?> }}\n-->" in out
    assert "A 12<!--src:results/sealed/decisions.json#n_rules q fmt=int--> and " + G2_SPEEDUP in out
    rc, rep = claims(tmp_path, capsys)
    assert rc == 0, rep["findings"]


def test_dev_number_via_a_grammar_in_sealed_sentence_fails(tmp_path, capsys):
    rc, _ = _q_ws(
        tmp_path,
        "# t\n\nOn the sealed set it was {{ accuracy/g2.json | metrics.speedup.value | x1 }}×.\n",
    )
    assert rc == 1 and "sealed" in capsys.readouterr().err


def test_logic_helpers_for_plain_json(tmp_path, capsys):
    body = (
        "{% set D = 'sealed/decisions.json' %}"
        "{{ exists(D) }} {{ exists('sealed/decisions') }} {{ exists(D, 'rules.D1.pass') }} "
        "{{ exists(D, 'rules.D7') }} {{ exists('bench/nope') }}\n"
        "{{ flag(D, 'overall_pass') }} {{ flag(D, 'rules.D9.pass') }}\n"
        "{{ keys(D, 'rules') | join(',') }} {{ keys(D, 'items') | join(',') }}\n"
        "{{ kind(D, 'rules') }} {{ kind(D, 'items') }} {{ kind(D, 'n_rules') }} "
        "{{ kind(D, 'rules.D9.pass') }} {{ kind(D, 'nope') }}\n"
    )
    rc, out = _q_ws(tmp_path, body)
    assert rc == 0, capsys.readouterr().err
    lines = out.splitlines()[1:]
    assert lines[0] == "True True True False False"
    assert lines[1] == "False None"
    assert lines[2] == "D1,D9 0,1"
    assert lines[3] == "object array number null missing"


def test_plain_json_is_not_a_results_file_but_broken_results_file_is(ws, capsys):
    (ws / "results" / "sealed" / "decisions.json").write_text(
        json.dumps({"overall_pass": False}), encoding="utf-8"
    )
    assert render(ws) == 0
    rc, _ = _render_one(ws, "{{ m('sealed/decisions', 'overall_pass') }}\n")
    assert rc == 1 and "not found" in capsys.readouterr().err
    (ws / "results" / "sealed" / "broken.json").write_text(
        json.dumps({"schema_id": "driftsense.results/1", "metrics": {}}), encoding="utf-8"
    )
    assert render(ws) == 1
    assert "broken.json: invalid ResultsFile" in capsys.readouterr().err


def test_root_submission_template_needs_flag_then_stays_generated(ws, capsys):
    write_template(ws, "SUBMISSION.md.j2", "# Devpost\n\nSpeed-up {{ m('accuracy/g2', 'speedup') }}.\n")
    (ws / "SUBMISSION.md").write_text("# legacy, hand-written\n", encoding="utf-8")
    assert render(ws) == 0
    assert (ws / "SUBMISSION.md").read_text(encoding="utf-8") == "# legacy, hand-written\n"
    assert "skipped docs/templates/SUBMISSION.md.j2" in capsys.readouterr().err
    assert render(ws, "--include-readme") == 0
    assert G2_SPEEDUP in (ws / "SUBMISSION.md").read_text(encoding="utf-8")
    assert not (ws / "docs" / "SUBMISSION.md").exists()
    assert render(ws, "--check") == 0  # generated root file is now picked up automatically
    rc, rep = claims(ws, capsys)
    assert rc == 0, rep["findings"]


# --------------------------------------------------------------------------- logic helpers


def _render_one(ws: Path, body: str, *extra: str) -> tuple[int, str]:
    only_templates(ws)
    write_template(ws, "T.md.j2", body)
    rc = render(ws, *extra)
    out = ws / "docs" / "T.md"
    return rc, out.read_text(encoding="utf-8") if out.exists() else ""


def test_logic_helpers(ws):
    body = (
        "{% set B = 'bench/cool_bench' %}"
        "{{ has(B) }} {{ has(B, 'meta.n_arms') }} {{ has(B, 'nope') }} {{ has('x/y') }}\n"
        "{{ metrics(B, 'cost.x86-pip.') | join(',') }}\n"
        "{{ ci_verdict(B, '" + CMP + ".kleidicv.geomean.tN') }} "
        "{{ ci_verdict(B, '" + CMP + ".numpy.geomean.tN') }} "
        "{{ ci_verdict(B, '" + CMP + ".control.geomean.t1') }} "
        "{{ ci_verdict(B, '" + CMP + ".e2e.throughput') }} "
        "{{ ci_verdict(B, 'meta.n_arms') }}\n"
        "{{ val(B, 'meta.n_arms') == 3 }} {{ val(B, 'meta.n_arms') > 2.5 }} "
        "{{ val(B, 'meta.n_arms', 'n') == 3 }} {{ val(B, 'meta.git_sha_consistent') | string "
        "if false else 'ok' }}\n"
    )
    rc, text = _render_one(ws, body)
    assert rc == 0, text
    lines = text.splitlines()[1:]
    assert lines[0] == "True True False False"
    assert lines[1] == "cost.x86-pip.usd_per_1000,cost.x86-pip.usd_per_hour"
    assert lines[2] == "higher includes lower no_ci no_ci"
    assert lines[3] == "True True False ok"


def test_val_cannot_be_printed(ws, capsys):
    rc, _ = _render_one(ws, "# t\n\n{{ val('bench/cool_bench', 'meta.n_arms') }}\n")
    assert rc == 1
    err = capsys.readouterr().err
    assert "docs/templates/T.md.j2:3" in err and "cannot be printed" in err


def test_logic_helpers_missing_keys(ws, capsys):
    rc, _ = _render_one(ws, "# t\n\n{{ ci_verdict('bench/cool_bench', 'nope') }}\n")
    assert rc == 1 and "metric 'nope' not in" in capsys.readouterr().err
    rc, text = _render_one(
        ws,
        "{{ ci_verdict('x/y', 'a') }} {{ metrics('x/y') | length }} "
        "{{ 'yes' if val('x/y', 'a') else 'no' }}\n",
        "--allow-missing",
    )
    assert rc == 0 and text.splitlines()[1] == "no_ci 0 no"


def test_sidecar_meta_json_is_not_a_results_file(ws, capsys):
    assert (ws / "results" / "bench" / "cool_bench_meta.json").exists()
    rc, _ = _render_one(ws, "{{ m('bench/cool_bench_meta', 'x') }}\n")
    assert rc == 1 and "not found" in capsys.readouterr().err
    (ws / "results" / "bench" / "broken_meta.json").write_text("{not json", encoding="utf-8")
    assert render(ws) == 1
    assert "sidecar is not valid JSON" in capsys.readouterr().err
