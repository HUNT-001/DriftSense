import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from bench.cool import common as C

ROOT = Path(__file__).resolve().parents[2]
RUN_ALL = ROOT / "bench" / "cool" / "run_all.sh"


def test_evidence_collector_off_ec2(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from bench.cool import collect_evidence as E

    # exercise the real IMDSv2 code path against a closed local port: must degrade to None quickly
    monkeypatch.delenv("BENCH_NO_IMDS", raising=False)
    monkeypatch.setattr(C, "IMDS_BASE", "http://127.0.0.1:9/latest")
    s = E.collect(tmp_path)
    assert s["imds"] == {k: None for k in E.IMDS_PATHS}
    for f in (
        "cv2_version.txt",
        "cv2_file.txt",
        "build_information.txt",
        "build_information_highlights.txt",
        "ldd.txt",
        "pip_freeze.txt",
        "uname.txt",
        "cpuinfo.txt",
        "imds.json",
        "cool.json",
        "evidence.json",
    ):
        assert (tmp_path / f).is_file(), f
    import cv2

    assert (tmp_path / "cv2_version.txt").read_text().strip() == cv2.__version__
    assert len((tmp_path / "build_information.txt").read_text()) > 500
    ev = json.loads((tmp_path / "evidence.json").read_text())
    assert ev["cool_detected"] is C.cv2_is_cool(cv2.__file__)
    assert ev["git_sha"] == "test-sha"


def test_cool_detection_by_path():
    assert C.cv2_is_cool("/opt/cool/venvs/python_3.12/lib/python3.12/site-packages/cv2/__init__.py")
    assert not C.cv2_is_cool("/home/ubuntu/.venvs/x/lib/python3.12/site-packages/cv2/__init__.py")


def test_cool_version_files_are_found_not_assumed(tmp_path):
    from bench.cool import collect_evidence as E

    root = tmp_path / "cool"
    (root / "share" / "doc").mkdir(parents=True)
    (root / "VERSION").write_text("3.1.0\n")
    (root / "share" / "doc" / "cool-release.txt").write_text("Cloud Optimized OpenCV 3.1\n")
    (root / "venvs" / "python_3.12").mkdir(parents=True)
    (root / "lib.so").write_text("x")
    info = E.cool_info(root)
    assert info["exists"] and info["venvs"] == ["python_3.12"]
    found = {Path(v["path"]).name: v["content"] for v in info["version_files"]}
    assert found["VERSION"].strip() == "3.1.0"
    assert "cool-release.txt" in found and "lib.so" not in found


def _cmd_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def test_run_all_never_pip_installs_opencv():
    text = RUN_ALL.read_bytes().decode("utf-8")
    assert "\r" not in text, "run_all.sh must have LF line endings"
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text
    cmds = _cmd_lines(text)
    installs = [ln for ln in cmds if re.search(r"\bpip3?\b.*\binstall\b", ln)]
    assert not [ln for ln in installs if re.search(r"opencv|cv2", ln, re.IGNORECASE)]
    # the COOL arm installs only requirements-cool.lock, with --no-deps (no transitive opencv)
    cool_installs = [ln for ln in installs if "requirements-cool.lock" in ln]
    assert cool_installs and all("--no-deps" in ln for ln in cool_installs)
    assert not any(re.search(r"pip3?\s.*install\s.*(-e\s+\.|\s\.$)", ln) for ln in cmds), (
        "never pip install ."
    )
    # the lock the COOL arm uses must not pin OpenCV
    lock = (ROOT / "requirements-cool.lock").read_text(encoding="utf-8")
    assert not re.search(r"(?im)^\s*opencv", lock)
    # guards come before the first benchmark step; the COOL guard is not overridable
    assert text.index('[[ "$CV2_FILE" == "$COOL_ROOT"/* ]]') < text.index("bench.cool.collect_evidence")
    assert (
        "BENCH_ALLOW_MISMATCH"
        not in text[
            text.index(
                "# ---------------------------------------------------------------- OpenCV guard"
            ) : text.index("ENGINE_GIT_SHA=")
        ]
    )


def test_run_all_contract_is_documented():
    text = RUN_ALL.read_text(encoding="utf-8")
    header = text.split("set -euo pipefail", 1)[0]
    for var in ("BENCH_ARM", "BENCH_RUN_ID", "BENCH_OUT_DIR", "BENCH_S3_URI", "BENCH_QUICK"):
        assert var in header, var
    for arm in ("x86-pip", "arm-pip", "arm-cool", "arm-cool-large"):
        assert arm in header
    for mod in (
        "collect_evidence",
        "census",
        "microbench",
        "stagebench",
        "e2e",
        "agent_bench",
        "equality dump",
    ):
        assert f"bench.cool.{mod}" in text, mod
    assert "aws s3 cp" in text
    readme = (ROOT / "bench" / "README.md").read_text(encoding="utf-8")
    for var in ("BENCH_ARM", "BENCH_RUN_ID", "BENCH_OUT_DIR", "BENCH_S3_URI"):
        assert var in readme
    assert (ROOT / "bench" / ".gitattributes").read_text().strip() == "*.sh text eol=lf"


def _bash() -> str | None:
    b = shutil.which("bash")
    if b and os.name == "nt" and "system32" in b.lower():  # WSL launcher, not Git Bash
        return None
    return b


def _run(env_extra: dict) -> subprocess.CompletedProcess:
    bash = _bash()
    if bash is None:
        pytest.skip("bash not available")
    env = {k: v for k, v in os.environ.items() if not k.startswith("BENCH_")}
    env.update(env_extra)
    return subprocess.run(
        [bash, "bench/cool/run_all.sh"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60
    )


def test_run_all_rejects_bad_inputs():
    r = _run({"BENCH_ARM": "graviton-cool", "BENCH_RUN_ID": "x"})
    assert r.returncode == 2 and "BENCH_ARM must be" in r.stderr
    r = _run({"BENCH_ARM": "arm-pip", "BENCH_RUN_ID": "bad id"})
    assert r.returncode == 2
    r = _run({"BENCH_ARM": "arm-pip", "BENCH_RUN_ID": "r1", "BENCH_S3_URI": "https://x"})
    assert r.returncode == 2 and "s3://" in r.stderr


def test_run_all_cool_arm_aborts_without_opt_cool_cv2():
    pytest.importorskip("cv2")
    import sys

    if C.cv2_is_cool():
        pytest.skip("running on a COOL host")
    r = _run(
        {
            "BENCH_ARM": "arm-cool",
            "BENCH_RUN_ID": "r1",
            "BENCH_SKIP_VENV": "1",
            "BENCH_ALLOW_MISMATCH": "1",
            "PYTHON": sys.executable.replace("\\", "/"),
        }
    )
    assert r.returncode == 3, r.stderr
    assert "not under /opt/cool" in r.stderr
