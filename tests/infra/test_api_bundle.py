"""The API Lambda bundler (infra/driftsense_infra/app_bundle.py). No network: pip is faked."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from .conftest import INFRA_DIR  # noqa: F401  (puts infra/ on sys.path)

app_bundle = pytest.importorskip("driftsense_infra.app_bundle")

REPO_ROOT = Path(__file__).resolve().parents[2]

LOCK = textwrap.dedent(
    """\
    # pip-compile header
    fastapi==0.156.1
        # via -r requirements.in
    mangum==0.22.0
    numpy==2.5.4
    opencv-python-headless==5.0.0.93
    pydantic==2.14.1
    pydantic-core==2.50.1
    """
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "service").mkdir(parents=True)
    (repo / "service" / "__init__.py").write_text("")
    (repo / "service" / "api.py").write_text("handler = object()\n")
    (repo / "service" / "worker.py").write_text("import cv2\n")
    (repo / "driftsense").mkdir()
    (repo / "driftsense" / "__init__.py").write_text("")
    (repo / "driftsense" / "contracts.py").write_text("X = 1\n")
    (repo / "driftsense" / "api.py").write_text("import numpy\n")
    (repo / "requirements.lock").write_text(LOCK)
    return repo


class FakePip:
    def __init__(self, extra: tuple[str, ...] = ()) -> None:
        self.calls: list[list[str]] = []
        self.extra = extra

    def __call__(self, cmd, check):
        assert check is True
        self.calls.append(cmd)
        target = Path(cmd[cmd.index("--target") + 1])
        for pkg in ("fastapi", "mangum", "pydantic", *self.extra):
            (target / pkg).mkdir(parents=True, exist_ok=True)
            (target / pkg / "__init__.py").write_text("")


def test_requirements_pinned_from_lock(fake_repo):
    assert app_bundle.api_requirements(fake_repo / "requirements.lock") == [
        "fastapi==0.156.1",
        "mangum==0.22.0",
        "pydantic==2.14.1",
    ]


def test_pip_command_targets_lambda_arm64_without_docker(fake_repo, tmp_path):
    cmd = app_bundle.pip_command(tmp_path / "out", fake_repo / "requirements.lock", python="py")
    joined = " ".join(cmd)
    assert cmd[:4] == ["py", "-m", "pip", "install"]
    assert "--platform manylinux2014_aarch64" in joined
    assert "--python-version 3.12" in joined and "--implementation cp" in joined
    assert "--only-binary=:all:" in cmd and "--target" in cmd
    assert "-c" in cmd  # the lock as constraints: transitive versions match requirements.lock
    assert not [c for c in cmd if "opencv" in c or c.startswith(("numpy", "boto3"))]
    assert "docker" not in joined.lower()


def test_bundle_contents_and_cache(fake_repo, tmp_path):
    pip = FakePip()
    out = app_bundle.build_api_bundle(repo_root=fake_repo, build_dir=tmp_path / "build", run=pip, python="py")
    assert (out / "service" / "api.py").is_file()
    assert (out / "driftsense" / "contracts.py").is_file()
    assert not (out / "driftsense" / "api.py").exists()  # cv2/numpy engine facade stays out
    assert not (out / "service" / "worker.py").exists()
    assert (out / "fastapi").is_dir()
    assert len(pip.calls) == 1
    # unchanged inputs: cached, pip not called again
    app_bundle.build_api_bundle(repo_root=fake_repo, build_dir=tmp_path / "build", run=pip, python="py")
    assert len(pip.calls) == 1
    # a source change rebuilds
    (fake_repo / "service" / "api.py").write_text("handler = 2\n")
    app_bundle.build_api_bundle(repo_root=fake_repo, build_dir=tmp_path / "build", run=pip, python="py")
    assert len(pip.calls) == 2


def test_bundle_rejects_cv2_or_numpy(fake_repo, tmp_path):
    with pytest.raises(app_bundle.BundleError, match="cv2"):
        app_bundle.build_api_bundle(
            repo_root=fake_repo, build_dir=tmp_path / "b", run=FakePip(("cv2",)), python="py"
        )


def test_bundle_requires_service(tmp_path):
    with pytest.raises(app_bundle.BundleError, match="api_asset=placeholder"):
        app_bundle.build_api_bundle(repo_root=tmp_path, build_dir=tmp_path / "b", run=FakePip(), python="py")


def test_worker_bundle_ships_engine_top_level_modules():
    # driftsense imports config, data_gen and localization as top-level modules
    for path in ("driftsense", "localization", "data_gen", "config.py", "service", "requirements-cool.lock"):
        assert path in app_bundle.WORKER_PATHS
    assert "requirements.lock" not in app_bundle.WORKER_PATHS  # PyPI opencv lock never reaches COOL
    assert "bench" in app_bundle.BENCH_PATHS


@pytest.mark.skipif(not (REPO_ROOT / "service" / "api.py").is_file(), reason="service/ not in this checkout")
def test_real_api_sources_import_without_cv2_numpy_boto3(tmp_path):
    """Stage the real API source subset and import service.api with cv2/numpy/boto3 blocked."""
    pytest.importorskip("fastapi")
    pytest.importorskip("mangum")
    for f in app_bundle._source_files(REPO_ROOT):
        dest = tmp_path / f.relative_to(REPO_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(f.read_bytes())
    code = textwrap.dedent(
        """\
        import importlib.abc, sys
        sys.path.insert(0, sys.argv[1])
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("cv2", "numpy", "boto3"):
                    raise ImportError("blocked " + name)
        sys.meta_path.insert(0, Block())
        import service.api
        assert service.api.handler is not None
        """
    )
    result = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
