"""Application code shipped by the CDK app.

Two kinds of bundle are built at synth time under ``infra/.build/`` (gitignored):

* **EC2 app bundle** (worker / bench): the allow-listed repo paths that exist are copied
  into ``.build/<name>/`` and that directory becomes an ``aws_s3_assets.Asset``. The asset
  hash changes exactly when shipped code changes, which changes the user-data and rolls
  the instances. Python dependencies are installed on the instance from the lock files.

* **API Lambda bundle**: ``service/`` plus the cv2-free part of ``driftsense`` that
  ``service.api`` imports (``driftsense/__init__.py`` + ``driftsense/contracts.py``), and the
  API's third-party deps (fastapi, mangum, pydantic and their closure) pip-installed for
  ``manylinux aarch64 / CPython 3.12`` **without Docker**::

      pip install --target <dir> --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64
          --implementation cp --python-version 3.12 --only-binary=:all: -c requirements.lock
          fastapi==<lock> mangum==<lock> pydantic==<lock>

  boto3/botocore come from the Lambda runtime. The result is cached by an input hash
  (pins + flags + source bytes) so a re-synth with unchanged inputs does not touch pip.
  Tests never call pip: they pass ``-c api_asset=placeholder`` (a stub handler).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from aws_cdk import aws_s3_assets as s3_assets
from constructs import Construct

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = Path(__file__).resolve().parents[1] / ".build"

# driftsense imports these top-level modules (sim/instrument.py: config, data_gen; v1 adapter and
# v2 cascade/engine: localization), so they ship with the package.
ENGINE_PATHS: tuple[str, ...] = ("driftsense", "localization", "data_gen", "config.py", "stage2_config.py")
WORKER_PATHS: tuple[str, ...] = (
    "service",
    *ENGINE_PATHS,
    "bench",
    "requirements-cool.lock",
    "pyproject.toml",
)
BENCH_PATHS: tuple[str, ...] = WORKER_PATHS + ("requirements.lock",)

EXCLUDE_PATTERNS: tuple[str, ...] = (
    "tests",
    "test_*.py",
    "__pycache__",
    "*.pyc",
    ".venv",
    "venv",
    "outputs",
    "data",
    "out",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
)

MARKER = "BUNDLE_CONTENTS.txt"


def stage(
    name: str, paths: Sequence[str], repo_root: Path = REPO_ROOT, build_dir: Path = BUILD_DIR
) -> tuple[Path, list[str]]:
    """Copy the existing allow-listed paths into ``build_dir/name``; return (dir, included paths)."""
    target = build_dir / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    ignore = shutil.ignore_patterns(*EXCLUDE_PATTERNS)
    included: list[str] = []
    for rel in paths:
        src = repo_root / rel
        if src.is_dir():
            shutil.copytree(src, target / rel, ignore=ignore)
            included.append(rel + "/")
        elif src.is_file():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target / rel)
            included.append(rel)

    # Always non-empty and self-describing (also makes an empty bundle obvious on the instance).
    (target / MARKER).write_text(
        "\n".join(included) + "\n" if included else "(no app paths existed at synth time)\n",
        encoding="utf-8",
    )
    return target, included


def app_asset(
    scope: Construct, construct_id: str, name: str, paths: Iterable[str]
) -> tuple[s3_assets.Asset, list[str]]:
    directory, included = stage(name, tuple(paths))
    return s3_assets.Asset(scope, construct_id, path=str(directory)), included


# --------------------------------------------------------------------------- API Lambda bundle

API_HANDLER = "service.api.handler"
# What service.api imports from the repo. driftsense/api.py, v2/, engines/ (cv2/numpy) stay out.
API_SOURCE_PATHS: tuple[str, ...] = ("service", "driftsense/__init__.py", "driftsense/contracts.py")
# service/ modules the Lambda never imports (worker side or local tooling).
API_SOURCE_EXCLUDES: tuple[str, ...] = ("worker.py", "e2e_local.py", "README.md")
API_REQUIREMENTS: tuple[str, ...] = ("fastapi", "mangum", "pydantic")
# Never allowed in the Lambda bundle: provided by the runtime (boto3) or must not be imported (cv2).
API_FORBIDDEN_TOP_LEVEL: tuple[str, ...] = ("cv2", "numpy", "boto3", "botocore")
LAMBDA_PIP_FLAGS: tuple[str, ...] = (
    "--platform",
    "manylinux2014_aarch64",
    "--platform",
    "manylinux_2_28_aarch64",
    "--implementation",
    "cp",
    "--python-version",
    "3.12",
    "--only-binary=:all:",
    "--no-compile",
    "--disable-pip-version-check",
    "--no-input",
)
LOCK_FILE = "requirements.lock"
STAMP = ".bundle-inputs.sha256"

_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;#]+)")


class BundleError(RuntimeError):
    pass


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(lock: Path) -> dict[str, str]:
    """``{normalized name: version}`` for every ``name==version`` line in a pip-compile lock."""
    pins: dict[str, str] = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        m = _PIN_RE.match(line.strip())
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    return pins


def api_requirements(lock: Path) -> list[str]:
    pins = lock_pins(lock)
    missing = [r for r in API_REQUIREMENTS if _norm(r) not in pins]
    if missing:
        raise BundleError(f"{lock.name} has no pin for {missing}; regenerate the lock")
    return [f"{r}=={pins[_norm(r)]}" for r in API_REQUIREMENTS]


def pip_command(target: Path, lock: Path, python: str = sys.executable) -> list[str]:
    return [
        python,
        "-m",
        "pip",
        "install",
        "--target",
        str(target),
        *LAMBDA_PIP_FLAGS,
        "-c",
        str(lock),
        *api_requirements(lock),
    ]


def _source_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in API_SOURCE_PATHS:
        src = repo_root / rel
        if src.is_file():
            files.append(src)
        elif src.is_dir():
            for f in sorted(src.rglob("*.py")):
                parts = f.relative_to(repo_root).parts
                if "__pycache__" in parts or "tests" in parts or f.name in API_SOURCE_EXCLUDES:
                    continue
                files.append(f)
    return files


def _inputs_hash(repo_root: Path, lock: Path) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([list(LAMBDA_PIP_FLAGS), api_requirements(lock)]).encode())
    for f in _source_files(repo_root):
        h.update(f.relative_to(repo_root).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def service_available(repo_root: Path = REPO_ROOT) -> bool:
    return (repo_root / "service" / "api.py").is_file() and (
        repo_root / "driftsense" / "contracts.py"
    ).is_file()


def build_api_bundle(
    repo_root: Path = REPO_ROOT,
    build_dir: Path = BUILD_DIR,
    run: Callable[..., object] = subprocess.run,
    python: str = sys.executable,
) -> Path:
    """Build (or reuse) ``build_dir/api_bundle`` for the arm64 Python 3.12 Lambda; return the directory."""
    if not service_available(repo_root):
        raise BundleError(
            "api_asset=service needs service/api.py and driftsense/contracts.py in the repo; "
            "use -c api_asset=placeholder for a stub API"
        )
    lock = repo_root / LOCK_FILE
    if not lock.is_file():
        raise BundleError(f"{LOCK_FILE} not found at the repo root")

    target = build_dir / "api_bundle"
    digest = _inputs_hash(repo_root, lock)
    stamp = target / STAMP
    if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == digest:
        return target

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    cmd = pip_command(target, lock, python)
    print("api bundle: " + " ".join(cmd), file=sys.stderr)
    run(cmd, check=True)

    for f in _source_files(repo_root):
        dest = target / f.relative_to(repo_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)

    for path in list(target.rglob("__pycache__")):
        shutil.rmtree(path, ignore_errors=True)
    forbidden = [name for name in API_FORBIDDEN_TOP_LEVEL if (target / name).exists()]
    if forbidden:
        raise BundleError(f"API bundle must not contain {forbidden} (pulled in by a dependency?)")
    stamp.write_text(digest + "\n", encoding="utf-8")
    return target


def api_bundle_dir(
    api_asset: str, placeholder: Path, repo_root: Path = REPO_ROOT, **kwargs
) -> tuple[Path, str, str | None]:
    """Resolve the API code directory: ``(dir, handler, warning)`` for ``api_asset`` auto/service/placeholder."""
    if api_asset == "placeholder":
        return placeholder, "handler.handler", None
    if api_asset == "auto" and not service_available(repo_root):
        return (
            placeholder,
            "handler.handler",
            "service/api.py is not in this checkout: the API Lambda gets the PLACEHOLDER handler. "
            "Deploy from a checkout that contains service/ (or pass -c api_asset=service to fail instead).",
        )
    return build_api_bundle(repo_root=repo_root, **kwargs), API_HANDLER, None
