"""Collect the COOL evidence pack for one benchmark arm (Overall_plan.md section 7).

Writes into ``--out-dir`` (default ``out/<arm>/evidence``):

  cv2_version.txt, cv2_file.txt         cv2.__version__ / cv2.__file__ (COOL lives under /opt/cool)
  build_information.txt                 full cv2.getBuildInformation()
  build_information_highlights.txt      lines mentioning KleidiCV / HAL / NEON / SVE / IPP / dispatch ...
  ldd.txt                               ldd of the cv2 extension module (Linux only)
  pip_freeze.txt                        pip freeze (falls back to importlib.metadata)
  uname.txt, cpuinfo.txt, lscpu.txt     host identity (cpuinfo/lscpu on Linux only)
  imds.json                             IMDSv2 instance-type, ami-id, region, AZ, instance-id (null off-EC2)
  cool.json                             /opt/cool listing + any VERSION/version/RELEASE-like files found
                                        there (recorded verbatim; nothing assumed)
  evidence.json                         summary of the above (+ git sha, python, numpy, nproc)

Never fails on a missing tool: each item records the error instead.

Usage::

    python -m bench.cool.collect_evidence --out-dir out/graviton-cool/evidence
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from bench.cool import common as C

HIGHLIGHT_RX = re.compile(
    r"kleidi|hal\b|hal:|custom hal|neon|sve|asimd|fp16|bf16|dotprod|i8mm|ipp|"
    r"baseline|dispatch|parallel framework|lapack|carotene|armpl|version control|"
    r"opencv modules|python 3|files\):",
    re.IGNORECASE,
)
COOL_ROOT = Path("/opt/cool")
VERSION_NAME_RX = re.compile(
    r"(^|[._-])(version|release|build[-_]?info|manifest)([._-]|$)|^VERSION", re.IGNORECASE
)
IMDS_PATHS = {
    "instance_type": "meta-data/instance-type",
    "ami_id": "meta-data/ami-id",
    "instance_id": "meta-data/instance-id",
    "region": "meta-data/placement/region",
    "availability_zone": "meta-data/placement/availability-zone",
    "product_codes": "meta-data/product-codes",
}


def _run(cmd: list[str], timeout: float = 30.0) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return (
            r.stdout
            + (f"\n[stderr]\n{r.stderr}" if r.stderr.strip() else "")
            + ("" if r.returncode == 0 else f"\n[exit {r.returncode}]")
        )
    except (OSError, subprocess.SubprocessError) as e:
        return f"[error] {' '.join(cmd)}: {type(e).__name__}: {e}"


def _write(d: Path, name: str, text: str) -> None:
    (d / name).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def cv2_binary(cv2_file: str) -> Path | None:
    """The compiled cv2 extension (cv2*.so / cv2*.pyd) inside the cv2 package."""
    pkg = Path(cv2_file).parent
    cands = sorted(pkg.rglob("cv2*.so")) + sorted(pkg.rglob("cv2*.pyd"))
    return cands[0] if cands else None


def find_cool_version_files(root: Path = COOL_ROOT, max_depth: int = 3, limit: int = 50) -> list[dict]:
    """Every file under ``root`` (depth <= max_depth, not inside venvs/site-packages) whose name looks
    like a version/release marker. Contents recorded verbatim (first 2 KB)."""
    found: list[dict] = []
    if not root.is_dir():
        return found
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - root_depth
        dirnames[:] = [
            d
            for d in dirnames
            if depth < max_depth and d not in ("site-packages", "__pycache__") and not d.startswith("python3")
        ]
        for fn in filenames:
            if VERSION_NAME_RX.search(fn):
                p = Path(dirpath) / fn
                try:
                    content = p.read_bytes()[:2048].decode("utf-8", "replace")
                except OSError as e:
                    content = f"[unreadable: {e}]"
                found.append({"path": str(p), "content": content})
                if len(found) >= limit:
                    return found
    return found


def cool_info(root: Path = COOL_ROOT) -> dict:
    info: dict = {"root": str(root), "exists": root.is_dir()}
    if not info["exists"]:
        return info
    try:
        info["listing"] = sorted(os.listdir(root))
    except OSError as e:
        info["listing_error"] = str(e)
    venvs = root / "venvs"
    if venvs.is_dir():
        info["venvs"] = sorted(os.listdir(venvs))
    info["version_files"] = find_cool_version_files(root)
    info["note"] = (
        "COOL version recorded from files found under /opt/cool (if any); "
        "nothing is assumed. Cross-check with the AMI ID/product code in imds.json."
    )
    return info


def collect(out_dir: Path) -> dict:
    import cv2
    import numpy as np

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "cv2_version": cv2.__version__,
        "cv2_file": cv2.__file__,
        "cool_detected": C.cv2_is_cool(cv2.__file__),
        "python": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "arch": C.normalized_arch(),
        "nproc": os.cpu_count(),
        "git_sha": C.git_sha(),
        "cv2_num_threads_default": cv2.getNumThreads(),
    }
    _write(d, "cv2_version.txt", cv2.__version__)
    _write(d, "cv2_file.txt", cv2.__file__ or "")

    bi = cv2.getBuildInformation()
    _write(d, "build_information.txt", bi)
    hl = [ln for ln in bi.splitlines() if HIGHLIGHT_RX.search(ln)]
    _write(d, "build_information_highlights.txt", "\n".join(hl) or "(no matching lines)")
    summary["build_mentions_kleidicv"] = bool(re.search(r"kleidi", bi, re.IGNORECASE))
    summary["build_custom_hal_lines"] = [
        ln.strip() for ln in bi.splitlines() if re.search(r"hal", ln, re.IGNORECASE)
    ]

    so = cv2_binary(cv2.__file__ or "")
    summary["cv2_binary"] = str(so) if so else None
    if sys.platform.startswith("linux") and so and shutil.which("ldd"):
        ldd = _run(["ldd", str(so)])
        summary["ldd_mentions_kleidicv"] = bool(re.search(r"kleidi", ldd, re.IGNORECASE))
        summary["ldd_mentions_opt_cool"] = "/opt/cool" in ldd
    else:
        ldd = f"[skipped] ldd not applicable on {sys.platform} (binary: {so})"
    _write(d, "ldd.txt", ldd)

    freeze = _run([sys.executable, "-m", "pip", "freeze", "--all"])
    if freeze.startswith("[error]") or "No module named pip" in freeze:
        from importlib import metadata

        freeze = "\n".join(sorted(f"{m.metadata['Name']}=={m.version}" for m in metadata.distributions()))
        freeze = "# pip unavailable; from importlib.metadata\n" + freeze
    _write(d, "pip_freeze.txt", freeze)
    summary["opencv_dists"] = [ln for ln in freeze.splitlines() if "opencv" in ln.lower()]

    _write(d, "uname.txt", _run(["uname", "-a"]) if shutil.which("uname") else " ".join(platform.uname()))
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        lines = cpuinfo.read_text(errors="replace").splitlines()
        keep = re.compile(
            r"^(model name|cpu model|CPU implementer|CPU architecture|CPU variant|CPU part|"
            r"CPU revision|Features|flags|processor)\s*:",
            re.IGNORECASE,
        )
        sel = [ln for ln in lines if keep.match(ln)]
        uniq = list(dict.fromkeys(ln for ln in sel if not ln.lower().startswith("processor")))
        n_proc = sum(1 for ln in sel if ln.lower().startswith("processor"))
        _write(d, "cpuinfo.txt", f"processors: {n_proc}\n" + "\n".join(uniq))
        summary["cpu_features_sve"] = any(re.search(r"\bsve\b", ln) for ln in uniq)
        summary["cpu_features_sve2"] = any(re.search(r"\bsve2\b", ln) for ln in uniq)
    else:
        _write(d, "cpuinfo.txt", f"[skipped] /proc/cpuinfo not present ({platform.processor()})")
    if shutil.which("lscpu"):
        _write(d, "lscpu.txt", _run(["lscpu"]))

    imds = {k: C.imds_get(p) for k, p in IMDS_PATHS.items()}
    C.write_json(d / "imds.json", imds)
    summary["imds"] = imds

    ci = cool_info()
    C.write_json(d / "cool.json", ci)
    summary["cool_root_exists"] = ci["exists"]
    summary["cool_version_files"] = [v["path"] for v in ci.get("version_files", [])]
    C.write_json(d / "evidence.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("out") / "local" / "evidence")
    a = ap.parse_args(argv)
    s = collect(a.out_dir)
    print(
        f"[evidence] cv2 {s['cv2_version']} at {s['cv2_file']} cool={s['cool_detected']} "
        f"kleidicv_in_build={s['build_mentions_kleidicv']} instance={s['imds']['instance_type']} "
        f"ami={s['imds']['ami_id']} region={s['imds']['region']}"
    )
    print(f"[evidence] wrote {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
