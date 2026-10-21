#!/usr/bin/env bash
# Run ONE COOL benchmark arm end to end on Ubuntu 24.04 and (optionally) upload it to S3.
#
#   BENCH_ARM=arm-cool BENCH_RUN_ID=r20261021a bash bench/cool/run_all.sh
#
# Contract (the infra bench stack calls exactly `bash bench/cool/run_all.sh` from the repo root):
#   BENCH_ARM      required. One of:
#                    x86-pip         c7i.4xlarge, pip venv from requirements.lock (opencv-python-headless)
#                    arm-pip         c8g.4xlarge, pip venv from requirements.lock
#                    arm-cool        c8g.4xlarge COOL AMI, venv on /opt/cool/venvs/python_3.12 +
#                                    requirements-cool.lock (OpenCV is NEVER pip-installed)
#                    arm-cool-large  c8g.large   COOL AMI, same as arm-cool (production size, cost claim)
#   BENCH_RUN_ID   required. [A-Za-z0-9._-]+, the same for every arm of one campaign.
#   BENCH_OUT_DIR  optional. Output root (default: out/$BENCH_RUN_ID). This arm writes to
#                  $BENCH_OUT_DIR/$BENCH_ARM/ and $BENCH_OUT_DIR/$BENCH_ARM.tar.gz.
#   BENCH_S3_URI   optional. s3:// prefix. Uploads with `aws s3 cp` to
#                  $BENCH_S3_URI/$BENCH_RUN_ID/$BENCH_ARM/ (directory) and .../$BENCH_ARM.tar.gz.
# Optional knobs:
#   BENCH_QUICK=1          smoke run (few runs, small counts): plumbing only, never results.
#   BENCH_SKIP_VENV=1      use $PYTHON as is (no venv/pip); every guard below still applies.
#   BENCH_VENV_DIR         venv location (default: $HOME/.venvs/driftsense-bench-$BENCH_ARM).
#   BENCH_BASE_PYTHON      interpreter for the pip venvs (default: python3.12, else python3).
#   BENCH_ALLOW_MISMATCH=1 continue when the CPU arch does not match the arm (never relaxes the
#                          COOL guard: a *cool arm always aborts unless cv2 is under /opt/cool).
#   ENGINE_GIT_SHA         recorded git sha (default: git rev-parse HEAD).
# Exit codes: 0 ok; 1 a benchmark step failed (partial results still uploaded); 2 bad input;
#             3 guard tripped (wrong OpenCV for the arm / opencv found in the COOL venv).
# After all arms finish:  python -m bench.cool.report --run-dir $BENCH_OUT_DIR --out results/bench/
set -euo pipefail

die() { echo "run_all.sh: $2" >&2; exit "$1"; }

[[ -f bench/cool/run_all.sh ]] || die 2 "run from the repository root"
ARM="${BENCH_ARM:-}"
RUN_ID="${BENCH_RUN_ID:-}"
case "$ARM" in
  x86-pip|arm-pip|arm-cool|arm-cool-large) ;;
  *) die 2 "BENCH_ARM must be x86-pip|arm-pip|arm-cool|arm-cool-large (got '$ARM')" ;;
esac
[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || die 2 "BENCH_RUN_ID must match [A-Za-z0-9._-]+ (got '$RUN_ID')"
OUT_ROOT="${BENCH_OUT_DIR:-out/$RUN_ID}"
S3_URI="${BENCH_S3_URI:-}"
if [[ -n "$S3_URI" && "$S3_URI" != s3://* ]]; then
  die 2 "BENCH_S3_URI must start with s3:// (got '$S3_URI')"
fi
COOL_ROOT=/opt/cool
COOL_PY="$COOL_ROOT/venvs/python_3.12/bin/python"
IS_COOL=0
[[ "$ARM" == *cool* ]] && IS_COOL=1

# ---------------------------------------------------------------- arch guard
MACHINE="$(uname -m 2>/dev/null || echo unknown)"
case "$MACHINE" in x86_64|amd64|AMD64) MACHINE=x86_64 ;; aarch64|arm64) MACHINE=aarch64 ;; esac
WANT_ARCH=aarch64
[[ "$ARM" == x86-* ]] && WANT_ARCH=x86_64
if [[ "$MACHINE" != "$WANT_ARCH" ]]; then
  if [[ "${BENCH_ALLOW_MISMATCH:-0}" == "1" ]]; then
    echo "run_all.sh: WARNING: arm $ARM expects $WANT_ARCH, host is $MACHINE (BENCH_ALLOW_MISMATCH=1)" >&2
  else
    die 3 "arm $ARM expects a $WANT_ARCH host, this one is $MACHINE"
  fi
fi

# ---------------------------------------------------------------- interpreter / venv
if [[ "${BENCH_SKIP_VENV:-0}" == "1" ]]; then
  if [[ -z "${PYTHON:-}" ]]; then
    if command -v python >/dev/null 2>&1; then PYTHON=python; else PYTHON=python3; fi
  fi
else
  VENV="${BENCH_VENV_DIR:-$HOME/.venvs/driftsense-bench-$ARM}"
  if [[ "$IS_COOL" == "1" ]]; then
    [[ -x "$COOL_PY" ]] || die 3 "COOL interpreter $COOL_PY not found: is this the COOL AMI?"
    # Guard 1: the COOL lock must not mention OpenCV at all.
    if grep -Eiq '^[[:space:]]*opencv' requirements-cool.lock; then
      die 3 "requirements-cool.lock pins an opencv package: refusing (COOL's cv2 must be used)"
    fi
    "$COOL_PY" -m venv --system-site-packages "$VENV"
    PYTHON="$VENV/bin/python"
    # A venv created from a venv inherits the BASE interpreter's site-packages, not the COOL
    # venv's: add the COOL venv's site dirs explicitly (after our own, so nothing shadows cv2).
    SITE="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    "$COOL_PY" -c 'import sysconfig; p = sysconfig.get_paths(); print(p["purelib"]); print(p["platlib"])' \
      | sort -u >"$SITE/zz_cool_site.pth"
    # Guard 2: --no-deps so no transitive dependency can pull an OpenCV wheel in.
    "$PYTHON" -m pip install --quiet --upgrade pip
    "$PYTHON" -m pip install --quiet --no-deps -r requirements-cool.lock
  else
    BASE_PY="${BENCH_BASE_PYTHON:-}"
    if [[ -z "$BASE_PY" ]]; then
      if command -v python3.12 >/dev/null 2>&1; then BASE_PY=python3.12; else BASE_PY=python3; fi
    fi
    "$BASE_PY" -m venv "$VENV"
    PYTHON="$VENV/bin/python"
    "$PYTHON" -m pip install --quiet --upgrade pip
    "$PYTHON" -m pip install --quiet -r requirements.lock
  fi
fi
# The repo is used from the checkout (PYTHONPATH), never `pip install .` (its dependencies include
# opencv-python-headless, which must not land in a COOL venv).
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------- OpenCV guard (before any step)
CV2_FILE="$("$PYTHON" -c 'import cv2; print(cv2.__file__)')" || die 3 "cannot import cv2 with $PYTHON"
CV2_VERSION="$("$PYTHON" -c 'import cv2; print(cv2.__version__)')"
echo "cv2.__file__ = $CV2_FILE"
echo "cv2.__version__ = $CV2_VERSION"
if [[ "$IS_COOL" == "1" ]]; then
  [[ "$CV2_FILE" == "$COOL_ROOT"/* ]] || die 3 "arm $ARM: cv2 is not under $COOL_ROOT ($CV2_FILE): aborting"
  # Guard 3: no opencv distribution may be installed outside /opt/cool in this interpreter.
  STRAY="$("$PYTHON" - <<'PY'
from importlib import metadata
bad = []
for d in metadata.distributions():
    name = (d.metadata["Name"] or "").lower()
    loc = str(d.locate_file(""))
    if "opencv" in name and not loc.startswith("/opt/cool"):
        bad.append(f"{name} at {loc}")
print("; ".join(bad))
PY
)"
  [[ -z "$STRAY" ]] || die 3 "arm $ARM: pip-installed OpenCV found outside /opt/cool: $STRAY"
else
  [[ "$CV2_FILE" != "$COOL_ROOT"/* ]] || die 3 "arm $ARM is a pip arm but cv2 is under $COOL_ROOT"
fi

if [[ -z "${ENGINE_GIT_SHA:-}" ]]; then
  ENGINE_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
fi
export ENGINE_GIT_SHA
echo "git sha = $ENGINE_GIT_SHA"

OUT="$OUT_ROOT/$ARM"
mkdir -p "$OUT"
LOG="$OUT/run.log"
QUICK=()
if [[ "${BENCH_QUICK:-0}" == "1" ]]; then
  QUICK=(--quick)
  echo "BENCH_QUICK=1: plumbing smoke only, NOT a result" | tee -a "$LOG"
fi

FAILED=()
# A failing step is recorded (exit 1 at the end) but does not stop the remaining steps, so partial
# results and the evidence pack are still uploaded.
step() {
  echo "=== [$ARM] $* ($(date -u +%Y-%m-%dT%H:%M:%SZ))" | tee -a "$LOG"
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    echo "!!! [$ARM] step failed: $*" | tee -a "$LOG" >&2
    FAILED+=("$*")
  fi
}

{
  echo "arm=$ARM run_id=$RUN_ID cv2_file=$CV2_FILE cv2_version=$CV2_VERSION git_sha=$ENGINE_GIT_SHA"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) nproc=$(nproc 2>/dev/null || echo '?')"
} >>"$LOG"

step "$PYTHON" -m bench.cool.collect_evidence --out-dir "$OUT/evidence"
step "$PYTHON" -m bench.cool.census "${QUICK[@]}" --out-dir "$OUT"
step "$PYTHON" -m bench.cool.microbench "${QUICK[@]}" --threads 1 all --out-dir "$OUT"
step "$PYTHON" -m bench.cool.stagebench "${QUICK[@]}" --threads 1 all --out-dir "$OUT"
step "$PYTHON" -m bench.cool.e2e "${QUICK[@]}" --out-dir "$OUT"
step "$PYTHON" -m bench.cool.agent_bench "${QUICK[@]}" --out-dir "$OUT"
step "$PYTHON" -m bench.cool.equality dump --out-dir "$OUT"
cp bench/cool/prices.json "$OUT/prices.json"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  printf 'FAILED\n%s\n' "${FAILED[@]}" >"$OUT/STATUS"
else
  echo OK >"$OUT/STATUS"
fi

TARBALL="$OUT_ROOT/$ARM.tar.gz"
tar -czf "$TARBALL" -C "$OUT_ROOT" "$ARM"
echo "wrote $TARBALL"

if [[ -n "$S3_URI" ]]; then
  DEST="${S3_URI%/}/$RUN_ID/$ARM/"
  aws s3 cp "$TARBALL" "${S3_URI%/}/$RUN_ID/$ARM.tar.gz"
  aws s3 cp --recursive "$OUT/" "$DEST"
  echo "uploaded to $DEST"
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "${#FAILED[@]} step(s) failed:" >&2
  printf '  %s\n' "${FAILED[@]}" >&2
  exit 1
fi
echo "[$ARM] all steps OK"
