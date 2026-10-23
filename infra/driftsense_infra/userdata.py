"""Bash fragments for EC2 user-data (worker + bench).

All fragments run under ``set -euxo pipefail`` as root on Ubuntu (COOL 3.1 AMI or
canonical Ubuntu 24.04). CloudFormation tokens (bucket names, queue URLs) are
interpolated by CDK into an ``Fn::Join``; no ``Fn::Sub`` is involved, so bash
``${VAR}`` syntax is safe here.
"""

from __future__ import annotations

from aws_cdk import aws_s3_assets as s3_assets

COOL_PYTHON = "/opt/cool/venvs/python_3.12/bin/python"
APP_DIR = "/opt/driftsense/app"
VENV_DIR = "/opt/driftsense/venv"
VENV_PY = f"{VENV_DIR}/bin/python"

PREAMBLE = """\
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="$PATH:/snap/bin:/usr/local/bin"
"""

ENSURE_SSM_AGENT = """\
# --- SSM agent: the ONLY way in (no SSH key, no inbound SG rules) ---
if systemctl cat snap.amazon-ssm-agent.amazon-ssm-agent.service >/dev/null 2>&1; then
  systemctl enable --now snap.amazon-ssm-agent.amazon-ssm-agent.service
elif systemctl cat amazon-ssm-agent.service >/dev/null 2>&1; then
  systemctl enable --now amazon-ssm-agent.service
else
  snap install amazon-ssm-agent --classic
  systemctl enable --now snap.amazon-ssm-agent.amazon-ssm-agent.service
fi
"""

ENSURE_AWS_CLI = """\
# --- AWS CLI (needed to fetch the app asset) ---
if ! command -v aws >/dev/null 2>&1; then
  snap install aws-cli --classic || { apt-get update -y && apt-get install -y awscli; }
fi
aws --version
"""


def fetch_app(asset: s3_assets.Asset, region: str) -> str:
    return f"""\
# --- application bundle (S3 asset) ---
rm -rf {APP_DIR} && mkdir -p {APP_DIR}
aws s3 cp "s3://{asset.s3_bucket_name}/{asset.s3_object_key}" /opt/driftsense/app.zip --region {region}
python3 -m zipfile -e /opt/driftsense/app.zip {APP_DIR}
cat {APP_DIR}/BUNDLE_CONTENTS.txt
"""


COOL_LOCK_FILTERED = "/opt/driftsense/requirements-cool.filtered.txt"

COOL_VENV = f"""\
# --- venv on top of COOL's OpenCV (NEVER pip-install opencv here) ---
if [ ! -x {COOL_PYTHON} ]; then
  echo "FATAL: {COOL_PYTHON} not found; is this the COOL 3.1 AMI?" >&2
  ls -la /opt/cool /opt/cool/venvs >&2 || true
  exit 1
fi
rm -rf {VENV_DIR}
{COOL_PYTHON} -m venv --system-site-packages {VENV_DIR}
# If COOL's python_3.12 is itself a venv, --system-site-packages only exposes the *base* interpreter,
# so bridge COOL's own site-packages explicitly (appended after the venv's, so it never shadows them).
VENV_SITE="$({VENV_PY} -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
{COOL_PYTHON} -c 'import site; print("\n".join(site.getsitepackages()))' > "$VENV_SITE/_cool_site.pth"
cd {APP_DIR}
test -f requirements-cool.lock
if grep -inE '^[[:space:]]*opencv' requirements-cool.lock; then
  echo "FATAL: requirements-cool.lock pins an opencv wheel; COOL's cv2 must be the only OpenCV" >&2
  exit 1
fi
# COOL's cv2 is built against COOL's numpy: keep that numpy (drop the lock's pin) when COOL ships one.
if {COOL_PYTHON} -c 'import numpy' >/dev/null 2>&1; then
  grep -viE '^[[:space:]]*numpy[[:space:]]*(==|$)' requirements-cool.lock > {COOL_LOCK_FILTERED}
else
  cp requirements-cool.lock {COOL_LOCK_FILTERED}
fi
# --no-deps: the lock is the complete pinned closure, so nothing outside it (no transitive opencv) can sneak in.
{VENV_PY} -m pip install --disable-pip-version-check --no-input --no-deps -r {COOL_LOCK_FILTERED}
{VENV_PY} -m pip check || echo "WARN: pip check found unmet requirements (cv2/numpy are provided by COOL)" >&2
"""

ASSERT_COOL_OPENCV = f"""\
# --- guard: OpenCV must come from COOL (/opt/cool), never from pip; fail the boot otherwise ---
if {VENV_PY} -m pip list --local --format=freeze | grep -i opencv; then
  echo "FATAL: an opencv wheel was pip-installed into {VENV_DIR}; COOL's cv2 must be used" >&2
  exit 1
fi
CV2_FILE="$({VENV_PY} -c 'import cv2; print(cv2.__file__)')"
echo "cv2: $({VENV_PY} -c 'import cv2; print(cv2.__version__)') $CV2_FILE"
case "$CV2_FILE" in
  /opt/cool/*) ;;
  *) echo "FATAL: cv2 is not loaded from /opt/cool: $CV2_FILE" >&2; exit 1 ;;
esac
"""


def assert_engine(engine: str) -> str:
    """Fail the boot if the app bundle cannot import the configured engine (catches a broken bundle early)."""
    return f"""\
# --- guard: the configured engine imports from the bundle ---
cd {APP_DIR}
PYTHONPATH={APP_DIR} {VENV_PY} -c 'import service, driftsense.api as a; e = a.available_engines(); print("engines:", e); \
assert "{engine}" in e, "engine {engine} not available"'
"""


PIP_VENV = f"""\
# --- plain pip venv (comparison arm: OpenCV from PyPI wheels via requirements.lock) ---
apt-get update -y
apt-get install -y python3-venv python3-pip
rm -rf {VENV_DIR}
python3 -m venv {VENV_DIR}
cd {APP_DIR}
test -f requirements.lock
{VENV_PY} -m pip install --disable-pip-version-check --no-input -r requirements.lock
{VENV_PY} -c 'import cv2; print("cv2:", cv2.__version__, cv2.__file__)'
"""


def worker_unit(env: dict[str, str], stop_timeout_s: int = 1020) -> str:
    env_lines = "\n".join(f"Environment={key}={value}" for key, value in env.items())
    return f"""\
# --- systemd unit (logs: journald; `journalctl -u driftsense-worker` over SSM) ---
id driftsense >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/driftsense --shell /usr/sbin/nologin driftsense
cat > /etc/systemd/system/driftsense-worker.service <<'UNIT'
[Unit]
Description=DriftSense SQS worker (COOL OpenCV, Graviton)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=driftsense
WorkingDirectory={APP_DIR}
{env_lines}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH={APP_DIR}
ExecStart={VENV_PY} -m service.worker
Restart=always
RestartSec=5
KillSignal=SIGTERM
# > approval window + tool slack, so SIGTERM lets an in-flight agent episode finish
TimeoutStopSec={stop_timeout_s}

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now driftsense-worker.service
"""
