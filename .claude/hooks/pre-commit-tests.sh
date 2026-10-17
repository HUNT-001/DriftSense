#!/usr/bin/env bash
# PreToolUse(Bash): run fast tests before any git commit; block on failure.
set -uo pipefail
deny() { echo "BLOCKED (DriftSense pre-commit-tests): $1" >&2; exit 2; }
INPUT=$(cat)
# jq-free, fail-closed: resolve a Python interpreter (python, then python3); DENY if none.
PY=""
for c in python python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
[[ -z "$PY" ]] && deny "no python interpreter found; cannot parse hook input (fail-closed)."
CMD=$(printf '%s' "$INPUT" | "$PY" -c "import sys,json; d=json.load(sys.stdin); print((d.get('tool_input') or {}).get('command') or '')" 2>/dev/null)
rc=$?
[[ $rc -ne 0 ]] && deny "could not parse hook JSON input (fail-closed)."
grep -Eq '(^|[;&|[:space:]])git[[:space:]]+commit' <<<"$CMD" || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
if ! make -s test-fast >/tmp/driftsense-test-fast.log 2>&1; then
  tail -n 40 /tmp/driftsense-test-fast.log >&2
  echo "BLOCKED: 'make test-fast' failed. Fix the failures, then commit." >&2
  exit 2
fi
exit 0
