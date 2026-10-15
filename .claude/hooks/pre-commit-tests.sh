#!/usr/bin/env bash
# PreToolUse(Bash): run fast tests before any git commit; block on failure.
set -uo pipefail
CMD=$(jq -r '.tool_input.command // empty')
grep -Eq '(^|[;&|[:space:]])git[[:space:]]+commit' <<<"$CMD" || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
if ! make -s test-fast >/tmp/driftsense-test-fast.log 2>&1; then
  tail -n 40 /tmp/driftsense-test-fast.log >&2
  echo "BLOCKED: 'make test-fast' failed. Fix the failures, then commit." >&2
  exit 2
fi
exit 0
