#!/usr/bin/env bash
# PreToolUse(Bash): git safety rails.
set -uo pipefail
deny() { echo "BLOCKED (DriftSense guard-bash): $1" >&2; exit 2; }
INPUT=$(cat)
# jq-free, fail-closed: resolve a Python interpreter (python, then python3); DENY if none.
PY=""
for c in python python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
[[ -z "$PY" ]] && deny "no python interpreter found; cannot parse hook input (fail-closed)."
CMD=$(printf '%s' "$INPUT" | "$PY" -c "import sys,json; d=json.load(sys.stdin); print((d.get('tool_input') or {}).get('command') or '')" 2>/dev/null)
rc=$?
[[ $rc -ne 0 ]] && deny "could not parse hook JSON input (fail-closed)."
BRANCH=$(git -C "$CLAUDE_PROJECT_DIR" branch --show-current 2>/dev/null || echo "")

if [[ "$BRANCH" == "main" ]] && grep -Eq '(^|[;&|[:space:]])git[[:space:]]+(commit|merge|cherry-pick|rebase)' <<<"$CMD"; then
  deny "never commit/merge on main. git switch -c <a|b>/p<N>-<topic> origin/main"
fi
grep -Eq 'git[[:space:]]+push([^|;&]*[[:space:]])(origin[[:space:]]+)?(HEAD:)?(refs/heads/)?main([[:space:]]|$)' <<<"$CMD" && deny "never push to main; use /open-pr."
grep -Eq 'git[[:space:]]+push[^|;&]*(--force([[:space:]]|$)|[[:space:]]-f([[:space:]]|$))' <<<"$CMD" && deny "use --force-with-lease on your own branch only."
grep -Eq 'git[[:space:]]+(reset[[:space:]]+--hard|clean[[:space:]]+-[a-z]*f)|rm[[:space:]]+-rf[[:space:]]+(/|~|\.git)' <<<"$CMD" && deny "destructive command; ask the human."
if grep -Eq 'git[[:space:]]+(tag|push)[^|;&]*(v1-munichtech|g[0-9]-pass|v2-freeze|protocol-v[0-9]|numbers-lock|rc[0-9]+|submission)' <<<"$CMD" && [[ "${DRIFTSENSE_RELEASE:-0}" != "1" ]]; then
  deny "gate tags are created only by a human via /tag-gate (start claude with DRIFTSENSE_RELEASE=1)."
fi
grep -Eq 'pip3?[[:space:]]+install[^|;&]*opencv' <<<"$CMD" && deny "never pip-install opencv ad hoc (and never on COOL hosts); use requirements.lock."
grep -Eq 'cdk[[:space:]]+destroy|aws[[:space:]].*(delete-bucket|rb[[:space:]])' <<<"$CMD" && deny "destructive AWS command; human only."
exit 0
