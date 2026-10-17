#!/usr/bin/env bash
# PreToolUse(Edit|Write|MultiEdit): enforce frozen paths, generated files and ownership.
set -uo pipefail
deny() { echo "BLOCKED (DriftSense guard-edit): $1" >&2; exit 2; }
INPUT=$(cat)
# jq-free, fail-closed: resolve a Python interpreter (python, then python3); DENY if none.
PY=""
for c in python python3; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
[[ -z "$PY" ]] && deny "no python interpreter found; cannot parse hook input (fail-closed)."
FP=$(printf '%s' "$INPUT" | "$PY" -c "import sys,json; d=(json.load(sys.stdin).get('tool_input') or {}); print(d.get('file_path') or d.get('notebook_path') or '')" 2>/dev/null)
rc=$?
[[ $rc -ne 0 ]] && deny "could not parse hook JSON input (fail-closed)."
[[ -z "$FP" ]] && exit 0
FP="${FP//\\//}"
REL="${FP#"$CLAUDE_PROJECT_DIR"/}"
OWNER=$(git -C "$CLAUDE_PROJECT_DIR" config --get driftsense.owner || echo "")
BRANCH=$(git -C "$CLAUDE_PROJECT_DIR" branch --show-current 2>/dev/null || echo "")

case "$REL" in
  driftsense/v1/*)            deny "driftsense/v1 is the frozen golden reference (v1-munichtech).";;
  localization/*|experiments/*|data_gen/*|config.py|stage2_config.py) deny "frozen v1 reference (v1-munichtech) — localization/, experiments/, data_gen/, config.py, stage2_config.py are read-only.";;
  eval/sealed/*|data/sealed/*|*SEALED_UNLOCK*) deny "sealed data/unlock files are never touched by an agent.";;
  requirements*.lock)         deny "lockfiles are generated: edit requirements.in and run 'make lock'.";;
  results/*.json|results/*/*.json) deny "results are generated: rerun the producing script.";;
esac
[[ "$BRANCH" == "main" ]] && deny "you are on main. Create a branch first (git switch -c a/pN-topic or b/pN-topic)."

while read -r pattern who; do
  [[ -z "${pattern:-}" || "$pattern" == \#* ]] && continue
  # shellcheck disable=SC2053
  if [[ "$REL" == $pattern ]]; then
    case "$who" in
      any) exit 0;;
      shared) [[ "$BRANCH" == shared/* || "$BRANCH" == int/* ]] || deny "$REL is a SHARED file. Use a shared/<topic> branch agreed at sync (see CLAUDE.md).";;
      *) if [[ -n "$OWNER" && "$who" != "$OWNER" ]]; then deny "$REL is owned by Person $who. Run /handoff to request the change."; fi;;
    esac
    exit 0
  fi
done < "$CLAUDE_PROJECT_DIR/.claude/ownership.txt"
exit 0
