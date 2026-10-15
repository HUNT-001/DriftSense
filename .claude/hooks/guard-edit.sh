#!/usr/bin/env bash
# PreToolUse(Edit|Write|MultiEdit): enforce frozen paths, generated files and ownership.
set -uo pipefail
INPUT=$(cat)
FP=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')
[[ -z "$FP" ]] && exit 0
FP="${FP//\\//}"
REL="${FP#"$CLAUDE_PROJECT_DIR"/}"
OWNER=$(git -C "$CLAUDE_PROJECT_DIR" config --get driftsense.owner || echo "")
BRANCH=$(git -C "$CLAUDE_PROJECT_DIR" branch --show-current 2>/dev/null || echo "")
deny() { echo "BLOCKED (DriftSense guard-edit): $1" >&2; exit 2; }

case "$REL" in
  driftsense/v1/*)            deny "driftsense/v1 is the frozen golden reference (v1-munichtech).";;
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
