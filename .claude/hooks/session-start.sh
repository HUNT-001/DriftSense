#!/usr/bin/env bash
cd "$CLAUDE_PROJECT_DIR" || exit 0
git fetch -q origin 2>/dev/null || true
OWNER=$(git config --get driftsense.owner || echo "UNSET - run: git config driftsense.owner A|B")
BR=$(git branch --show-current)
COUNTS=$(git rev-list --left-right --count origin/main...HEAD 2>/dev/null || echo "? ?")
echo "DriftSense session: owner=Person ${OWNER} branch=${BR} behind_main=${COUNTS%%[[:space:]]*} ahead=${COUNTS##*[[:space:]]}"
echo "Rules: CLAUDE.md. Start with /sync. Today's schedule: docs/team/PLAN.md."
