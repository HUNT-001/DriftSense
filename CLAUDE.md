# DriftSense Pilot — team rules for Claude Code
Two humans, two machines: Person A (engine, accuracy, virtual SEM, agent, sealed eval) and Person B (AWS infra, service, COOL bench, UI, docs pipeline). Read CLAUDE.local.md to learn who you are working for.

## Git rules (non-negotiable; hooks enforce most of them)
- Never commit on or push to `main`. Work on `a/p<N>-<topic>`, `b/p<N>-<topic>`, `shared/<topic>`, `int/g<N>-<topic>`, `fix/<topic>` only.
- Start every session with /sync. Rebase only your own branch onto origin/main; push with --force-with-lease, never --force.
- One PR per branch, opened with /open-pr, squash-merged after the other person approves and CI is green on amd64 + arm64.
- Never edit paths owned by the other person (.claude/ownership.txt). If you need a change there, run /handoff.
- Shared files (contracts, api.py, requirements.*, Makefile, pyproject, CLAUDE.md, .claude/, .github/) change only on a `shared/*` branch.
- Never edit requirements*.lock by hand: edit requirements.in, run `make lock`.
- Never edit results/*.json by hand: rerun the producing script. Never type a number into docs: docs are rendered from results by scripts/render_docs.py.
- Never create, move or delete tags. Humans use /tag-gate.
- Never touch driftsense/v1/ (frozen golden reference, tag v1-munichtech).
- Never read, list or download sealed data; only eval/sealed_run.py may, and only when a human sets DRIFTSENSE_ALLOW_SEALED=1.
- Before saying "done": `make check` passes, PR body has Summary / Files / Tests / Gate evidence / Handoff sections.
- When unsure about ownership, contracts or gates: stop and ask the human.

## Interface contract
- Public entry point: driftsense.api.localize(template_bytes, search_bytes, mode, budget) -> LocalizeResult (driftsense/contracts.py). Engines: fake | v1 | v2 via DRIFTSENSE_ENGINE.
- Contract changes: shared/contract-* branch, bump CONTRACT_VERSION, `make contracts`, both approvals. Additive-only after Oct 13.

## Commands
make setup | make test-fast | make test | make check (lint+tests+contracts+claims) | make contracts | make lock | make gate-g2 | make gate-g4 | make gate-g5 | make docs

## Project constraints
- Python 3.12; opencv-python-headless==5.0.0.93 and an exact NumPy 2.x pin in requirements.lock; no torch.
- On COOL workers cv2 comes from /opt/cool/venvs/python_3.12 (requirements-cool.lock): never pip-install opencv there.
- NumPy-2 safe: no np.cross on 2-vectors.
- SEALED data only when DRIFTSENSE_ALLOW_SEALED=1 AND eval/SEALED_UNLOCK exists (enforced in code); never re-tune after unlock.
- Engine and agent are frozen at v2-freeze: no change may alter a LocalizeResult or an agent decision behind results/.
- Exact metric names (candidate_set_size, never recall@60); label every number DEV, CALIB or SEALED.
- Prefer KleidiCV-listed ops in hot paths (GaussianBlur, sepFilter2D, Sobel, erode, dilate, remap, warpPerspective,
  resize, pyrDown, medianBlur, cvtColor, minMaxIdx); claim no COOL speed-up for dft or matchTemplate.
- Hard agent safety: never commit below tau (runner and MCP), <= 6 steps, every step traced to JSONL.
- No secrets, credentials, account IDs or cdk-outputs.json in git. Deploy region ap-south-1.
