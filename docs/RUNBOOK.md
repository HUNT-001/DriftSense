# DriftSense Pilot — Person A RUNBOOK (follow top to bottom)

Single source of truth for *how to proceed*. Each step says WHERE it runs
(CC = Claude Code paid · FREE = free-tier LLM · PASTE = copy from PHASE.md · YOU = by hand),
the exact prompt, and when to `/clear`. Companion files: PERSON_A_PLAYBOOK.md (why),
PROMPT_PACK.md (reusable prompt bodies).

Repo base: **D:\DriftSense-clean**, branch `fresh-main`. Tag `v1-munichtech` created. ✅

Already in the repo (done): driftsense/ package + v1_adapter + v2 stub, Makefile,
pyproject.toml, requirements.in, these docs. Old requirements.txt deleted. ✅

---

## STEP 0 — one-time machine setup (YOU, once per machine)
```bash
git config driftsense.owner A
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```
(Full AWS/SSO + CLAUDE.local.md setup = PHASE.md §5. B does the AWS admin side.)

---

## STEP 1 — BOOTSTRAP session  (CC + PASTE)  ← START HERE
Open Claude Code in the repo. Paste the Bootstrap prompt (**PROMPT_PACK.md §3**).
When CC asks, paste these verbatim blocks from PHASE.md, one at a time:
- S1 block: CLAUDE.md, .gitignore additions
- S2 block: .claude/** (hooks, settings.json, commands/*.md, agents/verifier.md)
- S3 block: .github/ci.yml, CODEOWNERS, pull_request_template.md
- S4 block: contracts.py, api.py, engines/fake.py, tests/contract/test_contracts.py
- docs/PLAN.md: the §4 schedule table

CC then runs `make lock && make contracts && make check` and opens the PR.
Do NOT let CC rewrite v1_adapter.py / v2 / Makefile / pyproject / requirements.in — they exist.
**Acceptance:** `make check` green (v2 contract test may skip), PR `shared/p0-bootstrap` open.
→ **/clear** after the PR is open.

## STEP 2 — rulesets  (PASTE + CC)
In a fresh CC session: paste PHASE.md S5. Paste the ruleset JSON, let CC run the `gh api` calls.
→ **/clear** after.  Then deliberately test the guards (ask CC to edit driftsense/v1 or push main — both must be blocked).

---

## STEP 3 — phase loop (repeat for A1 … A10)
For each phase, new CC session:
1. Prepend **PROMPT_PACK.md §2 hardening preamble**.
2. Paste the matching **PHASE.md prompt (A1, A2, …)**.
3. CC: plan → small steps → `make check` → `/open-pr`.
4. **/clear** at the PR boundary (gate phases: clear after the tag).

Offload BEFORE each CC session where marked — generate with a FREE prompt (PROMPT_PACK §4),
paste the file in, then let CC only wire + run:

| Phase | CC does | Offload to FREE first |
|---|---|---|
| A1 profiling | profiler + baseline JSON (must run) | test skeleton `test_profile_outputs.py` (T1) |
| A2 S0–S2 | ingest/lattice/rectify algorithm | synthetic-lattice fixtures, docstrings (T1) |
| A3 S3–S6 | roughness/score/cascade/contour | test fixtures (T1) |
| A4 S8/S9+G2 | subpixel/explain/gate (must run) | alt-text, colormap helper text |
| A5 DRAM | contour algorithm + diagnose | — |
| A6 confidence | calibration/τ (must run on CALIB) | — |
| A7 VirtualSEM | deterministic twin | virtual_sem.md prose (T2) |
| A8 agent+MCP+G5 | tools/rules/runner/trace/MCP (must run) | MCP README, LLM-planner prompt text (T2) |
| A9 protocol+sealed | ALL in CC, never offload | protocol.md prose, no numbers (T2) |
| A10 report | `make docs && make claims` only | REPORT.md, PRIOR_WORK.md prose w/ {{placeholders}} (T2) |

---

## Golden rules (token economy)
- One phase = one session = one `/clear` at the PR boundary. Never two phases in one context.
- Never regenerate anything verbatim in PHASE.md — PASTE it.
- Free LLMs: one spec + one file, no repo, no numbers. CC verifies with `make check`.
- Numbers come from a script run ONCE into results/*.json — never typed, never eyeballed by re-running.
- §2 preamble on every CC prompt → it plans first and stops after 2 failed tries (kills retry spirals).

## Handy
- Resume same unfinished branch next day: `claude -c` (don't clear mid-branch).
- Stuck/loop: stop CC, read the error yourself or send it to a FREE LLM, come back with the fix.
