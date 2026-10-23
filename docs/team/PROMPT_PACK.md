# DriftSense Pilot — Person A Prompt Pack (token-efficient Claude Code usage)

How to drive Claude Code (CC) so phases finish on the first try, with the exact
`/clear` points and the free-API offload prompts. Pairs with PERSON_A_PLAYBOOK.md.

---

## 1. The /clear cadence (one phase = one session = one clear)

CC keeps the whole transcript in context and bills it every turn, so a long-lived
session gets expensive fast. Rule: **one branch/PR per session, `/clear` the moment
its PR is open (or merged).** Never carry two phases in one context.

| Session | Prompt to paste | `/clear` AFTER |
|---|---|---|
| Bootstrap | §3 below | PR `shared/p0-bootstrap` opened |
| A1 profiling | PHASE.md prompt **A1** (prepend §2 preamble) | PR opened |
| A2 S0–S2 | PHASE.md **A2** | PR opened |
| A3 S3–S6 | PHASE.md **A3** | PR opened |
| A4 S8/S9+G2 | PHASE.md **A4** | after `g2-pass` tagged |
| A5 DRAM | PHASE.md **A5** | PR opened |
| A6 confidence+freeze | PHASE.md **A6** | after `v2-freeze` |
| A7 VirtualSEM | PHASE.md **A7** | PR opened |
| A8 agent+MCP+G5 | PHASE.md **A8** | after `g5-pass` |
| A9 protocol+sealed | PHASE.md **A9** | PR opened |
| A10 report | PHASE.md **A10** | PR opened |

Also `/clear` mid-session if one debugging detour runs long and is resolved — start the
next sub-task clean. Use `claude -c` / `--resume` only to pick up the *same* unfinished
branch the next morning; otherwise clear.

---

## 2. Hardening preamble (prepend to EVERY CC phase prompt)

Copy this block above any A-prompt. It kills the usual token-wasters: sprawling edits,
guessing on ambiguity, and eyeballing numbers by re-running the engine.

```
Before coding: output a 6–10 line PLAN (files to add/change, functions, tests). If the
diff will exceed ~300 lines or you must touch a file I don't own or the contract, STOP
and ask me — do not proceed on a guess.
Work in small steps; after each file run `make test-fast`. Use KleidiCV-listed ops in hot
paths. Never print metrics from memory: write the number to results/*.json via a script,
run it ONCE, and read the JSON back.
Do NOT edit driftsense/v1, shared files, or B's paths. If a test fails twice for the same
reason, stop and show me the failure instead of trying a third variant.
Done = `make check` green + the acceptance checks below + a filled PR body. Then STOP.
```

---

## 3. Bootstrap session (exact prompt — code is already written)

I already wrote `driftsense/{__init__,engines/__init__,engines/v1_adapter,v2/__init__,v2/engine}.py`
and the `tests/contract/__init__.py`. This prompt finishes the rest:

```
Read docs/PERSON_A_PLAYBOOK.md §0 first. We are on a dirty tree; confirm with `git status`
and STOP if it is not what I expect.
1. Create branch shared/p0-bootstrap from origin/main.
2. Create these files with the EXACT contents I paste next (I will paste each): CLAUDE.md,
   .gitignore additions, .claude/** (hooks, settings.json, commands/*.md, agents/verifier.md),
   .github/{workflows/ci.yml,CODEOWNERS,pull_request_template.md}, driftsense/contracts.py,
   driftsense/api.py, driftsense/engines/fake.py, tests/contract/test_contracts.py, docs/PLAN.md,
   requirements.in. chmod +x .claude/hooks/*.sh.
3. Do NOT rewrite driftsense/engines/v1_adapter.py or driftsense/v2/*.py — they exist; read
   them and confirm they import cleanly.
4. Add pydantic>=2.8,<3 and pip-tools to requirements.in; run `make lock`.
5. `make contracts && make check` must pass (the v2 contract test may skip). Fix only real
   breakage; if the fake-engine test fails, show me — do not edit contracts to make it pass.
6. Commit in logical commits; open PR with /open-pr "chore(shared): P0 bootstrap". Then STOP.
```
Then paste the verbatim blocks from PHASE.md §(a) S1/S2/S3/S4 one at a time when it asks.
After the PR is open: `/clear`.

---

## 4. Free-API offload prompts (ChatGPT / Gemini / Claude free)

Give these ONE at a time, with no repo attached. Paste the result into the file, then let
CC only run/verify.

**T1 — unit-test skeleton from a signature**
```
Here is a Python function signature and docstring:
<paste ONE signature>
Write a pytest test module with 4–6 focused tests (happy path, boundary, bad input,
determinism). Use only pytest + numpy + cv2. No network, no fixtures beyond a small
synthetic image. Output only the code.
```

**T2 — prose doc from bullet points (REPORT / virtual_sem / RESPONSIBLE_USE)**
```
Turn these bullet points into clean technical Markdown prose for section "<name>".
Keep every numeric value as a literal placeholder token like {{ results.sealed.accuracy }}
— never invent numbers. ~250 words, no headings beyond ##.
<paste bullets>
```

**T3 — Mermaid diagram from a description**
```
Produce a Mermaid <flowchart|sequence> diagram, nothing else, for:
<paste the component list and arrows>. Keep node labels under 4 words.
```

**T4 — commit message / PR body wording**
```
Write a Conventional-Commits title + 5-line body for this change summary:
<paste the plan/diff summary>. No metrics unless I give them.
```

Rule: free LLMs get a spec + one file's worth of context. They never see the repo, never
produce numbers, and their output is reviewed by CC's `make check`, not trusted blind.

---

## 5. Token-waste guardrails (the big ones)
- Don't regenerate anything printed verbatim in PHASE.md — paste it.
- One phase per session; `/clear` at every PR boundary (§1).
- Always prepend §2 so CC plans before editing and stops after two failed tries.
- Numbers come from a script run once into JSON, never from re-running the engine to look.
- Keep free-LLM drafts to text with placeholders; wiring + running stays in CC.
