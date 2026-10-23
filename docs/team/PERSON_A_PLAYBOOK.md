# DriftSense Pilot — Person A Playbook (token-aware)

Reconciles the PHASE.md two-person plan with the **actual repo state** on 2026-10-11,
and marks every unit of work as **[CC]** (do in Claude Code) or **[FREE]** (offload to a
free-tier LLM / paste verbatim) so you spend paid tokens only where they're needed.

You are **Person A** and the repo admin (HUNT-001). You own: engine (`driftsense/v2`),
accuracy, virtual SEM, agent, sealed eval, report narrative. You also run the shared
P0 bootstrap because you're admin.

---

## 0. Reality check (do this FIRST, by hand)

Base repo is **D:\\DriftSense-clean**, branch **fresh-main**, clean tree, no tags. Before any prompt:

```bash
# make fresh-main the integration trunk and freeze the golden reference
git tag -a v1-munichtech -m "MunichTech golden reference"   # frozen v1 point
# if origin/main should be this: git push origin fresh-main:main --tags   (your call)
git push origin fresh-main --tags
```

`v1` stays where it is (data_gen/, localization/, experiments/, config.py); the adapter
wraps it in place — no move needed.

**v1 entry point the adapter wraps (verified):**
`localization.ler_localizer.localize(ref, search, ...) -> LocalizationResult`
with `pred_x/pred_y` = **centre** coords. Contract `Offset` is **top-left**, so the adapter
does `x_px = pred_x - ref_w/2`, `y_px = pred_y - ref_h/2`. Already written in
`driftsense/engines/v1_adapter.py`.

## 1. The offload rule (how to save tokens)

| Bucket | Who | Why |
|---|---|---|
| **Verbatim files already in PHASE.md** (CLAUDE.md, all `.claude/` hooks+settings+commands+agents, CODEOWNERS, PR template, ci.yml, ruleset JSON, contracts.py, api.py, engines/fake.py, tests/contract) | **PASTE** (no model at all) | They're literally printed in your doc. Spending any model to "generate" them is pure waste. Create the files and paste. |
| **Self-contained text from a clear spec, no repo access** (unit-test skeletons from a signature, docstrings, README/REPORT prose, RESPONSIBLE_USE.md, PRIOR_WORK.md, Mermaid diagrams, commit/PR-body wording, requirements.in list) | **FREE** (ChatGPT/Gemini/Claude free) | One file at a time, give it the spec, paste the result back. No repo state needed. |
| **Must read the repo, run it, iterate against tests, invent algorithm, or touch git/AWS/hooks/CI** | **CC** (Claude Code, paid) | This is the only work that genuinely needs the agent in the loop. |

**Golden habit:** never paste the whole repo into a free LLM. Give it exactly one spec +
one file. Bring the draft into Claude Code only to wire, run `make check`, and commit.

---

## 2. Person A work, phase by phase, tagged

### P0 bootstrap (Oct 11, you as admin) — mostly FREE/paste
- **[PASTE]** S1–S4 files: `CLAUDE.md`, `.gitignore` additions, `.claude/**` (hooks, settings.json,
  commands/*.md, agents/verifier.md), `.github/` (ci.yml, CODEOWNERS, pull_request_template.md),
  `driftsense/contracts.py`, `driftsense/api.py`, `driftsense/engines/fake.py`,
  `tests/contract/test_contracts.py`, `driftsense/v2/engine.py` stub. All verbatim in PHASE.md.
- **[CC]** `driftsense/engines/v1_adapter.py` — the ONLY bootstrap file needing reasoning: it must
  wrap the real `ler_localizer.localize` and map centre→top-left into `LocalizeResult`. ~40 lines.
- **[CC]** `make contracts && make check` until green (v2 test skips), then commit + open the
  `shared/p0-bootstrap` PR. Run `/hooks` and deliberately test the guards (try to edit v1 / push main).
- **[PASTE+CC]** S5 rulesets: paste the JSON **[PASTE]**, run the `gh api` calls **[CC]**.
- **[FREE]** `requirements.in` list, `docs/PLAN.md` schedule table (copy from PHASE.md §4).

### A1 — P1 profiling + golden baseline (Oct 11–12) — CC
- **[CC]** `eval/profile_v1.py`, `results/accuracy/{baseline_v1,profile_v1}.json`. Must run cProfile
  on real pairs through `api.localize(engine="v1")`, compute DEV accuracy + bootstrap 95% CI, recall@20.
- **[FREE]** the test skeleton `tests/engine/test_profile_outputs.py` (just "load JSON, validate as
  ResultsFile") — draft free, run in CC.

### A2 — P2a v2 S0–S2 ingest/lattice/rectify (Oct 12–13) — CC
- **[CC]** real algorithm work; KleidiCV-op discipline; NumPy-2 cross-product fix; iterate vs synthetic
  lattice tests. This is core engine IP — keep in CC.
- **[FREE]** boilerplate: synthetic-lattice fixture generators, dataclass docstrings.

### A3 — P2b S3–S6 roughness/score/cascade/contour (Oct 14–15) — CC
- **[CC]** all of it (the roughness-residual thesis, cascade switch). **[FREE]** test fixtures only.

### A4 — P2c S8/S9 + entry point + Gate G2 (Oct 16) — CC
- **[CC]** `subpixel`, `explain`, full `engine.localize`, `eval/gate_g2.py`, run the gate.
- **[FREE]** the explain-stage colormap/overlay helper text, alt-text strings.

### A5 — P3a DRAM contour (Oct 15–17) — CC (algorithm + diagnose)
### A6 — P3b confidence + calibration + freeze (Oct 17–19) — CC (isotonic/τ, must run on CALIB)
### A7 — P5a VirtualSEM (Oct 18–19) — CC (reuses data_gen physics; must be deterministic)
  - **[FREE]** `docs/virtual_sem.md` assumptions prose (draft from your bullet points).
### A8 — P5b Agent + MCP + G5 (Oct 19–20) — CC
  - **[CC]** tools, rules policy, runner, trace, MCP server, run the episode.
  - **[FREE]** the optional LLM-planner prompt text, MCP README, trace-viewer sample fixture.
### A9 — P7 protocol + sealed run (Oct 21–22) — CC ONLY
  - **[CC]** never offload. Sealed data must never leave the repo/S3, and the single run must be
    executed and logged. **[FREE]** the `eval/protocol.md` prose template (no numbers).
### A10 — P9 report narrative (Oct 24) — FREE-heavy
  - **[FREE]** `docs/REPORT.md`, `docs/PRIOR_WORK.md` prose with `{{ placeholder }}` tokens.
  - **[CC]** only `make docs && make claims` to fill/verify numbers (never typed by hand).

---

## 3. What Claude Code must do that NOTHING else can
Running `make check` / gates, git + hooks + rulesets, reading real v1 to write the adapter,
inventing/iterating the v2 algorithm against tests, the AWS-adjacent integration swaps,
the sealed run, and the final clean-clone verifier session. Everything text-shaped and
spec-driven around those is free.

## 4. Anti-patterns that burn tokens
- Re-generating files already printed verbatim in PHASE.md.
- Pasting the whole repo into any model "for context".
- Letting CC write long prose (REPORT/README) — draft free, wire in CC.
- Running the engine many times in-session to eyeball numbers — write a script, run once, read JSON.
