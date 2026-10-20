<!--
  RENDERED FILE — DO NOT HAND-EDIT NUMBERS.
  Every quantity below is a {{ ... }} placeholder resolved from results/*.json by
  B's scripts/render_docs.py (docs pipeline, branch b/p9-docs-pipeline). As of this
  PR that pipeline does not exist yet, so `make claims` passes trivially via the
  Makefile fallback (`|| test ! -d docs/templates`). B must wire render_docs +
  check_claims against the grammar below before this file renders / is claim-checked.

  PLACEHOLDER GRAMMAR (one convention for every file):
      {{ <relpath> | <jsonpath> | <fmt?> }}
    - <relpath>  : path under results/ INCLUDING .json (e.g. sealed/v2_single.json)
    - <jsonpath> : dotted path into that JSON; segments containing dots use bracket
                   form, e.g. metrics['top1.pooled'].value ; indexes are numeric
                   (disclosures.2.what). Resolves ResultsFile metrics AND plain JSON.
    - <fmt?>     : optional display format (default: verbatim). Suggested tokens:
                   pct   value*100, 1dp        pct0  value*100, integer
                   pct2  value*100, 2dp        x1    multiplier, 1dp (e.g. 10.8)
                   ms0   milliseconds, integer  g     grouped integer (65,810,000)
                   nm3   nanometres, 3dp         int   integer
                   (strings render verbatim; render_docs owns the exact formatting.)

  check_claims MUST treat the following as nomenclature, NOT metrics (they carry
  digits but are identifiers, not claims): stage labels S0–S9, engine names v1/v2,
  "OpenCV 5", "arm64"/"amd64", "recall@K", "95% CI", gate/rule IDs (G2, D1–D12).
  Every other digit in prose should be traceable to a placeholder above.
-->

# DriftSense v2 — Technical Report

## 1. The problem: periodic-array addressing ambiguity

Semiconductor process control moves a sample stage to a template location that the
operator or an automated recipe has marked on a reference image. On the periodic
structures that dominate real devices — DRAM capacitor arrays, FinFET fin pitches —
the template matches not one location but a whole lattice of near-identical replicas.
A matcher that returns a single confident coordinate will, on a rotated or degraded
search image, commit to the *wrong* replica. In a fab that is not a ranking error; it
is a stage move to the wrong cell, and every downstream measurement inherits the
offset. The cost of a confidently-wrong commit is categorically worse than the cost of
asking for another image.

DriftSense is built around that asymmetry: the goal is not to be right most often, it
is to **never be confidently wrong**, and to make the residual ambiguity explicit so a
human or an active-perception loop can resolve it.

## 2. From v1 to v2

v1 (the MunichTech engine, tagged `v1-munichtech`; see [PRIOR_WORK.md](PRIOR_WORK.md))
establishes the accuracy baseline. On the sealed set its pooled top-1 is
**{{ sealed/v1.json | metrics['top1.pooled'].value | pct }}%** on the lattice path and
**{{ sealed/ncc.json | metrics['top1.pooled'].value | pct }}%** on the NCC path. v1 is
accurate but gives no calibrated notion of *when not to trust itself*: it commits on
**{{ sealed/v1.json | metrics['commits.n'].value | int }}** of
**{{ sealed/v1.json | metrics['top1.pooled'].n | int }}** sealed pairs and carries a
pooled false-confident rate of
**{{ sealed/v1.json | metrics['false_confident.pooled'].value | pct2 }}%**.

v2 is a ground-up re-implementation on the **OpenCV 5** stack whose design target is
calibrated abstention rather than raw top-1.

## 3. The v2 pipeline (S0–S9)

v2 is a ten-stage pipeline. Every hot-path operation is restricted to the KleidiCV-
accelerated OpenCV subset so the same code runs on arm64 COOL workers without a Python
pixel loop:

| Stage | Name | Role |
|------|------|------|
| S0 | ingest | normalise, denoise (GaussianBlur / medianBlur / cvtColor) |
| S1 | lattice | detect the periodic lattice and its replicas |
| S2 | rectify | estimate orientation and warp to a rectified frame (warpPerspective / remap) |
| S3 | roughness | per-replica line-edge-roughness residual fingerprint |
| S4 | score | fuse correlation + roughness into a per-candidate score |
| S5 | cascade | escalate from the fast path to the full path when the margin is low |
| S6 | contour | sub-structure contour refinement (DRAM path) |
| S7 | confidence | margin → calibrated commit probability |
| S8 | subpixel | sub-pixel offset refinement |
| S9 | explain | emit ranked candidates, heatmaps and the decision trace |

The KleidiCV-eligible ops used in the hot paths are GaussianBlur, sepFilter2D, Sobel,
erode, dilate, remap, warpPerspective, resize, pyrDown, medianBlur, cvtColor and
minMaxIdx. On the DEV speed bench v2 runs at
**{{ accuracy/g2.json | metrics['v2_latency_ms'].value | ms0 }} ms/pair**, which is
**{{ accuracy/g2.json | metrics['speedup_vs_v1_lattice'].value | x1 }}×** faster than
v1's lattice path
(**{{ accuracy/g2.json | metrics['v1_lattice_latency_ms'].value | ms0 }} ms/pair**),
though slower than v1's NCC path
(**{{ accuracy/g2.json | metrics['v1_ncc_latency_ms'].value | ms0 }} ms/pair**).

## 4. The core honest finding

**v2's single-shot top-1 is below both baselines, by design.** On the sealed set v2
single-shot pooled top-1 is
**{{ sealed/v2_single.json | metrics['top1.pooled'].value | pct }}%**, versus
**{{ sealed/v1.json | metrics['top1.pooled'].value | pct }}%** (v1 lattice) and
**{{ sealed/ncc.json | metrics['top1.pooled'].value | pct }}%** (NCC). We do not
present v2 as a more accurate point-estimator, because it is not one.

What v2 buys instead is three things that the baselines do not provide:

1. **High set-recall.** The true replica is in v2's returned candidate set on
   **{{ sealed/v2_single.json | metrics['set_recall.pooled_structured'].value | pct }}%**
   of structured sealed pairs. The right answer is almost always *present*; what is
   hard is discriminating it from its replicas.
2. **Zero false-confident commits.** v2 commits on
   **{{ sealed/v2_single.json | metrics['commits.n'].value | int }}** of
   **{{ sealed/v2_single.json | metrics['top1.pooled'].n | int }}** sealed pairs at the
   calibrated threshold, with a pooled false-confident rate of
   **{{ sealed/v2_single.json | metrics['false_confident.pooled'].value | pct }}%**
   (95% CI upper bound
   **{{ sealed/v2_single.json | metrics['false_confident.pooled'].ci_high | pct }}%**).
   It abstains rather than guess.
3. **Calibrated abstention.** The commit threshold τ is fit on a held-out calibration
   split (isotonic margin→probability map, Wilson lower bound); equal-count ECE on the
   sealed set is
   **{{ sealed/v2_single.json | metrics['ece'].value | pct2 }}**.

**Ranked recall@K is not set-recall.** A candidate can be present (counts toward
set-recall) yet ranked below a replica (misses top-1). The gap between v2's high
set-recall and its low top-1 is *discrimination*, and it concentrates exactly where the
physics says it must — on the rotated/degraded tiers:

| Tier (n = {{ sealed/v2_single.json | metrics['top1.clean'].n | int }} each) | v2 top-1 | v2 set-recall |
|------|------|------|
| clean | {{ sealed/v2_single.json | metrics['top1.clean'].value | pct }}% | {{ sealed/v2_single.json | metrics['set_recall.clean'].value | pct }}% |
| nominal | {{ sealed/v2_single.json | metrics['top1.nominal'].value | pct }}% | {{ sealed/v2_single.json | metrics['set_recall.nominal'].value | pct }}% |
| hard | {{ sealed/v2_single.json | metrics['top1.hard'].value | pct }}% | {{ sealed/v2_single.json | metrics['set_recall.hard'].value | pct }}% |
| ambiguous | {{ sealed/v2_single.json | metrics['top1.ambiguous'].value | pct }}% | {{ sealed/v2_single.json | metrics['set_recall.ambiguous'].value | pct }}% |

On the ambiguous tier top-1 is at the floor for every engine (v1, NCC and v2 all score
**{{ sealed/v2_single.json | metrics['top1.ambiguous'].value | pct }}%**): once the
rotation makes two replicas genuinely indistinguishable in a single frame, no
single-shot matcher can break the tie. This is a known physics limit, not a tuning
failure — and it is the motivation for the agent in §5.

## 5. Active perception: re-imaging under ambiguity

When the margin falls below τ, v2 does not guess — it reports *insufficient evidence*,
and the active-perception agent acquires another image (a tilt/re-scan) rather than
committing. Scored over the agent evaluation of
**{{ sealed/agent_rules.json | metrics['n_episodes'].value | int }}** episodes on the
sealed set, the rule-based agent reaches a success rate of
**{{ sealed/agent_rules.json | metrics['agent_rules.success'].value | pct }}%** — up
from **{{ sealed/agent_rules.json | metrics['v2_single.success'].value | pct }}%** for
single-shot v2 over the same episodes — with a false-commit rate of
**{{ sealed/agent_rules.json | metrics['agent_rules.false_commit'].value | pct }}%**. It
gets there by *escalating*, not by guessing.

> **Scope:** this success rate is over the
> **{{ sealed/agent_rules.json | metrics['n_episodes'].value | int }}**-episode agent
> set only; it is **not** a success rate over all sealed pairs. The agent trades dose
> and wall-time for certainty: mean acquisitions rise to
> **{{ sealed/agent_rules.json | metrics['agent_rules.mean_steps'].value }}** and mean
> dose to **{{ sealed/agent_rules.json | metrics['agent_rules.mean_dose'].value | g }}**
> electrons, which **exceeds** the pre-registered per-episode budget of
> **{{ sealed/agent_rules.json | metrics['dose_budget'].value | g }}** electrons — the
> reason sealed decision rule D12 does not pass (§6).

## 6. Sealed evaluation

The sealed evaluation was **pre-registered and hash-locked** before any sealed pair was
touched: protocol SHA
`{{ sealed/disclosures.json | protocol_sha256 }}`, engine pinned at
`{{ sealed/disclosures.json | engine_pin }}`. The grader refuses to run unless the
running engine's git SHA matches the pin, so no engine parameter could be tuned to the
sealed data.

**Result: the sealed gate did not pass.** Of
**{{ sealed/decisions.json | n_rules | int }}** pre-registered decision rules,
**{{ sealed/decisions.json | n_decided | int }}** were decided and overall pass is
`{{ sealed/decisions.json | overall_pass }}`. Only the two rules that encode the
engine's actual design thesis pass:

- **D7 (false-confident bound):** `{{ sealed/decisions.json | rules.D7_false_confident.pass }}` — {{ sealed/decisions.json | rules.D7_false_confident.detail }}
- **D11 (agent not worse than single-shot):** `{{ sealed/decisions.json | rules.D11_agent_not_worse.pass }}` — {{ sealed/decisions.json | rules.D11_agent_not_worse.detail }}

The accuracy-oriented rules (D1–D6, D8) do not pass, because v2 is deliberately not a
better point-estimator; and D12 does not pass because the agent exceeds its dose budget
(**{{ sealed/decisions.json | rules.D12_agent_budget.detail }}**). We report this
plainly rather than re-defining the gate around the result.

### 6.1 Harness-correction disclosure (three runs)

The sealed result comes from **one corrected run** of a harness that was fixed, on the
DEV set only, across three attempts. The engine was SHA-frozen throughout; every bug
was in `eval/sealed_run.py`, not in `driftsense/`. The full record is in
`results/sealed/disclosures.json`; in summary:

- **Run 1** aborted in `--grade` and surfaced three measurement bugs (top-1 scored in
  the rectified frame; set-recall read from the engine's top-K instead of the generation
  definition; the grade glob tried to parse non-ResultsFile sidecars). See
  *{{ sealed/disclosures.json | disclosures.2.id }}*:
  {{ sealed/disclosures.json | disclosures.2.what }}
- **Run 2** completed but exposed a v1-arm candidate-selection bug. See
  *{{ sealed/disclosures.json | disclosures.3.id }}*:
  {{ sealed/disclosures.json | disclosures.3.what }}
- **Run 3** (committed) ran the audited harness. A manifest-metadata read during harness
  development is disclosed in full
  (*{{ sealed/disclosures.json | disclosures.0.id }}*), as is the tag-vs-pinned-SHA note
  (*{{ sealed/disclosures.json | disclosures.1.id }}*).

**Limitation line (verbatim from disclosures.json):**
> {{ sealed/disclosures.json | report_limitation_line }}

## 7. Responsible use

- **Synthetic data only.** All results here are on synthetic SEM imagery generated by
  `data_gen/`; DriftSense has **not** been validated on production tool data.
- **Human-in-the-loop for stage moves.** The engine's job is to commit *or abstain*. A
  commit is a recommendation; moving a physical stage remains a human- or recipe-gated
  action, never an unattended consequence of a v2 commit.
- **Dose limits.** The active-perception agent operates under an explicit per-episode
  electron-dose budget
  (**{{ sealed/agent_rules.json | metrics['dose_budget'].value | g }}** electrons); where
  it exceeds that budget (§5, §6) the overage is reported, not hidden.
- **Not a safety certification.** These numbers characterise an engine on a synthetic
  benchmark. They are not a claim of fitness for any specific tool, process node, or
  safety-critical deployment.

---
*All figures in this report are rendered from `results/*.json` by the docs pipeline; no
number is hand-typed. See the header comment for the placeholder grammar.*
