# DriftSense sealed-set — pre-registered evaluation protocol

- **Protocol id:** `driftsense.sealed-eval`  · **version:** `1.0.0` · **status:** pre-registered
- **Registered on:** 2026-10-19 · **owner:** Person A
- **Machine-readable source of truth:** [`eval/protocol.json`](protocol.json)
- **Frozen sha256 (canonical JSON, see below):** `776ed76e7434087905e0a309ddc64a5b79689e8331ecd43936345a3851f6d6c0`

This document is **pre-registered before the sealed set is run**. It fixes the arms,
metrics, confidence-interval method, decision thresholds, engine revision, dataset
seed, and exact reproduction commands. Running the sealed set happens **once**, in a
separate session, via `eval/sealed_run.py`. **Nothing here reads, lists, generates, or
runs the sealed set.** After merge the hash below is placed in the `protocol-v1` tag
message; the sealed-run grader refuses to proceed unless the live protocol hash and the
running engine SHA match what is pinned here.

`protocol.md` and `protocol.json` must agree; `protocol.json` wins on any discrepancy.

## Engine (pinned)

| field | value |
|---|---|
| tag | `v2-freeze` |
| git sha | `f9395b84dd654ff1d537f6817c258a0f2b9373de` |
| selector | `DRIFTSENSE_ENGINE=v2` |
| entry point | `driftsense.api.localize(template_bytes, search_bytes, mode, budget) -> LocalizeResult` |
| contract version | `1.0.0` |

All v2 arms run against exactly this commit. The grader writes the running
`git_sha` into each results file and must refuse to grade if it differs.

## Dataset (pinned)

- **name:** `sealed` · **seed:** `9999`
- **tiers:** `clean`, `nominal`, `hard`, `ambiguous` (structured = `clean`, `nominal`,
  `hard`; adversarial = `ambiguous`)
- **families:** `finfet`, `dram`, `grid`, `staggered`
- **access rule:** only `eval/sealed_run.py` may read/list/download sealed data, and only
  when a human sets `DRIFTSENSE_ALLOW_SEALED=1`. The sealed set is run exactly once.
- Tier/family labels come from the sealed manifest (`config.py` TIERS,
  `contracts.LatticeType`). The `unknown` family and any unlisted tier are reported but
  excluded from per-family / per-tier decision rules.

## Confidence intervals

- **Binomial rates** (per-tier / per-family / pooled top-1, set-recall, false-confident,
  agent success/false-commit/escalation): **Wilson score interval, 95%, two-sided.**
- **Continuous metrics** (subpixel nm error): median and p95 with a 95% CI by **BCa
  bootstrap, 10000 resamples, seed 9999.**

## Metrics

| key | definition | unit | CI |
|---|---|---|---|
| `top1.<tier>` | rank-1 center inside the GT template window, per tier (same hit test as `eval/gate_g2.py`) | rate | Wilson |
| `top1.family.<family>` | top-1 hit rate per `LatticeType` family | rate | Wilson |
| `top1.pooled` | top-1 hit rate over the full sealed set (**primary endpoint**) | rate | Wilson |
| `set_recall.<tier>` | GT location present in the returned candidate set (generation set-recall, **not** recall@20) | rate | Wilson |
| `false_confident.pooled` | commits (`margin >= tau`) **and** rank-1 wrong | rate | Wilson |
| `ece` | expected calibration error of `Margin.p_correct` vs top-1 correctness (equal-count bins, as `eval/calibration.py`) | fraction | bootstrap |
| `subpixel_nm_error` | Euclidean nm distance of committed `Offset` to GT, over commits; median and p95 | nm | bootstrap |
| `<arm>.success` | correct commit within dose/step budget | rate | Wilson |
| `<arm>.false_commit` | commits but committed location wrong | rate | Wilson |
| `<arm>.escalation` | escalates / abstains instead of committing | rate | Wilson |
| `<arm>.mean_steps` | mean SEM acquisitions per episode | acquisitions | bootstrap |
| `<arm>.mean_dose` | mean cumulative electron dose per episode | electrons | bootstrap |

## Arms

1. **`ncc`** — single-scale normalized cross-correlation (OpenCV `matchTemplate`,
   `TM_CCOEFF_NORMED`) on the raw search frame. No rectification, no calibration (never
   commits under the margin rule; its false-confident is reported `n/a`).
2. **`v1`** — frozen v1 golden reference (`v1-munichtech`, `driftsense/v1/`), lattice
   path. Primary accuracy baseline, compared like-for-like (NCC-path, tier-pooled,
   sign-agnostic orientation — see the G2 notes).
3. **`v2_single`** — one v2 localize per pair at the pinned SHA, commit iff
   `Margin.decision == 'commit'`. Headline accuracy / calibration arm.
4. **`agent_rules`** — full active-perception runner (offline rules policy); re-images
   while `margin < tau` up to the dose/step budget, then commits or escalates. Shares the
   identical first search frame with `v2_single`.
5. **`ablation_ladder`** — v2 with one capability removed per rung, graded on the same
   metrics as `v2_single`:
   `full` → `no_cascade` → `no_contour_subpixel` → `no_calibration` →
   `no_roughness_fusion` → `lattice_only`.

## Decision thresholds (frozen)

| id | requirement |
|---|---|
| D1 | `v2_single` `top1.pooled` Wilson lower bound ≥ **0.90** |
| D2 | `v2_single` `top1.pooled` Wilson lower bound ≥ (`v1` `top1.pooled` point − **0.02**) — non-inferiority |
| D3 | `v2_single` `top1.pooled` point ≥ `ncc` `top1.pooled` point + **0.10** |
| D4 | each structured tier: `v2_single` `top1.<tier>` Wilson lower bound ≥ **0.85** |
| D5 | each family: `v2_single` `top1.family.<family>` Wilson lower bound ≥ **0.80** |
| D6 | `v2_single` set-recall pooled over structured tiers ≥ **0.98** |
| D7 | `v2_single` `false_confident.pooled` ≤ **0.02** and Wilson upper bound ≤ **0.05** |
| D8 | `v2_single` `ece` ≤ **0.10** |
| D9 | `v2_single` `subpixel_nm_error` median ≤ **0.50 nm** and p95 ≤ **1.00 nm** (structured tiers) |
| D10 | `agent_rules` `false_commit` ≤ **0.02** and Wilson upper bound ≤ **0.05** |
| D11 | `agent_rules` `success` Wilson lower bound ≥ `v2_single` `success` point — re-imaging must not reduce success |
| D12 | `agent_rules` `mean_steps` ≤ **3.0** and `mean_dose` ≤ per-episode dose budget (from results env) |

**Overall pass = all of D1..D12.** Any failure is reported as-is. Thresholds are frozen;
no metric, tier, family, or threshold may be added or changed after `protocol-v1` is
tagged. Anything beyond this battery is exploratory and labeled as such.

## Reproduce

```sh
git checkout f9395b84dd654ff1d537f6817c258a0f2b9373de   # == tag v2-freeze
make setup
export DRIFTSENSE_ALLOW_SEALED=1                        # human-set; required for sealed data

python -m eval.sealed_run --arm ncc         --seed 9999 --protocol eval/protocol.json
python -m eval.sealed_run --arm v1          --seed 9999 --protocol eval/protocol.json
python -m eval.sealed_run --arm v2_single   --seed 9999 --protocol eval/protocol.json
python -m eval.sealed_run --arm agent_rules --seed 9999 --protocol eval/protocol.json
python -m eval.sealed_run --arm ablation \
    --rung full --rung no_cascade --rung no_contour_subpixel \
    --rung no_calibration --rung no_roughness_fusion --rung lattice_only \
    --seed 9999 --protocol eval/protocol.json
```

Outputs land in `results/sealed/<arm>.json` (`ResultsFile`, `dataset='sealed'`), each
with `git_sha == engine.git_sha` and `protocol_sha256 == ` the frozen hash below.
`eval/sealed_run.py` is implemented in a separate PR/session.

## Freeze hash

The protocol is frozen by the sha256 of its **canonical JSON** — `json.dumps` with
`sort_keys=True`, `separators=(',', ':')`, `ensure_ascii=True`, computed over
`protocol.json` **with the top-level `hash` key removed**. This is line-ending and
whitespace independent (the repo uses `core.autocrlf=true`), so the value is stable
across platforms and safe to paste into the `protocol-v1` tag message.

```sh
python -c "import hashlib,json; d=json.load(open('eval/protocol.json')); d.pop('hash',None); print(hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest())"
# 776ed76e7434087905e0a309ddc64a5b79689e8331ecd43936345a3851f6d6c0
```

`tests/engine/test_protocol.py` recomputes this value and fails if `protocol.json`
changes.
