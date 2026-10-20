<!--
  RENDERED FILE — DO NOT HAND-EDIT NUMBERS.
  Same placeholder grammar as REPORT.md (see its header comment): resolved from
  results/*.json by B's scripts/render_docs.py. Nomenclature carrying digits
  (v1/v2, OpenCV 5, S0–S9, arm64, the tag name v1-munichtech) is not a metric.
-->

# Prior Work and Attribution

DriftSense v2 builds directly on a pre-existing engine. This document separates what was
**inherited** from what is **new in v2**, so that only the v2 deltas are claimed as this
project's contribution.

## Prior art: v1 (MunichTech)

The v1 engine is prior work from MunichTech. It is frozen in this repository under
`driftsense/v1/` and tagged **`v1-munichtech`**; it is never modified. v1 contributes:

- The **lattice-detection and NCC matching** approach to periodic-structure
  localisation, and the orientation estimate used to rectify the search image.
- The **accuracy baseline** this project measures against. On the sealed set v1's pooled
  top-1 is **{{ sealed/v1.json | metrics['top1.pooled'].value | pct }}%** (lattice path)
  and **{{ sealed/ncc.json | metrics['top1.pooled'].value | pct }}%** (NCC path), and its
  sub-pixel localisation error on committed pairs has median
  **{{ sealed/v1.json | metrics['subpixel_nm_error.median'].value | nm3 }} nm**.

We claim **none** of the above as new. v1's accuracy is the bar, not our result.

> **Sign convention note (inherited).** v1's `estimate_orientation` returns an angle of
> the opposite sign to v2's `detect_lattice`; the two are compared sign-agnostically in
> every evaluation. This is a known interface detail of the prior engine, recorded so
> that cross-engine numbers are not misread.

## New in v2 (this project)

Everything claimed as a contribution is a **delta over v1**, measured on the same sealed
set and benchmark harness:

1. **Calibrated abstention instead of unconditional commit.** v1 commits on every pair;
   v2 introduces a calibrated commit threshold τ and a margin→probability map, driving
   the pooled false-confident rate from
   **{{ sealed/v1.json | metrics['false_confident.pooled'].value | pct2 }}%** (v1) to
   **{{ sealed/v2_single.json | metrics['false_confident.pooled'].value | pct }}%** (v2),
   by committing on **{{ sealed/v2_single.json | metrics['commits.n'].value | int }}** of
   **{{ sealed/v2_single.json | metrics['top1.pooled'].n | int }}** pairs rather than
   **{{ sealed/v1.json | metrics['commits.n'].value | int }}**.

2. **A roughness-residual fingerprint (S3–S4)** that scores candidate replicas on
   line-edge-roughness residual in addition to correlation — a discriminator v1 does not
   have.

3. **An OpenCV 5 / KleidiCV hot path** that runs unchanged on arm64 COOL workers, giving
   a **{{ accuracy/g2.json | metrics['speedup_vs_v1_lattice'].value | x1 }}×** speedup
   over v1's lattice path on the DEV bench.

4. **Set-recall as the reported quantity** (the true replica is present on
   **{{ sealed/v2_single.json | metrics['set_recall.pooled_structured'].value | pct }}%**
   of structured pairs), making the ambiguity explicit rather than collapsing it into a
   single point estimate.

5. **An active-perception agent** that re-images under low margin, reaching
   **{{ sealed/agent_rules.json | metrics['agent_rules.success'].value | pct }}%** success
   over the **{{ sealed/agent_rules.json | metrics['n_episodes'].value | int }}**-episode
   agent set (scope and dose caveats in [REPORT.md](REPORT.md) §5).

What v2 explicitly does **not** claim as an improvement: single-shot top-1 accuracy,
which is **lower** than v1 by design
(**{{ sealed/v2_single.json | metrics['top1.pooled'].value | pct }}%** vs
**{{ sealed/v1.json | metrics['top1.pooled'].value | pct }}%**). See
[REPORT.md](REPORT.md) §4.

## Frozen-reference policy

`driftsense/v1/` is the golden reference and must not be edited; the `v1-munichtech` tag
marks the exact prior-art commit. All v2 code lives outside `driftsense/v1/`, and every
cross-engine comparison in this report runs both engines through the same public entry
point (`driftsense.api.localize`) so the deltas above are like-for-like.
