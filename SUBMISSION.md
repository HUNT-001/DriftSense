# DriftSense — Devpost Submission (MunichTech EXPO Hackathon)

*Physics-aware localization for semiconductor metrology — finding the one real
copy on a wafer full of identical ones.*

---

## Problem statement

Every advanced chip is inspected under a scanning electron microscope (SEM). A
core operation in metrology, defect review and overlay is **re-finding a known
reference pattern inside a much larger SEM image** — to measure it, compare it
across process steps, or align two scans of the same die.

On a memory array or a logic standard-cell field this is deceptively hard: the
structure is **periodic**, so the reference patch matches *hundreds of identical
copies* equally well. Classical template matching (normalized cross-correlation)
picks the copy with the highest photometric score — which, under the rotation,
scale drift and low-dose noise of real acquisition, is very often the **wrong**
copy. On our hard tier, classical NCC locates the correct copy only **17.5%** of
the time, and when it is wrong it can be off by the full width of the image
(hundreds of pixels).

A wrong match in metrology is not a cosmetic error: it corrupts the CD/overlay
measurement, mis-attributes a defect, and can send a false yield signal.

## Solution overview

DriftSense separates the problem into the two questions template matching
conflates, and answers each with the right physics:

> **Frequency tells you *where* a copy could be. Line-edge roughness tells you
> *which* copy is real.**

1. **Where (recall).** The reference is part of a crystal-like lattice, so it can
   only align at lattice-congruent positions. We estimate the reciprocal lattice
   from the 2-D FFT and enumerate *every* structurally valid placement — reducing
   ~10⁶ possible positions to ~1,500 candidates while keeping the true one
   **97.5%** of the time (vs 77.5% for the top-150 NCC peaks).

2. **Which (discrimination).** Line-edge roughness (LER) — the stochastic,
   spatially-*fixed* raggedness of each etched edge — is a genuine physical
   fingerprint of a specific location on the wafer: two separate scans of the
   same region see the *same* roughness. We extract an LER fingerprint and use it
   to pick the real copy among the candidates (measured discriminability
   d′ ≈ 4–5, vs ≈ 1.2 for raw intensity).

3. **Geometry.** A single global de-rotation/de-scale is estimated per pair and
   applied uniformly, correcting the acquisition distortion before the
   fingerprint comparison.

**Result:** on the hard tier DriftSense reaches **57.5% Acc@5 — a 3.3× gain over
classical matching (17.5%)** — with a median error of 0.6 px. It uses **no neural
network**: pure classical computer vision + digital signal processing, so it runs
on any CPU with no GPU, no toolchain and no network.

| Tier | Distortion | Classical NCC | **DriftSense** | recall@60 |
|---|---|---|---|---|
| clean | none | 100% | 97.5% | 97.5% |
| nominal | ±1.5° | 82.5% | **90.0%** | 97.5% |
| hard | ±5°, ±7% scale, low-dose | 17.5% | **57.5%** | 97.5% |

## Target users / industry

Semiconductor **fab metrology and defect-review** teams, SEM/e-beam tool vendors,
and yield-engineering groups — the people who run pattern-matching inside
CD-SEM, overlay and defect-review workflows. It also generalises to any periodic
micrograph localization task (photomask inspection, MEMS, displays).

## Why this matters for Europe

Semiconductor metrology and inspection is one of the few layers of the chip
supply chain where **Europe leads the world** — ASML (litho + metrology), Carl
Zeiss SMT (SEM optics), and a deep research base (imec, Fraunhofer). The EU Chips
Act is pouring public investment into fab capacity precisely here.

Yield in an advanced fab is gated by metrology and inspection, and every wrong
match is lost yield. DriftSense strengthens exactly the layer Europe is betting
on, and it does so in a way aligned with European priorities:

- **Sovereign & on-prem by construction.** No cloud, no GPU, no external API,
  no training data leaving the fab — it is CPU-only classical DSP. Fabs guard
  process data as their crown jewels; a method that needs none of it off-site is
  a fit for European data-sovereignty requirements.
- **Explainable, not a black box.** Every decision traces to a physical quantity
  (lattice geometry, LER), which matters for the auditability expectations
  emerging under the EU AI Act and for engineer trust.
- **Industry 4.0 for manufacturing.** Directly serves the deployable-automation,
  yield-and-quality mandate of European advanced manufacturing.

## Technical details

**Architecture**
```
1000×1000 SEM search image + reference patch
        │
        ▼
Stage 2A — Spectral candidate generation   (2-D FFT → reciprocal lattice →
        │                                    enumerate lattice-consistent placements)
        ▼
Stage 2C — Global de-rotation / de-scale    (power-weighted angular consensus;
        │                                    uniform correction, non-regressing)
        ▼
Stage 2B — LER fingerprint re-ranking       (the disambiguator: d′ ≈ 4–5)
        │
        ▼
Stage C  — Sub-pixel refinement → (x, y)
```

**Stack.** Python 3.10+, NumPy, SciPy, OpenCV, Matplotlib. Web demo: FastAPI +
uvicorn. **No ML frameworks, no GPU.**

**Datasets / models.** No trained model and no pre-trained weights. A
physics-based SEM simulator (9-stage acquisition model: LER, CD drift, shot/read
noise, charging, vignetting, blur, jitter) generates a tiered, reproducible,
sealed test set (DRAM + FinFET families). An `ambiguous` control tier with the
aperiodic physics switched *off* scores 0% by design — proving the gains come
from real physics, not overfitting.

**Validation.** 31+ passing regression tests, sealed held-out set, and honest
attribution diagnostics (recall vs discrimination reported separately, plus
`rescued`/`broken` counts so a net gain can't hide a trade-off).

## What we built for this hackathon

- **Completed the unified full-search benchmark.** The best pipeline
  (lattice + de-rotation) existed in the code but was never in the headline
  evaluation, which still reported the NCC-candidate path. Wiring it in raised
  the honest hard-tier number from 30% → **57.5%** and produced a clean,
  reproducible benchmark across all tiers (`docs/BENCHMARK.md`).
- **Ran an orientation ablation** (`localization/lattice_fit.py`,
  `experiments/orientation_error.py`) — three new estimators (RANSAC lattice-fit,
  Gauss-reduced, Fourier angular-correlation) — and *measured* that orientation
  is **not** the bottleneck: recall is solved and discrimination is the ceiling.
  A negative result that correctly redirected effort.
- **Built a deployable product:** a CLI (`python -m demo.cli locate …`) and a
  self-contained FastAPI web app (`demo/app.py`) that runs the pipeline on any
  uploaded or built-in SEM pair and shows the classical-vs-DriftSense result, the
  reciprocal lattice, and the LER decision live.

## Deployment considerations

CPU-only, single-process, ~6 s/image in the unstuned demo path; the spectral
candidate stage is the natural target for the obvious speed-ups (FFT reuse,
vectorised fingerprinting). Packages cleanly as an on-prem microservice or a
plug-in behind an existing SEM tool's matching step. No data leaves the host.

## Ethics, regulatory & sustainability

Explainable-by-construction (physical features, no opaque model), which supports
EU AI Act auditability and engineer trust. Sustainability: CPU-only inference
avoids GPU-scale energy; better first-pass localization reduces re-scans and
scrapped wafers.

## Next steps

- Close the recall→Acc@5 gap on the hard tier by strengthening the LER
  fingerprint under compound rotation+scale (the measured true bottleneck).
- Validate on **real** CD-SEM imagery with an industry partner.
- Optimise the spectral stage for tool-rate throughput and package as an on-prem
  service.

## Links

- Repository: https://github.com/HUNT-001/DriftSense
- Benchmark & reproduction: `docs/BENCHMARK.md`
- Demo: `demo/README.md`
