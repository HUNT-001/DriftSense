# DriftSense — PS2, Semicon India Hackathon

Reference-pattern localization in periodic semiconductor SEM imagery.
Given a small reference patch and a 1000×1000 search image, predict the
reference centre `(x, y)`.

**Stage 1 (this repo, complete):** a physics-based synthetic data engine, a
classical localization baseline, and a controlled failure analysis that
identifies *exactly* why classical template matching fails.

---

## The central finding

Classical matching does not fail randomly on these structures. **It fails by
locking onto the wrong period.**

| Tier | What changes | NCC median error | NCC Acc@5px | Failures that are period aliases |
|---|---|---|---|---|
| `clean` | no rotation/scale, low noise | **0.3 px** | **100.0%** | — (no failures) |
| `nominal` | ±1.5°, ±1.5% scale, mixed noise | 0.4 px | 82.5% | **100%** |
| `hard` | ±5°, ±7% scale, heavy drift, low dose | 360.6 px | 17.5% | 45% |
| `ambiguous` | **wafer physics disabled** | 347.6 px | **0.0%** | **100%** |

`clean` and `ambiguous` use *identical* noise, blur and geometry settings.
The only difference is whether line-edge roughness, CD variation and defects
are enabled. That single change moves NCC from **100% → 0%**.

Measured on the same image pair:

- perfectly periodic scene → **553** locations within 2% of the peak NCC score
- real wafer physics enabled → **2**

![ambiguity](outputs/figures/ambiguity_demo.png)

### Why this matters for the solution

A perfectly periodic scene is **information-theoretically unsolvable** — every
alias location is a genuinely equal explanation of the observation. No model,
of any size, can win there.

What makes the real task solvable is that wafers are *not* perfectly periodic.
**Line-edge roughness is a permanent physical property of the etched wafer**,
so two separate SEM scans of the same region observe the *same* roughness plus
independent detector noise. LER is therefore a positional fingerprint.

That is the design thesis for Stage 2: **the disambiguating signal is wafer
physics, not model capacity.**

---

## Data engine

### Structures (`data_gen/structures.py`)

| Family | Variants | Notes |
|---|---|---|
| DRAM | `grid`, `staggered` | staggered = 6F² honeycomb offset, as in modern DRAM |
| FinFET | `plain`, `fin_cut` | dense fins + sparse perpendicular gates + diffusion breaks |

Aperiodic content that breaks translational ambiguity:

- **Line-edge roughness** — spatially-correlated edge displacement, σ ≈ 0.55 px,
  correlation length ≈ 18 px. Generated **once per scene** and shared by both
  acquisitions, because roughness is a property of the wafer, not of the scan.
- **CD variation** — smooth linewidth drift across the exposure field (dose/focus).
- **Contrast drift** — smooth material-contrast variation.
- **Defects** — particles, bridges (shorts), opens. ~44 per scene.

### SEM rendering (`data_gen/sem_effects.py`)

Applied in physical acquisition order:

1. Beam blur (Gaussian PSF) — Bunday et al., *Proc. SPIE* 6152 (2006)
2. Edge brightening (SE yield) — Goldstein et al., *SEM and X-Ray Microanalysis*, 4th ed. (2018)
3. Specimen charging — Cazaux, *Ultramicroscopy* 60 (1995)
4. Vignetting (collection efficiency) — Goldstein et al. (2018)
5. Scan drift (raster shear) — Vladár & Postek, *Microscopy Today* 13(4) (2005)
6. Photometric response (brightness/contrast/gamma)
7. Scan-line gain jitter — Reimer, *Scanning Electron Microscopy*, 2nd ed. (1998)
8. Shot noise (Poisson, 25–700 e⁻/px) — Postek & Vladár, *Scanning* 33 (2011)
9. Detector read noise — Reimer (1998)

![effects](outputs/figures/effects_ladder.png)

### Ground truth

The reference is cropped from the unwarped scene; the search image is the
warped scene cropped to 1000×1000. The label is chosen first and the **full
forward chain is inverted analytically** — reference-crop drift shear → affine
rotation/scale → crop offset → search drift shear — to recover the scene
coordinate to crop from. Pixels and labels therefore cannot disagree.

Every generated pair is validated by a rotation/scale-compensated,
alias-immune correlation audit:

```
clean      pass@3px=100.0%  median_offset=0.35px  max=0.68px
nominal    pass@3px=100.0%  median_offset=0.39px  max=0.64px
hard       pass@3px=100.0%  median_offset=0.52px  max=1.33px
ambiguous  pass@3px=100.0%  median_offset=0.39px  max=0.68px
```

---

## Usage

```bash
pip install -r requirements.txt

python -m tests.test_datagen                       # 19 tests
python -m data_gen.dataset_gen --all --n_pairs 40  # 160 pairs, ~2.7 min
python -m experiments.run_baseline --all_tiers
python -m experiments.analyze_failures --all_tiers --method NCC
python -m data_gen.visualize --all --tier clean
```

## Layout

```
config.py                        tier presets, all tunable physics
data_gen/
  structures.py                  DRAM / FinFET generators + LER + defects
  sem_effects.py                 9-stage SEM acquisition model
  scene_composer.py              exact-geometry pair composition + GT audit
  dataset_gen.py                 tiered dataset driver
  visualize.py                   QC and presentation figures
localization/
  baseline.py                    NCC, phase correlation, multi-scale pyramid
experiments/
  run_baseline.py                per-tier benchmark
  analyze_failures.py            alias-vs-gross failure taxonomy
tests/test_datagen.py            19 correctness tests
```

## Bugs found and fixed while hardening Stage 1

1. **Line rendering was geometrically wrong.** The width profile was applied
   *along* each line instead of *across* it, so every "line" was a 1-pixel
   column holding a perpendicular intensity ramp. The generator produced faint
   blobs, never actual DRAM/FinFET structure — which invalidated the original
   baseline numbers entirely. Regression test: `test_bar_positions_and_width`.

2. **Ground truth ignored scan drift.** Drift is a horizontal shear
   (`x' = x + d·y`) displacing content by up to 110 px at the bottom of the
   raster, but labels were computed pre-drift while pixels were rendered
   post-drift. Regression tests: `test_ground_truth_correct_under_drift_and_rotation`,
   `test_drift_is_a_shear_of_known_magnitude`.

3. **Border artifacts.** Warping a 1000×1000 image with reflect padding
   fabricated mirrored structure and a false symmetry a matcher could exploit.
   Now an oversized scene is warped and the valid centre cropped.

4. **Self-audit was aliasing.** The original audit used a wide correlation
   window and no rotation compensation, so it reported correct labels as wrong.
   Now the window is narrower than half the smallest period and the known warp
   is undone first.

5. `np.exp` overflow; O(H) `warpAffine` calls for drift (now a single `remap`);
   O(n²) manifest rewriting; integer-quantized labels (now sub-pixel).

---

# Stage 2 — LER-aware localization

```
1000x1000 search
       |
       v
Stage A: NCC candidates + NMS -> top-K      (recall)
       |
       v
Stage B: LER fingerprint re-ranking         (precision / disambiguation)
       |
       v
Stage C: sub-pixel refinement
       |
       v
     (x, y)
```

## Does LER carry the signal? Yes — decisively

Factorial experiment, truth vs ~24 periodic replicas. Only the wafer-physics
factor varies; rotation, scale and drift are held at zero.

| physics | LER top-1 | separation | d′ |
|---|---|---|---|
| none (control) | 0–10% *(chance 4%)* | 0.01–0.10 | 0.06–0.46 |
| **LER only** | **95–100%** | **0.62–0.75** | **3.9–5.0** |
| full physics | 90–95% | 0.52–0.76 | 3.4–4.6 |

Turning LER on and nothing else moves discrimination from chance to ~100%.
The extractor recovers injected LER at **corr 0.997**, with recovered RMS
matching the injected σ, and still achieves corr 0.89 at 25 e⁻/px.

## Two results I expected and got wrong

**1. "Raw NCC's thin margin will collapse under distortion." It did not.**
In the 25-way replica test NCC holds 100% top-1 across every stressor
(rotation, drift, dose); its d′ even improves. The honest reading is that
*NCC is itself weakly sensitive to LER* — it reads the same physical signal
inefficiently, with a 0.08 margin versus the fingerprint's 0.70.

That test simply cannot separate the methods: 25-way discrimination is easy
enough that both saturate. The operational task searches ~10⁶ positions,
where beating *every* competitor demands far more margin. All method
comparison therefore moved to full search.

**2. A detrending bug was masquerading as a rotation limit.**
Discrimination collapsed above ~2.5°. The cause was not rotation itself but
that lines were detrended by removing only the **mean**. Rotation induces a
linear *ramp* along each line — ≈8.7 px over a 100 px patch at 5°, ~15× the
0.55 px LER amplitude — which mean-removal leaves untouched. Switching to
linear detrending recovered most of it:

| |rotation| | NCC | LER (mean-detrend) | LER (linear detrend) |
|---|---|---|---|
| 0–1° | 38% | 75% | **88%** |
| 1–2.5° | 12% | 50% | **50%** |
| 2.5–5° | 20% | 13% | **27%** |

## Full-search results (all 1000×1000, top-K = 150)

| Tier | recall@K | NCC Acc@5 | **LER-2stage Acc@5** | gain | rescued | broken |
|---|---|---|---|---|---|---|
| clean | 100% | 100% | 100% | +0.0% | 0 | 0 |
| nominal | 100% | 82.5% | **87.5%** | +5.0% | 6 | 4 |
| **hard** | 77.5% | 17.5% | **37.5%** | **+20.0%** | 9 | 1 |
| ambiguous | 40% | 0% | 0% | +0.0% | 0 | 0 |

`hard` median error drops 360.6 → 226.6 px. `ambiguous` stays at 0% for both,
exactly as it must — it is the control, and any method that beat it would be
leaking information.

## Where the remaining error actually lives

The `recall@K` and `rescued`/`broken` diagnostics exist to stop a net gain
from hiding a method that merely trades one error for another. On `hard`:

- **22.5% of failures belong to Stage A** — the truth is not in the candidate
  set at all, so re-ranking cannot recover it. This is the single largest
  remaining loss, and it is what motivates frequency-domain candidate
  generation (Stage 2A): estimating pitch and orientation lets candidates be
  placed *on the lattice* instead of at NCC peaks.
- **Of pairs where the truth IS a candidate, the re-ranker picks it 48%** (up
  from 39% before linear detrending), concentrated in the >2.5° rotation
  bucket. Explicit orientation estimation and de-rotation before extraction is
  the direct fix.

So frequency analysis is now empirically motivated as a *recall* mechanism,
not assumed to be the innovation. It characterizes the ambiguity; LER breaks it.

## Known limitations

- The extractor assumes an axis-aligned lattice; beyond ~2.5° relative
  rotation, discrimination degrades even with linear detrending.
- `ncc_weight` in the combined score is currently inert: all candidates have
  near-identical NCC (~0.9), so it adds a near-constant offset and never
  changes the ranking. Making it useful requires z-scoring NCC across
  candidates first.
- The generator models *line-position* roughness (both edges of a bar move
  together). Real lines have partially independent left/right edge roughness;
  the centroid estimator recovers only the common mode.
- Stage-2 hyperparameters (`top_k`, detrend order) were chosen on the same
  tiers used for reporting. A held-out split is needed before quoting these
  as generalization numbers.

## Stage 2 usage

```bash
python -m tests.test_ler_fingerprint       # 12 tests
python -m experiments.ler_hypothesis --n_pairs 20
python -m experiments.ler_stress --n_pairs 15
python -m experiments.run_stage2 --all_tiers --top_k 150
```
