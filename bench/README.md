# DriftSense benchmarks: COOL on AWS Graviton4

`bench/cool/` measures whether **Cloud Optimized OpenCV (COOL AMI, KleidiCV HAL)** on AWS
Graviton4 speeds up the real DriftSense v2 engine compared with pip
`opencv-python-headless==5.0.0.93` on Graviton and on x86, and what the difference costs per
1,000 localizations and per 1,000 agent episodes. The harness can show a speed-up, and it can
just as easily show none. We publish whatever it measures.

## Honesty rules (pre-committed)

1. **We publish whatever we measure**, including "no measurable difference" and "slower".
2. **Vendor claims are not our numbers.** "~20% faster", "1.5x" and similar figures used other
   baselines. The report never quotes them as results.
3. **If a CI includes 1.0, we say so.** `report.py` writes the phrase "no measurable difference"
   itself, whatever the point estimate.
4. **COOL is not expected to speed up unlisted ops** (matchTemplate, dft, magnitude, log,
   phaseCorrelate, findTransformECC, imencode, ...). They are labelled `control` and kept in the
   run. A "gain" on a control, or on the NumPy negative controls, is noise or a setup
   difference, and we investigate it. We do not claim it.
5. **Only arms with the same `git_sha` are compared.** The report warns on a mismatch.
6. **No speed-up counts without equality.** On the same instance type, pip and COOL offsets must
   agree to within 1e-3 px, and the agent episodes must reach the same outcomes and estimates.
7. **Smoke runs are not evidence.** `--quick` or off-EC2 output is named `*.local_smoke`, and
   `report.py` refuses to write it into `results/bench/` (anything outside
   `results/bench/local_smoke/`). Local Windows/laptop numbers are never committed as results.

## What is measured

| Workload | Module | What runs | Keys |
|---|---|---|---|
| Call census | `census.py` | `driftsense.api.localize(engine="v2")` on 12 DEV pairs per mode (fast, auto), with every engine cv2 call wrapped | `census.<mode>.<fn>@<module>.<func>.{calls,ms}_per_localize`, `census.<mode>.share.{kleidicv,control,non_cv2}` |
| Microbench | `microbench.py` | every engine cv2 call site, **replaying the arguments the engine passed** (captured by the census on DEV pair `hard/0000`) | `op.<fn>.<variant>.<t1/tN>.{median_ms,p95_ms}` |
| Stagebench | `stagebench.py` | `localize(engine="v2")` on 30 DEV pairs, modes fast and auto, threads 1 and N; per-stage `LocalizeResult.timings` | `stage.<mode>.<stage>.<tag>.*`, `check.<mode>.top1_rate.<tag>` |
| End to end | `e2e.py` | latency p50/p95 (threads N) and process-pool throughput (one worker per vCPU, cv2 threads 1), psutil CPU | `e2e.<mode>.latency.tN.*`, `e2e.<mode>.throughput.jobs_per_hour` |
| Agent | `agent_bench.py` | `driftsense.agent.runner.run_episode` on `driftsense.sim.VirtualSEM`, 8 seeds x {dram, finfet}; latency and episodes/hour | `agent.latency.tN.*`, `agent.throughput.episodes_per_hour`, `check.agent.success_rate` |
| Equality | `equality.py` | offsets of every (mode, pair), agent estimates/outcomes, and S0-S4 score maps (max abs diff, SHA-256) | `equality.<cmp>.*` |
| Evidence | `collect_evidence.py` | instance type, region, AMI id, product codes (IMDSv2), COOL version files under `/opt/cool`, `cv2.__version__`, `cv2.__file__`, `getBuildInformation()` (KleidiCV/HAL lines), `ldd`, `pip freeze`, `uname -a`, `/proc/cpuinfo`, `lscpu` | `evidence/*.txt`, `evidence/evidence.json` |

The stage names are the ones `driftsense/v2/engine.py` emits: ingest, lattice, rectify,
roughness, score, cascade, subpixel, explain. `auto` is the service default
(`JobCreate.mode`). `fast` is the cheapest production path.

**Inputs.** Committed DEV pairs only: `outputs/dataset/<tier>/manifest.json` and
`pair_<id>_{ref,search}.png` (1000x1000 uint8 search, 80-130 px template), taken
round-robin over clean/nominal/hard/ambiguous. `bench/cool/pairs.py` refuses any path
containing `dataset_test` or `sealed`. The agent workload is deterministic from
`(seed, family)`. Its operating point is copied from `eval/agent_eval.py:57-67`
(commit 46ff2f5). `eval/` is never imported, and `tests/bench/test_ops.py` checks that the
copy matches.

**Protocol.** One untimed first call, 5 warm-ups, then 30 timed runs with
`time.perf_counter_ns`, at `cv2.setNumThreads(1)` (`t1`) and `os.cpu_count()` (`tN`). Ops that
take under 1 ms are batched so that one sample lasts at least 1 ms, and the per-call time is
reported. Statistics: the median with a percentile-bootstrap 95% CI (2000 resamples, fixed seed)
and the p95. A speed-up is median(baseline) / median(candidate), with both arms resampled
independently. Group geometric means carry a joint bootstrap CI.

**Amdahl check.** The census measures what fraction of `total_ms` the KleidiCV-listed calls
take. `report.py` combines that with the per-group speed-ups into a predicted pipeline speed-up
and the upper bound if listed ops cost nothing. It prints both next to the measured
`stage.<mode>.total` speed-up. This bounds what COOL *can* do for this pipeline before
anyone reads the per-op numbers.

## Microbench op table (engine calls, as of the v2 engine at 81b4dd5)

Groups: `kleidicv` = the cv2 function is on the KleidiCV HAL list (GaussianBlur, sepFilter2D,
Sobel, medianBlur, dilate, erode, resize, pyrDown, transpose, rotate, remap, warpPerspective,
cvtColor, convertTo, minMaxIdx/minMaxLoc, absdiff/add/subtract/multiply, calcOpticalFlowPyrLK).
COOL may help, but only where KleidiCV supports the dtype and parameters. OpenCV silently falls
back otherwise, and the notes flag the risky variants (float32, REFLECT101 borders).
`control` = not listed. `numpy` = pure NumPy that runs in the pipeline (negative control).
`counterfactual` = **not an engine call**: the same op and kernel on uint8 data, reported
separately to show what the float32 design leaves on the table. Arguments in the table were
captured from the engine; the shapes depend on the capture pair.

<!-- generated from bench/cool/ops.py; tests/bench/test_ops.py checks every file:line cited here -->
| op key | stage | group | engine call site | production arguments (captured on hard/0000) | calls / localize | note |
|---|---|---|---|---|---|---|
| `imdecode.s0_png_unchanged` | S0 | control | `ingest.py:95` (ingest) | `uint8[899141], -1` | 2 (template, search) |  |
| `meanStdDev.s0_qc_u8` | S0 | control | `ingest.py:57` (_qc) | `uint8[1000, 1000]` | 2 |  |
| `minMaxLoc.s0_qc_u8` | S0 | kleidicv | `ingest.py:60` (_qc) | `uint8[1000, 1000]` | 2 | Python minMaxLoc wraps minMaxIdx |
| `createHanningWindow.s1_window_f32` | S1 | control | `lattice.py:52` (_power_spectrum) | `(1000,1000), 5` | 1 per detect_lattice |  |
| `dft.s1_complex_f32` | S1 | control | `lattice.py:57` (_power_spectrum) | `float32[1024, 1024], flags=16` | 1 per detect_lattice |  |
| `magnitude.s1_spectrum_f32` | S1 | control | `lattice.py:58` (_power_spectrum) | `float32[1024, 1024], float32[1024, 1024]` | 1 per detect_lattice |  |
| `log.s1_spectrum_f32` | S1 | control | `lattice.py:59` (_power_spectrum) | `float32[1024, 1024]` | 1 per detect_lattice |  |
| `minMaxLoc.s1_peak_f32` | S1 | kleidicv | `lattice.py:96` (strongest) | `float32[1024, 1024]` | 2-33 per detect_lattice (peak search) | float32 input: KleidiCV coverage unverified (generic OpenCV path if unsupported); Python minMaxLoc wraps minMaxIdx |
| `pyrDown.s2_coarse_u8` | S2 | kleidicv | `rectify.py:49` (rectify) | `uint8[1000, 1000]` | 1 |  |
| `warpPerspective.s2_hypothesis_u8` | S2 | kleidicv | `rectify.py:68` (rectify) | `uint8[500, 500], float64[3, 3], (500,500), flags=1` | up to 4 (hypotheses) | half-res coarse frame, INTER_LINEAR, BORDER_CONSTANT |
| `warpPerspective.s2_derotate_u8` | S2 | kleidicv | `engine.py:108` (_warp) | `uint8[1000, 1000], float64[3, 3], (1000,1000), flags=1, borderMode=4` | 0-2 (only when the axis residual >= 0.75 deg) | BORDER_REFLECT101: KleidiCV border coverage unverified |
| `sepFilter2D.s3_along_f32` | S3 | kleidicv | `roughness.py:111,113` (residual) | `float32[1000, 1000], 5, float32[1], float32[41]` | 2 | float32 input: KleidiCV coverage unverified (generic OpenCV path if unsupported); 1-tap x k along-line Gaussian, k = 2 lattice periods snapped to an odd size |
| `Sobel.s3_grad_f32` | S3 | kleidicv | `roughness.py:118,119` (residual) | `float32[1000, 1000], 5, 1, 0, ksize=3` | 4 (dx, dy) x 2 frames | float32 input: KleidiCV coverage unverified (generic OpenCV path if unsupported); ksize 3, dx and dy |
| `magnitude.s3_grad_f32` | S3 | control | `roughness.py:120` (residual) | `float32[1000, 1000], float32[1000, 1000]` | 2 |  |
| `erode.s3_band_u8` | S3 | kleidicv | `roughness.py:125` (residual) | `uint8[1000, 1000], uint8[3, 3]` | 2 | 3x3 ellipse |
| `dilate.s3_band_u8` | S3 | kleidicv | `roughness.py:126` (residual) | `uint8[1000, 1000], uint8[3, 3], iterations=2` | 2 | 3x3 ellipse, iterations=2 |
| `GaussianBlur.s3_localnorm_f32` | S3 | kleidicv | `roughness.py:134,135` (residual) | `float32[1000, 1000], (25,25), 8` | 4 (mean, energy) x 2 frames | float32 input: KleidiCV coverage unverified (generic OpenCV path if unsupported); 25x25, sigma 8 (norm_tile 24 snapped) |
| `matchTemplate.s4_ccoeff_normed_f32` | S4 | control | `score.py:77` (score_residual) | `float32[1000, 1000], float32[106, 105], 5` | 1 | residual-on-residual TM_CCOEFF_NORMED |
| `minMaxLoc.s4_anchor_f32` | S4 | kleidicv | `score.py:78` (score_residual) | `float32[895, 896]` | 1 | float32 input: KleidiCV coverage unverified (generic OpenCV path if unsupported); Python minMaxLoc wraps minMaxIdx |
| `dilate.s4_nms_f32` | S4 | kleidicv | `score.py:59` (_nms_peaks) | `float32[895, 896], uint8[5, 5]` | 1 | float32 input: KleidiCV coverage unverified (generic OpenCV path if unsupported); 5x5 rect NMS |
| `createHanningWindow.s8_window_f32` | S8 | control | `subpixel.py:70` (refine) | `(105,106), 5` | 1 |  |
| `phaseCorrelate.s8_hann_f32` | S8 | control | `subpixel.py:73` (refine) | `float32[106, 105], float32[106, 105], float32[106, 105]` | 1 |  |
| `findTransformECC.s8_translation_f32` | S8 | control | `subpixel.py:86` (refine) | `float32[106, 105], float32[106, 105], float32[2, 3], 0, (3,100,1e-05), None, 5` | 1 | MOTION_TRANSLATION, 100 iters, eps 1e-5, gaussFiltSize 5 |
| `convertScaleAbs.s9_norm_u8` | S9 | control | `explain.py:40` (_norm_u8) | `float32[1000, 1000], alpha=23.7401, beta=-0` | 2 | not listed (KleidiCV lists convertTo, not convertScaleAbs) |
| `applyColorMap.s9_heatmap` | S9 | control | `explain.py:61` (heatmap) | `uint8[1000, 1000], 14` | 2 |  |
| `imencode.s9_png` | S9 | control | `explain.py:53` (_png) | `'.png', uint8[1000, 1000, 3]` | 4 | largest of the 4 PNG encodes is timed |
| `resize.s9_gallery_nearest_u8` | S9 | kleidicv | `explain.py:83` (gallery) | `uint8[106, 105], (105,106), interpolation=0` | <=5 | INTER_NEAREST, template size |
| `cvtColor.s9_gallery_gray2bgr_u8` | S9 | kleidicv | `explain.py:84` (gallery) | `uint8[106, 105], 8` | <=5 | template-sized crop |
| `putText.s9_gallery` | S9 | control | `explain.py:85` (gallery) | `uint8[106, 105, 3], '#1 0.23', (3,14), 0, 0.4, (0,255,0), 1, 16` | <=5 |  |
| `copyMakeBorder.s9_gallery` | S9 | control | `explain.py:95` (gallery) | `uint8[106, 105, 3], 4, 4, 4, 4, 0, value=(30,30,30)` | <=5 |  |
| `hconcat.s9_gallery` | S9 | control | `explain.py:98` (gallery) | `[uint8[114, 113, 3],uint8[114, 113, 3],uint8[114, 113, 3],uint8[114, 113, 3],...` | 1 |  |
| `cvtColor.s9_overlay_gray2bgr_u8` | S9 | kleidicv | `explain.py:105` (overlay) | `uint8[1000, 1000], 8` | 1 | full search frame |
| `addWeighted.s9_overlay_u8` | S9 | control | `explain.py:106` (overlay) | `uint8[1000, 1000, 3], 0.6, uint8[1000, 1000, 3], 0.4, 0` | 1 |  |
| `rectangle.s9_overlay` | S9 | control | `explain.py:109` (overlay) | `uint8[1000, 1000, 3], (35,562), (140,668), (0,255,0), 2` | 1 |  |
| `drawMarker.s9_overlay` | S9 | control | `explain.py:110` (overlay) | `uint8[1000, 1000, 3], (87,615), (0,0,255), 0, 18, 2` | 1 |  |
| `putText.s9_overlay` | S9 | control | `explain.py:111` (overlay) | `uint8[1000, 1000, 3], 'score 0.229', (35,556), 0, 0.5, (0,255,0), 1, 16` | 1 |  |
| `numpy.s6_v1_fingerprint_rerank` | S6 | numpy | `cascade.py` (_v1_rerank) | `pool=64 template=[119, 94] search=[1000, 1000] pair=clean/0000` | 0-1 | localization.ler_fingerprint over the 64-candidate pool (cascade.rerank mode='v1'); runs in auto/accurate when the margin escalates |
| `numpy.s3_residual_elementwise` | S3 | numpy | `roughness.py` (residual) | `float32[1000, 1000]` | 1 | I - smooth, r * mask, r * r, local std and the final normalisation on the search frame |
| `GaussianBlur.cf_localnorm_u8` | S3 | counterfactual | not an engine call; mirrors `roughness.py` (residual) | `uint8[1000, 1000], ((25, 25), 8.0)` | - | same 25x25 / sigma as s3_localnorm_f32 on the uint8 search frame |
| `sepFilter2D.cf_along_u8` | S3 | counterfactual | not an engine call; mirrors `roughness.py` (residual) | `uint8[1000, 1000], -1, (1,), (41,)` | - | same 1 x k kernel as s3_along_f32 on the uint8 search frame (uint8 output) |
| `Sobel.cf_grad_u8_s16` | S3 | counterfactual | not an engine call; mirrors `roughness.py` (residual) | `uint8[1000, 1000] -> int16` | - | same ksize 3 dx as s3_grad_f32, uint8 -> int16 |

Engine cv2 calls that are deliberately not timed:

- `cv2.cvtColor` at `ingest.py:51` (_to_gray): only for 3-channel input; every DEV/agent frame is single-channel
- `cv2.getOptimalDFTSize` at `lattice.py:54` (_power_spectrum): scalar helper (returns 1024 here)
- `cv2.getRotationMatrix2D` at `rectify.py:41` (_homography): scalar 2x3 matrix
- `cv2.getRotationMatrix2D` at `engine.py:100` (_rotation): scalar 2x3 matrix
- `cv2.getGaussianKernel` at `roughness.py:109` (residual): builds a 1-D kernel (k taps)
- `cv2.getStructuringElement` at `roughness.py:123,124` (residual): builds a 3x3 element
- `cv2.getStructuringElement` at `score.py:59` (_nms_peaks): builds a 5x5 element

KleidiCV-listed functions that the v2 hot path never calls, so COOL cannot touch them here:
medianBlur, transpose, rotate, remap, convertTo, absdiff, add, subtract, multiply,
calcOpticalFlowPyrLK (`ops.LISTED_BUT_UNUSED`). The S3 element-wise maths (`I - smooth`,
`r * mask`, ...) is NumPy, not cv2.subtract/multiply.

Mode coverage: S6 re-rank and S9 explain run only in auto/accurate. The DENSE-only spectral
enumeration (`localization.spectral`) is not benchmarked. `tests/bench/test_ops.py` fails if an
engine cv2 call has no row (or no reason in `NOT_BENCHMARKED`), if a cited call site moved, or
if the engine's stage names change. `census.py` and `microbench.py` also report any call site
seen at runtime that is not in the table (`uncovered_sites`, which must be empty).

## Arms and `run_all.sh` contract (infra)

The infra bench stack runs, from the repository root of a checkout at the campaign SHA:

```bash
BENCH_ARM=arm-cool BENCH_RUN_ID=r20261021a BENCH_S3_URI=s3://<bucket>/bench bash bench/cool/run_all.sh
```

| Variable | Required | Meaning |
|---|---|---|
| `BENCH_ARM` | yes | `x86-pip` (c7i.4xlarge), `arm-pip` (c8g.4xlarge), `arm-cool` (c8g.4xlarge, COOL AMI), `arm-cool-large` (c8g.large, COOL AMI) |
| `BENCH_RUN_ID` | yes | `[A-Za-z0-9._-]+`, the same for all arms of one campaign |
| `BENCH_OUT_DIR` | no | output root, default `out/$BENCH_RUN_ID`. The arm writes `$BENCH_OUT_DIR/$BENCH_ARM/` and `$BENCH_OUT_DIR/$BENCH_ARM.tar.gz` |
| `BENCH_S3_URI` | no | `s3://` prefix. Uploads with `aws s3 cp` to `$BENCH_S3_URI/$BENCH_RUN_ID/$BENCH_ARM/` (directory) and `$BENCH_S3_URI/$BENCH_RUN_ID/$BENCH_ARM.tar.gz` |
| `BENCH_QUICK=1` | no | smoke run only (few runs) |
| `BENCH_SKIP_VENV=1` | no | use `$PYTHON` as is (local smoke). All guards still apply |
| `BENCH_VENV_DIR`, `BENCH_BASE_PYTHON`, `ENGINE_GIT_SHA`, `BENCH_ALLOW_MISMATCH=1` | no | venv path (default `~/.venvs/driftsense-bench-<arm>`), base interpreter for pip venvs (default `python3.12`), recorded SHA, continue on a CPU-arch mismatch |

Venvs (Ubuntu 24.04; the pip arms need `python3.12-venv`; the S3 upload needs the AWS CLI):

- **pip arms**: `python3.12 -m venv`, then `pip install -r requirements.lock`, which includes
  `opencv-python-headless==5.0.0.93`. The script aborts if cv2 turns out to be under `/opt/cool`.
- **COOL arms**: `/opt/cool/venvs/python_3.12/bin/python -m venv --system-site-packages`, plus a
  `.pth` that adds the COOL venv's own site-packages (a venv made from a venv inherits only the
  base interpreter's site). Then `pip install --no-deps -r requirements-cool.lock`, which pins
  no OpenCV; `--no-deps` stops any transitive OpenCV wheel. **OpenCV is never pip-installed.**
  The script aborts (exit 3) if the lock mentions opencv, if `cv2.__file__` is not under
  `/opt/cool`, or if any `opencv*` distribution is installed outside `/opt/cool`. Nothing,
  including `BENCH_ALLOW_MISMATCH`, overrides this guard. The repo is used via `PYTHONPATH`,
  never `pip install .` (its dependencies include opencv-python-headless).
  Caveat: the lock pins numpy 2.5.4, which shadows the COOL venv's numpy. If COOL's cv2 fails to
  import against it, the cv2 import check aborts the arm before any step runs.

Steps: `collect_evidence`, `census`, `microbench --threads 1 all`, `stagebench --threads 1 all`,
`e2e`, `agent_bench`, `equality dump`. All output, with a `run.log` and a `STATUS` file
(`OK`, or `FAILED` plus the failed steps), lands in `$BENCH_OUT_DIR/$BENCH_ARM/`. A failed step
does not stop the others. The script still uploads, then exits 1. Exit codes: 0 ok,
1 step failed, 2 bad input, 3 guard tripped.

## Report

```bash
aws s3 cp --recursive s3://<bucket>/bench/<run_id>/ out/<run_id>/ --exclude '*.tar.gz'
python -m bench.cool.report --run-dir out/<run_id> --out results/bench/
```

The report writes `results/bench/cool_bench.json` (a `driftsense.contracts.ResultsFile`,
`dataset="bench"`, `env` from the `arm-cool` arm, so `env.cool` is true only if that arm's cv2
was under `/opt/cool`), `cool_bench_docs.json` (flat docs-ready keys, including strings),
`cool_bench_summary.md`, `cool_bench_meta.json` and `charts/` (a speed-up forest plot per
comparison and thread tag, plus cost-per-1,000 bars).

Default comparisons: `arm-cool` vs `arm-pip` (the COOL effect, gated by equality), `arm-pip` vs
`x86-pip` and `arm-cool` vs `x86-pip`. `arm-cool-large` feeds the production-size cost
figure. `--compare cand:base` adds others. `--price type=usd` overrides a price.
`--instance-type label=type` fills in a missing IMDS value.

Cost per 1,000 = (EC2 $/h + COOL fee $/h) / throughput per hour x 1000, using
`bench/cool/prices.json` (ap-south-1 on-demand; every entry carries `source` and `as_of`). It is
computed for localizations (`e2e.fast`, `e2e.auto`) and for agent episodes.

### Keys in `cool_bench.json`

| Key | Unit |
|---|---|
| `speedup.<cand>_vs_<base>.<kleidicv/control/numpy/counterfactual>.<fn>.<variant>.<t1/tN>` | x (CI, n) |
| `speedup.<cmp>.<group>.geomean.<t1/tN>` | x |
| `speedup.<cmp>.stage.<mode>.<stage>.<t1/tN>`, `speedup.<cmp>.e2e.<mode>.{latency.tN,loaded_latency.t1}`, `speedup.<cmp>.agent.{latency.tN,loaded_latency.t1}` | x |
| `speedup.<cmp>.e2e.<mode>.throughput`, `speedup.<cmp>.agent.throughput` | x (cand/base rate) |
| `amdahl.<cmp>.<mode>.{predicted_speedup,upper_bound_speedup}` | x |
| `equality.<cmp>.{pass, offset.max_abs_diff_px, offset.pass, offset.top1_agreement, offset.n_missing, agent.max_abs_diff_nm, agent.outcome_agreement, agent.pass, scoremap.<map>.max_abs_diff, scoremap.<map>.sha256_equal}` | px / nm / bool |
| `cost.<arm>.usd_per_hour`, `cost.<arm>.e2e.<mode>.usd_per_1000_localizations`, `cost.<arm>.agent.usd_per_1000_episodes` | USD |
| `arm.<arm>.<sample key>.median_ms` (+ `.p95_ms` for stage/e2e/agent), `arm.<arm>.e2e.<mode>.throughput.per_hour`, `arm.<arm>.agent.throughput.per_hour`, `arm.<arm>.census.<mode>.share.*`, `arm.<arm>.census.<mode>.total_ms` | ms, /h, fraction |
| `headline.cool_vs_pip.{e2e_auto_throughput,e2e_fast_throughput,agent_throughput,kleidicv_geomean_t1,control_geomean_t1,numpy_geomean_t1}`, `headline.cool_vs_x86.e2e_auto_throughput`, `headline.graviton_vs_x86.e2e_auto_throughput`, `headline.equality.cool_vs_pip.pass`, `headline.cost.<arm>.auto.usd_per_1000` | copies of the keys above |
| `meta.n_arms`, `meta.git_sha_consistent`, `meta.local_smoke` | |

`cool_bench_docs.json` adds `arm.<arm>.{instance_type, region, ami_id, product_codes, arch,
cv2_version, cv2_file, cool, git_sha, numpy, python, build_mentions_kleidicv,
build_kleidicv_lines, ldd_mentions_kleidicv, cool_version_files, opencv_dists, quick}`, the
`<cmp>.geomean.<group>.<tag>` values with `.ci`/`.verdict`, the throughput and total-stage
speed-ups with verdicts, `headline.*` with `.ci`/`.verdict`, `cost.<arm>.*`,
`equality.<cmp>.*`, `prices.as_of`, `prices.region`, `smoke` and `verdict.no_diff_phrase`.

## Run locally (plumbing check only)

```bash
BENCH_ARM=x86-pip BENCH_RUN_ID=local BENCH_OUT_DIR=/tmp/bench BENCH_QUICK=1 BENCH_SKIP_VENV=1 \
  PYTHON=python bash bench/cool/run_all.sh          # about 1 minute
python -m bench.cool.report --run-dir /tmp/bench --out /tmp/bench/report
python -m bench.cool.microbench --list              # every op with engine file:line
pytest -q tests/bench                               # fast; add -m slow for the process-pool tests
```

Local numbers are **not results**. Turbo, thermals and background load move them by tens of
percent, and the NumPy controls exist to catch exactly that. Keep them in a temp directory.
