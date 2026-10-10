# DriftSense — Full-Search Benchmark

Operational task: locate a reference patch inside a 1000×1000 SEM image with no
prior knowledge of its position, competing against ~10⁶ candidate placements on a
periodic wafer. Acc@5 = fraction of pairs localized within 5 px of ground truth.

**Method definitions**
- `NCC` — classical normalized cross-correlation template matching (argmax).
- `DriftSense` — spectral lattice candidate generation (Stage 2A) + global
  de-rotation (Stage 2C) + LER-fingerprint re-ranking (Stage 2B).

**Result** (40 pairs/tier, `top_k = 60`, sealed test dataset)

| Tier    | Distortion                     | NCC Acc@5 | DriftSense Acc@5 | Gain    | recall@60 |
|---------|--------------------------------|-----------|------------------|---------|-----------|
| clean   | none                           | 100.0%    | 97.5%            | −2.5    | 97.5%     |
| nominal | ±1.5° rotation, mild noise     | 82.5%     | **90.0%**        | +7.5    | 97.5%     |
| hard    | ±5° rot, ±7% scale, low-dose   | 17.5%     | **57.5%**        | **+40.0** | 97.5%   |

On the hard tier classical template matching collapses to 17.5% because the true
peak is out-scored by ~150 periodic aliases; DriftSense reaches 57.5% (median
error 0.6 px, 17 pairs rescued / 1 broken).

**recall@K is the key diagnostic.** It is 97.5% on every tier — spectral candidate
generation almost always *contains* the truth. The remaining hard-tier gap
(97.5% recall → 57.5% Acc@5) is therefore attributable to Stage-2B *discrimination*
under compound rotation+scale distortion, not to candidate generation. An orientation
ablation (`experiments/orientation_error.py`, `localization/lattice_fit.py`)
confirmed that improving the de-rotation angle does not lift end-to-end accuracy:
the consensus estimator already has 0.83° median error, and the bottleneck is the
fingerprint, not the geometry.

**Reproduce**

```bash
python -m experiments.run_stage2 --tier hard    --method driftsense --top_k 60
python -m experiments.run_stage2 --tier nominal --method driftsense --top_k 60
python -m experiments.run_stage2 --tier clean   --method driftsense --top_k 60
# baseline for comparison:
python -m experiments.run_stage2 --all_tiers --method ncc
```
