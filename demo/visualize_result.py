"""
DriftSense demo — annotated result montage.

Runs the full pipeline on one (reference, search) SEM pair and renders a single
figure that tells the whole story:

  [1] Search image, classical NCC pick (wrong on periodic wafers) vs DriftSense
      pick vs ground truth  ->  "why this is hard, and that we solve it"
  [2] Search power spectrum with the detected reciprocal lattice
      ->  "frequency tells you WHERE a copy could be"
  [3] Reference patch vs located patch, with LER fingerprint similarity
      ->  "line-edge roughness tells you WHICH copy is real"

Usage
-----
    python -m demo.visualize_result --tier hard --pair 5
    python -m demo.visualize_result --ref path/ref.png --search path/search.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localization.baseline import ncc_localize
from localization.ler_localizer import localize
from localization.lattice_fit import detect_peaks
from localization.ler_fingerprint import extract_fingerprint, fingerprint_similarity

# Colour-blind-safe: truth=blue, DriftSense=green, NCC=orange/red.
C_TRUTH = "#2563EB"
C_DS = "#16A34A"
C_NCC = "#EA580C"


def _load(p):
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def _box(ax, cx, cy, w, h, color, label, lw=2.2, ls="-"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                           edgecolor=color, linewidth=lw, linestyle=ls,
                           label=label))


def render(ref, search, out_path, gt=None, title_extra=""):
    rh, rw = ref.shape

    ncc = ncc_localize(ref, search)
    res = localize(ref, search, top_k=60, ncc_weight=0.0,
                   candidate_source="lattice", derotate=True)

    # Located patch, and the fingerprint scores the pipeline actually used
    # (de-warped max), so the number reflects the real discrimination margin.
    lx = int(np.clip(res.pred_x - rw / 2, 0, search.shape[1] - rw))
    ly = int(np.clip(res.pred_y - rh / 2, 0, search.shape[0] - rh))
    found_patch = search[ly:ly + rh, lx:lx + rw]
    fp_sorted = sorted((c.fp for c in res.candidates if np.isfinite(c.fp)),
                       reverse=True)
    sim = fp_sorted[0] if fp_sorted else float(res.score)
    runner = fp_sorted[1] if len(fp_sorted) > 1 else float("nan")

    fig = plt.figure(figsize=(15, 5.4), dpi=130)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.22)

    # ---- Panel 1: search with all three markers ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(search, cmap="gray", interpolation="nearest")
    # a few strongest NCC alias candidates, faint
    for c in sorted(res.candidates, key=lambda c: -c.ncc)[:12]:
        ax1.plot(c.x, c.y, ".", color=C_NCC, ms=3, alpha=0.45)
    _box(ax1, ncc.pred_x, ncc.pred_y, rw, rh, C_NCC, "NCC pick (classical)", ls="--")
    _box(ax1, res.pred_x, res.pred_y, rw, rh, C_DS, "DriftSense pick")
    if gt is not None:
        _box(ax1, gt[0], gt[1], rw, rh, C_TRUTH, "Ground truth", lw=1.6, ls=":")
    ax1.set_title("Full-image search (1000×1000)", fontsize=11, fontweight="bold")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ---- Panel 2: power spectrum + detected lattice ----
    ax2 = fig.add_subplot(gs[0, 1])
    img = search - search.mean()
    F = np.fft.fftshift(np.abs(np.fft.fft2(img * np.outer(
        np.hanning(search.shape[0]), np.hanning(search.shape[1])))))
    ax2.imshow(np.log1p(F), cmap="magma", interpolation="nearest")
    H, W = F.shape
    peaks = detect_peaks(search, n_peaks=16)
    for kx, ky, _ in peaks:
        ax2.plot(kx * W + W / 2, ky * H + H / 2, "o", mfc="none",
                 mec="#22D3EE", mew=1.4, ms=9)
    ax2.set_title("Search spectrum → reciprocal lattice\n"
                  "“where a copy could be”", fontsize=10, fontweight="bold", pad=8)
    ax2.set_xticks([]); ax2.set_yticks([])

    # ---- Panel 3: reference vs located patch ----
    ax3 = fig.add_subplot(gs[0, 2])
    pair = np.hstack([ref, np.ones((rh, 6)), found_patch])
    ax3.imshow(pair, cmap="gray", interpolation="nearest")
    ax3.axvline(rw + 3, color=C_DS, lw=1)
    margin = f"   best alias {runner:.2f}" if np.isfinite(runner) else ""
    ax3.set_title(f"LER fingerprint — “which copy is real”\n"
                  f"winner {sim:.2f}{margin}", fontsize=10, fontweight="bold")
    ax3.set_xlabel("reference   |   located", fontsize=9)
    ax3.set_xticks([]); ax3.set_yticks([])

    # ---- header ----
    ncc_err = f"{ncc.error(*gt):.0f} px" if gt is not None else "n/a"
    ds_err = f"{res.error(*gt):.1f} px" if gt is not None else "n/a"
    verdict = "✓ LOCATED" if (gt is not None and res.error(*gt) <= 5) else "result"
    head = (f"DriftSense — physics-aware SEM patch localization      "
            f"NCC error: {ncc_err}   |   DriftSense error: {ds_err}   {verdict}")
    fig.suptitle(head + ("     " + title_extra if title_extra else ""),
                 fontsize=11.5, fontweight="bold", y=1.04)

    fig.subplots_adjust(top=0.82)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(ncc_error=(float(ncc.error(*gt)) if gt is not None else None),
                ds_error=(float(res.error(*gt)) if gt is not None else None),
                pred=(res.pred_x, res.pred_y), sim=float(sim),
                n_candidates=res.n_candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="hard")
    ap.add_argument("--pair", type=int, default=5)
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--ref"); ap.add_argument("--search")
    ap.add_argument("--out", default="outputs/figures/demo_result.png")
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.ref and a.search:
        ref, search, gt = _load(a.ref), _load(a.search), None
    else:
        ds = Path(a.dataset) / a.tier
        meta = {m["pair_id"]: m for m in json.load(open(ds / "manifest.json"))["pairs"]}[a.pair]
        ref = _load(ds / f"pair_{a.pair:04d}_ref.png")
        search = _load(ds / f"pair_{a.pair:04d}_search.png")
        gt = (meta["gt_x"], meta["gt_y"])
    info = render(ref, search, a.out, gt=gt,
                  title_extra=f"tier={a.tier}  pair={a.pair}" if not a.ref else "")
    print(json.dumps(info, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
