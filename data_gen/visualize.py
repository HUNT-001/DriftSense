"""
Dataset QC visualization.

Produces figures that are directly usable in the submission deck:

  contact_sheet_<tier>.png  — grid of search images with GT marked and the
                              reference patch inset, for eyeballing realism
  pair_detail_<id>.png      — one pair in full: reference, search, GT zoom
  effects_ladder.png        — the SEM rendering chain applied step by step
  structure_gallery.png     — DRAM grid / DRAM staggered / FinFET variants
  ambiguity_demo.png        — NCC score map showing multiple equal peaks

Usage
-----
    python -m data_gen.visualize --tier clean
    python -m data_gen.visualize --effects
    python -m data_gen.visualize --gallery
    python -m data_gen.visualize --ambiguity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (AperiodicityConfig, get_tier, SEARCH_H, SEARCH_W)
from data_gen.structures import (generate_dram, generate_finfet,
                                  DRAMParams, FinFETParams)
from data_gen.sem_effects import (sample_acquisition, apply_gaussian_blur,
                                   apply_edge_brightening, apply_charging,
                                   apply_vignetting, apply_drift_distortion,
                                   apply_shot_noise, apply_read_noise)
from data_gen.scene_composer import SceneComposer


# ---------------------------------------------------------------------------

def _load_pair(ds_dir: Path, pid: int):
    ref = cv2.imread(str(ds_dir / f"pair_{pid:04d}_ref.png"),
                     cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(str(ds_dir / f"pair_{pid:04d}_search.png"),
                        cv2.IMREAD_GRAYSCALE)
    return ref, search


# ---------------------------------------------------------------------------

def contact_sheet(tier: str, dataset_root: str, out_dir: str,
                   n: int = 12) -> Path:
    """Grid of search images with GT crosshair and reference inset."""
    ds_dir = Path(dataset_root) / tier
    with open(ds_dir / "manifest.json") as f:
        man = json.load(f)
    pairs = man["pairs"][:n]

    cols = 4
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, meta in zip(axes, pairs):
        ref, search = _load_pair(ds_dir, meta["pair_id"])
        if search is None:
            continue
        ax.imshow(search, cmap="gray", vmin=0, vmax=255)
        ax.plot(meta["gt_x"], meta["gt_y"], "+", color="lime",
                markersize=16, markeredgewidth=2)
        rect = plt.Rectangle(
            (meta["gt_x"] - meta["ref_w"] / 2, meta["gt_y"] - meta["ref_h"] / 2),
            meta["ref_w"], meta["ref_h"],
            fill=False, edgecolor="lime", linewidth=1.5)
        ax.add_patch(rect)
        # Reference inset (top-left)
        if ref is not None:
            inset = cv2.resize(ref, (150, 150),
                                interpolation=cv2.INTER_NEAREST)
            ax.imshow(inset, cmap="gray", vmin=0, vmax=255,
                      extent=(5, 155, 155, 5))
            ax.add_patch(plt.Rectangle((5, 5), 150, 150, fill=False,
                                        edgecolor="yellow", linewidth=2))
        ax.set_title(f"#{meta['pair_id']} {meta['structure_family']}"
                     f"/{meta['structure_variant']}\n"
                     f"noise={meta['noise_level']} "
                     f"rot={meta['rotation_deg']:.1f}deg",
                     fontsize=9)
        ax.set_xlim(0, SEARCH_W); ax.set_ylim(SEARCH_H, 0)
        ax.axis("off")

    for ax in axes[len(pairs):]:
        ax.axis("off")

    fig.suptitle(f"DriftSense dataset — tier '{tier}'  "
                 f"(green = ground truth, yellow inset = reference)",
                 fontsize=13)
    fig.tight_layout()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"contact_sheet_{tier}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def pair_detail(tier: str, pair_id: int, dataset_root: str,
                 out_dir: str) -> Path:
    """Reference, full search, and a zoom around the ground-truth location."""
    ds_dir = Path(dataset_root) / tier
    with open(ds_dir / "manifest.json") as f:
        man = json.load(f)
    meta = next(m for m in man["pairs"] if m["pair_id"] == pair_id)
    ref, search = _load_pair(ds_dir, pair_id)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].imshow(ref, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Reference  {meta['ref_w']}x{meta['ref_h']} px")

    axes[1].imshow(search, cmap="gray", vmin=0, vmax=255)
    axes[1].plot(meta["gt_x"], meta["gt_y"], "+", color="lime",
                 markersize=20, markeredgewidth=2)
    axes[1].set_title(f"Search 1000x1000 — GT=({meta['gt_x']:.1f}, "
                      f"{meta['gt_y']:.1f})")

    z = 90
    x0 = int(np.clip(meta["gt_x"] - z, 0, SEARCH_W - 2 * z))
    y0 = int(np.clip(meta["gt_y"] - z, 0, SEARCH_H - 2 * z))
    axes[2].imshow(search[y0:y0 + 2 * z, x0:x0 + 2 * z],
                   cmap="gray", vmin=0, vmax=255)
    axes[2].plot(meta["gt_x"] - x0, meta["gt_y"] - y0, "+", color="lime",
                 markersize=20, markeredgewidth=2)
    axes[2].set_title("Zoom at ground truth")

    for a in axes:
        a.axis("off")
    fig.suptitle(f"tier={tier}  {meta['structure_family']}"
                 f"/{meta['structure_variant']}  noise={meta['noise_level']}  "
                 f"period=({meta['period_x']:.1f}, {meta['period_y']:.1f}) px",
                 fontsize=12)
    fig.tight_layout()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"pair_detail_{tier}_{pair_id:04d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def effects_ladder(out_dir: str) -> Path:
    """Show each SEM effect applied cumulatively — a strong explainer slide."""
    aper = AperiodicityConfig()
    rng = np.random.default_rng(4)
    clean, _ = generate_dram(300, 300, DRAMParams(
        variant="grid", period_x=26, period_y=30,
        line_width_x=7, line_width_y=7, background=0.05),
        aper, rng)

    steps = [("1. Clean structure", clean)]
    x = apply_gaussian_blur(clean, 1.3)
    steps.append(("2. + Beam blur [E3]", x))
    x = apply_edge_brightening(x, 0.38)
    steps.append(("3. + Edge brightening [E1]", x))
    x = apply_charging(x, 0.10, np.random.default_rng(5))
    steps.append(("4. + Charging [E8]", x))
    x = apply_vignetting(x, 0.20, 0.35, 0.6)
    steps.append(("5. + Vignetting [E9]", x))
    x = apply_drift_distortion(x, 0.09)
    steps.append(("6. + Scan drift [E4]", x))
    x = apply_shot_noise(x, 60.0, np.random.default_rng(6))
    steps.append(("7. + Shot noise [E2]", x))
    x = apply_read_noise(x, 0.03, np.random.default_rng(7))
    steps.append(("8. + Read noise [E6]", x))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax, (title, im) in zip(axes.ravel(), steps):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("DriftSense SEM rendering chain (physically ordered)",
                 fontsize=14)
    fig.tight_layout()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "effects_ladder.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def structure_gallery(out_dir: str) -> Path:
    """DRAM grid / DRAM staggered / FinFET plain / FinFET fin-cut."""
    aper = AperiodicityConfig()
    variants = [
        ("DRAM — orthogonal grid",
         lambda r: generate_dram(280, 280, DRAMParams(
             variant="grid", period_x=24, period_y=28,
             line_width_x=7, line_width_y=7), aper, r)[0]),
        ("DRAM — staggered (6F^2)",
         lambda r: generate_dram(280, 280, DRAMParams(
             variant="staggered", period_x=26, period_y=30,
             line_width_x=6, line_width_y=6, cell_radius=5.0,
             row_offset_frac=0.5), aper, r)[0]),
        ("FinFET — fins + gates",
         lambda r: generate_finfet(280, 280, FinFETParams(
             fin_period=11, gate_period=44, fin_width=4, gate_width=9,
             enable_fin_cut=False), aper, r)[0]),
        ("FinFET — with fin cuts",
         lambda r: generate_finfet(280, 280, FinFETParams(
             fin_period=11, gate_period=44, fin_width=4, gate_width=9,
             enable_fin_cut=True, fin_cut_period=120,
             fin_cut_width=18), aper, r)[0]),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    for i, (title, fn) in enumerate(variants):
        img = fn(np.random.default_rng(20 + i))
        axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(title, fontsize=11)
        axes[0, i].axis("off")
        # 2-D log power spectrum
        F = np.fft.fftshift(np.abs(np.fft.fft2(img - img.mean())))
        axes[1, i].imshow(np.log1p(F), cmap="magma")
        axes[1, i].set_title("2-D power spectrum", fontsize=10)
        axes[1, i].axis("off")
    fig.suptitle("Structure families and their frequency signatures",
                 fontsize=14)
    fig.tight_layout()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "structure_gallery.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def ambiguity_demo(out_dir: str) -> Path:
    """
    The money figure — a controlled A/B experiment in one image.

    Top row    ('ambiguous' tier): LER, CD variation and defects DISABLED, so
                the scene is perfectly periodic.  The NCC surface becomes a
                lattice of near-identical peaks and localization is
                information-theoretically impossible.
    Bottom row ('clean' tier)    : identical noise/blur settings, but the real
                wafer physics is enabled.  A single dominant peak appears at
                the true location.

    The ONLY difference between the rows is the aperiodic physics.  This is
    the experimental justification for the entire Stage-2 design: the signal
    that resolves the ambiguity is LER, not a bigger model.
    """
    rows = [
        ("ambiguous", "Perfectly periodic (LER/CD/defects OFF)"),
        ("clean",     "Real wafer physics ON (LER/CD/defects)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    counts = {}

    for r, (tier_name, label) in enumerate(rows):
        tier = get_tier(tier_name)
        comp = SceneComposer(seed=31337)
        ref, search, meta = comp.compose(0, tier, structure_type="dram")

        score = cv2.matchTemplate(search.astype(np.float32),
                                   ref.astype(np.float32),
                                   cv2.TM_CCOEFF_NORMED)
        _, best, _, loc = cv2.minMaxLoc(score)
        n_near = int((score > (best - 0.02)).sum())
        counts[tier_name] = n_near

        pred_x = loc[0] + meta.ref_w / 2
        pred_y = loc[1] + meta.ref_h / 2
        err = float(np.hypot(pred_x - meta.gt_x, pred_y - meta.gt_y))

        axes[r, 0].imshow(ref, cmap="gray")
        axes[r, 0].set_title(f"Reference\n{label}", fontsize=10)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(search, cmap="gray")
        axes[r, 1].plot(meta.gt_x, meta.gt_y, "+", color="lime",
                        markersize=18, markeredgewidth=2, label="ground truth")
        axes[r, 1].plot(pred_x, pred_y, "x", color="red", markersize=14,
                        markeredgewidth=2, label="NCC best")
        axes[r, 1].legend(loc="upper right", fontsize=8)
        axes[r, 1].set_title(f"Search image — NCC error = {err:.1f} px",
                             fontsize=10)
        axes[r, 1].axis("off")

        im = axes[r, 2].imshow(score, cmap="viridis")
        axes[r, 2].set_title(f"NCC score map\n{n_near} locations within 2% "
                             f"of peak (best={best:.3f})", fontsize=10)
        axes[r, 2].axis("off")
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046)

    fig.suptitle("Periodicity ambiguity is resolved by wafer physics, "
                 "not by model capacity\n"
                 "top: perfectly periodic -> lattice of equal peaks   |   "
                 "bottom: LER/CD/defects enabled -> unique peak",
                 fontsize=13)
    fig.tight_layout()
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "ambiguity_demo.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  near-peak count — ambiguous: {counts['ambiguous']}, "
          f"clean: {counts['clean']}")
    return path


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DriftSense dataset visualization")
    ap.add_argument("--tier", default="clean")
    ap.add_argument("--dataset", default="outputs/dataset")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--effects", action="store_true")
    ap.add_argument("--gallery", action="store_true")
    ap.add_argument("--ambiguity", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    made = []
    if args.effects or args.all:
        made.append(effects_ladder(args.out))
    if args.gallery or args.all:
        made.append(structure_gallery(args.out))
    if args.ambiguity or args.all:
        made.append(ambiguity_demo(args.out))
    if not (args.effects or args.gallery or args.ambiguity) or args.all:
        ds = Path(args.dataset) / args.tier
        if (ds / "manifest.json").exists():
            made.append(contact_sheet(args.tier, args.dataset, args.out))
            made.append(pair_detail(args.tier, args.pair, args.dataset,
                                     args.out))

    for p in made:
        print(f"  wrote {p}")
