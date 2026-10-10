"""
Dataset generation driver — v2.

Generates tiered (reference, search, ground-truth) datasets and performs a
self-audit verifying that the declared ground truth is actually correct.

Usage
-----
    python -m data_gen.dataset_gen --tier clean     --n_pairs 40
    python -m data_gen.dataset_gen --tier nominal   --n_pairs 40
    python -m data_gen.dataset_gen --tier hard      --n_pairs 40
    python -m data_gen.dataset_gen --tier ambiguous --n_pairs 20
    python -m data_gen.dataset_gen --all            --n_pairs 40

Outputs (per tier, under outputs/dataset/<tier>/)
-------------------------------------------------
    pair_XXXX_ref.png
    pair_XXXX_search.png
    manifest.json    — ground truth + full acquisition params + tier config
    summary.txt      — human-readable distribution + self-audit report
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_tier, tier_to_dict, TIERS
from data_gen.scene_composer import (SceneComposer, save_pair, write_manifest,
                                      verify_pair_geometry)


# ---------------------------------------------------------------------------
# Composition plan
# ---------------------------------------------------------------------------

def build_plan(n_pairs: int, tier, rng: np.random.Generator):
    """
    Balanced allocation of structure families and noise levels.

    Guarantees an even DRAM/FinFET split (the challenge judges both equally)
    and an even spread over the tier's permitted noise levels.
    """
    families = ["dram", "finfet"] * (n_pairs // 2 + 1)
    families = families[:n_pairs]

    levels = list(tier.noise_levels)
    noise = [levels[i % len(levels)] for i in range(n_pairs)]

    rng.shuffle(families)
    rng.shuffle(noise)
    return list(zip(families, noise))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_dataset(tier_name: str,
                      n_pairs: int,
                      seed: int,
                      out_root: str,
                      audit: bool = True,
                      quiet: bool = False) -> dict:

    tier = get_tier(tier_name)
    out_path = Path(out_root) / tier_name
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    plan = build_plan(n_pairs, tier, rng)
    composer = SceneComposer(seed=seed + 991)

    if not quiet:
        print(f"\n=== Tier '{tier_name}' — generating {n_pairs} pairs ===")
        print(f"{'ID':>4} {'Family':>7} {'Variant':>10} {'Noise':>7} "
              f"{'gt_x':>8} {'gt_y':>8} {'rot':>6} {'scale':>7} {'t(s)':>6}")
        print("-" * 78)

    metas = []
    audit_offsets = []
    t_total = 0.0

    for i, (family, noise) in enumerate(plan):
        t0 = time.perf_counter()
        ref, search, meta = composer.compose(
            pair_id=i, tier=tier, structure_type=family, noise_level=noise)
        save_pair(ref, search, meta, out_path)
        metas.append(meta)
        dt = time.perf_counter() - t0
        t_total += dt

        # Self-audit: rotation/scale-compensated and alias-immune, so it is
        # valid for EVERY tier (see verify_pair_geometry docstring).
        if audit:
            ok, offset = verify_pair_geometry(ref, search, meta)
            audit_offsets.append(offset)

        if not quiet:
            print(f"{i:>4} {meta.structure_family:>7} {meta.structure_variant:>10} "
                  f"{meta.noise_level:>7} {meta.gt_x:>8.2f} {meta.gt_y:>8.2f} "
                  f"{meta.rotation_deg:>6.2f} {meta.scale:>7.4f} {dt:>6.2f}")

    # ---- Manifest ----
    manifest_path = write_manifest(
        metas, out_path,
        extra={
            "tier_config": tier_to_dict(tier),
            "seed": seed,
            "n_pairs": n_pairs,
            "generator_version": "2.0",
        })

    # ---- Summary + audit ----
    dram_n   = sum(1 for m in metas if m.structure_family == "dram")
    finfet_n = sum(1 for m in metas if m.structure_family == "finfet")

    audit_line = "  (audit disabled)"
    audit_pass = None
    if audit_offsets:
        med = statistics.median(audit_offsets)
        mx  = max(audit_offsets)
        n_ok = sum(1 for o in audit_offsets if o <= 3.0)
        audit_pass = n_ok / len(audit_offsets)
        audit_line = (f"  checked={len(audit_offsets)}  "
                      f"pass@3px={audit_pass:.1%}  "
                      f"median_offset={med:.2f}px  max={mx:.2f}px")

    with open(out_path / "summary.txt", "w") as f:
        f.write(f"DriftSense Dataset — tier '{tier_name}'\n")
        f.write("=" * 62 + "\n")
        f.write(f"pairs: {n_pairs}   seed: {seed}   "
                f"gen time: {t_total:.1f}s ({t_total/n_pairs:.2f}s/pair)\n\n")
        f.write(f"Structure families:  DRAM={dram_n}  FinFET={finfet_n}\n")
        for lvl in ("low", "medium", "high"):
            c = sum(1 for m in metas if m.noise_level == lvl)
            if c:
                f.write(f"  noise '{lvl}': {c}\n")
        f.write("\nGeometry ranges applied:\n")
        rots = [m.rotation_deg for m in metas]
        scls = [m.scale for m in metas]
        f.write(f"  rotation: [{min(rots):.2f}, {max(rots):.2f}] deg\n")
        f.write(f"  scale:    [{min(scls):.4f}, {max(scls):.4f}]\n")
        f.write("\nStructural periods (px, post-scale):\n")
        pxs = [m.period_x for m in metas]
        pys = [m.period_y for m in metas]
        f.write(f"  period_x: [{min(pxs):.1f}, {max(pxs):.1f}]\n")
        f.write(f"  period_y: [{min(pys):.1f}, {max(pys):.1f}]\n")
        f.write(f"\nDefects per scene: mean="
                f"{statistics.mean(m.n_defects for m in metas):.1f}\n")
        f.write("\nGROUND-TRUTH SELF-AUDIT\n")
        f.write("  Local NCC around declared GT must peak at GT.\n")
        f.write(audit_line + "\n")

    if not quiet:
        print("-" * 78)
        print(f"Done in {t_total:.1f}s ({t_total/n_pairs:.2f}s/pair)")
        print(f"GT self-audit:{audit_line}")
        print(f"Manifest → {manifest_path}")

    return {
        "tier": tier_name,
        "n_pairs": n_pairs,
        "time_s": t_total,
        "audit_pass": audit_pass,
        "path": str(out_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DriftSense dataset generator v2")
    ap.add_argument("--tier", default="nominal", choices=list(TIERS),
                    help="Difficulty tier to generate")
    ap.add_argument("--all", action="store_true",
                    help="Generate every tier")
    ap.add_argument("--n_pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/dataset")
    ap.add_argument("--no_audit", action="store_true")
    args = ap.parse_args()

    if args.n_pairs < 30:
        print("Challenge requires >= 30 pairs; raising to 30.")
        args.n_pairs = 30

    tiers = list(TIERS) if args.all else [args.tier]
    results = []
    for i, t in enumerate(tiers):
        results.append(generate_dataset(
            tier_name=t,
            n_pairs=args.n_pairs,
            seed=args.seed + i * 1000,
            out_root=args.out,
            audit=not args.no_audit,
        ))

    print("\n" + "=" * 62)
    print(f"{'Tier':>12} {'Pairs':>6} {'Time(s)':>8} {'GT audit':>10}")
    print("-" * 62)
    for r in results:
        ap_str = "n/a" if r["audit_pass"] is None else f"{r['audit_pass']:.1%}"
        print(f"{r['tier']:>12} {r['n_pairs']:>6} {r['time_s']:>8.1f} {ap_str:>10}")
