"""Failure gallery — the 12 worst committed-but-wrong cases from a sealed arm.

Reads the per-pair sidecar written by eval/sealed_run.py
(``results/sealed/<arm>.pairs.jsonl``), selects the committed pairs whose rank-1
is wrong (the false-confident cases the active-perception agent exists to catch),
ranks them by committed-centre pixel error, and renders the 12 worst as a montage
with the search frame, the committed location (red) and the ground truth (green).

Reading the pair IMAGES means touching the dataset root. For the sealed root this
is gated by the same unlock as eval/sealed_run.py; a self-test run points at the
DEV root instead.

    python -m eval.failure_gallery --arm v2_single
    python -m eval.failure_gallery --arm v2_single --root outputs/dataset   # self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval.sealed_run import OUT_DIR, SEALED_ROOT, require_unlock  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_WORST = 12


def load_false_confident(sidecar: Path) -> list[dict]:
    recs = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    fc = [r for r in recs if r.get("committed") and not r.get("hit")]
    fc.sort(key=lambda r: r.get("px_error", 0.0) or 0.0, reverse=True)
    return fc[:N_WORST]


def _search_png(root: Path, tier: str, pair_id: int) -> np.ndarray | None:
    p = root / tier / f"pair_{pair_id:04d}_search.png"
    if not p.exists():
        return None
    return cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)


def render(cases: list[dict], root: Path, out: Path) -> Path:
    n = max(1, len(cases))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, c in zip(axes, cases, strict=False):
        img = _search_png(root, c["tier"], c["pair_id"])
        if img is not None:
            ax.imshow(img, cmap="gray")
        ax.plot([c["gt_x"]], [c["gt_y"]], "+", color="#2ca02c", markersize=12, markeredgewidth=2)
        ax.plot([c["cx"]], [c["cy"]], "x", color="#d62728", markersize=10, markeredgewidth=2)
        pe = c.get("px_error")
        ax.set_title(f"{c['tier']}/{c['family']} #{c['pair_id']}\nerr={pe:.1f}px", fontsize=8)
    fig.suptitle(f"False-confident gallery — {len(cases)} worst (green=GT, red=committed)", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render the sealed false-confident gallery")
    ap.add_argument("--arm", default="v2_single")
    ap.add_argument("--in-dir", default=str(OUT_DIR))
    ap.add_argument("--root", default=None, help="dataset root for images (default: sealed, gated)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir)
    sidecar = in_dir / f"{args.arm}.pairs.jsonl"
    if not sidecar.exists():
        print(f"no sidecar at {sidecar}; run the arm first", file=sys.stderr)
        return 1

    if args.root:
        root = Path(args.root)
    else:
        require_unlock()  # sealed images are gated
        root = SEALED_ROOT

    cases = load_false_confident(sidecar)
    out = Path(args.out) if args.out else in_dir / f"{args.arm}_failures.png"
    render(cases, root, out)
    print(f"gallery: {len(cases)} case(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
