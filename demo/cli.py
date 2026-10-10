"""
DriftSense command-line inference.

Deployable, dependency-light entry point: locate a reference patch inside a
search image and print the result as JSON. No GPU, no toolchain, no network.

    python -m demo.cli locate --ref ref.png --search search.png
    python -m demo.cli locate --ref ref.png --search search.png --figure out.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from localization.ler_localizer import localize


def _load(p: str) -> np.ndarray:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"error: cannot read image '{p}'")
    return img.astype(np.float32) / 255.0


def locate(ref_path: str, search_path: str, top_k: int = 60) -> dict:
    ref, search = _load(ref_path), _load(search_path)
    t0 = time.perf_counter()
    res = localize(ref, search, top_k=top_k, ncc_weight=0.0,
                   candidate_source="lattice", derotate=True)
    return {
        "method": "driftsense",
        "pred_x": round(float(res.pred_x), 3),
        "pred_y": round(float(res.pred_y), 3),
        "score": round(float(res.score), 4),
        "n_candidates": int(res.n_candidates),
        "runtime_ms": round((time.perf_counter() - t0) * 1000, 1),
        "ref": ref_path,
        "search": search_path,
    }


def main():
    ap = argparse.ArgumentParser(prog="driftsense")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lo = sub.add_parser("locate", help="locate a reference patch in a search image")
    lo.add_argument("--ref", required=True)
    lo.add_argument("--search", required=True)
    lo.add_argument("--top_k", type=int, default=60)
    lo.add_argument("--figure", help="also write an annotated result montage here")
    a = ap.parse_args()

    if a.cmd == "locate":
        out = locate(a.ref, a.search, a.top_k)
        print(json.dumps(out, indent=2))
        if a.figure:
            from demo.visualize_result import render, _load as _l
            Path(a.figure).parent.mkdir(parents=True, exist_ok=True)
            render(_l(a.ref), _l(a.search), a.figure, gt=None)
            print("figure:", a.figure)


if __name__ == "__main__":
    main()
