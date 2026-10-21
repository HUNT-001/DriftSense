"""Deterministic benchmark inputs: the committed DEV pairs under ``outputs/dataset/``.

Every arm reads byte-identical PNGs from the repository checkout, so all arms see the
same inputs. The DEV split is ``outputs/dataset/<tier>/manifest.json`` plus
``pair_<id:04d>_ref.png`` (the template) and ``pair_<id:04d>_search.png``, for the tiers
clean, nominal, hard and ambiguous (40 pairs each; ``eval/gate_g2.py`` reads them the
same way). ``gt_x``/``gt_y`` in the manifest are the template *centre* in search
coordinates.

The sealed split (``outputs/dataset_test/``) is never read. ``load_dev_pairs`` refuses
any root whose path mentions it.

Pairs are taken round-robin over the tiers (clean 0, nominal 0, hard 0, ambiguous 0,
clean 1, ...), so a short ``--quick`` run still covers every tier, including the rotated
ones that exercise the de-rotation warp.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from bench.cool.common import REPO_ROOT

DEV_ROOT = REPO_ROOT / "outputs" / "dataset"
TIERS = ("clean", "nominal", "hard", "ambiguous")
SEALED_MARKERS = ("dataset_test", "sealed")
# Hit radius on the template centre, copied from eval/gate_g2.py (THR_PX = 3.0).
THR_PX = 3.0
# Pair the microbench captures its real op arguments from: hard/0000 is rotated by 2.66 deg,
# so both the search and the template exceed the 0.75 deg de-rotation threshold in
# driftsense/v2/engine.py and the capture includes the full-frame S2 warpPerspective.
CAPTURE_PAIR = ("hard", 0)


class SealedDataError(RuntimeError):
    """Raised when a caller points the benchmark at the sealed split."""


@dataclass(frozen=True)
class DevPair:
    tier: str
    pair_id: int
    template_png: bytes
    search_png: bytes
    gt_x: float  # template centre, search coordinates
    gt_y: float
    ref_w: int
    ref_h: int
    rotation_deg: float

    @property
    def key(self) -> str:
        return f"{self.tier}/{self.pair_id:04d}"

    def hit(self, x_px: float, y_px: float, thr: float = THR_PX) -> bool:
        """True when a top-left offset puts the template centre within ``thr`` px of the truth."""
        return math.hypot(x_px + self.ref_w / 2.0 - self.gt_x, y_px + self.ref_h / 2.0 - self.gt_y) <= thr


def _guard(root: Path) -> Path:
    p = Path(root).resolve()
    low = p.as_posix().lower()
    if any(m in low for m in SEALED_MARKERS):
        raise SealedDataError(f"refusing to read sealed data: {p}")
    return p


def _manifest(root: Path, tier: str) -> list[dict]:
    f = _guard(root) / tier / "manifest.json"
    return json.loads(f.read_text(encoding="utf-8"))["pairs"]


def load_pair(tier: str, pair_id: int, root: Path = DEV_ROOT) -> DevPair:
    root = _guard(root)
    for p in _manifest(root, tier):
        if int(p["pair_id"]) == int(pair_id):
            return _make(root, tier, p)
    raise KeyError(f"{tier}/{pair_id} not in {root / tier / 'manifest.json'}")


def _make(root: Path, tier: str, p: dict) -> DevPair:
    d = root / tier
    pid = int(p["pair_id"])
    return DevPair(
        tier=tier,
        pair_id=pid,
        template_png=(d / f"pair_{pid:04d}_ref.png").read_bytes(),
        search_png=(d / f"pair_{pid:04d}_search.png").read_bytes(),
        gt_x=float(p["gt_x"]),
        gt_y=float(p["gt_y"]),
        ref_w=int(p["ref_w"]),
        ref_h=int(p["ref_h"]),
        rotation_deg=float(p.get("rotation_deg", 0.0)),
    )


def load_dev_pairs(k: int, root: Path = DEV_ROOT, tiers: tuple[str, ...] = TIERS) -> list[DevPair]:
    """The first ``k`` DEV pairs, round-robin over ``tiers``. Deterministic."""
    root = _guard(root)
    manifests = {t: _manifest(root, t) for t in tiers}
    out: list[DevPair] = []
    i = 0
    while len(out) < k:
        progressed = False
        for t in tiers:
            if i < len(manifests[t]) and len(out) < k:
                out.append(_make(root, t, manifests[t][i]))
                progressed = True
        if not progressed:
            break
        i += 1
    if len(out) < k:
        raise ValueError(f"only {len(out)} DEV pairs available under {root}, asked for {k}")
    return out
