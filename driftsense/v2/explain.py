"""S9 — explainability artifacts.

Pure functions that turn a localisation into human-reviewable PNGs with alt text:
    - an NCC heatmap and a roughness-residual heatmap (applyColorMap),
    - a top-5 candidate gallery (ranked crops with score captions),
    - an overlay of the winning box on the search frame (addWeighted + drawMarker
      + putText).

Each builder returns (png_bytes, alt_text). Rendering/encoding ops are
KleidiCV-listed or plain drawing (resize, cvtColor, applyColorMap, addWeighted,
drawMarker, rectangle, putText, imencode); no matchTemplate and no per-pixel loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from driftsense.contracts import Candidate

_HEAT = cv2.COLORMAP_INFERNO


@dataclass
class ExplainArtifact:
    kind: str  # "heatmap" | "overlay" | "candidates"
    name: str  # short label, e.g. "ncc_heatmap"
    png: bytes  # PNG-encoded image
    alt: str  # alt text describing the artifact


def _norm_u8(a: np.ndarray) -> np.ndarray:
    """Scale an arbitrary float map to 0..255 uint8 for display."""
    a = np.asarray(a, dtype=np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros(a.shape, dtype=np.uint8)
    return cv2.convertScaleAbs(a, alpha=255.0 / (hi - lo), beta=-255.0 * lo / (hi - lo))


def _gray_u8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype == np.uint16:
        a = (a.astype(np.float32) / 257.0).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = _norm_u8(a)
    return a


def _png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("explain: PNG encode failed")
    return buf.tobytes()


def heatmap(score_map: np.ndarray, name: str, what: str) -> ExplainArtifact:
    """Colour-mapped heatmap of a score map (higher = hotter)."""
    colour = cv2.applyColorMap(_norm_u8(score_map), _HEAT)
    m = np.asarray(score_map, dtype=np.float32)
    h, w = m.shape[:2]
    py, px = (int(v) for v in np.unravel_index(int(np.nanargmax(m)), m.shape))
    alt = f"{what} heatmap, {w}x{h}px; peak {float(np.nanmax(m)):.3f} at (x={px}, y={py})."
    return ExplainArtifact(kind="heatmap", name=name, png=_png(colour), alt=alt)


def gallery(
    search: np.ndarray, candidates: list[Candidate], ref_w: int, ref_h: int, top: int = 5
) -> ExplainArtifact:
    """Horizontal strip of the top-N candidate crops, captioned with rank/score."""
    base = _gray_u8(search)
    tiles: list[np.ndarray] = []
    pad = 4
    used = candidates[:top]
    for c in used:
        x0 = int(round(c.x_px))
        y0 = int(round(c.y_px))
        crop = base[max(0, y0) : y0 + ref_h, max(0, x0) : x0 + ref_w]
        if crop.size == 0:
            crop = np.zeros((ref_h, ref_w), dtype=np.uint8)
        crop = cv2.resize(crop, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            tile,
            f"#{c.rank} {c.score:.2f}",
            (3, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        tiles.append(cv2.copyMakeBorder(tile, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(30, 30, 30)))
    if not tiles:
        tiles = [np.zeros((ref_h, ref_w, 3), dtype=np.uint8)]
    strip = cv2.hconcat(tiles)
    alt = f"Top-{len(used)} candidate gallery; ranks 1..{len(used)} with fused scores."
    return ExplainArtifact(kind="candidates", name="top5_gallery", png=_png(strip), alt=alt)


def overlay(search: np.ndarray, winner: Candidate, ref_w: int, ref_h: int) -> ExplainArtifact:
    """Winning box + centre marker over a dimmed copy of the search frame."""
    base = cv2.cvtColor(_gray_u8(search), cv2.COLOR_GRAY2BGR)
    dim = cv2.addWeighted(base, 0.6, np.zeros_like(base), 0.4, 0.0)
    x0, y0 = int(round(winner.x_px)), int(round(winner.y_px))
    cx, cy = x0 + ref_w // 2, y0 + ref_h // 2
    cv2.rectangle(dim, (x0, y0), (x0 + ref_w, y0 + ref_h), (0, 255, 0), 2)
    cv2.drawMarker(dim, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.putText(
        dim,
        f"score {winner.score:.3f}",
        (x0, max(14, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    alt = f"Localisation overlay: winning box top-left ({x0}, {y0}), size {ref_w}x{ref_h}, score {winner.score:.3f}."
    return ExplainArtifact(kind="overlay", name="overlay", png=_png(dim), alt=alt)


def build_all(
    search: np.ndarray,
    candidates: list[Candidate],
    ncc_map: np.ndarray,
    roughness_map: np.ndarray,
    ref_w: int,
    ref_h: int,
) -> list[ExplainArtifact]:
    """Assemble the full S9 artifact set for one localisation."""
    arts = [
        heatmap(ncc_map, "ncc_heatmap", "NCC"),
        heatmap(roughness_map, "roughness_heatmap", "roughness-residual"),
        gallery(search, candidates, ref_w, ref_h),
    ]
    if candidates:
        arts.append(overlay(search, candidates[0], ref_w, ref_h))
    return arts
