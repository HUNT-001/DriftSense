"""v2 engine — the full S0->S9 entry point.

localize() assembles the pipeline into one contract LocalizeResult:

    S0 ingest -> S1 lattice -> S2 rectify -> (de-rotate) -> S3 roughness ->
    S4 lattice-consistent candidate enumeration -> score -> S6 cascade ->
    S8 subpixel -> S9 explain.

Two corrections over the stage-by-stage prototypes:
  * Rectification is wired in: S3/S4 run on the de-rotated frame (the dominant
    lattice axis brought onto an image axis), and the winning offset is mapped
    back to native search coordinates. Running un-rectified was a correctness gap
    on the rotated tiers (nominal/hard).
  * S4 candidate generation enumerates all lattice-consistent placements (the v1
    spectral "lattice" path, imported read-only) instead of scoring only around
    the single global residual peak. The lattice set keeps the truth ~100% of the
    time; the single-peak path reached ~0.46 recall@20. This is the G2 blocker.

run_s0_s2 is retained for the S0-S2 tests. Hot-path discipline: this module calls
no template-matching or affine-warp cv2 primitives directly and runs no per-pixel
loops; the heavy ops live in the stage modules (the match in score, the
phase-correlate / ECC in subpixel). Rotation uses warpPerspective (3x3).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import cv2
import numpy as np

from driftsense.contracts import (
    Artifact,
    Budget,
    Candidate,
    DriftSenseInputError,
    LatticeType,
    LocalizeResult,
    Margin,
    Mode,
    Offset,
    StageTiming,
)
from driftsense.v2.cascade import CascadeConfig, rerank
from driftsense.v2.explain import build_all
from driftsense.v2.ingest import IngestResult, ingest
from driftsense.v2.lattice import Lattice, detect_lattice
from driftsense.v2.rectify import Rectified, rectify
from driftsense.v2.roughness import Residual, residual
from driftsense.v2.score import score_residual
from driftsense.v2.subpixel import refine

ENGINE_VERSION = "v2-p2"
_DEFAULT_TAU = 0.10
_DEROTATE_MIN_DEG = 0.75  # below this the frame is treated as already axis-aligned
_CASCADE_POOL = 64  # residual-ranked pool the cascade fingerprint re-ranks (> max_candidates
#                     so the re-rank can lift the truth into the top-K, raising recall@K)


@dataclass
class S0S2Result:
    template: IngestResult
    search: IngestResult
    lattice: Lattice
    rectified: list[Rectified]


def run_s0_s2(template_bytes: bytes, search_bytes: bytes) -> S0S2Result:
    """Run ingest -> lattice -> rectify on the search frame (QC both frames)."""
    tmpl = ingest(template_bytes)
    srch = ingest(search_bytes)
    if tmpl.image.shape[0] > srch.image.shape[0] or tmpl.image.shape[1] > srch.image.shape[1]:
        raise DriftSenseInputError("template larger than search")
    lattice = detect_lattice(srch.image)
    rectified = rectify(srch.image, lattice)
    return S0S2Result(template=tmpl, search=srch, lattice=lattice, rectified=rectified)


class _Clock:
    """Accumulate per-stage StageTiming entries."""

    def __init__(self) -> None:
        self.timings: list[StageTiming] = []

    def mark(self, stage: str, t0: float) -> None:
        self.timings.append(StageTiming(stage=stage, ms=(time.perf_counter() - t0) * 1e3))


def _axis_residual_deg(deg: float) -> float:
    a = (deg + 45.0) % 90.0 - 45.0
    return 45.0 if a == -45.0 else a


def _rotation(shape: tuple[int, int], angle_deg: float) -> np.ndarray:
    """3x3 rotation homography about the image centre (embeds a 2x3 affine)."""
    h, w = shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = m
    return H


def _warp(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def _map_point(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = H @ np.array([x, y, 1.0], dtype=np.float64)
    return float(v[0] / v[2]), float(v[1] / v[2])


def _lattice_type(lattice: Lattice) -> LatticeType:
    p1 = 1.0 / float(np.linalg.norm(lattice.k1) + 1e-12)
    p2 = 1.0 / float(np.linalg.norm(lattice.k2) + 1e-12)
    if lattice.confidence < 0.2:
        return LatticeType.UNKNOWN
    return LatticeType.GRID if min(p1, p2) / max(p1, p2) > 0.8 else LatticeType.FINFET


def _enumerate_lattice_candidates(
    search_r: np.ndarray, template_r: np.ndarray, score_map: np.ndarray, top_k: int
) -> list[Candidate]:
    """Lattice-consistent placements (v1 spectral path), ranked by residual score.

    Returns [] if the spectral basis fails, so the caller can fall back to the
    single-peak candidates. Coordinates are rectified-frame top-left offsets.
    """
    from localization.spectral import lattice_candidates, snap_to_local_ncc  # read-only (frozen v1)

    th, tw = template_r.shape[:2]
    centres, basis = lattice_candidates(search_r, template_r, max_candidates=4000)
    if not centres:
        return []
    centres = snap_to_local_ncc(centres, score_map, (th, tw), radius=5)
    h, w = score_map.shape
    scored: list[tuple[float, int, int]] = []
    for cx, cy in centres:
        tx, ty = int(round(cx - tw / 2.0)), int(round(cy - th / 2.0))
        if 0 <= tx < w and 0 <= ty < h:
            scored.append((float(score_map[ty, tx]), tx, ty))
    if not scored:
        return []
    scored.sort(key=lambda s: -s[0])
    seen: set[tuple[int, int]] = set()
    out: list[Candidate] = []
    for sc, tx, ty in scored:
        if (tx, ty) in seen:
            continue
        seen.add((tx, ty))
        out.append(Candidate(rank=len(out) + 1, x_px=float(tx), y_px=float(ty), score=sc, roughness_score=sc))
        if len(out) >= top_k:
            break
    return out


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def localize(template_bytes: bytes, search_bytes: bytes, mode: Mode, budget: Budget) -> LocalizeResult:
    """Full v2 localisation: S0->S9, returning a complete LocalizeResult."""
    clk = _Clock()
    t_total = time.perf_counter()
    warnings: list[str] = []

    # ---- S0 ingest ----
    t0 = time.perf_counter()
    tmpl, srch = ingest(template_bytes), ingest(search_bytes)
    if tmpl.image.shape[0] > srch.image.shape[0] or tmpl.image.shape[1] > srch.image.shape[1]:
        raise DriftSenseInputError("template larger than search")
    t_img, s_img = tmpl.image, srch.image
    th, tw = t_img.shape[:2]
    clk.mark("ingest", t0)

    # ---- S1 lattice ----
    t0 = time.perf_counter()
    lattice = detect_lattice(s_img)
    clk.mark("lattice", t0)

    # ---- S2 rectify + de-rotation wiring ----
    # The template is a crop of the un-rotated reference; the search carries the
    # relative rotation. Each frame is de-rotated by ITS OWN lattice-axis residual
    # so both land axis-aligned (and mutually aligned), then the lattice is
    # re-detected on the rectified search for S3/S4. Only the search transform
    # maps winner coordinates, so just its inverse is kept.
    t0 = time.perf_counter()
    hyps = rectify(s_img, lattice)
    phi_s = _axis_residual_deg(lattice.angle_deg)
    phi_t = _axis_residual_deg(detect_lattice(t_img).angle_deg)
    derotated = abs(phi_s) >= _DEROTATE_MIN_DEG
    Hs = _rotation(s_img.shape, phi_s)
    search_r = _warp(s_img, Hs) if derotated else s_img
    Hs_inv = np.linalg.inv(Hs) if derotated else np.eye(3)
    template_r = _warp(t_img, _rotation(t_img.shape, phi_t)) if abs(phi_t) >= _DEROTATE_MIN_DEG else t_img
    lattice_r = detect_lattice(search_r) if derotated else lattice
    if not hyps:
        warnings.append("rectify: no hypotheses; using native orientation")
    clk.mark("rectify", t0)

    # ---- S3 roughness (on the rectified frames) ----
    t0 = time.perf_counter()
    rt: Residual = residual(template_r, lattice_r)
    rs: Residual = residual(search_r, lattice_r)
    warnings.extend(rt.warnings)
    clk.mark("roughness", t0)

    # ---- S4 score + lattice-consistent candidate enumeration ----
    t0 = time.perf_counter()
    # The fast roughness-residual scan is the PRODUCTION DEFAULT: one matchTemplate
    # on residuals + lattice-consistent peaks (~150 ms/pair). The lattice-consistent
    # ENUMERATION (v1 spectral basis) has far higher candidate-set coverage but costs
    # ~2-3 s/pair -- it is reserved for the DENSE recall path, since on DEV it is a
    # ~20-35x slowdown for no gain in ranked top-1 (see eval/gate_g2.py).
    pool_size = max(budget.max_candidates, _CASCADE_POOL)
    sr = score_residual(rt, rs, lattice_r, max_candidates=pool_size)
    score_map = sr.score_map
    if mode == Mode.DENSE:
        enum = _enumerate_lattice_candidates(search_r, template_r, score_map, pool_size)
        pool = enum or sr.candidates
        if not enum:
            warnings.append("S4 DENSE: spectral basis unavailable; fell back to fast candidates")
    else:
        pool = sr.candidates
    if not pool:  # guarantee >=1 candidate from the global residual peak
        ax, ay = sr.anchor_xy
        pool = [Candidate(rank=1, x_px=float(ax), y_px=float(ay), score=float(score_map.max()))]
    dense = pool[: budget.max_candidates]
    clk.mark("score", t0)

    # ---- S6 cascade (AUTO/ACCURATE re-rank by fingerprint; FAST/DENSE stay dense) ----
    t0 = time.perf_counter()
    tau = budget.tau if budget.tau is not None else _DEFAULT_TAU
    if mode in (Mode.AUTO, Mode.ACCURATE):
        cres = rerank(pool, template_r, search_r, mode="cascade", tau=tau, cfg=CascadeConfig(top_k=len(pool)))
        cands = (cres.candidates or dense)[: budget.max_candidates]
        escalated = cres.escalated
        if escalated:
            warnings.append("cascade: escalated to v1 fingerprint re-rank")
    else:
        cands = dense
        escalated = False
    clk.mark("cascade", t0)

    # ---- S8 subpixel on the winner ----
    t0 = time.perf_counter()
    win = cands[0]
    dx = dy = 0.0
    wtx, wty = int(round(win.x_px)), int(round(win.y_px))
    patch = search_r[wty : wty + th, wtx : wtx + tw]
    if patch.shape[:2] == (th, tw):  # subpixel is ~ms; runs in every mode
        sp = refine(template_r, patch, pixel_size_nm=None)
        dx, dy = sp.dx_px, sp.dy_px
    clk.mark("subpixel", t0)

    # Winner centre in rectified coords -> map back to native search coords.
    cx_r, cy_r = win.x_px + dx + tw / 2.0, win.y_px + dy + th / 2.0
    cx, cy = _map_point(Hs_inv, cx_r, cy_r) if derotated else (cx_r, cy_r)
    off = Offset(x_px=cx - tw / 2.0, y_px=cy - th / 2.0)

    # ---- confidence / margin ----
    m = cands[0].score - (cands[1].score if len(cands) > 1 else cands[0].score - 1.0)
    margin = Margin(
        margin=float(m),
        tau=float(tau),
        calibrated=False,
        decision="commit" if m >= tau else "escalate",
    )

    # ---- S9 explain (only the accuracy-oriented modes need artifacts) ----
    t0 = time.perf_counter()
    artifacts: list[Artifact] = []
    if mode in (Mode.ACCURATE, Mode.AUTO):
        rough_disp = np.abs(rs.residual).astype(np.float32)
        for art in build_all(search_r, cands, score_map, rough_disp, tw, th):
            artifacts.append(Artifact(kind=art.kind, uri=f"{art.name}.png", sha256=_sha(art.png)))
    clk.mark("explain", t0)

    total_ms = (time.perf_counter() - t_total) * 1e3
    return LocalizeResult(
        engine="v2",
        engine_version=ENGINE_VERSION,
        mode=mode,
        lattice_type=_lattice_type(lattice),
        offset=off,
        candidates=[
            Candidate(
                rank=i + 1,
                x_px=c.x_px,
                y_px=c.y_px,
                score=c.score,
                lattice_score=c.lattice_score,
                roughness_score=c.roughness_score,
            )
            for i, c in enumerate(cands)
        ],
        margin=margin,
        timings=clk.timings,
        total_ms=total_ms,
        artifacts=artifacts,
        warnings=warnings,
    )
