"""Adapter: wrap the frozen v1 localizer behind the public contract.

v1 entry point (do NOT modify v1):
    localization.ler_localizer.localize(ref, search, ...) -> LocalizationResult
    with pred_x/pred_y = CENTRE coords and candidates ranked best-first.
The contract Offset is the template TOP-LEFT, so we convert.
"""
from __future__ import annotations

import time
from math import isfinite

import cv2
import numpy as np

from driftsense.contracts import (Budget, Candidate, DriftSenseInputError,
                                   LocalizeResult, Margin, Mode, Offset, StageTiming)

V1_VERSION = "v1-munichtech"


def _decode(b: bytes, name: str) -> np.ndarray:
    if not b:
        raise DriftSenseInputError(f"{name}: empty")
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise DriftSenseInputError(f"{name}: not a decodable image")
    return img


def _score(c) -> float:
    for attr in ("combined", "fp", "ncc"):
        v = getattr(c, attr, float("nan"))
        if v is not None and isfinite(float(v)):
            return float(v)
    return 0.0


def localize(template_bytes: bytes, search_bytes: bytes, mode: Mode, budget: Budget) -> LocalizeResult:
    from localization.ler_localizer import localize as v1_localize  # lazy: keeps import graph clean

    t0 = time.perf_counter()
    T = _decode(template_bytes, "template")
    S = _decode(search_bytes, "search")
    if T.shape[0] > S.shape[0] or T.shape[1] > S.shape[1]:
        raise DriftSenseInputError("template larger than search")
    rh, rw = T.shape[:2]

    res = v1_localize(T, S, top_k=max(budget.max_candidates, 40), return_candidates=True)
    total_ms = float(getattr(res, "runtime_ms", (time.perf_counter() - t0) * 1e3))

    raw = list(getattr(res, "candidates", []) or [])
    raw = raw[: budget.max_candidates]
    cands: list[Candidate] = []
    for i, c in enumerate(raw):
        cx, cy = float(c.x), float(c.y)  # v1 centre coords
        cands.append(Candidate(
            rank=i + 1,
            x_px=cx - rw / 2.0,
            y_px=cy - rh / 2.0,
            score=_score(c),
            lattice_score=(float(c.ncc) if isfinite(float(getattr(c, "ncc", float("nan")))) else None),
            roughness_score=(float(c.fp) if isfinite(float(getattr(c, "fp", float("nan")))) else None),
        ))

    if not cands:  # v1 returned nothing usable (e.g. empty score map)
        cands = [Candidate(rank=1,
                           x_px=float(res.pred_x) - rw / 2.0,
                           y_px=float(res.pred_y) - rh / 2.0,
                           score=float(getattr(res, "score", 0.0)) if isfinite(float(getattr(res, "score", float("nan")))) else 0.0)]

    margin = cands[0].score - (cands[1].score if len(cands) > 1 else cands[0].score - 1.0)
    tau = budget.tau if budget.tau is not None else 0.10
    return LocalizeResult(
        engine="v1",
        engine_version=V1_VERSION,
        mode=mode,
        offset=Offset(x_px=cands[0].x_px, y_px=cands[0].y_px),
        candidates=cands,
        margin=Margin(margin=float(margin), tau=float(tau), calibrated=False,
                      decision="commit" if margin >= tau else "escalate"),
        timings=[StageTiming(stage="v1_total", ms=total_ms)],
        total_ms=total_ms,
        warnings=["v1 adapter: margin/tau uncalibrated; offset converted centre->top-left"],
    )
