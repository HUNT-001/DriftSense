from __future__ import annotations

import time

import cv2
import numpy as np

from driftsense.contracts import (
    Budget,
    Candidate,
    DriftSenseInputError,
    LocalizeResult,
    Margin,
    Mode,
    Offset,
    StageTiming,
)


def _decode(b: bytes, name: str) -> np.ndarray:
    if not b:
        raise DriftSenseInputError(f"{name}: empty")
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise DriftSenseInputError(f"{name}: not a decodable image")
    return img


def localize(template_bytes: bytes, search_bytes: bytes, mode: Mode, budget: Budget) -> LocalizeResult:
    t0 = time.perf_counter()
    T, S = _decode(template_bytes, "template"), _decode(search_bytes, "search")
    if T.shape[0] > S.shape[0] or T.shape[1] > S.shape[1]:
        raise DriftSenseInputError("template larger than search")
    t1 = time.perf_counter()
    R = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
    h, w = T.shape
    cands = []
    for k in range(min(budget.max_candidates, 5)):
        _, mx, _, (x, y) = cv2.minMaxLoc(R)
        cands.append(Candidate(rank=k + 1, x_px=float(x), y_px=float(y), score=float(mx)))
        R[max(0, y - h // 2) : y + h // 2 + 1, max(0, x - w // 2) : x + w // 2 + 1] = -1.0
    t2 = time.perf_counter()
    m = cands[0].score - (cands[1].score if len(cands) > 1 else -1.0)
    tau = budget.tau if budget.tau is not None else 0.10
    return LocalizeResult(
        engine="fake",
        engine_version="fake-1",
        mode=mode,
        offset=Offset(x_px=cands[0].x_px, y_px=cands[0].y_px),
        candidates=cands,
        margin=Margin(margin=m, tau=tau, calibrated=False, decision="commit" if m >= tau else "escalate"),
        timings=[
            StageTiming(stage="ingest", ms=(t1 - t0) * 1e3),
            StageTiming(stage="score", ms=(t2 - t1) * 1e3),
        ],
        total_ms=(t2 - t0) * 1e3,
        warnings=["fake engine: not for reported numbers"],
    )
