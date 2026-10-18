"""v2 engine. P2 lands the pipeline stage by stage (prompts A2–A4).

S0–S2 (ingest, lattice, rectify) are implemented here as `run_s0_s2`. The public
`localize` entry point still raises NotImplementedError for the scoring stages
(S3+), except that under DRIFTSENSE_V2_PARTIAL=1 (tests only) it first runs
S0–S2 end to end, then raises for the unimplemented remainder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from driftsense.contracts import Budget, DriftSenseInputError, LocalizeResult, Mode
from driftsense.v2.ingest import IngestResult, ingest
from driftsense.v2.lattice import Lattice, detect_lattice
from driftsense.v2.rectify import Rectified, rectify


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


def localize(template_bytes: bytes, search_bytes: bytes, mode: Mode, budget: Budget) -> LocalizeResult:
    if os.environ.get("DRIFTSENSE_V2_PARTIAL") == "1":
        run_s0_s2(template_bytes, search_bytes)  # exercise S0–S2; raise for the rest
        raise NotImplementedError("v2 S0–S2 ran; scoring stages S3+ land in P2 (A3–A4)")
    raise NotImplementedError("v2 lands in P2")
