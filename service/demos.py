"""Demo registry and a deterministic synthetic (template, search) generator for local/dev.

The registry is importable without numpy/cv2 (the Lambda API only needs ids/titles). The synthetic
generator imports numpy + cv2 lazily; the worker uses it only when ``demos/<id>/...`` is missing in S3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from service.models import DemoInfo

SEARCH_SIZE = 384
TEMPLATE_SIZE = 96


@dataclass(frozen=True)
class DemoSpec:
    demo_id: str
    title: str
    lattice: str  # finfet | dram | grid | staggered
    difficulty: str  # easy | medium | hard
    seed: int
    roughness: float  # px of line-edge / placement roughness (higher = more unique)
    noise: float  # gaussian noise sigma (0-255 scale)
    description: str = ""

    def info(self) -> DemoInfo:
        return DemoInfo(
            demo_id=self.demo_id,
            title=self.title,
            lattice=self.lattice,
            difficulty=self.difficulty,
            description=self.description,
        )


DEMOS: dict[str, DemoSpec] = {
    d.demo_id: d
    for d in (
        DemoSpec(
            "demo_finfet_easy",
            "FinFET fins, strong roughness",
            "finfet",
            "easy",
            11,
            1.6,
            6.0,
            "Periodic fins with pronounced line-edge roughness: the fingerprint is unique.",
        ),
        DemoSpec(
            "demo_dram_hard",
            "DRAM cell array, weak roughness",
            "dram",
            "hard",
            23,
            0.05,
            40.0,
            "Hexagonal capacitor array, almost perfectly periodic and noisy: expect escalation.",
        ),
        DemoSpec(
            "demo_grid",
            "Contact grid",
            "grid",
            "medium",
            37,
            0.8,
            10.0,
            "Square contact grid with moderate placement roughness.",
        ),
        DemoSpec(
            "demo_staggered",
            "Staggered contacts",
            "staggered",
            "medium",
            41,
            0.8,
            10.0,
            "Staggered (brick) contact lattice with moderate placement roughness.",
        ),
    )
}


# ---------- agent episodes on a VirtualSEM: demo_id = vsem-<family>-<seed> ----------
VSEM_FAMILIES = ("dram", "finfet", "random")
MAX_SEED = 2**31 - 1
_VSEM_RE = re.compile(r"^vsem-(dram|finfet|random)-(0|[1-9][0-9]{0,9})$")


def parse_vsem(demo_id: str | None) -> tuple[str, int] | None:
    """``vsem-<family>-<seed>`` -> (family, seed); None if ``demo_id`` is not an episode id."""
    m = _VSEM_RE.match(demo_id or "")
    if not m or int(m.group(2)) > MAX_SEED:
        return None
    return m.group(1), int(m.group(2))


def vsem_demo_id(family: str, seed: int) -> str:
    return f"vsem-{family}-{seed}"


# Curated episodes (any vsem-<family>-<seed> is accepted; these are the ones the UI lists). Outcomes
# were observed with the G5 operating point (tau 0.10, max_steps 6) and the rules policy.
AGENT_DEMOS: tuple[DemoInfo, ...] = (
    DemoInfo(
        demo_id="vsem-dram-7",
        title="Agent: DRAM array, confident commit",
        lattice="dram",
        difficulty="easy",
        agent=True,
        family="dram",
        seed=7,
        description="Bootstrap frame clears tau: the agent commits the corrective stage move and verifies it.",
    ),
    DemoInfo(
        demo_id="vsem-finfet-3",
        title="Agent: FinFET fins, confident commit",
        lattice="finfet",
        difficulty="easy",
        agent=True,
        family="finfet",
        seed=3,
        description="Line-edge roughness fingerprint is unique on the first look: commit + verify.",
    ),
    DemoInfo(
        demo_id="vsem-dram-1",
        title="Agent: DRAM array, re-image then commit",
        lattice="dram",
        difficulty="medium",
        agent=True,
        family="dram",
        seed=1,
        description="First frame is below tau; the margin drives an extra acquisition before the commit.",
    ),
    DemoInfo(
        demo_id="vsem-dram-17",
        title="Agent: DRAM array, escalates to operator",
        lattice="dram",
        difficulty="hard",
        agent=True,
        family="dram",
        seed=17,
        description="Periodic ambiguity the agent cannot break within budget: it asks a human to pick.",
    ),
)


def list_demos() -> list[DemoInfo]:
    return [d.info() for d in DEMOS.values()] + list(AGENT_DEMOS)


def get_demo(demo_id: str) -> DemoSpec | None:
    """Classic (template, search) demo spec; None for unknown ids and for vsem-* episode ids."""
    return DEMOS.get(demo_id)


def is_known_demo(demo_id: str | None) -> bool:
    return bool(demo_id) and (demo_id in DEMOS or parse_vsem(demo_id) is not None)


def demo_keys(demo_id: str) -> tuple[str, str]:
    return f"demos/{demo_id}/template.png", f"demos/{demo_id}/search.png"


def _render(spec: DemoSpec, size: int, rng):
    import cv2
    import numpy as np

    img = np.full((size, size), 60, np.uint8)
    if spec.lattice == "finfet":
        pitch, width = 14, 6
        xs = np.arange(size)[None, :]
        # smooth per-row edge jitter = line-edge roughness, independent per fin
        for x0 in range(0, size, pitch):
            jl = cv2.GaussianBlur(rng.normal(0, spec.roughness * 3, (size, 1)), (1, 9), 0)
            jr = cv2.GaussianBlur(rng.normal(0, spec.roughness * 3, (size, 1)), (1, 9), 0)
            mask = (xs >= np.round(x0 + jl)) & (xs < np.round(x0 + width + jr))
            img[mask] = 190
        # gate cuts at irregular rows add a coarse fingerprint
        for y0 in range(0, size, 48):
            y = int(y0 + rng.integers(0, 20))
            img[y : y + 4, :] = 40
    else:
        if spec.lattice == "dram":
            px, py, stagger, shape = 18, 16, 9, "ellipse"
        elif spec.lattice == "staggered":
            px, py, stagger, shape = 20, 18, 10, "rect"
        else:  # grid
            px, py, stagger, shape = 20, 20, 0, "rect"
        for j, y0 in enumerate(range(-py, size + py, py)):
            off = stagger if j % 2 else 0
            for x0 in range(-px, size + px, px):
                cx = x0 + off + rng.normal(0, spec.roughness)
                cy = y0 + rng.normal(0, spec.roughness)
                r = 5 + rng.normal(0, spec.roughness * 0.5)
                if shape == "ellipse":
                    cv2.ellipse(
                        img,
                        (round(cx), round(cy)),
                        (max(2, round(r + 1)), max(2, round(r))),
                        0,
                        0,
                        360,
                        200,
                        -1,
                        cv2.LINE_AA,
                    )
                else:
                    s = max(2, round(r))
                    cv2.rectangle(
                        img,
                        (round(cx - s), round(cy - s)),
                        (round(cx + s), round(cy + s)),
                        200,
                        -1,
                        cv2.LINE_AA,
                    )
    return cv2.GaussianBlur(img, (3, 3), 0).astype(np.float32)


def _acquire(clean, noise: float, rng):
    """Simulated acquisition: independent shot noise per image."""
    import numpy as np

    noisy = clean + rng.normal(0, noise, clean.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def synthetic_truth(demo_id: str) -> tuple[int, int]:
    """Top-left (x, y) of the template inside the search image for a synthetic demo."""
    import numpy as np

    spec = DEMOS[demo_id]
    rng = np.random.default_rng(spec.seed + 1000)
    hi = SEARCH_SIZE - TEMPLATE_SIZE
    return int(rng.integers(8, hi - 8)), int(rng.integers(8, hi - 8))


def synthetic_pair(demo_id: str) -> tuple[bytes, bytes]:
    """Deterministic PNG-encoded (template, search) pair for ``demo_id``."""
    import cv2
    import numpy as np

    spec = DEMOS[demo_id]
    rng = np.random.default_rng(spec.seed)
    clean = _render(spec, SEARCH_SIZE, rng)
    x, y = synthetic_truth(demo_id)
    # the template is the same field re-imaged: same structure, independent noise
    template = _acquire(clean[y : y + TEMPLATE_SIZE, x : x + TEMPLATE_SIZE], spec.noise, rng)
    search = _acquire(clean, spec.noise, rng)
    ok_t, t_png = cv2.imencode(".png", template)
    ok_s, s_png = cv2.imencode(".png", search)
    if not (ok_t and ok_s):  # pragma: no cover
        raise RuntimeError("PNG encode failed")
    return t_png.tobytes(), s_png.tobytes()
