"""The microbench op table: every OpenCV call the v2 engine makes, with its group.

Each ``OpSpec`` names one call site in ``driftsense/v2`` as ``(file, enclosing function,
cv2 function)``. A site key survives line-number churn in the engine. The README table
cites file:line as of the engine commit the table was written against. The microbench
does not invent arguments: it replays the arguments the engine actually passed at that
site (captured by ``bench.cool.census`` while ``driftsense.api.localize(engine="v2")``
runs on a DEV pair), so dtype, image size, kernel size, border mode and iteration count
are the production ones by construction.

Groups:

* ``kleidicv``: the cv2 function is in the KleidiCV HAL list (``KLEIDICV_LISTED``).
  COOL can only help here, and only where KleidiCV supports the dtype/parameters
  (OpenCV falls back to its generic path otherwise, silently). The ``note`` says what
  is at risk of falling back (float32 inputs, non-default borders, large kernels).
* ``control``: not KleidiCV-listed (matchTemplate, dft, phaseCorrelate, findTransformECC,
  imencode, ...). COOL is not expected to speed these up. A "gain" here is noise or a
  setup difference until explained.
* ``numpy``: pure-NumPy code that really runs in the pipeline (the v1 fingerprint
  re-rank of S6 and the S3 element-wise residual maths). Negative control: expected
  speed-up about 1.0 between pip and COOL on the same instance type.
* ``counterfactual``: NOT called by the engine. The same op and kernel as an engine call,
  but on uint8/int16 data, where KleidiCV coverage is broad. Reported separately (never
  folded into pipeline numbers) to show what the float32 design choice leaves on the
  table.

This module is import-light (stdlib only) so tests can read the table without cv2.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_ROOT / "driftsense" / "v2"

# Arm's KleidiCV HAL list as used for the COOL claim (Python-visible names). minMaxIdx is
# reached from Python through minMaxLoc; convertTo has no Python binding.
KLEIDICV_LISTED = frozenset(
    {
        "GaussianBlur",
        "sepFilter2D",
        "Sobel",
        "medianBlur",
        "dilate",
        "erode",
        "resize",
        "pyrDown",
        "transpose",
        "rotate",
        "remap",
        "warpPerspective",
        "cvtColor",
        "convertTo",
        "minMaxIdx",
        "minMaxLoc",
        "absdiff",
        "add",
        "subtract",
        "multiply",
        "calcOpticalFlowPyrLK",
    }
)

KLEIDICV = "kleidicv"
CONTROL = "control"
NUMPY = "numpy"
COUNTERFACTUAL = "counterfactual"
GROUPS = (KLEIDICV, CONTROL, NUMPY, COUNTERFACTUAL)
GROUP_DOC = {
    KLEIDICV: "KleidiCV-listed; COOL may accelerate it where the dtype/parameters are supported",
    CONTROL: "not KleidiCV-listed; COOL is not expected to speed it up",
    NUMPY: "pure NumPy that runs in the pipeline; negative control, expected speed-up ~1.0",
    COUNTERFACTUAL: "NOT an engine call: same op/kernel on uint8/int16 data; shows the float32 cost",
}

AUTO_ONLY = ("auto", "accurate")  # S6 re-rank / S9 explain only run in these modes
ALL_MODES = ("fast", "auto", "accurate", "dense")


def group_of(fn: str) -> str:
    return KLEIDICV if fn in KLEIDICV_LISTED else CONTROL


@dataclass(frozen=True)
class OpSpec:
    fn: str  # cv2 function name (or a numpy/counterfactual label)
    variant: str
    stage: str  # S0..S9
    file: str  # repo-relative engine file of the call site
    func: str  # enclosing Python function at the call site
    group: str
    note: str = ""
    modes: tuple[str, ...] = ALL_MODES
    calls: str = ""  # human note on how often the site runs per localize

    @property
    def key(self) -> str:
        return f"op.{self.fn}.{self.variant}"

    @property
    def site(self) -> tuple[str, str, str]:
        return (self.file, self.func, self.fn)

    @property
    def is_engine_call(self) -> bool:
        return self.group in (KLEIDICV, CONTROL)


_I = "driftsense/v2/ingest.py"
_L = "driftsense/v2/lattice.py"
_R = "driftsense/v2/rectify.py"
_E = "driftsense/v2/engine.py"
_G = "driftsense/v2/roughness.py"
_S = "driftsense/v2/score.py"
_P = "driftsense/v2/subpixel.py"
_X = "driftsense/v2/explain.py"
_C = "driftsense/v2/cascade.py"

F32 = "float32 input: KleidiCV coverage unverified (generic OpenCV path if unsupported)"


def _k(fn: str) -> str:
    return group_of(fn)


ENGINE_OPS: tuple[OpSpec, ...] = (
    # ---- S0 ingest + QC (template and search) ----
    OpSpec("imdecode", "s0_png_unchanged", "S0", _I, "ingest", _k("imdecode"), calls="2 (template, search)"),
    OpSpec("meanStdDev", "s0_qc_u8", "S0", _I, "_qc", _k("meanStdDev"), calls="2"),
    OpSpec(
        "minMaxLoc",
        "s0_qc_u8",
        "S0",
        _I,
        "_qc",
        _k("minMaxLoc"),
        "Python minMaxLoc wraps minMaxIdx",
        calls="2",
    ),
    # ---- S1 lattice (1024x1024 padded spectrum; detect_lattice runs ~6-7x per localize) ----
    OpSpec(
        "createHanningWindow",
        "s1_window_f32",
        "S1",
        _L,
        "_power_spectrum",
        _k("createHanningWindow"),
        calls="1 per detect_lattice",
    ),
    OpSpec("dft", "s1_complex_f32", "S1", _L, "_power_spectrum", _k("dft"), calls="1 per detect_lattice"),
    OpSpec(
        "magnitude",
        "s1_spectrum_f32",
        "S1",
        _L,
        "_power_spectrum",
        _k("magnitude"),
        calls="1 per detect_lattice",
    ),
    OpSpec("log", "s1_spectrum_f32", "S1", _L, "_power_spectrum", _k("log"), calls="1 per detect_lattice"),
    OpSpec(
        "minMaxLoc",
        "s1_peak_f32",
        "S1",
        _L,
        "strongest",
        _k("minMaxLoc"),
        f"{F32}; Python minMaxLoc wraps minMaxIdx",
        calls="2-33 per detect_lattice (peak search)",
    ),
    # ---- S2 rectify + de-rotation ----
    OpSpec("pyrDown", "s2_coarse_u8", "S2", _R, "rectify", _k("pyrDown"), calls="1"),
    OpSpec(
        "warpPerspective",
        "s2_hypothesis_u8",
        "S2",
        _R,
        "rectify",
        _k("warpPerspective"),
        "half-res coarse frame, INTER_LINEAR, BORDER_CONSTANT",
        calls="up to 4 (hypotheses)",
    ),
    OpSpec(
        "warpPerspective",
        "s2_derotate_u8",
        "S2",
        _E,
        "_warp",
        _k("warpPerspective"),
        "BORDER_REFLECT101: KleidiCV border coverage unverified",
        calls="0-2 (only when the axis residual >= 0.75 deg)",
    ),
    # ---- S3 roughness residual (float32 frames; template and search) ----
    OpSpec(
        "sepFilter2D",
        "s3_along_f32",
        "S3",
        _G,
        "residual",
        _k("sepFilter2D"),
        f"{F32}; 1-tap x k along-line Gaussian, k = 2 lattice periods snapped to an odd size",
        calls="2",
    ),
    OpSpec(
        "Sobel",
        "s3_grad_f32",
        "S3",
        _G,
        "residual",
        _k("Sobel"),
        f"{F32}; ksize 3, dx and dy",
        calls="4 (dx, dy) x 2 frames",
    ),
    OpSpec("magnitude", "s3_grad_f32", "S3", _G, "residual", _k("magnitude"), calls="2"),
    OpSpec("erode", "s3_band_u8", "S3", _G, "residual", _k("erode"), "3x3 ellipse", calls="2"),
    OpSpec(
        "dilate", "s3_band_u8", "S3", _G, "residual", _k("dilate"), "3x3 ellipse, iterations=2", calls="2"
    ),
    OpSpec(
        "GaussianBlur",
        "s3_localnorm_f32",
        "S3",
        _G,
        "residual",
        _k("GaussianBlur"),
        f"{F32}; 25x25, sigma 8 (norm_tile 24 snapped)",
        calls="4 (mean, energy) x 2 frames",
    ),
    # ---- S4 score ----
    OpSpec(
        "matchTemplate",
        "s4_ccoeff_normed_f32",
        "S4",
        _S,
        "score_residual",
        _k("matchTemplate"),
        "residual-on-residual TM_CCOEFF_NORMED",
        calls="1",
    ),
    OpSpec(
        "minMaxLoc",
        "s4_anchor_f32",
        "S4",
        _S,
        "score_residual",
        _k("minMaxLoc"),
        f"{F32}; Python minMaxLoc wraps minMaxIdx",
        calls="1",
    ),
    OpSpec("dilate", "s4_nms_f32", "S4", _S, "_nms_peaks", _k("dilate"), f"{F32}; 5x5 rect NMS", calls="1"),
    # ---- S8 subpixel (template-sized patch) ----
    OpSpec("createHanningWindow", "s8_window_f32", "S8", _P, "refine", _k("createHanningWindow"), calls="1"),
    OpSpec("phaseCorrelate", "s8_hann_f32", "S8", _P, "refine", _k("phaseCorrelate"), calls="1"),
    OpSpec(
        "findTransformECC",
        "s8_translation_f32",
        "S8",
        _P,
        "refine",
        _k("findTransformECC"),
        "MOTION_TRANSLATION, 100 iters, eps 1e-5, gaussFiltSize 5",
        calls="1",
    ),
    # ---- S9 explain (auto/accurate only) ----
    OpSpec(
        "convertScaleAbs",
        "s9_norm_u8",
        "S9",
        _X,
        "_norm_u8",
        _k("convertScaleAbs"),
        "not listed (KleidiCV lists convertTo, not convertScaleAbs)",
        AUTO_ONLY,
        calls="2",
    ),
    OpSpec(
        "applyColorMap", "s9_heatmap", "S9", _X, "heatmap", _k("applyColorMap"), modes=AUTO_ONLY, calls="2"
    ),
    OpSpec(
        "imencode",
        "s9_png",
        "S9",
        _X,
        "_png",
        _k("imencode"),
        "largest of the 4 PNG encodes is timed",
        AUTO_ONLY,
        calls="4",
    ),
    OpSpec(
        "resize",
        "s9_gallery_nearest_u8",
        "S9",
        _X,
        "gallery",
        _k("resize"),
        "INTER_NEAREST, template size",
        AUTO_ONLY,
        calls="<=5",
    ),
    OpSpec(
        "cvtColor",
        "s9_gallery_gray2bgr_u8",
        "S9",
        _X,
        "gallery",
        _k("cvtColor"),
        "template-sized crop",
        AUTO_ONLY,
        calls="<=5",
    ),
    OpSpec("putText", "s9_gallery", "S9", _X, "gallery", _k("putText"), modes=AUTO_ONLY, calls="<=5"),
    OpSpec(
        "copyMakeBorder",
        "s9_gallery",
        "S9",
        _X,
        "gallery",
        _k("copyMakeBorder"),
        modes=AUTO_ONLY,
        calls="<=5",
    ),
    OpSpec("hconcat", "s9_gallery", "S9", _X, "gallery", _k("hconcat"), modes=AUTO_ONLY, calls="1"),
    OpSpec(
        "cvtColor",
        "s9_overlay_gray2bgr_u8",
        "S9",
        _X,
        "overlay",
        _k("cvtColor"),
        "full search frame",
        AUTO_ONLY,
        calls="1",
    ),
    OpSpec(
        "addWeighted", "s9_overlay_u8", "S9", _X, "overlay", _k("addWeighted"), modes=AUTO_ONLY, calls="1"
    ),
    OpSpec("rectangle", "s9_overlay", "S9", _X, "overlay", _k("rectangle"), modes=AUTO_ONLY, calls="1"),
    OpSpec("drawMarker", "s9_overlay", "S9", _X, "overlay", _k("drawMarker"), modes=AUTO_ONLY, calls="1"),
    OpSpec("putText", "s9_overlay", "S9", _X, "overlay", _k("putText"), modes=AUTO_ONLY, calls="1"),
)

PIPELINE_EXTRA_OPS: tuple[OpSpec, ...] = (
    OpSpec(
        "numpy",
        "s6_v1_fingerprint_rerank",
        "S6",
        _C,
        "_v1_rerank",
        NUMPY,
        "localization.ler_fingerprint over the 64-candidate pool (cascade.rerank mode='v1'); runs in "
        "auto/accurate when the margin escalates",
        AUTO_ONLY,
        calls="0-1",
    ),
    OpSpec(
        "numpy",
        "s3_residual_elementwise",
        "S3",
        _G,
        "residual",
        NUMPY,
        "I - smooth, r * mask, r * r, local std and the final normalisation on the search frame",
        calls="1",
    ),
)

COUNTERFACTUAL_OPS: tuple[OpSpec, ...] = (
    OpSpec(
        "GaussianBlur",
        "cf_localnorm_u8",
        "S3",
        _G,
        "residual",
        COUNTERFACTUAL,
        "same 25x25 / sigma as s3_localnorm_f32 on the uint8 search frame",
    ),
    OpSpec(
        "sepFilter2D",
        "cf_along_u8",
        "S3",
        _G,
        "residual",
        COUNTERFACTUAL,
        "same 1 x k kernel as s3_along_f32 on the uint8 search frame (uint8 output)",
    ),
    OpSpec(
        "Sobel",
        "cf_grad_u8_s16",
        "S3",
        _G,
        "residual",
        COUNTERFACTUAL,
        "same ksize 3 dx as s3_grad_f32, uint8 -> int16",
    ),
)

ALL_OPS: tuple[OpSpec, ...] = ENGINE_OPS + PIPELINE_EXTRA_OPS + COUNTERFACTUAL_OPS

# Engine cv2 call sites deliberately not timed, with the reason.
NOT_BENCHMARKED: dict[tuple[str, str, str], str] = {
    (_I, "_to_gray", "cvtColor"): "only for 3-channel input; every DEV/agent frame is single-channel",
    (_L, "_power_spectrum", "getOptimalDFTSize"): "scalar helper (returns 1024 here)",
    (_R, "_homography", "getRotationMatrix2D"): "scalar 2x3 matrix",
    (_E, "_rotation", "getRotationMatrix2D"): "scalar 2x3 matrix",
    (_G, "residual", "getGaussianKernel"): "builds a 1-D kernel (k taps)",
    (_G, "residual", "getStructuringElement"): "builds a 3x3 element",
    (_S, "_nms_peaks", "getStructuringElement"): "builds a 5x5 element",
}

# KleidiCV-listed functions the v2 hot path never calls (so COOL cannot touch them here).
_USED = {o.fn for o in ENGINE_OPS} | (
    {"minMaxIdx"} if any(o.fn == "minMaxLoc" for o in ENGINE_OPS) else set()
)
LISTED_BUT_UNUSED = tuple(sorted(KLEIDICV_LISTED - _USED))


def assert_unique_keys() -> None:
    keys = [o.key for o in ALL_OPS]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise AssertionError(f"duplicate op keys: {sorted(dup)}")


assert_unique_keys()


# ------------------------------------------------------------------ static engine scan
@dataclass(frozen=True)
class CallSite:
    file: str
    line: int
    func: str
    fn: str

    @property
    def site(self) -> tuple[str, str, str]:
        return (self.file, self.func, self.fn)


def _walk(node: ast.AST, func: str, rel: str, out: list[CallSite]) -> None:
    for ch in ast.iter_child_nodes(node):
        if isinstance(ch, ast.FunctionDef | ast.AsyncFunctionDef):
            _walk(ch, ch.name, rel, out)
            continue
        if (
            isinstance(ch, ast.Call)
            and isinstance(ch.func, ast.Attribute)
            and isinstance(ch.func.value, ast.Name)
            and ch.func.value.id == "cv2"
        ):
            out.append(CallSite(rel, ch.lineno, func, ch.func.attr))
        _walk(ch, func, rel, out)


@lru_cache(maxsize=1)
def engine_cv2_calls(engine_dir: Path = ENGINE_DIR) -> tuple[CallSite, ...]:
    """Every ``cv2.<fn>(...)`` call in the v2 engine package, from the AST (no import)."""
    out: list[CallSite] = []
    for f in sorted(Path(engine_dir).glob("*.py")):
        rel = f.resolve().relative_to(REPO_ROOT).as_posix()
        _walk(ast.parse(f.read_text(encoding="utf-8")), "<module>", rel, out)
    return tuple(out)


def engine_cv2_functions() -> tuple[str, ...]:
    return tuple(sorted({c.fn for c in engine_cv2_calls()}))


def lines_for(spec: OpSpec) -> list[int]:
    return sorted(c.line for c in engine_cv2_calls() if c.site == spec.site)
