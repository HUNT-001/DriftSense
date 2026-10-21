"""The microbench op table must match the real v2 engine (static AST scan; no cv2 needed)."""

import re
from pathlib import Path

from bench.cool import ops as O

ROOT = Path(__file__).resolve().parents[2]


def test_every_engine_op_is_a_real_cv2_call_in_the_named_function():
    sites = {c.site for c in O.engine_cv2_calls()}
    for o in O.ENGINE_OPS:
        assert o.site in sites, f"{o.key}: cv2.{o.fn} not called in {o.file}:{o.func}"
        lines = O.lines_for(o)
        src = (ROOT / o.file).read_text(encoding="utf-8").splitlines()
        assert all(re.search(rf"\bcv2\.{o.fn}\(", src[n - 1]) for n in lines), (o.key, lines)


def test_every_engine_cv2_call_is_benchmarked_or_explained():
    covered = {o.site for o in O.ENGINE_OPS} | set(O.NOT_BENCHMARKED)
    missing = sorted({c.site for c in O.engine_cv2_calls()} - covered)
    assert not missing, f"engine cv2 calls with no microbench entry: {missing}"
    stale = sorted(set(O.NOT_BENCHMARKED) - {c.site for c in O.engine_cv2_calls()})
    assert not stale, f"NOT_BENCHMARKED entries no longer in the engine: {stale}"


def test_groups_follow_the_kleidicv_list():
    for o in O.ENGINE_OPS:
        want = O.KLEIDICV if o.fn in O.KLEIDICV_LISTED else O.CONTROL
        assert o.group == want, o.key
    for fn in ("matchTemplate", "dft", "phaseCorrelate", "findTransformECC", "imencode", "magnitude", "log"):
        assert O.group_of(fn) == O.CONTROL
    for fn in (
        "GaussianBlur",
        "sepFilter2D",
        "Sobel",
        "dilate",
        "erode",
        "pyrDown",
        "warpPerspective",
        "minMaxLoc",
        "cvtColor",
        "resize",
    ):
        assert O.group_of(fn) == O.KLEIDICV
    assert {o.group for o in O.PIPELINE_EXTRA_OPS} == {O.NUMPY}
    assert all(not o.is_engine_call for o in O.COUNTERFACTUAL_OPS)


def test_listed_but_unused_and_unique_keys():
    O.assert_unique_keys()
    assert "minMaxIdx" not in O.LISTED_BUT_UNUSED  # reached via minMaxLoc
    for fn in ("medianBlur", "remap", "calcOpticalFlowPyrLK", "absdiff"):
        assert fn in O.LISTED_BUT_UNUSED


def test_stage_names_match_engine_clock():
    from bench.cool.stagebench import ENGINE_STAGES

    src = (ROOT / "driftsense" / "v2" / "engine.py").read_text(encoding="utf-8")
    marked = re.findall(r'clk\.mark\("([a-z_]+)"', src)
    assert tuple(marked) == ENGINE_STAGES


def test_readme_table_cites_every_engine_op():
    text = (ROOT / "bench" / "README.md").read_text(encoding="utf-8")
    for o in O.ENGINE_OPS:
        assert o.key.removeprefix("op.") in text, o.key
        cite = f"{Path(o.file).name}:{','.join(str(n) for n in O.lines_for(o))}"
        assert cite in text, f"README does not cite {cite} for {o.key}"


def test_agent_constants_match_eval():
    import ast

    from bench.cool import agent_bench as A

    tree = ast.parse((ROOT / "eval" / "agent_eval.py").read_text(encoding="utf-8"))
    consts = {
        t.targets[0].id: ast.literal_eval(t.value)
        for t in tree.body
        if isinstance(t, ast.Assign)
        and isinstance(t.targets[0], ast.Name)
        and t.targets[0].id.isupper()
        and isinstance(t.value, ast.Constant | ast.Tuple)
    }
    for k in (
        "TAU",
        "TOL_NM",
        "FAMILIES",
        "SITE_NM",
        "PIXEL_NM",
        "REF_FOV_NM",
        "REF_DWELL_US",
        "BASE_FOV_NM",
        "BASE_DWELL_US",
        "TEMPLATE_FRAC",
        "MAX_STEPS",
    ):
        assert getattr(A, k) == consts[k], k
