import json

import numpy as np
import pytest

from bench.cool import common as C
from bench.cool import equality as EQ

HDR = ["mode", "pair", "x_px", "y_px", "top1_hit", "engine", "engine_version"]
AHDR = ["episode", "outcome", "steps", "est_x_nm", "est_y_nm", "success"]


def _arm(d, offsets, agent=None, maps=None):
    d.mkdir(parents=True, exist_ok=True)
    C.write_csv(
        d / "e2e_offsets.csv",
        HDR,
        [["fast", f"clean/{i:04d}", repr(x), repr(y), 1, "v2", "t"] for i, (x, y) in enumerate(offsets)],
    )
    if agent is not None:
        C.write_csv(
            d / "agent_offsets.csv",
            AHDR,
            [
                [f"dram-{i}", o, 3, "" if x is None else repr(x), "" if y is None else repr(y), 1]
                for i, (o, x, y) in enumerate(agent)
            ],
        )
    if maps is not None:
        sd = d / "scoremaps"
        sd.mkdir()
        idx = {}
        for name, arr in maps.items():
            np.save(sd / f"{name}.npy", arr)
            idx[name] = {"sha256": EQ._sha(arr), "shape": list(arr.shape), "dtype": str(arr.dtype)}
        (sd / "hashes.json").write_text(json.dumps({"maps": idx}))
    return d


def test_offsets_within_tolerance_pass(tmp_path):
    a = _arm(tmp_path / "a", [(10.0, 20.0), (30.25, 40.5)])
    b = _arm(tmp_path / "b", [(10.0004, 20.0), (30.25, 40.5009)])
    r = EQ.compare(a, b)
    assert r["pass"] and r["offset"]["n"] == 2
    assert r["offset"]["max_abs_diff_px"] == pytest.approx(9e-4, abs=1e-6)
    m = EQ.to_metrics(r)
    assert m["equality.offset.pass"].value == 1.0 and m["equality.pass"].value == 1.0


def test_offsets_beyond_tolerance_or_missing_fail(tmp_path):
    a = _arm(tmp_path / "a", [(10.0, 20.0), (30.0, 40.0)])
    b = _arm(tmp_path / "b", [(10.0, 20.0), (30.002, 40.0)])
    r = EQ.compare(a, b)
    assert not r["pass"] and r["offset"]["n_exceeding"] == 1
    c = _arm(tmp_path / "c", [(10.0, 20.0)])
    r = EQ.compare(a, c)
    assert not r["offset"]["pass"] and r["offset"]["missing"] == ["fast:clean/0001"]


def test_agent_estimates_gate(tmp_path):
    a = _arm(tmp_path / "a", [(1.0, 1.0)], agent=[("commit", 5.0, -2.0), ("escalate", None, None)])
    b = _arm(tmp_path / "b", [(1.0, 1.0)], agent=[("commit", 5.0005, -2.0), ("escalate", None, None)])
    assert EQ.compare(a, b)["agent"]["pass"]
    c = _arm(tmp_path / "c", [(1.0, 1.0)], agent=[("commit", 5.0, -2.0), ("commit", 1.0, 1.0)])
    r = EQ.compare(a, c)
    assert not r["agent"]["pass"] and r["agent"]["outcome_agreement"] == 0.5 and not r["pass"]


def test_scoremap_max_abs_diff_reported_not_gated(tmp_path):
    m = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    a = _arm(tmp_path / "a", [(1.0, 1.0)], maps={"s": m})
    b = _arm(tmp_path / "b", [(1.0, 1.0)], maps={"s": m + np.float32(1e-6)})
    r = EQ.compare(a, b)
    assert r["pass"]
    assert r["scoremaps"]["s"]["sha256_equal"] is False
    assert r["scoremaps"]["s"]["max_abs_diff"] == pytest.approx(1e-6, rel=0.2)
    assert "equality.scoremap.s.max_abs_diff" in EQ.to_metrics(r)
