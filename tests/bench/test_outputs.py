"""Every bench output validates as driftsense.contracts.ResultsFile (tmp dirs only; never results/)."""

import json

import pytest

from bench.cool import common as C

pytest.importorskip("cv2")
pytest.importorskip("psutil")
contracts = pytest.importorskip("driftsense.contracts")


def _load(path):
    rf = contracts.ResultsFile.model_validate_json(path.read_text(encoding="utf-8"))
    assert rf.schema_id == "driftsense.results/1" and rf.dataset == "bench"
    assert rf.git_sha == "test-sha"
    assert rf.env.cool is C.cv2_is_cool()  # filled from this machine
    assert rf.env.instance_type is None  # IMDS disabled in tests
    return rf


def test_microbench_replays_engine_args(tmp_path):
    from bench.cool import microbench

    rx = "GaussianBlur|matchTemplate|findTransformECC|warpPerspective|residual_elementwise|cf_grad"
    assert (
        microbench.main(
            [
                "--quick",
                "--threads",
                "1",
                "--ops",
                rx,
                "--runs",
                "3",
                "--warmup",
                "1",
                "--min-sample-ms",
                "0",
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    rf = _load(tmp_path / "microbench.json")
    assert rf.produced_by == "bench/cool/microbench.py"
    meta = json.loads((tmp_path / "microbench_ops.json").read_text())
    ops = meta["ops"]
    assert not any(v["errors"] for v in ops.values()), {k: v["errors"] for k, v in ops.items() if v["errors"]}
    assert meta["uncovered_sites"] == []
    # production parameters, not synthetic ones
    assert ops["op.GaussianBlur.s3_localnorm_f32"]["signature"].startswith("float32[1000, 1000], (25,25), 8")
    assert ops["op.warpPerspective.s2_derotate_u8"]["signature"].startswith("uint8[1000, 1000]")
    assert ops["op.matchTemplate.s4_ccoeff_normed_f32"]["signature"].endswith(", 5")  # TM_CCOEFF_NORMED
    assert ops["op.matchTemplate.s4_ccoeff_normed_f32"]["group"] == "control"
    m = rf.metrics["op.GaussianBlur.s3_localnorm_f32.t1.median_ms"]
    assert m.unit == "ms" and m.n == 3 and m.ci_low <= m.value <= m.ci_high
    assert rf.metrics["meta.quick"].value == 1.0


def test_microbench_list(capsys):
    from bench.cool import microbench

    assert microbench.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "roughness.py:134,135" in out and "counterfactual" in out


def test_census_and_stagebench_on_dev_pairs(tmp_path):
    from bench.cool import census, stagebench

    assert census.main(["--quick", "--pairs", "1", "--modes", "fast", "--out-dir", str(tmp_path)]) == 0
    rf = _load(tmp_path / "census.json")
    share = rf.metrics["census.fast.share.kleidicv"].value
    assert 0.0 < share < 1.0
    assert rf.metrics["census.fast.matchTemplate@score.score_residual.calls_per_localize"].value == 1.0
    assert json.loads((tmp_path / "census_sites.json").read_text())["uncovered"] == []

    assert (
        stagebench.main(
            [
                "--quick",
                "--pairs",
                "2",
                "--modes",
                "fast",
                "--threads",
                "1",
                "--warmup",
                "0",
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    rf = _load(tmp_path / "stagebench.json")
    assert rf.name == "stagebench.v2"
    for st in stagebench.ENGINE_STAGES + ("total", "wall"):
        assert f"stage.fast.{st}.t1.median_ms" in rf.metrics, st
    assert "check.fast.top1_rate.t1" in rf.metrics


def test_e2e_latency_only(tmp_path):
    from bench.cool import e2e

    assert (
        e2e.main(
            [
                "--quick",
                "--pairs",
                "2",
                "--modes",
                "fast",
                "--warmup",
                "0",
                "--latency-threads",
                "1",
                "--no-throughput",
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    rf = _load(tmp_path / "e2e.json")
    assert rf.metrics["e2e.fast.latency.t1.p50_ms"].n == 2
    offs = C.read_csv(tmp_path / "e2e_offsets.csv")
    assert [r["pair"] for r in offs] == ["clean/0000", "nominal/0000"]


@pytest.mark.slow
def test_e2e_and_agent_with_process_pool(tmp_path):
    from bench.cool import agent_bench, e2e

    assert (
        e2e.main(
            [
                "--quick",
                "--pairs",
                "2",
                "--modes",
                "fast",
                "--workers",
                "2",
                "--jobs",
                "4",
                "--latency-threads",
                "1",
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    rf = _load(tmp_path / "e2e.json")
    assert rf.metrics["e2e.fast.throughput.jobs_per_hour"].value > 0
    assert agent_bench.main(["--quick", "--workers", "1", "--jobs", "1", "--out-dir", str(tmp_path)]) == 0
    rf = _load(tmp_path / "agent.json")
    assert rf.metrics["agent.throughput.episodes_per_hour"].value > 0
    assert rf.metrics["meta.episodes"].value == 2


@pytest.mark.slow
def test_equality_dump_is_reproducible(tmp_path):
    from bench.cool import equality as EQ

    a = EQ.dump(tmp_path / "a", pairs=(("clean", 0),))
    b = EQ.dump(tmp_path / "b", pairs=(("clean", 0),))
    assert a == b and any(k.endswith("score_map_f32") for k in a)
