"""report.py on synthetic per-arm fixtures: merge, verdicts, Amdahl, equality, cost, smoke guard."""

import json

import numpy as np
import pytest

from bench.cool import common as C
from bench.cool import report as R

contracts = pytest.importorskip("driftsense.contracts")

OPS = [
    ("op.GaussianBlur.s3_localnorm_f32", "kleidicv", 1.0),
    ("op.sepFilter2D.s3_along_f32", "kleidicv", 0.5),
    ("op.matchTemplate.s4_ccoeff_normed_f32", "control", 5.0),
    ("op.numpy.s6_v1_fingerprint_rerank", "numpy", 2.0),
]
MB_HDR = ["key", "op", "variant", "group", "stage", "threads", "tag", "inner", "run", "ms"]


def _noise(seed, n=30):
    return 1 + 0.02 * np.random.default_rng(seed).standard_normal(n)


def make_arm(
    root,
    label,
    kcv_scale,
    itype,
    cool,
    jph,
    eph=600.0,
    offsets=((10.0, 20.0), (30.0, 40.0)),
    arch="aarch64",
    quick=False,
    sha="abc123",
):
    d = root / label
    d.mkdir(parents=True)
    rows = []
    for i, (key, group, base) in enumerate(OPS):
        f = kcv_scale if group == "kleidicv" else 1.0  # same noise in every arm -> controls identical
        _, op, var = key.split(".", 2)
        rows += [
            [key, op, var, group, "S3", 1, "t1", 1, r, f"{v:.6f}"]
            for r, v in enumerate(base * f * _noise(100 + i))
        ]
    C.write_csv(d / "microbench_samples.csv", MB_HDR, rows)
    C.write_csv(
        d / "stagebench_samples.csv",
        ["mode", "tag", "threads", "pair", "repeat", "stage", "ms"],
        [
            ["auto", "t1", 1, f"p{i}", 0, "total", f"{v:.6f}"]
            for i, v in enumerate(400 * (0.9 + 0.1 * kcv_scale) * _noise(5))
        ],
    )
    C.write_csv(
        d / "e2e_samples.csv",
        ["mode", "phase", "tag", "threads", "pair_or_job", "repeat_or_trial", "ms", "cpu_ms"],
        [["auto", "latency", "tN", 16, f"p{i}", 0, f"{v:.6f}", ""] for i, v in enumerate(400 * _noise(7))],
    )
    C.write_csv(
        d / "e2e_throughput.csv",
        ["mode", "trial", "workers", "jobs", "wall_s", "jobs_per_hour", "cpu_pct"],
        [["auto", t, 16, 240, 1.0, f"{jph * (1 + 0.001 * t):.3f}", 99.0] for t in range(3)],
    )
    C.write_csv(
        d / "agent_throughput.csv",
        ["trial", "workers", "jobs", "wall_s", "episodes_per_hour", "cpu_pct"],
        [[t, 16, 64, 1.0, f"{eph * (1 + 0.001 * t):.3f}", 99.0] for t in range(3)],
    )
    C.write_csv(
        d / "e2e_offsets.csv",
        ["mode", "pair", "x_px", "y_px", "top1_hit", "engine", "engine_version"],
        [["auto", f"clean/{i:04d}", repr(x), repr(y), 1, "v2", "t"] for i, (x, y) in enumerate(offsets)],
    )
    env = contracts.EnvInfo(
        python="3.12.3",
        opencv="5.0.0",
        numpy="2.5.4",
        platform="Linux",
        arch=arch,
        instance_type=itype,
        cool=cool,
    )
    q = C.metric(1.0 if quick else 0.0, "bool")
    C.write_results(d / "e2e.json", "e2e.v2", {"meta.quick": q}, "bench/cool/e2e.py", env=env, sha=sha)
    C.write_results(
        d / "census.json",
        "census",
        {
            "census.auto.share.kleidicv": C.metric(0.25, "fraction"),
            "census.auto.share.control": C.metric(0.25, "fraction"),
            "meta.quick": q,
        },
        "bench/cool/census.py",
        env=env,
        sha=sha,
    )
    ev = d / "evidence"
    ev.mkdir()
    (ev / "evidence.json").write_text(
        json.dumps(
            {
                "cv2_file": "/opt/cool/venvs/python_3.12/lib/cv2/__init__.py"
                if cool
                else "/venv/cv2/__init__.py",
                "cv2_version": "5.0.0",
                "cool_detected": cool,
                "build_mentions_kleidicv": cool,
                "imds": {"instance_type": itype, "region": "ap-south-1", "ami_id": "ami-123"},
            }
        )
    )
    (ev / "build_information.txt").write_text(
        "  Custom HAL: YES (kleidicv (ver 0.6.0))\n" if cool else "HAL: NO\n"
    )
    return d


@pytest.fixture
def campaign(tmp_path):
    arms = tmp_path / "run"
    make_arm(arms, "arm-pip", 1.0, "c8g.4xlarge", False, 36000.0)
    make_arm(arms, "arm-cool", 0.5, "c8g.4xlarge", True, 72000.0)
    make_arm(arms, "x86-pip", 1.0, "c7i.4xlarge", False, 30000.0, arch="x86_64")
    make_arm(arms, "arm-cool-large", 0.5, "c8g.large", True, 6400.0)
    return arms


def test_cost_per_1000_arithmetic():
    assert R.cost_per_1000(0.054, 0.01, 6400.0) == pytest.approx(0.01)
    assert R.cost_per_1000(0.4318, 0.0, 43180.0) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        R.cost_per_1000(1.0, 0.0, 0.0)


def test_prices_file_has_sources_and_dates():
    prices = R.load_prices()
    assert prices["region"] == "ap-south-1" and prices["as_of"]
    for section in ("ec2", "cool_fee"):
        for itype, e in prices[section].items():
            assert e.get("source") and "as_of" in e, (section, itype)
    for itype in ("c7i.4xlarge", "c8g.4xlarge", "c8g.large"):
        assert prices["ec2"][itype]["usd_per_hour"] > 0
    ec2, fee, w = R.hourly_price(prices, "c8g.4xlarge", True, {})
    assert (ec2, fee, w) == (0.4318, 0.04, [])
    ec2, fee, w = R.hourly_price(prices, "c8g.4xlarge", False, {})
    assert fee == 0.0  # pip arms pay no COOL fee
    ec2, _, w = R.hourly_price(prices, "c7i.4xlarge", False, {"c7i.4xlarge": 0.8})
    assert ec2 == 0.8 and not w
    _, _, w = R.hourly_price(prices, "m9z.huge", False, {})
    assert any("no EC2 price" in x for x in w)


def test_report_merges_arms(campaign, tmp_path):
    out = tmp_path / "report"
    res = R.build_report(R.discover_arms(campaign), out, charts=False)
    rf = C.load_results(out / "cool_bench.json")
    assert rf.name == "cool_bench" and rf.dataset == "bench" and rf.git_sha == "abc123"
    assert rf.env.cool is True and rf.env.instance_type == "c8g.4xlarge"  # env from arm-cool
    m = rf.metrics
    cmp_ = "arm-cool_vs_arm-pip"
    gb = m[f"speedup.{cmp_}.kleidicv.GaussianBlur.s3_localnorm_f32.t1"]
    assert gb.value == pytest.approx(2.0, rel=0.03) and gb.ci_low > 1.0 and gb.unit == "x"
    mt = m[f"speedup.{cmp_}.control.matchTemplate.s4_ccoeff_normed_f32.t1"]
    assert mt.value == pytest.approx(1.0) and C.ci_includes(mt.ci_low, mt.ci_high, 1.0)
    npc = m[f"speedup.{cmp_}.numpy.numpy.s6_v1_fingerprint_rerank.t1"]
    assert C.ci_includes(npc.ci_low, npc.ci_high, 1.0)
    assert m[f"speedup.{cmp_}.kleidicv.geomean.t1"].value == pytest.approx(2.0, rel=0.03)
    assert m[f"speedup.{cmp_}.e2e.auto.throughput"].value == pytest.approx(2.0, rel=0.01)
    assert f"speedup.{cmp_}.e2e.auto.latency.tN" in m
    assert f"speedup.{cmp_}.stage.auto.total.t1" in m
    # Amdahl: 1 / (0.25/2 + 0.25/1 + 0.5/1) = 1.142857
    assert m[f"amdahl.{cmp_}.auto.predicted_speedup"].value == pytest.approx(1 / 0.875, rel=0.02)
    assert m[f"amdahl.{cmp_}.auto.upper_bound_speedup"].value == pytest.approx(1 / 0.75)
    assert m[f"equality.{cmp_}.offset.pass"].value == 1.0
    # cost: (EC2 + COOL fee) / per hour x 1000
    assert m["cost.arm-cool.e2e.auto.usd_per_1000_localizations"].value == pytest.approx(
        (0.4318 + 0.04) / 72072.0 * 1000, rel=1e-6
    )
    assert m["cost.arm-pip.e2e.auto.usd_per_1000_localizations"].value == pytest.approx(
        0.4318 / 36036.0 * 1000, rel=1e-6
    )
    assert m["cost.arm-cool-large.e2e.auto.usd_per_1000_localizations"].value == pytest.approx(
        (0.054 + 0.01) / 6406.4 * 1000, rel=1e-6
    )
    assert m["cost.x86-pip.e2e.auto.usd_per_1000_localizations"].value == pytest.approx(
        0.714 / 30030 * 1000, rel=1e-6
    )
    assert "cost.arm-cool.agent.usd_per_1000_episodes" in m
    assert m["headline.cool_vs_pip.e2e_auto_throughput"].value == pytest.approx(2.0, rel=0.01)
    assert m["meta.local_smoke"].value == 0.0
    meta = json.loads((out / "cool_bench_meta.json").read_text())
    rows = {r["key"]: r for r in meta["comparisons"][cmp_]["rows"]}
    assert rows["op.matchTemplate.s4_ccoeff_normed_f32.t1"]["verdict"] == R.NO_DIFF
    assert rows["op.GaussianBlur.s3_localnorm_f32.t1"]["verdict"] == R.FASTER
    md = (out / "cool_bench_summary.md").read_text()
    assert R.NO_DIFF in md and "Cost per 1,000" in md and "LOCAL SMOKE" not in md
    docs = json.loads((out / "cool_bench_docs.json").read_text())
    assert docs["arm.arm-cool.ami_id"] == "ami-123" and docs["arm.arm-cool.region"] == "ap-south-1"
    assert docs["arm.arm-cool.build_kleidicv_lines"] == ["Custom HAL: YES (kleidicv (ver 0.6.0))"]
    assert docs["headline.cool_vs_pip.e2e_auto_throughput.verdict"] == R.FASTER
    assert docs[f"equality.{cmp_}.pass"] is True
    assert res["meta"]["env_from_arm"] == "arm-cool"


def test_no_difference_is_said_when_ci_includes_one(tmp_path):
    arms = tmp_path / "run"
    make_arm(arms, "arm-pip", 1.0, "c8g.4xlarge", False, 36000.0)
    make_arm(arms, "arm-cool", 1.0, "c8g.4xlarge", True, 36000.0)
    out = tmp_path / "r"
    R.build_report(R.discover_arms(arms), out, charts=False)
    meta = json.loads((out / "cool_bench_meta.json").read_text())
    geo = meta["comparisons"]["arm-cool_vs_arm-pip"]["geomeans"]
    assert geo["kleidicv.t1"]["verdict"] == R.NO_DIFF
    docs = json.loads((out / "cool_bench_docs.json").read_text())
    assert docs["arm-cool_vs_arm-pip.geomean.kleidicv.t1.verdict"] == "no measurable difference"


def test_equality_failure_is_flagged(tmp_path):
    arms = tmp_path / "run"
    make_arm(arms, "arm-pip", 1.0, "c8g.4xlarge", False, 36000.0)
    make_arm(arms, "arm-cool", 0.5, "c8g.4xlarge", True, 72000.0, offsets=((10.0, 20.0), (30.01, 40.0)))
    res = R.build_report(R.discover_arms(arms), tmp_path / "r", charts=False)
    assert any("EQUALITY FAILED" in w for w in res["meta"]["warnings"])
    assert res["results"].metrics["equality.arm-cool_vs_arm-pip.pass"].value == 0.0


def test_smoke_guard(tmp_path):
    arms = tmp_path / "run"
    make_arm(arms, "arm-pip", 1.0, None, False, 36000.0, quick=True)
    with pytest.raises(R.SmokeGuardError):
        R.build_report(R.discover_arms(arms), C.REPO_ROOT / "results" / "bench", charts=False)
    res = R.build_report(R.discover_arms(arms), tmp_path / "out", charts=False)
    assert res["results"].name == "cool_bench.local_smoke"
    assert "LOCAL SMOKE" in (tmp_path / "out" / "cool_bench_summary.md").read_text()


@pytest.mark.slow
def test_charts_render(campaign, tmp_path):
    pytest.importorskip("matplotlib")
    res = R.build_report(R.discover_arms(campaign), tmp_path / "c", charts=True)
    names = {p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in res["meta"]["charts"]}
    assert "speedup_arm-cool_vs_arm-pip_t1.png" in names and "cost_per_1000_e2e_auto.png" in names


def test_verdict_rules():
    assert R.verdict(0.9, 1.1) == R.NO_DIFF
    assert R.verdict(1.05, 1.2) == R.FASTER
    assert R.verdict(0.7, 0.95) == R.SLOWER
    assert R.verdict(1.05, 1.2, n=1) == R.INSUFFICIENT
