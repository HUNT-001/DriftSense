import numpy as np
import pytest

from bench.cool import common as C


def test_bootstrap_ci_contains_median_and_is_deterministic():
    x = np.random.default_rng(1).lognormal(0.0, 0.3, 40)
    med, lo, hi = C.bootstrap_median_ci(x, seed=7)
    assert lo <= med <= hi
    assert med == pytest.approx(float(np.median(x)))
    assert C.bootstrap_median_ci(x, seed=7) == (med, lo, hi)
    assert C.bootstrap_median_ci(x, seed=8) != (med, lo, hi)


def test_bootstrap_ci_narrows_with_more_samples():
    rng = np.random.default_rng(2)
    _, lo1, hi1 = C.bootstrap_median_ci(rng.normal(10, 1, 10))
    _, lo2, hi2 = C.bootstrap_median_ci(rng.normal(10, 1, 1000))
    assert (hi2 - lo2) < (hi1 - lo1)


def test_single_sample_ci_degenerates():
    assert C.bootstrap_median_ci([3.0]) == (3.0, 3.0, 3.0)
    with pytest.raises(ValueError):
        C.bootstrap_median_ci([])


def test_ratio_ci_includes_one_for_identical_samples():
    x = np.random.default_rng(3).normal(5.0, 0.5, 30)
    r, lo, hi = C.bootstrap_ratio_ci(x, x.copy())
    assert r == pytest.approx(1.0)
    assert C.ci_includes(lo, hi, 1.0)


def test_ratio_ci_detects_two_x_speedup():
    rng = np.random.default_rng(4)
    base = rng.normal(10.0, 0.3, 30)
    cand = rng.normal(5.0, 0.15, 30)
    r, lo, hi = C.bootstrap_ratio_ci(base, cand)
    assert r == pytest.approx(2.0, rel=0.05)
    assert lo > 1.0 and not C.ci_includes(lo, hi, 1.0)
    assert C.bootstrap_ratio_ci(base, cand) == (r, lo, hi)


def test_summarize_fields():
    s = C.summarize(np.arange(1, 101, dtype=float))
    assert s["n"] == 100 and s["p50"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05)
    assert s["ci_low"] <= s["median"] <= s["ci_high"]


def test_time_callable_and_inner_batching():
    calls = []
    ms = C.time_callable(lambda: calls.append(1), warmup=2, runs=4, inner=3)
    assert ms.shape == (4,) and (ms >= 0).all()
    assert len(calls) == 2 + 4 * 3
    assert C.calibrate_inner(lambda: None, 0.0) == 1
    assert 1 <= C.calibrate_inner(lambda: None, 0.5, cap=50) <= 50


def test_thread_resolution():
    import os

    assert C.resolve_threads("1") == 1
    assert C.resolve_threads("all") == (os.cpu_count() or 1)
    assert C.thread_tag(1) == "t1"
    if (os.cpu_count() or 1) > 1:
        assert C.thread_tag(os.cpu_count()) == "tN"
    with pytest.raises(ValueError):
        C.resolve_threads("-2")


def test_cool_detection_from_path():
    assert C.cv2_is_cool("/opt/cool/venvs/python_3.12/lib/python3.12/site-packages/cv2/__init__.py")
    assert not C.cv2_is_cool("/home/ubuntu/venv/lib/python3.12/site-packages/cv2/__init__.py")
    assert not C.cv2_is_cool(r"C:\venv\Lib\site-packages\cv2\__init__.py")


def test_imds_disabled_returns_none():
    assert C.imds_get("meta-data/instance-type") is None
