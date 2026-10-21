import pytest

from bench.cool import pairs as P


def test_dev_pairs_round_robin_and_deterministic():
    a = P.load_dev_pairs(6)
    b = P.load_dev_pairs(6)
    assert [p.key for p in a] == [
        "clean/0000",
        "nominal/0000",
        "hard/0000",
        "ambiguous/0000",
        "clean/0001",
        "nominal/0001",
    ]
    assert [p.search_png for p in a] == [p.search_png for p in b]
    assert all(p.search_png[:8] == b"\x89PNG\r\n\x1a\n" for p in a)


def test_hit_uses_template_centre():
    p = P.load_pair("clean", 0)
    x0, y0 = p.gt_x - p.ref_w / 2.0, p.gt_y - p.ref_h / 2.0
    assert p.hit(x0, y0) and p.hit(x0 + 2.0, y0)
    assert not p.hit(x0 + 3.5, y0)


def test_sealed_split_is_refused(tmp_path):
    with pytest.raises(P.SealedDataError):
        P.load_dev_pairs(1, root=P.REPO_ROOT / "outputs" / "dataset_test")
    with pytest.raises(P.SealedDataError):
        P.load_pair("clean", 0, root=tmp_path / "sealed")


def test_capture_pair_is_rotated_past_derotation_threshold():
    p = P.load_pair(*P.CAPTURE_PAIR)
    assert abs(p.rotation_deg) >= 0.75  # driftsense/v2/engine.py _DEROTATE_MIN_DEG
