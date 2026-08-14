"""
Unit tests for the DriftSense data engine.

Runs with plain python (no pytest required):

    python -m tests.test_datagen

These tests are regression guards for the class of bug found in v1, where the
line-width profile was applied along the line instead of across it and the
generator silently produced near-empty images.  Every test therefore asserts
on MEASURABLE STRUCTURE, not just on shapes and dtypes.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (AperiodicityConfig, SEARCH_H, SEARCH_W, get_tier)
from data_gen.structures import (generate_dram, generate_finfet, DRAMParams,
                                  FinFETParams, make_ler_profile,
                                  generate_structure)
from data_gen.sem_effects import (sample_acquisition, render_sem_image,
                                   apply_shot_noise, build_affine,
                                   transform_point, invert_affine)
from data_gen.scene_composer import SceneComposer, verify_pair_geometry


NO_APER = AperiodicityConfig(enable_ler=False, enable_cd_variation=False,
                              enable_contrast_drift=False,
                              enable_defects=False)

_results = []


def test(fn):
    """Minimal test registration decorator."""
    _results.append(fn)
    return fn


# ===========================================================================
# Structure rendering
# ===========================================================================

@test
def test_bar_positions_and_width():
    """Bars must be centred at phase + k*period with the requested width."""
    p = DRAMParams(variant="grid", period_x=20, period_y=1000,
                   line_width_x=6, line_width_y=0.1,
                   phase_x=10, phase_y=999, background=0.0,
                   brightness_x=1.0, brightness_y=0.0)
    img, _ = generate_dram(60, 100, p, NO_APER, np.random.default_rng(0))

    row = img[10, :]
    lit = np.where(row > 0.5)[0]
    assert lit.size > 0, "no lit pixels — bars not rendered"

    # Group contiguous runs
    groups, cur = [], [lit[0]]
    for v in lit[1:]:
        if v == cur[-1] + 1:
            cur.append(v)
        else:
            groups.append(cur); cur = [v]
    groups.append(cur)

    centres = [np.mean(g) for g in groups]
    widths  = [len(g) for g in groups]

    expected = [10.0, 30.0, 50.0, 70.0, 90.0]
    assert len(centres) == len(expected), \
        f"expected {len(expected)} bars, got {len(centres)}: {centres}"
    for c, e in zip(centres, expected):
        assert abs(c - e) < 1.0, f"bar centre {c:.2f} != expected {e}"
    for w in widths:
        assert 5 <= w <= 7, f"bar width {w} not ~6 px"


@test
def test_periodicity_via_autocorrelation():
    """The rendered image's autocorrelation must peak at the true period."""
    p = DRAMParams(variant="grid", period_x=24, period_y=1000,
                   line_width_x=6, line_width_y=0.1,
                   phase_x=0, phase_y=999, background=0.0,
                   brightness_x=1.0, brightness_y=0.0)
    img, info = generate_dram(64, 480, p, NO_APER, np.random.default_rng(1))

    prof = img.mean(axis=0)
    prof = prof - prof.mean()
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    # First non-trivial peak should be at lag == period_x
    search = ac[5:60]
    lag = int(np.argmax(search)) + 5
    assert abs(lag - 24) <= 1, f"autocorrelation peak at lag {lag}, expected 24"
    assert abs(info["period_x"] - 24) < 1e-6


@test
def test_image_has_real_contrast():
    """Guard against the v1 near-empty-image failure."""
    for stype in ("dram", "finfet"):
        img, _ = generate_structure(stype, 300, 300, NO_APER,
                                     np.random.default_rng(3))
        assert img.mean() > 0.10, f"{stype} mean {img.mean():.4f} too dark"
        assert img.std() > 0.15, f"{stype} std {img.std():.4f} — no structure"
        assert img.max() > 0.8, f"{stype} max {img.max():.3f} — no bright lines"


@test
def test_finfet_is_anisotropic():
    """FinFET fin pitch and gate pitch must differ substantially."""
    p = FinFETParams(fin_period=10, gate_period=45, fin_axis=0,
                     background=0.0, enable_fin_cut=False)
    img, info = generate_finfet(300, 300, p, NO_APER,
                                 np.random.default_rng(4))
    # Dominant frequency along x (fins) vs along y (gates)
    fx = np.abs(np.fft.rfft(img.mean(axis=0) - img.mean()))
    fy = np.abs(np.fft.rfft(img.mean(axis=1) - img.mean()))
    peak_x = int(np.argmax(fx[1:])) + 1
    peak_y = int(np.argmax(fy[1:])) + 1
    period_x = 300 / peak_x
    period_y = 300 / peak_y
    assert abs(period_x - 10) < 2, f"fin period measured {period_x:.1f}"
    assert abs(period_y - 45) < 6, f"gate period measured {period_y:.1f}"
    assert period_y > 2 * period_x, "spectrum is not anisotropic"


@test
def test_ler_statistics():
    """LER profile must have the requested RMS and be smooth (correlated)."""
    rng = np.random.default_rng(5)
    prof = make_ler_profile(4000, sigma_px=0.6, correlation_px=20.0, rng=rng)
    assert abs(prof.std() - 0.6) < 0.05, f"LER std {prof.std():.3f} != 0.6"
    # Correlated noise: adjacent-sample difference much smaller than std
    d = np.diff(prof).std()
    assert d < prof.std() * 0.5, "LER is not spatially correlated"


@test
def test_ler_changes_image_but_preserves_period():
    """Enabling LER must perturb pixels without destroying periodicity."""
    p = DRAMParams(variant="grid", period_x=24, period_y=26,
                   line_width_x=6, line_width_y=6, phase_x=0, phase_y=0,
                   background=0.0)
    a, _ = generate_dram(256, 256, p, NO_APER, np.random.default_rng(6))
    aper = AperiodicityConfig(enable_ler=True, enable_cd_variation=False,
                               enable_contrast_drift=False,
                               enable_defects=False, ler_sigma_px=0.8)
    b, _ = generate_dram(256, 256, p, aper, np.random.default_rng(6))
    diff = np.abs(a - b).mean()
    assert diff > 0.001, "LER had no effect on the image"
    assert b.std() > 0.15, "LER destroyed the structure"


# ===========================================================================
# SEM effects
# ===========================================================================

@test
def test_shot_noise_follows_poisson_scaling():
    """Noise std must scale as 1/sqrt(dose)."""
    flat = np.full((256, 256), 0.5, dtype=np.float32)
    rng = np.random.default_rng(7)
    s_low  = apply_shot_noise(flat, 50.0, rng).std()
    s_high = apply_shot_noise(flat, 800.0, rng).std()
    ratio = s_low / s_high
    expected = np.sqrt(800.0 / 50.0)          # = 4.0
    assert abs(ratio - expected) / expected < 0.25, \
        f"noise ratio {ratio:.2f}, expected ~{expected:.2f}"


@test
def test_acquisitions_are_independent():
    """Two acquisitions of the same clean image must differ stochastically."""
    clean, _ = generate_structure("dram", 200, 200, NO_APER,
                                   np.random.default_rng(8))
    tier = get_tier("nominal")
    rng = np.random.default_rng(9)
    acq = sample_acquisition(tier.sem, "medium", rng)
    a = render_sem_image(clean, acq, np.random.default_rng(100))
    b = render_sem_image(clean, acq, np.random.default_rng(200))
    assert not np.allclose(a, b), "acquisitions share noise — not independent"
    assert np.abs(a - b).mean() > 0.005, "noise difference implausibly small"


@test
def test_render_is_deterministic_given_seed():
    """Same params + same seed must reproduce identical pixels."""
    clean, _ = generate_structure("finfet", 180, 180, NO_APER,
                                   np.random.default_rng(10))
    tier = get_tier("nominal")
    acq = sample_acquisition(tier.sem, "medium", np.random.default_rng(11))
    a = render_sem_image(clean, acq, np.random.default_rng(555))
    b = render_sem_image(clean, acq, np.random.default_rng(555))
    assert np.array_equal(a, b), "rendering is not reproducible"


@test
def test_output_range_valid():
    """All rendered output must stay in [0,1] and be finite."""
    clean, _ = generate_structure("dram", 200, 200, AperiodicityConfig(),
                                   np.random.default_rng(12))
    tier = get_tier("hard")
    for lvl in ("low", "medium", "high"):
        acq = sample_acquisition(tier.sem, lvl, np.random.default_rng(13))
        out = render_sem_image(clean, acq, np.random.default_rng(14))
        assert np.all(np.isfinite(out)), "non-finite pixels"
        assert out.min() >= 0.0 and out.max() <= 1.0, "output out of [0,1]"


# ===========================================================================
# Geometry
# ===========================================================================

@test
def test_affine_roundtrip():
    """M then M_inv must return the original point."""
    M = build_affine((500.0, 500.0), 4.2, 1.035)
    Mi = invert_affine(M)
    for (x, y) in [(0, 0), (123.4, 987.6), (999, 1)]:
        px, py = transform_point(M, x, y)
        rx, ry = transform_point(Mi, px, py)
        assert abs(rx - x) < 1e-3 and abs(ry - y) < 1e-3, \
            f"roundtrip failed for ({x},{y}) -> ({rx},{ry})"


@test
def test_ground_truth_is_correct_clean_tier():
    """
    The declared GT must coincide with the actual NCC peak location.

    This is the single most important test in the suite: it validates that
    the labels we train and evaluate on are truthful.
    """
    tier = get_tier("clean")
    comp = SceneComposer(seed=321)
    offsets = []
    for i in range(6):
        ref, search, meta = comp.compose(i, tier, structure_type="random")
        ok, off = verify_pair_geometry(ref, search, meta, tolerance_px=3.0)
        offsets.append(off)
        assert ok, f"pair {i}: GT off by {off:.2f}px"
    assert np.median(offsets) < 1.5, \
        f"median GT offset {np.median(offsets):.2f}px too large"


@test
def test_ground_truth_correct_under_drift_and_rotation():
    """
    REGRESSION: scan drift is a horizontal shear (x' = x + d*y) that displaces
    content by up to d*1000 px.  An earlier version computed the label BEFORE
    drift was applied to the pixels, so labels in the drift-bearing tiers were
    systematically wrong by ~d*gt_y pixels.  This test locks that fix in.
    """
    for tier_name in ("nominal", "hard"):
        tier = get_tier(tier_name)
        comp = SceneComposer(seed=1357)
        offsets = []
        for i in range(6):
            ref, search, meta = comp.compose(i, tier)
            ok, off = verify_pair_geometry(ref, search, meta, tolerance_px=3.0)
            offsets.append(off)
            assert ok, (f"[{tier_name}] pair {i}: GT off by {off:.2f}px "
                        f"(drift={meta.search_acq['drift_rate']:.4f}, "
                        f"rot={meta.rotation_deg:.2f})")
        assert np.median(offsets) < 1.5, \
            f"[{tier_name}] median GT offset {np.median(offsets):.2f}px"


@test
def test_drift_is_a_shear_of_known_magnitude():
    """Verify the drift model matches the analytic shear used by the labeller."""
    from data_gen.sem_effects import apply_drift_distortion
    img = np.zeros((200, 200), dtype=np.float32)
    img[:, 100] = 1.0                      # a vertical line at x=100
    d = 0.05
    out = apply_drift_distortion(img, d, axis=0)
    for y in (0, 50, 100, 150, 199):
        row = out[y]
        pos = float(np.argmax(row))
        expected = 100 + d * y             # x' = x + d*y
        assert abs(pos - expected) <= 1.0, \
            f"row {y}: line at {pos}, expected {expected}"


@test
def test_gt_is_subpixel():
    """GT must not be quantized to integers."""
    tier = get_tier("clean")
    comp = SceneComposer(seed=654)
    fracs = []
    for i in range(8):
        _, _, meta = comp.compose(i, tier)
        fracs.append(abs(meta.gt_x - round(meta.gt_x)))
    assert max(fracs) > 0.05, "GT appears integer-quantized"


@test
def test_shapes_and_reference_scale():
    """Search must be 1000x1000 and reference ~10x smaller."""
    tier = get_tier("nominal")
    comp = SceneComposer(seed=777)
    for i in range(4):
        ref, search, meta = comp.compose(i, tier)
        assert search.shape == (SEARCH_H, SEARCH_W), f"search {search.shape}"
        assert ref.shape == (meta.ref_h, meta.ref_w)
        ratio = SEARCH_W / meta.ref_w
        assert 7.0 <= ratio <= 13.0, f"ref/search ratio {ratio:.1f} not ~10x"


@test
def test_search_has_no_empty_border():
    """
    Warping must not leave black padding at the search-image border.

    v1 warped a 1000x1000 image directly, producing reflected/empty borders.
    v2 warps an oversized scene and crops, so all borders carry real content.
    """
    tier = get_tier("hard")
    comp = SceneComposer(seed=888)
    for i in range(4):
        _, search, _ = comp.compose(i, tier)
        for strip in (search[:6, :], search[-6:, :],
                      search[:, :6], search[:, -6:]):
            assert strip.mean() > 0.02, "black border detected after warp"


@test
def test_pair_reproducibility():
    """Same composer seed must reproduce identical pairs."""
    tier = get_tier("nominal")
    a_ref, a_search, a_meta = SceneComposer(seed=2024).compose(0, tier)
    b_ref, b_search, b_meta = SceneComposer(seed=2024).compose(0, tier)
    assert np.array_equal(a_ref, b_ref), "reference not reproducible"
    assert np.array_equal(a_search, b_search), "search not reproducible"
    assert a_meta.gt_x == b_meta.gt_x and a_meta.gt_y == b_meta.gt_y


@test
def test_ambiguous_tier_has_no_aperiodic_content():
    """The ablation tier must genuinely disable LER/CD/defects."""
    tier = get_tier("ambiguous")
    comp = SceneComposer(seed=999)
    _, _, meta = comp.compose(0, tier)
    assert meta.n_defects == 0, "defects present in ambiguous tier"
    assert tier.aperiodicity.enable_ler is False
    assert tier.aperiodicity.enable_cd_variation is False


# ===========================================================================
# Runner
# ===========================================================================

def main() -> int:
    passed, failed = 0, 0
    print(f"Running {len(_results)} tests\n" + "=" * 62)
    for fn in _results:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}\n          {e}")
            failed += 1
        except Exception:
            print(f"  ERROR {name}")
            traceback.print_exc()
            failed += 1
    print("=" * 62)
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
