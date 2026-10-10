"""
Scene composer — v2 (exact geometry).

Assembles (reference_image, search_image, ground_truth) triples.

CHANGES FROM v1
---------------
  * EXACT ground truth.  v1 warped the search image and then analytically
    re-derived the GT with a hand-written rotation formula plus a separate
    scale formula.  v2 builds ONE affine matrix, warps pixels with it, and
    maps the GT point through THE SAME matrix — so pixel content and GT can
    never disagree.
  * NO BORDER ARTIFACTS.  v1 warped a 1000x1000 image with BORDER_REFLECT_101,
    which fabricates mirrored (physically impossible) structure near the edges
    and, worse, creates a *false symmetry* that a matcher can exploit.  v2
    warps an oversized scene and crops the valid centre, so every pixel is
    real rendered content.
  * SUB-PIXEL ground truth.  The reference crop is taken at a fractional
    offset, so GT is not quantized to integers and sub-pixel refinement can
    actually be measured.
  * SHARED wafer physics, INDEPENDENT scan noise.  The clean scene (including
    its LER, CD variation and defects) is generated ONCE and shared by both
    acquisitions — this is what makes the task solvable.  Only the stochastic
    acquisition effects use independent generators.
  * Reference rendered WITH PADDING then centre-cropped, so blur and edge
    enhancement at the patch border match what the same region looks like
    inside the search image (v1 had mismatched border statistics, an
    unintended cue).

Coordinate systems
------------------
  scene coords   : the large clean canvas (SCENE_H x SCENE_W)
  warped coords  : scene after the affine M is applied (same canvas size)
  search coords  : warped canvas cropped to SEARCH_H x SEARCH_W at (ox, oy)

  Ground truth (gt_x, gt_y) is expressed in SEARCH coords, as the challenge
  requires: the centre of the reference pattern inside the search image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Tuple, Dict, Any

import cv2
import numpy as np

from config import (SEARCH_H, SEARCH_W, SCENE_H, SCENE_W,
                    REF_MIN_SIZE, REF_MAX_SIZE, REF_RENDER_PAD,
                    TierConfig, get_tier)
from data_gen.structures import generate_structure
from data_gen.sem_effects import (AcquisitionParams, sample_acquisition,
                                   render_sem_image, build_affine,
                                   warp_affine, transform_point,
                                   invert_affine)


# ===========================================================================
# Metadata record
# ===========================================================================

@dataclass
class PairMeta:
    pair_id: int
    tier: str
    structure_family: str
    structure_variant: str
    noise_level: str

    # --- Ground truth (search-image coordinates, sub-pixel) ---
    gt_x: float
    gt_y: float

    ref_h: int
    ref_w: int

    # --- True structural periods (used by the failure analysis) ---
    period_x: float
    period_y: float
    n_defects: int

    # --- Geometry relating reference to search ---
    rotation_deg: float
    scale: float
    subpixel_dx: float
    subpixel_dy: float

    # --- Acquisition parameters (independent per image) ---
    ref_acq: Dict[str, Any]
    search_acq: Dict[str, Any]

    seed: int


# ===========================================================================
# Composer
# ===========================================================================

class SceneComposer:
    """
    Generates one (reference, search, meta) triple per compose() call.

    A single master RNG derives one independent seed per pair, so pairs are
    reproducible individually and the dataset is reproducible as a whole.
    """

    def __init__(self, seed: int | None = None):
        self._master_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def compose(self,
                pair_id: int,
                tier: TierConfig,
                structure_type: str = "random",
                noise_level: str | None = None
                ) -> Tuple[np.ndarray, np.ndarray, PairMeta]:

        pair_seed = int(self._master_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(pair_seed)

        if noise_level is None:
            noise_level = str(rng.choice(list(tier.noise_levels)))

        # ---------------------------------------------------------------
        # 1. Clean scene — SHARED wafer physics (LER / CD / defects)
        # ---------------------------------------------------------------
        clean_scene, sinfo = generate_structure(
            structure_type, SCENE_H, SCENE_W, tier.aperiodicity, rng)

        # ---------------------------------------------------------------
        # 2. Geometry: one affine matrix used for BOTH pixels and GT
        # ---------------------------------------------------------------
        g = tier.geometry
        rot   = float(rng.uniform(*g.rotation_deg_range))
        scale = float(rng.uniform(*g.scale_range))

        scene_cx, scene_cy = SCENE_W / 2.0, SCENE_H / 2.0
        M     = build_affine((scene_cx, scene_cy), rot, scale)
        M_inv = invert_affine(M)

        warped_scene = warp_affine(clean_scene, M, (SCENE_W, SCENE_H))

        # Search window = centred crop of the warped scene
        ox = (SCENE_W - SEARCH_W) // 2
        oy = (SCENE_H - SEARCH_H) // 2
        clean_search = warped_scene[oy:oy + SEARCH_H, ox:ox + SEARCH_W].copy()

        # ---------------------------------------------------------------
        # 3. Acquisitions are sampled FIRST, because scan drift is a
        #    geometric shear that must enter the ground-truth bookkeeping.
        # ---------------------------------------------------------------
        ref_acq    = sample_acquisition(tier.sem, noise_level, rng)
        search_acq = sample_acquisition(tier.sem, noise_level, rng)

        ref_rng    = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        search_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))

        ref_h = int(rng.integers(REF_MIN_SIZE, REF_MAX_SIZE + 1))
        ref_w = int(rng.integers(REF_MIN_SIZE, REF_MAX_SIZE + 1))
        pad = REF_RENDER_PAD
        crop_w, crop_h = ref_w + 2 * pad, ref_h + 2 * pad

        # ---------------------------------------------------------------
        # 4. Choose ground truth FIRST, then invert the FULL forward chain
        #
        # Scan drift (sem_effects.apply_drift_distortion, axis=0) implements
        #     dst(x, y) = src(x - d*y, y)
        # so a feature at src (a, b) appears in dst at (a + d*b, b): a pure
        # horizontal SHEAR of magnitude d*y.  With d up to 0.11 and y up to
        # 1000 this displaces content by >100 px, so it CANNOT be ignored
        # when labelling.  (This was a real ground-truth bug: the label was
        # computed pre-drift while the pixels were rendered post-drift.)
        #
        # Forward chain, from the scene point placed at the reference crop
        # centre through to the final search-image coordinate:
        #
        #   (1) ref drift shear inside the reference crop
        #   (2) affine M (rotation + scale) applied to the scene
        #   (3) crop offset (ox, oy)
        #   (4) search drift shear across the full 1000-row raster
        #
        # We sample the desired label and invert (4)->(1) to recover the
        # scene coordinate the reference must be cropped from.
        # ---------------------------------------------------------------
        margin = int(max(ref_h, ref_w) * 0.75) + 10
        gt_x = float(rng.uniform(margin, SEARCH_W - margin))
        gt_y = float(rng.uniform(margin, SEARCH_H - margin))

        if g.enable_subpixel_offset:
            sub_dx = float(rng.uniform(-0.5, 0.5))
            sub_dy = float(rng.uniform(-0.5, 0.5))
        else:
            sub_dx = sub_dy = 0.0
        gt_x += sub_dx
        gt_y += sub_dy

        # (4)^-1  undo the search-image drift shear
        pre_y = gt_y
        pre_x = gt_x - search_acq.drift_rate * pre_y

        # (3)^-1  search coords -> warped scene coords
        warp_x, warp_y = pre_x + ox, pre_y + oy

        # (2)^-1  warped scene -> unwarped scene
        scene_pt_x, scene_pt_y = transform_point(M_inv, warp_x, warp_y)

        # (1)^-1  undo the reference-crop drift shear.  The content that ends
        # up at the reference patch centre originates, pre-drift, at
        # x = crop_w/2 - d_ref * crop_h/2.
        scene_x = scene_pt_x + ref_acq.drift_rate * (crop_h / 2.0)
        scene_y = scene_pt_y

        # ---------------------------------------------------------------
        # 5. Reference: crop from the UNWARPED clean scene, sub-pixel exact
        # ---------------------------------------------------------------
        tx = crop_w / 2.0 - scene_x
        ty = crop_h / 2.0 - scene_y
        T = np.float32([[1.0, 0.0, tx],
                        [0.0, 1.0, ty]])
        clean_ref_padded = cv2.warpAffine(
            clean_scene, T, (crop_w, crop_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101)

        # Render the reference WITH padding, then centre-crop, so border
        # blur/edge statistics match those inside the search image.
        ref_rendered = render_sem_image(clean_ref_padded, ref_acq, ref_rng)
        ref_image = ref_rendered[pad:pad + ref_h, pad:pad + ref_w].copy()

        search_image = render_sem_image(clean_search, search_acq, search_rng)

        # ---------------------------------------------------------------
        # 6. Metadata
        # ---------------------------------------------------------------
        meta = PairMeta(
            pair_id           = pair_id,
            tier              = tier.name,
            structure_family  = sinfo["family"],
            structure_variant = sinfo["variant"],
            noise_level       = noise_level,
            gt_x              = round(gt_x, 4),
            gt_y              = round(gt_y, 4),
            ref_h             = ref_h,
            ref_w             = ref_w,
            period_x          = round(float(sinfo["period_x"]) * scale, 4),
            period_y          = round(float(sinfo["period_y"]) * scale, 4),
            n_defects         = int(sinfo["n_defects"]),
            rotation_deg      = round(rot, 4),
            scale             = round(scale, 6),
            subpixel_dx       = round(sub_dx, 4),
            subpixel_dy       = round(sub_dy, 4),
            ref_acq           = ref_acq.to_dict(),
            search_acq        = search_acq.to_dict(),
            seed              = pair_seed,
        )

        return ref_image, search_image, meta


# ===========================================================================
# Verification helper (used by unit tests and dataset QC)
# ===========================================================================

def verify_pair_geometry(ref: np.ndarray,
                          search: np.ndarray,
                          meta: PairMeta,
                          tolerance_px: float = 3.0,
                          search_radius_px: int = 8) -> Tuple[bool, float]:
    """
    Independently verify that the declared ground truth is truthful.

    This is a *self-audit of the data engine*.  It must answer exactly one
    question — "is the label correct?" — without being contaminated by the two
    phenomena the dataset is designed to contain:

      1. PERIODIC AMBIGUITY.  A wide correlation window will happily lock onto
         an alias peak one period away, which would make a perfectly correct
         label look wrong.  We therefore restrict the correlation search to
         +/- `search_radius_px` (default 8 px), which is smaller than half the
         smallest structural period in the dataset (~9 px fins), so no alias
         peak can enter the window.

      2. ROTATION / SCALE.  The reference is cropped from the unwarped scene
         while the search image is warped, so at 1.5-5 degrees of rotation the
         patches genuinely do not align.  We first UNDO the known warp about
         the ground-truth point, using the same rotation and scale recorded in
         the manifest, and only then correlate.

    With both corrections the audit is valid for every tier, including
    'ambiguous' and 'hard'.

    Returns
    -------
    (ok, offset_px)  — offset of the local correlation peak from the label.
    """
    rh, rw = ref.shape[:2]
    sh, sw = search.shape[:2]
    cx, cy = meta.gt_x, meta.gt_y

    # --- Crop a generous window around the label ---
    half = int(max(rh, rw) * 0.75) + search_radius_px + 8
    x0 = int(np.clip(round(cx) - half, 0, sw - 1))
    y0 = int(np.clip(round(cy) - half, 0, sh - 1))
    x1 = int(np.clip(round(cx) + half, 1, sw))
    y1 = int(np.clip(round(cy) + half, 1, sh))
    patch = search[y0:y1, x0:x1].astype(np.float32)
    if patch.shape[0] < rh + 2 or patch.shape[1] < rw + 2:
        return False, float("inf")

    # --- Undo the known rotation/scale about the label position ---
    lx, ly = cx - x0, cy - y0            # label in patch coords
    if abs(meta.rotation_deg) > 1e-6 or abs(meta.scale - 1.0) > 1e-9:
        M = cv2.getRotationMatrix2D((lx, ly),
                                     -meta.rotation_deg,
                                     1.0 / meta.scale)
        patch = cv2.warpAffine(patch, M,
                                (patch.shape[1], patch.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)

    # --- Tight correlation window: aliases cannot enter ---
    r = search_radius_px
    tx0 = int(round(lx - rw / 2.0)) - r
    ty0 = int(round(ly - rh / 2.0)) - r
    tx1 = tx0 + rw + 2 * r
    ty1 = ty0 + rh + 2 * r
    if tx0 < 0 or ty0 < 0 or tx1 > patch.shape[1] or ty1 > patch.shape[0]:
        return False, float("inf")

    region = patch[ty0:ty1, tx0:tx1]
    score = cv2.matchTemplate(region, ref.astype(np.float32),
                               cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(score)

    found_cx = tx0 + loc[0] + rw / 2.0
    found_cy = ty0 + loc[1] + rh / 2.0
    offset = float(np.hypot(found_cx - lx, found_cy - ly))
    return offset <= tolerance_px, offset


# ===========================================================================
# Disk I/O
# ===========================================================================

def save_pair(ref_image: np.ndarray,
              search_image: np.ndarray,
              meta: PairMeta,
              output_dir: Path) -> None:
    """
    Write the pair as PNGs.  The manifest is written ONCE by the dataset
    generator (v1 rewrote the whole manifest per pair — O(n^2) I/O).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def to_uint8(img: np.ndarray) -> np.ndarray:
        return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)

    pid = meta.pair_id
    cv2.imwrite(str(output_dir / f"pair_{pid:04d}_ref.png"), to_uint8(ref_image))
    cv2.imwrite(str(output_dir / f"pair_{pid:04d}_search.png"), to_uint8(search_image))


def write_manifest(metas, output_dir: Path, extra: dict | None = None) -> Path:
    """Write the full manifest in a single pass."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pairs": [asdict(m) for m in metas],
    }
    if extra:
        payload.update(extra)
    path = output_dir / "manifest.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_manifest(dataset_dir: Path) -> dict:
    with open(Path(dataset_dir) / "manifest.json") as f:
        return json.load(f)
