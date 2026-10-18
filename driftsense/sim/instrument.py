"""
VirtualSEM — a stateful digital-twin SEM for the alignment agent.

WHY A TWIN (and not just the data engine)
------------------------------------------
``data_gen`` produces independent (reference, search) *pairs*: every call draws
a fresh scene, so two acquisitions never see the same physical wafer.  An
alignment agent, by contrast, drives a *single instrument* over *one* wafer and
must cope with the fact that:

  * the same site, re-imaged, shows the SAME line-edge roughness (LER) — it is a
    permanent property of the etched silicon, not per-scan noise.  This is the
    positional fingerprint the agent exploits;
  * the stage does not go exactly where it is told (bias + thermal random walk +
    per-move positioning error), so the agent must *localize* rather than trust
    the commanded coordinate;
  * every acquisition costs electron dose and the specimen/budget is finite.

VirtualSEM models exactly this.  It generates ONE clean wafer site at
construction (reusing the frozen ``data_gen`` physics) and keeps it for the
instrument's lifetime, so repeated ``acquire`` calls of a site return images
whose underlying structure — including LER — is identical, differing only in the
independent scan noise, the stage error at that moment, and any scan rotation.

GROUND TRUTH IS EVALUATOR-ONLY
------------------------------
The true stage error is what the agent is being asked to recover.  It is exposed
*only* through :meth:`VirtualSEM.verify`; it never appears in the ``(png_bytes,
meta)`` returned by :meth:`acquire`.  An agent that only ever calls ``acquire``
has no path to the answer.

All physics (structure synthesis, SEM acquisition chain) is imported READ-ONLY
from the frozen ``config`` / ``data_gen`` modules; this file adds only the
stateful instrument wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from config import AperiodicityConfig
from data_gen.sem_effects import AcquisitionParams, render_sem_image
from data_gen.structures import generate_structure

_VALID_FAMILIES = ("dram", "finfet", "random")

# Distinct, fixed stream labels so the wafer, the stage physics and the
# per-acquisition noise are drawn from independent — but fully reproducible —
# random streams given the instrument seed.
_STREAM_WAFER = 0x7AFE12
_STREAM_STAGE = 0x57A6E0


@dataclass(frozen=True)
class DriftModel:
    """
    Stage positioning-error model, all magnitudes in nanometres.

    bias       : constant systematic offset present from power-on.
    walk_sigma : per-acquisition thermal random-walk increment (RMS per axis).
    move_base  : positioning error incurred by any commanded stage move (RMS).
    move_frac  : additional move error proportional to the move distance.
    """

    bias_sigma: float = 6.0
    walk_sigma: float = 1.5
    move_base: float = 3.0
    move_frac: float = 0.01


class DoseError(RuntimeError):
    """Raised when an acquisition would exceed the electron-dose budget."""


class AcquisitionLimitError(RuntimeError):
    """Raised when the maximum number of acquisitions has been reached."""


class VirtualSEM:
    """
    A deterministic, stateful virtual SEM over a single synthetic wafer site.

    Parameters
    ----------
    seed          : master seed; the instrument is fully reproducible from it.
    family        : structure family — 'dram', 'finfet' or 'random'.
    site_size_nm  : physical extent of the (square) wafer site.
    pixel_nm      : nanometres per pixel of the detector grid.
    dose_per_us   : mean electrons per pixel delivered per microsecond of dwell
                    (ties shot noise to dwell: noise RMS scales as 1/sqrt(dwell)).
    max_dose      : cumulative electron-dose cap (electrons); None disables it.
    max_acquisitions : cap on the number of acquisitions; None disables it.
    drift         : stage :class:`DriftModel`; a default realistic model if None.
    """

    def __init__(
        self,
        seed: int,
        family: str = "dram",
        site_size_nm: float = 2000.0,
        pixel_nm: float = 1.0,
        *,
        dose_per_us: float = 50.0,
        max_dose: float | None = None,
        max_acquisitions: int | None = None,
        drift: DriftModel | None = None,
    ) -> None:
        if family not in _VALID_FAMILIES:
            raise ValueError(f"family must be one of {_VALID_FAMILIES}, got {family!r}")
        if site_size_nm <= 0 or pixel_nm <= 0:
            raise ValueError("site_size_nm and pixel_nm must be positive")
        if dose_per_us <= 0:
            raise ValueError("dose_per_us must be positive")

        self.seed = int(seed)
        self.family = family
        self.site_size_nm = float(site_size_nm)
        self.pixel_nm = float(pixel_nm)
        self.dose_per_us = float(dose_per_us)
        self.max_dose = max_dose
        self.max_acquisitions = max_acquisitions
        self._drift = drift if drift is not None else DriftModel()

        self._site_px = max(16, int(round(site_size_nm / pixel_nm)))

        # --- One permanent clean wafer site (persistent LER = the twin) ---
        wafer_rng = np.random.default_rng([self.seed, _STREAM_WAFER])
        resolved = family
        if resolved == "random":
            resolved = str(wafer_rng.choice(["dram", "finfet"]))
        self._resolved_family = resolved
        wafer, self._wafer_info = generate_structure(
            resolved, self._site_px, self._site_px, AperiodicityConfig(), wafer_rng
        )
        self._wafer = wafer.astype(np.float32)

        # --- Stage physics: one reproducible stream for bias/walk/move ---
        self._stage_rng = np.random.default_rng([self.seed, _STREAM_STAGE])
        self._bias_nm = self._stage_rng.normal(0.0, self._drift.bias_sigma, size=2)
        self._walk_nm = np.zeros(2, dtype=np.float64)

        # --- Counters / evaluator-only ground truth ---
        self._n_acq = 0
        self._total_dose = 0.0
        self._last_error_nm: np.ndarray | None = None

    # -----------------------------------------------------------------
    # Introspection (agent-visible; contains no ground truth)
    # -----------------------------------------------------------------
    @property
    def total_dose(self) -> float:
        """Cumulative electron dose delivered so far (electrons)."""
        return self._total_dose

    @property
    def n_acquisitions(self) -> int:
        """Number of acquisitions performed so far."""
        return self._n_acq

    def _nm_to_px(self, v_nm: float) -> float:
        """Map a nanometre coordinate (origin = site centre) to pixel coords."""
        return self._site_px / 2.0 + v_nm / self.pixel_nm

    # -----------------------------------------------------------------
    # Stage control
    # -----------------------------------------------------------------
    def commit_stage_move(self, dx_nm: float, dy_nm: float) -> None:
        """
        Command a relative stage move, incurring a positioning error.

        The move itself is assumed ideal (the agent tracks its own commanded
        position); what the instrument adds is the *error* of the move, which
        grows with the move distance and permanently shifts the stage's true
        offset from the commanded frame.
        """
        dist = float(np.hypot(dx_nm, dy_nm))
        sigma = self._drift.move_base + self._drift.move_frac * dist
        self._walk_nm += self._stage_rng.normal(0.0, sigma, size=2)

    def verify(self) -> dict[str, float]:
        """
        Return the TRUE stage error of the most recent acquisition — for the
        evaluator only.

        This is deliberately *not* reachable from :meth:`acquire`'s return value.
        An agent must recover this error itself (e.g. by localizing against a
        reference); ``verify`` exists so a harness can grade that estimate.
        """
        if self._last_error_nm is None:
            raise RuntimeError("verify() called before any acquire()")
        return {
            "error_x_nm": float(self._last_error_nm[0]),
            "error_y_nm": float(self._last_error_nm[1]),
            "error_mag_nm": float(np.hypot(*self._last_error_nm)),
        }

    # -----------------------------------------------------------------
    # Acquisition
    # -----------------------------------------------------------------
    def acquire(
        self,
        center_nm: tuple[float, float],
        fov_nm: float,
        dwell_us: float,
        scan_rotation_deg: float = 0.0,
    ) -> tuple[bytes, dict[str, Any]]:
        """
        Image the site around ``center_nm`` and return ``(png_bytes, meta)``.

        The image is centred not at the commanded coordinate but at
        ``commanded + stage_error`` (bias + accumulated random walk), so the
        agent observes the mis-registration it must solve.  Shot noise scales as
        ``1/sqrt(dwell_us)``; a small rotation is applied via ``warpPerspective``.
        """
        if fov_nm <= 0:
            raise ValueError("fov_nm must be positive")
        if dwell_us <= 0:
            raise ValueError("dwell_us must be positive")
        if self.max_acquisitions is not None and self._n_acq >= self.max_acquisitions:
            raise AcquisitionLimitError(f"acquisition limit reached ({self.max_acquisitions})")

        fov_px = max(8, int(round(fov_nm / self.pixel_nm)))
        electrons_per_pixel = self.dose_per_us * dwell_us
        dose_this = electrons_per_pixel * (fov_px * fov_px)
        if self.max_dose is not None and self._total_dose + dose_this > self.max_dose:
            raise DoseError(
                f"dose budget exceeded: {self._total_dose + dose_this:.0f} > {self.max_dose:.0f} electrons"
            )

        # --- Advance thermal drift one time-step, then read true stage error ---
        self._walk_nm += self._stage_rng.normal(0.0, self._drift.walk_sigma, size=2)
        error_nm = self._bias_nm + self._walk_nm
        true_cx = center_nm[0] + error_nm[0]
        true_cy = center_nm[1] + error_nm[1]

        # --- Crop clean content at the (sub-pixel) true centre, with a margin
        #     so the rotation does not pull in replicated border pixels. ---
        pad = int(fov_px * 0.15) + 8
        crop = fov_px + 2 * pad
        center_px = (self._nm_to_px(true_cx), self._nm_to_px(true_cy))
        patch = cv2.getRectSubPix(self._wafer, (crop, crop), center_px)

        # --- Small scan rotation about the patch centre via warpPerspective ---
        if abs(scan_rotation_deg) > 1e-9:
            m23 = cv2.getRotationMatrix2D((crop / 2.0, crop / 2.0), scan_rotation_deg, 1.0)
            h = np.vstack([m23, [0.0, 0.0, 1.0]]).astype(np.float32)
            patch = cv2.warpPerspective(
                patch,
                h,
                (crop, crop),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        clean = patch[pad : pad + fov_px, pad : pad + fov_px]

        # --- Full SEM acquisition chain (frozen physics), dwell-driven dose ---
        params = AcquisitionParams(
            blur_sigma=0.9,
            edge_strength=0.30,
            electrons_per_pixel=electrons_per_pixel,
            drift_rate=0.0,
            read_noise_sigma=0.008,
            scanline_jitter_sigma=0.004,
            charging_amplitude=0.0,
            vignette_strength=0.0,
            vignette_cx=0.5,
            vignette_cy=0.5,
            brightness_offset=0.0,
            contrast_gain=1.0,
            gamma=1.0,
        )
        acq_rng = np.random.default_rng([self.seed, self._n_acq])
        rendered = render_sem_image(clean, params, acq_rng)

        img_u8 = np.clip(rendered * 255.0 + 0.5, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".png", img_u8)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        png_bytes = buf.tobytes()

        # --- Commit state AFTER a successful acquisition ---
        self._total_dose += dose_this
        self._last_error_nm = error_nm.copy()
        meta = {
            "acq_index": self._n_acq,
            "requested_center_nm": (float(center_nm[0]), float(center_nm[1])),
            "fov_nm": float(fov_nm),
            "dwell_us": float(dwell_us),
            "scan_rotation_deg": float(scan_rotation_deg),
            "pixel_nm": self.pixel_nm,
            "width_px": fov_px,
            "height_px": fov_px,
            "electrons_per_pixel": electrons_per_pixel,
            "dose_this_acq": dose_this,
            "cumulative_dose": self._total_dose,
            "family": self._resolved_family,
        }
        self._n_acq += 1
        return png_bytes, meta
