"""NumPy-2 compatibility shim for the frozen v1 reference engine.

The v1 arm (localization/*, tag v1-munichtech) is frozen golden reference and
must not change. It calls ``np.cross`` on 2-element vectors to get the scalar
z-component of a planar cross product (e.g. spectral.py's non-collinearity test).
NumPy deprecated that form in 2.0 and *removed* it in 2.x: on the pinned
``numpy==2.5.4`` (see requirements.lock) ``np.cross`` on 2-vectors raises
``ValueError``, which crashes the v1 baseline arm in the sealed-run harness.

Current (non-frozen) engine code never uses this form — it uses explicit 2-D
determinants (driftsense/v2/lattice.py, localization/lattice_fit._cross2). So
restoring legacy 2-vector behaviour is invisible to everything except frozen v1.

``enable_legacy_cross2()`` is idempotent and wraps ``numpy.cross`` so that
2-vector inputs compute the old scalar result while all other shapes delegate
unchanged to the real implementation.
"""

from __future__ import annotations

import numpy as np

_PATCHED_FLAG = "_driftsense_legacy_cross2"


def enable_legacy_cross2() -> None:
    """Restore pre-NumPy-2 ``np.cross`` behaviour for 2-D vectors (idempotent)."""
    current = getattr(np, "cross", None)
    if current is None or getattr(current, _PATCHED_FLAG, False):
        return
    _real_cross = current

    def cross(a, b, *args, **kwargs):
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        # Only intercept the removed 2-vector form (planar scalar cross product);
        # everything else goes to the real np.cross untouched.
        if not args and not kwargs and a_arr.shape[-1] == 2 and b_arr.shape[-1] == 2:
            return a_arr[..., 0] * b_arr[..., 1] - a_arr[..., 1] * b_arr[..., 0]
        return _real_cross(a, b, *args, **kwargs)

    setattr(cross, _PATCHED_FLAG, True)
    np.cross = cross  # type: ignore[assignment]
