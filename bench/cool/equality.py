"""Numeric equality between two benchmark arms (gate: arm-pip vs arm-cool, same instance type).

A speed-up only counts if the answers are the same. The checks:

1. **Offsets** (gate). The ``e2e_offsets.csv`` files of both arms, joined on (mode, DEV pair).
   The max absolute offset difference in x and y must be <= ``--tol-px`` (default 1e-3 px)
   and no pair may be missing.
2. **Agent estimates** (gate when present). The ``agent_offsets.csv`` files, joined on episode.
   The final drift estimates must agree within ``tol_px * PIXEL_NM`` nm (1 nm/px in the agent
   operating point) and the outcomes (commit/escalate) must be identical.
3. **Score maps** (reported, not gated). ``equality dump`` runs the engine's S0-S4 stage
   functions (``ingest``, ``detect_lattice``, ``residual``, ``score_residual`` from
   ``driftsense.v2``) on one DEV pair per tier and saves the TM_CCOEFF_NORMED score map,
   the search roughness residual and the lattice vectors to ``<arm>/scoremaps/*.npy`` with
   SHA-256 hashes. ``compare`` reports the max abs difference per map and whether the hashes
   match. KleidiCV may round differently from the generic path by design, so this is
   informational. The offset gate is what counts.

Usage::

    python -m bench.cool.equality dump --out-dir out/run/arm-cool
    python -m bench.cool.equality compare --a out/run/arm-pip --b out/run/arm-cool --out eq.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from bench.cool import common as C

DEFAULT_TOL_PX = 1e-3
DUMP_PAIRS = (("clean", 0), ("nominal", 0), ("hard", 0), ("ambiguous", 0))


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def stage_outputs(tier: str, pair_id: int) -> dict[str, np.ndarray]:
    """S0-S4 intermediates of one DEV pair via the engine's own stage functions (native frame)."""
    from bench.cool.pairs import load_pair
    from driftsense.v2.engine import _CASCADE_POOL
    from driftsense.v2.ingest import ingest
    from driftsense.v2.lattice import detect_lattice
    from driftsense.v2.roughness import residual
    from driftsense.v2.score import score_residual

    p = load_pair(tier, pair_id)
    t, s = ingest(p.template_png).image, ingest(p.search_png).image
    lat = detect_lattice(s)
    rt, rs = residual(t, lat), residual(s, lat)
    sr = score_residual(rt, rs, lat, max_candidates=_CASCADE_POOL)
    tag = f"{tier}_{pair_id:04d}"
    return {
        f"{tag}_lattice_k1k2": np.array([*lat.k1, *lat.k2], dtype=np.float64),
        f"{tag}_search_residual_f32": rs.residual,
        f"{tag}_score_map_f32": sr.score_map,
    }


def dump(out_dir: Path, pairs=DUMP_PAIRS) -> dict:
    C.set_cv_threads(1)
    d = Path(out_dir) / "scoremaps"
    d.mkdir(parents=True, exist_ok=True)
    index = {}
    for tier, pid in pairs:
        for name, arr in stage_outputs(tier, pid).items():
            np.save(d / f"{name}.npy", arr, allow_pickle=False)
            index[name] = {"sha256": _sha(arr), "shape": list(arr.shape), "dtype": str(arr.dtype)}
    C.write_json(d / "hashes.json", {"pairs": [f"{t}/{i:04d}" for t, i in pairs], "maps": index})
    return index


def _csv(arm: Path, name: str) -> Path:
    return Path(arm) / name if Path(arm).is_dir() else Path(arm)


def _read_offsets(arm: Path) -> dict[tuple[str, str], tuple[float, float, int]]:
    rows = C.read_csv(_csv(arm, "e2e_offsets.csv"))
    return {(r["mode"], r["pair"]): (float(r["x_px"]), float(r["y_px"]), int(r["top1_hit"])) for r in rows}


def _read_agent(arm: Path) -> dict[str, tuple[str, float | None, float | None]] | None:
    f = Path(arm) / "agent_offsets.csv"
    if not f.is_file():
        return None

    def num(v: str) -> float | None:
        return float(v) if v not in ("", None) else None

    return {r["episode"]: (r["outcome"], num(r["est_x_nm"]), num(r["est_y_nm"])) for r in C.read_csv(f)}


def compare_offsets(oa: dict, ob: dict, tol_px: float) -> dict:
    common = sorted(set(oa) & set(ob))
    missing = sorted(f"{m}:{p}" for m, p in set(oa) ^ set(ob))
    diffs = [max(abs(oa[k][0] - ob[k][0]), abs(oa[k][1] - ob[k][1])) for k in common]
    max_diff = max(diffs) if diffs else math.inf
    agree = sum(oa[k][2] == ob[k][2] for k in common)
    return {
        "n": len(common),
        "missing": missing,
        "max_abs_diff_px": max_diff,
        "tol_px": tol_px,
        "pass": bool(common) and not missing and max_diff <= tol_px,
        "n_exceeding": int(sum(d > tol_px for d in diffs)),
        "top1_agreement": agree / len(common) if common else 0.0,
    }


def compare_agent(aa: dict, ab: dict, tol_nm: float) -> dict:
    common = sorted(set(aa) & set(ab))
    missing = sorted(set(aa) ^ set(ab))
    same_outcome = sum(aa[k][0] == ab[k][0] for k in common)
    diffs = []
    for k in common:
        xa, ya, xb, yb = aa[k][1], aa[k][2], ab[k][1], ab[k][2]
        if None in (xa, ya, xb, yb):
            diffs.append(0.0 if (xa is None) == (xb is None) else math.inf)
        else:
            diffs.append(max(abs(xa - xb), abs(ya - yb)))
    max_diff = max(diffs) if diffs else math.inf
    return {
        "n": len(common),
        "missing": missing,
        "max_abs_diff_nm": max_diff,
        "tol_nm": tol_nm,
        "outcome_agreement": same_outcome / len(common) if common else 0.0,
        "pass": bool(common) and not missing and same_outcome == len(common) and max_diff <= tol_nm,
    }


def compare(a: Path, b: Path, tol_px: float = DEFAULT_TOL_PX) -> dict:
    """Compare arm dirs (or offset CSVs) ``a`` and ``b``. Returns a plain dict summary."""
    from bench.cool.agent_bench import PIXEL_NM

    res: dict = {"offset": compare_offsets(_read_offsets(a), _read_offsets(b), tol_px), "scoremaps": {}}
    aa, ab = _read_agent(a), _read_agent(b)
    if aa is not None and ab is not None:
        res["agent"] = compare_agent(aa, ab, tol_px * PIXEL_NM)
    da, db = Path(a) / "scoremaps", Path(b) / "scoremaps"
    if da.is_dir() and db.is_dir():
        ha = json.loads((da / "hashes.json").read_text(encoding="utf-8"))
        hb = json.loads((db / "hashes.json").read_text(encoding="utf-8"))
        for name in sorted(set(ha["maps"]) & set(hb["maps"])):
            xa, xb = np.load(da / f"{name}.npy"), np.load(db / f"{name}.npy")
            entry: dict = {"sha256_equal": ha["maps"][name]["sha256"] == hb["maps"][name]["sha256"]}
            if xa.shape == xb.shape:
                entry["max_abs_diff"] = float(np.max(np.abs(xa.astype(np.float64) - xb.astype(np.float64))))
            else:
                entry["max_abs_diff"] = None
                entry["shape_mismatch"] = [list(xa.shape), list(xb.shape)]
            res["scoremaps"][name] = entry
    res["pass"] = res["offset"]["pass"] and res.get("agent", {"pass": True})["pass"]
    return res


def _finite(v: float) -> float:
    return v if math.isfinite(v) else -1.0


def to_metrics(res: dict) -> dict:
    o = res["offset"]
    m = {
        "equality.offset.max_abs_diff_px": C.metric(_finite(o["max_abs_diff_px"]), "px", n=o["n"]),
        "equality.offset.tol_px": C.metric(o["tol_px"], "px"),
        "equality.offset.pass": C.metric(1.0 if o["pass"] else 0.0, "bool", n=o["n"]),
        "equality.offset.n_exceeding": C.metric(o["n_exceeding"], "pairs", n=o["n"]),
        "equality.offset.n_missing": C.metric(len(o["missing"]), "pairs"),
        "equality.offset.top1_agreement": C.metric(o["top1_agreement"], "fraction", n=o["n"]),
        "equality.pass": C.metric(1.0 if res["pass"] else 0.0, "bool"),
    }
    if "agent" in res:
        g = res["agent"]
        m["equality.agent.max_abs_diff_nm"] = C.metric(_finite(g["max_abs_diff_nm"]), "nm", n=g["n"])
        m["equality.agent.outcome_agreement"] = C.metric(g["outcome_agreement"], "fraction", n=g["n"])
        m["equality.agent.pass"] = C.metric(1.0 if g["pass"] else 0.0, "bool", n=g["n"])
    for name, e in res["scoremaps"].items():
        m[f"equality.scoremap.{name}.sha256_equal"] = C.metric(1.0 if e["sha256_equal"] else 0.0, "bool")
        if e.get("max_abs_diff") is not None:
            m[f"equality.scoremap.{name}.max_abs_diff"] = C.metric(e["max_abs_diff"], "abs")
    return m


def env_of(arm: Path):
    """EnvInfo recorded by an arm's results (the comparison is attributed to the arm, not this host)."""
    for f in ("e2e.json", "microbench.json", "stagebench.json", "agent.json"):
        p = Path(arm) / f
        if p.is_file():
            try:
                return C.load_results(p).env
            except (ValueError, OSError) as e:  # pydantic ValidationError is a ValueError
                print(f"[equality] ignoring unreadable {p}: {e}", file=sys.stderr)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="write S0-S4 score maps + hashes into <out-dir>/scoremaps")
    d.add_argument("--out-dir", type=Path, required=True)
    d.add_argument("--quick", action="store_true", help="accepted for symmetry; the dump is always the same")
    c = sub.add_parser("compare", help="compare two arms; exit 1 if the gate fails")
    c.add_argument("--a", type=Path, required=True, help="baseline arm dir")
    c.add_argument("--b", type=Path, required=True, help="candidate arm dir")
    c.add_argument("--tol-px", type=float, default=DEFAULT_TOL_PX)
    c.add_argument("--out", type=Path, default=None, help="write a ResultsFile JSON here")
    a = ap.parse_args(argv)

    if a.cmd == "dump":
        idx = dump(a.out_dir)
        print(f"[equality] wrote {len(idx)} arrays to {a.out_dir / 'scoremaps'}")
        return 0

    res = compare(a.a, a.b, a.tol_px)
    o = res["offset"]
    print(
        f"[equality] offsets {a.a} vs {a.b}: n={o['n']} max|diff|={o['max_abs_diff_px']:.3g} px "
        f"(tol {o['tol_px']:g}) missing={o['missing']} -> {'PASS' if o['pass'] else 'FAIL'}"
    )
    if "agent" in res:
        g = res["agent"]
        print(
            f"[equality] agent: n={g['n']} max|diff|={g['max_abs_diff_nm']:.3g} nm outcome agreement "
            f"{g['outcome_agreement']:.0%} -> {'PASS' if g['pass'] else 'FAIL'}"
        )
    for name, e in res["scoremaps"].items():
        print(f"[equality]   {name}: max|diff|={e.get('max_abs_diff')} sha256_equal={e['sha256_equal']}")
    if a.out:
        C.write_results(
            a.out,
            f"equality.{Path(a.a).name}_vs_{Path(a.b).name}",
            to_metrics(res),
            C.produced_by(__file__),
            env=env_of(a.b),
        )
        print(f"[equality] wrote {a.out}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
