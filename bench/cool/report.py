"""Cross-arm COOL report: speed-ups with bootstrap CIs, equality, Amdahl bound, cost per 1,000.

Inputs are arm directories written by ``run_all.sh`` (``<BENCH_OUT_DIR>/<BENCH_ARM>/``),
given one by one with ``--arm`` or all at once with ``--run-dir`` (every sub-directory
holding results). Arm labels default to the directory name: ``x86-pip``, ``arm-pip``,
``arm-cool`` or ``arm-cool-large``. Use ``label=path`` to override.

Speed-up = median(baseline) / median(candidate); > 1 means the candidate is faster. The
CI resamples both arms' raw samples independently (2000 draws, fixed seed). **If the CI
includes 1.0, the report says "no measurable difference"**, whatever the point estimate.

Cost per 1,000 localizations = (EC2 $/h + COOL $/h) / localizations per hour x 1000, from
the e2e process-pool throughput (all vCPUs, cv2 threads=1 per worker) and
``bench/cool/prices.json``. The same is computed per 1,000 agent episodes.

Outputs (``--out``, default ``results/bench/``):

* ``cool_bench.json``: ResultsFile, ``dataset="bench"``. ``env`` comes from the
  ``arm-cool`` arm (its ``env.cool`` is True only if cv2 was imported from /opt/cool).
* ``cool_bench_docs.json``: flat, docs-ready keys (numbers, strings, verdict phrases and
  evidence such as the AMI id and the KleidiCV lines of getBuildInformation()).
* ``cool_bench_summary.md``, ``cool_bench_meta.json`` and ``charts/*.png`` (matplotlib).

Smoke guard: when any arm ran ``--quick`` or off EC2 (no instance type), the ResultsFile
is named ``cool_bench.local_smoke``. The report then refuses to write into the
repository's ``results/bench/`` unless the output path contains ``local_smoke``.

Usage::

    python -m bench.cool.report --run-dir out/<run_id> --out results/bench/
    python -m bench.cool.report --arm out/r1/arm-pip --arm out/r1/arm-cool --out /tmp/report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bench.cool import common as C
from bench.cool import equality as EQ
from bench.cool import ops as O

PRICES = Path(__file__).with_name("prices.json")
ARMS = ("x86-pip", "arm-pip", "arm-cool", "arm-cool-large")
DEFAULT_COMPARISONS = [
    ("arm-cool", "arm-pip", "COOL effect on the same Graviton4 instance type (c8g.4xlarge)"),
    ("arm-pip", "x86-pip", "Graviton4 vs x86 with the same pip OpenCV"),
    ("arm-cool", "x86-pip", "COOL on Graviton4 vs pip on x86 (combined)"),
]
PRIMARY = ("arm-cool", "arm-cool-large", "arm-pip", "x86-pip")
GEO_GROUPS = (O.KLEIDICV, O.CONTROL, O.NUMPY, O.COUNTERFACTUAL)
RESULT_FILES = ("microbench", "census", "stagebench", "e2e", "agent")
NO_DIFF = "no measurable difference"
FASTER = "candidate faster"
SLOWER = "candidate slower"
MIN_N = 3
INSUFFICIENT = f"insufficient samples (n<{MIN_N})"
LOCAL_SMOKE = "local_smoke"


class SmokeGuardError(RuntimeError):
    """Raised when smoke-test output would land in the campaign results directory."""


@dataclass
class Arm:
    label: str
    path: Path
    samples: dict[str, np.ndarray] = field(default_factory=dict)  # comparable key -> samples (ms)
    groups: dict[str, str] = field(default_factory=dict)  # key -> group
    throughput: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # "e2e.<mode>" / "agent" -> per-trial rate
    results: dict = field(default_factory=dict)  # name -> ResultsFile
    evidence: dict = field(default_factory=dict)

    def _first(self, attr: str):
        for n in ("e2e", "microbench", "stagebench", "agent", "census"):
            if n in self.results:
                return getattr(self.results[n], attr)
        return None

    @property
    def env(self):
        return self._first("env")

    @property
    def git_sha(self) -> str | None:
        return self._first("git_sha")

    @property
    def cool(self) -> bool:
        e = self.env
        return bool((e is not None and e.cool) or self.evidence.get("cool_detected"))

    @property
    def instance_type(self) -> str | None:
        e = self.env
        return (e.instance_type if e is not None else None) or (self.evidence.get("imds") or {}).get(
            "instance_type"
        )

    @property
    def quick(self) -> bool:
        return any(
            (rf.metrics.get("meta.quick") is not None and rf.metrics["meta.quick"].value)
            for rf in self.results.values()
        )


def load_arm(label: str, path: Path) -> Arm:
    arm = Arm(label=label, path=Path(path))
    p = arm.path
    if not p.is_dir():
        raise FileNotFoundError(f"arm dir not found: {p}")
    for n in RESULT_FILES:
        f = p / f"{n}.json"
        if f.is_file():
            arm.results[n] = C.load_results(f)
    ev = p / "evidence" / "evidence.json"
    if ev.is_file():
        arm.evidence = json.loads(ev.read_text(encoding="utf-8"))
    acc: dict[str, list[float]] = defaultdict(list)

    def add(k: str, v: str, group: str) -> None:
        acc[k].append(float(v))
        arm.groups[k] = group

    for r in _rows(p / "microbench_samples.csv"):
        add(f"{r['key']}.{r['tag']}", r["ms"], r["group"])
    for r in _rows(p / "stagebench_samples.csv"):
        add(f"stage.{r['mode']}.{r['stage']}.{r['tag']}", r["ms"], "stage")
    for r in _rows(p / "e2e_samples.csv"):
        add(
            f"e2e.{r['mode']}.{'latency' if r['phase'] == 'latency' else 'loaded_latency'}.{r['tag']}",
            r["ms"],
            "e2e",
        )
    for r in _rows(p / "agent_samples.csv"):
        add(
            f"agent.{'latency' if r['phase'] == 'latency' else 'loaded_latency'}.{r['tag']}", r["ms"], "agent"
        )
    tp: dict[str, list[float]] = defaultdict(list)
    for r in _rows(p / "e2e_throughput.csv"):
        tp[f"e2e.{r['mode']}"].append(float(r["jobs_per_hour"]))
    for r in _rows(p / "agent_throughput.csv"):
        tp["agent"].append(float(r["episodes_per_hour"]))
    arm.throughput = {k: np.asarray(v, dtype=np.float64) for k, v in tp.items()}
    arm.samples = {k: np.asarray(v, dtype=np.float64) for k, v in acc.items()}
    return arm


def _rows(f: Path) -> list[dict[str, str]]:
    return C.read_csv(f) if f.is_file() else []


def verdict(lo: float | None, hi: float | None, n: int = MIN_N) -> str:
    if n < MIN_N:
        return INSUFFICIENT
    if C.ci_includes(lo, hi, 1.0):
        return NO_DIFF
    return FASTER if lo is not None and lo > 1.0 else SLOWER


def compare_arms(cand: Arm, base: Arm) -> tuple[dict, list[dict], dict]:
    """Per-key speed-ups, group geometric means and throughput ratios. (metrics, rows, geomeans)."""
    cmp_ = f"{cand.label}_vs_{base.label}"
    metrics: dict = {}
    rows: list[dict] = []
    draws_by_group: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    points_by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for k in sorted(set(cand.samples) & set(base.samples)):
        b, c = base.samples[k], cand.samples[k]
        ratio, draws = C.bootstrap_ratio_draws(b, c)
        lo, hi = C.percentile_ci(draws, ratio)
        group = cand.groups.get(k, base.groups.get(k, "other"))
        if k.startswith(
            ("op.", "stage.")
        ):  # op.<fn>.<variant>.<tag> -> speedup.<cmp>.<group>.<fn>.<variant>.<tag>
            mkey = f"speedup.{cmp_}.{group}.{k.split('.', 1)[1]}"
        else:  # e2e.<mode>.latency.<tag> / agent.latency.<tag>
            mkey = f"speedup.{cmp_}.{k}"
        n = int(min(b.size, c.size))
        metrics[mkey] = C.metric(ratio, "x", lo, hi, n)
        rows.append(
            {
                "key": k,
                "group": group,
                "base_ms": float(np.median(b)),
                "cand_ms": float(np.median(c)),
                "speedup": ratio,
                "ci_low": lo,
                "ci_high": hi,
                "verdict": verdict(lo, hi, n),
                "metric": mkey,
            }
        )
        if group in GEO_GROUPS:
            ttag = k.rsplit(".", 1)[-1]
            draws_by_group[(group, ttag)].append(np.log(draws))
            points_by_group[(group, ttag)].append(math.log(ratio))
    geo: dict = {}
    for (group, ttag), dl in sorted(draws_by_group.items()):
        point = float(np.exp(np.mean(points_by_group[(group, ttag)])))
        lo, hi = C.percentile_ci(np.exp(np.mean(np.vstack(dl), axis=0)), point)
        metrics[f"speedup.{cmp_}.{group}.geomean.{ttag}"] = C.metric(point, "x", lo, hi, len(dl))
        geo[f"{group}.{ttag}"] = {
            "speedup": point,
            "ci_low": lo,
            "ci_high": hi,
            "n_ops": len(dl),
            "verdict": verdict(lo, hi),
        }
    for tk in sorted(set(cand.throughput) & set(base.throughput)):
        ct, bt = cand.throughput[tk], base.throughput[tk]
        if not ct.size or not bt.size:
            continue
        ratio, lo, hi = C.bootstrap_ratio_ci(ct, bt)  # higher rate is better: cand / base
        n = int(min(ct.size, bt.size))
        mkey = f"speedup.{cmp_}.{tk}.throughput"
        metrics[mkey] = C.metric(ratio, "x", lo, hi, n)
        rows.append(
            {
                "key": f"{tk}.throughput",
                "group": "throughput",
                "base_ms": None,
                "cand_ms": None,
                "base_rate": float(np.median(bt)),
                "cand_rate": float(np.median(ct)),
                "speedup": ratio,
                "ci_low": lo,
                "ci_high": hi,
                "verdict": verdict(lo, hi, n),
                "metric": mkey,
            }
        )
    return metrics, rows, geo


def amdahl(cand: Arm, base: Arm, geo: dict) -> dict:
    """Predicted pipeline speed-up from per-group op speed-ups and the baseline's census shares.

    predicted = 1 / (f_kleidicv / s_kleidicv + f_control / s_control + f_rest / s_numpy), at t1.
    Compared with the measured ``stage.<mode>.total.t1`` speed-up. If the prediction and the
    measurement disagree badly, the op mix in the microbench does not explain the pipeline.
    """
    out: dict = {}
    cen = base.results.get("census")
    if cen is None:
        return out
    s = {g: geo.get(f"{g}.t1", {}).get("speedup") for g in (O.KLEIDICV, O.CONTROL, O.NUMPY)}
    if s[O.KLEIDICV] is None or s[O.CONTROL] is None:
        return out
    s_rest = s[O.NUMPY] or 1.0
    for mode in sorted({k.split(".")[1] for k in cen.metrics if k.startswith("census.")}):
        fk = cen.metrics.get(f"census.{mode}.share.kleidicv")
        fc = cen.metrics.get(f"census.{mode}.share.control")
        if fk is None or fc is None:
            continue
        rest = max(0.0, 1.0 - fk.value - fc.value)
        pred = 1.0 / (fk.value / s[O.KLEIDICV] + fc.value / s[O.CONTROL] + rest / s_rest)
        bound = 1.0 / (1.0 - fk.value) if fk.value < 1 else math.inf  # COOL infinitely fast on listed ops
        out[mode] = {
            "predicted": pred,
            "kleidicv_share": fk.value,
            "control_share": fc.value,
            "upper_bound_if_listed_ops_free": bound,
        }
    return out


# ------------------------------------------------------------------------------------- cost
def load_prices(path: Path = PRICES) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hourly_price(
    prices: dict, itype: str | None, cool: bool, overrides: dict[str, float]
) -> tuple[float | None, float | None, list[str]]:
    """(ec2 $/h, cool $/h, warnings). None when unknown/unverified and not overridden."""
    warns: list[str] = []
    if not itype:
        return None, None, ["instance type unknown (not on EC2?): pass --instance-type label=type"]
    ec2: float | None = overrides.get(itype)
    if ec2 is None:
        e = prices.get("ec2", {}).get(itype)
        if e is None:
            warns.append(f"no EC2 price for {itype}: pass --price {itype}=<usd/h>")
        elif e.get("usd_per_hour") is None or e.get("unverified"):
            warns.append(
                f"EC2 price for {itype} is unverified/null in prices.json: pass --price {itype}=<usd/h>"
            )
        else:
            ec2 = float(e["usd_per_hour"])
    fee: float | None = 0.0
    if cool:
        f = prices.get("cool_fee", {}).get(itype)
        fee = None if f is None or f.get("usd_per_hour") is None else float(f["usd_per_hour"])
        if fee is None:
            warns.append(f"no COOL fee for {itype} in prices.json")
    return ec2, fee, warns


def cost_per_1000(ec2_per_h: float, cool_per_h: float, per_hour: float) -> float:
    if per_hour <= 0:
        raise ValueError("throughput must be positive")
    return (ec2_per_h + cool_per_h) / per_hour * 1000.0


def arm_costs(
    arm: Arm, prices: dict, overrides: dict[str, float], itype_override: str | None
) -> tuple[dict, dict]:
    itype = itype_override or arm.instance_type
    ec2, fee, warns = hourly_price(prices, itype, arm.cool, overrides)
    info: dict = {
        "instance_type": itype,
        "cool": arm.cool,
        "ec2_usd_per_hour": ec2,
        "cool_usd_per_hour": fee,
        "warnings": warns,
        "per_1000": {},
        "rate_per_hour": {},
    }
    metrics: dict = {}
    if not arm.throughput:
        info["warnings"].append("no throughput samples (--no-throughput?)")
    if ec2 is not None and fee is not None:
        metrics[f"cost.{arm.label}.usd_per_hour"] = C.metric(ec2 + fee, "USD/h")
    for tk, tp in sorted(arm.throughput.items()):
        if not tp.size:
            continue
        med, lo, hi = C.bootstrap_median_ci(tp)
        unit, what = ("episodes/h", "episodes") if tk == "agent" else ("localizations/h", "localizations")
        info["rate_per_hour"][tk] = med
        metrics[f"arm.{arm.label}.{tk}.throughput.per_hour"] = C.metric(med, unit, lo, hi, tp.size)
        if ec2 is None or fee is None:
            continue
        cost = cost_per_1000(ec2, fee, med)
        info["per_1000"][tk] = cost
        metrics[f"cost.{arm.label}.{tk}.usd_per_1000_{what}"] = C.metric(
            cost, f"USD/1000 {what}", cost_per_1000(ec2, fee, hi), cost_per_1000(ec2, fee, lo), tp.size
        )
    return metrics, info


def arm_absolute(arm: Arm) -> dict:
    m: dict = {}
    for k, v in sorted(arm.samples.items()):
        med, lo, hi = C.bootstrap_median_ci(v)
        m[f"arm.{arm.label}.{k}.median_ms"] = C.metric(med, "ms", lo, hi, v.size)
        if arm.groups.get(k) in ("e2e", "agent", "stage"):
            m[f"arm.{arm.label}.{k}.p95_ms"] = C.metric(float(np.percentile(v, 95)), "ms", n=v.size)
    cen = arm.results.get("census")
    if cen is not None:
        for k, v in cen.metrics.items():
            if ".share." in k or k.endswith(".total_ms"):
                m[f"arm.{arm.label}.{k}"] = v
    return m


# ------------------------------------------------------------------------------------- charts
GROUP_STYLE = {  # reference categorical palette slots 1-4 in fixed order; marker shape is the 2nd encoding
    O.KLEIDICV: ("#2a78d6", "o", "KleidiCV-listed"),
    O.CONTROL: ("#eb6834", "s", "not listed (control)"),
    O.NUMPY: ("#1baf7a", "D", "pure NumPy control"),
    O.COUNTERFACTUAL: ("#eda100", "P", "counterfactual uint8 (not an engine call)"),
    "stage": ("#52514e", "^", "pipeline stage"),
    "e2e": ("#0b0b0b", "v", "end-to-end"),
    "agent": ("#0b0b0b", "X", "agent episode"),
}
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[report] matplotlib not installed: skipping charts", file=sys.stderr)
        return None
    return plt


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)


def make_charts(comparisons: dict[str, list[dict]], costs: dict[str, dict], out_dir: Path) -> list[str]:
    plt = _plt()
    if plt is None:
        return []
    written = []
    cdir = out_dir / "charts"
    cdir.mkdir(parents=True, exist_ok=True)
    order = {g: i for i, g in enumerate(GROUP_STYLE)}
    for cmp_name, rows in comparisons.items():
        by_tag: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["group"] == "throughput":
                continue
            by_tag[r["key"].rsplit(".", 1)[-1]].append(r)
        for ttag, rs in sorted(by_tag.items()):
            rs = sorted(rs, key=lambda r: (order.get(r["group"], 9), r["key"]), reverse=True)
            fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.2 * len(rs) + 1.4)), dpi=130)
            fig.patch.set_facecolor(SURFACE)
            _style(ax)
            ax.axvline(1.0, color=INK2, lw=1, zorder=1)
            seen = set()
            for y, r in enumerate(rs):
                col, mk, lab = GROUP_STYLE.get(r["group"], (INK2, "o", r["group"]))
                ax.plot(
                    [r["ci_low"], r["ci_high"]], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2
                )
                ax.plot(
                    [r["speedup"]],
                    [y],
                    marker=mk,
                    ms=6,
                    color=col,
                    mec=SURFACE,
                    mew=1.5,
                    zorder=3,
                    ls="none",
                    label=None if r["group"] in seen else lab,
                )
                seen.add(r["group"])
            ax.set_yticks(range(len(rs)))
            ax.set_yticklabels([r["key"].rsplit(".", 1)[0] for r in rs], fontsize=6.5, color=INK)
            ax.set_xscale("log")
            lo = min(min(r["ci_low"] for r in rs), 0.9)
            hi = max(max(r["ci_high"] for r in rs), 1.1)
            ticks = [
                t
                for t in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
                if lo / 1.2 <= t <= hi * 1.2
            ]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}x" for t in ticks], fontsize=7, color=INK2)
            ax.minorticks_off()
            ax.set_xlabel(
                "speed-up = baseline median / candidate median, 95% bootstrap CI (1.0 = no change)",
                fontsize=8,
                color=INK2,
            )
            ax.set_title(f"{cmp_name.replace('_vs_', ' vs ')} ({ttag})", fontsize=10, color=INK, loc="left")
            ax.legend(fontsize=7, frameon=False, loc="lower right")
            fig.tight_layout()
            f = cdir / f"speedup_{cmp_name}_{ttag}.png"
            fig.savefig(f, facecolor=fig.get_facecolor())
            plt.close(fig)
            written.append(str(f))
    # cost per 1,000 localizations, one bar per arm (one series -> no legend; values labelled directly)
    for tk in sorted({tk for c in costs.values() for tk in c["per_1000"]}):
        items = [(lab, c["per_1000"][tk]) for lab, c in costs.items() if tk in c["per_1000"]]
        if not items:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(items) + 1.4), dpi=130)
        fig.patch.set_facecolor(SURFACE)
        _style(ax)
        ys = range(len(items))
        ax.barh(list(ys), [v for _, v in items], color="#2a78d6", height=0.55, edgecolor=SURFACE, linewidth=2)
        for y, (_, v) in zip(ys, items, strict=True):
            ax.text(v, y, f"  ${v:.4f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([lab for lab, _ in items], fontsize=8, color=INK)
        what = "agent episodes" if tk == "agent" else f"localizations ({tk.split('.', 1)[1]} mode)"
        ax.set_title(
            f"USD per 1,000 {what}, ap-south-1 on-demand + COOL fee", fontsize=9, color=INK, loc="left"
        )
        ax.set_xlim(0, max(v for _, v in items) * 1.35)
        fig.tight_layout()
        f = cdir / f"cost_per_1000_{tk.replace('.', '_')}.png"
        fig.savefig(f, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(str(f))
    return written


# ------------------------------------------------------------------------------------- summary
def _fmt_ci(r: dict) -> str:
    return f"{r['speedup']:.3f}x [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def summary_md(
    arms: list[Arm],
    comps: dict[str, dict],
    costs: dict[str, dict],
    eq: dict[str, dict],
    warnings: list[str],
    smoke: bool,
) -> str:
    intro = (
        "Generated by `bench/cool/report.py`. Speed-up = baseline median / candidate median (> 1: "
        "candidate faster), 95% percentile-bootstrap CI (2000 resamples, seeded). A CI that includes 1.0 "
        f"is reported as **{NO_DIFF}**. Vendor claims are not our numbers; these are."
    )
    L = ["# COOL benchmark summary", ""]
    if smoke:
        L += [
            "> **LOCAL SMOKE / QUICK RUN: plumbing check only. These numbers are not results and are not "
            "COOL evidence.**",
            "",
        ]
    L += [
        intro,
        "",
        "## Arms",
        "",
        "| arm | instance | arch | cv2 | cool | cv2.__file__ | git sha |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        e = a.env
        L.append(
            f"| {a.label} | {a.instance_type} | {e.arch if e else '?'} | {e.opencv if e else '?'} | {a.cool} | "
            f"`{a.evidence.get('cv2_file', '?')}` | `{(a.git_sha or '?')[:12]}` |"
        )
    if warnings:
        L += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]
    for name, c in comps.items():
        L += [
            "",
            f"## {name}",
            "",
            f"_{c['description']}_",
            "",
            "### Group geometric means",
            "",
            "| group.threads | speed-up [95% CI] | n ops | verdict |",
            "|---|---|---|---|",
        ]
        for g, v in c["geomeans"].items():
            L.append(f"| {g} | {_fmt_ci(v)} | {v['n_ops']} | {v['verdict']} |")
        if c.get("amdahl"):
            L += [
                "",
                "### Amdahl check (t1)",
                "",
                "| mode | KleidiCV share | predicted | measured total | bound if listed ops were free |",
                "|---|---|---|---|---|",
            ]
            for mode, v in c["amdahl"].items():
                meas = next((r for r in c["rows"] if r["key"] == f"stage.{mode}.total.t1"), None)
                L.append(
                    f"| {mode} | {v['kleidicv_share']:.1%} | {v['predicted']:.3f}x | "
                    f"{_fmt_ci(meas) if meas else 'n/a'} | {v['upper_bound_if_listed_ops_free']:.3f}x |"
                )
        L += [
            "",
            "### Per key",
            "",
            "| key | group | baseline | candidate | speed-up [95% CI] | verdict |",
            "|---|---|---|---|---|---|",
        ]
        for r in c["rows"]:
            note = ""
            if r["group"] == O.CONTROL and r["verdict"] == FASTER:
                note = " (not KleidiCV-listed: investigate before claiming)"
            if r["group"] == O.NUMPY and r["verdict"] != NO_DIFF:
                note = " (NumPy negative control moved: check machine noise before trusting other ratios)"
            if r["group"] == "throughput":
                L.append(
                    f"| {r['key']} (/h) | throughput | {r['base_rate']:,.0f} | {r['cand_rate']:,.0f} | "
                    f"{_fmt_ci(r)} | {r['verdict']} |"
                )
            else:
                L.append(
                    f"| {r['key']} | {r['group']} | {r['base_ms']:.4f} ms | {r['cand_ms']:.4f} ms | "
                    f"{_fmt_ci(r)} | {r['verdict']}{note} |"
                )
    L += [
        "",
        "## Cost per 1,000",
        "",
        "(EC2 $/h + COOL $/h) / throughput per hour x 1000; prices from `bench/cool/prices.json`.",
        "",
        "| arm | instance | EC2 $/h | COOL $/h | workload | per hour | $ per 1,000 |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, ci in costs.items():
        for tk, rate in ci["rate_per_hour"].items():
            usd = ci["per_1000"].get(tk)
            L.append(
                f"| {label} | {ci['instance_type']} | {ci['ec2_usd_per_hour']} | {ci['cool_usd_per_hour']} | "
                f"{tk} | {rate:,.0f} | {'n/a' if usd is None else f'{usd:.5f}'} |"
            )
    if eq:
        L += [
            "",
            "## Equality",
            "",
            "| comparison | pairs | max abs offset diff (px) | tol | agent | gate |",
            "|---|---|---|---|---|---|",
        ]
        for name, r in eq.items():
            o = r["offset"]
            g = r.get("agent")
            gtxt = (
                "n/a"
                if g is None
                else f"{g['max_abs_diff_nm']:.3g} nm, outcomes {g['outcome_agreement']:.0%}"
            )
            L.append(
                f"| {name} | {o['n']} | {o['max_abs_diff_px']:.3g} | {o['tol_px']:g} | {gtxt} | "
                f"{'PASS' if r['pass'] else 'FAIL'} |"
            )
    L += [
        "",
        "## Honesty rules",
        "",
        "- Every number above is published as measured, including no-difference and slower results.",
        f'- If a CI includes 1.0 we say "{NO_DIFF}".',
        "- KleidiCV acceleration applies only to supported dtypes/kernel sizes; unlisted ops are controls.",
        "- The NumPy rows are negative controls and should show ~1.0x between pip and COOL.",
        "- Counterfactual rows are not engine calls and are never folded into pipeline numbers.",
        "",
    ]
    return "\n".join(L)


# ------------------------------------------------------------------------------------- docs keys
def _build_info_kleidicv(arm: Arm) -> list[str]:
    f = arm.path / "evidence" / "build_information.txt"
    if not f.is_file():
        return []
    return [
        ln.strip()
        for ln in f.read_text(encoding="utf-8", errors="replace").splitlines()
        if "kleidi" in ln.lower()
    ]


def docs_keys(
    arms: list[Arm], comps: dict, costs: dict, eq: dict, metrics: dict, smoke: bool, prices: dict
) -> dict:
    """Flat key -> value map for templating docs (COOL_EVIDENCE.md). Strings allowed."""
    d: dict = {
        "smoke": smoke,
        "prices.as_of": prices.get("as_of"),
        "prices.region": prices.get("region"),
        "verdict.no_diff_phrase": NO_DIFF,
    }
    for a in arms:
        e = a.env
        imds = a.evidence.get("imds") or {}
        p = f"arm.{a.label}"
        d.update(
            {
                f"{p}.instance_type": a.instance_type,
                f"{p}.region": imds.get("region"),
                f"{p}.ami_id": imds.get("ami_id"),
                f"{p}.product_codes": imds.get("product_codes"),
                f"{p}.arch": e.arch if e else None,
                f"{p}.cv2_version": a.evidence.get("cv2_version") or (e.opencv if e else None),
                f"{p}.cv2_file": a.evidence.get("cv2_file"),
                f"{p}.cool": a.cool,
                f"{p}.git_sha": a.git_sha,
                f"{p}.numpy": e.numpy if e else None,
                f"{p}.python": e.python if e else None,
                f"{p}.build_mentions_kleidicv": a.evidence.get("build_mentions_kleidicv"),
                f"{p}.build_kleidicv_lines": _build_info_kleidicv(a),
                f"{p}.ldd_mentions_kleidicv": a.evidence.get("ldd_mentions_kleidicv"),
                f"{p}.cool_version_files": a.evidence.get("cool_version_files"),
                f"{p}.opencv_dists": a.evidence.get("opencv_dists"),
                f"{p}.quick": a.quick,
            }
        )
    for name, c in comps.items():
        for g, v in c["geomeans"].items():
            d[f"{name}.geomean.{g}"] = round(v["speedup"], 4)
            d[f"{name}.geomean.{g}.ci"] = [round(v["ci_low"], 4), round(v["ci_high"], 4)]
            d[f"{name}.geomean.{g}.verdict"] = v["verdict"]
        for r in c["rows"]:
            if r["group"] in ("throughput", "e2e", "agent", "stage") and (
                r["group"] == "throughput" or r["key"].endswith(("total.t1", "total.tN", ".tN", ".t1"))
            ):
                d[f"{name}.{r['key']}"] = round(r["speedup"], 4)
                d[f"{name}.{r['key']}.ci"] = [round(r["ci_low"], 4), round(r["ci_high"], 4)]
                d[f"{name}.{r['key']}.verdict"] = r["verdict"]
        for mode, v in c.get("amdahl", {}).items():
            d[f"{name}.amdahl.{mode}.predicted"] = round(v["predicted"], 4)
            d[f"{name}.amdahl.{mode}.kleidicv_share"] = round(v["kleidicv_share"], 4)
    for label, ci in costs.items():
        for tk, usd in ci["per_1000"].items():
            d[f"cost.{label}.{tk}.usd_per_1000"] = round(usd, 6)
        d[f"cost.{label}.usd_per_hour"] = (
            None
            if ci["ec2_usd_per_hour"] is None or ci["cool_usd_per_hour"] is None
            else round(ci["ec2_usd_per_hour"] + ci["cool_usd_per_hour"], 4)
        )
    for name, r in eq.items():
        d[f"equality.{name}.pass"] = r["pass"]
        d[f"equality.{name}.offset_max_abs_diff_px"] = r["offset"]["max_abs_diff_px"]
        if "agent" in r:
            d[f"equality.{name}.agent_max_abs_diff_nm"] = r["agent"]["max_abs_diff_nm"]
        sm = [v["max_abs_diff"] for v in r["scoremaps"].values() if v.get("max_abs_diff") is not None]
        if sm:
            d[f"equality.{name}.scoremap_max_abs_diff"] = max(sm)
    for k in (
        "headline.cool_vs_pip.e2e_auto_throughput",
        "headline.cool_vs_pip.e2e_fast_throughput",
        "headline.cool_vs_pip.kleidicv_geomean_t1",
        "headline.cool_vs_pip.agent_throughput",
    ):
        if k in metrics:
            m = metrics[k]
            d[k] = round(m.value, 4)
            d[k + ".ci"] = [m.ci_low, m.ci_high]
            d[k + ".verdict"] = verdict(m.ci_low, m.ci_high, m.n or MIN_N)
    return d


# ------------------------------------------------------------------------------------- driver
def parse_kv(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"bad {what} {it!r}: expected key=value")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def discover_arms(run_dir: Path) -> list[str]:
    return [
        str(p)
        for p in sorted(Path(run_dir).iterdir())
        if p.is_dir() and any((p / f"{n}.json").is_file() for n in RESULT_FILES)
    ]


def _is_repo_results(out_dir: Path) -> bool:
    try:
        Path(out_dir).resolve().relative_to((C.REPO_ROOT / "results").resolve())
        return True
    except ValueError:
        return False


HEADLINES = {
    "headline.cool_vs_pip.e2e_auto_throughput": "speedup.arm-cool_vs_arm-pip.e2e.auto.throughput",
    "headline.cool_vs_pip.e2e_fast_throughput": "speedup.arm-cool_vs_arm-pip.e2e.fast.throughput",
    "headline.cool_vs_pip.agent_throughput": "speedup.arm-cool_vs_arm-pip.agent.throughput",
    "headline.cool_vs_pip.kleidicv_geomean_t1": "speedup.arm-cool_vs_arm-pip.kleidicv.geomean.t1",
    "headline.cool_vs_pip.control_geomean_t1": "speedup.arm-cool_vs_arm-pip.control.geomean.t1",
    "headline.cool_vs_pip.numpy_geomean_t1": "speedup.arm-cool_vs_arm-pip.numpy.geomean.t1",
    "headline.cool_vs_x86.e2e_auto_throughput": "speedup.arm-cool_vs_x86-pip.e2e.auto.throughput",
    "headline.graviton_vs_x86.e2e_auto_throughput": "speedup.arm-pip_vs_x86-pip.e2e.auto.throughput",
    "headline.equality.cool_vs_pip.pass": "equality.arm-cool_vs_arm-pip.pass",
    "headline.cost.arm-cool-large.auto.usd_per_1000": "cost.arm-cool-large.e2e.auto.usd_per_1000_localizations",
    "headline.cost.arm-cool.auto.usd_per_1000": "cost.arm-cool.e2e.auto.usd_per_1000_localizations",
    "headline.cost.arm-pip.auto.usd_per_1000": "cost.arm-pip.e2e.auto.usd_per_1000_localizations",
    "headline.cost.x86-pip.auto.usd_per_1000": "cost.x86-pip.e2e.auto.usd_per_1000_localizations",
}


def build_report(
    arm_specs: list[str],
    out_dir: Path,
    comparisons: list[str] | None = None,
    price_overrides: dict[str, float] | None = None,
    instance_types: dict[str, str] | None = None,
    prices_path: Path = PRICES,
    charts: bool = True,
    tol_px: float = EQ.DEFAULT_TOL_PX,
) -> dict:
    arms: list[Arm] = []
    for spec in arm_specs:
        label, path = (
            spec.split("=", 1) if "=" in spec and not Path(spec).exists() else (Path(spec).name, spec)
        )
        arms.append(load_arm(label, Path(path)))
    if not arms:
        raise ValueError("no arms given")
    by = {a.label: a for a in arms}
    warnings: list[str] = []
    itypes = instance_types or {}
    smoke = any(a.quick or not (itypes.get(a.label) or a.instance_type) for a in arms)
    out_dir = Path(out_dir)
    if smoke and _is_repo_results(out_dir) and LOCAL_SMOKE not in out_dir.as_posix():
        raise SmokeGuardError(
            f"quick/off-EC2 arms would be written to {out_dir}: use a path containing "
            f"'{LOCAL_SMOKE}' (e.g. results/bench/{LOCAL_SMOKE}/) or a temp dir"
        )
    if smoke:
        warnings.append("LOCAL SMOKE: an arm ran --quick or off EC2; plumbing only, not COOL evidence")
    if comparisons:
        cmps = [(c.split(":", 1)[0], c.split(":", 1)[1], c.replace(":", " vs ")) for c in comparisons]
    else:
        cmps = [c for c in DEFAULT_COMPARISONS if c[0] in by and c[1] in by]
    shas = {a.label: a.git_sha for a in arms}
    if len(set(shas.values())) > 1:
        warnings.append(f"arms were measured at different git SHAs: {shas}")
    for a in arms:
        if "cool" in a.label and not a.cool:
            warnings.append(f"arm {a.label!r} is labelled COOL but cv2 is not under /opt/cool")
        if "pip" in a.label and a.cool:
            warnings.append(f"arm {a.label!r} is labelled pip but cv2 is under /opt/cool")
        if a.quick:
            warnings.append(f"arm {a.label!r} ran with --quick: plumbing only, not a result")

    metrics: dict = {}
    comps: dict[str, dict] = {}
    eq: dict[str, dict] = {}
    for cand, base, desc in cmps:
        if cand not in by or base not in by:
            warnings.append(f"comparison {cand} vs {base} skipped: arm missing")
            continue
        name = f"{cand}_vs_{base}"
        m, rows, geo = compare_arms(by[cand], by[base])
        metrics.update(m)
        am = amdahl(by[cand], by[base], geo)
        for mode, v in am.items():
            metrics[f"amdahl.{name}.{mode}.predicted_speedup"] = C.metric(v["predicted"], "x")
            metrics[f"amdahl.{name}.{mode}.upper_bound_speedup"] = C.metric(
                v["upper_bound_if_listed_ops_free"]
                if math.isfinite(v["upper_bound_if_listed_ops_free"])
                else -1.0,
                "x",
            )
        comps[name] = {"description": desc, "rows": rows, "geomeans": geo, "amdahl": am}
        if (by[cand].path / "e2e_offsets.csv").is_file() and (by[base].path / "e2e_offsets.csv").is_file():
            r = EQ.compare(by[base].path, by[cand].path, tol_px)
            same_type = by[cand].instance_type == by[base].instance_type
            r["gated"] = same_type
            eq[name] = r
            for k, v in EQ.to_metrics(r).items():
                metrics[k.replace("equality.", f"equality.{name}.", 1)] = v
            if same_type and not r["pass"]:
                warnings.append(f"EQUALITY FAILED for {name}: speed-ups on this pair do not count")

    prices = load_prices(prices_path)
    costs: dict[str, dict] = {}
    for a in arms:
        m, info = arm_costs(a, prices, dict(price_overrides or {}), itypes.get(a.label))
        metrics.update(m)
        costs[a.label] = info
        metrics.update(arm_absolute(a))
        warnings += [f"cost {a.label}: {w}" for w in info["warnings"]]
    for hk, src in HEADLINES.items():
        if src in metrics:
            metrics[hk] = metrics[src]
    metrics["meta.n_arms"] = C.metric(len(arms), "arms")
    metrics["meta.git_sha_consistent"] = C.metric(1.0 if len(set(shas.values())) <= 1 else 0.0, "bool")
    metrics["meta.local_smoke"] = C.metric(1.0 if smoke else 0.0, "bool")

    out_dir.mkdir(parents=True, exist_ok=True)
    primary = next((by[k] for k in PRIMARY if k in by), arms[0])
    name = "cool_bench" + (f".{LOCAL_SMOKE}" if smoke else "")
    rf = C.write_results(
        out_dir / "cool_bench.json",
        name,
        metrics,
        C.produced_by(__file__),
        env=primary.env,
        sha=primary.git_sha,
    )
    (out_dir / "cool_bench_summary.md").write_text(
        summary_md(arms, comps, costs, eq, warnings, smoke), encoding="utf-8"
    )
    chart_files = make_charts({k: v["rows"] for k, v in comps.items()}, costs, out_dir) if charts else []
    docs = docs_keys(arms, comps, costs, eq, metrics, smoke, prices)
    C.write_json(out_dir / "cool_bench_docs.json", docs)
    meta = {
        "arms": {
            a.label: {
                "path": str(a.path),
                "instance_type": a.instance_type,
                "cool": a.cool,
                "git_sha": a.git_sha,
                "cv2_file": a.evidence.get("cv2_file"),
                "quick": a.quick,
            }
            for a in arms
        },
        "env_from_arm": primary.label,
        "comparisons": comps,
        "costs": costs,
        "equality": eq,
        "warnings": warnings,
        "charts": chart_files,
        "prices_as_of": prices.get("as_of"),
        "smoke": smoke,
    }
    C.write_json(out_dir / "cool_bench_meta.json", meta)
    return {"results": rf, "meta": meta, "docs": docs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", default=[], help="arm dir, or label=dir (repeatable)")
    ap.add_argument("--run-dir", type=Path, default=None, help="directory whose sub-directories are arm dirs")
    ap.add_argument("--compare", action="append", default=None, help="candidate:baseline labels (repeatable)")
    ap.add_argument(
        "--price", action="append", default=[], help="override EC2 price: instance_type=usd_per_hour"
    )
    ap.add_argument(
        "--instance-type", action="append", default=[], help="label=instance_type when IMDS was unavailable"
    )
    ap.add_argument("--prices", type=Path, default=PRICES)
    ap.add_argument("--out", type=Path, default=Path("results") / "bench")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--tol-px", type=float, default=EQ.DEFAULT_TOL_PX)
    a = ap.parse_args(argv)
    specs = list(a.arm) + (discover_arms(a.run_dir) if a.run_dir else [])
    if not specs:
        ap.error("give --arm and/or --run-dir")
    try:
        res = build_report(
            specs,
            a.out,
            a.compare,
            {k: float(v) for k, v in parse_kv(a.price, "--price").items()},
            parse_kv(a.instance_type, "--instance-type"),
            a.prices,
            not a.no_charts,
            a.tol_px,
        )
    except SmokeGuardError as e:
        print(f"[report] REFUSED: {e}", file=sys.stderr)
        return 3
    meta = res["meta"]
    for name, c in meta["comparisons"].items():
        print(f"[report] {name}:")
        for g, v in c["geomeans"].items():
            print(f"    {g:<20} {_fmt_ci(v)}  {v['verdict']}  (n={v['n_ops']})")
        for r in c["rows"]:
            if r["group"] == "throughput":
                print(f"    {r['key']:<20} {_fmt_ci(r)}  {r['verdict']}")
    for label, ci in meta["costs"].items():
        for tk, usd in ci["per_1000"].items():
            print(f"[report] cost {label} {tk}: ${usd:.5f} per 1,000")
    for w in meta["warnings"]:
        print(f"[report] WARNING: {w}", file=sys.stderr)
    print(
        f"[report] wrote {a.out / 'cool_bench.json'} (+ cool_bench_docs.json, cool_bench_summary.md, "
        f"cool_bench_meta.json{', charts' if meta['charts'] else ''})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
