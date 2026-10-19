"""Charts for the sealed-set evaluation — rendered FROM results/sealed/*.json.

This script never recomputes a metric: it reads the ResultsFile JSON written by
eval/sealed_run.py and draws. Four figures (whichever arms are present):

  sealed_top1_by_tier.png     per-tier / per-family top-1 with Wilson bars
  sealed_ablation.png         ablation ladder (pooled top-1 per rung)
  sealed_agent_arms.png       agent arms: success / false-commit / escalation
  sealed_calibration.png      v2_single risk-coverage + per-tier false-confident

Reconstructed ablation rungs (``*.reconstructed = 1``) are hatched and labelled so
they are never read as engine configurations. The sealed ECE is equal-count and
is labelled as such, distinct from the A6 equal-width ECE.

    python -m eval.charts                       # results/sealed -> results/sealed/*.png
    python -m eval.charts --in DIR --out DIR
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from driftsense.contracts import ResultsFile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TIERS = ("clean", "nominal", "hard", "ambiguous")
RECON = ("no_roughness_fusion", "lattice_only")
LADDER = (
    "full",
    "no_cascade",
    "no_contour_subpixel",
    "no_calibration",
    "no_roughness_fusion",
    "lattice_only",
)
BLUE, RED, GRAY = "#1f77b4", "#d62728", "#7f7f7f"


def _load(in_dir: Path) -> dict[str, ResultsFile]:
    out: dict[str, ResultsFile] = {}
    for f in sorted(in_dir.glob("*.json")):
        if f.name == "decisions.json":
            continue
        try:
            rf = ResultsFile.model_validate_json(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        out[rf.name] = rf
    return out


def _m(rf: ResultsFile, key: str):
    return rf.metrics.get(key)


def _bars(ax, labels, vals, los, his, color, hatches=None):
    x = range(len(labels))
    err = [
        [max(0.0, v - lo) for v, lo in zip(vals, los, strict=True)],
        [max(0.0, hi - v) for v, hi in zip(vals, his, strict=True)],
    ]
    bars = ax.bar(x, vals, yerr=err, color=color, capsize=3, edgecolor="black", linewidth=0.5)
    if hatches:
        for b, h in zip(bars, hatches, strict=True):
            if h:
                b.set_hatch("///")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)


def chart_top1(data, out: Path) -> Path | None:
    v2 = data.get("v2_single")
    if not v2:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    tiers = [t for t in TIERS if _m(v2, f"top1.{t}")]
    tv = [_m(v2, f"top1.{t}") for t in tiers]
    _bars(
        a1,
        tiers,
        [m.value for m in tv],
        [m.ci_low or m.value for m in tv],
        [m.ci_high or m.value for m in tv],
        BLUE,
    )
    a1.axhline(0.85, ls="--", color=GRAY, lw=1)
    a1.set_title("v2_single top-1 by tier (Wilson 95%)")
    a1.set_ylabel("top-1 accuracy")
    fams = sorted(k.split("top1.family.")[1] for k in v2.metrics if k.startswith("top1.family."))
    fv = [_m(v2, f"top1.family.{f}") for f in fams]
    _bars(
        a2,
        fams,
        [m.value for m in fv],
        [m.ci_low or m.value for m in fv],
        [m.ci_high or m.value for m in fv],
        BLUE,
    )
    a2.axhline(0.80, ls="--", color=GRAY, lw=1)
    a2.set_title("v2_single top-1 by family")
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    return out


def chart_ablation(data, out: Path) -> Path | None:
    rungs = [r for r in LADDER if f"ablation_{r}" in data]
    if not rungs:
        return None
    vals, los, his, hatch = [], [], [], []
    for r in rungs:
        m = _m(data[f"ablation_{r}"], "top1.pooled")
        vals.append(m.value)
        los.append(m.ci_low or m.value)
        his.append(m.ci_high or m.value)
        hatch.append(r in RECON)
    fig, ax = plt.subplots(figsize=(7, 4))
    _bars(ax, rungs, vals, los, his, BLUE, hatches=hatch)
    ax.set_title("Ablation ladder — pooled top-1 (/// = eval-reconstructed, not engine config)")
    ax.set_ylabel("top-1 accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    return out


def chart_agent(data, out: Path) -> Path | None:
    ag = data.get("agent_rules")
    if not ag:
        return None
    arms = ["v2_single", "agent_rules"]
    metrics = ["success", "false_commit", "escalation"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    for i, met in enumerate(metrics):
        vals = [_m(ag, f"{a}.{met}").value if _m(ag, f"{a}.{met}") else 0.0 for a in arms]
        ax.bar([x + i * width for x in range(len(arms))], vals, width, label=met)
    ax.set_xticks([x + width for x in range(len(arms))])
    ax.set_xticklabels(arms)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Agent arms (episodes)")
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    return out


def chart_calibration(data, out: Path) -> Path | None:
    v2 = data.get("v2_single")
    if not v2:
        return None
    tiers = [t for t in TIERS if _m(v2, f"false_confident.{t}")]
    if not tiers:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [_m(v2, f"false_confident.{t}").value for t in tiers]
    ax.bar(range(len(tiers)), vals, color=RED, edgecolor="black", linewidth=0.5)
    ax.axhline(0.02, ls="--", color=GRAY, lw=1, label="target 0.02")
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers)
    ece = _m(v2, "ece")
    title = "v2_single false-confident by tier"
    if ece:
        title += f"  ·  ECE(equal-count)={ece.value:.3f}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor="white")
    plt.close(fig)
    return out


def render_all(in_dir: Path, out_dir: Path) -> list[Path]:
    data = _load(in_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for fn, name in (
        (chart_top1, "sealed_top1_by_tier.png"),
        (chart_ablation, "sealed_ablation.png"),
        (chart_agent, "sealed_agent_arms.png"),
        (chart_calibration, "sealed_calibration.png"),
    ):
        p = fn(data, out_dir / name)
        if p:
            made.append(p)
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render sealed-eval charts from results JSON")
    ap.add_argument("--in", dest="in_dir", default=str(ROOT / "results" / "sealed"))
    ap.add_argument("--out", dest="out_dir", default=None)
    args = ap.parse_args(argv)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    made = render_all(in_dir, out_dir)
    print(f"charts: wrote {len(made)} figure(s) to {out_dir}")
    for p in made:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
