"""
plot_appendix_preference_recovery.py
Appendix figure: preference-recovery metrics under the soft (alpha=0.3)
preference regime, for the three backbones (Qwen3-30B, DeepSeek-V4-Flash,
GPT-5.2).

For each backbone, we report three distribution-alignment metrics between
the agent's learned per-(user,domain) preference and the user's true
Dirichlet distribution:
  * Cosine Similarity     (higher is better)
  * KL Divergence         (lower is better)
  * Spearman Rank Corr.   (higher is better)

Layout: 3 rows (backbones) x 3 columns (metrics) = 9 panels, each is a
nine-agent bar chart with the canonical paper ordering and palette.

Source of statistics: summary.json[agents][name].preference_recovery_rate
(aggregated across all seeds x users x domains; no per-seed std is stored,
so error bars are omitted).

Output: figures/appendix_preference_recovery.{pdf,png} by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib / numpy not installed.", file=sys.stderr)
    sys.exit(1)


DISPLAY_NAME: List[Tuple[str, str]] = [
    ("Random",            "Random"),
    ("ZeroShot-LLM",      "ZeroShot-LLM"),
    ("Freq-Greedy",       "Freq-Greedy"),
    ("Pure-Bandit",       "Pure-Bandit"),
    ("InContext-Memory",  "InContext-Memory"),
    ("Profile-Memory",    "Profile-Memory"),
    ("Bandit+CoT",        "Bandit-as-Context"),
    ("Freq+Override",     "Freq-as-Override"),
    ("Bandit+Override",   "Bandit-as-Override"),
]

_PALETTE: Dict[str, str] = {
    "Random":             "#7f7f7f",
    "ZeroShot-LLM":       "#f4a261",
    "InContext-Memory":   "#e76f51",
    "Profile-Memory":     "#bc4749",
    "Freq-Greedy":        "#a8c5e6",
    "Pure-Bandit":        "#4a7ab8",
    "Bandit-as-Context":  "#9c89b8",
    "Freq-as-Override":   "#7fb069",
    "Bandit-as-Override": "#2a9d8f",
}

BACKBONES = [
    ("qwen3-30b-a3b-instruct-2507", "Qwen3-30B"),
    ("deepseek-v4-flash",           "DeepSeek-V4-Flash"),
    ("gpt-5.2",                     "GPT-5.2"),
]
METRICS = [
    ("cosine_sim",    r"Cosine Similarity $\uparrow$"),
    ("kl_divergence", r"KL Divergence $\downarrow$"),
    ("spearman",      r"Spearman Rank Corr. $\uparrow$"),
]


def load_panels(base_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns {model_tag: {metric_key: {agent_display_name: value}}}
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_tag, _ in BACKBONES:
        exp_dir = os.path.join(base_dir, f"main-soft0.3-{model_tag}")
        summary_path = os.path.join(exp_dir, "summary.json")
        if not os.path.isfile(summary_path):
            print(f"  [warn] missing summary.json in {exp_dir}", file=sys.stderr)
            continue
        with open(summary_path, "r") as f:
            summary = json.load(f)
        by_metric: Dict[str, Dict[str, float]] = {m: {} for m, _ in METRICS}
        for raw, disp in DISPLAY_NAME:
            ag = summary["agents"].get(raw)
            if ag is None:
                continue
            rec = ag.get("preference_recovery_rate", {})
            for metric_key, _ in METRICS:
                if metric_key in rec:
                    by_metric[metric_key][disp] = float(rec[metric_key])
        out[model_tag] = by_metric
    return out


def _configure_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot(base_dir: str = "paper-exp",
         output_path: Optional[str] = None,
         annotate: bool = True,
         figsize: Tuple[float, float] = (12.0, 8.5)) -> Tuple[str, str]:
    _configure_rc()
    panels = load_panels(base_dir)

    agent_disp = [disp for _, disp in DISPLAY_NAME]
    colors = [_PALETTE.get(d, "#444444") for d in agent_disp]

    # Pre-compute the per-metric global y-axis range so rows are comparable.
    metric_ranges: Dict[str, Tuple[float, float]] = {}
    for metric_key, _ in METRICS:
        vs: List[float] = []
        for model_tag, _ in BACKBONES:
            d = panels.get(model_tag, {}).get(metric_key, {})
            vs.extend(d.values())
        if not vs:
            metric_ranges[metric_key] = (0.0, 1.0)
            continue
        lo = min(0.0, min(vs))
        hi = max(vs)
        pad = (hi - lo) * 0.12 if hi > lo else 0.1
        metric_ranges[metric_key] = (lo, hi + pad)

    fig, axes = plt.subplots(len(BACKBONES), len(METRICS),
                             figsize=figsize)

    x = np.arange(len(agent_disp))
    width = 0.66

    for r, (model_tag, model_label) in enumerate(BACKBONES):
        for c, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[r, c]
            d = panels.get(model_tag, {}).get(metric_key, {})
            if not d:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            vals = np.array([d.get(disp, np.nan) for disp in agent_disp])
            bars = ax.bar(x, vals, width,
                          color=colors, edgecolor="black", linewidth=0.5)

            if annotate:
                for bar, v in zip(bars, vals):
                    if np.isnan(v):
                        continue
                    fmt = f"{v:.3f}" if metric_key != "kl_divergence" else f"{v:.2f}"
                    # Offset is metric-specific because scales differ.
                    lo, hi = metric_ranges[metric_key]
                    yoff = (hi - lo) * 0.018
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + yoff,
                            fmt, ha="center", va="bottom",
                            fontsize=7.5, color="#222222")

            ax.set_xticks(x)
            ax.set_xticklabels(agent_disp, rotation=28, ha="right")
            ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylim(*metric_ranges[metric_key])

            if r == 0:
                ax.set_title(metric_label, fontsize=10.5, pad=6)
            if c == 0:
                ax.set_ylabel(model_label, fontsize=10)
            if r < len(BACKBONES) - 1:
                # hide x labels for non-bottom rows to reduce clutter
                ax.set_xticklabels([""] * len(agent_disp))

    fig.tight_layout(h_pad=1.2, w_pad=1.6)

    if output_path is None:
        os.makedirs("figures", exist_ok=True)
        base = "figures/appendix_preference_recovery"
    elif os.path.isdir(output_path) or output_path.endswith("/"):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.join(output_path, "appendix_preference_recovery")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        base, _ = os.path.splitext(output_path)

    pdf_path, png_path = base + ".pdf", base + ".png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"  saved: {pdf_path}")
    print(f"         {png_path}")
    return pdf_path, png_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", default="paper-exp")
    p.add_argument("--output", default=None)
    p.add_argument("--no-annotate", dest="annotate", action="store_false")
    p.set_defaults(annotate=True)
    p.add_argument("--width", type=float, default=12.0)
    p.add_argument("--height", type=float, default=8.5)
    args = p.parse_args()
    plot(base_dir=args.base_dir, output_path=args.output,
         annotate=args.annotate, figsize=(args.width, args.height))


if __name__ == "__main__":
    main()
