"""
plot_appendix_per_domain.py
Appendix figure: per-domain accuracy for the six main experiments
(3 backbones x {one-hot, soft0.3}), rendered as a 2 x 3 grid of
heatmaps.

For each panel, rows are the nine agents (camera-ready ordering) and
columns are the ten ToolBench-60 domains. Each cell shows the
agent's training-round accuracy on that domain, averaged across all
seeds (ddof=1 std reported in textual annotation when --annotate is on).

Output: figures/appendix_per_domain.{pdf,png} by default.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
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

# Canonical ToolBench-60 domain order (matches the benchmark JSON).
DOMAINS_ORDER = [
    "Finance", "Sports", "Travel", "Entertainment", "Gaming",
    "Education", "Communication", "Location", "eCommerce", "Social",
]

BACKBONES = [
    ("qwen3-30b-a3b-instruct-2507", "Qwen3-30B"),
    ("deepseek-v4-flash",           "DeepSeek-V4-Flash"),
    ("gpt-5.2",                     "GPT-5.2"),
]
REGIMES = [
    ("onehot",  "One-hot Preference"),
    ("soft0.3", r"Soft Preference ($\alpha=0.3$)"),
]


def _find_seed_pkls(ckpt_dir: str) -> List[str]:
    paths = []
    for path in glob(os.path.join(ckpt_dir, "seed_*.pkl")):
        if "_progress" in os.path.basename(path):
            continue
        m = re.match(r"seed_(\d+)\.pkl$", os.path.basename(path))
        if m:
            paths.append((int(m.group(1)), path))
    paths.sort()
    return [p for _, p in paths]


def _per_domain_seed_accuracy(records) -> Dict[str, float]:
    """Mean reward per domain for one agent in one seed."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for r in records:
        sums[r.domain] = sums.get(r.domain, 0.0) + r.reward
        counts[r.domain] = counts.get(r.domain, 0) + 1
    return {d: sums[d] / counts[d] for d in sums}


def load_panel(exp_dir: str) -> Optional[np.ndarray]:
    """Returns shape (9 agents, 10 domains) of mean accuracy across seeds."""
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    agent_raw_to_idx = {raw: i for i, (raw, _) in enumerate(DISPLAY_NAME)}
    domain_to_idx = {d: i for i, d in enumerate(DOMAINS_ORDER)}
    n_agents, n_domains = len(DISPLAY_NAME), len(DOMAINS_ORDER)

    per_seed: List[np.ndarray] = []
    for path in _find_seed_pkls(ckpt_dir):
        try:
            with open(path, "rb") as f:
                sr = pickle.load(f)
        except Exception as e:
            print(f"  [warn] {path}: {e}", file=sys.stderr)
            continue
        mat = np.full((n_agents, n_domains), np.nan, dtype=np.float64)
        for ar in sr.agent_results:
            if ar.agent_name not in agent_raw_to_idx:
                continue
            ai = agent_raw_to_idx[ar.agent_name]
            for dom, acc in _per_domain_seed_accuracy(ar.records).items():
                if dom in domain_to_idx:
                    mat[ai, domain_to_idx[dom]] = acc
        per_seed.append(mat)
    if not per_seed:
        return None
    return np.nanmean(np.stack(per_seed), axis=0)


def _configure_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot(base_dir: str = "paper-exp",
         output_path: Optional[str] = None,
         annotate: bool = True,
         figsize: Tuple[float, float] = (13.5, 7.8)) -> Tuple[str, str]:
    _configure_rc()

    # Pre-load all panels
    panels: Dict[Tuple[str, str], np.ndarray] = {}
    for model_tag, _ in BACKBONES:
        for regime_tag, _ in REGIMES:
            exp = os.path.join(base_dir, f"main-{regime_tag}-{model_tag}")
            mat = load_panel(exp)
            if mat is not None:
                panels[(model_tag, regime_tag)] = mat
            else:
                print(f"  [warn] no data for {exp}", file=sys.stderr)

    fig, axes = plt.subplots(len(REGIMES), len(BACKBONES),
                             figsize=figsize)
    agent_disp = [disp for _, disp in DISPLAY_NAME]

    # Use a perceptually-uniform sequential colormap; values in [0, 1].
    cmap = cm.get_cmap("YlGnBu")
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    last_im = None
    for r, (regime_tag, regime_label) in enumerate(REGIMES):
        for c, (model_tag, model_label) in enumerate(BACKBONES):
            ax = axes[r, c]
            mat = panels.get((model_tag, regime_tag))
            if mat is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
            last_im = im
            # Per-cell numeric annotation
            if annotate:
                for ai in range(mat.shape[0]):
                    for di in range(mat.shape[1]):
                        v = mat[ai, di]
                        if np.isnan(v):
                            continue
                        ax.text(di, ai, f"{v * 100:.0f}",
                                ha="center", va="center",
                                fontsize=6.5,
                                color="white" if v > 0.55 else "#222222")

            ax.set_xticks(np.arange(len(DOMAINS_ORDER)))
            ax.set_yticks(np.arange(len(agent_disp)))
            ax.set_xticklabels(DOMAINS_ORDER, rotation=35, ha="right")
            ax.set_yticklabels(agent_disp if c == 0 else [""] * len(agent_disp))
            ax.set_title(model_label if r == 0 else "", fontsize=10.5, pad=6)
            if c == 0:
                # Use a y-axis label as the row label
                ax.set_ylabel(regime_label, fontsize=10)
            ax.tick_params(axis="both", which="both", length=0)

    # One shared colorbar to the right of the grid
    if last_im is not None:
        cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.65])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Accuracy", fontsize=9.5)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))

    if output_path is None:
        os.makedirs("figures", exist_ok=True)
        base = "figures/appendix_per_domain"
    elif os.path.isdir(output_path) or output_path.endswith("/"):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.join(output_path, "appendix_per_domain")
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
    p.add_argument("--width", type=float, default=13.5)
    p.add_argument("--height", type=float, default=7.8)
    args = p.parse_args()
    plot(base_dir=args.base_dir, output_path=args.output,
         annotate=args.annotate, figsize=(args.width, args.height))


if __name__ == "__main__":
    main()
