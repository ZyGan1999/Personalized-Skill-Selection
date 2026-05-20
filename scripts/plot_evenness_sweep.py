"""
plot_evenness_sweep.py
Publication-quality 1x3 figure for the preference-evenness ablation:
  (a) Cumulative regret  vs evenness level
  (b) Test-set accuracy  vs evenness level
  (c) Spearman rank corr vs evenness level  (only the three soft settings)

X-axis is the four (or three) categorical evenness levels:
  one-hot, alpha=0.1, alpha=0.3, alpha=1.0
plotted at equally-spaced positions (the actual alpha values are not on a
uniform scale, so categorical positioning is more faithful).

Each line corresponds to one of the nine agents (camera-ready names),
with a stable per-agent (color, marker) pair. The proposed agent
(Bandit-as-Override) is rendered slightly thicker so the eye finds it.

Sources of statistics
---------------------
- Regret: `summary.json[agents][name].final_cumulative_regret.{mean,std}`,
  already aggregated with `ddof=1`.
- Test accuracy: per-seed `test_results` from each `seed_N.pkl`,
  aggregated with `ddof=1` here.
- Spearman: `summary.json[agents][name].preference_recovery_rate.spearman`
  (mean across all users x domains x seeds; no per-seed std stored, so
  no error bars on subplot (c)).

Usage
-----
# Default: Qwen3-30B across {one-hot, soft0.1, soft0.3, soft1.0}
python scripts/plot_evenness_sweep.py

# Different model
python scripts/plot_evenness_sweep.py --model deepseek-v4-flash

# Custom output path
python scripts/plot_evenness_sweep.py --output figures/fig_evenness.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

# Make the project root importable so pickled SeedResult deserialise.
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


# ---------------------------------------------------------------------------
# Canonical paper-side agent ordering and display names.
# ---------------------------------------------------------------------------

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

# Family-grouped restrained palette (same as plot_test_accuracy.py).
_FAMILY_PALETTE: Dict[str, str] = {
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

# One marker per agent. Shapes are chosen to be unambiguous at print size.
_MARKER: Dict[str, str] = {
    "Random":             "X",
    "ZeroShot-LLM":       "v",
    "InContext-Memory":   "^",
    "Profile-Memory":     "*",
    "Freq-Greedy":        "s",
    "Pure-Bandit":        "D",
    "Bandit-as-Context":  "P",
    "Freq-as-Override":   "d",
    "Bandit-as-Override": "o",
}

# Evenness levels: (directory tag, x-axis label).
EVENNESS_LEVELS: List[Tuple[str, str]] = [
    ("onehot",  "one-hot"),
    ("soft0.1", r"$\alpha{=}0.1$"),
    ("soft0.3", r"$\alpha{=}0.3$"),
    ("soft1.0", r"$\alpha{=}1.0$"),
]
SOFT_ONLY_LEVELS = EVENNESS_LEVELS[1:]   # for the Spearman subplot


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _find_seed_pkls(ckpt_dir: str) -> List[str]:
    out: List[Tuple[int, str]] = []
    for path in glob(os.path.join(ckpt_dir, "seed_*.pkl")):
        base = os.path.basename(path)
        if "_progress" in base:
            continue
        m = re.match(r"seed_(\d+)\.pkl$", base)
        if not m:
            continue
        out.append((int(m.group(1)), path))
    out.sort()
    return [p for _, p in out]


def _load_summary(experiment_dir: str) -> dict:
    path = os.path.join(experiment_dir, "summary.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no summary.json in {experiment_dir}")
    with open(path, "r") as f:
        return json.load(f)


def _load_test_accs(experiment_dir: str) -> Dict[str, Tuple[float, float]]:
    """Per-agent (mean, std) of held-out test accuracy across all seeds (ddof=1)."""
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return {}

    per_agent: Dict[str, List[float]] = {raw: [] for raw, _ in DISPLAY_NAME}
    for path in _find_seed_pkls(ckpt_dir):
        try:
            with open(path, "rb") as f:
                sr = pickle.load(f)
        except Exception as e:
            print(f"  [warn] could not load {path}: {e}", file=sys.stderr)
            continue
        if not sr.test_results:
            continue
        for raw, _ in DISPLAY_NAME:
            r = sr.test_results.get(raw)
            if r is None:
                continue
            per_agent[raw].append(r["accuracy"])

    result: Dict[str, Tuple[float, float]] = {}
    for raw, vals in per_agent.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        result[raw] = (mean, std)
    return result


def load_all(base_dir: str, model_tag: str) -> Dict[str, dict]:
    """
    For each evenness level, load:
      - regret_mean / regret_std   from summary.json
      - acc_mean / acc_std         from per-seed test_results
      - spearman_mean              from summary.json (soft only)
    Returns:
      {level_tag: {agent_raw: {'regret_m','regret_s','acc_m','acc_s','spearman_m'}}}
    """
    out: Dict[str, dict] = {}
    for tag, _label in EVENNESS_LEVELS:
        exp = os.path.join(base_dir, f"main-{tag}-{model_tag}")
        if not os.path.isdir(exp):
            print(f"  [warn] missing experiment dir: {exp}", file=sys.stderr)
            out[tag] = {}
            continue
        try:
            summary = _load_summary(exp)
        except FileNotFoundError as e:
            print(f"  [warn] {e}", file=sys.stderr)
            out[tag] = {}
            continue
        test_accs = _load_test_accs(exp)

        per_agent: dict = {}
        for raw, _ in DISPLAY_NAME:
            ag = summary["agents"].get(raw)
            if ag is None:
                continue
            regret = ag.get("final_cumulative_regret", {})
            acc_m, acc_s = test_accs.get(raw, (None, None))
            rec = ag.get("preference_recovery_rate", {})
            per_agent[raw] = {
                "regret_m": regret.get("mean"),
                "regret_s": regret.get("std"),
                "acc_m":    acc_m,
                "acc_s":    acc_s,
                "spearman_m": rec.get("spearman"),
            }
        out[tag] = per_agent
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.3,
        "lines.markersize": 5.5,
    })


def _draw_panel(ax, agents_in_order, x_positions, x_labels,
                values_mean, values_std, *, ylabel, title,
                accent_agent="Bandit-as-Override",
                show_errorbars=True):
    """Draw one of the three line panels.

    values_mean : dict {agent_display_name: [v0, v1, ...]} length == len(x_positions)
    values_std  : same shape; pass None to disable.
    """
    for disp in agents_in_order:
        if disp not in values_mean:
            continue
        ys = np.asarray(values_mean[disp], dtype=float)
        if np.all(np.isnan(ys)):
            continue
        color = _FAMILY_PALETTE.get(disp, "#444444")
        marker = _MARKER.get(disp, "o")
        lw = 2.0 if disp == accent_agent else 1.2
        msz = 7.0 if disp == accent_agent else 5.5
        if show_errorbars and values_std is not None and disp in values_std:
            ys_s = np.asarray(values_std[disp], dtype=float)
            ax.errorbar(
                x_positions, ys, yerr=ys_s,
                color=color, marker=marker,
                linewidth=lw, markersize=msz,
                markeredgecolor="black", markeredgewidth=0.4,
                capsize=2.2, elinewidth=0.7,
                label=disp, zorder=3 if disp == accent_agent else 2,
            )
        else:
            ax.plot(
                x_positions, ys,
                color=color, marker=marker,
                linewidth=lw, markersize=msz,
                markeredgecolor="black", markeredgewidth=0.4,
                label=disp, zorder=3 if disp == accent_agent else 2,
            )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def plot_evenness_sweep(
    base_dir: str = "paper-exp",
    model_tag: str = "qwen3-30b-a3b-instruct-2507",
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12.5, 3.7),
) -> Tuple[str, str]:
    _rc()
    data = load_all(base_dir, model_tag)

    # Build per-agent value lists, keyed by *display name*, in plot order.
    agents_disp = [disp for _, disp in DISPLAY_NAME]
    raw_to_disp = {raw: disp for raw, disp in DISPLAY_NAME}

    def _collect(metric: str, levels):
        m: Dict[str, List[float]] = {}
        s: Dict[str, List[float]] = {}
        for raw, disp in DISPLAY_NAME:
            mv, sv = [], []
            for tag, _ in levels:
                ag = data.get(tag, {}).get(raw)
                if ag is None:
                    mv.append(np.nan)
                    sv.append(np.nan)
                    continue
                key_m = f"{metric}_m"
                key_s = f"{metric}_s"
                mv.append(ag.get(key_m) if ag.get(key_m) is not None else np.nan)
                sv.append(ag.get(key_s) if ag.get(key_s) is not None else np.nan)
            m[disp] = mv
            s[disp] = sv
        return m, s

    # Regret (all 4 levels), test acc (all 4), spearman (3 soft levels).
    regret_m, regret_s = _collect("regret", EVENNESS_LEVELS)
    acc_m, acc_s = _collect("acc", EVENNESS_LEVELS)
    spear_m, _ = _collect("spearman", SOFT_ONLY_LEVELS)

    # Convert acc to % for nicer y-axis numbers
    acc_m_pct = {k: [v * 100 if v is not None and not np.isnan(v) else np.nan
                      for v in vs] for k, vs in acc_m.items()}
    acc_s_pct = {k: [v * 100 if v is not None and not np.isnan(v) else np.nan
                      for v in vs] for k, vs in acc_s.items()}

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    x4 = np.arange(len(EVENNESS_LEVELS))
    lbl4 = [lbl for _, lbl in EVENNESS_LEVELS]
    x3 = np.arange(len(SOFT_ONLY_LEVELS))
    lbl3 = [lbl for _, lbl in SOFT_ONLY_LEVELS]

    _draw_panel(
        axes[0], agents_disp, x4, lbl4,
        values_mean=regret_m, values_std=regret_s,
        ylabel="Cumulative Regret",
        title=r"(a) Cumulative Regret $\downarrow$",
    )
    _draw_panel(
        axes[1], agents_disp, x4, lbl4,
        values_mean=acc_m_pct, values_std=acc_s_pct,
        ylabel="Test-pool Accuracy (%)",
        title=r"(b) Test-pool Accuracy $\uparrow$",
    )
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(round(v))}%")
    )
    _draw_panel(
        axes[2], agents_disp, x3, lbl3,
        values_mean=spear_m, values_std=None,
        ylabel="Spearman Rank Corr.",
        title=r"(c) Spearman Rank Correlation $\uparrow$",
        show_errorbars=False,
    )
    # Slight headroom for SRC since values cluster low.
    sm_concat = [v for vs in spear_m.values() for v in vs
                 if v is not None and not np.isnan(v)]
    if sm_concat:
        ymin = min(0.0, float(np.min(sm_concat)) - 0.05)
        ymax = max(0.6, float(np.max(sm_concat)) + 0.08)
        axes[2].set_ylim(ymin, ymax)

    # Common legend below the row of panels — 9 agents in 3 columns x 3 rows
    # gives an even rectangular legend block.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3, frameon=False,
        handlelength=1.8, handletextpad=0.5,
        columnspacing=2.0, labelspacing=0.4,
    )

    fig.tight_layout(rect=(0.0, 0.16, 1.0, 1.0))

    # Output path resolution
    if output_path is None:
        os.makedirs("figures", exist_ok=True)
        base = os.path.join("figures", f"evenness_sweep_{model_tag}")
    elif os.path.isdir(output_path) or output_path.endswith("/"):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.join(output_path, f"evenness_sweep_{model_tag}")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        base, _ = os.path.splitext(output_path)

    pdf_path = base + ".pdf"
    png_path = base + ".png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"  agents : {agents_disp}")
    print(f"  saved  : {pdf_path}")
    print(f"           {png_path}")
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-dir", default="paper-exp",
                   help="Parent directory of the experiment subdirs (default: paper-exp).")
    p.add_argument("--model", default="qwen3-30b-a3b-instruct-2507",
                   help="Model tag used in the experiment dir name "
                        "(default: qwen3-30b-a3b-instruct-2507).")
    p.add_argument("--output", default=None,
                   help="Output PDF path or directory. "
                        "Defaults to figures/evenness_sweep_<model>.{pdf,png}")
    p.add_argument("--width", type=float, default=12.5,
                   help="Figure width in inches (default: 12.5).")
    p.add_argument("--height", type=float, default=3.7,
                   help="Figure height in inches (default: 3.7).")
    args = p.parse_args()

    plot_evenness_sweep(
        base_dir=args.base_dir,
        model_tag=args.model,
        output_path=args.output,
        figsize=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
