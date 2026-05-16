"""
plot_test_accuracy.py
Publication-quality grouped-bar chart of held-out test-set accuracy for the
paper's main experiments. Two bars per agent: Overall Accuracy (over the full
test pool, 90% standard + 10% OOD by default) and Explicit Query Accuracy
(the OOD subset, where the query names a tool by name).

Reads each seed's `test_results` from
`<experiment-dir>/checkpoints/seed_*.pkl` and computes mean ± std (unbiased,
ddof=1) across seeds.

The figure follows top-tier-conference conventions: serif font, restrained
palette grouped by agent family, error bars (no per-bar numeric labels),
hairline grid, no chart title (handle that in the LaTeX caption).

Usage
-----
# Default: read paper-exp/<run-dir>, save to <dir>/images/test_accuracy_pub.{pdf,png}
python scripts/plot_test_accuracy.py \
    --dir paper-exp/main-soft0.3-qwen3-30b-a3b-instruct-2507

# Pick a custom output path; supports either a directory or a full file path
python scripts/plot_test_accuracy.py \
    --dir paper-exp/main-soft0.3-qwen3-30b-a3b-instruct-2507 \
    --output figures/test_acc_soft_qwen.pdf

# Restrict seeds
python scripts/plot_test_accuracy.py \
    --dir paper-exp/main-onehot-deepseek-v4-flash --seeds 0 1 2

# Annotate per-bar numeric labels (off by default to keep the figure clean)
python scripts/plot_test_accuracy.py --dir <...> --annotate
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

# Ensure the project root is importable so that pickles referencing
# `env.SeedResult`, `data_gen.UserPersona`, agent classes, etc. deserialise.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("ERROR: matplotlib / numpy not installed. "
          "Run inside `conda run -n tool-call ...`.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Canonical agent ordering and display names for the paper.
# Map checkpoint names (LHS) → camera-ready names (RHS), keeping plot order.
# ---------------------------------------------------------------------------

DISPLAY_NAME = [
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

# Restrained palette grouped by agent family. Two hues per family are visually
# distinct without being garish; the proposed agent (last) gets the strongest
# saturation in its family so the eye finds it in the bar chart.
_FAMILY_PALETTE: Dict[str, str] = {
    # No learning ---------- neutral gray
    "Random":            "#7f7f7f",
    # LLM only ------------- warm muted
    "ZeroShot-LLM":      "#f4a261",
    "InContext-Memory":  "#e76f51",
    "Profile-Memory":    "#bc4749",
    # Statistical only ----- cool muted
    "Freq-Greedy":       "#a8c5e6",
    "Pure-Bandit":       "#4a7ab8",
    # Hybrid (ours + ablations) -- highlight greens, with proposed deepest
    "Bandit-as-Context": "#9c89b8",   # purple (LLM as decision-maker)
    "Freq-as-Override":  "#7fb069",   # mid green (ablation)
    "Bandit-as-Override":"#2a9d8f",   # deep teal-green (proposed)
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _find_seed_pkls(ckpt_dir: str, allowed_seeds: Optional[List[int]]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for path in glob(os.path.join(ckpt_dir, "seed_*.pkl")):
        base = os.path.basename(path)
        if "_progress" in base:
            continue
        m = re.match(r"seed_(\d+)\.pkl$", base)
        if not m:
            continue
        seed_num = int(m.group(1))
        if allowed_seeds is not None and seed_num not in allowed_seeds:
            continue
        out.append((seed_num, path))
    out.sort()
    return out


def load_test_results(
    experiment_dir: str,
    allowed_seeds: Optional[List[int]] = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, np.ndarray], Dict[str, np.ndarray], List[int]]:
    """
    Returns
    -------
    (display_pairs, accs, oods, seeds_used)
        display_pairs : the (raw, display) name pairs (in plot order) that
                        are present in the data
        accs / oods   : agent_raw_name -> array of per-seed values (range 0..1)
        seeds_used    : list of seed numbers actually loaded
    """
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"no checkpoints/ directory in {experiment_dir}")

    seed_files = _find_seed_pkls(ckpt_dir, allowed_seeds)
    if not seed_files:
        raise FileNotFoundError(f"no finalized seed_*.pkl in {ckpt_dir}")

    accs: Dict[str, List[float]] = {raw: [] for raw, _ in DISPLAY_NAME}
    oods: Dict[str, List[float]] = {raw: [] for raw, _ in DISPLAY_NAME}
    seeds_loaded: List[int] = []

    for seed_num, path in seed_files:
        try:
            with open(path, "rb") as f:
                sr = pickle.load(f)
        except Exception as e:
            print(f"  [warn] could not load {path}: {e}", file=sys.stderr)
            continue
        if not sr.test_results:
            continue
        seeds_loaded.append(seed_num)
        for raw, _ in DISPLAY_NAME:
            r = sr.test_results.get(raw)
            if r is None:
                continue
            accs[raw].append(r["accuracy"])
            oods[raw].append(r["ood_accuracy"])

    # Keep only agents that actually have data
    display_pairs = [(raw, disp) for raw, disp in DISPLAY_NAME if accs[raw]]
    accs_arr = {raw: np.asarray(accs[raw], dtype=float) for raw, _ in display_pairs}
    oods_arr = {raw: np.asarray(oods[raw], dtype=float) for raw, _ in display_pairs}

    return display_pairs, accs_arr, oods_arr, seeds_loaded


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _configure_rc() -> None:
    """Conference-style rcParams. Conservative serif look; tweak if needed."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "pdf.fonttype": 42,   # embed Type-42 fonts (editable in Illustrator)
        "ps.fonttype": 42,
    })


def _lighten(hex_color: str, blend: float = 0.55) -> str:
    """Mix a hex color toward white. blend=0 returns the original; blend=1 returns white."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = int(round(r + (255 - r) * blend))
    g = int(round(g + (255 - g) * blend))
    b = int(round(b + (255 - b) * blend))
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_test_accuracy(
    experiment_dir: str,
    output_path: Optional[str] = None,
    allowed_seeds: Optional[List[int]] = None,
    annotate: bool = True,
    figsize: Tuple[float, float] = (8.0, 3.8),
) -> Tuple[str, str]:
    """
    Render the figure. Saves both .pdf and .png next to `output_path`.
    Returns the two saved paths.
    """
    _configure_rc()

    display_pairs, accs, oods, seeds_used = load_test_results(
        experiment_dir, allowed_seeds=allowed_seeds
    )
    if not display_pairs:
        raise RuntimeError(f"no test_results loaded from {experiment_dir}")

    raw_names = [raw for raw, _ in display_pairs]
    disp_names = [disp for _, disp in display_pairs]

    def _mean_std(arr: np.ndarray) -> Tuple[float, float]:
        if arr.size <= 1:
            return float(arr.mean()) if arr.size else 0.0, 0.0
        return float(arr.mean()), float(arr.std(ddof=1))

    acc_mean = np.array([_mean_std(accs[r])[0] for r in raw_names])
    acc_std  = np.array([_mean_std(accs[r])[1] for r in raw_names])
    ood_mean = np.array([_mean_std(oods[r])[0] for r in raw_names])
    ood_std  = np.array([_mean_std(oods[r])[1] for r in raw_names])

    full_colors  = [_FAMILY_PALETTE.get(d, "#444444") for d in disp_names]
    light_colors = [_lighten(c, blend=0.55) for c in full_colors]

    x = np.arange(len(disp_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=figsize)

    # Overall (saturated fill) vs Explicit Query (light fill, same hue).
    # Both share a thin black edge so the boundary stays crisp at print size.
    bars_overall = ax.bar(
        x - width / 2, acc_mean, width,
        yerr=acc_std, capsize=2.5,
        color=full_colors, edgecolor="black", linewidth=0.5,
        error_kw={"linewidth": 0.7, "ecolor": "black"},
        label="Overall Accuracy",
    )
    bars_explicit = ax.bar(
        x + width / 2, ood_mean, width,
        yerr=ood_std, capsize=2.5,
        color=light_colors, edgecolor="black", linewidth=0.5,
        error_kw={"linewidth": 0.7, "ecolor": "black"},
        label="Explicit Query Accuracy",
    )

    if annotate:
        for bar, m, s in zip(bars_overall, acc_mean, acc_std):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.012,
                    f"{m * 100:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color="#222222")
        for bar, m, s in zip(bars_explicit, ood_mean, ood_std):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.012,
                    f"{m * 100:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(disp_names, rotation=22, ha="right")
    # A small headroom above 100% so per-bar numeric labels remain visible.
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Test-pool Accuracy")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(round(v * 100))}%")
    )
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Legend: neutral-gray proxy patches so the legend conveys "saturated vs
    # light" without binding to any specific agent's color. Placed above the
    # axes to avoid colliding with tall bars / numeric labels.
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#666666", edgecolor="black", linewidth=0.5,
              label="Overall Accuracy"),
        Patch(facecolor=_lighten("#666666", 0.55), edgecolor="black", linewidth=0.5,
              label="Explicit Query Accuracy"),
    ]
    ax.legend(handles=legend_handles,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, ncol=2,
              handlelength=1.8, handletextpad=0.6, columnspacing=2.0)

    fig.tight_layout()

    if output_path is None:
        out_dir = os.path.join(experiment_dir, "images")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, "test_accuracy_pub")
    elif os.path.isdir(output_path) or output_path.endswith("/"):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.join(output_path, "test_accuracy_pub")
    else:
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        base, _ = os.path.splitext(output_path)

    pdf_path = base + ".pdf"
    png_path = base + ".png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"  seeds used : {seeds_used}  (n={len(seeds_used)})")
    print(f"  agents     : {disp_names}")
    print(f"  saved      : {pdf_path}")
    print(f"               {png_path}")
    return pdf_path, png_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dir", required=True,
                   help="Experiment directory containing checkpoints/seed_*.pkl")
    p.add_argument("--output", default=None,
                   help="Output PDF path or output directory. "
                        "Defaults to <dir>/images/test_accuracy_pub.{pdf,png}")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Optional seed whitelist (default: all finalized).")
    p.add_argument("--no-annotate", dest="annotate", action="store_false",
                   help="Suppress per-bar numeric labels (on by default).")
    p.set_defaults(annotate=True)
    p.add_argument("--width", type=float, default=8.0,
                   help="Figure width in inches (default: 8.0).")
    p.add_argument("--height", type=float, default=3.8,
                   help="Figure height in inches (default: 3.8).")
    args = p.parse_args()

    plot_test_accuracy(
        experiment_dir=args.dir,
        output_path=args.output,
        allowed_seeds=args.seeds,
        annotate=args.annotate,
        figsize=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
