"""
plot_explicit_query_acc.py
Single-bar variant of `plot_test_accuracy.py` that shows only the Explicit
Query Accuracy (OOD subset of the held-out test pool) for each agent.

Use when the two-bar grouped chart is too wide; this version is roughly
half the width of the grouped chart and fits comfortably in a
single-column figure slot.

Reads each seed's `test_results` from
`<experiment-dir>/checkpoints/seed_*.pkl` and computes mean ± std
(unbiased, ddof=1) across seeds.

Same conference-style conventions as `plot_test_accuracy.py`:
- serif font, family-grouped restrained palette
- thin error bars, per-bar numeric labels
- no chart title (handle that in the LaTeX caption)
- output both PDF (vector, Type-42 fonts) and PNG (300 dpi)

Usage
-----
# Default: save to <dir>/images/explicit_query_acc.{pdf,png}
python scripts/plot_explicit_query_acc.py \
    --dir paper-exp/main-soft0.3-qwen3-30b-a3b-instruct-2507

# Custom output (file path or directory)
python scripts/plot_explicit_query_acc.py \
    --dir paper-exp/main-soft0.3-qwen3-30b-a3b-instruct-2507 \
    --output figures/explicit_qwen_soft.pdf

# Restrict seeds
python scripts/plot_explicit_query_acc.py \
    --dir paper-exp/main-onehot-deepseek-v4-flash --seeds 0 1 2

# Suppress per-bar numeric labels
python scripts/plot_explicit_query_acc.py --dir <...> --no-annotate
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

# Ensure the project root is importable so pickled SeedResult objects
# (which reference env, data_gen, agent classes) deserialise.
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


def load_explicit_query_accs(
    experiment_dir: str,
    allowed_seeds: Optional[List[int]] = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, np.ndarray], List[int]]:
    """
    Returns
    -------
    (display_pairs, oods, seeds_used)
        display_pairs : (raw, display) name pairs for agents that have data,
                        in canonical plot order
        oods          : agent_raw_name -> array of per-seed values in [0,1]
        seeds_used    : list of seed numbers actually loaded
    """
    ckpt_dir = os.path.join(experiment_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"no checkpoints/ directory in {experiment_dir}")

    seed_files = _find_seed_pkls(ckpt_dir, allowed_seeds)
    if not seed_files:
        raise FileNotFoundError(f"no finalized seed_*.pkl in {ckpt_dir}")

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
            oods[raw].append(r["ood_accuracy"])

    display_pairs = [(raw, disp) for raw, disp in DISPLAY_NAME if oods[raw]]
    oods_arr = {raw: np.asarray(oods[raw], dtype=float) for raw, _ in display_pairs}
    return display_pairs, oods_arr, seeds_loaded


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _configure_rc() -> None:
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_explicit_query_acc(
    experiment_dir: str,
    output_path: Optional[str] = None,
    allowed_seeds: Optional[List[int]] = None,
    annotate: bool = True,
    figsize: Tuple[float, float] = (5.6, 3.4),
) -> Tuple[str, str]:
    """Render a single-bar Explicit Query Accuracy figure and return (pdf, png)."""
    _configure_rc()

    display_pairs, oods, seeds_used = load_explicit_query_accs(
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

    means = np.array([_mean_std(oods[r])[0] for r in raw_names])
    stds  = np.array([_mean_std(oods[r])[1] for r in raw_names])
    colors = [_FAMILY_PALETTE.get(d, "#444444") for d in disp_names]

    x = np.arange(len(disp_names))
    width = 0.62

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        x, means, width,
        yerr=stds, capsize=2.8,
        color=colors, edgecolor="black", linewidth=0.5,
        error_kw={"linewidth": 0.8, "ecolor": "black"},
    )

    if annotate:
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.012,
                    f"{m * 100:.1f}", ha="center", va="bottom",
                    fontsize=8, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(disp_names, rotation=22, ha="right")
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Explicit Query Accuracy")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(round(v * 100))}%")
    )
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.grid(True, axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if output_path is None:
        out_dir = os.path.join(experiment_dir, "images")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, "explicit_query_acc")
    elif os.path.isdir(output_path) or output_path.endswith("/"):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.join(output_path, "explicit_query_acc")
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
                        "Defaults to <dir>/images/explicit_query_acc.{pdf,png}")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Optional seed whitelist (default: all finalized).")
    p.add_argument("--no-annotate", dest="annotate", action="store_false",
                   help="Suppress per-bar numeric labels (on by default).")
    p.set_defaults(annotate=True)
    p.add_argument("--width", type=float, default=5.6,
                   help="Figure width in inches (default: 5.6).")
    p.add_argument("--height", type=float, default=3.4,
                   help="Figure height in inches (default: 3.4).")
    args = p.parse_args()

    plot_explicit_query_acc(
        experiment_dir=args.dir,
        output_path=args.output,
        allowed_seeds=args.seeds,
        annotate=args.annotate,
        figsize=(args.width, args.height),
    )


if __name__ == "__main__":
    main()
