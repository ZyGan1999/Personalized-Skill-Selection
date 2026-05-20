"""
plot_appendix_rolling_acc.py
Appendix figure: rolling-accuracy convergence curves for the six main
experiments (3 backbones x {one-hot, soft0.3}).

Layout: 2 rows x 3 columns. Rows = preference regime (one-hot, soft0.3);
columns = backbone (Qwen3-30B, DeepSeek-V4-Flash, GPT-5.2). Each panel
plots nine agent curves with 95% CI band; an Oracle reference line marks
the theoretical upper bound implied by the user's preference distribution.

Reads per-round rewards from `<dir>/checkpoints/seed_*.pkl`,
computes per-seed rolling accuracy with a window of W rounds (default 50),
and averages across seeds with t-distribution 95% CI.

Output: figures/appendix_rolling_acc.{pdf,png} by default.
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from collections import Counter
from glob import glob
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats as scipy_stats
except ImportError:
    print("ERROR: matplotlib / numpy / scipy not installed.", file=sys.stderr)
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


def _rolling(records, T: int, window: int) -> np.ndarray:
    """Rolling mean reward, length T - window + 1."""
    user_ids = sorted({r.user_id for r in records})
    uid_idx = {uid: i for i, uid in enumerate(user_ids)}
    mat = np.zeros((len(user_ids), T), dtype=np.float64)
    for rec in records:
        mat[uid_idx[rec.user_id], rec.round_idx] = rec.reward
    mean_reward = mat.mean(axis=0)
    kernel = np.ones(window) / window
    return np.convolve(mean_reward, kernel, mode="valid")


def _oracle_accuracy(records) -> float:
    """Theoretical ceiling: always pick the most frequent true_tool per (user, domain)."""
    ud_counter: Dict[Tuple[int, str], Counter] = {}
    for r in records:
        ud_counter.setdefault((r.user_id, r.domain), Counter())[r.true_tool] += 1
    correct = total = 0
    for r in records:
        best = ud_counter[(r.user_id, r.domain)].most_common(1)[0][0]
        if r.true_tool == best:
            correct += 1
        total += 1
    return correct / max(total, 1)


def load_panels(base_dir: str, window: int) -> Dict[Tuple[str, str], dict]:
    out: Dict[Tuple[str, str], dict] = {}
    for model_tag, _ in BACKBONES:
        for regime_tag, _ in REGIMES:
            exp_dir = os.path.join(base_dir, f"main-{regime_tag}-{model_tag}")
            ckpt_dir = os.path.join(exp_dir, "checkpoints")
            if not os.path.isdir(ckpt_dir):
                print(f"  [warn] missing: {exp_dir}", file=sys.stderr)
                continue
            per_agent: Dict[str, List[np.ndarray]] = {raw: [] for raw, _ in DISPLAY_NAME}
            oracles: List[float] = []
            T_seed = 0
            for path in _find_seed_pkls(ckpt_dir):
                try:
                    with open(path, "rb") as f:
                        sr = pickle.load(f)
                except Exception as e:
                    print(f"  [warn] {path}: {e}", file=sys.stderr)
                    continue
                T_seed = 1 + max((r.round_idx for ar in sr.agent_results
                                  for r in ar.records), default=0)
                W = min(window, T_seed)
                for ar in sr.agent_results:
                    if ar.agent_name in per_agent:
                        per_agent[ar.agent_name].append(_rolling(ar.records, T_seed, W))
                # oracle from any agent (same query stream)
                if sr.agent_results:
                    oracles.append(_oracle_accuracy(sr.agent_results[0].records))
            stacked = {raw: np.stack(arrs) for raw, arrs in per_agent.items() if arrs}
            out[(model_tag, regime_tag)] = {
                "data": stacked,
                "oracle": float(np.mean(oracles)) if oracles else None,
                "T": T_seed,
                "window": min(window, T_seed),
            }
    return out


def _ci95(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    if n <= 1:
        return mean, mean
    se = arr.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
    return mean - t_crit * se, mean + t_crit * se


def _configure_rc() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.1,
    })


def plot(base_dir: str = "paper-exp",
         output_path: Optional[str] = None,
         window: int = 50,
         figsize: Tuple[float, float] = (12.0, 6.4)) -> Tuple[str, str]:
    _configure_rc()
    panels = load_panels(base_dir, window)

    fig, axes = plt.subplots(len(REGIMES), len(BACKBONES),
                             figsize=figsize, sharex=False, sharey=True)

    for r, (regime_tag, regime_label) in enumerate(REGIMES):
        for c, (model_tag, model_label) in enumerate(BACKBONES):
            ax = axes[r, c]
            cell = panels.get((model_tag, regime_tag), {})
            data = cell.get("data", {})
            if not data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=9)
                ax.set_xticks([])
                continue

            T_eff = next(iter(data.values())).shape[1]
            W = cell.get("window", window)
            rounds = np.arange(W, W + T_eff)
            for raw, disp in DISPLAY_NAME:
                if raw not in data:
                    continue
                arr = data[raw]
                color = _PALETTE.get(disp, "#444444")
                lw = 1.7 if disp == "Bandit-as-Override" else 1.0
                mean = arr.mean(axis=0)
                lo, hi = _ci95(arr)
                ax.plot(rounds, mean, color=color, linewidth=lw, label=disp,
                        zorder=3 if disp == "Bandit-as-Override" else 2)
                ax.fill_between(rounds, lo, hi, color=color, alpha=0.12, linewidth=0)

            # Oracle reference
            oracle = cell.get("oracle")
            if oracle is not None:
                ax.axhline(oracle, color="black", linestyle="--",
                           linewidth=0.9, alpha=0.6,
                           label=f"Oracle ({oracle:.0%})" if r == 0 and c == 0 else None)

            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="both", linestyle="--", alpha=0.45, zorder=0)
            ax.set_axisbelow(True)

            if c == 0:
                ax.set_ylabel(f"{regime_label}\n\nRolling Accuracy",
                              fontsize=9.5, multialignment="center")
            if r == 0:
                ax.set_title(model_label, fontsize=10.5, pad=6)
            if r == len(REGIMES) - 1:
                ax.set_xlabel("Round")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=5, frameon=False, handlelength=2.2,
               handletextpad=0.6, columnspacing=1.8)

    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    if output_path is None:
        os.makedirs("figures", exist_ok=True)
        base = "figures/appendix_rolling_acc"
    elif os.path.isdir(output_path) or output_path.endswith("/"):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.join(output_path, "appendix_rolling_acc")
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
    p.add_argument("--window", type=int, default=50,
                   help="Rolling-window size in rounds (default 50).")
    p.add_argument("--width", type=float, default=12.0)
    p.add_argument("--height", type=float, default=6.4)
    args = p.parse_args()
    plot(base_dir=args.base_dir, output_path=args.output,
         window=args.window, figsize=(args.width, args.height))


if __name__ == "__main__":
    main()
