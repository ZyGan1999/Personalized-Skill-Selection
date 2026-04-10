"""
metrics.py
Metric computation and visualization for the personalized tool-selection benchmark.

Single-seed metrics
-------------------
  cumulative_regret(records, T)
  rolling_accuracy(records, T, window)
  ood_accuracy(records, late_cutoff)

Multi-seed metrics (with confidence intervals)
----------------------------------------------
  plot_cumulative_regret_ci(multi_result)   — regret curve + 95% CI band
  plot_rolling_accuracy_ci(multi_result)    — rolling accuracy + 95% CI band
  plot_ood_robustness(multi_result)         — OOD bar chart + error bars
  plot_preference_recovery(multi_result)    — bar chart of preference recovery rate
  plot_per_domain_accuracy(multi_result)    — heatmap: agent × domain accuracy

Summary & export
----------------
  print_summary(multi_result)
  export_csv(multi_result, path)            — raw records to CSV
  export_json_summary(multi_result, path)   — aggregated stats to JSON

All figures are saved to ./images/. plt.show() is never called.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats as scipy_stats

from env import MultiSeedResult

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


def set_output_dir(output_dir: str) -> None:
    """Set the directory where figures and data exports are saved."""
    global IMAGES_DIR
    IMAGES_DIR = os.path.join(output_dir, "images")


def _ensure_images_dir() -> None:
    os.makedirs(IMAGES_DIR, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> str:
    _ensure_images_dir()
    base = os.path.splitext(filename)[0]

    # Save PDF
    pdf_path = os.path.join(IMAGES_DIR, f"{base}.pdf")
    fig.tight_layout()
    fig.savefig(pdf_path)
    print(f"[metrics] Saved: {pdf_path}")

    # Save PNG
    png_path = os.path.join(IMAGES_DIR, f"{base}.png")
    fig.savefig(png_path, dpi=150)
    print(f"[metrics] Saved: {png_path}")

    plt.close(fig)
    return pdf_path


# Colour palette (consistent across all plots)
_PALETTE = plt.cm.tab10(np.linspace(0, 0.9, 10))


def _agent_color(idx: int) -> np.ndarray:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Single-seed scalar metrics
# ---------------------------------------------------------------------------


def cumulative_regret(records: List, T: int) -> np.ndarray:
    """Cumulative regret (1 - mean_reward per round) averaged across all users."""
    n_users = len(set(r.user_id for r in records))
    mat = np.zeros((n_users, T), dtype=np.float64)
    user_ids = sorted(set(r.user_id for r in records))
    uid_idx = {uid: i for i, uid in enumerate(user_ids)}
    for rec in records:
        mat[uid_idx[rec.user_id], rec.round_idx] = rec.reward
    mean_reward = mat.mean(axis=0)
    return np.cumsum(1.0 - mean_reward)


def rolling_accuracy(records: List, T: int, window: int = 10) -> np.ndarray:
    """Rolling accuracy (mean reward over last `window` rounds), averaged across users."""
    n_users = len(set(r.user_id for r in records))
    mat = np.zeros((n_users, T), dtype=np.float64)
    user_ids = sorted(set(r.user_id for r in records))
    uid_idx = {uid: i for i, uid in enumerate(user_ids)}
    for rec in records:
        mat[uid_idx[rec.user_id], rec.round_idx] = rec.reward
    mean_reward = mat.mean(axis=0)
    kernel = np.ones(window) / window
    return np.convolve(mean_reward, kernel, mode="valid")  # length T - window + 1


def ood_accuracy(records: List, late_cutoff: float = 0.6) -> Tuple[float, int]:
    """
    Accuracy on OOD queries in late rounds (round_idx >= late_cutoff * T).
    Returns (accuracy, n_ood_records).
    """
    if not records:
        return 0.0, 0
    T = max(r.round_idx for r in records) + 1
    cutoff = int(late_cutoff * T)
    late_ood = [r for r in records if r.is_ood and r.round_idx >= cutoff]
    if not late_ood:
        return 0.0, 0
    return float(np.mean([r.reward for r in late_ood])), len(late_ood)


def convergence_round(records: List, T: int, threshold: float = 0.90, window: int = 10) -> Optional[int]:
    """First round (1-indexed) at which rolling accuracy first exceeds threshold."""
    window = min(window, T)
    ra = rolling_accuracy(records, T, window)
    for i, acc in enumerate(ra):
        if acc >= threshold:
            return i + window  # convert back to 1-indexed round number
    return None


# ---------------------------------------------------------------------------
# Multi-seed aggregation helpers
# ---------------------------------------------------------------------------


def _multi_seed_regret(multi: MultiSeedResult) -> Dict[str, np.ndarray]:
    """
    Compute cumulative regret per seed per agent.
    Returns {agent_name: array of shape (n_seeds, T)}.
    """
    out: Dict[str, List[np.ndarray]] = {name: [] for name in multi.agent_names}
    for sr in multi.seed_results:
        for ar in sr.agent_results:
            out[ar.agent_name].append(cumulative_regret(ar.records, multi.T))
    return {name: np.stack(arrs) for name, arrs in out.items()}


def _multi_seed_rolling(multi: MultiSeedResult, window: int = 10) -> Dict[str, np.ndarray]:
    """Rolling accuracy per seed per agent. Shape: (n_seeds, T - window + 1)."""
    out: Dict[str, List[np.ndarray]] = {name: [] for name in multi.agent_names}
    for sr in multi.seed_results:
        for ar in sr.agent_results:
            out[ar.agent_name].append(rolling_accuracy(ar.records, multi.T, window))
    return {name: np.stack(arrs) for name, arrs in out.items()}


def _ci95(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    95% confidence interval for the mean across the first axis.
    Uses t-distribution (valid for small n_seeds).
    Returns (lower_bound, upper_bound) arrays of the same shape as arr[0].
    """
    n = arr.shape[0]
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1) if n > 1 else 0.0
    return mean - t_crit * se, mean + t_crit * se


# ---------------------------------------------------------------------------
# Plot 1: Cumulative Regret with 95% CI
# ---------------------------------------------------------------------------


def plot_cumulative_regret_ci(
    multi: MultiSeedResult,
    filename: str = "cumulative_regret.pdf",
) -> str:
    regret_data = _multi_seed_regret(multi)
    rounds = np.arange(1, multi.T + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, name in enumerate(multi.agent_names):
        arr = regret_data[name]          # (n_seeds, T)
        mean = arr.mean(axis=0)
        lo, hi = _ci95(arr)
        color = _agent_color(idx)
        ax.plot(rounds, mean, label=name, color=color, linewidth=2)
        ax.fill_between(rounds, lo, hi, alpha=0.15, color=color)

    ax.set_xlabel("Round (T)", fontsize=12)
    ax.set_ylabel("Cumulative Regret", fontsize=12)
    ax.set_title(
        f"Cumulative Regret over Time\n"
        f"({multi.n_users} users, {multi.n_seeds} seeds, 95% CI shaded)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Plot 2: Rolling Accuracy with 95% CI
# ---------------------------------------------------------------------------


def plot_rolling_accuracy_ci(
    multi: MultiSeedResult,
    window: int = 10,
    threshold: float = 0.90,
    filename: str = "rolling_accuracy.pdf",
) -> str:
    # Clamp window to at most T so there is always at least one data point
    window = min(window, multi.T)
    rolling_data = _multi_seed_rolling(multi, window)
    rounds = np.arange(window, multi.T + 1)

    # Compute oracle accuracy: best possible accuracy if always picking the
    # most frequent true_tool per (user, domain) in the query pool.
    # This represents the theoretical upper bound given the preference distribution.
    oracle_accs = []
    for sr in multi.seed_results:
        correct = 0
        total = 0
        for ar in sr.agent_results[:1]:  # just count from any agent's records
            from collections import Counter
            # Group by (user_id, domain) and find the most common true_tool
            ud_tools = {}  # (user_id, domain) -> Counter of true_tools
            for r in ar.records:
                key = (r.user_id, r.domain)
                if key not in ud_tools:
                    ud_tools[key] = Counter()
                ud_tools[key][r.true_tool] += 1
            for r in ar.records:
                key = (r.user_id, r.domain)
                best_tool = ud_tools[key].most_common(1)[0][0]
                if r.true_tool == best_tool:
                    correct += 1
                total += 1
        oracle_accs.append(correct / max(total, 1))
    oracle_mean = float(np.mean(oracle_accs))

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, name in enumerate(multi.agent_names):
        arr = rolling_data[name]         # (n_seeds, T_eff)
        mean = arr.mean(axis=0)
        lo, hi = _ci95(arr)
        color = _agent_color(idx)
        ax.plot(rounds, mean, label=name, color=color, linewidth=2)
        ax.fill_between(rounds, lo, hi, alpha=0.15, color=color)

    ax.axhline(oracle_mean, color="green", linestyle="-.", linewidth=1.5,
               label=f"Oracle ({oracle_mean:.0%})", alpha=0.7)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"{threshold:.0%} threshold")
    ax.set_xlabel("Round (T)", fontsize=12)
    ax.set_ylabel(f"Rolling Accuracy (window={window})", fontsize=12)
    ax.set_title(
        f"Convergence Rate\n"
        f"({multi.n_users} users, {multi.n_seeds} seeds, 95% CI shaded)",
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Plot 3: OOD Robustness (bar chart with error bars)
# ---------------------------------------------------------------------------


def plot_ood_robustness(
    multi: MultiSeedResult,
    late_cutoff: float = 0.6,
    filename: str = "ood_robustness.pdf",
) -> str:
    """Bar chart comparing OOD accuracy ± std across seeds for each agent."""
    means, stds = [], []
    for name in multi.agent_names:
        accs = []
        for sr in multi.seed_results:
            for ar in sr.agent_results:
                if ar.agent_name == name:
                    acc, _ = ood_accuracy(ar.records, late_cutoff)
                    accs.append(acc)
                    break
        means.append(float(np.mean(accs)))
        stds.append(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0)

    x = np.arange(len(multi.agent_names))
    colors = [_agent_color(i) for i in range(len(multi.agent_names))]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.8, error_kw={"linewidth": 1.5})

    for bar, m, s in zip(bars, means, stds):
        label = f"{m:.1%}"
        if s > 0:
            label += f"\n±{s:.1%}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                label, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(multi.agent_names, fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("OOD Accuracy (late rounds)", fontsize=12)
    ax.set_title(
        f"OOD Robustness — Late Rounds (after round {late_cutoff:.0%}×T)\n"
        f"({multi.n_seeds} seeds, error bars = ±1 std)",
        fontsize=13,
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Plot 4: Preference Recovery Rate
# ---------------------------------------------------------------------------


def plot_preference_recovery(
    recovery_rates: Dict[str, Dict[str, float]],
    agent_names: List[str],
    filename: str = "preference_recovery.pdf",
    soft_mode: bool = False,
) -> str:
    """
    Bar chart showing preference recovery quality after training.

    In one-hot mode: single bar chart of recovery rate.
    In soft mode: 3-panel subplot (Cosine Similarity, KL Divergence, Spearman Correlation).
    """
    if not soft_mode:
        # One-hot mode: single bar chart
        def _get_rate(v):
            return v.get("recovery_rate", 0.0) if isinstance(v, dict) else float(v)
        vals = [_get_rate(recovery_rates.get(name, 0.0)) for name in agent_names]
        x = np.arange(len(agent_names))
        colors = [_agent_color(i) for i in range(len(agent_names))]

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Preference Recovery Rate", fontsize=12)
        ax.set_title(
            "Preference Recovery Rate after Training\n"
            "(fraction of users where argmax learned dist = true preferred tool)",
            fontsize=12,
        )
        ax.axhline(1.0 / 4, color="gray", linestyle=":", linewidth=1.2, label="Random baseline (1/K)")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        return _save(fig, filename)

    # Soft mode: 3-panel subplot
    metric_configs = [
        ("cosine_sim", "Cosine Similarity", (0, 1.15), True, "{:.3f}"),
        ("kl_divergence", "KL Divergence", None, False, "{:.3f}"),
        ("spearman", "Spearman Rank Correlation", (-0.3, 1.15), True, "{:.3f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(agent_names))
    colors = [_agent_color(i) for i in range(len(agent_names))]

    for ax, (key, ylabel, ylim, higher_better, fmt) in zip(axes, metric_configs):
        def _get_metric(v, k=key):
            return v.get(k, 0.0) if isinstance(v, dict) else 0.0
        vals = [_get_metric(recovery_rates.get(name, {})) for name in agent_names]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.8)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    fmt.format(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, fontsize=9, rotation=15, ha="right")
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontsize=11)
        arrow = "↑" if higher_better else "↓"
        ax.set_title(f"{ylabel} ({arrow} better)", fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Distribution Alignment after Training (Soft Preferences)", fontsize=13, y=1.02)
    fig.tight_layout()
    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Plot 5: Per-Domain Accuracy Heatmap
# ---------------------------------------------------------------------------


def plot_per_domain_accuracy(
    multi: MultiSeedResult,
    filename: str = "per_domain_accuracy.pdf",
) -> str:
    """
    Heatmap of mean accuracy per (agent, domain) pair, averaged over all
    rounds and seeds.  Highlights domain-specific strengths and weaknesses.
    """
    from data_gen import DOMAINS

    domain_list = list(DOMAINS.keys())
    agent_list = multi.agent_names

    # acc_matrix[i_agent, j_domain] = mean accuracy
    acc_matrix = np.zeros((len(agent_list), len(domain_list)), dtype=np.float64)

    for i, agent_name in enumerate(agent_list):
        for j, domain in enumerate(domain_list):
            accs = []
            for sr in multi.seed_results:
                for ar in sr.agent_results:
                    if ar.agent_name == agent_name:
                        domain_recs = [r for r in ar.records if r.domain == domain]
                        if domain_recs:
                            accs.append(np.mean([r.reward for r in domain_recs]))
                        break
            acc_matrix[i, j] = float(np.mean(accs)) if accs else 0.0

    fig, ax = plt.subplots(figsize=(max(8, len(domain_list) * 1.5), max(4, len(agent_list))))
    im = ax.imshow(acc_matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Accuracy")

    ax.set_xticks(np.arange(len(domain_list)))
    ax.set_yticks(np.arange(len(agent_list)))
    ax.set_xticklabels(domain_list, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(agent_list, fontsize=10)
    ax.set_title(f"Per-Domain Accuracy (mean over all rounds, {multi.n_seeds} seeds)", fontsize=13)

    for i in range(len(agent_list)):
        for j in range(len(domain_list)):
            val = acc_matrix[i, j]
            text_color = "black" if val < 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold")

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Plot 6: Domain Classification Accuracy (for Bandit+CoT with domain inference)
# ---------------------------------------------------------------------------


def plot_domain_classification_accuracy(
    multi: MultiSeedResult,
    filename: str = "domain_classification_accuracy.pdf",
) -> str:
    """
    Bar chart showing domain classification accuracy.
    Reads from SeedResult.domain_predictions (shared classifier) or
    falls back to agent.domain_predictions (legacy per-agent tracking).
    """
    seed_accuracies = []

    for sr in multi.seed_results:
        # Prefer shared domain predictions from env.py
        preds = getattr(sr, 'domain_predictions', None)
        if preds:
            correct = sum(1 for true_d, pred_d in preds if true_d == pred_d)
            total = len(preds)
            seed_accuracies.append(correct / total if total > 0 else 0.0)
        else:
            # Legacy: check agents for domain_predictions
            for agent in sr.trained_agents:
                if hasattr(agent, 'domain_predictions') and agent.domain_predictions:
                    correct = sum(1 for true_d, pred_d in agent.domain_predictions if true_d == pred_d)
                    total = len(agent.domain_predictions)
                    seed_accuracies.append(correct / total if total > 0 else 0.0)
                    break

    if not seed_accuracies:
        return ""

    mean_acc = np.mean(seed_accuracies)
    std_acc = np.std(seed_accuracies) if len(seed_accuracies) > 1 else 0.0
    n_domains = len(multi.seed_results[0].agent_results[0].records[0].domain
                     if multi.seed_results else 5)

    from data_gen import DOMAINS
    n_domains = len(DOMAINS)
    random_baseline = 1.0 / max(n_domains, 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    bar = ax.bar(["Shared Domain\nClassifier"], [mean_acc], yerr=[std_acc],
                  capsize=5, color='steelblue', edgecolor='black', linewidth=0.8)
    ax.text(bar[0].get_x() + bar[0].get_width() / 2, bar[0].get_height() + 0.02,
            f"{mean_acc:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Domain Classification Accuracy", fontsize=12)
    ax.set_title("LLM Domain Inference Accuracy", fontsize=13, fontweight="bold")
    ax.axhline(random_baseline, color="gray", linestyle=":", linewidth=1.2,
               label=f"Random (1/{n_domains})")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    return _save(fig, filename)


# ---------------------------------------------------------------------------
# Plot 7: Preference Alignment Heatmap (per-user, per-domain)
# ---------------------------------------------------------------------------


# Agents whose get_learned_distribution is non-trivial (bandit-based)
_BANDIT_AGENTS = {"Pure-Bandit", "Bandit+CoT"}


def _build_alignment_matrices(
    multi: MultiSeedResult,
    metric_fn,
) -> Dict[str, np.ndarray]:
    """
    Build per-(user, domain) matrices for bandit agents using a custom metric_fn.

    metric_fn(learned: Dict[str, float], user: UserPersona, domain: str,
              domain_tools: List[str]) -> float
    """
    from data_gen import DOMAINS, STANDARD_QUERIES

    domain_list = list(DOMAINS.keys())
    n_domains = len(domain_list)
    bandit_names = [n for n in multi.agent_names if n in _BANDIT_AGENTS]

    alignment: Dict[str, np.ndarray] = {}
    for agent_name in bandit_names:
        seed_matrices = []
        for sr in multi.seed_results:
            agent = next((a for a in sr.trained_agents if a.name == agent_name), None)
            if agent is None:
                continue
            mat = np.zeros((multi.n_users, n_domains), dtype=np.float64)
            for ui, user in enumerate(sr.users):
                for dj, domain in enumerate(domain_list):
                    domain_tools = DOMAINS[domain]
                    sample_qs = STANDARD_QUERIES.get(domain, [])[:10]
                    learned = agent.get_learned_distribution(
                        user.user_id, domain, domain_tools, sample_qs
                    )
                    mat[ui, dj] = metric_fn(learned, user, domain, domain_tools)
            seed_matrices.append(mat)
        if seed_matrices:
            alignment[agent_name] = np.mean(seed_matrices, axis=0)
    return alignment


def _plot_alignment_heatmap(
    multi: MultiSeedResult,
    alignment: Dict[str, np.ndarray],
    metric_label: str,
    baseline_str: str,
    norm,
    filename: str,
    fmt: str = "{:.0%}",
) -> str:
    """Render a preference alignment heatmap and save it."""
    from data_gen import DOMAINS

    domain_list = list(DOMAINS.keys())
    n_domains = len(domain_list)
    n_agents = len(alignment)
    if n_agents == 0:
        return ""

    fig, axes = plt.subplots(
        1, n_agents,
        figsize=(5 * n_agents + 2, max(5, multi.n_users * 0.35 + 1.5)),
        squeeze=False,
    )

    for col, agent_name in enumerate(alignment):
        ax = axes[0, col]
        mat = alignment[agent_name]
        im = ax.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(np.arange(n_domains))
        ax.set_xticklabels(domain_list, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(np.arange(multi.n_users))
        ax.set_yticklabels([f"u{i}" for i in range(multi.n_users)], fontsize=8)
        ax.set_title(agent_name, fontsize=12, fontweight="bold")
        for i in range(multi.n_users):
            for j in range(n_domains):
                val = mat[i, j]
                text_color = "black" if 0.15 < val < 0.75 else "white"
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=7, color=text_color)

    fig.suptitle(
        f"Preference Alignment: {metric_label}\n"
        f"({multi.n_users} users, {multi.n_seeds} seeds avg, "
        f"{baseline_str})",
        fontsize=12, y=1.02,
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), label=metric_label,
                 shrink=0.8, pad=0.04, fraction=0.03)
    return _save(fig, filename)


def plot_preference_alignment(
    multi: MultiSeedResult,
    filename: str = "preference_alignment.pdf",
) -> str:
    """
    Per-user, per-domain heatmap(s) showing alignment between learned and true
    preference distributions.

    One-hot mode: single heatmap of P(true preferred tool).
    Soft mode: three heatmaps (Cosine Similarity, KL Divergence, Spearman Rank Corr).
    """
    from matplotlib.colors import TwoSlopeNorm

    bandit_names = [n for n in multi.agent_names if n in _BANDIT_AGENTS]
    if not bandit_names:
        return ""

    soft_mode = (multi.seed_results[0].users[0].soft_preferences is not None)

    if not soft_mode:
        # One-hot mode: P(true preferred tool)
        def _pref_prob(learned, user, domain, domain_tools):
            return learned.get(user.preferences[domain], 0.0)

        alignment = _build_alignment_matrices(multi, _pref_prob)
        norm = TwoSlopeNorm(vmin=0.0, vcenter=0.25, vmax=1.0)
        return _plot_alignment_heatmap(
            multi, alignment, "P(true preferred tool)", "random baseline = 25%",
            norm, filename,
        )

    # Soft mode: three metrics
    saved = []

    # 1. Cosine Similarity
    def _cosine(learned, user, domain, domain_tools):
        true_dist = user.soft_preferences[domain]
        tv = np.array([true_dist.get(t, 0.0) for t in domain_tools])
        lv = np.array([learned.get(t, 0.0) for t in domain_tools])
        return float(np.dot(tv, lv) / (np.linalg.norm(tv) * np.linalg.norm(lv) + 1e-12))

    alignment = _build_alignment_matrices(multi, _cosine)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    saved.append(_plot_alignment_heatmap(
        multi, alignment, "Cosine Similarity", "uniform baseline ≈ 0.5",
        norm, filename,
    ))

    # 2. KL Divergence
    def _kl(learned, user, domain, domain_tools):
        eps = 1e-12
        true_dist = user.soft_preferences[domain]
        tv = np.array([true_dist.get(t, 0.0) for t in domain_tools])
        lv = np.array([learned.get(t, 0.0) for t in domain_tools])
        tv = np.clip(tv, eps, None); tv = tv / tv.sum()
        lv = np.clip(lv, eps, None); lv = lv / lv.sum()
        return float(np.sum(tv * np.log(tv / lv)))

    alignment_kl = _build_alignment_matrices(multi, _kl)
    # KL: lower is better. Use inverted colormap.
    max_kl = max(m.max() for m in alignment_kl.values()) if alignment_kl else 1.0
    norm_kl = TwoSlopeNorm(vmin=0.0, vcenter=max_kl / 2, vmax=max(max_kl, 0.01))
    saved.append(_plot_alignment_heatmap(
        multi, alignment_kl, "KL Divergence (↓ better)", f"0 = perfect match",
        norm_kl, filename.replace(".pdf", "_kl.pdf"), fmt="{:.2f}",
    ))

    # 3. Spearman Rank Correlation
    def _spearman(learned, user, domain, domain_tools):
        from scipy.stats import spearmanr
        true_dist = user.soft_preferences[domain]
        tv = np.array([true_dist.get(t, 0.0) for t in domain_tools])
        lv = np.array([learned.get(t, 0.0) for t in domain_tools])
        corr, _ = spearmanr(tv, lv)
        return float(corr) if not np.isnan(corr) else 0.0

    alignment_sp = _build_alignment_matrices(multi, _spearman)
    norm_sp = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    saved.append(_plot_alignment_heatmap(
        multi, alignment_sp, "Spearman Rank Correlation", "0 = no correlation",
        norm_sp, filename.replace(".pdf", "_spearman.pdf"), fmt="{:.2f}",
    ))

    return saved[0]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(multi: MultiSeedResult, late_cutoff: float = 0.6) -> None:
    """Print a comprehensive summary table to stdout."""
    header = (
        f"{'Agent':<22} {'FinalRegret':>12} {'±':>5} "
        f"{'ConvergeRd':>11} {'OOD_Acc':>9} {'±':>5}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for name in multi.agent_names:
        # Cumulative regret (final value across seeds)
        final_regrets = []
        conv_rounds = []
        ood_accs = []

        for sr in multi.seed_results:
            for ar in sr.agent_results:
                if ar.agent_name == name:
                    cr = cumulative_regret(ar.records, multi.T)
                    final_regrets.append(cr[-1])
                    conv = convergence_round(ar.records, multi.T)
                    conv_rounds.append(conv if conv is not None else multi.T + 1)
                    acc, _ = ood_accuracy(ar.records, late_cutoff)
                    ood_accs.append(acc)
                    break

        mean_reg = np.mean(final_regrets)
        std_reg = np.std(final_regrets, ddof=1) if len(final_regrets) > 1 else 0.0
        mean_conv = np.mean(conv_rounds)
        mean_ood = np.mean(ood_accs)
        std_ood = np.std(ood_accs, ddof=1) if len(ood_accs) > 1 else 0.0

        conv_str = f"{mean_conv:.1f}" if mean_conv <= multi.T else "N/A"
        print(
            f"{name:<22} {mean_reg:>12.2f} {std_reg:>5.2f} "
            f"{conv_str:>11} {mean_ood:>9.1%} {std_ood:>5.1%}"
        )
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------


def significance_test(
    multi: MultiSeedResult,
    baseline_name: str,
    proposed_name: str,
) -> Dict[str, float]:
    """
    Paired t-test comparing final cumulative regret of proposed vs baseline
    across seeds (paired because each seed uses the same users).

    Returns dict with t-statistic, p-value, and Cohen's d effect size.
    """
    base_regrets, prop_regrets = [], []
    for sr in multi.seed_results:
        for ar in sr.agent_results:
            if ar.agent_name == baseline_name:
                base_regrets.append(cumulative_regret(ar.records, multi.T)[-1])
            if ar.agent_name == proposed_name:
                prop_regrets.append(cumulative_regret(ar.records, multi.T)[-1])

    if len(base_regrets) < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0, "n_seeds": len(base_regrets)}

    diffs = np.array(base_regrets) - np.array(prop_regrets)  # positive → proposed is better
    t_stat, p_value = scipy_stats.ttest_1samp(diffs, popmean=0.0)
    cohens_d = float(diffs.mean() / (diffs.std(ddof=1) + 1e-9))

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": cohens_d,
        "n_seeds": len(base_regrets),
    }


# ---------------------------------------------------------------------------
# CSV export (raw records)
# ---------------------------------------------------------------------------


def export_csv(multi: MultiSeedResult, path: str = "results.csv") -> str:
    """Export all RoundRecord data to a flat CSV file for downstream analysis."""
    fieldnames = [
        "seed", "agent_name", "user_id", "round_idx",
        "domain", "is_ood", "true_tool", "selected_tool", "reward",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sr in multi.seed_results:
            for ar in sr.agent_results:
                for rec in ar.records:
                    writer.writerow({
                        "seed": sr.seed,
                        "agent_name": ar.agent_name,
                        "user_id": rec.user_id,
                        "round_idx": rec.round_idx,
                        "domain": rec.domain,
                        "is_ood": int(rec.is_ood),
                        "true_tool": rec.true_tool,
                        "selected_tool": rec.selected_tool,
                        "reward": rec.reward,
                    })
    print(f"[metrics] CSV exported: {path}")
    return path


# ---------------------------------------------------------------------------
# JSON summary export
# ---------------------------------------------------------------------------


def export_json_summary(
    multi: MultiSeedResult,
    recovery_rates: Optional[Dict] = None,
    path: str = "summary.json",
) -> str:
    """Export aggregated statistics to JSON for reproducibility."""
    summary: Dict = {
        "config": {
            "n_users": multi.n_users,
            "T": multi.T,
            "n_seeds": multi.n_seeds,
            "seeds": [sr.seed for sr in multi.seed_results],
        },
        "agents": {},
    }

    for name in multi.agent_names:
        final_regrets, ood_accs, conv_rounds = [], [], []
        for sr in multi.seed_results:
            for ar in sr.agent_results:
                if ar.agent_name == name:
                    cr = cumulative_regret(ar.records, multi.T)
                    final_regrets.append(float(cr[-1]))
                    acc, _ = ood_accuracy(ar.records)
                    ood_accs.append(acc)
                    conv = convergence_round(ar.records, multi.T)
                    conv_rounds.append(conv)
                    break

        summary["agents"][name] = {
            "final_cumulative_regret": {
                "mean": float(np.mean(final_regrets)),
                "std": float(np.std(final_regrets, ddof=1)) if len(final_regrets) > 1 else 0.0,
                "per_seed": final_regrets,
            },
            "ood_accuracy_late": {
                "mean": float(np.mean(ood_accs)),
                "std": float(np.std(ood_accs, ddof=1)) if len(ood_accs) > 1 else 0.0,
            },
            "convergence_round": {
                "mean": float(np.mean([c for c in conv_rounds if c is not None]))
                if any(c is not None for c in conv_rounds) else None,
            },
            "preference_recovery_rate": recovery_rates.get(name) if recovery_rates else None,
        }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[metrics] JSON summary exported: {path}")
    return path


if __name__ == "__main__":
    print("metrics.py: run main.py to execute the full benchmark.")
