#!/usr/bin/env python3
"""
plot_progress.py
Read mid-seed checkpoint from a running experiment and generate metrics plots.

Usage (while main.py is still running):
    python plot_progress.py --dir gpt5.2-toolbench60-shared-t3

This reads the latest mid-seed checkpoint (agent state + partial records),
constructs a temporary MultiSeedResult, and generates all standard plots.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from in-progress experiment")
    parser.add_argument("--dir", required=True, help="Experiment output directory")
    args = parser.parse_args()

    output_dir = Path(args.dir)
    ckpt_dir = output_dir / "checkpoints"

    if not ckpt_dir.exists():
        print(f"No checkpoints directory found at {ckpt_dir}")
        sys.exit(1)

    # Load any completed seed results
    from env import SeedResult, MultiSeedResult, AgentResult

    seed_results = []

    # 1. Load completed seeds
    for p in sorted(ckpt_dir.glob("seed_*.pkl")):
        if "_progress" in p.name:
            continue
        try:
            with open(p, "rb") as f:
                sr = pickle.load(f)
            seed_results.append(sr)
            n_records = sum(len(ar.records) for ar in sr.agent_results)
            print(f"  Loaded completed seed {sr.seed}: {n_records} records")
        except Exception as e:
            print(f"  Failed to load {p}: {e}")

    # 2. Load any in-progress seed checkpoint
    for p in sorted(ckpt_dir.glob("seed_*_progress.pkl")):
        try:
            with open(p, "rb") as f:
                ckpt = pickle.load(f)
            seed_num = int(p.name.split("_")[1])
            agents = ckpt["agents"]
            partial_records = ckpt["partial_records"]
            completed_users = ckpt["completed_user_count"]

            # Reconstruct AgentResults
            agent_results = []
            for i, agent in enumerate(agents):
                ar = AgentResult(agent_name=agent.name, records=partial_records[i])
                agent_results.append(ar)

            # We need users for preference metrics — regenerate them
            # (they're deterministic from seed)
            import data_gen
            if (output_dir / "..").resolve() != Path.cwd().resolve():
                # Try to load benchmark config if it was used
                for bf in output_dir.glob("*.json"):
                    pass  # benchmark config is loaded via main.py, not stored in output

            users = data_gen.generate_users(n_users=completed_users, seed=seed_num)

            n_records = sum(len(ar.records) for ar in agent_results)
            print(f"  Loaded in-progress seed {seed_num}: {completed_users} users, {n_records} records")

            # Determine T from records
            if agent_results and agent_results[0].records:
                max_round = max(r.round_idx for r in agent_results[0].records) + 1
            else:
                max_round = 50

            domain_preds = []  # Not available from checkpoint

            sr = SeedResult(
                seed=seed_num,
                agent_results=agent_results,
                users=users,
                trained_agents=agents,
                domain_predictions=domain_preds,
            )
            seed_results.append(sr)
        except Exception as e:
            print(f"  Failed to load {p}: {e}")

    if not seed_results:
        print("No data found to plot.")
        sys.exit(1)

    # Determine T and n_users from records
    all_records = seed_results[0].agent_results[0].records
    if not all_records:
        print("No records found.")
        sys.exit(1)

    T = max(r.round_idx for r in all_records) + 1
    n_users = len(set(r.user_id for r in all_records))
    agent_names = [ar.agent_name for ar in seed_results[0].agent_results]

    print(f"\n  Total: {len(seed_results)} seed(s), {n_users} users, T={T}, {len(agent_names)} agents")
    print(f"  Agents: {', '.join(agent_names)}")

    multi = MultiSeedResult(
        seed_results=seed_results,
        agent_names=agent_names,
        T=T,
        n_users=n_users,
    )

    # Generate plots
    import metrics as _metrics_mod
    _metrics_mod.set_output_dir(str(output_dir))

    from metrics import (
        plot_cumulative_regret_ci,
        plot_rolling_accuracy_ci,
        plot_ood_robustness,
        plot_per_domain_accuracy,
        plot_domain_classification_accuracy,
        plot_test_accuracy,
        print_summary,
    )

    print(f"\nGenerating plots to {output_dir}/images/ ...")
    plot_cumulative_regret_ci(multi)
    plot_rolling_accuracy_ci(multi)
    plot_ood_robustness(multi)
    plot_per_domain_accuracy(multi)
    plot_domain_classification_accuracy(multi)
    # Test accuracy: only available for fully completed seeds
    if any(sr.test_results for sr in seed_results if hasattr(sr, 'test_results')):
        plot_test_accuracy(multi)
    print_summary(multi)
    print("Done.")


if __name__ == "__main__":
    main()
