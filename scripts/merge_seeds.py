"""
merge_seeds.py
Load seed checkpoints from one or more directories/files and regenerate
all metrics + plots as if they had been run together.

Usage
-----
# Merge soft GPT-5.2: original seed 0 + new seeds 1 and 2
python merge_seeds.py \\
    --ckpts gpt5.2-toolbench60-shared-t3-override-greedy-soft/checkpoints/seed_0.pkl \\
            paper-exp/main-soft0.3-gpt-5.2/checkpoints/seed_1.pkl \\
            paper-exp/main-soft0.3-gpt-5.2/checkpoints/seed_2.pkl \\
    --output-dir paper-exp/merged-soft0.3-gpt-5.2 \\
    --soft-preferences

# Merge one-hot GPT-5.2: seeds 1 and 2 only (seed 0 has incomplete Bandit+Override)
python merge_seeds.py \\
    --ckpts paper-exp/main-onehot-gpt-5.2/checkpoints/seed_1.pkl \\
            paper-exp/main-onehot-gpt-5.2/checkpoints/seed_2.pkl \\
    --output-dir paper-exp/merged-onehot-gpt-5.2

Alternatively, point to a directory and the script picks up all seed_N.pkl files:
python merge_seeds.py \\
    --dirs gpt5.2-toolbench60-shared-t3-override-greedy-soft \\
           paper-exp/main-soft0.3-gpt-5.2 \\
    --output-dir paper-exp/merged-soft0.3-gpt-5.2 \\
    --soft-preferences
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from glob import glob
from typing import List


def load_ckpt(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge seed checkpoints and regenerate metrics")
    parser.add_argument(
        "--ckpts", nargs="+", default=[],
        help="Explicit paths to seed_N.pkl checkpoint files",
    )
    parser.add_argument(
        "--dirs", nargs="+", default=[],
        help="Directories to scan for seed_N.pkl files (checks <dir>/checkpoints/seed_*.pkl)",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for merged results (images/, summary.json, etc.)",
    )
    parser.add_argument(
        "--soft-preferences", action="store_true",
        help="Treat as soft-preference experiment (enables KL/Spearman plots)",
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        help="Path to benchmark JSON (needed if built-in tool list differs from checkpoint)",
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Export raw records to results.csv",
    )
    parser.add_argument(
        "--no-significance", action="store_true",
        help="Skip significance tests (use when fewer than 3 seeds)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Collect checkpoint paths
    # ------------------------------------------------------------------
    ckpt_paths: List[str] = list(args.ckpts)
    for d in args.dirs:
        pattern = os.path.join(d, "checkpoints", "seed_*.pkl")
        found = sorted(glob(pattern))
        if not found:
            print(f"  WARNING: no seed_*.pkl found in {pattern}", flush=True)
        ckpt_paths.extend(found)

    if not ckpt_paths:
        print("ERROR: no checkpoint files specified (use --ckpts or --dirs)", file=sys.stderr)
        sys.exit(1)

    # Deduplicate while preserving order
    seen = set()
    unique_paths = []
    for p in ckpt_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    ckpt_paths = unique_paths

    print(f"Loading {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {p}")

    # ------------------------------------------------------------------
    # 2. Load checkpoints and validate
    # ------------------------------------------------------------------
    if args.benchmark:
        from data_gen import load_benchmark
        load_benchmark(args.benchmark)

    seed_results = []
    agent_names = None
    T = None
    n_users = None

    for path in ckpt_paths:
        if not os.path.exists(path):
            print(f"  ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        sr = load_ckpt(path)
        names = [ar.agent_name for ar in sr.agent_results]
        records_per_agent = {ar.agent_name: len(ar.records) for ar in sr.agent_results}

        # Infer T and n_users from first complete agent
        max_records = max(records_per_agent.values())

        # Detect incomplete agents (record count lower than the max)
        incomplete = {
            name: cnt for name, cnt in records_per_agent.items()
            if cnt < max_records
        }
        if incomplete:
            print(f"\n  WARNING: seed {sr.seed} ({os.path.basename(path)}) has incomplete agents:")
            for name, cnt in incomplete.items():
                expected = max_records
                users_done = cnt // (max_records // max(1, len(sr.users) if hasattr(sr, 'users') else 50))
                print(f"    {name}: {cnt} records (expected {expected})")
            print("  → These agents will be EXCLUDED from this seed to avoid biasing metrics.")
            print("    You can re-run just this seed to get complete data.")
            # Remove incomplete agents from this seed's records
            sr.agent_results = [
                ar for ar in sr.agent_results if ar.agent_name not in incomplete
            ]
            names = [ar.agent_name for ar in sr.agent_results]

        # Validate consistency across seeds
        if agent_names is None:
            agent_names = names
        elif agent_names != names:
            print(f"\n  WARNING: agent list mismatch for seed {sr.seed}.")
            print(f"    Expected: {agent_names}")
            print(f"    Got:      {names}")
            print("  → Using intersection of agents present in all seeds.")
            agent_names = [n for n in agent_names if n in names]

        cur_T = max(records_per_agent.values())
        cur_users = len(sr.users) if hasattr(sr, 'users') else 50

        if T is None:
            T = cur_T // cur_users
            n_users = cur_users
        else:
            inferred_T = cur_T // cur_users
            if inferred_T != T or cur_users != n_users:
                print(
                    f"  WARNING: seed {sr.seed} has T={inferred_T}, n_users={cur_users} "
                    f"(expected T={T}, n_users={n_users}). Skipping consistency check."
                )

        seed_results.append(sr)
        print(f"  Loaded seed {sr.seed}: {len(sr.agent_results)} agents, "
              f"T={T}, n_users={n_users}, test_results={'yes' if sr.test_results else 'no'}")

    if not agent_names:
        print("ERROR: no agents in common across seeds", file=sys.stderr)
        sys.exit(1)

    # Filter all seeds to only keep the common agent set
    for sr in seed_results:
        sr.agent_results = [ar for ar in sr.agent_results if ar.agent_name in agent_names]

    print(f"\nFinal: {len(seed_results)} seeds × {len(agent_names)} agents × T={T}, n_users={n_users}")
    print(f"Agents: {agent_names}")

    # ------------------------------------------------------------------
    # 3. Build MultiSeedResult
    # ------------------------------------------------------------------
    from env import MultiSeedResult, compute_preference_recovery

    multi = MultiSeedResult(
        seed_results=seed_results,
        agent_names=agent_names,
        T=T,
        n_users=n_users,
    )

    # ------------------------------------------------------------------
    # 4. Compute preference recovery
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nComputing preference recovery …")
    recovery = compute_preference_recovery(multi)
    for name, metrics_dict in recovery.items():
        if "recovery_rate" in metrics_dict:
            print(f"  {name:<22}: {metrics_dict['recovery_rate']:.1%}")
        else:
            parts = [f"{k}={v:.3f}" for k, v in metrics_dict.items()]
            print(f"  {name:<22}: {', '.join(parts)}")

    # ------------------------------------------------------------------
    # 5. Generate plots
    # ------------------------------------------------------------------
    from metrics import (
        export_csv,
        export_json_summary,
        plot_cumulative_regret_ci,
        plot_domain_classification_accuracy,
        plot_ood_robustness,
        plot_per_domain_accuracy,
        plot_preference_alignment,
        plot_preference_recovery,
        plot_rolling_accuracy_ci,
        plot_test_accuracy,
        print_summary,
        set_output_dir,
        significance_test,
    )

    print("Generating plots …")
    set_output_dir(args.output_dir)
    plot_cumulative_regret_ci(multi)
    plot_rolling_accuracy_ci(multi)
    plot_ood_robustness(multi)
    plot_preference_recovery(recovery, multi.agent_names, soft_mode=args.soft_preferences)
    plot_per_domain_accuracy(multi)
    plot_preference_alignment(multi)
    plot_domain_classification_accuracy(multi)
    plot_test_accuracy(multi)

    print_summary(multi)

    if not args.no_significance and len(seed_results) >= 3:
        proposed = "Bandit+Override"
        if proposed in agent_names:
            print("Significance tests (paired t-test, Bandit+Override vs baselines):")
            for baseline in agent_names:
                if baseline == proposed:
                    continue
                result = significance_test(multi, baseline, proposed)
                stars = (
                    "***" if result["p_value"] < 0.001
                    else "**" if result["p_value"] < 0.01
                    else "*" if result["p_value"] < 0.05
                    else "ns"
                )
                print(
                    f"  {proposed} vs {baseline:<22}: "
                    f"t={result['t_stat']:+.2f}, p={result['p_value']:.4f} {stars}, "
                    f"d={result['cohens_d']:+.2f}"
                )

    if args.export_csv:
        export_csv(multi, path=os.path.join(args.output_dir, "results.csv"))

    export_json_summary(multi, recovery_rates=recovery,
                        path=os.path.join(args.output_dir, "summary.json"))

    print(f"\nDone. Results saved to ./{args.output_dir}/")


if __name__ == "__main__":
    main()
