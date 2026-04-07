"""
main.py
Entry point for the personalized agent tool-selection benchmark.

Usage examples
--------------
# Full multi-seed run with default settings
python main.py

# Custom model, smaller scale for quick smoke-test
python main.py --model gpt-4o-mini --users 5 --rounds 20 --seeds 0 1 2

# Skip LLM calls (Random + PureBandit only) for fast debugging
python main.py --dry-run --users 5 --rounds 30 --seeds 0 1 2 3 4

# Change OOD ratio and export CSV
python main.py --ood-ratio 0.15 --export-csv

Environment variables
---------------------
OPENAI_API_KEY      required for OpenAI models
ANTHROPIC_API_KEY   required for Anthropic models via litellm
AGENT_MODEL         override LLM model (same as --model)
"""

from __future__ import annotations
import argparse
import os
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Personalized agent tool-selection benchmark"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        help="LLM model identifier (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--users", type=int, default=20,
        help="Number of synthetic users per seed (default: 20)",
    )
    parser.add_argument(
        "--rounds", type=int, default=50,
        help="Rounds per user (default: 50)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(range(5)),
        help="Random seeds for independent trials (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--ood-ratio", type=float, default=0.10,
        help="Fraction of OOD queries in each user's pool (default: 0.10)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=200,
        help="Queries per user pool (default: 200)",
    )
    parser.add_argument(
        "--no-random", action="store_true",
        help="Exclude the RandomAgent baseline from the run",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip LLM-based agents; run only Random + PureBandit for fast debugging",
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Export all raw round records to results.csv",
    )
    parser.add_argument(
        "--export-json", action="store_true", default=True,
        help="Export aggregated statistics to summary.json (default: on)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to store results (images/, results.csv, summary.json). "
             "Defaults to a new folder named after the model, e.g. ./gpt-4o-mini/",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-seed simulation progress",
    )
    parser.add_argument(
        "--soft-preferences", action="store_true",
        help="Use Dirichlet-sampled soft preferences instead of one-hot",
    )
    parser.add_argument(
        "--concentration", type=float, default=2.0,
        help="Dirichlet concentration parameter for soft preferences (higher = more peaked, default: 2.0)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available (saves progress after each user/seed)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["AGENT_MODEL"] = args.model

    output_dir = args.output_dir if args.output_dir else args.model
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  Personalized Agent Tool-Selection Benchmark")
    print("=" * 65)
    print(f"  Model     : {args.model}")
    print(f"  Users     : {args.users}")
    print(f"  Rounds    : {args.rounds}")
    print(f"  Seeds     : {args.seeds}")
    print(f"  OOD ratio : {args.ood_ratio:.0%}")
    print(f"  Soft prefs: {args.soft_preferences}")
    if args.soft_preferences:
        print(f"  Concentration: {args.concentration}")
    print(f"  Dry-run   : {args.dry_run}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Define agent factory (called fresh per seed)
    # ------------------------------------------------------------------
    from agents import (
        BanditPriorCoTAgent,
        InContextMemoryAgent,
        PureBanditAgent,
        RandomAgent,
        ZeroShotAgent,
    )

    def build_agents_fn():
        if args.dry_run:
            # Fast path: no LLM calls — useful for debugging metrics pipeline
            return [RandomAgent(), PureBanditAgent(alpha=1.0)]
        agents = []
        if not args.no_random:
            agents.append(RandomAgent())
        agents.extend([
            ZeroShotAgent(model=args.model),
            InContextMemoryAgent(model=args.model, memory_size=5),
            PureBanditAgent(alpha=1.0),
            BanditPriorCoTAgent(model=args.model, alpha=1.0),
        ])
        return agents

    # ------------------------------------------------------------------
    # 2. Run multi-seed experiment
    # ------------------------------------------------------------------
    from env import run_experiment, compute_preference_recovery

    print(f"\n[1/3] Running {len(args.seeds)}-seed experiment …")
    t0 = time.time()
    multi = run_experiment(
        build_agents_fn=build_agents_fn,
        n_users=args.users,
        T=args.rounds,
        seeds=args.seeds,
        ood_ratio=args.ood_ratio,
        pool_size=args.pool_size,
        verbose=args.verbose,
        soft_preferences=args.soft_preferences,
        concentration=args.concentration,
        output_dir=output_dir if args.resume else None,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  "
          f"({len(args.seeds)} seeds × {args.users} users × {args.rounds} rounds)")

    # ------------------------------------------------------------------
    # 3. Compute preference recovery rate
    # ------------------------------------------------------------------
    print("\n[2/3] Computing preference recovery …")
    recovery = compute_preference_recovery(multi)
    for name, metrics_dict in recovery.items():
        if "recovery_rate" in metrics_dict:
            print(f"  {name:<22}: {metrics_dict['recovery_rate']:.1%}")
        else:
            parts = []
            for k, v in metrics_dict.items():
                parts.append(f"{k}={v:.3f}")
            print(f"  {name:<22}: {', '.join(parts)}")

    # ------------------------------------------------------------------
    # 4. Plot metrics and print summary
    # ------------------------------------------------------------------
    from metrics import (
        export_csv,
        export_json_summary,
        plot_cumulative_regret_ci,
        plot_ood_robustness,
        plot_per_domain_accuracy,
        plot_preference_alignment,
        plot_preference_recovery,
        plot_rolling_accuracy_ci,
        plot_domain_classification_accuracy,
        print_summary,
        significance_test,
    )

    print("\n[3/3] Generating figures and summary …")
    import metrics as _metrics_mod
    _metrics_mod.set_output_dir(output_dir)
    plot_cumulative_regret_ci(multi)
    plot_rolling_accuracy_ci(multi)
    plot_ood_robustness(multi)
    plot_preference_recovery(recovery, multi.agent_names, soft_mode=args.soft_preferences)
    plot_per_domain_accuracy(multi)
    plot_preference_alignment(multi)
    plot_domain_classification_accuracy(multi)

    print_summary(multi)

    # Statistical significance: Bandit+CoT vs each baseline
    if not args.dry_run and len(args.seeds) >= 3:
        proposed = "Bandit+CoT"
        print("Significance tests (paired t-test, proposed vs baselines):")
        for baseline in multi.agent_names:
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
        print()

    if args.export_csv:
        export_csv(multi, path=os.path.join(output_dir, "results.csv"))

    if args.export_json:
        export_json_summary(multi, recovery_rates=recovery,
                            path=os.path.join(output_dir, "summary.json"))

    print(f"Benchmark complete. Results saved to ./{output_dir}/")


if __name__ == "__main__":
    main()
