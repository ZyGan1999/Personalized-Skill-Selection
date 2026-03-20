"""
env.py
Online evaluation environment for the personalized tool-selection benchmark.

Single-seed loop
----------------
run_simulation(agents, users, T, seed) -> List[AgentResult]
  For each user × T rounds: sample query, call each agent, compute reward, update.

Multi-seed experiment
---------------------
run_experiment(build_agents_fn, n_users, T, seeds, ...) -> MultiSeedResult
  Repeats the simulation across multiple random seeds, generating fresh users and
  fresh agent instances each time for statistically independent trials.
  Returns aggregated results suitable for plotting with confidence intervals.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from data_gen import DOMAINS, STANDARD_QUERIES, UserPersona, generate_users, sample_query


# ---------------------------------------------------------------------------
# Per-round record
# ---------------------------------------------------------------------------


@dataclass
class RoundRecord:
    round_idx: int        # 0-indexed round within a user's trajectory
    user_id: int
    domain: str
    query: str
    is_ood: bool
    true_tool: str
    selected_tool: str
    reward: float         # 1.0 if correct, 0.0 otherwise


# ---------------------------------------------------------------------------
# Per-agent result (one simulation run)
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    agent_name: str
    records: List[RoundRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Multi-seed aggregation structures
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """Results from one complete simulation run under a single random seed."""
    seed: int
    agent_results: List[AgentResult]
    users: List[UserPersona]   # kept for post-hoc preference alignment metrics
    # Agents after training — used to extract learned distributions
    trained_agents: List                # list of agent objects


@dataclass
class MultiSeedResult:
    """
    Aggregated results across multiple independent seeds.

    Attributes
    ----------
    seed_results   : one SeedResult per seed
    agent_names    : ordered list of agent names (same across all seeds)
    T              : number of rounds per user
    n_users        : number of users per seed
    """
    seed_results: List[SeedResult]
    agent_names: List[str]
    T: int
    n_users: int

    @property
    def n_seeds(self) -> int:
        return len(self.seed_results)

    def get_agent_results_by_name(self, agent_name: str) -> List[AgentResult]:
        """Return all AgentResult objects for a given agent across all seeds."""
        results = []
        for sr in self.seed_results:
            for ar in sr.agent_results:
                if ar.agent_name == agent_name:
                    results.append(ar)
                    break
        return results


# ---------------------------------------------------------------------------
# Single-seed simulation
# ---------------------------------------------------------------------------


def run_simulation(
    agents,
    users: List[UserPersona],
    T: int = 50,
    seed: int = 0,
    verbose: bool = False,  # noqa: ARG001 (kept for API compatibility)
) -> List[AgentResult]:
    """
    Run the online evaluation loop for one set of agents and users.

    Each round: all agents receive the same query (fair comparison).
    Reward is binary: 1.0 iff selected_tool == true_tool.
    Agent state is updated after each selection.

    Returns
    -------
    List[AgentResult], one per agent, in the same order as `agents`.
    """
    results: List[AgentResult] = [AgentResult(agent_name=a.name) for a in agents]
    rng = random.Random(seed)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(users) * T, unit="step",
                    desc=f"seed={seed}", leave=False)
    except ImportError:
        pbar = None

    for user in users:
        for t in range(T):
            query, domain, is_ood, true_tool = sample_query(user, rng)
            tools = DOMAINS[domain]

            for agent_idx, agent in enumerate(agents):
                selected = agent.select_tool(query, domain, user.user_id, tools)
                reward = 1.0 if selected == true_tool else 0.0

                results[agent_idx].records.append(RoundRecord(
                    round_idx=t,
                    user_id=user.user_id,
                    domain=domain,
                    query=query,
                    is_ood=is_ood,
                    true_tool=true_tool,
                    selected_tool=selected,
                    reward=reward,
                ))
                agent.update(query, domain, user.user_id, selected, reward)

            if pbar is not None:
                # show mean reward of first agent as live metric
                recent = results[0].records[-min(10, len(results[0].records)):]
                acc = sum(r.reward for r in recent) / len(recent)
                pbar.set_postfix({"acc": f"{acc:.0%}", "user": user.user_id})
                pbar.update(1)

    if pbar is not None:
        pbar.close()
    return results


# ---------------------------------------------------------------------------
# Multi-seed experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    build_agents_fn: Callable[[], List],
    n_users: int = 20,
    T: int = 50,
    seeds: Optional[List[int]] = None,
    ood_ratio: float = 0.10,
    pool_size: int = 200,
    verbose: bool = False,
) -> MultiSeedResult:
    """
    Run independent simulation trials across multiple random seeds.

    Each seed produces:
      - A fresh set of synthetic users (different preference assignments)
      - Fresh agent instances (clean state)
      - An independent simulation run

    This generates statistically independent trials for computing
    confidence intervals and running significance tests.

    Parameters
    ----------
    build_agents_fn : zero-argument callable returning a fresh list of agents
    seeds           : list of integer seeds (default: [0, 1, 2, 3, 4])
    ood_ratio       : fraction of OOD queries in each user's pool
    pool_size       : total queries per user pool
    verbose         : print per-seed progress
    """
    if seeds is None:
        seeds = list(range(5))

    seed_results: List[SeedResult] = []
    agent_names: Optional[List[str]] = None

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n[Seed {seed}] ({i + 1}/{len(seeds)})", flush=True)
        t0 = time.time()

        # Fresh users with this seed's preference assignments
        users = generate_users(
            n_users=n_users, seed=seed, pool_size=pool_size, ood_ratio=ood_ratio
        )

        # Fresh agents (clean state — no cross-seed contamination)
        agents = build_agents_fn()
        if agent_names is None:
            agent_names = [a.name for a in agents]

        # Run simulation
        agent_results = run_simulation(
            agents=agents, users=users, T=T, seed=seed, verbose=verbose
        )

        elapsed = time.time() - t0
        if verbose:
            n_correct = sum(r.reward for ar in agent_results for r in ar.records)
            n_total = sum(len(ar.records) for ar in agent_results)
            print(f"  Done in {elapsed:.1f}s | overall acc: {n_correct / n_total:.1%}")

        seed_results.append(SeedResult(
            seed=seed,
            agent_results=agent_results,
            users=users,
            trained_agents=agents,
        ))

    return MultiSeedResult(
        seed_results=seed_results,
        agent_names=agent_names or [],
        T=T,
        n_users=n_users,
    )


# ---------------------------------------------------------------------------
# Utility: per-round reward matrix
# ---------------------------------------------------------------------------


def rewards_matrix(result: AgentResult, n_users: int, T: int) -> np.ndarray:
    """
    Build a (n_users × T) reward matrix from an AgentResult.
    Entry [i, t] = reward of user i at round t.
    """
    mat = np.zeros((n_users, T), dtype=np.float64)
    user_ids = sorted(set(r.user_id for r in result.records))
    uid_idx = {uid: i for i, uid in enumerate(user_ids)}
    for rec in result.records:
        mat[uid_idx[rec.user_id], rec.round_idx] = rec.reward
    return mat


# ---------------------------------------------------------------------------
# Utility: preference alignment evaluation
# ---------------------------------------------------------------------------


def compute_preference_recovery(
    multi_result: MultiSeedResult,
) -> Dict[str, float]:
    """
    Preference recovery rate: fraction of (user, domain) pairs where the
    agent's learned distribution assigns the highest probability to the
    user's true preferred tool.

    Only applicable to agents that implement `get_learned_distribution`
    (bandit-based agents). Others return None.

    Returns
    -------
    Dict mapping agent_name → mean preference recovery rate across all
    (seed, user, domain) triples.
    """
    recovery: Dict[str, List[float]] = {name: [] for name in multi_result.agent_names}

    domains = list(DOMAINS.keys())

    for sr in multi_result.seed_results:
        agents_by_name = {a.name: a for a in sr.trained_agents}
        for user in sr.users:
            for domain in domains:
                tools = DOMAINS[domain]
                sample_qs = STANDARD_QUERIES.get(domain, [])[:10]
                true_pref = user.preferences[domain]

                for agent_name, agent in agents_by_name.items():
                    learned = agent.get_learned_distribution(
                        user.user_id, domain, tools, sample_qs
                    )
                    predicted = max(learned, key=learned.get)
                    recovery[agent_name].append(1.0 if predicted == true_pref else 0.0)

    return {name: float(np.mean(vals)) if vals else 0.0 for name, vals in recovery.items()}


if __name__ == "__main__":
    # Smoke test with a fast dummy agent
    from data_gen import generate_users as _gen

    class _DummyAgent:
        name = "Random"

        def select_tool(self, query, domain, user_id, tools):
            return random.choice(tools)

        def update(self, *args):
            pass

    users = _gen(n_users=2)
    results = run_simulation([_DummyAgent()], users, T=5, verbose=True)
    print(f"Records: {len(results[0].records)}")
    for rec in results[0].records[:3]:
        print(f"  round={rec.round_idx} user={rec.user_id} reward={rec.reward}")
