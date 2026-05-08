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

import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

import data_gen
from data_gen import UserPersona, generate_users, generate_users_soft, sample_query


def _agent_uses_llm(agent) -> bool:
    """LLM-using agents have a `model` attribute (set in their __init__)."""
    return hasattr(agent, "model")


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
    # Domain classification predictions (true_domain, inferred_domain) — tracked at env level
    domain_predictions: List = field(default_factory=list)
    # Test evaluation results: {agent_name: {"accuracy": float, "ood_accuracy": float, ...}}
    test_results: Optional[Dict] = field(default=None)


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
# Checkpoint utilities
# ---------------------------------------------------------------------------


def _ckpt_dir(output_dir: str) -> Path:
    p = Path(output_dir) / "checkpoints"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_seed_checkpoint(output_dir: str, seed: int, sr: SeedResult) -> None:
    path = _ckpt_dir(output_dir) / f"seed_{seed}.pkl"
    with open(path, "wb") as f:
        pickle.dump(sr, f)


def _load_completed_seeds(output_dir: str) -> Dict[int, SeedResult]:
    ckpt = Path(output_dir) / "checkpoints"
    if not ckpt.exists():
        return {}
    completed = {}
    for p in ckpt.glob("seed_*.pkl"):
        if "_progress" in p.name:
            continue
        try:
            with open(p, "rb") as f:
                sr = pickle.load(f)
            completed[sr.seed] = sr
        except Exception:
            pass
    return completed


def _save_mid_seed_checkpoint(
    output_dir: str, seed: int,
    agents, partial_records: List[List], completed_user_count: int,
    rng_state,
) -> None:
    path = _ckpt_dir(output_dir) / f"seed_{seed}_progress.pkl"
    data = {
        "completed_user_count": completed_user_count,
        "agents": agents,
        "partial_records": partial_records,
        "rng_state": rng_state,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def _load_mid_seed_checkpoint(output_dir: str, seed: int) -> Optional[dict]:
    path = Path(output_dir) / "checkpoints" / f"seed_{seed}_progress.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _remove_mid_seed_checkpoint(output_dir: str, seed: int) -> None:
    path = Path(output_dir) / "checkpoints" / f"seed_{seed}_progress.pkl"
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Shared domain classification
# ---------------------------------------------------------------------------


def _classify_domain(query: str, model: str) -> str:
    """
    Classify a query into a domain using an LLM call.
    Shared across all agents for fair comparison.
    Falls back to the first domain if parsing fails.
    """
    import json as _json
    from agents import _llm_call

    domain_summaries = []
    for d, d_tools in data_gen.DOMAINS.items():
        tool_names = ", ".join(d_tools[:4])
        if len(d_tools) > 4:
            tool_names += f", ... ({len(d_tools)} total)"
        domain_summaries.append(f"  - {d}: [{tool_names}]")
    domains_block = "\n".join(domain_summaries)

    prompt = (
        f"Which domain does this query belong to?\n\n"
        f"Domains and their tools:\n{domains_block}\n\n"
        f"Query: \"{query}\"\n"
        f"Reply with only JSON: {{\"domain\": \"<exact domain name>\"}}"
    )
    response = _llm_call(prompt, model)

    # Parse
    try:
        data = _json.loads(response)
        candidate = data.get("domain", "")
        if candidate in data_gen.DOMAINS:
            return candidate
    except (_json.JSONDecodeError, AttributeError):
        pass
    # Substring fallback
    for d in data_gen.DOMAINS:
        if d.lower() in response.lower():
            return d
    # Last resort: first domain
    return list(data_gen.DOMAINS.keys())[0]


# ---------------------------------------------------------------------------
# Single-seed simulation
# ---------------------------------------------------------------------------


def run_simulation(
    agents,
    users: List[UserPersona],
    T: int = 50,
    seed: int = 0,
    verbose: bool = False,  # noqa: ARG001 (kept for API compatibility)
    output_dir: Optional[str] = None,
    shared_domain_model: Optional[str] = None,
    wandb_run=None,
) -> tuple:
    """
    Run the online evaluation loop for one set of agents and users.

    Each round: all agents receive the same query (fair comparison).
    Reward is binary: 1.0 iff selected_tool == true_tool.
    Agent state is updated after each selection.

    If shared_domain_model is set, domain classification is done once per query
    via LLM and shared across all agents (fair comparison).
    Otherwise, ground truth domain is used.

    Returns
    -------
    (List[AgentResult], List[tuple]) — agent results and domain predictions.
    """
    results: List[AgentResult] = [AgentResult(agent_name=a.name) for a in agents]
    domain_predictions: List[tuple] = []  # (true_domain, inferred_domain)
    rng = random.Random(seed)

    # Try to resume from mid-seed checkpoint
    start_user_idx = 0
    if output_dir is not None:
        ckpt = _load_mid_seed_checkpoint(output_dir, seed)
        if ckpt is not None:
            start_user_idx = ckpt["completed_user_count"]
            # Restore agent state
            for i, saved_agent in enumerate(ckpt["agents"]):
                agents[i] = saved_agent
            # Restore partial records
            for i, records in enumerate(ckpt["partial_records"]):
                results[i].records = records
            # Restore RNG state
            rng.setstate(ckpt["rng_state"])
            if start_user_idx > 0:
                print(f"  [Resume] Seed {seed}: skipping {start_user_idx}/{len(users)} completed users", flush=True)

    remaining_steps = (len(users) - start_user_idx) * T
    try:
        from tqdm import tqdm
        pbar = tqdm(total=remaining_steps, unit="step",
                    desc=f"seed={seed}", leave=False)
    except ImportError:
        pbar = None

    # Thread pool for parallel LLM-agent select_tool calls within a round.
    # Non-LLM agents (Random, Pure-Bandit, Freq-Greedy) stay sequential to preserve
    # global RNG state ordering (RandomAgent uses random.choice on the global module).
    # LLM agents do not touch the global RNG, so their order is irrelevant; only the
    # post-selection update() phase, which mutates state, runs sequentially.
    llm_indices = [i for i, a in enumerate(agents) if _agent_uses_llm(a)]
    nonllm_indices = [i for i, a in enumerate(agents) if not _agent_uses_llm(a)]
    pool = ThreadPoolExecutor(max_workers=max(len(llm_indices), 1)) if llm_indices else None

    try:
        for user_idx, user in enumerate(users):
            if user_idx < start_user_idx:
                continue

            for t in range(T):
                query, domain, is_ood, true_tool = sample_query(user, rng)
                tools = data_gen.ALL_TOOLS  # agents select from the full pool across all domains

                # Shared domain classification: one LLM call, result shared by all agents
                if shared_domain_model:
                    inferred_domain = _classify_domain(query, shared_domain_model)
                    domain_predictions.append((domain, inferred_domain))
                    agent_domain = inferred_domain
                else:
                    agent_domain = domain  # ground truth fallback

                # Phase 1: non-LLM agents — sequential, preserves global RNG ordering
                selections: List[Optional[str]] = [None] * len(agents)
                for idx in nonllm_indices:
                    selections[idx] = agents[idx].select_tool(
                        query, agent_domain, user.user_id, tools
                    )

                # Phase 2: LLM agents — parallel select_tool (HTTP overlap)
                if pool is not None:
                    futures = {
                        idx: pool.submit(
                            agents[idx].select_tool,
                            query, agent_domain, user.user_id, tools,
                        )
                        for idx in llm_indices
                    }
                    for idx, fut in futures.items():
                        selections[idx] = fut.result()

                # Phase 3: sequential — record + update in original agent order
                for agent_idx, agent in enumerate(agents):
                    selected = selections[agent_idx]
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
                    agent.update(query, agent_domain, user.user_id, selected, reward)

                if pbar is not None:
                    recent = results[0].records[-min(10, len(results[0].records)):]
                    acc = sum(r.reward for r in recent) / len(recent)
                    pbar.set_postfix({"acc": f"{acc:.0%}", "user": user.user_id})
                    pbar.update(1)

            # Checkpoint after each user completes
            if output_dir is not None:
                _save_mid_seed_checkpoint(
                    output_dir, seed, agents,
                    [r.records for r in results],
                    user_idx + 1, rng.getstate(),
                )

            # Log per-user metrics to wandb (if configured)
            if wandb_run is not None:
                log_dict = {"user_idx": user_idx + 1, "seed": seed}
                for ar in results:
                    if not ar.records:
                        continue
                    # Cumulative regret = sum of (1 - reward) over all rounds so far
                    total_regret = sum(1.0 - r.reward for r in ar.records)
                    # Rolling accuracy: last 50 rounds (1 user worth)
                    window = min(50, len(ar.records))
                    recent_acc = sum(r.reward for r in ar.records[-window:]) / window
                    log_dict[f"regret/{ar.agent_name}"] = total_regret
                    log_dict[f"rolling_acc/{ar.agent_name}"] = recent_acc
                wandb_run.log(log_dict)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if pbar is not None:
            pbar.close()

    return results, domain_predictions


# ---------------------------------------------------------------------------
# Test evaluation (no agent updates)
# ---------------------------------------------------------------------------


def evaluate_test(
    agents,
    users: List[UserPersona],
    shared_domain_model: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate trained agents on held-out test queries. Agents select but do NOT update.

    Returns {agent_name: {"accuracy": float, "ood_accuracy": float, "n_total": int, "n_ood": int}}
    """
    stats = {
        a.name: {"correct": 0, "total": 0, "ood_correct": 0, "ood_total": 0}
        for a in agents
    }

    total_queries = sum(len(u.test_pool) for u in users)
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_queries, unit="query", desc="test_eval", leave=False)
    except ImportError:
        pbar = None

    # Same parallelization pattern as run_simulation: LLM agents in a pool,
    # non-LLM agents (Random uses global random.choice) sequential.
    llm_indices = [i for i, a in enumerate(agents) if _agent_uses_llm(a)]
    nonllm_indices = [i for i, a in enumerate(agents) if not _agent_uses_llm(a)]
    pool = ThreadPoolExecutor(max_workers=max(len(llm_indices), 1)) if llm_indices else None

    try:
        for user in users:
            if not user.test_pool:
                continue
            for query, domain, is_ood, true_tool in user.test_pool:
                if shared_domain_model:
                    agent_domain = _classify_domain(query, shared_domain_model)
                else:
                    agent_domain = domain

                selections: List[Optional[str]] = [None] * len(agents)
                for idx in nonllm_indices:
                    selections[idx] = agents[idx].select_tool(
                        query, agent_domain, user.user_id, data_gen.ALL_TOOLS
                    )
                if pool is not None:
                    futures = {
                        idx: pool.submit(
                            agents[idx].select_tool,
                            query, agent_domain, user.user_id, data_gen.ALL_TOOLS,
                        )
                        for idx in llm_indices
                    }
                    for idx, fut in futures.items():
                        selections[idx] = fut.result()

                for agent_idx, agent in enumerate(agents):
                    selected = selections[agent_idx]
                    correct = 1 if selected == true_tool else 0
                    stats[agent.name]["correct"] += correct
                    stats[agent.name]["total"] += 1
                    if is_ood:
                        stats[agent.name]["ood_correct"] += correct
                        stats[agent.name]["ood_total"] += 1

                if pbar is not None:
                    pbar.update(1)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if pbar is not None:
            pbar.close()

    results = {}
    for name, s in stats.items():
        results[name] = {
            "accuracy": s["correct"] / max(s["total"], 1),
            "ood_accuracy": s["ood_correct"] / max(s["ood_total"], 1),
            "n_total": s["total"],
            "n_ood": s["ood_total"],
        }

    if verbose:
        print("  Test evaluation:")
        for name, r in results.items():
            print(f"    {name:<22}: acc={r['accuracy']:.1%}, ood={r['ood_accuracy']:.1%} "
                  f"({r['n_total']} queries, {r['n_ood']} OOD)")

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
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_config: Optional[Dict] = None,
    test_pool_size: int = 50,
    soft_preferences: bool = False,
    concentration: float = 2.0,
    output_dir: Optional[str] = None,
    shared_domain_model: Optional[str] = None,
) -> MultiSeedResult:
    """
    Run independent simulation trials across multiple random seeds.

    Each seed produces:
      - A fresh set of synthetic users (different preference assignments)
      - Fresh agent instances (clean state)
      - An independent simulation run

    If output_dir is set, checkpoints are saved after each seed (and each user
    within a seed). On resume, completed seeds are loaded from checkpoint and
    partially completed seeds continue from where they left off.

    Parameters
    ----------
    build_agents_fn : zero-argument callable returning a fresh list of agents
    seeds           : list of integer seeds (default: [0, 1, 2, 3, 4])
    ood_ratio       : fraction of OOD queries in each user's pool
    pool_size       : total queries per user pool
    verbose         : print per-seed progress
    output_dir      : if set, enable checkpoint/resume to this directory
    """
    if seeds is None:
        seeds = list(range(5))

    # Load completed seed checkpoints
    completed = _load_completed_seeds(output_dir) if output_dir else {}
    if completed and verbose:
        print(f"  [Resume] Found {len(completed)} completed seed(s): {sorted(completed.keys())}", flush=True)

    seed_results: List[SeedResult] = []
    agent_names: Optional[List[str]] = None

    for i, seed in enumerate(seeds):
        # Skip fully completed seeds
        if seed in completed:
            sr = completed[seed]
            seed_results.append(sr)
            if agent_names is None:
                agent_names = [ar.agent_name for ar in sr.agent_results]
            if verbose:
                print(f"\n[Seed {seed}] ({i + 1}/{len(seeds)}) — loaded from checkpoint", flush=True)
            continue

        if verbose:
            print(f"\n[Seed {seed}] ({i + 1}/{len(seeds)})", flush=True)
        t0 = time.time()

        # Initialize wandb run for this seed (if configured)
        wandb_run = None
        if wandb_project:
            try:
                import wandb
                run_name = (wandb_run_name or "exp") + f"-seed{seed}"
                cfg = dict(wandb_config or {})
                cfg.update({"seed": seed, "n_users": n_users, "T": T})
                wandb_run = wandb.init(
                    project=wandb_project,
                    name=run_name,
                    config=cfg,
                    reinit=True,
                )
            except Exception as e:
                print(f"  [wandb] init failed: {e}", flush=True)
                wandb_run = None

        # Fresh users with this seed's preference assignments
        if soft_preferences:
            users = generate_users_soft(
                n_users=n_users, seed=seed, pool_size=pool_size,
                ood_ratio=ood_ratio, concentration=concentration,
                test_pool_size=test_pool_size,
            )
        else:
            users = generate_users(
                n_users=n_users, seed=seed, pool_size=pool_size, ood_ratio=ood_ratio,
                test_pool_size=test_pool_size,
            )

        # Fresh agents (clean state — no cross-seed contamination)
        agents = build_agents_fn()
        if agent_names is None:
            agent_names = [a.name for a in agents]

        # Run training simulation (with mid-seed checkpoint support)
        agent_results, domain_preds = run_simulation(
            agents=agents, users=users, T=T, seed=seed, verbose=verbose,
            output_dir=output_dir,
            shared_domain_model=shared_domain_model,
            wandb_run=wandb_run,
        )

        elapsed = time.time() - t0
        if verbose:
            n_correct = sum(r.reward for ar in agent_results for r in ar.records)
            n_total = sum(len(ar.records) for ar in agent_results)
            print(f"  Done in {elapsed:.1f}s | overall acc: {n_correct / n_total:.1%}")

        # Test evaluation on held-out queries (no agent updates)
        test_res = None
        if test_pool_size > 0 and any(u.test_pool for u in users):
            if verbose:
                print("  Running test evaluation ...", flush=True)
            test_res = evaluate_test(
                agents, users, shared_domain_model=shared_domain_model, verbose=verbose,
            )

        sr = SeedResult(
            seed=seed,
            agent_results=agent_results,
            users=users,
            trained_agents=agents,
            domain_predictions=domain_preds,
            test_results=test_res,
        )
        seed_results.append(sr)

        # Save seed-level checkpoint and clean up mid-seed checkpoint
        if output_dir is not None:
            _save_seed_checkpoint(output_dir, seed, sr)
            _remove_mid_seed_checkpoint(output_dir, seed)

        # Finalize wandb run for this seed
        if wandb_run is not None:
            try:
                # Log final test results and summary metrics
                final_log = {}
                if test_res:
                    for name, r in test_res.items():
                        final_log[f"test_acc/{name}"] = r["accuracy"]
                        final_log[f"test_ood_acc/{name}"] = r["ood_accuracy"]
                # Log domain classification accuracy
                if domain_preds:
                    correct = sum(1 for true_d, pred_d in domain_preds if true_d == pred_d)
                    final_log["domain_classification_acc"] = correct / len(domain_preds)
                if final_log:
                    wandb_run.log(final_log)
                wandb_run.finish()
            except Exception as e:
                print(f"  [wandb] finish failed: {e}", flush=True)

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


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) with smoothing to avoid log(0)."""
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _spearman_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two arrays."""
    from scipy.stats import spearmanr
    corr, _ = spearmanr(a, b)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_preference_recovery(
    multi_result: MultiSeedResult,
) -> Dict[str, Dict[str, float]]:
    """
    Preference recovery metrics for each agent.

    Returns
    -------
    In one-hot mode:
      {agent_name: {"recovery_rate": float}}
    In soft mode:
      {agent_name: {"cosine_sim": float, "kl_divergence": float, "spearman": float}}
    """
    # Detect soft mode
    soft_mode = (multi_result.seed_results[0].users[0].soft_preferences is not None)

    if soft_mode:
        cosine: Dict[str, List[float]] = {n: [] for n in multi_result.agent_names}
        kl: Dict[str, List[float]] = {n: [] for n in multi_result.agent_names}
        spearman: Dict[str, List[float]] = {n: [] for n in multi_result.agent_names}
    else:
        recovery: Dict[str, List[float]] = {n: [] for n in multi_result.agent_names}

    domains = list(data_gen.DOMAINS.keys())
    for sr in multi_result.seed_results:
        agents_by_name = {a.name: a for a in sr.trained_agents}
        for user in sr.users:
            for domain in domains:
                domain_tools = data_gen.DOMAINS[domain]

                for agent_name, agent in agents_by_name.items():
                    domain_sample_qs = data_gen.STANDARD_QUERIES.get(domain, [])[:10]
                    learned = agent.get_learned_distribution(
                        user.user_id, domain, domain_tools, domain_sample_qs
                    )
                    if soft_mode and user.soft_preferences is not None:
                        true_dist = user.soft_preferences[domain]
                        true_vec = np.array([true_dist.get(t, 0.0) for t in domain_tools])
                        learned_vec = np.array([learned.get(t, 0.0) for t in domain_tools])
                        # Cosine similarity
                        norm_prod = np.linalg.norm(true_vec) * np.linalg.norm(learned_vec)
                        cosine[agent_name].append(
                            float(np.dot(true_vec, learned_vec) / (norm_prod + 1e-12))
                        )
                        # KL divergence: KL(true || learned)
                        kl[agent_name].append(_kl_divergence(true_vec, learned_vec))
                        # Spearman rank correlation
                        spearman[agent_name].append(_spearman_rank_corr(true_vec, learned_vec))
                    else:
                        true_pref = user.preferences[domain]
                        predicted = max(domain_tools, key=lambda t: learned.get(t, 0.0))
                        recovery[agent_name].append(1.0 if predicted == true_pref else 0.0)

    if soft_mode:
        return {
            name: {
                "cosine_sim": float(np.mean(cosine[name])) if cosine[name] else 0.0,
                "kl_divergence": float(np.mean(kl[name])) if kl[name] else 0.0,
                "spearman": float(np.mean(spearman[name])) if spearman[name] else 0.0,
            }
            for name in multi_result.agent_names
        }
    else:
        return {
            name: {"recovery_rate": float(np.mean(recovery[name])) if recovery[name] else 0.0}
            for name in multi_result.agent_names
        }


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
