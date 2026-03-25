"""
bandits.py
Contextual bandit implementations used as personalization priors.

LinUCB (disjoint model): one arm per (user_id, domain, tool) triplet.
Features: 96-dim vector from feature-hashing over query + tool + domain text.

Key addition over baseline: `learned_preference_distribution()` which
estimates the agent's learned tool preferences for a given (user, domain)
by averaging bandit probabilities over a representative query sample.
This enables post-hoc evaluation of personalization quality (KL divergence,
preference recovery rate) without storing the full trajectory.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64  # dimensionality of the query hash subspace


def _hash_feature(text: str, dim: int = VOCAB_SIZE) -> np.ndarray:
    """
    Map a string to a fixed-size float vector via feature hashing (sign-hash trick).
    Each whitespace-separated token contributes to exactly one bucket.
    Output is L2-normalised.
    """
    vec = np.zeros(dim, dtype=np.float64)
    words = text.lower().split()
    for word in words:
        digest = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = digest % dim
        sign = 1 if (digest >> 8) % 2 == 0 else -1
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def context_vector(query: str, tool: str, domain: str) -> np.ndarray:
    """Concatenate query + tool + domain feature hashes into one context vector."""
    q_feat = _hash_feature(query, VOCAB_SIZE)          # 64-dim
    t_feat = _hash_feature(tool, VOCAB_SIZE // 4)      # 16-dim
    d_feat = _hash_feature(domain, VOCAB_SIZE // 4)    # 16-dim
    return np.concatenate([q_feat, t_feat, d_feat])    # 96-dim


CONTEXT_DIM = VOCAB_SIZE + VOCAB_SIZE // 4 + VOCAB_SIZE // 4  # 96


# ---------------------------------------------------------------------------
# LinUCB arm
# ---------------------------------------------------------------------------


class LinUCBArm:
    """One arm of a disjoint LinUCB bandit (ridge-regression ridge + UCB bonus)."""

    def __init__(self, d: int = CONTEXT_DIM, alpha: float = 1.0):
        self.alpha = alpha
        self.A = np.eye(d, dtype=np.float64)   # regularised Gram matrix  (d×d)
        self.b = np.zeros(d, dtype=np.float64)  # reward-weighted feature sum (d,)

    def ucb_score(self, x: np.ndarray) -> float:
        # Use lstsq for numerical stability; rcond filters near-zero singular values
        A_inv_x, _, _, _ = np.linalg.lstsq(self.A, x, rcond=None)
        theta, _, _, _ = np.linalg.lstsq(self.A, self.b, rcond=None)
        exploit = float(theta @ x)
        explore = self.alpha * float(np.sqrt(max(x @ A_inv_x, 0.0)))
        return exploit + explore

    def update(self, x: np.ndarray, reward: float) -> None:
        self.A += np.outer(x, x)
        self.b += reward * x

    @property
    def n_updates(self) -> int:
        """Approximate number of updates (trace minus initial identity rank)."""
        return max(0, int(np.trace(self.A)) - self.A.shape[0])


# ---------------------------------------------------------------------------
# LinUCB bandit (disjoint)
# ---------------------------------------------------------------------------


class LinUCBBandit:
    """
    Per-user, per-domain disjoint LinUCB bandit.

    Arms are keyed by (user_id, domain, tool_name).
    Provides:
      select()                       → greedy UCB selection
      scores() / probabilities()     → raw UCB / softmax distribution (prior for CoT)
      update()                       → Bayesian update after observing reward
      learned_preference_distribution()
                                     → post-hoc preference estimate via query averaging
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._arms: Dict[tuple, LinUCBArm] = {}

    def _get_arm(self, user_id: int, domain: str, tool: str) -> LinUCBArm:
        key = (user_id, domain, tool)
        if key not in self._arms:
            self._arms[key] = LinUCBArm(d=CONTEXT_DIM, alpha=self.alpha)
        return self._arms[key]

    def select(self, user_id: int, domain: str, query: str, tools: List[str]) -> str:
        """Return the tool with the highest UCB score."""
        scores = {tool: self._get_arm(user_id, domain, tool).ucb_score(
            context_vector(query, tool, domain)) for tool in tools}
        return max(scores, key=lambda t: scores[t])

    def scores(
        self, user_id: int, domain: str, query: str, tools: List[str]
    ) -> Dict[str, float]:
        """Return raw UCB scores for all tools."""
        return {
            tool: self._get_arm(user_id, domain, tool).ucb_score(
                context_vector(query, tool, domain)
            )
            for tool in tools
        }

    def probabilities(
        self, user_id: int, domain: str, query: str, tools: List[str]
    ) -> Dict[str, float]:
        """
        Softmax-normalised UCB scores → probability distribution over tools.
        Used as the 'statistical prior' injected into the CoT agent prompt.
        Numerically stable via max-shift before exponentiation.
        """
        raw = self.scores(user_id, domain, query, tools)
        vals = np.array([raw[t] for t in tools], dtype=np.float64)
        vals -= vals.max()
        exp_vals = np.exp(vals)
        probs = exp_vals / exp_vals.sum()
        return {tool: float(p) for tool, p in zip(tools, probs)}

    def update(
        self, user_id: int, domain: str, query: str, tool: str, reward: float
    ) -> None:
        """Update the arm for the selected (user, domain, tool) triplet."""
        x = context_vector(query, tool, domain)
        self._get_arm(user_id, domain, tool).update(x, reward)

    def learned_preference_distribution(
        self,
        user_id: int,
        domain: str,
        tools: List[str],
        sample_queries: List[str],
    ) -> Dict[str, float]:
        """
        Estimate the agent's learned marginal preference distribution for a
        (user, domain) pair by averaging softmax probabilities over a set of
        representative standard queries.

        This marginalises out query-specific effects and captures the
        persistent user-level preference the bandit has learned.
        Used for computing KL divergence and preference recovery rate.

        Parameters
        ----------
        sample_queries : representative queries from STANDARD_QUERIES[domain]
        """
        if not sample_queries:
            n = len(tools)
            return {tool: 1.0 / n for tool in tools}

        accumulated = np.zeros(len(tools), dtype=np.float64)
        for query in sample_queries:
            probs = self.probabilities(user_id, domain, query, tools)
            accumulated += np.array([probs[t] for t in tools])
        mean_probs = accumulated / len(sample_queries)
        return {tool: float(p) for tool, p in zip(tools, mean_probs)}

    def has_updates(self, user_id: int, domain: str) -> bool:
        """Return True if any arm for this (user, domain) has been updated."""
        return any(
            k[0] == user_id and k[1] == domain for k in self._arms
        )


if __name__ == "__main__":
    from data_gen import DOMAINS, STANDARD_QUERIES, generate_users

    bandit = LinUCBBandit(alpha=1.0)
    users = generate_users(n_users=2)
    u = users[0]
    domain = "food_delivery"
    tools = DOMAINS[domain]
    query = "Order some milk tea"

    probs = bandit.probabilities(u.user_id, domain, query, tools)
    print("Initial uniform-ish priors:", {k: f"{v:.3f}" for k, v in probs.items()})

    # Simulate positive updates for preferred tool
    preferred = u.preferences[domain]
    for _ in range(20):
        bandit.update(u.user_id, domain, query, preferred, reward=1.0)

    probs = bandit.probabilities(u.user_id, domain, query, tools)
    print(f"After 20 positive updates for '{preferred}':", {k: f"{v:.3f}" for k, v in probs.items()})

    # Learned preference distribution (averaged over standard queries)
    learned = bandit.learned_preference_distribution(
        u.user_id, domain, tools, STANDARD_QUERIES[domain]
    )
    print("Learned marginal preference:", {k: f"{v:.3f}" for k, v in learned.items()})
    print(f"Predicted preferred tool: {max(learned, key=learned.get)}")
    print(f"True preferred tool:      {preferred}")
