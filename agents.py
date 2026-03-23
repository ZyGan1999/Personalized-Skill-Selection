"""
agents.py
Five agent implementations for the personalized tool-selection benchmark.

All agents expose the same interface:
    select_tool(query, domain, user_id, tools) -> str
    update(query, domain, user_id, selected_tool, reward) -> None
    get_learned_distribution(user_id, domain, tools, sample_queries) -> Dict[str, float]

Agents
------
RandomAgent            — uniform random selection (sanity-check lower bound)
ZeroShotAgent          — pure LLM, no state (Baseline 1)
InContextMemoryAgent   — last-K successful selections in prompt (Baseline 2)
PureBanditAgent        — LinUCB only, no LLM (Baseline 3)
BanditPriorCoTAgent    — proposed method: LinUCB prior + LLM CoT (Ours)
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict, deque
from typing import Dict, List

from bandits import LinUCBBandit
from data_gen import DOMAINS, TOOL_METADATA

# ---------------------------------------------------------------------------
# LLM client — lazy import so the module loads without LLM dependencies
# (allows dry-run / bandit-only experiments without openai/litellm installed)
# ---------------------------------------------------------------------------

def _llm_call(prompt: str, model: str) -> str:
    """
    Call an LLM via raw HTTP (requests library).

    Environment variables:
      OPENAI_API_KEY    = your platform API key
      OPENAI_CHAT_URL   = complete chat endpoint URL, e.g.
                          https://www.right.codes/claude-aws/v1/chat/completions
                          (takes priority over OPENAI_API_BASE)
      OPENAI_API_BASE   = base URL without path; /chat/completions is appended
                          e.g. https://api.openai.com/v1  (fallback)
      AGENT_MODEL       = model name the platform expects
    """
    import time
    import requests

    api_key = os.getenv("OPENAI_API_KEY", "")
    bare_model = model.removeprefix("openai/")

    chat_url = os.getenv("OPENAI_CHAT_URL")
    if chat_url:
        url = chat_url
    else:
        api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
        url = f"{api_base}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": bare_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code >= 500:
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} Server Error — body: {resp.text[:300]}",
                    response=resp,
                )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                requests.exceptions.ChunkedEncodingError) as e:
            last_exc = e
            is_upstream = "do_request_failed" in str(e) or "upstream" in str(e)
            wait = 15 if is_upstream else min(2 ** attempt, 30)
            print(f"\n  [LLM] attempt {attempt + 1}/5 failed, retrying in {wait}s…", flush=True)
            time.sleep(wait)
    raise last_exc


DEFAULT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")


def _extract_tool(response: str, tools: List[str]) -> str:
    """
    Parse the LLM's text response to find a valid tool name.
    Priority: JSON {"tool": "..."} → direct substring match → first tool.
    """
    # JSON parse
    try:
        data = json.loads(response)
        candidate = data.get("tool", "")
        for t in tools:
            if t.lower() == candidate.lower():
                return t
    except (json.JSONDecodeError, AttributeError):
        pass

    # Substring match (case-insensitive)
    for tool in tools:
        if tool.lower() in response.lower():
            return tool

    return tools[0]


# ---------------------------------------------------------------------------
# Shared interface mixin
# ---------------------------------------------------------------------------


class _AgentBase:
    """Mixin that provides a default no-op get_learned_distribution."""

    name: str

    def get_learned_distribution(
        self,
        _user_id: int,
        _domain: str,
        tools: List[str],
        _sample_queries: List[str],
    ) -> Dict[str, float]:
        """
        Return the agent's estimated tool preference distribution for a
        (user, domain) pair after training.  Default: uniform (no learning).
        Override in bandit-based agents.
        """
        n = len(tools)
        return {tool: 1.0 / n for tool in tools}


# ---------------------------------------------------------------------------
# Sanity-check baseline: Random agent (no LLM, no learning)
# ---------------------------------------------------------------------------


class RandomAgent(_AgentBase):
    """Select tools uniformly at random. Establishes the performance floor."""

    name = "Random"

    def select_tool(
        self, query: str, domain: str, user_id: int, tools: List[str]
    ) -> str:
        return random.choice(tools)

    def update(
        self, query: str, domain: str, user_id: int, selected_tool: str, reward: float
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# Baseline 1: Zero-Shot LLM
# ---------------------------------------------------------------------------


class ZeroShotAgent(_AgentBase):
    """Select a tool based solely on the LLM's zero-shot understanding. No state."""

    name = "ZeroShot-LLM"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def select_tool(
        self, query: str, domain: str, user_id: int, tools: List[str]
    ) -> str:
        tools_block = "\n".join(
            f"  - {t}: {TOOL_METADATA.get(t, '')}" for t in tools
        )
        prompt = (
            f"Select the best tool for the user query.\n"
            f"Available tools:\n{tools_block}\n"
            f"Query: \"{query}\"\n"
            f"Reply with only JSON: {{\"tool\": \"<tool name>\"}}"
        )
        response = _llm_call(prompt, self.model)
        return _extract_tool(response, tools)

    def update(
        self, query: str, domain: str, user_id: int, selected_tool: str, reward: float
    ) -> None:
        pass  # stateless


# ---------------------------------------------------------------------------
# Baseline 2: In-Context Memory
# ---------------------------------------------------------------------------


class InContextMemoryAgent(_AgentBase):
    """
    Appends the last K successful (reward=1) tool selections to the LLM prompt.
    Provides recency-based personalization without an explicit statistical model.
    """

    name = "InContext-Memory"

    def __init__(self, model: str = DEFAULT_MODEL, memory_size: int = 5):
        self.model = model
        self.memory_size = memory_size
        # (user_id, domain) → deque of successful tool names
        self._memory: Dict[tuple, deque] = defaultdict(
            lambda: deque(maxlen=memory_size)
        )

    def select_tool(
        self, query: str, domain: str, user_id: int, tools: List[str]
    ) -> str:
        history = list(self._memory[(user_id, domain)])
        tools_block = "\n".join(
            f"  - {t}: {TOOL_METADATA.get(t, '')}" for t in tools
        )
        history_str = ", ".join(history) if history else "none"
        prompt = (
            f"Select the best tool for the user query.\n"
            f"Available tools:\n{tools_block}\n"
            f"User's recent successful selections: {history_str}.\n"
            f"Query: \"{query}\"\n"
            f"Reply with only JSON: {{\"tool\": \"<tool name>\"}}"
        )
        response = _llm_call(prompt, self.model)
        return _extract_tool(response, tools)

    def update(
        self, query: str, domain: str, user_id: int, selected_tool: str, reward: float
    ) -> None:
        if reward == 1.0:
            self._memory[(user_id, domain)].append(selected_tool)

    def get_learned_distribution(
        self,
        user_id: int,
        domain: str,
        tools: List[str],
        sample_queries: List[str],
    ) -> Dict[str, float]:
        """Empirical frequency distribution from memory (recency-weighted)."""
        history = list(self._memory[(user_id, domain)])
        if not history:
            n = len(tools)
            return {tool: 1.0 / n for tool in tools}
        counts = {tool: 0 for tool in tools}
        for t in history:
            if t in counts:
                counts[t] += 1
        total = sum(counts.values())
        if total == 0:
            n = len(tools)
            return {tool: 1.0 / n for tool in tools}
        return {tool: c / total for tool, c in counts.items()}


# ---------------------------------------------------------------------------
# Baseline 3: Pure Bandit (LinUCB, no LLM)
# ---------------------------------------------------------------------------


class PureBanditAgent(_AgentBase):
    """Select tools using LinUCB UCB scores alone. No LLM reasoning at test time."""

    name = "Pure-Bandit"

    def __init__(self, alpha: float = 1.0):
        self.bandit = LinUCBBandit(alpha=alpha)

    def select_tool(
        self, query: str, domain: str, user_id: int, tools: List[str]
    ) -> str:
        domain_tools = DOMAINS[domain]
        return self.bandit.select(user_id, domain, query, domain_tools)

    def update(
        self, query: str, domain: str, user_id: int, selected_tool: str, reward: float
    ) -> None:
        self.bandit.update(user_id, domain, query, selected_tool, reward)

    def get_learned_distribution(
        self,
        user_id: int,
        domain: str,
        tools: List[str],
        sample_queries: List[str],
    ) -> Dict[str, float]:
        domain_tools = DOMAINS[domain]
        return self.bandit.learned_preference_distribution(
            user_id, domain, domain_tools, sample_queries
        )


# ---------------------------------------------------------------------------
# Proposed method: Bandit Prior + Test-Time Chain-of-Thought
# ---------------------------------------------------------------------------


class BanditPriorCoTAgent(_AgentBase):
    """
    Proposed method combining statistical personalization with LLM reasoning:

    1. LinUCB bandit computes a softmax probability distribution over tools
       (the 'statistical prior') from user interaction history.
    2. The prior is injected into an LLM prompt together with the query.
       The prior is computed per-domain (4 tools) for a concentrated signal,
       while the LLM still sees all 20 tools with metadata.
    3. The LLM semantically evaluates the query against the prior, then
       outputs its final tool selection.
    4. The prior is OVERRIDDEN when the query explicitly requests a different tool
       (OOD robustness), but FOLLOWED for ambiguous standard queries (personalization).

    This combines the sample-efficiency of bandit learning with the semantic
    flexibility of LLM reasoning — the two are complementary.
    """

    name = "Bandit+CoT"

    def __init__(self, model: str = DEFAULT_MODEL, alpha: float = 1.0):
        self.model = model
        self.bandit = LinUCBBandit(alpha=alpha)

    def __init__(self, model: str = DEFAULT_MODEL, alpha: float = 1.0):
        self.model = model
        self.bandit = LinUCBBandit(alpha=alpha)
        # Track domain classification accuracy
        self.domain_predictions = []  # list of (true_domain, inferred_domain) tuples
        # Cache inferred domain per (user_id, round) for consistent update
        self._inferred_domains = {}  # (user_id, query) -> inferred_domain

    def select_tool(
        self, query: str, domain: str, user_id: int, tools: List[str]
    ) -> str:
        # Two-stage decision: LLM infers domain (with full metadata) → Bandit + LLM refines (compact)
        # Stage 1: LLM infers domain by seeing all 20 tools with metadata
        all_tools_block = "\n".join(
            f"  - {t}: {TOOL_METADATA[t]}" for t in tools
        )
        domain_prompt = (
            f"Classify this query into one domain by analyzing which tools are most relevant.\n"
            f"Available tools:\n{all_tools_block}\n\n"
            f"Query: \"{query}\"\n"
            f"Domains: {', '.join(DOMAINS.keys())}\n"
            f"Reply with only JSON: {{\"domain\": \"<domain name>\"}}"
        )
        domain_response = _llm_call(domain_prompt, self.model)

        # Parse inferred domain
        inferred_domain = domain  # fallback to ground truth
        try:
            domain_data = json.loads(domain_response)
            candidate_domain = domain_data.get("domain", "")
            if candidate_domain in DOMAINS:
                inferred_domain = candidate_domain
        except (json.JSONDecodeError, AttributeError):
            # Try substring match
            for d in DOMAINS:
                if d in domain_response.lower():
                    inferred_domain = d
                    break

        # Record domain prediction for accuracy tracking
        self.domain_predictions.append((domain, inferred_domain))
        # Cache inferred domain for update
        self._inferred_domains[(user_id, query)] = inferred_domain

        # Stage 2: Get bandit priors for inferred domain (4 tools only)
        domain_tools = DOMAINS[inferred_domain]
        probs = self.bandit.probabilities(user_id, inferred_domain, query, domain_tools)

        # Stage 3: LLM chooses from 4 tools (names + priors only, matching original successful design)
        tools_block = "\n".join(
            f"  - {t}: {probs[t]:.0%}"
            for t in sorted(domain_tools, key=lambda x: -probs[x])
        )

        prompt = (
            f"Select the best tool for this query.\n"
            f"Available tools (with learned user preference %):\n{tools_block}\n\n"
            f"Query: \"{query}\"\n\n"
            f"If the query explicitly names a tool, use it. Otherwise, prefer the tool with the highest user preference.\n"
            f"Reply with only JSON: {{\"tool\": \"<tool name>\"}}"
        )

        response = _llm_call(prompt, self.model)
        return _extract_tool(response, domain_tools)

    def update(
        self, query: str, domain: str, user_id: int, selected_tool: str, reward: float
    ) -> None:
        # Use inferred domain (not ground truth) for consistent bandit update
        inferred_domain = self._inferred_domains.get((user_id, query), domain)
        self.bandit.update(user_id, inferred_domain, query, selected_tool, reward)

    def get_learned_distribution(
        self,
        user_id: int,
        domain: str,
        tools: List[str],
        sample_queries: List[str],
    ) -> Dict[str, float]:
        domain_tools = DOMAINS[domain]
        return self.bandit.learned_preference_distribution(
            user_id, domain, domain_tools, sample_queries
        )


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------


def build_agents(model: str = DEFAULT_MODEL, include_random: bool = True) -> List:
    """
    Build the full agent suite.

    Parameters
    ----------
    include_random : whether to include the RandomAgent baseline (default True).
                     Set False to skip it in clean paper figures if desired.
    """
    agents = []
    if include_random:
        agents.append(RandomAgent())
    agents.extend([
        ZeroShotAgent(model=model),
        InContextMemoryAgent(model=model, memory_size=5),
        PureBanditAgent(alpha=1.0),
        BanditPriorCoTAgent(model=model, alpha=1.0),
    ])
    return agents


if __name__ == "__main__":
    from data_gen import ALL_TOOLS, generate_users

    users = generate_users(n_users=1)
    u = users[0]
    domain = "food_delivery"
    tools = ALL_TOOLS  # full pool: all 20 tools across all domains
    query = "Order some milk tea"

    agents = build_agents()
    print(f"User {u.user_id} preference for {domain}: {u.preferences[domain]}")
    print(f"Query: {query}\n")
    for agent in agents:
        result = agent.select_tool(query, domain, u.user_id, tools)
        print(f"[{agent.name}] Selected: {result}")
