# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Benchmark

```bash
# Install dependencies
pip install numpy matplotlib openai litellm

# Run with defaults (20 users, 50 rounds, gpt-4o-mini)
python main.py

# Override model, users, rounds
python main.py --model claude-haiku-3 --users 5 --rounds 20 --verbose

# Use Anthropic model via litellm
ANTHROPIC_API_KEY=sk-... python main.py --model claude-3-haiku-20240307

# Use OpenAI model
OPENAI_API_KEY=sk-... python main.py --model gpt-4o-mini
```

Output figures are saved to `./images/` automatically (never `plt.show()`).

## Quick Module Tests

```bash
python data_gen.py      # prints 3 sample users with query pool stats
python bandits.py       # shows LinUCB prior shift after 10 updates
python agents.py        # runs one select_tool call per agent (requires API key)
python env.py           # smoke test with a random dummy agent
```

## Architecture

The project implements a **Contextual Bandit + LLM Test-Time CoT** framework evaluated against three baselines in an online learning loop.

### Data Flow

```
data_gen.py  →  env.py  →  agents.py  →  metrics.py
                  ↑              ↑
             UserPersona    bandits.py
```

- **`data_gen.py`** — Generates 20 `UserPersona` objects. Each has a `preferences` dict (one ground-truth tool per domain) and a `query_pool` (200 pre-generated `(query, domain, is_ood, true_tool)` tuples; 10% are OOD queries that explicitly override the user's preference).

- **`bandits.py`** — `LinUCBBandit` maintains one `LinUCBArm` per `(user_id, domain, tool)` triplet. Features are 96-dim vectors from feature-hashing (query + tool + domain). Key method: `probabilities()` returns softmax-normalised UCB scores used as the statistical prior.

- **`agents.py`** — Four agents all share the interface `select_tool(query, domain, user_id, tools) -> str` + `update(...)`:
  1. `ZeroShotAgent` — pure LLM, no state
  2. `InContextMemoryAgent` — deque of last 3 successful selections per `(user_id, domain)`
  3. `PureBanditAgent` — wraps `LinUCBBandit.select()`, no LLM
  4. `BanditPriorCoTAgent` (**proposed**) — injects bandit `probabilities()` as a statistical prior into a `<thinking>...</thinking>` CoT prompt; LLM can override the prior for OOD queries

- **`env.py`** — `run_simulation()` iterates users × T rounds. For each round it calls every agent's `select_tool`, computes reward (exact string match against `true_tool`), then calls `update`. Returns `List[AgentResult]` where each result holds a flat list of `RoundRecord` objects.

- **`metrics.py`** — Computes and plots three metrics from `AgentResult.records`:
  1. Cumulative regret (`1 - mean_reward` summed over rounds)
  2. Rolling accuracy (convergence to 90% threshold)
  3. OOD accuracy in late rounds (after round `0.6 × T`)

### LLM Backend

`agents.py` tries `litellm` first; falls back to the `openai` SDK. The active model is controlled by the `AGENT_MODEL` env var (default `gpt-4o-mini`). `_extract_tool()` parses LLM responses: JSON `{"tool": "..."}` → regex → substring match → first tool as fallback.

### Reward Signal

Reward is binary: `1.0` if `selected_tool == true_tool`, else `0.0`. For standard queries `true_tool` is the user's preferred tool. For OOD queries `true_tool` is the tool explicitly named in the OOD query (breaking the user's usual preference).
