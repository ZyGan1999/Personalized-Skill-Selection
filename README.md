# Decouple Skill Selection as Local Harness for Personal Agents Under Implicit Preference

A sandbox for **personalized skill selection** in LLM personal agents, built around a **Local Harness** design: a lightweight, locally-running statistical policy makes the decision on every query, and the remote LLM is reserved as an exception handler for queries that explicitly name a tool. 
## Motivation

When an LLM agent must choose among multiple skills (e.g., which food-delivery app to call), the optimal choice depends on **user-specific preferences** that are not observable from the query text alone. Pure LLM policies are stateless and cannot personalize; pure statistical policies converge slowly and cannot read explicit user requests in the query. We study how to combine them so each handles what it is best at:

- **Local statistical prior** — accumulates personalized preference signals from feedback (frequency tables or bandit).
- **Remote LLM reasoning** — provides semantic understanding for queries that the prior cannot handle.

## Architecture

```
data_gen.py  ->  env.py  ->  agents.py  ->  metrics.py
                   ^             ^
              UserPersona   bandits.py
```

### Agents

Nine agents are evaluated, grouped into four families:

**No learning**
- **Random** — Uniform random selection (lower bound). 
- **ZeroShot-LLM** — Pure LLM, no personalization.

**LLM only**

- **InContext-Memory** — LLM with last-K successful selections in the prompt.
- **Profile-Memory** — LLM with structured success-rate profiles (simulates a real AI-agent memory module).

**Statistical only**
- **Freq-Greedy** — Pure frequency counting, no LLM, no exploration.
- **Pure-Bandit** — LinUCB only, no LLM at decision time.

**Local Harness hybrids (statistical + LLM)**
- **Bandit-as-Context** — LinUCB prior injected into the LLM CoT prompt; LLM produces the final choice.
- **Freq-as-Override** — Frequency-greedy by default; LLM only overrides for explicit queries.
- **Bandit-as-Override** (**proposed**) — LinUCB selects by default; LLM only overrides for explicit queries.

### Design

1. **Domain classification** (shared): one LLM call per query, result cached and shared across LLM-using agents.
2. **Skill selection**: Statistical priors chooses the skill with the highest score.
3. **LLM override**: a second LLM call checks whether the query explicitly names a skill; if yes, the bandit choice is overridden.

### Data
- **ToolBench-60**: 10 domains x 6 skills = 60 real APIs from ToolBench (via `--benchmark benchmark_data/toolbench_60.json`).
- **Per-user preferences**: one-hot (single preferred skill per domain) or soft (Dirichlet-sampled distribution, concentration alpha).
- **OOD queries**: explicitly name a non-preferred skill, testing override ability.

## Quick Start

### Install dependencies

```bash
pip install numpy matplotlib openai litellm tqdm scipy
```

### Run the benchmark

```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Default run (20 users, 50 rounds, gpt-4o-mini)
python main.py

# Custom model and configuration
python main.py --model gpt-4o-mini --users 20 --rounds 50 --verbose

# Quick debug (no LLM calls, Random + Pure-Bandit only)
python main.py --dry-run --users 5 --rounds 20

# With checkpoint / resume (recovers from API failures)
python main.py --model gpt-4o-mini --output-dir gpt-4o-mini --resume
```

### Soft-preference experiments

```bash
# Dirichlet-sampled preferences (concentration controls peakedness)
python main.py --soft-preferences --concentration 0.5 --output-dir soft-c0.5
python main.py --soft-preferences --concentration 2.0 --output-dir soft-c2.0

# Different OOD ratios
python main.py --ood-ratio 0.20 --output-dir ood20
python main.py --ood-ratio 0.50 --output-dir ood50
```

### Module-level tests

```bash
python data_gen.py     # prints sample users with query-pool stats
python bandits.py      # shows LinUCB prior shift after updates
python agents.py       # runs one select_tool call per agent (requires API key)
python env.py          # smoke test with a random dummy agent
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | gpt-4o-mini | LLM model identifier |
| `--users` | 20 | Synthetic users per seed |
| `--rounds` | 50 | Interaction rounds per user |
| `--seeds` | 0 1 2 3 4 | Random seeds for independent trials |
| `--ood-ratio` | 0.10 | Fraction of explicit queries |
| `--pool-size` | 200 | Queries per user pool |
| `--soft-preferences` | off | Use Dirichlet soft preferences |
| `--concentration` | 2.0 | Dirichlet concentration (lower = more peaked) |
| `--temperature` | 3.0 | Softmax temperature for bandit prior (higher = sharper) |
| `--benchmark` | none | Path to benchmark config JSON (e.g., `benchmark_data/toolbench_60.json`) |
| `--test-pool-size` | 50 | Held-out test queries per user (0 to disable) |
| `--wandb-project` | none | W&B project name (enables wandb logging) |
| `--wandb-run-name` | output-dir name | Base name for wandb runs (appended with `-seed{N}`) |
| `--resume` | off | Enable checkpoint / resume |
| `--output-dir` | {model}/ | Results directory |
| `--dry-run` | off | Skip LLM agents for fast debugging |
| `--export-csv` | off | Export raw records to CSV |
| `--verbose` | off | Print per-seed progress |

## Output

Results are saved to `{output-dir}/`:

```
{output-dir}/
  images/
    cumulative_regret.pdf/png       # Regret over rounds (lower = better)
    rolling_accuracy.pdf/png        # Convergence to 90% threshold
    ood_robustness.pdf/png          # Accuracy on OOD queries in late rounds
    preference_recovery.pdf/png     # How well agents learn user preferences
    preference_alignment.pdf/png    # Per-user per-domain alignment heatmap
    per_domain_accuracy.pdf/png     # Accuracy breakdown by domain
    domain_classification_accuracy.pdf/png  # LLM domain inference accuracy
  summary.json                      # Aggregated statistics
  results.csv                       # Raw per-round records (if --export-csv)
  checkpoints/                      # Resume checkpoints (if --resume)
```

In soft-preference mode, additional heatmaps are generated:
- `preference_alignment_kl.pdf` — KL divergence between learned and true distributions.
- `preference_alignment_spearman.pdf` — Spearman rank correlation.


## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI-compatible endpoint |
| `OPENAI_API_BASE` | Custom API base URL (for proxies) |
| `OPENAI_CHAT_URL` | Complete endpoint URL (takes priority over `OPENAI_API_BASE`); auto-detects Responses format when ending with `/responses` |
| `OPENAI_API_FORMAT` | `chat` (default) or `responses` — selects between `/v1/chat/completions` and `/v1/responses` payload formats |
| `ANTHROPIC_API_KEY` | API key for Anthropic models (via litellm) |
| `AGENT_MODEL` | Override model (same as `--model`) |

## License

MIT
