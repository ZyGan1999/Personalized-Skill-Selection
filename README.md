# Tool-Call-Bandit

A benchmark for **personalized tool selection** combining contextual bandits with LLM reasoning. The system learns which tools work best for each user through online interaction, using LinUCB to build statistical priors and LLM chain-of-thought to make context-aware decisions.

## Motivation

When an LLM-based assistant needs to choose among multiple tools (e.g., which food delivery app to use), the optimal choice depends on **user-specific preferences** that aren't observable from the query alone. This benchmark studies how to combine:

- **Bandit learning**: accumulate personalized preference signals from user feedback
- **LLM reasoning**: leverage semantic understanding of queries and tool capabilities

Neither alone is sufficient — bandits are slow to converge in limited rounds, and LLMs lack personalization without historical context.

## Architecture

```
data_gen.py  →  env.py  →  agents.py  →  metrics.py
                  ↑              ↑
             UserPersona    bandits.py
```

### Agents

| Agent | Description |
|-------|-------------|
| **RandomAgent** | Uniform random selection (lower bound) |
| **ZeroShot-LLM** | Pure LLM, no personalization |
| **InContext-Memory** | LLM with last-K successful selections in prompt |
| **Pure-Bandit** | LinUCB only, no LLM at test time |
| **Bandit+CoT** (proposed) | LinUCB prior injected into LLM CoT prompt |

### Bandit+CoT: Two-Stage Design

1. **Stage 1 — Domain Inference**: LLM sees all 20 tools with descriptions, infers the query's domain
2. **Stage 2 — Tool Selection**: LLM sees 4 domain-specific tools with bandit-learned preference percentages, selects the best tool

The bandit prior gives the LLM a "soft hint" about user preferences. For standard queries, the LLM follows the prior. For OOD queries (where the user explicitly names a different tool), the LLM overrides it.

### Data

- **5 domains**: financial, food delivery, navigation, shopping, entertainment
- **4 tools per domain** (20 total), with Chinese market tools (Meituan, Ele.me, DiDi, etc.)
- **Per-user preferences**: one-hot (fixed preferred tool) or soft (Dirichlet-sampled distribution)
- **OOD queries**: explicitly name a non-preferred tool, testing override ability

## Quick Start

### Install Dependencies

```bash
pip install numpy matplotlib openai litellm tqdm scipy
```

### Run Benchmark

```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Default run (20 users, 50 rounds, gpt-4o-mini)
python main.py

# Custom model and configuration
python main.py --model gpt-4o-mini --users 20 --rounds 50 --verbose

# Quick debug (no LLM calls, Random + PureBandit only)
python main.py --dry-run --users 5 --rounds 20

# With checkpoint/resume (recovers from API failures)
python main.py --model gpt-5.2 --output-dir gpt-5.2 --resume
```

### Soft Preference Experiments

```bash
# Dirichlet-sampled preferences (concentration controls peakedness)
python main.py --soft-preferences --concentration 0.5 --output-dir soft-c0.5
python main.py --soft-preferences --concentration 2.0 --output-dir soft-c2.0

# Different OOD ratios
python main.py --ood-ratio 0.20 --output-dir ood20
python main.py --ood-ratio 0.50 --output-dir ood50
```

### Module-Level Tests

```bash
python data_gen.py      # prints sample users with query pool stats
python bandits.py       # shows LinUCB prior shift after updates
python agents.py        # runs one select_tool call per agent (requires API key)
python env.py           # smoke test with a random dummy agent
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | gpt-4o-mini | LLM model identifier |
| `--users` | 20 | Synthetic users per seed |
| `--rounds` | 50 | Interaction rounds per user |
| `--seeds` | 0 1 2 3 4 | Random seeds for independent trials |
| `--ood-ratio` | 0.10 | Fraction of OOD queries |
| `--pool-size` | 200 | Queries per user pool |
| `--soft-preferences` | off | Use Dirichlet soft preferences |
| `--concentration` | 2.0 | Dirichlet concentration (higher = more peaked) |
| `--resume` | off | Enable checkpoint/resume |
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

In soft preference mode, additional heatmaps are generated:
- `preference_alignment_kl.pdf` — KL divergence between learned and true distributions
- `preference_alignment_spearman.pdf` — Spearman rank correlation

## Key Results

### One-Hot Preferences (GPT-5.2, 20 users, 50 rounds)

- **Bandit+CoT** achieves the lowest cumulative regret and highest accuracy
- **OOD robustness**: Bandit+CoT and LLM-based agents achieve ~100% OOD accuracy (can detect explicit tool requests); Pure-Bandit fails (~25%)
- **Convergence**: Bandit+CoT converges fastest due to LLM reasoning accelerating cold-start

### Soft Preferences

- With soft (stochastic) preferences, accuracy ceiling drops significantly (e.g., ~35% for concentration=2.0)
- **Spearman rank correlation** is the most discriminative metric (cosine similarity saturates at ~90%)
- Interesting finding: Pure-Bandit has better Spearman (cleaner learning signal) but worse accuracy than Bandit+CoT (LLM reasoning compensates)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI models |
| `OPENAI_API_BASE` | Custom API base URL (for proxies) |
| `ANTHROPIC_API_KEY` | API key for Anthropic models (via litellm) |
| `AGENT_MODEL` | Override model (same as `--model`) |

## License

MIT
