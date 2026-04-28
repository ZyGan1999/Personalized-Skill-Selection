# Experimental Plan

This document describes the full experimental protocol for the paper.

## Story

When LLM-based agents handle personalized tool selection, they face two challenges:
1. **Personalization** — user preferences must be discovered through interaction
2. **OOD robustness** — users may explicitly request specific tools, overriding learned preferences

Existing approaches handle one but not both:
- Pure statistical methods (LinUCB, frequency counting) learn preferences but fail on OOD
- LLM-based methods (ZeroShot, Profile-Memory) handle OOD but learn slowly

We propose **Bandit+Override**: LinUCB selects by default, LLM only intervenes for OOD detection. This decouples personalization (handled statistically) from semantic interpretation (handled by LLM).

## Agent Categories

Agents are organized into four categories with consistent visual encoding:

| Category | Color | Agents |
|----------|-------|--------|
| No learning | gray | Random |
| Statistical only | blues | Freq-Greedy, Pure-Bandit |
| LLM only | oranges/reds | ZeroShot-LLM, InContext-Memory, Profile-Memory |
| Statistical + LLM (hybrid) | greens/purples | Bandit+CoT, Freq+Override, **Bandit+Override** (proposed) |

## Default Setup

- Benchmark: `benchmark_data/toolbench_60.json` (10 domains × 6 tools = 60 real APIs from ToolBench)
- 50 users per seed
- 500 training rounds per user
- 50 held-out test queries per user
- 3 random seeds (0, 1, 2) for paired t-test
- Softmax temperature for bandit prior: 3.0
- OOD ratio: 0.10
- Shared LLM domain classification across agents

---

## Phase 1: Main Results (Required)

**Goal**: Table 1 (comprehensive comparison) and Figure 1 (cumulative regret), Figure 2 (OOD accuracy).

### Exp 1.1: One-hot Preferences (per model)

```bash
python main.py --benchmark benchmark_data/toolbench_60.json \
    --model <MODEL> --users 50 --rounds 500 --seeds 0 1 2 \
    --temperature 3.0 --resume --verbose \
    --output-dir paper-exp/main-onehot-<MODEL>
```

### Exp 1.2: Soft Preferences (concentration=0.3, per model)

```bash
python main.py --benchmark benchmark_data/toolbench_60.json \
    --model <MODEL> --users 50 --rounds 500 --seeds 0 1 2 \
    --soft-preferences --concentration 0.3 \
    --temperature 3.0 --resume --verbose \
    --output-dir paper-exp/main-soft0.3-<MODEL>
```

**Models to test**:
- API models: gpt-5.2 (done), glm-5, kimi-k2.5, minimax-m2.5, qwen3-max
- Local models: TBD (to be selected)

---

## Phase 2: Ablation Studies

### Exp 2.1: Concentration Sweep (Core Ablation)

Demonstrates that bandit's exploration mechanism becomes essential as preference uncertainty increases.

```bash
for c in 0.1 0.3 1.0 2.0 5.0; do
  python main.py --benchmark benchmark_data/toolbench_60.json \
      --model gpt-5.2 --users 50 --rounds 500 --seeds 0 1 2 \
      --soft-preferences --concentration $c \
      --temperature 3.0 --resume --verbose \
      --output-dir paper-exp/sweep-c$c
done
```

**Expected finding**: (Bandit+Override - Freq+Override) regret gap increases with concentration value.

### Exp 2.2: Rounds Sweep (Sample Efficiency)

Demonstrates bandit's advantage under data sparsity (cold-start scenarios).

```bash
for r in 50 100 200 500; do
  python main.py --benchmark benchmark_data/toolbench_60.json \
      --model gpt-5.2 --users 50 --rounds $r --seeds 0 1 2 \
      --soft-preferences --concentration 0.3 \
      --temperature 3.0 --resume --verbose \
      --output-dir paper-exp/rounds-$r
done
```

---

## Phase 3: Robustness Experiments

### Exp 3.1: OOD Ratio Sweep

```bash
for ood in 0.05 0.20 0.30; do
  python main.py --benchmark benchmark_data/toolbench_60.json \
      --model gpt-5.2 --users 50 --rounds 500 --seeds 0 1 2 \
      --ood-ratio $ood --soft-preferences --concentration 0.3 \
      --temperature 3.0 --resume --verbose \
      --output-dir paper-exp/ood-$ood
done
```

### Exp 3.2: Default 20-Tool Benchmark

```bash
python main.py --model gpt-5.2 --users 50 --rounds 500 --seeds 0 1 2 \
    --soft-preferences --concentration 0.3 \
    --temperature 3.0 --resume --verbose \
    --output-dir paper-exp/default20
```

---

## Phase 4: Analysis (Post-hoc, no new experiments)

- Domain classification accuracy vs final regret correlation
- Test set accuracy comparison (ensures no train/test leakage)
- Per-user variance and convergence patterns

---

## Required Tables and Figures

### Table 1: Main Results (Soft Preference, c=0.3)

| Agent | Final Regret ↓ | Test Acc ↑ | OOD Acc ↑ | Recovery ↑ |
|-------|----------------|------------|-----------|------------|
| Random | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| **Bandit+Override** | **...** | **...** | **...** | **...** |

Mark significance with paired t-test (\*, \*\*, \*\*\*).

### Figure 1: Cumulative Regret over Time

Soft preference setting, all 8 agents with 95% CI.

### Figure 2: OOD Robustness Bar Chart

Bar chart showing OOD accuracy. Highlights gap between statistical-only and override methods.

### Figure 3: Concentration Sweep (Ablation)

Line plot: x-axis = concentration, y-axis = regret. Shows Bandit+Override vs Freq+Override gap widens as concentration grows.

### Figure 4: Rounds Sweep (Sample Efficiency)

Bar chart at different rounds. Demonstrates bandit's cold-start advantage.

---

## Compute Budget

- Each main experiment: ~6-8 hours per seed × 3 seeds = ~20 hours
- 5 API models × 2 settings (one-hot + soft) = 10 experiments × 20h = **~200 hours of API calls**
- Phase 2-3 ablations: ~10 additional experiments × 20h = ~200 hours

**Total estimated wall-clock**: ~2 weeks of continuous experimentation.

## Execution Priority

1. **High priority** (must include):
   - Phase 1 (Exp 1.1, 1.2) for all models — main results
   - Phase 2.1 (Concentration sweep) on gpt-5.2 — core ablation

2. **Medium priority** (strongly recommended):
   - Phase 2.2 (Rounds sweep)
   - Phase 3.1 (OOD ratio sweep)

3. **Lower priority** (time permitting):
   - Phase 3.2 (Default 20-tool benchmark)
   - Local model experiments
