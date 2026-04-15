# Changelog

All notable changes to the Tool-Call-Bandit project are documented here.

---

## 2026-04-14 — Bandit+Override Agent, Frequency Greedy Baseline & Progress Viewer

### New Agent: Bandit+Override (Proposed Method 2)
- Bandit makes the default tool selection (greedy UCB argmax, same as Pure-Bandit)
- LLM is invoked only to check whether the query explicitly names a tool (OOD detection)
- If yes: LLM overrides the bandit's choice with the named tool
- If no: bandit's selection is used as-is
- Design rationale: eliminates LLM selection noise on standard queries while retaining OOD robustness
- Expected to achieve Pure-Bandit-level accuracy on standard queries + OOD handling

### New Baseline: Frequency Greedy
- Pure frequency counting — always selects the tool with the highest historical success rate
- No LLM, no exploration mechanism (greedy after initial round-robin)
- Demonstrates the value of LinUCB's exploration-exploitation mechanism vs naive statistics

### Softmax Temperature
- Added `temperature` parameter to `LinUCBBandit.probabilities()` (default 3.0)
- Higher temperature amplifies score differences, making bandit prior more informative for LLM
- Added `--temperature` CLI flag
- Example: UCB [1.21, 1.0, 1.0, 1.0] → temp=1: [29%, 24%, 24%, 24%] → temp=3: [38%, 21%, 21%, 21%]

### Progress Viewer
- Added `plot_progress.py`: reads mid-seed checkpoint from a running experiment and generates metrics plots
- Allows monitoring multi-day experiments without waiting for completion

### Oracle Baseline in Rolling Accuracy Plot
- Added "Oracle" line showing theoretical upper bound accuracy
- Computed as: fraction of queries where the most frequent true_tool per (user, domain) matches

### Key Experimental Insight
- With 10 domains × 6 tools and 50 rounds: each (user, domain) gets only ~5 rounds → insufficient for bandit convergence
- Increased to 500 rounds for proper evaluation
- Bandit+CoT underperforms Profile-Memory because LLM introduces selection noise even with correct bandit prior; Bandit+Override addresses this

---

## 2026-04-10 — Benchmark Scaling, Shared Domain Classification, ProfileMemory Baseline & Temperature Tuning

### Benchmark Scaling: ToolBench Integration
- Added `extract_tools.py`: extracts real API tools from ToolBench data (3,451 tools across 49 categories)
- Added `generate_queries.py`: LLM-based batch generation of standard and OOD queries for any tool set
- Added `load_benchmark()` to `data_gen.py`: loads tool definitions from JSON config files, replacing hardcoded 20-tool setup
- Added `--benchmark` CLI flag to `main.py`: switch between built-in benchmark and custom configs
- Changed module imports in `env.py` and `agents.py` to `import data_gen` (module reference) so `load_benchmark()` updates are visible globally
- Created `benchmark_data/toolbench_60.json`: 60 real APIs across 10 domains (Finance, Sports, Travel, Entertainment, Gaming, Education, Communication, Location, eCommerce, Social)

### Shared Domain Classification (Fair Comparison)
- Added `_classify_domain()` in `env.py`: one LLM call per query, result shared across all agents
- All agents (including Pure-Bandit) now use the same inferred domain — no agent gets ground truth for free
- Removed Stage 1 (domain inference) from `BanditPriorCoTAgent` — it now only does Stage 2 (bandit prior + LLM tool selection), saving one LLM call per round
- Domain classification accuracy tracked at `env.py` level via `SeedResult.domain_predictions`
- Updated `plot_domain_classification_accuracy()` in `metrics.py` to read from `SeedResult`

### New Baseline: ProfileMemoryAgent
- Simulates how current AI agents (Claude Memory, ChatGPT Memory) handle personalization
- Maintains per-(user, domain, tool) success/attempt statistics
- Injects structured profile into LLM prompt: "Tool X: 8/10 successful (80%)"
- A strong baseline that accumulates knowledge over all rounds (unlike InContext-Memory's 5-item window)

### Softmax Temperature for Bandit Prior
- Added `temperature` parameter to `LinUCBBandit.probabilities()` (default: 3.0)
- Higher temperature sharpens the prior distribution, making it more informative for the LLM
- Example: UCB [1.21, 1.0, 1.0, 1.0] → temp=1: [29%, 24%, 24%, 24%] → temp=3: [38%, 21%, 21%, 21%]
- Added `--temperature` CLI flag for easy tuning

### Oracle Baseline in Rolling Accuracy Plot
- Added "Oracle" line: theoretical upper bound if always picking the most frequent true_tool per (user, domain)
- For one-hot mode: ~90% (standard queries all correct, OOD all wrong)
- Provides visual reference for how far agents are from optimal

### Key Experimental Findings
- toolbench_100 (10 domains × 10 tools): domain classification only 74%, Bandit+CoT underperformed Pure-Bandit
- toolbench_60 (10 domains × 6 tools) + improved Stage 1 prompt: domain classification improved to 94.5%
- Shared domain classification eliminates the unfair advantage Pure-Bandit had from using ground truth domain

---

## 2026-04-07 — Soft Preferences, Multi-Metric Evaluation & Checkpoint/Resume

**Commit**: `ea732db`

### Soft Preference Support
- Added Dirichlet-sampled soft preferences (`--soft-preferences --concentration <float>`)
- New `_generate_query_pool_soft()`: standard queries sample `true_tool` from the user's soft distribution instead of always using a fixed preferred tool
- `generate_users_soft()` now uses the soft pool generator
- Enables ablation experiments across different concentration values (e.g., c=0.5, 1.0, 2.0, 5.0)

### Multi-Metric Evaluation for Soft Preferences
- Added KL divergence and Spearman rank correlation alongside cosine similarity
- `compute_preference_recovery()` now returns per-metric dictionaries in soft mode
- `plot_preference_recovery()` generates a 3-panel bar chart (Cosine Sim, KL Div, Spearman) in soft mode
- `plot_preference_alignment()` generates three separate heatmaps in soft mode (cosine, KL, Spearman)
- Refactored alignment heatmap code into reusable `_build_alignment_matrices()` and `_plot_alignment_heatmap()` helpers

### Checkpoint/Resume
- Two-level checkpoint system: seed-level (after each seed completes) and user-level (after each user's rounds complete)
- Checkpoints saved via pickle to `{output_dir}/checkpoints/`
- `--resume` flag: skips completed seeds, resumes mid-seed from last completed user
- Preserves agent state (bandit A/b matrices, memory deques), partial records, and RNG state

### Bug Fix
- Fixed `InContextMemoryAgent` pickle failure: replaced unpicklable lambda with `_DequeFactory` class

### Key Insight
- Cosine similarity is a poor metric for soft preferences (always ~90% due to non-negative vector geometry)
- Spearman rank correlation provides the best discrimination between agents
- Interesting finding: Bandit+CoT has lower Spearman than Pure-Bandit (LLM overrides pollute bandit learning signal) but higher accuracy (LLM reasoning compensates)

---

## 2026-03-25 — Robustness Improvements & Ablation Results

**Commit**: `be7d944`

- Added ablation experiment results for different OOD ratios
- Improved overall robustness of the experimental pipeline

---

## 2026-03-24 — Critical Fix: UCB vs Exploit-Only Scores

**Commit**: `e5ce170`

- **Root cause**: `probabilities()` was accidentally changed to use exploit-only scores, which produced uniform priors that the LLM ignored
- **Fix**: Restored UCB scores (exploit + explore) for `probabilities()`, matching the original successful design
- This was a critical regression that severely degraded Bandit+CoT performance

---

## 2026-03-23 — Two-Stage Architecture Stabilization

**Commits**: `83fd551`, `a1cfd72`, `4eaba31`, `bec9302`, `b15a9fc`

### Error Handling
- Added 30s retry wait for 5xx errors (API proxy is prone to transient failures)
- Catch `ChunkedEncodingError` and `JSONDecodeError` (empty body) in retry loop

### Prompt Engineering Iterations
- Settled on the final two-stage design:
  - **Stage 1**: LLM sees all 20 tools with metadata, infers domain
  - **Stage 2**: LLM sees only 4 domain tools with bandit prior percentages (no metadata)
- Key lesson: Stage 2 prompt must be simple (names + prior %) for the LLM to properly weight the bandit signal
- Including tool metadata in Stage 2 caused the LLM to ignore bandit priors

### Consistency Fix
- `BanditPriorCoTAgent.update()` now uses inferred domain (not ground-truth) for bandit updates, ensuring consistency between selection and learning

---

## 2026-03-22 — Two-Stage Bandit+CoT Design

**Commits**: `b18c6ef`, `140bbc9`, `9926ce7`, `563bd8b`, `ac64f01`, `accfa99`

### Architecture Evolution
1. **Initial approach**: Show all 20 tools to LLM with bandit priors — too many tools degraded performance
2. **Attempt 1**: Bandit pre-filters top-6 from 20, LLM refines — inconsistent candidate sets
3. **Attempt 2**: Guarantee domain tools in candidate set (4 domain + 4 top others) — still noisy
4. **Final design (Plan B)**: LLM infers domain first (Stage 1), then selects from 4 domain tools with bandit priors (Stage 2)

### New Features
- Domain classification accuracy tracking and plotting
- PNG output alongside PDF for all figures (150 dpi)
- Simplified to 4-tool per-domain selection for cleaner bandit signal

---

## 2026-03-20–21 — Initial Implementation

**Commits**: `0466856`, `81b1b6b`, `6af18e4`

### Initial Commit
- Full benchmark framework: `data_gen.py`, `bandits.py`, `agents.py`, `env.py`, `metrics.py`, `main.py`
- 5 domains (financial, food_delivery, navigation, shopping, entertainment) with 4 tools each
- 4 agents: ZeroShot, InContext-Memory, Pure-Bandit, Bandit+CoT (proposed)
- LinUCB with 96-dim feature-hashed context vectors
- OOD query system with explicit tool-naming templates
- Multi-seed evaluation with confidence intervals

### Early Fixes
- Added `ChunkedEncodingError` handling in LLM retry loop
- Experimented with feeding all 20 tools (with metadata) to agents — led to performance degradation that motivated the two-stage redesign
