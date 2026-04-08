# Changelog

All notable changes to the Tool-Call-Bandit project are documented here.

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
