# Personalized Tool Selection via Bandit Default + LLM Override

A paper-organized synthesis of the project's motivation, formulation, method, and empirical findings to date. Intended as a planning document for writing the actual paper, not a draft of the paper itself.

---

## 0. Working Title & One-Sentence Pitch

**Working title**: *Bandit by Default, LLM by Exception: Decoupling Personalization from Semantic Interpretation in Tool-Using Agents*

**One-sentence pitch**: When an LLM agent must pick the right tool for a user, the *who* (which user) is a statistical problem and the *what* (what does this specific query require) is a semantic problem; we show that handling them with the right primitive—a contextual bandit by default and the LLM only as an exception handler—dominates both pure-statistical and pure-LLM baselines, including LLM-with-memory variants that resemble current production agents.

---

## 1. Motivation & Problem Statement

### 1.1 The Personalized Tool-Selection Problem

Modern LLM-based agents (Claude, ChatGPT, Cursor, Cline, etc.) often face the following micro-decision repeatedly: given a user query, choose among a set of tools / APIs / sub-services. For many such choices the right answer depends on **who the user is**, not only on **what the query says**:

- "Order lunch" → Meituan or Ele.me? Depends on the user's preferred app.
- "Book a ride" → Didi or Caocao? Depends on local availability and personal preference.
- "Pay" → Alipay or WeChat Pay? Personal habit.
- "Look up code" → Stack Overflow or company internal wiki? Role-specific.

These preferences are not in the query text and not in the tool documentation. They emerge only through user feedback.

### 1.2 Two Coupled but Distinct Sub-Problems

Personalized tool selection is the composition of two sub-problems that the literature usually treats as one:

1. **Personalization (statistical)**: estimate $P(\text{best tool} \mid \text{user}, \text{domain})$ from a stream of (action, reward) pairs. This is fundamentally a **contextual bandit / online learning** problem.
2. **OOD / explicit override (semantic)**: when the user's query *explicitly* names a non-default tool (e.g. "Use *Alipay*, not WeChat Pay, for this transaction"), the learned preference must be **overridden** based on reading the query. This is fundamentally a **semantic / NLU** problem.

The two sub-problems have *opposite* failure modes:
- A pure bandit converges to the user's habitual choice and is **brittle to OOD requests** that explicitly demand a different tool.
- A pure LLM (zero-shot or with memory in context) handles the semantic case well, but **cannot reliably accumulate personalization signal** through interaction — its preference estimate is unstable, prompt-sensitive, and adds noise even to easy cases.

### 1.3 How Current AI Agents Handle This

Production LLM-based agents today (Claude with memory, ChatGPT memory, Cursor user-rules) take the **LLM-only** approach:
- Persist some structured "memory" or "profile" about the user.
- Stuff it into the prompt.
- Let the LLM make the decision.

This is essentially our `ProfileMemory` baseline. It has the right intuition (accumulate evidence over time) but the wrong primitive (LLM as decision-maker for a problem that is, at its core, *credit assignment under uncertainty*).

### 1.4 Our Thesis

> The right architecture is the **inverse** of the current default. Make the statistical learner the *primary* decision-maker, and invoke the LLM **only as an exception handler** for queries that override the statistical estimate.

This decoupling has three consequences worth proving empirically:

- **(C1)** Performance on standard (non-OOD) queries should approach the contextual-bandit ceiling, because the LLM does not introduce selection noise.
- **(C2)** Performance on OOD queries should approach the LLM ceiling, because the LLM is still consulted (just narrowly).
- **(C3)** The advantage over a frequency-counting variant (no exploration) should *grow* as user preferences become more stochastic — this is where exploration matters.

We provide a controlled benchmark and report results across **multiple LLM backbones** (different sizes, different reasoning styles) to demonstrate that the *architectural* claim holds across models, not just for one strong model.

---

## 2. Contributions

1. **Formalization**: We formulate personalized tool selection as a contextual-bandit problem with an OOD-override channel and identify the architectural trade-off that current LLM agents make implicitly.
2. **Method**: We propose `Bandit+Override`, in which a LinUCB bandit selects by default and the LLM is invoked only to check for an explicit-tool-name override. We also propose `Freq+Override` as an ablation to isolate the value of bandit exploration vs. naive frequency counting.
3. **Benchmark**: A controllable simulator with 60 real APIs (ToolBench-derived) across 10 domains, with both one-hot and Dirichlet-soft user preferences, a held-out test split, and a shared LLM domain classifier for fair cross-agent comparison.
4. **Empirical claims**, validated across at least three different LLM backbones:
   - Under one-hot preferences, statistical and override-based methods dominate; LLM-only methods (incl. ProfileMemory) cannot match them.
   - Under soft preferences (concentration = 0.3), `Bandit+Override` strictly dominates `Freq+Override` *and* `Pure-Bandit`, showing both exploration and OOD-override are needed.
   - `Bandit+CoT` (LLM uses bandit prior internally) underperforms `Bandit+Override` (LLM only overrides) for **all tested models** and degrades sharply on smaller backbones — i.e. handing the bandit's prior to the LLM as part of CoT *hurts*. This is the central methodological finding.

---

## 3. Related Work

This section needs filling out with specific citations, but the structure is:

### 3.1 Contextual Bandits for Recommendation / Tool Use
- LinUCB (Li et al. 2010) and follow-ups; non-stationary bandits; Thompson sampling. We use LinUCB but the framework is generic.
- Bandits in conversational recommendation.

### 3.2 LLM Agents and Tool Use
- Toolformer, ReAct, function-calling APIs (OpenAI / Anthropic).
- Recent work on tool / API selection from large pools (ToolBench, RestGPT, AnyTool).
- **Gap**: most of this literature evaluates *zero-shot* tool selection from documentation, not *personalized* selection that depends on user identity.

### 3.3 Memory-Augmented LLMs
- Long-term memory systems for chat agents (MemGPT, ChatGPT memory, Claude memory).
- Profile / persona-based personalization in dialog (PersonaChat etc.).
- **Gap**: these systems treat memory as additional context for the LLM, not as a separate statistical primitive. Our `ProfileMemory` baseline is a strong representative of this paradigm.

### 3.4 Hybrid LLM + Symbolic / Statistical Systems
- LLM-as-controller papers (e.g. AutoGen, HuggingGPT) — LLM dispatches to specialists.
- LLM-as-prior in RL (Chen et al., Yao et al.). Closest to our `Bandit+CoT` baseline.
- **Gap**: most hybrid work either (a) uses LLM as the high-level controller and a symbolic system as the worker, or (b) uses LLM as the worker and a learned policy as the controller. We argue for an inverted division of labor at the *decision* level itself: statistical primary, LLM exceptional.

### 3.5 Position to Stake Out
Our work sits in the intersection of (3.1) and (3.3): we agree with the memory literature that personalization is important, but disagree on *what should hold the memory and make the decision*. The contextual-bandit framing makes the problem precise and lets us measure both regret and OOD robustness on the same axis.

---

## 4. Problem Formulation

### 4.1 Setup

- A user $u \in \mathcal{U}$ at round $t$ issues a query $q_t \sim \mathcal{Q}_u$.
- A finite tool set $\mathcal{T}$ partitioned into domains $\mathcal{D}$; $\mathcal{T} = \bigsqcup_{d \in \mathcal{D}} \mathcal{T}_d$ with $|\mathcal{T}_d| = K$ tools per domain.
- Each query has a *latent* ground-truth domain $d^*(q_t)$ and a *latent* correct tool $a^*(q_t, u)$.
- The agent picks $a_t \in \mathcal{T}$ and receives reward $r_t = \mathbb{1}\{a_t = a^*(q_t, u)\}$.

### 4.2 Two Query Types

For each user $u$ we generate a query pool with two query types:

- **Standard queries** (90% by default): $a^*(q_t, u) = \pi_u(d^*(q_t))$, where $\pi_u$ is the user's preference function over domain tools. The correct answer is *unobservable from the query text alone* — only the user's preference matters.
- **OOD queries** (10% by default): the query *explicitly names* a specific tool $\tau$, and $a^*(q_t, u) = \tau$ regardless of $\pi_u$. The correct answer is *fully determined by the query text* and overrides personalization.

### 4.3 Two Preference Models

User preference $\pi_u(d)$ over the $K$ tools in domain $d$:

- **One-hot** (deterministic preference): $\pi_u(d)$ is a single tool; $a^*(q_t, u)$ is that tool for every standard query in domain $d$. Models a user with strong habit.
- **Soft / Dirichlet** (stochastic preference): $\pi_u(d) \sim \text{Dir}(\alpha \cdot \mathbf{1}_K)$ is a probability distribution; for each standard query, the correct tool is sampled from this distribution. Models a user with graded, probabilistic preference. Lower $\alpha$ (concentration) → more peaked distribution; higher $\alpha$ → more uniform. Default soft setting: $\alpha = 0.3$.

The soft setting is **essential** for revealing the bandit's true advantage: under deterministic preferences, simple frequency counting converges as fast as LinUCB; under stochastic preferences, exploration matters and the bandit's UCB term provides a real edge.

### 4.4 Performance Metrics

- **Cumulative regret**: $\sum_t (1 - r_t)$. Aggregate online performance.
- **Rolling accuracy**: 50-round window. Convergence dynamics.
- **OOD accuracy (late rounds)**: accuracy on OOD queries after round $0.6 \cdot T$ — measures whether the agent can break out of its learned habit when the query demands it.
- **Test accuracy** (held-out): each user has a separate test pool. Agents evaluate without updating. Eliminates train/test leakage concerns.
- **Preference recovery**:
  - One-hot: recovery rate = % of (user, domain) pairs where the agent's most-frequently-selected tool matches the user's preferred tool.
  - Soft: cosine similarity, KL divergence, and Spearman rank correlation between the agent's learned distribution and the ground-truth Dirichlet sample. Spearman is the most discriminative metric.

---

## 5. Method

### 5.1 The Shared Pipeline

To make all agents comparable, we share a single LLM-based domain classifier across all agents. For each query, **one** LLM call labels the domain $\hat{d}(q_t)$; this label is then passed to every agent. This eliminates the unfair advantage `Pure-Bandit` would otherwise receive from ground-truth domain access, and reduces total LLM cost. Domain-classification accuracy is reported as a separate diagnostic metric (~94.5% on GPT-5.2 with the 60-tool benchmark).

### 5.2 The Nine Agents

We categorize all agents into four groups; each group corresponds to a specific design philosophy:

| Category | Agents | Decision primitive |
|---|---|---|
| No learning | `Random` | uniform |
| Statistical only | `Freq-Greedy`, `Pure-Bandit` | success rate / LinUCB |
| LLM only | `ZeroShot-LLM`, `InContext-Memory`, `Profile-Memory` | LLM with varying memory |
| Statistical + LLM hybrid | `Bandit+CoT`, `Freq+Override`, **`Bandit+Override`** (proposed) | mixed |

Brief description of each:

- **`Random`** — sanity floor.
- **`Freq-Greedy`** — greedy success-rate, no exploration, no LLM. Tests whether simple counting is enough.
- **`Pure-Bandit`** — LinUCB only. Per-(user, domain, tool) arms. 96-dim feature-hashed context (64 query + 16 tool + 16 domain).
- **`ZeroShot-LLM`** — LLM with full tool list, no personalization. Random-tier on standard queries; near-100% on OOD queries (it can read the tool name from the query).
- **`InContext-Memory`** — LLM with the last 5 successful selections per (user, domain) in context. Recency-weighted personalization through prompting.
- **`Profile-Memory`** — LLM with a structured per-(user, domain, tool) success-rate profile in context. Simulates Claude Memory / ChatGPT Memory style production agents.
- **`Bandit+CoT`** — LLM is given the bandit's UCB-softmax prior as % numbers in the prompt and asked to reason about the query in light of this prior. This is the classic "give the LLM a calculator" hybrid.
- **`Freq+Override`** (ablation) — Frequency-greedy selects by default; LLM is invoked only to ask *"does this query explicitly name a tool?"* and overrides if yes.
- **`Bandit+Override`** (**proposed**) — LinUCB selects by default; LLM is invoked only for OOD-override detection. Identical to Freq+Override except for the default-selection mechanism.

### 5.3 Why `Bandit+Override` Should Work

The architectural argument:

1. **On standard queries**, the LLM in `Bandit+Override` is asked only a narrow yes/no question. If it correctly returns "no override", the bandit's choice is used unchanged. Thus the agent inherits **Pure-Bandit's accuracy ceiling** on standard queries.
2. **On OOD queries**, the LLM is asked the same yes/no question and is able to read the explicitly-named tool from the query text. Empirically this is ~93% accurate even with smaller backbones. The override path bypasses the bandit's habit and matches **ZeroShot-LLM's OOD ceiling**.
3. **The bandit's learning signal is preserved**: because the bandit's update uses the *executed* action and observed reward, the LLM's occasional override does not corrupt the bandit's posterior in a damaging way (in fact, it provides a small amount of off-policy training data on the non-default tool).

By contrast, `Bandit+CoT` exposes the bandit prior as part of the LLM's *selection* reasoning, not just as override information. The LLM then introduces selection noise on every query — even standard ones where the bandit alone would have been correct. This is the central failure mode we identify and measure.

### 5.4 Implementation Notes

- LinUCB hyperparameters: $\alpha = 1.0$ (UCB exploration coefficient), 96-dim feature-hashed context, softmax temperature 3.0 when exposing prior to the LLM.
- LLM calls at `temperature = 0.0` for determinism.
- Prompt engineering for `Bandit+CoT`: in early iterations, including tool *metadata* in the second-stage prompt caused the LLM to ignore the bandit prior. Final design: tool names + prior % only.
- Within each round, the up-to-6 LLM-using agents now call the API concurrently (engineering improvement; results are bit-identical to sequential execution at `temperature = 0.0`).

---

## 6. Experimental Setup

### 6.1 Benchmark

- **ToolBench-60**: 60 real APIs from ToolBench across 10 domains (Finance, Sports, Travel, Entertainment, Gaming, Education, Communication, Location, eCommerce, Social), 6 tools per domain.
- Standard and OOD query templates are LLM-generated per tool. OOD templates explicitly name a non-preferred tool.

### 6.2 Default Configuration

- 50 users per seed.
- 500 training rounds per user (so each (user, domain) gets ~50 rounds on average — sufficient for bandit convergence).
- 50 held-out test queries per user.
- 3 random seeds for paired t-tests.
- 10% OOD ratio.
- Shared LLM domain classification.

### 6.3 LLM Backbones

We test the architecture across multiple LLM backbones, spanning capability tiers:

| Backbone | Size class | Status |
|---|---|---|
| `gpt-5.2` (via packy proxy) | frontier-tier, reasoning | partial — main paper experiment in progress |
| `qwen3-30b-a3b-instruct-2507` | mid-tier instruct | complete (3 seeds × one-hot + soft) |
| `deepseek-v4-flash` | mid-tier instruct | partial — seeds 0/1 complete, seed 2 in progress |
| `qwen3-235b-a22b-thinking-2507` | frontier-tier reasoning | planned |

This is **essential** because the methodological finding — that LLM-as-CoT-decision-maker hurts compared to LLM-as-exception-handler — must hold across backbones to be a claim about architecture, not about model strength.

### 6.4 Hardware / API

API access through packyapi.com proxy and direct OpenAI Responses-API endpoints. All experiments run on a single machine (LLM is the bottleneck, not compute). Total wall-clock budget: estimated ~200 hours of API calls for all main + ablation runs.

---

## 7. Main Results

> Note: numbers below are from completed runs. GPT-5.2 and DeepSeek main results to be filled in as they finalize. Qwen3-30b is complete and reported in full.

### 7.1 One-Hot Preferences (Qwen3-30b-a3b-instruct-2507, 50 users × 500 rounds × 3 seeds)

| Agent | Final Cumulative Regret ↓ | OOD Acc (late) ↑ | Pref Recovery ↑ |
|---|---|---|---|
| Random | 492.1 ± 0.2 | 1.7% | 15.9% |
| ZeroShot-LLM | 377.2 ± 3.0 | 97.9% | 15.9% |
| InContext-Memory | 363.9 ± 3.8 | 97.9% | 62.5% |
| Profile-Memory | 269.5 ± 4.5 | 89.4% | 70.9% |
| Bandit+CoT | 344.2 ± 1.5 | 93.1% | 82.7% |
| Freq-Greedy | 167.6 ± 6.0 | 3.9% | 86.4% |
| Pure-Bandit | 140.2 ± 1.9 | 30.7% | 99.9% |
| **Freq+Override** | **126.3 ± 4.5** | 92.4% | 92.5% |
| **Bandit+Override** | 135.7 ± 0.7 | 92.7% | **100.0%** |

**Reading**:
- The four override / statistical-primary methods (`Freq-Greedy`, `Pure-Bandit`, `Freq+Override`, `Bandit+Override`) form the clear top tier.
- `Profile-Memory`, the LLM-with-memory representative of current production agents, is far behind (regret 269.5 vs 135.7 for `Bandit+Override`) — **the central comparison**.
- `Bandit+CoT` performs poorly (344.2) — worse than `Profile-Memory`. On a mid-tier model the LLM cannot reliably integrate the bandit prior into its reasoning, and introduces selection noise.
- On one-hot preferences, `Freq+Override` and `Bandit+Override` are very close (126.3 vs 135.7), reflecting that under deterministic preferences naive counting and LinUCB converge similarly.

### 7.2 Soft Preferences (Qwen3-30b, concentration = 0.3)

| Agent | Final Cumulative Regret ↓ | OOD Acc ↑ | KL ↓ | Spearman ↑ |
|---|---|---|---|---|
| Random | 491.2 ± 0.8 | 1.7% | 0.785 | 0.000 |
| ZeroShot-LLM | 377.5 ± 3.6 | 98.3% | 0.785 | 0.000 |
| InContext-Memory | 372.7 ± 3.7 | 98.3% | 11.731 | 0.282 |
| Bandit+CoT | 369.6 ± 2.9 | 93.1% | 0.595 | 0.373 |
| Profile-Memory | 344.2 ± 4.3 | 90.9% | 11.336 | 0.271 |
| Freq-Greedy | 328.5 ± 1.2 | 11.6% | 11.962 | 0.399 |
| Freq+Override | 295.3 ± 2.9 | 92.6% | 10.855 | 0.288 |
| Pure-Bandit | 282.0 ± 2.2 | 37.0% | 0.500 | 0.539 |
| **Bandit+Override** | **264.8 ± 2.4** | **93.3%** | **0.497** | **0.539** |

**Reading**:
- `Bandit+Override` strictly dominates every other method on regret. It also achieves the best OOD accuracy among statistical-primary methods.
- The gap to `Freq+Override` widens to ~30 regret (vs ~10 in one-hot), confirming **the exploration vs. counting hypothesis**: bandit exploration matters under stochastic preferences.
- KL and Spearman: `Pure-Bandit` and `Bandit+Override` are *identical* (Spearman 0.539, KL 0.497 vs 0.500) — confirming that the LLM override path does not corrupt the bandit's learned distribution.
- `Freq+Override` has KL ~22× worse than `Bandit+Override` because frequency counting cannot recover a *distribution*, only a mode.

### 7.3 Cross-Model Pattern (placeholder for GPT-5.2 / DeepSeek)

Once GPT-5.2 and DeepSeek runs complete, the cross-model comparison goes here. The central claim to validate empirically is:

> *Across all backbones,* `Bandit+Override` $\le$ `Bandit+CoT` in cumulative regret and `Bandit+Override` $\le$ `Profile-Memory` in cumulative regret.

The strongest version of the architectural claim is that the ordering of these methods is **model-independent**.

The single-seed GPT-5.2 results from earlier runs (in `gpt5.2-toolbench60-shared-t3-override-greedy/`) are consistent with this; once 3-seed results are in we report mean ± std and run paired t-tests.

### 7.4 Test-Set Accuracy

Cross-checks the online metrics with held-out generalization. Pattern from Qwen3 one-hot:
- Bandit+Override: overall 84.3%, OOD 92.4%
- Freq+Override: overall 82.5%, OOD 91.7%
- Pure-Bandit: overall 80.4%, OOD 3.3%
- Bandit+CoT: overall 34.1%, OOD 93.5%
- Profile-Memory: overall 53.4%, OOD 86.0%

The wide gap between overall and OOD accuracy for LLM-only methods is **a structural property** of the benchmark, not a flaw: OOD queries leak the answer in the text, while standard queries require learned preference. This makes the two accuracy bars on the same plot a clean visualization of the personalization-vs-semantic decomposition.

---

## 8. Ablation Studies

### 8.1 Concentration Sweep (planned, GPT-5.2)

Vary Dirichlet $\alpha \in \{0.1, 0.3, 1.0, 2.0, 5.0\}$, fix everything else. Plot regret vs $\alpha$ for `Bandit+Override` and `Freq+Override`.

**Expected**: the gap is small at $\alpha = 0.1$ (preferences near one-hot, frequency counting suffices), and widens monotonically as $\alpha \to 5$. This isolates the **exploration value** of LinUCB.

### 8.2 Rounds Sweep (planned, GPT-5.2)

Vary $T \in \{50, 100, 200, 500\}$, fixed soft preferences with $\alpha = 0.3$.

**Expected**: bandit-based methods benefit relatively more from longer horizons; LLM-only methods plateau early. Demonstrates sample-efficiency / cold-start dynamics.

### 8.3 OOD Ratio Sweep (planned, GPT-5.2)

Vary OOD ratio $\in \{0.05, 0.20, 0.30\}$.

**Expected**: as OOD ratio grows, Pure-Bandit (no override path) degrades sharply; Freq+Override and Bandit+Override stay stable; gap between them shrinks (because OOD dominates the metric).

### 8.4 Backbone-Size Ablation (already partial)

Comparing Qwen3-30b vs GPT-5.2 (once complete) reveals that:
- `Bandit+CoT` is **highly sensitive** to backbone strength — fails on Qwen3-30b (regret 344), works much better on GPT-5.2.
- `Bandit+Override` is **insensitive** to backbone strength — works on both. This is the *practitioner's takeaway*: if you're deploying to a setting where you cannot guarantee a frontier-tier reasoner, `Bandit+Override` is the safe architecture.

---

## 9. Analysis & Discussion

### 9.1 Why Does `Bandit+CoT` Fail on Mid-Tier Backbones?

Diagnostic: even with the bandit's UCB-softmax prior provided as percentages in the prompt, the LLM does not consistently respect the prior. On standard queries, it sometimes picks a low-prior tool because the query text *plausibly* matches another tool's description.

This is the same failure mode as "the LLM cannot do basic arithmetic with a calculator unless you force it to use the calculator". The bandit prior is in the prompt, but the LLM does not commit to it.

`Bandit+Override` solves this by **structurally** preventing the LLM from selecting a low-prior tool: the LLM is asked only whether to override, not what to choose. The override slot is constrained by JSON schema and validated against the domain tool list.

### 9.2 Why Doesn't the LLM-Memory Approach Scale?

`Profile-Memory` and `InContext-Memory` both plateau far below the bandit ceiling, even after 500 rounds. Diagnostic:
- The LLM treats memory as *evidence to reason over*, not as *posterior to commit to*.
- With enough memory entries, the LLM's prior on "most common selection" is correct, but it still occasionally overrides itself for spurious semantic reasons.

This suggests that for personalization-heavy decisions, **the right division of labor is to let the LLM update the bandit, not act as one**.

### 9.3 The Override Channel Is a Narrow Interface

A key design insight: the LLM is given a *narrow* task. It does not need to enumerate all tools, weigh them against priors, or reason about user history — it answers a single question: "does the query name a specific tool?". This is:
- Cheap (short prompt, short answer).
- Robust (a question almost any LLM can answer with high accuracy).
- Decoupled (the bandit's learning signal is unaffected by override decisions).

This pattern — *the LLM as an exception handler* — generalizes beyond tool selection.

### 9.4 Connection to Production AI Agents

Current AI agents (Claude with Memory, ChatGPT Memory) operate close to our `Profile-Memory` baseline: they store user info in a memory layer and feed it to the LLM. Our results suggest that adding a thin contextual-bandit layer for repeated, feedback-rich decisions would substantially improve their personalization quality without altering the LLM's behavior on novel queries.

---

## 10. Limitations

- **Simulated users**: preferences are sampled, not from human data. Real users may have more complex preference structures (multi-modal, context-dependent, time-varying). Soft preferences with low concentration is a partial mitigation but not a substitute.
- **Single-domain queries**: each query maps to one domain. Multi-domain queries (e.g. "book a flight and a hotel") would need a different decision granularity.
- **LLM calls dominate cost**: even with parallelization, a 500-round × 50-user × 3-seed run is dozens of CPU-hours of LLM API. This limits how exhaustively we can sweep hyperparameters.
- **Domain classifier as a shared component**: ~94.5% accuracy on the 60-tool benchmark. Errors propagate to every agent. We report domain classification accuracy as a diagnostic, but a more thorough study would jointly tune classifier and agent.
- **Override mechanism is binary**: a richer override channel could pass back a probability or confidence to the bandit (Bayesian fusion). We do not explore this.

---

## 11. Conclusion

We argue that the right architecture for personalized tool selection inverts the current LLM-with-memory paradigm: a contextual bandit makes the default selection, and the LLM is invoked only to handle explicit semantic overrides. Across benchmarks and multiple LLM backbones, this `Bandit+Override` design dominates both pure-statistical and pure-LLM baselines and is **robust to backbone strength**, unlike alternatives that bake the bandit prior into LLM reasoning. The recipe is simple, deployable on top of existing memory systems, and applies to any decision in an agent's workflow that is both personalization-heavy and feedback-rich.

---

## 12. Storytelling Notes (for paper writing)

A few framing choices to think about during writing:

- **Lead with the architectural argument, not the benchmark**. The reader should be told *up front* that the contribution is a design pattern (statistical primary + LLM exception), and the benchmark is the evidence. Otherwise the work risks reading as "yet another bandit-augmented LLM agent".
- **Use `Profile-Memory` as the foil**. Tying our baseline to actual production AI agents (Claude Memory, ChatGPT Memory) makes the contribution legible to the reader in 2 sentences. The reader will viscerally recognize the problem.
- **The OOD vs Standard split is the structural insight**. The two-axis test-accuracy plot (overall accuracy bar + OOD accuracy bar per agent) is probably the cleanest single figure for the paper. It visually shows the *decomposition* claim.
- **The cross-model ablation is methodologically central**. We are claiming an architectural pattern, so we must show it is not model-specific. Once 3 backbones (frontier + mid + smaller) are reported, this claim is well-supported.
- **The soft-preference setting is the hero result**. One-hot is necessary for completeness but does not distinguish frequency from bandit. The soft setting is where the proposed method's full advantage emerges.
- **Don't oversell**. The contribution is architectural, not algorithmic — we use vanilla LinUCB. The pitch should be: *we identify the right interface between a learned policy and an LLM for personalized decisions*, not *we invent a new bandit algorithm*.

---

## 13. Suggested Section Word Targets (12-page top-tier paper)

| Section | Target |
|---|---|
| Abstract | 200 words |
| Introduction | 1 page (motivation + thesis + contributions list) |
| Related Work | 0.75 page |
| Problem Formulation | 0.75 page |
| Method | 1.5 pages (incl. architecture figure) |
| Experimental Setup | 0.75 page |
| Main Results | 2 pages (Table 1, Figure 1 regret, Figure 2 OOD, cross-model summary) |
| Ablations | 1.5 pages (concentration, rounds, OOD ratio, backbone-size) |
| Analysis & Discussion | 1.25 pages |
| Limitations + Conclusion | 0.5 page |
| References + Appendix | rest |

---

## 14. Open Items Before Writing Starts

1. Complete GPT-5.2 main runs (in progress) and DeepSeek runs.
2. Run concentration sweep on GPT-5.2 (Phase 2.1 in `EXPERIMENTS.md`).
3. Decide whether to include a frontier reasoner (e.g. `qwen3-235b-a22b-thinking-2507`) as a fourth backbone — it would strengthen the cross-model claim.
4. Decide whether to swap the central comparison from `Bandit+Override` vs `Profile-Memory` to `Bandit+Override` vs both `Profile-Memory` and `Bandit+CoT` — currently the strongest competing baseline depends on the backbone.
5. Citations for `ProfileMemory`-style production agents (Claude Memory documentation, OpenAI memory blog posts, etc.).
