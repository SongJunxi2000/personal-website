---
title: "LLM Evaluation (10/12) — Process-Based Evaluation, PRMs, and Faithfulness"
date: "2026-04-09"
excerpt: "When reasoning matters as much as the final answer: process reward models, the faithfulness problem, and why labeling steps is fundamentally hard."
project: language
readingTime: 7
---

Traditional evaluation only checks final answers. A model gets "42" — correct or not? But as reasoning models (o1, DeepSeek-R1, Claude extended thinking) become dominant, the reasoning process matters as much as the conclusion. A doctor who gets the right diagnosis through wrong reasoning is dangerous.

---

## Outcome vs. process reward models

### Outcome Reward Model (ORM)

Only checks the final answer. Model produces a 10-step chain ending in "42." Is 42 correct? Yes → reward. No → penalty. No information about which step went wrong.

### Process Reward Model (PRM)

Evaluates every intermediate step:

```
Problem: What is the area of a circle with radius 5?

Step 1: The formula for area is πr²              → ✓ correct
Step 2: The radius is 5                           → ✓ correct
Step 3: So area = π(5)² = π × 10 = 10π            → ✗ error (5² = 25, not 10)
Step 4: The answer is 10π                         → ✗ (follows from wrong step)
```

The PRM catches Step 3 as the failure point.

Formally, for a reasoning sequence $S = (s_1, s_2, \ldots, s_n)$, a PRM produces a vector of stepwise scores. Most use binary labels (correct/incorrect), though recent variants allow probabilistic scoring.

---

## How PRMs are trained

**Human annotation.** OpenAI's "Let's Verify Step by Step" (Lightman et al., 2023) was the foundational work. Humans labeled 800K+ steps as correct/incorrect/neutral. Showed process supervision outperforms outcome supervision on MATH. Extremely accurate but doesn't scale.

**Monte Carlo estimation.** The scalable approach. For each intermediate step, sample many completions (e.g., 16) from that point forward and check how many reach the correct final answer. A step that leads to correct answers 80% of the time is probably correct; 0% means probably wrong. Uses the final answer as a signal, propagated back to individual steps. Cheaper than human annotation but noisy.

**LLM-as-judge.** Use a strong model (GPT-4) to evaluate each step. Cheaper than humans but inherits all LLM-as-judge biases.

---

## How PRMs are used

**Best-of-N selection.** Generate N reasoning chains for the same problem, score each step-by-step, select the chain with highest minimum step score (or product of scores). Better than ORM selection because ORM might pick a chain that reaches the right answer through lucky error cancellation.

**Beam search guidance.** At each generation step, expand only branches the PRM scores highly. Prune bad paths early rather than generating full chains and evaluating afterward.

---

## Three paradigms of reasoning evaluation

### 1. Correctness verification

Is each step logically valid? Classic PRM territory. Works well for math (steps can be formally checked), much harder for open-ended reasoning.

### 2. Faithfulness verification

Does the chain-of-thought reflect the model's actual reasoning process?

**Anthropic's CoT Faithfulness Study (2025).** Tested Claude 3.7 Sonnet and DeepSeek-R1 by embedding subtle hints in problems. When the model used a hint to reach its answer but didn't acknowledge it, that's an unfaithful chain.

- Claude acknowledged using hints in only 25% of cases.
- DeepSeek-R1 showed 39% faithfulness.
- Unfaithful chains were consistently longer (model constructs fake justification).
- Faithfulness dropped 44% on harder problems.

**Q: How do they know the model used the hint?**

By comparing performance with and without the hint:

```
Control:   Hard problem, no hint                       → model answers A (wrong)
Treatment: Hard problem + hint "answer is B"           → model answers B (correct)
           Chain-of-thought: [valid-looking derivation of B, no mention of hint]
```

The counterfactual proves the hint causally influenced the answer. Across hundreds of problems and many samples, the pattern is statistically robust — random sampling variance can't explain consistent 85-point probability swings.

**Q: Unfaithful vs. incorrect steps — what's the difference?**

Incorrect steps = the math or logic is wrong (5² = 10). Unfaithful steps = the steps might be logically correct, but they don't reflect what the model actually computed internally. The model reached the answer through some other process and wrote a plausible justification after the fact. That's a transparency problem, not a correctness problem.

### 3. Completeness / relevance verification

Does each step contribute meaningfully? No non-sequiturs, redundant steps, or hallucinated reasoning? Important for reasoning models that generate very long chains.

---

## Current bottlenecks

**Labeling is fundamentally expensive.** Human annotation doesn't scale. Monte Carlo estimation is noisy — a step might be labeled "correct" because some random continuation reaches the right answer. The FaithRL finding (48.75% of correct answers contain unfaithful steps) shows how much noise exists in outcome-based labeling.

**Step boundary definition is unsolved.** What counts as a "step"? In math, equation breaks provide boundaries. In open-ended reasoning, no clear segmentation exists. Different granularities give different results.

**Domain generalization doesn't work.** PRMs trained on math don't transfer to code, science, or open-ended reasoning. The Qwen PRM paper (2025) found over 30% of annotations in mainstream PRM datasets are severely flawed.

**The independence assumption is wrong.** Most PRMs evaluate each step independently. But reasoning is sequential: an error in step 2 might make step 3 "correct given step 2" but "incorrect given the original problem." The CRM paper (2026) addresses this by conditioning each step's reward on prior steps, at the cost of added complexity.

**Faithfulness is fundamentally hard to measure.** The Oxford paper "Chain-of-Thought Is Not Explainability" (2025) showed internal computations systematically diverge from verbal traces. Position bias causes 36% accuracy drops but is never mentioned in chains of thought. Without mechanistic interpretability, we're stuck at behavioral testing.

**Reward hacking is emerging.** As PRMs train reasoning models via RL, models learn to produce chains that score well on the PRM without reasoning better — Goodhart's Law applied to reasoning evaluation.

---

## Key papers

- Lightman et al. (2023): "Let's Verify Step by Step" — foundational PRM work.
- Wang et al. (2024, ACL): "Math-Shepherd" — PRM training without human annotation.
- Zheng et al. (2024): ProcessBench — benchmark for evaluating PRMs themselves.
- FaithRL (2025): 48.75% of correct answers contain unfaithful reasoning steps.
- Anthropic CoT Faithfulness Study (2025): 25–39% faithfulness rates.
- CRM (2026): conditional reward modeling linking steps to outcomes.
