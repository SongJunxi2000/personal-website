---
title: "LLM Evaluation (5/12) — Contamination, Gaming, and the Evaluation Crisis"
date: "2026-04-14"
excerpt: "Three diagnostic studies — C-BOD, GSM1K, option-reordering — and the v1→v2 Open LLM Leaderboard collapse."
project: language
readingTime: 7
---

Contamination is when the model has memorized **specific benchmark question-answer pairs** from training data, so it's pattern-matching the test rather than demonstrating the skill the test measures.

---

## Contamination vs. legitimate learning

**Q: If a model memorized "the capital of France is Paris," isn't that just learning?**

Yes. Knowing facts is learning, not contamination. Contamination is specifically when the model memorized the **benchmark Q&A pair as a unit**: "when this specific question appears with these specific four options, pick B." The TS-Guessing result — GPT-4 guessing missing MMLU options with 57% exact match — suggests models recognized questions as memorized artifacts.

A concrete example of real contamination: GSM8K problem says "Sally has 7 apples, gives 3 to Tom, buys 5 more. How many?" Model memorized "answer: 9." Rephrase to 8 apples — if it still says 9, it memorized the pair rather than learning arithmetic.

---

## Scale of the problem

- GPT-3.5 and GPT-4 were exposed to roughly **4.7 million benchmark samples** during training.
- The GPT-4 technical report acknowledged "portions of BIG-bench were inadvertently mixed into the training set."
- OpenAI deliberately included part of the GSM8K training set in pretraining data.
- Because LLMs train on web-scraped corpora and most benchmarks are publicly available, overlap is near-inevitable.

---

## Three diagnostic studies compared

These three studies all test the same core question — *is the benchmark score real or inflated?* — but probe different failure modes.

### C-BOD (2025)

Systematically perturbs MMLU prompts — changes wording, reorders options, alters formatting — while keeping semantic content identical. Detects prompt-pattern dependence: the model learned the format, not the content.

Finding: 20/26 models showed statistically significant drops. Models with **higher baseline accuracy dropped more** — the top performers were most reliant on recognizing fixed patterns.

### GSM1K (Zhang et al., 2024)

Creates entirely new grade-school math problems of the same type and difficulty as GSM8K. Detects specific Q&A memorization.

Finding: Phi and Mistral models dropped about 10% on fresh problems.

### Option-reordering (Zheng et al.; Pezeshkpour & Hruschka)

Only changes the position of answer choices. Detects positional/selection bias.

Finding: up to 15.2% score swings; can completely invert model rankings.

### Comparison

| | What changes | What stays the same | What it detects |
|---|---|---|---|
| C-BOD | Wording, format, option order | Semantic content, difficulty | Prompt-pattern dependence |
| GSM1K | Entire question (fresh) | Type, difficulty | Specific Q&A memorization |
| Option-reordering | Only answer positions | Question text, answer content | Positional/selection bias |

C-BOD is the broadest. Option-reordering is the narrowest. GSM1K is the most definitive.

---

## Accidental contamination vs. deliberate optimization

Evidence for **deliberate optimization**:

- Qwen3Guard-8B: 91% on public benchmarks, 34% on private prompts of identical difficulty — a 57-point gap.
- HELM Gaming Study: small models fine-tuned on test sets achieved near-top scores on trained scenarios but collapsed on held-out ones.
- Open LLM Leaderboard v1→v2 transition: models dropping 50 positions when benchmarks changed.

Evidence for **accidental contamination**:

- GPT-4's admission that BIG-Bench was "inadvertently mixed into training."
- The sheer volume (~4.7M benchmark samples in training data).

The clean diagnostic signal: compare original benchmark score vs. novel question score (same type, same difficulty). If they match, the benchmark is trustworthy. If novel drops, the benchmark score is inflated.

---

## Inference-time decontamination

Rephrase or perturb benchmark questions at inference time. If performance drops significantly on rephrased versions, that's evidence of memorization. Use the rephrased score as the "decontaminated" score.

Results: 22.9% reduction in inflated accuracy on GSM8K, 19.0% on MMLU.

**Q: Why not just rephrase every benchmark question and use those as the real test?**

A few problems:

1. Rephrasing changes difficulty — you're no longer measuring the same thing.
2. Some questions can't be meaningfully rephrased — "What year did WWII end?" has limited options.
3. You'd need to re-validate each rephrased version.
4. New versions get scraped into future training data anyway.

This is exactly why the field moved to **dynamic benchmarks** (LiveBench, MathArena) rather than trying to fix static ones.

---

## Leaderboard gaming

### Open LLM Leaderboard v1→v2

V1 (early 2023, archived June 2024): ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K. Backed by EleutherAI's lm-evaluation-harness. Open-weight models only. 2M+ unique visitors, 13,000+ models evaluated.

Why the transition: benchmarks saturated. Models hit near-ceiling on HellaSwag, ARC, MMLU. Contamination and gaming became rampant.

V2 (June 2024): replaced everything with harder benchmarks — MMLU-Pro, GPQA, MATH Level 5, MuSR, IFEval, BBH. Added normalized scoring (random baseline = 0, perfect = 100).

What happened: models jumped or dropped **up to 50 positions**. Models optimized for v1 collapsed on v2. Stable models (Qwen-2-72B, LLaMA-3-70B) proved genuine. Phi-3's strong v1 scores despite small size were partly explained by curated training data overlapping with v1 benchmarks.

V2 problems: IFEval saturated within 4 months. Same gaming incentives applied to new targets. Contamination detection remained unsolved. The leaderboard was eventually retired in 2025–2026 with the team explaining that "as model capabilities change, benchmarks need to follow — the leaderboard is slowly becoming obsolete."

### The Llama 4 controversy (April 2025)

Meta tested 27 private variants on Chatbot Arena. The variant scoring 1,417 Elo (2nd place) was a specially tuned chat-optimized version never released publicly. The publicly shipped model performed at about 32nd–35th place.

**Q: Why didn't Meta just ship the high-scoring variant?**

Because it wasn't the best general-purpose model. The high-scoring variant was optimized for Arena-style interactions — verbose, stylistically appealing — at the cost of reliability, instruction-following, or consistency. This is Goodhart's Law at the product level: optimizing for the leaderboard metric produces a model that wins comparisons but isn't the best model to deploy.

LMSYS responded that Meta's approach "did not match what we expect from model providers" and tightened submission policies.
