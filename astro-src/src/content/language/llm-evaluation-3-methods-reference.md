---
title: "LLM Evaluation (3/12) — Methods Reference Table"
date: "2026-04-16"
excerpt: "Twelve evaluation paradigms, contamination detection methods, and diagnostic studies — each with strengths, weaknesses, and current status."
project: language
readingTime: 5
---

A companion reference to the benchmarks post. If you want to know *how* a benchmark is scored, the method matters as much as the questions.

---

## Evaluation paradigms

| Method | How it works | Strengths | Weaknesses | Status (2026) |
|---|---|---|---|---|
| **Logit-based** | Compare log-probabilities of each answer option via forward pass; no text generation | Fast, deterministic; works well for base models | Fails for instruction-tuned models that generate formatted responses | Still used for base models |
| **Generative** | Model generates text; output is parsed to extract answer | Captures what the model actually does | Answer extraction can fail; slower | Default for instruction-tuned models since HF Leaderboard v2 |
| **Multiple-choice** | Select from predefined options A/B/C/D | Easy to score; standardized | Position bias up to 15.2%; random guessing baseline | Widespread but increasingly questioned |
| **Open-ended generation** | Free-form text scored against ground truth or by judge | Tests realistic usage; no guessing advantage | Harder to score; needs judge or verifiable answer | Growing, especially math/code |
| **Human preference (pairwise)** | Users compare two outputs side-by-side | Captures real-world preferences; hard to contaminate | Verbosity/style bias; expensive; English/coding heavy | Gold standard via Chatbot Arena |
| **LLM-as-judge** | Strong model (GPT-4, Claude) evaluates outputs | Cheap (~$10/run); scalable | Position, verbosity, self-preference bias; exploitable | Widely used but known to be flawed |
| **Static benchmarks** | Fixed test set, same questions every time | Reproducible; comparable across time | Saturate; contaminate; get gamed | Being replaced by dynamic |
| **Dynamic benchmarks** | Questions refreshed periodically | Contamination-resistant; stay discriminative | Hard to compare across periods | LiveBench, MathArena, LiveCodeBench |
| **Private/held-out** | Test problems kept secret | Immune to training-data contamination | Can't be independently verified | FrontierMath (12 public samples only) |
| **Process reward models** | Evaluate each reasoning step, not just final answer | Catches errors at specific steps | Labeling expensive; step boundaries unclear | Active research frontier |
| **Automated red-teaming** | Adversarial models iteratively probe safety | Finds vulnerabilities fixed prompts miss | Hard to standardize; attacker quality varies | Growing rapidly (PAIR, Crescendo, GOAT) |
| **Agent-based** | Multi-step task completion in realistic environments | Tests real capability end-to-end | Scaffolding affects results; slow | SWE-bench, WebArena, OSWorld |

---

## Contamination detection

| Method | How it works | Effectiveness |
|---|---|---|
| **N-gram overlap** | Check for exact text overlap between benchmark and training data | Simple, easily evaded by paraphrasing |
| **Membership inference** | Test if model "recognizes" specific benchmark items | Moderate; high false-positive rate |
| **Perplexity-based** | Check for unusually low perplexity on benchmark items | Noisy but detects memorization |
| **TS-Guessing** | Ask model to guess missing answer options | GPT-4 achieved 57% exact match on MMLU |
| **Kernel Divergence Score** | Fine-tune and observe differential effects on seen vs unseen | Principled but computationally expensive |
| **Inference-time decontamination** | Rephrase benchmark questions at test time | Reduced inflated accuracy by 22.9% on GSM8K, 19.0% on MMLU |
| **Fresh parallel benchmarks (GSM1K)** | Create new questions of same type/difficulty | Most definitive; some models dropped ~10% |
| **C-BOD perturbation** | Systematically distort prompts while preserving semantics | 20/26 models showed significant drops |

---

## Diagnostic studies

| Study | What it tests | Key finding |
|---|---|---|
| **C-BOD** | Prompt-pattern dependence | 20/26 models drop on semantically identical rephrasings |
| **GSM1K** | Q&A memorization vs. math reasoning | Phi, Mistral drop ~10% on fresh problems |
| **Option-reordering** | Positional/selection bias in MC | Up to 15.2% swings; can invert rankings |
| **Null model attack** | LLM-as-judge exploitability | 86.5% win rate on AlpacaEval 2.0 from a constant non-response |
| **Intent laundering** | Safety reliance on trigger words | ASR jumps 5.38% → 86.79% when triggers removed |
| **CoT faithfulness** | Whether reasoning traces reflect actual computation | 25–39% faithfulness; unfaithful chains are longer |
| **ReDial dialect study** | Fairness across English dialects | Almost all models degrade significantly on AAVE |
