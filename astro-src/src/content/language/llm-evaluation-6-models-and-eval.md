---
title: "LLM Evaluation (6/12) — How Model Releases Broke Evaluation"
date: "2026-04-13"
excerpt: "GPT-3, Chinchilla, ChatGPT, GPT-4, LLaMA, o1, DeepSeek-R1 — each release exposed a specific evaluation failure and catalyzed new methods."
project: language
readingTime: 6
---

Each major model release exposed specific evaluation failures and catalyzed new approaches. The result is a feedback loop: capability advances break the old measurements, which forces new ones.

---

## GPT-3 (June 2020)

**What it broke.** The entire fine-tuning evaluation paradigm. Benchmarks like SuperGLUE assumed task-specific training. GPT-3 performed tasks with few-shot prompting — no gradient updates, no task-specific heads.

**What it motivated.** MMLU's design as a few-shot knowledge benchmark. BIG-Bench's diverse capability assessment. The paradigm shifted from task-specific fine-tuning + accuracy metrics to few-shot prompting + broad capability measurement.

---

## Chinchilla (March 2022)

**What it proved.** Evaluation results could inform fundamental training decisions. Its 70B model trained on 4× more data outperformed the 280B Gopher by over 7% on MMLU, establishing the ~20 tokens-per-parameter compute-optimal ratio.

**Impact.** Benchmark scores became the evidence driving resource allocation. This directly shaped LLaMA's training recipe and produced a 142-fold reduction in model size needed to achieve 60% on MMLU between 2022 and 2024 (per Stanford 2025 AI Index).

---

## ChatGPT / InstructGPT (November 2022 / March 2022)

**What it broke.** The assumption that benchmark performance equals real-world utility. The 1.3B InstructGPT was preferred by humans over the 175B GPT-3, despite scoring lower on every traditional NLP benchmark.

The InstructGPT paper put it bluntly: "Public NLP datasets are not reflective of how our language models are used."

**What it motivated.** Chatbot Arena, MT-Bench, AlpacaEval, and the entire preference-based evaluation paradigm. The gold standard shifted from static benchmarks with ground-truth answers to human preference and pairwise comparison.

The paper — "Training language models to follow instructions with human feedback" (Ouyang et al., 2022) — introduced the three-step RLHF process (SFT → reward model → PPO) that became the foundation for ChatGPT.

---

## GPT-4 (March 2023)

**What it broke.** Mass benchmark saturation. 86.4% on MMLU, top-10% on the bar exam, near-ceiling on HellaSwag, ARC, GSM8K. Most existing benchmarks became non-discriminative overnight.

**What it motivated.** An explosion of harder benchmarks — GPQA, SWE-bench, MMLU-Pro, FrontierMath, Humanity's Last Exam. Also professional exam evaluation and scaling-law-based performance prediction.

---

## LLaMA (February 2023)

**What it enabled.** Open weights meant truly reproducible evaluation on identical hardware. Everyone with the same weights on the same hardware using the same harness gets the same number, bit-for-bit.

It also built the Hugging Face Open LLM Leaderboard and made evaluation research itself possible — contamination studies, prompt sensitivity analyses, position bias experiments all require full control over the model.

**The downside.** Benchmark gaming. Open weights + fine-tuning = models optimized specifically for leaderboard benchmarks. The Phi series raised persistent questions: Phi-3-mini at 3.8B achieving 60%+ on MMLU — remarkable efficiency, or benchmark exposure?

**What does "open weights enable reproducible evaluation" really mean?** With closed API models you can't control: exact decoding parameters, whether the model silently changed between eval runs, whether logit-based evaluation is possible, or whether hidden system prompts modify outputs. If your score differs from the provider's reported number, you can't diagnose why. Open weights: download, run locally, identical results. That's what "truly reproducible" means.

---

## Reasoning models: o1 / o3 (2024–2025) and DeepSeek-R1 (January 2025)

**What they shattered.** Existing evaluation paradigms for math and code. o1 scored 83% on IMO qualifying vs. GPT-4o's 13%. o3 reached 87.5% on ARC-AGI. These were category changes, not incremental improvements.

**New evaluation challenges:**

1. **Test-time compute as a variable.** Same model produces different scores depending on thinking time allocated. Evaluation protocols must specify reasoning effort.
2. **Hidden reasoning.** o1's chain-of-thought is hidden from users, creating a transparency crisis.
3. **DeepSeek-R1's contribution.** Open weights with visible reasoning traces let the community study reasoning processes directly. By May 2025, R1-0528 nearly doubled thinking tokens per problem and reached 87.5% on AIME 2025.

These models made most existing math/code benchmarks instantly inadequate and created the urgency for process-based evaluation — the subject of a later post in this series.
