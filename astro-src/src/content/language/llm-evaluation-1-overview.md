---
title: "LLM Evaluation (1/12) — Overview and the Benchmark Treadmill"
date: "2026-04-18"
excerpt: "Goodhart's Law, five phases of benchmark saturation, and why no single evaluation method has proven durably reliable."
project: language
readingTime: 6
---

The field of LLM evaluation since 2020 has been driven by a single recurring dynamic: **Goodhart's Law** — every benchmark that becomes a target ceases to be a good measure. Compounding this are data contamination, benchmark gaming, and the structural incentive that evaluation scores serve as marketing in a multibillion-dollar industry.

As of April 2026, no single evaluation method has proven durably reliable. A few hard-won lessons have stuck:

- Static public benchmarks have a shelf life measured in months.
- Human preference captures something benchmarks miss, but is itself gameable.
- The gap between benchmark performance and real-world utility remains the field's deepest unsolved problem.

---

## The benchmark treadmill: five phases

The evaluation landscape follows a recurring pattern documented by Kiela et al. (2021): benchmark introduced → models improve → saturation reached → contamination and gaming surface → harder benchmark created. What changed in the LLM era is the *speed* of this cycle. MNIST took 20+ years to saturate; SuperGLUE saturated in roughly 5 months.

### Phase 1 — NLU benchmarks (2018–2020)

GLUE and SuperGLUE tested natural language understanding through tasks like textual entailment and word sense disambiguation. They assumed task-specific fine-tuning — a paradigm GPT-3 would render obsolete. SuperGLUE saturated in about 5 months when T5 reached 89.3% against a human baseline of 89.8%.

### Phase 2 — Broad knowledge and early skills (2020–2022)

GPT-3's June 2020 release, demonstrating few-shot learning without gradient updates, motivated benchmarks testing knowledge and reasoning at scale:

- **MMLU** (Hendrycks et al., Sep 2020): 15,908 multiple-choice questions across 57 subjects. GPT-3 scored 43.9%.
- **GSM8K** (OpenAI, Oct 2021): grade-school math word problems.
- **HumanEval** (OpenAI, Jul 2021): code generation with pass@k.
- **MATH** (Hendrycks et al., 2021): competition-level math; GPT-3 scored ~5%.
- **BIG-Bench** (2022): 204 tasks from 450 contributors across 132 institutions.

This era established MMLU as the field's "north star" benchmark.

### Phase 3 — The great saturation (2023)

GPT-4 (March 2023) simultaneously saturated MMLU (86.4%), GSM8K (92%), HellaSwag (95.3%), and ARC (96.3%), triggering a diversification crisis. The response was threefold:

- **Chatbot Arena** (LMSYS, May 2023) — crowdsourced human preferences.
- **SWE-bench** (Princeton, Oct 2023) — real-world software engineering.
- **GPQA** (NYU, Nov 2023) — PhD-level "Google-proof" science questions.

### Phase 4 — Harder static + dynamic evaluation (2024)

- **MMLU-Pro** (Jun 2024): 10 answer options, reasoning-heavy; scores dropped 16–33% vs MMLU.
- **LiveBench** (Jun 2024): monthly-updated questions, automated objective scoring.
- **SimpleQA** (OpenAI, Nov 2024): adversarial factuality; GPT-4o scored below 40%.
- **FrontierMath** (Epoch AI, late 2024): unpublished research-level math, all models initially ~2%.

### Phase 5 — Frontier-only benchmarks (2025–2026)

- **Humanity's Last Exam** (CAIS, Jan 2025, published in Nature): 2,500 expert questions, top scores ~37% as of early 2026.
- **MathArena**: fresh competition problems evaluated as released.

The field shifted from "can AI match average humans?" to "can AI match world experts?" to "can AI do genuine research?"

---

## Benchmark lifespans

| Benchmark | Introduced | Saturated | Lifespan | Killed by |
|-----------|-----------|-----------|----------|-----------|
| SuperGLUE | 2019 | 2019 | ~5 months | T5 |
| MMLU | Sep 2020 | ~2023 | ~2.5 years | GPT-4 |
| GSM8K | Oct 2021 | ~2023 | ~2 years | GPT-4 |
| HumanEval | Jul 2021 | ~2024 | ~3 years | GPT-4o |
| MATH | 2021 | Late 2024 | ~3.5 years | o1 |
| BIG-Bench | 2022 | 2022 | ~10 months | PaLM 540B |
| IFEval | Nov 2023 | 2024 | ~4 months | LLaMA 3.3 70B |

---

## What remains discriminative (April 2026)

- **Chatbot Arena** — unsaturable by design (always fresh prompts, always new models).
- **SWE-bench Verified** — still differentiating.
- **SimpleQA** — around 55% F1 for frontier models.
- **FrontierMath Tier 4** — essentially unsolved.
- **Humanity's Last Exam** — about 37% top scores.
- **Dynamic benchmarks** (LiveBench, MathArena) — refreshed regularly.

The most durable benchmarks are either **dynamic** (refreshed regularly), **private** (problems not publicly available), or target **genuinely unsolved problems**.

---

## Pre-LLM benchmarks as historical reference

Two benchmarks illustrate how saturation timelines compressed:

- **MNIST** (1998): 70,000 handwritten digit images (28×28 grayscale). Took 20+ years to fully saturate.
- **SQuAD 2.0** (2018): reading comprehension over Wikipedia paragraphs. Saturated in about 2 years by BERT-era models.

Decades → years → months. That acceleration is the real story of the treadmill.
