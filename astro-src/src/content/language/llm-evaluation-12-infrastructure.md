---
title: "LLM Evaluation (12/12) — Infrastructure, Leaderboards, and the Industry-Academia Divide"
date: "2026-04-07"
excerpt: "EleutherAI, HELM, OpenAI Evals, the Hugging Face Open LLM Leaderboard's rise and retirement, and where evaluation goes from here."
project: language
readingTime: 6
---

The closing post in this series. If the first eleven described *what* we measure and *why* it keeps breaking, this one is about *who* builds the measuring stick.

---

## Key frameworks

### EleutherAI lm-evaluation-harness

Born from frustration with non-reproducible evaluation circa 2020. Became the backend for the Hugging Face Open LLM Leaderboard. Cited in over 5,000 research papers. The de facto standard for evaluating open-weight models.

### Stanford HELM (November 2022)

Holistic multi-metric evaluation across 7 dimensions (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency). A framework wrapping multiple benchmarks rather than a benchmark itself. Covered in depth in the previous post.

### OpenAI Evals API (April 2025)

A product for **application developers** (not frontier labs) who build products on top of OpenAI's API. Enables systematic testing rather than "vibe-based evals" (OpenAI's term for informal subjective assessment).

**Step 1 — define test cases** (inputs with expected outputs):

```json
{"ticket_text": "My laptop screen is cracked", "expected": "Hardware"}
{"ticket_text": "Excel keeps crashing", "expected": "Software"}
```

**Step 2 — define grading criteria.** Programmatic graders (exact string match, regex, JSON validation) are deterministic and free. Model-based graders (GPT-4 judges quality subjectively) cost API tokens.

**Step 3 — run evals.** Point at test data, prompt template, and model. Get aggregate metrics.

What developers use it for: prompt iteration, model swapping (compare GPT-4 vs GPT-4o-mini on a specific use case), CI/CD integration, safety testing.

**Q: Can't I just build this myself with the GPT API?**

Yes. The Evals API is convenience tooling that wraps a common pattern. What it adds is a dashboard for visualizing results across runs, a pre-built grader library, batch execution with rate limiting, and integration with OpenAI's fine-tuning and prompt optimizer. Many teams build custom pipelines instead. The API keeps developers inside OpenAI's ecosystem.

**Q: What are "vibe-based evals"?**

Industry shorthand for informal, subjective evaluation — chatting with a model and forming impressions like "this feels smarter." OpenAI warns against this for application developers: gut feelings don't scale, aren't reproducible, and can be misleading (a model might feel better because it's verbose, not accurate).

---

## The Hugging Face Open LLM Leaderboard

### V1 (~early 2023 to June 2024)

Benchmarks: ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K. 2M+ unique visitors, 13,000+ models evaluated, 300K monthly active users. Backed by EleutherAI lm-evaluation-harness. Open-weight models only.

What forced the transition: benchmark saturation (models hitting ceilings on HellaSwag, ARC, MMLU), contamination, deliberate gaming, models fine-tuned specifically on v1 benchmark data.

### V2 (June 2024)

New benchmarks: MMLU-Pro, GPQA, MATH Level 5, MuSR, IFEval, BBH. Normalized scoring (random baseline = 0, perfect = 100). Shifted to generative evaluation for some tasks (better for instruction-tuned models).

What happened: models jumped or dropped up to 50 positions. Models optimized for v1 collapsed on v2. Stable models (Qwen-2-72B, LLaMA-3-70B) proved genuine capability. Some chat-tuned models performed worse on math than their base versions — RLHF for helpfulness can impair specific capabilities.

### Retirement (2025–2026)

IFEval saturated within about 4 months. Same gaming incentives applied to new targets. Contamination detection remained unsolved. The team explained: "As model capabilities change, benchmarks need to follow — the leaderboard is slowly becoming obsolete."

---

## The industry–academia divide

### Industry labs

**OpenAI.** Emphasizes eval-driven development through the Evals API. Created SimpleQA. Warns against vibe-based evals.

**Anthropic.** Most safety-forward approach. Organized around AI Safety Levels (ASL-1 through ASL-4) with detailed system cards. Considered the gold standard for evaluation transparency. Conducted a joint safety evaluation with OpenAI in 2025.

**Google DeepMind.** Benchmark leadership. Contributed BIG-Bench and MMLU-Pro. Drew scrutiny when the Gemini Ultra launch claimed 90% on MMLU using chain-of-thought with 32 attempts rather than standard 5-shot evaluation (where it scored 83.7%).

**Meta.** Emphasizes reproducibility through open-source. Publishes full eval datasets with per-example details. The Llama 4 Arena controversy undermined trust.

### Academic groups

**EleutherAI.** Built lm-evaluation-harness, the field's most-used evaluation framework.

**Stanford CRFM.** Created HELM, introducing holistic multi-metric evaluation.

**LMSYS (UC Berkeley).** Created Chatbot Arena. Grew from 4,700 votes at launch to 6M+. Provided critical independent evaluation.

**Academic critics.** The "Leaderboard Illusion" paper (Cohere/AI2/Princeton/Stanford, 2025) documented Arena distortions. CMU's independent Gemini evaluation found materially lower scores than Google's claims.

### The structural tension

Benchmark scores now function as marketing. Each major release leads with numbers: Gemini 3 Pro's 1,501 Elo, Llama 4 at 1,417 Elo. Billions of dollars of investment are evaluated on these scores.

The conflict: labs investing in genuine evaluation may lose marketing races to labs optimizing for benchmark optics. Domain-specific models quietly outperform general-purpose leaders in energy, finance, and healthcare, but rank lower on leaderboards because they don't optimize for gaming.

Sebastian Raschka's assessment: benchmark numbers are "no longer trustworthy indicators of LLM performance."

---

## Where evaluation goes from here

The field is converging on the principle that robust evaluation requires a **portfolio approach**:

- Dynamic benchmarks (LiveBench, MathArena).
- Human preferences (Chatbot Arena, despite its flaws).
- Domain-specific assessment.
- Adversarial and safety testing.
- Process-based reasoning evaluation.
- And the one metric hardest to game: whether people actually choose to use these systems for real work.

That last point feels like the quiet conclusion of this whole series. Every benchmark eventually saturates, every judge can be fooled, every leaderboard can be gamed. What's left is use. A tool people keep reaching for, on real problems, when no one is watching — that's the measurement nobody has figured out how to fake.
