---
title: "Post-Training (7/8) — Reasoning Training (o1 / DeepSeek-R1)"
date: "2026-04-20"
excerpt: "How RL with verifiable rewards enables language models to discover reasoning strategies from scratch — the R1-Zero experiment and the hybrid R1 pipeline."
project: language
readingTime: 10
---

The o1/R1 era showed that reasoning strategies — backtracking, self-verification, extended thinking — can emerge from RL with verifiable rewards, without human-written demonstrations. This post covers DeepSeek-R1-Zero's pure RL experiment, the hybrid R1 pipeline, and what this means for the role of SFT vs. RL.

---

## The Paradigm Shift

Before 2024, the assumption was: reasoning capability comes from *imitating* high-quality reasoning traces via SFT. You collect chain-of-thought examples from humans or stronger models, fine-tune on them, and the model learns to reason.

The o1 / R1 era showed a different path: **reasoning can emerge from RL with verifiable rewards, without human-written reasoning traces.** The model discovers effective reasoning strategies by optimizing for "get the answer right."

This reframes the role of SFT and RL in post-training:
- **SFT teaches format** — how to present reasoning readably
- **RL teaches capability** — what strategies actually produce correct answers

SFT is bounded by demonstrations. RL is bounded by the reward signal. If a strategy isn't in the demonstrations, SFT can't produce it. If a strategy gets higher reward than alternatives, RL finds it — even if no demonstration contained it.

## The Core Recipe

```
Base model
   ↓
RL with verifiable rewards (usually GRPO)
   ↓
Reasoning model
```

**Verifiable rewards** means the reward is computed by a deterministic checker, not a learned model:
- Math: does the final boxed answer match ground truth? → 1 or 0
- Code: do the unit tests pass? → 1 (or fractional based on pass rate)
- Proof: does the theorem prover accept the proof? → 1 or 0

The policy gets no partial credit for "reasoning looked good but answer was wrong." This sparsity is a feature, not a bug — the signal is honest. There's no reward model to hack.

## DeepSeek-R1-Zero: Pure RL from Base Model

R1-Zero is the minimalist version of the recipe. Starting from DeepSeek-V3-Base (a raw pretrained model with no instruction tuning), DeepSeek applied GRPO with verifiable math and code rewards. No SFT stage. No reasoning demonstrations. Just: "here's a problem, you get +1 if the answer is right."

### What Emerged

Three behaviors developed during training that weren't explicitly rewarded:

**1. Response length grew dramatically.**
Early in training, responses were a few hundred tokens. By late training, thousands of tokens. Nothing in the reward function said "generate longer responses." The model discovered that thinking longer produced more correct answers, and GRPO reinforced that. This is the "test-time compute" phenomenon that o1 is built around — rediscovered from scratch.

**2. Self-reflection and verification emerged spontaneously.**
The model started producing phrases like "Wait, let me reconsider...", "Actually, that's wrong. Let me try again.", "Let me verify this by...". Nobody wrote these patterns into training data. The model discovered that backtracking and self-correction increased success rates, so those patterns got amplified.

DeepSeek called this the **"aha moment"** — a specific training checkpoint where response length jumped discontinuously as the model learned to allocate more thinking time by re-evaluating its approach.

**3. Strategy adaptation across problem types.**
The model learned to use different approaches for different problems — algebraic manipulation, case analysis, contradiction — without being taught these categories. Right strategy = whichever strategy produced the right answer.

### Why This Was Surprising

The standard assumption: reasoning *patterns* (backtracking, verification, strategic thinking) are human inventions that need to be taught. R1-Zero showed these patterns **emerge from optimization pressure** alone, given a base model with language and math knowledge from pretraining plus a verifiable reward signal.

Analogy: this is the AlphaZero moment for language model reasoning. AlphaGo learned from human games first. AlphaZero started from scratch and discovered its own strategies — some better than humans', some alien to experts. R1-Zero is the same story for reasoning.

### The Catch

R1-Zero's reasoning traces were **effective but ugly**. They mixed languages mid-sentence (Chinese and English), skipped explanations, used strange notation, and were hard for humans to follow. The model was optimizing for "get the answer right," not "produce traces humans find useful." So while R1-Zero was the scientific result, it wasn't the product.

## DeepSeek-R1: The Hybrid Pipeline

R1 is what they actually shipped. It takes R1-Zero's insights and adds SFT stages for human-readable output:

```
Base model
   ↓
SFT on small set of high-quality reasoning traces  ← "cold start" for format
   ↓
GRPO with verifiable rewards                       ← reasoning capability
   ↓
SFT on curated data (reasoning + general tasks)    ← broader behavior
   ↓
GRPO with mixed rewards (verifiable + preference)  ← alignment
   ↓
R1
```

Each stage has a distinct job:

**Stage 1 (cold-start SFT):** Narrow goal — teach the model a *readable* format before RL takes over. Not teaching how to reason; teaching how to communicate reasoning.

**Stage 2 (GRPO with verifiable rewards):** Where actual reasoning capability gets built. This is the R1-Zero insight — RL discovers effective strategies.

**Stage 3 (SFT on curated data):** Broaden the model beyond pure reasoning. Include writing, general Q&A, tool use.

**Stage 4 (GRPO with mixed rewards):** Final alignment pass, mixing verifiable rewards for reasoning with preference-based rewards for general helpfulness and safety.

### Why the SFT Stages Come Back

R1-Zero proves SFT isn't *necessary* for reasoning capability. So why does R1 use it?

Two reasons:

1. **Readability** — raw RL outputs are effective but messy. Cold-start SFT constrains the format to something humans can use.
2. **Convergence speed** — starting RL from a model that already has some reasoning format is faster than starting from a raw base model. The cold-start SFT gives RL a better initialization.

The SFT stages don't *teach* reasoning capability. They *shape how reasoning capability gets expressed*.

## Efficiency Notes

R1's total training cost was estimated at 147K H100-hours — less than typical estimates for o1-class models. The efficiency comes from:

- **Verifiable rewards** — no reward model training, no reward hacking to mitigate
- **GRPO** — no value network, so less memory and compute per step (see Part 6)
- **Sparse reward structure** — simple signal, easy to scale

This combination is why reasoning models became practical to train in 2024-2025 rather than 2026-2027. The algorithmic pieces (GRPO, verifiable rewards) composed into a training recipe that's an order of magnitude cheaper than RLHF-style alignment for capable reasoning.

## Limitations of Verifiable-Reward Training

The approach works where you can verify. It doesn't work where you can't:

- **Open-ended generation** (essays, creative writing) — no ground truth to check against
- **Subjective quality** (tone, helpfulness, style) — requires judgment, not verification
- **Long-horizon tasks** (research, planning) — outcomes are hard to evaluate automatically

For these domains, you still need preference-based methods (RLHF, DPO, CAI). Reasoning training is a major unlock, but it's not universal.

The open research question: can verifiable-style rewards be extended to broader domains? Process reward models (checking reasoning steps rather than just outcomes) are one attempt. Formal specification (verifying that a solution meets declared constraints) is another. Neither is solved yet.

## The Role of the Base Model

R1-Zero worked because the base model already had math and language knowledge from pretraining. RL didn't *teach* math — it taught *how to deploy existing math knowledge effectively*. This matters for understanding what RL can and can't do:

- If the base model can solve a problem with some probability > 0, RL can push that probability up
- If the base model has zero knowledge of the domain, RL cannot bootstrap from nothing

This is why reasoning training is a *post-training* technique, not an alternative to pretraining. It's extracting and sharpening capability that pretraining put there. The ceiling is set by the base model; RL approaches that ceiling more efficiently than imitation.

---

## Key Q&A

**Q: What's the key claim R1-Zero establishes?**
A: Reasoning strategies (backtracking, verification, extended thinking) can emerge from RL with verifiable rewards, without human-written reasoning demonstrations. The model discovers what effective reasoning looks like through optimization pressure rather than imitation.

**Q: Why does R1 add SFT stages back in if R1-Zero worked?**
A: For human-readable output, not for reasoning capability. R1-Zero's traces were effective but ugly (mixed languages, missing explanations). SFT cold-start gives the model a readable format before RL does the capability work.

**Q: What's the division of labor between SFT and RL in the reasoning context?**
A: SFT teaches format — how to present reasoning. RL teaches capability — what reasoning strategies actually work. SFT imitates demonstrations (bounded by demonstrator skill). RL optimizes against rewards (bounded by reward signal quality). For reasoning, RL can discover strategies not present in any demonstration.

**Q: Why do response lengths grow during training without being explicitly rewarded?**
A: Because longer responses produce correct answers more often. GRPO assigns higher advantage to correct responses, pushing up the probability of all tokens in those responses — including the "let me reconsider..." tokens that make responses long. The length increase is a second-order effect of optimizing for correctness.

**Q: Why do verifiable rewards matter for reasoning training specifically?**
A: Three reasons: (1) no reward hacking — the checker is external and honest; (2) no reward model to train or run — direct compute savings; (3) the signal doesn't degrade as the policy improves, so you can push RL much further than with a learned reward model.

**Q: Could R1-Zero work on a base model that hadn't seen math in pretraining?**
A: No. RL extracts and sharpens capability; it doesn't create capability from nothing. R1-Zero worked because DeepSeek-V3-Base already had math and language knowledge. The reward signal would have nothing to reinforce if the base model couldn't occasionally produce correct answers.

**Q: Why don't verifiable rewards work for creative writing or dialogue?**
A: Because there's no deterministic checker for quality in those domains. "Is this essay good?" has no 0/1 answer. For these domains, you need preference-based training (RLHF, DPO, CAI) where the reward signal comes from human or AI judgment.

**Q: Is reasoning training replacing RLHF?**
A: No — they cover different domains. Reasoning training works where rewards are verifiable. RLHF works where rewards require judgment. Modern post-training pipelines use both: verifiable-reward RL for math/code capabilities, preference-based RL for general alignment and helpfulness.
