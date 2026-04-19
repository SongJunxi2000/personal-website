---
title: "Post-Training (8/8) — Constitutional AI and RLAIF"
date: "2026-04-19"
excerpt: "How AI feedback grounded in written principles replaces human preference labels — scaling preference learning but widening the proxy chain."
project: language
readingTime: 11
---

Constitutional AI replaces human preference labelers with an AI labeler grounded in explicit written principles. This post covers the two-phase CAI pipeline, why the proxy chain is longer than RLHF's, and when to use CAI vs. alternatives.

---

## Motivation

RLHF's bottleneck is human preference labeling. Tens of thousands of comparisons per training round, each requiring careful labeling by trained humans. Every new capability, every new safety concern, every new model iteration wants more labels. This doesn't scale.

**RLAIF (Reinforcement Learning from AI Feedback)** replaces the human labeler with an AI labeler. Everything downstream stays the same — collect preference labels, train a reward model, run RL. The only change is who generates the labels.

**Constitutional AI (CAI)** is Anthropic's specific version of RLAIF, where the AI labeler's preferences are grounded in an explicit written constitution — a set of principles the labeler uses to judge responses.

## The Core Substitution

```
RLHF:  prompt + response_A + response_B  →  human labeler  →  "A > B"
RLAIF: prompt + response_A + response_B  →  AI labeler     →  "A > B"
CAI:   prompt + response_A + response_B + constitutional principle
                                         →  AI labeler     →  "A > B per this principle"
```

Everything after the label is identical to RLHF: Bradley-Terry loss on a reward model (see Part 2), then PPO or GRPO to optimize the policy.

## The Constitution

A constitution is a set of written principles intended to capture the values the model should embody. Example principles from Anthropic's original CAI work:

- "Please choose the response that is most helpful, honest, and harmless."
- "Please choose the response that a wise, ethical person would give."
- "Please choose the response that is less likely to cause harm."
- "Please choose the response that treats the human with respect."

Principles can be sourced from external documents (UN Declaration of Human Rights, ethical frameworks) or written specifically for the model. Anthropic's original constitution had ~16 principles.

**Why a written constitution matters:** In RLHF, the values being optimized for are implicit in labelers' choices. Biases and inconsistencies get baked into the reward model invisibly. In CAI, the values are **explicit in the document**. You can read them, audit them, debate them, change them. The alignment target is legible.

## The Two-Stage CAI Pipeline

CAI has two phases that target different problems. Both rely on the model's own evaluative capacity — the bet is that **evaluation is easier than generation**, so a model can judge its own outputs usefully even when it struggles to produce good ones from scratch.

### Phase 1: SL-CAI (Supervised Learning)

Goal: produce a dataset of (prompt, harmless response) pairs for SFT, without human labelers writing the harmless responses.

For each harmful prompt:

```
Step 1 — Generate: An RLHF-trained helpful model responds to a harmful prompt.
         Often produces something problematic (optimized for helpfulness above safety).

Step 2 — Critique: Ask the same model:
         "Identify ways in which your previous response was harmful,
          unethical, or problematic. [Cite a specific constitutional principle.]"
         The model produces a self-critique.

Step 3 — Revise: Ask the model:
         "Rewrite your response to remove the problematic content you identified."
         The model produces a revised response.
```

Repeat for thousands of prompts → dataset of (prompt, revised response) pairs → SFT on those pairs. The resulting SL-CAI model has internalized the behavior of generating responses that would survive its own critique.

**Why this works:** The model evaluates its own output against principles it can follow when asked to, even if it didn't follow them when generating. Critique-then-revise bridges the gap — the model uses its stronger evaluation capacity to improve its weaker generation capacity.

### Phase 2: RL-CAI (Reinforcement Learning)

Goal: continued preference-based training using AI labels, starting from the SL-CAI model.

Standard RLAIF loop:

```
Step 1 — Sample two responses to a prompt from the SL-CAI model
Step 2 — Show both responses + a constitutional principle to an AI labeler
         (often a different model or earlier checkpoint)
         Ask which response better satisfies the principle
Step 3 — Collect thousands of (prompt, pref_response, rej_response) triples
Step 4 — Train a reward model on this preference data
         (Bradley-Terry loss, exactly as in Part 2)
Step 5 — Run RL (PPO or GRPO) to optimize the policy against this reward model,
         with KL penalty to the SL-CAI model
```

### Why Two Phases?

They target different problems:

- **Phase 1** reshapes the model's *default behavior* via direct SFT on self-revised examples. Big, direct change to what the model produces.
- **Phase 2** provides *continued pressure* toward constitutional adherence via RL against an AI-preference reward model. Fine-grained optimization on top of the new default.

Phase 1 alone isn't enough because SFT can't easily express "A is slightly better than B" (see Part 1). Phase 2 adds the preference-based signal.

Phase 2 alone isn't enough because starting RL from a model with bad default behavior wastes compute — RL works better from a reasonable starting point.

## Models in Memory

During Phase 2 RL training:

| Model | Purpose |
|---|---|
| Policy (initialized from SL-CAI) | Being updated |
| Reference (frozen SL-CAI copy) | Anchor for KL penalty |
| Reward model (trained on AI labels) | Scores responses |
| AI labeler (for generating preferences) | Used during preference data collection, not during RL itself |

Same structure as RLHF's four models, just with the preference source swapped.

## The Proxy Chain: Why CAI Has More Attack Surface than RLHF

This is the key insight for understanding CAI's tradeoffs.

**RLHF proxy chain:**
```
True goal  →  Human preferences  →  Reward model  →  Policy optimization
```
Three gaps, three places for the policy to exploit.

**CAI proxy chain:**
```
True goal  →  Constitution  →  AI labeler interpretation  →  Reward model  →  Policy optimization
```
Four gaps, four places to exploit.

Each added link is a new specification-to-implementation gap:

1. **Goal → Constitution gap:** The written principles are a lossy compression of fuzzy human values. "Be helpful" leaves out a lot.
2. **Constitution → AI labeler gap:** The labeler's interpretation of principles may differ from authors' intent. "Choose the more honest response" means something slightly different to the labeler model than to a human evaluator.
3. **AI labeler → Reward model gap:** The reward model learns to predict labeler outputs, not the labeler's underlying judgments. Same reward-model-hacking surface as RLHF.
4. **Reward model → Policy gap:** The policy finds reward model blind spots, same as in RLHF.

A sufficiently capable policy can exploit any of these gaps:
- **Constitution gaps** — behaviors that satisfy principles as written but violate the spirit
- **Labeler quirks** — stylistic patterns the labeler associates with "good" that aren't actually good
- **Reward model blind spots** — the usual reward hacking

So CAI has *more* attack surface than RLHF, not less. The trade is: CAI buys scalability and value transparency, at the cost of a longer proxy chain.

## When to Use CAI vs. Alternatives

The three main options produce a clear tradeoff:

| Approach | Reward source | Scalability | Attack surface | Domain |
|---|---|---|---|---|
| RLHF | Human preferences | Labor-bound | Medium | Broad |
| CAI/RLAIF | AI preferences + constitution | Compute-bound | High (more proxy layers) | Broad |
| GRPO + verifiable reward | External checker | Excellent | Minimal | Narrow (reasoning) |

Moving down the table: more scalable, more grounded, but narrower in applicable domain. CAI sits in the middle — scalable like verifiable rewards, broad-domain like RLHF, but with the most attack surface of the three.

The practical choice depends on domain:
- Math/code → GRPO + verifiable rewards (R1-style)
- General helpfulness/safety → CAI for scale, RLHF for rigor
- Novel capabilities you can't verify → RLHF until you can automate the evaluation

Most modern frontier training pipelines combine multiple approaches in different stages.

## The Recursive Quality Problem

CAI depends on the AI labeler being good enough to judge responses reliably. But the labeler is itself a language model. Three failure modes:

1. **If the labeler is weaker than the policy being trained**, it can't evaluate outputs reliably — the policy can produce reasoning beyond the labeler's judgment.
2. **If the labeler misinterprets principles**, those misinterpretations get amplified through RL.
3. **If the labeler has systematic blind spots**, the trained policy will have the same blind spots — and learn to exploit them.

Anthropic's bet: for many alignment-relevant judgments, evaluation is easier than generation, so even similar-capability labelers can provide useful signal. This assumption gets weaker as models become more capable, which connects directly to the **scalable oversight** problem — how do you provide training signal for models whose outputs you can't directly judge?

CAI is one answer to scalable oversight (use AI to help evaluate AI). It's not a complete answer, because the labeler is subject to the same capability limits as the policy. Research directions like debate, recursive reward modeling, and interpretability-guided oversight are attempts to extend CAI-style approaches further.

---

## Key Q&A

**Q: What's the difference between RLAIF and Constitutional AI?**
A: RLAIF is the general pattern — AI feedback replacing human feedback for preference labels. CAI is Anthropic's specific version where the AI labeler is prompted with explicit written principles (the constitution). CAI is RLAIF with a specific grounding mechanism; RLAIF without a constitution is just "ask an AI what's better," which has less principled grounding.

**Q: Why does SL-CAI work? Isn't the model generating both the problem response and the fix?**
A: Yes, and that's the key — evaluation is easier than generation. The model can identify problems in its existing output even when it would make the same mistakes from scratch. Critique-then-revise uses the model's stronger evaluation capacity to bootstrap improvements in its generation capacity. It's the same model wearing different hats.

**Q: Why isn't SL-CAI alone enough?**
A: Because SFT can't express fine-grained preferences — it treats target responses as exact sequences (see Part 1). After SL-CAI fixes default behavior, Phase 2 (RL-CAI) uses preference data to express "A is slightly better than B" and push the model further. Same reason RLHF has both SFT and RL stages.

**Q: Is the AI labeler the same model as the policy being trained?**
A: Usually not in the same forward pass. Common setups: the labeler is an earlier checkpoint of the same model family, a different model entirely (e.g., a larger, slower model providing labels for a smaller one being trained), or even a committee of different labelers. The point is that the labeler provides judgment during training; it's not being trained itself during RL-CAI.

**Q: Does CAI avoid reward hacking?**
A: No — it has *more* attack surface than RLHF, not less. The reward model trained on AI labels is still learned and still hackable. Plus there are two extra layers above it: the constitution-to-labeler-interpretation gap, and the goal-to-constitution gap. CAI trades reduced human labor for increased specification risk.

**Q: Why bother with the constitution at all — why not just ask the AI "which is better"?**
A: Grounding. Without a constitution, the AI's preferences reflect its training biases and implicit values — opaque and arbitrary. With a constitution, the preferences are grounded in explicit written criteria that can be audited, changed, and debated. The constitution makes the alignment target legible, which matters for interpretability of the training target, not just the trained model.

**Q: How does CAI relate to scalable oversight?**
A: CAI is a partial answer to scalable oversight — use AI labelers to scale beyond human-labeling bottlenecks. But it's not a complete answer, because the labeler has its own capability limits. When the policy exceeds the labeler's judgment, CAI degrades. Full scalable oversight requires additional mechanisms (debate, recursive verification, interpretability) to extend evaluation capacity beyond the labeler itself.

**Q: Does CAI work for reasoning tasks?**
A: In principle yes, but it's usually not the best choice. For reasoning, verifiable rewards (math answer matching, code test passing) are cheaper, more grounded, and hack-resistant. CAI is more useful for domains where verification is hard — general helpfulness, tone, safety judgments, open-ended dialogue. The modern pipeline uses verifiable rewards for reasoning and CAI-style approaches for broader alignment.
