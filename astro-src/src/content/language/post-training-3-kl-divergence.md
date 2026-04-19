---
title: "Post-Training (3/8) — KL Divergence in RLHF"
date: "2026-04-24"
excerpt: "How KL divergence acts as a leash to prevent the policy model from drifting too far from the original SFT model."
project: language
readingTime: 5
---

KL divergence is the leash that keeps RLHF from going off the rails. This post explains what it measures, how it's computed efficiently during PPO training, and how the β coefficient controls the tradeoff between learning from preferences and staying close to the original model.

---

## Definition

KL (Kullback-Leibler) divergence measures how different two probability distributions are:

```
KL(A || B) = Σ A(x) × log(A(x) / B(x))
```

Properties:
- KL ≥ 0 always
- KL = 0 only when the two distributions are identical

### Concrete Example

Two coins: Coin A = 70% heads / 30% tails, Coin B = 50% heads / 50% tails:

```
KL(A || B) = 0.7 × log(0.7/0.5) + 0.3 × log(0.3/0.5)
           = 0.235 - 0.153
           = 0.082
```

## Role in RLHF

KL divergence is the "leash" that prevents the policy model from drifting too far from the reference model (the frozen SFT copy). Without it, the model could find degenerate ways to hack the reward model — e.g., repeating "I'm very helpful and knowledgeable" to score high even though it's unnatural text.

## Per-Token KL Computation

### Full KL vs. Monte Carlo Approximation

The full KL at each token position would require evaluating log probabilities for all ~50,000 vocabulary entries under both models. In practice, a Monte Carlo approximation is used — evaluate only the sampled token:

```
KL_at_position_t = log P_policy(token_t) - log P_reference(token_t)
```

The `P(x) ×` weighting from the full formula is handled implicitly: tokens with higher policy probability get sampled more often across many training steps, so they naturally contribute more to the average.

```
Cost comparison:
Full KL:       50,000 evaluations × sequence_length × 2 models
Simplified:    1 evaluation × sequence_length × 2 models
```

### Worked Example

Policy model generated "certainly" at position 5:

```
P_policy("certainly")   = 0.25 → log = -1.39
P_reference("certainly") = 0.05 → log = -3.00
KL at this position = -1.39 - (-3.00) = 1.61 (large — policy has drifted)
```

This means the policy model finds "certainly" 5× more likely than the reference model would — a significant behavioral drift.

## How KL Is Computed During PPO

The reference model **never generates anything**. It evaluates the policy model's choices:

1. Policy model generates tokens [t1, t2, t3, ...]
2. Feed the **same token sequence** through the reference model
3. At each position, get `log P_reference(token_t | context)` — "what probability would the frozen SFT model have assigned to this exact token?"
4. Compare with the policy's log probability at each position
5. Sum across all tokens to get total KL

## Adjusted Reward

```
adjusted_reward = reward_model_score - β × Σ KL_per_token
```

The β coefficient controls leash tightness:
- β too high → model barely changes from SFT, learns very little from preferences
- β too low → model drifts far from SFT, risks reward hacking

---

## Key Q&A

**Q: The full KL formula has `P(x) ×` weighting. Why does the simplified version drop it?**
A: The full formula `KL = Σ P(x) × log(P(x)/Q(x))` is an expectation under the policy distribution. When you sample one token, that sample is a single-sample estimate of this expectation. The weighting is handled implicitly — higher-probability tokens get sampled more often across many training iterations, so they contribute proportionally over time.

**Q: What's actually used in practice — full KL or the approximation?**
A: The simplified per-token version. This is what the original InstructGPT paper used and what libraries like HuggingFace TRL implement. Over thousands of training iterations, the Monte Carlo approximation converges to the true KL.

**Q: For KL computation, which model's tokens are used — the policy's or the reference's?**
A: The policy model's tokens. The policy generates the response, and then you ask the reference model "what probability would you have assigned to this same token in this same context?" The reference model never generates its own tokens.

**Q: If the KL divergence becomes very large, what does that mean?**
A: The policy model has drifted far from the original SFT behavior — generating text the SFT model would find very unlikely. This is often a sign of reward hacking, where the model exploits the reward model rather than genuinely improving.
