---
title: "Post-Training (4/8) — PPO (Proximal Policy Optimization)"
date: "2026-04-23"
excerpt: "How PPO uses policy gradients, value networks, and clipping to safely update the language model using reward signals."
project: language
readingTime: 12
---

PPO is the workhorse RL algorithm behind RLHF. This post walks through the full machinery: why discrete sampling breaks normal backpropagation, how policy gradients bypass that problem, what the value network and clipping do, and how the complete four-model training loop fits together.

---

## The Core Problem: Non-Differentiable Generation

In SFT, you know the correct next token at every position (teacher forcing), so you compute loss at each position and backpropagate normally. With a reward model, the situation is fundamentally different:

1. You only get a score for the **entire completed sequence**, not per-token feedback
2. To get that score, the model must **actually sample** tokens one by one
3. Sampling is a discrete operation — you can't take the gradient of "pick token #4523"

Analogy: SFT is like a teacher grading each test answer individually. RLHF is like receiving a single final grade for the whole test — you need a different strategy to figure out which answers to improve.

## Policy Gradient: The Key Idea

### RL Framing

- **Agent / Policy**: the language model's probability distribution over the vocabulary at each step
- **Action**: selecting a specific token from that distribution
- **State**: the tokens generated so far
- **Reward**: the reward model's score on the complete response

The policy is the full probability distribution itself, **not** "pick the highest probability token." During training, you sample from it for exploration — sometimes picking the 2nd or 10th most likely token to discover that alternative phrasings might score higher.

### The Policy Gradient Objective

The model generates tokens [t1, t2, ..., tN], recording the probability it assigned to each:

```
objective = R × Σ log P(token_t)
```

This is differentiable — the sampling already happened, and you're computing gradients of the log-probabilities (which depend on the weights) multiplied by the reward (a fixed scalar).

Effect:
- High reward → increase probability of all tokens in that response
- Low reward → decrease probability of all tokens

### The Credit Assignment Problem

This naive approach treats all tokens equally. If the overall response is good but some tokens were bad (or vice versa), every token gets the same signal. This is noisy and wasteful.

## The Value Network

### Purpose

The value network predicts the **expected reward at each token position**: "given the tokens so far, what reward do I expect at the end?" This enables per-token credit assignment.

### Architecture

An additional linear head (hidden_dim → 1) attached to the policy model's transformer body (shared weights). Unlike the reward model which outputs one score per response using only the last token, the value network outputs a score at **every token position**.

```
                    Transformer body
                    (shared weights)
                         |
                    hidden state
                    (dim = 4096)
                       /    \
                      /      \
        Vocab head           Value head
     4096 → 50,000          4096 → 1
           |                      |
   token probabilities      expected reward
```

A "head" is just a final output layer — a single linear (matrix multiply) projection that sits on top of the shared transformer body.

### Input and Output

```
Input: [prompt + response tokens generated so far]

At each position, the value head outputs a prediction:
Position 1 ("The"):     value = 4.2  → "I expect reward ≈ 4.2"
Position 2 ("answer"):  value = 4.5  → "I expect reward ≈ 4.5"
Position 3 ("is"):      value = 4.8  → "I expect reward ≈ 4.8"

Actual reward for complete response: 6.0
```

### Training

The value network has its own separate loss — squared error between prediction and actual reward:

```
L_value = (predicted_value - actual_adjusted_reward)²
```

Trained simultaneously with the policy model during PPO.

### Computing Advantages

```
advantage_t = adjusted_reward - value_prediction_t
```

- Positive advantage → "things went better than expected after this token" → push probability up
- Negative advantage → "things went worse than expected" → push probability down

The improved objective uses per-token advantages instead of a single reward:

```
objective = Σ advantage_t × log P(token_t)
```

## PPO Clipping

### The Stability Problem

Policy gradient estimates are noisy. A single lucky high-reward sample could cause a massive weight update that destabilizes training. Also, after a large update, the advantage estimates (computed under old probabilities) become stale and invalid.

### The Ratio

PPO runs multiple **mini-epochs** of gradient updates on each batch. The ratio tracks how far the policy has moved:

```
ratio_t = P_policy_now(token_t) / P_policy_old(token_t)
```

- `P_policy_old` = probability at generation time (fixed snapshot for this iteration)
- `P_policy_now` = probability under current weights (changes with each mini-epoch)

On the first mini-epoch, ratio ≈ 1.0 everywhere (weights haven't changed yet). It diverges from 1.0 in subsequent mini-epochs as weights are updated.

### The Clipped Objective

```
L = min(ratio_t × advantage_t, clip(ratio_t, 1-ε, 1+ε) × advantage_t)
```

With ε ≈ 0.2, the ratio is clamped to [0.8, 1.2]:

```
clip(0.5, 0.8, 1.2) → 0.8   (clamped up)
clip(0.9, 0.8, 1.2) → 0.9   (unchanged)
clip(1.15, 0.8, 1.2) → 1.15  (unchanged)
clip(1.5, 0.8, 1.2) → 1.2   (clamped down)
```

### How Clipping Works: Four Cases

| Advantage | Ratio | Min picks | Gradient effect |
|---|---|---|---|
| Positive (good token) | > 1.2 | Clipped (constant 1.2 × adv) | Gradient = 0 — no incentive to boost further |
| Positive (good token) | < 0.8 | Unclipped (small positive) | Weak gradient signal |
| Negative (bad token) | > 1.2 | Unclipped (very negative) | Active gradient — penalize hard |
| Negative (bad token) | < 0.8 | Clipped (constant 0.8 × adv) | Gradient = 0 — already declining |

**Why gradient becomes zero when clipped:** The clipped value (e.g., `1.2 × advantage`) is a constant — neither 1.2 nor the advantage depends on the current policy weights. The gradient of a constant with respect to the weights is zero. Compare with the unclipped `ratio × advantage`: the ratio involves `P_policy_now`, which depends on the weights, so the gradient is non-zero.

**Pattern:** Clipping creates a one-sided wall. Moving further in the "beneficial" direction → gradient goes to zero (no incentive). Moving in the "harmful" direction → gradient stays active (pulled back). The clipping is preventive — it shapes the loss function before the gradient is computed, so updates are never too large in the first place.

## Mini-Epochs

### Why Multiple Updates Per Batch

Generating responses and scoring them is expensive (autoregressive decoding + reward model + reference model forward passes). Mini-epochs squeeze 3-4 gradient updates from each expensive batch.

### How They Work

All mini-epoch updates are **real weight updates**, not temporary:

```
Start: weights = W₀, P_policy_old saved from generation

Mini-epoch 1:
  Forward pass with W₀ → P_policy_now
  ratio ≈ 1.0 → no clipping → normal gradient update
  Real update: W₁ = W₀ + gradient step

Mini-epoch 2:
  Forward pass with W₁ → P_policy_now
  ratio starts diverging → some tokens may clip
  Real update: W₂ = W₁ + gradient step

Mini-epoch 3:
  Forward pass with W₂ → P_policy_now
  more tokens clipped → smaller gradients
  Real update: W₃ = W₂ + gradient step

Mini-epoch 4:
  many tokens clipped → very small update
  Real update: W₄ = W₃ + gradient step
```

The clipping ensures mini-epoch reuse is safe — stale advantages remain approximately valid because the policy can't move far.

### The Loss Function Across Mini-Epochs

The same clipped objective is used every mini-epoch. The advantages and `P_policy_old` are computed once and stay fixed. Only `P_policy_now` changes as weights are updated.

## The Full PPO Training Loop

### Setup: Four Models in Memory

| Model | Updated during PPO? | Purpose |
|---|---|---|
| Policy model | Yes | Generates responses, gets improved |
| Reference model (frozen SFT copy) | No | Anchor for KL penalty |
| Reward model | No | Scores complete responses |
| Value network (head on policy model) | Yes | Predicts expected reward per token |

### One Iteration

**Step 1 — Generate:** Policy model samples responses to a batch of prompts (pure inference, no gradients). Save every token and its log probability as `P_policy_old`.

**Step 2 — Score:** Frozen reward model scores each complete (prompt + response) → one scalar reward per response.

**Step 3 — KL penalty:** Frozen reference model evaluates the same token sequences → compute per-token KL → `adjusted_reward = reward - β × ΣKL`.

**Step 4 — Value predictions:** Value head outputs expected reward at each token position.

**Step 5 — Advantages:** `advantage_t = adjusted_reward - value_prediction_t` at each position.

**Step 6 — PPO policy update:** Run 3-4 mini-epochs on this batch. Each mini-epoch: forward pass → compute ratio → clipped objective → gradient → real weight update.

**Step 7 — Value network update:** Minimize `(value_prediction - adjusted_reward)²` — separate loss, always active regardless of clipping.

**Step 8 — Repeat** from Step 1 with new prompts using the updated policy.

After training completes, only the policy model is deployed. Reference model, reward model, and value network are all discarded.

---

## Key Q&A

**Q: What is a "policy" exactly?**
A: The model's probability distribution over the vocabulary at each step. It's not "pick the highest probability token" — it's the full distribution. During training, you sample from it for exploration. During deployment, you can use any decoding strategy.

**Q: Why can't you backpropagate through the reward model into the language model?**
A: Because generating a response requires discrete sampling (choosing specific tokens), which is non-differentiable. Policy gradient sidesteps this by computing gradients of log-probabilities of already-sampled tokens, which are differentiable.

**Q: What is "vanilla policy gradient"?**
A: "Vanilla" means the simplest, most basic version — no modifications. Vanilla policy gradient = basic policy gradient without PPO's improvements (no baseline subtraction, no clipping). It works in principle but is noisy and unstable.

**Q: In PPO, is `P_policy_old` from the previous iteration's policy, and `P_policy_now` from the current one?**
A: Both refer to the **same response**. `P_policy_old` is the probability the model assigned at generation time (fixed snapshot). `P_policy_now` is the probability the model assigns to those same tokens after mini-epoch weight updates. The ratio tracks how much probabilities have shifted within this iteration's mini-epochs.

**Q: Are mini-epoch updates temporary or real?**
A: All real. Each mini-epoch does an actual weight update. The clipping prevents any single update from being too large. By the 3rd-4th mini-epoch, many tokens are clipped and updates become very small naturally.

**Q: Why does RLHF update the policy weights multiple times per batch?**
A: Cost efficiency. Generation and scoring are the expensive steps (full autoregressive decoding, multiple forward passes through large models). Mini-epochs squeeze more learning from each expensive batch. Clipping ensures this reuse is safe.

**Q: What is the value network's architecture?**
A: An additional linear head (4096 → 1) attached to the policy model's shared transformer body. "Head" means a final output layer — a linear projection sitting on top of the transformer. The policy model has two heads: the vocab head (4096 → 50,000) for token probabilities and the value head (4096 → 1) for expected reward prediction. Both read from the same hidden states produced by one forward pass.

**Q: What's the relationship between the "human feedback" and the PPO training loop?**
A: The human feedback already happened before PPO starts. Humans labeled preference pairs → trained the reward model. The reward model is frozen during PPO and acts as an automatic scoring function — a proxy for human judgment. No humans are in the loop during PPO training itself.
