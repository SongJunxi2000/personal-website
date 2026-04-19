---
title: "Post-Training (6/8) — GRPO (Group Relative Policy Optimization)"
date: "2026-04-21"
excerpt: "How GRPO replaces PPO's value network with a group-mean baseline, enabling efficient RL with verifiable rewards."
project: language
readingTime: 9
---

GRPO trades PPO's learned value network for a simple group-mean baseline — sample many responses to the same prompt, compare within the group. This post covers the algorithm, the bias-variance tradeoff against PPO, why uniform advantage works for reasoning tasks, and the computational costs.

---

## Motivation

PPO's value network is expensive. It's roughly a full model's worth of parameters, needs its own forward/backward passes, its own optimizer states, and its own training loss. For reasoning tasks (math, code), there's an additional issue: the reward is often binary (correct/incorrect), which makes the value network's job harder — it's trying to predict expected reward from a noisy, sparse signal.

GRPO asks: what if you skip the value network entirely and use a simpler baseline?

**Key idea:** Instead of learning a function that predicts expected reward, sample many responses to the same prompt and use the group mean as the baseline. Monte Carlo estimate instead of learned estimate.

## The Core Substitution

```
PPO:  advantage_t = adjusted_reward - value_prediction_t       (per-token, learned baseline)
GRPO: A_i         = (r_i - group_mean) / group_std            (per-response, sampled baseline)
```

Where:
- `r_i` is the reward for response *i* in the group
- `group_mean` and `group_std` are computed across the G sampled responses for the same prompt

Every token in response *i* gets the **same advantage** `A_i`. No per-token credit assignment.

## The Algorithm

### Setup

For each prompt:
1. Sample G responses from the current policy (typically G = 8, 16, or 32)
2. Score each response — usually with a verifiable checker (math answer match, code tests passing), sometimes with a learned reward model
3. Compute group mean and standard deviation of rewards
4. Normalize: `A_i = (r_i - mean) / std` for each response

### The GRPO Objective

```
L_GRPO = -E[ min(ratio_t × A_i, clip(ratio_t, 1-ε, 1+ε) × A_i) ]
         + β × KL(policy || reference)
```

This is PPO's clipped objective with two changes:
- `A_i` is the group-normalized advantage, constant across all tokens in response *i*
- KL penalty is added directly to the loss (rather than folded into the reward as in PPO)

The ratio, clipping, and mini-epoch structure work exactly as in Part 4.

### Worked Example

Group of 10 responses on a math problem. 6 get the answer right (reward 1), 4 get it wrong (reward 0):

```
mean = (6 × 1 + 4 × 0) / 10 = 0.6
variance = (6 × (1 - 0.6)² + 4 × (0 - 0.6)²) / 10
         = (6 × 0.16 + 4 × 0.36) / 10 = 0.24
std = √0.24 ≈ 0.49

Advantage for correct responses: (1 - 0.6) / 0.49 ≈ +0.82
Advantage for wrong responses:   (0 - 0.6) / 0.49 ≈ -1.22
```

Wrong responses get penalized harder than correct responses get rewarded. The std normalization keeps advantage magnitudes stable across groups with different correctness ratios.

### The Degenerate Case

If all G responses get the same reward (all correct or all wrong), the group mean equals each `r_i`, advantages are all zero, and no learning happens on that prompt. This is why GRPO needs prompts at the edge of the model's capability — hard enough that not all succeed, easy enough that not all fail.

## Why Uniform Advantage Works for Reasoning

Every token in response *i* gets the same `A_i`. This would be disastrous for open-ended generation — a mostly-good essay with one weak paragraph would push up the bad tokens alongside the good ones. But for reasoning, it works because correctness is a property of the **whole trajectory**:

- A math proof is valid or invalid as a unit — there's no meaningful sense in which "token 47 was great but token 113 was bad"
- Correct reasoning chains are coherent — the good tokens support each other
- Wrong reasoning chains are coherent in their wrongness — the error propagates

So uniform advantage across a trajectory is *appropriate* for reasoning tasks. It's inappropriate for tasks where reward reflects a weighted sum of per-token qualities.

This is why GRPO took off specifically in the reasoning era — it's well-matched to the settings that became important in 2024-2025, not universally superior to PPO.

## Models in Memory

| Model | PPO | GRPO |
|---|---|---|
| Policy | ✓ (trained) | ✓ (trained) |
| Reference | ✓ (frozen) | ✓ (frozen) |
| Reward model | ✓ (frozen) | Optional — replaced by verifier when possible |
| Value network | ✓ (trained) | ✗ (replaced by group mean) |

Dropping the value network is the big computational win — typically 30-50% memory reduction during training, plus no value-network forward/backward passes.

## Bias-Variance Tradeoff vs. PPO

This is the real story, and it's not just "GRPO is cheaper."

**PPO's value network:** low variance (smooth learned predictions), but **biased** — the value network is always chasing a moving target (the policy keeps changing), and an imperfect estimator introduces systematic error in the advantages.

**GRPO's group mean:** higher variance (only G samples, so the estimate is noisy), but **unbiased** — it's a direct Monte Carlo estimate computed fresh from the current policy.

For rapidly-improving policies (as in RL-driven reasoning), the unbiased estimate can be more valuable than the lower-variance biased one. PPO's value network struggles to keep up when the policy discovers new strategies; GRPO's group estimate doesn't have that lag.

## The Cost GRPO Pays

Nothing comes free. GRPO trades value-network cost for **more generations per prompt**:

- PPO: 1 generation per prompt
- GRPO with G=16: 16 generations per prompt

Since generation is autoregressive and therefore expensive, this is significant. The tradeoff works in GRPO's favor when:
- Generation is cheaper than maintaining a value network (often true for reasoning because responses are long but the model isn't huge)
- The group baseline is more useful than the learned one (often true when the reward is verifiable and sparse)

The variance of the group-mean baseline decreases as `1/G`, so larger G gives a better baseline at linearly higher cost.

---

## Key Q&A

**Q: What does the group mean replace?**
A: The value network's prediction. In PPO, advantage = reward - value_prediction. In GRPO, advantage = (reward - group_mean) / group_std. Both are baselines that prevent "push everything up when reward is positive."

**Q: Is the group mean a worse baseline than the value network?**
A: Worse in one dimension, better in another. Worse: it doesn't provide per-token credit assignment — every token in a response gets the same advantage. Better: it's unbiased (direct Monte Carlo estimate) and doesn't lag behind a changing policy. For reasoning tasks where trajectories are coherent, the per-token attribution isn't needed anyway.

**Q: Does GRPO require verifiable rewards?**
A: No — GRPO can use any reward source, including a learned reward model. The DeepSeekMath paper originally used learned rewards. The *synergy* is with verifiable rewards: GRPO removes the value network, and verifiable rewards remove the reward model, leaving just policy + reference + verifier. That combination is what unlocked efficient reasoning-focused RL.

**Q: Why doesn't GRPO need per-token advantages?**
A: Because for reasoning tasks, reward reflects whole-trajectory properties (a proof is valid or invalid as a unit). For open-ended generation where reward reflects a weighted sum of per-token qualities, uniform advantage across the response would be wrong. GRPO trades off this capability because reasoning doesn't need it.

**Q: What happens when all G responses get the same reward?**
A: No learning signal on that prompt. Mean equals each `r_i`, advantages are all zero. This is why GRPO needs a curriculum — prompts where the current model sometimes succeeds and sometimes fails. Prompts that are always right or always wrong are wasted compute.

**Q: Where does the "clipping" come from?**
A: Same as PPO — see Part 4. GRPO inherits PPO's clipped objective, mini-epoch structure, and all the stability machinery. The only change is the advantage estimator. Think of GRPO as "PPO with a different advantage function."

**Q: Why is the KL penalty added to the loss rather than folded into the reward?**
A: Implementation choice. In PPO (see Part 3), the KL penalty reduces the reward at each token, which then feeds into advantage computation. In GRPO's original formulation, the advantage is already computed from group statistics (no per-token reward to modify), so KL is added as a separate term in the final loss. The effect is similar — both penalize divergence from the reference model — but the math is cleaner when KL is a separate loss term.

**Q: If I used GRPO with a learned reward model, what's my attack surface?**
A: Exactly the same as RLHF — the learned reward model can be gamed. GRPO doesn't reduce reward-hacking risk; it only removes the value network. The reduction in attack surface comes from moving to verifiable rewards, which GRPO makes practical but doesn't require.
