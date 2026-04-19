---
title: "Post-Training (2/8) — The Reward Model"
date: "2026-04-25"
excerpt: "Training a model to score response quality using human preference data and the Bradley-Terry framework."
project: language
readingTime: 6
---

The reward model translates human preferences into a scalar signal that RL can optimize. This post covers how it's built from a pretrained transformer, trained on pairwise comparisons using the Bradley-Terry framework, and why the probabilistic foundation matters.

---

## Purpose

The reward model is a proxy for human judgment. Humans provide preference labels ("I prefer response A over response B"), and the reward model learns to assign scalar scores to responses such that preferred responses score higher.

## Architecture

Take a pretrained transformer (typically initialized from the SFT model), remove the final token-prediction head (hidden_dim → vocab_size), and replace it with a single linear layer (hidden_dim → 1).

```
Language Model head:  4096 → 50,000 (probability over vocabulary)
Reward Model head:    4096 → 1       (scalar score)
```

The transformer body is identical — same layers, same attention + FFN blocks. Only the final projection changes.

### How It Produces a Score

- The reward model **does not generate anything**
- It receives the full (prompt + response) concatenated as one input
- One forward pass through the transformer produces hidden states at every token position
- Take the hidden state at the **last token position** only (it has attended to everything via causal attention)
- Project that hidden state to a single scalar → the reward score
- Different response lengths don't matter — each produces a hidden state at its own last position

## Training

### Collecting Preference Data

1. Take the SFT model (single set of weights)
2. Give it a prompt
3. Sample two or more different completions (same weights, different random draws during sampling)
4. Show both completions to a human labeler
5. The human says "I prefer response A over response B"
6. Result: dataset of (prompt, preferred response, rejected response) triples

### Loss Function

For each training example, two forward passes:

```
[prompt + preferred response] → score_preferred
[prompt + rejected response]  → score_rejected
```

Loss:
```
L = -log(sigmoid(score_preferred - score_rejected))
```

### How the Loss Works

**Sigmoid** maps the score difference to [0, 1]:
```
sigmoid(x) = 1 / (1 + e^(-x))
```

**-log** converts to a loss:

| Scenario | Difference | sigmoid | -log | Loss |
|---|---|---|---|---|
| Correct ranking (large gap) | +4 | 0.98 | 0.02 | Small |
| Can't distinguish | 0 | 0.50 | 0.69 | Moderate |
| Wrong ranking | -4 | 0.02 | 4.02 | Large |

The model is penalized much more harshly for confident mistakes than it's rewarded for confident correct answers.

### Absolute Scale Doesn't Matter

Only the difference between scores matters. Scores of (95, 90) produce the same loss as (5, 0) — both have difference = 5. There's no incentive to push scores apart once the ranking is already correct.

## Bradley-Terry Foundation

The loss isn't an arbitrary penalty — it comes from the **Bradley-Terry model** of pairwise comparison:

```
P(i > j) = s_i / (s_i + s_j)
```

Reparameterize strengths as exponentials: `s_i = e^(r_i)`:

```
P(i > j) = e^(r_i) / (e^(r_i) + e^(r_j))
         = 1 / (1 + e^(-(r_i - r_j)))
         = sigmoid(r_i - r_j)
```

The reward model loss is the **negative log-likelihood** of observed human preferences under this model.

### Why the Probabilistic Interpretation Matters

- **Handles noisy labelers:** If 7/10 labelers prefer A, the model learns P(A > B) ≈ 0.7 — a moderate score gap, not an extreme one. Contradictory labels don't fight each other.
- **Calibration:** A score difference of 2 means ~88% preference probability everywhere (`sigmoid(2) ≈ 0.88`), making reward signals consistent across different prompts.
- **Theoretical guarantees:** Maximum likelihood estimation is consistent — with enough data, you converge to the true preference ordering.

---

## Key Q&A

**Q: What do you mean by "replace the final token prediction head with a single linear layer"?**
A: A "head" is just a final output layer on top of the transformer body. The language model head is a matrix multiplication from hidden_dim (4096) to vocab_size (50,000). The reward model head replaces this with a matrix multiplication from 4096 to 1 — a single scalar output. Both are linear layers (matrix multiply, no activation function).

**Q: What happens if responses have different lengths?**
A: Doesn't matter. Both go through one forward pass. Both produce a hidden state at their respective last token position. Both get projected to one number. The last token works as a summary because causal attention lets it attend to everything before it.

**Q: Does the response generation and reward model training happen at the same time?**
A: No, completely separate. First, generate two responses using the SFT model (autoregressive sampling). Then, feed each complete (prompt + response) into the reward model as a single forward pass and get scores. Generation and scoring are independent processes.

**Q: How is `sigmoid(score_A - score_B)` related to the Bradley-Terry model?**
A: Bradley-Terry uses positive ratio scores. The reward model uses unbounded real-valued scores. The bridge is exponential reparameterization: `s_i = e^(r_i)`. Substituting this into the Bradley-Terry formula and simplifying yields exactly `sigmoid(r_i - r_j)`.

**Q: Why use this probabilistic loss instead of any function that penalizes wrong rankings?**
A: Many functions could penalize wrong rankings (hinge loss, squared difference). The probabilistic version wins because: (1) it gracefully handles disagreement among labelers, (2) scores are calibrated consistently across examples, (3) maximum likelihood has theoretical convergence guarantees that arbitrary penalty functions lack.
