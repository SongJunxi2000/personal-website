---
title: "Training Fundamentals (4/4) — Scaling Laws: Kaplan, Chinchilla, and Beyond"
date: "2026-04-03"
excerpt: "How model performance relates to parameters, data, and compute — and why the industry shifted from 'make it bigger' to 'train it longer.'"
project: language
readingTime: 9
---

Given a fixed compute budget, how should you split it between model size and training data to get the best performance? This question has shaped every major LLM training decision since 2020, and the answer has changed dramatically.

---

## The core relationship

Three variables are in play:

| Variable | Symbol | Meaning |
|----------|--------|---------|
| Parameters | N | Total learnable weights in the model |
| Training data | D | Total tokens processed during training |
| Compute | C | Total floating-point operations (FLOPs) |

These are linked by a simple relationship:

$$C \approx 6ND$$

The factor of 6 comes from: forward pass ~2 FLOPs per parameter per token (one multiply, one add), backward pass ~4 FLOPs per parameter per token (~2× forward). This holds regardless of model size because it's just counting arithmetic in the matrix multiplies.

**Key implication:** with fixed C, making N bigger forces D smaller, and vice versa. That's the fundamental tradeoff.

---

**Q: What's a power law, and why does it matter here?**

The key empirical discovery: model performance (test loss L) follows a power law with respect to each variable:

$$L = a \cdot x^{-k}$$

**Critical property: diminishing returns.** Each proportional increase in x gives a smaller proportional decrease in loss. You need to keep making x proportionally bigger to get the same improvement.

How was this discovered? Purely empirically — train many Transformer models at different sizes, varying amounts of data, record test loss, and plot on log-log axes. The data points fell on a straight line. Taking log of both sides: log(L) = log(a) − k·log(x), which is linear with slope −k. A straight line on a log-log plot **is** a power law.

This is remarkably robust empirically (holds over 7+ orders of magnitude of compute), but not proven from theory. No one has derived from first principles that Transformers must follow power laws.

---

## Kaplan (OpenAI, 2020)

Trained Transformer language models across many sizes on WebText2 and fit separate power laws. The conclusion: **model size matters much more than data size.** If compute increases 10×, increase N by ~5.5× but D by only ~1.8×.

**The flaw:** Kaplan's team used the same cosine learning rate schedule for all model sizes without re-tuning. A cosine schedule decays from a high value to near zero over a fixed number of steps. Small models (cheap per step) can run many more steps than large models for the same compute budget — but if the schedule was designed for fewer steps, the learning rate bottoms out and the extra steps train at near-zero learning rate.

The comparison was unintentionally biased against smaller models trained on more data. Additional data appeared less useful than it actually was.

---

## Chinchilla (DeepMind, 2022)

**What they fixed:** Trained 400+ models across many (N, D) combinations and re-tuned the learning rate schedule independently for each run.

**The loss model:**

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

Where A/Nᵅ is loss from finite model size, B/Dᵝ is loss from finite data, and E is the irreducible entropy floor of natural language.

**The result:** Given fixed compute C = 6ND, minimizing L gives:

$$N_{\text{optimal}} \propto C^{0.5}, \quad D_{\text{optimal}} \propto C^{0.5}$$

Both scale as the square root of compute — N and D should be scaled **equally**. The fitted ratio:

$$D_{\text{optimal}} / N_{\text{optimal}} \approx 20$$

**For every parameter, train on about 20 tokens.**

DeepMind demonstrated this concretely: with the same compute used for Gopher (280B params, 300B tokens), they built Chinchilla (70B params, 1.4T tokens) and it outperformed Gopher despite being 4× smaller. This proved that Gopher, GPT-3, and most large models of that era were severely undertrained.

---

## Post-Chinchilla: the inference cost shift

**Q: What did Chinchilla miss?**

Chinchilla optimizes for training compute — minimizing loss for a fixed training budget. But training is a one-time cost. Inference — running the model for every user query — is an ongoing cost that scales with N (more parameters = more FLOPs per forward pass).

If you're serving billions of queries, inference cost dominates. The new strategy: **deliberately over-train smaller models** on far more data than Chinchilla optimal.

| Model | N | D | D/N | Strategy |
|-------|---|---|-----|----------|
| GPT-3 (2020) | 175B | 300B | 1.7× | Severely undertrained |
| Gopher (2021) | 280B | 300B | 1.1× | Severely undertrained |
| Chinchilla (2022) | 70B | 1.4T | 20× | Training-optimal |
| LLaMA 2 (2023) | 70B | 2T | 28.6× | Intentionally over-trained |
| LLaMA 3 (2024) | 70B | 15T | 214× | Heavily over-trained |

LLaMA 3's strategy: spend more on training (one-time) to get a smaller, cheaper-to-deploy model. The extra training cost is paid once; the inference savings compound across every query.

---

## The entropy floor

**Q: Is there a limit to how low loss can go?**

Yes. The theoretical floor is the **entropy rate of natural language** — the inherent unpredictability in what comes next. Key estimates for English converge around ~1 bit per character (bpc), meaning English is roughly 80% redundant relative to random text.

At some point, scaling gives almost nothing — you're spending exponentially more compute to squeeze out fractions of a bit. Modern LLMs have only recently approached this benchmark.

---

## The data wall

**Q: Where does the data come from for these massive training runs?**

The scaling law C = 6ND doesn't distinguish between 1 trillion unique tokens seen once and 100 billion unique tokens seen 10 times. But the model cares — repeated data gives diminishing returns compared to fresh unique data, and heavy repetition actively hurts via overfitting.

Chinchilla says a 70B model needs ~1.4T tokens. But high-quality English text on the internet is maybe a few trillion tokens. For the largest models, labs are running out of data. Responses include synthetic data generation, multilingual data, diverse sources (code, books, transcribed audio), and aggressive data quality curation.

---

## Modern complications

**Q: Do scaling laws still apply to modern architectures?**

Several things have changed:

**Mixture of Experts (MoE):** Original scaling laws assumed dense models where all N parameters are active for every token. MoE architectures route each token through only a subset. DeepSeek-V3 has 671B total parameters but activates only 37B per token. "Model size" now has two meanings — total parameters (storage/memory) vs. active parameters (inference compute).

**Secrecy:** Leading labs no longer publish parameter counts. This is partly competitive and partly because parameter count alone is increasingly uninformative.

**Post-training:** RLHF, instruction tuning, and chain-of-thought training can dramatically change model behavior without changing N or D. Scaling laws describe pre-training loss, not downstream task performance.

---

## The evolution of thinking

1. **Kaplan (2020):** "Scale N aggressively" → huge, undertrained models
2. **Chinchilla (2022):** "Scale N and D equally, D/N ≈ 20" → training-compute optimal
3. **Post-Chinchilla (2023+):** "Over-train smaller models" → total-deployment-cost optimal

The fundamental insight is the same: performance follows power laws with diminishing returns. What changed is the definition of "optimal" — from minimizing training cost to minimizing total cost of ownership including inference.
