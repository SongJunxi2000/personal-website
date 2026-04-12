---
title: "Transformer Series (1/9) — Token Embeddings & d_model"
date: "2026-03-28"
excerpt: "What is d_model, and how does a lookup table become the foundation of every LLM?"
project: language
readingTime: 4
---

Every transformer begins the same way: a word becomes a number, and that number gets replaced by a vector. This post traces that chain — from vocabulary size to embedding matrix to the single hyperparameter that governs the width of everything that follows.

---

**Q: What is `d_model`?**

`d_model` is the dimensionality of each token's embedding vector — how many numbers represent each token. It's the "width" of the representation that flows through the entire model. In the original Transformer paper, `d_model = 512`. In GPT-3, it's 12288. It's a hyperparameter — the model designer just picks a number before training.

---

**Q: Who defines `d_model` and why?**

It's a design choice, like deciding how wide a highway should be. Bigger means more expressive but more expensive. There's no formula — it's chosen based on compute budget, model size goals, and empirical results.

---

**Q: What are common `d_model` sizes used in the industry?**

The original Transformer (2017) used 512. GPT-2 used 1600. GPT-3 used 12288. LLaMA-7B uses 4096. LLaMA-70B uses 8192. These are almost always powers of 2 or multiples of 256 — that's for GPU hardware efficiency, not any mathematical reason.

---

**Q: What happens if `d_model` is too small?**

Each token's vector can't encode enough nuance. Think of it like describing a person with only 3 adjectives versus 100. "Cat" in "the cat sat" and "cat" in "cat-5 ethernet cable" need to be distinguishable — a tiny vector doesn't have enough room for that.

---

**Q: What happens if `d_model` is too large?**

Two problems. First, cost — parameter count of nearly every component scales with `d_model`. The embedding matrix is `(V × d_model)`, each attention projection is `(d_model × d_model)`, the feedforward layer is `(d_model × 4·d_model)`. Doubling `d_model` roughly quadruples compute in attention layers. Second, if `d_model` is huge relative to training data, the model can memorize rather than generalize.

---

**Q: Is there an optimal `d_model`?**

Not in isolation. Researchers have found it should be balanced against depth (number of layers), number of attention heads, and training data size. The "scaling laws" papers showed that for a fixed compute budget, there's an optimal ratio between model width (`d_model`) and depth. Extremely wide but shallow, or extremely deep but narrow, both underperform.

---

**Q: How is `d_model` related to vocabulary size?**

They're connected through the **embedding matrix**. The vocabulary has `V` tokens (e.g., 50,000). Each token starts as a token ID — just an integer like 8471 for "cat." The embedding matrix `W_embed` has shape `(V, d_model)`, so `(50000, 512)` in the original paper. Each row is one token's learned vector. To get the embedding for "cat" (ID 8471), you grab row 8471 — a vector of size `d_model`.

```
vocabulary size (V) = how many tokens exist
d_model             = how many numbers represent each token
embedding matrix    = (V, d_model) — the bridge between the two
```

They're independent choices. V determines the number of rows; `d_model` determines the number of columns.

---

**Q: Is `d_model` related to BPE at all?**

No. They live in completely separate stages. BPE operates before the model even exists — it's a preprocessing step that determines the vocabulary (V). `d_model` comes after and decides how rich the representation of each token is. The chain:

```
Raw text
  → BPE → token IDs (integers)        # determines V
  → Embedding lookup → vectors         # uses (V, d_model) matrix
  → into the Transformer layers        # everything is d_model-wide
```

BPE doesn't know or care what `d_model` is. `d_model` doesn't know or care how the tokens were created.

---

**Q: What exactly is the token embedding?**

It's just the `(V, d_model)` matrix — a big lookup table. The entire operation is: take a token ID, go to that row of the matrix, get a vector of size `d_model`. No computation, no multiplication, no activation function. Just grab the row.

```
token ID 2 ("cat") → go to row 2 of the (V, d_model) matrix → get a vector of size d_model
```

---

**Q: Is the embedding matrix learned?**

Yes. When training starts, every row is initialized with random numbers — totally meaningless. During training, backpropagation updates these numbers. Over time, the model discovers that tokens with similar meanings should have similar vectors:

```
Before training:  "cat" = [0.02, -0.31, 0.77, ...]   (random)
                  "dog" = [0.85,  0.12, -0.44, ...]   (random)

After training:   "cat" = [0.71, -0.23, 0.55, ...]
                  "dog" = [0.68, -0.19, 0.51, ...]    (close to "cat"!)
```

Nobody tells the model "cat and dog are similar." It figures that out because placing them close together in vector space helps it predict better during training. "Learned" means "updated by gradient descent during training, not hand-designed by a human."

---

**Q: How many parameters are in the embedding matrix?**

Simply `V × d_model`. If `V = 50,000` and `d_model = 512`, that's 25,600,000 parameters — just for the embedding matrix alone, before a single attention or feedforward layer.

---

**Q: Is the embedding matrix reused at the output?**

Often yes. Many architectures reuse the same `(V, d_model)` matrix (transposed) to convert vectors back into token probabilities at the output end. This is called **weight tying** — the same matrix serves double duty. This saves parameters and tends to improve performance.
