---
title: "Transformer Series (9/9) — Output Head & Training Essentials"
date: "2026-03-23"
excerpt: "How a hidden state becomes a probability distribution, and what cross-entropy loss, teacher forcing, and learning rate warmup actually do."
project: language
readingTime: 8
---

The last step of the transformer — converting a `d_model`-dimensional vector into a probability over 50,000 vocabulary entries — turns out to reuse the same matrix that started the whole pipeline. This post covers the output head, the training mechanics that make the whole thing work, and a complete end-to-end pipeline summary.

---

## Output Head

**Q: What is the output head?**

The final step that converts the model's internal representation into a probability distribution over the vocabulary. After all Transformer blocks, you have `final_hidden_state` of shape `(N, d_model)`. You need a probability for every token in the vocabulary.

```
logits = final_hidden_state @ W_embed^T    → (N, V)
probs = softmax(logits)                    → (N, V)
```

---

**Q: Is W_embed the same embedding matrix from the input?**

Yes — the same `(V, d_model)` lookup table used to convert token IDs into vectors at the input. The output head uses its transpose `(d_model, V)`. This is **weight tying**.

```
At input:   token ID → grab row from W_embed → vector of size d_model
At output:  hidden state → dot product with ALL rows of W_embed → score per vocab token
```

Not all models do this. Some use a separate output matrix `W_out` with its own learned parameters — adds `(d_model × V)` more parameters but gives more flexibility.

---

**Q: Why does weight tying make sense intuitively?**

The embedding matrix maps each token to a "meaning vector." The output head computes a dot product between the hidden state and every token's meaning vector. If the hidden state is close to the meaning vector of "cat," that dot product is large, and "cat" gets high probability.

```
Embedding (input):   token ID → meaning vector
Output head:         meaning vector → token ID
Same matrix, opposite directions.
```

---

**Q: What shape are the logits?**

`(N, V)` — one row per token position, one column per vocabulary entry. Softmax is applied **row-wise**, so each row independently becomes a probability distribution:

```
logits: shape (4, 50000) for "The cat sat on"

                    token0  token1  ... token49999
"The"  (pos 0):  [  2.1,   -0.5,  ...   1.8    ]
"cat"  (pos 1):  [  0.4,    1.9,  ...   0.1    ]
"sat"  (pos 2):  [  1.1,    0.3,  ...  -0.7    ]
"on"   (pos 3):  [ -0.3,    0.8,  ...   1.2    ]

After softmax (each row sums to 1.0):
"The"  (pos 0):  [  0.12,   0.01, ...   0.09   ]
"cat"  (pos 1):  [  0.03,   0.15, ...   0.01   ]
"sat"  (pos 2):  [  0.05,   0.02, ...   0.01   ]
"on"   (pos 3):  [  0.01,   0.04, ...   0.06   ]
```

Each row is an independent prediction for what comes after that position.

---

**Q: So you're training on N data sets at the same time?**

Yes. One forward pass through N tokens gives you N-1 training examples simultaneously:

```
"The cat sat on"  (4 tokens)

Example 1: "The"             → predict "cat"     (row 0)
Example 2: "The cat"         → predict "sat"     (row 1)
Example 3: "The cat sat"     → predict "on"      (row 2)

3 training examples from 1 forward pass.
```

Causal masking ensures each is honest. In practice with batches: a batch of 4 sequences of 1024 tokens = 4 × 1023 ≈ 4092 training examples per forward pass.

---

**Q: During inference, which row do you use?**

Only the **last row** — that's the prediction for the next token that doesn't exist yet:

```
Input: "The cat sat on"
→ look at row 3 (the "on" position)
→ probs = [0.01, 0.04, 0.02, ..., 0.06]
→ sample next token → "the"
→ feed "The cat sat on the" and repeat
```

---

**Q: How is the next token selected from the probability distribution?**

Several strategies:

```
Greedy:      pick highest probability token
Sampling:    randomly sample from the distribution
Top-k:       sample from the top k most likely tokens
Temperature: scale logits before softmax to control randomness
```

---

## Training Essentials

**Q: What is cross-entropy loss?**

The measure of "how wrong was my prediction." After softmax, you have probabilities over 50,000 tokens. The target is one correct token. The loss is:

```
loss = -log(probability assigned to the correct token)
```

Concrete examples:
```
Target: "cat"

Model assigns 0.9 to "cat":   loss = -log(0.9) = 0.105   ← small, good
Model assigns 0.1 to "cat":   loss = -log(0.1) = 2.302   ← large, bad
Model assigns 0.001 to "cat": loss = -log(0.001) = 6.908 ← huge, terrible
```

If the model is confident about the right answer, loss is small. If it puts almost no probability on the right answer, loss is enormous. The log makes the penalty grow sharply as probability approaches zero.

For a full sequence, average across positions:
```
"The cat sat on"
Position 0: predicted "cat" with prob 0.6  → loss = 0.51
Position 1: predicted "sat" with prob 0.3  → loss = 1.20
Position 2: predicted "on"  with prob 0.1  → loss = 2.30

Total loss = (0.51 + 1.20 + 2.30) / 3 = 1.34
```

Backpropagation computes gradients of this loss with respect to every parameter and updates them.

---

**Q: What is teacher forcing?**

During training, always feeding the **correct previous tokens** regardless of what the model predicted.

```
Inference (autoregressive):
  Input: "The"        → predicts "dog" (wrong!)
  Input: "The dog"    → predicts based on wrong "dog"
  Errors compound.

Teacher forcing (training):
  Input: "The"        → predicts "dog" (wrong!)
  Input: "The cat"    → fed correct "cat", not model's "dog"
  Input: "The cat sat"→ fed correct "sat"
```

Even when the model predicts wrong, the next step gets ground truth as input. This keeps training stable and enables parallel training — all positions computed simultaneously because inputs are known in advance.

---

**Q: What is learning rate warmup and scheduling?**

The learning rate controls how aggressively parameters are updated. The original Transformer used:

```
Phase 1 — Warmup (first ~4000 steps):
  Learning rate increases linearly from near 0 to peak

  Step 0:    lr ≈ 0
  Step 4000: lr = 0.001      ← peak

Phase 2 — Decay (after warmup):
  Learning rate decreases proportional to 1/sqrt(step)

  Step 4000:  lr = 0.001
  Step 10000: lr = 0.00063
  Step 40000: lr = 0.00032
```

---

**Q: Why warmup?**

At training start, all parameters are random. Gradients are noisy and unreliable. A large learning rate would cause wild parameter swings. Starting small lets the model find a reasonable region of parameter space before taking bigger steps.

---

**Q: Why decay?**

As training progresses, the model is closer to optimal. You want smaller, more precise updates — like taking big steps when far from your destination but small careful steps when close.

---

## Full Pipeline Summary

```
Raw text
  → BPE tokenizer → token IDs
  → Embedding lookup → (N, d_model)
  → + Positional encoding → X_final (N, d_model)
  → Transformer Block ×96:
      → Multi-head self-attention (with causal mask)
      → Residual + LayerNorm
      → FFN
      → Residual + LayerNorm
  → Output head: final_hidden @ W_embed^T → (N, V)
  → Softmax → probabilities
  → Sample next token
  → Repeat
```
