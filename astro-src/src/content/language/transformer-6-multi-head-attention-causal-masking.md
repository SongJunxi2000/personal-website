---
title: "Transformer Series (6/9) — Multi-Head Attention & Causal Masking"
date: "2026-03-23"
excerpt: "One attention head can only look for one thing at a time. Multi-head attention lets different heads specialize — and causal masking keeps training honest."
project: language
readingTime: 6
---

Single-head attention forces every token to compress all its relational needs into one distribution. Multi-head attention solves this by running several attention operations in parallel, each at a fraction of the cost. Causal masking then enforces the rule that governs generation: you can only see the past.

---

## Multi-Head Attention

**Q: What problem does multi-head attention solve?**

In single-head attention, each token produces one set of attention weights — one distribution over all other tokens. But a token might need to attend to different tokens for different reasons simultaneously:

```
"The cat that was sitting on the mat ate the fish"

For "ate", it needs to simultaneously figure out:
  - who ate → "cat" (subject relationship)
  - what was eaten → "fish" (object relationship)
```

With a single head, there's only one set of weights — it has to compromise between attending to "cat" and "fish" with a single distribution. Multi-head attention runs multiple attention operations in parallel, each learning different relationship types.

---

**Q: Why split `d_model` across heads instead of running full-sized attention multiple times?**

Efficiency. Running 8 full-sized heads with `d_model = 512` would mean 8× the compute and 8× the parameters. By splitting:

```
Single head:   Q, K, V each (N, 512)    → one attention
Multi-head:    Q, K, V each (N, 64)     → per head, 8 heads

Total parameters: roughly the same
Total compute: roughly the same
But now you get 8 different attention patterns
```

---

**Q: How does multi-head attention work step by step?**

With `d_model = 512`, `num_heads = 8`:

```
Step 1: Project Q, K, V as usual
   Q = X @ W_Q    → (N, 512)
   K = X @ W_K    → (N, 512)
   V = X @ W_V    → (N, 512)

Step 2: Split each into 8 heads
   Q → [Q₁, Q₂, ..., Q₈]    each (N, 64)
   K → [K₁, K₂, ..., K₈]    each (N, 64)
   V → [V₁, V₂, ..., V₈]    each (N, 64)

Step 3: Each head runs attention independently
   head₁ = softmax(Q₁ @ K₁^T / sqrt(64)) @ V₁    → (N, 64)
   head₂ = softmax(Q₂ @ K₂^T / sqrt(64)) @ V₂    → (N, 64)
   ...
   head₈ = softmax(Q₈ @ K₈^T / sqrt(64)) @ V₈    → (N, 64)

Step 4: Concatenate all heads
   concat = [head₁ | head₂ | ... | head₈]    → (N, 512)

Step 5: Final projection
   output = concat @ W_O    → (N, 512)
```

Output is back to `(N, d_model)`.

---

**Q: What is W_O?**

The output projection matrix, shape `(d_model, d_model)`. After concatenating heads, the first 64 dimensions came from head 1, next 64 from head 2, etc. The information is segregated by head. W_O lets the model **mix information across heads**.

Head 1 might have found "the subject is cat" and head 2 might have found "the action is ate." W_O combines those findings into a single unified representation. It's another set of learned parameters — the model learns not just what each head should attend to, but how to combine all heads' results.

---

**Q: What does each head learn in practice?**

Different heads specialize in different patterns. Researchers have found examples like:

```
Head 1: attends to the previous token
Head 2: attends to the subject of the sentence
Head 3: attends to tokens with matching syntactic role
Head 5: attends to the beginning of the sentence
```

No one programs this — the heads discover these patterns through training.

---

**Q: What's the tradeoff of multi-head attention?**

Each head sees a smaller slice of the embedding (64 dims instead of 512), but you get 8 independent attention patterns for roughly the same cost as one full-sized attention. In practice this is a huge win.

---

**Q: What is the relationship between `d_k` and `d_model` in multi-head attention?**

`d_k = d_model / num_heads`. With `d_model = 512` and 8 heads, `d_k = 64`. This is why the attention formula uses `d_k` instead of `d_model` — to stay general across both single-head and multi-head settings.

---

## Causal Masking

**Q: What problem does causal masking solve?**

During training, the model processes an entire sequence at once and learns from all positions simultaneously. But each position should only see tokens before it — otherwise it's cheating by looking at the answer.

```
Training on "The cat sat on":
  "The"         → predict "cat"
  "The cat"     → predict "sat"
  "The cat sat" → predict "on"

All trained simultaneously, but each must be honest.
```

---

**Q: How does causal masking work?**

Before softmax, set all "future" positions to negative infinity:

```
Raw scores:
            The    cat    sat    on
   The  [  2.0,   1.5,   0.8,   0.3]
   cat  [  1.2,   1.8,   0.9,   0.4]
   sat  [  0.5,   1.1,   2.1,   0.7]
   on   [  0.3,   0.6,   1.3,   1.9]

After masking (upper triangle → -∞):
            The    cat    sat    on
   The  [  2.0,   -∞,    -∞,    -∞ ]
   cat  [  1.2,   1.8,   -∞,    -∞ ]
   sat  [  0.5,   1.1,   2.1,   -∞ ]
   on   [  0.3,   0.6,   1.3,   1.9]

After softmax (softmax(-∞) = 0):
            The    cat    sat    on
   The  [  1.00,  0.00,  0.00,  0.00]
   cat  [  0.35,  0.65,  0.00,  0.00]
   sat  [  0.14,  0.26,  0.60,  0.00]
   on   [  0.08,  0.11,  0.22,  0.59]
```

Each token can only attend to itself and tokens before it.

---

**Q: How does masking enable training on N examples simultaneously?**

Each row in the masked attention matrix is an independent computation that only uses past tokens. So position 0 genuinely only sees "The", position 1 genuinely only sees "The" and "cat", etc. All positions can be computed in parallel in one forward pass, giving you N-1 training examples from a single sequence. At inference time, you generate tokens sequentially — future tokens genuinely don't exist yet.

---

**Q: Do all models use causal masking?**

Only decoder models (GPT, LLaMA, Claude). Encoder models like BERT use **no masking** — every token sees every other token in both directions. This is because BERT's task (fill in the blank) requires full context, not autoregressive generation.

---

**Q: What is the difference between self-attention and cross-attention?**

The word "self" clarifies where Q, K, V come from:

```
Self-attention:   Q, K, V all come from the SAME source
  Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V

Cross-attention:  Q from one source, K and V from ANOTHER
  Q = X_dec @ W_Q,  K = enc_output @ W_K,  V = enc_output @ W_V
```

In decoder-only models (GPT, Claude), there's only self-attention (with causal masking). Cross-attention exists in encoder-decoder models for tasks like translation.
