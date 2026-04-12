---
title: "Transformer Series (4/9) — Scaled Dot-Product Attention"
date: "2026-03-25"
excerpt: "The full attention formula, where Q, K, V come from, and why dot products measure relevance."
project: language
readingTime: 5
---

The full attention formula is five characters: `softmax(Q @ K^T / sqrt(d_k)) @ V`. Each part has a specific motivation. This post traces every step — from input matrix X to the context-enriched output — with concrete numerical examples.

---

**Q: What is the full attention formula?**

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

---

**Q: Where do Q, K, V come from?**

Each is the input X multiplied by a separate learned weight matrix:

```
Q = X @ W_Q    # (N, d_model) @ (d_model, d_model) = (N, d_model)
K = X @ W_K    # same shape
V = X @ W_V    # same shape
```

X is the output of `token_embedding + positional_encoding`. W_Q, W_K, W_V are learned parameters — updated by gradient descent during training.

---

**Q: What do Q, K, V intuitively represent?**

They represent three different "roles" for each token:

```
Q (queries): "what am I looking for?"
K (keys):    "what do I advertise to others?"
V (values):  "what information do I transmit when selected?"
```

Think of a conference: Q is your question, K is someone's name badge, V is the actual knowledge they share once you find them.

---

**Q: Why not just use X directly? Isn't X already the value carried by each token?**

You could do `softmax(X @ X^T) @ X` and it would technically work. But X contains everything about a token packed into one vector. When "dog" attends to "bites," maybe it only needs one specific aspect — like the fact that it's an action verb — not everything about "bites."

The separate projections let the model decouple three roles:
```
W_Q: learns what each token should search for
W_K: learns what each token should advertise
W_V: learns what each token should transmit when selected
```

Using raw X for all three forces one representation to serve three different purposes. Separate weight matrices give the model more flexibility.

---

**Q: Why was Q @ K^T designed this way? Is it just an arbitrary algorithm?**

No — it follows from a clear motivation. The fundamental problem: each token needs to gather information from other tokens. "It" needs to figure out that "it" refers to "cat." So you need a mechanism where each token can ask "which other tokens are relevant to me?"

The simplest way to measure relevance between two vectors is a **dot product** — similar directions = large value, unrelated = small value.

But what a token "looks for" differs from what it "advertises." So instead of comparing raw embeddings, separate projections were introduced. The Q/K/V dot-product approach turned out to be both expressive and fast (matrix multiplication is extremely efficient on GPUs).

Could the authors have designed it differently? Yes — earlier attention mechanisms used learned additive scores instead of dot products. But the dot-product approach won because of efficiency.

---

**Q: What does `Q @ K^T` produce?**

An `(N, N)` matrix where entry `(i, j)` is the dot product between token i's query and token j's key. Higher values mean "token i finds token j more relevant."

```
Q is (N, d_model), K^T is (d_model, N), result is (N, N).

Example for "dog bites man" (with positional encoding):

            dog    bites   man
   dog  [  1.25,   0.80,  -0.20]
   bites[  0.80,   1.09,  -0.40]
   man  [ -0.20,  -0.40,   1.16]
```

Row 0 says: "dog" finds itself most relevant (1.25), "bites" somewhat (0.80), "man" barely (-0.20).

---

**Q: Can you show every matrix step-by-step for "dog bites man"?**

Using `d_model = 4`, identity W_Q and W_K, with positional encoding added:

```
X = [[1.0,  0.5, 0.0, 0.0],    ← dog + pos0
     [0.3,  1.0, 0.0, 0.0],    ← bites + pos1
     [0.0, -0.4, 1.0, 0.0]]    ← man + pos2

Q = X (identity W_Q)
K = X (identity W_K)

K^T = [[1.0,  0.3,  0.0],
       [0.5,  1.0, -0.4],
       [0.0,  0.0,  1.0],
       [0.0,  0.0,  0.0]]

Q @ K^T computed via dot products:

Row 0 = [1.0, 0.5, 0.0, 0.0] (dog)
  dot col 0: (1.0×1.0)+(0.5×0.5)+0+0 = 1.25
  dot col 1: (1.0×0.3)+(0.5×1.0)+0+0 = 0.80
  dot col 2: (1.0×0.0)+(0.5×-0.4)+0+0 = -0.20

Row 1 = [0.3, 1.0, 0.0, 0.0] (bites)
  dot col 0: (0.3×1.0)+(1.0×0.5)+0+0 = 0.80
  dot col 1: (0.3×0.3)+(1.0×1.0)+0+0 = 1.09
  dot col 2: (0.3×0.0)+(1.0×-0.4)+0+0 = -0.40

Row 2 = [0.0, -0.4, 1.0, 0.0] (man)
  dot col 0: (0.0×1.0)+(-0.4×0.5)+0+0 = -0.20
  dot col 1: (0.0×0.3)+(-0.4×1.0)+0+0 = -0.40
  dot col 2: (0.0×0.0)+(-0.4×-0.4)+(1.0×1.0)+0 = 1.16

Result:
            dog    bites   man
   dog  [  1.25,   0.80,  -0.20]
   bites[  0.80,   1.09,  -0.40]
   man  [ -0.20,  -0.40,   1.16]
```

---

**Q: What is `d_k` and is it the same as `d_model`?**

In single-head attention, yes — `d_k = d_model`. But `d_k` exists as a separate term because in multi-head attention, `d_k = d_model / num_heads`. With `d_model = 512` and 8 heads, `d_k = 64`. The formula uses `d_k` to stay general.
