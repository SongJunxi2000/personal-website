---
title: "Transformer (5/9) — sqrt(d_k) Scaling & the @V Step"
date: "2026-03-24"
excerpt: "Why dividing by sqrt(d_k) isn't arbitrary — and what it means for a token to 'attend' to another."
project: language
readingTime: 5
---

Two steps finish the attention computation after the dot products: dividing by `sqrt(d_k)` to prevent softmax collapse, then multiplying by V to blend information across tokens. The scaling factor has an elegant statistical justification. The `@ V` step is where attention stops being a similarity measure and starts being actual communication.

---

**Q: Why divide by `sqrt(d_k)`?**

When `d_k` is large, dot products produce large numbers, which makes softmax collapse into a near one-hot distribution:

```
softmax([2.0, 1.0, 1.0])   → [0.50, 0.25, 0.25]   ← spread out
softmax([20,  10,  10])     → [0.99, 0.005, 0.005]  ← almost all on one token
softmax([200, 100, 100])    → [1.00, 0.00, 0.00]    ← completely concentrated
```

If each token only attends to one other token, it can't gather context from multiple relevant tokens simultaneously. Dividing by `sqrt(d_k)` brings values back into a range where softmax produces a smooth distribution.

```
Without scaling (d_k=512):  dot products ≈ [200, 100, 100]
                            softmax → [1.00, 0.00, 0.00]

With scaling (÷√512≈22.6):  scaled ≈ [8.8, 4.4, 4.4]
                            softmax → [0.90, 0.05, 0.05]
```

---

**Q: Why specifically `sqrt(d_k)` and not `d_k^(2/3)` or some other power?**

It's not arbitrary — there's a clean statistical derivation. Assume elements of Q and K are independent random variables with mean 0 and variance 1. The dot product of two vectors of dimension `d_k` is:

```
dot product = q₁k₁ + q₂k₂ + ... + q_dk k_dk

Each term qᵢkᵢ has variance 1 (product of two unit-variance variables).
Summing d_k such terms:

mean of dot product = 0
variance of dot product = d_k
standard deviation = sqrt(d_k)
```

Dividing by `sqrt(d_k)` rescales the variance back to 1:

```
var(dot product / sqrt(d_k)) = d_k / d_k = 1
```

Any other power either over-corrects or under-corrects:

```
divide by d_k^(1/3):  variance = d_k / d_k^(2/3) = d_k^(1/3)  ← still grows
divide by d_k^(2/3):  variance = d_k / d_k^(4/3) = d_k^(-1/3) ← shrinks toward 0
divide by d_k^(1/2):  variance = d_k / d_k = 1                  ← exactly right
```

`sqrt(d_k)` is the exact normalizer that gives unit variance under standard initialization assumptions.

---

**Q: What does softmax do after scaling?**

Softmax normalizes each row of the `(N, N)` score matrix into a probability distribution:

```
Raw scores after scaling:
            dog    bites   man
   dog  [  1.25,   0.80,  -0.20]
   bites[  0.80,   1.09,  -0.40]
   man  [ -0.20,  -0.40,   1.16]

After softmax (each row sums to 1):
            dog    bites   man
   dog  [  0.44,   0.35,   0.21]
   bites[  0.38,   0.41,   0.21]
   man  [  0.19,   0.15,   0.66]
```

Each row is a probability distribution saying "how much should this token attend to each other token."

---

**Q: What is V and how is it different from the token embedding?**

V = X @ W_V — it's a learned projection of X, not the raw embedding. X contains everything about a token. V is a **filtered version** that selects what information to transmit.

Analogy: A person's full resume (X) contains everything — hobbies, address, education. But when responding to your query about databases, they give you a filtered subset (V) — just the relevant database knowledge. W_V is the filter that learns what's worth passing along.

---

**Q: What does the `@ V` step do?**

It computes a **weighted blend** of all tokens' value vectors for each token.

```
V (value vectors for each token):
         [[0.9,  0.1, -0.3,  0.5],   ← v_dog
          [0.2,  0.8,  0.1, -0.4],   ← v_bites
          [0.4, -0.2,  0.7,  0.3]]   ← v_man

For "dog" (using attention weights from row 0):
output_dog = 0.44 × v_dog + 0.35 × v_bites + 0.21 × v_man

= 0.44 × [0.9, 0.1, -0.3, 0.5]
+ 0.35 × [0.2, 0.8,  0.1,-0.4]
+ 0.21 × [0.4,-0.2,  0.7, 0.3]

= [0.550, 0.282, 0.050, 0.143]
```

"Dog" gets 44% of its own information, 35% of "bites"'s information, and 21% of "man"'s information, mixed into a single new vector.

---

**Q: What's the point of this blending?**

Before attention, "dog" only knew about itself. After attention, "dog" has gathered context from the entire sentence. In a real trained model, this is how "it" learns to pull information from "cat" 10 tokens back, or how "sat" learns that its subject is "dog."

Each token's output is a cocktail of information from all the tokens it finds relevant. That's the core idea of attention.

---

**Q: What are the input and output shapes of the full attention operation?**

```
X_final                         → (N, d_model)     raw input
Q, K, V = X @ W_Q, W_K, W_V    → each (N, d_model) three projections
Q @ K^T / sqrt(d_k)            → (N, N)            raw similarity scores
softmax(...)                    → (N, N)            attention weights
weights @ V                     → (N, d_model)      context-aware output
```

Input is `(N, d_model)`, output is `(N, d_model)`. Same shape — but now every token's vector contains information from the tokens it attended to.
