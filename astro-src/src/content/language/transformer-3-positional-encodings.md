---
title: "Transformer Series (3/9) — Positional Encodings"
date: "2026-03-26"
excerpt: "Attention is permutation-invariant. Without a positional signal, 'dog bites man' and 'man bites dog' look identical."
project: language
readingTime: 6
---

Attention computes dot products between all pairs of tokens. The operation doesn't care about order — it's purely about similarity. This is powerful, but it means the model has no idea whether "dog" came first or last. Positional encodings are the fix: a position-specific vector added to each token's embedding before anything else happens.

---

**Q: Why do we need positional encodings?**

Attention computes `Q @ K^T` — dot products between all pairs of tokens. This operation is **permutation invariant**, meaning it doesn't care about token order. Without positional information:

```
"dog bites man"  and  "man bites dog"
```

would produce the exact same internal representations, even though they mean completely different things. The model has no idea what order the tokens are in.

---

**Q: How is position information added?**

Before feeding X into the Transformer, we add a positional signal to each token's embedding:

```
X_final = token_embedding + positional_encoding
```

Both have shape `(N, d_model)`. You're literally adding a position-specific vector to each token's vector, element-wise. Here `token_embedding` is not the big lookup table itself — it's the result of looking up your specific sentence's tokens in that table.

---

**Q: Can you show concrete dimensions and values?**

Using `d_model = 4` and "The cat sat":

```
token_embedding (looked up from the big table):
         dim0   dim1   dim2   dim3
"The"  [ 0.2,   0.5,  -0.1,   0.8]
"cat"  [ 0.7,  -0.3,   0.6,   0.1]
"sat"  [-0.4,   0.9,   0.2,  -0.5]

positional_encoding (from sinusoidal formula):
           dim0    dim1    dim2    dim3
pos 0:  [ 0.000,  1.000,  0.000,  1.000]
pos 1:  [ 0.841,  0.540,  0.010,  0.999]
pos 2:  [ 0.909, -0.416,  0.020,  0.999]

X_final = token_embedding + positional_encoding:
               dim0    dim1    dim2    dim3
"The"@pos0: [ 0.200,  1.500, -0.100,  1.800]
"cat"@pos1: [ 1.541,  0.240,  0.610,  1.099]
"sat"@pos2: [ 0.509,  0.484,  0.220,  0.499]
```

Now "cat" at position 1 has a different vector than "cat" would have at position 0.

---

**Q: Why not use simple incrementing numbers like 0.0001, 0.0002, etc.?**

Two problems. First, every dimension has the same value, so you're wasting dimensions — they carry no additional information. Second, the differences are tiny (0.0001) compared to token embedding values like 0.7 or -0.3. The positional signal gets drowned out. And if you make the numbers bigger, they overwhelm the token meaning instead. You need values that stay in a reasonable range — between -1 and 1 — so they don't dominate or get lost.

---

**Q: Why not use random numbers between -1 and 1 with slight increments per position?**

Like this:
```
pos 0: [0.12, -0.45,  0.78, -0.23]   ← random
pos 1: [0.14, -0.43,  0.80, -0.21]   ← slightly shifted
pos 2: [0.16, -0.41,  0.82, -0.19]   ← slightly shifted again
```

This works to some extent — each position is unique. But the difference between any two adjacent positions is always the same constant vector. This makes it harder for the model to distinguish "5 positions apart" vs "50 positions apart" because differences just scale linearly. The sinusoidal approach uses multiple frequencies, allowing relative positions to be captured as linear transformations — a mathematical property specific to sin/cos.

That said, the original authors tested learned positional embeddings (random vectors that get trained) and found nearly identical performance. So the simple approach isn't broken — sin/cos just gives a clean solution with nice mathematical properties.

---

**Q: How do the sinusoidal positional encodings work?**

Each dimension oscillates at a different frequency:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

With `d_model = 4`:
```
dimension 0: sin(pos / 1)      ← fast wave
dimension 1: cos(pos / 1)      ← fast wave
dimension 2: sin(pos / 100)    ← slow wave
dimension 3: cos(pos / 100)    ← slow wave

         dim0(fast)  dim1(fast)  dim2(slow)  dim3(slow)
pos 0: [ 0.000,      1.000,      0.000,      1.000]
pos 1: [ 0.841,      0.540,      0.010,      0.999]
pos 2: [ 0.909,     -0.416,      0.020,      0.999]
pos 3: [ 0.141,     -0.990,      0.030,      0.999]
```

Low dimensions change fast (like a clock's second hand), high dimensions change slowly (like the hour hand). The combination gives each position a unique fingerprint. And sin/cos values always stay between -1 and 1.

---

**Q: Why does the multi-frequency approach help with relative distances?**

Consider "The cat that I saw yesterday at the park sat on the mat." The word "sat" needs to attend to "cat" (its subject) 9 positions back. In another sentence, the subject might be 2 positions back. The model needs to learn patterns like "look N positions back." The multi-frequency approach makes this easier because the relationship between any two positions can be captured as a simple linear transformation of their sin/cos values.

---

**Q: Is the positional encoding learned or fixed?**

In the original Transformer — **fixed**. Computed once using sinusoidal formulas, never updated during training. Some later models (like GPT-2) switched to **learned positional embeddings** — a second lookup table of shape `(max_sequence_length, d_model)` trained by backpropagation. Modern models like GPT and LLaMA use **RoPE** (Rotary Position Embeddings), a different approach entirely.

```
token_embedding:      always learned
positional_encoding:  fixed (original), learned (GPT-2), or RoPE (modern)
```

---

**Q: Can you prove that without positional encoding, position doesn't matter?**

Yes. Let P be a permutation matrix that reorders rows. For permuted input PX:

```
Q' = PX W_Q = PQ,   K' = PK,   V' = PV

Q'K'^T = PQ(PK)^T = PQK^T P^T

softmax(PQK^T P^T) = P softmax(QK^T) P^T    (softmax is row-wise)

Attention(PX) = P softmax(QK^T) P^T @ PV
              = P softmax(QK^T) (P^T P) V
              = P softmax(QK^T) V             (since P^T P = I)
              = P Attention(X)
```

Result: `Attention(PX) = P × Attention(X)`. The model computes the identical representation for each token regardless of position — outputs just get reordered. This is **permutation equivariance**. Adding positional encoding breaks this because `P(X + PE) ≠ PX + PE`.

---

**Q: Can you show Q @ K^T with and without positional encoding for "dog bites man" vs "man bites dog"?**

Without positional encoding — identical:
```
"dog bites man":              "man bites dog":
            dog  bites  man               man  bites  dog
   dog  [  1,    0,     0]      man  [  1,    0,     0]
   bites[  0,    1,     0]      bites[  0,    1,     0]
   man  [  0,    0,     1]      dog  [  0,    0,     1]
```

With positional encoding — different:
```
"dog bites man":              "man bites dog":
            dog  bites  man               man  bites  dog
   dog  [ 1.25, 0.80, -0.20]    man  [ 1.25, 0.50, -0.20]
   bites[ 0.80, 1.09, -0.40]    bites[ 0.50, 1.09, -0.10]
   man  [-0.20,-0.40,  1.16]    dog  [-0.20,-0.10,  1.16]
```

"bites" attends to the first word with score 0.80 in one sentence but 0.50 in the other. The model can now distinguish word order.
