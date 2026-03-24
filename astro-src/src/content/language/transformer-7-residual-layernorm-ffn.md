---
title: "Transformer Series (7/9) — Residual Connections, Layer Norm & FFN"
date: "2026-03-23"
excerpt: "The three components that wrap every attention block — and why without them, deep networks couldn't be trained at all."
project: language
readingTime: 6
---

Attention is only half the story inside a transformer block. The other half is infrastructure: residual connections that let gradients flow through 96 layers, layer normalization that prevents values from drifting, and a feedforward network that does the actual per-token computation. Each piece has a clear reason to exist.

---

## Residual Connections

**Q: What is a residual connection?**

After multi-head attention produces its output, you **add the input back**:

```
output = X + MultiHeadAttention(X)
```

That's it — a simple element-wise addition.

---

**Q: What are X and MultiHeadAttention(X) in this formula?**

**X** is the input to this layer — shape `(N, d_model)`. If it's the first layer, this is `X_final` (token embeddings + positional encodings). If it's a deeper layer, it's the output of the previous layer.

**MultiHeadAttention(X)** is the result of the full attention mechanism — Q/K/V projections, split into heads, attention computation, concatenate, multiply by W_O. Also shape `(N, d_model)`.

The addition means:
```
X                        = what each token already knows
MultiHeadAttention(X)    = new context gathered from other tokens
X + MultiHeadAttention(X)= original knowledge + new context
```

Attention can only *add* to what a token knows, not overwrite it.

---

**Q: Why do we need residual connections?**

Without them, stacking many layers (GPT-3 has 96) means data passes through 96 sequential transformations. During backpropagation, gradients must flow backward through all of them. They either **explode** (grow exponentially) or **vanish** (shrink to near zero). Either way, early layers can't learn.

The residual connection creates a **shortcut path** — gradients can flow directly through the addition, skipping the attention computation:

```
Without residual:  X → Attn → Attn → Attn → ...
                   Gradients must survive 96 transformations

With residual:     X → X + Attn(X) → X' + Attn(X') → ...
                   Gradients have a direct highway through additions
```

---

## Layer Normalization

**Q: What is layer normalization?**

After the residual connection, layer norm stabilizes the values by normalizing each token's vector independently:

```
For each token's vector x = [x₁, x₂, ..., x_dmodel]:

1. Compute mean:     μ = (x₁ + x₂ + ... + x_dmodel) / d_model
2. Compute variance: σ² = average of (xᵢ - μ)²
3. Normalize:        x̂ᵢ = (xᵢ - μ) / sqrt(σ² + ε)
4. Scale and shift:  outᵢ = γ · x̂ᵢ + β
```

---

**Q: Can you show a concrete example?**

```
x = [4.0, 2.0, 0.0, -2.0]

μ = (4 + 2 + 0 + -2) / 4 = 1.0
σ² = (3² + 1² + 1² + 3²) / 4 = 5.0
sqrt(σ²) ≈ 2.24

x̂ = [(4-1)/2.24, (2-1)/2.24, (0-1)/2.24, (-2-1)/2.24]
   = [1.34, 0.45, -0.45, -1.34]
```

After normalization, values are centered around 0 with roughly unit variance.

---

**Q: Why does layer norm exist?**

As data flows through many layers, the scale of vectors can drift — some dimensions grow huge, others shrink. This makes training unstable because gradients become erratic. Layer norm keeps values in a nice, stable range.

---

**Q: What are γ and β?**

Learned parameters (vectors of size `d_model`). They let the model undo the normalization if it wants to. This sounds counterintuitive — why normalize then un-normalize? The key is the model gets to *choose* the scale and shift that works best, rather than having the scale drift unpredictably.

---

## Feedforward Network (FFN)

**Q: What is the FFN?**

Two linear transformations with an activation function in between:

```
FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂
```

With `d_model = 512`:
```
W₁: (512, 2048)    ← expand to 4× the size
b₁: (2048,)
W₂: (2048, 512)    ← compress back down
b₂: (512,)

x            → (N, 512)
x @ W₁ + b₁  → (N, 2048)   ← expand
ReLU          → (N, 2048)   ← zero out negatives
x @ W₂ + b₂  → (N, 512)    ← compress back
```

---

**Q: What's the intuition behind the FFN?**

Attention's job is to **move information between tokens** — letting "dog" gather context from "bites" and "man." But attention doesn't do much thinking about the information. It's mostly weighted averaging.

The FFN is where **per-token computation** happens:

```
Attention: "Let me gather relevant information from other tokens"
FFN:       "Now let me process and think about what I gathered"
```

---

**Q: Why expand then compress (512 → 2048 → 512)?**

The wider hidden layer gives the network a larger space to do computation in before squeezing back down. Research has shown these wider hidden layers are where the model stores **factual knowledge** — like "Paris is the capital of France" is encoded in the FFN weights.

The 4× multiplier is a design choice from the original paper — not magical. Some models use 8/3× or other ratios.

---

## Complete Transformer Block

**Q: How do all these pieces compose into one block?**

```
X
→ MultiHeadAttention(X)
→ X + MultiHeadAttention(X)         ← residual
→ LayerNorm(...)                     ← normalize
→ FFN(...)
→ LayerNorm output + FFN(output)     ← another residual
→ LayerNorm(...)                     ← normalize again
→ final output of ONE block
```

Notice there's a **second residual connection and layer norm** around the FFN — same pattern as around attention. Every sub-component gets wrapped in residual + layer norm.

---

**Q: How is a full Transformer built from these blocks?**

```
X → Block₁ → Block₂ → Block₃ → ... → Block₉₆ → output
```

GPT-3 has 96 blocks. Each block has the same structure but its own learned parameters (W_Q, W_K, W_V, W_O, W₁, W₂, γ, β).
