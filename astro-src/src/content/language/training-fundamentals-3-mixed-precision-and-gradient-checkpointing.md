---
title: "Training Fundamentals (3/4) — Mixed Precision Training & Gradient Checkpointing"
date: "2026-04-04"
excerpt: "Two essential memory optimization techniques: reducing the size of stored values (FP16/BF16) and reducing the number of stored values (checkpointing)."
project: language
readingTime: 8
---

Two essential techniques for training large models efficiently. They attack the same core problem — memory and compute constraints — from different angles: mixed precision reduces the **size** of stored values, while gradient checkpointing reduces the **number** of stored values.

---

## Part 1: Floating Point Representation

**Q: How does floating point work?**

Floating point is binary scientific notation. Just as decimal notation writes 5750 as 5.750 × 10³, floating point represents numbers as:

```
(-1)^sign × 1.mantissa × 2^exponent
```

Three components: a **sign bit** (positive/negative), **exponent bits** (controlling range — how large or small), and **mantissa bits** (controlling precision — how many significant digits).

---

**Q: What are FP32, FP16, and BF16?**

| Format | Sign | Exponent | Mantissa | Total Bits |
|--------|------|----------|----------|------------|
| FP32 | 1 | 8 | 23 | 32 |
| FP16 | 1 | 5 | 10 | 16 |
| BF16 | 1 | 8 | 7 | 16 |

BF16 (Brain Floating Point) was designed by Google Brain specifically for deep learning. The key insight: it keeps the same 8 exponent bits as FP32, trading mantissa bits instead.

---

**Q: Can you show a concrete example?**

Representing 5.75 in FP32:

1. Convert to binary: 5 = 101, 0.75 = .11, so 5.75 = 101.11
2. Normalize: 101.11 → 1.0111 × 2²
3. Extract fields: sign = 0, exponent = 2 + 127 (bias) = 129 = 10000001, mantissa = 0111...

```
0  10000001  01110000000000000000000
│  ────┬───  ──────────┬────────────
sign  exp(129)      mantissa
```

In BF16, the exponent is identical (both have 8 bits). Only the mantissa is truncated to 7 bits. For 5.75, no information is lost because the trailing bits were zeros. But for 5.743, BF16's ~2–3 decimal digits of precision can't distinguish it from 5.744.

---

## Part 2: Why Range Matters More Than Precision

**Q: What happens when you lose range vs. precision?**

**Losing precision** (fewer mantissa bits): 1003 rounds to 1000. Slightly off, but training is inherently noisy, so this barely matters.

**Losing range** (fewer exponent bits): a gradient of 100,000 becomes **infinity/NaN**. A tiny gradient of 0.0000001 rounds to **exactly zero**. Both are catastrophic — NaN poisons every computation, and zeroed gradients mean the model stops learning.

| Format | Exponent Bits | Approximate Range |
|--------|--------------|-------------------|
| FP32 | 8 | ~10⁻³⁸ to ~10³⁸ |
| BF16 | 8 | ~10⁻³⁸ to ~10³⁸ |
| FP16 | 5 | ~6×10⁻⁵ to ~65,504 |

BF16 matches FP32's range. FP16's range is dramatically smaller — gradient values regularly underflow to zero during training.

---

**Q: How did people make FP16 work before BF16 existed?**

With **loss scaling**. Before the backward pass, multiply the loss by a large factor (e.g., 1024). By the chain rule, every gradient gets scaled up — tiny gradients that would underflow now land in representable range. After computing gradients, divide by the scale factor to recover true values.

**Dynamic loss scaling** handles the risk of overflow: start with a large scale (e.g., 2¹⁶), check for NaN after backward, halve if overflow detected, occasionally try doubling. The scale hunts for the sweet spot between underflow and overflow.

BF16 doesn't need any of this because its range matches FP32.

---

## Part 3: The Swamping Problem — Why FP32 Is Still Needed

**Q: If BF16 has enough range, why keep FP32 at all?**

Because of the **weight update**. Example: weight = 1000.0, gradient update = 0.001.

In BF16 (~2–3 decimal digits of precision), representable values near 1000 are spaced roughly:

```
... 992, 1000, 1008, 1016 ...
```

1000.0 + 0.001 = 1000.001, but the nearest BF16 value is still **1000**. The update is completely rounded away to nothing. This is **swamping**: a small but meaningful value lost when added to a much larger value.

In FP32 (~7 decimal digits), values near 1000 are spaced ~0.0001 apart. The update survives. Since every weight update adds a small gradient to a relatively large weight, doing updates in half precision causes the model to stop learning entirely.

---

## Part 4: Mixed Precision Training

**Q: How does mixed precision keep the best of both worlds?**

Half-precision for speed, FP32 for the critical update step:

1. **Cast** FP32 master weights → FP16/BF16 (nearly free)
2. **Forward pass** in FP16/BF16 (fast — Tensor Cores do ~2× faster)
3. **Backward pass** in FP16/BF16 (fast, half the memory for activations)
4. **Cast** gradients → FP32 (nearly free)
5. **Update** FP32 master weights using FP32 gradients (precise — no swamping)

What's stored where:

| Component | Precision | Why |
|-----------|-----------|-----|
| Master weights | FP32 | Accumulates small updates — needs precision |
| Optimizer state (m, v) | FP32 | Running averages — same swamping risk |
| Weight copy for forward/backward | FP16/BF16 | Fast matrix multiplications |
| Activations | FP16/BF16 | Intermediate values during forward pass |
| Gradients | FP16/BF16 | Computed during backward pass |

The real memory savings come from **activations**, which scale with batch_size × sequence_length × hidden_dim × number_of_layers and easily dwarf parameter memory.

---

## Part 5: Gradient Checkpointing

**Q: What's the activation memory problem?**

During the forward pass, backpropagation requires storing the intermediate output of every operation at every layer. For a single Transformer layer, that's roughly ~12 tensors (layer norm outputs, Q/K/V projections, attention scores, softmax weights, FFN intermediates, residual outputs).

For a 96-layer Transformer, you're storing intermediates at every layer. Activations are typically the single largest memory consumer during training.

---

**Q: What's the core idea of gradient checkpointing?**

Don't store all activations. Throw most away during the forward pass, and **recompute them during the backward pass** when needed.

Keep only a few strategically chosen activations called **checkpoints**. To regenerate any discarded intermediate, re-run the forward pass from the nearest checkpoint.

---

**Q: Can you give a concrete example?**

24 layers, 4 checkpoints at layers 6, 12, 18, and 24:

**Without checkpointing:** ~12 intermediates × 24 layers = ~288 tensors stored.

**With checkpointing:** 4 checkpoint tensors stored permanently. During backward pass, recompute one 6-layer segment at a time (~72 intermediates at peak).

---

**Q: What's the compute cost?**

One additional forward pass, regardless of checkpoint count. Every layer gets recomputed exactly once during the backward pass. The backward pass is ~2× the cost of a forward pass, so total goes from ~3× (1 forward + 2 backward) to ~4× — roughly a **33% increase in total compute**.

The mathematically optimal number of checkpoints is **√N** for N layers (Chen et al., 2016). Activation memory drops from O(N) to O(√N).

---

**Q: How do mixed precision and gradient checkpointing work together?**

They're multiplicative. Mixed precision reduces the **size** of each tensor (FP16 = 2 bytes vs FP32 = 4 bytes). Gradient checkpointing reduces the **count** of stored tensors (O(√N) instead of O(N)). Fewer tensors × smaller tensors — this combination is what makes training very large models possible on limited GPU memory.

| Technique | Saves Memory By | Costs |
|-----------|----------------|-------|
| Mixed precision | Halving activation/gradient size | Loss scaling complexity (FP16 only); ~no compute cost |
| Gradient checkpointing | Storing O(√N) instead of O(N) activations | ~33% more compute (one extra forward pass) |
