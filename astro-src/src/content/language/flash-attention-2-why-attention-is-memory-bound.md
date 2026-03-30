---
title: "FlashAttention Series (2/4) — Why Attention Is Memory-Bound: The N×N Problem"
date: "2026-03-26"
excerpt: "The attention score matrix is always N×N per head regardless of d_k, and naive attention writes it to HBM three times. Here's why."
project: language
readingTime: 5
---

The previous post established that attention is memory-bound — bottlenecked by data movement, not arithmetic. This post digs into exactly why: the N×N attention matrix is unavoidably large, and naive attention shuffles it between HBM and SRAM three separate times.

---

## The N×N attention matrix never shrinks

In multi-head attention, d_model is split across heads: d_k = d_model / num_heads. So with d_model=512 and 8 heads, each head works with d_k=64. Q and K per head are both [N × d_k].

But look at what happens in the matmul:

```
Q @ Kᵀ = [N × d_k] @ [d_k × N] = [N × N]
```

The inner dimension d_k cancels out. Whether d_k is 16, 64, or 512, the score matrix is always **N × N**. The number of heads changes how "rich" each dot product is (comparing vectors of length 16 vs 64), but the number of pairwise comparisons is always N² — every token attends to every other token.

More heads doesn't mean smaller matrices. It means **more** N×N matrices. With 32 heads, you have 32 independent N×N matrices — each representing different learned attention patterns. Head 1 might attend to adjacent tokens while head 7 tracks syntactic dependencies. They can't be merged into one matmul without destroying this expressiveness.

In practice, all heads are computed in parallel via a single batched matmul by reshaping:

```
[batch, N, d_model] → [batch, num_heads, N, d_k]

Q @ Kᵀ: [batch, 32, N, d_k] @ [batch, 32, d_k, N] → [batch, 32, N, N]
```

The GPU treats the batch and heads dimensions as independent lanes. It's one kernel launch, but the output tensor is still [batch, 32, N, N] — 32 separate N×N matrices stacked together.

---

## How big is this?

With N=4096, 32 heads, fp16 (2 bytes per element), batch=1:

```
32 × 4096 × 4096 × 2 bytes ≈ 1 GB
```

The A100's total on-chip SRAM is ~20 MB. Even a single head's N×N matrix at N=4096 is 32 MB — already too big. The GPU has no choice but to store these matrices in HBM.

---

## Naive attention: three round trips to HBM

Here's the data flow when the N×N matrix doesn't fit in SRAM:

1. **Compute Q @ Kᵀ**: The GPU streams small chunks of Q and K from HBM into SRAM, computes partial dot products, and **writes the resulting scores back to HBM** piece by piece because the output won't fit in SRAM.

2. **Apply softmax**: Read the entire N×N score matrix **back from HBM** into SRAM in chunks, compute softmax, **write the N×N weight matrix back to HBM**.

3. **Multiply by V**: Read the N×N weight matrix **back from HBM** yet again, multiply by V, write the output to HBM.

That's three round trips for the same N×N data. The arithmetic at each step is fast — it's the repeated loading and storing of gigabytes of intermediate data that kills performance.

---

## The real waste

Notice what's wasteful: the scores and softmax weights are **intermediate results**. You don't need them in the final output — you only need them to compute the weighted sum of V. But because they're too large for SRAM, they get materialized in HBM as a staging area.

What if you could avoid ever writing those intermediates to HBM? What if you could compute the scores, apply softmax, and multiply by V all in one pass — keeping everything in SRAM?

That's exactly what FlashAttention does.
