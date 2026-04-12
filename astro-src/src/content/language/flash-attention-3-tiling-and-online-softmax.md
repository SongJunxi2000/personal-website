---
title: "FlashAttention Series (3/4) — Tiling and Online Softmax Explained"
date: "2026-03-31"
excerpt: "A step-by-step walkthrough of how FlashAttention eliminates the N×N memory bottleneck using tiling and incremental softmax — with concrete numbers."
project: language
readingTime: 8
---

The previous posts established that attention is memory-bound and that the N×N score matrix is the culprit. FlashAttention's answer: never materialize that matrix at all. Instead, divide Q and K into small blocks and compute the attention output tile by tile, keeping everything in SRAM. This post walks through exactly how.

---

## Block structure and loop order

Q is divided into row-blocks and K into row-blocks (of the same size). V is split at the same token boundaries as K.

The algorithm uses two nested loops:

- **Outer loop**: iterate over Q blocks (rows of the output)
- **Inner loop**: for each Q block, sweep across all K blocks (columns of the score matrix)

The inner loop sweeps across K blocks because softmax is computed **row-wise** — each row of the score matrix becomes a probability distribution over all key tokens. You need to see all K blocks before a row's softmax is finalized. Sweeping in the other direction (Q blocks for a fixed K block) would make incremental softmax much harder because you'd be partially computing many rows without finishing any of them.

---

## What lives in SRAM at any moment

At any point during computation, SRAM holds only:

- One Q block (a few rows of Q)
- One K block (a few rows of K, loaded fresh each inner iteration)
- One V block (corresponding rows of V)
- Running accumulators: max (1 scalar per row), sum (1 scalar per row), partial output (d_k values per row)

Previous K/V blocks are discarded. Scores and weights from previous tiles are also discarded — they've already been folded into the accumulators.

---

## The fused pipeline

For each (Q block, K block) pair, the entire pipeline is fused into one pass:

1. Compute score tile = Q_block @ K_blockᵀ
2. Update running max and running sum for softmax
3. Rescale old partial output if the max changed
4. Compute new softmax weights, multiply by V_block, add to partial output
5. **Discard scores and weights** — they never touch HBM

After processing all K blocks for a given Q block, the output rows are final and get written to HBM exactly once.

---

## The softmax problem

Standard softmax for row *i* is:

```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

You need the sum across all columns. But when you're processing K block 0, you've only seen a fraction of the row. How can you compute softmax without the full row?

---

## Online softmax: the incremental rescaling trick

You maintain a **running max** and **running sum**, correcting earlier results when new values arrive. Here's how it works:

**After K block 0** (say scores = [3, 1, 2, 4]):

- Running max: m₀ = 4
- Running sum: s₀ = exp(3−4) + exp(1−4) + exp(2−4) + exp(4−4) = 1.320
- Compute tentative weights, multiply by V block 0, store as partial output o₀
- **Discard raw scores and weights**

**K block 1 arrives** (scores = [5, 1, 3, 1]). The new local max is 5, which is bigger than the old max of 4.

**Rescaling step:**

```
m_new = max(4, 5) = 5

Correction factor = exp(m_old − m_new) = exp(4 − 5) = 0.368

s₀_corrected = 1.320 × 0.368 = 0.486
o₀_corrected = o₀ × 0.368
```

Then add block 1's contribution (computed with global max = 5), combine, and normalize by the final sum.

---

## Why rescaling is mathematically exact

This isn't an approximation. It relies on a simple identity:

```
exp(x − m_old) × exp(m_old − m_new) = exp(x − m_new)
```

Multiplying the old exponential by the correction factor produces exactly the value you'd have computed if you'd known the global max from the start. The final result is bit-identical to standard softmax.

---

## What persists between blocks

Only three tiny quantities are carried forward, all in SRAM:

| Accumulator | Size | Purpose |
|---|---|---|
| Running max | 1 scalar per row | Numerical stability |
| Running sum | 1 scalar per row | Softmax denominator |
| Partial output | d_k values per row | Accumulated weighted V |

Everything else — raw scores, softmax weights — is ephemeral. Computed, used (multiplied into the partial output), and gone.

---

## Concrete walkthrough

Let's trace the full algorithm with real numbers. Setup: N=8, d_k=4, block size=4 (2 blocks).

**One Q row**: q = [1, 0, 2, 1]

**K matrix** (split into 2 blocks of 4 rows):

```
K block 0:  k₀=[1,1,0,0]  k₁=[0,1,1,0]  k₂=[1,0,1,1]  k₃=[0,0,1,0]
K block 1:  k₄=[2,1,1,1]  k₅=[0,1,0,1]  k₆=[1,1,1,0]  k₇=[0,0,0,1]
```

**V matrix** (same split):

```
V block 0:  v₀=[2,1,0,3]  v₁=[1,0,1,2]  v₂=[0,2,1,1]  v₃=[3,1,0,0]
V block 1:  v₄=[1,3,2,0]  v₅=[0,1,0,2]  v₆=[2,0,1,1]  v₇=[1,0,0,3]
```

### Processing K block 0

**Compute scores** (dot products, all in SRAM):

```
q·k₀ = 1×1 + 0×1 + 2×0 + 1×0 = 1
q·k₁ = 1×0 + 0×1 + 2×1 + 1×0 = 2
q·k₂ = 1×1 + 0×0 + 2×1 + 1×1 = 4
q·k₃ = 1×0 + 0×0 + 2×1 + 1×0 = 2

scores = [1, 2, 4, 2]
```

**Partial softmax**:

```
m₀ = max(1, 2, 4, 2) = 4
s₀ = exp(−3) + exp(−2) + exp(0) + exp(−2)
   = 0.050 + 0.135 + 1.000 + 0.135 = 1.320

weights = [0.038, 0.102, 0.758, 0.102]
```

**Multiply by V block 0** (in SRAM):

```
o₀ = 0.038×[2,1,0,3] + 0.102×[1,0,1,2]
   + 0.758×[0,2,1,1] + 0.102×[3,1,0,0]
   = [0.485, 1.621, 0.860, 1.071]
```

**Discard** scores [1,2,4,2] and weights [0.038, 0.102, 0.758, 0.102]. They never touch HBM. Keep only: m₀=4, s₀=1.320, o₀=[0.485, 1.621, 0.860, 1.071].

### Processing K block 1

**Compute scores**:

```
q·k₄ = 5, q·k₅ = 1, q·k₆ = 3, q·k₇ = 1
scores = [5, 1, 3, 1]
```

**Rescale** (new local max 5 > old max 4):

```
m_new  = 5
factor = exp(4 − 5) = 0.368

s₀_corrected = 1.320 × 0.368 = 0.486
o₀_corrected = [0.485, 1.621, 0.860, 1.071] × 0.368
             = [0.178, 0.596, 0.316, 0.394]
```

**Add block 1's contribution**:

```
s₁ = exp(5−5) + exp(1−5) + exp(3−5) + exp(1−5)
   = 1.000 + 0.018 + 0.135 + 0.018 = 1.171

s_final = 0.486 + 1.171 = 1.657

o₁ = (unnormalized weights) × V block 1
   = [0.779, 1.823, 1.290, 0.115]
```

**Final output**:

```
output = (o₀_corrected + o₁) / s_final
       = ([0.178, 0.596, 0.316, 0.394] + [0.779, 1.823, 1.290, 0.115]) / 1.657
       = [0.578, 1.460, 0.970, 0.307]
```

Written to HBM **once**. Identical to computing standard softmax on the full score vector [1,2,4,2,5,1,3,1] and multiplying by the full V matrix.

---

## I/O comparison

| | Naive attention | FlashAttention |
|---|---|---|
| HBM writes per Q row | scores [1×N] + weights [1×N] + output [1×d_k] | output [1×d_k] only |
| HBM reads per Q row | scores (for softmax) + weights (for V multiply) | Q, K, V blocks (streamed once) |
| Round trips | 3 | 1 (final write) |
| N×N in HBM? | Yes — fully materialized | Never |
| FLOPs | O(N² d_k) | O(N² d_k) + small rescaling overhead |
| Speedup source | — | I/O reduction, not FLOP reduction |

FlashAttention does slightly more arithmetic (the rescaling corrections), but dramatically less memory traffic. Since attention is memory-bound, reducing I/O is what matters. The result is typically a 2–4× wall-clock speedup.
