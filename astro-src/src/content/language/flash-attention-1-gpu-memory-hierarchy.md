---
title: "FlashAttention Series (1/4) — GPU Memory Hierarchy: HBM, SRAM, and Why It Matters"
date: "2026-03-26"
excerpt: "Understanding the two-level memory system in modern GPUs and why some operations are memory-bound while others are compute-bound."
project: language
readingTime: 4
---

Every optimization in GPU computing comes down to one thing: minimizing the number of round trips between two types of memory. Before understanding FlashAttention, you need to understand the hardware it was designed around.

---

## The two types of memory on a GPU

Modern GPUs have two fundamentally different types of memory. Every matrix multiply, every softmax, every operation in a transformer flows through both of them.

**SRAM (Static RAM)** lives on-chip, right next to the compute cores. On an A100, you get about 20 MB total across all streaming multiprocessors. It's extremely fast (~19 TB/s bandwidth) but tiny. Think of it as a small cutting board right next to your hands.

**HBM (High Bandwidth Memory)** sits off-chip but still on the same package. It's much larger — 40–80 GB on an A100 — but accessing it is roughly 10× slower (~2 TB/s). Think of it as a large fridge you have to walk to.

---

## The mandatory data path

Here's the critical thing most people miss: the GPU's arithmetic units (ALUs) can **only** operate on data that's in SRAM or registers. They cannot reach into HBM directly. There is no shortcut.

```
HBM (big, slow) → SRAM (tiny, fast) → ALUs (compute) → SRAM → HBM
```

When you say "load data into the GPU for computation," you're really saying "load data into SRAM." They're the same thing. SRAM isn't an optional detour — it's the only path to computation.

This means every optimization in GPU computing is about minimizing the number of round trips between HBM and SRAM.

---

## Memory-bound vs compute-bound

The ratio of compute operations to bytes moved is called **arithmetic intensity**. This determines whether an operation is bottlenecked by memory I/O or by arithmetic.

| | Memory-bound | Compute-bound |
|---|---|---|
| Bottleneck | Data movement (HBM ↔ SRAM) | Arithmetic (FLOPs) |
| Arithmetic intensity | Low — few ops per byte loaded | High — many ops per byte loaded |
| Transformer example | Attention (naive) | FFN layers |

Why do attention and FFN behave differently if they both use matrix multiplications?

**FFN layers** load a large weight matrix W₁ once and reuse it across every token in the batch. One load, many FLOPs. High arithmetic intensity → compute-bound.

**Naive attention** generates huge intermediate N×N matrices (the attention scores) that get written to HBM and read back multiple times — once for the score computation, once for softmax, once for the V multiplication. Low compute per byte moved → memory-bound.

The distinction isn't about the type of operation (both are matmuls). It's about the **reuse ratio** — how much compute you extract from each byte before it has to travel back to HBM.

---

## Why this matters

If attention is memory-bound, then making the GPU do arithmetic faster won't help much. The bottleneck is the data movement. To speed up attention, you need to reduce how much data travels between HBM and SRAM.

This is exactly what FlashAttention does — and it starts with understanding why the N×N attention matrix is the root of the problem.
