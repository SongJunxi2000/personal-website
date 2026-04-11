---
title: "Inference Optimization — KV Cache, Quantization, and Batching"
date: "2026-04-09"
excerpt: "Why autoregressive generation is expensive, and the three core techniques that make serving LLMs practical: caching key-value pairs, reducing precision, and amortizing weight loads across users."
project: language
readingTime: 10
---

Training a language model is expensive, but it happens once. Inference — actually generating text — happens millions of times per day and must be fast. This post covers the three core optimizations that make LLM serving practical: KV caching, quantization, and batching.

---

## Why inference is expensive

**Q: Why is autoregressive generation inherently sequential?**

"Autoregressive" means each output token is conditioned on all previous tokens — the model's own prior outputs become its next inputs. During training, the entire sequence is processed in parallel (all positions attend simultaneously via the full attention matrix). During inference, you get one forward pass **per output token**, sequentially. Generating 500 tokens means ~500 serial forward passes.

This sequential dependency is fundamental and cannot be eliminated — token 502 genuinely depends on token 501.

---

**Q: Are there alternatives to autoregressive inference?**

| Approach | How it works | Tradeoff |
|----------|-------------|----------|
| Masked/bidirectional (BERT) | Fills in blanks, not left-to-right generation | Not suitable for open-ended generation |
| Diffusion models for text | Generate all tokens simultaneously, iteratively refine | Research area, not yet dominant |
| Non-autoregressive (NAR) | Generate all output tokens in one shot | Much faster, but significantly lower quality |

Autoregressive remains dominant because the sequential token-by-token dependency is what produces coherent text.

---

**Q: Does backpropagation happen during inference?**

No. Backpropagation is purely a training concept — it computes gradients to update weights. During inference, weights are frozen and only forward passes occur.

---

## Inference forward pass: step by step

**Q: What actually happens in one forward pass during generation?**

Using a reference model: d_model = 4096, n_heads = 32, d_head = 128, n_layers = 32, vocab_size = 50,000.

**Step 1 — Embedding lookup.** The embedding matrix E has shape (50,000 × 4096). For a 10-token prompt, this produces a (10 × 4096) tensor.

**Step 2 — Attention (per layer, per head).** Three learned weight matrices per head:

- **W_Q** (4096 × 128) → Query: "what am I looking for?"
- **W_K** (4096 × 128) → Key: "what do I contain?"
- **W_V** (4096 × 128) → Value: "what do I give you if you attend to me?"

Projections:

$$Q = X \cdot W_Q \rightarrow (10 \times 128)$$
$$K = X \cdot W_K \rightarrow (10 \times 128)$$
$$V = X \cdot W_V \rightarrow (10 \times 128)$$

Attention computation:

$$\text{Attention} = \text{softmax}\!\left(\frac{Q \cdot K^T}{\sqrt{128}}\right) \rightarrow (10 \times 10)$$
$$\text{Output} = \text{Attention} \cdot V \rightarrow (10 \times 128)$$

All 32 heads are concatenated → (10 × 4096), then multiplied by the output projection W_O (4096 × 4096) → (10 × 4096).

**Step 3 — Feed-forward network (per layer).** Two weight matrices: W₁ (4096 × 16384) expands, W₂ (16384 × 4096) compresses back down.

**Step 4 — Repeat steps 2–3 for all 32 layers.** Each layer takes (10 × 4096) in and produces (10 × 4096) out.

**Step 5 — Final prediction.** The last token's vector (1 × 4096) is multiplied by the unembedding matrix (4096 × 50,000) to get logits, then softmax and sample one token.

---

## Optimization 1: KV cache

**Q: What's the redundant computation problem in autoregressive generation?**

Generating token 502 naively recomputes K and V for tokens 1–501 across all layers. But K and V for tokens 1–500 were already computed when generating token 501. They're deterministic functions of those tokens and the frozen weights — they never change.

---

**Q: Why can't old tokens' K and V values change when new tokens are added?**

Because of the **causal mask**. Token 5 can only attend to tokens 1–5. Adding token 11 to the sequence cannot change token 5's output — old tokens never see future tokens, so their K and V vectors are frozen forever once computed.

---

**Q: How does the KV cache work?**

Store K and V vectors for every previously processed token. Each generation step:

1. **Compute** q, k, v for the one new token — (1 × 128) each per head
2. **Append** new k and v to the cache
3. **Attend**: new q × full K cache → (1 × seq_len) attention scores
4. **Aggregate**: attention weights × full V cache → (1 × 128) output
5. **Discard** q (never needed again — only useful at the moment of generation)

No full (seq_len × seq_len) attention matrix is needed. Only one row: (1 × seq_len).

---

**Q: Why cache K and V but not Q?**

**Q (Query)** is only used at the moment a token is generated — the new token asks "what should I attend to?" Once answered, Q is discarded. **K (Key)** must persist because future tokens need to compare their queries against all previous keys. **V (Value)** must persist because future tokens need to retrieve content from attended positions.

---

**Q: How much memory does the KV cache use?**

Per layer: 32 heads × 2 (K and V) × seq_len × 128 = 2 × seq_len × 4096.

Total across 32 layers:

$$32 \times 2 \times \text{seq\_len} \times 4096$$

For 500 tokens: 32 × 2 × 500 × 4096 = **131 million values ≈ 250 MB** in float16.

Each new token adds: 32 layers × 2 × 4096 = 262,144 values ≈ **0.5 MB**.

This grows linearly with sequence length — at 100K tokens, the KV cache alone becomes enormous.

---

**Q: Is the attention score matrix cached too?**

No. The attention score matrix (seq_len × seq_len) is a temporary computation — it's computed, used for the weighted sum over V, then discarded. The next step only needs a (1 × seq_len) vector, not the full matrix.

---

## Optimization 2: Quantization

**Q: Why is inference memory-bound rather than compute-bound?**

The GPU spends most of its time waiting for data (weights) to transfer from HBM to compute units, not doing arithmetic. This makes inference **memory-bandwidth-bound**.

---

**Q: How does quantization help?**

Quantization represents each parameter with fewer bits — same architecture, same parameter count, less precision per value.

| Precision | Bits | 70B model size |
|-----------|------|---------------|
| float16 | 16 | 140 GB |
| int8 | 8 | 70 GB |
| int4 | 4 | 35 GB |

A weight stored as 0.0372841 in float16 might become ~0.04 in 4-bit. Small precision loss, massive memory savings.

Fewer bits per weight → less data to transfer from memory → faster weight loading → faster matrix multiplications. The bandwidth savings matter more than the cheaper arithmetic.

---

**Q: Why is precision loss acceptable at inference but not training?**

During **training**, precise gradients are needed for tiny, careful weight updates — precision matters. During **inference**, weights are frozen and the model is just multiplying — small rounding errors average out across billions of operations.

---

**Q: How much quality do you lose from quantization?**

Rule of thumb: 4-bit quantization costs roughly one "size class." A 70B model at 4-bit performs roughly like a 50–60B model at float16 — still well above a 30B model at float16.

Aggressive quantization (2-bit, 1-bit) degrades quality significantly.

---

**Q: Can the KV cache itself be quantized?**

Yes. Storing K and V in int8 instead of float16 halves the cache memory, allowing more concurrent users or longer contexts within the same memory budget.

---

## Optimization 3: Batching

**Q: Why is serving one user at a time wasteful?**

Without batching, each generation step loads all 70B weights from memory just to multiply against a single (1 × 4096) vector. That's a terrible ratio of memory transfer to computation.

---

**Q: How does batching fix this?**

Stack multiple users' token embeddings together:

$$[x_{\text{user1}};\; x_{\text{user2}};\; \ldots;\; x_{\text{user100}}] \times W_Q \rightarrow (100 \times 128)$$

Load weights **once**, get 100 results. Same memory transfer, 100× the useful computation.

---

**Q: What can and can't be batched?**

**Can be batched:** all weight multiplications (W_Q, W_K, W_V, W_O, FFN). Weights are the same for every user. Token position doesn't matter — a user's 3rd token and another's 50th token both produce (1 × 4096) embeddings that stack together.

**Cannot be batched:** attention against KV cache. Each user's query must attend to their own KV cache — different sequences, different lengths, different content. These remain independent computations.

---

**Q: What's arithmetic intensity and why does it matter here?**

Arithmetic intensity is the ratio of compute to bytes moved. Without batching, this ratio is low — the GPU is memory-bound, with compute units sitting idle. With batching, the ratio increases, pushing toward compute-bound territory where GPUs excel.

---

**Q: What's the difference between static and continuous batching?**

**Static batching** waits for an entire batch to finish before accepting new requests. This wastes slots when some users finish early. **Continuous batching** inserts a new request into a slot the moment one finishes, keeping the system fully utilized.

---

## The core tension

**Q: How do KV cache and batching interact?**

They **compete for GPU memory**. Every user in the batch carries their own KV cache (~250 MB per user at 500 tokens). 100 concurrent users means ~25 GB just for KV caches. More memory for KV caches means fewer users in the batch (lower throughput). More users in the batch means less room for KV caches (shorter contexts).

Quantizing KV caches helps ease this tension.

---

## Summary

| Optimization | What it does | Tradeoff |
|-------------|-------------|----------|
| KV Cache | Avoids recomputing K,V for old tokens | Trades memory for compute savings |
| Quantization | Fewer bits per parameter | Trades small accuracy loss for memory/bandwidth savings |
| Batching | Amortizes weight loading across users | Trades per-user memory (KV cache) for throughput |

The fundamental constraint: GPU memory is finite and shared by model weights, KV caches, and activations. Inference serving is the art of balancing model quality, per-user latency, and total throughput within that memory budget.
