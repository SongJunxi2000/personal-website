---
title: "FlashAttention Series (4/4) — FlashAttention in Code"
date: "2026-03-26"
excerpt: "A minimal Python/PyTorch implementation of FlashAttention's online softmax and tiling algorithm, with comparison to naive attention."
project: language
readingTime: 6
---

The previous posts covered the hardware motivation, the N×N bottleneck, and the tiling algorithm with online softmax. This post puts it all into code — a naive baseline, a simplified FlashAttention implementation, and notes on what the real CUDA/Triton version does differently.

---

## Naive attention (the baseline)

This is the standard implementation that materializes the full N×N matrix. Simple to read, expensive in memory.

```python
import torch
import torch.nn.functional as F

def naive_attention(Q, K, V):
    """
    Q, K, V: [batch, num_heads, N, d_k]
    Returns:  [batch, num_heads, N, d_k]
    """
    d_k = Q.shape[-1]

    # Step 1: Compute full N×N score matrix → written to HBM
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)

    # Step 2: Read scores from HBM, apply softmax → write weights to HBM
    weights = F.softmax(scores, dim=-1)

    # Step 3: Read weights from HBM, multiply by V → write output to HBM
    output = weights @ V

    return output

    # The scores and weights tensors are both [batch, heads, N, N]
    # and each one gets written/read from HBM — that's the waste.
```

---

## FlashAttention (simplified Python)

This implementation mirrors the algorithm we walked through. It's not optimized (a real implementation uses Triton or CUDA kernels), but it shows the exact logic.

```python
import torch

def flash_attention(Q, K, V, block_size=128):
    """
    FlashAttention with online softmax.

    Q, K, V: [batch, num_heads, N, d_k]
    Returns:  [batch, num_heads, N, d_k]

    The full N×N score matrix is never materialized.
    """
    B, H, N, d_k = Q.shape
    scale = d_k ** -0.5

    # Output accumulator — same shape as Q, NOT N×N
    output = torch.zeros_like(Q)

    # Running softmax accumulators (per row)
    row_max = torch.full((B, H, N, 1), float('-inf'), device=Q.device)
    row_sum = torch.zeros((B, H, N, 1), device=Q.device)

    # Outer loop: iterate over Q blocks
    # (In this simplified version we process all Q rows at once
    #  and tile only over K/V blocks — the key insight is the same)

    num_k_blocks = (N + block_size - 1) // block_size

    for j in range(num_k_blocks):
        # --- Load one K block and one V block ---
        k_start = j * block_size
        k_end = min(k_start + block_size, N)

        K_block = K[:, :, k_start:k_end, :]   # [B, H, block, d_k]
        V_block = V[:, :, k_start:k_end, :]   # [B, H, block, d_k]

        # --- Compute score tile (NOT the full N×N) ---
        # Shape: [B, H, N, block] — only block columns, not all N
        scores_tile = (Q @ K_block.transpose(-2, -1)) * scale

        # --- Online softmax: update running max ---
        tile_max = scores_tile.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(row_max, tile_max)

        # --- Rescale old accumulators ---
        # This is the exp(m_old - m_new) correction
        correction = torch.exp(row_max - new_max)
        row_sum = row_sum * correction
        output = output * correction

        # --- Compute new contributions ---
        # Exponentiate with the global max for stability
        exp_scores = torch.exp(scores_tile - new_max)

        # Update running sum
        row_sum = row_sum + exp_scores.sum(dim=-1, keepdim=True)

        # Multiply weights by V block and accumulate
        output = output + exp_scores @ V_block

        # Update running max
        row_max = new_max

        # scores_tile and exp_scores are now DISCARDED
        # They never touch HBM in a real GPU implementation

    # --- Normalize by final sum ---
    output = output / row_sum

    return output
```

---

## Verifying correctness

The two implementations should produce identical results (up to floating-point precision):

```python
# Create test data
B, H, N, d_k = 2, 8, 256, 64
Q = torch.randn(B, H, N, d_k)
K = torch.randn(B, H, N, d_k)
V = torch.randn(B, H, N, d_k)

# Compare
naive_out = naive_attention(Q, K, V)
flash_out = flash_attention(Q, K, V, block_size=64)

print(f"Max difference: {(naive_out - flash_out).abs().max().item():.2e}")
# Should be ~1e-6 or smaller (float32 precision)
print(f"Allclose: {torch.allclose(naive_out, flash_out, atol=1e-5)}")
# Should be True
```

---

## What the real implementation does differently

The Python code above demonstrates the algorithm but runs entirely in PyTorch on standard GPU kernels — so it doesn't actually achieve the SRAM-level optimizations. A production FlashAttention implementation (like [Dao et al.'s](https://github.com/Dao-AILab/flash-attention)) differs in several ways:

**Written as a fused CUDA/Triton kernel.** The entire inner loop — score computation, online softmax, V multiply — runs as a single GPU kernel. No intermediate tensors are allocated in HBM between steps.

**Tiles over Q blocks too.** Our simplified version processes all Q rows and tiles only over K/V blocks. The real implementation tiles over both dimensions so that the Q block, K block, V block, and accumulators all fit within a single streaming multiprocessor's SRAM.

**Backward pass also tiled.** FlashAttention recomputes the attention scores during the backward pass rather than storing them for backpropagation. This trades extra compute for memory savings — the same memory-bound insight applies.

**Block size tuned to hardware.** The block size is chosen based on the specific GPU's SRAM capacity and memory bandwidth. Typical values are 64–256, tuned per GPU architecture.

---

## Using FlashAttention in practice

In PyTorch 2.0+, FlashAttention is available out of the box:

```python
import torch.nn.functional as F

# PyTorch automatically uses FlashAttention when possible
output = F.scaled_dot_product_attention(Q, K, V)

# Or force a specific backend:
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(Q, K, V)
```

With the HuggingFace Transformers library:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    attn_implementation="flash_attention_2"
)
```

---

## Key takeaways

The algorithm is elegant because it reduces a memory problem to a math trick:

1. **Tiling** avoids materializing the N×N matrix by processing small blocks that fit in SRAM.
2. **Online softmax** makes tiling possible by incrementally computing softmax with a running max/sum and a single-multiplication rescaling correction.
3. **Pipeline fusion** computes scores → softmax → V multiply in one pass per block, so intermediates never touch HBM.
4. The result is **mathematically identical** to standard attention — no approximation, just smarter memory access.
5. Speedup comes from **I/O reduction**, not FLOP reduction. FlashAttention does slightly more arithmetic but dramatically less memory traffic.
