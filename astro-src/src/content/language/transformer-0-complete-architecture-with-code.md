---
title: "Transformer Series (0/9) — Complete Architecture Walkthrough with Code"
date: "2026-03-23"
excerpt: "The full decoder-only Transformer in working PyTorch — from BPE tokenization to text generation, with every piece explained."
project: language
readingTime: 10
---

This is the overview post for the Transformer series. Instead of Q&A, it walks through the entire architecture end-to-end with working PyTorch code. Each section corresponds to a detailed post in the series (1–9). Read this for the full picture first, then go deeper on any piece.

We follow data through the complete pipeline:

```
Raw text
  → BPE tokenizer → token IDs
  → Embedding lookup → (N, d_model)
  → + Positional encoding → X_final (N, d_model)
  → Transformer Block × num_layers:
      → Multi-head self-attention (with causal mask)
      → Residual + LayerNorm
      → FFN
      → Residual + LayerNorm
  → Output head: final_hidden @ W_embed^T → (N, V)
  → Softmax → probabilities
  → Sample next token
```

---

## 1. Tokenization (BPE)

BPE is a preprocessing step — not part of the neural network. It converts raw text into token IDs using learned merge rules and a vocabulary.

```python
# In practice you'd use a library like tiktoken or sentencepiece.
# Here's a simplified illustration of what BPE does:

# BPE training produces:
#   1. Merge rules (ordered): [("t","h") → "th", ("th","e") → "the", ...]
#   2. Vocabulary (token → ID): {"the": 5000, "cat": 2368, " ": 32, ...}

# Encoding: text → token IDs
def simple_tokenize(text, vocab):
    """Simplified tokenizer — real BPE applies merge rules sequentially."""
    tokens = text.split()  # oversimplified; real BPE works at subword level
    return [vocab[t] for t in tokens]

# Decoding: token IDs → text
def simple_detokenize(ids, id_to_token):
    """Reverse lookup — just map IDs back to strings and concatenate."""
    return " ".join(id_to_token[i] for i in ids)

# Example:
# "The cat sat" → [464, 2368, 3290]
# [464, 2368, 3290] → "The cat sat"
```

BPE's only contribution to the model is **V** (vocabulary size). Once token IDs exist, BPE's job is done.

---

## 2. Token Embeddings

The embedding matrix is a lookup table of shape `(V, d_model)`. Each row is a learned vector for one token.

```python
import torch
import torch.nn as nn
import math

# Hyperparameters
V = 50000        # vocabulary size
d_model = 512    # embedding dimension
max_seq_len = 1024
num_heads = 8
num_layers = 6
d_ff = 2048      # feedforward hidden dimension

# The embedding matrix — initialized randomly, learned during training
token_embedding = nn.Embedding(V, d_model)

# Example: convert token IDs to vectors
token_ids = torch.tensor([464, 2368, 3290])  # "The cat sat"
X = token_embedding(token_ids)  # shape: (3, 512)

# X[0] is the 512-dim vector for "The" (row 464 of the embedding matrix)
# X[1] is the 512-dim vector for "cat" (row 2368)
# X[2] is the 512-dim vector for "sat" (row 3290)
print(X.shape)  # torch.Size([3, 512])
```

At initialization, these vectors are random. Through training, tokens with similar meanings end up with similar vectors.

---

## 3. Positional Encoding

Attention is permutation-invariant — without position information, "dog bites man" and "man bites dog" produce identical representations. Positional encoding adds a unique position-dependent vector to each token.

```python
def sinusoidal_positional_encoding(max_len, d_model):
    """
    Original Transformer positional encoding using sin/cos at different frequencies.

    Each dimension oscillates at a different frequency:
      - Low dimensions change fast (like a clock's second hand)
      - High dimensions change slowly (like the hour hand)

    This gives each position a unique fingerprint, with values bounded in [-1, 1].
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)

    # Compute the division term: 10000^(2i/d_model)
    # Using log-space for numerical stability:
    #   10000^(2i/d_model) = exp(2i * log(10000) / d_model)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )  # shape: (d_model/2,)

    # Even dimensions get sin, odd dimensions get cos
    pe[:, 0::2] = torch.sin(position * div_term)  # sin(pos / 10000^(2i/d_model))
    pe[:, 1::2] = torch.cos(position * div_term)  # cos(pos / 10000^(2i/d_model))

    return pe  # shape: (max_len, d_model)

# Create positional encodings (fixed, not learned)
PE = sinusoidal_positional_encoding(max_seq_len, d_model)

# Add positional encoding to token embeddings
# X is (N, d_model), PE[:N] is (N, d_model)
N = X.shape[0]  # number of tokens in our sequence
X_final = X + PE[:N]  # element-wise addition, shape unchanged: (3, 512)

# Now "cat" at position 1 has a different vector than "cat" would at position 0
# because different positional vectors were added.
```

This `X_final` is the actual input to the Transformer. From here, everything operates on `(N, d_model)` tensors.

---

## 4. Scaled Dot-Product Attention

The core mechanism for tokens to "look at" each other. Each token gets three projections: Q (what am I looking for?), K (what do I advertise?), V (what information do I carry?).

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        Q: queries,  shape (batch, num_heads, N, d_k)
        K: keys,     shape (batch, num_heads, N, d_k)
        V: values,   shape (batch, num_heads, N, d_k)
        mask: optional causal mask, shape (N, N)

    Returns:
        output: shape (batch, num_heads, N, d_k)
        attention_weights: shape (batch, num_heads, N, N)
    """
    d_k = Q.shape[-1]

    # Step 1: Q @ K^T — compute similarity between all pairs of tokens
    # (batch, heads, N, d_k) @ (batch, heads, d_k, N) → (batch, heads, N, N)
    scores = Q @ K.transpose(-2, -1)

    # Step 2: Scale by sqrt(d_k) to prevent softmax saturation
    # Without this, large d_k produces large dot products,
    # which makes softmax nearly one-hot (only one token gets attention)
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply causal mask (for decoder models)
    # Set future positions to -inf so softmax gives them 0 weight
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Softmax — normalize each row into a probability distribution
    # Each row says "how much should token i attend to each token j"
    attention_weights = torch.softmax(scores, dim=-1)

    # Step 5: Multiply by V — weighted blend of value vectors
    # (batch, heads, N, N) @ (batch, heads, N, d_k) → (batch, heads, N, d_k)
    # Each token's output is a cocktail of information from tokens it found relevant
    output = attention_weights @ V

    return output, attention_weights
```

---

## 5. Multi-Head Attention

Instead of one attention operation on full `d_model` vectors, we split into multiple heads — each attending to a different aspect of the input. This gives us multiple "perspectives" for roughly the same compute cost.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # each head works with smaller vectors

        # Projection matrices — these are the learned parameters
        # W_Q, W_K, W_V each (d_model, d_model)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        # W_O: output projection to mix information across heads
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X, mask=None):
        """
        Args:
            X: input, shape (batch, N, d_model)
            mask: causal mask, shape (N, N)
        Returns:
            output: shape (batch, N, d_model)
        """
        batch_size, N, _ = X.shape

        # Step 1: Project X into Q, K, V — three different "views" of the input
        # Each is (batch, N, d_model)
        Q = self.W_Q(X)
        K = self.W_K(X)
        V = self.W_V(X)

        # Step 2: Split d_model into num_heads chunks of size d_k
        # Reshape: (batch, N, d_model) → (batch, N, num_heads, d_k) → (batch, num_heads, N, d_k)
        Q = Q.view(batch_size, N, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, N, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, N, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Run attention independently on each head
        # Each head can learn to focus on different relationship types
        #   e.g., head 1: subject-verb, head 2: adjective-noun, etc.
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: (batch, num_heads, N, d_k)

        # Step 4: Concatenate heads back together
        # (batch, num_heads, N, d_k) → (batch, N, num_heads * d_k) = (batch, N, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, N, self.d_model)

        # Step 5: Final projection through W_O to mix information across heads
        output = self.W_O(attn_output)  # (batch, N, d_model)

        return output
```

---

## 6. Causal Mask

For decoder models (GPT, LLaMA, Claude), each token can only attend to itself and tokens before it. This prevents "seeing the future" during training.

```python
def create_causal_mask(N):
    """
    Creates a lower-triangular mask of shape (N, N).

    Entry (i, j) = 1 if j <= i (can attend), 0 if j > i (blocked).

    For N=4 ("The cat sat on"):
        [[1, 0, 0, 0],    ← "The" can only see itself
         [1, 1, 0, 0],    ← "cat" sees "The" and itself
         [1, 1, 1, 0],    ← "sat" sees "The", "cat", itself
         [1, 1, 1, 1]]    ← "on" sees everything before it

    Positions with 0 get set to -inf before softmax → softmax(-inf) = 0 attention.
    This enables training on all positions in parallel while keeping each honest.
    """
    mask = torch.tril(torch.ones(N, N))
    return mask

# Example
mask = create_causal_mask(4)
print(mask)
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])
```

---

## 7. Feedforward Network (FFN)

The "thinking" layer. Attention moves information between tokens; the FFN processes that information per-token through a wider hidden layer. Research suggests factual knowledge is stored here.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Expand to 4× (d_ff = 4 * d_model typically), then compress back
        self.W1 = nn.Linear(d_model, d_ff)    # (512, 2048) — expand
        self.W2 = nn.Linear(d_ff, d_model)    # (2048, 512) — compress
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        FFN(x) = W2 · ReLU(W1 · x + b1) + b2

        (N, 512) → (N, 2048) → ReLU → (N, 512)

        The expand-then-compress gives the network a larger space
        to do computation before squeezing back down.
        """
        return self.W2(self.relu(self.W1(x)))
```

---

## 8. Complete Transformer Block

Each block wraps attention and FFN with residual connections and layer normalization. Residual connections create gradient highways for training deep networks. Layer norm stabilizes activations.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)  # learned γ and β, size d_model
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        """
        One Transformer block:

        X
          → Multi-head self-attention
          → X + attention_output          ← residual (never lose original info)
          → LayerNorm                     ← stabilize values
          → FFN
          → normed + FFN_output           ← another residual
          → LayerNorm                     ← stabilize again
          → output of this block

        Input and output are both (batch, N, d_model).
        """
        # Sub-layer 1: Multi-head attention + residual + norm
        attn_output = self.attention(X, mask)
        X = self.norm1(X + self.dropout(attn_output))   # residual + layer norm

        # Sub-layer 2: FFN + residual + norm
        ffn_output = self.ffn(X)
        X = self.norm2(X + self.dropout(ffn_output))    # residual + layer norm

        return X
```

---

## 9. Full Decoder-Only Transformer

Stack everything together: embeddings → positional encoding → N blocks → output head.

```python
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, V, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()

        # Token embedding: the (V, d_model) lookup table — learned
        self.token_embedding = nn.Embedding(V, d_model)

        # Positional encoding: fixed sinusoidal vectors
        self.register_buffer('PE', sinusoidal_positional_encoding(max_seq_len, d_model))

        # Stack of Transformer blocks (GPT-3 uses 96, we use 6 here)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, token_ids):
        """
        Full forward pass: token IDs → next-token probabilities.

        Args:
            token_ids: (batch, N) tensor of integer token IDs
        Returns:
            logits: (batch, N, V) raw scores for each vocab token at each position
        """
        batch_size, N = token_ids.shape

        # Step 1: Token embedding lookup
        X = self.token_embedding(token_ids)  # (batch, N, d_model)

        # Step 2: Add positional encoding
        X = X + self.PE[:N]  # broadcast over batch dimension

        # Step 3: Create causal mask
        mask = create_causal_mask(N).to(X.device)

        # Step 4: Pass through all Transformer blocks
        for block in self.blocks:
            X = block(X, mask)

        X = self.final_norm(X)

        # Step 5: Output head — project back to vocabulary size
        # Weight tying: reuse the embedding matrix transposed
        # (batch, N, d_model) @ (d_model, V) → (batch, N, V)
        logits = X @ self.token_embedding.weight.T

        return logits
```

---

## 10. Training Loop

Cross-entropy loss measures prediction quality. Teacher forcing feeds correct tokens as input.

```python
def train_step(model, optimizer, token_ids):
    """
    One training step.

    Given a sequence of N tokens, we get N-1 training examples:
      position 0 predicts token 1
      position 1 predicts token 2
      ...
      position N-2 predicts token N-1

    Teacher forcing: the input always contains the correct previous tokens,
    regardless of what the model would have predicted.
    """
    # Input: all tokens except the last
    # Target: all tokens except the first (shifted by 1)
    input_ids = token_ids[:, :-1]    # (batch, N-1)
    target_ids = token_ids[:, 1:]    # (batch, N-1)

    # Forward pass — get logits for every position
    logits = model(input_ids)  # (batch, N-1, V)

    # Cross-entropy loss: -log(probability assigned to correct token)
    # Reshape for PyTorch's cross_entropy: (batch*(N-1), V) vs (batch*(N-1),)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V),
        target_ids.reshape(-1)
    )

    # Backward pass — compute gradients for ALL parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Initialize model and optimizer
model = DecoderOnlyTransformer(V, d_model, num_heads, d_ff, num_layers, max_seq_len)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Example training step (with dummy data)
dummy_batch = torch.randint(0, V, (4, 128))  # batch of 4, sequence length 128
loss = train_step(model, optimizer, dummy_batch)
# This single step trains on 4 × 127 = 508 examples simultaneously
```

---

## 11. Inference (Text Generation)

At inference, we generate one token at a time, feeding each prediction back as input.

```python
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=50, temperature=1.0):
    """
    Autoregressive generation: predict one token, append it, repeat.

    Args:
        model: trained DecoderOnlyTransformer
        prompt_ids: (1, N) tensor of prompt token IDs
        max_new_tokens: how many tokens to generate
        temperature: controls randomness (lower = more deterministic)

    Returns:
        generated token IDs
    """
    generated = prompt_ids.clone()

    for _ in range(max_new_tokens):
        # Forward pass on entire sequence so far
        logits = model(generated)  # (1, current_len, V)

        # Only look at the LAST position's prediction
        next_token_logits = logits[:, -1, :]  # (1, V)

        # Apply temperature scaling
        # Low temperature → sharp distribution (more deterministic)
        # High temperature → flat distribution (more random)
        next_token_logits = next_token_logits / temperature

        # Convert to probabilities
        probs = torch.softmax(next_token_logits, dim=-1)  # (1, V)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Append to sequence and continue
        generated = torch.cat([generated, next_token], dim=1)

    return generated

# Example:
# prompt = tokenize("The cat sat")
# output = generate(model, prompt)
# print(detokenize(output))
```

---

## 12. Parameter Count Summary

For a model with `V=50000, d_model=512, num_heads=8, d_ff=2048, num_layers=6`:

```
Embedding matrix:          V × d_model           = 50000 × 512   = 25,600,000

Per Transformer block:
  W_Q, W_K, W_V, W_O:     4 × d_model²          = 4 × 262,144   = 1,048,576
  FFN (W1 + W2):           d_model×d_ff + d_ff×d_model = 2 × 1,048,576 = 2,097,152
  Layer norms (γ, β):      4 × d_model            = 2,048
  Block total:             ≈ 3,147,776

All 6 blocks:              6 × 3,147,776          = 18,886,656

Output head (weight tied):  0 (reuses embedding matrix)

Total:                     ≈ 44,486,656 parameters
```

GPT-3 (175B) uses `d_model=12288, num_heads=96, num_layers=96, d_ff=49152` — same architecture, just scaled up.

---

## Quick Reference: The Complete Data Flow

```
"The cat sat on the mat"
    │
    ▼
[BPE Tokenizer] ─── token IDs: [464, 2368, 3290, 319, 262, 2603]
    │
    ▼
[Embedding Lookup] ─── (6, 512) — each token → d_model vector
    │
    ▼
[+ Positional Encoding] ─── (6, 512) — inject position info
    │
    ▼
[Transformer Block 1]
    ├── Multi-Head Self-Attention (causal masked)
    │     Q, K, V = X @ W_Q, W_K, W_V
    │     scores = Q @ K^T / sqrt(d_k)
    │     mask future positions → -inf
    │     weights = softmax(scores)
    │     output = weights @ V
    ├── Residual + LayerNorm
    ├── FFN (expand 512→2048, ReLU, compress 2048→512)
    └── Residual + LayerNorm
    │
    ▼
[Transformer Block 2] ... [Block 6]
    │
    ▼
[Output Head] ─── (6, 512) @ (512, 50000) → (6, 50000) logits
    │
    ▼
[Softmax] ─── (6, 50000) probabilities, one distribution per position
    │
    ▼
[Sample / Argmax] ─── next token prediction
```
