---
title: "Transformer (8/9) — Encoder vs Decoder Architecture"
date: "2026-03-21"
excerpt: "Why the industry converged on decoder-only — and what was lost when the encoder was left behind."
project: language
readingTime: 7
---

The original Transformer had two halves: an encoder that read the source language with full attention, and a decoder that generated the target language one token at a time. Today, almost every major LLM uses decoder-only. This post traces the architectural difference, the training data advantage that settled it, and what cross-attention actually does.

---

**Q: What do "encoder" and "decoder" mean in the Transformer context?**

They refer to two different architectural designs with different **attention patterns** — NOT the same as BPE encoding/decoding (which is simple text ↔ integer lookup).

---

**Q: What is the Transformer encoder?**

A stack of blocks where every token can attend to EVERY other token (no masking):

```
"The cat sat on the mat"

            The  cat  sat  on  the  mat
   The    [  ✓    ✓    ✓    ✓    ✓    ✓ ]
   cat    [  ✓    ✓    ✓    ✓    ✓    ✓ ]
   sat    [  ✓    ✓    ✓    ✓    ✓    ✓ ]
   ...
```

Full bidirectional attention. Good for tasks like "fill in the blank" (BERT), but can't be used for generation because it would be cheating.

---

**Q: What is the Transformer decoder?**

A stack of blocks where each token can only attend to itself and previous tokens (causal masking):

```
            The  cat  sat  on  the  mat
   The    [  ✓    ✗    ✗    ✗    ✗    ✗ ]
   cat    [  ✓    ✓    ✗    ✗    ✗    ✗ ]
   sat    [  ✓    ✓    ✓    ✗    ✗    ✗ ]
   ...
```

This enables autoregressive generation — predict the next token using only what came before.

---

**Q: How did the original Transformer use both?**

It was designed for translation and used both stacks connected by cross-attention:

```
Encoder: reads "Je suis étudiant" with full bidirectional attention
         → produces enc_output, a rich representation of the French sentence

Decoder: generates "I am a student" one token at a time
         → causal masking for the English side
         → cross-attention to look at the encoder's output
```

---

**Q: What does the encoder look like step by step?**

```
Input: "Je suis étudiant"

Step 1: Token embedding + positional encoding → X_enc (3, d_model)

Step 2: Pass through 6 encoder blocks

Each encoder block:
  → Multi-head self-attention (NO causal mask)
  → Residual + LayerNorm
  → FFN
  → Residual + LayerNorm

Output: enc_output, shape (3, d_model)
```

Each position's vector now encodes not just that token but its relationship to all other tokens.

---

**Q: What does the decoder look like step by step?**

Each decoder block has **three** sub-layers (not two like encoder):

```
Sub-layer 1: Masked self-attention (causal mask)
  → Q, K, V all from decoder input
  → "am" can see "I" and itself, but NOT "a" or "student"
  → Residual + LayerNorm

Sub-layer 2: Cross-attention (the key new piece)
  → Q comes from decoder
  → K and V come from enc_output
  → Residual + LayerNorm

Sub-layer 3: FFN
  → Residual + LayerNorm
```

---

**Q: What exactly is cross-attention?**

Regular attention, but Q comes from one source and K/V from another:

```
Self-attention:
  Q = X_dec @ W_Q     ← from decoder
  K = X_dec @ W_K     ← from decoder
  V = X_dec @ W_V     ← from decoder

Cross-attention:
  Q = X_dec @ W_Q      ← from decoder ("what am I looking for?")
  K = enc_output @ W_K  ← from encoder ("what does the French contain?")
  V = enc_output @ W_V  ← from encoder ("what French info to grab?")
```

The decoder token "am" sends a query asking "what French words are relevant to me?" and attends most to "suis" (which means "am" in French).

---

**Q: How does inference work with encoder-decoder?**

```
Step 1: Encode "Je suis étudiant" → enc_output (done ONCE)

Step 2: Start decoder with "<start>"
  → self-attention + cross-attention to enc_output → predicts "I"

Step 3: Feed "<start> I"
  → predicts "am"

Step 4: Feed "<start> I am"
  → predicts "a"

... continue until model predicts "<end>"
```

The encoder runs once. The decoder runs one step per output token, each time cross-attending to the same encoder output.

---

**Q: Before decoder-only models, did each task require a separate model?**

Yes. In the 2017-2018 era:

```
Translation model:      trained on English-French pairs → can only translate
Summarization model:    trained on article-summary pairs → can only summarize
QA model:              trained on question-answer pairs → can only answer questions
```

Each was a separate model with separate paired training data.

T5 (2020) tried to fix this by framing every task as text-to-text within encoder-decoder, but still needed paired data per task.

---

**Q: What training data does encoder-decoder need vs decoder-only?**

Encoder-decoder needs **paired** data:
```
encoder input:  "Je suis étudiant"
decoder target: "I am a student"
```

Decoder-only just needs **raw text**:
```
"The cat sat on the mat"
→ every position predicts the next token
→ no pairing or labeling needed
```

This is why decoder-only won — the internet is your training set. No need for carefully curated pairs.

---

**Q: How does a decoder-only model handle tasks like translation?**

The paired structure is embedded within a single continuous sequence:

```
"Translate French to English: Je suis étudiant → I am a student"
```

The model sees this as continuous text and learns next-token prediction. The "question then answer" pattern emerges naturally. This is the insight behind instruction tuning and RLHF.

---

**Q: Why did the industry converge on decoder-only?**

```
BERT (2018):     encoder only  → understanding tasks
GPT (2018):      decoder only  → generation
T5 (2020):       encoder-decoder → both

Modern LLMs (GPT-4, Claude, LLaMA): all decoder-only
```

Decoder-only can get both understanding AND generation from one architecture with enough scale and data. And it can use unlimited raw text instead of requiring task-specific paired data.
