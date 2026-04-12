---
title: "Transformer Series (2/9) — BPE Tokenization"
date: "2026-03-27"
excerpt: "How raw text becomes integers — and why the merge rules matter as much as the vocabulary."
project: language
readingTime: 5
---

Before a transformer can process a single word, it needs to convert text into integers. BPE — Byte Pair Encoding — is how that conversion is learned and applied. It's also the source of a persistent naming confusion: BPE has an encoder and decoder, and so does the Transformer, but they're completely different things.

---

**Q: What is BPE (Byte Pair Encoding)?**

BPE is a tokenization algorithm that determines how to split raw text into tokens. It builds a vocabulary of subword units from a training corpus. Its output is what feeds into the Transformer's embedding matrix.

---

**Q: How does BPE training work?**

BPE training happens once on a large corpus and produces merge rules plus a vocabulary:

```
Start with individual characters as the vocabulary:
  [a, b, c, ..., z, A, B, ..., Z, ', !, ...]

Repeat:
  1. Look at all adjacent pairs in the corpus
  2. Find the most frequent pair
  3. Merge them into a new token, add to vocabulary

Example iterations:
  Most frequent pair: "t" + "h" → merge into "th"
  Next most frequent:  "th" + "e" → merge into "the"
  Next most frequent:  "i" + "n" → merge into "in"
  Next most frequent:  "u" + "n" → merge into "un"
  ... continue until vocab reaches desired size (e.g. 50,000)
```

This produces a list of merge rules in priority order.

---

**Q: What does BPE training output?**

Two things:

```
1. Merge rules (ordered list):    ["t"+"h" → "th",  "th"+"e" → "the",  "u"+"n" → "un", ...]
2. Vocabulary (token → ID map):   {"a": 0, "b": 1, ..., "th": 200, "the": 5000, "un": 5001, ...}
```

Encoding needs both. Decoding only needs the vocabulary.

---

**Q: How does BPE encoding work (text → token IDs)?**

It uses the merge rules in two steps:

```
Step 1: Apply merge rules in priority order to segment the text

  Input: "unhappiness"
  Start with characters: ["u","n","h","a","p","p","i","n","e","s","s"]
  Apply rule "u"+"n" → "un": ["un","h","a","p","p","i","n","e","s","s"]
  Apply rule "h"+"a" → "ha": ["un","ha","p","p","i","n","e","s","s"]
  ... continue applying rules until no more merges possible

Step 2: Look up each segment in the vocabulary to get IDs
  ["un", "happi", "ness"] → [412, 8837, 1053]
```

---

**Q: Is BPE encoding just a find-and-replace for common subwords?**

No — it's not a frequency-based find-and-replace. It applies **merge rules in a specific learned priority order**, and each merge might create new pairs that enable further merges. The process is sequential and order-dependent.

---

**Q: Why can't you skip the merge rules and just use the vocabulary for encoding?**

Because the vocabulary alone doesn't tell you how to segment. If your vocabulary contains "un", "sure", "uns", and "ure", you'd have multiple valid ways to split "unsure":

```
"un" + "sure"    ← correct (what merge rules produce)
"uns" + "ure"    ← wrong segmentation
```

The merge rules resolve this ambiguity by enforcing a specific priority order.

---

**Q: How does BPE decoding work (token IDs → text)?**

Much simpler — it only needs the vocabulary (reversed):

```
[412, 8837, 1053] → look up each ID → ["un", "happi", "ness"] → "unhappiness"
```

No merge rules needed for decoding. Just map IDs back to strings and concatenate.

---

**Q: What happens if the model encounters a word that wasn't in the vocabulary?**

BPE handles this gracefully by falling back to smaller subword units. Since the vocabulary always includes individual characters (from the initial step), any word can be broken down into character-level tokens as a last resort. So a rare technical term might be split into many small pieces, but it never produces an "unknown token" error.

---

**Q: Is the BPE encoder/decoder the same as the Transformer encoder/decoder?**

No — completely different concepts that unfortunately share the same names:

```
Tokenization:   encode/decode = text ↔ integers (simple lookup)
Transformer:    encoder/decoder = architectural design (attention pattern)
```

BPE encoding is a deterministic text preprocessing step. Transformer encoder/decoder refers to neural network architecture choices (bidirectional vs causal attention).

---

**Q: What is V (vocabulary size)?**

V is the total number of unique tokens your model knows about — the number of entries in the vocabulary that BPE produces. If the vocabulary has 50,000 entries, then `V = 50,000`. Every possible token the model can ever see or produce is somewhere in that list.

---

**Q: What does BPE contribute to the Transformer model?**

Only one thing — **V** (the vocabulary size). Once BPE converts text to token IDs, its job is done. The token IDs are then looked up in the embedding matrix of shape `(V, d_model)` to produce vectors that the Transformer processes.
