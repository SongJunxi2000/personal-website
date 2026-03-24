---
title: "BPE Tokenization, Under the Hood (2/2)"
date: "2026-03-22"
excerpt: "Pre-tokenization, the merge loop, and the frequency count trick that makes BPE training fast enough to actually use."
project: language
readingTime: 5
---

*Part 2 of 2. This covers how BPE training actually works — pre-tokenization, the merge loop, and efficiency. See Part 1 for the conceptual overview and comparison with other algorithms.*

---

**Q: What is pre-tokenization and why is it needed?**

Pre-tokenization splits text into word-level chunks **before** BPE merging, so that merges can never cross word boundaries.

Without it, BPE could merge the `e` at the end of `"the"` with the space and `d` of `"dog"` into a meaningless token `e d`. That is a boundary that should not exist. Pre-tokenization defines hard boundaries that the merge loop cannot cross.

In practice, this means the merge loop operates on short byte sequences (pre-tokens) rather than on the whole corpus as one long stream.

---

**Q: What does the GPT-2 pre-tokenization regex actually do?**

It splits text into chunks that roughly correspond to:

- Whitespace-prefixed words: `" the"`, `" dog"` (leading space stays attached to the following word)
- Contractions: `"'m"`, `"'s"`, `"'t"`, `"'re"`
- Numbers and punctuation as their own chunks

Example: `"Hello, how are you?"` → `["Hello", ",", " how", " are", " you", "?"]`

Notice that `" how"` has the space in front, not behind. This means the space is part of the token, not a separator. It is a subtle but important detail — it affects how the tokenizer handles spaces at the start of lines, in prompts, and so on.

This regex requires the `regex` library (not Python's stdlib `re`) because it needs Unicode character classes like `\p{L}` (any Unicode letter) and `\p{N}` (any Unicode number).

---

**Q: What is the correct ordering — file chunking first, or pre-tokenization first?**

File chunking first, pre-tokenization second:

1. **File-level chunking** — splits the corpus at `<|endoftext|>` boundaries into large chunks for parallel processing
2. **GPT-2 regex split** — runs independently on each chunk to produce pre-tokens

This is correct for two reasons: each chunk can run the regex in parallel without sharing state, and `<|endoftext|>` boundaries are document boundaries so you are guaranteed not to split a word in half.

---

**Q: Why does pre-tokenization happen once, not inside the merge loop?**

Pre-tokenization defines fixed boundaries. Those boundaries do not change during training — only the internal representation of each pre-token changes as merges combine adjacent bytes.

Running the regex on every iteration would be both slow (regex is expensive at scale) and wrong (re-splitting would change the boundary structure mid-training). What changes inside the merge loop is the tuple representation of each pre-token: `(b'H', b'e', b'l', b'l', b'o')` gets shorter as merges happen.

---

**Q: What is the full `train_bpe()` pipeline?**

```
1. Initialize vocab = {0: b'\x00', ..., 255: b'\xff'}  (256 bytes)
2. Reserve special tokens (e.g., <|endoftext|>) in vocab
3. Compute num_merges = vocab_size - len(vocab)
4. Read corpus file
5. Chunk file at special token boundaries (enables parallelism)
6. For each chunk (parallelizable):
     - GPT-2 regex split → list of pre-tokens
     - Count frequency of each unique pre-token
7. Merge frequency counts across all chunks
   Result: { (b'H', b'e', b'l', b'l', b'o'): 47, ... }
8. Merge loop — runs num_merges times (sequential):
     - Count all adjacent pairs, weighted by pre-token frequency
     - Find most frequent pair (a, b)
     - Record: merges.append((a, b))
     - Record: vocab[len(vocab)] = a + b
     - Update pre-token tuples: replace (a, b) → merged token
9. Return (vocab, merges)
```

Steps 5–7 are parallelizable. Step 8 is sequential by necessity.

---

**Q: Why must special tokens be reserved before the merge loop?**

It is a counting problem. If `vocab_size = 500` and you have 1 special token, the math is:

`256 (base bytes) + 1 (special) + 243 (merges) = 500`

If you added special tokens after the loop, you would overshoot the target vocabulary size. Special tokens also must never participate in merging — they are atomic units that should never be split or combined with adjacent bytes.

---

**Q: Why does the merge loop run sequentially rather than in parallel?**

Each merge changes the pair frequencies for the next iteration. After merging `(b'l', b'l')` into `b'll'`, new pairs like `(b'e', b'll')` appear that did not exist before. The input to iteration N depends on the output of iteration N−1, which means the loop cannot be parallelized.

This is the fundamental bottleneck of BPE training. You can parallelize the corpus preprocessing (steps 5–7), but the merge loop itself is inherently sequential.

---

**Q: Why is working with pre-token frequency counts so much faster than the naive approach?**

The naive approach rescans the entire corpus on every iteration. If the corpus has 1 million tokens, you iterate over 1 million elements per merge, and for 40,000 merges that is 40 billion operations.

With pre-token frequency counts, if the word `" the"` appears 5,000 times, you process it **once** and multiply any pair counts by 5,000. You iterate over unique pre-tokens (typically 10K–100K) instead of the full corpus (potentially millions of tokens).

The difference between a corpus with 1M tokens and one with 50K unique pre-tokens is a 20× speedup on each merge iteration. Over 40K merges, this is the difference between training in seconds and training for hours.

---

*Day 1 notes — CS336: Language Modeling from Scratch*
