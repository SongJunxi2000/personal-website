---
title: "BPE Tokenization, Explained (1/2)"
date: "2026-03-22"
excerpt: "What tokenization is, why BPE became dominant, and how it compares to other approaches — from word-level to Unigram LM."
project: language
readingTime: 5
---

*Part 1 of 2. This covers the what and why of BPE tokenization, plus a comparison with other algorithms. Part 2 covers the implementation.*

---

**Q: What is BPE and why does it matter for LLMs?**

BPE (Byte Pair Encoding) is a tokenization algorithm that breaks text into subword units. It is the literal first step of any LLM pipeline — before a model can process text, it must convert strings into sequences of integer token IDs.

BPE is the dominant tokenization method used by GPT, LLaMA, Mistral, and most modern LLMs. It sits at the boundary between the raw world of strings and the numerical world where neural networks operate. Get it wrong and everything downstream breaks.

---

**Q: Why does BPE start with bytes (256 base vocab) rather than characters?**

Starting with the 256 possible single bytes means the tokenizer can handle **any input** — ASCII, Unicode, emojis, even binary data. There is no "unknown token" problem.

A character like `🙃` is 4 bytes in UTF-8, so it starts as 4 separate tokens. Through BPE training, those 4 bytes can get merged into a single token if `🙃` appears frequently enough. If you started with characters instead, you would need a massive base vocabulary to cover all of Unicode — and you would still miss some. Starting with bytes makes the system universal by design.

---

**Q: How does the number of merges affect the vocabulary?**

BPE starts with 256 base tokens and grows the vocabulary by performing merges. Each merge adds one new token. So:

- **Few merges (e.g., 1K vocab)**: Short sequences become many tokens. Rare or unseen words break into smaller pieces gracefully. But longer sequences = more computation.
- **Many merges (e.g., 50K vocab)**: Common words become single tokens. Sequences are shorter and faster to process. But rare words still fragment, and the embedding matrix gets large.

This is a compression tradeoff. More merges = better compression of common patterns at the cost of a larger vocabulary.

---

**Q: How does BPE relate to compression algorithms like Huffman coding?**

Both are greedy algorithms that exploit frequency. Huffman coding assigns shorter bit sequences to more frequent symbols. BPE merges more frequent adjacent pairs first.

BPE is essentially dictionary-based compression: the "dictionary" is the vocabulary of merged byte sequences. When you run `encode()`, you are compressing a string into a shorter sequence of integers drawn from that dictionary. The compression ratio depends on how well the training corpus matches the text you are encoding.

---

## How BPE Compares to Other Tokenization Algorithms

BPE is not the only game in town. Here is a quick survey of the landscape:

**Word-level tokenization** splits on whitespace and punctuation. Simple, intuitive — and it breaks immediately on out-of-vocabulary words. A typo, a proper noun, a rare technical term: all become `<UNK>`. Also requires one embedding per word, so the vocabulary gets enormous. Used in early NLP; largely abandoned.

**Character-level tokenization** treats every character as one token. Zero OOV problem, handles any language. The cost is sequence length: the word "hello" is 5 tokens instead of 1. Longer sequences mean more computation, and the model has to learn what a word even is from scratch. Some recent models (e.g., MegaByte) revisit this for specific reasons, but it is not the default.

**WordPiece** is used by BERT and DistilBERT. The algorithm is similar to BPE but with a different merge criterion: instead of merging the most frequent pair, it merges the pair that maximizes the likelihood of the training data under a language model. Concretely, it prefers merges where `score(a, b) = freq(ab) / (freq(a) × freq(b))` is highest — pairs that co-occur more than you would expect by chance. This tends to produce slightly more linguistically motivated splits.

**Unigram Language Model** (used in SentencePiece by Google, and in XLNet and ALBERT) takes the opposite approach from BPE. Instead of starting small and growing, it starts with a large candidate vocabulary and prunes it down using an EM algorithm. At each step, it removes tokens that minimally increase the total loss. The result is a probabilistic tokenizer: a single string can have multiple valid tokenizations, each with a probability. This allows sampling different tokenizations during training, which can improve robustness.

---

### Quick Comparison

| Algorithm | Strategy | Used By | Strengths | Weaknesses |
|---|---|---|---|---|
| Word-level | Split on whitespace | Early NLP | Simple | OOV problem, huge vocab |
| Character-level | One token per char | Rare / research | Universal, no OOV | Very long sequences |
| BPE | Merge most frequent pairs | GPT, LLaMA, Mistral | Universal (byte-level), fast, simple | Greedy, order-dependent |
| WordPiece | Merge by likelihood gain | BERT, DistilBERT | More principled merges | Harder to implement, slower training |
| Unigram LM | Prune from large vocab | SentencePiece, XLNet | Probabilistic, multiple tokenizations | More complex, EM required |

The practical winner for most modern LLMs has been BPE — specifically byte-level BPE. It is simple, fast to implement, universally applicable, and produces good results. WordPiece and Unigram LM are more principled but more complex; they are worth understanding when you want to know what BERT and SentencePiece are actually doing.

---

*Part 2 covers the actual BPE training pipeline — the merge loop, pre-tokenization, and the efficiency insight that makes it fast.*

*Day 1 notes — CS336: Language Modeling from Scratch*
