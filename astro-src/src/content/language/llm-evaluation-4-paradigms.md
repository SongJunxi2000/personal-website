---
title: "LLM Evaluation (4/12) — Paradigms: Fine-Tuning, Few-Shot, Logit vs. Generative"
date: "2026-04-15"
excerpt: "From BERT-on-SQuAD to GPT-3's few-shot prompting, and the subtle but crucial gap between logit-based and generative evaluation."
project: language
readingTime: 8
---

Every time the field changed how it *trained* models, it also had to change how it *evaluated* them. This post walks through three shifts: task-specific fine-tuning, few-shot prompting, and the logit-vs-generative distinction that still trips people up today.

---

## The pre-LLM paradigm: task-specific fine-tuning

Before GPT-3, evaluation assumed each task required a separately fine-tuned model. Here's a concrete example — BERT on SQuAD 2.0.

**Step 1 — fine-tuning.** Take pretrained BERT and add a task-specific head: two learned vectors $S$ (start) and $E$ (end), each of dimension 768. Train on SQuAD's 130K examples with gradient updates, modifying all of BERT's weights.

**Step 2 — evaluation input.**

```
Context: "The Normans were the people who in the 10th and 11th 
centuries gave their name to Normandy, a region in France. They 
were descended from Norse Vikings."

Question: "In what country is Normandy located?"
```

**Step 3 — start/end probabilities.** BERT processes the concatenated input. Each token gets a final hidden state $h_i$. The task head computes:

```
start_score(i) = dot(S, h_i)
end_score(i)   = dot(E, h_i)

start_probs = softmax([start_score(0), ..., start_score(N)])
end_probs   = softmax([end_score(0), ..., end_score(N)])
```

Pick the highest-probability start and end (with `end >= start`), extract the span.

```
Tokens:    [The] [Normans] ... [Normandy] [,] [a] [region] [in] [France] [.]
Position:    0      1     ...     20      21  22    23     24    25     26

Start position = 25 (France) → P = 0.82
End position   = 25 (France) → P = 0.89
Answer: "France"
```

**Step 4 — metrics.** Exact Match and token-level F1.

**Why this broke.** Each task required a separate fine-tuned model with different weights. You couldn't meaningfully compare models across tasks. GPT-3 skipped this entirely — one model, all tasks, same weights, no fine-tuning.

---

## Few-shot learning

**Q: What does "few-shot" mean?**

"Shot" = "example." The terminology comes from computer vision where "one-shot learning" meant recognizing a new object from a single image.

- **Zero-shot**: no examples, just the task instruction.
- **One-shot**: one demonstration.
- **Few-shot**: typically 2–5 demonstrations.

**Q: Why did GPT-3 need few-shot examples at all?**

In 2020, GPT-3 wasn't reliable enough for zero-shot. You had to prime the format:

```
Translate English to French:
"hello" -> "bonjour"
"the cat" -> "le chat"
"good morning" -> "bon matin"
"how are you" ->
```

The zero-shot ability we associate with ChatGPT came later with instruction tuning (InstructGPT, 2022), where the model was fine-tuned specifically to follow natural language instructions.

**Q: Why was few-shot surprising?**

Nobody expected a frozen model (no weight changes) to figure out a task from a few prompt examples. The assumption was that learning a new task required gradient updates. GPT-3 showed that at sufficient scale (~175B parameters), the model develops an emergent ability to pattern-match from context — purely through the attention mechanism during a single forward pass. The weights don't change. Examples sit in the context window and influence output through attention. No backpropagation. Pure inference.

One hypothesis: during pretraining on massive web text, GPT-3 encountered countless implicit task demonstrations — text that naturally takes the form "here's an example, here's another, now here's a new one." It learned a meta-pattern for recognizing and completing task formats.

---

## Logit-based vs. generative evaluation

### Logit-based

For a question like "What is the capital of France? A) London B) Paris C) Berlin D) Madrid", construct four prompts and compare log-probabilities:

```
Prompt 1: "...Answer: London"  → log P = -8.2
Prompt 2: "...Answer: Paris"   → log P = -1.3
Prompt 3: "...Answer: Berlin"  → log P = -7.9
Prompt 4: "...Answer: Madrid"  → log P = -6.5
```

Pick highest → **Paris**. No text is generated.

How the log-probabilities are computed: the model processes tokens sequentially. At each position the final layer outputs a logits vector over the entire vocabulary, softmaxed into probabilities. The log-probability of a completion is the sum of log-probs of each token given all preceding ones. For a multi-token completion:

```
log P("London") = log P("Lon" | ...Answer:) + log P("don" | ...Answer: Lon)
```

Same forward pass as training — you just skip loss, backward, and update.

### Generative

Give the model the full prompt and let it produce text:

```
Prompt: "What is the capital of France?
A) London  B) Paris  C) Berlin  D) Madrid
Answer:"

Model generates: "The answer is B) Paris."
```

Parse the output to extract "B" and check against ground truth.

### Why the distinction matters

A **base pretrained model** might assign high probability to "Paris" but, when generating, ramble: "Well, the capital of France has historically been..." Logit-based eval works well here.

An **instruction-tuned model** (ChatGPT) has been trained to produce "The answer is B." Its probability distribution over bare tokens differs from a base model's. Logit-based eval can undercount its ability. The HF Open LLM Leaderboard v2 switched some tasks to generative evaluation for exactly this reason.

---

## Multiple-choice format problems

**Position bias.** Zheng et al. (ICLR 2024) showed LLMs exhibit "selection bias" — preferring specific option positions. On MMLU, moving golden answers to position A boosted LLaMA-30B by 15.2 percentage points, completely inverting some model rankings.

**Option sensitivity.** Pezeshkpour and Hruschka (NAACL 2024) showed performance gaps of 13–75% from reordering options alone.

**Benchmark errors.** Gema et al. (June 2024) manually reviewed 5,700 MMLU questions and found an estimated 6.49% error rate — with 57% of Virology questions containing errors.

**MMLU-Pro's response.** Expanded to 10 answer options (reducing random guessing from 25% to 10%) but introduced even more position-bias surface area.

---

## The timeline of paradigm shifts

1. **Pre-2020**: task-specific fine-tuning + accuracy metrics (BERT on SQuAD).
2. **2020–2022**: few-shot prompting + broad capability measurement (GPT-3 on MMLU).
3. **2022–2023**: instruction-following + human preference (InstructGPT/ChatGPT → Chatbot Arena).
4. **2024–2026**: reasoning chain evaluation + process-based metrics (o1/R1 → PRMs).
