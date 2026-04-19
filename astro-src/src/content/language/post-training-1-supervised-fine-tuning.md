---
title: "Post-Training (1/8) — Supervised Fine-Tuning (SFT)"
date: "2026-04-26"
excerpt: "How SFT transforms a base model into an instruction-following model using curated conversation data — the first step in the post-training pipeline."
project: language
readingTime: 4
---

SFT is the bridge between pretraining and instruction following — it reshapes a base model's behavior using curated (instruction, response) pairs. This post covers why it works with surprisingly few examples, how LoRA makes it efficient, and what SFT can't do that motivates RLHF.

---

## What SFT Is

SFT fine-tunes the pretrained base model on a curated dataset of (instruction, response) pairs. The training objective is identical to pretraining — cross-entropy next-token prediction — but the data distribution changes from internet documents to curated conversations.

**Pretraining example:**
```
The capital of France is Paris. It is located along the Seine river and has been...
```

**SFT example:**
```
[User]: What is the capital of France?
[Assistant]: The capital of France is Paris.
```

**Summary:** Pretraining learns *what* to say. SFT learns *when and how* to say it.

## Key Details

- **Loss is computed only on assistant tokens.** The user's input is treated as context. The model is learning to predict answers, not questions.
- **The model learns the EOS (end-of-sequence) token.** This teaches the model *when to stop*. Base models don't have a strong stop signal.
- **SFT datasets are tiny compared to pretraining.** Tens of thousands to low hundreds of thousands of examples (e.g., InstructGPT used ~13,000 demonstrations) vs. trillions of pretraining tokens.
- **Why so few examples work:** SFT isn't teaching new knowledge — the model already has that from pretraining. It's teaching a new *format*: "when you see an instruction, respond helpfully and stop." That's a relatively simple behavioral shift.

## SFT vs. Few-Shot Prompting

Few-shot prompting doesn't change the model's weights. You're crafting the input so that document-continuation behavior happens to produce the output you want. SFT actually updates the weights — after SFT, the model responds correctly even with zero-shot input.

## Which Weights to Update

**Full fine-tuning:** Update all weights. Maximum flexibility but expensive (full optimizer states needed) and risks **catastrophic forgetting** — the model overfits on the small SFT dataset and loses pretrained knowledge.

**Parameter-efficient fine-tuning (PEFT) / LoRA:** Freeze all original weights. Add a small low-rank decomposition (two small matrices A and B) to each weight matrix. Only train these small matrices. This works because SFT is a simple behavioral shift, not a fundamental restructuring — a low-rank update captures that efficiently with far fewer trainable parameters.

## What SFT Can't Do

SFT requires someone to *write* the ideal response for every instruction. For any given prompt, there are many possible good answers that vary in subtle ways. SFT treats it as "this exact sequence of tokens is the target." It can't express "response A is slightly better than response B, which is slightly better than C." It's much easier for a human to *compare* two responses than to *write* the perfect one from scratch. This motivates RLHF.

---

## Key Q&A

**Q: Is SFT the same thing as "fine-tuning"?**
A: The term is overloaded. "Fine-tuning" broadly means continuing training on a narrower dataset — that could be SFT for instruction following, or fine-tuning on medical papers, or on a company's codebase. SFT specifically refers to fine-tuning on (instruction, response) pairs. SFT is a *type* of fine-tuning, but not all fine-tuning is SFT.

**Q: Does SFT need a large dataset?**
A: No. The model already has all knowledge from pretraining. SFT is just redirecting behavior from "continue this document" to "answer this question," which requires relatively few examples.

**Q: Is few-shot prompting like a small SFT?**
A: No — they're fundamentally different. Few-shot prompting doesn't change weights at all. You're tricking the base model into the right behavior by making the input look like a document containing Q&A pairs. SFT updates weights so the model behaves differently even with zero examples in the prompt.

**Q: Why would a low-rank update (LoRA) be sufficient for SFT?**
A: Because SFT is a relatively small behavioral shift — adjusting the output distribution — not restructuring the model's entire knowledge representation. A low-rank update captures that kind of systematic but simple transformation efficiently.
