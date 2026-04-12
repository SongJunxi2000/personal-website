---
title: "LLM Evaluation (11/12) — Fairness, HELM's Seven Dimensions, and Multilingual Gaps"
date: "2026-04-08"
excerpt: "HELM's multi-dimensional framework, the ReDial dialect study on AAVE, and why multilingual ≠ multicultural."
project: language
readingTime: 6
---

Stanford's HELM paper (Liang et al., November 2022) introduced holistic multi-metric evaluation. It found that prior to HELM, models were evaluated on only 17.9% of core scenarios. HELM improved this to 96.0% by evaluating 30 models across 42 scenarios on 7 dimensions.

---

## The seven dimensions

### 1. Accuracy

Standard correctness: fraction of correct predictions. HELM applies this across 16 diverse scenarios rather than cherry-picking favorable ones.

### 2. Calibration

Does the model's confidence match its actual correctness?

$$\text{Calibration Error} = \mathbb{E}\big[(p_i - \mathbb{1}(y_i = y_i^*))^2\big]$$

If a model says "90% confident" on 100 questions, it should get about 90 right. A model saying 90% but getting 60% is poorly calibrated. In high-stakes domains (medical, legal), you need to know when the model is uncertain. A poorly calibrated model gives false confidence.

### 3. Robustness

Accuracy under perturbation. Apply transformations (typos, paraphrasing, misspellings) and measure how much accuracy drops.

HELM finding: TNLG v2 (530B) dropped from 72.6% to 38.9% on NarrativeQA under robustness perturbations — nearly halved by typos and rephrasing.

### 4. Fairness

Whether accuracy varies across demographic groups. Tested by applying dialect transformations (standard English → AAVE) and checking whether accuracy changes.

HELM finding: OPT (175B) dropped from 51.3% to 8.8% on the Black demographic split under robustness perturbations, while the White split dropped from 50.8% to 24.3%.

### 5. Bias

Whether model *outputs* contain harmful stereotypes, independent of accuracy. About what the model says, not whether it gets the right answer. Measured via associations between demographic terms and stereotypical attributes in generated text.

The distinction from fairness: fairness asks whether the model performs differently for different groups; bias asks whether the model produces stereotypical content.

### 6. Toxicity

Propensity to generate harmful, offensive, or unsafe content. Measured using toxicity classifiers (Perspective API). Even in benign tasks like summarization, a model might introduce toxic language.

### 7. Efficiency

Computational cost — tokens per second, inference time, parameters, training compute. Makes the cost-quality trade-off explicit.

### The key HELM insight

These dimensions often trade off. A model might be the most accurate but poorly calibrated, or very robust but biased. By measuring all seven simultaneously, HELM exposed that optimizing for accuracy alone masks serious problems in the other six.

---

## Dialect fairness: the ReDial study

**Paper.** "One Language, Many Gaps: Evaluating Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks" (Lin et al., ACL 2025). arXiv: [2410.11005](https://arxiv.org/abs/2410.11005).

**Method.** Hired AAVE speakers, including CS experts, to rewrite seven standard benchmarks (HumanEval, GSM8K, etc.) into AAVE. Created 1,200+ parallel question pairs — same question, same difficulty, different dialect.

**Key findings.**

Almost every model showed significant drops on AAVE — GPT-4o, GPT-4, GPT-3.5, LLaMA-3.1/3, Mistral, Phi-3, across the board.

Models trained on curated data were most unfair. Phi-3-Mini (3.8B) outperformed the larger LLaMA-3-8B on standard English (0.528 vs 0.488 pass rate), but on AAVE Phi-3 dropped by 0.067 while LLaMA-3 dropped by under 0.016. Curating training data to be "clean" removes dialectal variation, making models more brittle.

AAVE hurt more than random misspellings. Models handled typos in standard English better than grammatically correct AAVE. They specifically underperformed on a legitimate English dialect.

Scaling didn't reliably help. Mixtral-8x7B had bigger drops than the smaller Mistral-7B. Mixture-of-Experts doesn't inherently improve dialect robustness.

Asking the model to rephrase didn't fix it. "First translate to standard English, then solve" didn't close the gap and added cost.

---

## Multilingual performance gaps

Models perform substantially better in English than in other languages. It's one of the least contested findings in the field.

**The performance hierarchy.**

- **English**: best performance across all tasks.
- **High-resource European** (French, German, Spanish): close to English, noticeable gap on complex reasoning.
- **Medium-resource** (Chinese, Japanese, Korean, Arabic): larger gaps, especially on nuanced tasks.
- **Low-resource** (most African, indigenous, many Southeast Asian languages): dramatically worse, sometimes barely functional.

**Why.** Training corpora are massively English-skewed. Common Crawl is roughly 45–50% English. Most curated datasets (Wikipedia, books, code) are predominantly English. When you train on 20× more English tokens than Hindi tokens, the model is simply better at English.

**Specific findings.**

- **INCLUDE** (ICLR 2025): 197,243 QA pairs from local exam sources across 44 languages, using indigenous content rather than translated English.
- **OneRuler** (2025): the gap between high- and low-resource languages widens from 11% at 8K context to 34% at 128K context.
- Models "often revert to English reasoning even under non-English prompts."
- Cultural biases persist: multilingual ≠ multicultural. Handling French grammar doesn't mean understanding French cultural context.
- Safety alignment degrades 20–25 percentage points from English to low-resource languages.

**The safety implication.** Models that reliably refuse dangerous requests in English will comply when the same request is made in Zulu or Bengali. Safety training was done primarily in English, so switching languages bypasses it.
