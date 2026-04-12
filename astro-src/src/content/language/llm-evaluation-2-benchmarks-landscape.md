---
title: "LLM Evaluation (2/12) — Benchmarks Landscape (1998–2026)"
date: "2026-04-17"
excerpt: "A complete reference table of the benchmarks that shaped LLM evaluation, organized by category."
project: language
readingTime: 6
---

A reference post. If the previous entry sketched the treadmill, this one lays out the individual sleepers underneath it — every benchmark that mattered, when it appeared, and whether it still does.

---

## Complete benchmark reference

| Benchmark | Year | About | Format | Saturated? | Creator |
|---|---|---|---|---|---|
| **MNIST** | 1998 | Handwritten digit recognition | Image classification | ~2020s (20+ years) | Yann LeCun et al. |
| **GLUE** | 2018 | NLU: sentiment, entailment, etc. | Multi-task | ~2019 | NYU/UW/DeepMind |
| **SQuAD 2.0** | 2018 | Reading comprehension with unanswerable questions | Extractive span | ~2020 | Stanford |
| **ARC** | 2018 | Grade-school science | 4-option MC | ~2023 (GPT-4, 96.3%) | Allen AI |
| **SuperGLUE** | 2019 | Harder NLU successor | Multi-task | ~2019 (~5 months) | Same consortium |
| **HellaSwag** | 2019 | Commonsense completion | 4-option MC | ~2023 (GPT-4, 95.3%) | UW (Zellers et al.) |
| **ARC-AGI** | 2019 | Abstract reasoning / fluid intelligence | Visual pattern completion | o3 reached 87.5% (late 2024) | François Chollet |
| **MMLU** | Sep 2020 | 57-subject knowledge test | 4-option MC, few-shot | ~2023 (GPT-4) | Hendrycks et al. |
| **GSM8K** | Oct 2021 | Grade-school math word problems | Open-ended, verify final number | ~2023 (GPT-4) | OpenAI |
| **HumanEval** | Jul 2021 | Function-level code generation | Unit tests (pass@k) | ~2024 (GPT-4o) | OpenAI |
| **MATH** | 2021 | Competition mathematics | Open-ended | Late 2024 (o1) | Hendrycks et al. |
| **TruthfulQA** | 2021 | Tendency to reproduce common falsehoods | Generation + MC | Not fully saturated | Oxford (Lin et al.) |
| **BIG-Bench** | 2022 | 204 diverse capability tasks | Mixed | ~2022 (~10 months, PaLM 540B) | Google + 132 institutions |
| **HELM** | Nov 2022 | Holistic multi-metric framework | Framework | N/A | Stanford CRFM |
| **Chatbot Arena** | May 2023 | Real-world human preference | Blind pairwise, Bradley-Terry/Elo | Unsaturable by design | LMSYS (UC Berkeley) |
| **MT-Bench** | Jun 2023 | Multi-turn conversation quality | LLM-as-judge 1–10 | Largely saturated | LMSYS |
| **AlpacaEval** | 2023 | Instruction-following quality | LLM-as-judge pairwise | Gamed (null model) | Stanford |
| **SWE-bench** | Oct 2023 | Real-world software engineering | Patch + test suite | Still differentiating | Princeton |
| **GPQA** | Nov 2023 | PhD-level science | 4-option MC, "Google-proof" | Approaching ceiling | NYU (Rein et al.) |
| **IFEval** | Nov 2023 | Instruction-following with verifiable constraints | Deterministic verification | ~2024 (~4 months) | Google |
| **WebArena** | 2023 | Web agent task completion | Multi-step on real websites | Not yet | CMU |
| **GSM1K** | 2024 | Fresh GSM8K-style problems (contamination control) | Same format as GSM8K | N/A (diagnostic) | Zhang et al. |
| **RULER** | Apr 2024 | Long-context evaluation (13 tasks) | Retrieval/reasoning | Not yet | NVIDIA |
| **Berkeley Function Calling** | 2024 | Tool use / function calling | Serial, parallel, multi-turn | Not yet | UC Berkeley |
| **LiveCodeBench** | 2024 | Time-segmented coding problems | Code + tests | Refreshes periodically | Academic group |
| **OSWorld** | 2024 | OS-level computer-use agent | Multi-step desktop tasks | Not yet | Academic group |
| **MMLU-Pro** | Jun 2024 | Harder MMLU, 10-option, reasoning-heavy | 10-option MC | Not yet | Tiger Lab |
| **LiveBench** | Jun 2024 | Contamination-resistant, monthly-refreshed | Objective ground truth | Refreshes monthly | Abramovich et al. |
| **SimpleQA** | Nov 2024 | Short-form factuality (adversarially curated) | Open-ended vs ground truth | Not yet (~55% F1) | OpenAI |
| **FrontierMath** | Late 2024 | Unpublished research-level math | Open-ended | Mostly unsolved (~2%) | Epoch AI |
| **Humanity's Last Exam** | Jan 2025 | 2,500 expert questions across all fields | Mixed | Not yet (~37% top) | CAIS (Nature) |
| **MathArena** | 2025 | Fresh competition math as released | Open-ended | Dynamic, by design | ETH Zürich |
| **SWE-bench Pro** | 2025 | Extended SWE-bench: 1,865 problems, 41 repos, 123 languages | Patch + tests | Not yet | ICLR 2026 |
| **INCLUDE** | 2025 | Multilingual with regional/cultural knowledge | MC from local exams, 44 languages | Not yet | ICLR 2025 |

---

## Categories

**Knowledge & reasoning**: MMLU, MMLU-Pro, GPQA, BIG-Bench, TruthfulQA, Humanity's Last Exam.

**Mathematics**: GSM8K, MATH, FrontierMath, MathArena, GSM1K (diagnostic).

**Code**: HumanEval, SWE-bench, SWE-bench Pro, LiveCodeBench.

**Human preference**: Chatbot Arena, MT-Bench, AlpacaEval.

**Instruction following**: IFEval.

**Long context**: RULER.

**Agent / tool use**: WebArena, OSWorld, Berkeley Function Calling Leaderboard.

**Safety & factuality**: TruthfulQA, SimpleQA.

**Multilingual**: INCLUDE.

**Dynamic / anti-contamination**: LiveBench, LiveCodeBench, MathArena.

**Frameworks**: HELM (multi-metric), Open LLM Leaderboard (aggregator).
