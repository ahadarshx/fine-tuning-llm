# MIT 6.034 Course Assistant — QLoRA Fine-Tuned LLM

<p align="center">
  <img src="https://img.shields.io/badge/LLM-Qwen2.5--1.5B-blue">
  <img src="https://img.shields.io/badge/Fine--Tuning-QLoRA-green">
  <img src="https://img.shields.io/badge/Framework-PyTorch-red">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow">
  <img src="https://img.shields.io/badge/UI-Gradio-orange">
  <img src="https://img.shields.io/badge/GPU-GTX%201650-lightgrey">
</p>


## Overview

This project explores adapting a small open-source Large Language Model into a domain-specific AI teaching assistant.

The assistant is trained on MIT 6.034 Artificial Intelligence course material using **QLoRA (Quantized Low-Rank Adaptation)** to make fine-tuning possible on consumer hardware.

The goal was not only to fine-tune a model, but to build and evaluate an end-to-end LLM adaptation pipeline:

- Course data preparation
- Synthetic instruction dataset generation
- Parameter-efficient fine-tuning
- Evaluation pipeline
- Interactive chatbot deployment


## Motivation

Large language models contain broad knowledge, but they are not optimized for specific courses or institutional material.

The objective was to answer:

> Can a small language model be adapted into a course-specific teaching assistant using limited hardware?

Key constraints:

- Consumer GPU only
- Limited VRAM
- Small domain-specific corpus
- Need for explainable educational responses


## Model

Base model:

```
Qwen/Qwen2.5-1.5B-Instruct
```

Why this model?

- Strong instruction-following capabilities
- Small enough for local experimentation
- Supports efficient QLoRA fine-tuning
- Good Hugging Face ecosystem compatibility


## System Architecture

```
MIT Course Notes
        |
        v
Markdown Processing
        |
        v
Synthetic Q&A Generation
        |
        v
Instruction Dataset
        |
        v
Qwen2.5-1.5B
        +
QLoRA Adapter
        |
        v
Fine-Tuned Course Assistant
        |
        v
Gradio Chat Interface
```


## Why QLoRA?

Traditional fine-tuning updates every model parameter.

For a 1.5B parameter model, this requires significant GPU memory.

QLoRA solves this by:

1. Loading the base model in 4-bit precision
2. Freezing original model weights
3. Training small LoRA adapter matrices

In this project:

```
Total Parameters:
~1.5 Billion

Trainable Parameters:
~2.1 Million

Trainable:
~0.14%
```

This enabled fine-tuning on:

```
NVIDIA GTX 1650
4 GB VRAM
```


## Training Configuration

### Quantization

```python
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=True
```

Purpose:

- Reduce VRAM usage
- Preserve model performance
- Enable training locally


### LoRA Configuration

```python
r = 8
alpha = 16
dropout = 0.05
```

Reasoning:

- `r` controls adapter capacity
- `alpha` controls adapter influence
- dropout reduces overfitting


### Target Modules

```python
q_proj
k_proj
v_proj
o_proj
```

Attention projection layers were targeted because they provide strong adaptation capability with minimal trainable parameters.


### Batch Strategy

```python
batch_size = 1
gradient_accumulation_steps = 16
```

Effective batch size:

```
1 × 16 = 16
```

Gradient accumulation allowed stable training despite limited GPU memory.


# Dataset Evolution

The project went through multiple iterations.

---

# Version 1: Direct Note Fine-Tuning

Initial pipeline:

```
Course Notes
      |
      v
Question Templates
      |
      v
Training Examples
```

Example:

Question:

```
Explain Section 3.3 — Dynamic Ordering
```

Answer:

```
Choose variable with smallest remaining domain...
```

## Problems Found

The model began overfitting.

Root causes:

### 1. Artificial Questions

Generated prompts did not match real student questions.

The model learned:

```
Section Heading → Notes
```

instead of:

```
Student Question → Explanation
```


### 2. Raw Note Responses

Answers were copied directly from notes.

The model learned note formatting rather than conversational teaching style.


### 3. Repetitive Dataset

Limited question templates caused repeated examples.

This increased memorization.


### 4. Small Corpus Size

A 1.5B parameter model could easily memorize a small course corpus.


## Solution

Redesign the dataset generation pipeline.


# Version 2: Synthetic Instruction Dataset

New pipeline:

```
Course Notes
      |
      v
Qwen2.5 Generator
      |
      v
Natural Q&A Dataset
      |
      v
Fine-Tuning
```

Instead of:

```
Explain A*
```

The dataset contained:

```
Why is A* guaranteed to find an optimal path?
```

This better matched real student interactions.


## Dataset Statistics

| Split | Examples |
|---|---:|
| Train | 663 |
| Validation | 142 |
| Test | 119 |


## Training Results

Validation loss:

| Epoch | Validation Loss |
|-|-|
| 1 | 1.0859 |
| 2 | 1.0498 |
| 3 | 1.0455 |

Validation loss continued improving, showing healthier learning behavior compared to the first version.


# Evaluation

Evaluation used:

- Required concept recall
- ROUGE-L similarity
- Manual factual review


## Results

| Model | ROUGE-L |
|-|-:|
| Base Qwen2.5-1.5B | 0.127 |
| Fine-Tuned Adapter | 0.244 |

The fine-tuned model produced responses more aligned with the course reference material.


# Interface

A Gradio application was created for interaction.

Example questions:

```
How do BFS and DFS differ?

Why is A* optimal?

Does alpha-beta pruning change minimax results?
```


Run:

```bash
python app.py
```

Open:

```
http://127.0.0.1:7860
```


# Current Limitations

The model still answers some out-of-domain questions.

Example:

```
Who won the FIFA World Cup?
```

Reason:

The dataset contained many more answering examples than refusal examples, so the model learned answering behavior more strongly than domain boundaries.


# Future Improvements

## Retrieval-Augmented Generation (RAG)

Next iteration:

```
User Question
        |
        v
Retrieve Relevant Course Material
        |
        v
LLM Generates Grounded Answer
```

Expected benefits:

- Better factual accuracy
- Less hallucination
- Easier course updates
- Support for PDFs, slides, and lecture material


## Larger Models

Future experiments:

- Qwen2.5-7B
- Llama 3.1 8B
- Larger instruction models


# Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- bitsandbytes
- QLoRA
- Gradio
