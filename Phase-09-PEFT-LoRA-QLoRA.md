# Phase 09 - PEFT, LoRA & QLoRA

## Goal

Learn parameter-efficient fine-tuning (PEFT), implement LoRA and QLoRA, and understand when to use fine-tuning vs RAG vs prompting in real-world systems.

**Estimated Duration:** 3–5 Weeks

---

## Why this Phase?

Not every problem needs fine-tuning.

A strong AI Engineer knows:

- When prompting is enough
- When RAG is better
- When fine-tuning is required

👉 This phase is about **making the right decision**, not just training models.

---

## Prerequisites

- Complete Phase 08
- Comfortable with LLMs
- Comfortable with RAG
- Basic Hugging Face usage

---

## ⚠️ Cost Policy

- Use FREE GPUs only (Kaggle / Colab)
- Use SMALL models (no large expensive models)
- Keep training lightweight
- Avoid long training runs

---

## 9.1 PEFT Fundamentals

### Resource

https://huggingface.co/docs/peft

---

### Learn (Core)

- What is PEFT
- Why PEFT vs full fine-tuning
- Adapter-based tuning
- Parameter efficiency concept

---

## 9.2 LoRA

### Resource

https://huggingface.co/docs/transformers/training

---

### Learn

- Low-Rank Adaptation (LoRA)
- Trainable adapters
- Rank (r)
- Target modules
- Training workflow

👉 Focus on implementation, not math

---

## 9.3 QLoRA

### Learn

- Quantization basics
- 4-bit quantization
- Memory optimization
- Training on low GPU

👉 This is key for low-resource systems

---

## 9.4 Training Setup (FREE)

### Platforms

- Kaggle (Preferred)
- Google Colab

---

### Learn

#### Dataset Preparation

- Instruction format
- Prompt format
- Train/validation split

#### Evaluation

- Loss
- Accuracy (if classification)
- Manual evaluation
- Baseline comparison

---

## Core Concepts (VERY IMPORTANT)

### Decision Making

- Prompting vs Fine-Tuning
- RAG vs Fine-Tuning
- LoRA vs QLoRA
- When NOT to fine-tune

---

### Optimization

- Quantization
- Memory usage
- Inference speed

---

## Flagship Project

### Fine-Tuned Ticket Classification Model

---

## Project Goal

Build a system that:

- Fine-tunes a model using LoRA
- Fine-tunes using QLoRA
- Compares multiple approaches
- Documents engineering decisions

---

## Suggested Models (Use Small Models)

- DistilBERT (classification)
- Tiny LLaMA / small instruct models

👉 Avoid large models (costly)

---

## Build Order (VERY IMPORTANT)

1. Prepare dataset (clean + format)
2. Train baseline model (existing ML model)
3. Fine-tune using LoRA
4. Evaluate results
5. Fine-tune using QLoRA
6. Compare LoRA vs QLoRA
7. Compare with prompting approach
8. Compare with RAG approach
9. Document results (VERY IMPORTANT)

---

## What to Compare

- Accuracy / performance
- Cost
- Latency
- Complexity

---

## Deliverables

Project (New Repo)

- Fine-Tuned Ticket Classification System

---

## Repository Structure

```text id="9xrepo"
ticket-classification-finetuning/
│
├── dataset/
├── notebooks/
├── models/
├── evaluation/
├── configs/
├── scripts/
├── requirements.txt
└── README.md
```
