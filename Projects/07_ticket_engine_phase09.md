# Project 07 - AI Ticket Workflow Engine (Fine-Tuning Upgrade)

## Phase Mapping

- Phase 09
- Extends: Project 06 (AI Agent System)

---

## Goal

Improve the AI Ticket Workflow Engine by applying fine-tuning (LoRA / QLoRA) and comparing it with other approaches like RAG and prompting.

---

## Why Fine-Tuning Here?

This is NOT a separate project.

Fine-tuning is applied to improve the **existing AI system** built in Phase 06.

👉 This ensures:

- Better classification accuracy
- Better workflow decisions
- Real-world system improvement

---

## What You Already Have

From Phase 06:

- AI agent system
- Ticket workflow automation
- Tool-based execution (LangGraph + MCP)
- Basic classification logic (rule-based or ML)

---

## What You Will Build NOW

### Model Improvement Layer

- Fine-tuned classification model
- Comparative evaluation system

---

## Tech Stack

- Hugging Face Transformers
- PEFT (LoRA / QLoRA)
- PyTorch
- Kaggle / Colab (FREE GPU)

---

## Features

### Fine-Tuning

- Train model using LoRA
- Train model using QLoRA
- Use lightweight models (important)

---

### Evaluation (MOST IMPORTANT)

Compare multiple approaches:

- Existing baseline (rule-based / ML)
- Fine-tuned model (LoRA)
- Fine-tuned model (QLoRA)
- RAG-based approach
- Prompting approach

---

### Integration

- Replace classification logic in agent system
- Improve decision accuracy
- Plug improved model into workflow

---

## Architecture (Updated)

```text
User Input
   ↓
Agent (LangGraph)
   ↓
Classification Layer (Improved Model)
   ↓
Tool Selection
   ↓
MCP Tools
   ↓
Database / APIs
   ↓
Response
```
