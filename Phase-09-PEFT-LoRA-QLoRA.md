# Phase 09 - PEFT, LoRA & QLoRA

## Goal

Understand Parameter-Efficient Fine-Tuning (PEFT) techniques, learn LoRA and QLoRA, and compare fine-tuning with prompting and RAG for real-world AI applications.

**Estimated Duration:** 4–6 Weeks

---

## Why this Phase?

Not every AI problem requires fine-tuning.

A good AI Engineer should know:

- When prompting is enough.
- When RAG is a better solution.
- When fine-tuning is actually required.

This phase focuses on making the right engineering decision rather than fine-tuning every model.

---

## Prerequisites

- Complete Phase 08
- Comfortable with LLMs
- Comfortable with RAG
- Basic understanding of Hugging Face

---

## 9.1 PEFT (Parameter-Efficient Fine-Tuning)

### Resources

1. Hugging Face PEFT Documentation

https://huggingface.co/docs/peft

---

### Learn

PEFT Fundamentals

- What is PEFT?
- Why PEFT?
- Full Fine-Tuning vs PEFT
- Adapter-based Fine-Tuning

---

## 9.2 LoRA

### Resources

1. Hugging Face Transformers Training Guide

https://huggingface.co/docs/transformers/training

---

### Learn

LoRA

- Low-Rank Adaptation
- Trainable Adapters
- Rank (r)
- Target Modules
- Training Workflow

---

## 9.3 QLoRA

### Learn

- Quantization
- 4-bit Quantization
- Memory Optimization
- QLoRA Workflow
- GPU Memory Saving

---

## 9.4 Training Environment

### Resources

1. Kaggle Notebooks

https://www.kaggle.com/code

2. Google Colab

https://colab.research.google.com/

Use free GPUs whenever available.

---

## Learn

Dataset Preparation

- Dataset Formatting
- Instruction Format
- Prompt Format
- Train / Validation Split

Evaluation

- Loss
- Accuracy
- Human Evaluation
- Baseline Comparison

---

## Concepts to Learn

Decision Making

- Prompting vs Fine-Tuning
- RAG vs Fine-Tuning
- LoRA vs QLoRA
- When to Fine-Tune
- Cost Comparison

Optimization

- Quantization
- Memory Usage
- Inference Speed

---

## Flagship Project

### Fine-Tuned Ticket Classification Model

---

### Tech Stack

- Hugging Face Transformers
- PEFT
- LoRA
- QLoRA
- PyTorch

---

### Build Order

1. Prepare Dataset
2. Choose Base Model
3. Fine-Tune using LoRA
4. Evaluate Results
5. Fine-Tune using QLoRA
6. Compare Performance
7. Compare with Prompt Engineering
8. Compare with XGBoost Baseline
9. Compare with RAG Approach
10. Document Findings

---

## Deliverables

Project

- Fine-Tuned Ticket Classification Model

Repository

```text
ticket-classification-finetuning/
│
├── dataset/
├── notebooks/
├── models/
├── evaluation/
├── configs/
├── requirements.txt
└── README.md
```

Artifacts

- LoRA Model
- QLoRA Model
- Evaluation Report
- Performance Comparison
- Training Configuration

---

## Skills Gained

- PEFT
- LoRA
- QLoRA
- Dataset Preparation
- Fine-Tuning
- Quantization
- Model Evaluation
- Cost Optimization
- Model Comparison

---

## Completion Criteria

Complete this phase only after:

- Learned PEFT fundamentals.
- Fine-tuned a model using LoRA.
- Fine-tuned a model using QLoRA.
- Compared LoRA and QLoRA.
- Compared Prompting vs RAG vs Fine-Tuning.
- Documented all experimental results.

---

## Final Portfolio

After completing the roadmap, your portfolio should contain:

### Project 1

Production Customer Churn Prediction System

Stack

- XGBoost
- SHAP
- FastAPI
- PostgreSQL
- Docker
- GitHub Actions
- AWS EC2
- MLflow
- DVC
- Prefect
- Pandera
- Evidently
- Kubernetes

---

### Project 2

RAG Knowledge Assistant

Stack

- LangChain
- FAISS
- ChromaDB
- Ollama / OpenAI / Gemini
- Ragas
- Phoenix
- FastAPI
- Docker
- AWS EC2 / Render

---

### Project 3

AI Ticket Workflow Engine

Stack

- FastAPI
- React
- LangGraph
- MCP
- RAG
- AI Agents
- PostgreSQL
- Docker Compose

---

### Project 4

Fine-Tuned Ticket Classification

Stack

- Hugging Face Transformers
- PEFT
- LoRA
- QLoRA
- PyTorch

---

## Roadmap Complete

By completing all phases, you will have practical experience with:

- Python Engineering
- SQL & PostgreSQL
- Machine Learning
- XGBoost
- Deep Learning
- PyTorch
- LLM Engineering
- Prompt Engineering
- RAG
- Vector Databases
- LangChain
- LangGraph
- MCP
- AI Agents
- MLOps
- Kubernetes
- PEFT
- LoRA
- QLoRA
- Cloud Deployment
- CI/CD
- Docker
- AWS

These skills prepare you for roles such as:

- AI Engineer
- Machine Learning Engineer
- LLM Engineer
- Applied AI Engineer
- Entry-Level MLOps Engineer

The roadmap doesn't end here—it gives you a strong foundation to specialize further as your interests and career evolve.
