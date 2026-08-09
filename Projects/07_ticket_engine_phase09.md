# 🚀 Project 07 - AI Ticket Workflow Engine (Phase 09 - Fine-Tuning Upgrade)

---

## 🎯 Objective

Improve the AI agent system by applying fine-tuning and comparing it with other approaches.

---

## ⚠️ Core Principle

This is NOT a standalone project.

It upgrades:

- Classification layer
- Decision-making quality

---

## 🧠 System Architecture

User Input → Agent → Classification Layer → Tool Selection → MCP → Response

---

## 📁 Project Structure

```text
project/
│
├── src/
│   ├── training/
│   │   └── finetune.py
│   │
│   ├── inference/
│   │   └── predictor.py
│   │
│   └── evaluation/
│       └── compare.py
│
├── data/
│   └── tickets_dataset.json
│
├── models/
│   └── finetuned_model
│
├── notebooks/
│   └── experiments.ipynb
│
└── requirements.txt
```

---

## 🧩 Dataset Format

JSON format:

[
{
"text": "User cannot login",
"label": "authentication_issue"
}
]

---

## 🧩 Training Setup

- Use Hugging Face Transformers
- Use PEFT (LoRA / QLoRA)
- Use lightweight models:
  - distilbert
  - tinyllama

---

## 🧩 Training Steps

1. Load dataset
2. Tokenize data
3. Apply LoRA / QLoRA
4. Train model
5. Save model

---

## 🧩 Evaluation (CRITICAL)

Compare:

- Baseline model
- LoRA model
- QLoRA model
- RAG approach
- Prompt-based approach

Metrics:

- Accuracy
- F1 Score
- Latency

---

## 🧩 Integration

- Replace classifier in agent system
- Use finetuned model for inference

---

## 🧩 Inference Flow

Input → Model → Prediction → Agent Decision

---

## 📦 Output

- Improved classification model
- Performance comparison report

---

## 🚫 Constraints

- Use free GPU (Kaggle/Colab)
- Keep models lightweight

---

## 🎯 Final Outcome

A significantly improved AI system with better decision accuracy
