# Phase 04 - LLM Engineering Foundations

## Goal

Understand how Large Language Models (LLMs) work and build real-world AI applications using APIs and local models.

**Estimated Duration:** 4–5 Weeks

---

## Why this Phase?

This is where you transition from:

👉 ML Engineer → AI Engineer

You will learn:

- How LLMs actually work (practical understanding)
- How to control LLM outputs
- How to build reliable AI systems
- How to use APIs and local models

This phase is REQUIRED before RAG and AI Agents.

---

## Prerequisites

- Complete Phase 03
- Basic understanding of Neural Networks
- Basic understanding of Transformers

---

## 4.1 LLM Fundamentals (Focused)

### Resources

1. Hugging Face LLM Course

https://huggingface.co/learn/llm-course

👉 Complete ONLY these:

- Chapter 1 - Transformers
- Chapter 2 - Using Transformers
- Chapter 3 - Fine-tuning (Concept Only)
- Chapter 5 - Tokenizers

⚠️ Skip heavy training sections

---

2. Generative AI with Large Language Models - DeepLearning.AI

https://www.deeplearning.ai/courses/generative-ai-with-llms/

👉 Optional but Recommended

Focus only on:

- LLM lifecycle
- Pretraining vs Fine-tuning
- Inference concepts
- Evaluation basics

⚠️ Do NOT go deep — use for intuition only

---

### Learn (What Actually Matters)

#### Core Concepts

- What is an LLM
- Tokens & Tokenization
- Context Window
- Temperature, Top-k, Top-p
- Max Tokens & Stop Sequences

#### Transformers (Practical Understanding)

- Encoder vs Decoder
- Attention (intuition only)
- Why Transformers replaced RNNs

#### Prompt Engineering (VERY IMPORTANT)

- Zero-shot
- Few-shot
- Role-based prompting
- Prompt Templates
- Output structuring

---

## 4.2 Local LLMs (Ollama)

### Resource

https://docs.ollama.com/

---

### Learn

- Install Ollama
- Run local models
- Model management
- Basic API usage

---

### Practice

Use models like:

- Llama
- Gemma
- Mistral

---

## 4.3 LLM APIs (Core Engineering)

### Learn

#### API Usage

- Chat-based APIs
- System / User / Assistant roles
- Prompt structuring

#### Structured Outputs (VERY IMPORTANT)

- JSON outputs
- Pydantic validation
- Output parsing
- Response formatting

#### Reliability Engineering

- Retry logic
- Handling invalid outputs
- Rate limits
- Token usage awareness

---

## ⚠️ Cost Policy

- Prefer local models (Ollama)
- Use free API credits only
- Keep requests minimal
- Avoid unnecessary API calls

---

## Practice (Mini Apps)

Build small tools:

- Text Summarizer
- Email Generator
- Resume Improver
- Blog Generator

👉 Focus on:

- Prompt quality
- Output structure
- Reliability

---

## Flagship Project (MANDATORY)

### Structured AI Assistant

---

## Project Goal

Build a production-style AI system that:

- Accepts user input
- Generates structured JSON output
- Validates output
- Handles failures
- Returns reliable responses

---

## Tech Stack

- FastAPI
- Pydantic
- Ollama (Primary)
- OpenAI or Gemini (Optional)

---

## Build Order

1. Setup FastAPI
2. Connect LLM (start with Ollama)
3. Design prompt templates
4. Generate structured JSON
5. Validate using Pydantic
6. Handle invalid outputs
7. Implement retry logic
8. Add logging
9. Dockerize application

---

## Deliverables

Project (Implemented in separate repo)

- Structured AI Assistant

---

## Repository Structure (Project Repo)

```text
structured-ai-assistant/
│
├── app/
├── prompts/
├── schemas/
├── services/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```
