# Phase 04 - LLM Engineering Foundations

## Goal

Understand how Large Language Models (LLMs) work and build AI applications using modern LLM APIs before learning RAG, AI Agents and LangGraph.

**Estimated Duration:** 4–6 Weeks

---

## Why this Phase?

Before building AI Agents or RAG systems, you need to understand how LLMs work.

This phase focuses on using LLMs effectively rather than training them.

You will learn:

- How LLMs generate text
- How to communicate with LLMs
- How to structure AI outputs
- How to run local models
- How to build reliable AI applications

---

## Prerequisites

- Complete Phase 03
- Basic understanding of Neural Networks
- Basic understanding of Transformers

---

## 4.1 LLM Fundamentals

### Resources

1. Hugging Face LLM Course

   https://huggingface.co/learn/llm-course

Complete these chapters:

- Chapter 1 - Transformers
- Chapter 2 - Using Transformers
- Chapter 3 - Fine-tuning Basics (Concept Only)
- Chapter 5 - Tokenizers
- Chapter 8 - Sharing Models

> Skip chapters that focus heavily on training large models. We'll revisit fine-tuning in Phase 09.

---

### Learn

#### LLM Basics

- What is an LLM?
- Tokens
- Tokenization
- Context Window
- Embeddings (Concept)
- Temperature
- Top-k
- Top-p
- Max Tokens
- Stop Sequences

#### Transformers

- Encoder vs Decoder
- Self Attention (Concept)
- Attention Mechanism
- Positional Encoding (Concept)

#### Prompt Engineering

- Zero-shot Prompting
- One-shot Prompting
- Few-shot Prompting
- Role Prompting
- Chain of Thought (Concept)
- Prompt Templates

---

## 4.2 Ollama

### Resource

1. Ollama Documentation

https://docs.ollama.com/

---

### Learn

- Install Ollama
- Download Models
- Run Local Models
- Model Management
- Basic API Usage

Practice using models like:

- Llama
- Gemma
- Mistral

---

## 4.3 LLM APIs

### Resource

1. Generative AI with Large Language Models

https://www.deeplearning.ai/courses/generative-ai-with-llms/

> Optional

Take this course only if it's affordable or available through financial aid.

---

### Learn

Using LLM APIs

- System Prompts
- User Prompts
- Assistant Messages
- Chat Completion APIs

Structured Outputs

- JSON Responses
- Pydantic Validation
- Output Parsing
- Response Validation

Reliability

- Retry Logic
- Error Handling
- Rate Limits
- Cost Optimization
- Token Usage

---

## Practice

Build small programs using an LLM API.

Examples

- Text Summarizer
- Email Generator
- Resume Improver
- Blog Title Generator

Focus on learning prompts and structured outputs.

---

## Flagship Project

### Structured AI Assistant

### Tech Stack

- FastAPI
- Pydantic
- Ollama
- OpenAI or Gemini API

---

### Features

- Accept user input
- Generate structured JSON
- Validate output using Pydantic
- Handle invalid responses
- Retry on parsing failures
- Return consistent API responses

---

### Build Order

1. Setup FastAPI
2. Connect LLM
3. Design Prompts
4. Generate Structured JSON
5. Validate using Pydantic
6. Handle Errors
7. Retry Failed Responses
8. Dockerize Application

---

## Deliverables

Project

- Structured AI Assistant

Repository

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

Artifacts

- FastAPI API
- Prompt Templates
- Pydantic Schemas
- Dockerized Application

---

## Skills Gained

- LLM Fundamentals
- Transformers Basics
- Prompt Engineering
- Ollama
- Local LLMs
- LLM APIs
- Structured Outputs
- Pydantic Validation
- Retry Strategies
- Cost Optimization

---

## Completion Criteria

Move to **Phase 05** only after:

- Completed the Hugging Face LLM Course.
- Used Ollama to run local models.
- Built applications using an LLM API.
- Generated structured JSON responses.
- Validated responses using Pydantic.
- Built the Structured AI Assistant.
- Dockerized the application.

---

## Ready for Next Phase

In **Phase 05**, you'll learn Retrieval-Augmented Generation (RAG), Vector Databases, document ingestion, retrieval pipelines, evaluation using Ragas, observability with Phoenix, and build a production-ready Knowledge Assistant.
