# Phase 05 - Retrieval-Augmented Generation (RAG)

## Goal

Build production-ready RAG systems using document retrieval, vector databases, evaluation, and observability.

**Estimated Duration:** 5–7 Weeks

---

## Why this Phase?

LLMs cannot access your private data.

RAG allows you to:

- Retrieve relevant documents
- Inject context into prompts
- Generate accurate and grounded responses

👉 This is one of the MOST important skills for AI Engineers.

---

## Prerequisites

- Complete Phase 04
- Comfortable with FastAPI
- Understand LLM fundamentals
- Experience with Ollama or LLM APIs

---

## 5.1 RAG Fundamentals (LangChain - Focused)

### Resources

1. LangChain RAG Tutorial

https://python.langchain.com/docs/tutorials/rag/

---

### Learn (Core Only)

- Documents
- Document Loaders
- Text Splitters
- Embeddings
- Retrievers
- Prompt Templates
- Output Parsing

⚠️ Ignore agents for now (covered in Phase 06)

---

## 5.2 Vector Databases (FAISS First)

### Resource

https://github.com/facebookresearch/faiss

---

### Learn

- What are embeddings
- Vector similarity search
- Indexing
- Top-K retrieval

---

## 5.3 Persistence Layer (ChromaDB)

### Resource

https://docs.trychroma.com/

---

### Learn

- Persistent storage
- Collections
- Metadata filtering

👉 Use only after FAISS is clear

---

## 5.4 Evaluation (Ragas)

### Resource

https://docs.ragas.io/

---

### Learn

- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy

👉 Do NOT go too deep — focus on usage

---

## 5.5 Observability (Optional but Valuable)

### Resource

https://docs.arize.com/phoenix

---

### Learn

- Tracing
- Retrieval debugging
- Prompt debugging

👉 Optional but recommended for deeper understanding

---

## Core Concepts (Must Understand)

### Document Processing

- PDF / DOCX / TXT loading
- Chunking strategies

### Embeddings

- Sentence embeddings
- Embedding models (local preferred)

### Retrieval

- Similarity search
- Top-K retrieval
- Metadata filtering

### Generation

- Prompt construction
- Context injection
- Source citations

### Reliability

- Hallucination handling
- "I don't know" fallback
- Grounded responses

---

## ⚠️ Cost Policy

- Prefer local embeddings (SentenceTransformers)
- Prefer local LLM (Ollama)
- Avoid heavy API usage
- Keep datasets small

---

## Flagship Project (MANDATORY)

### RAG Knowledge Assistant

---

## Project Goal

Build a real-world AI system that:

- Accepts documents
- Builds a knowledge base
- Answers questions using retrieved context
- Provides source-backed answers

---

## Tech Stack

- FastAPI
- LangChain
- FAISS (Primary)
- ChromaDB (Optional)
- Ollama (Primary)
- OpenAI / Gemini (Optional)
- Docker

---

## Features

### Document Ingestion

- Upload PDF / DOCX / TXT
- Chunk documents
- Generate embeddings
- Store in FAISS

---

### Question Answering

- Retrieve relevant chunks
- Inject into prompt
- Generate response
- Return source citations

---

### Conversation Support

- Maintain chat history
- Context-aware responses

---

### Evaluation

- Create ~15–20 test questions
- Run Ragas evaluation
- Analyze results

---

### Observability (Optional)

- Trace retrieval
- Debug prompts
- Analyze responses

---

## Build Order (IMPORTANT)

1. Document Upload API
2. Document Chunking
3. Embedding Generation (local)
4. FAISS Index Setup
5. Retrieval Pipeline
6. Prompt Design
7. LLM Integration
8. Add Source Citations
9. Add Conversation Memory
10. Ragas Evaluation
11. (Optional) Phoenix Observability
12. Dockerize Application
13. Deploy (FREE option preferred)

---

## Deployment (FREE FIRST)

- Backend → Render / Railway
- UI (optional) → Hugging Face Spaces
- AWS EC2 → Optional (Free Tier only)

---

## Deliverables

Project (Implemented in separate repo)

- RAG Knowledge Assistant

---

## Repository Structure (Project Repo)

```text
rag-knowledge-assistant/
│
├── app/
├── documents/
├── embeddings/
├── vectorstore/
├── prompts/
├── evaluation/
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```
