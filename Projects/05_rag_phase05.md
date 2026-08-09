# Project 05 - RAG Knowledge Assistant (Full System)

## Phase Mapping

- Phase 05
- Extends: Project 04

---

## Goal

Build a complete RAG system that answers user queries using retrieved knowledge and LLM.

---

## What You Already Have

From Phase 04:

- Document chunks
- Embeddings
- Vector database (FAISS)

---

## What You Will Build NOW

### Full RAG Pipeline

- Query processing
- Retrieval
- Context injection
- LLM response generation
- API layer

---

## Tech Stack

- LangChain
- FAISS / ChromaDB
- Ollama (preferred) / OpenAI / Gemini
- FastAPI
- Docker

---

## Architecture

```text
User Query
   ↓
Embedding
   ↓
Vector Search
   ↓
Top-K Documents
   ↓
Context Injection
   ↓
LLM
   ↓
Final Answer
```
