# Project 04 - RAG Knowledge Assistant (Foundations)

## Phase Mapping

- Phase 04 ONLY
- This is the BASE for RAG system
- Will be extended in Phase 05

---

## Goal

Understand and implement the core building blocks required for a RAG system.

---

## What You Are Building NOW

### Core Components

- Text preprocessing
- Chunking strategy
- Embedding generation
- Vector storage

---

## Tech Stack

- Python
- Sentence Transformers / OpenAI Embeddings
- FAISS (preferred for simplicity)

---

## Concepts to Implement

### 1. Text Processing

- Load documents (PDF / text)
- Clean text
- Split into chunks

---

### 2. Chunking

- Fixed size chunks
- Overlap strategy

---

### 3. Embeddings

- Convert text → vectors
- Use embedding model

---

### 4. Vector Storage

- Store embeddings in FAISS
- Basic similarity search

---

## Build Steps

1. Load sample documents
2. Clean and preprocess text
3. Split into chunks
4. Generate embeddings
5. Store in FAISS
6. Perform similarity search

---

## Output

- Working vector database
- Ability to retrieve relevant chunks

---

## IMPORTANT (Next Phase)

⚠️ This project will be EXTENDED in Phase 05

You will ADD:

- LLM integration
- Query → response pipeline
- FastAPI
- Evaluation (Ragas)
- Docker deployment

---

## Constraints

- Keep it simple
- Use local embeddings if possible
- No heavy APIs required

---

## Final Outcome

A working retrieval system (without LLM yet)
