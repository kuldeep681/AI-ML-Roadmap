# 🚀 Project 04 - RAG Knowledge Assistant (Phase 04 - Foundations)

---

## 🎯 Objective

Build the core retrieval system required for a Retrieval-Augmented Generation (RAG) pipeline.

This phase focuses ONLY on:

- Document processing
- Chunking
- Embedding
- Vector storage
- Retrieval

No LLM is used in this phase.

---

## ⚠️ Core Requirements

- Modular design
- No hardcoded paths
- Reusable for next phase
- Efficient vector search

---

## 🧠 System Architecture

Documents → Preprocessing → Chunking → Embedding → Vector Store (FAISS) → Retrieval

---

## 📁 Project Structure

```text
project/
│
├── data/
│ └── documents/
│
├── src/
│ ├── data/
│ │ └── loader.py
│ │
│ ├── processing/
│ │ └── text_cleaner.py
│ │
│ ├── chunking/
│ │ └── chunker.py
│ │
│ ├── embeddings/
│ │ └── embedder.py
│ │
│ ├── vectorstore/
│ │ └── faiss_store.py
│ │
│ └── retrieval/
│ └── retriever.py
│
├── vector_db/
│ └── faiss_index
│
├── config.yaml
└── run.py
```

---

## ⚙️ Configuration

config.yaml should define:

- chunk_size: 500
- chunk_overlap: 50
- embedding_model: all-MiniLM-L6-v2
- top_k: 5

---

## 🧩 Implementation Steps

### Step 1 — Load Documents

- Support PDF and text files
- Extract raw text

---

### Step 2 — Text Preprocessing

- Remove noise
- Normalize whitespace
- Clean unwanted characters

---

### Step 3 — Chunking

- Fixed-size chunking
- Use overlap to preserve context

Example:

- chunk_size = 500
- overlap = 50

---

### Step 4 — Embedding Generation

- Use SentenceTransformers
- Convert each chunk → vector

---

### Step 5 — Vector Storage (FAISS)

- Create FAISS index
- Store embeddings
- Save index locally

---

### Step 6 — Retrieval

- Convert query → embedding
- Perform similarity search
- Return top_k chunks

---

## 📦 Output

- FAISS index stored locally
- Retrieval function working

---

## 🚫 Constraints

- No LLM usage
- No API
- Keep system lightweight

---

## 🔮 Future Compatibility

Will be extended with:

- LLM (Phase 05)
- API
- Evaluation
- Docker

---

## 🎯 Final Outcome

A working semantic search system capable of retrieving relevant document chunks
