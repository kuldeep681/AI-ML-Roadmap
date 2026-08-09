# 🚀 Project 05 - RAG Knowledge Assistant (Phase 05 - Full System)

---

## 🎯 Objective

Build a complete RAG system that answers user queries using retrieved knowledge and LLM.

---

## 🧠 System Architecture

User Query → Embedding → Vector Search → Top-K Chunks → Context Injection → LLM → Response

---

## 📁 Project Structure

```text
project/
│
├── src/
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   ├── llm/
│   │   └── generator.py
│   │
│   ├── pipeline/
│   │   └── rag_pipeline.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   └── routes.py
│   │
│   └── prompts/
│       └── template.txt
│
├── Dockerfile
├── requirements.txt
└── run.py
```

---

## ⚙️ Configuration

config.yaml should include:

- top_k: 5
- temperature: 0.7
- model_name: llama3

---

## 🧩 Implementation Steps

### Step 1 — Query Processing

- Accept user query
- Clean input

---

### Step 2 — Retrieval

- Convert query → embedding
- Retrieve top_k documents

---

### Step 3 — Context Injection

- Combine retrieved chunks into context
- Limit token size

---

### Step 4 — Prompt Template

Example:

You are a helpful assistant.
Answer the question based ONLY on the context below.

Context:
{context}

Question:
{question}

Answer:

---

### Step 5 — LLM Integration

- Use Ollama (preferred) or OpenAI
- Generate response from prompt

---

### Step 6 — RAG Pipeline

Combine:

- Query → Retrieval → Prompt → LLM → Output

---

### Step 7 — API (FastAPI)

POST /ask

Input:
{
"query": "your question"
}

Output:
{
"answer": "generated response"
}

---

### Step 8 — Dockerization

- Containerize FastAPI app
- Lightweight image

---

## 📦 Output

- Working RAG system
- API endpoint
- LLM-powered answers

---

## 🚫 Constraints

- Optimize for latency
- Keep context small
- Avoid hallucination

---

## 🎯 Final Outcome

A complete RAG system capable of answering queries using retrieved knowledge + LLM
