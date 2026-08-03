# Phase 05 - Retrieval-Augmented Generation (RAG)

## Goal

Learn how to build production-ready RAG applications using vector databases, document retrieval, evaluation, and observability.

**Estimated Duration:** 6–8 Weeks

---

## Why this Phase?

Large Language Models don't know your private documents.

RAG allows an LLM to retrieve relevant information before generating a response, making AI applications more accurate, reliable and useful.

This is one of the most in-demand skills for AI Engineers today.

---

## Prerequisites

- Complete Phase 04
- Comfortable with FastAPI
- Understand LLM Fundamentals
- Comfortable using Ollama or an LLM API

---

## 5.1 LangChain for RAG

### Resources

1. LangChain RAG Tutorial

https://python.langchain.com/docs/tutorials/rag/

---

### Learn

- Documents
- Document Loaders
- Text Splitters
- Embeddings
- Retrievers
- Chains
- Prompt Templates
- Output Parsers

> Focus only on the RAG components of LangChain.
> Agent-related topics will be covered in Phase 06.

---

## 5.2 FAISS

### Resource

1. FAISS GitHub Repository

https://github.com/facebookresearch/faiss

---

### Learn

- What is a Vector Database?
- Vector Embeddings
- Similarity Search
- Indexing
- Top-K Retrieval

---

## 5.3 ChromaDB

### Resource

1. Chroma Documentation

https://docs.trychroma.com/

---

### Learn

- Persistent Vector Storage
- Collections
- Metadata
- Filtering

> Use ChromaDB only when persistence is required.
> Start with FAISS first.

---

## 5.4 RAG Evaluation

### Resource

1. Ragas Documentation

https://docs.ragas.io/

---

### Learn

- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy
- Test Dataset Generation

---

## 5.5 Observability

### Resource

1. Arize Phoenix Documentation

https://docs.arize.com/phoenix

---

### Learn

- Tracing
- Retrieval Debugging
- Prompt Debugging
- Response Analysis

---

## Concepts to Learn

### Document Processing

- PDF Loading
- DOCX Loading
- TXT Loading
- Document Chunking

### Embeddings

- Sentence Embeddings
- Embedding Models

### Retrieval

- Similarity Search
- Metadata Filtering
- Top-K Retrieval

### Response Generation

- Prompt Construction
- Context Injection
- Source Citations

### Reliability

- Hallucination Handling
- "I Don't Know" Responses
- Response Grounding

---

## Flagship Project

### RAG Knowledge Assistant

### Tech Stack

- FastAPI
- LangChain
- FAISS
- ChromaDB
- Ollama or OpenAI/Gemini
- Docker

---

### Features

Document Upload

- PDF
- DOCX
- TXT

Knowledge Base

- Document Chunking
- Embedding Generation
- FAISS Index
- ChromaDB Persistence (Optional)

Question Answering

- Context Retrieval
- Source Citations
- Conversation History

Evaluation

- At least 20 Evaluation Questions
- Ragas Evaluation Report

Deployment

- Docker
- Hugging Face Spaces (UI)
- AWS EC2 or Render (Backend)

---

### Build Order

1. Document Upload
2. Document Chunking
3. Embedding Generation
4. FAISS Retrieval
5. Prompt Construction
6. LLM Integration
7. Source Citations
8. Conversation History
9. Ragas Evaluation
10. Phoenix Observability
11. Dockerize Application
12. Deploy Application

---

## Deliverables

Project

- Production RAG Knowledge Assistant

Repository

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

Artifacts

- Dockerized Application
- FAISS Index
- Ragas Evaluation Report
- Phoenix Traces
- Deployment

---

## Skills Gained

- LangChain
- RAG Pipeline
- Document Chunking
- Embeddings
- FAISS
- ChromaDB
- Vector Search
- Source Grounding
- RAG Evaluation
- Phoenix Observability

---

## Completion Criteria

Move to **Phase 06** only after:

- Completed the LangChain RAG Tutorial.
- Built a complete RAG pipeline.
- Implemented document ingestion.
- Implemented FAISS retrieval.
- Added source citations.
- Maintained conversation history.
- Evaluated using Ragas.
- Debugged using Phoenix.
- Dockerized and deployed the application.

---

## Ready for Next Phase

In **Phase 06**, you'll learn AI Agents, LangGraph and MCP to build reliable multi-step AI workflows with tool calling, memory, human approval and structured execution.
