# AI Ticket Workflow Engine

## Goal

Build an enterprise-grade AI-powered ticket management system that automates ticket classification, knowledge retrieval, workflow execution and human approvals using Machine Learning, RAG, AI Agents, LangGraph and MCP.

The project starts as a traditional ticket management application and gradually evolves into an intelligent AI platform capable of assisting support teams through automated workflows.

---

## Why this Project?

Modern enterprises receive thousands of support tickets every day.

Many of these tickets require:

- Ticket classification
- Priority prediction
- Knowledge retrieval
- Multi-step workflows
- Human approvals
- Audit logging
- Secure tool execution

This project teaches how production AI systems automate these workflows safely and reliably.

By the end of the roadmap, this project becomes an enterprise-ready AI automation platform.

---

## Industry Use Cases

- IT Service Desk
- Customer Support
- HR Helpdesk
- Banking Support
- Insurance Claims
- Internal Enterprise Helpdesk
- Technical Support
- Incident Management

---

## Project Timeline

| Phase              | Progress                         |
| ------------------ | -------------------------------- |
| Existing Knowledge | Backend & Frontend Development   |
| Phase 06           | AI Agents, LangGraph & MCP       |
| Phase 09           | Fine-Tuned Ticket Classification |

---

## Final Architecture

```text
                    User Creates Ticket
                             │
                             ▼
                    FastAPI Backend API
                             │
               ┌─────────────┴─────────────┐
               │                           │
               ▼                           ▼
     Ticket Classification          RAG Knowledge Search
               │                           │
               └─────────────┬─────────────┘
                             ▼
                      AI Agent (LangGraph)
                             │
          ┌───────────┬───────────┬────────────┐
          ▼           ▼           ▼            ▼
     Get Ticket   Search KB   Update Ticket  Workflow Log
          │           │           │            │
          └───────────┴───────────┴────────────┘
                             │
                    Human Approval (if required)
                             │
                             ▼
                      Final AI Response
                             │
                             ▼
                     Audit Logs & Database
```

---

# Tech Stack

## Frontend

- React
- JavaScript
- HTML
- CSS

---

## Backend

- FastAPI
- Pydantic
- SQLAlchemy

---

## Database

- PostgreSQL

---

## Machine Learning

- XGBoost
- SHAP

---

## LLM

- Ollama

or

- OpenAI

or

- Google Gemini

---

## RAG

- LangChain
- FAISS
- ChromaDB

---

## AI Agents

- LangGraph

---

## MCP

- MCP Server
- MCP Client
- Custom MCP Tools

---

## Deployment

- Docker
- Docker Compose

---

# Learning Roadmap

## Existing Project

### Goal

Build the complete backend and frontend of a ticket management application.

### Topics Covered

Backend

- Authentication
- REST APIs
- PostgreSQL
- File Upload
- Notifications
- Permissions

Frontend

- React
- API Integration
- UI Components

---

## Phase 06

### Goal

Transform the application into an AI Agent.

### Topics Covered

Machine Learning

- Ticket Classification
- Priority Prediction

Knowledge

- RAG
- Vector Search
- Source Citations

Agents

- Tool Calling
- LangGraph
- State Management

MCP

- MCP Server
- MCP Tools
- Tool Validation

Security

- Prompt Injection Protection
- Tool Permissions
- Human Approval

### Build

- AI Agent
- RAG Integration
- LangGraph Workflow
- MCP Server
- AI Decision Making
- Human Approval Workflow

---

## Phase 09

### Goal

Improve ticket classification using Fine-Tuning.

### Topics Covered

- PEFT
- LoRA
- QLoRA
- Model Evaluation

### Build

- Fine-Tuned Ticket Classifier
- Performance Comparison
- LoRA vs Prompting
- LoRA vs RAG
- LoRA vs XGBoost

---

# AI Workflow

## Step 1

User submits a ticket.

---

## Step 2

Machine Learning model predicts:

- Category
- Priority

---

## Step 3

AI searches the Knowledge Base using RAG.

---

## Step 4

Relevant documents are retrieved.

---

## Step 5

LangGraph Agent decides:

- Answer directly
- Search more documents
- Update ticket
- Escalate ticket
- Request approval

---

## Step 6

The Agent calls MCP tools.

Examples

- Get Ticket
- Update Ticket
- Search Knowledge Base
- Create Workflow Log

---

## Step 7

If confidence is low:

Human approval is requested.

---

## Step 8

The ticket is updated.

Audit logs are stored.

---

# AI Agent Workflow

Agent Capabilities

- Ticket Understanding
- Tool Calling
- Multi-Step Reasoning
- Workflow Execution
- State Management
- Memory
- Error Recovery

---

# MCP Tools

Build the following MCP tools.

Ticket Tools

- Get Ticket
- Update Ticket
- Create Ticket

Knowledge Tools

- Search Knowledge Base
- Retrieve Documents

Workflow Tools

- Create Workflow Log
- Escalate Ticket
- Assign Ticket

---

# Security Features

Authentication

- JWT

Authorization

- Role-Based Access Control

AI Security

- Prompt Injection Protection
- Tool Validation
- Permission Checks
- Human Approval

Auditing

- Audit Logs
- Workflow History

---

# Final Features

Ticket Management

- Create Ticket
- Update Ticket
- Search Ticket
- Track Ticket

Machine Learning

- Category Prediction
- Priority Prediction

Knowledge Assistant

- RAG Search
- Source Citations

AI Agent

- LangGraph Workflow
- Tool Calling
- MCP Integration

Security

- RBAC
- Audit Logs
- Human Approval

Deployment

- Docker Compose

---

# Folder Structure

```text
ai-ticket-workflow-engine/

├── backend/
├── frontend/
├── agents/
├── workflows/
├── mcp/
├── tools/
├── prompts/
├── evaluation/
├── tests/
├── docker-compose.yml
└── README.md
```

---

# Skills Covered

Backend

- FastAPI
- REST APIs
- PostgreSQL

Machine Learning

- Classification
- Explainability

LLM Engineering

- Prompt Engineering
- Structured Outputs

RAG

- Retrieval
- Vector Databases
- Citations

AI Agents

- LangGraph
- Tool Calling
- State Management

MCP

- MCP Server
- MCP Client
- Tool Development

Security

- RBAC
- Prompt Injection Protection
- Human Approval

Deployment

- Docker Compose

---

# Interview Topics Covered

Backend

- FastAPI
- Authentication
- PostgreSQL

Machine Learning

- XGBoost
- Feature Engineering
- SHAP

LLM Engineering

- Prompt Engineering
- Tool Calling
- Structured Outputs

RAG

- Embeddings
- FAISS
- ChromaDB
- Retrieval
- Hallucination Reduction

AI Agents

- LangGraph
- Agent Workflows
- State Management

MCP

- Client/Server Architecture
- Tool Registration
- Tool Security

Enterprise AI

- RBAC
- Audit Logs
- Human-in-the-loop
- AI Workflow Automation

---

# Future Improvements

Possible enhancements after completing the roadmap:

- Multi-Agent Collaboration
- Voice-Based Ticket Creation
- Email & Slack Integration
- Real-Time Notifications
- Automatic Ticket Summarization
- Sentiment Analysis
- Root Cause Analysis
- Calendar & CRM Integrations
- Workflow Analytics Dashboard
- Kubernetes Deployment
