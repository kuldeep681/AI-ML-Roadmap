# Phase 06 - AI Agents, LangGraph & MCP

## Goal

Build reliable AI Agents using LangGraph and MCP that can perform real-world tasks using tools, workflows, and structured execution.

**Estimated Duration:** 5–7 Weeks

---

## Why this Phase?

LLMs alone cannot perform real-world tasks.

AI Agents enable LLMs to:

- Use tools (APIs, DBs)
- Execute workflows
- Maintain state
- Make decisions
- Request human approval

👉 This is how real production AI systems are built.

---

## Prerequisites

- Complete Phase 05
- Comfortable with RAG pipelines
- Comfortable with FastAPI
- Understand structured outputs (JSON + Pydantic)

---

## 6.1 Tool Calling Fundamentals

### Resources

1. OpenAI Function Calling Guide  
   https://platform.openai.com/docs/guides/function-calling

2. Google Gemini Function Calling  
   https://ai.google.dev/gemini-api/docs/function-calling

---

### Learn (Core Only)

- Function / Tool calling
- JSON schema for tools
- Tool input/output validation
- Structured outputs

#### Agent Basics

- Tool selection
- Multi-step reasoning (basic understanding)
- Error handling

---

## 6.2 LangGraph (Core Engine)

### Resource

https://langchain-ai.github.io/langgraph/tutorials/introduction/

---

### Learn (Important)

#### Core Concepts

- Nodes
- Edges
- State
- Graph execution
- Conditional routing

#### Advanced (Focus, don’t overdo)

- Persistence
- Memory
- Checkpoints
- Human-in-the-loop

---

## 6.3 Model Context Protocol (MCP)

### Resources

1. https://huggingface.co/learn/mcp-course/en/unit0/introduction
2. https://huggingface.co/learn/mcp-course/en/unit1/introduction
3. https://huggingface.co/learn/mcp-course/unit3/build-mcp-server

---

### Learn

#### Core Concepts

- MCP architecture
- MCP client & server
- Tools & resources

#### Implementation

- Build MCP server
- Register tools
- Tool permissions
- Tool validation

---

## 6.4 Agent Reliability & Security

### Learn

#### Security

- Prompt injection
- Tool injection
- Permission control
- Input validation
- Output validation

#### Reliability

- Retry logic
- Error recovery
- Logging
- Audit trail

#### Human-in-the-loop

- Approval workflows
- Low-confidence handling

---

## ⚠️ Cost Policy

- Prefer local LLMs (Ollama)
- Use APIs only when needed
- Avoid unnecessary tool calls
- Keep workflows lightweight

---

## Core Concepts (Must Understand)

### AI Agents

- Agent architecture
- Planning vs execution
- Tool usage
- State management

### LangGraph

- Workflow graphs
- Conditional execution
- Persistent state

### MCP

- Tool-based communication
- Safe execution layer

---

## Flagship Project (MANDATORY)

### AI Ticket Workflow Engine (Upgrade)

👉 Continue your existing project

---

## Project Goal

Build a production-level AI system that:

- Automates ticket workflows
- Uses tools safely
- Executes multi-step logic
- Integrates ML + RAG + Agents

---

## Tech Stack

- FastAPI
- React (optional UI)
- LangGraph
- MCP
- Ollama (Primary)
- OpenAI/Gemini (Optional)
- PostgreSQL
- Docker Compose

---

## Features

### ML Integration

- Ticket classification
- Priority prediction

---

### Knowledge (RAG)

- Search support documents
- Context-aware responses

---

### Agent Workflow

- Multi-step LangGraph flow
- Decision-based routing

---

### MCP Tools

- Get ticket
- Update ticket
- Search knowledge base
- Create workflow logs

---

### Security

- Role-based access control (RBAC)
- Tool permissions

---

### Reliability

- Human approval (low confidence)
- Retry logic
- Logging

---

### Observability (Basic)

- Execution logs
- Workflow tracing (simple)

---

## Build Order (IMPORTANT)

1. Integrate ML model (classification)
2. Integrate RAG pipeline
3. Build basic LangGraph workflow
4. Add tool calling logic
5. Build MCP server
6. Register tools
7. Add conditional routing
8. Add human approval step
9. Add RBAC
10. Add logging & audit logs
11. Docker Compose setup
12. Deploy (free-first approach)

---

## Deployment (FREE FIRST)

- Backend → Render / Railway
- Database → Free tier PostgreSQL (or local)
- AWS EC2 → Optional (Free Tier only)

---

## Optional (After Completion)

### CrewAI

Resource: https://docs.crewai.com/

👉 Learn only after completing this phase  
👉 Not required for roadmap

---

## Deliverables

Project (Implemented in separate repo)

- AI Ticket Workflow Engine (Production Version)

---

## Repository Structure (Project Repo)

```text
ai-ticket-workflow-engine/
│
├── backend/
├── frontend/
├── agents/
├── workflows/
├── mcp/
├── tools/
├── prompts/
├── tests/
├── docker-compose.yml
└── README.md
```
