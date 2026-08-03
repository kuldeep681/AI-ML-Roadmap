# Phase 06 - AI Agents, LangGraph & MCP

## Goal

Learn how to build reliable AI Agents using LangGraph and Model Context Protocol (MCP), enabling LLMs to safely interact with external tools, APIs, and workflows.

**Estimated Duration:** 6–8 Weeks

---

## Why this Phase?

LLMs can answer questions, but they cannot perform real-world tasks on their own.

AI Agents extend LLMs by allowing them to:

- Use external tools
- Access databases
- Search knowledge bases
- Call APIs
- Execute multi-step workflows
- Request human approval when needed

This phase teaches you how production AI Agents are built.

---

## Prerequisites

- Complete Phase 05
- Comfortable with RAG
- Comfortable with FastAPI
- Understand Structured Outputs

---

## 6.1 Tool Calling Fundamentals

### Resources

1. OpenAI Function Calling Guide

https://platform.openai.com/docs/guides/function-calling

2. Google Gemini Function Calling

https://ai.google.dev/gemini-api/docs/function-calling

> Learn the concepts even if you primarily use local models.

---

### Learn

Tool Calling

- Function Calling
- Tool Definitions
- Tool Parameters
- Structured Outputs
- JSON Schema
- Tool Validation

Agent Basics

- Tool Selection
- Multi-step Reasoning
- Error Handling

---

## 6.2 LangGraph

### Resources

1. LangGraph Introduction

https://langchain-ai.github.io/langgraph/tutorials/introduction/

---

### Learn

Core Concepts

- Nodes
- Edges
- State
- Graph Execution
- Conditional Routing

Advanced Concepts

- Persistence
- Memory
- Checkpoints
- Human-in-the-loop
- Error Recovery

---

## 6.3 Model Context Protocol (MCP)

### Resources

1. Hugging Face MCP Course - Unit 0

https://huggingface.co/learn/mcp-course/en/unit0/introduction

2. Hugging Face MCP Course - Unit 1

https://huggingface.co/learn/mcp-course/en/unit1/introduction

3. Build an MCP Server

https://huggingface.co/learn/mcp-course/unit3/build-mcp-server

---

### Learn

MCP Basics

- MCP Architecture
- MCP Client
- MCP Server
- Resources
- Tools

Implementation

- Build an MCP Server
- Register Tools
- Tool Permissions
- Tool Validation

---

## 6.4 AI Agent Security

### Learn

Security

- Prompt Injection
- Tool Injection
- Permission Control
- Input Validation
- Output Validation

Reliability

- Retry Logic
- Error Recovery
- Logging
- Audit Trail

Human Approval

- Human Review
- Approval Workflow
- High-Risk Action Handling

---

## Concepts to Learn

### AI Agents

- Agent Architecture
- Planning
- Tool Usage
- State Management

### LangGraph

- Graph-based Workflows
- State Persistence
- Conditional Execution

### MCP

- Client-Server Communication
- Tool Registration
- Tool Execution

---

## Flagship Project

### AI Ticket Workflow Engine

Upgrade your existing project.

---

### Tech Stack

- FastAPI
- React
- LangGraph
- MCP
- Ollama or OpenAI/Gemini
- PostgreSQL
- Docker

---

### Add Features

Machine Learning

- Ticket Classification
- Priority Prediction

Knowledge

- RAG over Support Documents

Workflow

- LangGraph Workflow
- Multi-step Execution

MCP Tools

- Get Ticket
- Update Ticket
- Search Knowledge Base
- Create Workflow Log

Security

- Role-Based Access Control (RBAC)
- Tool Permissions
- Audit Logs

Reliability

- Human Approval for Low Confidence Actions

Deployment

- Docker Compose

---

### Build Order

1. Integrate Ticket Classification
2. Integrate RAG
3. Build LangGraph Workflow
4. Build MCP Server
5. Register MCP Tools
6. Add Human Approval
7. Add RBAC
8. Add Audit Logs
9. Docker Compose
10. Deploy Application

---

## Optional

### CrewAI

Learn CrewAI **only after** completing this phase.

Resource

https://docs.crewai.com/

> This is optional and not required for your roadmap.

---

## Deliverables

Project

- AI Ticket Workflow Engine (Production Version)

Repository

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

Artifacts

- LangGraph Workflow
- MCP Server
- AI Agent
- Docker Compose Setup
- Audit Logs
- Human Approval Workflow

---

## Skills Gained

- AI Agents
- Tool Calling
- Structured Outputs
- LangGraph
- Workflow Design
- State Management
- MCP
- Human-in-the-loop
- RBAC
- Prompt Injection Defense
- AI Workflow Development

---

## Completion Criteria

Move to **Phase 07** only after:

- Completed the LangGraph Introduction Tutorial.
- Completed the Hugging Face MCP Course.
- Built an MCP Server.
- Built a LangGraph workflow.
- Integrated RAG into the AI Ticket Workflow Engine.
- Implemented tool calling.
- Added human approval.
- Added RBAC.
- Added audit logs.
- Dockerized the application.

---

## Ready for Next Phase

In **Phase 07**, you'll learn real-world MLOps by versioning data, tracking experiments, automating training pipelines, monitoring models, and deploying reproducible Machine Learning systems using MLflow, DVC, Prefect, Pandera and Evidently.
