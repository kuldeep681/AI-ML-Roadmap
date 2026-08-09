# Project 06 - AI Ticket Workflow Engine (Agent System)

## Phase Mapping

- Phase 06 ONLY
- This is the BASE AI system
- Will be extended in Phase 09

---

## Goal

Build a production-style AI system that automates ticket workflows using Agents, LangGraph, and MCP.

---

## What You Are Building NOW

### Core AI System

- Ticket classification
- Multi-step workflow execution
- Tool-based decision system

---

## Tech Stack

- FastAPI
- LangGraph
- MCP
- PostgreSQL
- Docker

---

## Features

### AI Layer

- Ticket classification (ML or rule-based)
- Decision making
- Multi-step workflow

---

### Agent System

- Tool usage
- Conditional routing
- State management

---

### MCP Tools

- Get ticket
- Update ticket
- Search knowledge base

---

## Architecture

```text
User Input
   ↓
Agent (Decision)
   ↓
LangGraph Workflow
   ↓
Tool Calls
   ↓
MCP Server
   ↓
Database / APIs
   ↓
Response
```
