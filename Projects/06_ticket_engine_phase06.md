# 🚀 Project 06 - AI Ticket Workflow Engine (Phase 06 - Agent System)

---

## 🎯 Objective

Build a production-style AI system that automates ticket workflows using:

- Agent-based decision making
- LangGraph workflow orchestration
- MCP-based tool execution
- FastAPI for interaction

---

## ⚠️ Core Requirements

- Deterministic workflow execution (no randomness)
- Clear agent state management
- Tool-driven actions ONLY
- Modular architecture for future upgrades

---

## 🧠 System Architecture

User Input → API → Agent → LangGraph Workflow → Tool Calls → MCP Server → Database → Response

---

## 📁 Project Structure

```text
project/
│
├── src/
│ ├── api/
│ │ ├── main.py
│ │ └── routes.py
│ │
│ ├── agent/
│ │ ├── agent.py
│ │ └── state.py
│ │
│ ├── graph/
│ │ └── workflow.py
│ │
│ ├── tools/
│ │ ├── get_ticket.py
│ │ ├── update_ticket.py
│ │ └── search_kb.py
│ │
│ ├── mcp/
│ │ └── client.py
│ │
│ ├── db/
│ │ └── models.py
│ │
│ └── services/
│ └── classifier.py
│
├── docker/
│ └── Dockerfile
│
├── requirements.txt
└── run.py
```

---

## 🧩 Agent State (CRITICAL)

State must include:

- ticket_id
- ticket_text
- classification
- action
- tool_response
- status

---

## 🧩 Workflow Design (LangGraph)

Define nodes:

1. classify_ticket
2. decide_action
3. execute_tool
4. finalize_response

---

## 🧩 Tool Interface

Each tool must follow:

Input:

- ticket_id or query

Output:

- structured JSON response

---

## 🧩 MCP Integration

- MCP acts as execution layer
- Agent sends tool requests
- MCP returns result

---

## 🧩 Database (PostgreSQL)

Ticket schema:

- id
- title
- description
- status
- priority
- created_at

---

## 🧩 Implementation Steps

1. Build FastAPI endpoint
2. Define agent state
3. Create LangGraph workflow
4. Implement tools
5. Integrate MCP client
6. Connect PostgreSQL
7. Execute end-to-end flow

---

## 📦 Output

- Working AI workflow system
- Automated ticket handling

---

## 🚫 Constraints

- No fine-tuning yet
- Keep classification simple (ML or rules)

---

## 🔮 Future Compatibility

Will be extended with:

- Fine-tuning (Phase 09)
- Advanced reasoning
- Multi-agent systems

---

## 🎯 Final Outcome

A production-style AI agent system capable of automating ticket workflows
