---
tags:
- copilot-sdk
- mcp
- tools-and-sdks
- integration
type: tooling-note
created: 2026-04-16
updated: 2026-04-16
---

# Copilot SDK Frontend And Jules MCP Runtime

## Summary

Jules is moving toward a split architecture:

- **Frontend:** GitHub Copilot SDK session layer
- **Backend:** Jules daemon and wiki-backed runtime
- **Bridge:** local MCP server that exposes Jules capabilities to the frontend

This avoids rebuilding a generic chat/session substrate inside Jules while
keeping Jules responsible for SSH execution, workflow state, status semantics,
and test knowledge.

## Why This Split

Copilot SDK is a better fit for:

- chat-first interaction
- session persistence
- streaming events
- custom agent selection
- MCP server orchestration

Jules is a better fit for:

- test-specific workflow planning
- SSH target resolution
- prerequisite-aware workflows
- run status and history
- wiki-backed learned test knowledge
- eventual approval-gated execution and escalation

## Current First Slice

The current implementation introduces:

- `src/jules_daemon/mcp_server.py`
  - local stdio MCP server over the existing Jules daemon
- `jules-mcp`
  - Python entrypoint for that server
- `copilot-sdk-frontend/`
  - a minimal interactive Copilot SDK frontend

## Current MCP Tool Surface

The first slice intentionally exposes a safe, read/control-first subset:

- `jules_chat`
- `jules_status`
- `jules_history`
- `jules_health`
- `jules_cancel`

These tools are enough to prove:

- Copilot SDK can act as the chat/session frontend
- Jules can act as the MCP-backed runtime
- workflow/test knowledge can be queried without forcing users into CLI verbs

## Deliberately Deferred

This slice does **not** yet expose full workflow execution tools over MCP.

The missing design work is a clean approval bridge between:

- Copilot SDK tool permissions
- Jules daemon SSH command approvals
- multi-step workflow prompts and confirmations

That is the next implementation target.

## Direction

The intended long-term flow is:

1. user talks to Copilot SDK frontend
2. custom Jules agent uses Jules MCP tools
3. Jules MCP server talks to the daemon/runtime
4. daemon executes workflows, tracks status, interprets results
5. frontend handles session UX, memory, and event streaming
