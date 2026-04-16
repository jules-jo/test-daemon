# Copilot SDK Frontend

This directory contains the first Copilot SDK frontend scaffold for Jules.

Current shape:

- **Frontend:** GitHub Copilot SDK interactive REPL
- **Backend:** Jules daemon exposed through a local MCP server
- **Transport:** Copilot SDK -> local stdio MCP server -> Jules daemon socket

## What Works

- chat-style questions against Jules via `jules_chat`
- current status via `jules_status`
- recent history via `jules_history`
- liveness via `jules_health`
- cancel via `jules_cancel`

## What Is Intentionally Deferred

This first slice does **not** expose workflow execution tools over MCP yet.

The missing piece is a clean approval bridge:

- Copilot SDK tool permission approvals
- Jules daemon SSH command approvals
- multi-step workflow execution prompts

That is the next integration slice.

## Prerequisites

1. Jules daemon running from the repo root
2. `uv sync` or equivalent so `.venv` and `src/` are current
3. Copilot CLI/SDK authentication available on the machine

## Install

```bash
cd copilot-sdk-frontend
npm install
```

## Run

From the repository root:

```bash
.venv/bin/python -m jules_daemon --wiki-dir "$HOME/.jules/wiki" --log-level DEBUG
```

In another terminal:

```bash
cd copilot-sdk-frontend
npm start
```

## Environment

- `COPILOT_MODEL`
  - Optional Copilot model name, default: `gpt-5`
- `JULES_SOCKET_PATH`
  - Optional explicit daemon socket path
- `JULES_MCP_COMMAND`
  - Optional override for the Python executable used to launch the MCP server
- `JULES_MCP_ARGS`
  - Optional override for the MCP server launch args

## Notes

The frontend currently restricts Copilot tool permissions to MCP-only use.
Local shell/write/url tools are denied so the session stays focused on Jules.
