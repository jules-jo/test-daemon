---
tags:
- architecture
- repo
- daemon
- agent-loop
type: repo-overview
created: 2026-04-13
updated: 2026-04-16
sources:
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/pyproject.toml
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/__main__.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/ipc/request_handler.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/agent/agent_loop.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/execution/run_pipeline.py
- /workspaces/jules/wiki/pages/daemon/jules-implementation.md
- /workspaces/jules/wiki/log.md
- /home/vscode/.ouroboros/data/interview_interview_20260412_003810.json
- /workspaces/jules/wiki/pages/daemon/phase-2-5-tools.md
---

# Repo Overview

*High-level map of the `jules-daemon` codebase, its entrypoints, and the currently implemented execution paths.*

## Scope

This repository contains a Python package named `jules-daemon`, a single-user daemon for running remote test workflows over SSH with wiki-backed persistence and optional LLM-assisted command handling.

## Current Phase

This snapshot is best understood as a Phase 2.5 codebase: the repo contains the agent-loop foundation that the external planning wiki and Ouroboros interview defined on April 12, 2026, and most of the core scenario plumbing is now present in the default runtime path.

## Entry Points

- `jules-daemon` -> `jules_daemon.__main__:main`
- `jules` -> `jules_daemon.cli_main:main`
- `jules-demo` -> `jules_daemon.demo_runner:main`

The main daemon lifecycle starts in `src/jules_daemon/__main__.py`. It initializes the wiki, attempts crash recovery, runs startup checks, builds the IPC request handler, and then serves the Unix domain socket.

## Main Runtime Flow

1. CLI connects to the daemon over a Unix domain socket.
2. `RequestHandler` validates the incoming verb and routes it.
3. `run` requests choose between two paths:
   - direct shell-like commands -> one-shot approval and execution path
   - natural-language commands -> iterative agent loop when LLM config is present
4. SSH execution is handled by `execution.run_pipeline.execute_run`.
5. Wiki state is written during the run and promoted into history when terminal.
6. `watch`, `status`, `history`, and recovery queries read a mix of wiki state and handler-owned in-memory state.

## Major Packages

- `agent/`: iterative tool-calling loop, OpenAI adapter, approval-gated tools
- `audit/`: audit record models and writers
- `classifier/`: deterministic structured-versus-NL classification
- `cli/`: newer typed CLI parsing and dispatch pipeline
- `execution/`: SSH execution orchestration, output summarization, knowledge extraction
- `ipc/`: socket framing, server, request routing, streaming, notification primitives
- `llm/`: older one-shot translation and summarization stack
- `monitor/`: newer output broadcasting, anomaly detection, queue-draining, and watch helpers
- `ssh/`: connection, reconnect, reattach, and command helpers
- `startup/`: lifecycle, collision detection, crash-recovery wiring
- `thin_client/`: protocol-first client implementation
- `wiki/`: persistence format, layout, current-run state, recovery, promotion, knowledge pages

## Current Architectural Notes

- The packaged `jules` entrypoint still targets `cli_main.py`, while a newer `cli/` pipeline exists in parallel.
- Natural-language `run` requests prefer the agent loop when LLM configuration is available; direct commands intentionally bypass it for latency and backward-compatibility.
- The runtime wiki is not just documentation. It is part of the daemon's persistence model.
- Agent-started `execute_ssh` runs now reuse the daemon's background run lifecycle, so `status` and `read_output` can inspect live partial output before terminal completion.
- The default daemon startup path now instantiates a `NotificationBroadcaster`, and `notify_user` prefers broadcaster-backed delivery when subscribers exist.
- Active-run watch/status/live-output paths now prefer `JobOutputBroadcaster` and default monitor detectors in `RequestHandler`; the older in-memory buffer/queue path remains as a fallback for completed runs and legacy watch behavior.
- Current request interpretation is now daemon-first: `cli_main` forwards nearly all prompts into daemon-side `interpret`, non-action prompts can be answered conversationally from workflow/test knowledge context, and action-like prompts still flow through deterministic validation plus daemon-side LLM interpretation before execution.
- A newer design direction is to model multi-step test workflows explicitly, so Jules can reason about prerequisites such as calibration, answer status queries mid-run, and summarize composite workflows rather than only single commands.
- That workflow direction now has a concrete implementation-plan page describing the proposed records, services, tool primitives, and staged rollout for generic multi-step test execution.
- The current codebase now also supports workflow-aware test-knowledge fields such as `workflow_steps`, `prerequisites`, `artifact_requirements`, and `when_missing_artifact_ask`, plus a first deterministic planner/preflight layer under `src/jules_daemon/workflows/planner.py`.
- The active natural-language run path now uses that workflow-aware knowledge before entering the agent loop: Jules can match a test spec, probe explicit remote artifact paths, ask the user whether to run prerequisite steps first when artifacts are missing or unverifiable, and inject that preflight context into the agent loop and persisted workflow record.
- When the user approves prerequisite execution, the active run path can now bypass the agent loop and use a deterministic background workflow runner: Jules resolves each workflow step to test knowledge, collects any missing step arguments up front, asks for explicit approval per step command, then executes the approved steps sequentially while persisting workflow-step state.
- Workflow steps now have a generic interpreter layer and persisted `parsed_status`, so sequential workflows can expose parsed per-step progress through `status` and emit step-transition alerts through the notification broadcaster while the workflow is running.
- A new integration direction is now scaffolded under `src/jules_daemon/mcp_server.py` and `copilot-sdk-frontend/`: Copilot SDK is intended to become the chat/session frontend while Jules remains the backend runtime exposed through a local MCP adapter.

## Current Snapshot

- Local source file count under `src/jules_daemon/`: 435 files in this workspace snapshot
- Local test file count under `tests/`: 454 files in this workspace snapshot
- `.venv/bin/pytest --collect-only -q` collected 9331 tests on 2026-04-13

## Documentation Drift To Watch

- Older external wiki pages still cite 5146, 5181, or 9298 tests depending on when they were written.
- Older external planning pages still cite an 11-tool baseline or a default around 5 iterations, while the current codebase uses 10 tools and a 15-iteration default.
- The repo-local architecture pages should be preferred over the external planning wiki when the two disagree about current runtime behavior.

## Related

- [Wiki Runtime Architecture](wiki-runtime-architecture.md)
- [Phase 2.5 Design Record](phase-2-5-design-record.md)
- [Phase 2.5 Alignment Checklist](phase-2-5-alignment-checklist.md)
- [Phase 2.5 Implementation Backlog](phase-2-5-implementation-backlog.md)
- [Agent-Driven Request Interpretation](agent-driven-request-interpretation.md)
- [Stateful Test Workflow Orchestration](stateful-test-workflow-orchestration.md)
- [Stateful Test Workflow Implementation Plan](stateful-test-workflow-implementation-plan.md)
- [Project Wiki System](../concepts/project-wiki-system.md)
- [Workflow-Aware Test Knowledge Schema](../concepts/workflow-aware-test-knowledge-schema.md)
