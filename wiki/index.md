---
tags:
- wiki-structure
- index
- project
- jules-daemon
type: wiki-index
created: 2026-04-13
updated: 2026-04-16
sources:
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/AGENTS.md
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/wiki/schema/AGENTS.md
---

# Wiki Index

Content-oriented catalog for the repo-local codebase wiki and the daemon's runtime persistence layout.

## Special Files

- [log.md](log.md) -- chronological record of wiki ingest, updates, and notable repo observations
- [schema/AGENTS.md](schema/AGENTS.md) -- maintenance contract for future agents working in this repo

## Architecture

- [Repo Overview](pages/architecture/repo-overview.md) -- top-level map of entrypoints, packages, execution modes, and known doc drift
- [Wiki Runtime Architecture](pages/architecture/wiki-runtime-architecture.md) -- how `wiki/` functions as daemon persistence and where user-maintained docs should live
- [Phase 2.5 Design Record](pages/architecture/phase-2-5-design-record.md) -- pre-build design decisions, tool-surface evolution, and acceptance criteria for the agent-loop foundation
- [Phase 2.5 Alignment Checklist](pages/architecture/phase-2-5-alignment-checklist.md) -- concrete aligned versus partial versus not-yet-aligned comparison against the interview and scenario
- [Phase 2.5 Implementation Backlog](pages/architecture/phase-2-5-implementation-backlog.md) -- prioritized follow-up work needed to close the remaining scenario gaps
- [Agent-Driven Request Interpretation](pages/architecture/agent-driven-request-interpretation.md) -- daemon-side interpret path is now chat-first for informational prompts and still uses structured LLM interpretation plus deterministic enforcement for action requests
- [Stateful Test Workflow Orchestration](pages/architecture/stateful-test-workflow-orchestration.md) -- target architecture and state model for prerequisite-aware multi-step test workflows such as `LT -> calibration -> LT result summary`
- [Stateful Test Workflow Implementation Plan](pages/architecture/stateful-test-workflow-implementation-plan.md) -- concrete primitives, module boundaries, and phased delivery plan for generic multi-step test workflows

## Concepts

- [Project Wiki System](pages/concepts/project-wiki-system.md) -- local adaptation of the Karpathy LLM Wiki pattern for this repository
- [Workflow-Aware Test Knowledge Schema](pages/concepts/workflow-aware-test-knowledge-schema.md) -- exact frontmatter fields and example markdown for prerequisite-aware test knowledge pages

## Tools And SDKs

- [Copilot SDK Frontend And Jules MCP Runtime](pages/tools-and-sdks/copilot-sdk-frontend-and-jules-mcp-runtime.md) -- staged architecture for using GitHub Copilot SDK as the chat/session frontend while Jules remains the daemon-backed MCP runtime

## Daemon-Managed Directories

These directories are automatically managed by the daemon.
Do not manually edit files in daemon-managed directories.

- **daemon** (`pages/daemon/`) -- Daemon state files (current-run, recovery-log)
- **history** (`pages/daemon/history/`) -- Completed run history (one file per terminal run)
- **results** (`pages/daemon/results/`) -- Assembled test results (one file per run)
- **translations** (`pages/daemon/translations/`) -- NL-to-command translation mappings for learning
- **audit** (`pages/daemon/audit/`) -- Per-command audit trail (one file per execution event)
- **archive** (`pages/daemon/audit/archive/`) -- Archived audit logs (moved with explicit user approval)
- **queue** (`pages/daemon/queue/`) -- Pending command queue (one file per queued command)
- **knowledge** (`pages/daemon/knowledge/`) -- Per-test learned knowledge (one file per known test command)

## User-Managed Directories

These directories are for user-curated knowledge and notes.

- **agents** (`pages/agents/`) -- Agent documentation and research
- **architecture** (`pages/architecture/`) -- Architecture notes and design decisions
- **concepts** (`pages/concepts/`) -- General concepts and knowledge base
- **security** (`pages/security/`) -- Security notes, audits, and patterns
- **systems** (`pages/systems/`) -- Named system definitions for SSH target aliases
- **tools-and-sdks** (`pages/tools-and-sdks/`) -- Tool and SDK documentation
- **raw** (`raw/`) -- Unprocessed research notes and raw material
- **schema** (`schema/`) -- Schema documentation and reference
