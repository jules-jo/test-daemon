---
tags:
- architecture
- wiki
- persistence
- crash-recovery
type: wiki-runtime-architecture
created: 2026-04-13
updated: 2026-04-13
sources:
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/wiki/layout.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/wiki/current_run.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/startup/crash_recovery_wire.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/wiki/recovery_orchestrator.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/wiki/README.md
---

# Wiki Runtime Architecture

*How the repo-local `wiki/` directory doubles as the daemon's persistence backbone and how that interacts with user-maintained project pages.*

## Core Idea

In this repository, `wiki/` is not only a knowledge base. It is also the daemon's state store. The `jules_daemon.wiki` package reads and writes markdown files with YAML frontmatter to track current runs, history, queue entries, audit records, translations, and learned test knowledge.

## Layout Split

The code explicitly separates wiki ownership into daemon-managed and user-managed areas.

### Daemon-Managed

- `pages/daemon/`
- `pages/daemon/history/`
- `pages/daemon/results/`
- `pages/daemon/translations/`
- `pages/daemon/audit/`
- `pages/daemon/audit/archive/`
- `pages/daemon/queue/`
- `pages/daemon/knowledge/`

These directories are runtime state. The daemon creates and updates their contents.

### User-Managed

- `pages/agents/`
- `pages/architecture/`
- `pages/concepts/`
- `pages/security/`
- `pages/tools-and-sdks/`
- `raw/`
- `schema/`

These directories are where the repo-local knowledge base should live.

## Important Runtime Records

- `pages/daemon/current-run.md`: active state machine record for the current run
- `pages/daemon/history/`: promoted terminal runs
- `pages/daemon/results/`: assembled result records
- `pages/daemon/translations/`: NL-to-command learning artifacts
- `pages/daemon/audit/`: per-command audit trail
- `pages/daemon/queue/`: queued run requests
- `pages/daemon/knowledge/`: per-test learned knowledge pages

## Lifecycle

### Startup

`initialize_wiki()` creates the expected directory structure and README stubs if they are missing. The daemon then attempts crash recovery before opening the IPC server.

### Active Run

`current_run.py` persists the current state through transitions such as idle, pending approval, running, completed, failed, and cancelled. The markdown body contains a human-readable snapshot while frontmatter stores the structured state.

### Completion

After a run terminates, promotion logic moves durable records into history and result pages. Audit and learned knowledge may be updated as part of the same broader execution flow.

### Crash Recovery

`startup/crash_recovery_wire.py` bridges startup into the deeper recovery orchestration code. The current wiring is best-effort and may mark interrupted runs as failed when full reattachment is unavailable.

## Repo-Wiki Implication

Because the repo ships with a `wiki/` directory that is also used at runtime, architecture notes and persistent project documentation must stay out of `pages/daemon/*`. The safe place for repo-level wiki content is the user-managed area created by the layout module.

## Current Observation

The repo-local wiki created for codebase orientation is compatible with the runtime layout because it only adds files to user-managed directories plus root special files such as `index.md` and `log.md`.

## Related

- [Repo Overview](repo-overview.md)
- [Project Wiki System](../concepts/project-wiki-system.md)
