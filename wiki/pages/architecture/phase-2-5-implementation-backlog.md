---
tags:
- architecture
- phase-2-5
- backlog
- agent-loop
type: implementation-backlog
created: 2026-04-13
updated: 2026-04-13
sources:
- /home/vscode/.ouroboros/data/interview_interview_20260412_003810.json
- /workspaces/jules/wiki/pages/daemon/phase-2-5-tools.md
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/__main__.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/ipc/request_handler.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/agent/tools/execute_ssh.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/agent/ipc_bridge.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/agent/tools/lookup_test_spec.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/execution/test_discovery.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/wiki/test_knowledge.py
---

# Phase 2.5 Implementation Backlog

*Shortest path from the current Phase 2.5 foundation to the fuller pre-build scenario.*

## Status Update

Implemented on 2026-04-13:

- agent-loop `execute_ssh` now launches daemon-managed background runs and shares the one-shot lifecycle
- `read_output` and `status` can inspect live partial output from active runs
- default daemon startup now instantiates a `NotificationBroadcaster`
- notification subscriptions are handled by `RequestHandler`, and `notify_user` now prefers broadcaster-backed delivery
- `summary_fields` is now a first-class test-knowledge field returned by `lookup_test_spec` and consumed by `parse_test_output`, `status`, and `summarize_run`
- active watch/status/live-output now prefer `JobOutputBroadcaster`, and the default runtime collects/emits conservative monitor alerts for error keywords and failure-rate spikes

Main remaining gaps from the original backlog:

- the shipped status path is still procedural enrichment, not a literal end-to-end agent/tool invocation
- the default detector set is intentionally conservative; stall-style triggers and richer LLM-on-trigger orchestration are still future work
- some repo-local architecture notes still need ongoing maintenance as the runtime evolves

## Priority Order

This backlog is ordered by scenario alignment impact, not by implementation neatness.
The first four items were substantially completed on 2026-04-13. The remaining work is now narrower: status design choices, future monitor-trigger expansion, and ongoing doc cleanup.

## P0: Make agent-loop execution backgrounded and monitorable

Current gap:

- `execute_ssh` awaits `execute_run()` to terminal completion, so the agent-loop path cannot start a run and then inspect it while it is still active.

Why this matters:

- this is the largest mismatch with the interview scenario
- `read_output`, `watch`, and `status` cannot behave like scenario tools if agent-started runs are already finished before control returns

Starting files:

- `src/jules_daemon/agent/tools/execute_ssh.py`
- `src/jules_daemon/ipc/request_handler.py`
- `src/jules_daemon/execution/run_pipeline.py`

Implementation direction:

- split "start run" from "wait for run completion"
- make the agent-loop execution path reuse the daemon's background-run state, current-run wiki updates, and output-stream plumbing instead of creating a separate blocking execution mode
- treat one-shot and agent-started runs as different front doors into the same run lifecycle

Done when:

- `execute_ssh` returns a `run_id` quickly after launch
- `status` can report an active run started by the agent loop
- `read_output` can retrieve partial output before the run ends
- `watch` and completion handling work the same way for one-shot and agent-started runs

## P1: Wire persistent notifications as the default runtime path

Current gap:

- `RequestHandlerConfig` supports a broadcaster, but daemon startup does not create one by default
- `notify_user` currently writes a direct `STREAM` message to the active client instead of going through the subscription channel

Why this matters:

- the scenario assumes completion and alert notifications survive beyond the single in-flight request
- without default broadcaster wiring, push-style behavior stays optional infrastructure instead of shipped behavior

Starting files:

- `src/jules_daemon/__main__.py`
- `src/jules_daemon/ipc/request_handler.py`
- `src/jules_daemon/agent/ipc_bridge.py`
- `src/jules_daemon/ipc/notification_broadcaster.py`

Implementation direction:

- instantiate a `NotificationBroadcaster` in normal daemon startup
- route agent-loop notifications through the same broadcaster-backed delivery path used for daemon notifications
- keep direct request/response streaming for interactive prompts, but stop treating that as the long-lived notification mechanism

Done when:

- the default daemon process owns a broadcaster
- completion notifications from both one-shot and agent-loop runs reach subscribed clients
- `notify_user` can reach subscribers even if the original requesting client is gone

## P1: Add scenario-grade status summarization

Current gap:

- `status` is still a procedural handler over in-memory state and wiki state
- it does not compose `lookup_test_spec`, `read_output`, and `parse_test_output` into a test-specific status summary

Why this matters:

- the interview scenario depends on richer "what is happening in this test" answers, not only generic running/completed state
- this is the feature that makes the tool catalog feel agentic instead of only agent-triggered

Starting files:

- `src/jules_daemon/ipc/request_handler.py`
- `src/jules_daemon/agent/tools/lookup_test_spec.py`
- `src/jules_daemon/agent/tools/read_output.py`
- `src/jules_daemon/agent/tools/parse_test_output.py`

Implementation direction:

- keep the existing procedural `status` path as the reliable base
- add a shared status-enrichment layer that can attach test-aware summaries when a matching spec and readable output exist
- avoid making basic `status` depend on a full agent-loop invocation if a smaller shared summarizer is enough

Done when:

- `status` still works with no spec present
- known tests return a structured or clearly summarized test-specific status
- the same summary logic can be reused for completion reporting

## P2: Promote test-catalog schema to a first-class contract

Current gap:

- planning artifacts assume fields such as `summary_fields`
- current code persists knowledge under `pages/daemon/knowledge/`, which still differs from the older planned `tests/` namespace

Why this matters:

- the scenario's parsing and status behavior depends on stable test metadata
- the current mismatch between planned schema and runtime schema makes future tool work more brittle than it needs to be

Starting files:

- `src/jules_daemon/wiki/test_knowledge.py`
- `src/jules_daemon/execution/test_discovery.py`
- `src/jules_daemon/agent/tools/lookup_test_spec.py`
- `wiki/schema/`

Implementation direction:

- decide whether `pages/daemon/knowledge/` is the intended long-term location or whether a migration to a `tests/` namespace is actually needed
- add any missing first-class fields needed by status and completion summarization, especially `summary_fields`
- make discovery, save/load, tool lookup, and schema docs agree on one contract

Done when:

- catalog location and schema are explicitly documented
- `lookup_test_spec` returns the metadata needed by downstream summarizers
- new test knowledge pages round-trip through discovery and load without dropping planned fields

## P2: Remove Phase 2.5 doc and runtime drift

Current gap:

- external planning artifacts, repo-local architecture pages, and inline comments can still drift from one another even after the runtime changes land

Why this matters:

- the project now has planning artifacts, external wiki pages, and repo-local wiki pages; drift compounds quickly if the code comments stay stale
- this is cheap cleanup that reduces future confusion while Phase 2.5 is still fresh

Starting files:

- `src/jules_daemon/agent/agent_loop.py`
- `src/jules_daemon/ipc/request_handler.py`
- `src/jules_daemon/agent/tools/execute_ssh.py`
- `wiki/pages/architecture/`

Implementation direction:

- align comments and prompts with the current 10-tool surface and 15-iteration default unless the code is intentionally changed back
- remove stale wording about second confirmation where the implementation no longer does it
- use the repo-local wiki to record intentional deviations from the older external planning pages

Done when:

- inline comments match runtime behavior
- repo-local architecture pages explain the current defaults and known deliberate deviations
- future readers no longer need to reconcile obviously conflicting numbers and flows by hand

## Suggested Sequence

1. Background agent-loop execution
2. Default broadcaster wiring
3. Status enrichment on top of the shared run lifecycle
4. Test-catalog schema cleanup
5. Doc drift cleanup while the previous changes are still local and easy to verify

## Related

- [Phase 2.5 Alignment Checklist](phase-2-5-alignment-checklist.md)
- [Phase 2.5 Design Record](phase-2-5-design-record.md)
- [Repo Overview](repo-overview.md)
