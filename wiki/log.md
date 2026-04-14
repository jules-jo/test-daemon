---
tags:
- wiki-log
- project
- jules-daemon
type: wiki-log
created: 2026-04-13
updated: 2026-04-14
---

# Wiki Log

## [2026-04-13] wiki | Initialized repo-local wiki scaffold
Ran `jules_daemon.wiki.layout.initialize_wiki()` to materialize the repo-local wiki structure under `wiki/`, including daemon-managed and user-managed directories plus directory `README.md` files.

## [2026-04-13] docs | Added schema and initial codebase pages
Added a repo-local wiki schema, a content-oriented index, and initial pages covering the repo overview, the runtime wiki architecture, and how the Karpathy LLM Wiki pattern is adapted for this codebase.

## [2026-04-13] analysis | Recorded current repo snapshot
Documented that the packaged entrypoints are `jules-daemon`, `jules`, and `jules-demo`; natural-language `run` requests prefer the agent loop when LLM config is present; and a local `.venv/bin/pytest --collect-only -q` run collected 9331 tests in this workspace snapshot.

## [2026-04-13] docs | Added Phase 2.5 design record from planning artifacts
Reviewed the pre-implementation Ouroboros interview transcript at `/home/vscode/.ouroboros/data/interview_interview_20260412_003810.json` and the external Phase 2.5 tool-catalog page. Filed the durable design decisions, tool-surface evolution, and verification criteria into `wiki/pages/architecture/phase-2-5-design-record.md`.

## [2026-04-13] analysis | Added Phase 2.5 alignment checklist
Compared the current codebase against the Phase 2.5 interview and scenario, then filed a concrete `aligned` / `partially aligned` / `not yet aligned` assessment into `wiki/pages/architecture/phase-2-5-alignment-checklist.md`.

## [2026-04-13] planning | Added Phase 2.5 implementation backlog
Converted the alignment gaps into a short prioritized backlog in `wiki/pages/architecture/phase-2-5-implementation-backlog.md`, with concrete starting files and "done when" criteria for background execution, notifications, status enrichment, test-catalog schema cleanup, and doc-drift cleanup.

## [2026-04-13] implementation | Closed most of the Phase 2.5 runtime backlog
Background agent-loop execution now launches daemon-managed runs, live output can be read while those runs are active, the default daemon startup path owns a `NotificationBroadcaster`, `notify_user` prefers broadcaster-backed delivery, and `summary_fields` is now a first-class field carried through lookup, parsing, status, and run summarization. Updated the repo-local Phase 2.5 architecture pages to reflect the new current state.

## [2026-04-13] implementation | Wired the default live-run path into the monitor stack
`RequestHandler` now registers active runs with `JobOutputBroadcaster`, routes thread-based SSH output back onto the event loop safely, prefers broadcaster-backed live output for active `watch` and `status`, and collects default monitor alerts for error keywords and failure-rate spikes. Updated the Phase 2.5 alignment/backlog pages so they no longer describe the monitor path as entirely unwired.

## [2026-04-14] implementation | Added wiki-backed named systems for run requests
Added a user-managed `pages/systems/` wiki directory plus daemon resolution for `system_name` on run requests. The CLI can now interpret prompts like `run the smoke tests in system tuto`, send the named system to the daemon, and let the daemon resolve host/user/port from a markdown page instead of requiring `root@<IP>` in the prompt.

## [2026-04-14] docs | Clarified daemon wiki-dir behavior for named systems
Documented that system aliases are loaded from the daemon's configured `--wiki-dir`, not automatically from the repo-local `wiki/` folder. Added a copyable `wiki/pages/systems/example-system.md` template file for defining named SSH targets.

## [2026-04-14] implementation | Added richer target details to SSH approval prompts
SSH command approval prompts now show named-system context alongside the resolved SSH target. When a system page provides optional `hostname` or `ip_address` fields, the prompt includes those fields so the user can see the friendly system name and the concrete remote endpoint before approving execution.
