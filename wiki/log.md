---
tags:
- wiki-log
- project
- jules-daemon
type: wiki-log
created: 2026-04-13
updated: 2026-04-16
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

## [2026-04-14] design | Recorded hybrid plan for broader natural-language request handling
Added `wiki/pages/architecture/agent-driven-request-interpretation.md` to capture the next-step design direction: keep only a thin deterministic front door for obvious structured commands, let unresolved conversational run requests fall into a daemon-side agent interpretation path, and keep final target validation, approvals, execution, monitoring, and recovery deterministic in daemon code.

## [2026-04-14] implementation | Switched the active CLI path to daemon-side interpretation
The legacy CLI front door is still present in code for fallback/reference, but the active `cli_main` flow now forwards nearly all user requests through a daemon-side `interpret` verb. The daemon uses the LLM intent classifier to map conversational prompts to structured verbs, retries once with a follow-up clarification question when confidence or validation is insufficient, and then dispatches into the existing `run` / `status` / `watch` / `cancel` / `history` / `discover` handlers.

## [2026-04-16] design | Added stateful workflow architecture for prerequisite-driven tests
Added `wiki/pages/architecture/stateful-test-workflow-orchestration.md` to capture the next-step design target for more agentic Jules behavior. The page uses an `LT -> calibration -> LT summary` example to define a workflow-oriented state model, prerequisite reasoning, mid-run status answers, notification transitions, and future JIRA escalation after approval.

## [2026-04-16] design | Clarified that workflow orchestration is generic, not LT-specific
Updated `wiki/pages/architecture/stateful-test-workflow-orchestration.md` to make the intended scope explicit: LT is just one motivating example, while the actual architecture should support many different sequential test workflows through generic workflow templates, test knowledge, artifact checks, and step-specific parsers.

## [2026-04-16] planning | Added concrete implementation plan for stateful test workflows
Added `wiki/pages/architecture/stateful-test-workflow-implementation-plan.md` to convert the workflow architecture note into an executable plan. The new page defines the core records, services, tool/runtime primitives, suggested module boundaries, knowledge-schema additions, phased delivery sequence, and acceptance scenarios for generic prerequisite-aware test workflows.

## [2026-04-16] implementation | Added Phase 1 workflow state foundation
Implemented the first workflow slice in code: new workflow models, a wiki-backed workflow/step store, a read-side workflow status service, and request-handler integration that creates one workflow per launched run, updates it on completion or cancellation, and exposes workflow snapshots through `status`.

## [2026-04-16] implementation | Added workflow-aware test knowledge schema and planner groundwork
Extended `TestKnowledge` with workflow-specific fields such as `workflow_steps`, `prerequisites`, `artifact_requirements`, `when_missing_artifact_ask`, `success_criteria`, and `failure_criteria`. Added a deterministic workflow planner/preflight module, exposed the new fields through `lookup_test_spec`, `read_wiki`, and status test context, and documented the authoring format in `wiki/pages/concepts/workflow-aware-test-knowledge-schema.md`.

## [2026-04-16] implementation | Wired workflow preflight into the active run path
Natural-language runs now resolve workflow-aware test knowledge before the agent loop, probe explicit remote artifact paths over SSH when possible, ask a deterministic preflight question when artifacts are missing or unverifiable, and carry the resulting workflow context into the agent loop system prompt plus the persisted workflow record. Updated the workflow implementation-plan and schema pages to reflect the new current state and remaining gap around automatic multi-step advancement.

## [2026-04-16] implementation | Added the first sequential workflow runner
When workflow preflight says prerequisite steps should run first, Jules can now resolve those workflow steps into concrete test specs, collect any missing step arguments up front, ask for explicit approval per step command, and execute the approved steps sequentially in one background workflow task. Workflow status now persists multiple step records for the same workflow instead of only a single `primary-run` step, and the implementation-plan page now records that automatic `prerequisite -> main step` advancement is live in the first deterministic slice.
