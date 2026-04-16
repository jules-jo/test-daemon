---
tags:
- architecture
- implementation-plan
- workflow
- orchestration
- state-model
- agent-loop
type: implementation-plan
created: 2026-04-16
updated: 2026-04-16
sources:
- /workspaces/jules/experiments-codex/jules-rescope-standalone/wiki/pages/architecture/stateful-test-workflow-orchestration.md
- /workspaces/jules/experiments-codex/jules-rescope-standalone/wiki/pages/architecture/agent-driven-request-interpretation.md
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/ipc/request_handler.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/agent/agent_loop.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/wiki/test_knowledge.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/wiki/layout.py
---

# Stateful Test Workflow Implementation Plan

## Summary

This page turns the workflow-orchestration design into a concrete implementation plan for Jules.

The target is:

- one active workflow at a time per conversation or run context
- many different workflow templates across the product
- workflow-aware status, notifications, summaries, and approvals
- prerequisite-aware execution such as `check artifact -> ask user -> run setup -> run main test`

This plan assumes the current daemon-backed codebase as the starting point. It does not require immediate daemon removal. A more model-centric or daemonless version can be built later on the same workflow primitives.

## Current Implementation Status

As of 2026-04-16, the Phase 1 foundation is now implemented in code:

- `src/jules_daemon/workflows/models.py`
- `src/jules_daemon/workflows/store.py`
- `src/jules_daemon/workflows/status.py`
- `src/jules_daemon/ipc/request_handler.py`
- `src/jules_daemon/wiki/layout.py`

What exists now:

- wiki-backed `WorkflowRecord` and `WorkflowStepRecord` persistence
- daemon-managed `pages/daemon/workflows/` and `pages/daemon/workflow-steps/`
- one workflow per launched run with a single `primary-run` step
- workflow-aware `status` output, including latest persisted workflow state
- workflow-aware test-knowledge schema fields such as `workflow_steps`, `prerequisites`, and `artifact_requirements`
- a first deterministic planner/preflight module in `src/jules_daemon/workflows/planner.py`
- active NL run handling now resolves matching workflow-aware test knowledge before the agent loop
- explicit path-like artifact requirements can now be checked remotely over SSH in `src/jules_daemon/workflows/preflight.py`
- missing or unverifiable artifacts can now trigger a preflight user question before the agent loop starts
- approved/declined preflight context is now carried into the agent loop system prompt and persisted on the started workflow record
- approved prerequisite-aware workflows can now switch onto a deterministic background workflow runner in `src/jules_daemon/workflows/runner.py`
- the run path can now resolve step specs, ask for missing step arguments up front, ask for per-step approval, and then execute `prerequisite -> main step` sequentially while persisting step-by-step workflow state
- workflow steps now support persisted `parsed_status` fields, backed by a generic interpreter registry under `src/jules_daemon/workflows/interpreters/`
- active `status` responses can now attach live interpreted progress for the active workflow step instead of only raw output tails
- sequential workflow execution now emits step-transition notifications (`started`, `completed`, `failed`) through the existing notification broadcaster when subscribers are present

What still remains for later phases:

- step-family-specific interpreters beyond the current generic parser-first fallback
- richer approval/edit semantics once a workflow is already running
- workflow-specific summaries and pass/fail interpretation driven by step interpreters instead of generic command results

## Scope

### In scope

- prerequisite-aware workflows
- workflow state persistence
- step-aware status answers
- step-aware output parsing
- workflow transition notifications
- generic support for multiple test families

### Out of scope for the first implementation

- concurrent multi-workflow orchestration
- fully daemonless runtime
- autonomous JIRA filing
- multi-agent coordination as a core runtime dependency

## Core Primitive Set

The implementation should introduce a small set of explicit primitives rather than embedding workflow behavior in ad hoc conditionals.

### 1. WorkflowRecord

Persistent top-level state for a user-visible workflow.

Suggested fields:

- `workflow_id: str`
- `conversation_id: str | None`
- `user_request: str`
- `test_slug: str | None`
- `workflow_kind: str`
- `system_name: str | None`
- `target_host: str | None`
- `target_user: str | None`
- `status: WorkflowStatus`
- `active_step_id: str | None`
- `step_ids: list[str]`
- `artifact_state: dict[str, ArtifactState]`
- `latest_summary: str`
- `final_outcome: str | None`
- `created_at: datetime`
- `updated_at: datetime`

### 2. WorkflowStepRecord

Persistent state for one step inside the workflow.

Suggested fields:

- `step_id: str`
- `workflow_id: str`
- `kind: str`
- `display_name: str`
- `command: str | None`
- `status: WorkflowStepStatus`
- `run_id: str | None`
- `session_id: str | None`
- `started_at: datetime | None`
- `ended_at: datetime | None`
- `exit_code: int | None`
- `summary: str`
- `parsed_status: dict[str, Any]`
- `artifact_updates: dict[str, Any]`

### 3. ArtifactState

Durable representation of a prerequisite artifact or generated output.

Suggested fields:

- `name: str`
- `path: str`
- `exists: bool`
- `freshness: str | None`
- `last_checked_at: datetime | None`
- `source_step_id: str | None`

### 4. WorkflowPlan

Short-lived planning object created by the agent or planner before execution begins.

Suggested fields:

- `workflow_kind: str`
- `test_slug: str`
- `primary_step_kind: str`
- `prerequisite_steps: list[WorkflowStepTemplate]`
- `main_steps: list[WorkflowStepTemplate]`
- `post_steps: list[WorkflowStepTemplate]`
- `required_user_questions: list[WorkflowQuestion]`

### 5. WorkflowQuestion

Explicit question state when the workflow needs user input or approval.

Suggested fields:

- `question_id: str`
- `workflow_id: str`
- `kind: str`
- `prompt: str`
- `blocking_step_kind: str | None`
- `choices: list[str] | None`

### 6. ParsedStepStatus

Structured interpretation result from step-specific output parsing.

Suggested fields:

- `state: str`
- `progress_message: str`
- `summary_fields: dict[str, Any]`
- `artifact_evidence: dict[str, Any]`
- `success_detected: bool`
- `failure_detected: bool`
- `raw_evidence_lines: list[str]`

## Service Layer

These services should be separate from the agent loop itself so the workflow engine is testable and reusable.

### WorkflowStore

Responsibilities:

- create/load/update workflow records
- create/load/update step records
- persist artifact state
- expose read APIs for status queries

Suggested initial location:

- `src/jules_daemon/workflows/store.py`

Suggested persistence backing:

- repo wiki if staying close to current Jules design
- or SQLite if later moving away from daemon/wiki persistence

### WorkflowPlanner

Responsibilities:

- resolve a user request into a workflow plan
- combine test knowledge, aliases, prerequisites, and workflow templates
- determine whether preflight checks are needed
- decide whether user clarification or approval is needed before execution

Suggested initial location:

- `src/jules_daemon/workflows/planner.py`

This planner can start as deterministic-plus-agent-assisted:

- deterministic loading of test knowledge
- agent reasoning for ambiguous workflow choices

### PreflightEvaluator

Responsibilities:

- run artifact checks before execution
- inspect whether a prerequisite artifact exists
- decide whether a prerequisite step is needed

Suggested initial location:

- `src/jules_daemon/workflows/preflight.py`

This should be a separate primitive from the main test execution step.

### WorkflowRunner

Responsibilities:

- execute one step at a time
- update workflow state transitions
- hand step output to interpreters
- advance to the next step when conditions are satisfied

Suggested initial location:

- `src/jules_daemon/workflows/runner.py`

This should own the generic workflow state machine, not test-specific business logic.

### StepInterpreterRegistry

Responsibilities:

- map a step kind or test family to a parser/interpreter
- return `ParsedStepStatus`
- support a generic fallback parser

Suggested initial location:

- `src/jules_daemon/workflows/interpreters/registry.py`
- `src/jules_daemon/workflows/interpreters/base.py`
- `src/jules_daemon/workflows/interpreters/generic.py`

Future per-family examples:

- `lt.py`
- `calibration.py`
- `smoke.py`

### WorkflowNotifier

Responsibilities:

- emit workflow lifecycle notifications
- convert step transitions into user-facing messages
- support both active-request responses and subscriber notifications

Suggested initial location:

- `src/jules_daemon/workflows/notifier.py`

### WorkflowStatusService

Responsibilities:

- answer `what is the current status?`
- format current phase, active step, and latest parsed summary
- provide a single stable status snapshot API

Suggested initial location:

- `src/jules_daemon/workflows/status.py`

## Knowledge Schema Additions

`test_knowledge` should expand to support workflow planning, not only argument lookup.

### New candidate fields

- `display_name`
- `aliases`
- `workflow_kind`
- `test_file_path`
- `prerequisites`
- `artifact_requirements`
- `preflight_checks`
- `workflow_steps`
- `success_criteria`
- `failure_criteria`
- `status_patterns`
- `summary_fields`
- `followup_suggestions`
- `escalation_policy`

### Minimal first slice

The smallest useful addition is:

- `workflow_kind`
- `prerequisites`
- `artifact_requirements`
- `workflow_steps`
- `success_criteria`
- `failure_criteria`

This is enough to support the first prerequisite-aware workflows.

## Concrete Tool and Runtime Primitives

The agent should not directly improvise the entire workflow every time. It should use explicit tools or service APIs.

### Must-have primitives

1. `resolve_test_workflow(test_name)`
   - loads the workflow-aware test knowledge record
   - returns prerequisites, artifact checks, and step templates

2. `check_artifact(path, system_name | target)`
   - verifies existence and optionally freshness
   - returns structured artifact state

3. `start_workflow_step(...)`
   - allocates a step record
   - launches the command via the existing runtime
   - returns a step/run handle

4. `read_step_output(step_id | run_id)`
   - returns incremental output for the active step

5. `interpret_step_output(step_kind, output, test_context)`
   - returns `ParsedStepStatus`

6. `advance_workflow(workflow_id)`
   - evaluates whether the next step should start

7. `get_workflow_status(workflow_id)`
   - returns a stable workflow-aware status payload

8. `emit_workflow_notification(workflow_id, event_type, payload)`
   - sends a user-facing progress or completion signal

### Nice-to-have later

- `draft_jira_from_workflow(workflow_id)`
- `suggest_followup_tests(workflow_id)`
- `resume_workflow(workflow_id)`

## Suggested Module Placement

The implementation should avoid burying workflow logic directly inside `request_handler.py`.

Suggested package:

- `src/jules_daemon/workflows/`

Suggested initial files:

- `models.py`
- `store.py`
- `planner.py`
- `preflight.py`
- `runner.py`
- `status.py`
- `notifier.py`
- `service.py`
- `interpreter_registry.py`

And optionally:

- `interpreters/base.py`
- `interpreters/generic.py`
- `interpreters/<family>.py`

## Integration Points In The Current Codebase

### Request entry

Current likely entry point:

- `src/jules_daemon/ipc/request_handler.py`

Initial integration:

- after request interpretation resolves a test, call `WorkflowPlanner`
- if the plan is single-step and has no prerequisites, reuse the existing simple path
- if the plan is multi-step, instantiate a workflow and route through `WorkflowRunner`

### Agent loop

Current core:

- `src/jules_daemon/agent/agent_loop.py`

Integration direction:

- the agent loop remains the orchestrator for request understanding and follow-up questions
- workflow execution itself should move into explicit workflow services
- avoid letting the model reconstruct all workflow state from raw chat history alone

### Test knowledge

Current location:

- `src/jules_daemon/wiki/test_knowledge.py`

Integration direction:

- extend the schema and accessors first
- keep backward compatibility so older test pages still load
- let discovered pages evolve into workflow-aware records gradually

### Status and notifications

Current home:

- `request_handler.py`
- monitor stack
- notification broadcaster stack

Integration direction:

- workflow status should become a first-class layer above raw run status
- step transitions should drive notifications
- the current monitor stack can remain the live-output source beneath the workflow layer

## Recommended Delivery Sequence

### Phase 1: Workflow state scaffolding

Build:

- `WorkflowRecord`
- `WorkflowStepRecord`
- `WorkflowStore`
- `WorkflowStatusService`

Done when:

- a workflow can be created and updated independently of the current single-run state
- `status` can return workflow-aware output for a mocked workflow

Status:

- implemented on 2026-04-16 as the first workflow foundation slice

### Phase 2: Knowledge schema and planner

Build:

- minimal knowledge schema additions
- `resolve_test_workflow`
- `WorkflowPlanner`
- `PreflightEvaluator`

Done when:

- a test spec can express prerequisites and artifact checks
- a request like `run lt test` can produce a plan with `calibration -> lt`
- the planner can tell whether a user question is needed before execution

Status:

- partially implemented on 2026-04-16 via workflow-aware test-knowledge fields plus a deterministic planner/preflight layer

### Phase 3: Step runner and workflow transitions

Build:

- `WorkflowRunner`
- `start_workflow_step`
- `advance_workflow`

Done when:

- Jules can run one prerequisite step, then advance to the primary step automatically
- workflow state transitions are persisted, not inferred only from logs

Status:

- implemented on 2026-04-16 for the first deterministic slice
- current behavior resolves step specs before launch, collects missing step args up front, asks for step-level approval, then runs the approved steps sequentially in one background workflow task
- current limitations: no mid-workflow command re-planning, no step-specific output interpreters yet, and no dedicated workflow notification UX beyond existing status/watch plumbing

### Phase 4: Output interpretation

Build:

- `ParsedStepStatus`
- interpreter registry
- generic interpreter
- first family-specific interpreter

Done when:

- Jules can answer status during a running workflow
- Jules can classify completion as pass/fail from interpreted step output
- the final summary comes from interpreted step state, not only raw shell output

### Phase 5: Notifications and follow-up UX

Build:

- `WorkflowNotifier`
- event-driven progress notifications
- improved status responses for conversational queries

Done when:

- users receive useful progress messages during prerequisites and main steps
- users can ask `what is the current status?` and get a workflow-aware answer

### Phase 6: Future escalation hooks

Build later:

- JIRA drafting from workflow state
- approval-gated filing step
- follow-up test suggestions

Done when:

- workflow failure can produce a draft downstream action without bypassing user approval

## Acceptance Scenarios

The implementation should be judged against scenarios, not only unit primitives.

### Scenario A: prerequisite artifact missing

- user requests a known test
- preflight check shows missing artifact
- Jules asks approval to run the prerequisite
- Jules runs prerequisite then main test
- Jules answers status while either is active
- Jules returns a final summary

### Scenario B: prerequisite already satisfied

- preflight finds the artifact
- Jules skips the prerequisite
- Jules runs the main test directly

### Scenario C: prerequisite fails

- prerequisite exits unsuccessfully
- Jules marks workflow failed
- Jules does not run the main test
- status and summary explain that the failure occurred in the prerequisite step

### Scenario D: multiple test families

- at least two unrelated test families each use different workflow templates
- the engine handles both without hard-coded LT-specific logic

## Design Constraint

The engine should be:

- generic over test family
- explicit about state
- deterministic in execution/runtime behavior
- agent-assisted in planning and explanation

That balance is the main implementation goal. The model should provide reasoning, but workflow state, step transitions, and approval boundaries should not live only in the conversation transcript.

## Related

- [Stateful Test Workflow Orchestration](stateful-test-workflow-orchestration.md)
- [Agent-Driven Request Interpretation](agent-driven-request-interpretation.md)
- [Phase 2.5 Implementation Backlog](phase-2-5-implementation-backlog.md)
- [Repo Overview](repo-overview.md)
