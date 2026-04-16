---
tags:
- architecture
- workflow
- agent-loop
- orchestration
- state-model
- notifications
type: architecture-note
created: 2026-04-16
updated: 2026-04-16
sources:
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/agent/agent_loop.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/ipc/request_handler.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/wiki/layout.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/wiki/pages/architecture/agent-driven-request-interpretation.md
- /workspaces/jules/experiments-codex/jules-rescope-standalone/wiki/pages/architecture/repo-overview.md
---

# Stateful Test Workflow Orchestration

## Summary

Jules should evolve from a single-command test runner into a stateful workflow agent that can reason about prerequisites, run multi-step test sequences, track progress, answer status questions mid-run, and summarize results in test-aware language.

The motivating example in this page is:

1. User asks to run `lt test`.
2. Agent notices the calibration artifact is missing.
3. Agent asks whether it should run calibration first.
4. Agent runs calibration.
5. Agent runs LT after calibration succeeds.
6. Agent keeps the user informed during both steps.
7. Agent can answer `what is the current status?` at any point.
8. Agent interprets LT output into pass/fail plus a summary.
9. Future: on failure, the agent drafts a JIRA and files it after user approval.

The LT case is only an example. The target design is not LT-specific and should support many different tests and workflow templates over time. Jules should be able to handle one test workflow at a time per active conversation or run context, but it should not be limited to a single hard-coded test family.

## Why This Needs More Agentic Behavior

This flow requires more than command generation:

- infer hidden prerequisites from test knowledge
- inspect environment state before execution
- ask a clarification or permission question when a prerequisite is missing
- remember that calibration and LT belong to the same higher-level workflow
- interpret output differently for calibration versus LT
- answer status questions while work is still in progress
- transition automatically from prerequisite completion into the next approved step

This is a good fit for a more model-centric Jules, but it does not require a multi-agent core.

The same reasoning pattern applies to many other tests:

- smoke tests that require environment setup first
- diagnostics that should run only when a preflight check fails
- tests that need generated input artifacts before execution
- tests that branch into different recovery or follow-up sequences depending on parsed output

## Recommended Architecture

The recommended design is:

1. One orchestration agent
2. One stateful execution runtime
3. One structured output-interpretation layer
4. One notification/status layer
5. Optional future escalation tools such as JIRA drafting

This architecture should be generic across test families. Jules should not encode a single privileged workflow such as LT. Instead, it should load workflow behavior from test knowledge plus runtime observation.

### Orchestration agent

The orchestration agent owns:

- interpreting the user's high-level intent
- expanding a named test request into a workflow plan
- deciding whether a prerequisite must run first
- asking approval questions for side actions
- deciding when to advance from one workflow step to the next
- answering user follow-up questions against workflow state

### Execution runtime

The execution runtime owns:

- connecting to the named system or SSH target
- opening and reusing a remote terminal session
- executing approved commands
- streaming stdout/stderr incrementally
- tracking process identifiers, start times, completion, and exit codes

This runtime can be daemon-backed or daemonless. If the daemon is removed, the runtime still needs to preserve durable workflow state somewhere if Jules should survive reconnects or answer later status queries.

### Output interpretation layer

The interpretation layer owns:

- test-specific parsing for calibration output
- test-specific parsing for LT output
- artifact detection
- success/failure judgment
- summary generation from raw output plus test knowledge

### Notification and status layer

The notification layer owns:

- pushing progress updates during calibration and LT
- surfacing important transitions such as `starting calibration`, `calibration complete`, `starting LT`, `LT failed`, `LT passed`
- answering synchronous status questions with workflow-aware summaries

## State Model

Jules should track a workflow object rather than only a single command.

### Workflow record

Suggested top-level fields:

- `workflow_id`
- `user_request`
- `system_name` or resolved target
- `workflow_type`
- `status`
- `active_step`
- `steps`
- `workflow_template`
- `artifact_state`
- `latest_summary`
- `final_outcome`
- `approval_history`
- `notification_history`
- `created_at`
- `updated_at`

### Workflow status

Suggested states:

- `received`
- `planning`
- `awaiting_prerequisite_approval`
- `running_prerequisite`
- `awaiting_primary_run_approval`
- `running_primary_test`
- `completed_success`
- `completed_failure`
- `cancelled`
- `error`

### Step record

Each workflow step should carry:

- `step_id`
- `kind` such as `calibration` or `lt_test`
- `command`
- `status`
- `started_at`
- `ended_at`
- `exit_code`
- `run_id` or terminal-session handle
- `artifact_checks`
- `parsed_status`
- `summary`

### Sequential scope

The intended scope is:

- one active workflow at a time in the current conversational/run context
- many different workflow templates across the product
- no requirement for concurrent multi-test orchestration in the first version

That means Jules should be able to run calibration-driven tests, smoke tests, diagnostics, and similar workflows sequentially without being architecturally limited to one specific named test.

### Artifact state

For prerequisite-driven tests, Jules also needs a durable artifact view:

- `artifact_name`
- `artifact_path`
- `exists`
- `last_checked_at`
- `source_step`

For LT, an example artifact could be the calibration file path that must exist before the main test runs.

## Knowledge Model Extensions

To support this kind of reasoning, test knowledge should expand beyond `required_args`.

Suggested fields:

- `display_name`
- `aliases`
- `test_file_path`
- `prerequisites`
- `artifact_requirements`
- `preflight_checks`
- `workflow_steps`
- `workflow_kind`
- `status_patterns`
- `success_criteria`
- `failure_criteria`
- `summary_fields`
- `followup_suggestions`
- `escalation_policy`

### Example LT-oriented knowledge

Conceptually:

- LT requires a calibration artifact
- if calibration is missing, Jules should offer calibration first
- calibration output has its own success/failure markers
- LT output has a different parser and different summary fields
- LT failure may recommend JIRA drafting after user approval

## Execution Flow For The LT Example

This section is illustrative only. The same workflow engine should support other prerequisite-driven tests by swapping in a different knowledge record and parser set.

### 1. Interpret request

User says: `Run lt test`.

The orchestration agent:

- resolves `lt test` to the LT spec
- loads prerequisite and artifact metadata
- checks whether calibration is required

### 2. Check prerequisite state

Jules runs a preflight check against the remote system:

- confirm the calibration file path
- confirm whether it exists and is current enough

### 3. Ask user when prerequisite is missing

If the artifact is missing, Jules replies with a workflow-aware question such as:

`There is no calibration file for LT. Do you want me to run calibration first?`

This should be modeled as part of the workflow state, not as an isolated prompt detached from later execution.

### 4. Run calibration

After approval, Jules:

- starts the calibration step
- emits a notification that calibration has started
- streams and interprets calibration output
- updates the workflow status and current summary while it runs

### 5. Run LT automatically after calibration

If calibration succeeds and the user already approved the prerequisite flow, Jules:

- marks calibration complete
- verifies the artifact exists
- starts the LT step
- updates the workflow status and notifications

### 6. Answer status questions mid-run

When the user asks:

- `what is the current status?`
- `what is it doing now?`
- `did calibration finish?`

Jules should answer from workflow state, not by starting from scratch.

The answer should include:

- active step
- current phase
- latest parsed output
- whether it is blocked, running, passed, or failed

### 7. Completion and final summary

When LT completes, Jules should:

- emit a completion notification
- store the final parsed result
- provide a summary with test-aware fields
- classify the workflow as success or failure

## Interpreting Output

Output interpretation should be step-specific.

### Calibration

Calibration needs:

- progress markers
- artifact-produced markers
- success markers
- failure markers

### LT test

LT needs:

- progress markers
- result counters or key fields
- success markers
- failure markers
- summary fields relevant to LT rather than generic shell output

This argues for structured parsers per test family or per workflow step, not only a generic summarizer.

## Notifications

Useful notifications for this workflow are:

- `Checking LT prerequisites`
- `Calibration file missing`
- `Waiting for your approval to run calibration`
- `Calibration started`
- `Calibration completed successfully`
- `Starting LT`
- `LT failed`
- `LT passed`
- `Workflow complete`

The notification system should be driven from workflow state transitions so it stays consistent with `status` answers.

## JIRA As A Future Step

JIRA filing should not be part of the critical path yet, but the workflow model should leave room for it.

Recommended future shape:

- on LT failure, the orchestration agent can propose a JIRA
- the agent drafts title, body, and evidence from workflow state
- the user explicitly approves filing
- the filing step becomes a separate approval-gated workflow action

## Why One Orchestrator Is Enough

This example justifies a more agentic Jules, but not necessarily a multi-agent core.

Recommended baseline:

- one orchestration agent
- one execution runtime
- one interpretation layer
- one workflow state store

Optional future sidecars:

- failure triage agent
- JIRA drafting agent
- follow-up test recommendation agent

Those can be added later without making the core workflow depend on agent-to-agent coordination.

## Generalization Requirement

The design target should be phrased as:

- Jules supports multiple test families and workflow templates
- each test can define its own prerequisites, artifact checks, workflow steps, parsers, and summary rules
- the orchestration engine is generic
- the LT example is one acceptance scenario, not the definition of the product

## Implications For A More Model-Centric Jules

If Jules becomes more model-centric or even daemonless, this workflow should be one of the acceptance targets.

The new architecture should prove that Jules can:

- plan a multi-step workflow from a short natural-language request
- detect prerequisite gaps from test knowledge and environment state
- ask the right question before taking a side action
- continue the workflow after the prerequisite succeeds
- answer status questions from persistent workflow state
- interpret completion into a test-aware success/failure summary

That is a better target than merely "can generate the right SSH command".
