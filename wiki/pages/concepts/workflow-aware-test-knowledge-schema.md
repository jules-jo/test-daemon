---
tags:
- concepts
- schema
- workflow
- test-knowledge
type: concept
created: 2026-04-16
updated: 2026-04-16
sources:
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/wiki/test_knowledge.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/workflows/planner.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/agent/tools/lookup_test_spec.py
---

# Workflow-Aware Test Knowledge Schema

This page documents the current markdown/frontmatter format for giving workflow-aware test knowledge to Jules.

The live files should be created in the daemon's configured wiki under:

- `pages/daemon/knowledge/test-<slug>.md`

In a typical Windows local setup, that means:

- `$env:USERPROFILE\.jules\wiki\pages\daemon\knowledge\test-<slug>.md`

## Current Schema

### Base test fields

- `type`
- `test_slug`
- `command_pattern`
- `purpose`
- `output_format`
- `test_file_path`
- `summary_fields`
- `normal_behavior`
- `required_args`
- `common_failures`
- `runs_observed`
- `last_updated`

### Workflow-aware fields

- `workflow_steps`
  Ordered logical step names for the workflow.
- `prerequisites`
  Prerequisite step or capability names that should be satisfied before the primary step.
- `artifact_requirements`
  Named artifacts that should exist before the primary step is ready.
- `when_missing_artifact_ask`
  User-facing prompt to ask when one or more required artifacts are missing.
- `success_criteria`
  Human-readable description of what success means for the workflow.
- `failure_criteria`
  Human-readable description of what failure means for the workflow.

## Example

```md
---
tags:
  - daemon
  - test-knowledge
  - learning
type: test-knowledge
test_slug: lt-test
command_pattern: python3 /root/lt.py --target {target}
test_file_path: /root/lt.py
purpose: Runs the LT validation flow.
output_format: Progress lines plus a final pass/fail summary.
summary_fields:
  - passed
  - failed
required_args:
  - target
workflow_steps:
  - calibration
  - lt_test
prerequisites:
  - calibration
artifact_requirements:
  - calibration_file
when_missing_artifact_ask: There is no calibration file. Do you want me to run calibration first?
success_criteria: LT summary reports zero failures.
failure_criteria: Calibration fails or LT reports any failure.
common_failures:
  - calibration file missing
  - timeout waiting for device
runs_observed: 0
last_updated: '2026-04-16T00:00:00+00:00'
---
```

## What Uses These Fields Today

- `lookup_test_spec` now returns the workflow-aware fields directly to the agent.
- `read_wiki` now returns the same fields in its `test_knowledge` payload.
- `status` test context now surfaces the workflow-aware fields when matching test knowledge exists.
- `workflows.planner.resolve_test_workflow()` can build a deterministic plan from `workflow_steps`, `prerequisites`, and `artifact_requirements`.
- `workflows.planner.evaluate_workflow_preflight()` can turn missing artifact facts into a deterministic user question.

## What Is Not Wired Yet

- automatic remote artifact probing
- automatic prerequisite execution from the main run path
- agent-loop prompt/tool changes that make workflow planning mandatory before execution
- workflow-driven notifications and composite summaries

So this schema is now implemented and usable, but it is still groundwork for the later multi-step execution phases.
