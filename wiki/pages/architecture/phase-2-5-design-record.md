---
tags:
- architecture
- phase-2-5
- agent-loop
- planning
type: phase-design-record
created: 2026-04-13
updated: 2026-04-13
sources:
- /home/vscode/.ouroboros/data/interview_interview_20260412_003810.json
- /workspaces/jules/wiki/pages/daemon/phase-2-5-tools.md
- /workspaces/jules/wiki/pages/daemon/future-plans.md
- /workspaces/jules/wiki/log.md
---

# Phase 2.5 Design Record

*Durable record of the Phase 2.5 design intent captured before implementation on April 12, 2026, plus the main planning drift to watch when comparing plan versus code.*

## Context

Phase 2.5 was the transition from the v1.2 MVP's one-shot NL-to-command flow to a true tool-calling agent loop. The pre-build design source for this phase lives in two places:

- the Ouroboros interview transcript at `/home/vscode/.ouroboros/data/interview_interview_20260412_003810.json`
- the external planning page [phase-2-5-tools.md](/workspaces/jules/wiki/pages/daemon/phase-2-5-tools.md:1)

These are planning sources, not implementation truth. They are still useful because they explain why several Phase 2.5 components exist and what behavior they were supposed to enforce.

## Core Intent

The intended Phase 2.5 change was:

- agent loop as the default path for natural-language `run` requests
- one-shot path retained as fallback
- explicit human approval preserved for SSH execution and other state-changing actions
- a small extensible tool surface from Day 1
- daemon-owned monitoring and wiki persistence kept outside the LLM loop

## Resolved Design Decisions

### Approval Boundary

- Read-only tools and planning operations are free.
- SSH execution and other state-changing tools require explicit human approval.
- Approval bypass modes were considered, but explicitly deferred and must remain opt-in if added later.

This explains the security posture of the Phase 2.5 design: the LLM may inspect and reason autonomously, but it must not execute or mutate state silently.

### Iteration Model

One iteration means one full think-act-observe cycle:

1. LLM receives the current message history
2. LLM returns one or more tool calls
3. daemon executes those tool calls
4. results are appended to history

Multiple tool calls in a single LLM response were intentionally allowed.

### Stop Conditions

The stop conditions were OR-composed:

- hard max iteration cap
- LLM signals completion
- explicit user cancellation

The interview consistently described the cap as configurable, with a default of 5 at planning time.

### Integration Strategy

The agent loop was supposed to become the primary path for natural-language commands, while one-shot translation remained as a fallback when:

- LLM credentials are missing
- agent-loop initialization fails
- a one-shot flag is explicitly requested

### Error Handling

The planning behavior was:

- transient LLM or infra errors: retry twice within the same iteration, then fall back to one-shot
- permanent errors: terminate immediately
- `execute_ssh` timeout: treat as permanent
- hitting the max-iteration cap: terminate with a gave-up style message

## Tool Surface

The design evolved during the interview.

### Early Baseline

The early baseline was a 6-tool set:

- `read_wiki`
- `check_remote_processes`
- `propose_ssh_command`
- `execute_ssh`
- `read_output`
- `summarize_run`

### Later Baseline

The later and more complete baseline expanded to 10 tools:

- `read_wiki`
- `lookup_test_spec`
- `check_remote_processes`
- `propose_ssh_command`
- `execute_ssh`
- `read_output`
- `parse_test_output`
- `summarize_run`
- `ask_user_question`
- `notify_user`

This later 10-tool baseline is the more important design record because it captures the missing-argument flow, test catalog lookup, and notification path that the earlier 6-tool framing did not yet include.

## Phase 2.5-Specific Product Decisions

### Test Catalog

The test catalog was intended to be hybrid:

- user provides starter spec fields such as command template and required args
- daemon augments with learned fields such as durations, failures, preferred hosts, and summary fields

### Missing Arguments

The design was explicit: always ask the user for missing required arguments. No guessing and no defaulting from history for required fields.

### Notifications

The preferred notification design was a persistent subscription channel where the CLI opens a long-lived connection and the daemon pushes completion or alert events.

### Real-Time Analysis

The design was hybrid:

- daemon performs lightweight continuous regex or pattern checks
- LLM is only invoked on demand or on triggers such as status requests, anomalies, or completion

### JIRA

JIRA was explicitly deferred to Phase 2.6+, along with draft review and ticket creation tools.

## Verification Criteria

The interview defined three must-pass demo scenarios:

1. End-to-end named test execution from NL request through summary
2. Self-correction after an initial failed command
3. Missing-argument resolution through the wiki-backed test catalog

It also set quality expectations:

- at least 80% coverage on new agent code
- no regressions in the existing suite at planning time
- one-shot backward compatibility
- no performance regression on direct commands

## Planning Drift Worth Remembering

Several planning artifacts disagree with each other:

- the interview began with a 6-tool baseline, then expanded to 10
- the external tool-catalog page says "11 tools" while its listed baseline is actually 10
- planning documents and the built code later diverged on details such as iteration defaults, notification wiring, and exact tool behavior

This page exists so future work can distinguish:

- what Phase 2.5 intended to build
- what this repository currently implements

## Related

- [Repo Overview](repo-overview.md)
- [Wiki Runtime Architecture](wiki-runtime-architecture.md)
