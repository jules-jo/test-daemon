---
tags:
- architecture
- phase-2-5
- alignment
- agent-loop
type: alignment-checklist
created: 2026-04-13
updated: 2026-04-13
sources:
- /home/vscode/.ouroboros/data/interview_interview_20260412_003810.json
- /workspaces/jules/wiki/pages/daemon/phase-2-5-tools.md
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/ipc/request_handler.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/agent/agent_loop.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/agent/tools/registry_factory.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/agent/tools/execute_ssh.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/agent/ipc_bridge.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/execution/test_discovery.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/wiki/test_knowledge.py
---

# Phase 2.5 Alignment Checklist

*Concrete comparison between the current repository and the Phase 2.5 interview, scenario, and tool-catalog plan.*

## Overall Verdict

The current project is **substantially aligned** with the Phase 2.5 interview and scenario, with a few narrower deviations still remaining.

It is strongly aligned on the core architectural intent of Phase 2.5:

- natural-language requests go through an agent loop by default
- one-shot remains as fallback
- approval-gated SSH execution is preserved
- a concrete Phase 2.5 tool surface exists

The biggest remaining deviations are that `status` is still a procedural enrichment path instead of a literal end-to-end tool invocation, and the default detector set is intentionally conservative rather than enabling every monitor trigger described in the earlier planning artifacts.

## Aligned

### Agent loop is the primary NL path

Planned:

- agent loop should be the default for natural-language commands
- one-shot should remain as fallback

Current state:

- `RequestHandler._handle_run()` routes natural-language commands into `_handle_run_agent_loop()`
- direct commands and certain failures fall back to `_handle_run_oneshot()`

Verdict: aligned

### Tool registry model exists

Planned:

- extensible ToolRegistry from day 1
- baseline Phase 2.5 tools behind tool schemas and execution wrappers

Current state:

- `agent/tool_registry.py` exists
- `agent/tools/registry_factory.py` wires the Phase 2.5 tool set

Verdict: aligned

### Core 10-tool Phase 2.5 surface is present

Planned later baseline:

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

Current state:

- all ten appear in the Phase 2.5 registry factory

Verdict: aligned

### Approval boundary matches the design intent

Planned:

- read-only operations are free
- SSH execution and state-changing actions are approval-gated

Current state:

- `propose_ssh_command` records approval
- `execute_ssh` consumes only previously approved commands
- read-only tools such as `lookup_test_spec`, `read_output`, and `parse_test_output` do not require approval

Verdict: aligned

### Missing required arguments are modeled as ask-the-user flow

Planned:

- never guess required arguments
- always ask the user

Current state:

- `ask_user_question` exists as a first-class tool
- `lookup_test_spec` returns `required_args`
- the system prompt in `RequestHandler._build_agent_system_prompt()` explicitly tells the LLM to ask for each missing required arg

Verdict: aligned

### Agent-loop execution now matches the scenario's long-running run model

Planned scenario:

1. agent proposes and approves a command
2. daemon starts the run in the background
3. run can be monitored while still executing
4. `read_output` can inspect partial output during execution

Current state:

- agent-started `execute_ssh` now launches daemon-managed background runs
- the agent-loop path shares the same current-run wiki state and in-memory output lifecycle as one-shot runs
- `read_output` can read live active-run output, and `status` can summarize active partial output

Verdict: aligned

### Persistent notification delivery is now part of the default runtime path

Planned:

- persistent subscription channel on CLI startup
- daemon push for completion and proactive alerts

Current state:

- default daemon startup instantiates and passes a `NotificationBroadcaster`
- `subscribe_notifications` and `unsubscribe_notifications` are fully handled by `RequestHandler`
- `notify_user` now prefers broadcaster-backed delivery when subscribers exist and falls back to direct stream delivery only when needed

Verdict: aligned

### Test-spec `summary_fields` is now a first-class runtime field

Planned scenario:

- test spec lookup returns summary configuration used to parse status and completion output

Current state:

- `test_knowledge.py` persists `summary_fields`
- `lookup_test_spec` returns `summary_fields`
- `parse_test_output`, `status`, and `summarize_run` all consume the focused summary view when it is available

Verdict: aligned

## Partially Aligned

### Test catalog exists, but not exactly in the planned shape

Planned:

- hybrid test catalog with user starter spec plus daemon augmentation
- page location described as `wiki/pages/daemon/tests/test-<name>.md`
- scenario text assumes fields such as `summary_fields`

Current state:

- the repo uses `pages/daemon/knowledge/test-{slug}.md`
- `test_knowledge.py` clearly supports `required_args`, `summary_fields`, and learned knowledge fields
- `discover_test` and `save_discovered_spec` reuse that `knowledge/` path
- `lookup_test_spec` returns `required_args` and `summary_fields`

Verdict: partially aligned

### The scenario's "status via agent parsing" is not the default implementation

Planned scenario:

- user asks `status`
- agent reads output
- agent consults test spec
- agent parses output and returns test-specific structured summary

Current state:

- `status` is handled procedurally in `RequestHandler._handle_status()`
- it reports active/completed state from wiki and in-memory handler state
- it now attaches test context, parses active/completed output snapshots, and formats `summary_fields`-aware summaries
- it still does not literally route through the agent loop or invoke the tool layer end to end

Verdict: partially aligned

### Notification delivery and proactive monitor alerts now flow through the default runtime

Planned:

- persistent subscription channel on CLI startup
- daemon push for completion and proactive alerts

Current state:

- notification broadcaster and emitter modules exist
- default daemon startup instantiates and passes a broadcaster
- agent-loop completion events and subscriber heartbeats flow through the broadcaster-backed channel
- `notify_user` prefers that broadcaster-backed channel
- `RequestHandler` now runs default error-keyword and failure-rate detectors against live output and emits alert notifications through the broadcaster-backed channel

Verdict: aligned

### Real-time analysis stack is now wired into the active live-run path, but the shipped detector set stays conservative

Planned:

- daemon performs continuous lightweight analysis
- LLM only on triggers such as status, anomaly, or completion

Current state:

- the repo has substantial `monitor/` infrastructure for output parsing, anomaly detection, output broadcasting, and queue consumers
- the main request handler now registers active runs with `JobOutputBroadcaster` and prefers that path for active watch/status/live-output access
- the request handler still keeps its older in-memory buffer and queue as a fallback for completed-run and legacy watch behavior
- the default detector set currently enables error-keyword and failure-rate signals, not every possible monitor trigger

Verdict: partially aligned

### Planning defaults drifted from implementation defaults

Planned:

- planning repeatedly described max iterations as configurable with a default around 5

Current state:

- code defaults are 15 iterations in both `AgentLoopConfig` and `RequestHandlerConfig`

Verdict: partially aligned

## Biggest Remaining Gaps

### Status remains procedural enrichment instead of a literal tool-driven flow

Planned scenario:

- user asks `status`
- agent reads output, consults test spec, parses it, and replies using the tool layer end to end

Current state:

- `status` now sees live broadcaster-backed output and collected alert summaries
- test-specific enrichment is present and uses shared parsing helpers
- the shipped implementation still stays in `RequestHandler._handle_status()` rather than invoking the tool layer as the runtime path

Verdict: partially aligned

### Default monitor triggers are narrower than the full planning sketch

Planned scenario:

- daemon performs continuous lightweight analysis
- proactive alerts can be emitted on meaningful anomalies such as error keywords, failure spikes, and stalls
- LLM is only invoked on user request, completion, or daemon-detected triggers

Current state:

- the default runtime now wires active output into `JobOutputBroadcaster`, detector dispatch, alert collection, and broadcaster-backed alert delivery
- the shipped detector set is conservative: error keywords and failure-rate spikes are on by default
- stall-style triggers and any LLM-on-trigger orchestration are not yet default runtime behavior

Verdict: partially aligned

## Bottom Line

If the question is whether the current repo implements the **Phase 2.5 architectural foundation**, the answer is yes.

If the question is whether it fully implements the **pre-build Phase 2.5 scenario as written**, the answer is still no. The main missing pieces are now:

- deciding whether procedural status enrichment should stay the shipped design or be replaced by a more literal tool-driven flow
- expanding the default monitor trigger set beyond conservative error-keyword and failure-rate alerts

## Related

- [Phase 2.5 Design Record](phase-2-5-design-record.md)
- [Repo Overview](repo-overview.md)
