---
tags:
- architecture
- agent-loop
- request-routing
- cli
- llm
type: architecture-note
created: 2026-04-14
updated: 2026-04-14
sources:
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/cli_main.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/ipc/request_handler.py
- /workspaces/jules/experiments-codex/jules-rescope-standalone/src/jules_daemon/agent/agent_loop.py
- /workspaces/jules/wiki/pages/daemon/phase-2-5-tools.md
---

# Agent-Driven Request Interpretation

## Summary

The project should move toward a thinner deterministic front door and a broader agent-driven interpretation layer for natural-language requests.

## Current Status

As of 2026-04-14, the active `cli_main` path now forwards nearly all user requests into a daemon-side `interpret` verb instead of classifying `run` / `status` / `watch` / `cancel` locally. The older front-door helpers still exist in `cli_main.py`, but they are no longer the primary user path.

As of 2026-04-16, the daemon-side `interpret` path is also chat-first for non-action prompts. When a user asks an informational question such as "do you know about test X?" or "what's the current status?", Jules first tries to answer directly from daemon context, workflow state, and matching test knowledge instead of forcing the request through a verb-classification/clarification loop.

The current system works best when the CLI can already identify:

- the canonical verb
- the SSH target or named system
- a clean natural-language test request

That is too brittle for real user prompts. When the front door fails early, the request can be blocked, distorted, or locally prompted before it ever reaches the agent loop.

## Problem

The current split is:

1. CLI/front-door heuristics try to determine the verb and transport target.
2. The daemon validates and resolves the target.
3. The agent loop then reasons about the test request, wiki spec, missing args, and command proposal.

This creates a failure mode:

- if the front door cannot confidently parse the target or intent, the LLM never gets the chance to interpret the request
- if the front door partially parses the request incorrectly, the agent loop receives distorted task text

Examples of fragile prompts:

- `run test A in tuto. 1 iteration`
- `can you run test A on tuto with 3 iterations`
- `please run the smoke tests against tutorial and name it nightly`

## Design Direction

The active design is now a hybrid:

### Thin front door

Keep deterministic handling for obvious structured cases:

- explicit `run user@host ...`
- explicit `run --system NAME ...`
- explicit `status`, `watch`, `cancel`, `history`

But when a request is conversational or ambiguous, do not force the CLI to fully resolve it.

### Agent-driven interpretation

Let the daemon answer non-action prompts conversationally and pass unresolved natural-language action requests into an agent interpretation path. The agent should infer:

- whether the request is actually a `run`
- what part of the prompt is the test request
- whether a target reference likely maps to a known named system
- which pieces look like test arguments versus transport metadata
- whether more information is still needed

For purely informational prompts, the daemon should instead:

- gather relevant workflow state
- gather matching test knowledge
- answer directly without forcing the user into command syntax

### Deterministic enforcement after interpretation

Even with broader LLM interpretation, the daemon should still own:

- final system alias resolution against the wiki
- validation that the target is real and allowed
- approval gating
- SSH execution lifecycle
- monitoring, persistence, and recovery

So the interface becomes more chat-like, but the runtime stays tool- and daemon-governed.

## Recommended Flow

1. User sends free-form input.
2. Front door handles obvious structured commands directly.
3. Non-action prompts are answered directly from daemon/wiki context when possible.
4. Unresolved or ambiguous action-like requests are forwarded to daemon-side interpretation.
5. The agent produces a structured interpretation, for example:
   - intent: `run`
   - system reference: `tuto`
   - task text: `test A`
   - provided args: `iterations=1`
   - missing info: `name`
6. Daemon resolves the system reference against `wiki/pages/systems/`.
7. If target resolution succeeds, the agent loop continues with wiki test lookup, missing-arg questioning, command proposal, approval, and execution.
8. If target resolution or task interpretation is still insufficient, only then ask the user a follow-up question.

## Why Not Make Everything Purely Agentic

Fully removing the front door would make the product feel more like a chatbot, but it would also make basic routing and safety less predictable.

The preferred design is not "heuristics only" and not "LLM only". It is:

- deterministic where transport and safety need hard guarantees
- LLM-assisted where prompt variety and ambiguity are the real problem

## Implications For Future Work

- CLI should stop acting as the final gatekeeper for unresolved run requests.
- The daemon needs an interpretation mode for raw conversational run prompts.
- Missing-argument questioning remains appropriate inside the agent loop.
- System alias resolution should stay daemon-side and wiki-backed, even when the LLM proposes the alias.
