---
tags:
- concept
- wiki
- karpathy
- maintenance
type: project-wiki-system
created: 2026-04-13
updated: 2026-04-13
sources:
- https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- /workspaces/jules/wiki/pages/concepts/llm-wiki-pattern.md
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/src/jules_daemon/wiki/layout.py
- /workspaces/jules/experiments-codex/jules-agent-loop-rescope-codex/jules-agent-loop-rescope/AGENTS.md
---

# Project Wiki System

*Adaptation of the Karpathy LLM Wiki pattern for a living software repository whose most important sources are the code, tests, and design drift over time.*

## Why This Repo Needs A Wiki

This codebase is large enough that session-by-session rediscovery is wasteful. The important facts are not only what modules exist, but how the runtime flows are wired, where the newer and older stacks coexist, and where the external planning wiki has drifted from the implementation.

## Adapted Three-Layer Model

### Sources

For this project, sources are primarily:

- code under `src/`
- tests under `tests/`
- project metadata such as `pyproject.toml`
- explicitly referenced external docs, especially `/workspaces/jules/wiki/`

`wiki/raw/` is reserved for immutable imported documents when needed, but the repo itself remains the primary source corpus.

### Wiki

The synthesized knowledge lives under `wiki/` and should answer the repeated questions:

- What are the main runtime paths?
- Which modules are foundational versus partially integrated?
- How does the wiki persistence model work?
- What is stale in the external docs?

### Schema

The maintenance rules live in the root `AGENTS.md` and `wiki/schema/AGENTS.md`.

## Supported Operations

### Ingest

Use ingest when:

- a package is added or substantially refactored
- runtime behavior changes
- tests counts or coverage narratives materially change
- external reference docs are added or corrected

Expected output:

- update the relevant page
- update `wiki/index.md`
- append to `wiki/log.md`

### Query

Use query by reading `wiki/index.md` first, then the most relevant pages, then source files only where the wiki is incomplete or needs verification.

### Lint

Use lint to check:

- stale version, test-count, or roadmap claims
- mismatch between comments and implementation
- mismatch between the repo-local wiki and `/workspaces/jules/wiki/`
- orphan pages not linked from `wiki/index.md`

## Repo-Specific Rule

Because the daemon itself uses `wiki/` as runtime storage, durable prose belongs only in user-managed directories. `pages/daemon/*` must stay reserved for daemon-managed state records.

## Relationship To The External Wiki

The external `/workspaces/jules/wiki/` directory is a useful historical and planning reference. This repo-local wiki is different:

- external wiki: broader research and planning corpus
- repo-local wiki: implementation-focused map of this repository

The local wiki should cite external pages when useful, but remain anchored to the current code.
