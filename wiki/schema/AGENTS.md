# Wiki Schema

This repo-local wiki adapts Andrej Karpathy's LLM Wiki pattern to a software project whose primary source material is the codebase itself.

## Purpose

The goal is to keep a durable, synthesized map of the repository so future work does not start from zero every session. The wiki should accumulate architectural understanding, implementation notes, and known documentation drift.

## Layers

### Source Layer

Primary sources for this wiki are:

- Repository files under `src/`, `tests/`, `config/`, and project metadata files such as `pyproject.toml`
- The repo-local runtime wiki implementation in `src/jules_daemon/wiki/`
- Explicitly referenced external documents, especially `/workspaces/jules/wiki/`
- The Karpathy LLM Wiki gist: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

`wiki/raw/` is reserved for immutable external documents or snapshots that need to be filed into the repo-local knowledge base. Do not treat synthesized notes as raw sources.

### Wiki Layer

The maintained knowledge base lives in `wiki/`.

- `wiki/index.md`: content-oriented catalog of important pages
- `wiki/log.md`: chronological append-only record of ingest and maintenance activity
- `wiki/pages/architecture/`: repo architecture, runtime flow, package maps
- `wiki/pages/concepts/`: maintenance model, conventions, durable cross-cutting ideas
- `wiki/pages/daemon/*`: daemon-managed runtime state only; do not use for hand-written overview pages

### Schema Layer

The maintenance contract for the wiki is:

- repository root `AGENTS.md`
- `wiki/schema/AGENTS.md` (this file)

These documents tell future agents how to update the wiki and what belongs where.

## Operations

### Ingest

Use ingest when the repo changes materially or when a new external source is introduced.

1. Read the relevant source files.
2. Update existing wiki pages or add narrowly scoped new pages in user-managed directories.
3. Update `wiki/index.md`.
4. Append a concise dated entry to `wiki/log.md`.

### Query

When answering a repo question:

1. Read `wiki/index.md` first.
2. Read the most relevant wiki pages next.
3. Only then drill into source files that are still needed.
4. If the answer produces durable knowledge, file it back into the wiki.

### Lint

Periodically check for:

- stale counts or package descriptions
- drift between repo code and `/workspaces/jules/wiki/`
- pages that should link to each other but do not
- orphan pages not referenced by `wiki/index.md`
- statements about runtime behavior that no longer match the implementation

## Page Conventions

- Prefer YAML frontmatter with `tags`, `type`, `created`, `updated`, and `sources` when applicable.
- Keep pages focused and durable. Avoid dumping temporary debugging notes into permanent pages.
- Use relative markdown links or simple wiki-style references between pages.
- Put implementation facts close to the code paths they describe.
- Record dates explicitly when citing test counts, roadmap state, or documentation drift.

## Ownership Rules

- `wiki/pages/daemon/*` is daemon-managed. Do not manually write overview content there.
- User-curated repo knowledge belongs in `wiki/pages/architecture/`, `wiki/pages/concepts/`, `wiki/pages/security/`, and similar user-managed directories.
- External `/workspaces/jules/wiki/` content may be cited as a source, but should not be modified as part of maintaining this repo-local wiki unless the user explicitly asks for that too.

## Topics To Keep Current

- top-level entrypoints and runtime flow
- one-shot versus agent-loop execution paths
- the runtime wiki persistence model
- crash recovery behavior
- notification and watch-stream wiring
- test inventory and major doc drift from older external wiki pages
