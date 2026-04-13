# Repo Wiki Workflow

This repository uses `wiki/` as a Karpathy-style project wiki for durable codebase knowledge.

Before answering architecture questions or making broad changes:

1. Read `wiki/index.md`.
2. Open any relevant pages under `wiki/pages/`.
3. Use `wiki/schema/AGENTS.md` as the maintenance contract.

Rules:

- Treat `src/`, `tests/`, `config/`, `pyproject.toml`, and explicitly referenced external docs as source material.
- Never manually write explanatory prose into `wiki/pages/daemon/*`; those directories are daemon-managed runtime state.
- Put durable human/LLM-maintained notes only in user-managed areas such as `wiki/pages/architecture/` and `wiki/pages/concepts/`.
- When you learn something durable about the repo, update the relevant wiki page, `wiki/index.md`, and `wiki/log.md`.
- Keep edits incremental. Prefer updating existing pages over creating near-duplicates.
- The external reference wiki at `/workspaces/jules/wiki/` is a source for context, not part of this repo-local wiki.
