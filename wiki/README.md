# Jules Daemon Wiki

The local `wiki/` directory serves two roles:

1. it is the daemon's persistence backbone for runtime state
2. it is the repo-local project wiki for durable architecture knowledge

That only works if the ownership split is respected. Runtime state lives in
daemon-managed directories. Human or LLM-maintained project documentation
belongs only in user-managed directories such as `pages/architecture/`,
`pages/concepts/`, and `schema/`.

This repo-local wiki follows the Karpathy LLM Wiki pattern: sources feed a
maintained wiki, and the maintenance workflow is defined by a schema file.
For this repository, the schema lives in `schema/AGENTS.md`, the catalog
lives in `index.md`, and the append-only maintenance record lives in
`log.md`.

## Directory Structure

When the daemon starts, it initializes this structure automatically via
`jules_daemon.wiki.layout.initialize_wiki()`.

```
wiki/
  index.md                           # Auto-generated wiki index
  pages/
    daemon/                          # Daemon-managed: run state
      current-run.md                 # Active state record (single file)
      recovery-log.md                # Recovery attempt log
      startup-event.md               # Last startup lifecycle audit
      history/                       # Completed run archives
        run-{run_id}.md              # One file per completed run
      results/                       # Assembled test results
        result-{run_id}.md           # One file per result set
      translations/                  # NL-to-command mappings (learning)
        {slug}--{id}.md              # One file per translation
      audit/                         # Per-command audit trail
        audit-{event_id}.md          # One file per execution event
        archive/                     # Archived audit logs
          audit-{event_id}.md
      queue/                         # Pending command queue
        {seq}-{run_id}.md            # One file per queued command
    agents/                          # User-managed: agent documentation
    architecture/                    # User-managed: architecture notes
    concepts/                        # User-managed: general concepts
    security/                        # User-managed: security notes
    systems/                         # User-managed: named SSH target aliases
    tools-and-sdks/                  # User-managed: tool documentation
  raw/                               # User-managed: unprocessed notes
  schema/                            # User-managed: schema documentation
  log.md                             # Chronological repo-local wiki log
```

## Ownership Model

- **Daemon-managed directories** (`pages/daemon/*`): Files are created,
  updated, and archived automatically by the daemon. Do not manually
  edit files in these directories.

- **User-managed directories** (`pages/agents`, `pages/concepts`, etc.):
  Users create, edit, and organize files here. The daemon will not
  modify files in these directories.

## Frontmatter Convention

Every wiki file uses YAML frontmatter with at minimum:
- `tags`: List of classification tags
- `type`: Document type identifier

Example:
```yaml
---
tags: [daemon, state, current-run]
type: daemon-state
status: idle
---
# Current Run

No active run.
```

## Crash Recovery

On startup, the daemon reads `pages/daemon/current-run.md` to detect
incomplete runs from a previous crash. The scan-probe-mark pipeline
checks liveness of any active sessions and marks stale ones, ensuring
recovery within 30 seconds.

## Repo-Local Wiki Guidance

- Read `index.md` first when using the wiki for codebase orientation.
- Use `schema/AGENTS.md` as the maintenance contract.
- Update `log.md` when you add durable project knowledge or revise a
  previously recorded understanding.
