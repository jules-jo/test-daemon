# Jules Daemon Wiki

The wiki is the sole persistence backbone for the Jules daemon. All state,
history, audit logs, and queue data are stored as markdown files with YAML
frontmatter.

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
    tools-and-sdks/                  # User-managed: tool documentation
  raw/                               # User-managed: unprocessed notes
  schema/                            # User-managed: schema documentation
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
