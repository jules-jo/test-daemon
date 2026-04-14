---
tags:
- wiki-structure
type: wiki-directory
kind: user_managed
path: pages/systems
---

# Systems

*User-Managed directory*

Named system definitions for SSH target aliases.

## Ownership

This directory is user-managed. Create, edit, and organize
files here as needed. The daemon will not modify files
in this directory.

## System Page Format

Create one markdown file per named system. Example:

```yaml
---
type: system-info
system_name: tuto
aliases:
  - tutorial
host: 10.0.0.10
user: root
port: 22
description: Tutorial box for smoke-test runs.
---
```

Then users can say things like:

- `run the smoke tests in system tuto`
- `run --system tuto run the smoke tests`
