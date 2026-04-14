---
type: system-info-template
template_for: system-info
system_name: tuto
aliases:
  - tutorial
  - tuto-box
host: 203.0.113.10
user: root
port: 22
description: Example system definition. Copy this file and change values.
---

# Example System Template

This file is documentation only.

To use it:

1. Copy it into the daemon's live wiki directory under `pages/systems/`.
2. Rename it to something like `tuto.md`.
3. Change `type` from `system-info-template` to `system-info`.
4. Replace `host`, `user`, `aliases`, and `description`.

Example live file:

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
