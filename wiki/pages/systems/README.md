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

The daemon resolves these aliases from its configured wiki root.
If you start the daemon with `--wiki-dir /some/path/wiki`, put
system pages under `/some/path/wiki/pages/systems/`, not only under
this repo checkout.

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

## Important

If your daemon is started like this:

```text
python -m jules_daemon --wiki-dir C:\Users\you\.jules\wiki
```

then the live system file must be created here:

```text
C:\Users\you\.jules\wiki\pages\systems\tuto.md
```

If you only create `wiki/pages/systems/tuto.md` inside the git repo,
the CLI will not see it unless the daemon was also started with that
repo `wiki/` as its `--wiki-dir`.

## Template File

There is also a copyable template in this directory:

- `example-system.md`

Copy it to your daemon wiki, rename it to something like `tuto.md`,
replace the host/user values, and change `type` to `system-info`.
