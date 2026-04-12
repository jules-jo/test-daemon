"""Sample wiki test spec entries for demo scenarios.

Each constant is the raw markdown content (YAML frontmatter + body) of a
wiki test knowledge page.  These mirror what the daemon persists under
``wiki/pages/daemon/knowledge/test-{slug}.md``.

The specs follow the hybrid wiki format:
    - User creates starter fields: command_template, required_args
    - Daemon augments with learned fields: typical_duration, failure_patterns,
      summary_fields, runs_observed, last_updated

Usage in tests::

    from tests.fixtures.wiki_test_specs import AGENT_TEST_SPEC_RAW
    from jules_daemon.wiki.frontmatter import parse

    doc = parse(AGENT_TEST_SPEC_RAW)
    assert doc.frontmatter["test_slug"] == "agent-test-py"
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# agent_test.py -- the primary demo test (Demo 1, Demo 3)
# ---------------------------------------------------------------------------

AGENT_TEST_SPEC_RAW: str = """\
---
tags:
  - daemon
  - test-knowledge
  - learning
type: test-knowledge
test_slug: agent-test-py
command_pattern: python3 ~/agent_test.py
purpose: >-
  Runs the agent loop stress test with configurable iterations and
  concurrency. Verifies that the agent can maintain state across
  multiple think-act cycles under load.
output_format: >-
  Line-delimited progress: 'Iteration N/M ... OK|FAIL'. Final summary
  line: 'Result: X passed, Y failed, Z skipped in Ns'.
normal_behavior: >-
  All iterations complete within the timeout. Exit code 0 with
  'Result: N passed, 0 failed, 0 skipped' on the final line.
required_args:
  - iterations
  - host
common_failures:
  - 'timeout on large iteration counts (>500)'
  - 'connection refused when SSH agent is not forwarded'
  - 'ImportError: missing dependency on fresh hosts'
runs_observed: 42
last_updated: '2026-04-10T14:30:00+00:00'
---

# Test Knowledge: agent-test-py

*Auto-curated knowledge accumulated across runs of this test.*
*You may edit this file by hand -- the daemon will preserve
non-empty fields when merging new observations.*

## Command Pattern

```bash
python3 ~/agent_test.py
```

## Purpose

Runs the agent loop stress test with configurable iterations and
concurrency. Verifies that the agent can maintain state across
multiple think-act cycles under load.

## Output Format

Line-delimited progress: 'Iteration N/M ... OK|FAIL'. Final summary
line: 'Result: X passed, Y failed, Z skipped in Ns'.

## Normal Behavior

All iterations complete within the timeout. Exit code 0 with
'Result: N passed, 0 failed, 0 skipped' on the final line.

## Required Arguments

- `iterations`
- `host`

## Common Failures

- timeout on large iteration counts (>500)
- connection refused when SSH agent is not forwarded
- ImportError: missing dependency on fresh hosts

## Statistics

- Runs observed: 42
- Last updated: 2026-04-10T14:30:00+00:00
"""


# ---------------------------------------------------------------------------
# pytest integration suite -- a second spec for variety (Demo 2)
# ---------------------------------------------------------------------------

PYTEST_INTEGRATION_SPEC_RAW: str = """\
---
tags:
  - daemon
  - test-knowledge
  - learning
type: test-knowledge
test_slug: pytest-tests-integration
command_pattern: pytest tests/integration/ -v --tb=short
purpose: >-
  Runs the integration test suite against a live database. Covers
  API endpoints, message queue consumers, and cache invalidation.
output_format: >-
  Standard pytest verbose output with short tracebacks. Summary line:
  'X passed, Y failed, Z warnings in Ns'.
normal_behavior: >-
  All tests pass. Typical duration 45-90 seconds depending on DB latency.
  Exit code 0.
required_args: []
common_failures:
  - 'FAILED tests/integration/test_api.py::test_health - ConnectionError'
  - 'fixture "db_session" not found (missing conftest on remote)'
runs_observed: 18
last_updated: '2026-04-09T09:15:00+00:00'
---

# Test Knowledge: pytest-tests-integration

*Auto-curated knowledge accumulated across runs of this test.*

## Command Pattern

```bash
pytest tests/integration/ -v --tb=short
```

## Purpose

Runs the integration test suite against a live database.

## Normal Behavior

All tests pass. Typical duration 45-90 seconds depending on DB latency.

## Statistics

- Runs observed: 18
- Last updated: 2026-04-09T09:15:00+00:00
"""


# ---------------------------------------------------------------------------
# smoke_test.sh -- minimal test with no required args
# ---------------------------------------------------------------------------

SMOKE_TEST_SPEC_RAW: str = """\
---
tags:
  - daemon
  - test-knowledge
  - learning
type: test-knowledge
test_slug: smoke-test-sh
command_pattern: ./smoke_test.sh
purpose: Quick sanity check that critical services are reachable.
output_format: 'PASS/FAIL per service, one line each.'
normal_behavior: All lines show PASS. Exit code 0. Duration under 10 seconds.
required_args: []
common_failures:
  - 'FAIL: redis -- Connection refused on port 6379'
runs_observed: 87
last_updated: '2026-04-11T22:00:00+00:00'
---

# Test Knowledge: smoke-test-sh

*Auto-curated knowledge accumulated across runs of this test.*

## Command Pattern

```bash
./smoke_test.sh
```

## Purpose

Quick sanity check that critical services are reachable.

## Statistics

- Runs observed: 87
- Last updated: 2026-04-11T22:00:00+00:00
"""
