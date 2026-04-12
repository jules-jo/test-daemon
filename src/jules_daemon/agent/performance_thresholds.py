"""Performance threshold constants for agent loop latency enforcement.

Defines the maximum acceptable latencies for each stage of command
processing.  These thresholds ensure that the agent loop introduction
does not regress the v1.2-mvp direct-command execution path, which
is the latency-critical fast path (no LLM involvement).

The thresholds are organized into three tiers:

    1. **Direct-command detection** -- sub-millisecond classifier that
       decides whether input is a shell command (bypass agent loop) or
       natural language (enter agent loop).

    2. **One-shot translation** -- the v1.2-mvp LLM path with a 5-second
       deadline.  Retained as a fallback when the agent loop is
       unavailable or when ``--one-shot`` is explicitly requested.

    3. **Agent loop overhead** -- the incremental latency added by each
       iteration of the think-act-observe cycle.  Measured as wall-clock
       time from cycle entry to tool dispatch (excluding tool execution
       time itself).

All thresholds are in **seconds** unless otherwise noted.  Timing
assertions in the test suite compare ``time.monotonic()`` deltas against
these constants.

Environment variable overrides are intentionally *not* supported for
thresholds -- they are compile-time constants that define the performance
contract.  To change them, modify this module and re-run the test suite.

Usage::

    from jules_daemon.agent.performance_thresholds import (
        DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    )

    elapsed = _measure(detect_direct_command, "pytest -v tests/")
    assert elapsed < DIRECT_COMMAND_DETECTION_THRESHOLD_S
"""

from __future__ import annotations

__all__ = [
    "BASELINE_ITERATION_COUNT",
    "DIRECT_COMMAND_DETECTION_THRESHOLD_S",
    "ONESHOT_TRANSLATION_DEADLINE_S",
    "ONESHOT_PROMPT_BUDGET_S",
    "ONESHOT_PARSE_BUDGET_S",
    "AGENT_LOOP_OVERHEAD_PER_ITERATION_S",
    "AGENT_LOOP_FIRST_TOOL_CALL_THRESHOLD_S",
    "IPC_ENVELOPE_ROUNDTRIP_THRESHOLD_S",
    "CLASSIFY_INPUT_THRESHOLD_S",
    "TOOL_REGISTRY_LOOKUP_THRESHOLD_S",
    "WARM_DETECTION_THRESHOLD_S",
]


# ---------------------------------------------------------------------------
# Tier 1: Direct-command detection (fast path, no LLM)
# ---------------------------------------------------------------------------

DIRECT_COMMAND_DETECTION_THRESHOLD_S: float = 0.001
"""Maximum acceptable wall-clock time for ``detect_direct_command()``.

The v1.2-mvp baseline measurement shows sub-0.1ms execution on modern
hardware.  The 1ms threshold provides a 10x safety margin to account
for CI variability, GC pauses, and cold-cache scenarios.

This is the critical gate: if detection exceeds this threshold, the
user-perceived latency for direct commands (which should be near-instant)
degrades unacceptably.
"""

WARM_DETECTION_THRESHOLD_S: float = 0.0005
"""Maximum wall-clock time for a *warm* detection call.

After the first call has compiled regex patterns and populated the
module-level frozenset, subsequent calls should be even faster.
This 0.5ms threshold catches regressions in the hot path.
"""


# ---------------------------------------------------------------------------
# Tier 1b: Input classification (verb resolution + structuredness)
# ---------------------------------------------------------------------------

CLASSIFY_INPUT_THRESHOLD_S: float = 0.002
"""Maximum acceptable time for full input classification pipeline.

Covers verb registry lookup, structuredness scoring, and NL extraction
attempt.  The classification pipeline is pure CPU (no I/O) and should
complete well under 2ms.
"""


# ---------------------------------------------------------------------------
# Tier 2: One-shot LLM translation (v1.2-mvp path)
# ---------------------------------------------------------------------------

ONESHOT_TRANSLATION_DEADLINE_S: float = 5.0
"""Hard deadline for the full NL-to-command one-shot pipeline.

Inherited from ``command_translator.DEFAULT_DEADLINE_SECONDS``.
Kept here as the canonical reference for performance tests that
assert deadline compliance.
"""

ONESHOT_PROMPT_BUDGET_S: float = 0.1
"""Budget reserved for system prompt construction (cached).

Sub-millisecond in practice; 100ms budget is generous headroom.
"""

ONESHOT_PARSE_BUDGET_S: float = 0.1
"""Budget reserved for LLM response parsing and validation.

JSON extraction, Pydantic validation, and SSHCommand mapping are
all local operations completing in sub-millisecond time.
"""


# ---------------------------------------------------------------------------
# Tier 3: Agent loop overhead
# ---------------------------------------------------------------------------

AGENT_LOOP_OVERHEAD_PER_ITERATION_S: float = 0.050
"""Maximum acceptable *framework* overhead per think-act-observe cycle.

This measures only the loop machinery -- history assembly, message
serialization, result observation -- *excluding* LLM call time and
tool execution time.  50ms per iteration is generous; the actual
overhead should be under 5ms.
"""

AGENT_LOOP_FIRST_TOOL_CALL_THRESHOLD_S: float = 0.010
"""Maximum time from loop entry to first tool dispatch (framework only).

Measures history initialization, system prompt injection, and the
first ``get_tool_calls()`` dispatch setup.  Excludes the actual LLM
network call.  10ms threshold catches initialization regressions.
"""


# ---------------------------------------------------------------------------
# Supporting thresholds
# ---------------------------------------------------------------------------

IPC_ENVELOPE_ROUNDTRIP_THRESHOLD_S: float = 0.005
"""Maximum time for IPC envelope encode + decode roundtrip.

Covers ``encode_frame()`` and ``decode_frame()`` for a typical
command request envelope (~500 bytes JSON payload).
"""

TOOL_REGISTRY_LOOKUP_THRESHOLD_S: float = 0.001
"""Maximum time for ``ToolRegistry.get()`` and ``list_tools()`` calls.

Registry lookups are dict-based O(1) operations.  The 1ms threshold
catches accidental O(n) scans or serialization overhead.
"""


# ---------------------------------------------------------------------------
# Baseline measurement parameters
# ---------------------------------------------------------------------------

BASELINE_ITERATION_COUNT: int = 100
"""Number of iterations for baseline latency measurement.

Performance fixtures run ``detect_direct_command()`` and classification
functions this many times to compute stable median/p95/p99 statistics.
Using 100 iterations balances measurement precision against CI runtime.
"""
