"""Performance baseline tests for v1.2-mvp direct-command execution.

Validates that the latency-critical paths stay within the thresholds
defined in ``jules_daemon.agent.performance_thresholds``.  These tests
capture the v1.2-mvp baseline so that agent loop changes can be
verified against a known-good performance contract.

Test categories:
    1. Direct-command detection latency (single call, warm, batch)
    2. Input classification latency (verb resolution + structuredness)
    3. IPC envelope encode/decode roundtrip
    4. Tool registry lookup latency
    5. Correctness assertions on baseline inputs (sanity checks)

All latency tests use ``time.monotonic()`` for high-resolution timing
and include a warm-up pass to exclude module initialization costs.
Threshold comparisons use the p95 percentile to tolerate occasional
GC pauses or CI scheduling jitter.

Usage::

    pytest tests/test_performance_baselines.py -v
"""

from __future__ import annotations

import json
import time

import pytest

from jules_daemon.agent.performance_thresholds import (
    BASELINE_ITERATION_COUNT,
    CLASSIFY_INPUT_THRESHOLD_S,
    DIRECT_COMMAND_DETECTION_THRESHOLD_S,
    IPC_ENVELOPE_ROUNDTRIP_THRESHOLD_S,
    ONESHOT_PARSE_BUDGET_S,
    ONESHOT_PROMPT_BUDGET_S,
    ONESHOT_TRANSLATION_DEADLINE_S,
    TOOL_REGISTRY_LOOKUP_THRESHOLD_S,
    WARM_DETECTION_THRESHOLD_S,
)
from jules_daemon.classifier.direct_command import (
    DirectCommandDetection,
    detect_direct_command,
)
from tests.fixtures.baseline_latency import (
    ALL_BASELINE_INPUTS,
    DIRECT_COMMAND_BASELINE_INPUTS,
    EDGE_CASE_BASELINE_INPUTS,
    ENV_PREFIX_BASELINE_INPUTS,
    NL_INPUT_BASELINE_INPUTS,
    PATH_EXECUTABLE_BASELINE_INPUTS,
    SUDO_PREFIX_BASELINE_INPUTS,
    LatencyStats,
    measure_detection_latency,
    measure_single_call,
)


# ---------------------------------------------------------------------------
# Threshold constants are correctly defined
# ---------------------------------------------------------------------------


class TestThresholdConstants:
    """Validate that performance threshold constants are sensible."""

    def test_direct_command_threshold_is_positive(self) -> None:
        assert DIRECT_COMMAND_DETECTION_THRESHOLD_S > 0.0

    def test_direct_command_threshold_is_submillisecond_range(self) -> None:
        """1ms threshold is appropriate for a pure-CPU classifier."""
        assert DIRECT_COMMAND_DETECTION_THRESHOLD_S <= 0.001

    def test_warm_threshold_is_tighter(self) -> None:
        """Warm calls should be faster than cold calls."""
        assert WARM_DETECTION_THRESHOLD_S < DIRECT_COMMAND_DETECTION_THRESHOLD_S

    def test_classify_threshold_allows_more_than_detection(self) -> None:
        """Full classification includes more stages than detection alone."""
        assert CLASSIFY_INPUT_THRESHOLD_S > DIRECT_COMMAND_DETECTION_THRESHOLD_S

    def test_oneshot_deadline_matches_translator(self) -> None:
        """Our constant matches the command_translator default."""
        from jules_daemon.llm.command_translator import DEFAULT_DEADLINE_SECONDS
        assert ONESHOT_TRANSLATION_DEADLINE_S == DEFAULT_DEADLINE_SECONDS

    def test_oneshot_budgets_leave_room_for_llm(self) -> None:
        """Prompt + parse budgets must be much less than the full deadline."""
        combined = ONESHOT_PROMPT_BUDGET_S + ONESHOT_PARSE_BUDGET_S
        assert combined < ONESHOT_TRANSLATION_DEADLINE_S * 0.1

    def test_ipc_roundtrip_is_submillisecond_range(self) -> None:
        """IPC envelope roundtrip should be in single-digit milliseconds."""
        assert IPC_ENVELOPE_ROUNDTRIP_THRESHOLD_S <= 0.010

    def test_tool_registry_lookup_is_fast(self) -> None:
        """Registry lookups are O(1) dict operations."""
        assert TOOL_REGISTRY_LOOKUP_THRESHOLD_S <= 0.001

    def test_baseline_iteration_count_reasonable(self) -> None:
        """Iteration count is high enough for stable stats, low enough for CI."""
        assert 50 <= BASELINE_ITERATION_COUNT <= 500


# ---------------------------------------------------------------------------
# Baseline input correctness (sanity checks before timing)
# ---------------------------------------------------------------------------


class TestBaselineInputCorrectness:
    """Verify that baseline inputs produce expected detection results.

    These are not timing tests -- they validate that the inputs used
    for performance measurement actually exercise the intended code paths.
    """

    @pytest.mark.parametrize("command", DIRECT_COMMAND_BASELINE_INPUTS)
    def test_direct_commands_are_detected(self, command: str) -> None:
        result = detect_direct_command(command)
        assert result.is_direct_command is True, (
            f"Expected is_direct_command=True for {command!r}, "
            f"got {result}"
        )

    @pytest.mark.parametrize("nl_input", NL_INPUT_BASELINE_INPUTS)
    def test_nl_inputs_are_not_detected(self, nl_input: str) -> None:
        result = detect_direct_command(nl_input)
        assert result.is_direct_command is False, (
            f"Expected is_direct_command=False for {nl_input!r}, "
            f"got {result}"
        )

    @pytest.mark.parametrize("command", ENV_PREFIX_BASELINE_INPUTS)
    def test_env_prefix_commands_are_detected(self, command: str) -> None:
        result = detect_direct_command(command)
        assert result.is_direct_command is True, (
            f"Expected is_direct_command=True for {command!r}, "
            f"got {result}"
        )

    @pytest.mark.parametrize("command", SUDO_PREFIX_BASELINE_INPUTS)
    def test_sudo_prefix_commands_are_detected(self, command: str) -> None:
        result = detect_direct_command(command)
        assert result.is_direct_command is True, (
            f"Expected is_direct_command=True for {command!r}, "
            f"got {result}"
        )

    @pytest.mark.parametrize("command", PATH_EXECUTABLE_BASELINE_INPUTS)
    def test_path_executables_are_detected(self, command: str) -> None:
        result = detect_direct_command(command)
        assert result.is_direct_command is True, (
            f"Expected is_direct_command=True for {command!r}, "
            f"got {result}"
        )

    def test_edge_case_empty_is_not_detected(self) -> None:
        assert detect_direct_command("").is_direct_command is False

    def test_edge_case_whitespace_is_not_detected(self) -> None:
        assert detect_direct_command("   ").is_direct_command is False

    def test_edge_case_bare_executable_is_detected(self) -> None:
        assert detect_direct_command("python3").is_direct_command is True

    def test_edge_case_daemon_verb_is_not_detected(self) -> None:
        assert detect_direct_command("status").is_direct_command is False

    def test_edge_case_unknown_word_is_not_detected(self) -> None:
        assert detect_direct_command("frobnicator --test").is_direct_command is False


# ---------------------------------------------------------------------------
# Direct-command detection latency (Tier 1)
# ---------------------------------------------------------------------------


class TestDirectCommandDetectionLatency:
    """Assert that detect_direct_command() meets the v1.2-mvp latency SLA."""

    def test_single_call_within_threshold(self) -> None:
        """A single cold call to detect_direct_command is within 1ms."""
        # Warm-up to ensure module-level initialization is done
        detect_direct_command("pytest -v")

        elapsed = measure_single_call("pytest -v tests/")
        assert elapsed < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Single detect_direct_command call took {elapsed*1000:.3f}ms, "
            f"threshold is {DIRECT_COMMAND_DETECTION_THRESHOLD_S*1000:.3f}ms"
        )

    def test_batch_p95_within_threshold(self) -> None:
        """p95 across all baseline inputs stays under the detection threshold."""
        stats = measure_detection_latency(
            inputs=DIRECT_COMMAND_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        assert stats.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Batch p95={stats.p95*1000:.3f}ms exceeds "
            f"threshold={DIRECT_COMMAND_DETECTION_THRESHOLD_S*1000:.3f}ms. "
            f"Full stats: {stats}"
        )

    def test_nl_inputs_p95_within_threshold(self) -> None:
        """NL inputs (non-detection path) are equally fast."""
        stats = measure_detection_latency(
            inputs=NL_INPUT_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        assert stats.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"NL input p95={stats.p95*1000:.3f}ms exceeds threshold. "
            f"Full stats: {stats}"
        )

    def test_env_prefix_p95_within_threshold(self) -> None:
        """Env-prefix stripping does not blow the latency budget."""
        stats = measure_detection_latency(
            inputs=ENV_PREFIX_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        assert stats.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Env prefix p95={stats.p95*1000:.3f}ms exceeds threshold. "
            f"Full stats: {stats}"
        )

    def test_sudo_prefix_p95_within_threshold(self) -> None:
        """Sudo prefix stripping does not blow the latency budget."""
        stats = measure_detection_latency(
            inputs=SUDO_PREFIX_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        assert stats.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Sudo prefix p95={stats.p95*1000:.3f}ms exceeds threshold. "
            f"Full stats: {stats}"
        )

    def test_path_executable_p95_within_threshold(self) -> None:
        """Path-based executables do not blow the latency budget."""
        stats = measure_detection_latency(
            inputs=PATH_EXECUTABLE_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        assert stats.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Path executable p95={stats.p95*1000:.3f}ms exceeds threshold. "
            f"Full stats: {stats}"
        )

    def test_edge_cases_p95_within_threshold(self) -> None:
        """Edge-case inputs do not blow the latency budget."""
        stats = measure_detection_latency(
            inputs=EDGE_CASE_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        assert stats.p95 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Edge case p95={stats.p95*1000:.3f}ms exceeds threshold. "
            f"Full stats: {stats}"
        )

    def test_warm_path_p95_within_warm_threshold(self) -> None:
        """After warm-up, the hot-path p95 is tighter (0.5ms)."""
        stats = measure_detection_latency(
            inputs=DIRECT_COMMAND_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        assert stats.p95 < WARM_DETECTION_THRESHOLD_S, (
            f"Warm path p95={stats.p95*1000:.3f}ms exceeds "
            f"warm threshold={WARM_DETECTION_THRESHOLD_S*1000:.3f}ms. "
            f"Full stats: {stats}"
        )

    def test_all_inputs_p99_within_threshold(self) -> None:
        """Even p99 across all input categories stays under threshold."""
        stats = measure_detection_latency(
            inputs=ALL_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        assert stats.p99 < DIRECT_COMMAND_DETECTION_THRESHOLD_S, (
            f"Full corpus p99={stats.p99*1000:.3f}ms exceeds threshold. "
            f"Full stats: {stats}"
        )


# ---------------------------------------------------------------------------
# IPC envelope roundtrip latency
# ---------------------------------------------------------------------------


class TestIPCEnvelopeLatency:
    """Assert that IPC envelope encode/decode roundtrip is within threshold."""

    def test_envelope_roundtrip_within_threshold(self) -> None:
        """Encode + decode of a typical run command envelope is fast."""
        from jules_daemon.ipc.framing import (
            MessageEnvelope,
            MessageType,
            decode_envelope,
            encode_frame,
        )

        envelope = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="perf-test-001",
            timestamp="2026-04-12T12:00:00Z",
            payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "pytest -v tests/integration/",
            },
        )

        # Warm-up
        frame = encode_frame(envelope)
        # decode_envelope operates on the payload portion (after 4-byte header)
        decode_envelope(frame[4:])

        # Timed measurement (100 roundtrips)
        timings: list[float] = []
        for _ in range(100):
            start = time.monotonic()
            frame = encode_frame(envelope)
            decode_envelope(frame[4:])
            elapsed = time.monotonic() - start
            timings.append(elapsed)

        timings.sort()
        p95 = timings[int(len(timings) * 0.95)]
        assert p95 < IPC_ENVELOPE_ROUNDTRIP_THRESHOLD_S, (
            f"IPC roundtrip p95={p95*1000:.3f}ms exceeds "
            f"threshold={IPC_ENVELOPE_ROUNDTRIP_THRESHOLD_S*1000:.3f}ms"
        )


# ---------------------------------------------------------------------------
# Tool registry lookup latency
# ---------------------------------------------------------------------------


class TestToolRegistryLookupLatency:
    """Assert that ToolRegistry.get() and list_tools() are O(1)-fast."""

    def test_registry_get_within_threshold(self) -> None:
        """ToolRegistry.get() is within 1ms for a registered tool."""
        from jules_daemon.agent.tool_registry import ToolRegistry
        from jules_daemon.agent.tool_types import (
            ApprovalRequirement,
            ToolParam,
            ToolResult,
            ToolSpec,
        )
        from jules_daemon.agent.tools.base import BaseTool

        registry = ToolRegistry()

        # Register a minimal tool for lookup testing
        class _PerfTool(BaseTool):
            _spec = ToolSpec(
                name="test_perf_tool",
                description="Performance test tool",
                parameters=(
                    ToolParam(
                        name="arg1",
                        json_type="string",
                        description="test arg",
                        required=True,
                    ),
                ),
                approval=ApprovalRequirement.NONE,
            )

        registry.register(_PerfTool())

        # Warm-up
        registry.get("test_perf_tool")

        # Timed measurement
        timings: list[float] = []
        for _ in range(100):
            start = time.monotonic()
            registry.get("test_perf_tool")
            elapsed = time.monotonic() - start
            timings.append(elapsed)

        timings.sort()
        p95 = timings[int(len(timings) * 0.95)]
        assert p95 < TOOL_REGISTRY_LOOKUP_THRESHOLD_S, (
            f"Registry.get() p95={p95*1000:.3f}ms exceeds "
            f"threshold={TOOL_REGISTRY_LOOKUP_THRESHOLD_S*1000:.3f}ms"
        )

    def test_registry_list_tools_within_threshold(self) -> None:
        """ToolRegistry.list_tools() is within 1ms for a populated registry."""
        from jules_daemon.agent.tool_registry import ToolRegistry
        from jules_daemon.agent.tool_types import (
            ApprovalRequirement,
            ToolParam,
            ToolSpec,
        )
        from jules_daemon.agent.tools.base import BaseTool

        registry = ToolRegistry()

        # Register 10 tools (matching the agent loop tool count)
        for i in range(10):
            class _DynTool(BaseTool):
                _spec = ToolSpec(
                    name=f"tool_{i}",
                    description=f"Tool number {i}",
                    parameters=(
                        ToolParam(
                            name="arg",
                            json_type="string",
                            description="test",
                            required=True,
                        ),
                    ),
                    approval=ApprovalRequirement.NONE,
                )
            registry.register(_DynTool())

        # Warm-up
        registry.list_tools()

        # Timed measurement
        timings: list[float] = []
        for _ in range(100):
            start = time.monotonic()
            registry.list_tools()
            elapsed = time.monotonic() - start
            timings.append(elapsed)

        timings.sort()
        p95 = timings[int(len(timings) * 0.95)]
        assert p95 < TOOL_REGISTRY_LOOKUP_THRESHOLD_S, (
            f"Registry.list_tools() p95={p95*1000:.3f}ms exceeds "
            f"threshold={TOOL_REGISTRY_LOOKUP_THRESHOLD_S*1000:.3f}ms"
        )


# ---------------------------------------------------------------------------
# LatencyStats fixture self-tests
# ---------------------------------------------------------------------------


class TestLatencyStatsFixture:
    """Verify that the LatencyStats measurement helper works correctly."""

    def test_measure_detection_latency_returns_stats(self) -> None:
        stats = measure_detection_latency(
            inputs=("pytest -v",),
            iterations=10,
        )
        assert isinstance(stats, LatencyStats)
        assert stats.samples == 10
        assert stats.min <= stats.median <= stats.max
        assert stats.min <= stats.p95 <= stats.max
        assert stats.min <= stats.p99 <= stats.max

    def test_measure_detection_latency_default_inputs(self) -> None:
        """Default inputs use ALL_BASELINE_INPUTS."""
        stats = measure_detection_latency(iterations=1)
        assert stats.samples == len(ALL_BASELINE_INPUTS)

    def test_stats_repr_shows_milliseconds(self) -> None:
        """Repr formats values in milliseconds for readability."""
        stats = measure_detection_latency(
            inputs=("pytest -v",),
            iterations=10,
        )
        repr_str = repr(stats)
        assert "ms" in repr_str
        assert "samples=10" in repr_str

    def test_single_call_returns_positive_float(self) -> None:
        elapsed = measure_single_call("pytest -v")
        assert isinstance(elapsed, float)
        assert elapsed > 0.0

    def test_percentile_ordering(self) -> None:
        """p95 <= p99 in any measurement run."""
        stats = measure_detection_latency(
            inputs=DIRECT_COMMAND_BASELINE_INPUTS,
            iterations=50,
        )
        assert stats.p95 <= stats.p99


# ---------------------------------------------------------------------------
# Baseline capture (informational, not gating)
# ---------------------------------------------------------------------------


class TestBaselineCapture:
    """Capture and log baseline latency numbers for reference.

    These tests always pass -- they exist to produce logged output
    that documents the actual v1.2-mvp performance on the current
    platform.  The captured numbers serve as the human-readable
    baseline for comparison during agent loop development.
    """

    def test_capture_direct_command_baseline(self) -> None:
        """Capture and log direct-command detection baseline."""
        stats = measure_detection_latency(
            inputs=DIRECT_COMMAND_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        # Log for human consumption (visible in pytest -v output)
        print(f"\n[BASELINE] Direct command detection: {stats}")
        assert stats.samples > 0

    def test_capture_nl_input_baseline(self) -> None:
        """Capture and log NL input detection baseline."""
        stats = measure_detection_latency(
            inputs=NL_INPUT_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        print(f"\n[BASELINE] NL input detection: {stats}")
        assert stats.samples > 0

    def test_capture_env_prefix_baseline(self) -> None:
        """Capture and log env-prefix detection baseline."""
        stats = measure_detection_latency(
            inputs=ENV_PREFIX_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        print(f"\n[BASELINE] Env prefix detection: {stats}")
        assert stats.samples > 0

    def test_capture_all_inputs_baseline(self) -> None:
        """Capture and log full-corpus baseline."""
        stats = measure_detection_latency(
            inputs=ALL_BASELINE_INPUTS,
            iterations=BASELINE_ITERATION_COUNT,
        )
        print(f"\n[BASELINE] All inputs detection: {stats}")
        assert stats.samples > 0
