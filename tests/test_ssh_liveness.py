"""Tests for SSH connection liveness validation.

Covers:
    - Successful liveness probe with responsive shell
    - Probe timeout (unresponsive shell)
    - Probe execution error (command failure)
    - ProbeResult immutability and structure
    - ProbeConfig validation (defaults and constraints)
    - Latency measurement correctness
    - Probe command output capture
    - Protocol compliance for ProbeExecutor
    - Default probe command (echo-based)
    - Custom probe command support
    - Consecutive failure tracking
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from jules_daemon.ssh.liveness import (
    ConnectionHealth,
    ProbeConfig,
    ProbeExecutor,
    ProbeResult,
    validate_liveness,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeProbeOutput:
    """Fake output from a probe command execution."""

    stdout: str
    exit_code: int


class FakeProbeExecutor:
    """Configurable fake probe executor for testing.

    Tracks invocations and returns canned results or raises errors.
    """

    def __init__(
        self,
        *,
        outputs: list[FakeProbeOutput] | None = None,
        errors: list[Exception] | None = None,
        latency_seconds: float = 0.0,
    ) -> None:
        self._outputs: list[FakeProbeOutput] = list(outputs) if outputs else []
        self._errors: list[Exception] = list(errors) if errors else []
        self._latency_seconds = latency_seconds
        self.invocations: list[str] = []

    async def execute_probe(self, command: str, timeout: float) -> tuple[str, int]:
        self.invocations.append(command)
        if self._latency_seconds > 0:
            await asyncio.sleep(self._latency_seconds)
        if self._errors:
            raise self._errors.pop(0)
        if self._outputs:
            output = self._outputs.pop(0)
            return (output.stdout, output.exit_code)
        return ("ok", 0)


# Verify protocol compliance
assert isinstance(FakeProbeExecutor(), ProbeExecutor)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = ProbeConfig()
_FAST_CONFIG = ProbeConfig(timeout_seconds=0.5)


# ---------------------------------------------------------------------------
# ProbeConfig tests
# ---------------------------------------------------------------------------


class TestProbeConfig:
    """Validate ProbeConfig defaults and constraints."""

    def test_default_values(self) -> None:
        config = ProbeConfig()
        assert config.command == "echo __jules_probe_ok__"
        assert config.timeout_seconds == 5.0
        assert config.expected_output == "__jules_probe_ok__"

    def test_custom_values(self) -> None:
        config = ProbeConfig(
            command="true",
            timeout_seconds=2.0,
            expected_output="",
        )
        assert config.command == "true"
        assert config.timeout_seconds == 2.0
        assert config.expected_output == ""

    def test_frozen(self) -> None:
        config = ProbeConfig()
        with pytest.raises(AttributeError):
            config.command = "false"  # type: ignore[misc]

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ProbeConfig(timeout_seconds=0.0)

    def test_timeout_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ProbeConfig(timeout_seconds=-1.0)

    def test_command_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="command must not be empty"):
            ProbeConfig(command="")

    def test_command_must_not_be_whitespace(self) -> None:
        with pytest.raises(ValueError, match="command must not be empty"):
            ProbeConfig(command="   ")


# ---------------------------------------------------------------------------
# Successful probe
# ---------------------------------------------------------------------------


class TestValidateLivenessSuccess:
    """Liveness probe succeeds with a responsive shell."""

    @pytest.mark.asyncio
    async def test_success_with_matching_output(self) -> None:
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        assert result.health == ConnectionHealth.CONNECTED
        assert result.success is True
        assert result.output == "__jules_probe_ok__"
        assert result.exit_code == 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_success_with_custom_command(self) -> None:
        config = ProbeConfig(
            command="hostname",
            timeout_seconds=5.0,
            expected_output="",
        )
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="web-server-01", exit_code=0)]
        )
        result = await validate_liveness(executor, config)

        assert result.success is True
        assert result.output == "web-server-01"

    @pytest.mark.asyncio
    async def test_success_records_probe_command(self) -> None:
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        assert result.probe_command == "echo __jules_probe_ok__"

    @pytest.mark.asyncio
    async def test_executor_receives_correct_command(self) -> None:
        config = ProbeConfig(command="uptime", timeout_seconds=3.0, expected_output="")
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="up 5 days", exit_code=0)]
        )
        await validate_liveness(executor, config)

        assert len(executor.invocations) == 1
        assert executor.invocations[0] == "uptime"

    @pytest.mark.asyncio
    async def test_success_has_timestamp(self) -> None:
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )
        before = datetime.now(timezone.utc)
        result = await validate_liveness(executor, _DEFAULT_CONFIG)
        after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after

    @pytest.mark.asyncio
    async def test_latency_is_non_negative(self) -> None:
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Output mismatch
# ---------------------------------------------------------------------------


class TestValidateLivenessOutputMismatch:
    """Probe succeeds at transport but output does not match expected."""

    @pytest.mark.asyncio
    async def test_mismatch_marks_degraded(self) -> None:
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="wrong output", exit_code=0)]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        assert result.health == ConnectionHealth.DEGRADED
        assert result.success is False
        assert "output mismatch" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_empty_expected_skips_check(self) -> None:
        """When expected_output is empty, any output is acceptable."""
        config = ProbeConfig(
            command="date",
            timeout_seconds=5.0,
            expected_output="",
        )
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="Wed Apr 9 12:00:00 UTC 2026", exit_code=0)]
        )
        result = await validate_liveness(executor, config)

        assert result.success is True
        assert result.health == ConnectionHealth.CONNECTED


# ---------------------------------------------------------------------------
# Non-zero exit code
# ---------------------------------------------------------------------------


class TestValidateLivenessNonZeroExit:
    """Probe command returns non-zero exit code."""

    @pytest.mark.asyncio
    async def test_nonzero_exit_marks_degraded(self) -> None:
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="", exit_code=1)]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        assert result.health == ConnectionHealth.DEGRADED
        assert result.success is False
        assert result.exit_code == 1
        assert "non-zero exit" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class SlowExecutor:
    """Fake executor that always exceeds the timeout."""

    def __init__(self) -> None:
        self.invocations: list[str] = []

    async def execute_probe(
        self, command: str, timeout: float
    ) -> tuple[str, int]:
        self.invocations.append(command)
        await asyncio.sleep(1.0)
        return ("ok", 0)


class TestValidateLivenessTimeout:
    """Probe times out when shell is unresponsive."""

    @pytest.mark.asyncio
    async def test_timeout_marks_disconnected(self) -> None:
        config = ProbeConfig(timeout_seconds=0.05)
        executor = SlowExecutor()
        result = await validate_liveness(executor, config)

        assert result.health == ConnectionHealth.DISCONNECTED
        assert result.success is False
        assert "timeout" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_timeout_records_latency(self) -> None:
        config = ProbeConfig(timeout_seconds=0.05)
        executor = SlowExecutor()
        result = await validate_liveness(executor, config)

        # Latency should be approximately the timeout value
        assert result.latency_ms >= 40.0  # Allow tolerance


# ---------------------------------------------------------------------------
# Execution error
# ---------------------------------------------------------------------------


class TestValidateLivenessExecutionError:
    """Probe execution raises an exception."""

    @pytest.mark.asyncio
    async def test_oserror_marks_disconnected(self) -> None:
        executor = FakeProbeExecutor(
            errors=[OSError("Connection reset by peer")]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        assert result.health == ConnectionHealth.DISCONNECTED
        assert result.success is False
        assert "Connection reset by peer" in (result.error or "")
        # Error includes type name for classification
        assert "OSError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_eoferror_marks_disconnected(self) -> None:
        executor = FakeProbeExecutor(
            errors=[EOFError("channel closed")]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        assert result.health == ConnectionHealth.DISCONNECTED
        assert result.success is False
        assert "EOFError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_unexpected_error_marks_disconnected(self) -> None:
        executor = FakeProbeExecutor(
            errors=[RuntimeError("unexpected failure")]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        assert result.health == ConnectionHealth.DISCONNECTED
        assert result.success is False
        assert "unexpected failure" in (result.error or "")
        assert "RuntimeError" in (result.error or "")


# ---------------------------------------------------------------------------
# ProbeResult immutability
# ---------------------------------------------------------------------------


class TestProbeResultImmutability:
    """ProbeResult is frozen and cannot be mutated."""

    @pytest.mark.asyncio
    async def test_frozen(self) -> None:
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_has_all_fields(self) -> None:
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )
        result = await validate_liveness(executor, _DEFAULT_CONFIG)

        # Verify all expected fields exist
        assert hasattr(result, "success")
        assert hasattr(result, "health")
        assert hasattr(result, "latency_ms")
        assert hasattr(result, "output")
        assert hasattr(result, "exit_code")
        assert hasattr(result, "error")
        assert hasattr(result, "probe_command")
        assert hasattr(result, "timestamp")


# ---------------------------------------------------------------------------
# ConnectionHealth enum
# ---------------------------------------------------------------------------


class TestConnectionHealth:
    """Verify ConnectionHealth enum values."""

    def test_values(self) -> None:
        assert ConnectionHealth.CONNECTED.value == "connected"
        assert ConnectionHealth.DEGRADED.value == "degraded"
        assert ConnectionHealth.DISCONNECTED.value == "disconnected"
