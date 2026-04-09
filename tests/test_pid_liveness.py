"""Tests for remote PID liveness validation.

Verifies that the PID liveness validator:
- Checks whether a remote process is alive via kill -0 (primary)
- Falls back to /proc/<PID> check when kill -0 is inconclusive
- Returns structured alive/dead/unknown result
- Handles PID validation (positive integer, injection prevention)
- Handles executor timeout gracefully
- Handles executor transport errors gracefully
- Classifies kill -0 "Operation not permitted" as ALIVE (EPERM)
- Classifies kill -0 "No such process" as DEAD (ESRCH)
- Measures latency for both checks
- Records which method confirmed the result
- Is immutable (frozen dataclass result)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from jules_daemon.ssh.pid_liveness import (
    PidCheckConfig,
    PidCheckMethod,
    PidLivenessResult,
    PidStatus,
    validate_pid_liveness,
)
from jules_daemon.ssh.liveness import ProbeExecutor


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeCommandResult:
    """Canned result for a remote command execution."""

    stdout: str
    exit_code: int


class FakePidExecutor:
    """Configurable fake executor for PID liveness tests.

    Tracks invocations and returns canned results or raises errors.
    Supports per-command responses by matching command prefixes.
    """

    def __init__(
        self,
        *,
        results: list[FakeCommandResult] | None = None,
        errors: list[Exception] | None = None,
        latency_seconds: float = 0.0,
    ) -> None:
        self._results: list[FakeCommandResult] = list(results) if results else []
        self._errors: list[Exception] = list(errors) if errors else []
        self._latency_seconds = latency_seconds
        self.invocations: list[str] = []

    async def execute_probe(
        self, command: str, timeout: float
    ) -> tuple[str, int]:
        self.invocations.append(command)
        if self._latency_seconds > 0:
            await asyncio.sleep(self._latency_seconds)
        if self._errors:
            raise self._errors.pop(0)
        if self._results:
            result = self._results.pop(0)
            return (result.stdout, result.exit_code)
        return ("", 0)


# Verify protocol compliance
assert isinstance(FakePidExecutor(), ProbeExecutor)


class SlowPidExecutor:
    """Executor that always exceeds the timeout."""

    def __init__(self) -> None:
        self.invocations: list[str] = []

    async def execute_probe(
        self, command: str, timeout: float
    ) -> tuple[str, int]:
        self.invocations.append(command)
        await asyncio.sleep(1.0)
        return ("", 0)


# ---------------------------------------------------------------------------
# PidStatus enum
# ---------------------------------------------------------------------------


class TestPidStatus:
    def test_all_values_exist(self) -> None:
        assert PidStatus.ALIVE.value == "alive"
        assert PidStatus.DEAD.value == "dead"
        assert PidStatus.UNKNOWN.value == "unknown"


# ---------------------------------------------------------------------------
# PidCheckMethod enum
# ---------------------------------------------------------------------------


class TestPidCheckMethod:
    def test_all_values_exist(self) -> None:
        assert PidCheckMethod.KILL_ZERO.value == "kill_zero"
        assert PidCheckMethod.PROC_FALLBACK.value == "proc_fallback"


# ---------------------------------------------------------------------------
# PidCheckConfig
# ---------------------------------------------------------------------------


class TestPidCheckConfig:
    def test_default_timeout(self) -> None:
        config = PidCheckConfig()
        assert config.timeout_seconds == 5.0

    def test_custom_timeout(self) -> None:
        config = PidCheckConfig(timeout_seconds=10.0)
        assert config.timeout_seconds == 10.0

    def test_frozen(self) -> None:
        config = PidCheckConfig()
        with pytest.raises(AttributeError):
            config.timeout_seconds = 3.0  # type: ignore[misc]

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            PidCheckConfig(timeout_seconds=0.0)

    def test_timeout_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            PidCheckConfig(timeout_seconds=-1.0)


# ---------------------------------------------------------------------------
# PidLivenessResult immutability
# ---------------------------------------------------------------------------


class TestPidLivenessResult:
    def test_frozen(self) -> None:
        result = PidLivenessResult(
            pid=1234,
            alive=True,
            status=PidStatus.ALIVE,
            method=PidCheckMethod.KILL_ZERO,
            kill_zero_exit_code=0,
            kill_zero_output="",
            proc_exit_code=None,
            proc_output="",
            error=None,
            latency_ms=5.0,
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            result.alive = False  # type: ignore[misc]

    def test_has_all_fields(self) -> None:
        result = PidLivenessResult(
            pid=42,
            alive=False,
            status=PidStatus.DEAD,
            method=PidCheckMethod.PROC_FALLBACK,
            kill_zero_exit_code=1,
            kill_zero_output="No such process",
            proc_exit_code=1,
            proc_output="",
            error=None,
            latency_ms=10.5,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.pid == 42
        assert result.alive is False
        assert result.status == PidStatus.DEAD
        assert result.method == PidCheckMethod.PROC_FALLBACK
        assert result.kill_zero_exit_code == 1
        assert result.kill_zero_output == "No such process"
        assert result.proc_exit_code == 1
        assert result.proc_output == ""
        assert result.error is None
        assert result.latency_ms == 10.5


# ---------------------------------------------------------------------------
# PID validation
# ---------------------------------------------------------------------------


class TestPidValidation:
    @pytest.mark.asyncio
    async def test_negative_pid_raises(self) -> None:
        executor = FakePidExecutor()
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            await validate_pid_liveness(executor, pid=-1)

    @pytest.mark.asyncio
    async def test_zero_pid_raises(self) -> None:
        executor = FakePidExecutor()
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            await validate_pid_liveness(executor, pid=0)

    @pytest.mark.asyncio
    async def test_valid_pid_does_not_raise(self) -> None:
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=1234)
        assert result.pid == 1234


# ---------------------------------------------------------------------------
# kill -0: Process is alive (exit code 0)
# ---------------------------------------------------------------------------


class TestKillZeroAlive:
    """kill -0 returns exit code 0 -- process is alive and signalable."""

    @pytest.mark.asyncio
    async def test_alive_status(self) -> None:
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is True
        assert result.status == PidStatus.ALIVE

    @pytest.mark.asyncio
    async def test_method_is_kill_zero(self) -> None:
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.method == PidCheckMethod.KILL_ZERO

    @pytest.mark.asyncio
    async def test_kill_zero_exit_code_recorded(self) -> None:
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.kill_zero_exit_code == 0

    @pytest.mark.asyncio
    async def test_no_error(self) -> None:
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.error is None

    @pytest.mark.asyncio
    async def test_proc_not_invoked(self) -> None:
        """When kill -0 is definitive, /proc fallback is not needed."""
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.proc_exit_code is None
        # Only one command should have been invoked
        assert len(executor.invocations) == 1
        assert "kill" in executor.invocations[0]

    @pytest.mark.asyncio
    async def test_has_timestamp(self) -> None:
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        before = datetime.now(timezone.utc)
        result = await validate_pid_liveness(executor, pid=5678)
        after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after

    @pytest.mark.asyncio
    async def test_latency_is_non_negative(self) -> None:
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_correct_command_sent(self) -> None:
        """Verify the kill -0 command includes the correct PID."""
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        await validate_pid_liveness(executor, pid=9999)

        assert len(executor.invocations) == 1
        assert "9999" in executor.invocations[0]
        assert "kill" in executor.invocations[0]


# ---------------------------------------------------------------------------
# kill -0: Operation not permitted (EPERM) -- process alive, different user
# ---------------------------------------------------------------------------


class TestKillZeroEperm:
    """kill -0 returns non-zero with 'Operation not permitted' -- ALIVE."""

    @pytest.mark.asyncio
    async def test_eperm_is_alive(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(
                    stdout="kill: (5678): Operation not permitted",
                    exit_code=1,
                )
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is True
        assert result.status == PidStatus.ALIVE
        assert result.method == PidCheckMethod.KILL_ZERO

    @pytest.mark.asyncio
    async def test_eperm_case_insensitive(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(
                    stdout="kill: (5678): operation not permitted",
                    exit_code=1,
                )
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is True
        assert result.status == PidStatus.ALIVE

    @pytest.mark.asyncio
    async def test_eperm_records_output(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(
                    stdout="kill: (5678): Operation not permitted",
                    exit_code=1,
                )
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert "Operation not permitted" in result.kill_zero_output

    @pytest.mark.asyncio
    async def test_eperm_does_not_fall_through_to_proc(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(
                    stdout="kill: (5678): Operation not permitted",
                    exit_code=1,
                )
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert len(executor.invocations) == 1
        assert result.proc_exit_code is None


# ---------------------------------------------------------------------------
# kill -0: No such process (ESRCH) -- process is dead
# ---------------------------------------------------------------------------


class TestKillZeroNoSuchProcess:
    """kill -0 with 'No such process' output -- DEAD."""

    @pytest.mark.asyncio
    async def test_no_such_process_is_dead(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(
                    stdout="kill: (5678): No such process",
                    exit_code=1,
                )
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is False
        assert result.status == PidStatus.DEAD
        assert result.method == PidCheckMethod.KILL_ZERO

    @pytest.mark.asyncio
    async def test_no_such_process_case_insensitive(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(
                    stdout="kill: (5678): no such process",
                    exit_code=1,
                )
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is False
        assert result.status == PidStatus.DEAD

    @pytest.mark.asyncio
    async def test_no_such_process_records_output(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(
                    stdout="kill: (5678): No such process",
                    exit_code=1,
                )
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert "No such process" in result.kill_zero_output
        assert result.kill_zero_exit_code == 1


# ---------------------------------------------------------------------------
# kill -0: Ambiguous failure -- falls through to /proc
# ---------------------------------------------------------------------------


class TestKillZeroAmbiguousFallsToProc:
    """kill -0 gives unrecognized error -- /proc fallback is attempted."""

    @pytest.mark.asyncio
    async def test_ambiguous_kill_falls_to_proc_alive(self) -> None:
        """Unrecognized kill -0 error + /proc exists -> ALIVE via PROC."""
        executor = FakePidExecutor(
            results=[
                # kill -0: ambiguous error
                FakeCommandResult(stdout="kill: unknown error", exit_code=1),
                # /proc fallback: directory exists
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is True
        assert result.status == PidStatus.ALIVE
        assert result.method == PidCheckMethod.PROC_FALLBACK

    @pytest.mark.asyncio
    async def test_ambiguous_kill_falls_to_proc_dead(self) -> None:
        """Unrecognized kill -0 error + /proc absent -> DEAD via PROC."""
        executor = FakePidExecutor(
            results=[
                # kill -0: ambiguous error
                FakeCommandResult(stdout="kill: unknown error", exit_code=1),
                # /proc fallback: directory does not exist
                FakeCommandResult(stdout="", exit_code=1),
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is False
        assert result.status == PidStatus.DEAD
        assert result.method == PidCheckMethod.PROC_FALLBACK

    @pytest.mark.asyncio
    async def test_two_commands_invoked(self) -> None:
        """Both kill -0 and /proc commands should be invoked."""
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(stdout="kill: something weird", exit_code=1),
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        await validate_pid_liveness(executor, pid=5678)

        assert len(executor.invocations) == 2
        assert "kill" in executor.invocations[0]
        assert "proc" in executor.invocations[1].lower() or "/proc/" in executor.invocations[1]

    @pytest.mark.asyncio
    async def test_proc_command_contains_pid(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(stdout="kill: error", exit_code=1),
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        await validate_pid_liveness(executor, pid=7777)

        assert "7777" in executor.invocations[1]

    @pytest.mark.asyncio
    async def test_proc_exit_code_recorded(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(stdout="kill: error", exit_code=1),
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.proc_exit_code == 0

    @pytest.mark.asyncio
    async def test_kill_zero_output_still_recorded(self) -> None:
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(stdout="kill: unknown error", exit_code=1),
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert "unknown error" in result.kill_zero_output
        assert result.kill_zero_exit_code == 1


# ---------------------------------------------------------------------------
# kill -0 timeout -- falls through to /proc
# ---------------------------------------------------------------------------


class TestKillZeroTimeout:
    """kill -0 times out -- /proc fallback is attempted."""

    @pytest.mark.asyncio
    async def test_kill_timeout_falls_to_proc(self) -> None:
        """kill -0 times out, /proc succeeds -> ALIVE via PROC."""
        call_count = 0

        class TimeoutThenSuccessExecutor:
            def __init__(self) -> None:
                self.invocations: list[str] = []

            async def execute_probe(
                self, command: str, timeout: float
            ) -> tuple[str, int]:
                self.invocations.append(command)
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call (kill -0) times out
                    raise TimeoutError("kill -0 timed out")
                # Second call (/proc) succeeds
                return ("", 0)

        executor = TimeoutThenSuccessExecutor()
        config = PidCheckConfig(timeout_seconds=0.5)
        result = await validate_pid_liveness(executor, pid=5678, config=config)

        assert result.alive is True
        assert result.status == PidStatus.ALIVE
        assert result.method == PidCheckMethod.PROC_FALLBACK
        assert len(executor.invocations) == 2


# ---------------------------------------------------------------------------
# kill -0 transport error -- falls through to /proc
# ---------------------------------------------------------------------------


class TestKillZeroTransportError:
    """kill -0 raises transport error -- /proc fallback is attempted."""

    @pytest.mark.asyncio
    async def test_oserror_falls_to_proc(self) -> None:
        call_count = 0

        class ErrorThenSuccessExecutor:
            def __init__(self) -> None:
                self.invocations: list[str] = []

            async def execute_probe(
                self, command: str, timeout: float
            ) -> tuple[str, int]:
                self.invocations.append(command)
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("Connection reset by peer")
                return ("", 0)

        executor = ErrorThenSuccessExecutor()
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is True
        assert result.method == PidCheckMethod.PROC_FALLBACK


# ---------------------------------------------------------------------------
# Both kill -0 and /proc fail -- UNKNOWN
# ---------------------------------------------------------------------------


class TestBothChecksFail:
    """Both kill -0 and /proc fail -> UNKNOWN status."""

    @pytest.mark.asyncio
    async def test_both_timeout_returns_unknown(self) -> None:
        class AlwaysTimeoutExecutor:
            def __init__(self) -> None:
                self.invocations: list[str] = []

            async def execute_probe(
                self, command: str, timeout: float
            ) -> tuple[str, int]:
                self.invocations.append(command)
                raise TimeoutError("timed out")

        executor = AlwaysTimeoutExecutor()
        config = PidCheckConfig(timeout_seconds=0.5)
        result = await validate_pid_liveness(executor, pid=5678, config=config)

        assert result.alive is False
        assert result.status == PidStatus.UNKNOWN
        assert result.method is None
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_both_error_returns_unknown(self) -> None:
        class AlwaysErrorExecutor:
            def __init__(self) -> None:
                self.invocations: list[str] = []

            async def execute_probe(
                self, command: str, timeout: float
            ) -> tuple[str, int]:
                self.invocations.append(command)
                raise OSError("connection lost")

        executor = AlwaysErrorExecutor()
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is False
        assert result.status == PidStatus.UNKNOWN
        assert result.method is None
        assert result.error is not None
        assert len(executor.invocations) == 2

    @pytest.mark.asyncio
    async def test_unknown_has_error_description(self) -> None:
        class AlwaysErrorExecutor:
            def __init__(self) -> None:
                self.invocations: list[str] = []

            async def execute_probe(
                self, command: str, timeout: float
            ) -> tuple[str, int]:
                self.invocations.append(command)
                raise RuntimeError("unexpected failure")

        executor = AlwaysErrorExecutor()
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.error is not None
        assert len(result.error) > 0


# ---------------------------------------------------------------------------
# /proc fallback: process alive
# ---------------------------------------------------------------------------


class TestProcFallbackAlive:
    """kill -0 is inconclusive, /proc shows process exists."""

    @pytest.mark.asyncio
    async def test_proc_alive_with_output(self) -> None:
        executor = FakePidExecutor(
            results=[
                # kill -0: empty/ambiguous
                FakeCommandResult(stdout="", exit_code=127),
                # /proc: directory exists
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is True
        assert result.status == PidStatus.ALIVE
        assert result.method == PidCheckMethod.PROC_FALLBACK
        assert result.proc_exit_code == 0


# ---------------------------------------------------------------------------
# /proc fallback: process dead
# ---------------------------------------------------------------------------


class TestProcFallbackDead:
    """kill -0 is inconclusive, /proc shows process does not exist."""

    @pytest.mark.asyncio
    async def test_proc_dead(self) -> None:
        executor = FakePidExecutor(
            results=[
                # kill -0: ambiguous
                FakeCommandResult(stdout="", exit_code=127),
                # /proc: no directory
                FakeCommandResult(stdout="", exit_code=1),
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.alive is False
        assert result.status == PidStatus.DEAD
        assert result.method == PidCheckMethod.PROC_FALLBACK
        assert result.proc_exit_code == 1


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    @pytest.mark.asyncio
    async def test_uses_default_config_when_none(self) -> None:
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=1234, config=None)

        assert result.alive is True

    @pytest.mark.asyncio
    async def test_custom_config_is_used(self) -> None:
        config = PidCheckConfig(timeout_seconds=2.0)
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=1234, config=config)

        assert result.alive is True


# ---------------------------------------------------------------------------
# Command injection prevention
# ---------------------------------------------------------------------------


class TestCommandInjectionPrevention:
    """Verify PID is validated before being used in shell commands."""

    @pytest.mark.asyncio
    async def test_rejects_non_integer_like_values(self) -> None:
        """Even though pid is typed as int, verify validation is solid."""
        executor = FakePidExecutor()
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            await validate_pid_liveness(executor, pid=-100)

    @pytest.mark.asyncio
    async def test_large_pid_is_valid(self) -> None:
        """Linux supports PIDs up to 2^22 (4194304). Should work fine."""
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=4194304)
        assert result.pid == 4194304
        assert result.alive is True


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


class TestLatencyMeasurement:
    @pytest.mark.asyncio
    async def test_latency_includes_all_checks(self) -> None:
        """Latency should cover the full check duration (both kill + proc)."""
        executor = FakePidExecutor(
            results=[
                FakeCommandResult(stdout="kill: error", exit_code=1),
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        # We can't assert exact timing, but latency must be non-negative
        assert result.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_single_check_latency(self) -> None:
        executor = FakePidExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await validate_pid_liveness(executor, pid=5678)

        assert result.latency_ms >= 0.0
