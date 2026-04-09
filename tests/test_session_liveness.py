"""Tests for per-session liveness aggregator.

Verifies that the session liveness aggregator:
- Combines process-state verdict and SSH-connection verdict into a unified
  LivenessResult for each session
- Exposes a single check_session_liveness(session, executor) entry point
- Returns SessionHealth.HEALTHY when both probes pass
- Returns SessionHealth.DEGRADED when process is alive but SSH is degraded
- Returns SessionHealth.CONNECTION_LOST when process is alive but SSH
  is disconnected
- Returns SessionHealth.PROCESS_DEAD when local daemon process is dead
  (regardless of SSH state)
- Returns SessionHealth.UNKNOWN when process state is ERROR and SSH
  cannot be determined
- Handles missing daemon PID gracefully (returns UNKNOWN)
- Handles missing SSH executor gracefully (returns partial result)
- Is immutable (frozen dataclass result)
- Records timestamps and latency for the composite check
- Preserves sub-results (ProcessCheckResult and ProbeResult) for diagnostics
- Provides a boolean .alive property for quick checks
- Validates session input (must have required fields)
"""

from __future__ import annotations

import asyncio
import errno
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from jules_daemon.monitor.process_state import (
    ProcessCheckResult,
    ProcessVerdict,
)
from jules_daemon.monitor.session_liveness import (
    LivenessResult,
    SessionHealth,
    SessionInfo,
    _derive_health,
    check_session_liveness,
)
from jules_daemon.ssh.liveness import (
    ConnectionHealth,
    ProbeConfig,
    ProbeExecutor,
    ProbeResult,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeProbeOutput:
    """Canned output from a probe command execution."""

    stdout: str
    exit_code: int


class FakeProbeExecutor:
    """Configurable fake probe executor for testing."""

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

    async def execute_probe(
        self, command: str, timeout: float
    ) -> tuple[str, int]:
        self.invocations.append(command)
        if self._latency_seconds > 0:
            await asyncio.sleep(self._latency_seconds)
        if self._errors:
            raise self._errors.pop(0)
        if self._outputs:
            output = self._outputs.pop(0)
            return (output.stdout, output.exit_code)
        return ("__jules_probe_ok__", 0)


# Verify protocol compliance
assert isinstance(FakeProbeExecutor(), ProbeExecutor)


# ---------------------------------------------------------------------------
# SessionHealth enum
# ---------------------------------------------------------------------------


class TestSessionHealth:
    """Verify SessionHealth enum values."""

    def test_all_values_exist(self) -> None:
        assert SessionHealth.HEALTHY.value == "healthy"
        assert SessionHealth.DEGRADED.value == "degraded"
        assert SessionHealth.CONNECTION_LOST.value == "connection_lost"
        assert SessionHealth.PROCESS_DEAD.value == "process_dead"
        assert SessionHealth.UNKNOWN.value == "unknown"

    def test_member_count(self) -> None:
        assert len(SessionHealth) == 5


# ---------------------------------------------------------------------------
# SessionInfo input validation
# ---------------------------------------------------------------------------


class TestSessionInfo:
    """Validate SessionInfo construction and constraints."""

    def test_valid_construction(self) -> None:
        info = SessionInfo(session_id="run-123", daemon_pid=1234)
        assert info.session_id == "run-123"
        assert info.daemon_pid == 1234

    def test_frozen(self) -> None:
        info = SessionInfo(session_id="run-1", daemon_pid=100)
        with pytest.raises(AttributeError):
            info.session_id = "run-2"  # type: ignore[misc]

    def test_session_id_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="session_id must not be empty"):
            SessionInfo(session_id="", daemon_pid=1)

    def test_daemon_pid_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="daemon_pid must be a positive integer"):
            SessionInfo(session_id="run-1", daemon_pid=0)

    def test_daemon_pid_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="daemon_pid must be a positive integer"):
            SessionInfo(session_id="run-1", daemon_pid=-5)


# ---------------------------------------------------------------------------
# LivenessResult structure and immutability
# ---------------------------------------------------------------------------


class TestLivenessResult:
    """Verify LivenessResult immutability and structure."""

    def test_frozen(self) -> None:
        result = LivenessResult(
            session_id="run-1",
            health=SessionHealth.HEALTHY,
            alive=True,
            process_result=ProcessCheckResult(
                pid=1,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            ),
            ssh_result=None,
            errors=(),
            latency_ms=1.0,
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            result.health = SessionHealth.DEAD  # type: ignore[misc]

    def test_has_all_required_fields(self) -> None:
        ts = datetime.now(timezone.utc)
        proc = ProcessCheckResult(
            pid=42,
            verdict=ProcessVerdict.ALIVE,
            error=None,
            latency_ms=0.5,
            timestamp=ts,
        )
        result = LivenessResult(
            session_id="run-42",
            health=SessionHealth.HEALTHY,
            alive=True,
            process_result=proc,
            ssh_result=None,
            errors=(),
            latency_ms=2.0,
            timestamp=ts,
        )
        assert result.session_id == "run-42"
        assert result.health == SessionHealth.HEALTHY
        assert result.alive is True
        assert result.process_result == proc
        assert result.ssh_result is None
        assert result.errors == ()
        assert result.latency_ms == 2.0
        assert result.timestamp == ts

    def test_errors_is_tuple(self) -> None:
        """Errors must be a tuple (immutable sequence) not a list."""
        result = LivenessResult(
            session_id="run-1",
            health=SessionHealth.UNKNOWN,
            alive=False,
            process_result=None,
            ssh_result=None,
            errors=("some error",),
            latency_ms=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert isinstance(result.errors, tuple)


# ---------------------------------------------------------------------------
# HEALTHY: both probes pass
# ---------------------------------------------------------------------------


class TestCheckSessionLivenessHealthy:
    """check_session_liveness returns HEALTHY when both probes pass."""

    @pytest.mark.asyncio
    async def test_both_probes_pass(self) -> None:
        session = SessionInfo(session_id="run-1", daemon_pid=1234)
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=1234,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor)

        assert result.health == SessionHealth.HEALTHY
        assert result.alive is True
        assert result.session_id == "run-1"
        assert result.process_result is not None
        assert result.process_result.verdict == ProcessVerdict.ALIVE
        assert result.ssh_result is not None
        assert result.ssh_result.health == ConnectionHealth.CONNECTED
        assert result.errors == ()

    @pytest.mark.asyncio
    async def test_healthy_has_timestamp(self) -> None:
        session = SessionInfo(session_id="run-1", daemon_pid=1234)
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=1234,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            before = datetime.now(timezone.utc)
            result = await check_session_liveness(session, executor)
            after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after

    @pytest.mark.asyncio
    async def test_healthy_has_non_negative_latency(self) -> None:
        session = SessionInfo(session_id="run-1", daemon_pid=1234)
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=1234,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor)

        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# DEGRADED: process alive, SSH degraded
# ---------------------------------------------------------------------------


class TestCheckSessionLivenessDegraded:
    """check_session_liveness returns DEGRADED when SSH is degraded."""

    @pytest.mark.asyncio
    async def test_ssh_degraded_output_mismatch(self) -> None:
        session = SessionInfo(session_id="run-2", daemon_pid=5678)
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="wrong output", exit_code=0)]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=5678,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor)

        assert result.health == SessionHealth.DEGRADED
        assert result.alive is True
        assert result.ssh_result is not None
        assert result.ssh_result.health == ConnectionHealth.DEGRADED

    @pytest.mark.asyncio
    async def test_ssh_degraded_nonzero_exit(self) -> None:
        session = SessionInfo(session_id="run-3", daemon_pid=5678)
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="", exit_code=1)]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=5678,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor)

        assert result.health == SessionHealth.DEGRADED
        assert result.alive is True


# ---------------------------------------------------------------------------
# CONNECTION_LOST: process alive, SSH disconnected
# ---------------------------------------------------------------------------


class TestCheckSessionLivenessConnectionLost:
    """check_session_liveness returns CONNECTION_LOST on SSH failure."""

    @pytest.mark.asyncio
    async def test_ssh_disconnected(self) -> None:
        session = SessionInfo(session_id="run-4", daemon_pid=9999)
        executor = FakeProbeExecutor(
            errors=[OSError("Connection reset by peer")]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=9999,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor)

        assert result.health == SessionHealth.CONNECTION_LOST
        assert result.alive is False
        assert result.ssh_result is not None
        assert result.ssh_result.health == ConnectionHealth.DISCONNECTED

    @pytest.mark.asyncio
    async def test_ssh_timeout(self) -> None:
        session = SessionInfo(session_id="run-5", daemon_pid=9999)

        class SlowExecutor:
            async def execute_probe(
                self, command: str, timeout: float
            ) -> tuple[str, int]:
                await asyncio.sleep(1.0)
                return ("ok", 0)

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=9999,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            config = ProbeConfig(timeout_seconds=0.05)
            result = await check_session_liveness(
                session, SlowExecutor(), probe_config=config
            )

        assert result.health == SessionHealth.CONNECTION_LOST
        assert result.alive is False


# ---------------------------------------------------------------------------
# PROCESS_DEAD: local daemon process is dead
# ---------------------------------------------------------------------------


class TestCheckSessionLivenessProcessDead:
    """check_session_liveness returns PROCESS_DEAD when daemon is gone."""

    @pytest.mark.asyncio
    async def test_process_dead_skips_ssh(self) -> None:
        """When the daemon process is dead, SSH check is skipped."""
        session = SessionInfo(session_id="run-6", daemon_pid=1111)
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=1111,
                verdict=ProcessVerdict.DEAD,
                error="No such process",
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor)

        assert result.health == SessionHealth.PROCESS_DEAD
        assert result.alive is False
        assert result.process_result is not None
        assert result.process_result.verdict == ProcessVerdict.DEAD
        # SSH check should be skipped when process is dead
        assert result.ssh_result is None
        # executor should not have been called
        assert len(executor.invocations) == 0

    @pytest.mark.asyncio
    async def test_process_dead_records_error(self) -> None:
        session = SessionInfo(session_id="run-7", daemon_pid=2222)
        executor = FakeProbeExecutor()

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=2222,
                verdict=ProcessVerdict.DEAD,
                error="No such process",
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor)

        assert result.process_result is not None
        assert result.process_result.error == "No such process"


# ---------------------------------------------------------------------------
# UNKNOWN: process check returns ERROR
# ---------------------------------------------------------------------------


class TestCheckSessionLivenessUnknown:
    """check_session_liveness returns UNKNOWN on ambiguous state."""

    @pytest.mark.asyncio
    async def test_process_error_verdict(self) -> None:
        """ProcessVerdict.ERROR with SSH connected -> UNKNOWN."""
        session = SessionInfo(session_id="run-8", daemon_pid=3333)
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=3333,
                verdict=ProcessVerdict.ERROR,
                error="OSError(errno=22): Invalid argument",
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor)

        assert result.health == SessionHealth.UNKNOWN
        assert result.alive is False
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_process_error_and_ssh_disconnected(self) -> None:
        """Both probes problematic -> UNKNOWN."""
        session = SessionInfo(session_id="run-9", daemon_pid=4444)
        executor = FakeProbeExecutor(
            errors=[OSError("Connection refused")]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=4444,
                verdict=ProcessVerdict.ERROR,
                error="unexpected error",
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor)

        assert result.health == SessionHealth.UNKNOWN
        assert result.alive is False
        assert len(result.errors) >= 2


# ---------------------------------------------------------------------------
# No executor provided (SSH check skipped)
# ---------------------------------------------------------------------------


class TestCheckSessionLivenessNoExecutor:
    """When no executor is provided, only process check runs."""

    @pytest.mark.asyncio
    async def test_no_executor_process_alive(self) -> None:
        session = SessionInfo(session_id="run-10", daemon_pid=5555)

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=5555,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor=None)

        assert result.health == SessionHealth.DEGRADED
        assert result.alive is True
        assert result.ssh_result is None
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_no_executor_process_dead(self) -> None:
        session = SessionInfo(session_id="run-11", daemon_pid=6666)

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=6666,
                verdict=ProcessVerdict.DEAD,
                error="No such process",
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(session, executor=None)

        assert result.health == SessionHealth.PROCESS_DEAD
        assert result.alive is False


# ---------------------------------------------------------------------------
# Custom ProbeConfig
# ---------------------------------------------------------------------------


class TestCheckSessionLivenessCustomConfig:
    """check_session_liveness accepts an optional ProbeConfig."""

    @pytest.mark.asyncio
    async def test_custom_config_forwarded(self) -> None:
        session = SessionInfo(session_id="run-12", daemon_pid=7777)
        config = ProbeConfig(
            command="uptime",
            timeout_seconds=2.0,
            expected_output="",
        )
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="up 5 days", exit_code=0)]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=7777,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            result = await check_session_liveness(
                session, executor, probe_config=config
            )

        assert result.health == SessionHealth.HEALTHY
        assert result.ssh_result is not None
        assert result.ssh_result.probe_command == "uptime"


# ---------------------------------------------------------------------------
# Correct PID forwarding
# ---------------------------------------------------------------------------


class TestCheckSessionLivenessPidForwarding:
    """check_session_liveness passes the correct daemon PID to check_pid."""

    @pytest.mark.asyncio
    async def test_forwards_daemon_pid(self) -> None:
        session = SessionInfo(session_id="run-13", daemon_pid=42)
        executor = FakeProbeExecutor(
            outputs=[FakeProbeOutput(stdout="__jules_probe_ok__", exit_code=0)]
        )

        with patch(
            "jules_daemon.monitor.session_liveness.check_pid"
        ) as mock_check:
            mock_check.return_value = ProcessCheckResult(
                pid=42,
                verdict=ProcessVerdict.ALIVE,
                error=None,
                latency_ms=0.1,
                timestamp=datetime.now(timezone.utc),
            )
            await check_session_liveness(session, executor)

        mock_check.assert_called_once_with(42)


# ---------------------------------------------------------------------------
# Result consistency
# ---------------------------------------------------------------------------


class TestLivenessResultConsistency:
    """Verify alive property aligns with health values."""

    def test_healthy_is_alive(self) -> None:
        result = LivenessResult(
            session_id="x",
            health=SessionHealth.HEALTHY,
            alive=True,
            process_result=None,
            ssh_result=None,
            errors=(),
            latency_ms=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.alive is True

    def test_degraded_is_alive(self) -> None:
        result = LivenessResult(
            session_id="x",
            health=SessionHealth.DEGRADED,
            alive=True,
            process_result=None,
            ssh_result=None,
            errors=(),
            latency_ms=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.alive is True

    def test_connection_lost_is_not_alive(self) -> None:
        result = LivenessResult(
            session_id="x",
            health=SessionHealth.CONNECTION_LOST,
            alive=False,
            process_result=None,
            ssh_result=None,
            errors=(),
            latency_ms=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.alive is False

    def test_process_dead_is_not_alive(self) -> None:
        result = LivenessResult(
            session_id="x",
            health=SessionHealth.PROCESS_DEAD,
            alive=False,
            process_result=None,
            ssh_result=None,
            errors=(),
            latency_ms=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.alive is False

    def test_unknown_is_not_alive(self) -> None:
        result = LivenessResult(
            session_id="x",
            health=SessionHealth.UNKNOWN,
            alive=False,
            process_result=None,
            ssh_result=None,
            errors=(),
            latency_ms=0.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert result.alive is False


# ---------------------------------------------------------------------------
# _derive_health internal helper (direct unit tests)
# ---------------------------------------------------------------------------


class TestDeriveHealth:
    """Direct unit tests for the _derive_health classification function.

    These cover the defensive branches that the public API fast-paths
    around (e.g., DEAD process is short-circuited before _derive_health
    is called in check_session_liveness).
    """

    def test_dead_process_any_ssh(self) -> None:
        health, alive = _derive_health(ProcessVerdict.DEAD, None)
        assert health == SessionHealth.PROCESS_DEAD
        assert alive is False

    def test_dead_process_connected_ssh(self) -> None:
        health, alive = _derive_health(
            ProcessVerdict.DEAD, ConnectionHealth.CONNECTED
        )
        assert health == SessionHealth.PROCESS_DEAD
        assert alive is False

    def test_error_process_no_ssh(self) -> None:
        health, alive = _derive_health(ProcessVerdict.ERROR, None)
        assert health == SessionHealth.UNKNOWN
        assert alive is False

    def test_alive_process_connected_ssh(self) -> None:
        health, alive = _derive_health(
            ProcessVerdict.ALIVE, ConnectionHealth.CONNECTED
        )
        assert health == SessionHealth.HEALTHY
        assert alive is True

    def test_alive_process_degraded_ssh(self) -> None:
        health, alive = _derive_health(
            ProcessVerdict.ALIVE, ConnectionHealth.DEGRADED
        )
        assert health == SessionHealth.DEGRADED
        assert alive is True

    def test_alive_process_disconnected_ssh(self) -> None:
        health, alive = _derive_health(
            ProcessVerdict.ALIVE, ConnectionHealth.DISCONNECTED
        )
        assert health == SessionHealth.CONNECTION_LOST
        assert alive is False

    def test_alive_process_no_ssh(self) -> None:
        health, alive = _derive_health(ProcessVerdict.ALIVE, None)
        assert health == SessionHealth.DEGRADED
        assert alive is True
