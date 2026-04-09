"""Tests for SSH re-establishment from recovered run records.

Covers:
    - Extracting SSHTarget from CrashRecoveryResult
    - Validation of incomplete connection parameters
    - Successful re-establishment on first attempt
    - Successful re-establishment after transient failures
    - Permanent error causes immediate failure
    - Timeout configuration is respected
    - ReestablishmentResult structure and properties
    - on_progress callback invocation
    - Recovery from RUNNING state (reconnect action)
    - Recovery from PENDING_APPROVAL state is rejected
    - FRESH_START action is rejected
    - Key path is forwarded to SSHTarget when present
    - Missing host or user produces clear error
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from jules_daemon.ssh.backoff import BackoffConfig
from jules_daemon.ssh.errors import (
    SSHAuthenticationError,
    SSHConnectionError,
    SSHHostKeyError,
)
from jules_daemon.ssh.reestablish import (
    ReestablishmentResult,
    extract_ssh_target,
    reestablish_ssh,
)
from jules_daemon.ssh.reconnect import SSHConnectionHandle, SSHConnector
from jules_daemon.wiki.crash_recovery import CrashRecoveryResult, RecoveryAction
from jules_daemon.wiki.models import RunStatus, SSHTarget


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeHandle:
    """Fake SSH connection handle for testing."""

    session_id: str = "reestablish-session-001"


assert isinstance(FakeHandle(), SSHConnectionHandle)


class FakeConnector:
    """Configurable fake SSH connector for testing."""

    def __init__(
        self,
        *,
        errors: list[Exception] | None = None,
        handle: FakeHandle | None = None,
    ) -> None:
        self._errors: list[Exception] = list(errors) if errors else []
        self._handle = handle or FakeHandle()
        self.connect_attempts: list[SSHTarget] = []
        self.closed_handles: list[FakeHandle] = []

    async def connect(self, target: SSHTarget) -> FakeHandle:
        self.connect_attempts.append(target)
        if self._errors:
            raise self._errors.pop(0)
        return self._handle

    async def close(self, handle: FakeHandle) -> None:
        self.closed_handles.append(handle)

    async def is_alive(self, handle: FakeHandle) -> bool:
        return True


assert isinstance(FakeConnector(), SSHConnector)


# ---------------------------------------------------------------------------
# Fixtures: CrashRecoveryResult builders
# ---------------------------------------------------------------------------


def _make_recovery_result(
    *,
    action: RecoveryAction = RecoveryAction.RECONNECT,
    host: str | None = "staging.example.com",
    user: str | None = "deploy",
    port: int | None = 22,
    key_path: str | None = None,
    remote_pid: int | None = 67890,
    run_id: str = "test-run-001",
    status: RunStatus = RunStatus.RUNNING,
    resolved_shell: str | None = "pytest -v",
    natural_language_command: str | None = "run tests",
    progress_percent: float = 45.0,
) -> CrashRecoveryResult:
    """Build a CrashRecoveryResult for testing."""
    return CrashRecoveryResult(
        action=action,
        reason="Test recovery",
        run_id=run_id,
        status=status,
        host=host,
        user=user,
        port=port,
        key_path=key_path,
        remote_pid=remote_pid,
        daemon_pid=12345,
        resolved_shell=resolved_shell,
        natural_language_command=natural_language_command,
        progress_percent=progress_percent,
        error=None,
        source_path=Path("/tmp/wiki/pages/daemon/current-run.md"),
    )


_FAST_CONFIG = BackoffConfig(
    base_delay=0.001,
    max_delay=0.01,
    multiplier=2.0,
    jitter_factor=0.0,
    max_retries=3,
)


# ---------------------------------------------------------------------------
# extract_ssh_target
# ---------------------------------------------------------------------------


class TestExtractSSHTarget:
    """Extracting SSHTarget from CrashRecoveryResult fields."""

    def test_extracts_basic_target(self) -> None:
        recovery = _make_recovery_result()
        target = extract_ssh_target(recovery)

        assert target.host == "staging.example.com"
        assert target.user == "deploy"
        assert target.port == 22
        assert target.key_path is None

    def test_extracts_target_with_key_path(self) -> None:
        recovery = _make_recovery_result(key_path="/home/deploy/.ssh/id_ed25519")
        target = extract_ssh_target(recovery)

        assert target.key_path == "/home/deploy/.ssh/id_ed25519"

    def test_extracts_custom_port(self) -> None:
        recovery = _make_recovery_result(port=2222)
        target = extract_ssh_target(recovery)

        assert target.port == 2222

    def test_default_port_when_none(self) -> None:
        recovery = _make_recovery_result(port=None)
        target = extract_ssh_target(recovery)

        assert target.port == 22

    def test_raises_when_host_missing(self) -> None:
        recovery = _make_recovery_result(host=None)
        with pytest.raises(ValueError, match="host"):
            extract_ssh_target(recovery)

    def test_raises_when_user_missing(self) -> None:
        recovery = _make_recovery_result(user=None)
        with pytest.raises(ValueError, match="user"):
            extract_ssh_target(recovery)

    def test_raises_when_both_missing(self) -> None:
        recovery = _make_recovery_result(host=None, user=None)
        with pytest.raises(ValueError, match="host"):
            extract_ssh_target(recovery)


# ---------------------------------------------------------------------------
# reestablish_ssh: validation
# ---------------------------------------------------------------------------


class TestReestablishValidation:
    """Validation of recovery result before attempting connection."""

    @pytest.mark.asyncio
    async def test_rejects_fresh_start_action(self) -> None:
        recovery = _make_recovery_result(action=RecoveryAction.FRESH_START)
        connector = FakeConnector()

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.success is False
        assert "RECONNECT" in (result.error or "")
        assert result.handle is None
        assert len(connector.connect_attempts) == 0

    @pytest.mark.asyncio
    async def test_rejects_resume_approval_action(self) -> None:
        recovery = _make_recovery_result(
            action=RecoveryAction.RESUME_APPROVAL,
            status=RunStatus.PENDING_APPROVAL,
        )
        connector = FakeConnector()

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.success is False
        assert "RECONNECT" in (result.error or "")
        assert len(connector.connect_attempts) == 0

    @pytest.mark.asyncio
    async def test_rejects_missing_connection_params(self) -> None:
        recovery = _make_recovery_result(host=None)
        connector = FakeConnector()

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.success is False
        assert "host" in (result.error or "").lower()
        assert len(connector.connect_attempts) == 0


# ---------------------------------------------------------------------------
# reestablish_ssh: successful connection
# ---------------------------------------------------------------------------


class TestReestablishSuccess:
    """SSH re-establishment succeeds from recovered run record."""

    @pytest.mark.asyncio
    async def test_first_attempt_success(self) -> None:
        recovery = _make_recovery_result()
        connector = FakeConnector()

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.success is True
        assert result.handle is not None
        assert result.handle.session_id == "reestablish-session-001"
        assert result.error is None
        assert result.attempts == 1
        assert result.total_duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_target_forwarded_to_connector(self) -> None:
        recovery = _make_recovery_result()
        connector = FakeConnector()

        await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert len(connector.connect_attempts) == 1
        target = connector.connect_attempts[0]
        assert target.host == "staging.example.com"
        assert target.user == "deploy"
        assert target.port == 22

    @pytest.mark.asyncio
    async def test_result_contains_recovery_metadata(self) -> None:
        recovery = _make_recovery_result(
            run_id="run-abc-123",
            remote_pid=99999,
            resolved_shell="pytest -x",
        )
        connector = FakeConnector()

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.run_id == "run-abc-123"
        assert result.remote_pid == 99999
        assert result.resolved_shell == "pytest -x"

    @pytest.mark.asyncio
    async def test_result_contains_target(self) -> None:
        recovery = _make_recovery_result()
        connector = FakeConnector()

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.target is not None
        assert result.target.host == "staging.example.com"

    @pytest.mark.asyncio
    async def test_key_path_forwarded(self) -> None:
        recovery = _make_recovery_result(key_path="/home/deploy/.ssh/id_rsa")
        connector = FakeConnector()

        await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        target = connector.connect_attempts[0]
        assert target.key_path == "/home/deploy/.ssh/id_rsa"


# ---------------------------------------------------------------------------
# reestablish_ssh: retry behavior
# ---------------------------------------------------------------------------


class TestReestablishRetry:
    """Retry and backoff behavior during re-establishment."""

    @pytest.mark.asyncio
    async def test_succeeds_after_transient_failure(self) -> None:
        recovery = _make_recovery_result()
        connector = FakeConnector(
            errors=[SSHConnectionError("connection refused")]
        )

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.success is True
        assert result.attempts == 2
        assert len(result.retry_history) == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_multiple_transient_failures(self) -> None:
        recovery = _make_recovery_result()
        connector = FakeConnector(
            errors=[
                SSHConnectionError("refused"),
                TimeoutError("timed out"),
                OSError("network down"),
            ]
        )

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.success is True
        assert result.attempts == 4
        assert len(result.retry_history) == 3

    @pytest.mark.asyncio
    async def test_permanent_error_fails_immediately(self) -> None:
        recovery = _make_recovery_result()
        connector = FakeConnector(
            errors=[SSHAuthenticationError("invalid key")]
        )

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.success is False
        assert result.attempts == 1
        assert "Permanent error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_host_key_error_fails_immediately(self) -> None:
        recovery = _make_recovery_result()
        connector = FakeConnector(
            errors=[SSHHostKeyError("key changed")]
        )

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.success is False
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self) -> None:
        config = BackoffConfig(
            base_delay=0.001,
            max_delay=0.01,
            multiplier=2.0,
            jitter_factor=0.0,
            max_retries=2,
        )
        recovery = _make_recovery_result()
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
                SSHConnectionError("fail 3"),
            ]
        )

        result = await reestablish_ssh(recovery, connector, config=config)

        assert result.success is False
        assert result.attempts == 3
        assert "exhausted" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# reestablish_ssh: on_progress callback
# ---------------------------------------------------------------------------


class TestReestablishCallback:
    """Verify on_progress callback is invoked during re-establishment."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_transient_failure_then_success(self) -> None:
        messages: list[str] = []

        async def on_progress(msg: str) -> None:
            messages.append(msg)

        recovery = _make_recovery_result()
        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )

        result = await reestablish_ssh(
            recovery, connector, config=_FAST_CONFIG, on_progress=on_progress
        )

        assert result.success is True
        assert len(messages) >= 1  # At least one progress message

    @pytest.mark.asyncio
    async def test_callback_invoked_on_clean_success(self) -> None:
        """Verify progress messages on first-attempt success."""
        messages: list[str] = []

        async def on_progress(msg: str) -> None:
            messages.append(msg)

        recovery = _make_recovery_result()
        connector = FakeConnector()

        result = await reestablish_ssh(
            recovery, connector, config=_FAST_CONFIG, on_progress=on_progress
        )

        assert result.success is True
        assert len(messages) == 2  # "Re-establishing..." + "re-established..."
        assert "staging.example.com" in messages[0]
        assert "re-established" in messages[1].lower()

    @pytest.mark.asyncio
    async def test_callback_error_is_swallowed(self) -> None:
        """on_progress errors must not break re-establishment."""

        async def broken_callback(msg: str) -> None:
            raise RuntimeError("callback broke")

        recovery = _make_recovery_result()
        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )

        result = await reestablish_ssh(
            recovery, connector, config=_FAST_CONFIG, on_progress=broken_callback
        )

        assert result.success is True


# ---------------------------------------------------------------------------
# ReestablishmentResult structure
# ---------------------------------------------------------------------------


class TestReestablishmentResult:
    """Verify ReestablishmentResult properties."""

    @pytest.mark.asyncio
    async def test_result_is_frozen(self) -> None:
        recovery = _make_recovery_result()
        connector = FakeConnector()

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_retry_history_is_tuple(self) -> None:
        recovery = _make_recovery_result()
        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert isinstance(result.retry_history, tuple)

    @pytest.mark.asyncio
    async def test_failed_result_has_no_handle(self) -> None:
        recovery = _make_recovery_result(host=None)
        connector = FakeConnector()

        result = await reestablish_ssh(recovery, connector, config=_FAST_CONFIG)

        assert result.handle is None
        assert result.success is False

    @pytest.mark.asyncio
    async def test_default_config_used_when_none(self) -> None:
        """When config is None, default backoff config is used."""
        recovery = _make_recovery_result()
        connector = FakeConnector()

        # Should not raise -- uses default config internally
        result = await reestablish_ssh(recovery, connector)

        assert result.success is True
