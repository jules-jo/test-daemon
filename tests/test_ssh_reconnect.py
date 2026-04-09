"""Tests for SSH reconnection with exponential backoff.

Covers:
    - Successful first-attempt connection
    - Successful connection after transient failures
    - Permanent error causes immediate failure (no further retries)
    - All retries exhausted with transient errors
    - RetryRecord correctness (attempt, error type, delay, transient flag)
    - ReconnectionResult structure and properties
    - on_retry callback invocation
    - Backoff delays are respected (within tolerance)
    - Zero-retry config (max_retries=0)
    - Mixed transient then permanent errors
    - raise_on_failure convenience function
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from jules_daemon.ssh.backoff import BackoffConfig
from jules_daemon.ssh.errors import (
    SSHAuthenticationError,
    SSHConnectionError,
    SSHHostKeyError,
    SSHReconnectionExhaustedError,
)
from jules_daemon.ssh.reconnect import (
    ReconnectionResult,
    RetryRecord,
    SSHConnectionHandle,
    SSHConnector,
    raise_on_failure,
    reconnect_ssh,
)
from jules_daemon.wiki.models import SSHTarget


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeHandle:
    """Fake SSH connection handle for testing."""

    session_id: str = "fake-session-001"


# Verify protocol compliance
assert isinstance(FakeHandle(), SSHConnectionHandle)


class FakeConnector:
    """Configurable fake SSH connector for testing.

    Tracks connection attempts and raises errors from a queue.
    When the error queue is empty, returns a successful handle.
    """

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
        self.alive_checks: list[FakeHandle] = []

    async def connect(self, target: SSHTarget) -> FakeHandle:
        self.connect_attempts.append(target)
        if self._errors:
            raise self._errors.pop(0)
        return self._handle

    async def close(self, handle: FakeHandle) -> None:  # type: ignore[override]
        self.closed_handles.append(handle)

    async def is_alive(self, handle: FakeHandle) -> bool:  # type: ignore[override]
        self.alive_checks.append(handle)
        return True


# Verify protocol compliance
assert isinstance(FakeConnector(), SSHConnector)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


_TARGET = SSHTarget(host="test.example.com", user="deploy", port=22)
_FAST_CONFIG = BackoffConfig(
    base_delay=0.001,  # Near-zero delays for fast tests
    max_delay=0.01,
    multiplier=2.0,
    jitter_factor=0.0,
    max_retries=3,
)


# ---------------------------------------------------------------------------
# Successful connection
# ---------------------------------------------------------------------------


class TestReconnectSuccess:
    """Connection succeeds on first attempt."""

    @pytest.mark.asyncio
    async def test_first_attempt_success(self) -> None:
        connector = FakeConnector()
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert result.success is True
        assert result.attempts == 1
        assert result.handle is not None
        assert result.handle.session_id == "fake-session-001"
        assert result.error is None
        assert result.retry_history == ()
        assert result.target is _TARGET
        assert result.total_duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_connector_receives_target(self) -> None:
        connector = FakeConnector()
        await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert len(connector.connect_attempts) == 1
        assert connector.connect_attempts[0] is _TARGET


# ---------------------------------------------------------------------------
# Success after transient failures
# ---------------------------------------------------------------------------


class TestReconnectAfterTransientFailures:
    """Connection fails with transient errors, then succeeds."""

    @pytest.mark.asyncio
    async def test_succeeds_after_one_failure(self) -> None:
        connector = FakeConnector(
            errors=[SSHConnectionError("refused")]
        )
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert result.success is True
        assert result.attempts == 2
        assert len(result.retry_history) == 1
        assert result.retry_history[0].is_transient is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_succeeds_after_multiple_failures(self) -> None:
        connector = FakeConnector(
            errors=[
                SSHConnectionError("refused"),
                TimeoutError("timed out"),
                OSError("network unreachable"),
            ]
        )
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert result.success is True
        assert result.attempts == 4  # 3 failures + 1 success
        assert len(result.retry_history) == 3

    @pytest.mark.asyncio
    async def test_retry_records_capture_error_types(self) -> None:
        connector = FakeConnector(
            errors=[
                SSHConnectionError("refused"),
                TimeoutError("timed out"),
            ]
        )
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert result.retry_history[0].error_type == "SSHConnectionError"
        assert result.retry_history[0].error_message == "refused"
        assert result.retry_history[1].error_type == "TimeoutError"
        assert result.retry_history[1].error_message == "timed out"

    @pytest.mark.asyncio
    async def test_retry_records_have_timestamps(self) -> None:
        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )
        before = datetime.now(timezone.utc)
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)
        after = datetime.now(timezone.utc)

        record = result.retry_history[0]
        assert before <= record.timestamp <= after


# ---------------------------------------------------------------------------
# Permanent error: immediate failure
# ---------------------------------------------------------------------------


class TestReconnectPermanentError:
    """Permanent errors cause immediate failure without retries."""

    @pytest.mark.asyncio
    async def test_auth_error_fails_immediately(self) -> None:
        connector = FakeConnector(
            errors=[SSHAuthenticationError("invalid key")]
        )
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert result.success is False
        assert result.attempts == 1
        assert result.handle is None
        assert "Permanent error" in (result.error or "")
        assert len(result.retry_history) == 1
        assert result.retry_history[0].is_transient is False

    @pytest.mark.asyncio
    async def test_host_key_error_fails_immediately(self) -> None:
        connector = FakeConnector(
            errors=[SSHHostKeyError("key mismatch")]
        )
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert result.success is False
        assert result.attempts == 1
        assert "Permanent error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_no_extra_attempts_after_permanent(self) -> None:
        connector = FakeConnector(
            errors=[SSHAuthenticationError("bad key")]
        )
        await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        # Only 1 connection attempt was made
        assert len(connector.connect_attempts) == 1


# ---------------------------------------------------------------------------
# Exhausted retries
# ---------------------------------------------------------------------------


class TestReconnectExhausted:
    """All retry attempts exhausted with transient errors."""

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self) -> None:
        config = BackoffConfig(
            base_delay=0.001,
            max_delay=0.01,
            multiplier=2.0,
            jitter_factor=0.0,
            max_retries=2,
        )
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
                SSHConnectionError("fail 3"),
            ]
        )
        result = await reconnect_ssh(_TARGET, connector, config)

        assert result.success is False
        assert result.attempts == 3  # 1 initial + 2 retries
        assert result.handle is None
        assert "exhausted" in (result.error or "").lower()
        assert len(result.retry_history) == 3

    @pytest.mark.asyncio
    async def test_exhaustion_records_all_attempts(self) -> None:
        config = BackoffConfig(
            base_delay=0.001,
            max_delay=0.01,
            multiplier=2.0,
            jitter_factor=0.0,
            max_retries=3,
        )
        connector = FakeConnector(
            errors=[
                SSHConnectionError("e1"),
                OSError("e2"),
                TimeoutError("e3"),
                ConnectionResetError("e4"),
            ]
        )
        result = await reconnect_ssh(_TARGET, connector, config)

        assert result.success is False
        assert result.attempts == 4
        assert len(result.retry_history) == 4
        for record in result.retry_history:
            assert record.is_transient is True


# ---------------------------------------------------------------------------
# Mixed transient then permanent
# ---------------------------------------------------------------------------


class TestReconnectMixedErrors:
    """Transient errors followed by a permanent error."""

    @pytest.mark.asyncio
    async def test_transient_then_permanent(self) -> None:
        connector = FakeConnector(
            errors=[
                SSHConnectionError("retry me"),
                SSHAuthenticationError("bad creds"),
            ]
        )
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert result.success is False
        assert result.attempts == 2
        assert len(result.retry_history) == 2
        assert result.retry_history[0].is_transient is True
        assert result.retry_history[1].is_transient is False
        assert "Permanent error" in (result.error or "")


# ---------------------------------------------------------------------------
# Zero retries config
# ---------------------------------------------------------------------------


class TestReconnectZeroRetries:
    """Config with max_retries=0 means only one attempt."""

    @pytest.mark.asyncio
    async def test_success_on_single_attempt(self) -> None:
        config = BackoffConfig(base_delay=0.001, max_delay=0.01, max_retries=0)
        connector = FakeConnector()
        result = await reconnect_ssh(_TARGET, connector, config)

        assert result.success is True
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_failure_on_single_attempt(self) -> None:
        config = BackoffConfig(base_delay=0.001, max_delay=0.01, max_retries=0)
        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )
        result = await reconnect_ssh(_TARGET, connector, config)

        assert result.success is False
        assert result.attempts == 1
        assert len(result.retry_history) == 1


# ---------------------------------------------------------------------------
# on_retry callback
# ---------------------------------------------------------------------------


class TestReconnectOnRetryCallback:
    """Verify on_retry callback is invoked for each failed attempt."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_failure(self) -> None:
        records: list[RetryRecord] = []

        async def on_retry(record: RetryRecord) -> None:
            records.append(record)

        connector = FakeConnector(
            errors=[SSHConnectionError("fail 1"), SSHConnectionError("fail 2")]
        )
        result = await reconnect_ssh(
            _TARGET, connector, _FAST_CONFIG, on_retry=on_retry
        )

        assert result.success is True  # succeeds on 3rd attempt
        assert len(records) == 2
        assert records[0].attempt == 0
        assert records[1].attempt == 1

    @pytest.mark.asyncio
    async def test_callback_not_invoked_on_success(self) -> None:
        records: list[RetryRecord] = []

        async def on_retry(record: RetryRecord) -> None:
            records.append(record)

        connector = FakeConnector()
        await reconnect_ssh(
            _TARGET, connector, _FAST_CONFIG, on_retry=on_retry
        )

        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_callback_error_is_swallowed(self) -> None:
        """on_retry callback errors must not break reconnection."""

        async def broken_callback(record: RetryRecord) -> None:
            raise RuntimeError("callback broke")

        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )
        result = await reconnect_ssh(
            _TARGET, connector, _FAST_CONFIG, on_retry=broken_callback
        )

        assert result.success is True
        assert result.attempts == 2


# ---------------------------------------------------------------------------
# ReconnectionResult structure
# ---------------------------------------------------------------------------


class TestReconnectionResult:
    """Verify ReconnectionResult properties."""

    @pytest.mark.asyncio
    async def test_result_has_target(self) -> None:
        connector = FakeConnector()
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert result.target is _TARGET

    @pytest.mark.asyncio
    async def test_duration_is_non_negative(self) -> None:
        connector = FakeConnector()
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert result.total_duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_retry_history_is_tuple(self) -> None:
        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        assert isinstance(result.retry_history, tuple)


# ---------------------------------------------------------------------------
# RetryRecord structure
# ---------------------------------------------------------------------------


class TestRetryRecord:
    """Verify RetryRecord is frozen and has correct fields."""

    def test_frozen(self) -> None:
        record = RetryRecord(
            attempt=0,
            error_type="SSHConnectionError",
            error_message="fail",
            delay_seconds=1.0,
            is_transient=True,
        )
        with pytest.raises(AttributeError):
            record.attempt = 1  # type: ignore[misc]

    def test_default_timestamp(self) -> None:
        record = RetryRecord(
            attempt=0,
            error_type="SSHConnectionError",
            error_message="fail",
            delay_seconds=1.0,
            is_transient=True,
        )
        assert isinstance(record.timestamp, datetime)


# ---------------------------------------------------------------------------
# raise_on_failure convenience function
# ---------------------------------------------------------------------------


class TestRaiseOnFailure:
    """Verify raise_on_failure behavior."""

    @pytest.mark.asyncio
    async def test_does_not_raise_on_success(self) -> None:
        connector = FakeConnector()
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        # Should not raise
        raise_on_failure(result)

    @pytest.mark.asyncio
    async def test_raises_on_failure(self) -> None:
        connector = FakeConnector(
            errors=[SSHAuthenticationError("bad key")]
        )
        result = await reconnect_ssh(_TARGET, connector, _FAST_CONFIG)

        with pytest.raises(SSHReconnectionExhaustedError) as exc_info:
            raise_on_failure(result)

        assert exc_info.value.attempts == 1
        assert exc_info.value.last_error == "bad key"

    @pytest.mark.asyncio
    async def test_raises_with_exhausted_retries(self) -> None:
        config = BackoffConfig(
            base_delay=0.001,
            max_delay=0.01,
            max_retries=1,
            jitter_factor=0.0,
        )
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
            ]
        )
        result = await reconnect_ssh(_TARGET, connector, config)

        with pytest.raises(SSHReconnectionExhaustedError) as exc_info:
            raise_on_failure(result)

        assert exc_info.value.attempts == 2
        assert exc_info.value.last_error == "fail 2"


# ---------------------------------------------------------------------------
# Backoff timing (approximate)
# ---------------------------------------------------------------------------


class TestReconnectBackoffTiming:
    """Verify that backoff delays are approximately respected.

    Uses a config with small but measurable delays to confirm that
    actual elapsed time is at least the expected backoff.
    """

    @pytest.mark.asyncio
    async def test_delays_accumulate(self) -> None:
        """Total duration should exceed the sum of backoff delays."""
        config = BackoffConfig(
            base_delay=0.02,  # 20ms
            max_delay=1.0,
            multiplier=2.0,
            jitter_factor=0.0,
            max_retries=2,
        )
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
            ]
        )
        result = await reconnect_ssh(_TARGET, connector, config)

        assert result.success is True
        assert result.attempts == 3
        # Expected delays: attempt 0 = 0.02s, attempt 1 = 0.04s
        # Total minimum = 0.06s
        assert result.total_duration_seconds >= 0.05  # Allow small tolerance
