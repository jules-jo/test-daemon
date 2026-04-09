"""Tests for SSH command executor with 3-retry exponential backoff.

AC 5: 3 SSH retries with exponential backoff on connection failure.

Covers:
    - DEFAULT_SSH_BACKOFF enforces exactly max_retries=3
    - Executor makes exactly 4 attempts (1 initial + 3 retries) before giving up
    - Exponential delays grow: ~1s, ~2s, ~4s (base_delay=1, multiplier=2)
    - Transient errors trigger retries; permanent errors fail immediately
    - Successful connection after 1, 2, or 3 transient failures
    - ExecutionAttempt record captures each attempt's metadata
    - execute_ssh_command returns structured ExecutionOutcome
    - on_attempt callback fires for every attempt (including success)
    - create_ssh_backoff factory produces correct config
    - Config is immutable (frozen dataclass)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from jules_daemon.ssh.backoff import BackoffConfig
from jules_daemon.ssh.errors import (
    SSHAuthenticationError,
    SSHConnectionError,
    SSHHostKeyError,
    SSHReconnectionExhaustedError,
)
from jules_daemon.ssh.executor import (
    DEFAULT_SSH_BACKOFF,
    ExecutionAttempt,
    ExecutionOutcome,
    create_ssh_backoff,
    execute_ssh_command,
)
from jules_daemon.ssh.reconnect import (
    SSHConnectionHandle,
    SSHConnector,
)
from jules_daemon.wiki.models import SSHTarget


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeHandle:
    """Fake SSH connection handle for testing."""

    session_id: str = "exec-test-session"


assert isinstance(FakeHandle(), SSHConnectionHandle)


class FakeConnector:
    """Configurable fake connector that raises errors from a queue."""

    def __init__(
        self,
        *,
        errors: list[Exception] | None = None,
        handle: FakeHandle | None = None,
    ) -> None:
        self._errors: list[Exception] = list(errors) if errors else []
        self._handle = handle or FakeHandle()
        self.connect_attempts: list[SSHTarget] = []

    async def connect(self, target: SSHTarget) -> FakeHandle:
        self.connect_attempts.append(target)
        if self._errors:
            raise self._errors.pop(0)
        return self._handle

    async def close(self, handle: FakeHandle) -> None:
        pass

    async def is_alive(self, handle: FakeHandle) -> bool:
        return True


assert isinstance(FakeConnector(), SSHConnector)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TARGET = SSHTarget(host="test.example.com", user="deploy", port=22)

# Fast config for tests: same structure as DEFAULT_SSH_BACKOFF but with
# near-zero delays so tests do not block.
_FAST_3_RETRY = create_ssh_backoff(
    base_delay=0.001,
    max_delay=0.01,
)


# ---------------------------------------------------------------------------
# DEFAULT_SSH_BACKOFF constants
# ---------------------------------------------------------------------------


class TestDefaultSSHBackoff:
    """Verify the project-wide SSH backoff constant enforces 3 retries."""

    def test_max_retries_is_three(self) -> None:
        assert DEFAULT_SSH_BACKOFF.max_retries == 3

    def test_base_delay_is_one_second(self) -> None:
        assert DEFAULT_SSH_BACKOFF.base_delay == 1.0

    def test_multiplier_is_two(self) -> None:
        assert DEFAULT_SSH_BACKOFF.multiplier == 2.0

    def test_max_delay_is_thirty_seconds(self) -> None:
        assert DEFAULT_SSH_BACKOFF.max_delay == 30.0

    def test_jitter_factor_is_reasonable(self) -> None:
        assert 0.0 <= DEFAULT_SSH_BACKOFF.jitter_factor <= 0.5

    def test_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            DEFAULT_SSH_BACKOFF.max_retries = 5  # type: ignore[misc]

    def test_is_backoff_config_instance(self) -> None:
        assert isinstance(DEFAULT_SSH_BACKOFF, BackoffConfig)


# ---------------------------------------------------------------------------
# create_ssh_backoff factory
# ---------------------------------------------------------------------------


class TestCreateSSHBackoff:
    """Verify the factory always produces max_retries=3."""

    def test_default_produces_three_retries(self) -> None:
        config = create_ssh_backoff()
        assert config.max_retries == 3

    def test_custom_base_delay(self) -> None:
        config = create_ssh_backoff(base_delay=0.5)
        assert config.base_delay == 0.5
        assert config.max_retries == 3

    def test_custom_max_delay(self) -> None:
        config = create_ssh_backoff(max_delay=10.0)
        assert config.max_delay == 10.0
        assert config.max_retries == 3

    def test_custom_multiplier(self) -> None:
        config = create_ssh_backoff(multiplier=3.0)
        assert config.multiplier == 3.0
        assert config.max_retries == 3

    def test_custom_jitter_factor(self) -> None:
        config = create_ssh_backoff(jitter_factor=0.3)
        assert config.jitter_factor == 0.3
        assert config.max_retries == 3

    def test_returns_backoff_config(self) -> None:
        config = create_ssh_backoff()
        assert isinstance(config, BackoffConfig)


# ---------------------------------------------------------------------------
# 3-retry enforcement via executor
# ---------------------------------------------------------------------------


class TestExactlyThreeRetries:
    """The executor must make exactly 4 total attempts (1 + 3 retries)."""

    @pytest.mark.asyncio
    async def test_four_total_attempts_on_exhaustion(self) -> None:
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
                SSHConnectionError("fail 3"),
                SSHConnectionError("fail 4"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.success is False
        assert outcome.total_attempts == 4
        assert len(connector.connect_attempts) == 4

    @pytest.mark.asyncio
    async def test_no_fifth_attempt(self) -> None:
        """With 5 errors queued, only 4 should be consumed."""
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
                SSHConnectionError("fail 3"),
                SSHConnectionError("fail 4"),
                SSHConnectionError("should not reach"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.total_attempts == 4
        # One error should remain unconsumed in the fake
        assert len(connector._errors) == 1


# ---------------------------------------------------------------------------
# Successful connection scenarios
# ---------------------------------------------------------------------------


class TestExecutorSuccess:
    """Connection succeeds at various retry points."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        connector = FakeConnector()
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.success is True
        assert outcome.total_attempts == 1
        assert outcome.handle is not None
        assert outcome.handle.session_id == "exec-test-session"
        assert outcome.error is None

    @pytest.mark.asyncio
    async def test_success_after_one_retry(self) -> None:
        connector = FakeConnector(
            errors=[SSHConnectionError("transient")]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.success is True
        assert outcome.total_attempts == 2

    @pytest.mark.asyncio
    async def test_success_after_two_retries(self) -> None:
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                TimeoutError("fail 2"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.success is True
        assert outcome.total_attempts == 3

    @pytest.mark.asyncio
    async def test_success_after_three_retries(self) -> None:
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                OSError("fail 2"),
                ConnectionResetError("fail 3"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.success is True
        assert outcome.total_attempts == 4


# ---------------------------------------------------------------------------
# Permanent error: no retries
# ---------------------------------------------------------------------------


class TestExecutorPermanentError:
    """Permanent errors cause immediate failure without using retries."""

    @pytest.mark.asyncio
    async def test_auth_error_no_retries(self) -> None:
        connector = FakeConnector(
            errors=[SSHAuthenticationError("bad key")]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.success is False
        assert outcome.total_attempts == 1
        assert len(connector.connect_attempts) == 1

    @pytest.mark.asyncio
    async def test_host_key_error_no_retries(self) -> None:
        connector = FakeConnector(
            errors=[SSHHostKeyError("mismatch")]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.success is False
        assert outcome.total_attempts == 1

    @pytest.mark.asyncio
    async def test_permanent_after_transient(self) -> None:
        connector = FakeConnector(
            errors=[
                SSHConnectionError("transient"),
                SSHAuthenticationError("permanent"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.success is False
        assert outcome.total_attempts == 2
        assert "Permanent" in (outcome.error or "")


# ---------------------------------------------------------------------------
# ExecutionOutcome structure
# ---------------------------------------------------------------------------


class TestExecutionOutcome:
    """Verify ExecutionOutcome properties."""

    @pytest.mark.asyncio
    async def test_has_target(self) -> None:
        connector = FakeConnector()
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.target is _TARGET

    @pytest.mark.asyncio
    async def test_has_duration(self) -> None:
        connector = FakeConnector()
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.total_duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_attempts_list_is_tuple(self) -> None:
        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert isinstance(outcome.attempts, tuple)

    @pytest.mark.asyncio
    async def test_attempts_capture_error_details(self) -> None:
        connector = FakeConnector(
            errors=[SSHConnectionError("network down")]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        # First attempt failed
        assert len(outcome.attempts) >= 1
        failed_attempt = outcome.attempts[0]
        assert failed_attempt.success is False
        assert failed_attempt.error_type == "SSHConnectionError"
        assert "network down" in failed_attempt.error_message

    @pytest.mark.asyncio
    async def test_successful_attempt_recorded(self) -> None:
        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        # Last attempt succeeded
        last = outcome.attempts[-1]
        assert last.success is True
        assert last.error_type is None

    @pytest.mark.asyncio
    async def test_config_stored_in_outcome(self) -> None:
        connector = FakeConnector()
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        assert outcome.config.max_retries == 3


# ---------------------------------------------------------------------------
# ExecutionAttempt structure
# ---------------------------------------------------------------------------


class TestExecutionAttempt:
    """Verify ExecutionAttempt is frozen and has correct fields."""

    @pytest.mark.asyncio
    async def test_attempt_is_frozen(self) -> None:
        connector = FakeConnector()
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        attempt = outcome.attempts[0]
        with pytest.raises(AttributeError):
            attempt.attempt_number = 99  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_attempt_has_timestamp(self) -> None:
        connector = FakeConnector()
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        from datetime import datetime
        attempt = outcome.attempts[0]
        assert isinstance(attempt.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_attempt_delay_for_retries(self) -> None:
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY
        )

        # Failed attempts record the backoff delay slept before the next try.
        # Both failed attempts should have non-zero delays since retries remain.
        failed_attempts = [a for a in outcome.attempts if not a.success]
        assert len(failed_attempts) == 2
        for attempt in failed_attempts:
            assert attempt.delay_before_seconds > 0.0

        # The successful final attempt has delay=0 (no sleep needed after success)
        success_attempt = [a for a in outcome.attempts if a.success]
        assert len(success_attempt) == 1
        assert success_attempt[0].delay_before_seconds == 0.0


# ---------------------------------------------------------------------------
# on_attempt callback
# ---------------------------------------------------------------------------


class TestExecutorCallback:
    """Verify on_attempt callback fires for each attempt."""

    @pytest.mark.asyncio
    async def test_callback_fires_for_all_attempts(self) -> None:
        records: list[ExecutionAttempt] = []

        async def on_attempt(record: ExecutionAttempt) -> None:
            records.append(record)

        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )
        await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY,
            on_attempt=on_attempt,
        )

        # 1 failed attempt + 1 successful attempt = 2 callbacks
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_callback_not_invoked_when_none(self) -> None:
        connector = FakeConnector()
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY,
            on_attempt=None,
        )

        assert outcome.success is True

    @pytest.mark.asyncio
    async def test_callback_error_does_not_break_execution(self) -> None:
        async def broken(record: ExecutionAttempt) -> None:
            raise RuntimeError("callback exploded")

        connector = FakeConnector(
            errors=[SSHConnectionError("fail")]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=_FAST_3_RETRY,
            on_attempt=broken,
        )

        assert outcome.success is True


# ---------------------------------------------------------------------------
# Exponential backoff timing (approximate)
# ---------------------------------------------------------------------------


class TestExponentialBackoffTiming:
    """Verify delays grow exponentially: ~1s, ~2s, ~4s for default config.

    Uses measurable but small delays to avoid slow tests while confirming
    the exponential growth pattern.
    """

    @pytest.mark.asyncio
    async def test_delays_grow_exponentially(self) -> None:
        """Delays with no jitter should follow base * 2^attempt."""
        config = create_ssh_backoff(
            base_delay=0.01,   # 10ms
            max_delay=1.0,
            jitter_factor=0.0,
        )
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
                SSHConnectionError("fail 3"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=config
        )

        assert outcome.success is True
        assert outcome.total_attempts == 4

        # Retry delays should be approximately: 0.01, 0.02, 0.04
        delays = [a.delay_before_seconds for a in outcome.attempts if not a.success]
        assert len(delays) == 3
        # First retry delay should be approximately base_delay (0.01)
        assert 0.005 <= delays[0] <= 0.02
        # Second delay should be approximately 2x first
        assert delays[1] > delays[0] * 1.5
        # Third delay should be approximately 2x second
        assert delays[2] > delays[1] * 1.5

    @pytest.mark.asyncio
    async def test_total_duration_exceeds_sum_of_delays(self) -> None:
        config = create_ssh_backoff(
            base_delay=0.01,
            max_delay=1.0,
            jitter_factor=0.0,
        )
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=config
        )

        # Expected: 0.01 + 0.02 = 0.03s minimum delay
        assert outcome.total_duration_seconds >= 0.025


# ---------------------------------------------------------------------------
# Default config used when none provided
# ---------------------------------------------------------------------------


class TestExecutorDefaultConfig:
    """When no config is passed, executor uses DEFAULT_SSH_BACKOFF."""

    @pytest.mark.asyncio
    async def test_uses_default_config(self) -> None:
        """Verify that omitting config uses the 3-retry default.

        We queue exactly 4 errors to exhaust all retries (1 + 3),
        using near-zero-delay transient errors. The default config has
        real delays (1s, 2s, 4s) so we override with fast config.
        """
        # This test validates the config parameter default
        # We cannot wait for real 1s+2s+4s delays, so just verify
        # the config attribute matches.
        connector = FakeConnector()
        outcome = await execute_ssh_command(
            _TARGET, connector,
        )

        assert outcome.config.max_retries == 3
        assert outcome.config is DEFAULT_SSH_BACKOFF


# ---------------------------------------------------------------------------
# Safety override: max_retries forced to 3
# ---------------------------------------------------------------------------


class TestMaxRetriesOverride:
    """Executor forces max_retries=3 even if a different config is passed."""

    @pytest.mark.asyncio
    async def test_overrides_max_retries_5_to_3(self) -> None:
        config_with_five = BackoffConfig(
            base_delay=0.001,
            max_delay=0.01,
            multiplier=2.0,
            jitter_factor=0.0,
            max_retries=5,
        )
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
                SSHConnectionError("fail 3"),
                SSHConnectionError("fail 4"),
                SSHConnectionError("fail 5"),
                SSHConnectionError("fail 6"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=config_with_five
        )

        # Should have been capped at 4 attempts (1 + 3), not 6
        assert outcome.total_attempts == 4
        assert outcome.config.max_retries == 3

    @pytest.mark.asyncio
    async def test_overrides_max_retries_1_to_3(self) -> None:
        config_with_one = BackoffConfig(
            base_delay=0.001,
            max_delay=0.01,
            multiplier=2.0,
            jitter_factor=0.0,
            max_retries=1,
        )
        connector = FakeConnector(
            errors=[
                SSHConnectionError("fail 1"),
                SSHConnectionError("fail 2"),
                SSHConnectionError("fail 3"),
                SSHConnectionError("fail 4"),
            ]
        )
        outcome = await execute_ssh_command(
            _TARGET, connector, config=config_with_one
        )

        # Should have been expanded to 4 attempts (1 + 3), not 2
        assert outcome.total_attempts == 4
        assert outcome.config.max_retries == 3

    @pytest.mark.asyncio
    async def test_does_not_override_when_already_3(self) -> None:
        outcome = await execute_ssh_command(
            _TARGET, FakeConnector(), config=_FAST_3_RETRY
        )

        assert outcome.config.max_retries == 3
