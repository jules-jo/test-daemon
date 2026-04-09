"""Tests for CLI signal trap handlers (SIGINT, SIGTERM).

Verifies that signal handlers perform graceful detach from the daemon
socket without interrupting the running job. Tests cover:

- Signal handler installation and removal
- Graceful socket detach on SIGINT/SIGTERM
- User feedback messages on signal receipt
- Running job is not interrupted (disconnect_resilience)
- Double-signal (force quit) behavior
- Cleanup of resources on detach
- Integration with ClientConnection lifecycle
"""

from __future__ import annotations

import asyncio
import signal
import sys
from dataclasses import dataclass
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.cli.signal_handler import (
    DetachReason,
    DetachResult,
    SignalState,
    SignalTrapConfig,
    SignalTrapHandler,
    create_signal_handler,
    format_detach_message,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def output_stream() -> StringIO:
    """Provide a captured output stream for testing user feedback."""
    return StringIO()


@pytest.fixture
def mock_connection() -> AsyncMock:
    """Provide a mock ClientConnection with async close()."""
    conn = AsyncMock()
    conn.close = AsyncMock()
    conn.state = MagicMock()
    conn.state.value = "connected"
    return conn


@pytest.fixture
def default_config() -> SignalTrapConfig:
    """Provide default signal handler configuration."""
    return SignalTrapConfig()


@pytest.fixture
def handler_with_output(
    output_stream: StringIO,
    default_config: SignalTrapConfig,
) -> SignalTrapHandler:
    """Provide a SignalTrapHandler with captured output."""
    return SignalTrapHandler(config=default_config, output=output_stream)


# ---------------------------------------------------------------------------
# SignalTrapConfig tests
# ---------------------------------------------------------------------------


class TestSignalTrapConfig:
    """Tests for the immutable SignalTrapConfig dataclass."""

    def test_default_values(self) -> None:
        config = SignalTrapConfig()
        assert config.force_quit_timeout == 2.0
        assert config.detach_timeout == 5.0
        assert config.signals == (signal.SIGINT, signal.SIGTERM)

    def test_custom_values(self) -> None:
        config = SignalTrapConfig(
            force_quit_timeout=3.0,
            detach_timeout=10.0,
            signals=(signal.SIGINT,),
        )
        assert config.force_quit_timeout == 3.0
        assert config.detach_timeout == 10.0
        assert config.signals == (signal.SIGINT,)

    def test_frozen(self) -> None:
        config = SignalTrapConfig()
        with pytest.raises(AttributeError):
            config.force_quit_timeout = 99.0  # type: ignore[misc]

    def test_negative_force_quit_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="force_quit_timeout"):
            SignalTrapConfig(force_quit_timeout=-1.0)

    def test_zero_force_quit_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="force_quit_timeout"):
            SignalTrapConfig(force_quit_timeout=0.0)

    def test_negative_detach_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="detach_timeout"):
            SignalTrapConfig(detach_timeout=-1.0)

    def test_zero_detach_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="detach_timeout"):
            SignalTrapConfig(detach_timeout=0.0)

    def test_empty_signals_rejected(self) -> None:
        with pytest.raises(ValueError, match="signals"):
            SignalTrapConfig(signals=())


# ---------------------------------------------------------------------------
# DetachResult tests
# ---------------------------------------------------------------------------


class TestDetachResult:
    """Tests for the immutable DetachResult dataclass."""

    def test_graceful_detach(self) -> None:
        result = DetachResult(
            reason=DetachReason.GRACEFUL_DETACH,
            signal_received=signal.SIGINT,
            error=None,
        )
        assert result.reason == DetachReason.GRACEFUL_DETACH
        assert result.signal_received == signal.SIGINT
        assert result.error is None
        assert result.is_clean is True

    def test_force_quit(self) -> None:
        result = DetachResult(
            reason=DetachReason.FORCE_QUIT,
            signal_received=signal.SIGINT,
            error=None,
        )
        assert result.reason == DetachReason.FORCE_QUIT
        assert result.is_clean is False

    def test_error_detach(self) -> None:
        result = DetachResult(
            reason=DetachReason.ERROR,
            signal_received=signal.SIGTERM,
            error="socket close failed",
        )
        assert result.reason == DetachReason.ERROR
        assert result.error == "socket close failed"
        assert result.is_clean is False

    def test_frozen(self) -> None:
        result = DetachResult(
            reason=DetachReason.GRACEFUL_DETACH,
            signal_received=signal.SIGINT,
            error=None,
        )
        with pytest.raises(AttributeError):
            result.reason = DetachReason.FORCE_QUIT  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SignalState tests
# ---------------------------------------------------------------------------


class TestSignalState:
    """Tests for the SignalState enum."""

    def test_values(self) -> None:
        assert SignalState.IDLE.value == "idle"
        assert SignalState.DETACHING.value == "detaching"
        assert SignalState.FORCE_QUITTING.value == "force_quitting"
        assert SignalState.DETACHED.value == "detached"


# ---------------------------------------------------------------------------
# format_detach_message tests
# ---------------------------------------------------------------------------


class TestFormatDetachMessage:
    """Tests for the user feedback message formatter."""

    def test_sigint_first_signal(self) -> None:
        msg = format_detach_message(
            sig=signal.SIGINT,
            is_first=True,
        )
        assert "detaching" in msg.lower()
        assert "daemon continues" in msg.lower() or "running" in msg.lower()

    def test_sigint_second_signal(self) -> None:
        msg = format_detach_message(
            sig=signal.SIGINT,
            is_first=False,
        )
        assert "force" in msg.lower()

    def test_sigterm_first_signal(self) -> None:
        msg = format_detach_message(
            sig=signal.SIGTERM,
            is_first=True,
        )
        assert "detaching" in msg.lower()

    def test_sigterm_second_signal(self) -> None:
        msg = format_detach_message(
            sig=signal.SIGTERM,
            is_first=False,
        )
        assert "force" in msg.lower()


# ---------------------------------------------------------------------------
# SignalTrapHandler lifecycle tests
# ---------------------------------------------------------------------------


class TestSignalTrapHandlerLifecycle:
    """Tests for handler installation, removal, and state transitions."""

    def test_initial_state_is_idle(
        self,
        handler_with_output: SignalTrapHandler,
    ) -> None:
        assert handler_with_output.state == SignalState.IDLE

    def test_install_and_remove(self) -> None:
        """Verify install/remove restores original signal handlers."""
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=StringIO())

        original = signal.getsignal(signal.SIGINT)
        handler.install()
        assert handler._installed is True

        # The current handler should be different from the original
        current = signal.getsignal(signal.SIGINT)
        assert current is not original

        handler.remove()
        restored = signal.getsignal(signal.SIGINT)
        # Original handler should be restored
        assert restored is original
        assert handler._installed is False

    def test_double_install_is_idempotent(self) -> None:
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=StringIO())

        handler.install()
        try:
            # Second install should not raise
            handler.install()
        finally:
            handler.remove()

    def test_remove_without_install_is_safe(
        self,
        handler_with_output: SignalTrapHandler,
    ) -> None:
        # Should not raise
        handler_with_output.remove()

    def test_context_manager_installs_and_removes(self) -> None:
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=StringIO())

        original = signal.getsignal(signal.SIGINT)
        with handler:
            assert handler._installed is True
        assert handler._installed is False
        restored = signal.getsignal(signal.SIGINT)
        assert restored is original


# ---------------------------------------------------------------------------
# Signal handling behavior tests
# ---------------------------------------------------------------------------


class TestSignalHandling:
    """Tests for signal receipt behavior and state transitions."""

    def test_first_signal_transitions_to_detaching(self) -> None:
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        handler.install()
        try:
            handler._handle_signal(signal.SIGINT, None)
            assert handler.state == SignalState.DETACHING
        finally:
            handler.remove()

    def test_first_signal_prints_detach_message(self) -> None:
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        handler.install()
        try:
            handler._handle_signal(signal.SIGINT, None)
            text = output.getvalue()
            assert "detaching" in text.lower()
        finally:
            handler.remove()

    def test_second_signal_transitions_to_force_quitting(self) -> None:
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        handler.install()
        try:
            handler._handle_signal(signal.SIGINT, None)
            assert handler.state == SignalState.DETACHING

            handler._handle_signal(signal.SIGINT, None)
            assert handler.state == SignalState.FORCE_QUITTING
        finally:
            handler.remove()

    def test_second_signal_prints_force_quit_message(self) -> None:
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        handler.install()
        try:
            handler._handle_signal(signal.SIGINT, None)
            handler._handle_signal(signal.SIGINT, None)
            text = output.getvalue()
            assert "force" in text.lower()
        finally:
            handler.remove()

    def test_signal_count_increments(self) -> None:
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        handler.install()
        try:
            assert handler.signal_count == 0
            handler._handle_signal(signal.SIGINT, None)
            assert handler.signal_count == 1
            handler._handle_signal(signal.SIGINT, None)
            assert handler.signal_count == 2
        finally:
            handler.remove()


# ---------------------------------------------------------------------------
# Async detach tests
# ---------------------------------------------------------------------------


class TestAsyncDetach:
    """Tests for the async graceful_detach() method."""

    @pytest.mark.asyncio
    async def test_graceful_detach_closes_connection(
        self,
        mock_connection: AsyncMock,
    ) -> None:
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        result = await handler.graceful_detach(
            connection=mock_connection,
            signal_received=signal.SIGINT,
        )

        mock_connection.close.assert_awaited_once()
        assert result.reason == DetachReason.GRACEFUL_DETACH
        assert result.is_clean is True
        assert handler.state == SignalState.DETACHED

    @pytest.mark.asyncio
    async def test_graceful_detach_handles_close_error(
        self,
        mock_connection: AsyncMock,
    ) -> None:
        mock_connection.close = AsyncMock(side_effect=OSError("socket gone"))

        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        result = await handler.graceful_detach(
            connection=mock_connection,
            signal_received=signal.SIGINT,
        )

        assert result.reason == DetachReason.ERROR
        assert result.error is not None
        assert "socket gone" in result.error

    @pytest.mark.asyncio
    async def test_graceful_detach_with_none_connection(self) -> None:
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        result = await handler.graceful_detach(
            connection=None,
            signal_received=signal.SIGINT,
        )

        assert result.reason == DetachReason.GRACEFUL_DETACH
        assert result.is_clean is True
        assert handler.state == SignalState.DETACHED

    @pytest.mark.asyncio
    async def test_graceful_detach_prints_feedback(
        self,
        mock_connection: AsyncMock,
    ) -> None:
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        await handler.graceful_detach(
            connection=mock_connection,
            signal_received=signal.SIGINT,
        )

        text = output.getvalue()
        assert "detached" in text.lower() or "disconnected" in text.lower()

    @pytest.mark.asyncio
    async def test_detach_timeout_produces_error_result(self) -> None:
        """Connection close that exceeds timeout should produce error result."""
        slow_conn = AsyncMock()

        async def slow_close() -> None:
            await asyncio.sleep(10.0)

        slow_conn.close = slow_close

        output = StringIO()
        config = SignalTrapConfig(
            signals=(signal.SIGINT,),
            detach_timeout=0.1,
        )
        handler = SignalTrapHandler(config=config, output=output)

        result = await handler.graceful_detach(
            connection=slow_conn,
            signal_received=signal.SIGINT,
        )

        assert result.reason == DetachReason.ERROR
        assert result.error is not None
        assert "timed out" in result.error.lower()


# ---------------------------------------------------------------------------
# Detach event callback tests
# ---------------------------------------------------------------------------


class TestDetachCallback:
    """Tests for the optional on_detach callback."""

    @pytest.mark.asyncio
    async def test_on_detach_callback_invoked(
        self,
        mock_connection: AsyncMock,
    ) -> None:
        callback = MagicMock()
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(
            config=config,
            output=output,
            on_detach=callback,
        )

        result = await handler.graceful_detach(
            connection=mock_connection,
            signal_received=signal.SIGINT,
        )

        callback.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_on_detach_callback_exception_does_not_break_detach(
        self,
        mock_connection: AsyncMock,
    ) -> None:
        callback = MagicMock(side_effect=RuntimeError("callback crashed"))
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(
            config=config,
            output=output,
            on_detach=callback,
        )

        result = await handler.graceful_detach(
            connection=mock_connection,
            signal_received=signal.SIGINT,
        )

        # Detach should still succeed even if callback fails
        assert result.reason == DetachReason.GRACEFUL_DETACH


# ---------------------------------------------------------------------------
# Factory function tests
# ---------------------------------------------------------------------------


class TestCreateSignalHandler:
    """Tests for the create_signal_handler convenience factory."""

    def test_creates_handler_with_defaults(self) -> None:
        handler = create_signal_handler()
        assert isinstance(handler, SignalTrapHandler)
        assert handler.state == SignalState.IDLE

    def test_creates_handler_with_custom_config(self) -> None:
        config = SignalTrapConfig(force_quit_timeout=10.0)
        handler = create_signal_handler(config=config)
        assert handler._config.force_quit_timeout == 10.0

    def test_creates_handler_with_custom_output(self) -> None:
        output = StringIO()
        handler = create_signal_handler(output=output)
        assert handler._output is output


# ---------------------------------------------------------------------------
# Disconnect resilience tests (daemon continues)
# ---------------------------------------------------------------------------


class TestDisconnectResilience:
    """Verify that signal detach does NOT send any kill/stop to the daemon.

    The daemon owns the running job. CLI disconnect must never attempt
    to cancel, stop, or interfere with the daemon's running test.
    """

    @pytest.mark.asyncio
    async def test_detach_does_not_send_cancel_to_connection(
        self,
        mock_connection: AsyncMock,
    ) -> None:
        """Detach should only close, never send cancel/stop commands."""
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        await handler.graceful_detach(
            connection=mock_connection,
            signal_received=signal.SIGINT,
        )

        # close() should be called, but send() should NOT be called
        mock_connection.close.assert_awaited_once()
        mock_connection.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_feedback_mentions_daemon_continues(
        self,
        mock_connection: AsyncMock,
    ) -> None:
        """User feedback should clearly state the daemon continues running."""
        output = StringIO()
        config = SignalTrapConfig(signals=(signal.SIGINT,))
        handler = SignalTrapHandler(config=config, output=output)

        await handler.graceful_detach(
            connection=mock_connection,
            signal_received=signal.SIGINT,
        )

        text = output.getvalue().lower()
        assert "daemon" in text or "continues" in text or "running" in text
