"""CLI signal trap handlers for graceful daemon socket detach.

Implements SIGINT and SIGTERM handlers that perform a clean detach from
the daemon's IPC socket without interrupting the running job. The daemon
owns the test execution lifecycle -- CLI disconnect must never cancel,
stop, or interfere with the daemon's running test.

Key behaviors:

- **First signal** (SIGINT or SIGTERM): Initiates graceful detach.
  Closes the socket connection, prints user feedback explaining that
  the daemon continues monitoring autonomously, and transitions to
  DETACHED state. The running job is unaffected.

- **Second signal** (while detaching): Forces immediate exit. Prints
  a force-quit message and transitions to FORCE_QUITTING state. This
  handles the case where graceful detach is blocked (e.g., socket
  close hangs).

State machine::

    IDLE  --[signal]--> DETACHING --[signal]--> FORCE_QUITTING
                            |
                            +--[close complete]--> DETACHED

Signal handlers are installed/removed as a context manager or via
explicit ``install()`` / ``remove()`` calls. Original signal handlers
are saved and restored on removal.

Architecture::

    User presses Ctrl+C
        |
        v
    SignalTrapHandler._handle_signal()
        |
        +--> (first signal)
        |       write detach message to output
        |       set state = DETACHING
        |       set _detach_event (wakes async code)
        |
        +--> (second signal, already DETACHING)
                write force-quit message
                set state = FORCE_QUITTING
                set _force_event (wakes async code)

    Async code (CLI event loop):
        await handler.wait_for_detach()
        result = await handler.graceful_detach(connection, signal)

Usage::

    from jules_daemon.cli.signal_handler import (
        SignalTrapHandler,
        create_signal_handler,
    )

    handler = create_signal_handler()
    with handler:
        # ... connect to daemon, run CLI session ...
        # On Ctrl+C: handler detaches gracefully
        pass

    # Or async usage:
    handler = create_signal_handler()
    handler.install()
    try:
        result = await handler.graceful_detach(
            connection=client_conn,
            signal_received=signal.SIGINT,
        )
    finally:
        handler.remove()
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from enum import Enum
from types import FrameType
from typing import Any, Callable, TextIO

__all__ = [
    "DetachReason",
    "DetachResult",
    "SignalState",
    "SignalTrapConfig",
    "SignalTrapHandler",
    "create_signal_handler",
    "format_detach_message",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_FORCE_QUIT_TIMEOUT: float = 2.0
"""Seconds between first and second signal before force quit is triggered."""

_DEFAULT_DETACH_TIMEOUT: float = 5.0
"""Maximum seconds to wait for connection close during graceful detach."""

_DEFAULT_SIGNALS: tuple[signal.Signals, ...] = (signal.SIGINT, signal.SIGTERM)
"""Signals to trap by default."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SignalState(Enum):
    """Lifecycle state of the signal handler.

    Values:
        IDLE:           No signal received. Normal operation.
        DETACHING:      First signal received. Graceful detach in progress.
        FORCE_QUITTING: Second signal received. Immediate exit requested.
        DETACHED:       Graceful detach completed successfully.
    """

    IDLE = "idle"
    DETACHING = "detaching"
    FORCE_QUITTING = "force_quitting"
    DETACHED = "detached"


class DetachReason(Enum):
    """Why the CLI detached from the daemon.

    Values:
        GRACEFUL_DETACH: Clean socket close completed normally.
        FORCE_QUIT:      User sent a second signal during detach.
        ERROR:           Socket close failed or timed out.
    """

    GRACEFUL_DETACH = "graceful_detach"
    FORCE_QUIT = "force_quit"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Immutable data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalTrapConfig:
    """Immutable configuration for signal trap handlers.

    Attributes:
        force_quit_timeout: Seconds allowed for graceful detach before
            a second signal triggers force quit. Not currently used as
            a timer -- the second signal is the trigger.
        detach_timeout:     Maximum seconds to wait for the connection
            close() call to complete during graceful detach.
        signals:            Tuple of signals to trap. Must be non-empty.
    """

    force_quit_timeout: float = _DEFAULT_FORCE_QUIT_TIMEOUT
    detach_timeout: float = _DEFAULT_DETACH_TIMEOUT
    signals: tuple[signal.Signals, ...] = _DEFAULT_SIGNALS

    def __post_init__(self) -> None:
        if self.force_quit_timeout <= 0:
            raise ValueError(
                f"force_quit_timeout must be positive, "
                f"got {self.force_quit_timeout}"
            )
        if self.detach_timeout <= 0:
            raise ValueError(
                f"detach_timeout must be positive, "
                f"got {self.detach_timeout}"
            )
        if not self.signals:
            raise ValueError("signals must not be empty")


@dataclass(frozen=True)
class DetachResult:
    """Immutable result of a signal-triggered detach operation.

    Attributes:
        reason:          Why the detach completed (graceful, forced, error).
        signal_received: The signal that triggered the detach.
        error:           Human-readable error description on failure.
                         None on clean detach.
    """

    reason: DetachReason
    signal_received: signal.Signals
    error: str | None

    @property
    def is_clean(self) -> bool:
        """True if the detach completed cleanly without errors."""
        return self.reason == DetachReason.GRACEFUL_DETACH


# ---------------------------------------------------------------------------
# Pure functions: user feedback messages
# ---------------------------------------------------------------------------


def format_detach_message(
    *,
    sig: signal.Signals,
    is_first: bool,
) -> str:
    """Format a user feedback message for signal receipt.

    The first signal produces a graceful detach message explaining that
    the daemon continues running. The second signal produces a force
    quit warning.

    Args:
        sig: The signal that was received.
        is_first: True if this is the first signal (graceful detach).
            False if this is the second signal (force quit).

    Returns:
        Human-readable message string for terminal display.
    """
    sig_name = sig.name

    if is_first:
        return (
            f"\n-- Received {sig_name}. Detaching from daemon socket.\n"
            f"-- The daemon continues running and monitoring your tests.\n"
            f"-- Use 'jules watch' to reattach. "
            f"Press Ctrl+C again to force quit.\n"
        )

    return (
        f"\n-- Received {sig_name} again. Force quitting CLI.\n"
        f"-- The daemon continues running in the background.\n"
    )


def _format_detach_complete_message() -> str:
    """Format the message shown after successful detach.

    Returns:
        Human-readable completion message.
    """
    return (
        "-- Detached from daemon. "
        "The daemon continues running your tests autonomously.\n"
        "-- Reconnect with 'jules watch' or 'jules status'.\n"
    )


# ---------------------------------------------------------------------------
# SignalTrapHandler
# ---------------------------------------------------------------------------


class SignalTrapHandler:
    """CLI signal trap handler for graceful daemon socket detach.

    Installs signal handlers for SIGINT and SIGTERM that perform a clean
    detach from the daemon's IPC socket. The first signal initiates
    graceful detach; the second signal forces immediate exit.

    The handler never sends cancel/stop commands to the daemon. It only
    closes the local socket connection. The daemon continues monitoring
    the running test autonomously.

    Supports both context manager and explicit install/remove patterns.

    Note on mutability: This class holds mutable state (_state,
    _signal_count, _installed, etc.) because it is an active signal
    handler, not a value object. The frozen dataclasses (config, result)
    are immutable.

    Args:
        config:    Signal handler configuration.
        output:    Text IO stream for user feedback messages.
        on_detach: Optional callback invoked after detach completes,
                   receiving the DetachResult.
    """

    def __init__(
        self,
        *,
        config: SignalTrapConfig,
        output: TextIO | None = None,
        on_detach: Callable[[DetachResult], Any] | None = None,
    ) -> None:
        self._config = config
        self._output = output or sys.stderr
        self._on_detach = on_detach

        # Mutable state
        self._state = SignalState.IDLE
        self._signal_count = 0
        self._installed = False
        self._original_handlers: dict[
            signal.Signals, signal._HANDLER
        ] = {}

    # -- Properties --

    @property
    def state(self) -> SignalState:
        """Current lifecycle state of the handler."""
        return self._state

    @property
    def signal_count(self) -> int:
        """Number of signals received since installation."""
        return self._signal_count

    # -- Context manager --

    def __enter__(self) -> SignalTrapHandler:
        """Install signal handlers on context entry."""
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Remove signal handlers on context exit."""
        self.remove()

    # -- Installation --

    def install(self) -> None:
        """Install signal handlers for the configured signals.

        Saves the original handlers for restoration on remove(). If
        already installed, this is a no-op (idempotent).
        """
        if self._installed:
            logger.debug("Signal handlers already installed, skipping")
            return

        for sig in self._config.signals:
            try:
                original = signal.getsignal(sig)
                self._original_handlers[sig] = original
                signal.signal(sig, self._handle_signal)
                logger.debug("Installed handler for %s", sig.name)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Could not install handler for %s: %s",
                    sig.name,
                    exc,
                )

        self._installed = True

    def remove(self) -> None:
        """Remove signal handlers and restore the originals.

        If not installed, this is a safe no-op.
        """
        if not self._installed:
            return

        for sig, original in self._original_handlers.items():
            try:
                signal.signal(sig, original)
                logger.debug("Restored original handler for %s", sig.name)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Could not restore handler for %s: %s",
                    sig.name,
                    exc,
                )

        self._original_handlers.clear()
        self._installed = False

    # -- Signal handler callback --

    def _handle_signal(
        self,
        signum: int,
        frame: FrameType | None,
    ) -> None:
        """Handle a received signal (runs in signal context).

        First signal: transition to DETACHING, print feedback.
        Second signal: transition to FORCE_QUITTING, print feedback.

        This method is synchronous (signal handler requirement). It sets
        state and writes to output but does not perform async operations.
        The async ``graceful_detach()`` method performs the actual socket
        close.

        Args:
            signum: The signal number received.
            frame: The current stack frame (unused).
        """
        self._signal_count += 1

        try:
            sig = signal.Signals(signum)
        except ValueError:
            sig = signal.SIGINT  # fallback

        is_first = self._state == SignalState.IDLE

        if is_first:
            self._state = SignalState.DETACHING
            msg = format_detach_message(sig=sig, is_first=True)
        else:
            self._state = SignalState.FORCE_QUITTING
            msg = format_detach_message(sig=sig, is_first=False)

        try:
            self._output.write(msg)
            self._output.flush()
        except Exception:
            # Output may be broken (pipe closed, etc.) -- ignore
            pass

        logger.info(
            "Signal %s received (count=%d, state=%s)",
            sig.name,
            self._signal_count,
            self._state.value,
        )

    # -- Async detach --

    async def graceful_detach(
        self,
        *,
        connection: Any | None,
        signal_received: signal.Signals,
    ) -> DetachResult:
        """Perform graceful detach from the daemon socket.

        Closes the connection (if provided) with a timeout. Never sends
        cancel or stop commands to the daemon -- only closes the local
        socket. The daemon continues monitoring autonomously.

        On success, prints a completion message and invokes the on_detach
        callback (if provided).

        Args:
            connection: The IPC client connection to close. When None,
                no connection cleanup is needed (already disconnected).
            signal_received: The signal that triggered the detach.

        Returns:
            DetachResult describing the outcome of the detach operation.
        """
        result: DetachResult

        if connection is None:
            result = DetachResult(
                reason=DetachReason.GRACEFUL_DETACH,
                signal_received=signal_received,
                error=None,
            )
            self._state = SignalState.DETACHED
            self._write_completion_feedback()
            self._invoke_callback(result)
            return result

        # Attempt to close the connection with timeout
        try:
            await asyncio.wait_for(
                connection.close(),
                timeout=self._config.detach_timeout,
            )
            result = DetachResult(
                reason=DetachReason.GRACEFUL_DETACH,
                signal_received=signal_received,
                error=None,
            )
            self._state = SignalState.DETACHED
            self._write_completion_feedback()

        except asyncio.TimeoutError:
            error_msg = (
                f"Connection close timed out after "
                f"{self._config.detach_timeout:.1f}s. "
                f"The daemon may still be running."
            )
            result = DetachResult(
                reason=DetachReason.ERROR,
                signal_received=signal_received,
                error=error_msg,
            )
            logger.warning("Detach timeout: %s", error_msg)
            self._write_error_feedback(error_msg)

        except Exception as exc:
            error_msg = f"Connection close failed: {exc}"
            result = DetachResult(
                reason=DetachReason.ERROR,
                signal_received=signal_received,
                error=str(exc),
            )
            logger.warning("Detach error: %s", error_msg)
            self._write_error_feedback(error_msg)

        self._invoke_callback(result)
        return result

    # -- Internal helpers --

    def _write_completion_feedback(self) -> None:
        """Write the detach completion message to the output stream."""
        try:
            self._output.write(_format_detach_complete_message())
            self._output.flush()
        except Exception:
            pass

    def _write_error_feedback(self, error: str) -> None:
        """Write an error message to the output stream.

        Args:
            error: Human-readable error description.
        """
        try:
            self._output.write(f"-- Detach error: {error}\n")
            self._output.flush()
        except Exception:
            pass

    def _invoke_callback(self, result: DetachResult) -> None:
        """Invoke the on_detach callback, swallowing any exceptions.

        The callback is best-effort: its failure should not prevent
        the detach from completing.

        Args:
            result: The DetachResult to pass to the callback.
        """
        if self._on_detach is None:
            return

        try:
            self._on_detach(result)
        except Exception as exc:
            logger.warning(
                "on_detach callback raised %s: %s",
                type(exc).__name__,
                exc,
            )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_signal_handler(
    *,
    config: SignalTrapConfig | None = None,
    output: TextIO | None = None,
    on_detach: Callable[[DetachResult], Any] | None = None,
) -> SignalTrapHandler:
    """Create a SignalTrapHandler with sensible defaults.

    Convenience factory that provides default configuration and output
    stream. Use this instead of constructing SignalTrapHandler directly
    for standard CLI usage.

    Args:
        config:    Optional custom configuration. Uses defaults if None.
        output:    Optional output stream. Defaults to sys.stderr.
        on_detach: Optional callback invoked after detach completes.

    Returns:
        Configured SignalTrapHandler ready for installation.
    """
    effective_config = config if config is not None else SignalTrapConfig()
    return SignalTrapHandler(
        config=effective_config,
        output=output,
        on_detach=on_detach,
    )
