"""Async polling loop for autonomous SSH output monitoring.

Polls the SSH output reader at a configurable interval (default <=10s),
feeds each chunk to the MonitorStatus immutable update methods, and
exposes start/stop/reconfigure controls.

The loop is procedural daemon code -- the "monitoring" middle layer between
the LLM bookends (setup + summarization). It runs autonomously even when
the CLI is disconnected.

Key design decisions:
    - Uses asyncio.Task for the background loop
    - Feeds SSHOutput chunks to MonitorStatus.with_output() / .with_exit()
    - Skips empty reads to avoid noisy status updates
    - Auto-stops when the SSH session reaches a terminal state
    - Gracefully handles transient reader errors with configurable max
    - All state transitions produce new immutable instances
    - StatusCallback protocol for decoupled notification

Usage:
    loop = PollingLoop(
        channel=channel,
        session_id="run-abc",
        on_status_update=my_callback,
    )
    async with loop:
        # loop polls autonomously until channel exits or stop() is called
        await some_other_work()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Optional, Protocol

from jules_daemon.ssh.reader import SSHChannelHandle, SSHOutput, read_ssh_output
from jules_daemon.wiki.monitor_status import MonitorStatus

__all__ = [
    "PollingConfig",
    "PollingLoop",
    "PollingState",
    "StatusCallback",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback protocol
# ---------------------------------------------------------------------------


class StatusCallback(Protocol):
    """Protocol for receiving MonitorStatus updates from the polling loop."""

    def __call__(self, status: MonitorStatus) -> Awaitable[None]: ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PollingConfig:
    """Immutable configuration for the polling loop.

    Attributes:
        interval_seconds: Seconds between poll iterations. Must be positive.
            Default is 10.0 (the maximum allowed by the status freshness SLA).
        max_consecutive_errors: Number of consecutive read errors before
            the loop transitions to ERROR state. Zero means unlimited.
    """

    interval_seconds: float = 10.0
    max_consecutive_errors: int = 5

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, got {self.interval_seconds}"
            )
        if self.max_consecutive_errors < 0:
            raise ValueError(
                f"max_consecutive_errors must be non-negative, "
                f"got {self.max_consecutive_errors}"
            )


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class PollingState(Enum):
    """Lifecycle states for the polling loop."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _build_output_text(ssh_output: SSHOutput) -> str:
    """Combine stdout and stderr into a single text chunk.

    Decodes bytes to UTF-8 with replacement for invalid sequences.
    Stderr lines are prefixed for clarity when both streams have data.
    """
    parts: list[str] = []

    if ssh_output.stdout:
        parts.append(ssh_output.stdout.decode("utf-8", errors="replace"))

    if ssh_output.stderr:
        stderr_text = ssh_output.stderr.decode("utf-8", errors="replace")
        if ssh_output.stdout:
            parts.append(f"[stderr] {stderr_text}")
        else:
            parts.append(stderr_text)

    return "".join(parts)


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------


class PollingLoop:
    """Async polling loop for SSH output monitoring.

    Reads SSH output at ``config.interval_seconds`` intervals and invokes
    ``on_status_update`` with each new MonitorStatus snapshot. The loop
    auto-stops when the SSH channel reaches a terminal state (exit code
    received or channel closed).

    Supports:
        - start() / stop() lifecycle management
        - reconfigure() for live interval changes
        - async context manager protocol
        - Graceful error handling with configurable max consecutive errors

    Thread safety:
        This class is NOT thread-safe. All methods must be called from
        the same asyncio event loop.
    """

    def __init__(
        self,
        *,
        channel: SSHChannelHandle,
        session_id: str,
        on_status_update: StatusCallback,
        config: PollingConfig | None = None,
    ) -> None:
        if not session_id:
            raise ValueError("session_id must not be empty")

        self._channel = channel
        self._session_id = session_id
        self._on_status_update = on_status_update
        self._config = config or PollingConfig()
        self._state = PollingState.IDLE
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()
        self._latest_status: Optional[MonitorStatus] = None
        self._consecutive_errors: int = 0

    # -- Public properties --

    @property
    def state(self) -> PollingState:
        """Current lifecycle state of the polling loop."""
        return self._state

    @property
    def session_id(self) -> str:
        """Session identifier for the monitoring session."""
        return self._session_id

    @property
    def config(self) -> PollingConfig:
        """Current polling configuration (immutable snapshot)."""
        return self._config

    @property
    def latest_status(self) -> Optional[MonitorStatus]:
        """Most recent MonitorStatus snapshot, or None if no reads yet."""
        return self._latest_status

    @property
    def consecutive_errors(self) -> int:
        """Number of consecutive read errors since last success."""
        return self._consecutive_errors

    # -- Lifecycle controls --

    async def start(self) -> None:
        """Start the polling loop as a background asyncio task.

        Raises:
            RuntimeError: If the loop is already running.
        """
        if self._state == PollingState.RUNNING:
            raise RuntimeError("Polling loop is already running")

        self._state = PollingState.RUNNING
        self._stop_event.clear()
        self._consecutive_errors = 0
        self._task = asyncio.create_task(
            self._poll_loop(),
            name=f"polling-loop-{self._session_id}",
        )

    async def stop(self) -> None:
        """Stop the polling loop gracefully.

        If the loop is not running, this is a no-op. Sets the stop event
        and waits for the background task to complete.
        """
        if self._state not in (PollingState.RUNNING, PollingState.STOPPING):
            return

        self._state = PollingState.STOPPING
        self._stop_event.set()

        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Polling loop %s did not stop within 5s, cancelling",
                    self._session_id,
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None

        self._state = PollingState.STOPPED

    def reconfigure(self, config: PollingConfig) -> None:
        """Update the polling configuration.

        Takes effect on the next iteration (does not interrupt a sleep
        already in progress).

        Args:
            config: New polling configuration.
        """
        self._config = config

    # -- Async context manager --

    async def __aenter__(self) -> PollingLoop:
        await self.start()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: object, exc_tb: object) -> None:
        await self.stop()

    # -- Internal polling loop --

    async def _poll_loop(self) -> None:
        """Main polling coroutine.

        Reads SSH output, updates MonitorStatus, and invokes the callback.
        Runs until stop_event is set or a terminal condition is reached.
        """
        sequence_number = 0

        logger.info(
            "Polling loop started: session=%s interval=%.2fs",
            self._session_id,
            self._config.interval_seconds,
        )

        try:
            while not self._stop_event.is_set():
                try:
                    ssh_output = await read_ssh_output(self._channel)
                    self._consecutive_errors = 0
                except Exception as exc:
                    self._consecutive_errors += 1
                    logger.warning(
                        "SSH read error (attempt %d/%d): %s",
                        self._consecutive_errors,
                        self._config.max_consecutive_errors,
                        exc,
                    )
                    if (
                        self._config.max_consecutive_errors > 0
                        and self._consecutive_errors
                        >= self._config.max_consecutive_errors
                    ):
                        logger.error(
                            "Max consecutive errors reached (%d) for session %s",
                            self._config.max_consecutive_errors,
                            self._session_id,
                        )
                        self._state = PollingState.ERROR
                        return

                    # Wait before retrying
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=self._config.interval_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                # Determine if this is a terminal read
                is_terminal = (
                    ssh_output.exit_code is not None
                    or ssh_output.channel_closed
                )

                # Build the combined output text
                output_text = _build_output_text(ssh_output)

                # Skip empty non-terminal reads (no data to report)
                if not ssh_output.has_data and not is_terminal:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=self._config.interval_seconds,
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                # Build the new MonitorStatus snapshot
                now = _now_utc()
                sequence_number += 1

                if is_terminal:
                    exit_code = ssh_output.exit_code if ssh_output.exit_code is not None else -1
                    if self._latest_status is not None:
                        new_status = self._latest_status.with_exit(
                            timestamp=now,
                            exit_status=exit_code,
                            raw_output_chunk=output_text if output_text else None,
                        )
                    else:
                        new_status = MonitorStatus(
                            session_id=self._session_id,
                            timestamp=now,
                            raw_output_chunk=output_text,
                            exit_status=exit_code,
                            sequence_number=sequence_number,
                        )
                else:
                    if self._latest_status is not None:
                        new_status = self._latest_status.with_output(
                            timestamp=now,
                            raw_output_chunk=output_text,
                        )
                    else:
                        new_status = MonitorStatus(
                            session_id=self._session_id,
                            timestamp=now,
                            raw_output_chunk=output_text,
                            sequence_number=sequence_number,
                        )

                self._latest_status = new_status

                # Invoke the callback
                try:
                    await self._on_status_update(new_status)
                except Exception as exc:
                    logger.warning(
                        "Status callback error for session %s: %s",
                        self._session_id,
                        exc,
                    )

                # Auto-stop on terminal state
                if is_terminal:
                    logger.info(
                        "Terminal state reached for session %s: exit_code=%s",
                        self._session_id,
                        ssh_output.exit_code,
                    )
                    return

                # Sleep until next poll (interruptible by stop_event)
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._config.interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass

        except asyncio.CancelledError:
            logger.info(
                "Polling loop cancelled for session %s", self._session_id
            )
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error in polling loop for session %s: %s",
                self._session_id,
                exc,
            )
            self._state = PollingState.ERROR
        finally:
            if self._state not in (PollingState.STOPPING, PollingState.ERROR):
                self._state = PollingState.STOPPED
            elif self._state == PollingState.STOPPING:
                self._state = PollingState.STOPPED
            logger.info(
                "Polling loop ended: session=%s state=%s",
                self._session_id,
                self._state.value,
            )
