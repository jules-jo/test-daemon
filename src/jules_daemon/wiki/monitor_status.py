"""Timestamped status model for SSH monitoring sessions.

Each MonitorStatus is an immutable snapshot of what the daemon observed at a
specific point in time during test execution. The daemon creates a new instance
for every status update -- never mutating an existing one.

Fields:
    session_id       -- identifies the SSH monitoring session
    timestamp        -- UTC datetime when this snapshot was captured
    raw_output_chunk -- raw stdout/stderr from the remote process
    parsed_state     -- structured interpretation of the output
    exit_status      -- remote process exit code (None while running)
    sequence_number  -- monotonically increasing counter per session
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Union

__all__ = ["OutputPhase", "ParsedState", "MonitorStatus"]


class OutputPhase(Enum):
    """Phase of test execution being monitored."""

    CONNECTING = "connecting"
    SETUP = "setup"
    COLLECTING = "collecting"
    RUNNING = "running"
    TEARDOWN = "teardown"
    COMPLETE = "complete"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ParsedState:
    """Structured interpretation of raw SSH output.

    Immutable -- all fields are set at creation time.
    """

    phase: OutputPhase = OutputPhase.UNKNOWN
    tests_discovered: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_total: int = 0
    current_test: str = ""
    error_message: str = ""

    def __post_init__(self) -> None:
        if self.tests_discovered < 0:
            raise ValueError("tests_discovered must not be negative")
        if self.tests_passed < 0:
            raise ValueError("tests_passed must not be negative")
        if self.tests_failed < 0:
            raise ValueError("tests_failed must not be negative")
        if self.tests_skipped < 0:
            raise ValueError("tests_skipped must not be negative")
        if self.tests_total < 0:
            raise ValueError("tests_total must not be negative")


# Sentinel for distinguishing "not provided" from None in with_update().
# This is needed because exit_status legitimately takes None as a value.
class _SentinelType:
    """Singleton sentinel for unset keyword arguments."""

    _instance: _SentinelType | None = None

    def __new__(cls) -> _SentinelType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNSET>"


_SENTINEL = _SentinelType()


@dataclass(frozen=True)
class MonitorStatus:
    """Timestamped status snapshot from SSH monitoring.

    Immutable -- state transitions produce new instances via the with_*
    methods. The sequence_number increases monotonically across updates
    for a given session, enabling ordering without relying solely on
    timestamp precision.
    """

    session_id: str
    timestamp: datetime
    raw_output_chunk: str = ""
    parsed_state: ParsedState = field(default_factory=ParsedState)
    exit_status: int | None = None
    sequence_number: int = 0

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must not be negative")

    # -- Computed properties --

    @property
    def is_terminal(self) -> bool:
        """True if the remote process has exited."""
        return self.exit_status is not None

    @property
    def is_success(self) -> bool:
        """True if the remote process exited with code 0."""
        return self.exit_status == 0

    # -- Immutable update methods --

    def with_update(
        self,
        *,
        timestamp: datetime | None = None,
        raw_output_chunk: str | None = None,
        parsed_state: ParsedState | None = None,
        exit_status: Union[int, None, _SentinelType] = _SENTINEL,
        sequence_number: int | None = None,
    ) -> MonitorStatus:
        """Return a new MonitorStatus with the specified fields replaced.

        Only the fields that are explicitly provided are changed; all others
        are carried forward from the current instance. Pass exit_status=None
        explicitly to clear a previously-set exit status.
        """
        kwargs: dict[str, object] = {}
        if timestamp is not None:
            kwargs["timestamp"] = timestamp
        if raw_output_chunk is not None:
            kwargs["raw_output_chunk"] = raw_output_chunk
        if parsed_state is not None:
            kwargs["parsed_state"] = parsed_state
        if not isinstance(exit_status, _SentinelType):
            kwargs["exit_status"] = exit_status
        if sequence_number is not None:
            kwargs["sequence_number"] = sequence_number
        return replace(self, **kwargs)

    def with_output(
        self,
        *,
        timestamp: datetime,
        raw_output_chunk: str,
    ) -> MonitorStatus:
        """Return a new snapshot with fresh output and auto-incremented sequence."""
        return replace(
            self,
            timestamp=timestamp,
            raw_output_chunk=raw_output_chunk,
            sequence_number=self.sequence_number + 1,
        )

    def with_parsed_state(
        self,
        *,
        timestamp: datetime,
        parsed_state: ParsedState,
    ) -> MonitorStatus:
        """Return a new snapshot with an updated parsed state."""
        return replace(
            self,
            timestamp=timestamp,
            parsed_state=parsed_state,
            sequence_number=self.sequence_number + 1,
        )

    def with_exit(
        self,
        *,
        timestamp: datetime,
        exit_status: int,
        parsed_state: ParsedState | None = None,
        raw_output_chunk: str | None = None,
    ) -> MonitorStatus:
        """Return a terminal snapshot with the process exit status."""
        kwargs: dict[str, object] = {
            "timestamp": timestamp,
            "exit_status": exit_status,
            "sequence_number": self.sequence_number + 1,
        }
        if parsed_state is not None:
            kwargs["parsed_state"] = parsed_state
        if raw_output_chunk is not None:
            kwargs["raw_output_chunk"] = raw_output_chunk
        return replace(self, **kwargs)
