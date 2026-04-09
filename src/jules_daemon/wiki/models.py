"""Immutable data models for daemon current-run state.

All models are frozen dataclasses -- state transitions produce new instances,
never mutate existing ones.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class RunStatus(Enum):
    """Lifecycle states for a test run."""

    IDLE = "idle"
    PENDING_APPROVAL = "pending_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SSHTarget:
    """Remote host connection details."""

    host: str
    user: str
    port: int = 22
    key_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("SSH host must not be empty")
        if not self.user:
            raise ValueError("SSH user must not be empty")
        if not (1 <= self.port <= 65535):
            raise ValueError(f"SSH port must be 1-65535, got {self.port}")
        if self.key_path is not None:
            from pathlib import PurePosixPath
            p = PurePosixPath(self.key_path)
            if not p.is_absolute():
                raise ValueError(
                    f"SSH key_path must be an absolute path, got: {self.key_path!r}"
                )


@dataclass(frozen=True)
class Command:
    """The natural-language command and its resolved shell form."""

    natural_language: str
    resolved_shell: str = ""
    approved: bool = False
    approved_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.natural_language:
            raise ValueError("Natural language command must not be empty")

    def with_approval(self, resolved_shell: str) -> Command:
        """Return a new Command marked as approved with the resolved shell."""
        return replace(
            self,
            resolved_shell=resolved_shell,
            approved=True,
            approved_at=datetime.now(timezone.utc),
        )


@dataclass(frozen=True)
class ProcessIDs:
    """Process identifiers for the daemon and remote process."""

    daemon: Optional[int] = None
    remote: Optional[int] = None


@dataclass(frozen=True)
class Progress:
    """Progress checkpoint for a running test suite."""

    percent: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_total: int = 0
    last_output_line: str = ""
    checkpoint_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.percent <= 100.0):
            raise ValueError(
                f"Progress percent must be 0-100, got {self.percent}"
            )
        if self.tests_passed < 0 or self.tests_failed < 0:
            raise ValueError("Test counts must not be negative")


def _generate_run_id() -> str:
    """Generate a new unique run identifier."""
    return str(uuid.uuid4())


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class CurrentRun:
    """Complete snapshot of the daemon's current run state.

    This is the top-level record that gets serialized to the wiki file.
    Transitions produce new instances via the with_* methods.
    """

    status: RunStatus = RunStatus.IDLE
    run_id: str = field(default_factory=_generate_run_id)
    ssh_target: Optional[SSHTarget] = None
    command: Optional[Command] = None
    pids: ProcessIDs = field(default_factory=ProcessIDs)
    progress: Progress = field(default_factory=Progress)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    # -- State transition methods (each returns a new frozen instance) --

    def with_pending_approval(
        self,
        ssh_target: SSHTarget,
        command: Command,
        daemon_pid: int,
    ) -> CurrentRun:
        """Transition to pending_approval: user must confirm the command."""
        return replace(
            self,
            status=RunStatus.PENDING_APPROVAL,
            ssh_target=ssh_target,
            command=command,
            pids=ProcessIDs(daemon=daemon_pid),
            progress=Progress(),
            started_at=None,
            completed_at=None,
            error=None,
            updated_at=_now_utc(),
        )

    def with_running(
        self,
        resolved_shell: str,
        remote_pid: Optional[int] = None,
    ) -> CurrentRun:
        """Transition to running: command approved and execution started."""
        if self.command is None:
            raise ValueError("Cannot start running without a command")
        return replace(
            self,
            status=RunStatus.RUNNING,
            command=self.command.with_approval(resolved_shell),
            pids=ProcessIDs(
                daemon=self.pids.daemon,
                remote=remote_pid,
            ),
            started_at=_now_utc(),
            updated_at=_now_utc(),
        )

    def with_progress(self, progress: Progress) -> CurrentRun:
        """Update the progress checkpoint."""
        return replace(
            self,
            progress=progress,
            updated_at=_now_utc(),
        )

    def with_completed(self, final_progress: Progress) -> CurrentRun:
        """Transition to completed: test suite finished successfully."""
        return replace(
            self,
            status=RunStatus.COMPLETED,
            progress=final_progress,
            completed_at=_now_utc(),
            updated_at=_now_utc(),
        )

    def with_failed(self, error: str, final_progress: Progress) -> CurrentRun:
        """Transition to failed: test suite or connection failed."""
        return replace(
            self,
            status=RunStatus.FAILED,
            progress=final_progress,
            error=error,
            completed_at=_now_utc(),
            updated_at=_now_utc(),
        )

    def with_cancelled(self) -> CurrentRun:
        """Transition to cancelled: user cancelled the run."""
        return replace(
            self,
            status=RunStatus.CANCELLED,
            completed_at=_now_utc(),
            updated_at=_now_utc(),
        )

    @property
    def is_active(self) -> bool:
        """True if the run is in a non-terminal state."""
        return self.status in (RunStatus.PENDING_APPROVAL, RunStatus.RUNNING)

    @property
    def is_terminal(self) -> bool:
        """True if the run reached a terminal state."""
        return self.status in (
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        )
