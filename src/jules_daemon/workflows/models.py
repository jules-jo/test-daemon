"""Immutable workflow models for stateful test orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum


def _now_utc() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


class WorkflowStatus(Enum):
    """Top-level workflow lifecycle."""

    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED_SUCCESS = "completed_success"
    COMPLETED_FAILURE = "completed_failure"
    CANCELLED = "cancelled"
    ERROR = "error"

    @property
    def is_terminal(self) -> bool:
        """Return True when the workflow reached a terminal state."""
        return self in {
            WorkflowStatus.COMPLETED_SUCCESS,
            WorkflowStatus.COMPLETED_FAILURE,
            WorkflowStatus.CANCELLED,
            WorkflowStatus.ERROR,
        }


class WorkflowStepStatus(Enum):
    """Lifecycle for an individual workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED_SUCCESS = "completed_success"
    COMPLETED_FAILURE = "completed_failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

    @property
    def is_terminal(self) -> bool:
        """Return True when the step reached a terminal state."""
        return self in {
            WorkflowStepStatus.COMPLETED_SUCCESS,
            WorkflowStepStatus.COMPLETED_FAILURE,
            WorkflowStepStatus.CANCELLED,
            WorkflowStepStatus.SKIPPED,
        }


class ArtifactStatus(Enum):
    """Availability state for a prerequisite artifact."""

    UNKNOWN = "unknown"
    PRESENT = "present"
    MISSING = "missing"


@dataclass(frozen=True)
class ArtifactState:
    """Durable snapshot of an artifact check."""

    name: str
    status: ArtifactStatus = ArtifactStatus.UNKNOWN
    details: str | None = None
    checked_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ArtifactState.name must not be empty")


@dataclass(frozen=True)
class WorkflowStepRecord:
    """Immutable state for one workflow step."""

    workflow_id: str
    step_id: str
    name: str
    kind: str = "run"
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    run_id: str | None = None
    command: str | None = None
    target_host: str | None = None
    target_user: str | None = None
    exit_code: int | None = None
    summary: str | None = None
    error: str | None = None
    last_output_line: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        if not self.workflow_id.strip():
            raise ValueError("WorkflowStepRecord.workflow_id must not be empty")
        if not self.step_id.strip():
            raise ValueError("WorkflowStepRecord.step_id must not be empty")
        if not self.name.strip():
            raise ValueError("WorkflowStepRecord.name must not be empty")

    def with_running(
        self,
        *,
        run_id: str | None = None,
        command: str | None = None,
        target_host: str | None = None,
        target_user: str | None = None,
    ) -> WorkflowStepRecord:
        """Transition the step to running."""
        return replace(
            self,
            status=WorkflowStepStatus.RUNNING,
            run_id=run_id if run_id is not None else self.run_id,
            command=command if command is not None else self.command,
            target_host=(
                target_host if target_host is not None else self.target_host
            ),
            target_user=(
                target_user if target_user is not None else self.target_user
            ),
            started_at=self.started_at or _now_utc(),
            updated_at=_now_utc(),
            error=None,
        )

    def with_completed_success(
        self,
        *,
        summary: str | None = None,
        last_output_line: str | None = None,
        exit_code: int | None = None,
    ) -> WorkflowStepRecord:
        """Transition the step to success."""
        return replace(
            self,
            status=WorkflowStepStatus.COMPLETED_SUCCESS,
            summary=summary if summary is not None else self.summary,
            last_output_line=(
                last_output_line
                if last_output_line is not None
                else self.last_output_line
            ),
            exit_code=exit_code,
            completed_at=_now_utc(),
            updated_at=_now_utc(),
            error=None,
        )

    def with_completed_failure(
        self,
        *,
        error: str,
        summary: str | None = None,
        last_output_line: str | None = None,
        exit_code: int | None = None,
    ) -> WorkflowStepRecord:
        """Transition the step to failure."""
        return replace(
            self,
            status=WorkflowStepStatus.COMPLETED_FAILURE,
            error=error,
            summary=summary if summary is not None else self.summary,
            last_output_line=(
                last_output_line
                if last_output_line is not None
                else self.last_output_line
            ),
            exit_code=exit_code,
            completed_at=_now_utc(),
            updated_at=_now_utc(),
        )

    def with_cancelled(
        self,
        *,
        summary: str | None = None,
    ) -> WorkflowStepRecord:
        """Transition the step to cancelled."""
        return replace(
            self,
            status=WorkflowStepStatus.CANCELLED,
            summary=summary if summary is not None else self.summary,
            completed_at=_now_utc(),
            updated_at=_now_utc(),
        )


@dataclass(frozen=True)
class WorkflowRecord:
    """Immutable top-level workflow state."""

    workflow_id: str
    request_text: str
    workflow_kind: str = "test_run"
    status: WorkflowStatus = WorkflowStatus.PLANNING
    run_id: str | None = None
    current_step_id: str | None = None
    target_host: str | None = None
    target_user: str | None = None
    summary: str | None = None
    error: str | None = None
    artifact_states: tuple[ArtifactState, ...] = ()
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        if not self.workflow_id.strip():
            raise ValueError("WorkflowRecord.workflow_id must not be empty")
        if not self.request_text.strip():
            raise ValueError("WorkflowRecord.request_text must not be empty")
        if not self.workflow_kind.strip():
            raise ValueError("WorkflowRecord.workflow_kind must not be empty")

    def with_running(
        self,
        *,
        current_step_id: str,
        run_id: str | None = None,
        target_host: str | None = None,
        target_user: str | None = None,
    ) -> WorkflowRecord:
        """Transition the workflow to running."""
        return replace(
            self,
            status=WorkflowStatus.RUNNING,
            current_step_id=current_step_id,
            run_id=run_id if run_id is not None else self.run_id,
            target_host=target_host if target_host is not None else self.target_host,
            target_user=target_user if target_user is not None else self.target_user,
            started_at=self.started_at or _now_utc(),
            updated_at=_now_utc(),
            error=None,
        )

    def with_completed_success(
        self,
        *,
        summary: str | None = None,
        current_step_id: str | None = None,
    ) -> WorkflowRecord:
        """Transition the workflow to success."""
        return replace(
            self,
            status=WorkflowStatus.COMPLETED_SUCCESS,
            summary=summary if summary is not None else self.summary,
            current_step_id=(
                current_step_id
                if current_step_id is not None
                else self.current_step_id
            ),
            completed_at=_now_utc(),
            updated_at=_now_utc(),
            error=None,
        )

    def with_completed_failure(
        self,
        *,
        error: str,
        summary: str | None = None,
        current_step_id: str | None = None,
    ) -> WorkflowRecord:
        """Transition the workflow to failure."""
        return replace(
            self,
            status=WorkflowStatus.COMPLETED_FAILURE,
            error=error,
            summary=summary if summary is not None else self.summary,
            current_step_id=(
                current_step_id
                if current_step_id is not None
                else self.current_step_id
            ),
            completed_at=_now_utc(),
            updated_at=_now_utc(),
        )

    def with_cancelled(
        self,
        *,
        summary: str | None = None,
        current_step_id: str | None = None,
    ) -> WorkflowRecord:
        """Transition the workflow to cancelled."""
        return replace(
            self,
            status=WorkflowStatus.CANCELLED,
            summary=summary if summary is not None else self.summary,
            current_step_id=(
                current_step_id
                if current_step_id is not None
                else self.current_step_id
            ),
            completed_at=_now_utc(),
            updated_at=_now_utc(),
        )

