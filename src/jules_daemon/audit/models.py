"""Immutable audit record data model for the command execution pipeline.

Captures every pipeline stage -- NL input, parsed command, confirmation
decision, SSH execution details, and structured result -- linked by a
correlation ID that ties the full chain together.

Each command execution event produces exactly one AuditRecord. The record
is built up immutably as each pipeline stage completes: a new frozen
instance is returned by each ``with_*`` transition method. The correlation
ID remains constant through all transitions, providing the chain link.

Pipeline stages (in order):
    NL_INPUT           -- User submits a natural-language command
    COMMAND_PARSED     -- LLM translates NL to a shell command
    CONFIRMATION       -- Human approves, denies, or edits the command
    SSH_DISPATCHED     -- Approved command sent to remote host
    EXECUTION_COMPLETE -- Execution finished, structured results captured

Wiki persistence: One audit file per command execution event, stored at
``pages/daemon/audit/audit-{correlation_id}.md``. The AuditRecord is
serialized to YAML frontmatter + markdown body via the wiki layer.

Usage::

    from jules_daemon.audit.models import (
        AuditRecord,
        NLInputRecord,
        PipelineStage,
    )

    nl = NLInputRecord(
        raw_input="run the full test suite",
        timestamp=datetime.now(timezone.utc),
        source="cli",
    )
    record = AuditRecord.create(run_id="run-001", nl_input=nl)
    record = record.with_parsed_command(parsed)
    record = record.with_confirmation(confirmation)
    record = record.with_ssh_execution(ssh)
    record = record.with_structured_result(result)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

__all__ = [
    "AuditRecord",
    "ConfirmationDecision",
    "ConfirmationRecord",
    "NLInputRecord",
    "ParsedCommandRecord",
    "PipelineStage",
    "SSHExecutionRecord",
    "StructuredResultRecord",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

_VALID_RISK_LEVELS = frozenset({"low", "medium", "high", "critical"})


class PipelineStage(Enum):
    """Ordered stages of the command execution pipeline.

    Each stage represents a distinct phase that produces audit-worthy
    data. The stages proceed in order; a record's current stage
    indicates how far the pipeline has progressed.
    """

    NL_INPUT = "nl_input"
    COMMAND_PARSED = "command_parsed"
    CONFIRMATION = "confirmation"
    SSH_DISPATCHED = "ssh_dispatched"
    EXECUTION_COMPLETE = "execution_complete"


class ConfirmationDecision(Enum):
    """Human decision on a proposed SSH command.

    APPROVED: Command accepted as-is.
    DENIED: Command rejected; pipeline stops.
    EDITED: Command modified by human before approval.
    """

    APPROVED = "approved"
    DENIED = "denied"
    EDITED = "edited"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_non_empty(value: str, field_name: str) -> str:
    """Strip and validate a string is not empty."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


# ---------------------------------------------------------------------------
# Stage sub-models (all frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NLInputRecord:
    """Natural-language input stage of the audit chain.

    Captures the raw user input, when it was submitted, and the
    source channel (e.g., "cli", "ipc").
    """

    raw_input: str
    timestamp: datetime
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "raw_input", _require_non_empty(self.raw_input, "raw_input")
        )
        _require_non_empty(self.source, "source")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki YAML persistence."""
        return {
            "raw_input": self.raw_input,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NLInputRecord:
        """Deserialize from a plain dict."""
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            raw_input=data["raw_input"],
            timestamp=ts,
            source=data["source"],
        )


@dataclass(frozen=True)
class ParsedCommandRecord:
    """LLM command translation stage of the audit chain.

    Records what the LLM produced: the resolved shell command, risk
    classification, explanation, and which model was used.
    """

    natural_language: str
    resolved_shell: str
    model_id: str
    risk_level: str
    explanation: str
    affected_paths: tuple[str, ...]
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_non_empty(self.natural_language, "natural_language")
        _require_non_empty(self.resolved_shell, "resolved_shell")
        _require_non_empty(self.model_id, "model_id")
        if self.risk_level not in _VALID_RISK_LEVELS:
            raise ValueError(
                f"risk_level must be one of {sorted(_VALID_RISK_LEVELS)}, "
                f"got {self.risk_level!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki YAML persistence."""
        return {
            "natural_language": self.natural_language,
            "resolved_shell": self.resolved_shell,
            "model_id": self.model_id,
            "risk_level": self.risk_level,
            "explanation": self.explanation,
            "affected_paths": list(self.affected_paths),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedCommandRecord:
        """Deserialize from a plain dict."""
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        paths = data.get("affected_paths", ())
        if isinstance(paths, list):
            paths = tuple(paths)
        return cls(
            natural_language=data["natural_language"],
            resolved_shell=data["resolved_shell"],
            model_id=data["model_id"],
            risk_level=data["risk_level"],
            explanation=data["explanation"],
            affected_paths=paths,
            timestamp=ts,
        )


@dataclass(frozen=True)
class ConfirmationRecord:
    """Human confirmation decision stage of the audit chain.

    Records whether the human approved, denied, or edited the command,
    along with both the original and final command strings.
    """

    decision: ConfirmationDecision
    original_command: str
    final_command: str
    decided_by: str
    timestamp: datetime

    def __post_init__(self) -> None:
        _require_non_empty(self.original_command, "original_command")
        _require_non_empty(self.decided_by, "decided_by")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki YAML persistence."""
        return {
            "decision": self.decision.value,
            "original_command": self.original_command,
            "final_command": self.final_command,
            "decided_by": self.decided_by,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfirmationRecord:
        """Deserialize from a plain dict."""
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            decision=ConfirmationDecision(data["decision"]),
            original_command=data["original_command"],
            final_command=data["final_command"],
            decided_by=data["decided_by"],
            timestamp=ts,
        )


@dataclass(frozen=True)
class SSHExecutionRecord:
    """SSH execution details stage of the audit chain.

    Captures connection details, the dispatched command, remote process
    identity, and timing/exit information.
    """

    host: str
    user: str
    port: int
    command: str
    session_id: str
    started_at: datetime
    remote_pid: int | None = None
    completed_at: datetime | None = None
    exit_code: int | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.host, "host")
        _require_non_empty(self.user, "user")
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be 1-65535, got {self.port}")
        _require_non_empty(self.command, "command")
        _require_non_empty(self.session_id, "session_id")
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("duration_seconds must not be negative")

    @property
    def is_complete(self) -> bool:
        """True if the SSH command has finished execution."""
        return self.exit_code is not None

    @property
    def is_success(self) -> bool:
        """True if the SSH command exited with code 0."""
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki YAML persistence."""
        return {
            "host": self.host,
            "user": self.user,
            "port": self.port,
            "command": self.command,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "remote_pid": self.remote_pid,
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at is not None
                else None
            ),
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SSHExecutionRecord:
        """Deserialize from a plain dict."""
        started = data["started_at"]
        if isinstance(started, str):
            started = datetime.fromisoformat(started)
        completed = data.get("completed_at")
        if isinstance(completed, str):
            completed = datetime.fromisoformat(completed)
        return cls(
            host=data["host"],
            user=data["user"],
            port=data["port"],
            command=data["command"],
            session_id=data["session_id"],
            started_at=started,
            remote_pid=data.get("remote_pid"),
            completed_at=completed,
            exit_code=data.get("exit_code"),
            duration_seconds=data.get("duration_seconds"),
        )


@dataclass(frozen=True)
class StructuredResultRecord:
    """Structured test result stage of the audit chain.

    Captures the parsed test outcome: counts, exit code, success flag,
    and a human-readable summary.
    """

    tests_passed: int
    tests_failed: int
    tests_skipped: int
    tests_total: int
    exit_code: int
    success: bool
    error_message: str | None
    summary: str
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.tests_passed < 0:
            raise ValueError("tests_passed must not be negative")
        if self.tests_failed < 0:
            raise ValueError("tests_failed must not be negative")
        if self.tests_skipped < 0:
            raise ValueError("tests_skipped must not be negative")
        if self.tests_total < 0:
            raise ValueError("tests_total must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki YAML persistence."""
        return {
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "tests_total": self.tests_total,
            "exit_code": self.exit_code,
            "success": self.success,
            "error_message": self.error_message,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredResultRecord:
        """Deserialize from a plain dict."""
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            tests_passed=data["tests_passed"],
            tests_failed=data["tests_failed"],
            tests_skipped=data["tests_skipped"],
            tests_total=data["tests_total"],
            exit_code=data["exit_code"],
            success=data["success"],
            error_message=data.get("error_message"),
            summary=data["summary"],
            timestamp=ts,
        )


# ---------------------------------------------------------------------------
# Top-level audit record
# ---------------------------------------------------------------------------


def _generate_correlation_id() -> str:
    """Generate a new unique correlation identifier."""
    return str(uuid.uuid4())


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AuditRecord:
    """Full-chain audit record for a single command execution event.

    Links every pipeline stage -- from natural-language input through
    to structured test results -- via a single ``correlation_id``.
    Immutable: each stage transition returns a new instance with the
    additional data, leaving the original unchanged.

    The ``pipeline_stage`` field indicates how far the pipeline has
    progressed. Optional stage fields (``parsed_command``, etc.) are
    ``None`` until their corresponding stage completes.

    Attributes:
        correlation_id: Unique UUID linking all stages of this execution.
        run_id: Links to the daemon's CurrentRun record.
        pipeline_stage: Current (latest) stage reached.
        nl_input: NL input stage data (always present).
        parsed_command: LLM translation data (None before COMMAND_PARSED).
        confirmation: Human decision data (None before CONFIRMATION).
        ssh_execution: SSH dispatch data (None before SSH_DISPATCHED).
        structured_result: Test results (None before EXECUTION_COMPLETE).
        created_at: UTC timestamp when the record was first created.
        completed_at: UTC timestamp when the pipeline reached a terminal
            state (EXECUTION_COMPLETE or denied at CONFIRMATION).
    """

    correlation_id: str
    run_id: str
    pipeline_stage: PipelineStage
    nl_input: NLInputRecord
    parsed_command: Optional[ParsedCommandRecord] = None
    confirmation: Optional[ConfirmationRecord] = None
    ssh_execution: Optional[SSHExecutionRecord] = None
    structured_result: Optional[StructuredResultRecord] = None
    created_at: datetime = field(default_factory=_now_utc)
    completed_at: Optional[datetime] = None

    # -- Factory --

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        nl_input: NLInputRecord,
    ) -> AuditRecord:
        """Create a new AuditRecord at the NL_INPUT stage.

        Generates a fresh correlation_id and sets the pipeline stage
        to NL_INPUT.

        Args:
            run_id: Identifier linking to the daemon's CurrentRun.
            nl_input: The natural-language input record.

        Returns:
            New AuditRecord at the NL_INPUT stage.
        """
        return cls(
            correlation_id=_generate_correlation_id(),
            run_id=run_id,
            pipeline_stage=PipelineStage.NL_INPUT,
            nl_input=nl_input,
        )

    # -- Immutable stage transitions --

    def with_parsed_command(
        self, parsed_command: ParsedCommandRecord
    ) -> AuditRecord:
        """Return a new record advanced to the COMMAND_PARSED stage."""
        return replace(
            self,
            pipeline_stage=PipelineStage.COMMAND_PARSED,
            parsed_command=parsed_command,
        )

    def with_confirmation(
        self, confirmation: ConfirmationRecord
    ) -> AuditRecord:
        """Return a new record advanced to the CONFIRMATION stage.

        If the decision is DENIED, the completed_at timestamp is set
        since the pipeline will not proceed further.
        """
        completed = (
            _now_utc()
            if confirmation.decision == ConfirmationDecision.DENIED
            else self.completed_at
        )
        return replace(
            self,
            pipeline_stage=PipelineStage.CONFIRMATION,
            confirmation=confirmation,
            completed_at=completed,
        )

    def with_ssh_execution(
        self, ssh_execution: SSHExecutionRecord
    ) -> AuditRecord:
        """Return a new record advanced to the SSH_DISPATCHED stage."""
        return replace(
            self,
            pipeline_stage=PipelineStage.SSH_DISPATCHED,
            ssh_execution=ssh_execution,
        )

    def with_structured_result(
        self, structured_result: StructuredResultRecord
    ) -> AuditRecord:
        """Return a new record advanced to the EXECUTION_COMPLETE stage."""
        return replace(
            self,
            pipeline_stage=PipelineStage.EXECUTION_COMPLETE,
            structured_result=structured_result,
            completed_at=_now_utc(),
        )

    # -- Computed properties --

    @property
    def is_complete(self) -> bool:
        """True if all pipeline stages have data."""
        return self.pipeline_stage == PipelineStage.EXECUTION_COMPLETE

    @property
    def is_denied(self) -> bool:
        """True if the human denied the command at confirmation."""
        if self.confirmation is None:
            return False
        return self.confirmation.decision == ConfirmationDecision.DENIED

    # -- Serialization --

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for wiki YAML persistence.

        Enum values are serialized as strings. Datetimes are ISO 8601.
        None sub-records remain None.
        """
        return {
            "correlation_id": self.correlation_id,
            "run_id": self.run_id,
            "pipeline_stage": self.pipeline_stage.value,
            "nl_input": self.nl_input.to_dict(),
            "parsed_command": (
                self.parsed_command.to_dict()
                if self.parsed_command is not None
                else None
            ),
            "confirmation": (
                self.confirmation.to_dict()
                if self.confirmation is not None
                else None
            ),
            "ssh_execution": (
                self.ssh_execution.to_dict()
                if self.ssh_execution is not None
                else None
            ),
            "structured_result": (
                self.structured_result.to_dict()
                if self.structured_result is not None
                else None
            ),
            "created_at": self.created_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditRecord:
        """Deserialize from a plain dict (e.g., parsed from wiki YAML).

        Restores the full AuditRecord including all present sub-records.
        """
        created_at = data["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        completed_at = data.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)

        nl_input = NLInputRecord.from_dict(data["nl_input"])

        parsed_command = None
        if data.get("parsed_command") is not None:
            parsed_command = ParsedCommandRecord.from_dict(
                data["parsed_command"]
            )

        confirmation = None
        if data.get("confirmation") is not None:
            confirmation = ConfirmationRecord.from_dict(data["confirmation"])

        ssh_execution = None
        if data.get("ssh_execution") is not None:
            ssh_execution = SSHExecutionRecord.from_dict(
                data["ssh_execution"]
            )

        structured_result = None
        if data.get("structured_result") is not None:
            structured_result = StructuredResultRecord.from_dict(
                data["structured_result"]
            )

        return cls(
            correlation_id=data["correlation_id"],
            run_id=data["run_id"],
            pipeline_stage=PipelineStage(data["pipeline_stage"]),
            nl_input=nl_input,
            parsed_command=parsed_command,
            confirmation=confirmation,
            ssh_execution=ssh_execution,
            structured_result=structured_result,
            created_at=created_at,
            completed_at=completed_at,
        )
