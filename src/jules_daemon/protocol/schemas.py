"""Pydantic request/response schemas for the Jules IPC protocol.

Defines the versioned envelope wrapper and typed payload models for
every command category exchanged between the CLI (thin IPC client)
and the daemon (owner of LLM calls and execution).

Architecture:
    Every IPC message is wrapped in an ``Envelope`` that carries a
    ``MessageHeader`` (protocol_version, message_id, timestamp,
    message_type) and a typed ``payload``.

    Payload models are grouped by command category:
        - Run:     Submit and acknowledge test execution requests
        - Status:  Query current run state
        - Watch:   Stream live output from running tests
        - Cancel:  Cancel an active run
        - Confirm: Security approval flow for SSH commands
        - Health:  Daemon liveness and metadata
        - History: Past run summaries
        - Error:   Structured error responses

    Each payload model carries a ``payload_type`` literal discriminator
    field that enables Pydantic to unambiguously deserialize the correct
    type from a union.

All models are frozen (immutable) to prevent accidental mutation.
Validators enforce domain constraints (non-empty strings, port ranges,
percentage bounds, non-negative counts, timezone-aware datetimes).

Usage::

    from jules_daemon.protocol.schemas import (
        Envelope,
        RunRequest,
        SSHTargetInfo,
        create_envelope,
    )
    from jules_daemon.protocol.types import MessageKind

    payload = RunRequest(
        natural_language_command="Run pytest on the auth module",
        ssh_target=SSHTargetInfo(host="staging.example.com", user="ci"),
    )
    envelope = create_envelope(
        message_type=MessageKind.REQUEST,
        payload=payload,
    )
    wire_json = envelope.model_dump_json()
"""

import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from jules_daemon.protocol.types import (
    PROTOCOL_VERSION,
    MessageKind,
    StatusCode,
)

__all__ = [
    "ApprovalDecision",
    "CancelRequest",
    "CancelResponse",
    "ConfirmPromptPayload",
    "ConfirmReplyPayload",
    "Envelope",
    "ErrorPayload",
    "HealthRequest",
    "HealthResponse",
    "HistoryRequest",
    "HistoryResponse",
    "HistoryRunSummary",
    "MessageHeader",
    "PayloadType",
    "ProgressSnapshot",
    "RunRequest",
    "RunResponse",
    "SSHTargetInfo",
    "StatusRequest",
    "StatusResponse",
    "StreamChunk",
    "WatchRequest",
    "create_envelope",
]


# ---------------------------------------------------------------------------
# Base configuration -- internal, not exported
# ---------------------------------------------------------------------------


class _FrozenModel(BaseModel):
    """Base for all protocol models -- frozen to enforce immutability.

    All datetime fields in derived models use ``AwareDatetime`` to
    enforce timezone-awareness at the type level, preventing naive
    datetimes from entering the protocol boundary.
    """

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Message header (envelope metadata)
# ---------------------------------------------------------------------------


class MessageHeader(_FrozenModel):
    """Versioned header for every IPC message.

    Fields:
        protocol_version: Semver string identifying the protocol revision.
        message_id: Unique identifier for this message (typically a UUID).
        timestamp: UTC-aware datetime when the message was created.
        message_type: The kind of message (request, response, etc.).
    """

    protocol_version: str = Field(..., min_length=1)
    message_id: str = Field(..., min_length=1)
    timestamp: AwareDatetime
    message_type: MessageKind


# ---------------------------------------------------------------------------
# SSH target (reusable across payloads)
# ---------------------------------------------------------------------------


class SSHTargetInfo(_FrozenModel):
    """SSH connection target details carried in request/response payloads.

    Fields:
        host: Remote hostname or IP address.
        user: SSH username.
        port: TCP port (1-65535, default 22).
        key_path: Optional absolute path to the SSH private key.
            Must be an absolute path with no path traversal components.
    """

    host: str = Field(..., min_length=1)
    user: str = Field(..., min_length=1)
    port: int = Field(default=22, ge=1, le=65535)
    key_path: str | None = None

    @field_validator("key_path")
    @classmethod
    def _key_path_must_be_safe_absolute(cls, v: str | None) -> str | None:
        """Validate that key_path is an absolute path without traversal.

        Checks the raw input for ``..`` components before normalizing,
        to reject paths that attempt traversal even if they would
        resolve to a valid absolute path.
        """
        if v is None:
            return v
        if not os.path.isabs(v):
            raise ValueError("key_path must be an absolute path")
        if ".." in v.split("/"):
            raise ValueError(
                "key_path must not contain path traversal components"
            )
        return os.path.normpath(v)


# ---------------------------------------------------------------------------
# Progress snapshot (reusable)
# ---------------------------------------------------------------------------


class ProgressSnapshot(_FrozenModel):
    """Test execution progress at a point in time.

    Fields:
        percent: Completion percentage (0.0 to 100.0).
        tests_passed: Number of tests that passed.
        tests_failed: Number of tests that failed.
        tests_skipped: Number of tests skipped.
        tests_total: Total number of tests discovered.
        last_output_line: Most recent stdout/stderr line (optional).
    """

    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    tests_passed: int = Field(default=0, ge=0)
    tests_failed: int = Field(default=0, ge=0)
    tests_skipped: int = Field(default=0, ge=0)
    tests_total: int = Field(default=0, ge=0)
    last_output_line: str | None = None


# ---------------------------------------------------------------------------
# Run: submit and acknowledge test execution
# ---------------------------------------------------------------------------


class RunRequest(_FrozenModel):
    """CLI -> Daemon: Submit a natural-language test command.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        natural_language_command: The user's intent in plain English.
        ssh_target: Remote system to execute on.
    """

    payload_type: Literal["run_request"] = "run_request"
    natural_language_command: str = Field(..., min_length=1)
    ssh_target: SSHTargetInfo


class RunResponse(_FrozenModel):
    """Daemon -> CLI: Acknowledge a submitted run request.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        run_id: Unique identifier assigned to this run.
        status_code: Outcome code (ACCEPTED, BUSY, etc.).
        message: Human-readable status message.
        queue_position: Position in the command queue (if queued).
    """

    payload_type: Literal["run_response"] = "run_response"
    run_id: str = Field(..., min_length=1)
    status_code: StatusCode
    message: str
    queue_position: int | None = None


# ---------------------------------------------------------------------------
# Status: query current run state
# ---------------------------------------------------------------------------


class StatusRequest(_FrozenModel):
    """CLI -> Daemon: Request current run status.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        run_id: Optional specific run to query. If None, returns current.
    """

    payload_type: Literal["status_request"] = "status_request"
    run_id: str | None = None


class StatusResponse(_FrozenModel):
    """Daemon -> CLI: Current run state and progress.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        run_id: Identifier of the run being reported.
        status: Lifecycle state string (idle, running, completed, etc.).
        status_code: Outcome code for this response.
        progress: Test execution progress (if running).
        ssh_target: Remote target details (if active).
        natural_language_command: Original user command (if active).
        resolved_shell: Resolved shell command (if approved).
        error: Error message (if failed).
        started_at: When execution began (UTC).
        completed_at: When execution ended (UTC).
    """

    payload_type: Literal["status_response"] = "status_response"
    run_id: str = Field(..., min_length=1)
    status: str
    status_code: StatusCode
    progress: ProgressSnapshot | None = None
    ssh_target: SSHTargetInfo | None = None
    natural_language_command: str | None = None
    resolved_shell: str | None = None
    error: str | None = None
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None


# ---------------------------------------------------------------------------
# Watch: live stream output
# ---------------------------------------------------------------------------


class WatchRequest(_FrozenModel):
    """CLI -> Daemon: Subscribe to live output for a run.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        run_id: The run to watch.
        from_sequence: Resume from this sequence number (for reconnect).
    """

    payload_type: Literal["watch_request"] = "watch_request"
    run_id: str = Field(..., min_length=1)
    from_sequence: int | None = None


class StreamChunk(_FrozenModel):
    """Daemon -> CLI: A single chunk of streaming output.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        run_id: The run this chunk belongs to.
        sequence_number: Monotonically increasing sequence per run.
        output_line: The stdout/stderr content.
        timestamp: When the output was captured (UTC).
        is_terminal: True if this is the final chunk.
        exit_status: Remote process exit code (only on terminal chunk).
    """

    payload_type: Literal["stream_chunk"] = "stream_chunk"
    run_id: str = Field(..., min_length=1)
    sequence_number: int
    output_line: str
    timestamp: AwareDatetime
    is_terminal: bool = False
    exit_status: int | None = None


# ---------------------------------------------------------------------------
# Cancel: stop an active run
# ---------------------------------------------------------------------------


class CancelRequest(_FrozenModel):
    """CLI -> Daemon: Cancel an active run.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        run_id: The run to cancel.
        reason: Optional human-readable cancellation reason.
    """

    payload_type: Literal["cancel_request"] = "cancel_request"
    run_id: str = Field(..., min_length=1)
    reason: str | None = None


class CancelResponse(_FrozenModel):
    """Daemon -> CLI: Acknowledge a cancellation request.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        run_id: The run that was targeted.
        status_code: Outcome code.
        message: Human-readable result.
        cancelled: Whether the run was actually cancelled.
    """

    payload_type: Literal["cancel_response"] = "cancel_response"
    run_id: str = Field(..., min_length=1)
    status_code: StatusCode
    message: str
    cancelled: bool


# ---------------------------------------------------------------------------
# Confirm: security approval flow for SSH commands
# ---------------------------------------------------------------------------


class ApprovalDecision(str, Enum):
    """User's decision on an SSH command confirmation prompt.

    Values:
        ALLOW: User approves the command for execution.
        DENY:  User rejects the command.
    """

    ALLOW = "allow"
    DENY = "deny"


class ConfirmPromptPayload(_FrozenModel):
    """Daemon -> CLI: Present an SSH command for user approval.

    Every SSH command requires explicit human approval before execution.
    The daemon sends this payload with the LLM-resolved shell command
    for the user to review, edit, or deny.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        run_id: The run this confirmation belongs to.
        natural_language_command: The original user intent.
        resolved_shell: The shell command the LLM generated.
        ssh_target: The remote system the command will run on.
        llm_explanation: Optional LLM explanation of what the command does.
    """

    payload_type: Literal["confirm_prompt"] = "confirm_prompt"
    run_id: str = Field(..., min_length=1)
    natural_language_command: str
    resolved_shell: str = Field(..., min_length=1)
    ssh_target: SSHTargetInfo
    llm_explanation: str | None = None


class ConfirmReplyPayload(_FrozenModel):
    """CLI -> Daemon: User's approval or denial of an SSH command.

    When decision is ALLOW, the edited_command (if provided) must be
    a non-empty, non-whitespace string. This is enforced by a model
    validator to prevent empty commands from reaching execution.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        run_id: The run this reply corresponds to.
        decision: ALLOW or DENY.
        edited_command: If the user edited the command before approving.
            Must be non-blank when provided with ALLOW decision.
        reason: Optional reason (especially useful for denials).
    """

    payload_type: Literal["confirm_reply"] = "confirm_reply"
    run_id: str = Field(..., min_length=1)
    decision: ApprovalDecision
    edited_command: str | None = Field(default=None, min_length=1)
    reason: str | None = None

    @model_validator(mode="after")
    def _edited_command_not_blank_on_allow(self) -> "ConfirmReplyPayload":
        """Ensure edited_command is not whitespace-only when ALLOW."""
        if (
            self.decision == ApprovalDecision.ALLOW
            and self.edited_command is not None
            and not self.edited_command.strip()
        ):
            raise ValueError(
                "edited_command must not be blank when decision is ALLOW"
            )
        return self


# ---------------------------------------------------------------------------
# Health: daemon liveness
# ---------------------------------------------------------------------------


class HealthRequest(_FrozenModel):
    """CLI -> Daemon: Check daemon health. No fields required."""

    payload_type: Literal["health_request"] = "health_request"


class HealthResponse(_FrozenModel):
    """Daemon -> CLI: Daemon health and metadata.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        status_code: Health check outcome.
        daemon_uptime_seconds: How long the daemon has been running.
        active_run_id: Current active run (None if idle).
        wiki_root: Path to the wiki root directory (None if not set).
        protocol_version: Protocol version the daemon supports.
        queue_depth: Number of commands waiting in the queue.
    """

    payload_type: Literal["health_response"] = "health_response"
    status_code: StatusCode
    daemon_uptime_seconds: float
    active_run_id: str | None = None
    wiki_root: str | None = None
    protocol_version: str = PROTOCOL_VERSION
    queue_depth: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# History: past run summaries
# ---------------------------------------------------------------------------


class HistoryRequest(_FrozenModel):
    """CLI -> Daemon: Request past run summaries.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        limit: Maximum number of runs to return (minimum 1).
        offset: Number of runs to skip (for pagination).
        status_filter: Optional filter by run status.
    """

    payload_type: Literal["history_request"] = "history_request"
    limit: int = Field(default=20, ge=1)
    offset: int = Field(default=0, ge=0)
    status_filter: str | None = None


class HistoryRunSummary(_FrozenModel):
    """Summary of a single past run for history listings.

    Fields:
        run_id: Unique run identifier.
        status: Terminal lifecycle state.
        natural_language_command: The original user command.
        started_at: When execution began (None if never started, UTC).
        completed_at: When execution ended (None if not yet done, UTC).
        tests_passed: Final count of passed tests.
        tests_failed: Final count of failed tests.
        tests_total: Final total test count.
        error: Error message if the run failed.
    """

    run_id: str = Field(..., min_length=1)
    status: str
    natural_language_command: str
    started_at: AwareDatetime | None = None
    completed_at: AwareDatetime | None = None
    tests_passed: int = Field(default=0, ge=0)
    tests_failed: int = Field(default=0, ge=0)
    tests_total: int = Field(default=0, ge=0)
    error: str | None = None


class HistoryResponse(_FrozenModel):
    """Daemon -> CLI: List of past run summaries.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        status_code: Outcome code.
        runs: List of run summaries.
        total: Total number of runs matching the query (for pagination).
    """

    payload_type: Literal["history_response"] = "history_response"
    status_code: StatusCode
    runs: list[HistoryRunSummary]
    total: int


# ---------------------------------------------------------------------------
# Error: structured error payload
# ---------------------------------------------------------------------------


class ErrorPayload(_FrozenModel):
    """Daemon -> CLI: Structured error information.

    Fields:
        payload_type: Discriminator literal for union deserialization.
        status_code: Error category code.
        error: Human-readable error message.
        details: Optional machine-readable error details.
            Values must be JSON-serializable primitives.
        run_id: Associated run ID (if applicable).
    """

    payload_type: Literal["error"] = "error"
    status_code: StatusCode
    error: str = Field(..., min_length=1)
    details: dict[str, Any] | None = None
    run_id: str | None = None


# ---------------------------------------------------------------------------
# Envelope: versioned wrapper with discriminated union
# ---------------------------------------------------------------------------

# Discriminated union of all possible payload types keyed on payload_type.
PayloadType = Annotated[
    Union[
        RunRequest,
        RunResponse,
        StatusRequest,
        StatusResponse,
        WatchRequest,
        StreamChunk,
        CancelRequest,
        CancelResponse,
        ConfirmPromptPayload,
        ConfirmReplyPayload,
        HealthRequest,
        HealthResponse,
        HistoryRequest,
        HistoryResponse,
        ErrorPayload,
    ],
    Field(discriminator="payload_type"),
]


class Envelope(_FrozenModel):
    """Versioned envelope wrapping every IPC message.

    Every message exchanged between CLI and daemon is wrapped in an
    Envelope that carries a ``MessageHeader`` (with protocol version,
    unique message ID, timestamp, and message type) and a typed payload.

    The payload is a discriminated union keyed on the ``payload_type``
    field, which enables unambiguous deserialization from JSON.

    Fields:
        header: Message metadata and routing information.
        payload: The typed message content (discriminated by payload_type).
    """

    header: MessageHeader
    payload: PayloadType


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_envelope(
    *,
    message_type: MessageKind,
    payload: PayloadType,
    message_id: str | None = None,
) -> Envelope:
    """Create an Envelope with auto-populated header fields.

    Generates a UUID message_id and UTC timestamp automatically.

    Args:
        message_type: The kind of message being sent.
        payload: The typed payload model.
        message_id: Optional override for the message ID.

    Returns:
        A fully populated, immutable Envelope.
    """
    header = MessageHeader(
        protocol_version=PROTOCOL_VERSION,
        message_id=message_id or str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        message_type=message_type,
    )
    return Envelope(header=header, payload=payload)
