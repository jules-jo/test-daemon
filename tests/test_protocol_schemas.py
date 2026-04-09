"""Tests for IPC protocol Pydantic request/response schemas.

Covers the versioned envelope wrapper, all typed payload models for
each command category, serialization round-trips, validation, the
payload_type discriminator, and security-related validators.
"""

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from jules_daemon.protocol.schemas import (
    ApprovalDecision,
    CancelRequest,
    CancelResponse,
    ConfirmPromptPayload,
    ConfirmReplyPayload,
    Envelope,
    ErrorPayload,
    HealthRequest,
    HealthResponse,
    HistoryRequest,
    HistoryResponse,
    HistoryRunSummary,
    MessageHeader,
    ProgressSnapshot,
    RunRequest,
    RunResponse,
    SSHTargetInfo,
    StatusRequest,
    StatusResponse,
    StreamChunk,
    WatchRequest,
    create_envelope,
)
from jules_daemon.protocol.types import (
    PROTOCOL_VERSION,
    MessageKind,
    StatusCode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
_MSG_ID = str(uuid.uuid4())


def _make_header(
    message_type: MessageKind = MessageKind.REQUEST,
) -> MessageHeader:
    return MessageHeader(
        protocol_version=PROTOCOL_VERSION,
        message_id=_MSG_ID,
        timestamp=_NOW,
        message_type=message_type,
    )


# ---------------------------------------------------------------------------
# MessageHeader
# ---------------------------------------------------------------------------


class TestMessageHeader:
    """MessageHeader model tests."""

    def test_all_fields_present(self) -> None:
        header = _make_header()
        assert header.protocol_version == PROTOCOL_VERSION
        assert header.message_id == _MSG_ID
        assert header.timestamp == _NOW
        assert header.message_type == MessageKind.REQUEST

    def test_timestamp_must_be_utc_aware(self) -> None:
        """Headers with naive datetimes are rejected."""
        naive = datetime(2026, 1, 1, 0, 0, 0)
        with pytest.raises(ValidationError):
            MessageHeader(
                protocol_version=PROTOCOL_VERSION,
                message_id=_MSG_ID,
                timestamp=naive,
                message_type=MessageKind.REQUEST,
            )

    def test_empty_message_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MessageHeader(
                protocol_version=PROTOCOL_VERSION,
                message_id="",
                timestamp=_NOW,
                message_type=MessageKind.REQUEST,
            )

    def test_empty_protocol_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MessageHeader(
                protocol_version="",
                message_id=_MSG_ID,
                timestamp=_NOW,
                message_type=MessageKind.REQUEST,
            )

    def test_serialization_roundtrip(self) -> None:
        header = _make_header()
        data = header.model_dump(mode="json")
        restored = MessageHeader.model_validate(data)
        assert restored == header

    def test_json_roundtrip(self) -> None:
        header = _make_header()
        json_str = header.model_dump_json()
        restored = MessageHeader.model_validate_json(json_str)
        assert restored == header

    def test_is_frozen(self) -> None:
        header = _make_header()
        with pytest.raises(ValidationError):
            header.message_id = "something-else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SSHTargetInfo
# ---------------------------------------------------------------------------


class TestSSHTargetInfo:
    """SSHTargetInfo payload model tests."""

    def test_basic_target(self) -> None:
        target = SSHTargetInfo(host="prod.example.com", user="deploy")
        assert target.host == "prod.example.com"
        assert target.user == "deploy"
        assert target.port == 22

    def test_custom_port(self) -> None:
        target = SSHTargetInfo(host="staging", user="ci", port=2222)
        assert target.port == 2222

    def test_with_key_path(self) -> None:
        target = SSHTargetInfo(
            host="staging",
            user="ci",
            key_path="/home/ci/.ssh/id_ed25519",
        )
        assert target.key_path == "/home/ci/.ssh/id_ed25519"

    def test_empty_host_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SSHTargetInfo(host="", user="deploy")

    def test_empty_user_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SSHTargetInfo(host="example.com", user="")

    def test_port_range_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            SSHTargetInfo(host="example.com", user="ci", port=0)

    def test_port_range_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            SSHTargetInfo(host="example.com", user="ci", port=70000)

    def test_valid_port_boundaries(self) -> None:
        t1 = SSHTargetInfo(host="a", user="b", port=1)
        assert t1.port == 1
        t2 = SSHTargetInfo(host="a", user="b", port=65535)
        assert t2.port == 65535

    def test_relative_key_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute path"):
            SSHTargetInfo(host="a", user="b", key_path="relative/path")

    def test_path_traversal_in_key_path_rejected(self) -> None:
        with pytest.raises(ValidationError, match="traversal"):
            SSHTargetInfo(
                host="a", user="b", key_path="/home/user/../../../etc/passwd"
            )

    def test_none_key_path_accepted(self) -> None:
        target = SSHTargetInfo(host="a", user="b", key_path=None)
        assert target.key_path is None


# ---------------------------------------------------------------------------
# RunRequest / RunResponse
# ---------------------------------------------------------------------------


class TestRunRequest:
    """RunRequest payload model tests."""

    def test_basic_request(self) -> None:
        req = RunRequest(
            natural_language_command="Run pytest on the auth module",
            ssh_target=SSHTargetInfo(host="staging", user="ci"),
        )
        assert req.natural_language_command == "Run pytest on the auth module"
        assert req.ssh_target.host == "staging"
        assert req.payload_type == "run_request"

    def test_empty_command_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunRequest(
                natural_language_command="",
                ssh_target=SSHTargetInfo(host="staging", user="ci"),
            )

    def test_serialization_roundtrip(self) -> None:
        req = RunRequest(
            natural_language_command="Run unit tests",
            ssh_target=SSHTargetInfo(
                host="prod.example.com", user="deploy", port=2222
            ),
        )
        data = req.model_dump(mode="json")
        restored = RunRequest.model_validate(data)
        assert restored == req


class TestRunResponse:
    """RunResponse payload model tests."""

    def test_basic_response(self) -> None:
        resp = RunResponse(
            run_id="abc-123",
            status_code=StatusCode.ACCEPTED,
            message="Run queued",
        )
        assert resp.run_id == "abc-123"
        assert resp.status_code == StatusCode.ACCEPTED
        assert resp.message == "Run queued"
        assert resp.queue_position is None
        assert resp.payload_type == "run_response"

    def test_with_queue_position(self) -> None:
        resp = RunResponse(
            run_id="abc-123",
            status_code=StatusCode.BUSY,
            message="Command queued",
            queue_position=2,
        )
        assert resp.queue_position == 2

    def test_empty_run_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunResponse(
                run_id="",
                status_code=StatusCode.ACCEPTED,
                message="Queued",
            )


# ---------------------------------------------------------------------------
# StatusRequest / StatusResponse
# ---------------------------------------------------------------------------


class TestStatusRequest:
    """StatusRequest payload model tests."""

    def test_default_request(self) -> None:
        req = StatusRequest()
        assert req.run_id is None
        assert req.payload_type == "status_request"

    def test_with_run_id(self) -> None:
        req = StatusRequest(run_id="abc-123")
        assert req.run_id == "abc-123"


class TestStatusResponse:
    """StatusResponse payload model tests."""

    def test_idle_status(self) -> None:
        resp = StatusResponse(
            run_id="abc-123",
            status="idle",
            status_code=StatusCode.OK,
        )
        assert resp.status == "idle"
        assert resp.progress is None
        assert resp.ssh_target is None
        assert resp.payload_type == "status_response"

    def test_running_status_with_progress(self) -> None:
        progress = ProgressSnapshot(
            percent=42.5,
            tests_passed=10,
            tests_failed=1,
            tests_skipped=2,
            tests_total=30,
        )
        resp = StatusResponse(
            run_id="abc-123",
            status="running",
            status_code=StatusCode.OK,
            progress=progress,
            ssh_target=SSHTargetInfo(host="staging", user="ci"),
        )
        assert resp.progress is not None
        assert resp.progress.percent == 42.5
        assert resp.ssh_target is not None

    def test_serialization_roundtrip(self) -> None:
        progress = ProgressSnapshot(
            percent=75.0,
            tests_passed=15,
            tests_failed=0,
            tests_skipped=0,
            tests_total=20,
            last_output_line="test_auth.py::test_login PASSED",
        )
        resp = StatusResponse(
            run_id="r-001",
            status="running",
            status_code=StatusCode.OK,
            progress=progress,
            ssh_target=SSHTargetInfo(host="prod", user="deploy", port=2222),
            natural_language_command="Run all tests",
            resolved_shell="pytest -v",
        )
        data = resp.model_dump(mode="json")
        restored = StatusResponse.model_validate(data)
        assert restored == resp

    def test_naive_started_at_rejected(self) -> None:
        naive = datetime(2026, 1, 1, 0, 0, 0)
        with pytest.raises(ValidationError):
            StatusResponse(
                run_id="abc-123",
                status="running",
                status_code=StatusCode.OK,
                started_at=naive,
            )


# ---------------------------------------------------------------------------
# ProgressSnapshot
# ---------------------------------------------------------------------------


class TestProgressSnapshot:
    """ProgressSnapshot model tests."""

    def test_defaults(self) -> None:
        snap = ProgressSnapshot()
        assert snap.percent == 0.0
        assert snap.tests_passed == 0
        assert snap.tests_total == 0
        assert snap.last_output_line is None

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProgressSnapshot(tests_passed=-1)

    def test_percent_over_100_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProgressSnapshot(percent=101.0)

    def test_percent_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ProgressSnapshot(percent=-1.0)


# ---------------------------------------------------------------------------
# WatchRequest / StreamChunk
# ---------------------------------------------------------------------------


class TestWatchRequest:
    """WatchRequest payload model tests."""

    def test_basic_watch(self) -> None:
        req = WatchRequest(run_id="abc-123")
        assert req.run_id == "abc-123"
        assert req.from_sequence is None
        assert req.payload_type == "watch_request"

    def test_with_sequence(self) -> None:
        req = WatchRequest(run_id="abc-123", from_sequence=42)
        assert req.from_sequence == 42

    def test_empty_run_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WatchRequest(run_id="")


class TestStreamChunk:
    """StreamChunk payload model tests."""

    def test_basic_chunk(self) -> None:
        chunk = StreamChunk(
            run_id="abc-123",
            sequence_number=1,
            output_line="test_login.py PASSED",
            timestamp=_NOW,
        )
        assert chunk.sequence_number == 1
        assert chunk.output_line == "test_login.py PASSED"
        assert chunk.is_terminal is False
        assert chunk.payload_type == "stream_chunk"

    def test_terminal_chunk(self) -> None:
        chunk = StreamChunk(
            run_id="abc-123",
            sequence_number=99,
            output_line="",
            timestamp=_NOW,
            is_terminal=True,
            exit_status=0,
        )
        assert chunk.is_terminal is True
        assert chunk.exit_status == 0

    def test_naive_timestamp_rejected(self) -> None:
        naive = datetime(2026, 1, 1, 0, 0, 0)
        with pytest.raises(ValidationError):
            StreamChunk(
                run_id="abc-123",
                sequence_number=1,
                output_line="test",
                timestamp=naive,
            )


# ---------------------------------------------------------------------------
# CancelRequest / CancelResponse
# ---------------------------------------------------------------------------


class TestCancelRequest:
    """CancelRequest payload model tests."""

    def test_basic_cancel(self) -> None:
        req = CancelRequest(run_id="abc-123")
        assert req.run_id == "abc-123"
        assert req.reason is None
        assert req.payload_type == "cancel_request"

    def test_with_reason(self) -> None:
        req = CancelRequest(run_id="abc-123", reason="User requested")
        assert req.reason == "User requested"

    def test_empty_run_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CancelRequest(run_id="")


class TestCancelResponse:
    """CancelResponse payload model tests."""

    def test_basic_response(self) -> None:
        resp = CancelResponse(
            run_id="abc-123",
            status_code=StatusCode.OK,
            message="Run cancelled",
            cancelled=True,
        )
        assert resp.cancelled is True
        assert resp.payload_type == "cancel_response"


# ---------------------------------------------------------------------------
# ConfirmPromptPayload / ConfirmReplyPayload
# ---------------------------------------------------------------------------


class TestConfirmPromptPayload:
    """ConfirmPromptPayload model tests."""

    def test_basic_prompt(self) -> None:
        prompt = ConfirmPromptPayload(
            run_id="abc-123",
            natural_language_command="Run pytest on staging",
            resolved_shell="cd /app && pytest -v --tb=short",
            ssh_target=SSHTargetInfo(host="staging", user="ci"),
        )
        assert prompt.resolved_shell == "cd /app && pytest -v --tb=short"
        assert prompt.ssh_target.host == "staging"
        assert prompt.payload_type == "confirm_prompt"

    def test_empty_shell_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ConfirmPromptPayload(
                run_id="abc-123",
                natural_language_command="Run pytest",
                resolved_shell="",
                ssh_target=SSHTargetInfo(host="staging", user="ci"),
            )


class TestApprovalDecision:
    """ApprovalDecision enum tests."""

    def test_allow_value(self) -> None:
        assert ApprovalDecision.ALLOW.value == "allow"

    def test_deny_value(self) -> None:
        assert ApprovalDecision.DENY.value == "deny"


class TestConfirmReplyPayload:
    """ConfirmReplyPayload model tests."""

    def test_allow_reply(self) -> None:
        reply = ConfirmReplyPayload(
            run_id="abc-123",
            decision=ApprovalDecision.ALLOW,
        )
        assert reply.decision == ApprovalDecision.ALLOW
        assert reply.edited_command is None
        assert reply.payload_type == "confirm_reply"

    def test_deny_reply(self) -> None:
        reply = ConfirmReplyPayload(
            run_id="abc-123",
            decision=ApprovalDecision.DENY,
            reason="Suspicious command",
        )
        assert reply.decision == ApprovalDecision.DENY
        assert reply.reason == "Suspicious command"

    def test_allow_with_edited_command(self) -> None:
        reply = ConfirmReplyPayload(
            run_id="abc-123",
            decision=ApprovalDecision.ALLOW,
            edited_command="pytest -v --tb=long",
        )
        assert reply.edited_command == "pytest -v --tb=long"

    def test_empty_edited_command_rejected(self) -> None:
        """Empty edited_command is rejected by min_length=1."""
        with pytest.raises(ValidationError):
            ConfirmReplyPayload(
                run_id="abc-123",
                decision=ApprovalDecision.ALLOW,
                edited_command="",
            )

    def test_whitespace_only_edited_command_rejected_on_allow(self) -> None:
        """Whitespace-only edited_command with ALLOW is rejected by model validator."""
        with pytest.raises(ValidationError, match="must not be blank"):
            ConfirmReplyPayload(
                run_id="abc-123",
                decision=ApprovalDecision.ALLOW,
                edited_command="   ",
            )


# ---------------------------------------------------------------------------
# HealthRequest / HealthResponse
# ---------------------------------------------------------------------------


class TestHealthRequest:
    """HealthRequest payload model tests."""

    def test_health_request_has_payload_type(self) -> None:
        req = HealthRequest()
        assert req.payload_type == "health_request"

    def test_health_request_dump(self) -> None:
        req = HealthRequest()
        data = req.model_dump()
        assert data["payload_type"] == "health_request"


class TestHealthResponse:
    """HealthResponse payload model tests."""

    def test_basic_health(self) -> None:
        resp = HealthResponse(
            status_code=StatusCode.OK,
            daemon_uptime_seconds=3600.0,
            active_run_id=None,
            wiki_root="/workspaces/jules/wiki",
        )
        assert resp.daemon_uptime_seconds == 3600.0
        assert resp.active_run_id is None
        assert resp.payload_type == "health_response"

    def test_with_active_run(self) -> None:
        resp = HealthResponse(
            status_code=StatusCode.OK,
            daemon_uptime_seconds=120.0,
            active_run_id="abc-123",
            wiki_root="/workspaces/jules/wiki",
        )
        assert resp.active_run_id == "abc-123"

    def test_wiki_root_defaults_to_none(self) -> None:
        resp = HealthResponse(
            status_code=StatusCode.OK,
            daemon_uptime_seconds=0.0,
        )
        assert resp.wiki_root is None

    def test_negative_queue_depth_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HealthResponse(
                status_code=StatusCode.OK,
                daemon_uptime_seconds=0.0,
                queue_depth=-1,
            )


# ---------------------------------------------------------------------------
# HistoryRequest / HistoryResponse
# ---------------------------------------------------------------------------


class TestHistoryRequest:
    """HistoryRequest payload model tests."""

    def test_default_request(self) -> None:
        req = HistoryRequest()
        assert req.limit == 20
        assert req.offset == 0
        assert req.payload_type == "history_request"

    def test_custom_limits(self) -> None:
        req = HistoryRequest(limit=5, offset=10)
        assert req.limit == 5
        assert req.offset == 10

    def test_zero_limit_rejected(self) -> None:
        """Limit of 0 is rejected; minimum is 1."""
        with pytest.raises(ValidationError):
            HistoryRequest(limit=0)

    def test_negative_limit_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HistoryRequest(limit=-1)

    def test_negative_offset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HistoryRequest(offset=-1)


class TestHistoryRunSummary:
    """HistoryRunSummary model tests."""

    def test_basic_summary(self) -> None:
        summary = HistoryRunSummary(
            run_id="r-001",
            status="completed",
            natural_language_command="Run all tests",
            started_at=_NOW,
            completed_at=_NOW,
            tests_passed=42,
            tests_failed=0,
            tests_total=42,
        )
        assert summary.run_id == "r-001"
        assert summary.tests_passed == 42

    def test_failed_run_summary(self) -> None:
        summary = HistoryRunSummary(
            run_id="r-002",
            status="failed",
            natural_language_command="Run integration tests",
            started_at=_NOW,
            error="SSH connection refused",
        )
        assert summary.error == "SSH connection refused"

    def test_negative_tests_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HistoryRunSummary(
                run_id="r-001",
                status="completed",
                natural_language_command="Test",
                tests_passed=-1,
            )

    def test_naive_datetime_rejected(self) -> None:
        naive = datetime(2026, 1, 1, 0, 0, 0)
        with pytest.raises(ValidationError):
            HistoryRunSummary(
                run_id="r-001",
                status="completed",
                natural_language_command="Test",
                started_at=naive,
            )

    def test_empty_run_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HistoryRunSummary(
                run_id="",
                status="completed",
                natural_language_command="Test",
            )


class TestHistoryResponse:
    """HistoryResponse payload model tests."""

    def test_empty_history(self) -> None:
        resp = HistoryResponse(
            status_code=StatusCode.OK,
            runs=[],
            total=0,
        )
        assert len(resp.runs) == 0
        assert resp.payload_type == "history_response"

    def test_with_runs(self) -> None:
        runs = [
            HistoryRunSummary(
                run_id="r-001",
                status="completed",
                natural_language_command="Run tests",
                started_at=_NOW,
            ),
        ]
        resp = HistoryResponse(
            status_code=StatusCode.OK,
            runs=runs,
            total=1,
        )
        assert len(resp.runs) == 1
        assert resp.runs[0].run_id == "r-001"


# ---------------------------------------------------------------------------
# ErrorPayload
# ---------------------------------------------------------------------------


class TestErrorPayload:
    """ErrorPayload model tests."""

    def test_basic_error(self) -> None:
        err = ErrorPayload(
            status_code=StatusCode.INTERNAL_ERROR,
            error="Something went wrong",
        )
        assert err.error == "Something went wrong"
        assert err.details is None
        assert err.payload_type == "error"

    def test_with_details(self) -> None:
        err = ErrorPayload(
            status_code=StatusCode.BAD_REQUEST,
            error="Invalid SSH target",
            details={"field": "port", "reason": "must be 1-65535"},
        )
        assert err.details is not None
        assert err.details["field"] == "port"

    def test_empty_error_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ErrorPayload(
                status_code=StatusCode.INTERNAL_ERROR,
                error="",
            )


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


class TestEnvelope:
    """Envelope wrapper model tests."""

    def test_basic_envelope(self) -> None:
        payload = StatusRequest()
        env = Envelope(
            header=_make_header(),
            payload=payload,
        )
        assert env.header.message_type == MessageKind.REQUEST
        assert isinstance(env.payload, StatusRequest)

    def test_envelope_with_run_request(self) -> None:
        payload = RunRequest(
            natural_language_command="Run pytest",
            ssh_target=SSHTargetInfo(host="staging", user="ci"),
        )
        env = Envelope(
            header=_make_header(MessageKind.REQUEST),
            payload=payload,
        )
        assert isinstance(env.payload, RunRequest)

    def test_envelope_json_roundtrip(self) -> None:
        payload = RunRequest(
            natural_language_command="Run pytest",
            ssh_target=SSHTargetInfo(host="staging", user="ci"),
        )
        env = Envelope(
            header=_make_header(MessageKind.REQUEST),
            payload=payload,
        )
        json_str = env.model_dump_json()
        restored = Envelope.model_validate_json(json_str)
        assert restored.header == env.header
        assert isinstance(restored.payload, RunRequest)
        assert restored.payload.natural_language_command == "Run pytest"

    def test_envelope_is_frozen(self) -> None:
        payload = StatusRequest()
        env = Envelope(
            header=_make_header(),
            payload=payload,
        )
        with pytest.raises(ValidationError):
            env.header = _make_header()  # type: ignore[misc]

    def test_envelope_discriminates_payload_types(self) -> None:
        """Envelope correctly deserializes different payload types via discriminator."""
        for payload_cls, expected_type in [
            (
                RunRequest(
                    natural_language_command="Run tests",
                    ssh_target=SSHTargetInfo(host="h", user="u"),
                ),
                RunRequest,
            ),
            (StatusRequest(), StatusRequest),
            (HealthRequest(), HealthRequest),
            (
                ErrorPayload(
                    status_code=StatusCode.INTERNAL_ERROR,
                    error="fail",
                ),
                ErrorPayload,
            ),
        ]:
            env = Envelope(
                header=_make_header(),
                payload=payload_cls,
            )
            json_str = env.model_dump_json()
            restored = Envelope.model_validate_json(json_str)
            assert isinstance(restored.payload, expected_type)


# ---------------------------------------------------------------------------
# create_envelope factory
# ---------------------------------------------------------------------------


class TestCreateEnvelope:
    """create_envelope() factory function tests."""

    def test_creates_valid_envelope(self) -> None:
        payload = StatusRequest()
        env = create_envelope(
            message_type=MessageKind.REQUEST,
            payload=payload,
        )
        assert env.header.protocol_version == PROTOCOL_VERSION
        assert env.header.message_type == MessageKind.REQUEST
        assert isinstance(env.payload, StatusRequest)
        # message_id should be a valid UUID
        uuid.UUID(env.header.message_id)

    def test_timestamp_is_utc(self) -> None:
        payload = HealthRequest()
        env = create_envelope(
            message_type=MessageKind.REQUEST,
            payload=payload,
        )
        assert env.header.timestamp.tzinfo is not None

    def test_unique_message_ids(self) -> None:
        ids = set()
        for _ in range(10):
            env = create_envelope(
                message_type=MessageKind.REQUEST,
                payload=HealthRequest(),
            )
            ids.add(env.header.message_id)
        assert len(ids) == 10

    def test_with_custom_message_id(self) -> None:
        custom_id = "custom-msg-001"
        env = create_envelope(
            message_type=MessageKind.RESPONSE,
            payload=HealthResponse(
                status_code=StatusCode.OK,
                daemon_uptime_seconds=0.0,
                active_run_id=None,
                wiki_root="/wiki",
            ),
            message_id=custom_id,
        )
        assert env.header.message_id == custom_id

    def test_with_error_payload(self) -> None:
        env = create_envelope(
            message_type=MessageKind.ERROR,
            payload=ErrorPayload(
                status_code=StatusCode.INTERNAL_ERROR,
                error="Something broke",
            ),
        )
        assert env.header.message_type == MessageKind.ERROR
        assert isinstance(env.payload, ErrorPayload)

    def test_with_confirm_prompt(self) -> None:
        env = create_envelope(
            message_type=MessageKind.CONFIRM_PROMPT,
            payload=ConfirmPromptPayload(
                run_id="r-001",
                natural_language_command="Run tests",
                resolved_shell="pytest -v",
                ssh_target=SSHTargetInfo(host="staging", user="ci"),
            ),
        )
        assert env.header.message_type == MessageKind.CONFIRM_PROMPT

    def test_with_stream_chunk(self) -> None:
        env = create_envelope(
            message_type=MessageKind.STREAM,
            payload=StreamChunk(
                run_id="r-001",
                sequence_number=5,
                output_line="PASSED test_auth.py",
                timestamp=_NOW,
            ),
        )
        assert env.header.message_type == MessageKind.STREAM
