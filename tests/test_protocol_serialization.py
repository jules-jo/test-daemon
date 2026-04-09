"""Tests for IPC protocol JSON wire-format serialization utilities.

Covers envelope framing (wrap/unwrap), length-prefixed stream framing
(encode_frame/decode_frame/FrameBuffer), round-trip serialize/deserialize
for every protocol message type, and edge cases including malformed input,
oversized messages, and encoding errors.
"""

from __future__ import annotations

import json
import struct
import uuid
from datetime import datetime, timezone

import pytest

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


def _ssh_target() -> SSHTargetInfo:
    return SSHTargetInfo(host="staging.example.com", user="ci")


# ---------------------------------------------------------------------------
# Import the module under test (deferred so missing module gives clear error)
# ---------------------------------------------------------------------------

from jules_daemon.protocol.serialization import (
    FrameBuffer,
    SerializationError,
    decode_frame,
    deserialize_envelope,
    deserialize_payload,
    encode_frame,
    serialize_envelope,
    serialize_payload,
    unwrap_payload,
    wrap_payload,
)


# ===================================================================
# serialize_envelope / deserialize_envelope
# ===================================================================


class TestSerializeEnvelope:
    """Envelope -> bytes -> Envelope round-trip via JSON wire format."""

    def test_roundtrip_run_request(self) -> None:
        payload = RunRequest(
            natural_language_command="Run pytest -v",
            ssh_target=_ssh_target(),
        )
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        assert isinstance(wire, bytes)

        restored = deserialize_envelope(wire)
        assert restored.header.message_id == envelope.header.message_id
        assert restored.header.message_type == MessageKind.REQUEST
        assert isinstance(restored.payload, RunRequest)
        assert (
            restored.payload.natural_language_command == "Run pytest -v"
        )

    def test_roundtrip_status_request(self) -> None:
        payload = StatusRequest(run_id="abc-123")
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, StatusRequest)
        assert restored.payload.run_id == "abc-123"

    def test_roundtrip_status_response_with_progress(self) -> None:
        progress = ProgressSnapshot(
            percent=55.0,
            tests_passed=11,
            tests_failed=2,
            tests_skipped=1,
            tests_total=20,
            last_output_line="test_auth.py::test_login PASSED",
        )
        payload = StatusResponse(
            run_id="r-001",
            status="running",
            status_code=StatusCode.OK,
            progress=progress,
            ssh_target=_ssh_target(),
            natural_language_command="Run auth tests",
            resolved_shell="pytest tests/auth/ -v",
            started_at=_NOW,
        )
        envelope = create_envelope(
            message_type=MessageKind.RESPONSE, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, StatusResponse)
        assert restored.payload.progress is not None
        assert restored.payload.progress.percent == 55.0
        assert restored.payload.progress.tests_passed == 11

    def test_roundtrip_watch_request(self) -> None:
        payload = WatchRequest(run_id="r-001", from_sequence=10)
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, WatchRequest)
        assert restored.payload.from_sequence == 10

    def test_roundtrip_stream_chunk(self) -> None:
        payload = StreamChunk(
            run_id="r-001",
            sequence_number=42,
            output_line="PASSED test_api.py::test_create_user",
            timestamp=_NOW,
            is_terminal=False,
        )
        envelope = create_envelope(
            message_type=MessageKind.STREAM, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, StreamChunk)
        assert restored.payload.sequence_number == 42
        assert restored.payload.output_line == "PASSED test_api.py::test_create_user"

    def test_roundtrip_cancel_request(self) -> None:
        payload = CancelRequest(run_id="r-001", reason="Taking too long")
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, CancelRequest)
        assert restored.payload.reason == "Taking too long"

    def test_roundtrip_cancel_response(self) -> None:
        payload = CancelResponse(
            run_id="r-001",
            status_code=StatusCode.OK,
            message="Run cancelled",
            cancelled=True,
        )
        envelope = create_envelope(
            message_type=MessageKind.RESPONSE, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, CancelResponse)
        assert restored.payload.cancelled is True

    def test_roundtrip_confirm_prompt(self) -> None:
        payload = ConfirmPromptPayload(
            run_id="r-001",
            natural_language_command="Run integration tests",
            resolved_shell="cd /app && pytest tests/integration/ -v",
            ssh_target=_ssh_target(),
            llm_explanation="This runs the integration test suite",
        )
        envelope = create_envelope(
            message_type=MessageKind.CONFIRM_PROMPT, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, ConfirmPromptPayload)
        assert "integration" in restored.payload.resolved_shell

    def test_roundtrip_confirm_reply_allow(self) -> None:
        payload = ConfirmReplyPayload(
            run_id="r-001",
            decision=ApprovalDecision.ALLOW,
            edited_command="pytest tests/integration/ -v --tb=short",
        )
        envelope = create_envelope(
            message_type=MessageKind.CONFIRM_REPLY, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, ConfirmReplyPayload)
        assert restored.payload.decision == ApprovalDecision.ALLOW

    def test_roundtrip_confirm_reply_deny(self) -> None:
        payload = ConfirmReplyPayload(
            run_id="r-001",
            decision=ApprovalDecision.DENY,
            reason="Command looks suspicious",
        )
        envelope = create_envelope(
            message_type=MessageKind.CONFIRM_REPLY, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, ConfirmReplyPayload)
        assert restored.payload.decision == ApprovalDecision.DENY

    def test_roundtrip_health_request(self) -> None:
        payload = HealthRequest()
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, HealthRequest)

    def test_roundtrip_health_response(self) -> None:
        payload = HealthResponse(
            status_code=StatusCode.OK,
            daemon_uptime_seconds=3600.5,
            active_run_id="r-001",
            wiki_root="/workspaces/jules/wiki",
            queue_depth=2,
        )
        envelope = create_envelope(
            message_type=MessageKind.RESPONSE, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, HealthResponse)
        assert restored.payload.daemon_uptime_seconds == 3600.5
        assert restored.payload.queue_depth == 2

    def test_roundtrip_history_request(self) -> None:
        payload = HistoryRequest(limit=5, offset=10, status_filter="completed")
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, HistoryRequest)
        assert restored.payload.limit == 5
        assert restored.payload.status_filter == "completed"

    def test_roundtrip_history_response(self) -> None:
        runs = [
            HistoryRunSummary(
                run_id="r-001",
                status="completed",
                natural_language_command="Run all tests",
                started_at=_NOW,
                completed_at=_NOW,
                tests_passed=42,
                tests_failed=0,
                tests_total=42,
            ),
            HistoryRunSummary(
                run_id="r-002",
                status="failed",
                natural_language_command="Run integration tests",
                started_at=_NOW,
                error="SSH connection refused",
            ),
        ]
        payload = HistoryResponse(
            status_code=StatusCode.OK,
            runs=runs,
            total=2,
        )
        envelope = create_envelope(
            message_type=MessageKind.RESPONSE, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, HistoryResponse)
        assert len(restored.payload.runs) == 2
        assert restored.payload.runs[0].tests_passed == 42

    def test_roundtrip_error_payload(self) -> None:
        payload = ErrorPayload(
            status_code=StatusCode.INTERNAL_ERROR,
            error="Unexpected daemon failure",
            details={"traceback": "line 42", "module": "agent"},
            run_id="r-001",
        )
        envelope = create_envelope(
            message_type=MessageKind.ERROR, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, ErrorPayload)
        assert restored.payload.details is not None
        assert restored.payload.details["module"] == "agent"

    def test_roundtrip_run_response(self) -> None:
        payload = RunResponse(
            run_id="r-003",
            status_code=StatusCode.ACCEPTED,
            message="Run accepted",
            queue_position=None,
        )
        envelope = create_envelope(
            message_type=MessageKind.RESPONSE, payload=payload
        )
        wire = serialize_envelope(envelope)
        restored = deserialize_envelope(wire)
        assert isinstance(restored.payload, RunResponse)
        assert restored.payload.run_id == "r-003"

    def test_wire_bytes_are_valid_json(self) -> None:
        """Serialized bytes are valid, parseable JSON."""
        payload = HealthRequest()
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        parsed = json.loads(wire)
        assert "header" in parsed
        assert "payload" in parsed

    def test_wire_bytes_are_utf8(self) -> None:
        """Wire format is UTF-8 encoded."""
        payload = RunRequest(
            natural_language_command="Run tests with unicode: \u00e9\u00e8\u00ea",
            ssh_target=_ssh_target(),
        )
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        text = wire.decode("utf-8")
        assert "\u00e9\u00e8\u00ea" in text

    def test_wire_bytes_end_with_newline(self) -> None:
        """Wire format uses newline delimiter for line-oriented transport."""
        payload = HealthRequest()
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        assert wire.endswith(b"\n")


# ===================================================================
# Deserialization error cases
# ===================================================================


class TestDeserializeErrors:
    """Error handling for malformed wire input."""

    def test_empty_bytes_raises(self) -> None:
        with pytest.raises(SerializationError):
            deserialize_envelope(b"")

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(SerializationError):
            deserialize_envelope(b"not valid json{{{")

    def test_valid_json_but_wrong_structure_raises(self) -> None:
        data = json.dumps({"foo": "bar"}).encode("utf-8")
        with pytest.raises(SerializationError):
            deserialize_envelope(data)

    def test_missing_header_raises(self) -> None:
        data = json.dumps({"payload": {"payload_type": "health_request"}}).encode("utf-8")
        with pytest.raises(SerializationError):
            deserialize_envelope(data)

    def test_missing_payload_raises(self) -> None:
        header_data = {
            "protocol_version": PROTOCOL_VERSION,
            "message_id": str(uuid.uuid4()),
            "timestamp": _NOW.isoformat(),
            "message_type": "request",
        }
        data = json.dumps({"header": header_data}).encode("utf-8")
        with pytest.raises(SerializationError):
            deserialize_envelope(data)

    def test_unknown_payload_type_raises(self) -> None:
        header_data = {
            "protocol_version": PROTOCOL_VERSION,
            "message_id": str(uuid.uuid4()),
            "timestamp": _NOW.isoformat(),
            "message_type": "request",
        }
        data = json.dumps({
            "header": header_data,
            "payload": {"payload_type": "unknown_type_xyz"},
        }).encode("utf-8")
        with pytest.raises(SerializationError):
            deserialize_envelope(data)

    def test_non_utf8_bytes_raises(self) -> None:
        with pytest.raises(SerializationError):
            deserialize_envelope(b"\x80\x81\x82\x83")


# ===================================================================
# wrap_payload / unwrap_payload
# ===================================================================


class TestWrapUnwrap:
    """Envelope framing: wrap a payload into wire bytes, unwrap back."""

    def test_wrap_creates_envelope_wire_bytes(self) -> None:
        payload = HealthRequest()
        wire = wrap_payload(MessageKind.REQUEST, payload)
        assert isinstance(wire, bytes)
        # Should be valid JSON with header + payload
        parsed = json.loads(wire)
        assert "header" in parsed
        assert "payload" in parsed
        assert parsed["header"]["message_type"] == "request"

    def test_wrap_with_custom_message_id(self) -> None:
        payload = HealthRequest()
        custom_id = "custom-id-001"
        wire = wrap_payload(
            MessageKind.REQUEST, payload, message_id=custom_id
        )
        parsed = json.loads(wire)
        assert parsed["header"]["message_id"] == custom_id

    def test_wrap_auto_generates_message_id(self) -> None:
        payload = HealthRequest()
        wire = wrap_payload(MessageKind.REQUEST, payload)
        parsed = json.loads(wire)
        # Should be a valid UUID
        uuid.UUID(parsed["header"]["message_id"])

    def test_wrap_auto_generates_timestamp(self) -> None:
        payload = HealthRequest()
        wire = wrap_payload(MessageKind.REQUEST, payload)
        parsed = json.loads(wire)
        assert "timestamp" in parsed["header"]
        # Should be parseable as ISO format
        datetime.fromisoformat(parsed["header"]["timestamp"])

    def test_wrap_includes_protocol_version(self) -> None:
        payload = HealthRequest()
        wire = wrap_payload(MessageKind.REQUEST, payload)
        parsed = json.loads(wire)
        assert parsed["header"]["protocol_version"] == PROTOCOL_VERSION

    def test_unwrap_returns_header_and_payload(self) -> None:
        payload = RunRequest(
            natural_language_command="Run tests",
            ssh_target=_ssh_target(),
        )
        wire = wrap_payload(MessageKind.REQUEST, payload)
        header, unwrapped = unwrap_payload(wire)
        assert isinstance(header, MessageHeader)
        assert isinstance(unwrapped, RunRequest)
        assert unwrapped.natural_language_command == "Run tests"

    def test_wrap_unwrap_roundtrip_all_message_kinds(self) -> None:
        """All MessageKind values produce valid wrapped messages."""
        payload_for_kind = {
            MessageKind.REQUEST: HealthRequest(),
            MessageKind.RESPONSE: HealthResponse(
                status_code=StatusCode.OK,
                daemon_uptime_seconds=0.0,
            ),
            MessageKind.NOTIFICATION: StatusResponse(
                run_id="r-001",
                status="running",
                status_code=StatusCode.OK,
            ),
            MessageKind.ERROR: ErrorPayload(
                status_code=StatusCode.INTERNAL_ERROR,
                error="test error",
            ),
            MessageKind.STREAM: StreamChunk(
                run_id="r-001",
                sequence_number=1,
                output_line="test",
                timestamp=_NOW,
            ),
            MessageKind.CONFIRM_PROMPT: ConfirmPromptPayload(
                run_id="r-001",
                natural_language_command="Run tests",
                resolved_shell="pytest",
                ssh_target=_ssh_target(),
            ),
            MessageKind.CONFIRM_REPLY: ConfirmReplyPayload(
                run_id="r-001",
                decision=ApprovalDecision.ALLOW,
            ),
        }
        for kind, payload in payload_for_kind.items():
            wire = wrap_payload(kind, payload)
            header, unwrapped = unwrap_payload(wire)
            assert header.message_type == kind
            assert type(unwrapped) == type(payload)

    def test_unwrap_invalid_bytes_raises(self) -> None:
        with pytest.raises(SerializationError):
            unwrap_payload(b"garbage")


# ===================================================================
# serialize_payload / deserialize_payload
# ===================================================================


class TestSerializeDeserializePayload:
    """Payload-level serialization without the envelope wrapper."""

    def test_roundtrip_run_request(self) -> None:
        payload = RunRequest(
            natural_language_command="Run smoke tests",
            ssh_target=_ssh_target(),
        )
        data = serialize_payload(payload)
        assert isinstance(data, dict)
        assert data["payload_type"] == "run_request"
        restored = deserialize_payload(data)
        assert isinstance(restored, RunRequest)
        assert restored.natural_language_command == "Run smoke tests"

    def test_roundtrip_status_response(self) -> None:
        payload = StatusResponse(
            run_id="r-001",
            status="completed",
            status_code=StatusCode.OK,
            progress=ProgressSnapshot(
                percent=100.0,
                tests_passed=20,
                tests_failed=0,
                tests_total=20,
            ),
        )
        data = serialize_payload(payload)
        restored = deserialize_payload(data)
        assert isinstance(restored, StatusResponse)
        assert restored.progress is not None
        assert restored.progress.percent == 100.0

    def test_roundtrip_error_payload(self) -> None:
        payload = ErrorPayload(
            status_code=StatusCode.BAD_REQUEST,
            error="Invalid field",
            details={"field": "port"},
        )
        data = serialize_payload(payload)
        restored = deserialize_payload(data)
        assert isinstance(restored, ErrorPayload)
        assert restored.details is not None
        assert restored.details["field"] == "port"

    def test_missing_payload_type_raises(self) -> None:
        with pytest.raises(SerializationError):
            deserialize_payload({"foo": "bar"})

    def test_unknown_payload_type_raises(self) -> None:
        with pytest.raises(SerializationError):
            deserialize_payload({"payload_type": "nonexistent"})

    def test_all_fifteen_payload_types_roundtrip(self) -> None:
        """Every payload type in the protocol can be serialized and deserialized."""
        payloads = [
            RunRequest(
                natural_language_command="Test",
                ssh_target=_ssh_target(),
            ),
            RunResponse(
                run_id="r-1",
                status_code=StatusCode.ACCEPTED,
                message="OK",
            ),
            StatusRequest(),
            StatusResponse(
                run_id="r-1",
                status="idle",
                status_code=StatusCode.OK,
            ),
            WatchRequest(run_id="r-1"),
            StreamChunk(
                run_id="r-1",
                sequence_number=1,
                output_line="test",
                timestamp=_NOW,
            ),
            CancelRequest(run_id="r-1"),
            CancelResponse(
                run_id="r-1",
                status_code=StatusCode.OK,
                message="Cancelled",
                cancelled=True,
            ),
            ConfirmPromptPayload(
                run_id="r-1",
                natural_language_command="Test",
                resolved_shell="pytest",
                ssh_target=_ssh_target(),
            ),
            ConfirmReplyPayload(
                run_id="r-1",
                decision=ApprovalDecision.ALLOW,
            ),
            HealthRequest(),
            HealthResponse(
                status_code=StatusCode.OK,
                daemon_uptime_seconds=0.0,
            ),
            HistoryRequest(),
            HistoryResponse(
                status_code=StatusCode.OK,
                runs=[],
                total=0,
            ),
            ErrorPayload(
                status_code=StatusCode.INTERNAL_ERROR,
                error="test",
            ),
        ]
        for payload in payloads:
            data = serialize_payload(payload)
            restored = deserialize_payload(data)
            assert type(restored) == type(payload), (
                f"Mismatch for {type(payload).__name__}"
            )


# ===================================================================
# encode_frame / decode_frame (length-prefixed framing)
# ===================================================================


class TestFrameEncoding:
    """Length-prefixed frame encoding/decoding for stream transport."""

    def test_encode_produces_length_prefix(self) -> None:
        data = b'{"test": true}'
        frame = encode_frame(data)
        # First 4 bytes are big-endian uint32 length
        length = struct.unpack("!I", frame[:4])[0]
        assert length == len(data)
        assert frame[4:] == data

    def test_decode_extracts_message_and_remainder(self) -> None:
        data = b'{"test": true}'
        frame = encode_frame(data)
        extra = b"leftover bytes"
        message, remainder = decode_frame(frame + extra)
        assert message == data
        assert remainder == extra

    def test_decode_exact_frame_no_remainder(self) -> None:
        data = b"hello"
        frame = encode_frame(data)
        message, remainder = decode_frame(frame)
        assert message == data
        assert remainder == b""

    def test_decode_incomplete_header_returns_none(self) -> None:
        """Less than 4 bytes of header returns None."""
        result = decode_frame(b"\x00\x00")
        assert result is None

    def test_decode_incomplete_body_returns_none(self) -> None:
        """Header present but body incomplete returns None."""
        data = b"hello world"
        frame = encode_frame(data)
        # Truncate the body
        result = decode_frame(frame[:6])
        assert result is None

    def test_encode_decode_roundtrip(self) -> None:
        """Full roundtrip for a real serialized envelope."""
        payload = HealthRequest()
        envelope = create_envelope(
            message_type=MessageKind.REQUEST, payload=payload
        )
        wire = serialize_envelope(envelope)
        frame = encode_frame(wire)
        decoded, remainder = decode_frame(frame)
        assert decoded == wire
        assert remainder == b""
        restored = deserialize_envelope(decoded)
        assert isinstance(restored.payload, HealthRequest)

    def test_multiple_frames_in_sequence(self) -> None:
        """Multiple frames concatenated can be decoded sequentially."""
        messages = [b"msg_one", b"msg_two", b"msg_three"]
        stream = b"".join(encode_frame(m) for m in messages)

        decoded_messages: list[bytes] = []
        remaining = stream
        while remaining:
            result = decode_frame(remaining)
            if result is None:
                break
            message, remaining = result
            decoded_messages.append(message)

        assert decoded_messages == messages

    def test_empty_message_encodes(self) -> None:
        frame = encode_frame(b"")
        message, remainder = decode_frame(frame)
        assert message == b""
        assert remainder == b""

    def test_encode_frame_with_large_payload(self) -> None:
        """Handles payloads up to the max frame size."""
        data = b"x" * 10_000
        frame = encode_frame(data)
        message, remainder = decode_frame(frame)
        assert message == data


# ===================================================================
# FrameBuffer (streaming frame accumulator)
# ===================================================================


class TestFrameBuffer:
    """FrameBuffer incrementally accumulates bytes and yields complete frames."""

    def test_single_complete_frame(self) -> None:
        buf = FrameBuffer()
        data = b"hello"
        frame = encode_frame(data)
        frames = buf.feed(frame)
        assert len(frames) == 1
        assert frames[0] == data

    def test_incremental_feed(self) -> None:
        """Feeding bytes one-at-a-time eventually yields the frame."""
        buf = FrameBuffer()
        data = b"hello"
        frame = encode_frame(data)

        results: list[bytes] = []
        for byte in frame:
            results.extend(buf.feed(bytes([byte])))

        assert len(results) == 1
        assert results[0] == data

    def test_multiple_frames_in_one_feed(self) -> None:
        """Feeding multiple frames at once yields all of them."""
        buf = FrameBuffer()
        messages = [b"alpha", b"beta", b"gamma"]
        blob = b"".join(encode_frame(m) for m in messages)
        frames = buf.feed(blob)
        assert frames == messages

    def test_partial_frame_buffered(self) -> None:
        """Partial frames are buffered until complete."""
        buf = FrameBuffer()
        data = b"complete message"
        frame = encode_frame(data)

        # Feed first half
        mid = len(frame) // 2
        frames1 = buf.feed(frame[:mid])
        assert frames1 == []

        # Feed second half
        frames2 = buf.feed(frame[mid:])
        assert len(frames2) == 1
        assert frames2[0] == data

    def test_empty_feed(self) -> None:
        buf = FrameBuffer()
        frames = buf.feed(b"")
        assert frames == []

    def test_buffer_pending_bytes(self) -> None:
        """pending property reports buffered byte count."""
        buf = FrameBuffer()
        assert buf.pending == 0

        data = b"test"
        frame = encode_frame(data)
        buf.feed(frame[:3])
        assert buf.pending == 3

    def test_clear_resets_buffer(self) -> None:
        """clear() discards all buffered data."""
        buf = FrameBuffer()
        buf.feed(b"partial data")
        buf.clear()
        assert buf.pending == 0

    def test_integration_with_envelope_roundtrip(self) -> None:
        """Full integration: create envelope -> serialize -> frame -> buffer -> deserialize."""
        buf = FrameBuffer()

        payloads_in = [
            RunRequest(
                natural_language_command="Run tests",
                ssh_target=_ssh_target(),
            ),
            HealthRequest(),
            ErrorPayload(
                status_code=StatusCode.INTERNAL_ERROR,
                error="test error",
            ),
        ]
        kinds = [MessageKind.REQUEST, MessageKind.REQUEST, MessageKind.ERROR]

        # Serialize all envelopes into frames
        stream = b""
        for kind, payload in zip(kinds, payloads_in):
            wire = wrap_payload(kind, payload)
            stream += encode_frame(wire)

        # Feed all at once
        frames = buf.feed(stream)
        assert len(frames) == 3

        # Deserialize each frame back
        for i, frame_data in enumerate(frames):
            header, payload_out = unwrap_payload(frame_data)
            assert type(payload_out) == type(payloads_in[i])
            assert header.message_type == kinds[i]


# ===================================================================
# SerializationError
# ===================================================================


class TestSerializationError:
    """SerializationError carries structured context."""

    def test_inherits_from_exception(self) -> None:
        err = SerializationError("test message")
        assert isinstance(err, Exception)

    def test_message_preserved(self) -> None:
        err = SerializationError("bad data")
        assert str(err) == "bad data"

    def test_wraps_cause(self) -> None:
        cause = ValueError("inner problem")
        err = SerializationError("wrapper", cause=cause)
        assert err.__cause__ is cause
