"""Tests for IPC message framing protocol.

Validates the length-prefixed wire format, MessageEnvelope dataclass,
MessageType enum, and standalone encode/decode functions.
"""

from __future__ import annotations

import json
import struct

import pytest

from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MAX_PAYLOAD_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    pack_header,
    unpack_header,
)


# ---------------------------------------------------------------------------
# MessageType enum
# ---------------------------------------------------------------------------


class TestMessageType:
    """Tests for the MessageType enum."""

    def test_request_value(self) -> None:
        assert MessageType.REQUEST.value == "request"

    def test_response_value(self) -> None:
        assert MessageType.RESPONSE.value == "response"

    def test_stream_value(self) -> None:
        assert MessageType.STREAM.value == "stream"

    def test_confirm_prompt_value(self) -> None:
        assert MessageType.CONFIRM_PROMPT.value == "confirm_prompt"

    def test_confirm_reply_value(self) -> None:
        assert MessageType.CONFIRM_REPLY.value == "confirm_reply"

    def test_error_value(self) -> None:
        assert MessageType.ERROR.value == "error"

    def test_all_six_types_exist(self) -> None:
        assert len(MessageType) == 6


# ---------------------------------------------------------------------------
# MessageEnvelope dataclass
# ---------------------------------------------------------------------------


class TestMessageEnvelope:
    """Tests for the frozen MessageEnvelope dataclass."""

    def test_create_request_envelope(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="abc-123",
            timestamp="2026-04-09T12:00:00Z",
            payload={"verb": "status"},
        )
        assert env.msg_type is MessageType.REQUEST
        assert env.msg_id == "abc-123"
        assert env.timestamp == "2026-04-09T12:00:00Z"
        assert env.payload == {"verb": "status"}

    def test_frozen(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="abc",
            timestamp="2026-04-09T12:00:00Z",
            payload={},
        )
        with pytest.raises(AttributeError):
            env.msg_type = MessageType.ERROR  # type: ignore[misc]

    def test_empty_msg_id_raises(self) -> None:
        with pytest.raises(ValueError, match="msg_id must not be empty"):
            MessageEnvelope(
                msg_type=MessageType.REQUEST,
                msg_id="",
                timestamp="2026-04-09T12:00:00Z",
                payload={},
            )

    def test_whitespace_msg_id_raises(self) -> None:
        with pytest.raises(ValueError, match="msg_id must not be empty"):
            MessageEnvelope(
                msg_type=MessageType.REQUEST,
                msg_id="   ",
                timestamp="2026-04-09T12:00:00Z",
                payload={},
            )

    def test_empty_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timestamp must not be empty"):
            MessageEnvelope(
                msg_type=MessageType.REQUEST,
                msg_id="abc",
                timestamp="",
                payload={},
            )

    def test_payload_must_be_dict(self) -> None:
        with pytest.raises(TypeError, match="payload must be a dict"):
            MessageEnvelope(
                msg_type=MessageType.REQUEST,
                msg_id="abc",
                timestamp="2026-04-09T12:00:00Z",
                payload="not a dict",  # type: ignore[arg-type]
            )

    def test_non_string_msg_id_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="msg_id must be a str"):
            MessageEnvelope(
                msg_type=MessageType.REQUEST,
                msg_id=123,  # type: ignore[arg-type]
                timestamp="2026-04-09T12:00:00Z",
                payload={},
            )

    def test_non_string_timestamp_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="timestamp must be a str"):
            MessageEnvelope(
                msg_type=MessageType.REQUEST,
                msg_id="abc",
                timestamp=None,  # type: ignore[arg-type]
                payload={},
            )

    def test_payload_defaults_to_empty_dict(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="abc",
            timestamp="2026-04-09T12:00:00Z",
        )
        assert env.payload == {}

    def test_to_dict_roundtrip(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="xyz-789",
            timestamp="2026-04-09T12:00:00Z",
            payload={"result": "ok", "data": [1, 2, 3]},
        )
        d = env.to_dict()
        assert d["msg_type"] == "response"
        assert d["msg_id"] == "xyz-789"
        assert d["timestamp"] == "2026-04-09T12:00:00Z"
        assert d["payload"] == {"result": "ok", "data": [1, 2, 3]}

    def test_from_dict_roundtrip(self) -> None:
        original = MessageEnvelope(
            msg_type=MessageType.STREAM,
            msg_id="stream-001",
            timestamp="2026-04-09T12:00:00Z",
            payload={"line": "test output"},
        )
        restored = MessageEnvelope.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_unknown_msg_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown msg_type"):
            MessageEnvelope.from_dict({
                "msg_type": "unknown",
                "msg_id": "abc",
                "timestamp": "2026-04-09T12:00:00Z",
                "payload": {},
            })

    def test_from_dict_missing_field_raises(self) -> None:
        with pytest.raises(KeyError):
            MessageEnvelope.from_dict({"msg_type": "request"})

    def test_all_message_types_roundtrip(self) -> None:
        """Every MessageType survives to_dict -> from_dict."""
        for mt in MessageType:
            env = MessageEnvelope(
                msg_type=mt,
                msg_id=f"test-{mt.value}",
                timestamp="2026-04-09T12:00:00Z",
                payload={"type_check": mt.value},
            )
            assert MessageEnvelope.from_dict(env.to_dict()) == env


# ---------------------------------------------------------------------------
# pack_header / unpack_header
# ---------------------------------------------------------------------------


class TestHeaderPacking:
    """Tests for the low-level 4-byte length header."""

    def test_header_size_is_four(self) -> None:
        assert HEADER_SIZE == 4

    def test_pack_header_returns_four_bytes(self) -> None:
        header = pack_header(100)
        assert len(header) == 4

    def test_pack_unpack_roundtrip(self) -> None:
        for length in (0, 1, 255, 65535, 1_000_000, MAX_PAYLOAD_SIZE):
            header = pack_header(length)
            assert unpack_header(header) == length

    def test_pack_big_endian_encoding(self) -> None:
        """Verify the exact bytes match big-endian uint32."""
        header = pack_header(256)
        expected = struct.pack("!I", 256)
        assert header == expected

    def test_pack_negative_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            pack_header(-1)

    def test_pack_exceeds_max_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds maximum"):
            pack_header(MAX_PAYLOAD_SIZE + 1)

    def test_unpack_wrong_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Header must be exactly"):
            unpack_header(b"\x00\x00")

    def test_unpack_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="Header must be exactly"):
            unpack_header(b"")

    def test_unpack_five_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="Header must be exactly"):
            unpack_header(b"\x00\x00\x00\x00\x00")

    def test_unpack_oversized_length_raises(self) -> None:
        """A malicious peer sending an oversized length is rejected."""
        oversized = struct.pack("!I", MAX_PAYLOAD_SIZE + 1)
        with pytest.raises(ValueError, match="exceeds maximum"):
            unpack_header(oversized)


# ---------------------------------------------------------------------------
# encode_frame / decode_envelope (full roundtrip)
# ---------------------------------------------------------------------------


class TestEncodeDecodeFrame:
    """Tests for the high-level encode_frame and decode_envelope functions."""

    def test_encode_produces_header_plus_payload(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="test-1",
            timestamp="2026-04-09T12:00:00Z",
            payload={"verb": "status"},
        )
        frame = encode_frame(env)
        length = unpack_header(frame[:HEADER_SIZE])
        payload_bytes = frame[HEADER_SIZE:]
        assert len(payload_bytes) == length

    def test_encode_decode_roundtrip(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.CONFIRM_PROMPT,
            msg_id="confirm-001",
            timestamp="2026-04-09T12:00:00Z",
            payload={
                "command": "pytest tests/",
                "host": "prod-server",
            },
        )
        frame = encode_frame(env)
        payload_bytes = frame[HEADER_SIZE:]
        restored = decode_envelope(payload_bytes)
        assert restored == env

    def test_encode_utf8_payload(self) -> None:
        """Non-ASCII characters in the payload are handled correctly."""
        env = MessageEnvelope(
            msg_type=MessageType.STREAM,
            msg_id="utf8-test",
            timestamp="2026-04-09T12:00:00Z",
            payload={"output": "Ergebnis: bestanden \u2714"},
        )
        frame = encode_frame(env)
        payload_bytes = frame[HEADER_SIZE:]
        restored = decode_envelope(payload_bytes)
        assert restored.payload["output"] == "Ergebnis: bestanden \u2714"

    def test_decode_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to decode"):
            decode_envelope(b"not json at all")

    def test_decode_non_dict_json_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a JSON object"):
            decode_envelope(json.dumps([1, 2, 3]).encode("utf-8"))

    def test_encode_empty_payload(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.REQUEST,
            msg_id="empty-payload",
            timestamp="2026-04-09T12:00:00Z",
            payload={},
        )
        frame = encode_frame(env)
        payload_bytes = frame[HEADER_SIZE:]
        restored = decode_envelope(payload_bytes)
        assert restored.payload == {}

    def test_encode_nested_payload(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="nested-1",
            timestamp="2026-04-09T12:00:00Z",
            payload={
                "result": {
                    "tests_passed": 42,
                    "tests_failed": 0,
                    "details": [
                        {"name": "test_foo", "status": "pass"},
                        {"name": "test_bar", "status": "pass"},
                    ],
                }
            },
        )
        frame = encode_frame(env)
        payload_bytes = frame[HEADER_SIZE:]
        restored = decode_envelope(payload_bytes)
        assert restored == env

    def test_frame_is_bytes(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.ERROR,
            msg_id="err-1",
            timestamp="2026-04-09T12:00:00Z",
            payload={"error": "something went wrong"},
        )
        frame = encode_frame(env)
        assert isinstance(frame, bytes)

    def test_multiple_frames_concatenated(self) -> None:
        """Simulate reading multiple frames from a stream buffer."""
        envelopes = [
            MessageEnvelope(
                msg_type=MessageType.REQUEST,
                msg_id=f"multi-{i}",
                timestamp="2026-04-09T12:00:00Z",
                payload={"index": i},
            )
            for i in range(3)
        ]
        buffer = b"".join(encode_frame(e) for e in envelopes)

        offset = 0
        decoded = []
        while offset < len(buffer):
            length = unpack_header(buffer[offset : offset + HEADER_SIZE])
            offset += HEADER_SIZE
            payload_bytes = buffer[offset : offset + length]
            decoded.append(decode_envelope(payload_bytes))
            offset += length

        assert decoded == envelopes

    def test_oversized_payload_raises_on_encode(self) -> None:
        """Payloads exceeding MAX_PAYLOAD_SIZE are rejected at encode time."""
        # Build a payload large enough to exceed the limit when JSON-encoded
        large_value = "x" * (MAX_PAYLOAD_SIZE + 1)
        env = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="oversize-1",
            timestamp="2026-04-09T12:00:00Z",
            payload={"data": large_value},
        )
        with pytest.raises(ValueError, match="exceeds maximum"):
            encode_frame(env)
