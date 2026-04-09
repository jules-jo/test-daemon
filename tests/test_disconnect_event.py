"""Tests for disconnect event detection and classification.

Validates:
    - DisconnectType enum covers all expected disconnect signal categories
    - DisconnectEvent model is immutable and captures event type + metadata
    - classify_disconnect correctly maps each exception type to its category:
        * asyncio.IncompleteReadError -> EOF
        * BrokenPipeError -> BROKEN_PIPE
        * ConnectionResetError -> CONNECTION_RESET
        * asyncio.TimeoutError -> SOCKET_TIMEOUT
        * OSError -> OS_ERROR
        * Unknown exceptions -> UNKNOWN
    - Metadata includes exception class name, message, client_id, and timestamp
    - DisconnectEvent.to_event_payload() produces a dict suitable for EventBus
    - classify_disconnect handles edge cases (empty messages, None-like causes)
    - DisconnectEvent validation rejects empty client_id and event_type
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from jules_daemon.ipc.disconnect_event import (
    DISCONNECT_EVENT_TYPE,
    DisconnectEvent,
    DisconnectType,
    classify_disconnect,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00Z"
_CLIENT = "client-abc123"


# ---------------------------------------------------------------------------
# DisconnectType enum tests
# ---------------------------------------------------------------------------


class TestDisconnectType:
    """Verify DisconnectType enum values and coverage."""

    def test_eof_value(self) -> None:
        assert DisconnectType.EOF.value == "eof"

    def test_broken_pipe_value(self) -> None:
        assert DisconnectType.BROKEN_PIPE.value == "broken_pipe"

    def test_connection_reset_value(self) -> None:
        assert DisconnectType.CONNECTION_RESET.value == "connection_reset"

    def test_socket_timeout_value(self) -> None:
        assert DisconnectType.SOCKET_TIMEOUT.value == "socket_timeout"

    def test_os_error_value(self) -> None:
        assert DisconnectType.OS_ERROR.value == "os_error"

    def test_unknown_value(self) -> None:
        assert DisconnectType.UNKNOWN.value == "unknown"

    def test_all_members_present(self) -> None:
        expected = {
            "EOF",
            "BROKEN_PIPE",
            "CONNECTION_RESET",
            "SOCKET_TIMEOUT",
            "OS_ERROR",
            "UNKNOWN",
        }
        assert set(DisconnectType.__members__.keys()) == expected


# ---------------------------------------------------------------------------
# DisconnectEvent model tests
# ---------------------------------------------------------------------------


class TestDisconnectEvent:
    """Verify DisconnectEvent immutability, fields, and serialization."""

    def test_create_with_required_fields(self) -> None:
        event = DisconnectEvent(
            disconnect_type=DisconnectType.EOF,
            client_id=_CLIENT,
            timestamp=_TS,
            reason="Stream ended unexpectedly",
        )
        assert event.disconnect_type == DisconnectType.EOF
        assert event.client_id == _CLIENT
        assert event.timestamp == _TS
        assert event.reason == "Stream ended unexpectedly"

    def test_default_metadata_is_empty_dict(self) -> None:
        event = DisconnectEvent(
            disconnect_type=DisconnectType.BROKEN_PIPE,
            client_id=_CLIENT,
            timestamp=_TS,
            reason="Pipe broken",
        )
        assert event.metadata == {}

    def test_metadata_preserved(self) -> None:
        meta = {"errno": 32, "exception_class": "BrokenPipeError"}
        event = DisconnectEvent(
            disconnect_type=DisconnectType.BROKEN_PIPE,
            client_id=_CLIENT,
            timestamp=_TS,
            reason="Pipe broken",
            metadata=meta,
        )
        assert event.metadata == meta

    def test_frozen_immutability(self) -> None:
        event = DisconnectEvent(
            disconnect_type=DisconnectType.EOF,
            client_id=_CLIENT,
            timestamp=_TS,
            reason="EOF",
        )
        with pytest.raises(AttributeError):
            event.client_id = "other"  # type: ignore[misc]

    def test_empty_client_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="client_id must not be empty"):
            DisconnectEvent(
                disconnect_type=DisconnectType.EOF,
                client_id="",
                timestamp=_TS,
                reason="EOF",
            )

    def test_whitespace_client_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="client_id must not be empty"):
            DisconnectEvent(
                disconnect_type=DisconnectType.EOF,
                client_id="   ",
                timestamp=_TS,
                reason="EOF",
            )

    def test_empty_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="timestamp must not be empty"):
            DisconnectEvent(
                disconnect_type=DisconnectType.EOF,
                client_id=_CLIENT,
                timestamp="",
                reason="EOF",
            )

    def test_to_event_payload_structure(self) -> None:
        meta = {"errno": 104}
        event = DisconnectEvent(
            disconnect_type=DisconnectType.CONNECTION_RESET,
            client_id=_CLIENT,
            timestamp=_TS,
            reason="Connection reset by peer",
            metadata=meta,
        )
        payload = event.to_event_payload()

        assert payload == {
            "disconnect_type": "connection_reset",
            "client_id": _CLIENT,
            "timestamp": _TS,
            "reason": "Connection reset by peer",
            "metadata": {"errno": 104},
        }

    def test_to_event_payload_returns_new_dict(self) -> None:
        event = DisconnectEvent(
            disconnect_type=DisconnectType.EOF,
            client_id=_CLIENT,
            timestamp=_TS,
            reason="EOF",
        )
        p1 = event.to_event_payload()
        p2 = event.to_event_payload()
        assert p1 is not p2
        assert p1 == p2

    def test_event_type_constant(self) -> None:
        assert DISCONNECT_EVENT_TYPE == "client_disconnect"

    def test_invalid_disconnect_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="disconnect_type must be"):
            DisconnectEvent(
                disconnect_type="eof",  # type: ignore[arg-type]
                client_id=_CLIENT,
                timestamp=_TS,
                reason="EOF",
            )

    def test_none_disconnect_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="disconnect_type must be"):
            DisconnectEvent(
                disconnect_type=None,  # type: ignore[arg-type]
                client_id=_CLIENT,
                timestamp=_TS,
                reason="EOF",
            )

    def test_empty_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason must not be empty"):
            DisconnectEvent(
                disconnect_type=DisconnectType.EOF,
                client_id=_CLIENT,
                timestamp=_TS,
                reason="",
            )

    def test_whitespace_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason must not be empty"):
            DisconnectEvent(
                disconnect_type=DisconnectType.EOF,
                client_id=_CLIENT,
                timestamp=_TS,
                reason="   ",
            )

    def test_metadata_is_immutable_after_creation(self) -> None:
        meta = {"errno": 32}
        event = DisconnectEvent(
            disconnect_type=DisconnectType.BROKEN_PIPE,
            client_id=_CLIENT,
            timestamp=_TS,
            reason="Pipe broken",
            metadata=meta,
        )
        # Original dict mutation must not affect event
        meta["errno"] = 999
        assert event.metadata["errno"] == 32

    def test_metadata_mapping_proxy_prevents_mutation(self) -> None:
        event = DisconnectEvent(
            disconnect_type=DisconnectType.OS_ERROR,
            client_id=_CLIENT,
            timestamp=_TS,
            reason="OS error",
            metadata={"key": "value"},
        )
        with pytest.raises(TypeError):
            event.metadata["key"] = "mutated"  # type: ignore[index]

    def test_to_event_payload_metadata_is_isolated(self) -> None:
        event = DisconnectEvent(
            disconnect_type=DisconnectType.OS_ERROR,
            client_id=_CLIENT,
            timestamp=_TS,
            reason="OS error",
            metadata={"errno": 22},
        )
        payload = event.to_event_payload()
        payload["metadata"]["errno"] = 999
        assert event.metadata["errno"] == 22

    def test_empty_reason_from_build_reason_is_class_name(self) -> None:
        """BrokenPipeError() with no args produces reason == class name."""
        exc = BrokenPipeError()
        event = classify_disconnect(
            exception=exc,
            client_id=_CLIENT,
            timestamp=_TS,
        )
        assert event.reason == "BrokenPipeError"


# ---------------------------------------------------------------------------
# classify_disconnect tests
# ---------------------------------------------------------------------------


class TestClassifyDisconnect:
    """Verify exception-to-DisconnectEvent classification logic."""

    def test_incomplete_read_error_classified_as_eof(self) -> None:
        exc = asyncio.IncompleteReadError(partial=b"part", expected=100)
        event = classify_disconnect(
            exception=exc,
            client_id=_CLIENT,
        )
        assert event.disconnect_type == DisconnectType.EOF
        assert event.client_id == _CLIENT
        assert "IncompleteReadError" in event.reason
        assert event.metadata["exception_class"] == "IncompleteReadError"
        assert event.metadata["bytes_received"] == 4
        assert event.metadata["bytes_expected"] == 100

    def test_incomplete_read_with_none_expected(self) -> None:
        exc = asyncio.IncompleteReadError(partial=b"", expected=None)
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.EOF
        assert event.metadata["bytes_received"] == 0
        assert event.metadata["bytes_expected"] is None

    def test_broken_pipe_classified(self) -> None:
        exc = BrokenPipeError("Broken pipe")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.BROKEN_PIPE
        assert "BrokenPipeError" in event.reason
        assert event.metadata["exception_class"] == "BrokenPipeError"

    def test_broken_pipe_with_errno(self) -> None:
        exc = BrokenPipeError(32, "Broken pipe")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.BROKEN_PIPE
        assert event.metadata.get("errno") == 32

    def test_connection_reset_classified(self) -> None:
        exc = ConnectionResetError("Connection reset by peer")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.CONNECTION_RESET
        assert "ConnectionResetError" in event.reason
        assert event.metadata["exception_class"] == "ConnectionResetError"

    def test_connection_reset_with_errno(self) -> None:
        exc = ConnectionResetError(104, "Connection reset by peer")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.CONNECTION_RESET
        assert event.metadata.get("errno") == 104

    def test_timeout_error_classified(self) -> None:
        exc = asyncio.TimeoutError()
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.SOCKET_TIMEOUT
        assert "TimeoutError" in event.reason
        assert event.metadata["exception_class"] == "TimeoutError"

    def test_timeout_error_with_message(self) -> None:
        exc = asyncio.TimeoutError("read timed out after 30s")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.SOCKET_TIMEOUT
        assert "read timed out" in event.reason

    def test_os_error_classified(self) -> None:
        # Use an errno that does not map to a more specific subclass
        exc = OSError(22, "Invalid argument")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.OS_ERROR
        assert event.metadata["exception_class"] == "OSError"
        assert event.metadata.get("errno") == 22

    def test_generic_os_error_without_errno(self) -> None:
        exc = OSError("Something went wrong")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.OS_ERROR
        assert event.metadata["exception_class"] == "OSError"

    def test_unknown_exception_classified(self) -> None:
        exc = RuntimeError("unexpected failure")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.UNKNOWN
        assert "RuntimeError" in event.reason
        assert event.metadata["exception_class"] == "RuntimeError"

    def test_value_error_classified_as_unknown(self) -> None:
        exc = ValueError("bad data")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.UNKNOWN

    def test_timestamp_is_populated(self) -> None:
        exc = BrokenPipeError("broken")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        # Should be a valid ISO 8601 string
        assert event.timestamp
        # Parse to verify format
        dt = datetime.fromisoformat(event.timestamp)
        assert dt.tzinfo is not None

    def test_custom_timestamp_preserved(self) -> None:
        exc = BrokenPipeError("broken")
        event = classify_disconnect(
            exception=exc,
            client_id=_CLIENT,
            timestamp=_TS,
        )
        assert event.timestamp == _TS

    def test_exception_message_in_reason(self) -> None:
        exc = BrokenPipeError("pipe is toast")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert "pipe is toast" in event.reason

    def test_empty_exception_message_handled(self) -> None:
        exc = BrokenPipeError()
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.BROKEN_PIPE
        assert event.reason  # should still have a non-empty reason

    def test_connection_aborted_classified_as_os_error(self) -> None:
        exc = ConnectionAbortedError("Connection aborted")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        # ConnectionAbortedError is a subclass of ConnectionError -> OSError
        # but is not BrokenPipe or ConnectionReset, so OS_ERROR
        assert event.disconnect_type == DisconnectType.OS_ERROR

    def test_eof_error_classified_as_eof(self) -> None:
        exc = EOFError("end of stream")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.EOF
        assert event.metadata["exception_class"] == "EOFError"

    def test_metadata_includes_client_id(self) -> None:
        exc = BrokenPipeError("broken")
        event = classify_disconnect(exception=exc, client_id="client-xyz")
        assert event.client_id == "client-xyz"
        # client_id should also be in the top-level event, not just metadata
        payload = event.to_event_payload()
        assert payload["client_id"] == "client-xyz"


# ---------------------------------------------------------------------------
# classify_disconnect ordering tests (subclass priority)
# ---------------------------------------------------------------------------


class TestClassifySubclassPriority:
    """Verify that more-specific exception types take precedence.

    BrokenPipeError and ConnectionResetError are both subclasses of OSError,
    so the classifier must check for them before the generic OSError branch.
    """

    def test_broken_pipe_before_os_error(self) -> None:
        exc = BrokenPipeError("broken pipe")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.BROKEN_PIPE

    def test_connection_reset_before_os_error(self) -> None:
        exc = ConnectionResetError("reset by peer")
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.CONNECTION_RESET

    def test_incomplete_read_before_eof(self) -> None:
        exc = asyncio.IncompleteReadError(partial=b"", expected=10)
        event = classify_disconnect(exception=exc, client_id=_CLIENT)
        assert event.disconnect_type == DisconnectType.EOF
        # Should have the specific IncompleteReadError metadata
        assert "bytes_received" in event.metadata
