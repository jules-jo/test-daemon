"""Tests for the protocol package public API (protocol/__init__.py).

Verifies that all re-exported symbols are importable from the top-level
``jules_daemon.protocol`` namespace, that convenience aliases point to the
correct underlying functions, and that the ``__all__`` list is accurate.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

import jules_daemon.protocol as protocol


# ---------------------------------------------------------------------------
# __all__ completeness
# ---------------------------------------------------------------------------


class TestAllExports:
    """Ensure __all__ is defined and every name resolves to an attribute."""

    def test_all_is_defined(self) -> None:
        assert hasattr(protocol, "__all__")
        assert isinstance(protocol.__all__, list)

    def test_all_entries_are_strings(self) -> None:
        for name in protocol.__all__:
            assert isinstance(name, str), f"__all__ entry {name!r} is not a string"

    def test_all_entries_resolve(self) -> None:
        for name in protocol.__all__:
            assert hasattr(protocol, name), (
                f"__all__ lists {name!r} but it is not an attribute of the module"
            )

    def test_no_extra_public_names(self) -> None:
        """Every non-underscore public attr should be in __all__."""
        public_attrs = {
            name
            for name in dir(protocol)
            if not name.startswith("_") and not isinstance(getattr(protocol, name), types.ModuleType)
        }
        all_set = set(protocol.__all__)
        extras = public_attrs - all_set
        assert not extras, (
            f"Public attributes not in __all__: {sorted(extras)}"
        )


# ---------------------------------------------------------------------------
# Protocol constants re-exported from types
# ---------------------------------------------------------------------------


class TestProtocolConstantsReexport:
    """Verify protocol identity constants are accessible at package level."""

    def test_protocol_name(self) -> None:
        assert protocol.PROTOCOL_NAME == "jules-ipc"

    def test_protocol_version(self) -> None:
        assert protocol.PROTOCOL_VERSION == "1.0.0"

    def test_protocol_version_major(self) -> None:
        assert protocol.PROTOCOL_VERSION_MAJOR == 1

    def test_protocol_version_minor(self) -> None:
        assert protocol.PROTOCOL_VERSION_MINOR == 0

    def test_protocol_version_patch(self) -> None:
        assert protocol.PROTOCOL_VERSION_PATCH == 0


# ---------------------------------------------------------------------------
# Enum re-exports
# ---------------------------------------------------------------------------


class TestEnumReexports:
    """Verify MessageKind and StatusCode are accessible at package level."""

    def test_message_kind_available(self) -> None:
        assert hasattr(protocol, "MessageKind")
        assert protocol.MessageKind.REQUEST.value == "request"

    def test_status_code_available(self) -> None:
        assert hasattr(protocol, "StatusCode")
        assert protocol.StatusCode.OK.value == 200

    def test_approval_decision_available(self) -> None:
        assert hasattr(protocol, "ApprovalDecision")
        assert protocol.ApprovalDecision.ALLOW.value == "allow"


# ---------------------------------------------------------------------------
# Type helper re-exports from types
# ---------------------------------------------------------------------------


class TestTypeHelperReexports:
    """Verify helper functions from types.py are accessible."""

    def test_parse_message_kind(self) -> None:
        result = protocol.parse_message_kind("request")
        assert result is protocol.MessageKind.REQUEST

    def test_parse_status_code(self) -> None:
        result = protocol.parse_status_code(200)
        assert result is protocol.StatusCode.OK

    def test_is_terminal_message(self) -> None:
        assert protocol.is_terminal_message(protocol.MessageKind.RESPONSE) is True
        assert protocol.is_terminal_message(protocol.MessageKind.STREAM) is False

    def test_is_success(self) -> None:
        assert protocol.is_success(protocol.StatusCode.OK) is True

    def test_is_client_error(self) -> None:
        assert protocol.is_client_error(protocol.StatusCode.BAD_REQUEST) is True

    def test_is_server_error(self) -> None:
        assert protocol.is_server_error(protocol.StatusCode.INTERNAL_ERROR) is True

    def test_status_code_to_reason(self) -> None:
        assert protocol.status_code_to_reason(protocol.StatusCode.OK) == "OK"


# ---------------------------------------------------------------------------
# Schema model re-exports
# ---------------------------------------------------------------------------


class TestSchemaReexports:
    """Verify Pydantic schema models are accessible at package level."""

    def test_envelope_class(self) -> None:
        assert hasattr(protocol, "Envelope")

    def test_message_header_class(self) -> None:
        assert hasattr(protocol, "MessageHeader")

    def test_payload_type_union(self) -> None:
        assert hasattr(protocol, "PayloadType")

    def test_create_envelope_factory(self) -> None:
        assert callable(protocol.create_envelope)

    @pytest.mark.parametrize(
        "model_name",
        [
            "RunRequest",
            "RunResponse",
            "StatusRequest",
            "StatusResponse",
            "WatchRequest",
            "StreamChunk",
            "CancelRequest",
            "CancelResponse",
            "ConfirmPromptPayload",
            "ConfirmReplyPayload",
            "HealthRequest",
            "HealthResponse",
            "HistoryRequest",
            "HistoryResponse",
            "HistoryRunSummary",
            "ErrorPayload",
            "SSHTargetInfo",
            "ProgressSnapshot",
        ],
    )
    def test_payload_model_available(self, model_name: str) -> None:
        assert hasattr(protocol, model_name), (
            f"Payload model {model_name!r} not re-exported from protocol package"
        )


# ---------------------------------------------------------------------------
# Serialization re-exports
# ---------------------------------------------------------------------------


class TestSerializationReexports:
    """Verify serialization functions are accessible at package level."""

    def test_serialize_alias(self) -> None:
        """serialize should be an alias for serialize_envelope."""
        from jules_daemon.protocol.serialization import serialize_envelope

        assert protocol.serialize is serialize_envelope

    def test_deserialize_alias(self) -> None:
        """deserialize should be an alias for deserialize_envelope."""
        from jules_daemon.protocol.serialization import deserialize_envelope

        assert protocol.deserialize is deserialize_envelope

    def test_serialize_envelope(self) -> None:
        assert callable(protocol.serialize_envelope)

    def test_deserialize_envelope(self) -> None:
        assert callable(protocol.deserialize_envelope)

    def test_serialize_payload(self) -> None:
        assert callable(protocol.serialize_payload)

    def test_deserialize_payload(self) -> None:
        assert callable(protocol.deserialize_payload)

    def test_wrap_payload(self) -> None:
        assert callable(protocol.wrap_payload)

    def test_unwrap_payload(self) -> None:
        assert callable(protocol.unwrap_payload)

    def test_encode_frame(self) -> None:
        assert callable(protocol.encode_frame)

    def test_decode_frame(self) -> None:
        assert callable(protocol.decode_frame)

    def test_frame_buffer_class(self) -> None:
        assert hasattr(protocol, "FrameBuffer")
        buf = protocol.FrameBuffer()
        assert buf.pending == 0

    def test_serialization_error(self) -> None:
        assert issubclass(protocol.SerializationError, Exception)


# ---------------------------------------------------------------------------
# Validation re-exports
# ---------------------------------------------------------------------------


class TestValidationReexports:
    """Verify validation functions are accessible at package level."""

    def test_validate_message(self) -> None:
        assert callable(protocol.validate_message)

    def test_message_validation_error(self) -> None:
        assert issubclass(protocol.MessageValidationError, Exception)

    def test_validation_detail(self) -> None:
        detail = protocol.ValidationDetail(
            field="test", message="test msg", code="test_code"
        )
        assert detail.field == "test"

    def test_get_payload_schema(self) -> None:
        assert callable(protocol.get_payload_schema)

    def test_get_envelope_schema(self) -> None:
        assert callable(protocol.get_envelope_schema)

    def test_list_payload_types(self) -> None:
        assert callable(protocol.list_payload_types)
        result = protocol.list_payload_types()
        assert isinstance(result, tuple)
        assert len(result) > 0

    def test_check_version_compatible(self) -> None:
        assert callable(protocol.check_version_compatible)
        assert protocol.check_version_compatible("1.0.0") is True


# ---------------------------------------------------------------------------
# Convenience alias round-trip
# ---------------------------------------------------------------------------


class TestConvenienceAliasRoundTrip:
    """Verify serialize/deserialize aliases work end-to-end."""

    def test_serialize_deserialize_round_trip(self) -> None:
        """Create an envelope, serialize via alias, deserialize via alias."""
        payload = protocol.HealthRequest()
        envelope = protocol.create_envelope(
            message_type=protocol.MessageKind.REQUEST,
            payload=payload,
        )
        wire_bytes = protocol.serialize(envelope)
        assert isinstance(wire_bytes, bytes)

        restored = protocol.deserialize(wire_bytes)
        assert isinstance(restored, protocol.Envelope)
        assert restored.header.message_type is protocol.MessageKind.REQUEST
        assert isinstance(restored.payload, protocol.HealthRequest)


# ---------------------------------------------------------------------------
# Module docstring
# ---------------------------------------------------------------------------


class TestModuleDocstring:
    """Verify that the module docstring contains protocol specification."""

    def test_docstring_exists(self) -> None:
        assert protocol.__doc__ is not None
        assert len(protocol.__doc__) > 100

    def test_docstring_documents_wire_format(self) -> None:
        doc = protocol.__doc__
        assert "wire format" in doc.lower() or "wire-format" in doc.lower()

    def test_docstring_documents_envelope(self) -> None:
        doc = protocol.__doc__
        assert "envelope" in doc.lower()

    def test_docstring_documents_message_types(self) -> None:
        doc = protocol.__doc__
        assert "message" in doc.lower()
        # Should mention the key message kinds
        assert "request" in doc.lower()
        assert "response" in doc.lower()

    def test_docstring_documents_framing(self) -> None:
        doc = protocol.__doc__
        assert "frame" in doc.lower() or "framing" in doc.lower()
