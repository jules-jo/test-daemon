"""Tests for IPC protocol validation helpers.

Covers JSON Schema generation for each message type, the
validate_message() entry point with structured error reporting,
and protocol version compatibility checks.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from jules_daemon.protocol.types import (
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MAJOR,
    PROTOCOL_VERSION_MINOR,
    MessageKind,
    StatusCode,
)
from jules_daemon.protocol.validation import (
    MessageValidationError,
    ValidationDetail,
    check_version_compatible,
    get_envelope_schema,
    get_payload_schema,
    list_payload_types,
    validate_message,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW_ISO = "2026-04-09T12:00:00+00:00"
_MSG_ID = str(uuid.uuid4())


def _valid_envelope(
    payload_type: str = "health_request",
    payload: dict[str, Any] | None = None,
    *,
    message_type: str = "request",
    protocol_version: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Build a minimal valid envelope dict for testing."""
    if payload is None:
        payload = {"payload_type": payload_type}
    return {
        "header": {
            "protocol_version": protocol_version,
            "message_id": _MSG_ID,
            "timestamp": _NOW_ISO,
            "message_type": message_type,
        },
        "payload": payload,
    }


def _valid_run_request_envelope() -> dict[str, Any]:
    """Build a valid envelope containing a RunRequest payload."""
    return _valid_envelope(
        payload_type="run_request",
        payload={
            "payload_type": "run_request",
            "natural_language_command": "Run pytest on auth module",
            "ssh_target": {"host": "staging.example.com", "user": "ci"},
        },
    )


# ---------------------------------------------------------------------------
# ValidationDetail
# ---------------------------------------------------------------------------


class TestValidationDetail:
    """ValidationDetail frozen dataclass tests."""

    def test_creation(self) -> None:
        detail = ValidationDetail(
            field="header.protocol_version",
            message="Version mismatch",
            code="version_incompatible",
        )
        assert detail.field == "header.protocol_version"
        assert detail.message == "Version mismatch"
        assert detail.code == "version_incompatible"

    def test_to_dict(self) -> None:
        detail = ValidationDetail(
            field="payload.run_id",
            message="Must not be empty",
            code="value_error",
        )
        result = detail.to_dict()
        assert result == {
            "field": "payload.run_id",
            "message": "Must not be empty",
            "code": "value_error",
        }

    def test_immutable(self) -> None:
        detail = ValidationDetail(
            field="f", message="m", code="c"
        )
        with pytest.raises(AttributeError):
            detail.field = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MessageValidationError
# ---------------------------------------------------------------------------


class TestMessageValidationError:
    """MessageValidationError structured exception tests."""

    def test_single_detail(self) -> None:
        detail = ValidationDetail(
            field="header",
            message="Missing required field",
            code="missing_field",
        )
        err = MessageValidationError(
            summary="Invalid message structure",
            details=(detail,),
        )
        assert "Invalid message structure" in str(err)
        assert len(err.details) == 1
        assert err.details[0].field == "header"

    def test_multiple_details(self) -> None:
        details = (
            ValidationDetail(field="a", message="m1", code="c1"),
            ValidationDetail(field="b", message="m2", code="c2"),
        )
        err = MessageValidationError(
            summary="Multiple issues",
            details=details,
        )
        assert len(err.details) == 2

    def test_to_dict(self) -> None:
        detail = ValidationDetail(field="f", message="m", code="c")
        err = MessageValidationError(summary="Bad", details=(detail,))
        result = err.to_dict()
        assert result["summary"] == "Bad"
        assert len(result["details"]) == 1
        assert result["details"][0]["field"] == "f"

    def test_empty_details(self) -> None:
        err = MessageValidationError(summary="General error", details=())
        assert len(err.details) == 0
        result = err.to_dict()
        assert result["details"] == []


# ---------------------------------------------------------------------------
# JSON Schema generation: get_payload_schema
# ---------------------------------------------------------------------------


class TestGetPayloadSchema:
    """get_payload_schema() JSON Schema generation tests."""

    def test_returns_dict_for_known_type(self) -> None:
        schema = get_payload_schema("run_request")
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"

    def test_schema_has_properties(self) -> None:
        schema = get_payload_schema("run_request")
        assert "properties" in schema

    def test_run_request_schema_fields(self) -> None:
        schema = get_payload_schema("run_request")
        props = schema["properties"]
        assert "natural_language_command" in props
        assert "ssh_target" in props
        assert "payload_type" in props

    def test_health_request_schema(self) -> None:
        schema = get_payload_schema("health_request")
        props = schema["properties"]
        assert "payload_type" in props

    def test_confirm_reply_schema(self) -> None:
        schema = get_payload_schema("confirm_reply")
        props = schema["properties"]
        assert "decision" in props
        assert "run_id" in props

    def test_error_payload_schema(self) -> None:
        schema = get_payload_schema("error")
        props = schema["properties"]
        assert "status_code" in props
        assert "error" in props

    def test_unknown_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown payload type"):
            get_payload_schema("not_a_type")

    def test_all_known_types_have_schemas(self) -> None:
        """Every registered payload type returns a valid schema."""
        for pt in list_payload_types():
            schema = get_payload_schema(pt)
            assert isinstance(schema, dict)
            assert "properties" in schema


# ---------------------------------------------------------------------------
# JSON Schema generation: get_envelope_schema
# ---------------------------------------------------------------------------


class TestGetEnvelopeSchema:
    """get_envelope_schema() JSON Schema generation tests."""

    def test_returns_dict(self) -> None:
        schema = get_envelope_schema()
        assert isinstance(schema, dict)

    def test_has_header_and_payload(self) -> None:
        schema = get_envelope_schema()
        props = schema.get("properties", {})
        assert "header" in props
        assert "payload" in props

    def test_required_fields(self) -> None:
        schema = get_envelope_schema()
        required = schema.get("required", [])
        assert "header" in required
        assert "payload" in required

    def test_schema_is_valid_json(self) -> None:
        """Schema can be serialized to JSON without errors."""
        schema = get_envelope_schema()
        json_str = json.dumps(schema)
        restored = json.loads(json_str)
        assert restored == schema


# ---------------------------------------------------------------------------
# list_payload_types
# ---------------------------------------------------------------------------


class TestListPayloadTypes:
    """list_payload_types() registry listing tests."""

    def test_returns_sorted_tuple(self) -> None:
        result = list_payload_types()
        assert isinstance(result, tuple)
        assert result == tuple(sorted(result))

    def test_contains_all_expected_types(self) -> None:
        expected = {
            "run_request",
            "run_response",
            "status_request",
            "status_response",
            "watch_request",
            "stream_chunk",
            "cancel_request",
            "cancel_response",
            "confirm_prompt",
            "confirm_reply",
            "health_request",
            "health_response",
            "history_request",
            "history_response",
            "error",
        }
        result = set(list_payload_types())
        assert expected == result


# ---------------------------------------------------------------------------
# Version compatibility: check_version_compatible
# ---------------------------------------------------------------------------


class TestCheckVersionCompatible:
    """check_version_compatible() version checks."""

    def test_same_version_compatible(self) -> None:
        assert check_version_compatible(PROTOCOL_VERSION) is True

    def test_same_major_higher_minor_compatible(self) -> None:
        bumped = f"{PROTOCOL_VERSION_MAJOR}.{PROTOCOL_VERSION_MINOR + 1}.0"
        assert check_version_compatible(bumped) is True

    def test_same_major_lower_minor_compatible(self) -> None:
        if PROTOCOL_VERSION_MINOR > 0:
            lower = f"{PROTOCOL_VERSION_MAJOR}.{PROTOCOL_VERSION_MINOR - 1}.0"
            assert check_version_compatible(lower) is True
        else:
            # Minor is already 0, just verify current is compatible
            assert check_version_compatible(PROTOCOL_VERSION) is True

    def test_different_major_incompatible(self) -> None:
        incompatible = f"{PROTOCOL_VERSION_MAJOR + 1}.0.0"
        assert check_version_compatible(incompatible) is False

    def test_major_zero_incompatible(self) -> None:
        if PROTOCOL_VERSION_MAJOR != 0:
            assert check_version_compatible("0.1.0") is False

    def test_higher_patch_compatible(self) -> None:
        bumped = f"{PROTOCOL_VERSION_MAJOR}.{PROTOCOL_VERSION_MINOR}.99"
        assert check_version_compatible(bumped) is True

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid version"):
            check_version_compatible("")

    def test_nonsense_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid version"):
            check_version_compatible("not.a.version")

    def test_two_part_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid version"):
            check_version_compatible("1.0")

    def test_four_part_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid version"):
            check_version_compatible("1.0.0.0")


# ---------------------------------------------------------------------------
# validate_message: input forms
# ---------------------------------------------------------------------------


class TestValidateMessageInputForms:
    """validate_message() accepts dict, str, and bytes."""

    def test_accepts_dict(self) -> None:
        envelope = validate_message(_valid_envelope())
        assert envelope.header.message_type == MessageKind.REQUEST

    def test_accepts_json_string(self) -> None:
        data = json.dumps(_valid_envelope())
        envelope = validate_message(data)
        assert envelope.header.message_type == MessageKind.REQUEST

    def test_accepts_json_bytes(self) -> None:
        data = json.dumps(_valid_envelope()).encode("utf-8")
        envelope = validate_message(data)
        assert envelope.header.message_type == MessageKind.REQUEST


# ---------------------------------------------------------------------------
# validate_message: structure validation
# ---------------------------------------------------------------------------


class TestValidateMessageStructure:
    """validate_message() structural validation tests."""

    def test_missing_header_raises(self) -> None:
        data = {"payload": {"payload_type": "health_request"}}
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message(data)
        assert any(d.field == "header" for d in exc_info.value.details)

    def test_missing_payload_raises(self) -> None:
        data = {
            "header": {
                "protocol_version": PROTOCOL_VERSION,
                "message_id": _MSG_ID,
                "timestamp": _NOW_ISO,
                "message_type": "request",
            },
        }
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message(data)
        assert any(d.field == "payload" for d in exc_info.value.details)

    def test_non_dict_input_raises(self) -> None:
        with pytest.raises(MessageValidationError, match="must be a JSON object"):
            validate_message("[]")

    def test_invalid_json_string_raises(self) -> None:
        with pytest.raises(MessageValidationError, match="Invalid JSON"):
            validate_message("{bad json")

    def test_invalid_json_bytes_raises(self) -> None:
        with pytest.raises(MessageValidationError, match="Invalid JSON"):
            validate_message(b"{bad json")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(MessageValidationError):
            validate_message("")

    def test_empty_bytes_raises(self) -> None:
        with pytest.raises(MessageValidationError):
            validate_message(b"")

    def test_none_header_raises(self) -> None:
        data = {"header": None, "payload": {"payload_type": "health_request"}}
        with pytest.raises(MessageValidationError):
            validate_message(data)


# ---------------------------------------------------------------------------
# validate_message: version checks
# ---------------------------------------------------------------------------


class TestValidateMessageVersion:
    """validate_message() protocol version validation tests."""

    def test_matching_version_passes(self) -> None:
        envelope = validate_message(_valid_envelope())
        assert envelope.header.protocol_version == PROTOCOL_VERSION

    def test_incompatible_major_version_raises(self) -> None:
        incompatible_version = f"{PROTOCOL_VERSION_MAJOR + 1}.0.0"
        data = _valid_envelope(protocol_version=incompatible_version)
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message(data)
        assert any(
            d.code == "version_incompatible" for d in exc_info.value.details
        )

    def test_compatible_minor_version_passes(self) -> None:
        compat_version = f"{PROTOCOL_VERSION_MAJOR}.{PROTOCOL_VERSION_MINOR + 1}.0"
        data = _valid_envelope(protocol_version=compat_version)
        envelope = validate_message(data)
        assert envelope.header.protocol_version == compat_version

    def test_missing_protocol_version_raises(self) -> None:
        data = _valid_envelope()
        del data["header"]["protocol_version"]
        with pytest.raises(MessageValidationError):
            validate_message(data)


# ---------------------------------------------------------------------------
# validate_message: payload validation
# ---------------------------------------------------------------------------


class TestValidateMessagePayload:
    """validate_message() payload content validation tests."""

    def test_valid_run_request(self) -> None:
        envelope = validate_message(_valid_run_request_envelope())
        assert envelope.payload.payload_type == "run_request"

    def test_missing_payload_type_raises(self) -> None:
        data = _valid_envelope(
            payload={
                "natural_language_command": "Run tests",
                "ssh_target": {"host": "staging", "user": "ci"},
            },
        )
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message(data)
        assert any(
            d.code == "missing_discriminator" for d in exc_info.value.details
        )

    def test_unknown_payload_type_raises(self) -> None:
        data = _valid_envelope(
            payload={"payload_type": "nonexistent_type"},
        )
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message(data)
        assert any(
            d.code == "unknown_payload_type" for d in exc_info.value.details
        )

    def test_invalid_payload_content_raises(self) -> None:
        """RunRequest with empty command string should fail validation."""
        data = _valid_envelope(
            payload={
                "payload_type": "run_request",
                "natural_language_command": "",
                "ssh_target": {"host": "staging", "user": "ci"},
            },
        )
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message(data)
        assert any(
            d.code == "payload_validation_error" for d in exc_info.value.details
        )

    def test_invalid_ssh_port_raises(self) -> None:
        """Port out of range should fail."""
        data = _valid_envelope(
            payload={
                "payload_type": "run_request",
                "natural_language_command": "Run tests",
                "ssh_target": {"host": "staging", "user": "ci", "port": 99999},
            },
        )
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message(data)
        assert any(
            d.code == "payload_validation_error" for d in exc_info.value.details
        )

    def test_valid_status_request(self) -> None:
        data = _valid_envelope(payload_type="status_request")
        envelope = validate_message(data)
        assert envelope.payload.payload_type == "status_request"

    def test_valid_health_request(self) -> None:
        data = _valid_envelope(payload_type="health_request")
        envelope = validate_message(data)
        assert envelope.payload.payload_type == "health_request"

    def test_valid_cancel_request(self) -> None:
        data = _valid_envelope(
            payload={
                "payload_type": "cancel_request",
                "run_id": "r-001",
            },
        )
        envelope = validate_message(data)
        assert envelope.payload.payload_type == "cancel_request"

    def test_valid_confirm_reply_allow(self) -> None:
        data = _valid_envelope(
            message_type="confirm_reply",
            payload={
                "payload_type": "confirm_reply",
                "run_id": "r-001",
                "decision": "allow",
            },
        )
        envelope = validate_message(data)
        assert envelope.payload.payload_type == "confirm_reply"

    def test_valid_confirm_reply_deny(self) -> None:
        data = _valid_envelope(
            message_type="confirm_reply",
            payload={
                "payload_type": "confirm_reply",
                "run_id": "r-001",
                "decision": "deny",
                "reason": "Command looks suspicious",
            },
        )
        envelope = validate_message(data)
        assert envelope.payload.payload_type == "confirm_reply"

    def test_valid_history_request(self) -> None:
        data = _valid_envelope(
            payload={
                "payload_type": "history_request",
                "limit": 10,
                "offset": 0,
            },
        )
        envelope = validate_message(data)
        assert envelope.payload.payload_type == "history_request"

    def test_valid_watch_request(self) -> None:
        data = _valid_envelope(
            payload={
                "payload_type": "watch_request",
                "run_id": "r-001",
            },
        )
        envelope = validate_message(data)
        assert envelope.payload.payload_type == "watch_request"


# ---------------------------------------------------------------------------
# validate_message: return value
# ---------------------------------------------------------------------------


class TestValidateMessageReturnValue:
    """validate_message() returns a properly typed Envelope on success."""

    def test_returns_envelope_type(self) -> None:
        from jules_daemon.protocol.schemas import Envelope

        envelope = validate_message(_valid_envelope())
        assert isinstance(envelope, Envelope)

    def test_returns_frozen_envelope(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        envelope = validate_message(_valid_envelope())
        with pytest.raises(PydanticValidationError):
            envelope.header = envelope.header  # type: ignore[misc]

    def test_header_fields_populated(self) -> None:
        envelope = validate_message(_valid_envelope())
        assert envelope.header.protocol_version == PROTOCOL_VERSION
        assert envelope.header.message_id == _MSG_ID
        assert envelope.header.message_type == MessageKind.REQUEST

    def test_payload_correctly_typed(self) -> None:
        from jules_daemon.protocol.schemas import RunRequest

        envelope = validate_message(_valid_run_request_envelope())
        assert isinstance(envelope.payload, RunRequest)
        assert (
            envelope.payload.natural_language_command
            == "Run pytest on auth module"
        )


# ---------------------------------------------------------------------------
# validate_message: error detail structure
# ---------------------------------------------------------------------------


class TestValidateMessageErrorDetails:
    """Structured error details from validate_message() failures."""

    def test_error_has_summary(self) -> None:
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message("{}")
        assert exc_info.value.summary != ""

    def test_error_has_details_list(self) -> None:
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message("{}")
        assert isinstance(exc_info.value.details, tuple)

    def test_error_to_dict_is_json_serializable(self) -> None:
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message("{}")
        result = exc_info.value.to_dict()
        json_str = json.dumps(result)
        assert json_str  # round-trips without error

    def test_pydantic_errors_mapped_to_details(self) -> None:
        """Pydantic validation errors produce meaningful detail entries."""
        data = _valid_envelope(
            payload={
                "payload_type": "run_request",
                "natural_language_command": "",
                "ssh_target": {"host": "", "user": "ci"},
            },
        )
        with pytest.raises(MessageValidationError) as exc_info:
            validate_message(data)
        assert len(exc_info.value.details) >= 1
