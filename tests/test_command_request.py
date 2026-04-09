"""Tests for the CommandRequest model and validate function.

Covers:
- Field validation and sanitization
- Immutability enforcement
- Structured error reporting from validate()
- Security-oriented input sanitization (control chars, length limits)
- Serialization round-trips (to_dict / from_dict / to_json / from_json)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from jules_daemon.models.command_request import (
    CommandRequest,
    FieldError,
    ValidationResult,
    validate_command_request,
    MAX_NL_COMMAND_LENGTH,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_LENGTH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_input() -> dict[str, Any]:
    """Return a minimal valid input dict for validate_command_request."""
    return {
        "natural_language_command": "run the full test suite",
        "target_host": "staging.example.com",
    }


def _full_input() -> dict[str, Any]:
    """Return a fully-populated valid input dict."""
    return {
        "command_id": str(uuid.uuid4()),
        "natural_language_command": "run integration tests on staging",
        "target_host": "staging.example.com",
        "target_user": "deploy",
        "target_port": 2222,
        "metadata": {"ci_pipeline": "nightly", "branch": "main"},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ===========================================================================
# CommandRequest model -- direct construction
# ===========================================================================


class TestCommandRequestConstruction:
    """Tests for direct CommandRequest construction."""

    def test_minimal_construction(self) -> None:
        req = CommandRequest(
            natural_language_command="run all tests",
            target_host="host.example.com",
        )
        assert req.natural_language_command == "run all tests"
        assert req.target_host == "host.example.com"
        assert req.target_user == ""
        assert req.target_port == 22
        assert req.metadata == {}
        assert req.command_id  # auto-generated UUID
        assert req.created_at  # auto-generated timestamp

    def test_full_construction(self) -> None:
        cid = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)
        req = CommandRequest(
            command_id=cid,
            natural_language_command="run integration tests",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=2222,
            metadata={"branch": "main"},
            created_at=ts,
        )
        assert req.command_id == cid
        assert req.natural_language_command == "run integration tests"
        assert req.target_host == "staging.example.com"
        assert req.target_user == "deploy"
        assert req.target_port == 2222
        assert req.metadata == {"branch": "main"}
        assert req.created_at == ts

    def test_auto_generated_command_id_is_valid_uuid(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        # Should not raise
        uuid.UUID(req.command_id)

    def test_auto_generated_created_at_is_utc(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        assert req.created_at.tzinfo is not None


class TestCommandRequestImmutability:
    """Tests that CommandRequest instances are frozen/immutable."""

    def test_frozen_model(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        with pytest.raises(Exception):
            req.natural_language_command = "modified"  # type: ignore[misc]

    def test_with_changes_returns_new_instance(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        updated = req.with_changes(target_host="new-host.example.com")
        assert updated.target_host == "new-host.example.com"
        assert req.target_host == "host.example.com"
        assert updated.command_id == req.command_id  # preserves other fields


# ===========================================================================
# Field validation -- natural_language_command
# ===========================================================================


class TestNaturalLanguageCommandValidation:
    """Tests for natural_language_command field validation."""

    def test_whitespace_stripped(self) -> None:
        req = CommandRequest(
            natural_language_command="  run all tests  ",
            target_host="host.example.com",
        )
        assert req.natural_language_command == "run all tests"

    def test_empty_raises(self) -> None:
        with pytest.raises(
            Exception, match="natural_language_command.*empty"
        ):
            CommandRequest(
                natural_language_command="",
                target_host="host.example.com",
            )

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(
            Exception, match="natural_language_command.*empty"
        ):
            CommandRequest(
                natural_language_command="   ",
                target_host="host.example.com",
            )

    def test_exceeds_max_length_raises(self) -> None:
        long_cmd = "a" * (MAX_NL_COMMAND_LENGTH + 1)
        with pytest.raises(Exception, match="exceed"):
            CommandRequest(
                natural_language_command=long_cmd,
                target_host="host.example.com",
            )

    def test_at_max_length_succeeds(self) -> None:
        cmd = "a" * MAX_NL_COMMAND_LENGTH
        req = CommandRequest(
            natural_language_command=cmd,
            target_host="host.example.com",
        )
        assert len(req.natural_language_command) == MAX_NL_COMMAND_LENGTH

    def test_control_characters_stripped(self) -> None:
        """Control characters (null bytes, ANSI escapes) are removed."""
        req = CommandRequest(
            natural_language_command="run\x00 all\x07 tests",
            target_host="host.example.com",
        )
        assert "\x00" not in req.natural_language_command
        assert "\x07" not in req.natural_language_command
        assert req.natural_language_command == "run all tests"


# ===========================================================================
# Field validation -- target_host
# ===========================================================================


class TestTargetHostValidation:
    """Tests for target_host field validation."""

    def test_whitespace_stripped(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="  host.example.com  ",
        )
        assert req.target_host == "host.example.com"

    def test_empty_raises(self) -> None:
        with pytest.raises(Exception, match="target_host.*empty"):
            CommandRequest(
                natural_language_command="run tests",
                target_host="",
            )

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(Exception, match="target_host.*empty"):
            CommandRequest(
                natural_language_command="run tests",
                target_host="   ",
            )

    def test_valid_hostname(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="staging-01.example.com",
        )
        assert req.target_host == "staging-01.example.com"

    def test_valid_ip_address(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="192.168.1.100",
        )
        assert req.target_host == "192.168.1.100"

    def test_control_characters_stripped(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host\x00.example.com",
        )
        assert "\x00" not in req.target_host


# ===========================================================================
# Field validation -- target_port
# ===========================================================================


class TestTargetPortValidation:
    """Tests for target_port field validation."""

    def test_default_is_22(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        assert req.target_port == 22

    def test_valid_port(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
            target_port=2222,
        )
        assert req.target_port == 2222

    def test_port_zero_raises(self) -> None:
        with pytest.raises(Exception, match="port.*1.*65535"):
            CommandRequest(
                natural_language_command="run tests",
                target_host="host.example.com",
                target_port=0,
            )

    def test_port_negative_raises(self) -> None:
        with pytest.raises(Exception, match="port.*1.*65535"):
            CommandRequest(
                natural_language_command="run tests",
                target_host="host.example.com",
                target_port=-1,
            )

    def test_port_above_65535_raises(self) -> None:
        with pytest.raises(Exception, match="port.*1.*65535"):
            CommandRequest(
                natural_language_command="run tests",
                target_host="host.example.com",
                target_port=65536,
            )

    def test_port_1_valid(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
            target_port=1,
        )
        assert req.target_port == 1

    def test_port_65535_valid(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
            target_port=65535,
        )
        assert req.target_port == 65535


# ===========================================================================
# Field validation -- metadata
# ===========================================================================


class TestMetadataValidation:
    """Tests for metadata field validation."""

    def test_empty_metadata_valid(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
            metadata={},
        )
        assert req.metadata == {}

    def test_valid_metadata(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
            metadata={"branch": "main", "ci_run": "12345"},
        )
        assert req.metadata == {"branch": "main", "ci_run": "12345"}

    def test_too_many_keys_raises(self) -> None:
        big_meta = {f"key_{i}": f"val_{i}" for i in range(MAX_METADATA_KEYS + 1)}
        with pytest.raises(Exception, match="metadata.*keys"):
            CommandRequest(
                natural_language_command="run tests",
                target_host="host.example.com",
                metadata=big_meta,
            )

    def test_value_exceeds_max_length_raises(self) -> None:
        long_value = "x" * (MAX_METADATA_VALUE_LENGTH + 1)
        with pytest.raises(Exception, match="metadata.*value.*exceed"):
            CommandRequest(
                natural_language_command="run tests",
                target_host="host.example.com",
                metadata={"key": long_value},
            )

    def test_control_characters_in_values_stripped(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
            metadata={"key": "value\x00with\x07control"},
        )
        assert "\x00" not in req.metadata["key"]
        assert "\x07" not in req.metadata["key"]

    def test_metadata_keys_are_alphanumeric(self) -> None:
        """Metadata keys must be alphanumeric with underscores/hyphens."""
        with pytest.raises(Exception, match="metadata.*key"):
            CommandRequest(
                natural_language_command="run tests",
                target_host="host.example.com",
                metadata={"invalid key!": "value"},
            )


# ===========================================================================
# Field validation -- command_id
# ===========================================================================


class TestCommandIdValidation:
    """Tests for command_id field validation."""

    def test_auto_generated_when_not_provided(self) -> None:
        result = validate_command_request(_valid_input())
        assert result.is_valid
        assert result.command is not None
        uuid.UUID(result.command.command_id)  # must be valid UUID

    def test_explicit_valid_uuid(self) -> None:
        cid = str(uuid.uuid4())
        data = {**_valid_input(), "command_id": cid}
        result = validate_command_request(data)
        assert result.is_valid
        assert result.command is not None
        assert result.command.command_id == cid


# ===========================================================================
# Serialization
# ===========================================================================


class TestSerialization:
    """Tests for to_dict / from_dict / to_json / from_json round-trips."""

    def test_to_dict(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
            target_user="deploy",
            target_port=2222,
            metadata={"branch": "main"},
        )
        d = req.to_dict()
        assert d["natural_language_command"] == "run tests"
        assert d["target_host"] == "host.example.com"
        assert d["target_user"] == "deploy"
        assert d["target_port"] == 2222
        assert d["metadata"] == {"branch": "main"}
        assert "command_id" in d
        assert "created_at" in d

    def test_from_dict_round_trip(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
            metadata={"branch": "main"},
        )
        d = req.to_dict()
        restored = CommandRequest.from_dict(d)
        assert restored.natural_language_command == req.natural_language_command
        assert restored.target_host == req.target_host
        assert restored.command_id == req.command_id
        assert restored.metadata == req.metadata

    def test_to_json(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        j = req.to_json()
        parsed = json.loads(j)
        assert parsed["natural_language_command"] == "run tests"

    def test_from_json_round_trip(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        j = req.to_json()
        restored = CommandRequest.from_json(j)
        assert restored.command_id == req.command_id
        assert restored.natural_language_command == req.natural_language_command
        assert restored.target_host == req.target_host


# ===========================================================================
# validate_command_request -- success cases
# ===========================================================================


class TestValidateSuccess:
    """Tests for the validate_command_request function -- success paths."""

    def test_minimal_valid_input(self) -> None:
        result = validate_command_request(_valid_input())
        assert result.is_valid
        assert result.command is not None
        assert len(result.errors) == 0
        assert result.command.natural_language_command == "run the full test suite"
        assert result.command.target_host == "staging.example.com"

    def test_full_valid_input(self) -> None:
        result = validate_command_request(_full_input())
        assert result.is_valid
        assert result.command is not None
        assert result.command.target_user == "deploy"
        assert result.command.target_port == 2222
        assert result.command.metadata == {
            "ci_pipeline": "nightly",
            "branch": "main",
        }

    def test_whitespace_stripped_in_validate(self) -> None:
        data = {
            "natural_language_command": "  run tests  ",
            "target_host": "  host.example.com  ",
        }
        result = validate_command_request(data)
        assert result.is_valid
        assert result.command is not None
        assert result.command.natural_language_command == "run tests"
        assert result.command.target_host == "host.example.com"


# ===========================================================================
# validate_command_request -- error cases
# ===========================================================================


class TestValidateErrors:
    """Tests for the validate_command_request function -- error paths."""

    def test_missing_natural_language_command(self) -> None:
        data = {"target_host": "host.example.com"}
        result = validate_command_request(data)
        assert not result.is_valid
        assert result.command is None
        assert any(
            e.field == "natural_language_command" for e in result.errors
        )

    def test_missing_target_host(self) -> None:
        data = {"natural_language_command": "run tests"}
        result = validate_command_request(data)
        assert not result.is_valid
        assert result.command is None
        assert any(e.field == "target_host" for e in result.errors)

    def test_empty_natural_language_command(self) -> None:
        data = {"natural_language_command": "", "target_host": "host.example.com"}
        result = validate_command_request(data)
        assert not result.is_valid
        assert any(
            e.field == "natural_language_command" for e in result.errors
        )

    def test_empty_target_host(self) -> None:
        data = {"natural_language_command": "run tests", "target_host": ""}
        result = validate_command_request(data)
        assert not result.is_valid
        assert any(e.field == "target_host" for e in result.errors)

    def test_multiple_errors_collected(self) -> None:
        data: dict[str, Any] = {}
        result = validate_command_request(data)
        assert not result.is_valid
        assert len(result.errors) >= 2
        fields = {e.field for e in result.errors}
        assert "natural_language_command" in fields
        assert "target_host" in fields

    def test_invalid_port_type(self) -> None:
        data = {
            **_valid_input(),
            "target_port": "not-a-port",
        }
        result = validate_command_request(data)
        assert not result.is_valid
        assert any(e.field == "target_port" for e in result.errors)

    def test_invalid_port_range(self) -> None:
        data = {
            **_valid_input(),
            "target_port": 99999,
        }
        result = validate_command_request(data)
        assert not result.is_valid
        assert any(e.field == "target_port" for e in result.errors)

    def test_metadata_not_dict(self) -> None:
        data = {
            **_valid_input(),
            "metadata": "not-a-dict",
        }
        result = validate_command_request(data)
        assert not result.is_valid
        assert any(e.field == "metadata" for e in result.errors)

    def test_unknown_fields_ignored(self) -> None:
        """Extra fields in input are silently ignored."""
        data = {
            **_valid_input(),
            "unknown_field": "should be ignored",
        }
        result = validate_command_request(data)
        assert result.is_valid


# ===========================================================================
# FieldError model
# ===========================================================================


class TestFieldError:
    """Tests for the FieldError frozen dataclass."""

    def test_construction(self) -> None:
        err = FieldError(
            field="target_host",
            message="must not be empty",
            code="required",
        )
        assert err.field == "target_host"
        assert err.message == "must not be empty"
        assert err.code == "required"

    def test_frozen(self) -> None:
        err = FieldError(field="f", message="m", code="c")
        with pytest.raises(AttributeError):
            err.field = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        err = FieldError(field="f", message="m", code="c")
        d = err.to_dict()
        assert d == {"field": "f", "message": "m", "code": "c"}


# ===========================================================================
# ValidationResult model
# ===========================================================================


class TestValidationResult:
    """Tests for the ValidationResult frozen dataclass."""

    def test_valid_result(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        result = ValidationResult(command=req, errors=())
        assert result.is_valid
        assert result.command is req
        assert len(result.errors) == 0

    def test_invalid_result(self) -> None:
        err = FieldError(field="f", message="m", code="c")
        result = ValidationResult(command=None, errors=(err,))
        assert not result.is_valid
        assert result.command is None
        assert len(result.errors) == 1

    def test_frozen(self) -> None:
        result = ValidationResult(command=None, errors=())
        with pytest.raises(AttributeError):
            result.command = None  # type: ignore[misc]

    def test_error_messages(self) -> None:
        errs = (
            FieldError(field="f1", message="m1", code="c1"),
            FieldError(field="f2", message="m2", code="c2"),
        )
        result = ValidationResult(command=None, errors=errs)
        messages = result.error_messages()
        assert messages == ("f1: m1", "f2: m2")

    def test_to_dict_valid(self) -> None:
        req = CommandRequest(
            natural_language_command="run tests",
            target_host="host.example.com",
        )
        result = ValidationResult(command=req, errors=())
        d = result.to_dict()
        assert d["is_valid"] is True
        assert d["command"] is not None
        assert d["errors"] == []

    def test_to_dict_invalid(self) -> None:
        err = FieldError(field="f", message="m", code="c")
        result = ValidationResult(command=None, errors=(err,))
        d = result.to_dict()
        assert d["is_valid"] is False
        assert d["command"] is None
        assert len(d["errors"]) == 1
