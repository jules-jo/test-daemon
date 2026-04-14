"""Tests for IPC request payload validation.

Validates that incoming MessageEnvelope payloads are structurally correct,
contain recognized verbs, and have the required fields for each verb.
"""

from __future__ import annotations

import pytest

from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.ipc.request_validator import (
    ValidationError,
    ValidationResult,
    validate_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    payload: dict,
    msg_type: MessageType = MessageType.REQUEST,
    msg_id: str = "test-001",
) -> MessageEnvelope:
    """Build a MessageEnvelope with the given payload."""
    return MessageEnvelope(
        msg_type=msg_type,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload=payload,
    )


# ---------------------------------------------------------------------------
# ValidationError model tests
# ---------------------------------------------------------------------------


class TestValidationError:
    """Tests for the immutable ValidationError dataclass."""

    def test_create(self) -> None:
        err = ValidationError(
            field="verb",
            message="verb is required",
            code="missing_field",
        )
        assert err.field == "verb"
        assert err.message == "verb is required"
        assert err.code == "missing_field"

    def test_frozen(self) -> None:
        err = ValidationError(
            field="verb",
            message="verb is required",
            code="missing_field",
        )
        with pytest.raises(AttributeError):
            err.field = "other"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        err = ValidationError(
            field="verb",
            message="verb is required",
            code="missing_field",
        )
        d = err.to_dict()
        assert d == {
            "field": "verb",
            "message": "verb is required",
            "code": "missing_field",
        }

    def test_empty_field_raises(self) -> None:
        with pytest.raises(ValueError, match="field must not be empty"):
            ValidationError(field="", message="msg", code="code")

    def test_empty_message_raises(self) -> None:
        with pytest.raises(ValueError, match="message must not be empty"):
            ValidationError(field="verb", message="", code="code")

    def test_empty_code_raises(self) -> None:
        with pytest.raises(ValueError, match="code must not be empty"):
            ValidationError(field="verb", message="msg", code="")


# ---------------------------------------------------------------------------
# ValidationResult model tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for the immutable ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        result = ValidationResult(
            is_valid=True,
            verb="status",
            errors=(),
        )
        assert result.is_valid is True
        assert result.verb == "status"
        assert result.errors == ()
        assert result.parsed_payload == {}

    def test_invalid_result_with_errors(self) -> None:
        err = ValidationError(
            field="verb",
            message="missing verb",
            code="missing_field",
        )
        result = ValidationResult(
            is_valid=False,
            verb=None,
            errors=(err,),
        )
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_frozen(self) -> None:
        result = ValidationResult(is_valid=True, verb="status", errors=())
        with pytest.raises(AttributeError):
            result.is_valid = False  # type: ignore[misc]

    def test_errors_to_dicts(self) -> None:
        err1 = ValidationError(field="verb", message="m1", code="c1")
        err2 = ValidationError(field="host", message="m2", code="c2")
        result = ValidationResult(
            is_valid=False,
            verb=None,
            errors=(err1, err2),
        )
        dicts = result.errors_to_dicts()
        assert len(dicts) == 2
        assert dicts[0]["field"] == "verb"
        assert dicts[1]["field"] == "host"

    def test_parsed_payload_included(self) -> None:
        result = ValidationResult(
            is_valid=True,
            verb="run",
            errors=(),
            parsed_payload={
                "target_host": "example.com",
                "target_user": "deploy",
                "natural_language": "run tests",
            },
        )
        assert result.parsed_payload["target_host"] == "example.com"


# ---------------------------------------------------------------------------
# validate_request: message type checks
# ---------------------------------------------------------------------------


class TestValidateRequestMessageType:
    """Tests for message type validation."""

    def test_non_request_type_is_invalid(self) -> None:
        envelope = _make_envelope(
            payload={"verb": "status"},
            msg_type=MessageType.RESPONSE,
        )
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.code == "invalid_message_type" for e in result.errors)

    def test_stream_type_is_invalid(self) -> None:
        envelope = _make_envelope(
            payload={"verb": "status"},
            msg_type=MessageType.STREAM,
        )
        result = validate_request(envelope)
        assert result.is_valid is False

    def test_error_type_is_invalid(self) -> None:
        envelope = _make_envelope(
            payload={"verb": "status"},
            msg_type=MessageType.ERROR,
        )
        result = validate_request(envelope)
        assert result.is_valid is False

    def test_request_type_is_accepted(self) -> None:
        envelope = _make_envelope(
            payload={"verb": "status"},
            msg_type=MessageType.REQUEST,
        )
        result = validate_request(envelope)
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# validate_request: verb validation
# ---------------------------------------------------------------------------


class TestValidateRequestVerb:
    """Tests for verb field validation."""

    def test_missing_verb_is_invalid(self) -> None:
        envelope = _make_envelope(payload={})
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.code == "missing_field" for e in result.errors)
        assert any(e.field == "verb" for e in result.errors)

    def test_empty_verb_is_invalid(self) -> None:
        envelope = _make_envelope(payload={"verb": ""})
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.code == "invalid_verb" for e in result.errors)

    def test_whitespace_verb_is_invalid(self) -> None:
        envelope = _make_envelope(payload={"verb": "   "})
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.code == "invalid_verb" for e in result.errors)

    def test_unknown_verb_is_invalid(self) -> None:
        envelope = _make_envelope(payload={"verb": "teleport"})
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.code == "unknown_verb" for e in result.errors)

    def test_non_string_verb_is_invalid(self) -> None:
        envelope = _make_envelope(payload={"verb": 42})
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.code == "invalid_verb" for e in result.errors)

    @pytest.mark.parametrize(
        "verb",
        [
            "status",
            "watch",
            "run",
            "queue",
            "cancel",
            "history",
            "subscribe_notifications",
            "unsubscribe_notifications",
        ],
    )
    def test_all_valid_verbs_accepted(self, verb: str) -> None:
        payload: dict = {"verb": verb}
        if verb in ("run", "queue"):
            payload.update({
                "target_host": "host.example.com",
                "target_user": "deploy",
                "natural_language": "run the tests",
            })
        elif verb == "unsubscribe_notifications":
            payload["subscription_id"] = "nsub-123"
        envelope = _make_envelope(payload=payload)
        result = validate_request(envelope)
        assert result.is_valid is True, f"verb {verb!r} should be valid: {result.errors}"
        assert result.verb == verb

    def test_case_insensitive_verb(self) -> None:
        envelope = _make_envelope(payload={"verb": "STATUS"})
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.verb == "status"

    def test_verb_with_whitespace_trimmed(self) -> None:
        envelope = _make_envelope(payload={"verb": "  status  "})
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.verb == "status"


class TestValidateRequestNotificationSubscriptionFields:
    """Tests for notification subscribe/unsubscribe validation."""

    def test_subscribe_event_filter_parsed(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "subscribe_notifications",
            "event_filter": ["completion", "ALERT"],
        })
        result = validate_request(envelope)
        assert result.is_valid is True
        assert len(result.parsed_payload["event_filter"]) == 2

    def test_subscribe_invalid_event_filter_type(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "subscribe_notifications",
            "event_filter": "completion",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "event_filter" for e in result.errors)

    def test_subscribe_invalid_event_filter_value(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "subscribe_notifications",
            "event_filter": ["completion", "telepathy"],
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "event_filter[1]" for e in result.errors)

    def test_unsubscribe_requires_subscription_id(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "unsubscribe_notifications",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "subscription_id" for e in result.errors)
        assert result.verb == "unsubscribe_notifications"


# ---------------------------------------------------------------------------
# validate_request: run verb field validation
# ---------------------------------------------------------------------------


class TestValidateRequestRunFields:
    """Tests for run verb required fields."""

    def test_run_missing_target_host(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_user": "deploy",
            "natural_language": "run tests",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "target_host" for e in result.errors)
        assert any(e.code == "missing_field" for e in result.errors)

    def test_run_missing_target_user(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "host.example.com",
            "natural_language": "run tests",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "target_user" for e in result.errors)

    def test_run_missing_natural_language(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "host.example.com",
            "target_user": "deploy",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "natural_language" for e in result.errors)

    def test_run_empty_target_host(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "",
            "target_user": "deploy",
            "natural_language": "run tests",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "target_host" for e in result.errors)

    def test_run_valid_complete(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run all unit tests",
        })
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.verb == "run"
        assert result.parsed_payload["target_host"] == "staging.example.com"

    def test_run_valid_with_system_name(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "system_name": "tuto",
            "natural_language": "run smoke tests",
        })
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.parsed_payload["system_name"] == "tuto"

    def test_run_valid_with_infer_target(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "infer_target": True,
            "natural_language": "run smoke tests in tuto",
        })
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.parsed_payload["infer_target"] is True

    def test_run_system_name_conflicts_with_explicit_target(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "system_name": "tuto",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run smoke tests",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.code == "conflicting_fields" for e in result.errors)

    def test_run_infer_target_conflicts_with_explicit_target(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "infer_target": True,
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run smoke tests in tuto",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.code == "conflicting_fields" for e in result.errors)

    def test_run_with_optional_fields(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run smoke tests",
            "target_port": 2222,
            "key_path": "/home/user/.ssh/id_rsa",
        })
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.parsed_payload["target_port"] == 2222
        assert result.parsed_payload["key_path"] == "/home/user/.ssh/id_rsa"

    def test_run_invalid_port(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "host.example.com",
            "target_user": "deploy",
            "natural_language": "run tests",
            "target_port": 0,
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "target_port" for e in result.errors)

    def test_run_port_too_high(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "host.example.com",
            "target_user": "deploy",
            "natural_language": "run tests",
            "target_port": 70000,
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "target_port" for e in result.errors)

    def test_run_relative_key_path_invalid(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "host.example.com",
            "target_user": "deploy",
            "natural_language": "run tests",
            "key_path": "relative/path/key",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "key_path" for e in result.errors)

    def test_run_key_path_traversal_rejected(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "host.example.com",
            "target_user": "deploy",
            "natural_language": "run tests",
            "key_path": "/home/user/../../etc/passwd",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(
            e.field == "key_path" and "traversal" in e.message
            for e in result.errors
        )

    def test_run_key_path_null_byte_rejected(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "host.example.com",
            "target_user": "deploy",
            "natural_language": "run tests",
            "key_path": "/home/user/.ssh/id_rsa\x00.evil",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(
            e.field == "key_path" and "null" in e.message
            for e in result.errors
        )

    def test_run_key_path_normalized(self) -> None:
        """Valid absolute key_path is normalized (extra slashes removed)."""
        envelope = _make_envelope(payload={
            "verb": "run",
            "target_host": "host.example.com",
            "target_user": "deploy",
            "natural_language": "run tests",
            "key_path": "/home/user/.ssh//id_rsa",
        })
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.parsed_payload["key_path"] == "/home/user/.ssh/id_rsa"


# ---------------------------------------------------------------------------
# validate_request: queue verb field validation
# ---------------------------------------------------------------------------


class TestValidateRequestQueueFields:
    """Tests for queue verb required fields."""

    def test_queue_missing_required_fields(self) -> None:
        envelope = _make_envelope(payload={"verb": "queue"})
        result = validate_request(envelope)
        assert result.is_valid is False
        field_names = {e.field for e in result.errors}
        assert "target_host" in field_names
        assert "target_user" in field_names
        assert "natural_language" in field_names

    def test_queue_valid_complete(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "queue",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run integration tests",
        })
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.verb == "queue"

    def test_queue_with_priority(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "queue",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run smoke tests",
            "priority": 20,
        })
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.parsed_payload["priority"] == 20

    def test_queue_negative_priority_invalid(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "queue",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run tests",
            "priority": -1,
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "priority" for e in result.errors)


# ---------------------------------------------------------------------------
# validate_request: cancel verb fields
# ---------------------------------------------------------------------------


class TestValidateRequestCancelFields:
    """Tests for cancel verb optional fields."""

    def test_cancel_no_args_valid(self) -> None:
        envelope = _make_envelope(payload={"verb": "cancel"})
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.verb == "cancel"

    def test_cancel_with_run_id(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "cancel",
            "run_id": "run-abc123",
        })
        result = validate_request(envelope)
        assert result.is_valid is True
        assert result.parsed_payload.get("run_id") == "run-abc123"

    def test_cancel_empty_run_id_invalid(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "cancel",
            "run_id": "",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "run_id" for e in result.errors)


# ---------------------------------------------------------------------------
# validate_request: watch verb fields
# ---------------------------------------------------------------------------


class TestValidateRequestWatchFields:
    """Tests for watch verb optional fields."""

    def test_watch_no_args_valid(self) -> None:
        envelope = _make_envelope(payload={"verb": "watch"})
        result = validate_request(envelope)
        assert result.is_valid is True

    def test_watch_invalid_output_format(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "watch",
            "output_format": "xml",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "output_format" for e in result.errors)

    def test_watch_zero_tail_lines_invalid(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "watch",
            "tail_lines": 0,
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "tail_lines" for e in result.errors)


# ---------------------------------------------------------------------------
# validate_request: history verb fields
# ---------------------------------------------------------------------------


class TestValidateRequestHistoryFields:
    """Tests for history verb optional fields."""

    def test_history_no_args_valid(self) -> None:
        envelope = _make_envelope(payload={"verb": "history"})
        result = validate_request(envelope)
        assert result.is_valid is True

    def test_history_invalid_status_filter(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "history",
            "status_filter": "dancing",
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "status_filter" for e in result.errors)

    def test_history_zero_limit_invalid(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "history",
            "limit": 0,
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "limit" for e in result.errors)

    def test_history_exceeds_max_limit(self) -> None:
        envelope = _make_envelope(payload={
            "verb": "history",
            "limit": 5000,
        })
        result = validate_request(envelope)
        assert result.is_valid is False
        assert any(e.field == "limit" for e in result.errors)


# ---------------------------------------------------------------------------
# validate_request: multiple errors accumulated
# ---------------------------------------------------------------------------


class TestValidateRequestMultipleErrors:
    """Tests for multiple validation errors in a single request."""

    def test_run_missing_all_required_fields(self) -> None:
        envelope = _make_envelope(payload={"verb": "run"})
        result = validate_request(envelope)
        assert result.is_valid is False
        assert len(result.errors) >= 3  # target_host, target_user, natural_language

    def test_errors_are_immutable_tuple(self) -> None:
        envelope = _make_envelope(payload={"verb": "run"})
        result = validate_request(envelope)
        assert isinstance(result.errors, tuple)
