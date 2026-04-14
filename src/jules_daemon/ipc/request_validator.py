"""IPC request payload validation layer.

Validates incoming ``MessageEnvelope`` payloads before they reach the
command dispatch pipeline. Checks:

1. Message type is REQUEST (not RESPONSE, STREAM, etc.)
2. Payload contains a recognized ``verb`` field
3. Verb-specific required fields are present and valid
4. Optional fields (when present) conform to expected constraints

The validator accumulates all errors (does not short-circuit on the first
failure) so the CLI can display a complete list of issues in a single
round trip.

Usage::

    from jules_daemon.ipc.request_validator import validate_request

    result = validate_request(envelope)
    if result.is_valid:
        # proceed with dispatch
        verb = result.verb
        payload = result.parsed_payload
    else:
        # return error response with result.errors_to_dicts()
        ...
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jules_daemon.ipc.framing import MessageEnvelope, MessageType
from jules_daemon.protocol.notifications import parse_notification_event_type

__all__ = [
    "ValidationError",
    "ValidationResult",
    "validate_request",
]


# ---------------------------------------------------------------------------
# Valid verbs (kept in sync with cli/verbs.py Verb enum)
# ---------------------------------------------------------------------------

_VALID_VERBS: frozenset[str] = frozenset({
    "status", "watch", "run", "queue", "cancel", "history", "handshake",
    "discover", "subscribe_notifications", "unsubscribe_notifications",
})

# Valid output formats for the watch verb
_VALID_OUTPUT_FORMATS: frozenset[str] = frozenset({"text", "json", "summary"})

# Valid status filter values for the history verb
_VALID_STATUS_FILTERS: frozenset[str] = frozenset({
    "idle", "pending_approval", "running", "completed", "failed", "cancelled",
})

# Maximum history limit
_MAX_HISTORY_LIMIT: int = 1000


# ---------------------------------------------------------------------------
# ValidationError dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationError:
    """A single field-level validation error.

    Attributes:
        field:   Name of the field that failed validation.
        message: Human-readable description of the error.
        code:    Machine-readable error code for programmatic handling.
    """

    field: str
    message: str
    code: str

    def __post_init__(self) -> None:
        if not self.field or not self.field.strip():
            raise ValueError("field must not be empty")
        if not self.message or not self.message.strip():
            raise ValueError("message must not be empty")
        if not self.code or not self.code.strip():
            raise ValueError("code must not be empty")

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain dict for JSON responses.

        Returns:
            Dict with field, message, and code keys.
        """
        return {
            "field": self.field,
            "message": self.message,
            "code": self.code,
        }


# ---------------------------------------------------------------------------
# ValidationResult dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Immutable result of request payload validation.

    Attributes:
        is_valid:       True if the request passed all validation checks.
        verb:           Normalized canonical verb (None if verb is invalid).
        errors:         Tuple of all accumulated validation errors.
        parsed_payload: Cleaned and validated payload fields (empty dict
                        if validation failed).
    """

    is_valid: bool
    verb: str | None
    errors: tuple[ValidationError, ...]
    parsed_payload: dict[str, Any] = field(default_factory=dict)

    def errors_to_dicts(self) -> list[dict[str, str]]:
        """Convert all errors to a list of plain dicts.

        Returns:
            List of dicts, each with field, message, and code keys.
        """
        return [e.to_dict() for e in self.errors]


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _validate_message_type(
    envelope: MessageEnvelope,
    errors: list[ValidationError],
) -> bool:
    """Check that the envelope is a REQUEST type.

    Appends a ValidationError if the type is wrong.

    Returns:
        True if the message type is valid.
    """
    if envelope.msg_type != MessageType.REQUEST:
        errors.append(ValidationError(
            field="msg_type",
            message=(
                f"Expected message type 'request', "
                f"got '{envelope.msg_type.value}'"
            ),
            code="invalid_message_type",
        ))
        return False
    return True


def _extract_verb(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> str | None:
    """Extract and normalize the verb from the payload.

    Appends ValidationErrors for missing, empty, non-string, or
    unrecognized verb values.

    Returns:
        Normalized verb string, or None if invalid.
    """
    raw_verb = payload.get("verb")

    if raw_verb is None:
        errors.append(ValidationError(
            field="verb",
            message="The 'verb' field is required",
            code="missing_field",
        ))
        return None

    if not isinstance(raw_verb, str):
        errors.append(ValidationError(
            field="verb",
            message=f"verb must be a string, got {type(raw_verb).__name__}",
            code="invalid_verb",
        ))
        return None

    normalized = raw_verb.strip().lower()
    if not normalized:
        errors.append(ValidationError(
            field="verb",
            message="verb must not be empty",
            code="invalid_verb",
        ))
        return None

    if normalized not in _VALID_VERBS:
        valid_list = ", ".join(sorted(_VALID_VERBS))
        errors.append(ValidationError(
            field="verb",
            message=f"Unknown verb '{normalized}'. Valid verbs: {valid_list}",
            code="unknown_verb",
        ))
        return None

    return normalized


def _require_non_empty_string(
    payload: dict[str, Any],
    field_name: str,
    errors: list[ValidationError],
) -> str | None:
    """Validate that a required string field is present and non-empty.

    Returns:
        The stripped string value, or None if invalid.
    """
    value = payload.get(field_name)

    if value is None:
        errors.append(ValidationError(
            field=field_name,
            message=f"'{field_name}' is required",
            code="missing_field",
        ))
        return None

    if not isinstance(value, str):
        errors.append(ValidationError(
            field=field_name,
            message=f"'{field_name}' must be a string",
            code="invalid_type",
        ))
        return None

    stripped = value.strip()
    if not stripped:
        errors.append(ValidationError(
            field=field_name,
            message=f"'{field_name}' must not be empty",
            code="empty_field",
        ))
        return None

    return stripped


def _validate_optional_non_empty_string(
    payload: dict[str, Any],
    field_name: str,
    errors: list[ValidationError],
) -> str | None:
    """Validate an optional string field -- if present, must be non-empty.

    Returns:
        The stripped string value, or None if not present.
    """
    value = payload.get(field_name)
    if value is None:
        return None

    if not isinstance(value, str):
        errors.append(ValidationError(
            field=field_name,
            message=f"'{field_name}' must be a string",
            code="invalid_type",
        ))
        return None

    stripped = value.strip()
    if not stripped:
        errors.append(ValidationError(
            field=field_name,
            message=f"'{field_name}' must not be empty when provided",
            code="empty_field",
        ))
        return None

    return stripped


def _validate_port(
    payload: dict[str, Any],
    field_name: str,
    errors: list[ValidationError],
    default: int = 22,
) -> int:
    """Validate an optional port number (1-65535).

    Returns:
        Validated port number, or the default.
    """
    value = payload.get(field_name)
    if value is None:
        return default

    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(ValidationError(
            field=field_name,
            message=f"'{field_name}' must be an integer",
            code="invalid_type",
        ))
        return default

    if not (1 <= value <= 65535):
        errors.append(ValidationError(
            field=field_name,
            message=f"'{field_name}' must be between 1 and 65535, got {value}",
            code="out_of_range",
        ))
        return default

    return value


def _validate_key_path(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> str | None:
    """Validate optional key_path -- must be a safe absolute path if present.

    Rejects relative paths, paths containing traversal segments (``..``),
    and embedded null bytes to prevent path injection attacks.
    """
    value = payload.get("key_path")
    if value is None:
        return None

    if not isinstance(value, str):
        errors.append(ValidationError(
            field="key_path",
            message="'key_path' must be a string",
            code="invalid_type",
        ))
        return None

    # Reject null bytes (path injection)
    if "\x00" in value:
        errors.append(ValidationError(
            field="key_path",
            message="'key_path' must not contain null bytes",
            code="invalid_path",
        ))
        return None

    if not value.startswith("/"):
        errors.append(ValidationError(
            field="key_path",
            message="'key_path' must be an absolute path (starting with /)",
            code="invalid_path",
        ))
        return None

    # Reject path traversal segments in the raw input (before normalization
    # resolves them). This catches "/home/user/../../etc/passwd" even
    # though normpath would resolve it to "/etc/passwd".
    if ".." in value.split("/"):
        errors.append(ValidationError(
            field="key_path",
            message="'key_path' must not contain path traversal segments (..)",
            code="invalid_path",
        ))
        return None

    # Normalize the path (collapse redundant separators and dots)
    normalized = os.path.normpath(value)
    return normalized


# ---------------------------------------------------------------------------
# Verb-specific validators
# ---------------------------------------------------------------------------


def _validate_run_fields(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> dict[str, Any]:
    """Validate run verb required and optional fields.

    Returns:
        Parsed payload dict with validated fields.
    """
    parsed: dict[str, Any] = {}

    nl = _require_non_empty_string(payload, "natural_language", errors)
    if nl is not None:
        parsed["natural_language"] = nl

    system_name = _validate_optional_non_empty_string(payload, "system_name", errors)
    infer_target = payload.get("infer_target") is True
    interpret_request = payload.get("interpret_request") is True
    if system_name is not None:
        parsed["system_name"] = system_name

        conflicting_fields = [
            field_name
            for field_name in (
                "target_host",
                "target_user",
                "target_port",
                "key_path",
                "infer_target",
                "interpret_request",
            )
            if payload.get(field_name) is not None
        ]
        if conflicting_fields:
            errors.append(ValidationError(
                field="system_name",
                message=(
                    "'system_name' cannot be combined with explicit target "
                    f"fields: {', '.join(conflicting_fields)}"
                ),
                code="conflicting_fields",
            ))
        return parsed

    if infer_target:
        parsed["infer_target"] = True
        conflicting_fields = [
            field_name
            for field_name in (
                "target_host",
                "target_user",
                "target_port",
                "key_path",
                "interpret_request",
            )
            if payload.get(field_name) is not None
        ]
        if conflicting_fields:
            errors.append(ValidationError(
                field="infer_target",
                message=(
                    "'infer_target' cannot be combined with explicit target "
                    f"fields: {', '.join(conflicting_fields)}"
                ),
                code="conflicting_fields",
            ))
        return parsed

    if interpret_request:
        parsed["interpret_request"] = True
        conflicting_fields = [
            field_name
            for field_name in (
                "target_host",
                "target_user",
                "target_port",
                "key_path",
                "infer_target",
            )
            if payload.get(field_name) is not None
        ]
        if conflicting_fields:
            errors.append(ValidationError(
                field="interpret_request",
                message=(
                    "'interpret_request' cannot be combined with explicit target "
                    f"fields: {', '.join(conflicting_fields)}"
                ),
                code="conflicting_fields",
            ))
        return parsed

    host = _require_non_empty_string(payload, "target_host", errors)
    if host is not None:
        parsed["target_host"] = host

    user = _require_non_empty_string(payload, "target_user", errors)
    if user is not None:
        parsed["target_user"] = user

    parsed["target_port"] = _validate_port(payload, "target_port", errors)

    key_path = _validate_key_path(payload, errors)
    if key_path is not None:
        parsed["key_path"] = key_path

    return parsed


def _validate_queue_fields(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> dict[str, Any]:
    """Validate queue verb required and optional fields.

    Returns:
        Parsed payload dict with validated fields.
    """
    parsed = _validate_run_fields(payload, errors)

    # Optional priority field
    priority = payload.get("priority")
    if priority is not None:
        if not isinstance(priority, int) or isinstance(priority, bool):
            errors.append(ValidationError(
                field="priority",
                message="'priority' must be an integer",
                code="invalid_type",
            ))
        elif priority < 0:
            errors.append(ValidationError(
                field="priority",
                message=f"'priority' must not be negative, got {priority}",
                code="out_of_range",
            ))
        else:
            parsed["priority"] = priority

    return parsed


def _validate_cancel_fields(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> dict[str, Any]:
    """Validate cancel verb optional fields."""
    parsed: dict[str, Any] = {}

    run_id = _validate_optional_non_empty_string(payload, "run_id", errors)
    if run_id is not None:
        parsed["run_id"] = run_id

    force = payload.get("force")
    if force is not None:
        if not isinstance(force, bool):
            errors.append(ValidationError(
                field="force",
                message="'force' must be a boolean",
                code="invalid_type",
            ))
        else:
            parsed["force"] = force

    reason = _validate_optional_non_empty_string(payload, "reason", errors)
    if reason is not None:
        parsed["reason"] = reason

    return parsed


def _validate_watch_fields(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> dict[str, Any]:
    """Validate watch verb optional fields."""
    parsed: dict[str, Any] = {}

    run_id = _validate_optional_non_empty_string(payload, "run_id", errors)
    if run_id is not None:
        parsed["run_id"] = run_id

    tail_lines = payload.get("tail_lines")
    if tail_lines is not None:
        if not isinstance(tail_lines, int) or isinstance(tail_lines, bool):
            errors.append(ValidationError(
                field="tail_lines",
                message="'tail_lines' must be an integer",
                code="invalid_type",
            ))
        elif tail_lines < 1:
            errors.append(ValidationError(
                field="tail_lines",
                message=f"'tail_lines' must be positive, got {tail_lines}",
                code="out_of_range",
            ))
        else:
            parsed["tail_lines"] = tail_lines

    follow = payload.get("follow")
    if follow is not None:
        if not isinstance(follow, bool):
            errors.append(ValidationError(
                field="follow",
                message="'follow' must be a boolean",
                code="invalid_type",
            ))
        else:
            parsed["follow"] = follow

    output_format = payload.get("output_format")
    if output_format is not None:
        if not isinstance(output_format, str):
            errors.append(ValidationError(
                field="output_format",
                message="'output_format' must be a string",
                code="invalid_type",
            ))
        elif output_format not in _VALID_OUTPUT_FORMATS:
            valid_list = ", ".join(sorted(_VALID_OUTPUT_FORMATS))
            errors.append(ValidationError(
                field="output_format",
                message=(
                    f"'output_format' must be one of {valid_list}, "
                    f"got '{output_format}'"
                ),
                code="invalid_value",
            ))
        else:
            parsed["output_format"] = output_format

    return parsed


def _validate_history_fields(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> dict[str, Any]:
    """Validate history verb optional fields."""
    parsed: dict[str, Any] = {}

    limit = payload.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool):
            errors.append(ValidationError(
                field="limit",
                message="'limit' must be an integer",
                code="invalid_type",
            ))
        elif limit < 1:
            errors.append(ValidationError(
                field="limit",
                message=f"'limit' must be positive, got {limit}",
                code="out_of_range",
            ))
        elif limit > _MAX_HISTORY_LIMIT:
            errors.append(ValidationError(
                field="limit",
                message=(
                    f"'limit' must not exceed {_MAX_HISTORY_LIMIT}, "
                    f"got {limit}"
                ),
                code="out_of_range",
            ))
        else:
            parsed["limit"] = limit

    status_filter = payload.get("status_filter")
    if status_filter is not None:
        if not isinstance(status_filter, str):
            errors.append(ValidationError(
                field="status_filter",
                message="'status_filter' must be a string",
                code="invalid_type",
            ))
        elif not status_filter.strip():
            errors.append(ValidationError(
                field="status_filter",
                message="'status_filter' must not be empty when provided",
                code="empty_field",
            ))
        elif status_filter.strip() not in _VALID_STATUS_FILTERS:
            valid_list = ", ".join(sorted(_VALID_STATUS_FILTERS))
            errors.append(ValidationError(
                field="status_filter",
                message=(
                    f"Invalid status_filter '{status_filter.strip()}'. "
                    f"Valid values: {valid_list}"
                ),
                code="invalid_value",
            ))
        else:
            parsed["status_filter"] = status_filter.strip()

    host_filter = _validate_optional_non_empty_string(
        payload, "host_filter", errors
    )
    if host_filter is not None:
        parsed["host_filter"] = host_filter

    verbose = payload.get("verbose")
    if verbose is not None:
        if not isinstance(verbose, bool):
            errors.append(ValidationError(
                field="verbose",
                message="'verbose' must be a boolean",
                code="invalid_type",
            ))
        else:
            parsed["verbose"] = verbose

    return parsed


def _validate_status_fields(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> dict[str, Any]:
    """Validate status verb optional fields."""
    parsed: dict[str, Any] = {}

    verbose = payload.get("verbose")
    if verbose is not None:
        if not isinstance(verbose, bool):
            errors.append(ValidationError(
                field="verbose",
                message="'verbose' must be a boolean",
                code="invalid_type",
            ))
        else:
            parsed["verbose"] = verbose

    return parsed


def _validate_discover_fields(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> dict[str, Any]:
    """Validate discover verb required and optional fields.

    Returns:
        Parsed payload dict with validated fields.
    """
    parsed: dict[str, Any] = {}

    host = _require_non_empty_string(payload, "target_host", errors)
    if host is not None:
        parsed["target_host"] = host

    user = _require_non_empty_string(payload, "target_user", errors)
    if user is not None:
        parsed["target_user"] = user

    command = _require_non_empty_string(payload, "command", errors)
    if command is not None:
        parsed["command"] = command

    parsed["target_port"] = _validate_port(payload, "target_port", errors)

    return parsed


def _validate_subscribe_notification_fields(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> dict[str, Any]:
    """Validate subscribe_notifications optional fields."""
    parsed: dict[str, Any] = {}

    raw_filter = payload.get("event_filter")
    if raw_filter is None:
        return parsed

    if not isinstance(raw_filter, (list, tuple)):
        errors.append(ValidationError(
            field="event_filter",
            message="'event_filter' must be a list of notification event types",
            code="invalid_type",
        ))
        return parsed

    event_types = []
    for index, raw_item in enumerate(raw_filter):
        field_name = f"event_filter[{index}]"
        if not isinstance(raw_item, str):
            errors.append(ValidationError(
                field=field_name,
                message=f"'{field_name}' must be a string",
                code="invalid_type",
            ))
            continue
        try:
            event_types.append(parse_notification_event_type(raw_item))
        except ValueError as exc:
            errors.append(ValidationError(
                field=field_name,
                message=str(exc),
                code="invalid_value",
            ))

    parsed["event_filter"] = frozenset(event_types)
    return parsed


def _validate_unsubscribe_notification_fields(
    payload: dict[str, Any],
    errors: list[ValidationError],
) -> dict[str, Any]:
    """Validate unsubscribe_notifications required fields."""
    parsed: dict[str, Any] = {}
    subscription_id = _require_non_empty_string(
        payload, "subscription_id", errors,
    )
    if subscription_id is not None:
        parsed["subscription_id"] = subscription_id
    return parsed


# Type alias for verb-specific field validators
_VerbValidator = Callable[[dict[str, Any], list[ValidationError]], dict[str, Any]]

# Verb -> field validator mapping
_VERB_VALIDATORS: dict[str, _VerbValidator] = {
    "run": _validate_run_fields,
    "queue": _validate_queue_fields,
    "cancel": _validate_cancel_fields,
    "watch": _validate_watch_fields,
    "history": _validate_history_fields,
    "status": _validate_status_fields,
    "discover": _validate_discover_fields,
    "subscribe_notifications": _validate_subscribe_notification_fields,
    "unsubscribe_notifications": _validate_unsubscribe_notification_fields,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_request(envelope: MessageEnvelope) -> ValidationResult:
    """Validate an incoming IPC request envelope.

    Checks message type, verb, and verb-specific fields. Accumulates
    all errors so the caller can return a complete error report.

    Args:
        envelope: The incoming MessageEnvelope to validate.

    Returns:
        ValidationResult with is_valid, verb, errors, and parsed_payload.
    """
    errors: list[ValidationError] = []

    # 1. Check message type
    if not _validate_message_type(envelope, errors):
        return ValidationResult(
            is_valid=False,
            verb=None,
            errors=tuple(errors),
        )

    # 2. Extract and validate verb
    verb = _extract_verb(envelope.payload, errors)
    if verb is None:
        return ValidationResult(
            is_valid=False,
            verb=None,
            errors=tuple(errors),
        )

    # 3. Validate verb-specific fields
    validator = _VERB_VALIDATORS.get(verb)
    parsed_payload: dict[str, Any] = {}
    if validator is not None:
        parsed_payload = validator(envelope.payload, errors)

    # 4. Build result (deepcopy payload on both paths for immutability)
    if errors:
        return ValidationResult(
            is_valid=False,
            verb=verb,
            errors=tuple(errors),
            parsed_payload=copy.deepcopy(parsed_payload),
        )

    return ValidationResult(
        is_valid=True,
        verb=verb,
        errors=(),
        parsed_payload=copy.deepcopy(parsed_payload),
    )
