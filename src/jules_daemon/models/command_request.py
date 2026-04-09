"""Command request model and validation for CLI-to-daemon requests.

Defines the ``CommandRequest`` model -- the inbound data contract for
natural-language command requests sent from the CLI to the daemon.
This is the first touchpoint in the command execution pipeline:

    CLI input --> CommandRequest --> LLM translation --> SSH execution

The model is frozen (immutable) and uses Pydantic for field validation
and input sanitization. A standalone ``validate_command_request()``
function provides structured error reporting for raw dict input,
collecting all validation errors rather than failing on the first one.

Security considerations:
    - Control characters (null bytes, bell, ANSI escapes) are stripped
      from all string fields to prevent injection via terminal output
    - String lengths are bounded to prevent resource exhaustion
    - Metadata keys are restricted to alphanumeric + underscore/hyphen
    - Metadata size is capped (key count and value length)

Usage::

    from jules_daemon.models.command_request import (
        CommandRequest,
        validate_command_request,
    )

    # Direct construction (raises on invalid input)
    req = CommandRequest(
        natural_language_command="run the full test suite",
        target_host="staging.example.com",
        target_user="deploy",
    )

    # Validation with structured errors (never raises)
    result = validate_command_request({
        "natural_language_command": "run all tests",
        "target_host": "staging.example.com",
    })
    if result.is_valid:
        print(result.command)
    else:
        for err in result.errors:
            print(f"{err.field}: {err.message}")
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    field_validator,
    model_validator,
)
from typing import Self

__all__ = [
    "CommandRequest",
    "FieldError",
    "ValidationResult",
    "validate_command_request",
    "MAX_NL_COMMAND_LENGTH",
    "MAX_METADATA_KEYS",
    "MAX_METADATA_VALUE_LENGTH",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NL_COMMAND_LENGTH: int = 2048
"""Maximum length for the natural language command string."""

MAX_METADATA_KEYS: int = 32
"""Maximum number of metadata key-value pairs."""

MAX_METADATA_VALUE_LENGTH: int = 1024
"""Maximum length for a single metadata value string."""

# Control character pattern: matches C0 control chars (0x00-0x1F) except
# common whitespace (tab=0x09, newline=0x0A, carriage return=0x0D),
# plus DEL (0x7F).
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)

# Metadata key pattern: letters, digits, underscores, hyphens.
_METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------


def _strip_control_chars(value: str) -> str:
    """Remove control characters from a string.

    Preserves tabs, newlines, and carriage returns since those appear
    in legitimate multi-line commands. Removes null bytes, bell,
    backspace, escape sequences, and other C0/DEL control characters.
    """
    return _CONTROL_CHAR_RE.sub("", value)


def _sanitize_string_field(value: str, field_name: str) -> str:
    """Strip whitespace and control characters, then validate non-empty.

    Returns the cleaned string. Raises ValueError if the result is empty.
    """
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = _strip_control_chars(value).strip()
    if not cleaned:
        raise ValueError(
            f"{field_name} must not be empty or whitespace-only"
        )
    return cleaned


# ---------------------------------------------------------------------------
# CommandRequest model
# ---------------------------------------------------------------------------


def _generate_command_id() -> str:
    """Generate a new UUID v4 command identifier."""
    return str(uuid.uuid4())


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


class CommandRequest(BaseModel):
    """Immutable command request from CLI to daemon.

    Represents a natural-language command targeting a remote SSH host.
    This is the entry point of the execution pipeline -- before LLM
    translation, risk classification, or human approval.

    Attributes:
        command_id: Unique identifier for this request (UUID v4).
            Auto-generated if not provided.
        natural_language_command: The user's natural-language description
            of what they want to execute (e.g., "run the full test suite").
        target_host: SSH hostname or IP address of the remote system.
        target_user: SSH username for the remote connection. Empty string
            means the daemon should use its configured default.
        target_port: SSH port number (1-65535, default 22).
        metadata: Arbitrary key-value pairs for additional context
            (e.g., CI pipeline ID, branch name, environment label).
        created_at: UTC timestamp when this request was created.
            Auto-generated if not provided.
    """

    model_config = ConfigDict(frozen=True, validate_default=True)

    command_id: str = ""
    natural_language_command: str
    target_host: str
    target_user: str = ""
    target_port: int = 22
    metadata: dict[str, str] = {}
    created_at: datetime = datetime.min

    # -- Field validators --

    @field_validator("command_id", mode="before")
    @classmethod
    def _default_command_id(cls, value: Any) -> str:
        """Auto-generate a UUID if not provided or empty."""
        if not value:
            return _generate_command_id()
        if not isinstance(value, str):
            raise ValueError("command_id must be a string")
        stripped = value.strip()
        if not stripped:
            return _generate_command_id()
        return stripped

    @field_validator("natural_language_command", mode="before")
    @classmethod
    def _sanitize_nl_command(cls, value: Any) -> str:
        """Strip whitespace, remove control characters, validate length."""
        cleaned = _sanitize_string_field(value, "natural_language_command")
        if len(cleaned) > MAX_NL_COMMAND_LENGTH:
            raise ValueError(
                f"natural_language_command must not exceed "
                f"{MAX_NL_COMMAND_LENGTH} characters, got {len(cleaned)}"
            )
        return cleaned

    @field_validator("target_host", mode="before")
    @classmethod
    def _sanitize_target_host(cls, value: Any) -> str:
        """Strip whitespace, remove control characters, validate non-empty."""
        return _sanitize_string_field(value, "target_host")

    @field_validator("target_user", mode="before")
    @classmethod
    def _sanitize_target_user(cls, value: Any) -> str:
        """Strip whitespace and control characters. Empty is allowed."""
        if not value:
            return ""
        if not isinstance(value, str):
            raise ValueError("target_user must be a string")
        return _strip_control_chars(value).strip()

    @field_validator("target_port", mode="before")
    @classmethod
    def _validate_target_port(cls, value: Any) -> int:
        """Validate port is an integer in 1-65535."""
        if isinstance(value, bool):
            raise ValueError("target_port must be an integer, not boolean")
        if not isinstance(value, int):
            raise ValueError("target_port must be an integer")
        if not (1 <= value <= 65535):
            raise ValueError(
                f"target_port must be 1-65535, got {value}"
            )
        return value

    @field_validator("created_at", mode="before")
    @classmethod
    def _default_created_at(cls, value: Any) -> datetime:
        """Auto-generate UTC timestamp if not provided or is the sentinel."""
        if value is None or value == datetime.min:
            return _now_utc()
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        if isinstance(value, datetime):
            return value
        raise ValueError("created_at must be a datetime or ISO 8601 string")

    @model_validator(mode="after")
    def _validate_metadata(self) -> Self:
        """Validate metadata keys and values after all fields are set."""
        meta = self.metadata
        if not isinstance(meta, dict):
            raise ValueError("metadata must be a dict")

        if len(meta) > MAX_METADATA_KEYS:
            raise ValueError(
                f"metadata must not have more than {MAX_METADATA_KEYS} keys, "
                f"got {len(meta)}"
            )

        sanitized: dict[str, str] = {}
        for key, val in meta.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"metadata key must be a string, got {type(key).__name__}"
                )
            if not _METADATA_KEY_RE.match(key):
                raise ValueError(
                    f"metadata key {key!r} must contain only letters, "
                    f"digits, underscores, or hyphens"
                )
            if not isinstance(val, str):
                raise ValueError(
                    f"metadata value for key {key!r} must be a string, "
                    f"got {type(val).__name__}"
                )
            cleaned_val = _strip_control_chars(val)
            if len(cleaned_val) > MAX_METADATA_VALUE_LENGTH:
                raise ValueError(
                    f"metadata value for key {key!r} must not exceed "
                    f"{MAX_METADATA_VALUE_LENGTH} characters, "
                    f"got {len(cleaned_val)}"
                )
            sanitized[key] = cleaned_val

        # Replace metadata with sanitized version if any values changed.
        # Pydantic frozen models require object.__setattr__ for post-init
        # mutation.
        if sanitized != meta:
            object.__setattr__(self, "metadata", sanitized)

        return self

    # -- Serialization helpers --

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for wiki YAML persistence.

        The created_at field is converted to an ISO 8601 string.
        """
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandRequest:
        """Deserialize from a plain dict (e.g., parsed from wiki YAML)."""
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> CommandRequest:
        """Deserialize from a JSON string."""
        return cls.model_validate_json(payload)

    # -- Immutable copy --

    def with_changes(self, **kwargs: Any) -> CommandRequest:
        """Return a new CommandRequest with the specified fields replaced.

        The original instance is never mutated. All validation rules
        are re-applied on the resulting instance.
        """
        current = self.model_dump()
        current.update(kwargs)
        return CommandRequest.model_validate(current)


# ---------------------------------------------------------------------------
# Structured validation errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldError:
    """A single field-level validation error.

    Attributes:
        field: Name of the field that failed validation.
        message: Human-readable description of the error.
        code: Machine-readable error code (e.g., "required", "invalid",
            "too_long", "sanitization_failed").
    """

    field: str
    message: str
    code: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to a plain dict."""
        return {
            "field": self.field,
            "message": self.message,
            "code": self.code,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a command request.

    Contains either a valid CommandRequest or a tuple of FieldErrors.
    Never both -- if errors is non-empty, command is None.

    Attributes:
        command: The validated CommandRequest, or None if validation failed.
        errors: Tuple of field-level errors. Empty tuple on success.
    """

    command: CommandRequest | None
    errors: tuple[FieldError, ...]

    @property
    def is_valid(self) -> bool:
        """True if validation succeeded with no errors."""
        return self.command is not None and len(self.errors) == 0

    def error_messages(self) -> tuple[str, ...]:
        """Return formatted error messages as (field: message) strings."""
        return tuple(f"{e.field}: {e.message}" for e in self.errors)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for IPC or logging."""
        return {
            "is_valid": self.is_valid,
            "command": self.command.to_dict() if self.command else None,
            "errors": [e.to_dict() for e in self.errors],
        }


# ---------------------------------------------------------------------------
# Validation function
# ---------------------------------------------------------------------------


def _check_required_string(
    data: dict[str, Any],
    field_name: str,
    errors: list[FieldError],
) -> str | None:
    """Check that a required string field exists and is non-empty.

    Appends a FieldError to ``errors`` and returns None if the check fails.
    Returns the raw value (un-sanitized) if the field exists.
    """
    value = data.get(field_name)
    if value is None:
        errors.append(
            FieldError(
                field=field_name,
                message=f"{field_name} is required",
                code="required",
            )
        )
        return None

    if not isinstance(value, str):
        errors.append(
            FieldError(
                field=field_name,
                message=f"{field_name} must be a string",
                code="invalid_type",
            )
        )
        return None

    stripped = _strip_control_chars(value).strip()
    if not stripped:
        errors.append(
            FieldError(
                field=field_name,
                message=f"{field_name} must not be empty or whitespace-only",
                code="required",
            )
        )
        return None

    return value


def validate_command_request(data: dict[str, Any]) -> ValidationResult:
    """Validate raw input data into a CommandRequest.

    Collects all field errors before returning, so the caller sees every
    problem at once rather than fixing them one at a time.

    This function never raises exceptions -- all validation failures are
    reported through the returned ValidationResult.

    Args:
        data: Raw input dictionary, typically from CLI or IPC layer.
            Expected keys: natural_language_command (required),
            target_host (required), plus optional command_id,
            target_user, target_port, metadata, created_at.

    Returns:
        ValidationResult with either a valid CommandRequest or a tuple
        of FieldError instances describing what went wrong.
    """
    errors: list[FieldError] = []

    # -- Required fields --
    _check_required_string(data, "natural_language_command", errors)
    _check_required_string(data, "target_host", errors)

    # -- Optional typed fields --
    target_port = data.get("target_port")
    if target_port is not None:
        if isinstance(target_port, bool) or not isinstance(target_port, int):
            errors.append(
                FieldError(
                    field="target_port",
                    message="target_port must be an integer",
                    code="invalid_type",
                )
            )
        elif not (1 <= target_port <= 65535):
            errors.append(
                FieldError(
                    field="target_port",
                    message=f"target_port must be 1-65535, got {target_port}",
                    code="out_of_range",
                )
            )

    metadata = data.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            errors.append(
                FieldError(
                    field="metadata",
                    message="metadata must be a dict",
                    code="invalid_type",
                )
            )
        else:
            if len(metadata) > MAX_METADATA_KEYS:
                errors.append(
                    FieldError(
                        field="metadata",
                        message=(
                            f"metadata must not have more than "
                            f"{MAX_METADATA_KEYS} keys, got {len(metadata)}"
                        ),
                        code="too_many_keys",
                    )
                )
            for key, val in metadata.items():
                if not isinstance(key, str) or not _METADATA_KEY_RE.match(key):
                    errors.append(
                        FieldError(
                            field="metadata",
                            message=(
                                f"metadata key {key!r} must contain only "
                                f"letters, digits, underscores, or hyphens"
                            ),
                            code="invalid_key",
                        )
                    )
                    break  # one key error is enough
                if isinstance(val, str) and len(val) > MAX_METADATA_VALUE_LENGTH:
                    errors.append(
                        FieldError(
                            field="metadata",
                            message=(
                                f"metadata value for key {key!r} must not "
                                f"exceed {MAX_METADATA_VALUE_LENGTH} characters"
                            ),
                            code="value_too_long",
                        )
                    )
                    break  # one value error is enough

    # -- If we found errors in pre-checks, return early --
    if errors:
        return ValidationResult(command=None, errors=tuple(errors))

    # -- Construct via Pydantic (catches any remaining validation) --
    try:
        # Build the constructor kwargs from the input data, only including
        # fields that the model recognizes.
        model_fields = {
            "natural_language_command",
            "target_host",
            "command_id",
            "target_user",
            "target_port",
            "metadata",
            "created_at",
        }
        filtered = {k: v for k, v in data.items() if k in model_fields}
        command = CommandRequest(**filtered)
        return ValidationResult(command=command, errors=())

    except ValidationError as exc:
        pydantic_errors: list[FieldError] = []
        for err in exc.errors():
            loc = err.get("loc", ())
            field_name = str(loc[0]) if loc else "unknown"
            pydantic_errors.append(
                FieldError(
                    field=field_name,
                    message=err.get("msg", "validation failed"),
                    code=err.get("type", "validation_error"),
                )
            )
        return ValidationResult(command=None, errors=tuple(pydantic_errors))

    except Exception as exc:
        return ValidationResult(
            command=None,
            errors=(
                FieldError(
                    field="__root__",
                    message=str(exc),
                    code="unexpected_error",
                ),
            ),
        )
