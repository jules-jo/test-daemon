"""Inbound message validation helpers for the Jules IPC protocol.

Provides JSON Schema definitions for each message type, a
``validate_message()`` entry point that raises structured errors on
invalid input, and protocol version compatibility checks.

Architecture:
    This module sits between the raw wire format (handled by
    ``serialization.py``) and the application logic. It provides:

    1. **JSON Schema generation** -- ``get_payload_schema()`` and
       ``get_envelope_schema()`` produce standard JSON Schema dicts
       from the Pydantic models for use by external validators,
       documentation generators, or non-Python clients.

    2. **Structured validation** -- ``validate_message()`` accepts raw
       input (dict, str, or bytes), performs multi-phase validation
       (structural checks, version compatibility, payload validation),
       and either returns a validated ``Envelope`` or raises a
       ``MessageValidationError`` with machine-readable detail records.

    3. **Version compatibility** -- ``check_version_compatible()``
       implements semver major-version gating so that CLI and daemon
       can detect protocol mismatches early.

Error reporting:
    ``MessageValidationError`` carries a tuple of ``ValidationDetail``
    frozen dataclasses, each with ``field``, ``message``, and ``code``
    attributes. The ``to_dict()`` methods on both classes produce
    JSON-serializable dicts suitable for wire transmission.

Usage::

    from jules_daemon.protocol.validation import (
        validate_message,
        get_payload_schema,
        check_version_compatible,
    )

    # Validate raw inbound data (dict, str, or bytes)
    try:
        envelope = validate_message(raw_data)
    except MessageValidationError as exc:
        for detail in exc.details:
            log.warning("%s: %s (%s)", detail.field, detail.message, detail.code)

    # Get JSON Schema for a payload type
    schema = get_payload_schema("run_request")

    # Check version compatibility
    if not check_version_compatible("2.0.0"):
        log.error("Protocol version incompatible")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from jules_daemon.protocol.schemas import (
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
    RunRequest,
    RunResponse,
    StatusRequest,
    StatusResponse,
    StreamChunk,
    WatchRequest,
)
from jules_daemon.protocol.types import (
    PROTOCOL_VERSION_MAJOR,
)

__all__ = [
    "MessageValidationError",
    "ValidationDetail",
    "check_version_compatible",
    "get_envelope_schema",
    "get_payload_schema",
    "list_payload_types",
    "validate_message",
]


# ---------------------------------------------------------------------------
# Structured error types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationDetail:
    """A single validation issue with location, message, and error code.

    Attributes:
        field: Dot-separated path to the problematic field
            (e.g., ``"header.protocol_version"``, ``"payload.run_id"``).
        message: Human-readable description of the issue.
        code: Machine-readable error code for programmatic handling
            (e.g., ``"missing_field"``, ``"version_incompatible"``).
    """

    field: str
    message: str
    code: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable dict representation."""
        return {
            "field": self.field,
            "message": self.message,
            "code": self.code,
        }


class MessageValidationError(Exception):
    """Raised when inbound message validation fails.

    Carries a human-readable summary and a tuple of
    ``ValidationDetail`` records describing each issue.

    Attributes:
        summary: High-level description of what went wrong.
        details: Tuple of individual validation issues.
    """

    def __init__(
        self,
        summary: str,
        details: tuple[ValidationDetail, ...] = (),
    ) -> None:
        super().__init__(summary)
        self.summary: str = summary
        self.details: tuple[ValidationDetail, ...] = details

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation."""
        return {
            "summary": self.summary,
            "details": [d.to_dict() for d in self.details],
        }


# ---------------------------------------------------------------------------
# Payload type registry (payload_type string -> Pydantic model class)
# ---------------------------------------------------------------------------

_PAYLOAD_MODEL_REGISTRY: dict[str, type] = {}


def _build_registry() -> None:
    """Populate the payload model registry from known payload classes.

    Each model class must have a ``payload_type`` field with a Literal
    default value. The default value is used as the registry key.
    """
    all_payload_classes: tuple[type, ...] = (
        RunRequest,
        RunResponse,
        StatusRequest,
        StatusResponse,
        WatchRequest,
        StreamChunk,
        CancelRequest,
        CancelResponse,
        ConfirmPromptPayload,
        ConfirmReplyPayload,
        HealthRequest,
        HealthResponse,
        HistoryRequest,
        HistoryResponse,
        ErrorPayload,
    )
    for cls in all_payload_classes:
        field_info = cls.model_fields.get("payload_type")
        if field_info is not None and field_info.default is not None:
            _PAYLOAD_MODEL_REGISTRY[field_info.default] = cls


_build_registry()


# ---------------------------------------------------------------------------
# JSON Schema generation
# ---------------------------------------------------------------------------


# Cache for generated schemas (keyed by payload_type string)
_PAYLOAD_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}

# Cache for the envelope schema
_ENVELOPE_SCHEMA_CACHE: dict[str, Any] | None = None


def get_payload_schema(payload_type: str) -> dict[str, Any]:
    """Return the JSON Schema dict for a specific payload type.

    Uses Pydantic's ``model_json_schema()`` to generate a standard
    JSON Schema definition. Results are cached for performance.

    Args:
        payload_type: The payload_type discriminator string
            (e.g., ``"run_request"``, ``"health_request"``).

    Returns:
        A JSON Schema dict with ``type``, ``properties``, etc.

    Raises:
        ValueError: If the payload_type is not in the registry.
    """
    if payload_type in _PAYLOAD_SCHEMA_CACHE:
        return _PAYLOAD_SCHEMA_CACHE[payload_type]

    model_cls = _PAYLOAD_MODEL_REGISTRY.get(payload_type)
    if model_cls is None:
        known = ", ".join(sorted(_PAYLOAD_MODEL_REGISTRY))
        raise ValueError(
            f"Unknown payload type {payload_type!r}. "
            f"Known types: {known}"
        )

    schema = model_cls.model_json_schema()
    _PAYLOAD_SCHEMA_CACHE[payload_type] = schema
    return schema


def get_envelope_schema() -> dict[str, Any]:
    """Return the JSON Schema dict for the full Envelope wrapper.

    Uses Pydantic's ``model_json_schema()`` to generate a standard
    JSON Schema definition for the ``Envelope`` model, including
    the discriminated payload union. The result is cached.

    Returns:
        A JSON Schema dict for the complete envelope structure.
    """
    global _ENVELOPE_SCHEMA_CACHE  # noqa: PLW0603
    if _ENVELOPE_SCHEMA_CACHE is not None:
        return _ENVELOPE_SCHEMA_CACHE

    schema = Envelope.model_json_schema()
    _ENVELOPE_SCHEMA_CACHE = schema
    return schema


def list_payload_types() -> tuple[str, ...]:
    """Return a sorted tuple of all registered payload type strings.

    Useful for documentation, auto-completion, and error messages.

    Returns:
        Sorted tuple of payload_type discriminator strings.
    """
    return tuple(sorted(_PAYLOAD_MODEL_REGISTRY))


# ---------------------------------------------------------------------------
# Version compatibility
# ---------------------------------------------------------------------------


def _parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semver string into (major, minor, patch) integers.

    Args:
        version: A string in ``MAJOR.MINOR.PATCH`` format.

    Returns:
        A tuple of three non-negative integers.

    Raises:
        ValueError: If the string is not valid semver.
    """
    stripped = version.strip()
    if not stripped:
        raise ValueError("Invalid version string: must not be empty")

    parts = stripped.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid version string {version!r}: "
            f"expected MAJOR.MINOR.PATCH format"
        )

    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        raise ValueError(
            f"Invalid version string {version!r}: "
            f"all components must be integers"
        ) from None

    if major < 0 or minor < 0 or patch < 0:
        raise ValueError(
            f"Invalid version string {version!r}: "
            f"components must be non-negative"
        )

    return major, minor, patch


def check_version_compatible(version: str) -> bool:
    """Check whether a protocol version is compatible with this daemon.

    Compatibility is determined by semver major-version gating:
    two versions are compatible if and only if they share the same
    major version number. Minor and patch differences are allowed
    (backward-compatible additions and fixes).

    Args:
        version: The protocol version string to check (``MAJOR.MINOR.PATCH``).

    Returns:
        True if the version is compatible, False otherwise.

    Raises:
        ValueError: If the version string is not valid semver.
    """
    remote_major, _, _ = _parse_semver(version)
    return remote_major == PROTOCOL_VERSION_MAJOR


# ---------------------------------------------------------------------------
# Message validation
# ---------------------------------------------------------------------------


def _parse_raw_input(
    data: dict[str, Any] | str | bytes,
) -> dict[str, Any]:
    """Parse raw input into a dict, handling str and bytes JSON.

    Args:
        data: Raw message data (dict, JSON string, or JSON bytes).

    Returns:
        The parsed dict.

    Raises:
        MessageValidationError: If the input cannot be parsed.
    """
    if isinstance(data, bytes):
        if not data:
            raise MessageValidationError(
                "Cannot validate empty input",
                details=(
                    ValidationDetail(
                        field="<root>",
                        message="Input bytes are empty",
                        code="empty_input",
                    ),
                ),
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise MessageValidationError(
                "Input is not valid UTF-8",
                details=(
                    ValidationDetail(
                        field="<root>",
                        message=str(exc),
                        code="encoding_error",
                    ),
                ),
            ) from exc
        data = text

    if isinstance(data, str):
        stripped = data.strip()
        if not stripped:
            raise MessageValidationError(
                "Cannot validate empty input",
                details=(
                    ValidationDetail(
                        field="<root>",
                        message="Input string is empty or whitespace-only",
                        code="empty_input",
                    ),
                ),
            )
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise MessageValidationError(
                "Invalid JSON input",
                details=(
                    ValidationDetail(
                        field="<root>",
                        message=f"JSON parse error: {exc}",
                        code="json_parse_error",
                    ),
                ),
            ) from exc
        data = parsed

    if not isinstance(data, dict):
        raise MessageValidationError(
            "Message must be a JSON object",
            details=(
                ValidationDetail(
                    field="<root>",
                    message=f"Expected object, got {type(data).__name__}",
                    code="invalid_type",
                ),
            ),
        )

    return data


def _check_structure(data: dict[str, Any]) -> list[ValidationDetail]:
    """Check that required top-level fields are present and non-null.

    Args:
        data: The parsed message dict.

    Returns:
        A list of validation issues (empty if structure is valid).
    """
    issues: list[ValidationDetail] = []

    if "header" not in data:
        issues.append(
            ValidationDetail(
                field="header",
                message="Missing required field 'header'",
                code="missing_field",
            )
        )
    elif data["header"] is None:
        issues.append(
            ValidationDetail(
                field="header",
                message="Field 'header' must not be null",
                code="null_field",
            )
        )

    if "payload" not in data:
        issues.append(
            ValidationDetail(
                field="payload",
                message="Missing required field 'payload'",
                code="missing_field",
            )
        )
    elif data["payload"] is None:
        issues.append(
            ValidationDetail(
                field="payload",
                message="Field 'payload' must not be null",
                code="null_field",
            )
        )

    return issues


def _check_version_in_header(
    header: dict[str, Any],
) -> list[ValidationDetail]:
    """Check protocol version in the header for compatibility.

    Args:
        header: The header dict from the message.

    Returns:
        A list of validation issues (empty if version is compatible).
    """
    issues: list[ValidationDetail] = []

    version = header.get("protocol_version")
    if version is None:
        issues.append(
            ValidationDetail(
                field="header.protocol_version",
                message="Missing required field 'protocol_version'",
                code="missing_field",
            )
        )
        return issues

    if not isinstance(version, str) or not version.strip():
        issues.append(
            ValidationDetail(
                field="header.protocol_version",
                message="Protocol version must be a non-empty string",
                code="invalid_version_format",
            )
        )
        return issues

    try:
        compatible = check_version_compatible(version)
    except ValueError:
        issues.append(
            ValidationDetail(
                field="header.protocol_version",
                message=f"Invalid version format: {version!r}",
                code="invalid_version_format",
            )
        )
        return issues

    if not compatible:
        issues.append(
            ValidationDetail(
                field="header.protocol_version",
                message=(
                    f"Protocol version {version!r} is incompatible "
                    f"with daemon (major version {PROTOCOL_VERSION_MAJOR})"
                ),
                code="version_incompatible",
            )
        )

    return issues


def _check_payload_discriminator(
    payload: dict[str, Any],
) -> list[ValidationDetail]:
    """Check that the payload has a valid payload_type discriminator.

    Args:
        payload: The payload dict from the message.

    Returns:
        A list of validation issues (empty if discriminator is valid).
    """
    issues: list[ValidationDetail] = []

    payload_type = payload.get("payload_type")
    if payload_type is None:
        issues.append(
            ValidationDetail(
                field="payload.payload_type",
                message="Missing required field 'payload_type'",
                code="missing_discriminator",
            )
        )
        return issues

    if payload_type not in _PAYLOAD_MODEL_REGISTRY:
        known = ", ".join(sorted(_PAYLOAD_MODEL_REGISTRY))
        issues.append(
            ValidationDetail(
                field="payload.payload_type",
                message=(
                    f"Unknown payload type {payload_type!r}. "
                    f"Known types: {known}"
                ),
                code="unknown_payload_type",
            )
        )

    return issues


def _map_pydantic_errors(
    exc: PydanticValidationError,
) -> tuple[ValidationDetail, ...]:
    """Convert Pydantic validation errors to ValidationDetail records.

    Maps each Pydantic error to a ValidationDetail with the field path
    derived from the error's location tuple.

    Args:
        exc: A Pydantic ValidationError.

    Returns:
        A tuple of ValidationDetail records.
    """
    details: list[ValidationDetail] = []
    for error in exc.errors():
        loc_parts = [str(part) for part in error.get("loc", ())]
        field = ".".join(loc_parts) if loc_parts else "<root>"
        details.append(
            ValidationDetail(
                field=field,
                message=error.get("msg", "Validation failed"),
                code="payload_validation_error",
            )
        )
    return tuple(details)


def validate_message(
    data: dict[str, Any] | str | bytes,
) -> Envelope:
    """Validate an inbound IPC message and return a typed Envelope.

    Performs multi-phase validation:

    1. **Parse** -- Accepts dict, JSON string, or JSON bytes.
       Raises on empty input, invalid JSON, or non-object types.

    2. **Structure** -- Checks for required top-level fields
       (``header`` and ``payload``) and null values.

    3. **Version** -- Extracts the protocol version from the header
       and checks compatibility via major-version gating.

    4. **Discriminator** -- Checks that the payload contains a valid
       ``payload_type`` field that maps to a known model.

    5. **Full validation** -- Deserializes the entire message through
       Pydantic's ``Envelope.model_validate()`` which validates all
       field constraints, types, and cross-field invariants.

    Args:
        data: Raw message data in one of three forms:
            - ``dict`` -- already parsed JSON object
            - ``str``  -- JSON-encoded string
            - ``bytes`` -- UTF-8 encoded JSON bytes

    Returns:
        A validated, immutable ``Envelope`` instance with a typed
        payload selected via the ``payload_type`` discriminator.

    Raises:
        MessageValidationError: If validation fails at any phase.
            The exception carries a ``summary`` string and a ``details``
            tuple of ``ValidationDetail`` records describing each issue.
    """
    # Phase 1: Parse raw input
    parsed = _parse_raw_input(data)

    # Phase 2: Structural checks
    structural_issues = _check_structure(parsed)
    if structural_issues:
        raise MessageValidationError(
            "Invalid message structure",
            details=tuple(structural_issues),
        )

    # Phase 3: Version compatibility (header exists from phase 2)
    header_data = parsed["header"]
    if isinstance(header_data, dict):
        version_issues = _check_version_in_header(header_data)
        if version_issues:
            raise MessageValidationError(
                "Protocol version check failed",
                details=tuple(version_issues),
            )

    # Phase 4: Payload discriminator check (payload exists from phase 2)
    payload_data = parsed["payload"]
    if isinstance(payload_data, dict):
        discriminator_issues = _check_payload_discriminator(payload_data)
        if discriminator_issues:
            raise MessageValidationError(
                "Invalid payload type",
                details=tuple(discriminator_issues),
            )

    # Phase 5: Full Pydantic validation
    try:
        return Envelope.model_validate(parsed)
    except PydanticValidationError as exc:
        details = _map_pydantic_errors(exc)
        raise MessageValidationError(
            "Message validation failed",
            details=details,
        ) from exc
