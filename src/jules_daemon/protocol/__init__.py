"""Jules IPC protocol -- public API for CLI-daemon communication.

This package defines the complete IPC protocol used between the Jules CLI
(thin client) and the Jules daemon (owner of LLM calls and execution).
Import all protocol symbols from this top-level namespace.

Protocol Specification
======================

Protocol Identity
-----------------

    Name:     ``jules-ipc``
    Version:  ``1.0.0`` (semver: MAJOR.MINOR.PATCH)

    Compatibility rule: two endpoints are compatible when they share the
    same major version. Minor and patch differences are permitted (they
    represent backward-compatible additions and fixes respectively).

Wire Format
-----------

    Messages are serialized as compact, single-line UTF-8 JSON terminated
    by a newline character (``\\n``). This newline-delimited JSON (NDJSON)
    encoding enables simple line-oriented parsing when the transport
    already provides message boundaries (e.g., a pipe per message).

    For byte-stream transports (Unix domain sockets, TCP), messages are
    wrapped in length-prefixed frames:

        +------------------+---------------------+
        | 4 bytes (BE u32) | N bytes (UTF-8 JSON)|
        | message length   | envelope JSON       |
        +------------------+---------------------+

    The 4-byte header is a big-endian unsigned 32-bit integer encoding
    the byte length of the JSON payload that follows. Maximum frame size
    is 16 MiB. Use ``encode_frame`` / ``decode_frame`` / ``FrameBuffer``
    for stream framing.

Envelope Structure
------------------

    Every IPC message is wrapped in an ``Envelope`` containing a
    ``MessageHeader`` and a typed ``payload``:

    .. code-block:: json

        {
            "header": {
                "protocol_version": "1.0.0",
                "message_id": "<uuid>",
                "timestamp": "<ISO-8601 UTC>",
                "message_type": "<message_kind>"
            },
            "payload": {
                "payload_type": "<discriminator>",
                ...payload fields...
            }
        }

    Header fields:
        ``protocol_version``  Semver string (``MAJOR.MINOR.PATCH``).
        ``message_id``        UUID v4 string uniquely identifying this message.
        ``timestamp``         ISO-8601 UTC datetime when the message was created.
        ``message_type``      One of the ``MessageKind`` enum values (see below).

    The ``payload`` is a discriminated union keyed on ``payload_type``,
    allowing unambiguous deserialization into the correct Pydantic model.
    Use ``create_envelope`` to build envelopes with auto-populated headers.

Supported Message Kinds (``MessageKind``)
-----------------------------------------

    ``request``         CLI sends a command to the daemon.
    ``response``        Daemon returns a result (terminal -- ends the exchange).
    ``notification``    Daemon pushes an unsolicited event (non-terminal).
    ``error``           Error from either side (terminal -- ends the exchange).
    ``stream``          Daemon pushes streaming output (non-terminal).
    ``confirm_prompt``  Daemon asks CLI for SSH command approval.
    ``confirm_reply``   CLI sends approval or denial back to daemon.

    Terminal kinds (``response``, ``error``) end a request-response cycle.
    Non-terminal kinds may be followed by additional messages.

Supported Payload Types
-----------------------

    Each payload is identified by a ``payload_type`` literal discriminator:

    Run (test execution):
        ``run_request``        Submit a natural-language test command.
        ``run_response``       Acknowledge a submitted run.

    Status (run state query):
        ``status_request``     Query current run state.
        ``status_response``    Return current run state and progress.

    Watch (live output streaming):
        ``watch_request``      Subscribe to live output for a run.
        ``stream_chunk``       A single chunk of streaming output.

    Cancel (stop active run):
        ``cancel_request``     Cancel an active run.
        ``cancel_response``    Acknowledge a cancellation.

    Confirm (SSH security approval):
        ``confirm_prompt``     Present SSH command for user approval.
        ``confirm_reply``      User's approval (ALLOW) or denial (DENY).

    Health (daemon liveness):
        ``health_request``     Check daemon health.
        ``health_response``    Return daemon health and metadata.

    History (past run summaries):
        ``history_request``    Request past run summaries.
        ``history_response``   Return paginated run history.

    Error:
        ``error``              Structured error information.

Status Codes (``StatusCode``)
-----------------------------

    Numeric codes modelled after HTTP semantics:

    2xx Success:
        200 OK, 202 Accepted, 204 No Content

    4xx Client Errors:
        400 Bad Request, 401 Unauthorized, 403 Forbidden,
        404 Not Found, 409 Conflict, 422 Unprocessable

    5xx Server Errors:
        500 Internal Error, 501 Not Implemented,
        502 Service Unavailable, 503 Busy, 504 Timeout

Quick Start
-----------

    Serialize and send::

        from jules_daemon.protocol import (
            serialize, create_envelope, encode_frame,
            MessageKind, HealthRequest,
        )

        envelope = create_envelope(
            message_type=MessageKind.REQUEST,
            payload=HealthRequest(),
        )
        wire_bytes = serialize(envelope)
        frame = encode_frame(wire_bytes)

    Receive and validate::

        from jules_daemon.protocol import (
            deserialize, validate_message, decode_frame,
        )

        # From stream transport:
        msg_bytes, remainder = decode_frame(raw_stream)
        envelope = deserialize(msg_bytes)

        # Or validate raw input (dict, str, or bytes):
        envelope = validate_message(raw_input)

    Wrap/unwrap shorthand::

        from jules_daemon.protocol import (
            wrap_payload, unwrap_payload, MessageKind, RunRequest,
            SSHTargetInfo,
        )

        wire = wrap_payload(
            MessageKind.REQUEST,
            RunRequest(
                natural_language_command="Run pytest on auth",
                ssh_target=SSHTargetInfo(host="ci.example.com", user="ci"),
            ),
        )
        header, payload = unwrap_payload(wire)

Submodules
----------

    ``types``          Protocol version, MessageKind/StatusCode enums, helpers.
    ``schemas``        Pydantic request/response models, Envelope wrapper.
    ``serialization``  JSON wire-format encoding/decoding, stream framing.
    ``validation``     Inbound validation, JSON Schema generation, version checks.
"""

# ---------------------------------------------------------------------------
# Protocol identity and enums (from types)
# ---------------------------------------------------------------------------

from jules_daemon.protocol.types import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_MAJOR,
    PROTOCOL_VERSION_MINOR,
    PROTOCOL_VERSION_PATCH,
    MessageKind,
    StatusCode,
    is_client_error,
    is_server_error,
    is_success,
    is_terminal_message,
    parse_message_kind,
    parse_status_code,
    status_code_to_reason,
)

# ---------------------------------------------------------------------------
# Pydantic schema models and envelope helpers (from schemas)
# ---------------------------------------------------------------------------

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
    PayloadType,
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

# ---------------------------------------------------------------------------
# Serialization (from serialization)
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

# ---------------------------------------------------------------------------
# Validation (from validation)
# ---------------------------------------------------------------------------

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
# Convenience aliases
#
# ``serialize`` and ``deserialize`` are the primary entry points for
# converting between typed Envelope objects and wire bytes. They alias
# the envelope-level functions since every IPC exchange uses envelopes.
# ---------------------------------------------------------------------------

serialize = serialize_envelope
"""Serialize an Envelope to UTF-8 JSON bytes with trailing newline.

Alias for ``serialize_envelope``. This is the primary serialization
entry point for the IPC protocol -- all messages are envelopes.
"""

deserialize = deserialize_envelope
"""Deserialize UTF-8 JSON bytes into a validated Envelope.

Alias for ``deserialize_envelope``. This is the primary deserialization
entry point for the IPC protocol -- all wire messages are envelopes.
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # -- Protocol identity --
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_MAJOR",
    "PROTOCOL_VERSION_MINOR",
    "PROTOCOL_VERSION_PATCH",
    # -- Enums --
    "ApprovalDecision",
    "MessageKind",
    "StatusCode",
    # -- Type helpers --
    "is_client_error",
    "is_server_error",
    "is_success",
    "is_terminal_message",
    "parse_message_kind",
    "parse_status_code",
    "status_code_to_reason",
    # -- Schema models --
    "CancelRequest",
    "CancelResponse",
    "ConfirmPromptPayload",
    "ConfirmReplyPayload",
    "Envelope",
    "ErrorPayload",
    "HealthRequest",
    "HealthResponse",
    "HistoryRequest",
    "HistoryResponse",
    "HistoryRunSummary",
    "MessageHeader",
    "PayloadType",
    "ProgressSnapshot",
    "RunRequest",
    "RunResponse",
    "SSHTargetInfo",
    "StatusRequest",
    "StatusResponse",
    "StreamChunk",
    "WatchRequest",
    # -- Envelope factory --
    "create_envelope",
    # -- Serialization --
    "FrameBuffer",
    "SerializationError",
    "decode_frame",
    "deserialize_envelope",
    "deserialize_payload",
    "encode_frame",
    "serialize_envelope",
    "serialize_payload",
    "unwrap_payload",
    "wrap_payload",
    # -- Convenience aliases --
    "serialize",
    "deserialize",
    # -- Validation --
    "MessageValidationError",
    "ValidationDetail",
    "check_version_compatible",
    "get_envelope_schema",
    "get_payload_schema",
    "list_payload_types",
    "validate_message",
]
