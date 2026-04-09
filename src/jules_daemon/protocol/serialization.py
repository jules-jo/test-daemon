"""JSON wire-format serialization utilities for the Jules IPC protocol.

Provides encoding/decoding between typed Pydantic protocol models and
raw bytes suitable for IPC transport. Three abstraction levels:

1. **Envelope level** (``serialize_envelope`` / ``deserialize_envelope``):
   Full Envelope <-> UTF-8 JSON bytes with newline delimiter.

2. **Framing level** (``wrap_payload`` / ``unwrap_payload``):
   Shorthand to create an Envelope from a bare payload + MessageKind,
   serialize to wire bytes, and reverse the process.

3. **Payload level** (``serialize_payload`` / ``deserialize_payload``):
   Convert a single payload model to/from a plain dict (useful for
   embedding in other structures or logging).

4. **Stream framing** (``encode_frame`` / ``decode_frame`` / ``FrameBuffer``):
   Length-prefixed (4-byte big-endian uint32) framing for reliable
   transport over byte streams (Unix domain sockets, pipes, TCP).

All functions are pure (no side effects) and raise ``SerializationError``
on malformed input. Models are never mutated.

Usage::

    from jules_daemon.protocol.serialization import (
        wrap_payload,
        unwrap_payload,
        encode_frame,
        FrameBuffer,
    )
    from jules_daemon.protocol.schemas import HealthRequest
    from jules_daemon.protocol.types import MessageKind

    # Sender
    wire = wrap_payload(MessageKind.REQUEST, HealthRequest())
    frame = encode_frame(wire)
    socket.sendall(frame)

    # Receiver
    buf = FrameBuffer()
    for chunk in socket_reads():
        for msg in buf.feed(chunk):
            header, payload = unwrap_payload(msg)
"""

from __future__ import annotations

import json
import struct
from typing import Any

from pydantic import ValidationError

from jules_daemon.protocol.schemas import (
    Envelope,
    PayloadType,
    MessageHeader,
    create_envelope,
)
from jules_daemon.protocol.types import MessageKind

__all__ = [
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
]

# Length-prefix header: 4-byte unsigned big-endian integer
_FRAME_HEADER_SIZE: int = 4
_FRAME_HEADER_FMT: str = "!I"

# Maximum wire message size (16 MiB) as a safety limit
_MAX_FRAME_SIZE: int = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class SerializationError(Exception):
    """Raised when serialization or deserialization fails.

    Wraps the underlying cause (JSON decode error, Pydantic validation
    error, etc.) for structured error handling without leaking internals.

    Args:
        message: Human-readable description of the failure.
        cause: Optional underlying exception.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        if cause is not None:
            self.__cause__ = cause


# ---------------------------------------------------------------------------
# Envelope-level serialization
# ---------------------------------------------------------------------------


def serialize_envelope(envelope: Envelope) -> bytes:
    """Serialize an Envelope to UTF-8 JSON bytes with a trailing newline.

    The output is a single line of compact JSON followed by ``\\n``,
    suitable for newline-delimited transport.

    Args:
        envelope: A fully populated, immutable Envelope instance.

    Returns:
        UTF-8 encoded JSON bytes ending with a newline.

    Raises:
        SerializationError: If the envelope cannot be serialized.
    """
    try:
        json_str = envelope.model_dump_json()
        return json_str.encode("utf-8") + b"\n"
    except Exception as exc:
        raise SerializationError(
            f"Failed to serialize envelope: {exc}", cause=exc
        ) from exc


def deserialize_envelope(data: bytes) -> Envelope:
    """Deserialize UTF-8 JSON bytes into an Envelope.

    Accepts data with or without a trailing newline. Validates the full
    structure including the discriminated payload union.

    Args:
        data: UTF-8 encoded JSON bytes representing an Envelope.

    Returns:
        A validated, immutable Envelope instance.

    Raises:
        SerializationError: If the data is empty, not valid UTF-8,
            not valid JSON, or fails Pydantic validation.
    """
    if not data:
        raise SerializationError("Cannot deserialize empty bytes")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SerializationError(
            "Wire data is not valid UTF-8", cause=exc
        ) from exc

    stripped = text.strip()
    if not stripped:
        raise SerializationError("Cannot deserialize empty/whitespace-only data")

    try:
        return Envelope.model_validate_json(stripped)
    except (ValidationError, json.JSONDecodeError) as exc:
        raise SerializationError(
            f"Failed to deserialize envelope: {exc}", cause=exc
        ) from exc


# ---------------------------------------------------------------------------
# Envelope framing (wrap / unwrap)
# ---------------------------------------------------------------------------


def wrap_payload(
    message_type: MessageKind,
    payload: PayloadType,
    *,
    message_id: str | None = None,
) -> bytes:
    """Create an Envelope from a payload, then serialize to wire bytes.

    Convenience shorthand that combines ``create_envelope`` and
    ``serialize_envelope`` into a single call. Automatically generates
    a UUID message_id and UTC timestamp.

    Args:
        message_type: The MessageKind for the envelope header.
        payload: The typed payload model to wrap.
        message_id: Optional override for the auto-generated message ID.

    Returns:
        UTF-8 JSON bytes (newline-delimited) representing the full envelope.

    Raises:
        SerializationError: If envelope creation or serialization fails.
    """
    try:
        envelope = create_envelope(
            message_type=message_type,
            payload=payload,
            message_id=message_id,
        )
        return serialize_envelope(envelope)
    except SerializationError:
        raise
    except Exception as exc:
        raise SerializationError(
            f"Failed to wrap payload: {exc}", cause=exc
        ) from exc


def unwrap_payload(data: bytes) -> tuple[MessageHeader, PayloadType]:
    """Deserialize wire bytes and extract the header and typed payload.

    Inverse of ``wrap_payload``. Returns a tuple of (header, payload)
    so callers can inspect envelope metadata alongside the payload.

    Args:
        data: UTF-8 JSON bytes representing a full envelope.

    Returns:
        A tuple of (MessageHeader, typed payload model).

    Raises:
        SerializationError: If deserialization fails.
    """
    envelope = deserialize_envelope(data)
    return envelope.header, envelope.payload


# ---------------------------------------------------------------------------
# Payload-level serialization (dict, not bytes)
# ---------------------------------------------------------------------------

# Lookup table mapping payload_type discriminator strings to their Pydantic
# model classes. Built from the PayloadType union members.
_PAYLOAD_TYPE_REGISTRY: dict[str, type] = {}


def _build_payload_registry() -> None:
    """Populate the payload type registry from the PayloadType union.

    Introspects the Annotated union to extract each member class and
    its ``payload_type`` literal default value.
    """
    from jules_daemon.protocol.schemas import (
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

    # Each model has a payload_type field with a Literal default
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
            _PAYLOAD_TYPE_REGISTRY[field_info.default] = cls


_build_payload_registry()


def serialize_payload(payload: PayloadType) -> dict[str, Any]:
    """Serialize a typed payload model to a JSON-compatible dict.

    Uses Pydantic's ``model_dump(mode="json")`` to produce a dict
    with JSON-native types (strings, numbers, bools, None, lists, dicts).

    Args:
        payload: A typed payload model instance.

    Returns:
        A plain dict suitable for JSON serialization or embedding.

    Raises:
        SerializationError: If serialization fails.
    """
    try:
        return payload.model_dump(mode="json")
    except Exception as exc:
        raise SerializationError(
            f"Failed to serialize payload: {exc}", cause=exc
        ) from exc


def deserialize_payload(data: dict[str, Any]) -> PayloadType:
    """Deserialize a dict into the correct typed payload model.

    Uses the ``payload_type`` discriminator field to select the right
    Pydantic model class from the registry, then validates the data.

    Args:
        data: A dict containing at minimum a ``payload_type`` key.

    Returns:
        A validated, immutable payload model instance.

    Raises:
        SerializationError: If ``payload_type`` is missing, unknown,
            or the data fails validation.
    """
    payload_type_str = data.get("payload_type")
    if payload_type_str is None:
        raise SerializationError(
            "Missing 'payload_type' discriminator in payload data"
        )

    model_cls = _PAYLOAD_TYPE_REGISTRY.get(payload_type_str)
    if model_cls is None:
        known = ", ".join(sorted(_PAYLOAD_TYPE_REGISTRY))
        raise SerializationError(
            f"Unknown payload_type {payload_type_str!r}. "
            f"Known types: {known}"
        )

    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise SerializationError(
            f"Payload validation failed for {payload_type_str!r}: {exc}",
            cause=exc,
        ) from exc


# ---------------------------------------------------------------------------
# Length-prefixed stream framing
# ---------------------------------------------------------------------------


def encode_frame(data: bytes) -> bytes:
    """Encode a message as a length-prefixed frame for stream transport.

    Format: 4-byte big-endian unsigned integer (message length) + message bytes.

    Args:
        data: Raw message bytes to frame.

    Returns:
        Length-prefixed frame bytes (header + data).

    Raises:
        SerializationError: If the message exceeds the maximum frame size.
    """
    length = len(data)
    if length > _MAX_FRAME_SIZE:
        raise SerializationError(
            f"Message size {length} exceeds maximum frame size {_MAX_FRAME_SIZE}"
        )
    header = struct.pack(_FRAME_HEADER_FMT, length)
    return header + data


def decode_frame(stream: bytes) -> tuple[bytes, bytes] | None:
    """Extract one complete frame from a byte stream.

    Returns the extracted message and the remaining bytes, or None if
    the stream does not yet contain a complete frame.

    Args:
        stream: Accumulated byte stream (may contain partial frames).

    Returns:
        A tuple of (message_bytes, remaining_bytes) if a complete frame
        is available, or None if more data is needed.
    """
    if len(stream) < _FRAME_HEADER_SIZE:
        return None

    (length,) = struct.unpack(_FRAME_HEADER_FMT, stream[:_FRAME_HEADER_SIZE])

    total_needed = _FRAME_HEADER_SIZE + length
    if len(stream) < total_needed:
        return None

    message = stream[_FRAME_HEADER_SIZE:total_needed]
    remainder = stream[total_needed:]
    return message, remainder


class FrameBuffer:
    """Incremental frame accumulator for streaming byte transport.

    Accepts arbitrary chunks of bytes via ``feed()`` and yields
    complete, length-prefixed messages as they become available.
    Handles partial frames transparently by buffering incomplete data.

    Thread safety: NOT thread-safe. Use external synchronization
    if feeding from multiple threads.

    Usage::

        buf = FrameBuffer()
        for chunk in socket.recv_chunks():
            for message in buf.feed(chunk):
                header, payload = unwrap_payload(message)
    """

    def __init__(self) -> None:
        self._buffer: bytearray = bytearray()

    @property
    def pending(self) -> int:
        """Number of bytes currently buffered (waiting for a complete frame)."""
        return len(self._buffer)

    def clear(self) -> None:
        """Discard all buffered data."""
        self._buffer.clear()

    def feed(self, data: bytes) -> list[bytes]:
        """Add bytes to the buffer and return any complete messages.

        May return zero, one, or many messages depending on how much
        data is available. Incomplete frames remain buffered.

        Args:
            data: New bytes received from the transport.

        Returns:
            A list of complete message payloads (without frame headers).
        """
        if not data:
            return []

        self._buffer.extend(data)
        messages: list[bytes] = []

        while True:
            if len(self._buffer) < _FRAME_HEADER_SIZE:
                break

            (length,) = struct.unpack(
                _FRAME_HEADER_FMT,
                bytes(self._buffer[:_FRAME_HEADER_SIZE]),
            )
            total_needed = _FRAME_HEADER_SIZE + length
            if len(self._buffer) < total_needed:
                break

            message = bytes(self._buffer[_FRAME_HEADER_SIZE:total_needed])
            del self._buffer[:total_needed]
            messages.append(message)

        return messages
