"""IPC message framing protocol with length-prefixed encoding.

Provides the wire format for CLI-daemon communication:

    [4 bytes: payload length (big-endian uint32)] [N bytes: JSON payload]

The ``MessageEnvelope`` dataclass is the canonical message container that
wraps every IPC exchange. ``MessageType`` enumerates the six message
categories used across the confirmation flow, streaming output, and
request-response cycles.

Standalone functions handle encoding and decoding at two levels:

    Low-level:
        ``pack_header``   -- create 4-byte length prefix
        ``unpack_header`` -- extract payload length from header bytes

    High-level:
        ``encode_frame``    -- serialize a MessageEnvelope to a complete frame
        ``decode_envelope`` -- deserialize JSON payload bytes to a MessageEnvelope

Usage::

    # Sender
    frame = encode_frame(envelope)
    transport.write(frame)

    # Receiver
    header = transport.read(HEADER_SIZE)
    length = unpack_header(header)
    payload = transport.read(length)
    envelope = decode_envelope(payload)
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "HEADER_SIZE",
    "MAX_PAYLOAD_SIZE",
    "MessageEnvelope",
    "MessageType",
    "decode_envelope",
    "encode_frame",
    "pack_header",
    "unpack_header",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADER_SIZE: int = 4
"""Number of bytes in the length-prefix header (unsigned 32-bit big-endian)."""

MAX_PAYLOAD_SIZE: int = 16 * 1024 * 1024  # 16 MiB
"""Safety limit on payload size to prevent unbounded memory allocation."""

_STRUCT_FORMAT = "!I"  # network byte order (big-endian), unsigned 32-bit int


# ---------------------------------------------------------------------------
# MessageType enum
# ---------------------------------------------------------------------------


class MessageType(Enum):
    """IPC message categories for CLI-daemon communication.

    Values:
        REQUEST:        CLI sends a command to the daemon.
        RESPONSE:       Daemon sends a result back to the CLI.
        STREAM:         Daemon pushes streaming output (watch verb).
        CONFIRM_PROMPT: Daemon asks the CLI to display a confirmation
                        dialog for an SSH command (security approval).
        CONFIRM_REPLY:  CLI sends the user's approval or denial back
                        to the daemon.
        ERROR:          Error response from either side.
    """

    REQUEST = "request"
    RESPONSE = "response"
    STREAM = "stream"
    CONFIRM_PROMPT = "confirm_prompt"
    CONFIRM_REPLY = "confirm_reply"
    ERROR = "error"


# Lookup table for deserialization: lowered value -> MessageType
_MSG_TYPE_LOOKUP: dict[str, MessageType] = {mt.value: mt for mt in MessageType}


# ---------------------------------------------------------------------------
# MessageEnvelope dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MessageEnvelope:
    """Typed, immutable envelope for every IPC message.

    Every message exchanged between the CLI and daemon is wrapped in
    this envelope. The ``msg_id`` enables request-response correlation,
    the ``timestamp`` provides an ordering guarantee, and the
    ``payload`` carries the verb-specific data.

    Attributes:
        msg_type:  Category of this message (request, response, etc.).
        msg_id:    Unique identifier for request-response correlation.
        timestamp: ISO 8601 timestamp of message creation.
        payload:   JSON-serializable dict with verb-specific data.
                   Defaults to an empty dict.
    """

    msg_type: MessageType
    msg_id: str
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.msg_id, str):
            raise TypeError(
                f"msg_id must be a str, got {type(self.msg_id).__name__}"
            )
        if not self.msg_id.strip():
            raise ValueError("msg_id must not be empty")
        if not isinstance(self.timestamp, str):
            raise TypeError(
                f"timestamp must be a str, got {type(self.timestamp).__name__}"
            )
        if not self.timestamp.strip():
            raise ValueError("timestamp must not be empty")
        if not isinstance(self.payload, dict):
            raise TypeError(
                f"payload must be a dict, got {type(self.payload).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON encoding.

        Returns:
            Dict with ``msg_type`` as its string value, plus all
            other fields.
        """
        return {
            "msg_type": self.msg_type.value,
            "msg_id": self.msg_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MessageEnvelope:
        """Deserialize from a plain dict (e.g., parsed JSON).

        Args:
            data: Dict with at least ``msg_type``, ``msg_id``,
                ``timestamp``, and ``payload`` keys.

        Returns:
            Reconstructed MessageEnvelope.

        Raises:
            ValueError: If ``msg_type`` is not a recognized value.
            KeyError: If required fields are missing.
        """
        raw_type = data["msg_type"]
        msg_type = _MSG_TYPE_LOOKUP.get(raw_type)
        if msg_type is None:
            valid = ", ".join(sorted(_MSG_TYPE_LOOKUP))
            raise ValueError(
                f"Unknown msg_type {raw_type!r}. Valid types: {valid}"
            )

        return MessageEnvelope(
            msg_type=msg_type,
            msg_id=data["msg_id"],
            timestamp=data["timestamp"],
            payload=data.get("payload", {}),
        )


# ---------------------------------------------------------------------------
# Low-level header functions
# ---------------------------------------------------------------------------


def pack_header(length: int) -> bytes:
    """Create a 4-byte big-endian length-prefix header.

    Args:
        length: Payload size in bytes. Must be between 0 and
            ``MAX_PAYLOAD_SIZE`` inclusive.

    Returns:
        4-byte header encoding the length as unsigned 32-bit
        big-endian integer.

    Raises:
        ValueError: If length is negative or exceeds MAX_PAYLOAD_SIZE.
    """
    if length < 0:
        raise ValueError(f"Payload length must not be negative, got {length}")
    if length > MAX_PAYLOAD_SIZE:
        raise ValueError(
            f"Payload length {length} exceeds maximum "
            f"of {MAX_PAYLOAD_SIZE} bytes"
        )
    return struct.pack(_STRUCT_FORMAT, length)


def unpack_header(header: bytes) -> int:
    """Extract the payload length from a 4-byte header.

    Args:
        header: Exactly 4 bytes of big-endian uint32 data.

    Returns:
        The decoded payload length.

    Raises:
        ValueError: If the header is not exactly 4 bytes, or if the
            decoded length exceeds MAX_PAYLOAD_SIZE.
    """
    if len(header) != HEADER_SIZE:
        raise ValueError(
            f"Header must be exactly {HEADER_SIZE} bytes, "
            f"got {len(header)}"
        )
    (length,) = struct.unpack(_STRUCT_FORMAT, header)
    if length > MAX_PAYLOAD_SIZE:
        raise ValueError(
            f"Decoded payload length {length} exceeds maximum "
            f"of {MAX_PAYLOAD_SIZE} bytes"
        )
    return length


# ---------------------------------------------------------------------------
# High-level encode / decode
# ---------------------------------------------------------------------------


def encode_frame(envelope: MessageEnvelope) -> bytes:
    """Serialize a MessageEnvelope to a length-prefixed wire frame.

    The frame layout is::

        [4 bytes: payload length] [N bytes: UTF-8 JSON payload]

    Args:
        envelope: The message to serialize.

    Returns:
        Complete frame bytes ready for transport.

    Raises:
        ValueError: If the serialized payload exceeds MAX_PAYLOAD_SIZE.
    """
    payload_bytes = json.dumps(
        envelope.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    payload_length = len(payload_bytes)
    if payload_length > MAX_PAYLOAD_SIZE:
        raise ValueError(
            f"Serialized payload ({payload_length} bytes) exceeds maximum "
            f"of {MAX_PAYLOAD_SIZE} bytes"
        )

    return pack_header(payload_length) + payload_bytes


def decode_envelope(payload_bytes: bytes) -> MessageEnvelope:
    """Deserialize JSON payload bytes into a MessageEnvelope.

    This function operates on the payload portion only (after the
    length-prefix header has been stripped). Use ``unpack_header``
    to extract the length from the header first.

    Args:
        payload_bytes: Raw bytes of the JSON-encoded envelope.

    Returns:
        Deserialized MessageEnvelope.

    Raises:
        ValueError: If the bytes are not valid JSON, not a JSON object,
            or contain an unknown message type.
    """
    try:
        raw = json.loads(payload_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to decode payload as JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Payload must be a JSON object, got {type(raw).__name__}"
        )

    return MessageEnvelope.from_dict(raw)
