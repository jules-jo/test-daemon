"""SSH output buffer reader for stale and interrupted sessions.

Retrieves buffered but unprocessed output from two sources:

1. **Live SSH channel** (Paramiko transport layer): Reads any buffered
   stdout/stderr bytes that have arrived but not yet been processed by
   the monitoring loop. Uses the existing ``read_ssh_output()`` function
   from ``ssh.reader`` and wraps the result in a structured
   ``BufferReadResult``.

2. **Local session log files**: For stale or interrupted sessions where
   the SSH channel is no longer available, reads from local log files
   that the daemon writes during normal operation. Supports reading
   from an offset (to skip already-processed output) and respects a
   byte limit to avoid unbounded memory allocation.

Both paths return the same ``BufferReadResult`` frozen dataclass,
enabling uniform processing by the caller regardless of the source.

The ``read_buffered_output()`` function is the unified entry point that
tries the channel first (if provided), then falls back to the log file.
When both sources are provided and both have data, results are merged
into a single ``COMBINED`` result.

Usage:
    # Live channel:
    result = await read_channel_buffer(channel)
    if result.has_data:
        for line in result.lines:
            process(line)

    # Stale session log:
    entry = LogFileEntry(path=Path("session.log"), run_id="abc")
    result = read_session_log(entry)
    if result.has_data:
        for line in result.lines:
            process(line)

    # Unified (tries both):
    result = await read_buffered_output(channel=ch, log_entry=entry)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from jules_daemon.ssh.reader import SSHChannelHandle, read_ssh_output

__all__ = [
    "BufferReadResult",
    "BufferSource",
    "LogFileEntry",
    "read_buffered_output",
    "read_channel_buffer",
    "read_session_log",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_BYTES = 65536
_DEFAULT_ENCODING = "utf-8"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BufferSource(Enum):
    """Origin of the buffered output data.

    CHANNEL: Data read from a live SSH channel (Paramiko transport).
    LOG_FILE: Data read from a local session log file.
    COMBINED: Data merged from both a channel and a log file.
    """

    CHANNEL = "channel"
    LOG_FILE = "log_file"
    COMBINED = "combined"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# LogFileEntry model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogFileEntry:
    """Immutable specification for reading a session log file.

    Attributes:
        path: Absolute path to the session log file on the local filesystem.
        run_id: Unique identifier for the run this log belongs to.
        offset: Byte offset to start reading from (skip already-processed
            output). Defaults to 0 (read from the beginning). Must not
            be negative.
        encoding: Character encoding for decoding the log file bytes.
            Invalid bytes are replaced (errors='replace'). Defaults
            to 'utf-8'.
    """

    path: Path
    run_id: str
    offset: int = 0
    encoding: str = _DEFAULT_ENCODING

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(
                f"offset must not be negative, got {self.offset}"
            )


# ---------------------------------------------------------------------------
# BufferReadResult model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BufferReadResult:
    """Immutable structured result of reading buffered SSH output.

    Contains decoded text from stdout/stderr, extracted lines, metadata
    about the read operation, and any error encountered.

    Attributes:
        source: Which source provided the data (channel, log file, or
            combined).
        stdout_text: Decoded text from stdout (channel) or the primary
            content of the log file. Empty string if no data.
        stderr_text: Decoded text from stderr (channel only). Empty
            string for log file sources.
        lines: Tuple of individual output lines extracted from the text.
            Trailing whitespace is stripped from each line. Empty lines
            are excluded.
        is_complete: True if the source has been fully consumed (channel
            EOF/closed, or log file read to the end without byte limit).
        exit_code: Remote process exit code (channel only). None if not
            available or if the source is a log file.
        error: Human-readable error description if the read failed.
            None on success.
        bytes_read: Total number of raw bytes read across all streams.
        log_file_path: Path to the log file that was read (None for
            channel-only reads).
        timestamp: UTC datetime when this result was captured.
    """

    source: BufferSource
    stdout_text: str
    stderr_text: str
    lines: tuple[str, ...]
    is_complete: bool = False
    exit_code: int | None = None
    error: str | None = None
    bytes_read: int = 0
    log_file_path: Path | None = None
    timestamp: datetime = field(default_factory=_now_utc)

    @property
    def has_data(self) -> bool:
        """True if any output lines or text were read."""
        return (
            len(self.lines) > 0
            or len(self.stdout_text) > 0
            or len(self.stderr_text) > 0
        )

    @property
    def line_count(self) -> int:
        """Number of extracted output lines."""
        return len(self.lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_lines(text: str) -> tuple[str, ...]:
    """Split text into lines, stripping trailing whitespace from each.

    Empty lines are excluded from the result.

    Args:
        text: The text to split into lines.

    Returns:
        Tuple of non-empty, stripped lines.
    """
    raw_lines = text.split("\n")
    return tuple(
        stripped
        for line in raw_lines
        if (stripped := line.rstrip())
    )


def _build_empty_result(
    source: BufferSource,
    *,
    error: str | None = None,
    log_file_path: Path | None = None,
) -> BufferReadResult:
    """Build an empty result for cases with no data."""
    return BufferReadResult(
        source=source,
        stdout_text="",
        stderr_text="",
        lines=(),
        is_complete=False,
        exit_code=None,
        error=error,
        bytes_read=0,
        log_file_path=log_file_path,
        timestamp=_now_utc(),
    )


# ---------------------------------------------------------------------------
# Public API: read_channel_buffer
# ---------------------------------------------------------------------------


async def read_channel_buffer(
    channel: SSHChannelHandle,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    encoding: str = _DEFAULT_ENCODING,
) -> BufferReadResult:
    """Read buffered output from a live SSH channel.

    Retrieves any available stdout/stderr bytes from the Paramiko
    transport layer without blocking the event loop. Decodes the raw
    bytes to text, extracts individual lines, and returns a structured
    ``BufferReadResult``.

    This function catches channel errors gracefully and returns them
    in the result's ``error`` field rather than raising.

    Args:
        channel: An object satisfying the SSHChannelHandle protocol
            (e.g., a paramiko.Channel).
        max_bytes: Maximum bytes to read per stream. Must be positive.
            Defaults to 65536.
        encoding: Character encoding for decoding. Invalid bytes are
            replaced. Defaults to 'utf-8'.

    Returns:
        Immutable BufferReadResult with the decoded output. Never raises
        for channel I/O errors -- they are captured in the result.
    """
    try:
        ssh_output = await read_ssh_output(channel, max_bytes=max_bytes)
    except (OSError, TimeoutError) as exc:
        logger.warning(
            "Error reading from SSH channel: %s: %s",
            type(exc).__name__,
            exc,
        )
        return _build_empty_result(
            BufferSource.CHANNEL,
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        # Catch-all for unexpected SSH library errors (e.g., Paramiko
        # SSHException). These are logged and returned as error results
        # to preserve the never-raise contract.
        logger.warning(
            "Unexpected error reading from SSH channel: %s: %s",
            type(exc).__name__,
            exc,
        )
        return _build_empty_result(
            BufferSource.CHANNEL,
            error=f"{type(exc).__name__}: {exc}",
        )

    # Decode bytes to text with replacement for invalid sequences
    stdout_text = ssh_output.stdout.decode(encoding, errors="replace")
    stderr_text = ssh_output.stderr.decode(encoding, errors="replace")

    # Combine both streams for line extraction
    combined_text = stdout_text + stderr_text

    # Determine completion state
    is_complete = ssh_output.is_eof or ssh_output.channel_closed

    return BufferReadResult(
        source=BufferSource.CHANNEL,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
        lines=_extract_lines(combined_text),
        is_complete=is_complete,
        exit_code=ssh_output.exit_code,
        error=None,
        bytes_read=ssh_output.total_bytes,
        log_file_path=None,
        timestamp=_now_utc(),
    )


# ---------------------------------------------------------------------------
# Public API: read_session_log
# ---------------------------------------------------------------------------


def read_session_log(
    entry: LogFileEntry,
    *,
    max_bytes: int = 0,
) -> BufferReadResult:
    """Read buffered output from a local session log file.

    Reads from the specified log file starting at the given byte offset.
    This is used for stale or interrupted sessions where the SSH channel
    is no longer available but the daemon wrote output to a local file.

    The file is read as raw bytes and decoded with the specified encoding
    using replacement for invalid byte sequences (binary-safe).

    Args:
        entry: LogFileEntry specifying the file path, run_id, offset,
            and encoding.
        max_bytes: Maximum number of bytes to read. 0 means no limit
            (read the entire remaining file). Defaults to 0.

    Returns:
        Immutable BufferReadResult with the decoded output. Never raises
        for file I/O errors -- they are captured in the result.
    """
    log_path = entry.path

    # Guard: file must exist
    if not log_path.exists():
        error_msg = f"Session log file not found: {log_path}"
        logger.warning(error_msg)
        return _build_empty_result(
            BufferSource.LOG_FILE,
            error=error_msg,
            log_file_path=log_path,
        )

    try:
        # Read raw bytes for binary safety
        raw_bytes = log_path.read_bytes()

        # Apply offset
        if entry.offset > 0:
            if entry.offset >= len(raw_bytes):
                # Offset past end of file -- nothing new to read
                return BufferReadResult(
                    source=BufferSource.LOG_FILE,
                    stdout_text="",
                    stderr_text="",
                    lines=(),
                    is_complete=True,
                    exit_code=None,
                    error=None,
                    bytes_read=0,
                    log_file_path=log_path,
                    timestamp=_now_utc(),
                )
            raw_bytes = raw_bytes[entry.offset:]

        # Apply max_bytes limit
        if max_bytes > 0 and len(raw_bytes) > max_bytes:
            raw_bytes = raw_bytes[:max_bytes]
            is_complete = False
        else:
            is_complete = True

        bytes_read = len(raw_bytes)

        # Decode with replacement for binary safety
        text = raw_bytes.decode(entry.encoding, errors="replace")

        # Extract lines
        lines = _extract_lines(text)

        return BufferReadResult(
            source=BufferSource.LOG_FILE,
            stdout_text=text,
            stderr_text="",
            lines=lines,
            is_complete=is_complete,
            exit_code=None,
            error=None,
            bytes_read=bytes_read,
            log_file_path=log_path,
            timestamp=_now_utc(),
        )

    except OSError as exc:
        error_msg = f"Error reading session log {log_path}: {exc}"
        logger.warning(error_msg)
        return _build_empty_result(
            BufferSource.LOG_FILE,
            error=error_msg,
            log_file_path=log_path,
        )


# ---------------------------------------------------------------------------
# Public API: read_buffered_output (unified entry point)
# ---------------------------------------------------------------------------


async def read_buffered_output(
    *,
    channel: SSHChannelHandle | None = None,
    log_entry: LogFileEntry | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    encoding: str = _DEFAULT_ENCODING,
) -> BufferReadResult:
    """Unified entry point for reading buffered SSH output.

    Tries the live SSH channel first (if provided), then reads from
    the local session log file (if provided). When both sources have
    data, results are merged into a single ``COMBINED`` result.

    This is the primary function the daemon calls during crash recovery
    and session resumption to collect any output that was buffered but
    not yet processed.

    Args:
        channel: Optional live SSH channel. When provided, buffered
            data is read from the Paramiko transport layer.
        log_entry: Optional session log file specification. When
            provided, data is read from the local log file.
        max_bytes: Maximum bytes to read per source. Defaults to 65536.
        encoding: Character encoding. Defaults to 'utf-8'.

    Returns:
        Immutable BufferReadResult. Source is CHANNEL if only channel
        data was found, LOG_FILE if only log data was found, or
        COMBINED if both had data. Never raises.
    """
    channel_result: BufferReadResult | None = None
    log_result: BufferReadResult | None = None

    # Try channel first
    if channel is not None:
        channel_result = await read_channel_buffer(
            channel,
            max_bytes=max_bytes,
            encoding=encoding,
        )

    # Try log file
    if log_entry is not None:
        log_result = read_session_log(log_entry, max_bytes=max_bytes)

    # Determine which results have data
    channel_has_data = channel_result is not None and channel_result.has_data
    log_has_data = log_result is not None and log_result.has_data

    # Both sources have data -- merge into COMBINED result
    if channel_has_data and log_has_data and channel_result is not None and log_result is not None:
        # Merge: channel stdout/stderr + log stdout into combined text
        merged_stdout = channel_result.stdout_text + log_result.stdout_text
        merged_stderr = channel_result.stderr_text

        # Merge lines: log lines first (older), then channel lines (newer)
        merged_lines = log_result.lines + channel_result.lines

        return BufferReadResult(
            source=BufferSource.COMBINED,
            stdout_text=merged_stdout,
            stderr_text=merged_stderr,
            lines=merged_lines,
            is_complete=channel_result.is_complete,
            exit_code=channel_result.exit_code,
            error=None,
            bytes_read=channel_result.bytes_read + log_result.bytes_read,
            log_file_path=log_result.log_file_path,
            timestamp=_now_utc(),
        )

    # Only channel has data
    if channel_has_data and channel_result is not None:
        return channel_result

    # Only log has data
    if log_has_data and log_result is not None:
        return log_result

    # Neither has data -- return whichever we have (prefer channel)
    if channel_result is not None:
        return channel_result

    if log_result is not None:
        return log_result

    # No sources at all
    return _build_empty_result(BufferSource.CHANNEL)
