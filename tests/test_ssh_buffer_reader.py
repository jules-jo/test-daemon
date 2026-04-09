"""Tests for SSH output buffer reader.

Covers:
    - BufferReadResult frozen dataclass behavior and defaults
    - LogFileEntry frozen dataclass validation
    - BufferSource enum values
    - read_channel_buffer() -- retrieves buffered but unprocessed output
      from a live SSH channel (Paramiko transport layer)
    - read_session_log() -- reads from local session log files for stale
      or interrupted sessions
    - read_buffered_output() -- unified entry point combining both sources
    - Edge cases: empty channel, empty log file, missing log file,
      binary-safe decoding, large buffers, concurrent reads
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.ssh.buffer_reader import (
    BufferReadResult,
    BufferSource,
    LogFileEntry,
    read_buffered_output,
    read_channel_buffer,
    read_session_log,
)


# ---------------------------------------------------------------------------
# Fake SSH channel for testing (same pattern as test_ssh_reader.py)
# ---------------------------------------------------------------------------


class FakeSSHChannel:
    """Fake SSH channel implementing the SSHChannelHandle protocol."""

    def __init__(
        self,
        *,
        stdout_buffer: bytes = b"",
        stderr_buffer: bytes = b"",
        is_closed: bool = False,
        exit_status: int | None = None,
        exit_status_ready: bool = False,
        is_eof: bool = False,
        raise_on_recv: Exception | None = None,
    ) -> None:
        self._stdout_buffer = stdout_buffer
        self._stderr_buffer = stderr_buffer
        self._closed = is_closed
        self._exit_status = exit_status
        self._exit_status_ready = exit_status_ready
        self._is_eof = is_eof
        self._raise_on_recv = raise_on_recv

    def recv_ready(self) -> bool:
        return len(self._stdout_buffer) > 0

    def recv(self, nbytes: int) -> bytes:
        if self._raise_on_recv is not None:
            raise self._raise_on_recv
        chunk = self._stdout_buffer[:nbytes]
        self._stdout_buffer = self._stdout_buffer[nbytes:]
        return chunk

    def recv_stderr_ready(self) -> bool:
        return len(self._stderr_buffer) > 0

    def recv_stderr(self, nbytes: int) -> bytes:
        if self._raise_on_recv is not None:
            raise self._raise_on_recv
        chunk = self._stderr_buffer[:nbytes]
        self._stderr_buffer = self._stderr_buffer[nbytes:]
        return chunk

    @property
    def closed(self) -> bool:
        return self._closed

    def eof_received(self) -> bool:
        return self._is_eof

    def exit_status_is_ready(self) -> bool:
        return self._exit_status_ready

    def get_exit_status(self) -> int:
        if self._exit_status is None:
            raise RuntimeError(
                "get_exit_status called before exit status is ready"
            )
        return self._exit_status


# ---------------------------------------------------------------------------
# BufferReadResult tests
# ---------------------------------------------------------------------------


class TestBufferReadResult:
    def test_create_with_defaults(self) -> None:
        result = BufferReadResult(
            source=BufferSource.CHANNEL,
            stdout_text="",
            stderr_text="",
            lines=(),
        )
        assert result.source == BufferSource.CHANNEL
        assert result.stdout_text == ""
        assert result.stderr_text == ""
        assert result.lines == ()
        assert result.is_complete is False
        assert result.exit_code is None
        assert result.error is None
        assert result.bytes_read == 0
        assert isinstance(result.timestamp, datetime)

    def test_create_with_data(self) -> None:
        now = datetime.now(timezone.utc)
        result = BufferReadResult(
            source=BufferSource.LOG_FILE,
            stdout_text="PASSED test_one\nPASSED test_two\n",
            stderr_text="warning: deprecated\n",
            lines=("PASSED test_one", "PASSED test_two"),
            is_complete=True,
            exit_code=0,
            error=None,
            bytes_read=48,
            log_file_path=Path("/tmp/session.log"),
            timestamp=now,
        )
        assert result.source == BufferSource.LOG_FILE
        assert result.lines == ("PASSED test_one", "PASSED test_two")
        assert result.is_complete is True
        assert result.exit_code == 0
        assert result.bytes_read == 48
        assert result.log_file_path == Path("/tmp/session.log")
        assert result.timestamp == now

    def test_frozen(self) -> None:
        result = BufferReadResult(
            source=BufferSource.CHANNEL,
            stdout_text="data",
            stderr_text="",
            lines=(),
        )
        with pytest.raises(FrozenInstanceError):
            result.stdout_text = "other"  # type: ignore[misc]

    def test_has_data_property(self) -> None:
        result_with = BufferReadResult(
            source=BufferSource.CHANNEL,
            stdout_text="data",
            stderr_text="",
            lines=("data",),
        )
        assert result_with.has_data is True

        result_without = BufferReadResult(
            source=BufferSource.CHANNEL,
            stdout_text="",
            stderr_text="",
            lines=(),
        )
        assert result_without.has_data is False

    def test_line_count(self) -> None:
        result = BufferReadResult(
            source=BufferSource.CHANNEL,
            stdout_text="line1\nline2\n",
            stderr_text="",
            lines=("line1", "line2"),
        )
        assert result.line_count == 2


class TestBufferSource:
    def test_enum_values(self) -> None:
        assert BufferSource.CHANNEL.value == "channel"
        assert BufferSource.LOG_FILE.value == "log_file"
        assert BufferSource.COMBINED.value == "combined"


# ---------------------------------------------------------------------------
# LogFileEntry tests
# ---------------------------------------------------------------------------


class TestLogFileEntry:
    def test_create(self) -> None:
        entry = LogFileEntry(
            path=Path("/tmp/test.log"),
            run_id="abc-123",
        )
        assert entry.path == Path("/tmp/test.log")
        assert entry.run_id == "abc-123"
        assert entry.offset == 0
        assert entry.encoding == "utf-8"

    def test_with_offset(self) -> None:
        entry = LogFileEntry(
            path=Path("/tmp/test.log"),
            run_id="abc-123",
            offset=1024,
        )
        assert entry.offset == 1024

    def test_frozen(self) -> None:
        entry = LogFileEntry(
            path=Path("/tmp/test.log"),
            run_id="abc-123",
        )
        with pytest.raises(FrozenInstanceError):
            entry.path = Path("/other")  # type: ignore[misc]

    def test_negative_offset_raises(self) -> None:
        with pytest.raises(ValueError, match="offset must not be negative"):
            LogFileEntry(
                path=Path("/tmp/test.log"),
                run_id="abc-123",
                offset=-1,
            )


# ---------------------------------------------------------------------------
# read_channel_buffer() async tests
# ---------------------------------------------------------------------------


class TestReadChannelBuffer:
    @pytest.mark.asyncio
    async def test_reads_stdout_from_channel(self) -> None:
        channel = FakeSSHChannel(stdout_buffer=b"PASSED test_foo\n")
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert result.source == BufferSource.CHANNEL
        assert "PASSED test_foo" in result.stdout_text
        assert result.has_data is True

    @pytest.mark.asyncio
    async def test_reads_stderr_from_channel(self) -> None:
        channel = FakeSSHChannel(stderr_buffer=b"ERROR: timeout\n")
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert result.source == BufferSource.CHANNEL
        assert "ERROR: timeout" in result.stderr_text
        assert result.has_data is True

    @pytest.mark.asyncio
    async def test_reads_both_streams(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"output\n",
            stderr_buffer=b"warning\n",
        )
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert "output" in result.stdout_text
        assert "warning" in result.stderr_text
        assert result.has_data is True

    @pytest.mark.asyncio
    async def test_empty_channel_returns_empty_result(self) -> None:
        channel = FakeSSHChannel()
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert result.has_data is False
        assert result.stdout_text == ""
        assert result.stderr_text == ""

    @pytest.mark.asyncio
    async def test_channel_closed_reports_is_complete(self) -> None:
        channel = FakeSSHChannel(
            is_closed=True,
            is_eof=True,
            exit_status_ready=True,
            exit_status=0,
        )
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert result.is_complete is True
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_extracts_lines_from_stdout(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"line1\nline2\nline3\n",
        )
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert "line1" in result.lines
        assert "line2" in result.lines
        assert "line3" in result.lines
        assert result.line_count == 3

    @pytest.mark.asyncio
    async def test_handles_partial_line_no_trailing_newline(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"line1\npartial",
        )
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        # Partial lines should still be included
        assert "line1" in result.lines
        assert "partial" in result.lines

    @pytest.mark.asyncio
    async def test_binary_safe_decode_replaces_invalid(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"valid\xff\xfeinvalid\n",
        )
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert result.has_data is True
        # Invalid bytes should be replaced, not raise
        assert "valid" in result.stdout_text

    @pytest.mark.asyncio
    async def test_captures_oserror_from_channel(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"data",
            raise_on_recv=OSError("Connection reset"),
        )
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert result.error is not None
        assert "Connection reset" in result.error

    @pytest.mark.asyncio
    async def test_captures_unexpected_exception_from_channel(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"data",
            raise_on_recv=RuntimeError("Unexpected SSH library error"),
        )
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "Unexpected SSH library error" in result.error

    @pytest.mark.asyncio
    async def test_bytes_read_tracks_total(self) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"hello",
            stderr_buffer=b"world",
        )
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        assert result.bytes_read == 10

    @pytest.mark.asyncio
    async def test_respects_max_bytes(self) -> None:
        channel = FakeSSHChannel(stdout_buffer=b"0123456789")
        result = await read_channel_buffer(
            channel,  # type: ignore[arg-type]
            max_bytes=5,
        )
        assert result.bytes_read <= 5

    @pytest.mark.asyncio
    async def test_result_has_utc_timestamp(self) -> None:
        channel = FakeSSHChannel()
        before = datetime.now(timezone.utc)
        result = await read_channel_buffer(channel)  # type: ignore[arg-type]
        after = datetime.now(timezone.utc)
        assert before <= result.timestamp <= after


# ---------------------------------------------------------------------------
# read_session_log() tests
# ---------------------------------------------------------------------------


class TestReadSessionLog:
    def test_reads_full_log_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "session.log"
        log_file.write_text(
            "PASSED test_one\nPASSED test_two\nFAILED test_three\n",
            encoding="utf-8",
        )
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
        )
        result = read_session_log(entry)
        assert result.source == BufferSource.LOG_FILE
        assert result.has_data is True
        assert result.line_count == 3
        assert "PASSED test_one" in result.lines
        assert "PASSED test_two" in result.lines
        assert "FAILED test_three" in result.lines
        assert result.log_file_path == log_file

    def test_reads_from_offset(self, tmp_path: Path) -> None:
        log_file = tmp_path / "session.log"
        content = "PASSED test_one\nPASSED test_two\nFAILED test_three\n"
        log_file.write_text(content, encoding="utf-8")
        # Offset past the first line ("PASSED test_one\n" = 16 bytes)
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
            offset=16,
        )
        result = read_session_log(entry)
        assert result.has_data is True
        assert "PASSED test_one" not in result.lines
        assert "PASSED test_two" in result.lines

    def test_missing_log_file_returns_error(self) -> None:
        entry = LogFileEntry(
            path=Path("/nonexistent/session.log"),
            run_id="test-run-1",
        )
        result = read_session_log(entry)
        assert result.has_data is False
        assert result.error is not None
        assert "not found" in result.error.lower() or "no such" in result.error.lower()

    def test_empty_log_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "empty.log"
        log_file.write_text("", encoding="utf-8")
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
        )
        result = read_session_log(entry)
        assert result.has_data is False
        assert result.line_count == 0

    def test_binary_safe_decode(self, tmp_path: Path) -> None:
        log_file = tmp_path / "binary.log"
        log_file.write_bytes(b"valid\xff\xfeline\nmore\n")
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
        )
        result = read_session_log(entry)
        assert result.has_data is True
        # Invalid bytes should be replaced, not raise
        assert "valid" in result.stdout_text

    def test_large_file_with_max_bytes(self, tmp_path: Path) -> None:
        log_file = tmp_path / "large.log"
        # Write a large file
        content = "x" * 100_000 + "\n"
        log_file.write_text(content, encoding="utf-8")
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
        )
        result = read_session_log(entry, max_bytes=1024)
        assert result.bytes_read <= 1024

    def test_bytes_read_accurate(self, tmp_path: Path) -> None:
        log_file = tmp_path / "session.log"
        content = "line one\nline two\n"
        log_file.write_text(content, encoding="utf-8")
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
        )
        result = read_session_log(entry)
        assert result.bytes_read == len(content.encode("utf-8"))

    def test_is_complete_for_fully_read_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "session.log"
        log_file.write_text("one\ntwo\n", encoding="utf-8")
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
        )
        result = read_session_log(entry)
        # Reading from beginning with no max_bytes means the whole file
        assert result.is_complete is True

    def test_offset_past_end_of_file_returns_empty(
        self, tmp_path: Path
    ) -> None:
        log_file = tmp_path / "short.log"
        log_file.write_text("short\n", encoding="utf-8")
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
            offset=99999,
        )
        result = read_session_log(entry)
        assert result.has_data is False
        assert result.is_complete is True
        assert result.bytes_read == 0

    def test_oserror_during_read_returns_error(
        self, tmp_path: Path
    ) -> None:
        log_dir = tmp_path / "unreadable.log"
        log_dir.mkdir()  # directory, not a file -- reading it causes IsADirectoryError
        entry = LogFileEntry(
            path=log_dir,
            run_id="test-run-1",
        )
        result = read_session_log(entry)
        assert result.error is not None
        assert result.has_data is False

    def test_custom_encoding(self, tmp_path: Path) -> None:
        log_file = tmp_path / "latin.log"
        log_file.write_bytes("caf\xe9\n".encode("latin-1"))
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
            encoding="latin-1",
        )
        result = read_session_log(entry)
        assert result.has_data is True
        assert "caf" in result.stdout_text


# ---------------------------------------------------------------------------
# read_buffered_output() tests
# ---------------------------------------------------------------------------


class TestReadBufferedOutput:
    @pytest.mark.asyncio
    async def test_channel_only(self) -> None:
        channel = FakeSSHChannel(stdout_buffer=b"from channel\n")
        result = await read_buffered_output(
            channel=channel,  # type: ignore[arg-type]
        )
        assert result.source == BufferSource.CHANNEL
        assert "from channel" in result.stdout_text

    @pytest.mark.asyncio
    async def test_log_file_only(self, tmp_path: Path) -> None:
        log_file = tmp_path / "session.log"
        log_file.write_text("from log\n", encoding="utf-8")
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
        )
        result = await read_buffered_output(log_entry=entry)
        assert result.source == BufferSource.LOG_FILE
        assert "from log" in result.stdout_text

    @pytest.mark.asyncio
    async def test_combined_sources(self, tmp_path: Path) -> None:
        channel = FakeSSHChannel(stdout_buffer=b"from channel\n")
        log_file = tmp_path / "session.log"
        log_file.write_text("from log\n", encoding="utf-8")
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
        )
        result = await read_buffered_output(
            channel=channel,  # type: ignore[arg-type]
            log_entry=entry,
        )
        assert result.source == BufferSource.COMBINED
        assert result.has_data is True

    @pytest.mark.asyncio
    async def test_no_sources_returns_empty(self) -> None:
        result = await read_buffered_output()
        assert result.has_data is False
        assert result.source == BufferSource.CHANNEL

    @pytest.mark.asyncio
    async def test_neither_has_data_prefers_channel_result(self) -> None:
        """When both channel and log are provided but neither has data."""
        channel = FakeSSHChannel()  # empty channel
        result = await read_buffered_output(
            channel=channel,  # type: ignore[arg-type]
        )
        assert result.source == BufferSource.CHANNEL
        assert result.has_data is False

    @pytest.mark.asyncio
    async def test_log_only_no_data_returns_log_result(
        self, tmp_path: Path
    ) -> None:
        """When only log is provided but the file is empty."""
        log_file = tmp_path / "empty.log"
        log_file.write_text("", encoding="utf-8")
        entry = LogFileEntry(path=log_file, run_id="test-run-1")
        result = await read_buffered_output(log_entry=entry)
        assert result.source == BufferSource.LOG_FILE
        assert result.has_data is False

    @pytest.mark.asyncio
    async def test_channel_error_falls_back_to_log(
        self, tmp_path: Path
    ) -> None:
        channel = FakeSSHChannel(
            stdout_buffer=b"data",
            raise_on_recv=OSError("broken"),
        )
        log_file = tmp_path / "session.log"
        log_file.write_text("fallback data\n", encoding="utf-8")
        entry = LogFileEntry(
            path=log_file,
            run_id="test-run-1",
        )
        result = await read_buffered_output(
            channel=channel,  # type: ignore[arg-type]
            log_entry=entry,
        )
        # Should still return data from the log file
        assert result.has_data is True
        assert "fallback data" in result.lines
