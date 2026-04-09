"""Tests for the IPC client-side watch command.

Validates that the WatchClient correctly:
    - Connects to the daemon socket, sends a watch request, and reads
      the subscription response.
    - Enters a streaming-read loop that prints STREAM envelope payloads.
    - Detects job completion (end-of-stream) and cleanly exits.
    - Detects user interrupt (KeyboardInterrupt) and sends an unwatch
      request before disconnecting.
    - Handles error envelopes from the daemon (unregistered jobs, etc.).
    - Handles connection failures and timeouts gracefully.
    - Formats output lines with optional timestamps and sequence numbers.
"""

from __future__ import annotations

import asyncio
from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from jules_daemon.ipc.framing import (
    MessageEnvelope,
    MessageType,
    encode_frame,
)
from jules_daemon.ipc.watch_client import (
    WatchClient,
    WatchClientConfig,
    WatchClientResult,
    WatchExitReason,
    format_output_line,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TS = "2026-04-09T12:00:00Z"


# ---------------------------------------------------------------------------
# Helpers: build envelopes
# ---------------------------------------------------------------------------


def _make_watch_response(
    job_id: str = "job-abc",
    subscriber_id: str = "sub-001",
    buffered_lines: int = 0,
    msg_id: str = "req-001",
) -> MessageEnvelope:
    """Build a successful watch subscription response envelope."""
    return MessageEnvelope(
        msg_type=MessageType.RESPONSE,
        msg_id=msg_id,
        timestamp=_TS,
        payload={
            "verb": "watch",
            "status": "subscribed",
            "job_id": job_id,
            "subscriber_id": subscriber_id,
            "buffered_lines": buffered_lines,
        },
    )


def _make_stream_envelope(
    line: str,
    job_id: str = "job-abc",
    sequence: int = 0,
    is_end: bool = False,
    msg_id: str = "stream-001",
) -> MessageEnvelope:
    """Build a STREAM envelope carrying an output line."""
    return MessageEnvelope(
        msg_type=MessageType.STREAM,
        msg_id=msg_id,
        timestamp=_TS,
        payload={
            "job_id": job_id,
            "line": line,
            "sequence": sequence,
            "is_end": is_end,
        },
    )


def _make_error_envelope(
    error: str,
    msg_id: str = "req-001",
) -> MessageEnvelope:
    """Build an ERROR envelope."""
    return MessageEnvelope(
        msg_type=MessageType.ERROR,
        msg_id=msg_id,
        timestamp=_TS,
        payload={"error": error},
    )


def _make_unwatch_response(
    job_id: str = "job-abc",
    subscriber_id: str = "sub-001",
    msg_id: str = "req-002",
) -> MessageEnvelope:
    """Build a successful unwatch response envelope."""
    return MessageEnvelope(
        msg_type=MessageType.RESPONSE,
        msg_id=msg_id,
        timestamp=_TS,
        payload={
            "verb": "unwatch",
            "status": "unsubscribed",
            "job_id": job_id,
            "subscriber_id": subscriber_id,
        },
    )


def _build_stream_reader(envelopes: list[MessageEnvelope]) -> asyncio.StreamReader:
    """Build a StreamReader pre-loaded with framed envelope data."""
    reader = asyncio.StreamReader()
    for env in envelopes:
        reader.feed_data(encode_frame(env))
    reader.feed_eof()
    return reader


def _make_writer_mock() -> AsyncMock:
    """Build a mock StreamWriter that records writes."""
    writer = AsyncMock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = False
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    return writer


# ---------------------------------------------------------------------------
# WatchClientConfig tests
# ---------------------------------------------------------------------------


class TestWatchClientConfig:
    """Tests for the immutable WatchClientConfig."""

    def test_defaults(self) -> None:
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        assert config.socket_path == "/tmp/test.sock"
        assert config.connect_timeout == 5.0
        assert config.read_timeout == 10.0
        assert config.show_timestamps is False
        assert config.show_sequence is False

    def test_frozen(self) -> None:
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        with pytest.raises(AttributeError):
            config.socket_path = "/mutated"  # type: ignore[misc]

    def test_empty_socket_path_raises(self) -> None:
        with pytest.raises(ValueError, match="socket_path must not be empty"):
            WatchClientConfig(socket_path="")

    def test_negative_connect_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="connect_timeout must be positive"):
            WatchClientConfig(socket_path="/tmp/s.sock", connect_timeout=-1.0)

    def test_zero_read_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="read_timeout must be positive"):
            WatchClientConfig(socket_path="/tmp/s.sock", read_timeout=0.0)


# ---------------------------------------------------------------------------
# WatchExitReason tests
# ---------------------------------------------------------------------------


class TestWatchExitReason:
    """Tests for the WatchExitReason enum."""

    def test_all_values(self) -> None:
        assert WatchExitReason.JOB_COMPLETED.value == "job_completed"
        assert WatchExitReason.USER_INTERRUPT.value == "user_interrupt"
        assert WatchExitReason.DAEMON_ERROR.value == "daemon_error"
        assert WatchExitReason.CONNECTION_LOST.value == "connection_lost"


# ---------------------------------------------------------------------------
# WatchClientResult tests
# ---------------------------------------------------------------------------


class TestWatchClientResult:
    """Tests for the immutable WatchClientResult."""

    def test_create(self) -> None:
        result = WatchClientResult(
            exit_reason=WatchExitReason.JOB_COMPLETED,
            lines_received=42,
            job_id="job-abc",
            error_message=None,
        )
        assert result.exit_reason == WatchExitReason.JOB_COMPLETED
        assert result.lines_received == 42
        assert result.job_id == "job-abc"
        assert result.error_message is None

    def test_frozen(self) -> None:
        result = WatchClientResult(
            exit_reason=WatchExitReason.JOB_COMPLETED,
            lines_received=0,
            job_id="job-abc",
            error_message=None,
        )
        with pytest.raises(AttributeError):
            result.lines_received = 99  # type: ignore[misc]

    def test_with_error(self) -> None:
        result = WatchClientResult(
            exit_reason=WatchExitReason.DAEMON_ERROR,
            lines_received=0,
            job_id="job-abc",
            error_message="Job not registered",
        )
        assert result.error_message == "Job not registered"


# ---------------------------------------------------------------------------
# format_output_line tests
# ---------------------------------------------------------------------------


class TestFormatOutputLine:
    """Tests for the line formatting function."""

    def test_plain_line(self) -> None:
        result = format_output_line(
            line="PASSED test_foo",
            sequence=0,
            timestamp=_TS,
            show_timestamps=False,
            show_sequence=False,
        )
        assert result == "PASSED test_foo"

    def test_with_timestamp(self) -> None:
        result = format_output_line(
            line="PASSED test_foo",
            sequence=0,
            timestamp=_TS,
            show_timestamps=True,
            show_sequence=False,
        )
        assert result == f"[{_TS}] PASSED test_foo"

    def test_with_sequence(self) -> None:
        result = format_output_line(
            line="PASSED test_foo",
            sequence=7,
            timestamp=_TS,
            show_timestamps=False,
            show_sequence=True,
        )
        assert result == "[#7] PASSED test_foo"

    def test_with_both_timestamp_and_sequence(self) -> None:
        result = format_output_line(
            line="line text",
            sequence=42,
            timestamp=_TS,
            show_timestamps=True,
            show_sequence=True,
        )
        assert result == f"[{_TS}] [#42] line text"


# ---------------------------------------------------------------------------
# WatchClient -- _send_watch_request
# ---------------------------------------------------------------------------


class TestWatchClientSendRequest:
    """Tests for the watch request sending logic."""

    @pytest.mark.asyncio
    async def test_send_watch_request_writes_frame(self) -> None:
        """Sending a watch request writes a framed envelope to the writer."""
        writer = _make_writer_mock()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        await client._send_watch_request(writer, job_id="job-xyz")

        writer.write.assert_called_once()
        writer.drain.assert_awaited_once()

        # Verify the frame is a valid REQUEST envelope
        frame_bytes = writer.write.call_args[0][0]
        assert isinstance(frame_bytes, bytes)
        assert len(frame_bytes) > 4  # header + payload

    @pytest.mark.asyncio
    async def test_send_watch_request_includes_job_id(self) -> None:
        """The sent envelope includes the job_id in the payload."""
        writer = _make_writer_mock()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        await client._send_watch_request(writer, job_id="my-job-123")

        frame_bytes = writer.write.call_args[0][0]
        # The job_id should appear in the serialized JSON
        assert b"my-job-123" in frame_bytes


# ---------------------------------------------------------------------------
# WatchClient -- _send_unwatch_request
# ---------------------------------------------------------------------------


class TestWatchClientSendUnwatchRequest:
    """Tests for the unwatch request sending logic."""

    @pytest.mark.asyncio
    async def test_send_unwatch_request_writes_frame(self) -> None:
        writer = _make_writer_mock()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        await client._send_unwatch_request(
            writer, job_id="job-abc", subscriber_id="sub-001"
        )

        writer.write.assert_called_once()
        writer.drain.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_unwatch_includes_ids(self) -> None:
        writer = _make_writer_mock()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        await client._send_unwatch_request(
            writer, job_id="job-xyz", subscriber_id="sub-999"
        )

        frame_bytes = writer.write.call_args[0][0]
        assert b"job-xyz" in frame_bytes
        assert b"sub-999" in frame_bytes


# ---------------------------------------------------------------------------
# WatchClient -- _read_envelope
# ---------------------------------------------------------------------------


class TestWatchClientReadEnvelope:
    """Tests for reading a single framed envelope from the stream."""

    @pytest.mark.asyncio
    async def test_read_valid_envelope(self) -> None:
        env = _make_watch_response()
        reader = _build_stream_reader([env])
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        result = await client._read_envelope(reader)

        assert result is not None
        assert result.msg_type == MessageType.RESPONSE
        assert result.payload["status"] == "subscribed"

    @pytest.mark.asyncio
    async def test_read_returns_none_on_eof(self) -> None:
        reader = asyncio.StreamReader()
        reader.feed_eof()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        result = await client._read_envelope(reader)
        assert result is None

    @pytest.mark.asyncio
    async def test_read_returns_none_on_incomplete_payload(self) -> None:
        """If the connection drops mid-payload, returns None."""
        from jules_daemon.ipc.framing import pack_header

        reader = asyncio.StreamReader()
        reader.feed_data(pack_header(100))
        reader.feed_data(b"short")
        reader.feed_eof()

        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        result = await client._read_envelope(reader)
        assert result is None


# ---------------------------------------------------------------------------
# WatchClient -- _stream_loop (the core streaming loop)
# ---------------------------------------------------------------------------


class TestWatchClientStreamLoop:
    """Tests for the streaming output loop."""

    @pytest.mark.asyncio
    async def test_prints_stream_lines_and_exits_on_end(self) -> None:
        """Lines are printed in order; end sentinel causes clean exit."""
        envelopes = [
            _make_stream_envelope("line 1", sequence=0),
            _make_stream_envelope("line 2", sequence=1),
            _make_stream_envelope("", sequence=2, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        exit_reason, lines_count = await client._stream_loop(reader)

        assert exit_reason == WatchExitReason.JOB_COMPLETED
        assert lines_count == 2

        printed = output.getvalue()
        assert "line 1" in printed
        assert "line 2" in printed

    @pytest.mark.asyncio
    async def test_exits_on_eof(self) -> None:
        """EOF (connection lost) exits with CONNECTION_LOST."""
        envelopes = [
            _make_stream_envelope("partial line", sequence=0),
        ]
        reader = _build_stream_reader(envelopes)
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        exit_reason, lines_count = await client._stream_loop(reader)

        assert exit_reason == WatchExitReason.CONNECTION_LOST
        assert lines_count == 1

    @pytest.mark.asyncio
    async def test_formats_with_timestamp(self) -> None:
        envelopes = [
            _make_stream_envelope("test line", sequence=5),
            _make_stream_envelope("", sequence=6, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        output = StringIO()
        config = WatchClientConfig(
            socket_path="/tmp/test.sock",
            show_timestamps=True,
        )
        client = WatchClient(config=config, output=output)

        await client._stream_loop(reader)

        printed = output.getvalue()
        assert f"[{_TS}]" in printed

    @pytest.mark.asyncio
    async def test_formats_with_sequence(self) -> None:
        envelopes = [
            _make_stream_envelope("test line", sequence=3),
            _make_stream_envelope("", sequence=4, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        output = StringIO()
        config = WatchClientConfig(
            socket_path="/tmp/test.sock",
            show_sequence=True,
        )
        client = WatchClient(config=config, output=output)

        await client._stream_loop(reader)

        printed = output.getvalue()
        assert "[#3]" in printed

    @pytest.mark.asyncio
    async def test_handles_error_envelope_in_stream(self) -> None:
        """An ERROR envelope during streaming produces DAEMON_ERROR exit."""
        envelopes = [
            _make_stream_envelope("good line", sequence=0),
            _make_error_envelope("something went wrong", msg_id="err-1"),
        ]
        reader = _build_stream_reader(envelopes)
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        exit_reason, lines_count = await client._stream_loop(reader)

        assert exit_reason == WatchExitReason.DAEMON_ERROR
        assert lines_count == 1

    @pytest.mark.asyncio
    async def test_ignores_non_stream_non_error_envelopes(self) -> None:
        """RESPONSE envelopes during streaming are skipped silently."""
        envelopes = [
            _make_stream_envelope("line 1", sequence=0),
            # A stray RESPONSE envelope (maybe from an unwatch ack)
            MessageEnvelope(
                msg_type=MessageType.RESPONSE,
                msg_id="stray-resp",
                timestamp=_TS,
                payload={"verb": "unwatch", "status": "unsubscribed"},
            ),
            _make_stream_envelope("line 2", sequence=1),
            _make_stream_envelope("", sequence=2, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        exit_reason, lines_count = await client._stream_loop(reader)

        assert exit_reason == WatchExitReason.JOB_COMPLETED
        assert lines_count == 2


# ---------------------------------------------------------------------------
# WatchClient -- _parse_subscription_response
# ---------------------------------------------------------------------------


class TestParseSubscriptionResponse:
    """Tests for parsing the daemon's watch subscription response."""

    def test_parse_success_response(self) -> None:
        env = _make_watch_response(
            job_id="job-abc",
            subscriber_id="sub-001",
            buffered_lines=10,
        )
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        job_id, sub_id, buffered = client._parse_subscription_response(env)

        assert job_id == "job-abc"
        assert sub_id == "sub-001"
        assert buffered == 10

    def test_parse_error_response_raises(self) -> None:
        env = _make_error_envelope("Job not registered")
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        with pytest.raises(ValueError, match="Job not registered"):
            client._parse_subscription_response(env)

    def test_parse_missing_subscriber_id_raises(self) -> None:
        env = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="req-001",
            timestamp=_TS,
            payload={"verb": "watch", "status": "subscribed", "job_id": "j"},
        )
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        with pytest.raises(ValueError, match="subscriber_id"):
            client._parse_subscription_response(env)


# ---------------------------------------------------------------------------
# WatchClient -- full run integration (using mock transport)
# ---------------------------------------------------------------------------


class TestWatchClientRun:
    """Integration tests for WatchClient.run() using mock transport."""

    @pytest.mark.asyncio
    async def test_full_run_prints_lines_and_completes(self) -> None:
        """Happy path: subscribe, receive lines, end-of-stream, clean exit."""
        envelopes = [
            _make_watch_response(),
            _make_stream_envelope("output 1", sequence=0),
            _make_stream_envelope("output 2", sequence=1),
            _make_stream_envelope("", sequence=2, is_end=True),
        ]
        reader = _build_stream_reader(envelopes)
        writer = _make_writer_mock()
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        result = await client._run_with_transport(
            reader=reader,
            writer=writer,
            job_id="job-abc",
        )

        assert result.exit_reason == WatchExitReason.JOB_COMPLETED
        assert result.lines_received == 2
        assert result.job_id == "job-abc"
        assert result.error_message is None

        printed = output.getvalue()
        assert "output 1" in printed
        assert "output 2" in printed

    @pytest.mark.asyncio
    async def test_run_with_daemon_error(self) -> None:
        """Daemon returns ERROR instead of subscription response."""
        envelopes = [
            _make_error_envelope("Job 'job-abc' is not registered"),
        ]
        reader = _build_stream_reader(envelopes)
        writer = _make_writer_mock()
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        result = await client._run_with_transport(
            reader=reader,
            writer=writer,
            job_id="job-abc",
        )

        assert result.exit_reason == WatchExitReason.DAEMON_ERROR
        assert result.lines_received == 0
        assert "not registered" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_run_connection_lost_during_subscribe(self) -> None:
        """Connection drops before subscription response is received."""
        reader = asyncio.StreamReader()
        reader.feed_eof()
        writer = _make_writer_mock()
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        result = await client._run_with_transport(
            reader=reader,
            writer=writer,
            job_id="job-abc",
        )

        assert result.exit_reason == WatchExitReason.CONNECTION_LOST
        assert result.lines_received == 0

    @pytest.mark.asyncio
    async def test_run_connection_lost_during_stream(self) -> None:
        """Connection drops after subscribe but during streaming."""
        envelopes = [
            _make_watch_response(),
            _make_stream_envelope("partial", sequence=0),
            # EOF follows (no end sentinel)
        ]
        reader = _build_stream_reader(envelopes)
        writer = _make_writer_mock()
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        result = await client._run_with_transport(
            reader=reader,
            writer=writer,
            job_id="job-abc",
        )

        assert result.exit_reason == WatchExitReason.CONNECTION_LOST
        assert result.lines_received == 1

    @pytest.mark.asyncio
    async def test_run_sends_unwatch_on_completion(self) -> None:
        """After job completes, an unwatch request is sent for cleanup."""
        envelopes = [
            _make_watch_response(subscriber_id="sub-cleanup"),
            _make_stream_envelope("done", sequence=0),
            _make_stream_envelope("", sequence=1, is_end=True),
            # The daemon might respond with an unwatch response
            _make_unwatch_response(subscriber_id="sub-cleanup"),
        ]
        reader = _build_stream_reader(envelopes)
        writer = _make_writer_mock()
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        result = await client._run_with_transport(
            reader=reader,
            writer=writer,
            job_id="job-abc",
        )

        assert result.exit_reason == WatchExitReason.JOB_COMPLETED
        # Should have sent at least 2 frames: watch + unwatch
        assert writer.write.call_count >= 2

    @pytest.mark.asyncio
    async def test_run_with_job_id_none_uses_current(self) -> None:
        """When job_id is None, the payload should not include job_id."""
        envelopes = [
            _make_watch_response(job_id="current-run"),
            _make_stream_envelope("running", sequence=0, job_id="current-run"),
            _make_stream_envelope("", sequence=1, is_end=True, job_id="current-run"),
        ]
        reader = _build_stream_reader(envelopes)
        writer = _make_writer_mock()
        output = StringIO()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=output)

        result = await client._run_with_transport(
            reader=reader,
            writer=writer,
            job_id=None,
        )

        assert result.exit_reason == WatchExitReason.JOB_COMPLETED
        assert result.job_id == "current-run"


# ---------------------------------------------------------------------------
# WatchClient -- close_transport
# ---------------------------------------------------------------------------


class TestCloseTransport:
    """Tests for clean transport teardown."""

    @pytest.mark.asyncio
    async def test_close_transport_closes_writer(self) -> None:
        writer = _make_writer_mock()
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        await client._close_transport(writer)

        writer.close.assert_called_once()
        writer.wait_closed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_transport_handles_already_closing(self) -> None:
        writer = _make_writer_mock()
        writer.is_closing.return_value = True
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        # Should not raise
        await client._close_transport(writer)
        writer.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_transport_handles_oserror(self) -> None:
        writer = _make_writer_mock()
        writer.close.side_effect = OSError("connection reset")
        config = WatchClientConfig(socket_path="/tmp/test.sock")
        client = WatchClient(config=config, output=StringIO())

        # Should not raise -- errors are swallowed
        await client._close_transport(writer)


# ---------------------------------------------------------------------------
# WatchClient -- cancellation (user interrupt) path
# ---------------------------------------------------------------------------


class TestWatchClientCancellation:
    """Tests for Ctrl+C / task cancellation handling."""

    @pytest.mark.asyncio
    async def test_stream_loop_returns_user_interrupt_on_cancel(self) -> None:
        """Cancelling the stream loop task returns USER_INTERRUPT
        with the correct partial line count."""
        # StreamReader that never sends EOF -- will block forever
        reader = asyncio.StreamReader()
        # Feed a couple of lines then leave the reader open (no EOF)
        for env in [
            _make_stream_envelope("line 1", sequence=0),
            _make_stream_envelope("line 2", sequence=1),
        ]:
            reader.feed_data(encode_frame(env))

        output = StringIO()
        config = WatchClientConfig(
            socket_path="/tmp/test.sock",
            read_timeout=60.0,  # high timeout so it blocks on the 3rd read
        )
        client = WatchClient(config=config, output=output)

        task = asyncio.create_task(client._stream_loop(reader))
        # Let the loop consume the two lines
        await asyncio.sleep(0.05)
        task.cancel()

        exit_reason, lines_count = await task

        assert exit_reason == WatchExitReason.USER_INTERRUPT
        # Should preserve the 2 lines already printed
        assert lines_count == 2

        printed = output.getvalue()
        assert "line 1" in printed
        assert "line 2" in printed

    @pytest.mark.asyncio
    async def test_full_run_cancel_preserves_line_count(self) -> None:
        """Cancelling _run_with_transport mid-stream preserves the
        partial line count in the result."""
        # Subscription response + two lines, then hang
        reader = asyncio.StreamReader()
        for env in [
            _make_watch_response(),
            _make_stream_envelope("output A", sequence=0),
            _make_stream_envelope("output B", sequence=1),
        ]:
            reader.feed_data(encode_frame(env))
        # Do NOT feed EOF -- reader will block on the next read

        writer = _make_writer_mock()
        output = StringIO()
        config = WatchClientConfig(
            socket_path="/tmp/test.sock",
            read_timeout=60.0,
        )
        client = WatchClient(config=config, output=output)

        task = asyncio.create_task(
            client._run_with_transport(
                reader=reader,
                writer=writer,
                job_id="job-abc",
            )
        )
        # Let the loop consume the subscription + two lines
        await asyncio.sleep(0.05)
        task.cancel()

        result = await task

        assert result.exit_reason == WatchExitReason.USER_INTERRUPT
        assert result.lines_received == 2
        assert result.job_id == "job-abc"
        assert result.error_message is None


# ---------------------------------------------------------------------------
# WatchClient -- read_timeout enforcement
# ---------------------------------------------------------------------------


class TestWatchClientReadTimeout:
    """Tests that read_timeout is enforced during streaming."""

    @pytest.mark.asyncio
    async def test_read_envelope_returns_none_on_timeout(self) -> None:
        """When no data arrives within read_timeout, returns None."""
        reader = asyncio.StreamReader()
        # Do not feed any data -- reader will block
        config = WatchClientConfig(
            socket_path="/tmp/test.sock",
            read_timeout=0.05,  # 50ms timeout
        )
        client = WatchClient(config=config, output=StringIO())

        result = await client._read_envelope(reader)

        assert result is None

    @pytest.mark.asyncio
    async def test_stream_loop_exits_connection_lost_on_timeout(self) -> None:
        """Stream loop treats read timeout as CONNECTION_LOST."""
        reader = asyncio.StreamReader()
        # Feed one line, then hang
        reader.feed_data(
            encode_frame(_make_stream_envelope("before timeout", sequence=0))
        )

        output = StringIO()
        config = WatchClientConfig(
            socket_path="/tmp/test.sock",
            read_timeout=0.05,
        )
        client = WatchClient(config=config, output=output)

        exit_reason, lines_count = await client._stream_loop(reader)

        assert exit_reason == WatchExitReason.CONNECTION_LOST
        assert lines_count == 1
        assert "before timeout" in output.getvalue()
