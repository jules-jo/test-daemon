"""Tests for process output re-attachment via SSH.

Verifies that the reattach module:
- Probes available reattach methods (/proc/PID/fd, log file)
- Selects the best method based on remote host capabilities
- Falls back from /proc/PID/fd to log file when /proc is unavailable
- Returns error when no method is available
- Builds correct tail commands for each method
- Validates PID input (positive integer only)
- Validates config (timeout positive, encoding non-empty)
- Returns immutable (frozen dataclass) results
- Streams decoded output lines from an SSH channel
- Handles partial line buffering correctly
- Handles EOF and channel closure
- Handles encoding errors gracefully
- Measures probe latency
- Records timestamps
- Uses sequence numbers for line ordering
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest

from jules_daemon.ssh.reattach import (
    OutputLine,
    OutputStreamType,
    ReattachConfig,
    ReattachMethod,
    ReattachStrategy,
    build_reattach_command,
    probe_reattach_strategy,
    stream_output_lines,
)
from jules_daemon.ssh.liveness import ProbeExecutor
from jules_daemon.ssh.reader import SSHChannelHandle


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeCommandResult:
    """Canned result for a remote command execution."""

    stdout: str
    exit_code: int


class FakeReattachExecutor:
    """Configurable fake executor for reattach tests.

    Tracks invocations and returns canned results by command prefix
    matching or sequential order.
    """

    def __init__(
        self,
        *,
        results: list[FakeCommandResult] | None = None,
        errors: list[Exception] | None = None,
    ) -> None:
        self._results: list[FakeCommandResult] = list(results) if results else []
        self._errors: list[Exception] = list(errors) if errors else []
        self.invocations: list[str] = []

    async def execute_probe(
        self, command: str, timeout: float
    ) -> tuple[str, int]:
        self.invocations.append(command)
        if self._errors:
            raise self._errors.pop(0)
        if self._results:
            result = self._results.pop(0)
            return (result.stdout, result.exit_code)
        return ("", 0)


# Verify protocol compliance
assert isinstance(FakeReattachExecutor(), ProbeExecutor)


class FakeStreamChannel:
    """Fake SSH channel for testing stream_output_lines.

    Delivers chunks of data sequentially, then signals EOF.
    """

    def __init__(
        self,
        *,
        chunks: list[bytes] | None = None,
        stderr_chunks: list[bytes] | None = None,
        exit_code: int | None = None,
    ) -> None:
        self._stdout_chunks: list[bytes] = list(chunks) if chunks else []
        self._stderr_chunks: list[bytes] = list(stderr_chunks) if stderr_chunks else []
        self._exit_code = exit_code
        self._delivered_all = False
        self._read_count = 0

    def recv_ready(self) -> bool:
        return len(self._stdout_chunks) > 0

    def recv(self, nbytes: int) -> bytes:
        if self._stdout_chunks:
            chunk = self._stdout_chunks.pop(0)
            return chunk[:nbytes]
        return b""

    def recv_stderr_ready(self) -> bool:
        return len(self._stderr_chunks) > 0

    def recv_stderr(self, nbytes: int) -> bytes:
        if self._stderr_chunks:
            chunk = self._stderr_chunks.pop(0)
            return chunk[:nbytes]
        return b""

    @property
    def closed(self) -> bool:
        return (
            not self._stdout_chunks
            and not self._stderr_chunks
            and self._delivered_all
        )

    def eof_received(self) -> bool:
        self._read_count += 1
        # Signal EOF after all chunks are consumed and at least one
        # empty read has occurred
        if not self._stdout_chunks and not self._stderr_chunks:
            self._delivered_all = True
            return True
        return False

    def exit_status_is_ready(self) -> bool:
        return self._delivered_all and self._exit_code is not None

    def get_exit_status(self) -> int:
        if self._exit_code is None:
            raise RuntimeError("exit status not ready")
        return self._exit_code


# Verify protocol compliance
assert isinstance(FakeStreamChannel(), SSHChannelHandle)


# ---------------------------------------------------------------------------
# ReattachMethod enum
# ---------------------------------------------------------------------------


class TestReattachMethod:
    def test_all_values_exist(self) -> None:
        assert ReattachMethod.PROC_FD.value == "proc_fd"
        assert ReattachMethod.LOG_FILE.value == "log_file"


# ---------------------------------------------------------------------------
# OutputStreamType enum
# ---------------------------------------------------------------------------


class TestOutputStreamType:
    def test_all_values_exist(self) -> None:
        assert OutputStreamType.STDOUT.value == "stdout"
        assert OutputStreamType.STDERR.value == "stderr"
        assert OutputStreamType.COMBINED.value == "combined"


# ---------------------------------------------------------------------------
# ReattachConfig
# ---------------------------------------------------------------------------


class TestReattachConfig:
    def test_defaults(self) -> None:
        config = ReattachConfig()
        assert config.timeout_seconds == 5.0
        assert config.log_file_path is None
        assert config.encoding == "utf-8"

    def test_custom_values(self) -> None:
        config = ReattachConfig(
            timeout_seconds=10.0,
            log_file_path="/var/log/test.log",
            encoding="ascii",
        )
        assert config.timeout_seconds == 10.0
        assert config.log_file_path == "/var/log/test.log"
        assert config.encoding == "ascii"

    def test_frozen(self) -> None:
        config = ReattachConfig()
        with pytest.raises(AttributeError):
            config.timeout_seconds = 3.0  # type: ignore[misc]

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ReattachConfig(timeout_seconds=0.0)

    def test_timeout_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ReattachConfig(timeout_seconds=-1.0)

    def test_encoding_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="encoding must not be empty"):
            ReattachConfig(encoding="")

    def test_encoding_must_not_be_whitespace(self) -> None:
        with pytest.raises(ValueError, match="encoding must not be empty"):
            ReattachConfig(encoding="   ")


# ---------------------------------------------------------------------------
# ReattachStrategy immutability
# ---------------------------------------------------------------------------


class TestReattachStrategy:
    def test_frozen(self) -> None:
        strategy = ReattachStrategy(
            success=True,
            method=ReattachMethod.PROC_FD,
            pid=1234,
            command="tail -f /proc/1234/fd/1 2>/dev/null",
            error=None,
            latency_ms=5.0,
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            strategy.success = False  # type: ignore[misc]

    def test_has_all_fields(self) -> None:
        ts = datetime.now(timezone.utc)
        strategy = ReattachStrategy(
            success=False,
            method=None,
            pid=42,
            command="",
            error="No reattach method available",
            latency_ms=10.5,
            timestamp=ts,
        )
        assert strategy.success is False
        assert strategy.method is None
        assert strategy.pid == 42
        assert strategy.command == ""
        assert strategy.error == "No reattach method available"
        assert strategy.latency_ms == 10.5
        assert strategy.timestamp == ts


# ---------------------------------------------------------------------------
# OutputLine immutability
# ---------------------------------------------------------------------------


class TestOutputLine:
    def test_frozen(self) -> None:
        line = OutputLine(
            text="PASSED test_foo",
            stream=OutputStreamType.STDOUT,
            timestamp=datetime.now(timezone.utc),
            sequence=1,
        )
        with pytest.raises(AttributeError):
            line.text = "changed"  # type: ignore[misc]

    def test_has_all_fields(self) -> None:
        ts = datetime.now(timezone.utc)
        line = OutputLine(
            text="test output line",
            stream=OutputStreamType.COMBINED,
            timestamp=ts,
            sequence=42,
        )
        assert line.text == "test output line"
        assert line.stream == OutputStreamType.COMBINED
        assert line.timestamp == ts
        assert line.sequence == 42


# ---------------------------------------------------------------------------
# PID validation
# ---------------------------------------------------------------------------


class TestPidValidation:
    @pytest.mark.asyncio
    async def test_negative_pid_raises(self) -> None:
        executor = FakeReattachExecutor()
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            await probe_reattach_strategy(executor, pid=-1)

    @pytest.mark.asyncio
    async def test_zero_pid_raises(self) -> None:
        executor = FakeReattachExecutor()
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            await probe_reattach_strategy(executor, pid=0)

    @pytest.mark.asyncio
    async def test_valid_pid_does_not_raise(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await probe_reattach_strategy(executor, pid=1234)
        assert result.pid == 1234


# ---------------------------------------------------------------------------
# probe_reattach_strategy: /proc/PID/fd method (primary)
# ---------------------------------------------------------------------------


class TestProbeProcFd:
    """test -r /proc/<PID>/fd/1 succeeds -> PROC_FD method selected."""

    @pytest.mark.asyncio
    async def test_proc_fd_readable_returns_success(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await probe_reattach_strategy(executor, pid=5678)

        assert result.success is True
        assert result.method == ReattachMethod.PROC_FD

    @pytest.mark.asyncio
    async def test_proc_fd_command_contains_pid(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await probe_reattach_strategy(executor, pid=9999)

        assert result.pid == 9999
        assert "9999" in result.command

    @pytest.mark.asyncio
    async def test_proc_fd_command_uses_tail(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await probe_reattach_strategy(executor, pid=5678)

        assert "tail" in result.command
        assert "/proc/5678/fd/1" in result.command

    @pytest.mark.asyncio
    async def test_no_error_on_success(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await probe_reattach_strategy(executor, pid=5678)

        assert result.error is None

    @pytest.mark.asyncio
    async def test_has_timestamp(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        before = datetime.now(timezone.utc)
        result = await probe_reattach_strategy(executor, pid=5678)
        after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after

    @pytest.mark.asyncio
    async def test_latency_is_non_negative(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await probe_reattach_strategy(executor, pid=5678)

        assert result.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_probe_command_is_test_readable(self) -> None:
        """The probe should use 'test -r' to check readability."""
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        await probe_reattach_strategy(executor, pid=5678)

        assert len(executor.invocations) >= 1
        assert "test -r" in executor.invocations[0]
        assert "/proc/5678/fd/1" in executor.invocations[0]


# ---------------------------------------------------------------------------
# probe_reattach_strategy: /proc not readable, log file available
# ---------------------------------------------------------------------------


class TestProbeLogFileFallback:
    """/proc/PID/fd not readable but log_file_path exists -> LOG_FILE."""

    @pytest.mark.asyncio
    async def test_log_file_fallback_success(self) -> None:
        executor = FakeReattachExecutor(
            results=[
                # /proc/PID/fd/1 not readable
                FakeCommandResult(stdout="", exit_code=1),
                # log file exists
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        config = ReattachConfig(log_file_path="/tmp/test-output.log")
        result = await probe_reattach_strategy(executor, pid=5678, config=config)

        assert result.success is True
        assert result.method == ReattachMethod.LOG_FILE

    @pytest.mark.asyncio
    async def test_log_file_command_uses_path(self) -> None:
        executor = FakeReattachExecutor(
            results=[
                FakeCommandResult(stdout="", exit_code=1),
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        config = ReattachConfig(log_file_path="/var/log/tests.log")
        result = await probe_reattach_strategy(executor, pid=5678, config=config)

        assert "/var/log/tests.log" in result.command
        assert "tail" in result.command

    @pytest.mark.asyncio
    async def test_log_file_probe_checks_existence(self) -> None:
        """Should use 'test -f' to verify log file exists."""
        executor = FakeReattachExecutor(
            results=[
                FakeCommandResult(stdout="", exit_code=1),
                FakeCommandResult(stdout="", exit_code=0),
            ]
        )
        config = ReattachConfig(log_file_path="/tmp/output.log")
        await probe_reattach_strategy(executor, pid=5678, config=config)

        assert len(executor.invocations) == 2
        assert "test -f" in executor.invocations[1]
        assert "/tmp/output.log" in executor.invocations[1]

    @pytest.mark.asyncio
    async def test_log_file_not_found_returns_error(self) -> None:
        """Both /proc and log file fail -> error result."""
        executor = FakeReattachExecutor(
            results=[
                # /proc not readable
                FakeCommandResult(stdout="", exit_code=1),
                # log file does not exist
                FakeCommandResult(stdout="", exit_code=1),
            ]
        )
        config = ReattachConfig(log_file_path="/tmp/missing.log")
        result = await probe_reattach_strategy(executor, pid=5678, config=config)

        assert result.success is False
        assert result.method is None
        assert result.error is not None


# ---------------------------------------------------------------------------
# probe_reattach_strategy: no log file configured, /proc fails
# ---------------------------------------------------------------------------


class TestProbeNoFallback:
    """/proc not readable and no log file configured -> error."""

    @pytest.mark.asyncio
    async def test_proc_fail_no_log_file_returns_error(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=1)]
        )
        result = await probe_reattach_strategy(executor, pid=5678)

        assert result.success is False
        assert result.method is None
        assert result.error is not None
        assert result.command == ""

    @pytest.mark.asyncio
    async def test_error_message_is_descriptive(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=1)]
        )
        result = await probe_reattach_strategy(executor, pid=5678)

        assert result.error is not None
        assert len(result.error) > 0


# ---------------------------------------------------------------------------
# probe_reattach_strategy: executor timeout/error
# ---------------------------------------------------------------------------


class TestProbeExecutorErrors:
    """Executor raises errors during probing."""

    @pytest.mark.asyncio
    async def test_proc_probe_timeout_with_log_fallback(self) -> None:
        """Proc probe times out but log file probe succeeds."""
        call_count = 0

        class TimeoutThenSuccessExecutor:
            def __init__(self) -> None:
                self.invocations: list[str] = []

            async def execute_probe(
                self, command: str, timeout: float
            ) -> tuple[str, int]:
                self.invocations.append(command)
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise TimeoutError("probe timed out")
                return ("", 0)

        executor = TimeoutThenSuccessExecutor()
        config = ReattachConfig(log_file_path="/tmp/test.log")
        result = await probe_reattach_strategy(executor, pid=5678, config=config)

        assert result.success is True
        assert result.method == ReattachMethod.LOG_FILE

    @pytest.mark.asyncio
    async def test_all_probes_error_returns_failure(self) -> None:
        """All probes raise errors -> failure result."""
        executor = FakeReattachExecutor(
            errors=[
                TimeoutError("proc probe timed out"),
                OSError("log file probe failed"),
            ]
        )
        config = ReattachConfig(log_file_path="/tmp/test.log")
        result = await probe_reattach_strategy(executor, pid=5678, config=config)

        assert result.success is False
        assert result.method is None
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_proc_probe_error_no_log_fallback(self) -> None:
        """Proc probe errors with no log file -> failure."""
        executor = FakeReattachExecutor(
            errors=[OSError("connection lost")]
        )
        result = await probe_reattach_strategy(executor, pid=5678)

        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# probe_reattach_strategy: config usage
# ---------------------------------------------------------------------------


class TestProbeConfigUsage:
    @pytest.mark.asyncio
    async def test_uses_default_config_when_none(self) -> None:
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await probe_reattach_strategy(executor, pid=1234, config=None)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_custom_config_is_used(self) -> None:
        config = ReattachConfig(timeout_seconds=2.0)
        executor = FakeReattachExecutor(
            results=[FakeCommandResult(stdout="", exit_code=0)]
        )
        result = await probe_reattach_strategy(
            executor, pid=1234, config=config
        )

        assert result.success is True


# ---------------------------------------------------------------------------
# build_reattach_command
# ---------------------------------------------------------------------------


class TestBuildReattachCommand:
    def test_proc_fd_command(self) -> None:
        cmd = build_reattach_command(ReattachMethod.PROC_FD, pid=5678)
        assert "tail" in cmd
        assert "/proc/5678/fd/1" in cmd
        assert "2>/dev/null" in cmd

    def test_proc_fd_uses_follow(self) -> None:
        cmd = build_reattach_command(ReattachMethod.PROC_FD, pid=1234)
        assert "-f" in cmd or "--follow" in cmd

    def test_log_file_command(self) -> None:
        cmd = build_reattach_command(
            ReattachMethod.LOG_FILE,
            pid=5678,
            log_file_path="/var/log/test.log",
        )
        assert "tail" in cmd
        assert "/var/log/test.log" in cmd
        assert "2>/dev/null" in cmd

    def test_log_file_uses_follow(self) -> None:
        cmd = build_reattach_command(
            ReattachMethod.LOG_FILE,
            pid=5678,
            log_file_path="/tmp/out.log",
        )
        assert "-f" in cmd or "--follow" in cmd

    def test_log_file_requires_path(self) -> None:
        with pytest.raises(ValueError, match="log_file_path is required"):
            build_reattach_command(ReattachMethod.LOG_FILE, pid=5678)

    def test_pid_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            build_reattach_command(ReattachMethod.PROC_FD, pid=-1)

    def test_pid_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            build_reattach_command(ReattachMethod.PROC_FD, pid=0)

    def test_proc_fd_includes_pid_in_path(self) -> None:
        cmd = build_reattach_command(ReattachMethod.PROC_FD, pid=99999)
        assert "/proc/99999/fd/1" in cmd

    def test_log_file_command_does_not_contain_proc(self) -> None:
        cmd = build_reattach_command(
            ReattachMethod.LOG_FILE,
            pid=5678,
            log_file_path="/tmp/test.log",
        )
        assert "/proc/" not in cmd


# ---------------------------------------------------------------------------
# stream_output_lines: basic line delivery
# ---------------------------------------------------------------------------


class TestStreamOutputLinesBasic:
    @pytest.mark.asyncio
    async def test_yields_complete_lines(self) -> None:
        channel = FakeStreamChannel(chunks=[b"line1\nline2\n"])
        lines = [line async for line in stream_output_lines(channel)]

        texts = [line.text for line in lines]
        assert "line1" in texts
        assert "line2" in texts

    @pytest.mark.asyncio
    async def test_each_line_has_timestamp(self) -> None:
        channel = FakeStreamChannel(chunks=[b"hello\n"])
        before = datetime.now(timezone.utc)
        lines = [line async for line in stream_output_lines(channel)]
        after = datetime.now(timezone.utc)

        assert len(lines) >= 1
        for line in lines:
            assert before <= line.timestamp <= after

    @pytest.mark.asyncio
    async def test_each_line_has_sequence_number(self) -> None:
        channel = FakeStreamChannel(chunks=[b"a\nb\nc\n"])
        lines = [line async for line in stream_output_lines(channel)]

        sequences = [line.sequence for line in lines]
        assert sequences == sorted(sequences)
        assert all(s > 0 for s in sequences)

    @pytest.mark.asyncio
    async def test_sequence_numbers_are_monotonic(self) -> None:
        channel = FakeStreamChannel(chunks=[b"x\ny\nz\n"])
        lines = [line async for line in stream_output_lines(channel)]

        for i in range(1, len(lines)):
            assert lines[i].sequence > lines[i - 1].sequence

    @pytest.mark.asyncio
    async def test_stream_type_is_combined(self) -> None:
        channel = FakeStreamChannel(chunks=[b"out\n"])
        lines = [line async for line in stream_output_lines(channel)]

        assert len(lines) >= 1
        assert lines[0].stream == OutputStreamType.COMBINED


# ---------------------------------------------------------------------------
# stream_output_lines: partial line buffering
# ---------------------------------------------------------------------------


class TestStreamPartialLines:
    @pytest.mark.asyncio
    async def test_buffers_partial_lines_across_chunks(self) -> None:
        """Partial line at end of chunk is buffered until newline arrives."""
        channel = FakeStreamChannel(
            chunks=[b"hel", b"lo\n"]
        )
        lines = [line async for line in stream_output_lines(channel)]

        texts = [line.text for line in lines]
        assert "hello" in texts

    @pytest.mark.asyncio
    async def test_flushes_partial_line_on_eof(self) -> None:
        """Remaining buffer is flushed when channel reaches EOF."""
        channel = FakeStreamChannel(chunks=[b"no newline"])
        lines = [line async for line in stream_output_lines(channel)]

        texts = [line.text for line in lines]
        assert "no newline" in texts


# ---------------------------------------------------------------------------
# stream_output_lines: empty channel
# ---------------------------------------------------------------------------


class TestStreamEmptyChannel:
    @pytest.mark.asyncio
    async def test_empty_channel_yields_nothing(self) -> None:
        channel = FakeStreamChannel()
        lines = [line async for line in stream_output_lines(channel)]
        assert lines == []


# ---------------------------------------------------------------------------
# stream_output_lines: stderr data
# ---------------------------------------------------------------------------


class TestStreamStderr:
    @pytest.mark.asyncio
    async def test_stderr_data_is_yielded(self) -> None:
        channel = FakeStreamChannel(stderr_chunks=[b"error output\n"])
        lines = [line async for line in stream_output_lines(channel)]

        texts = [line.text for line in lines]
        assert any("error output" in t for t in texts)


# ---------------------------------------------------------------------------
# stream_output_lines: mixed stdout + stderr
# ---------------------------------------------------------------------------


class TestStreamMixed:
    @pytest.mark.asyncio
    async def test_both_streams_produce_lines(self) -> None:
        channel = FakeStreamChannel(
            chunks=[b"stdout line\n"],
            stderr_chunks=[b"stderr line\n"],
        )
        lines = [line async for line in stream_output_lines(channel)]

        texts = [line.text for line in lines]
        assert any("stdout" in t for t in texts)
        assert any("stderr" in t for t in texts)


# ---------------------------------------------------------------------------
# stream_output_lines: encoding
# ---------------------------------------------------------------------------


class TestStreamEncoding:
    @pytest.mark.asyncio
    async def test_utf8_decoding_by_default(self) -> None:
        channel = FakeStreamChannel(
            chunks=["hello world\n".encode("utf-8")]
        )
        lines = [line async for line in stream_output_lines(channel)]

        assert len(lines) >= 1
        assert lines[0].text == "hello world"

    @pytest.mark.asyncio
    async def test_invalid_bytes_replaced(self) -> None:
        """Invalid UTF-8 bytes should be replaced, not raise."""
        channel = FakeStreamChannel(
            chunks=[b"valid \xff invalid\n"]
        )
        lines = [line async for line in stream_output_lines(channel)]

        assert len(lines) >= 1
        # The replacement character should appear for the invalid byte
        assert "\ufffd" in lines[0].text or "invalid" in lines[0].text

    @pytest.mark.asyncio
    async def test_custom_encoding(self) -> None:
        channel = FakeStreamChannel(
            chunks=["hello\n".encode("ascii")]
        )
        lines = [line async for line in stream_output_lines(channel, encoding="ascii")]

        assert len(lines) >= 1
        assert lines[0].text == "hello"


# ---------------------------------------------------------------------------
# stream_output_lines: channel with exit code
# ---------------------------------------------------------------------------


class TestStreamExitCode:
    @pytest.mark.asyncio
    async def test_stops_on_eof(self) -> None:
        """Generator terminates when channel signals EOF."""
        channel = FakeStreamChannel(
            chunks=[b"final\n"],
            exit_code=0,
        )
        lines = [line async for line in stream_output_lines(channel)]

        # Should have at least the "final" line
        texts = [line.text for line in lines]
        assert "final" in texts

    @pytest.mark.asyncio
    async def test_stops_on_channel_close(self) -> None:
        """Generator terminates when channel closes."""
        channel = FakeStreamChannel(exit_code=1)
        lines = [line async for line in stream_output_lines(channel)]

        # No data -> no lines
        assert len(lines) == 0


# ---------------------------------------------------------------------------
# stream_output_lines: multiple newlines
# ---------------------------------------------------------------------------


class TestStreamMultipleNewlines:
    @pytest.mark.asyncio
    async def test_empty_lines_preserved(self) -> None:
        """Empty lines (consecutive newlines) should be yielded."""
        channel = FakeStreamChannel(chunks=[b"a\n\nb\n"])
        lines = [line async for line in stream_output_lines(channel)]

        texts = [line.text for line in lines]
        assert "a" in texts
        assert "" in texts  # empty line between a and b
        assert "b" in texts


# ---------------------------------------------------------------------------
# stream_output_lines: large output
# ---------------------------------------------------------------------------


class TestStreamLargeOutput:
    @pytest.mark.asyncio
    async def test_many_lines(self) -> None:
        """Handles many lines without issues."""
        data = "".join(f"line-{i}\n" for i in range(100))
        channel = FakeStreamChannel(chunks=[data.encode("utf-8")])
        lines = [line async for line in stream_output_lines(channel)]

        assert len(lines) >= 100
        assert lines[0].text == "line-0"
        assert lines[99].text == "line-99"
