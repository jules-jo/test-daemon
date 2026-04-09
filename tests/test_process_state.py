"""Tests for local process-state checker.

Verifies that the process-state checker:
- Accepts a list of local PIDs and returns alive/dead verdict per PID
- Uses os.kill(pid, 0) to probe the OS process table
- Returns ProcessVerdict.ALIVE when the process is running (no error)
- Returns ProcessVerdict.ALIVE when PermissionError occurs (EPERM: process
  exists but caller lacks permission to signal it)
- Returns ProcessVerdict.DEAD when ProcessLookupError occurs (ESRCH: no
  such process)
- Returns ProcessVerdict.DEAD when OSError with ESRCH errno occurs
- Returns ProcessVerdict.ERROR for unexpected OSError (e.g., EINVAL)
- Handles empty PID list gracefully (returns empty mapping)
- Validates PID inputs (positive integers only)
- Returns immutable (frozen dataclass) results
- Handles race conditions (process exits between check and result use)
- Measures check latency
- Records timestamps
- Works with a single PID convenience wrapper
"""

from __future__ import annotations

import errno
import os
import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from jules_daemon.monitor.process_state import (
    ProcessCheckResult,
    ProcessVerdict,
    check_pid,
    check_pids,
)


# ---------------------------------------------------------------------------
# ProcessVerdict enum values
# ---------------------------------------------------------------------------


class TestProcessVerdict:
    def test_all_values_exist(self) -> None:
        assert ProcessVerdict.ALIVE.value == "alive"
        assert ProcessVerdict.DEAD.value == "dead"
        assert ProcessVerdict.ERROR.value == "error"


# ---------------------------------------------------------------------------
# ProcessCheckResult immutability
# ---------------------------------------------------------------------------


class TestProcessCheckResult:
    def test_frozen(self) -> None:
        result = ProcessCheckResult(
            pid=1234,
            verdict=ProcessVerdict.ALIVE,
            error=None,
            latency_ms=1.0,
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            result.verdict = ProcessVerdict.DEAD  # type: ignore[misc]

    def test_has_all_fields(self) -> None:
        ts = datetime.now(timezone.utc)
        result = ProcessCheckResult(
            pid=42,
            verdict=ProcessVerdict.DEAD,
            error="No such process",
            latency_ms=0.5,
            timestamp=ts,
        )
        assert result.pid == 42
        assert result.verdict == ProcessVerdict.DEAD
        assert result.error == "No such process"
        assert result.latency_ms == 0.5
        assert result.timestamp == ts


# ---------------------------------------------------------------------------
# check_pid: process alive (os.kill succeeds with no error)
# ---------------------------------------------------------------------------


class TestCheckPidAlive:
    def test_alive_process(self) -> None:
        """os.kill(pid, 0) succeeds -> ALIVE."""
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            mock_kill.return_value = None
            result = check_pid(1234)

        assert result.pid == 1234
        assert result.verdict == ProcessVerdict.ALIVE
        assert result.error is None
        mock_kill.assert_called_once_with(1234, 0)

    def test_alive_has_timestamp(self) -> None:
        with patch("jules_daemon.monitor.process_state.os.kill"):
            before = datetime.now(timezone.utc)
            result = check_pid(1234)
            after = datetime.now(timezone.utc)

        assert before <= result.timestamp <= after

    def test_alive_has_non_negative_latency(self) -> None:
        with patch("jules_daemon.monitor.process_state.os.kill"):
            result = check_pid(1234)

        assert result.latency_ms >= 0.0

    def test_current_process_is_alive(self) -> None:
        """Sanity check: our own PID should be alive."""
        result = check_pid(os.getpid())
        assert result.verdict == ProcessVerdict.ALIVE
        assert result.error is None


# ---------------------------------------------------------------------------
# check_pid: PermissionError (EPERM) -> process is ALIVE
# ---------------------------------------------------------------------------


class TestCheckPidPermissionError:
    def test_eperm_is_alive(self) -> None:
        """PermissionError means process exists but we lack permission."""
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            mock_kill.side_effect = PermissionError(
                errno.EPERM, "Operation not permitted"
            )
            result = check_pid(1234)

        assert result.verdict == ProcessVerdict.ALIVE
        assert result.error is None

    def test_eperm_records_pid(self) -> None:
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            mock_kill.side_effect = PermissionError(
                errno.EPERM, "Operation not permitted"
            )
            result = check_pid(9999)

        assert result.pid == 9999


# ---------------------------------------------------------------------------
# check_pid: ProcessLookupError (ESRCH) -> process is DEAD
# ---------------------------------------------------------------------------


class TestCheckPidProcessLookupError:
    def test_esrch_is_dead(self) -> None:
        """ProcessLookupError means no such process -> DEAD."""
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            mock_kill.side_effect = ProcessLookupError(
                errno.ESRCH, "No such process"
            )
            result = check_pid(1234)

        assert result.verdict == ProcessVerdict.DEAD
        assert result.error is not None
        assert "No such process" in result.error

    def test_esrch_records_pid(self) -> None:
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            mock_kill.side_effect = ProcessLookupError(
                errno.ESRCH, "No such process"
            )
            result = check_pid(8888)

        assert result.pid == 8888


# ---------------------------------------------------------------------------
# check_pid: OSError with ESRCH errno -> DEAD
# ---------------------------------------------------------------------------


class TestCheckPidOSErrorEsrch:
    def test_oserror_esrch_is_dead(self) -> None:
        """OSError with errno ESRCH should also be treated as DEAD."""
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            err = OSError(errno.ESRCH, "No such process")
            mock_kill.side_effect = err
            result = check_pid(1234)

        assert result.verdict == ProcessVerdict.DEAD

    def test_oserror_eperm_is_alive(self) -> None:
        """OSError with errno EPERM should be treated as ALIVE."""
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            err = OSError(errno.EPERM, "Operation not permitted")
            mock_kill.side_effect = err
            result = check_pid(1234)

        assert result.verdict == ProcessVerdict.ALIVE


# ---------------------------------------------------------------------------
# check_pid: unexpected OSError -> ERROR
# ---------------------------------------------------------------------------


class TestCheckPidUnexpectedError:
    def test_unexpected_oserror_is_error(self) -> None:
        """Unexpected OSError (e.g., EINVAL) -> ERROR verdict."""
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            err = OSError(errno.EINVAL, "Invalid argument")
            mock_kill.side_effect = err
            result = check_pid(1234)

        assert result.verdict == ProcessVerdict.ERROR
        assert result.error is not None
        assert "Invalid argument" in result.error

    def test_unexpected_exception_is_error(self) -> None:
        """Non-OSError exceptions -> ERROR verdict with description."""
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            mock_kill.side_effect = RuntimeError("something broke")
            result = check_pid(1234)

        assert result.verdict == ProcessVerdict.ERROR
        assert result.error is not None
        assert "something broke" in result.error


# ---------------------------------------------------------------------------
# check_pid: PID validation
# ---------------------------------------------------------------------------


class TestCheckPidValidation:
    def test_negative_pid_raises(self) -> None:
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            check_pid(-1)

    def test_zero_pid_raises(self) -> None:
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            check_pid(0)

    def test_positive_pid_does_not_raise(self) -> None:
        with patch("jules_daemon.monitor.process_state.os.kill"):
            result = check_pid(1)
        assert result.pid == 1

    def test_large_pid_is_valid(self) -> None:
        """Large PIDs should be accepted."""
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            mock_kill.side_effect = ProcessLookupError(
                errno.ESRCH, "No such process"
            )
            result = check_pid(4194304)
        assert result.pid == 4194304


# ---------------------------------------------------------------------------
# check_pids: batch processing
# ---------------------------------------------------------------------------


class TestCheckPids:
    def test_empty_list_returns_empty_mapping(self) -> None:
        results = check_pids([])
        assert results == {}

    def test_single_pid(self) -> None:
        with patch("jules_daemon.monitor.process_state.os.kill"):
            results = check_pids([1234])

        assert len(results) == 1
        assert 1234 in results
        assert results[1234].verdict == ProcessVerdict.ALIVE

    def test_multiple_pids_mixed_results(self) -> None:
        """Test batch with a mix of alive, dead, and error PIDs."""

        def side_effect(pid: int, sig: int) -> None:
            if pid == 100:
                return None  # alive
            if pid == 200:
                raise ProcessLookupError(errno.ESRCH, "No such process")
            if pid == 300:
                raise PermissionError(errno.EPERM, "Operation not permitted")
            raise OSError(errno.EINVAL, "Invalid argument")

        with patch(
            "jules_daemon.monitor.process_state.os.kill",
            side_effect=side_effect,
        ):
            results = check_pids([100, 200, 300, 400])

        assert results[100].verdict == ProcessVerdict.ALIVE
        assert results[200].verdict == ProcessVerdict.DEAD
        assert results[300].verdict == ProcessVerdict.ALIVE
        assert results[400].verdict == ProcessVerdict.ERROR

    def test_preserves_all_pids_in_output(self) -> None:
        """Every input PID must appear in the output mapping."""
        pids = [10, 20, 30]
        with patch("jules_daemon.monitor.process_state.os.kill"):
            results = check_pids(pids)

        assert set(results.keys()) == {10, 20, 30}

    def test_invalid_pid_in_list_raises(self) -> None:
        """Any invalid PID in the list should raise ValueError."""
        with pytest.raises(ValueError, match="PID must be a positive integer"):
            check_pids([1, -5, 10])

    def test_duplicate_pids_deduplicated(self) -> None:
        """Duplicate PIDs in input should produce one result each."""
        with patch("jules_daemon.monitor.process_state.os.kill"):
            results = check_pids([1234, 1234, 1234])

        assert len(results) == 1
        assert 1234 in results

    def test_returns_immutable_mapping(self) -> None:
        """Returned mapping should be a read-only view."""
        with patch("jules_daemon.monitor.process_state.os.kill"):
            results = check_pids([1234])

        # types.MappingProxyType raises TypeError on mutation
        with pytest.raises(TypeError):
            results[9999] = results[1234]  # type: ignore[index]

    def test_each_result_has_timestamp(self) -> None:
        with patch("jules_daemon.monitor.process_state.os.kill"):
            before = datetime.now(timezone.utc)
            results = check_pids([1, 2])
            after = datetime.now(timezone.utc)

        for result in results.values():
            assert before <= result.timestamp <= after


# ---------------------------------------------------------------------------
# Race condition handling
# ---------------------------------------------------------------------------


class TestRaceConditions:
    def test_process_dies_between_checks(self) -> None:
        """If a process dies between two check_pid calls, that is expected.

        This test verifies that check_pid does not cache or assume state
        persistence -- each call probes the OS fresh.
        """
        call_count = 0

        def side_effect(pid: int, sig: int) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None  # first call: alive
            raise ProcessLookupError(errno.ESRCH, "No such process")

        with patch(
            "jules_daemon.monitor.process_state.os.kill",
            side_effect=side_effect,
        ):
            result1 = check_pid(1234)
            result2 = check_pid(1234)

        assert result1.verdict == ProcessVerdict.ALIVE
        assert result2.verdict == ProcessVerdict.DEAD

    def test_no_global_state_between_calls(self) -> None:
        """check_pid must be stateless -- no caching."""
        with patch("jules_daemon.monitor.process_state.os.kill") as mock_kill:
            mock_kill.return_value = None
            r1 = check_pid(1)
            r2 = check_pid(2)

        assert r1.pid == 1
        assert r2.pid == 2
        assert mock_kill.call_count == 2


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------


class TestLatencyMeasurement:
    def test_latency_is_non_negative(self) -> None:
        with patch("jules_daemon.monitor.process_state.os.kill"):
            result = check_pid(1234)
        assert result.latency_ms >= 0.0

    def test_batch_latency_is_non_negative(self) -> None:
        with patch("jules_daemon.monitor.process_state.os.kill"):
            results = check_pids([1, 2, 3])
        for result in results.values():
            assert result.latency_ms >= 0.0
