"""Tests for the timestamped monitoring status model."""

from datetime import datetime, timezone

import pytest

from jules_daemon.wiki.monitor_status import (
    MonitorStatus,
    OutputPhase,
    ParsedState,
)


# -- OutputPhase --


class TestOutputPhase:
    def test_all_phases_exist(self) -> None:
        assert OutputPhase.CONNECTING.value == "connecting"
        assert OutputPhase.SETUP.value == "setup"
        assert OutputPhase.COLLECTING.value == "collecting"
        assert OutputPhase.RUNNING.value == "running"
        assert OutputPhase.TEARDOWN.value == "teardown"
        assert OutputPhase.COMPLETE.value == "complete"
        assert OutputPhase.ERROR.value == "error"
        assert OutputPhase.UNKNOWN.value == "unknown"

    def test_from_string(self) -> None:
        assert OutputPhase("running") == OutputPhase.RUNNING
        assert OutputPhase("error") == OutputPhase.ERROR


# -- ParsedState --


class TestParsedState:
    def test_defaults(self) -> None:
        state = ParsedState()
        assert state.phase == OutputPhase.UNKNOWN
        assert state.tests_discovered == 0
        assert state.tests_passed == 0
        assert state.tests_failed == 0
        assert state.tests_skipped == 0
        assert state.tests_total == 0
        assert state.current_test == ""
        assert state.error_message == ""

    def test_create_with_values(self) -> None:
        state = ParsedState(
            phase=OutputPhase.RUNNING,
            tests_discovered=20,
            tests_passed=5,
            tests_failed=1,
            tests_skipped=0,
            tests_total=20,
            current_test="test_login_flow",
            error_message="",
        )
        assert state.phase == OutputPhase.RUNNING
        assert state.tests_discovered == 20
        assert state.tests_passed == 5
        assert state.current_test == "test_login_flow"

    def test_frozen(self) -> None:
        state = ParsedState()
        with pytest.raises(AttributeError):
            state.phase = OutputPhase.RUNNING  # type: ignore[misc]

    def test_negative_counts_raise(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            ParsedState(tests_passed=-1)
        with pytest.raises(ValueError, match="must not be negative"):
            ParsedState(tests_failed=-1)
        with pytest.raises(ValueError, match="must not be negative"):
            ParsedState(tests_skipped=-1)
        with pytest.raises(ValueError, match="must not be negative"):
            ParsedState(tests_total=-1)
        with pytest.raises(ValueError, match="must not be negative"):
            ParsedState(tests_discovered=-1)

    def test_error_state_with_message(self) -> None:
        state = ParsedState(
            phase=OutputPhase.ERROR,
            error_message="Connection refused",
        )
        assert state.phase == OutputPhase.ERROR
        assert state.error_message == "Connection refused"


# -- MonitorStatus --


class TestMonitorStatus:
    def _make_timestamp(self) -> datetime:
        return datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)

    def test_create_minimal(self) -> None:
        ts = self._make_timestamp()
        status = MonitorStatus(session_id="abc-123", timestamp=ts)
        assert status.session_id == "abc-123"
        assert status.timestamp == ts
        assert status.raw_output_chunk == ""
        assert status.parsed_state == ParsedState()
        assert status.exit_status is None
        assert status.sequence_number == 0

    def test_empty_session_id_raises(self) -> None:
        ts = self._make_timestamp()
        with pytest.raises(ValueError, match="session_id must not be empty"):
            MonitorStatus(session_id="", timestamp=ts)

    def test_negative_sequence_raises(self) -> None:
        ts = self._make_timestamp()
        with pytest.raises(ValueError, match="sequence_number must not be negative"):
            MonitorStatus(session_id="abc", timestamp=ts, sequence_number=-1)

    def test_create_full(self) -> None:
        ts = self._make_timestamp()
        parsed = ParsedState(
            phase=OutputPhase.RUNNING,
            tests_passed=3,
            tests_total=10,
            current_test="test_checkout",
        )
        status = MonitorStatus(
            session_id="run-456",
            timestamp=ts,
            raw_output_chunk="PASSED test_checkout\n",
            parsed_state=parsed,
            exit_status=None,
            sequence_number=7,
        )
        assert status.session_id == "run-456"
        assert status.raw_output_chunk == "PASSED test_checkout\n"
        assert status.parsed_state.tests_passed == 3
        assert status.exit_status is None
        assert status.sequence_number == 7

    def test_frozen(self) -> None:
        ts = self._make_timestamp()
        status = MonitorStatus(session_id="abc", timestamp=ts)
        with pytest.raises(AttributeError):
            status.session_id = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            status.exit_status = 0  # type: ignore[misc]

    def test_is_terminal_when_exit_status_set(self) -> None:
        ts = self._make_timestamp()
        status = MonitorStatus(session_id="abc", timestamp=ts, exit_status=0)
        assert status.is_terminal is True

    def test_is_terminal_when_exit_status_none(self) -> None:
        ts = self._make_timestamp()
        status = MonitorStatus(session_id="abc", timestamp=ts, exit_status=None)
        assert status.is_terminal is False

    def test_is_terminal_nonzero_exit(self) -> None:
        ts = self._make_timestamp()
        status = MonitorStatus(session_id="abc", timestamp=ts, exit_status=1)
        assert status.is_terminal is True

    def test_is_success_zero_exit(self) -> None:
        ts = self._make_timestamp()
        status = MonitorStatus(session_id="abc", timestamp=ts, exit_status=0)
        assert status.is_success is True

    def test_is_success_nonzero_exit(self) -> None:
        ts = self._make_timestamp()
        status = MonitorStatus(session_id="abc", timestamp=ts, exit_status=1)
        assert status.is_success is False

    def test_is_success_none_exit(self) -> None:
        ts = self._make_timestamp()
        status = MonitorStatus(session_id="abc", timestamp=ts, exit_status=None)
        assert status.is_success is False


# -- Immutable update method --


class TestMonitorStatusWithUpdate:
    def _make_base(self) -> MonitorStatus:
        ts = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        return MonitorStatus(
            session_id="run-789",
            timestamp=ts,
            raw_output_chunk="initial output\n",
            parsed_state=ParsedState(
                phase=OutputPhase.CONNECTING,
            ),
            sequence_number=0,
        )

    def test_with_update_returns_new_instance(self) -> None:
        original = self._make_base()
        new_ts = datetime(2026, 4, 9, 12, 0, 10, tzinfo=timezone.utc)
        updated = original.with_update(
            timestamp=new_ts,
            raw_output_chunk="new output\n",
        )
        assert updated is not original
        assert type(updated) is MonitorStatus

    def test_with_update_preserves_unchanged_fields(self) -> None:
        original = self._make_base()
        new_ts = datetime(2026, 4, 9, 12, 0, 10, tzinfo=timezone.utc)
        updated = original.with_update(
            timestamp=new_ts,
            raw_output_chunk="new output\n",
        )
        # Unchanged fields
        assert updated.session_id == original.session_id
        assert updated.exit_status is None
        assert updated.parsed_state == original.parsed_state
        # Changed fields
        assert updated.timestamp == new_ts
        assert updated.raw_output_chunk == "new output\n"

    def test_with_update_does_not_mutate_original(self) -> None:
        original = self._make_base()
        original_ts = original.timestamp
        original_output = original.raw_output_chunk
        new_ts = datetime(2026, 4, 9, 12, 0, 10, tzinfo=timezone.utc)
        _ = original.with_update(
            timestamp=new_ts,
            raw_output_chunk="changed",
        )
        # Original is untouched
        assert original.timestamp == original_ts
        assert original.raw_output_chunk == original_output

    def test_with_update_all_fields(self) -> None:
        original = self._make_base()
        new_ts = datetime(2026, 4, 9, 12, 5, 0, tzinfo=timezone.utc)
        new_parsed = ParsedState(
            phase=OutputPhase.COMPLETE,
            tests_passed=10,
            tests_total=10,
        )
        updated = original.with_update(
            timestamp=new_ts,
            raw_output_chunk="all passed\n",
            parsed_state=new_parsed,
            exit_status=0,
            sequence_number=42,
        )
        assert updated.timestamp == new_ts
        assert updated.raw_output_chunk == "all passed\n"
        assert updated.parsed_state == new_parsed
        assert updated.exit_status == 0
        assert updated.sequence_number == 42

    def test_with_update_increments_sequence(self) -> None:
        """Convenience: auto-increment sequence when not explicitly provided."""
        original = self._make_base()
        assert original.sequence_number == 0
        new_ts = datetime(2026, 4, 9, 12, 0, 10, tzinfo=timezone.utc)
        updated = original.with_output(
            timestamp=new_ts,
            raw_output_chunk="more output\n",
        )
        assert updated.sequence_number == 1

    def test_with_output_updates_timestamp_and_chunk(self) -> None:
        original = self._make_base()
        new_ts = datetime(2026, 4, 9, 12, 0, 5, tzinfo=timezone.utc)
        updated = original.with_output(
            timestamp=new_ts,
            raw_output_chunk="PASSED test_foo\n",
        )
        assert updated.timestamp == new_ts
        assert updated.raw_output_chunk == "PASSED test_foo\n"
        assert updated.session_id == original.session_id

    def test_with_parsed_state(self) -> None:
        original = self._make_base()
        new_ts = datetime(2026, 4, 9, 12, 0, 15, tzinfo=timezone.utc)
        new_state = ParsedState(
            phase=OutputPhase.RUNNING,
            tests_passed=5,
            tests_failed=0,
            tests_total=20,
            current_test="test_payment",
        )
        updated = original.with_parsed_state(
            timestamp=new_ts,
            parsed_state=new_state,
        )
        assert updated.parsed_state == new_state
        assert updated.timestamp == new_ts
        assert updated.sequence_number == original.sequence_number + 1

    def test_with_exit_status(self) -> None:
        original = self._make_base()
        new_ts = datetime(2026, 4, 9, 12, 10, 0, tzinfo=timezone.utc)
        final_parsed = ParsedState(
            phase=OutputPhase.COMPLETE,
            tests_passed=10,
            tests_total=10,
        )
        completed = original.with_exit(
            timestamp=new_ts,
            exit_status=0,
            parsed_state=final_parsed,
            raw_output_chunk="10 passed in 5.2s\n",
        )
        assert completed.exit_status == 0
        assert completed.is_terminal is True
        assert completed.is_success is True
        assert completed.parsed_state.phase == OutputPhase.COMPLETE
        assert completed.raw_output_chunk == "10 passed in 5.2s\n"
        assert completed.sequence_number == original.sequence_number + 1

    def test_with_update_explicitly_clears_exit_status(self) -> None:
        """Sentinel: passing exit_status=None clears a previously-set value."""
        ts = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        terminal = MonitorStatus(
            session_id="sentinel-test",
            timestamp=ts,
            exit_status=1,
        )
        assert terminal.is_terminal is True
        cleared = terminal.with_update(exit_status=None)
        assert cleared.exit_status is None
        assert cleared.is_terminal is False

    def test_with_update_omitting_exit_status_preserves_it(self) -> None:
        """Sentinel: omitting exit_status leaves the existing value unchanged."""
        ts = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        terminal = MonitorStatus(
            session_id="sentinel-test",
            timestamp=ts,
            exit_status=42,
        )
        new_ts = datetime(2026, 4, 9, 12, 0, 5, tzinfo=timezone.utc)
        updated = terminal.with_update(timestamp=new_ts)
        assert updated.exit_status == 42
        assert updated.is_terminal is True

    def test_chained_updates(self) -> None:
        """Multiple immutable updates produce correct chain."""
        ts1 = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 4, 9, 12, 0, 5, tzinfo=timezone.utc)
        ts3 = datetime(2026, 4, 9, 12, 0, 10, tzinfo=timezone.utc)

        s0 = MonitorStatus(session_id="chain", timestamp=ts1)
        s1 = s0.with_output(
            timestamp=ts2,
            raw_output_chunk="collecting...\n",
        )
        s2 = s1.with_parsed_state(
            timestamp=ts3,
            parsed_state=ParsedState(
                phase=OutputPhase.COLLECTING,
                tests_discovered=15,
            ),
        )

        # Each instance is distinct
        assert s0.sequence_number == 0
        assert s1.sequence_number == 1
        assert s2.sequence_number == 2

        # Original untouched
        assert s0.raw_output_chunk == ""
        assert s0.parsed_state.phase == OutputPhase.UNKNOWN

        # Latest has all updates
        assert s2.raw_output_chunk == s1.raw_output_chunk  # preserved from s1
        assert s2.parsed_state.phase == OutputPhase.COLLECTING
        assert s2.parsed_state.tests_discovered == 15
