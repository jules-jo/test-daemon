"""Tests for structured logging of stale sessions during startup scan.

Verifies that the stale session logger:
- Emits structured log records via Python's logging module for each
  stale session detected during the startup scan
- Includes all required schema fields: session_id, host,
  last_activity_timestamp, staleness_reason
- Produces immutable StaleSessionLogEntry records
- Handles missing host gracefully (unknown host)
- Handles batch logging of multiple stale sessions
- Does not emit logs for alive/skipped sessions
- Produces JSON-serializable log extra data
- Returns tuple of log entries for callers to inspect
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.monitor.session_liveness import SessionHealth
from jules_daemon.startup.stale_session_logger import (
    StaleSessionLogEntry,
    build_stale_log_entry,
    log_stale_sessions_from_verdicts,
)
from jules_daemon.startup.scan_probe_mark import SessionVerdict
from jules_daemon.wiki.models import RunStatus
from jules_daemon.wiki.session_scanner import SessionEntry
from jules_daemon.wiki.stale_session_marker import MarkOutcome, MarkResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_TIMESTAMP = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


def _make_session_entry(
    *,
    run_id: str = "run-abc-123",
    host: str | None = "prod.example.com",
    user: str | None = "ci",
    port: int | None = 22,
    updated_at: datetime | None = None,
    daemon_pid: int | None = 1234,
) -> SessionEntry:
    """Build a SessionEntry for testing."""
    return SessionEntry(
        source_path=Path("/tmp/wiki/pages/daemon/current-run.md"),
        run_id=run_id,
        status=RunStatus.RUNNING,
        daemon_pid=daemon_pid,
        remote_pid=5678,
        ssh_host=host,
        ssh_user=user,
        ssh_port=port,
        started_at=_SAMPLE_TIMESTAMP,
        updated_at=updated_at or _SAMPLE_TIMESTAMP,
    )


def _make_verdict(
    *,
    entry: SessionEntry | None = None,
    health: SessionHealth = SessionHealth.PROCESS_DEAD,
    alive: bool = False,
    process_alive: bool | None = False,
    endpoint_reachable: bool | None = None,
) -> SessionVerdict:
    """Build a SessionVerdict for testing."""
    if entry is None:
        entry = _make_session_entry()
    return SessionVerdict(
        session_entry=entry,
        process_alive=process_alive,
        endpoint_reachable=endpoint_reachable,
        health=health,
        alive=alive,
    )


def _make_mark_result(
    *,
    session_id: str = "run-abc-123",
    outcome: MarkOutcome = MarkOutcome.MARKED_STALE,
    reason: str | None = "Daemon process dead -- PID no longer exists",
) -> MarkResult:
    """Build a MarkResult for testing."""
    return MarkResult(
        session_id=session_id,
        outcome=outcome,
        source_path=Path("/tmp/wiki/pages/daemon/current-run.md"),
        stale_path=Path("/tmp/wiki/pages/daemon/current-run.stale.20260409-120000-000000.md"),
        reason=reason,
        detected_at=_SAMPLE_TIMESTAMP,
        error=None,
    )


# ---------------------------------------------------------------------------
# StaleSessionLogEntry model tests
# ---------------------------------------------------------------------------


class TestStaleSessionLogEntry:
    """Verify the immutable log entry model."""

    def test_create_with_all_fields(self) -> None:
        entry = StaleSessionLogEntry(
            session_id="run-abc-123",
            host="prod.example.com",
            last_activity_timestamp=_SAMPLE_TIMESTAMP,
            staleness_reason="Daemon process dead -- PID no longer exists",
        )
        assert entry.session_id == "run-abc-123"
        assert entry.host == "prod.example.com"
        assert entry.last_activity_timestamp == _SAMPLE_TIMESTAMP
        assert entry.staleness_reason == "Daemon process dead -- PID no longer exists"

    def test_frozen(self) -> None:
        entry = StaleSessionLogEntry(
            session_id="run-abc",
            host="host",
            last_activity_timestamp=_SAMPLE_TIMESTAMP,
            staleness_reason="dead",
        )
        with pytest.raises(AttributeError):
            entry.session_id = "changed"  # type: ignore[misc]

    def test_to_log_dict_returns_serializable_dict(self) -> None:
        entry = StaleSessionLogEntry(
            session_id="run-abc-123",
            host="prod.example.com",
            last_activity_timestamp=_SAMPLE_TIMESTAMP,
            staleness_reason="Daemon process dead -- PID no longer exists",
        )
        log_dict = entry.to_log_dict()
        assert isinstance(log_dict, dict)
        assert log_dict["session_id"] == "run-abc-123"
        assert log_dict["host"] == "prod.example.com"
        assert log_dict["last_activity_timestamp"] == "2026-04-09T12:00:00+00:00"
        assert log_dict["staleness_reason"] == "Daemon process dead -- PID no longer exists"

    def test_to_log_dict_is_json_serializable(self) -> None:
        entry = StaleSessionLogEntry(
            session_id="run-abc-123",
            host="prod.example.com",
            last_activity_timestamp=_SAMPLE_TIMESTAMP,
            staleness_reason="Connection lost",
        )
        serialized = json.dumps(entry.to_log_dict())
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed["session_id"] == "run-abc-123"

    def test_unknown_host_when_none(self) -> None:
        entry = StaleSessionLogEntry(
            session_id="run-xyz",
            host="unknown",
            last_activity_timestamp=_SAMPLE_TIMESTAMP,
            staleness_reason="Connection lost",
        )
        assert entry.host == "unknown"


# ---------------------------------------------------------------------------
# build_stale_log_entry tests
# ---------------------------------------------------------------------------


class TestBuildStaleLogEntry:
    """Verify log entry construction from verdict + mark result."""

    def test_builds_from_dead_process_verdict(self) -> None:
        entry = _make_session_entry(
            run_id="run-dead-001",
            host="prod.example.com",
            updated_at=_SAMPLE_TIMESTAMP,
        )
        verdict = _make_verdict(
            entry=entry,
            health=SessionHealth.PROCESS_DEAD,
            alive=False,
        )
        mark = _make_mark_result(
            session_id="run-dead-001",
            reason="Daemon process dead -- PID no longer exists",
        )

        log_entry = build_stale_log_entry(verdict, mark)

        assert log_entry.session_id == "run-dead-001"
        assert log_entry.host == "prod.example.com"
        assert log_entry.last_activity_timestamp == _SAMPLE_TIMESTAMP
        assert log_entry.staleness_reason == "Daemon process dead -- PID no longer exists"

    def test_builds_from_connection_lost_verdict(self) -> None:
        entry = _make_session_entry(
            run_id="run-conn-002",
            host="staging.example.com",
            updated_at=_SAMPLE_TIMESTAMP,
        )
        verdict = _make_verdict(
            entry=entry,
            health=SessionHealth.CONNECTION_LOST,
            alive=False,
            process_alive=True,
            endpoint_reachable=False,
        )
        mark = _make_mark_result(
            session_id="run-conn-002",
            reason="SSH connection lost -- remote host unreachable",
        )

        log_entry = build_stale_log_entry(verdict, mark)

        assert log_entry.session_id == "run-conn-002"
        assert log_entry.host == "staging.example.com"
        assert log_entry.staleness_reason == "SSH connection lost -- remote host unreachable"

    def test_missing_host_yields_unknown(self) -> None:
        entry = _make_session_entry(
            run_id="run-nohost",
            host=None,
        )
        verdict = _make_verdict(
            entry=entry,
            health=SessionHealth.UNKNOWN,
            alive=False,
        )
        mark = _make_mark_result(
            session_id="run-nohost",
            reason="Health undetermined",
        )

        log_entry = build_stale_log_entry(verdict, mark)

        assert log_entry.host == "unknown"

    def test_missing_reason_yields_fallback(self) -> None:
        entry = _make_session_entry(run_id="run-noreason")
        verdict = _make_verdict(
            entry=entry,
            health=SessionHealth.PROCESS_DEAD,
            alive=False,
        )
        mark = _make_mark_result(
            session_id="run-noreason",
            reason=None,
        )

        log_entry = build_stale_log_entry(verdict, mark)

        assert log_entry.staleness_reason != ""
        assert "process_dead" in log_entry.staleness_reason.lower()


# ---------------------------------------------------------------------------
# log_stale_sessions_from_verdicts tests
# ---------------------------------------------------------------------------


class TestLogStaleSessionsFromVerdicts:
    """Verify the batch logging function."""

    def test_logs_stale_sessions(self, caplog: pytest.LogCaptureFixture) -> None:
        """Stale sessions produce structured log records."""
        entry = _make_session_entry(run_id="run-stale-001")
        verdict = _make_verdict(entry=entry, health=SessionHealth.PROCESS_DEAD)
        mark = _make_mark_result(
            session_id="run-stale-001",
            outcome=MarkOutcome.MARKED_STALE,
            reason="Daemon process dead -- PID no longer exists",
        )

        with caplog.at_level(logging.WARNING, logger="jules_daemon.startup.stale_session_logger"):
            results = log_stale_sessions_from_verdicts(
                verdicts=(verdict,),
                mark_results=(mark,),
            )

        assert len(results) == 1
        assert results[0].session_id == "run-stale-001"

        # Verify log message was emitted
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert "run-stale-001" in record.message
        assert "prod.example.com" in record.message

    def test_skips_alive_sessions(self, caplog: pytest.LogCaptureFixture) -> None:
        """Alive sessions (SKIPPED_ALIVE) do not produce log entries."""
        entry = _make_session_entry(run_id="run-alive-001")
        verdict = _make_verdict(
            entry=entry,
            health=SessionHealth.HEALTHY,
            alive=True,
            process_alive=True,
            endpoint_reachable=True,
        )
        mark = _make_mark_result(
            session_id="run-alive-001",
            outcome=MarkOutcome.SKIPPED_ALIVE,
            reason=None,
        )

        with caplog.at_level(logging.WARNING, logger="jules_daemon.startup.stale_session_logger"):
            results = log_stale_sessions_from_verdicts(
                verdicts=(verdict,),
                mark_results=(mark,),
            )

        assert len(results) == 0
        assert len(caplog.records) == 0

    def test_multiple_stale_sessions(self, caplog: pytest.LogCaptureFixture) -> None:
        """Multiple stale sessions each produce their own log entry."""
        entry1 = _make_session_entry(
            run_id="run-001",
            host="host-a.example.com",
        )
        entry2 = _make_session_entry(
            run_id="run-002",
            host="host-b.example.com",
        )

        verdict1 = _make_verdict(entry=entry1, health=SessionHealth.PROCESS_DEAD)
        verdict2 = _make_verdict(
            entry=entry2,
            health=SessionHealth.CONNECTION_LOST,
            process_alive=True,
            endpoint_reachable=False,
        )

        mark1 = _make_mark_result(
            session_id="run-001",
            outcome=MarkOutcome.MARKED_STALE,
            reason="Daemon process dead",
        )
        mark2 = _make_mark_result(
            session_id="run-002",
            outcome=MarkOutcome.MARKED_STALE,
            reason="SSH connection lost",
        )

        with caplog.at_level(logging.WARNING, logger="jules_daemon.startup.stale_session_logger"):
            results = log_stale_sessions_from_verdicts(
                verdicts=(verdict1, verdict2),
                mark_results=(mark1, mark2),
            )

        assert len(results) == 2
        assert results[0].session_id == "run-001"
        assert results[1].session_id == "run-002"
        assert len(caplog.records) == 2

    def test_mixed_stale_and_alive(self, caplog: pytest.LogCaptureFixture) -> None:
        """Only stale sessions are logged when mixed with alive ones."""
        stale_entry = _make_session_entry(run_id="run-stale")
        alive_entry = _make_session_entry(run_id="run-alive")

        stale_verdict = _make_verdict(
            entry=stale_entry,
            health=SessionHealth.PROCESS_DEAD,
        )
        alive_verdict = _make_verdict(
            entry=alive_entry,
            health=SessionHealth.HEALTHY,
            alive=True,
            process_alive=True,
            endpoint_reachable=True,
        )

        stale_mark = _make_mark_result(
            session_id="run-stale",
            outcome=MarkOutcome.MARKED_STALE,
            reason="Dead process",
        )
        alive_mark = _make_mark_result(
            session_id="run-alive",
            outcome=MarkOutcome.SKIPPED_ALIVE,
            reason=None,
        )

        with caplog.at_level(logging.WARNING, logger="jules_daemon.startup.stale_session_logger"):
            results = log_stale_sessions_from_verdicts(
                verdicts=(stale_verdict, alive_verdict),
                mark_results=(stale_mark, alive_mark),
            )

        assert len(results) == 1
        assert results[0].session_id == "run-stale"
        assert len(caplog.records) == 1

    def test_empty_verdicts(self) -> None:
        """Empty input produces empty output."""
        results = log_stale_sessions_from_verdicts(
            verdicts=(),
            mark_results=(),
        )
        assert results == ()

    def test_log_record_contains_structured_extra(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log records carry structured data as extra attributes."""
        entry = _make_session_entry(
            run_id="run-extra-001",
            host="extra.example.com",
            updated_at=_SAMPLE_TIMESTAMP,
        )
        verdict = _make_verdict(entry=entry, health=SessionHealth.PROCESS_DEAD)
        mark = _make_mark_result(
            session_id="run-extra-001",
            reason="Daemon process dead",
        )

        with caplog.at_level(logging.WARNING, logger="jules_daemon.startup.stale_session_logger"):
            log_stale_sessions_from_verdicts(
                verdicts=(verdict,),
                mark_results=(mark,),
            )

        assert len(caplog.records) == 1
        record = caplog.records[0]
        # Verify structured data is available as log record attributes
        assert hasattr(record, "stale_session")
        extra = record.stale_session  # type: ignore[attr-defined]
        assert extra["session_id"] == "run-extra-001"
        assert extra["host"] == "extra.example.com"
        assert extra["last_activity_timestamp"] == "2026-04-09T12:00:00+00:00"
        assert extra["staleness_reason"] == "Daemon process dead"

    def test_mark_result_error_still_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Even if mark outcome is ERROR, the stale session is still logged."""
        entry = _make_session_entry(run_id="run-error-001")
        verdict = _make_verdict(entry=entry, health=SessionHealth.PROCESS_DEAD)
        mark = MarkResult(
            session_id="run-error-001",
            outcome=MarkOutcome.ERROR,
            source_path=Path("/tmp/test.md"),
            stale_path=None,
            reason="Daemon process dead",
            detected_at=None,
            error="Write failed: permission denied",
        )

        with caplog.at_level(logging.WARNING, logger="jules_daemon.startup.stale_session_logger"):
            results = log_stale_sessions_from_verdicts(
                verdicts=(verdict,),
                mark_results=(mark,),
            )

        assert len(results) == 1
        assert len(caplog.records) == 1

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched verdicts and mark_results lengths raise ValueError."""
        verdict = _make_verdict()
        with pytest.raises(ValueError, match="must have the same length"):
            log_stale_sessions_from_verdicts(
                verdicts=(verdict,),
                mark_results=(),
            )
