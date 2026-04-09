"""Tests for stale SSH session detector.

Verifies that the stale session detector:
- Monitors active SSH sessions for staleness using heartbeat/keepalive
  timeouts and transport-layer status checks
- Tracks per-session heartbeat timestamps with configurable timeouts
- Uses transport-layer liveness probes to validate session health
- Marks sessions as stale with timestamp and reason when either:
  (a) heartbeat timeout exceeded, or
  (b) transport check reports DISCONNECTED/DEAD
- Produces immutable StalenessDetection result records
- Supports configurable heartbeat timeout, keepalive interval, and
  max missed heartbeats
- Handles edge cases: no sessions, already-stale sessions, mixed
  healthy/stale batches
- Integrates with ProbeExecutor protocol for transport checks
- Never mutates existing data -- all state transitions produce new
  frozen instances
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jules_daemon.monitor.stale_session_detector import (
    DetectorConfig,
    HeartbeatRecord,
    HeartbeatTracker,
    StalenessDetection,
    StalenessReason,
    detect_batch_staleness,
    detect_session_staleness,
)
from jules_daemon.ssh.liveness import (
    ConnectionHealth,
    ProbeResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


def _make_probe_result(
    *,
    health: ConnectionHealth = ConnectionHealth.CONNECTED,
    success: bool = True,
    latency_ms: float = 5.0,
    exit_code: int | None = 0,
    error: str | None = None,
    output: str = "__jules_probe_ok__",
) -> ProbeResult:
    """Create a minimal ProbeResult for testing."""
    return ProbeResult(
        success=success,
        health=health,
        latency_ms=latency_ms,
        output=output,
        exit_code=exit_code,
        error=error,
        probe_command="echo __jules_probe_ok__",
        timestamp=_NOW,
    )


# ---------------------------------------------------------------------------
# DetectorConfig tests
# ---------------------------------------------------------------------------


class TestDetectorConfig:
    """Verify the immutable configuration model."""

    def test_default_values(self) -> None:
        config = DetectorConfig()
        assert config.heartbeat_timeout_seconds == 30.0
        assert config.keepalive_interval_seconds == 10.0
        assert config.max_missed_heartbeats == 3
        assert config.transport_check_timeout_seconds == 5.0

    def test_custom_values(self) -> None:
        config = DetectorConfig(
            heartbeat_timeout_seconds=60.0,
            keepalive_interval_seconds=15.0,
            max_missed_heartbeats=5,
            transport_check_timeout_seconds=10.0,
        )
        assert config.heartbeat_timeout_seconds == 60.0
        assert config.keepalive_interval_seconds == 15.0
        assert config.max_missed_heartbeats == 5
        assert config.transport_check_timeout_seconds == 10.0

    def test_frozen(self) -> None:
        config = DetectorConfig()
        with pytest.raises(AttributeError):
            config.heartbeat_timeout_seconds = 99.0  # type: ignore[misc]

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DetectorConfig(heartbeat_timeout_seconds=-1.0)

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DetectorConfig(heartbeat_timeout_seconds=0.0)

    def test_negative_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DetectorConfig(keepalive_interval_seconds=-1.0)

    def test_negative_max_missed_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            DetectorConfig(max_missed_heartbeats=-1)

    def test_negative_transport_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DetectorConfig(transport_check_timeout_seconds=-1.0)


# ---------------------------------------------------------------------------
# HeartbeatRecord tests
# ---------------------------------------------------------------------------


class TestHeartbeatRecord:
    """Verify the immutable heartbeat tracking record."""

    def test_create(self) -> None:
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        assert record.session_id == "run-abc"
        assert record.last_heartbeat_at == _NOW
        assert record.consecutive_failures == 0
        assert record.last_transport_health is None

    def test_frozen(self) -> None:
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        with pytest.raises(AttributeError):
            record.session_id = "changed"  # type: ignore[misc]

    def test_with_heartbeat_returns_new_record(self) -> None:
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
            consecutive_failures=3,
        )
        new_time = _NOW + timedelta(seconds=10)
        updated = record.with_heartbeat(new_time)

        assert updated is not record
        assert updated.session_id == "run-abc"
        assert updated.last_heartbeat_at == new_time
        assert updated.consecutive_failures == 0  # Reset on success
        assert updated.last_transport_health == ConnectionHealth.CONNECTED

    def test_with_failure_increments_counter(self) -> None:
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
            consecutive_failures=2,
        )
        failed = record.with_failure(
            health=ConnectionHealth.DISCONNECTED,
            checked_at=_NOW + timedelta(seconds=5),
        )

        assert failed is not record
        assert failed.consecutive_failures == 3
        assert failed.last_transport_health == ConnectionHealth.DISCONNECTED
        # last_heartbeat_at should NOT advance on failure
        assert failed.last_heartbeat_at == _NOW

    def test_with_failure_records_degraded(self) -> None:
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        failed = record.with_failure(
            health=ConnectionHealth.DEGRADED,
            checked_at=_NOW + timedelta(seconds=1),
        )
        assert failed.consecutive_failures == 1
        assert failed.last_transport_health == ConnectionHealth.DEGRADED

    def test_heartbeat_age_seconds(self) -> None:
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        now = _NOW + timedelta(seconds=25)
        assert record.heartbeat_age_seconds(now) == 25.0

    def test_heartbeat_age_negative_clamps_to_zero(self) -> None:
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        past = _NOW - timedelta(seconds=5)
        assert record.heartbeat_age_seconds(past) == 0.0

    def test_empty_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            HeartbeatRecord(session_id="", last_heartbeat_at=_NOW)

    def test_whitespace_only_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            HeartbeatRecord(session_id="   ", last_heartbeat_at=_NOW)


# ---------------------------------------------------------------------------
# HeartbeatTracker tests
# ---------------------------------------------------------------------------


class TestHeartbeatTracker:
    """Verify the heartbeat tracking state container."""

    def test_empty_tracker(self) -> None:
        tracker = HeartbeatTracker()
        assert tracker.session_count == 0
        assert tracker.get("nonexistent") is None

    def test_register_and_get(self) -> None:
        tracker = HeartbeatTracker()
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        updated = tracker.register(record)
        assert updated.session_count == 1
        assert updated.get("run-abc") is not None
        assert updated.get("run-abc").session_id == "run-abc"

    def test_register_does_not_mutate_original(self) -> None:
        tracker = HeartbeatTracker()
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        updated = tracker.register(record)
        assert tracker.session_count == 0
        assert updated.session_count == 1

    def test_update_replaces_record(self) -> None:
        record1 = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        record2 = record1.with_heartbeat(_NOW + timedelta(seconds=10))

        tracker = HeartbeatTracker().register(record1)
        updated = tracker.update(record2)

        assert updated.get("run-abc").last_heartbeat_at == _NOW + timedelta(
            seconds=10
        )

    def test_update_unknown_session_raises(self) -> None:
        tracker = HeartbeatTracker()
        record = HeartbeatRecord(
            session_id="unknown",
            last_heartbeat_at=_NOW,
        )
        with pytest.raises(KeyError, match="unknown"):
            tracker.update(record)

    def test_remove(self) -> None:
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        tracker = HeartbeatTracker().register(record)
        removed = tracker.remove("run-abc")
        assert removed.session_count == 0

    def test_remove_unknown_is_noop(self) -> None:
        tracker = HeartbeatTracker()
        result = tracker.remove("nonexistent")
        assert result.session_count == 0

    def test_all_records(self) -> None:
        r1 = HeartbeatRecord(session_id="run-1", last_heartbeat_at=_NOW)
        r2 = HeartbeatRecord(session_id="run-2", last_heartbeat_at=_NOW)
        tracker = HeartbeatTracker().register(r1).register(r2)

        records = tracker.all_records()
        assert len(records) == 2
        ids = {r.session_id for r in records}
        assert ids == {"run-1", "run-2"}


# ---------------------------------------------------------------------------
# StalenessDetection model tests
# ---------------------------------------------------------------------------


class TestStalenessDetection:
    """Verify the immutable detection result model."""

    def test_create_stale(self) -> None:
        detection = StalenessDetection(
            session_id="run-abc",
            is_stale=True,
            reason=StalenessReason.HEARTBEAT_TIMEOUT,
            reason_detail="No heartbeat for 35.0 seconds (threshold: 30.0s)",
            detected_at=_NOW,
            heartbeat_age_seconds=35.0,
            transport_health=ConnectionHealth.DISCONNECTED,
            consecutive_failures=3,
        )
        assert detection.is_stale is True
        assert detection.reason == StalenessReason.HEARTBEAT_TIMEOUT
        assert detection.consecutive_failures == 3

    def test_create_healthy(self) -> None:
        detection = StalenessDetection(
            session_id="run-abc",
            is_stale=False,
            reason=None,
            reason_detail=None,
            detected_at=_NOW,
            heartbeat_age_seconds=5.0,
            transport_health=ConnectionHealth.CONNECTED,
            consecutive_failures=0,
        )
        assert detection.is_stale is False
        assert detection.reason is None

    def test_frozen(self) -> None:
        detection = StalenessDetection(
            session_id="run-abc",
            is_stale=True,
            reason=StalenessReason.HEARTBEAT_TIMEOUT,
            reason_detail="timeout",
            detected_at=_NOW,
            heartbeat_age_seconds=35.0,
            transport_health=None,
            consecutive_failures=0,
        )
        with pytest.raises(AttributeError):
            detection.is_stale = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StalenessReason enum tests
# ---------------------------------------------------------------------------


class TestStalenessReason:
    """Verify staleness reason classification."""

    def test_heartbeat_timeout(self) -> None:
        assert StalenessReason.HEARTBEAT_TIMEOUT.value == "heartbeat_timeout"

    def test_transport_disconnected(self) -> None:
        assert StalenessReason.TRANSPORT_DISCONNECTED.value == "transport_disconnected"

    def test_max_failures_exceeded(self) -> None:
        assert StalenessReason.MAX_FAILURES_EXCEEDED.value == "max_failures_exceeded"


# ---------------------------------------------------------------------------
# detect_session_staleness -- single session
# ---------------------------------------------------------------------------


class TestDetectSessionStaleness:
    """Verify single-session staleness detection logic."""

    def test_healthy_session_not_stale(self) -> None:
        """A session with a recent heartbeat and healthy transport is not stale."""
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        probe_result = _make_probe_result(health=ConnectionHealth.CONNECTED)
        config = DetectorConfig(heartbeat_timeout_seconds=30.0)

        detection = detect_session_staleness(
            record=record,
            probe_result=probe_result,
            config=config,
            now=_NOW + timedelta(seconds=5),
        )

        assert detection.is_stale is False
        assert detection.reason is None
        assert detection.heartbeat_age_seconds == 5.0
        assert detection.transport_health == ConnectionHealth.CONNECTED

    def test_heartbeat_timeout_triggers_stale(self) -> None:
        """A session that exceeds heartbeat timeout is stale.

        Uses a DEGRADED probe (not DISCONNECTED) so that the transport-
        disconnect check does not fire first. The heartbeat timeout
        check fires because 35s > 30s threshold.
        """
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        probe_result = _make_probe_result(
            health=ConnectionHealth.DEGRADED,
            success=False,
            exit_code=1,
            error="Non-zero exit code",
        )
        config = DetectorConfig(heartbeat_timeout_seconds=30.0)

        detection = detect_session_staleness(
            record=record,
            probe_result=probe_result,
            config=config,
            now=_NOW + timedelta(seconds=35),
        )

        assert detection.is_stale is True
        assert detection.reason == StalenessReason.HEARTBEAT_TIMEOUT
        assert detection.heartbeat_age_seconds == 35.0
        assert "35.0" in detection.reason_detail
        assert "30.0" in detection.reason_detail

    def test_transport_disconnected_triggers_stale(self) -> None:
        """A session with DISCONNECTED transport is stale regardless of heartbeat age."""
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        probe_result = _make_probe_result(
            health=ConnectionHealth.DISCONNECTED,
            success=False,
            exit_code=None,
            error="Connection reset",
        )
        config = DetectorConfig(heartbeat_timeout_seconds=30.0)

        detection = detect_session_staleness(
            record=record,
            probe_result=probe_result,
            config=config,
            now=_NOW + timedelta(seconds=5),  # Well within timeout
        )

        assert detection.is_stale is True
        assert detection.reason == StalenessReason.TRANSPORT_DISCONNECTED
        assert detection.transport_health == ConnectionHealth.DISCONNECTED

    def test_max_failures_exceeded_triggers_stale(self) -> None:
        """A session that exceeds max consecutive failures is stale."""
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
            consecutive_failures=2,
        )
        probe_result = _make_probe_result(
            health=ConnectionHealth.DEGRADED,
            success=False,
            exit_code=1,
            error="Non-zero exit code: 1",
        )
        config = DetectorConfig(
            heartbeat_timeout_seconds=30.0,
            max_missed_heartbeats=3,
        )

        detection = detect_session_staleness(
            record=record,
            probe_result=probe_result,
            config=config,
            now=_NOW + timedelta(seconds=5),
        )

        assert detection.is_stale is True
        assert detection.reason == StalenessReason.MAX_FAILURES_EXCEEDED
        assert detection.consecutive_failures == 3

    def test_degraded_under_max_failures_not_stale(self) -> None:
        """A degraded session under max failures is not yet stale."""
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
            consecutive_failures=1,
        )
        probe_result = _make_probe_result(
            health=ConnectionHealth.DEGRADED,
            success=False,
            exit_code=1,
            error="Non-zero exit code: 1",
        )
        config = DetectorConfig(
            heartbeat_timeout_seconds=30.0,
            max_missed_heartbeats=3,
        )

        detection = detect_session_staleness(
            record=record,
            probe_result=probe_result,
            config=config,
            now=_NOW + timedelta(seconds=5),
        )

        assert detection.is_stale is False
        assert detection.consecutive_failures == 2

    def test_none_probe_result_uses_heartbeat_only(self) -> None:
        """When no probe result is available, only heartbeat age matters."""
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        config = DetectorConfig(heartbeat_timeout_seconds=30.0)

        # Within timeout
        detection = detect_session_staleness(
            record=record,
            probe_result=None,
            config=config,
            now=_NOW + timedelta(seconds=10),
        )
        assert detection.is_stale is False

        # Exceeds timeout
        detection_stale = detect_session_staleness(
            record=record,
            probe_result=None,
            config=config,
            now=_NOW + timedelta(seconds=35),
        )
        assert detection_stale.is_stale is True
        assert detection_stale.reason == StalenessReason.HEARTBEAT_TIMEOUT

    def test_connected_probe_resets_heartbeat_concern(self) -> None:
        """A connected transport resets the consecutive failure count."""
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
            consecutive_failures=2,
        )
        probe_result = _make_probe_result(
            health=ConnectionHealth.CONNECTED,
            success=True,
        )
        config = DetectorConfig(heartbeat_timeout_seconds=30.0)

        detection = detect_session_staleness(
            record=record,
            probe_result=probe_result,
            config=config,
            now=_NOW + timedelta(seconds=5),
        )

        assert detection.is_stale is False
        assert detection.consecutive_failures == 0

    def test_detection_timestamp_matches_now(self) -> None:
        """The detection timestamp is the 'now' parameter."""
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
        )
        check_time = _NOW + timedelta(seconds=7)

        detection = detect_session_staleness(
            record=record,
            probe_result=_make_probe_result(),
            config=DetectorConfig(),
            now=check_time,
        )

        assert detection.detected_at == check_time

    def test_session_id_propagated(self) -> None:
        """The detection result carries the session ID from the record."""
        record = HeartbeatRecord(
            session_id="run-specific-id",
            last_heartbeat_at=_NOW,
        )
        detection = detect_session_staleness(
            record=record,
            probe_result=_make_probe_result(),
            config=DetectorConfig(),
            now=_NOW,
        )
        assert detection.session_id == "run-specific-id"

    def test_zero_max_missed_heartbeats_disables_failure_tracking(self) -> None:
        """When max_missed_heartbeats is 0, failure count never triggers stale."""
        record = HeartbeatRecord(
            session_id="run-abc",
            last_heartbeat_at=_NOW,
            consecutive_failures=100,
        )
        probe_result = _make_probe_result(
            health=ConnectionHealth.DEGRADED,
            success=False,
            exit_code=1,
            error="Non-zero exit",
        )
        config = DetectorConfig(
            heartbeat_timeout_seconds=30.0,
            max_missed_heartbeats=0,
        )

        detection = detect_session_staleness(
            record=record,
            probe_result=probe_result,
            config=config,
            now=_NOW + timedelta(seconds=5),
        )

        # Not stale because max_missed_heartbeats=0 disables that check
        # and heartbeat is within timeout
        assert detection.is_stale is False


# ---------------------------------------------------------------------------
# detect_batch_staleness -- multiple sessions
# ---------------------------------------------------------------------------


class TestDetectBatchStaleness:
    """Verify batch staleness detection for multiple sessions."""

    def test_empty_returns_empty(self) -> None:
        results = detect_batch_staleness(
            records=(),
            probe_results={},
            config=DetectorConfig(),
            now=_NOW,
        )
        assert results == ()

    def test_single_healthy_session(self) -> None:
        record = HeartbeatRecord(
            session_id="run-1",
            last_heartbeat_at=_NOW,
        )
        probe_results = {
            "run-1": _make_probe_result(health=ConnectionHealth.CONNECTED),
        }

        results = detect_batch_staleness(
            records=(record,),
            probe_results=probe_results,
            config=DetectorConfig(),
            now=_NOW + timedelta(seconds=5),
        )

        assert len(results) == 1
        assert results[0].is_stale is False

    def test_mixed_healthy_and_stale(self) -> None:
        healthy_record = HeartbeatRecord(
            session_id="run-healthy",
            last_heartbeat_at=_NOW,
        )
        stale_record = HeartbeatRecord(
            session_id="run-stale",
            last_heartbeat_at=_NOW - timedelta(seconds=60),
        )
        probe_results = {
            "run-healthy": _make_probe_result(health=ConnectionHealth.CONNECTED),
            "run-stale": _make_probe_result(
                health=ConnectionHealth.DISCONNECTED,
                success=False,
                exit_code=None,
                error="Timeout",
            ),
        }

        results = detect_batch_staleness(
            records=(healthy_record, stale_record),
            probe_results=probe_results,
            config=DetectorConfig(heartbeat_timeout_seconds=30.0),
            now=_NOW + timedelta(seconds=5),
        )

        assert len(results) == 2
        by_id = {r.session_id: r for r in results}
        assert by_id["run-healthy"].is_stale is False
        assert by_id["run-stale"].is_stale is True

    def test_missing_probe_result_uses_none(self) -> None:
        """When a session has no probe result, detection proceeds with None."""
        record = HeartbeatRecord(
            session_id="run-no-probe",
            last_heartbeat_at=_NOW,
        )

        results = detect_batch_staleness(
            records=(record,),
            probe_results={},  # No probe result for this session
            config=DetectorConfig(heartbeat_timeout_seconds=30.0),
            now=_NOW + timedelta(seconds=5),
        )

        assert len(results) == 1
        assert results[0].is_stale is False

    def test_preserves_order(self) -> None:
        """Results are returned in the same order as input records."""
        records = tuple(
            HeartbeatRecord(
                session_id=f"run-{i}",
                last_heartbeat_at=_NOW,
            )
            for i in range(5)
        )

        results = detect_batch_staleness(
            records=records,
            probe_results={},
            config=DetectorConfig(),
            now=_NOW,
        )

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.session_id == f"run-{i}"
