"""Tests for periodic stale subscriber sweep.

Validates:
    - SweepConfig validation (positive intervals, thresholds)
    - SubscriberMetadata immutability and age/idle calculations
    - StaleSubscriberDetection and SweepResult frozen dataclasses
    - Detection of orphaned subscribers (client no longer registered)
    - Detection of subscribers with excessive delivery failures
    - Detection of idle subscribers exceeding max age
    - Priority ordering: orphaned > failures > idle
    - Sweep removes detected stale subscribers via cleanup_subscriber
    - Background task lifecycle (start, stop, sweep_once)
    - Metadata tracker registration, activity recording, deregistration
    - Idempotent sweep (no removals when all subscribers are healthy)
    - Error isolation during cleanup
    - Sweep disabled by config
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from jules_daemon.cleanup.subscriber_sweep import (
    StaleSubscriberDetection,
    StaleSubscriberReason,
    StaleSubscriberSweep,
    SubscriberMetadata,
    SweepConfig,
    SweepResult,
    detect_stale_subscribers,
    sweep_stale_subscribers,
    _MetadataTracker,
)
from jules_daemon.ipc.connection_manager import ClientInfo, ConnectionManager
from jules_daemon.ipc.notification_broadcaster import NotificationBroadcaster
from jules_daemon.protocol.notifications import (
    CompletionNotification,
    NotificationEnvelope,
    NotificationEventType,
    create_notification_envelope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _minutes_ago(minutes: float) -> datetime:
    return _utc_now() - timedelta(minutes=minutes)


def _seconds_ago(seconds: float) -> datetime:
    return _utc_now() - timedelta(seconds=seconds)


def _make_completion_envelope(
    run_id: str = "run-001",
) -> NotificationEnvelope:
    payload = CompletionNotification(
        run_id=run_id,
        natural_language_command="Run pytest",
        exit_status=0,
    )
    return create_notification_envelope(
        event_type=NotificationEventType.COMPLETION,
        payload=payload,
    )


# ---------------------------------------------------------------------------
# SweepConfig
# ---------------------------------------------------------------------------


class TestSweepConfig:
    """Tests for SweepConfig validation."""

    def test_defaults(self) -> None:
        config = SweepConfig()
        assert config.sweep_interval_seconds == 60.0
        assert config.max_idle_seconds == 300.0
        assert config.failure_count_threshold == 3
        assert config.enabled is True

    def test_custom_values(self) -> None:
        config = SweepConfig(
            sweep_interval_seconds=30.0,
            max_idle_seconds=120.0,
            failure_count_threshold=2,
            enabled=False,
        )
        assert config.sweep_interval_seconds == 30.0
        assert config.max_idle_seconds == 120.0
        assert config.failure_count_threshold == 2
        assert config.enabled is False

    def test_frozen(self) -> None:
        config = SweepConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore[misc]

    def test_zero_sweep_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="sweep_interval_seconds must be positive"):
            SweepConfig(sweep_interval_seconds=0)

    def test_negative_sweep_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="sweep_interval_seconds must be positive"):
            SweepConfig(sweep_interval_seconds=-10.0)

    def test_zero_max_idle_raises(self) -> None:
        with pytest.raises(
            ValueError, match="max_idle_seconds must be positive"
        ):
            SweepConfig(max_idle_seconds=0)

    def test_negative_max_idle_raises(self) -> None:
        with pytest.raises(
            ValueError, match="max_idle_seconds must be positive"
        ):
            SweepConfig(max_idle_seconds=-5.0)

    def test_zero_failure_threshold_raises(self) -> None:
        with pytest.raises(
            ValueError, match="failure_count_threshold must be at least 1"
        ):
            SweepConfig(failure_count_threshold=0)

    def test_negative_failure_threshold_raises(self) -> None:
        with pytest.raises(
            ValueError, match="failure_count_threshold must be at least 1"
        ):
            SweepConfig(failure_count_threshold=-1)


# ---------------------------------------------------------------------------
# SubscriberMetadata
# ---------------------------------------------------------------------------


class TestSubscriberMetadata:
    """Tests for SubscriberMetadata frozen dataclass."""

    def test_basic_creation(self) -> None:
        now = _utc_now()
        meta = SubscriberMetadata(
            subscription_id="nsub-abc123",
            client_id="client-xyz",
            registered_at=now,
            last_active_at=now,
        )
        assert meta.subscription_id == "nsub-abc123"
        assert meta.client_id == "client-xyz"
        assert meta.registered_at == now
        assert meta.last_active_at == now

    def test_none_client_id(self) -> None:
        now = _utc_now()
        meta = SubscriberMetadata(
            subscription_id="nsub-abc123",
            client_id=None,
            registered_at=now,
            last_active_at=now,
        )
        assert meta.client_id is None

    def test_frozen(self) -> None:
        now = _utc_now()
        meta = SubscriberMetadata(
            subscription_id="nsub-abc123",
            client_id=None,
            registered_at=now,
            last_active_at=now,
        )
        with pytest.raises(AttributeError):
            meta.client_id = "changed"  # type: ignore[misc]

    def test_empty_subscription_id_raises(self) -> None:
        now = _utc_now()
        with pytest.raises(ValueError, match="subscription_id must not be empty"):
            SubscriberMetadata(
                subscription_id="",
                client_id=None,
                registered_at=now,
                last_active_at=now,
            )

    def test_age_seconds(self) -> None:
        registered = _minutes_ago(5)
        now = _utc_now()
        meta = SubscriberMetadata(
            subscription_id="nsub-test",
            client_id=None,
            registered_at=registered,
            last_active_at=registered,
        )
        age = meta.age_seconds(now)
        assert 290.0 <= age <= 310.0  # approximately 5 minutes

    def test_idle_seconds(self) -> None:
        registered = _minutes_ago(10)
        active_at = _minutes_ago(3)
        now = _utc_now()
        meta = SubscriberMetadata(
            subscription_id="nsub-test",
            client_id=None,
            registered_at=registered,
            last_active_at=active_at,
        )
        idle = meta.idle_seconds(now)
        assert 170.0 <= idle <= 190.0  # approximately 3 minutes

    def test_age_never_negative(self) -> None:
        future = _utc_now() + timedelta(hours=1)
        meta = SubscriberMetadata(
            subscription_id="nsub-test",
            client_id=None,
            registered_at=future,
            last_active_at=future,
        )
        assert meta.age_seconds(_utc_now()) == 0.0

    def test_naive_registered_at_raises(self) -> None:
        naive_dt = datetime(2026, 1, 1, 12, 0, 0)  # no tzinfo
        with pytest.raises(ValueError, match="registered_at must be timezone-aware"):
            SubscriberMetadata(
                subscription_id="nsub-test",
                client_id=None,
                registered_at=naive_dt,
                last_active_at=_utc_now(),
            )

    def test_naive_last_active_at_raises(self) -> None:
        naive_dt = datetime(2026, 1, 1, 12, 0, 0)  # no tzinfo
        with pytest.raises(
            ValueError, match="last_active_at must be timezone-aware"
        ):
            SubscriberMetadata(
                subscription_id="nsub-test",
                client_id=None,
                registered_at=_utc_now(),
                last_active_at=naive_dt,
            )


# ---------------------------------------------------------------------------
# StaleSubscriberDetection
# ---------------------------------------------------------------------------


class TestStaleSubscriberDetection:
    """Tests for StaleSubscriberDetection frozen dataclass."""

    def test_creation(self) -> None:
        now = _utc_now()
        detection = StaleSubscriberDetection(
            subscription_id="nsub-abc",
            reason=StaleSubscriberReason.ORPHANED_NO_CLIENT,
            reason_detail="Client gone",
            detected_at=now,
        )
        assert detection.subscription_id == "nsub-abc"
        assert detection.reason == StaleSubscriberReason.ORPHANED_NO_CLIENT
        assert detection.reason_detail == "Client gone"
        assert detection.detected_at == now

    def test_frozen(self) -> None:
        now = _utc_now()
        detection = StaleSubscriberDetection(
            subscription_id="nsub-abc",
            reason=StaleSubscriberReason.IDLE_TIMEOUT,
            reason_detail="Idle too long",
            detected_at=now,
        )
        with pytest.raises(AttributeError):
            detection.reason = StaleSubscriberReason.EXCESSIVE_FAILURES  # type: ignore[misc]

    def test_empty_subscription_id_raises(self) -> None:
        with pytest.raises(ValueError, match="subscription_id must not be empty"):
            StaleSubscriberDetection(
                subscription_id="  ",
                reason=StaleSubscriberReason.IDLE_TIMEOUT,
                reason_detail="Detail",
                detected_at=_utc_now(),
            )

    def test_empty_reason_detail_raises(self) -> None:
        with pytest.raises(ValueError, match="reason_detail must not be empty"):
            StaleSubscriberDetection(
                subscription_id="nsub-abc",
                reason=StaleSubscriberReason.IDLE_TIMEOUT,
                reason_detail="  ",
                detected_at=_utc_now(),
            )


# ---------------------------------------------------------------------------
# SweepResult
# ---------------------------------------------------------------------------


class TestSweepResult:
    """Tests for SweepResult frozen dataclass."""

    def test_clean_result(self) -> None:
        result = SweepResult(
            sweep_id="sweep-abc",
            swept_at=_utc_now(),
            subscribers_checked=5,
            stale_detected=0,
            removed_count=0,
        )
        assert result.clean is True
        assert result.has_errors is False
        assert result.subscribers_checked == 5

    def test_result_with_removals(self) -> None:
        result = SweepResult(
            sweep_id="sweep-abc",
            swept_at=_utc_now(),
            subscribers_checked=5,
            stale_detected=2,
            removed_count=2,
            duration_ms=12.5,
        )
        assert result.clean is False
        assert result.stale_detected == 2
        assert result.removed_count == 2

    def test_result_with_errors(self) -> None:
        result = SweepResult(
            sweep_id="sweep-abc",
            swept_at=_utc_now(),
            subscribers_checked=3,
            stale_detected=1,
            removed_count=0,
            errors=("Failed to clean nsub-x",),
        )
        assert result.has_errors is True

    def test_frozen(self) -> None:
        result = SweepResult(
            sweep_id="sweep-abc",
            swept_at=_utc_now(),
            subscribers_checked=0,
            stale_detected=0,
            removed_count=0,
        )
        with pytest.raises(AttributeError):
            result.removed_count = 5  # type: ignore[misc]

    def test_empty_sweep_id_raises(self) -> None:
        with pytest.raises(ValueError, match="sweep_id must not be empty"):
            SweepResult(
                sweep_id="",
                swept_at=_utc_now(),
                subscribers_checked=0,
                stale_detected=0,
                removed_count=0,
            )

    def test_negative_subscribers_checked_raises(self) -> None:
        with pytest.raises(
            ValueError, match="subscribers_checked must not be negative"
        ):
            SweepResult(
                sweep_id="sweep-x",
                swept_at=_utc_now(),
                subscribers_checked=-1,
                stale_detected=0,
                removed_count=0,
            )

    def test_negative_stale_detected_raises(self) -> None:
        with pytest.raises(ValueError, match="stale_detected must not be negative"):
            SweepResult(
                sweep_id="sweep-x",
                swept_at=_utc_now(),
                subscribers_checked=0,
                stale_detected=-1,
                removed_count=0,
            )

    def test_negative_removed_count_raises(self) -> None:
        with pytest.raises(ValueError, match="removed_count must not be negative"):
            SweepResult(
                sweep_id="sweep-x",
                swept_at=_utc_now(),
                subscribers_checked=0,
                stale_detected=0,
                removed_count=-1,
            )

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_ms must not be negative"):
            SweepResult(
                sweep_id="sweep-x",
                swept_at=_utc_now(),
                subscribers_checked=0,
                stale_detected=0,
                removed_count=0,
                duration_ms=-1.0,
            )


# ---------------------------------------------------------------------------
# _MetadataTracker
# ---------------------------------------------------------------------------


class TestMetadataTracker:
    """Tests for the internal metadata tracker."""

    def test_register_and_get(self) -> None:
        tracker = _MetadataTracker()
        now = _utc_now()
        meta = tracker.register("nsub-abc", "client-1", now)

        assert meta.subscription_id == "nsub-abc"
        assert meta.client_id == "client-1"
        assert meta.registered_at == now

        retrieved = tracker.get("nsub-abc")
        assert retrieved == meta

    def test_register_without_client(self) -> None:
        tracker = _MetadataTracker()
        meta = tracker.register("nsub-abc", None, _utc_now())
        assert meta.client_id is None

    def test_get_nonexistent_returns_none(self) -> None:
        tracker = _MetadataTracker()
        assert tracker.get("nsub-missing") is None

    def test_record_activity(self) -> None:
        tracker = _MetadataTracker()
        old_time = _minutes_ago(5)
        tracker.register("nsub-abc", "client-1", old_time)

        new_time = _utc_now()
        assert tracker.record_activity("nsub-abc", new_time) is True

        meta = tracker.get("nsub-abc")
        assert meta is not None
        assert meta.last_active_at == new_time
        assert meta.registered_at == old_time  # unchanged

    def test_record_activity_nonexistent(self) -> None:
        tracker = _MetadataTracker()
        assert tracker.record_activity("nsub-missing", _utc_now()) is False

    def test_deregister(self) -> None:
        tracker = _MetadataTracker()
        tracker.register("nsub-abc", "client-1", _utc_now())

        assert tracker.deregister("nsub-abc") is True
        assert tracker.get("nsub-abc") is None
        assert tracker.count == 0

    def test_deregister_nonexistent(self) -> None:
        tracker = _MetadataTracker()
        assert tracker.deregister("nsub-missing") is False

    def test_all_metadata(self) -> None:
        tracker = _MetadataTracker()
        now = _utc_now()
        tracker.register("nsub-a", "client-1", now)
        tracker.register("nsub-b", "client-2", now)

        all_meta = tracker.all_metadata()
        assert len(all_meta) == 2
        ids = {m.subscription_id for m in all_meta}
        assert ids == {"nsub-a", "nsub-b"}

    def test_count(self) -> None:
        tracker = _MetadataTracker()
        assert tracker.count == 0
        tracker.register("nsub-a", None, _utc_now())
        assert tracker.count == 1
        tracker.register("nsub-b", None, _utc_now())
        assert tracker.count == 2
        tracker.deregister("nsub-a")
        assert tracker.count == 1


# ---------------------------------------------------------------------------
# detect_stale_subscribers -- orphaned (no client)
# ---------------------------------------------------------------------------


class TestDetectOrphaned:
    """Tests for detecting orphaned subscribers (client no longer registered)."""

    @pytest.mark.asyncio
    async def test_orphaned_subscriber_detected(self) -> None:
        broadcaster = NotificationBroadcaster()
        connection_manager = ConnectionManager()
        tracker = _MetadataTracker()
        config = SweepConfig()
        now = _utc_now()

        # Create subscriber and register metadata
        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, "client-gone", now)

        # Client is NOT in connection manager -> orphaned
        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 1
        assert detections[0].subscription_id == handle.subscription_id
        assert detections[0].reason == StaleSubscriberReason.ORPHANED_NO_CLIENT
        assert "client-gone" in detections[0].reason_detail

    @pytest.mark.asyncio
    async def test_subscriber_with_active_client_not_orphaned(self) -> None:
        broadcaster = NotificationBroadcaster()
        connection_manager = ConnectionManager()
        tracker = _MetadataTracker()
        config = SweepConfig()
        now = _utc_now()

        # Add client to connection manager
        client_info = ClientInfo(
            client_id="client-active",
            connected_at=now.isoformat(),
        )
        await connection_manager.add_client(client_info)

        # Create subscriber associated with the active client
        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, "client-active", now)

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_subscriber_with_none_client_not_orphaned(self) -> None:
        """Subscribers without a client_id should not be flagged as orphaned."""
        broadcaster = NotificationBroadcaster()
        connection_manager = ConnectionManager()
        tracker = _MetadataTracker()
        config = SweepConfig()
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, None, now)

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 0


# ---------------------------------------------------------------------------
# detect_stale_subscribers -- excessive failures
# ---------------------------------------------------------------------------


class TestDetectExcessiveFailures:
    """Tests for detecting subscribers with excessive delivery failures."""

    @pytest.mark.asyncio
    async def test_high_failure_count_detected(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig(failure_count_threshold=3)
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, None, now)

        # Simulate failures
        broadcaster._failure_counts[handle.subscription_id] = 3

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 1
        assert detections[0].reason == StaleSubscriberReason.EXCESSIVE_FAILURES
        assert "3" in detections[0].reason_detail

    @pytest.mark.asyncio
    async def test_below_threshold_not_detected(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig(failure_count_threshold=5)
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, None, now)

        # Failures below threshold
        broadcaster._failure_counts[handle.subscription_id] = 4

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 0


# ---------------------------------------------------------------------------
# detect_stale_subscribers -- idle timeout
# ---------------------------------------------------------------------------


class TestDetectIdleTimeout:
    """Tests for detecting idle subscribers exceeding max age."""

    @pytest.mark.asyncio
    async def test_idle_subscriber_detected(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig(max_idle_seconds=60.0)
        now = _utc_now()

        handle = await broadcaster.subscribe()
        # Registered 2 minutes ago, no activity since
        old_time = now - timedelta(seconds=120)
        tracker.register(handle.subscription_id, None, old_time)

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 1
        assert detections[0].reason == StaleSubscriberReason.IDLE_TIMEOUT
        assert "120" in detections[0].reason_detail or "119" in detections[0].reason_detail

    @pytest.mark.asyncio
    async def test_active_subscriber_not_idle(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig(max_idle_seconds=60.0)
        now = _utc_now()

        handle = await broadcaster.subscribe()
        # Registered 2 minutes ago but active 10 seconds ago
        old_time = now - timedelta(seconds=120)
        tracker.register(handle.subscription_id, None, old_time)
        tracker.record_activity(handle.subscription_id, now - timedelta(seconds=10))

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 0

    @pytest.mark.asyncio
    async def test_subscriber_without_metadata_skips_idle_check(self) -> None:
        """Subscribers not tracked in metadata are not flagged for idle."""
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig(max_idle_seconds=60.0)
        now = _utc_now()

        # Create subscriber without registering in tracker
        await broadcaster.subscribe()

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        # No metadata -> no idle or orphan detection possible
        assert len(detections) == 0


# ---------------------------------------------------------------------------
# detect_stale_subscribers -- priority ordering
# ---------------------------------------------------------------------------


class TestDetectPriorityOrdering:
    """Tests that detection criteria are evaluated in priority order."""

    @pytest.mark.asyncio
    async def test_orphan_takes_priority_over_failures(self) -> None:
        broadcaster = NotificationBroadcaster()
        connection_manager = ConnectionManager()
        tracker = _MetadataTracker()
        config = SweepConfig(failure_count_threshold=2)
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, "client-gone", now)
        broadcaster._failure_counts[handle.subscription_id] = 5

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 1
        # Orphan should win over failures
        assert detections[0].reason == StaleSubscriberReason.ORPHANED_NO_CLIENT

    @pytest.mark.asyncio
    async def test_failures_takes_priority_over_idle(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig(
            failure_count_threshold=2,
            max_idle_seconds=60.0,
        )
        now = _utc_now()

        handle = await broadcaster.subscribe()
        old_time = now - timedelta(seconds=120)
        tracker.register(handle.subscription_id, None, old_time)
        broadcaster._failure_counts[handle.subscription_id] = 3

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 1
        # Failures should win over idle
        assert detections[0].reason == StaleSubscriberReason.EXCESSIVE_FAILURES


# ---------------------------------------------------------------------------
# detect_stale_subscribers -- no connection manager
# ---------------------------------------------------------------------------


class TestDetectWithoutConnectionManager:
    """Tests for detection when ConnectionManager is None (orphan check skipped)."""

    @pytest.mark.asyncio
    async def test_orphan_check_skipped_without_manager(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig()
        now = _utc_now()

        handle = await broadcaster.subscribe()
        # Register with a client_id but no ConnectionManager to check against
        tracker.register(handle.subscription_id, "client-maybe-gone", now)

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        # Should not detect as orphaned without connection manager
        assert len(detections) == 0


# ---------------------------------------------------------------------------
# detect_stale_subscribers -- multiple subscribers
# ---------------------------------------------------------------------------


class TestDetectMultipleSubscribers:
    """Tests for batch detection across multiple subscribers."""

    @pytest.mark.asyncio
    async def test_mixed_healthy_and_stale(self) -> None:
        broadcaster = NotificationBroadcaster()
        connection_manager = ConnectionManager()
        tracker = _MetadataTracker()
        config = SweepConfig(
            failure_count_threshold=3,
            max_idle_seconds=60.0,
        )
        now = _utc_now()

        # Add one active client
        client_info = ClientInfo(
            client_id="client-active",
            connected_at=now.isoformat(),
        )
        await connection_manager.add_client(client_info)

        # Healthy subscriber
        handle_healthy = await broadcaster.subscribe()
        tracker.register(handle_healthy.subscription_id, "client-active", now)

        # Orphaned subscriber
        handle_orphan = await broadcaster.subscribe()
        tracker.register(handle_orphan.subscription_id, "client-gone", now)

        # Failing subscriber (no client association, just high failures)
        handle_failing = await broadcaster.subscribe()
        tracker.register(handle_failing.subscription_id, None, now)
        broadcaster._failure_counts[handle_failing.subscription_id] = 5

        # Idle subscriber
        handle_idle = await broadcaster.subscribe()
        old_time = now - timedelta(seconds=120)
        tracker.register(handle_idle.subscription_id, None, old_time)

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(detections) == 3

        reasons = {d.subscription_id: d.reason for d in detections}
        assert reasons[handle_orphan.subscription_id] == StaleSubscriberReason.ORPHANED_NO_CLIENT
        assert reasons[handle_failing.subscription_id] == StaleSubscriberReason.EXCESSIVE_FAILURES
        assert reasons[handle_idle.subscription_id] == StaleSubscriberReason.IDLE_TIMEOUT

    @pytest.mark.asyncio
    async def test_no_subscribers_returns_empty(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig()

        detections = detect_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
        )

        assert detections == ()


# ---------------------------------------------------------------------------
# sweep_stale_subscribers -- removes detected subscribers
# ---------------------------------------------------------------------------


class TestSweepStaleSubscribers:
    """Tests for the sweep function that detects and removes."""

    @pytest.mark.asyncio
    async def test_sweep_removes_orphaned_subscriber(self) -> None:
        broadcaster = NotificationBroadcaster()
        connection_manager = ConnectionManager()
        tracker = _MetadataTracker()
        config = SweepConfig()
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, "client-gone", now)

        assert broadcaster.subscriber_count == 1

        result = await sweep_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert result.stale_detected == 1
        assert result.removed_count == 1
        assert broadcaster.subscriber_count == 0
        assert not result.has_errors
        assert result.sweep_id.startswith("sweep-")

    @pytest.mark.asyncio
    async def test_sweep_removes_failing_subscriber(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig(failure_count_threshold=2)
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, None, now)
        broadcaster._failure_counts[handle.subscription_id] = 3

        result = await sweep_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert result.stale_detected == 1
        assert result.removed_count == 1
        assert broadcaster.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_sweep_deregisters_metadata(self) -> None:
        broadcaster = NotificationBroadcaster()
        connection_manager = ConnectionManager()
        tracker = _MetadataTracker()
        config = SweepConfig()
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, "client-gone", now)
        assert tracker.count == 1

        await sweep_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert tracker.count == 0

    @pytest.mark.asyncio
    async def test_clean_sweep_no_removals(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig()
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, None, now)

        result = await sweep_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert result.clean is True
        assert result.removed_count == 0
        assert broadcaster.subscriber_count == 1

    @pytest.mark.asyncio
    async def test_sweep_result_has_duration(self) -> None:
        broadcaster = NotificationBroadcaster()
        tracker = _MetadataTracker()
        config = SweepConfig()

        result = await sweep_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=None,
            metadata_tracker=tracker,
            config=config,
        )

        assert result.duration_ms >= 0.0

    @pytest.mark.asyncio
    async def test_sweep_includes_cleanup_results(self) -> None:
        broadcaster = NotificationBroadcaster()
        connection_manager = ConnectionManager()
        tracker = _MetadataTracker()
        config = SweepConfig()
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, "client-gone", now)

        # Add an event to the queue so we can see it drained
        envelope = _make_completion_envelope()
        await broadcaster.broadcast(envelope)

        result = await sweep_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert len(result.removal_results) == 1
        assert result.removal_results[0].found is True
        assert result.removal_results[0].items_drained == 1


# ---------------------------------------------------------------------------
# sweep_stale_subscribers -- idempotent
# ---------------------------------------------------------------------------


class TestSweepIdempotent:
    """Tests that sweeping twice is harmless."""

    @pytest.mark.asyncio
    async def test_double_sweep_no_double_removal(self) -> None:
        broadcaster = NotificationBroadcaster()
        connection_manager = ConnectionManager()
        tracker = _MetadataTracker()
        config = SweepConfig()
        now = _utc_now()

        handle = await broadcaster.subscribe()
        tracker.register(handle.subscription_id, "client-gone", now)

        result1 = await sweep_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )
        result2 = await sweep_stale_subscribers(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            metadata_tracker=tracker,
            config=config,
            now=now,
        )

        assert result1.removed_count == 1
        assert result2.removed_count == 0
        assert result2.clean is True


# ---------------------------------------------------------------------------
# StaleSubscriberSweep -- lifecycle
# ---------------------------------------------------------------------------


class TestStaleSubscriberSweepLifecycle:
    """Tests for the background sweep task lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
            config=SweepConfig(sweep_interval_seconds=0.1),
        )

        assert sweep.is_running is False
        await sweep.start()
        assert sweep.is_running is True

        await sweep.stop()
        assert sweep.is_running is False

    @pytest.mark.asyncio
    async def test_start_when_disabled(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
            config=SweepConfig(enabled=False),
        )

        await sweep.start()
        assert sweep.is_running is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
            config=SweepConfig(sweep_interval_seconds=0.1),
        )

        await sweep.start()
        await sweep.start()  # should be no-op
        assert sweep.is_running is True

        await sweep.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
        )

        # Should not raise
        await sweep.stop()

    @pytest.mark.asyncio
    async def test_sweep_once_without_task(self) -> None:
        """Manual sweep_once() works without starting the background task."""
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
        )

        result = await sweep.sweep_once()
        assert result.clean is True
        assert sweep.sweep_count == 1

    @pytest.mark.asyncio
    async def test_sweep_count_increments(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
        )

        assert sweep.sweep_count == 0
        await sweep.sweep_once()
        assert sweep.sweep_count == 1
        await sweep.sweep_once()
        assert sweep.sweep_count == 2

    @pytest.mark.asyncio
    async def test_last_result_updated(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
        )

        assert sweep.last_result is None
        result = await sweep.sweep_once()
        assert sweep.last_result is result


# ---------------------------------------------------------------------------
# StaleSubscriberSweep -- subscriber registration
# ---------------------------------------------------------------------------


class TestStaleSubscriberSweepRegistration:
    """Tests for subscriber metadata registration on the sweep."""

    @pytest.mark.asyncio
    async def test_register_subscriber(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(broadcaster=broadcaster)

        meta = sweep.register_subscriber("nsub-abc", client_id="client-1")
        assert meta.subscription_id == "nsub-abc"
        assert meta.client_id == "client-1"
        assert sweep.tracked_subscriber_count == 1

    @pytest.mark.asyncio
    async def test_record_activity(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(broadcaster=broadcaster)

        sweep.register_subscriber("nsub-abc", client_id="client-1")
        assert sweep.record_activity("nsub-abc") is True
        assert sweep.record_activity("nsub-missing") is False

    @pytest.mark.asyncio
    async def test_deregister_subscriber(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(broadcaster=broadcaster)

        sweep.register_subscriber("nsub-abc")
        assert sweep.deregister_subscriber("nsub-abc") is True
        assert sweep.tracked_subscriber_count == 0

    @pytest.mark.asyncio
    async def test_get_metadata(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(broadcaster=broadcaster)

        sweep.register_subscriber("nsub-abc", client_id="client-1")
        meta = sweep.get_metadata("nsub-abc")
        assert meta is not None
        assert meta.client_id == "client-1"

    @pytest.mark.asyncio
    async def test_get_metadata_nonexistent(self) -> None:
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(broadcaster=broadcaster)

        assert sweep.get_metadata("nsub-missing") is None


# ---------------------------------------------------------------------------
# StaleSubscriberSweep -- integration with sweep_once
# ---------------------------------------------------------------------------


class TestStaleSubscriberSweepIntegration:
    """Integration tests using sweep_once to detect and remove stale subscribers."""

    @pytest.mark.asyncio
    async def test_sweep_removes_orphaned_via_sweep_once(self) -> None:
        connection_manager = ConnectionManager()
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
        )

        handle = await broadcaster.subscribe()
        sweep.register_subscriber(
            handle.subscription_id, client_id="client-gone"
        )

        result = await sweep.sweep_once()
        assert result.stale_detected == 1
        assert result.removed_count == 1
        assert broadcaster.subscriber_count == 0
        assert sweep.tracked_subscriber_count == 0

    @pytest.mark.asyncio
    async def test_sweep_preserves_healthy_subscribers(self) -> None:
        connection_manager = ConnectionManager()
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
        )

        # Add active client
        client_info = ClientInfo(
            client_id="client-active",
            connected_at=_utc_now().isoformat(),
        )
        await connection_manager.add_client(client_info)

        handle = await broadcaster.subscribe()
        sweep.register_subscriber(
            handle.subscription_id, client_id="client-active"
        )

        result = await sweep.sweep_once()
        assert result.clean is True
        assert broadcaster.subscriber_count == 1
        assert sweep.tracked_subscriber_count == 1

    @pytest.mark.asyncio
    async def test_background_loop_runs_sweep(self) -> None:
        """Verify the background task actually runs sweeps."""
        connection_manager = ConnectionManager()
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            config=SweepConfig(sweep_interval_seconds=0.05),  # 50ms
        )

        handle = await broadcaster.subscribe()
        sweep.register_subscriber(
            handle.subscription_id, client_id="client-gone"
        )

        await sweep.start()
        # Wait enough for at least one sweep cycle
        await asyncio.sleep(0.15)
        await sweep.stop()

        assert sweep.sweep_count >= 1
        assert broadcaster.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_sweep_with_mixed_subscribers(self) -> None:
        connection_manager = ConnectionManager()
        broadcaster = NotificationBroadcaster()
        sweep = StaleSubscriberSweep(
            broadcaster=broadcaster,
            connection_manager=connection_manager,
            config=SweepConfig(
                failure_count_threshold=2,
                max_idle_seconds=30.0,
            ),
        )

        now = _utc_now()

        # Active client
        client_info = ClientInfo(
            client_id="client-active",
            connected_at=now.isoformat(),
        )
        await connection_manager.add_client(client_info)

        # Healthy subscriber
        handle_healthy = await broadcaster.subscribe()
        sweep.register_subscriber(
            handle_healthy.subscription_id,
            client_id="client-active",
            now=now,
        )

        # Orphaned subscriber
        handle_orphan = await broadcaster.subscribe()
        sweep.register_subscriber(
            handle_orphan.subscription_id,
            client_id="client-dead",
            now=now,
        )

        # Failing subscriber
        handle_fail = await broadcaster.subscribe()
        sweep.register_subscriber(
            handle_fail.subscription_id,
            now=now,
        )
        broadcaster._failure_counts[handle_fail.subscription_id] = 3

        result = await sweep.sweep_once()

        assert result.stale_detected == 2
        assert result.removed_count == 2
        assert broadcaster.subscriber_count == 1  # only healthy remains
        assert broadcaster.has_subscriber(handle_healthy.subscription_id)


# ---------------------------------------------------------------------------
# StaleSubscriberReason enum
# ---------------------------------------------------------------------------


class TestStaleSubscriberReason:
    """Tests for the StaleSubscriberReason enum."""

    def test_values(self) -> None:
        assert StaleSubscriberReason.ORPHANED_NO_CLIENT.value == "orphaned_no_client"
        assert StaleSubscriberReason.EXCESSIVE_FAILURES.value == "excessive_failures"
        assert StaleSubscriberReason.IDLE_TIMEOUT.value == "idle_timeout"

    def test_all_members(self) -> None:
        assert len(StaleSubscriberReason) == 3
