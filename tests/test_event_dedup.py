"""Tests for last-seen event ID tracker and deduplication filter.

Covers EventIdTracker, EventDeduplicationConfig, DeduplicationVerdict,
and the filter_envelope convenience method. Validates:

- First-seen event IDs are not duplicates
- Repeated event IDs are detected as duplicates
- last_seen_event_id tracks the most recently recorded ID
- Capacity limits evict oldest entries (FIFO)
- Empty/blank event IDs are rejected
- NotificationEnvelope integration via filter_envelope
- Thread safety under concurrent access
- clear() resets all state
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from jules_daemon.ipc.event_dedup import (
    DeduplicationVerdict,
    EventDeduplicationConfig,
    EventIdTracker,
)
from jules_daemon.protocol.notifications import (
    HeartbeatNotification,
    NotificationEnvelope,
    NotificationEventType,
    create_notification_envelope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_envelope(event_id: str = "evt-test-001") -> NotificationEnvelope:
    """Create a minimal notification envelope for testing."""
    return NotificationEnvelope(
        channel_version="1.0.0",
        event_id=event_id,
        timestamp=datetime.now(timezone.utc),
        event_type=NotificationEventType.HEARTBEAT,
        payload=HeartbeatNotification(
            daemon_uptime_seconds=42.0,
        ),
    )


# ---------------------------------------------------------------------------
# EventDeduplicationConfig tests
# ---------------------------------------------------------------------------


class TestEventDeduplicationConfig:
    """Tests for the immutable config dataclass."""

    def test_default_values(self) -> None:
        """Default config uses sensible capacity."""
        config = EventDeduplicationConfig()
        assert config.max_tracked_ids > 0

    def test_custom_capacity(self) -> None:
        """Config accepts a custom max_tracked_ids value."""
        config = EventDeduplicationConfig(max_tracked_ids=50)
        assert config.max_tracked_ids == 50

    def test_frozen(self) -> None:
        """Config is immutable (frozen dataclass)."""
        config = EventDeduplicationConfig()
        with pytest.raises(AttributeError):
            config.max_tracked_ids = 999  # type: ignore[misc]

    def test_zero_capacity_rejected(self) -> None:
        """Zero capacity raises ValueError."""
        with pytest.raises(ValueError, match="max_tracked_ids"):
            EventDeduplicationConfig(max_tracked_ids=0)

    def test_negative_capacity_rejected(self) -> None:
        """Negative capacity raises ValueError."""
        with pytest.raises(ValueError, match="max_tracked_ids"):
            EventDeduplicationConfig(max_tracked_ids=-5)


# ---------------------------------------------------------------------------
# DeduplicationVerdict tests
# ---------------------------------------------------------------------------


class TestDeduplicationVerdict:
    """Tests for the immutable verdict dataclass."""

    def test_construction(self) -> None:
        """Verdict can be constructed with required fields."""
        verdict = DeduplicationVerdict(
            event_id="evt-123",
            is_duplicate=False,
        )
        assert verdict.event_id == "evt-123"
        assert verdict.is_duplicate is False

    def test_frozen(self) -> None:
        """Verdict is immutable (frozen dataclass)."""
        verdict = DeduplicationVerdict(event_id="evt-1", is_duplicate=True)
        with pytest.raises(AttributeError):
            verdict.is_duplicate = False  # type: ignore[misc]

    def test_empty_event_id_rejected(self) -> None:
        """Empty event_id raises ValueError."""
        with pytest.raises(ValueError, match="event_id"):
            DeduplicationVerdict(event_id="", is_duplicate=False)

    def test_whitespace_event_id_rejected(self) -> None:
        """Whitespace-only event_id raises ValueError."""
        with pytest.raises(ValueError, match="event_id"):
            DeduplicationVerdict(event_id="   ", is_duplicate=False)


# ---------------------------------------------------------------------------
# EventIdTracker -- basic recording
# ---------------------------------------------------------------------------


class TestEventIdTrackerBasic:
    """Tests for basic record/query behaviour."""

    def test_first_event_not_duplicate(self) -> None:
        """First time seeing an event ID returns is_duplicate=False."""
        tracker = EventIdTracker()
        verdict = tracker.record("evt-001")
        assert verdict.is_duplicate is False
        assert verdict.event_id == "evt-001"

    def test_second_event_is_duplicate(self) -> None:
        """Recording the same event ID again returns is_duplicate=True."""
        tracker = EventIdTracker()
        tracker.record("evt-001")
        verdict = tracker.record("evt-001")
        assert verdict.is_duplicate is True

    def test_different_events_not_duplicates(self) -> None:
        """Different event IDs are not duplicates of each other."""
        tracker = EventIdTracker()
        v1 = tracker.record("evt-001")
        v2 = tracker.record("evt-002")
        assert v1.is_duplicate is False
        assert v2.is_duplicate is False

    def test_empty_event_id_rejected(self) -> None:
        """Recording an empty event ID raises ValueError."""
        tracker = EventIdTracker()
        with pytest.raises(ValueError, match="event_id"):
            tracker.record("")

    def test_whitespace_event_id_rejected(self) -> None:
        """Recording a whitespace-only event ID raises ValueError."""
        tracker = EventIdTracker()
        with pytest.raises(ValueError, match="event_id"):
            tracker.record("   ")


# ---------------------------------------------------------------------------
# EventIdTracker -- last_seen_event_id property
# ---------------------------------------------------------------------------


class TestEventIdTrackerLastSeen:
    """Tests for last_seen_event_id tracking."""

    def test_initially_none(self) -> None:
        """Before any events, last_seen_event_id is None."""
        tracker = EventIdTracker()
        assert tracker.last_seen_event_id is None

    def test_tracks_most_recent(self) -> None:
        """last_seen_event_id reflects the most recently recorded ID."""
        tracker = EventIdTracker()
        tracker.record("evt-001")
        assert tracker.last_seen_event_id == "evt-001"

        tracker.record("evt-002")
        assert tracker.last_seen_event_id == "evt-002"

    def test_duplicate_updates_last_seen(self) -> None:
        """Recording a duplicate still updates last_seen_event_id."""
        tracker = EventIdTracker()
        tracker.record("evt-001")
        tracker.record("evt-002")
        assert tracker.last_seen_event_id == "evt-002"

        # Re-record evt-001 -- should update last_seen
        tracker.record("evt-001")
        assert tracker.last_seen_event_id == "evt-001"


# ---------------------------------------------------------------------------
# EventIdTracker -- is_duplicate query (non-recording)
# ---------------------------------------------------------------------------


class TestEventIdTrackerIsDuplicate:
    """Tests for the non-recording is_duplicate() method."""

    def test_unknown_not_duplicate(self) -> None:
        """An event ID not yet recorded is not a duplicate."""
        tracker = EventIdTracker()
        assert tracker.is_duplicate("evt-unknown") is False

    def test_known_is_duplicate(self) -> None:
        """An event ID that has been recorded is a duplicate."""
        tracker = EventIdTracker()
        tracker.record("evt-001")
        assert tracker.is_duplicate("evt-001") is True

    def test_does_not_mutate_state(self) -> None:
        """is_duplicate() does not record the event or change last_seen."""
        tracker = EventIdTracker()
        tracker.record("evt-001")
        assert tracker.last_seen_event_id == "evt-001"

        # Query without recording
        tracker.is_duplicate("evt-002")
        assert tracker.last_seen_event_id == "evt-001"
        assert tracker.tracked_count == 1

    def test_empty_event_id_rejected(self) -> None:
        """Querying an empty event ID raises ValueError."""
        tracker = EventIdTracker()
        with pytest.raises(ValueError, match="event_id"):
            tracker.is_duplicate("")


# ---------------------------------------------------------------------------
# EventIdTracker -- contains()
# ---------------------------------------------------------------------------


class TestEventIdTrackerContains:
    """Tests for the contains() membership check."""

    def test_unknown_returns_false(self) -> None:
        """Unknown event ID returns False."""
        tracker = EventIdTracker()
        assert tracker.contains("evt-nope") is False

    def test_known_returns_true(self) -> None:
        """Recorded event ID returns True."""
        tracker = EventIdTracker()
        tracker.record("evt-yes")
        assert tracker.contains("evt-yes") is True


# ---------------------------------------------------------------------------
# EventIdTracker -- capacity and eviction
# ---------------------------------------------------------------------------


class TestEventIdTrackerCapacity:
    """Tests for capacity limits and FIFO eviction."""

    def test_evicts_oldest_when_full(self) -> None:
        """When capacity is reached, the oldest entry is evicted."""
        config = EventDeduplicationConfig(max_tracked_ids=3)
        tracker = EventIdTracker(config=config)

        tracker.record("evt-001")
        tracker.record("evt-002")
        tracker.record("evt-003")
        assert tracker.tracked_count == 3

        # Adding a 4th should evict evt-001
        tracker.record("evt-004")
        assert tracker.tracked_count == 3
        assert tracker.contains("evt-001") is False
        assert tracker.contains("evt-002") is True
        assert tracker.contains("evt-003") is True
        assert tracker.contains("evt-004") is True

    def test_evicts_multiple_when_needed(self) -> None:
        """Multiple oldest entries evicted when capacity is exceeded."""
        config = EventDeduplicationConfig(max_tracked_ids=2)
        tracker = EventIdTracker(config=config)

        tracker.record("evt-a")
        tracker.record("evt-b")
        tracker.record("evt-c")  # Evicts evt-a

        assert tracker.tracked_count == 2
        assert tracker.contains("evt-a") is False
        assert tracker.contains("evt-b") is True
        assert tracker.contains("evt-c") is True

    def test_duplicate_does_not_increase_count(self) -> None:
        """Recording a duplicate does not increase tracked_count."""
        config = EventDeduplicationConfig(max_tracked_ids=3)
        tracker = EventIdTracker(config=config)

        tracker.record("evt-001")
        tracker.record("evt-002")
        assert tracker.tracked_count == 2

        tracker.record("evt-001")  # Duplicate
        assert tracker.tracked_count == 2

    def test_capacity_of_one(self) -> None:
        """Capacity of 1 means only the latest event is tracked."""
        config = EventDeduplicationConfig(max_tracked_ids=1)
        tracker = EventIdTracker(config=config)

        tracker.record("evt-a")
        assert tracker.contains("evt-a") is True

        tracker.record("evt-b")
        assert tracker.contains("evt-a") is False
        assert tracker.contains("evt-b") is True
        assert tracker.tracked_count == 1


# ---------------------------------------------------------------------------
# EventIdTracker -- clear()
# ---------------------------------------------------------------------------


class TestEventIdTrackerClear:
    """Tests for clear() state reset."""

    def test_clear_returns_removed_count(self) -> None:
        """clear() returns the number of entries removed."""
        tracker = EventIdTracker()
        tracker.record("evt-001")
        tracker.record("evt-002")
        count = tracker.clear()
        assert count == 2

    def test_clear_resets_tracked_count(self) -> None:
        """After clear(), tracked_count is 0."""
        tracker = EventIdTracker()
        tracker.record("evt-001")
        tracker.clear()
        assert tracker.tracked_count == 0

    def test_clear_resets_last_seen(self) -> None:
        """After clear(), last_seen_event_id is None."""
        tracker = EventIdTracker()
        tracker.record("evt-001")
        tracker.clear()
        assert tracker.last_seen_event_id is None

    def test_clear_empty_returns_zero(self) -> None:
        """Clearing an empty tracker returns 0."""
        tracker = EventIdTracker()
        count = tracker.clear()
        assert count == 0

    def test_events_after_clear_not_duplicates(self) -> None:
        """After clear(), previously seen events are not duplicates."""
        tracker = EventIdTracker()
        tracker.record("evt-001")
        tracker.clear()
        verdict = tracker.record("evt-001")
        assert verdict.is_duplicate is False


# ---------------------------------------------------------------------------
# EventIdTracker -- filter_envelope()
# ---------------------------------------------------------------------------


class TestEventIdTrackerFilterEnvelope:
    """Tests for the filter_envelope convenience method."""

    def test_new_envelope_not_duplicate(self) -> None:
        """First envelope with a given event_id is not a duplicate."""
        tracker = EventIdTracker()
        envelope = _make_envelope("evt-100")
        verdict = tracker.filter_envelope(envelope)
        assert verdict.is_duplicate is False
        assert verdict.event_id == "evt-100"

    def test_repeated_envelope_is_duplicate(self) -> None:
        """Re-filtering the same envelope returns is_duplicate=True."""
        tracker = EventIdTracker()
        envelope = _make_envelope("evt-100")
        tracker.filter_envelope(envelope)
        verdict = tracker.filter_envelope(envelope)
        assert verdict.is_duplicate is True

    def test_updates_last_seen(self) -> None:
        """filter_envelope updates last_seen_event_id."""
        tracker = EventIdTracker()
        tracker.filter_envelope(_make_envelope("evt-100"))
        assert tracker.last_seen_event_id == "evt-100"

        tracker.filter_envelope(_make_envelope("evt-200"))
        assert tracker.last_seen_event_id == "evt-200"

    def test_different_envelopes_not_duplicates(self) -> None:
        """Envelopes with different event_ids are not duplicates."""
        tracker = EventIdTracker()
        v1 = tracker.filter_envelope(_make_envelope("evt-a"))
        v2 = tracker.filter_envelope(_make_envelope("evt-b"))
        assert v1.is_duplicate is False
        assert v2.is_duplicate is False


# ---------------------------------------------------------------------------
# EventIdTracker -- tracked_count property
# ---------------------------------------------------------------------------


class TestEventIdTrackerTrackedCount:
    """Tests for the tracked_count property."""

    def test_initially_zero(self) -> None:
        """Before any recordings, tracked_count is 0."""
        tracker = EventIdTracker()
        assert tracker.tracked_count == 0

    def test_increments_with_new_events(self) -> None:
        """tracked_count increments for each new event ID."""
        tracker = EventIdTracker()
        tracker.record("evt-1")
        assert tracker.tracked_count == 1
        tracker.record("evt-2")
        assert tracker.tracked_count == 2

    def test_unchanged_on_duplicates(self) -> None:
        """tracked_count does not change on duplicate recordings."""
        tracker = EventIdTracker()
        tracker.record("evt-1")
        tracker.record("evt-1")
        assert tracker.tracked_count == 1


# ---------------------------------------------------------------------------
# EventIdTracker -- thread safety
# ---------------------------------------------------------------------------


class TestEventIdTrackerThreadSafety:
    """Tests for concurrent access safety."""

    def test_concurrent_recordings(self) -> None:
        """Multiple threads can record concurrently without errors."""
        tracker = EventIdTracker()
        errors: list[Exception] = []

        def worker(prefix: str, count: int) -> None:
            try:
                for i in range(count):
                    tracker.record(f"{prefix}-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(f"t{t}", 100))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # All unique IDs should be tracked (up to capacity)
        # 5 threads * 100 events = 500 unique IDs
        # Default capacity is 10000, so all should fit
        assert tracker.tracked_count == 500

    def test_concurrent_record_and_query(self) -> None:
        """Recording and querying concurrently does not raise."""
        tracker = EventIdTracker()
        errors: list[Exception] = []

        def recorder() -> None:
            try:
                for i in range(200):
                    tracker.record(f"rec-{i}")
            except Exception as exc:
                errors.append(exc)

        def querier() -> None:
            try:
                for i in range(200):
                    tracker.is_duplicate(f"rec-{i}")
                    _ = tracker.last_seen_event_id
                    _ = tracker.tracked_count
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=recorder)
        t2 = threading.Thread(target=querier)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []


# ---------------------------------------------------------------------------
# EventIdTracker -- repr
# ---------------------------------------------------------------------------


class TestEventIdTrackerRepr:
    """Tests for __repr__."""

    def test_repr_includes_count(self) -> None:
        """repr shows tracked count."""
        tracker = EventIdTracker()
        tracker.record("evt-1")
        tracker.record("evt-2")
        result = repr(tracker)
        assert "2" in result
        assert "EventIdTracker" in result

    def test_repr_empty(self) -> None:
        """repr works for empty tracker."""
        tracker = EventIdTracker()
        result = repr(tracker)
        assert "0" in result


# ---------------------------------------------------------------------------
# EventIdTracker -- default config
# ---------------------------------------------------------------------------


class TestEventIdTrackerDefaultConfig:
    """Tests for default configuration handling."""

    def test_default_config_when_none(self) -> None:
        """When no config is provided, a default is used."""
        tracker = EventIdTracker()
        # Should accept many events without eviction
        for i in range(100):
            tracker.record(f"evt-{i}")
        assert tracker.tracked_count == 100

    def test_custom_config_respected(self) -> None:
        """Custom config controls capacity."""
        config = EventDeduplicationConfig(max_tracked_ids=5)
        tracker = EventIdTracker(config=config)
        for i in range(10):
            tracker.record(f"evt-{i}")
        assert tracker.tracked_count == 5
