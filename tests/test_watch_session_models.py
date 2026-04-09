"""Tests for watch-mode session metadata models."""

from datetime import datetime, timezone

import pytest

from jules_daemon.wiki.watch_session_models import (
    StreamRecord,
    StreamStatus,
    WatcherRecord,
    WatcherStatus,
    WatchSessionSnapshot,
)


# ---------------------------------------------------------------------------
# WatcherStatus enum
# ---------------------------------------------------------------------------


class TestWatcherStatus:
    def test_values(self) -> None:
        assert WatcherStatus.ACTIVE.value == "active"
        assert WatcherStatus.DISCONNECTED.value == "disconnected"
        assert WatcherStatus.COMPLETED.value == "completed"

    def test_roundtrip_from_string(self) -> None:
        assert WatcherStatus("active") == WatcherStatus.ACTIVE
        assert WatcherStatus("disconnected") == WatcherStatus.DISCONNECTED
        assert WatcherStatus("completed") == WatcherStatus.COMPLETED


# ---------------------------------------------------------------------------
# StreamStatus enum
# ---------------------------------------------------------------------------


class TestStreamStatus:
    def test_values(self) -> None:
        assert StreamStatus.LIVE.value == "live"
        assert StreamStatus.ENDED.value == "ended"
        assert StreamStatus.IDLE.value == "idle"


# ---------------------------------------------------------------------------
# WatcherRecord
# ---------------------------------------------------------------------------


class TestWatcherRecord:
    def test_creation(self) -> None:
        now = datetime.now(timezone.utc)
        record = WatcherRecord(
            watcher_id="w-abc123",
            client_id="cli-001",
            job_id="job-xyz",
            subscriber_id="sub-def456",
            connected_at=now,
        )
        assert record.watcher_id == "w-abc123"
        assert record.client_id == "cli-001"
        assert record.job_id == "job-xyz"
        assert record.subscriber_id == "sub-def456"
        assert record.connected_at == now
        assert record.status == WatcherStatus.ACTIVE
        assert record.last_sequence == 0
        assert record.lines_received == 0

    def test_frozen(self) -> None:
        record = WatcherRecord(
            watcher_id="w-1",
            client_id="c-1",
            job_id="j-1",
            subscriber_id="s-1",
            connected_at=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            record.status = WatcherStatus.DISCONNECTED  # type: ignore[misc]

    def test_empty_watcher_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="watcher_id must not be empty"):
            WatcherRecord(
                watcher_id="",
                client_id="c-1",
                job_id="j-1",
                subscriber_id="s-1",
                connected_at=datetime.now(timezone.utc),
            )

    def test_empty_client_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="client_id must not be empty"):
            WatcherRecord(
                watcher_id="w-1",
                client_id="",
                job_id="j-1",
                subscriber_id="s-1",
                connected_at=datetime.now(timezone.utc),
            )

    def test_empty_job_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="job_id must not be empty"):
            WatcherRecord(
                watcher_id="w-1",
                client_id="c-1",
                job_id="",
                subscriber_id="s-1",
                connected_at=datetime.now(timezone.utc),
            )

    def test_empty_subscriber_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="subscriber_id must not be empty"):
            WatcherRecord(
                watcher_id="w-1",
                client_id="c-1",
                job_id="j-1",
                subscriber_id="",
                connected_at=datetime.now(timezone.utc),
            )

    def test_negative_last_sequence_rejected(self) -> None:
        with pytest.raises(ValueError, match="last_sequence must not be negative"):
            WatcherRecord(
                watcher_id="w-1",
                client_id="c-1",
                job_id="j-1",
                subscriber_id="s-1",
                connected_at=datetime.now(timezone.utc),
                last_sequence=-1,
            )

    def test_negative_lines_received_rejected(self) -> None:
        with pytest.raises(ValueError, match="lines_received must not be negative"):
            WatcherRecord(
                watcher_id="w-1",
                client_id="c-1",
                job_id="j-1",
                subscriber_id="s-1",
                connected_at=datetime.now(timezone.utc),
                lines_received=-1,
            )

    def test_with_status_returns_new_instance(self) -> None:
        record = WatcherRecord(
            watcher_id="w-1",
            client_id="c-1",
            job_id="j-1",
            subscriber_id="s-1",
            connected_at=datetime.now(timezone.utc),
        )
        updated = record.with_status(WatcherStatus.DISCONNECTED)
        assert updated is not record
        assert updated.status == WatcherStatus.DISCONNECTED
        assert record.status == WatcherStatus.ACTIVE  # original unchanged

    def test_with_progress_returns_new_instance(self) -> None:
        record = WatcherRecord(
            watcher_id="w-1",
            client_id="c-1",
            job_id="j-1",
            subscriber_id="s-1",
            connected_at=datetime.now(timezone.utc),
        )
        updated = record.with_progress(last_sequence=42, lines_received=100)
        assert updated is not record
        assert updated.last_sequence == 42
        assert updated.lines_received == 100
        assert record.last_sequence == 0  # original unchanged


# ---------------------------------------------------------------------------
# StreamRecord
# ---------------------------------------------------------------------------


class TestStreamRecord:
    def test_creation_defaults(self) -> None:
        record = StreamRecord(job_id="job-xyz")
        assert record.job_id == "job-xyz"
        assert record.status == StreamStatus.IDLE
        assert record.buffer_size == 0
        assert record.total_lines_published == 0
        assert record.subscriber_count == 0
        assert record.last_publish_at is None

    def test_frozen(self) -> None:
        record = StreamRecord(job_id="job-1")
        with pytest.raises(AttributeError):
            record.status = StreamStatus.LIVE  # type: ignore[misc]

    def test_empty_job_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="job_id must not be empty"):
            StreamRecord(job_id="")

    def test_negative_buffer_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="buffer_size must not be negative"):
            StreamRecord(job_id="j-1", buffer_size=-1)

    def test_negative_total_lines_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_lines_published must not be negative"):
            StreamRecord(job_id="j-1", total_lines_published=-1)

    def test_negative_subscriber_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="subscriber_count must not be negative"):
            StreamRecord(job_id="j-1", subscriber_count=-1)

    def test_with_status_returns_new_instance(self) -> None:
        record = StreamRecord(job_id="j-1")
        updated = record.with_status(StreamStatus.LIVE)
        assert updated is not record
        assert updated.status == StreamStatus.LIVE
        assert record.status == StreamStatus.IDLE

    def test_with_publish_update(self) -> None:
        now = datetime.now(timezone.utc)
        record = StreamRecord(job_id="j-1")
        updated = record.with_publish_update(
            total_lines=50,
            buffer_size=25,
            last_publish_at=now,
        )
        assert updated.total_lines_published == 50
        assert updated.buffer_size == 25
        assert updated.last_publish_at == now

    def test_with_subscriber_count(self) -> None:
        record = StreamRecord(job_id="j-1")
        updated = record.with_subscriber_count(3)
        assert updated.subscriber_count == 3
        assert record.subscriber_count == 0


# ---------------------------------------------------------------------------
# WatchSessionSnapshot
# ---------------------------------------------------------------------------


class TestWatchSessionSnapshot:
    def test_creation_defaults(self) -> None:
        snap = WatchSessionSnapshot()
        assert snap.watchers == ()
        assert snap.streams == ()
        assert snap.snapshot_at is not None
        assert snap.daemon_pid is None

    def test_frozen(self) -> None:
        snap = WatchSessionSnapshot()
        with pytest.raises(AttributeError):
            snap.daemon_pid = 123  # type: ignore[misc]

    def test_with_watchers(self) -> None:
        now = datetime.now(timezone.utc)
        w1 = WatcherRecord(
            watcher_id="w-1",
            client_id="c-1",
            job_id="j-1",
            subscriber_id="s-1",
            connected_at=now,
        )
        snap = WatchSessionSnapshot()
        updated = snap.with_watcher_added(w1)
        assert len(updated.watchers) == 1
        assert updated.watchers[0].watcher_id == "w-1"
        assert len(snap.watchers) == 0  # original unchanged

    def test_with_watcher_removed(self) -> None:
        now = datetime.now(timezone.utc)
        w1 = WatcherRecord(
            watcher_id="w-1",
            client_id="c-1",
            job_id="j-1",
            subscriber_id="s-1",
            connected_at=now,
        )
        w2 = WatcherRecord(
            watcher_id="w-2",
            client_id="c-2",
            job_id="j-1",
            subscriber_id="s-2",
            connected_at=now,
        )
        snap = WatchSessionSnapshot(watchers=(w1, w2))
        updated = snap.with_watcher_removed("w-1")
        assert len(updated.watchers) == 1
        assert updated.watchers[0].watcher_id == "w-2"

    def test_with_watcher_removed_nonexistent_is_noop(self) -> None:
        snap = WatchSessionSnapshot()
        updated = snap.with_watcher_removed("nonexistent")
        assert len(updated.watchers) == 0

    def test_with_stream_added(self) -> None:
        s1 = StreamRecord(job_id="j-1", status=StreamStatus.LIVE)
        snap = WatchSessionSnapshot()
        updated = snap.with_stream_added(s1)
        assert len(updated.streams) == 1
        assert updated.streams[0].job_id == "j-1"

    def test_with_stream_removed(self) -> None:
        s1 = StreamRecord(job_id="j-1")
        s2 = StreamRecord(job_id="j-2")
        snap = WatchSessionSnapshot(streams=(s1, s2))
        updated = snap.with_stream_removed("j-1")
        assert len(updated.streams) == 1
        assert updated.streams[0].job_id == "j-2"

    def test_with_stream_updated(self) -> None:
        s1 = StreamRecord(job_id="j-1", status=StreamStatus.IDLE)
        snap = WatchSessionSnapshot(streams=(s1,))
        new_s1 = s1.with_status(StreamStatus.LIVE)
        updated = snap.with_stream_updated(new_s1)
        assert updated.streams[0].status == StreamStatus.LIVE

    def test_with_stream_updated_appends_if_not_found(self) -> None:
        snap = WatchSessionSnapshot()
        s1 = StreamRecord(job_id="j-new", status=StreamStatus.LIVE)
        updated = snap.with_stream_updated(s1)
        assert len(updated.streams) == 1
        assert updated.streams[0].job_id == "j-new"

    def test_active_watcher_count(self) -> None:
        now = datetime.now(timezone.utc)
        w1 = WatcherRecord(
            watcher_id="w-1", client_id="c-1", job_id="j-1",
            subscriber_id="s-1", connected_at=now, status=WatcherStatus.ACTIVE,
        )
        w2 = WatcherRecord(
            watcher_id="w-2", client_id="c-2", job_id="j-1",
            subscriber_id="s-2", connected_at=now, status=WatcherStatus.DISCONNECTED,
        )
        w3 = WatcherRecord(
            watcher_id="w-3", client_id="c-3", job_id="j-1",
            subscriber_id="s-3", connected_at=now, status=WatcherStatus.ACTIVE,
        )
        snap = WatchSessionSnapshot(watchers=(w1, w2, w3))
        assert snap.active_watcher_count == 2

    def test_live_stream_count(self) -> None:
        s1 = StreamRecord(job_id="j-1", status=StreamStatus.LIVE)
        s2 = StreamRecord(job_id="j-2", status=StreamStatus.ENDED)
        s3 = StreamRecord(job_id="j-3", status=StreamStatus.LIVE)
        snap = WatchSessionSnapshot(streams=(s1, s2, s3))
        assert snap.live_stream_count == 2

    def test_find_watcher(self) -> None:
        now = datetime.now(timezone.utc)
        w1 = WatcherRecord(
            watcher_id="w-target", client_id="c-1", job_id="j-1",
            subscriber_id="s-1", connected_at=now,
        )
        snap = WatchSessionSnapshot(watchers=(w1,))
        assert snap.find_watcher("w-target") is w1
        assert snap.find_watcher("nonexistent") is None

    def test_find_stream(self) -> None:
        s1 = StreamRecord(job_id="j-target")
        snap = WatchSessionSnapshot(streams=(s1,))
        assert snap.find_stream("j-target") is s1
        assert snap.find_stream("nonexistent") is None

    def test_watchers_for_job(self) -> None:
        now = datetime.now(timezone.utc)
        w1 = WatcherRecord(
            watcher_id="w-1", client_id="c-1", job_id="j-1",
            subscriber_id="s-1", connected_at=now,
        )
        w2 = WatcherRecord(
            watcher_id="w-2", client_id="c-2", job_id="j-2",
            subscriber_id="s-2", connected_at=now,
        )
        w3 = WatcherRecord(
            watcher_id="w-3", client_id="c-3", job_id="j-1",
            subscriber_id="s-3", connected_at=now,
        )
        snap = WatchSessionSnapshot(watchers=(w1, w2, w3))
        job1_watchers = snap.watchers_for_job("j-1")
        assert len(job1_watchers) == 2
        assert all(w.job_id == "j-1" for w in job1_watchers)
