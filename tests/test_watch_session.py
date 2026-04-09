"""Tests for watch-mode session wiki persistence."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki import watch_session
from jules_daemon.wiki.frontmatter import parse
from jules_daemon.wiki.watch_session_models import (
    StreamRecord,
    StreamStatus,
    WatcherRecord,
    WatcherStatus,
    WatchSessionSnapshot,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_watcher(
    watcher_id: str = "w-1",
    client_id: str = "c-1",
    job_id: str = "j-1",
    subscriber_id: str = "s-1",
    status: WatcherStatus = WatcherStatus.ACTIVE,
    last_sequence: int = 0,
    lines_received: int = 0,
) -> WatcherRecord:
    return WatcherRecord(
        watcher_id=watcher_id,
        client_id=client_id,
        job_id=job_id,
        subscriber_id=subscriber_id,
        connected_at=datetime.now(timezone.utc),
        status=status,
        last_sequence=last_sequence,
        lines_received=lines_received,
    )


def _make_stream(
    job_id: str = "j-1",
    status: StreamStatus = StreamStatus.LIVE,
    buffer_size: int = 100,
    total_lines: int = 50,
    subscriber_count: int = 2,
) -> StreamRecord:
    return StreamRecord(
        job_id=job_id,
        status=status,
        buffer_size=buffer_size,
        total_lines_published=total_lines,
        subscriber_count=subscriber_count,
        last_publish_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


class TestWrite:
    def test_creates_file_and_directories(self, wiki_root: Path) -> None:
        snap = WatchSessionSnapshot()
        result_path = watch_session.write(wiki_root, snap)

        assert result_path.exists()
        assert result_path.name == "watch-sessions.md"
        assert "pages/daemon" in str(result_path)

    def test_empty_snapshot_persists(self, wiki_root: Path) -> None:
        snap = WatchSessionSnapshot()
        watch_session.write(wiki_root, snap)

        loaded = watch_session.read(wiki_root)
        assert loaded is not None
        assert loaded.watchers == ()
        assert loaded.streams == ()

    def test_full_snapshot_roundtrip(self, wiki_root: Path) -> None:
        w1 = _make_watcher(
            watcher_id="w-abc",
            client_id="cli-001",
            job_id="job-xyz",
            subscriber_id="sub-def",
            last_sequence=42,
            lines_received=100,
        )
        w2 = _make_watcher(
            watcher_id="w-ghi",
            client_id="cli-002",
            job_id="job-xyz",
            subscriber_id="sub-jkl",
            status=WatcherStatus.DISCONNECTED,
            last_sequence=30,
            lines_received=80,
        )
        s1 = _make_stream(
            job_id="job-xyz",
            status=StreamStatus.LIVE,
            buffer_size=200,
            total_lines=150,
            subscriber_count=2,
        )

        snap = WatchSessionSnapshot(
            watchers=(w1, w2),
            streams=(s1,),
            daemon_pid=9876,
        )
        watch_session.write(wiki_root, snap)
        loaded = watch_session.read(wiki_root)

        assert loaded is not None
        assert len(loaded.watchers) == 2
        assert loaded.watchers[0].watcher_id == "w-abc"
        assert loaded.watchers[0].client_id == "cli-001"
        assert loaded.watchers[0].job_id == "job-xyz"
        assert loaded.watchers[0].subscriber_id == "sub-def"
        assert loaded.watchers[0].status == WatcherStatus.ACTIVE
        assert loaded.watchers[0].last_sequence == 42
        assert loaded.watchers[0].lines_received == 100
        assert loaded.watchers[0].connected_at is not None

        assert loaded.watchers[1].watcher_id == "w-ghi"
        assert loaded.watchers[1].status == WatcherStatus.DISCONNECTED
        assert loaded.watchers[1].last_sequence == 30

        assert len(loaded.streams) == 1
        assert loaded.streams[0].job_id == "job-xyz"
        assert loaded.streams[0].status == StreamStatus.LIVE
        assert loaded.streams[0].buffer_size == 200
        assert loaded.streams[0].total_lines_published == 150
        assert loaded.streams[0].subscriber_count == 2
        assert loaded.streams[0].last_publish_at is not None

        assert loaded.daemon_pid == 9876

    def test_file_is_valid_wiki_format(self, wiki_root: Path) -> None:
        snap = WatchSessionSnapshot()
        path = watch_session.write(wiki_root, snap)

        raw = path.read_text(encoding="utf-8")
        doc = parse(raw)
        assert "tags" in doc.frontmatter
        assert "daemon" in doc.frontmatter["tags"]
        assert "watch-session" in doc.frontmatter["tags"]
        assert doc.frontmatter["type"] == "watch-session-state"
        assert "# Watch Sessions" in doc.body

    def test_overwrites_existing(self, wiki_root: Path) -> None:
        snap1 = WatchSessionSnapshot()
        watch_session.write(wiki_root, snap1)

        w1 = _make_watcher()
        snap2 = WatchSessionSnapshot(watchers=(w1,))
        watch_session.write(wiki_root, snap2)

        loaded = watch_session.read(wiki_root)
        assert loaded is not None
        assert len(loaded.watchers) == 1


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


class TestRead:
    def test_returns_none_when_no_file(self, wiki_root: Path) -> None:
        result = watch_session.read(wiki_root)
        assert result is None

    def test_reads_back_written_state(self, wiki_root: Path) -> None:
        snap = WatchSessionSnapshot(daemon_pid=42)
        watch_session.write(wiki_root, snap)
        loaded = watch_session.read(wiki_root)
        assert loaded is not None
        assert loaded.daemon_pid == 42


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_raises_when_no_file(self, wiki_root: Path) -> None:
        snap = WatchSessionSnapshot()
        with pytest.raises(FileNotFoundError, match="No watch-sessions record"):
            watch_session.update(wiki_root, snap)

    def test_updates_existing_record(self, wiki_root: Path) -> None:
        snap = WatchSessionSnapshot()
        watch_session.write(wiki_root, snap)

        w1 = _make_watcher()
        updated = snap.with_watcher_added(w1)
        watch_session.update(wiki_root, updated)

        loaded = watch_session.read(wiki_root)
        assert loaded is not None
        assert len(loaded.watchers) == 1

    def test_watcher_lifecycle(self, wiki_root: Path) -> None:
        """Test adding, updating, and removing watchers through wiki persistence."""
        # Start with empty
        snap = WatchSessionSnapshot(daemon_pid=1000)
        watch_session.write(wiki_root, snap)

        # Add first watcher
        w1 = _make_watcher(watcher_id="w-1", job_id="job-a")
        snap = snap.with_watcher_added(w1)
        watch_session.update(wiki_root, snap)
        loaded = watch_session.read(wiki_root)
        assert loaded is not None
        assert len(loaded.watchers) == 1

        # Add second watcher
        w2 = _make_watcher(watcher_id="w-2", client_id="c-2", job_id="job-a", subscriber_id="s-2")
        snap = snap.with_watcher_added(w2)
        watch_session.update(wiki_root, snap)
        loaded = watch_session.read(wiki_root)
        assert loaded is not None
        assert len(loaded.watchers) == 2

        # Remove first watcher
        snap = snap.with_watcher_removed("w-1")
        watch_session.update(wiki_root, snap)
        loaded = watch_session.read(wiki_root)
        assert loaded is not None
        assert len(loaded.watchers) == 1
        assert loaded.watchers[0].watcher_id == "w-2"


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_to_empty(self, wiki_root: Path) -> None:
        w1 = _make_watcher()
        s1 = _make_stream()
        snap = WatchSessionSnapshot(watchers=(w1,), streams=(s1,), daemon_pid=42)
        watch_session.write(wiki_root, snap)

        watch_session.clear(wiki_root)
        loaded = watch_session.read(wiki_root)
        assert loaded is not None
        assert loaded.watchers == ()
        assert loaded.streams == ()
        assert loaded.daemon_pid is None

    def test_clear_creates_file_if_missing(self, wiki_root: Path) -> None:
        watch_session.clear(wiki_root)

        loaded = watch_session.read(wiki_root)
        assert loaded is not None
        assert loaded.watchers == ()

    def test_file_persists_after_clear(self, wiki_root: Path) -> None:
        watch_session.clear(wiki_root)
        assert watch_session.exists(wiki_root)


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestExists:
    def test_false_when_no_file(self, wiki_root: Path) -> None:
        assert watch_session.exists(wiki_root) is False

    def test_true_after_write(self, wiki_root: Path) -> None:
        watch_session.write(wiki_root, WatchSessionSnapshot())
        assert watch_session.exists(wiki_root) is True


# ---------------------------------------------------------------------------
# file_path
# ---------------------------------------------------------------------------


class TestFilePath:
    def test_returns_expected_path(self, wiki_root: Path) -> None:
        path = watch_session.file_path(wiki_root)
        assert path == wiki_root / "pages" / "daemon" / "watch-sessions.md"


# ---------------------------------------------------------------------------
# Crash recovery scenario
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    """Verify that watch session state survives a simulated daemon crash."""

    def test_recovery_with_active_watchers(self, wiki_root: Path) -> None:
        """Daemon crashes while clients are watching; new daemon reads wiki state."""
        w1 = _make_watcher(
            watcher_id="w-active-1",
            client_id="cli-a",
            job_id="job-run-42",
            subscriber_id="sub-111",
            last_sequence=200,
            lines_received=195,
        )
        w2 = _make_watcher(
            watcher_id="w-active-2",
            client_id="cli-b",
            job_id="job-run-42",
            subscriber_id="sub-222",
            last_sequence=180,
            lines_received=175,
        )
        s1 = _make_stream(
            job_id="job-run-42",
            status=StreamStatus.LIVE,
            buffer_size=500,
            total_lines=200,
            subscriber_count=2,
        )

        snap = WatchSessionSnapshot(
            watchers=(w1, w2),
            streams=(s1,),
            daemon_pid=12345,
        )
        watch_session.write(wiki_root, snap)

        # Simulate crash: new daemon reads state
        recovered = watch_session.read(wiki_root)

        assert recovered is not None
        assert len(recovered.watchers) == 2
        assert recovered.watchers[0].watcher_id == "w-active-1"
        assert recovered.watchers[0].last_sequence == 200
        assert recovered.watchers[0].lines_received == 195
        assert recovered.watchers[1].watcher_id == "w-active-2"

        assert len(recovered.streams) == 1
        assert recovered.streams[0].job_id == "job-run-42"
        assert recovered.streams[0].status == StreamStatus.LIVE
        assert recovered.streams[0].total_lines_published == 200

        assert recovered.daemon_pid == 12345


# ---------------------------------------------------------------------------
# Markdown body content
# ---------------------------------------------------------------------------


class TestMarkdownBody:
    def test_empty_snapshot_body(self, wiki_root: Path) -> None:
        snap = WatchSessionSnapshot()
        path = watch_session.write(wiki_root, snap)
        raw = path.read_text(encoding="utf-8")
        doc = parse(raw)
        assert "No active watchers" in doc.body

    def test_active_watchers_listed_in_body(self, wiki_root: Path) -> None:
        w1 = _make_watcher(
            watcher_id="w-1",
            client_id="cli-001",
            job_id="job-xyz",
            subscriber_id="sub-1",
        )
        s1 = _make_stream(job_id="job-xyz")
        snap = WatchSessionSnapshot(watchers=(w1,), streams=(s1,))
        path = watch_session.write(wiki_root, snap)
        raw = path.read_text(encoding="utf-8")
        doc = parse(raw)

        assert "## Active Watchers" in doc.body
        assert "cli-001" in doc.body
        assert "job-xyz" in doc.body

    def test_streams_listed_in_body(self, wiki_root: Path) -> None:
        s1 = _make_stream(job_id="job-abc", status=StreamStatus.LIVE, total_lines=42)
        snap = WatchSessionSnapshot(streams=(s1,))
        path = watch_session.write(wiki_root, snap)
        raw = path.read_text(encoding="utf-8")
        doc = parse(raw)

        assert "## Stream State" in doc.body
        assert "job-abc" in doc.body
        assert "live" in doc.body


# ---------------------------------------------------------------------------
# Multiple watchers on different jobs
# ---------------------------------------------------------------------------


class TestMultiJobWatchers:
    def test_watchers_on_different_jobs_roundtrip(self, wiki_root: Path) -> None:
        w1 = _make_watcher(watcher_id="w-1", job_id="job-a", subscriber_id="s-1")
        w2 = _make_watcher(
            watcher_id="w-2", client_id="c-2", job_id="job-b", subscriber_id="s-2"
        )
        s1 = _make_stream(job_id="job-a")
        s2 = _make_stream(job_id="job-b", status=StreamStatus.ENDED)

        snap = WatchSessionSnapshot(watchers=(w1, w2), streams=(s1, s2))
        watch_session.write(wiki_root, snap)
        loaded = watch_session.read(wiki_root)

        assert loaded is not None
        assert len(loaded.watchers) == 2
        assert loaded.watchers[0].job_id == "job-a"
        assert loaded.watchers[1].job_id == "job-b"
        assert len(loaded.streams) == 2
        assert loaded.streams[0].status == StreamStatus.LIVE
        assert loaded.streams[1].status == StreamStatus.ENDED
