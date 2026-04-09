"""Persist watch-mode session metadata to a wiki markdown file.

The watch-session record is a single wiki file with YAML frontmatter
containing all structured state, and a markdown body with human-readable
summaries of active watchers and stream states. This module provides the
sole persistence API for watch-mode session metadata.

Wiki file location: {wiki_root}/pages/daemon/watch-sessions.md

The daemon writes this file periodically to persist:
- Active watcher subscriptions (client -> job mappings)
- Stream state per job (buffer size, publish counts, subscriber counts)
- Daemon PID for crash recovery correlation

On crash recovery, the new daemon reads this file to discover which
watchers were active and which streams were live, enabling it to clean
up stale subscriptions and re-establish monitoring.

Usage:
    from jules_daemon.wiki import watch_session
    from jules_daemon.wiki.watch_session_models import WatchSessionSnapshot

    snap = WatchSessionSnapshot(watchers=(w1,), streams=(s1,), daemon_pid=os.getpid())
    watch_session.write(wiki_root, snap)

    loaded = watch_session.read(wiki_root)
    if loaded and loaded.active_watcher_count > 0:
        ...
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument
from jules_daemon.wiki.watch_session_models import (
    StreamRecord,
    StreamStatus,
    WatcherRecord,
    WatcherStatus,
    WatchSessionSnapshot,
)

__all__ = [
    "clear",
    "exists",
    "file_path",
    "read",
    "update",
    "write",
]

logger = logging.getLogger(__name__)

_WATCH_SESSION_FILENAME = "watch-sessions.md"
_DAEMON_DIR = "pages/daemon"
_WIKI_TAGS = ["daemon", "state", "watch-session"]
_WIKI_TYPE = "watch-session-state"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _wiki_file_path(wiki_root: Path) -> Path:
    """Resolve the absolute path to the watch-sessions wiki file."""
    return wiki_root / _DAEMON_DIR / _WATCH_SESSION_FILENAME


def _ensure_directory(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _datetime_to_iso(dt: datetime | None) -> str | None:
    """Convert datetime to ISO 8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_datetime(value: str | None) -> datetime | None:
    """Parse ISO 8601 string to timezone-aware datetime, or None.

    If the parsed datetime is naive (no timezone info), UTC is assumed.
    """
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Watcher serialization
# ---------------------------------------------------------------------------


def _watcher_to_dict(watcher: WatcherRecord) -> dict[str, Any]:
    """Serialize a WatcherRecord to a YAML-compatible dict."""
    return {
        "watcher_id": watcher.watcher_id,
        "client_id": watcher.client_id,
        "job_id": watcher.job_id,
        "subscriber_id": watcher.subscriber_id,
        "connected_at": _datetime_to_iso(watcher.connected_at),
        "status": watcher.status.value,
        "last_sequence": watcher.last_sequence,
        "lines_received": watcher.lines_received,
    }


def _dict_to_watcher(data: dict[str, Any]) -> WatcherRecord:
    """Deserialize a WatcherRecord from a YAML dict."""
    return WatcherRecord(
        watcher_id=data["watcher_id"],
        client_id=data["client_id"],
        job_id=data["job_id"],
        subscriber_id=data["subscriber_id"],
        connected_at=_iso_to_datetime(data["connected_at"])
        or datetime.now(timezone.utc),
        status=WatcherStatus(data.get("status", "active")),
        last_sequence=int(data.get("last_sequence", 0)),
        lines_received=int(data.get("lines_received", 0)),
    )


# ---------------------------------------------------------------------------
# Stream serialization
# ---------------------------------------------------------------------------


def _stream_to_dict(stream: StreamRecord) -> dict[str, Any]:
    """Serialize a StreamRecord to a YAML-compatible dict."""
    return {
        "job_id": stream.job_id,
        "status": stream.status.value,
        "buffer_size": stream.buffer_size,
        "total_lines_published": stream.total_lines_published,
        "subscriber_count": stream.subscriber_count,
        "last_publish_at": _datetime_to_iso(stream.last_publish_at),
    }


def _dict_to_stream(data: dict[str, Any]) -> StreamRecord:
    """Deserialize a StreamRecord from a YAML dict."""
    return StreamRecord(
        job_id=data["job_id"],
        status=StreamStatus(data.get("status", "idle")),
        buffer_size=int(data.get("buffer_size", 0)),
        total_lines_published=int(data.get("total_lines_published", 0)),
        subscriber_count=int(data.get("subscriber_count", 0)),
        last_publish_at=_iso_to_datetime(data.get("last_publish_at")),
    )


# ---------------------------------------------------------------------------
# Snapshot serialization
# ---------------------------------------------------------------------------


def _snapshot_to_frontmatter(snap: WatchSessionSnapshot) -> dict[str, Any]:
    """Convert a WatchSessionSnapshot to a YAML-serializable frontmatter dict."""
    return {
        "tags": list(_WIKI_TAGS),
        "type": _WIKI_TYPE,
        "snapshot_at": _datetime_to_iso(snap.snapshot_at),
        "daemon_pid": snap.daemon_pid,
        "watchers": [_watcher_to_dict(w) for w in snap.watchers],
        "streams": [_stream_to_dict(s) for s in snap.streams],
    }


def _frontmatter_to_snapshot(fm: dict[str, Any]) -> WatchSessionSnapshot:
    """Reconstruct a WatchSessionSnapshot from a parsed frontmatter dict."""
    raw_watchers = fm.get("watchers")
    watchers: tuple[WatcherRecord, ...] = ()
    if raw_watchers and isinstance(raw_watchers, list):
        watchers = tuple(_dict_to_watcher(w) for w in raw_watchers)

    raw_streams = fm.get("streams")
    streams: tuple[StreamRecord, ...] = ()
    if raw_streams and isinstance(raw_streams, list):
        streams = tuple(_dict_to_stream(s) for s in raw_streams)

    snapshot_at = _iso_to_datetime(fm.get("snapshot_at"))
    if snapshot_at is None:
        snapshot_at = datetime.now(timezone.utc)

    return WatchSessionSnapshot(
        watchers=watchers,
        streams=streams,
        snapshot_at=snapshot_at,
        daemon_pid=fm.get("daemon_pid"),
    )


# ---------------------------------------------------------------------------
# Markdown body generation
# ---------------------------------------------------------------------------


def _build_body(snap: WatchSessionSnapshot) -> str:
    """Generate the human-readable markdown body for the watch-sessions file."""
    lines = [
        "# Watch Sessions",
        "",
        f"*Watch-mode session metadata -- {snap.active_watcher_count} active "
        f"watcher(s), {snap.live_stream_count} live stream(s)*",
        "",
    ]

    if not snap.watchers and not snap.streams:
        lines.append("No active watchers or streams. The daemon is idle.")
        return "\n".join(lines)

    # Active Watchers section
    if snap.watchers:
        lines.extend([
            "## Active Watchers",
            "",
        ])
        for watcher in snap.watchers:
            lines.extend([
                f"### Watcher: {watcher.watcher_id}",
                "",
                f"- **Client:** {watcher.client_id}",
                f"- **Job:** {watcher.job_id}",
                f"- **Subscriber:** {watcher.subscriber_id}",
                f"- **Status:** {watcher.status.value}",
                f"- **Connected:** {_datetime_to_iso(watcher.connected_at)}",
                f"- **Last Sequence:** {watcher.last_sequence}",
                f"- **Lines Received:** {watcher.lines_received}",
                "",
            ])
    else:
        lines.extend([
            "No active watchers.",
            "",
        ])

    # Stream State section
    if snap.streams:
        lines.extend([
            "## Stream State",
            "",
        ])
        for stream in snap.streams:
            lines.extend([
                f"### Stream: {stream.job_id}",
                "",
                f"- **Status:** {stream.status.value}",
                f"- **Buffer Size:** {stream.buffer_size}",
                f"- **Total Lines Published:** {stream.total_lines_published}",
                f"- **Subscribers:** {stream.subscriber_count}",
            ])
            if stream.last_publish_at:
                lines.append(
                    f"- **Last Publish:** {_datetime_to_iso(stream.last_publish_at)}"
                )
            lines.append("")
    else:
        lines.extend([
            "No active streams.",
            "",
        ])

    # Snapshot metadata
    lines.extend([
        "## Snapshot Metadata",
        "",
        f"- **Snapshot At:** {_datetime_to_iso(snap.snapshot_at)}",
    ])
    if snap.daemon_pid is not None:
        lines.append(f"- **Daemon PID:** {snap.daemon_pid}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write(wiki_root: Path, snap: WatchSessionSnapshot) -> Path:
    """Write a watch-session snapshot to the wiki file.

    Creates the file and parent directories if needed. Overwrites any
    existing content (each write is a complete snapshot).

    Args:
        wiki_root: Path to the wiki root directory.
        snap: The watch session snapshot to persist.

    Returns:
        Path to the written wiki file.
    """
    target = _wiki_file_path(wiki_root)
    _ensure_directory(target)

    doc = WikiDocument(
        frontmatter=_snapshot_to_frontmatter(snap),
        body=_build_body(snap),
    )
    content = frontmatter.serialize(doc)

    # Atomic write: write to temp file then rename
    tmp_path = target.with_suffix(".md.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(str(tmp_path), str(target))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info(
        "Wrote watch-session snapshot to %s: %d watchers, %d streams",
        target,
        len(snap.watchers),
        len(snap.streams),
    )
    return target


def read(wiki_root: Path) -> WatchSessionSnapshot | None:
    """Read the watch-session snapshot from the wiki file.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        The deserialized WatchSessionSnapshot, or None if the file
        does not exist.
    """
    target = _wiki_file_path(wiki_root)
    if not target.exists():
        return None

    raw = target.read_text(encoding="utf-8")
    doc = frontmatter.parse(raw)

    try:
        return _frontmatter_to_snapshot(doc.frontmatter)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Failed to parse watch-session snapshot from %s: %s",
            target,
            exc,
        )
        return None


def update(wiki_root: Path, snap: WatchSessionSnapshot) -> Path:
    """Update the watch-session record, validating that a file exists.

    This is semantically identical to write() but validates that a record
    already exists. Use this for updates to an existing session.

    Args:
        wiki_root: Path to the wiki root directory.
        snap: The updated snapshot.

    Returns:
        Path to the written wiki file.

    Raises:
        FileNotFoundError: If no watch-sessions record exists yet.
    """
    target = _wiki_file_path(wiki_root)
    if not target.exists():
        raise FileNotFoundError(
            f"No watch-sessions record at {target}. Use write() first."
        )
    return write(wiki_root, snap)


def clear(wiki_root: Path) -> Path:
    """Reset the watch-session record to empty state.

    Writes a clean empty snapshot, clearing all watchers and streams.
    The file is kept (not deleted) so the wiki always has the record.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        Path to the written wiki file.
    """
    empty = WatchSessionSnapshot()
    return write(wiki_root, empty)


def exists(wiki_root: Path) -> bool:
    """Check whether a watch-sessions wiki file exists.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        True if the file exists.
    """
    return _wiki_file_path(wiki_root).exists()


def file_path(wiki_root: Path) -> Path:
    """Return the expected path to the watch-sessions wiki file.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        Path (may or may not exist yet).
    """
    return _wiki_file_path(wiki_root)
