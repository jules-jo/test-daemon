"""Thread-safe command queue with wiki-backed persistence.

Each queued command is persisted as a Karpathy-style wiki file (YAML
frontmatter + markdown body) in pages/daemon/queue/. The queue supports:

- Enqueue: add a new command, persist to wiki file
- Dequeue: remove and return the highest-priority, oldest command
- Peek: inspect the next command without removing it
- Cancel: mark a command as cancelled, remove its wiki file
- List: return all pending commands sorted by priority and sequence
- Size: count of pending commands

Thread safety is ensured by a threading.Lock that serializes all
queue mutations. File I/O uses atomic writes (write-to-tmp + rename)
to prevent corruption from crashes.

Wiki file layout:
  {wiki_root}/pages/daemon/queue/{sequence:06d}-{queue_id}.md

Recovery: On initialization, the queue scans the wiki directory for
existing entries and restores the in-memory state. This enables
crash recovery -- the daemon simply re-creates the CommandQueue and
all pending commands are restored.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument
from jules_daemon.wiki.queue_models import (
    QueuedCommand,
    QueuePriority,
    QueueStatus,
)

__all__ = ["CommandQueue"]

logger = logging.getLogger(__name__)

_QUEUE_DIR = "pages/daemon/queue"
_WIKI_TAGS = ["daemon", "queue"]
_WIKI_TYPE = "queued-command"


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO 8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 string to datetime, or None."""
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _command_to_frontmatter(cmd: QueuedCommand) -> dict[str, Any]:
    """Convert a QueuedCommand to a YAML-serializable frontmatter dict."""
    return {
        "tags": list(_WIKI_TAGS),
        "type": _WIKI_TYPE,
        "queue_id": cmd.queue_id,
        "sequence": cmd.sequence,
        "status": cmd.status.value,
        "priority": cmd.priority.value,
        "natural_language": cmd.natural_language,
        "ssh_host": cmd.ssh_host,
        "ssh_user": cmd.ssh_user,
        "ssh_port": cmd.ssh_port,
        "queued_at": _datetime_to_iso(cmd.queued_at),
        "started_at": _datetime_to_iso(cmd.started_at),
        "completed_at": _datetime_to_iso(cmd.completed_at),
        "error": cmd.error,
    }


def _frontmatter_to_command(fm: dict[str, Any]) -> QueuedCommand:
    """Reconstruct a QueuedCommand from a parsed frontmatter dict."""
    return QueuedCommand(
        queue_id=fm["queue_id"],
        sequence=int(fm["sequence"]),
        natural_language=fm["natural_language"],
        status=QueueStatus(fm.get("status", "queued")),
        priority=QueuePriority(int(fm.get("priority", QueuePriority.NORMAL.value))),
        ssh_host=fm.get("ssh_host"),
        ssh_user=fm.get("ssh_user"),
        ssh_port=int(fm.get("ssh_port", 22)),
        queued_at=_iso_to_datetime(fm.get("queued_at")) or datetime.now(timezone.utc),
        started_at=_iso_to_datetime(fm.get("started_at")),
        completed_at=_iso_to_datetime(fm.get("completed_at")),
        error=fm.get("error"),
    )


def _build_body(cmd: QueuedCommand) -> str:
    """Generate a human-readable markdown body for a queued command."""
    lines = [
        "# Queued Command",
        "",
        f"*Queue entry -- status: {cmd.status.value}, "
        f"priority: {cmd.priority.name.lower()}*",
        "",
        "## Command",
        "",
        f"> {cmd.natural_language}",
        "",
    ]

    if cmd.ssh_host:
        lines.extend([
            "## Target",
            "",
            f"- **Host:** {cmd.ssh_host}",
        ])
        if cmd.ssh_user:
            lines.append(f"- **User:** {cmd.ssh_user}")
        if cmd.ssh_port != 22:
            lines.append(f"- **Port:** {cmd.ssh_port}")
        lines.append("")

    lines.extend([
        "## Metadata",
        "",
        f"- **Queue ID:** {cmd.queue_id}",
        f"- **Sequence:** {cmd.sequence}",
        f"- **Priority:** {cmd.priority.name.lower()}",
        f"- **Queued At:** {_datetime_to_iso(cmd.queued_at)}",
    ])

    if cmd.started_at:
        lines.append(f"- **Started At:** {_datetime_to_iso(cmd.started_at)}")
    if cmd.completed_at:
        lines.append(f"- **Completed At:** {_datetime_to_iso(cmd.completed_at)}")
    if cmd.error:
        lines.extend([
            "",
            "## Error",
            "",
            "```",
            cmd.error,
            "```",
        ])

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _queue_dir_path(wiki_root: Path) -> Path:
    """Resolve the absolute path to the queue directory."""
    return wiki_root / _QUEUE_DIR


def _entry_file_path(wiki_root: Path, cmd: QueuedCommand) -> Path:
    """Resolve the absolute path for a queue entry's wiki file."""
    return _queue_dir_path(wiki_root) / f"{cmd.file_stem}.md"


def _ensure_directory(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_entry(wiki_root: Path, cmd: QueuedCommand) -> Path:
    """Atomically write a queue entry to its wiki file.

    Uses write-to-tmp + rename for crash safety.
    """
    file_path = _entry_file_path(wiki_root, cmd)
    _ensure_directory(file_path)

    doc = WikiDocument(
        frontmatter=_command_to_frontmatter(cmd),
        body=_build_body(cmd),
    )
    content = frontmatter.serialize(doc)

    tmp_path = file_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(file_path))

    return file_path


def _delete_entry(wiki_root: Path, cmd: QueuedCommand) -> bool:
    """Delete a queue entry's wiki file. Returns True if file existed."""
    file_path = _entry_file_path(wiki_root, cmd)
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def _scan_entries(wiki_root: Path) -> list[QueuedCommand]:
    """Scan the queue directory and parse all valid queue entries.

    Skips files that cannot be parsed (logs a warning for each).
    Returns entries sorted by sort_key (priority desc, sequence asc).
    """
    queue_dir = _queue_dir_path(wiki_root)
    if not queue_dir.is_dir():
        return []

    entries: list[QueuedCommand] = []
    for md_file in sorted(queue_dir.glob("*.md")):
        if md_file.name == "README.md":
            continue
        try:
            raw = md_file.read_text(encoding="utf-8")
            doc = frontmatter.parse(raw)
            cmd = _frontmatter_to_command(doc.frontmatter)
            if cmd.status == QueueStatus.QUEUED:
                entries.append(cmd)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Skipping invalid queue file %s: %s",
                md_file.name,
                exc,
            )

    entries.sort(key=lambda c: c.sort_key)
    return entries


def _scan_entries_by_status(
    wiki_root: Path,
    target_status: QueueStatus,
) -> list[QueuedCommand]:
    """Scan the queue directory for entries matching a specific status.

    Used for crash recovery to find commands left in ACTIVE state.
    Returns entries sorted by sort_key (priority desc, sequence asc).
    """
    queue_dir = _queue_dir_path(wiki_root)
    if not queue_dir.is_dir():
        return []

    entries: list[QueuedCommand] = []
    for md_file in sorted(queue_dir.glob("*.md")):
        if md_file.name == "README.md":
            continue
        try:
            raw = md_file.read_text(encoding="utf-8")
            doc = frontmatter.parse(raw)
            cmd = _frontmatter_to_command(doc.frontmatter)
            if cmd.status == target_status:
                entries.append(cmd)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Skipping invalid queue file %s: %s",
                md_file.name,
                exc,
            )

    entries.sort(key=lambda c: c.sort_key)
    return entries


# ---------------------------------------------------------------------------
# Thread-safe command queue
# ---------------------------------------------------------------------------


class CommandQueue:
    """Thread-safe command queue backed by wiki markdown files.

    Each queued command is persisted as an individual wiki file with YAML
    frontmatter and a markdown body. The in-memory index is rebuilt from
    the wiki directory on initialization, enabling crash recovery.

    Thread safety: All public methods acquire a single lock before
    mutating state. This serializes access and prevents data races
    between concurrent enqueue/dequeue/cancel operations.

    Args:
        wiki_root: Path to the wiki root directory.
    """

    def __init__(self, wiki_root: Path) -> None:
        self._wiki_root = wiki_root
        self._lock = threading.Lock()
        # In-memory index: queue_id -> QueuedCommand
        self._entries: dict[str, QueuedCommand] = {}
        # Atomic sequence counter
        self._next_seq: int = 1

        # Recover any existing entries from the wiki
        self._recover_from_wiki()

    def _recover_from_wiki(self) -> None:
        """Load existing queue entries from wiki files into memory.

        Called once during initialization. Loads both QUEUED and ACTIVE
        entries. ACTIVE entries are included so crash recovery can find
        and resolve them. Sets the sequence counter to one beyond the
        highest existing sequence number.
        """
        queued = _scan_entries(self._wiki_root)
        active = _scan_entries_by_status(self._wiki_root, QueueStatus.ACTIVE)
        all_entries = queued + active

        max_seq = 0
        for cmd in all_entries:
            self._entries[cmd.queue_id] = cmd
            if cmd.sequence > max_seq:
                max_seq = cmd.sequence
        self._next_seq = max_seq + 1

        if all_entries:
            logger.info(
                "Recovered %d queue entries from wiki "
                "(queued=%d, active=%d, next_seq=%d)",
                len(all_entries),
                len(queued),
                len(active),
                self._next_seq,
            )

    def _allocate_sequence(self) -> int:
        """Allocate the next sequence number. Must be called under lock."""
        seq = self._next_seq
        self._next_seq = seq + 1
        return seq

    def _sorted_pending(self) -> tuple[QueuedCommand, ...]:
        """Return all pending entries sorted by sort_key. Must be called under lock."""
        pending = [
            cmd for cmd in self._entries.values()
            if cmd.status == QueueStatus.QUEUED
        ]
        pending.sort(key=lambda c: c.sort_key)
        return tuple(pending)

    # -- Public API --

    def enqueue(
        self,
        natural_language: str,
        *,
        ssh_host: Optional[str] = None,
        ssh_user: Optional[str] = None,
        ssh_port: int = 22,
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> QueuedCommand:
        """Add a new command to the queue.

        Creates a wiki file for the entry and adds it to the in-memory
        index. The command starts in QUEUED status.

        Args:
            natural_language: The user's command text.
            ssh_host: Optional target SSH hostname.
            ssh_user: Optional target SSH username.
            ssh_port: Target SSH port (default 22).
            priority: Priority tier for ordering.

        Returns:
            The newly created QueuedCommand.

        Raises:
            ValueError: If natural_language is empty.
        """
        with self._lock:
            seq = self._allocate_sequence()
            cmd = QueuedCommand(
                natural_language=natural_language,
                sequence=seq,
                ssh_host=ssh_host,
                ssh_user=ssh_user,
                ssh_port=ssh_port,
                priority=priority,
            )
            _write_entry(self._wiki_root, cmd)
            self._entries[cmd.queue_id] = cmd

            logger.debug(
                "Enqueued command seq=%d id=%s: %s",
                cmd.sequence,
                cmd.queue_id,
                cmd.natural_language[:80],
            )
            return cmd

    def dequeue(self) -> Optional[QueuedCommand]:
        """Remove and return the next command from the queue.

        Returns the highest-priority, lowest-sequence QUEUED command,
        transitions it to ACTIVE status, and deletes its wiki file.

        Returns:
            The activated QueuedCommand, or None if the queue is empty.
        """
        with self._lock:
            pending = self._sorted_pending()
            if not pending:
                return None

            cmd = pending[0]
            activated = cmd.with_activated()

            # Remove the wiki file and in-memory entry
            _delete_entry(self._wiki_root, cmd)
            del self._entries[cmd.queue_id]

            logger.debug(
                "Dequeued command seq=%d id=%s: %s",
                activated.sequence,
                activated.queue_id,
                activated.natural_language[:80],
            )
            return activated

    def peek(self) -> Optional[QueuedCommand]:
        """Inspect the next command without removing it.

        Returns:
            The next QueuedCommand that would be dequeued, or None.
        """
        with self._lock:
            pending = self._sorted_pending()
            if not pending:
                return None
            return pending[0]

    def list_pending(self) -> tuple[QueuedCommand, ...]:
        """Return all pending commands sorted by priority and sequence.

        Returns:
            Tuple of QueuedCommand instances in dequeue order.
        """
        with self._lock:
            return self._sorted_pending()

    def cancel(self, queue_id: str) -> bool:
        """Cancel a queued command and remove its wiki file.

        Args:
            queue_id: The queue_id of the command to cancel.

        Returns:
            True if the command was found and cancelled, False otherwise.
        """
        with self._lock:
            cmd = self._entries.get(queue_id)
            if cmd is None:
                return False

            if cmd.status != QueueStatus.QUEUED:
                return False

            _delete_entry(self._wiki_root, cmd)
            del self._entries[cmd.queue_id]

            logger.debug(
                "Cancelled command seq=%d id=%s",
                cmd.sequence,
                cmd.queue_id,
            )
            return True

    def size(self) -> int:
        """Return the number of pending commands in the queue.

        Returns:
            Count of QUEUED-status entries.
        """
        with self._lock:
            return self._pending_count()

    def _pending_count(self) -> int:
        """Count pending entries. Must be called under lock."""
        return sum(
            1 for cmd in self._entries.values()
            if cmd.status == QueueStatus.QUEUED
        )

    def try_enqueue(
        self,
        natural_language: str,
        *,
        max_size: int,
        ssh_host: Optional[str] = None,
        ssh_user: Optional[str] = None,
        ssh_port: int = 22,
        priority: QueuePriority = QueuePriority.NORMAL,
    ) -> tuple[QueuedCommand | None, int]:
        """Atomically check capacity and enqueue a command.

        Performs the size check and enqueue under a single lock
        acquisition, eliminating TOCTOU races. If the queue is at or
        above ``max_size``, no enqueue is performed and the current size
        is returned.

        Args:
            natural_language: The user's command text.
            max_size: Maximum pending entries allowed.
            ssh_host: Optional target SSH hostname.
            ssh_user: Optional target SSH username.
            ssh_port: Target SSH port (default 22).
            priority: Priority tier for ordering.

        Returns:
            A tuple of ``(queued_command, pending_count)``:
            - On success: ``(QueuedCommand, size_after_enqueue)``
            - On full: ``(None, current_size)``

        Raises:
            ValueError: If natural_language is empty.
        """
        with self._lock:
            current = self._pending_count()
            if current >= max_size:
                return (None, current)

            seq = self._allocate_sequence()
            cmd = QueuedCommand(
                natural_language=natural_language,
                sequence=seq,
                ssh_host=ssh_host,
                ssh_user=ssh_user,
                ssh_port=ssh_port,
                priority=priority,
            )
            _write_entry(self._wiki_root, cmd)
            self._entries[cmd.queue_id] = cmd

            size_after = self._pending_count()
            logger.debug(
                "try_enqueue: seq=%d id=%s size=%d: %s",
                cmd.sequence,
                cmd.queue_id,
                size_after,
                cmd.natural_language[:80],
            )
            return (cmd, size_after)

    def get(self, queue_id: str) -> Optional[QueuedCommand]:
        """Look up a specific queue entry by ID.

        Args:
            queue_id: The queue_id to look up.

        Returns:
            The QueuedCommand if found, or None.
        """
        with self._lock:
            return self._entries.get(queue_id)

    # -- Lifecycle transition methods --

    def activate(self, queue_id: str) -> Optional[QueuedCommand]:
        """Transition a QUEUED command to ACTIVE, updating the wiki file.

        The command stays in the in-memory index with ACTIVE status.
        The wiki file is updated in-place (not deleted) to reflect the
        transition. Only QUEUED commands can be activated.

        Args:
            queue_id: The queue_id of the command to activate.

        Returns:
            The activated QueuedCommand, or None if not found or not QUEUED.
        """
        with self._lock:
            cmd = self._entries.get(queue_id)
            if cmd is None:
                return None
            if cmd.status != QueueStatus.QUEUED:
                return None

            activated = cmd.with_activated()
            _write_entry(self._wiki_root, activated)
            self._entries[queue_id] = activated

            logger.debug(
                "Activated command seq=%d id=%s: %s",
                activated.sequence,
                activated.queue_id,
                activated.natural_language[:80],
            )
            return activated

    def mark_completed(self, queue_id: str) -> Optional[QueuedCommand]:
        """Transition an ACTIVE command to COMPLETED, updating the wiki file.

        The wiki file is updated with the terminal state. The command is
        removed from the in-memory index (terminal state) but its wiki
        file persists for audit trail.

        Args:
            queue_id: The queue_id of the command to complete.

        Returns:
            The completed QueuedCommand, or None if not found or not ACTIVE.
        """
        with self._lock:
            cmd = self._entries.get(queue_id)
            if cmd is None:
                return None
            if cmd.status != QueueStatus.ACTIVE:
                return None

            completed = cmd.with_completed()
            _write_entry(self._wiki_root, completed)
            del self._entries[queue_id]

            logger.debug(
                "Completed command seq=%d id=%s",
                completed.sequence,
                completed.queue_id,
            )
            return completed

    def mark_failed(
        self,
        queue_id: str,
        error: str,
    ) -> Optional[QueuedCommand]:
        """Transition an ACTIVE command to FAILED, updating the wiki file.

        The wiki file is updated with the terminal state and error message.
        The command is removed from the in-memory index (terminal state)
        but its wiki file persists for audit trail.

        Args:
            queue_id: The queue_id of the command to fail.
            error: Human-readable error description.

        Returns:
            The failed QueuedCommand, or None if not found or not ACTIVE.
        """
        with self._lock:
            cmd = self._entries.get(queue_id)
            if cmd is None:
                return None
            if cmd.status != QueueStatus.ACTIVE:
                return None

            failed = cmd.with_failed(error)
            _write_entry(self._wiki_root, failed)
            del self._entries[queue_id]

            logger.debug(
                "Failed command seq=%d id=%s: %s",
                failed.sequence,
                failed.queue_id,
                error[:80],
            )
            return failed

    def scan_active(self) -> list[QueuedCommand]:
        """Scan wiki files for commands left in ACTIVE state.

        Used for crash recovery: after a daemon restart, any commands
        found in ACTIVE state represent interrupted executions that
        need to be resolved (failed or re-queued).

        Returns:
            List of QueuedCommand instances with ACTIVE status,
            sorted by sort_key.
        """
        return _scan_entries_by_status(self._wiki_root, QueueStatus.ACTIVE)
