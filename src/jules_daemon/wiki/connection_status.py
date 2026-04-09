"""Wiki persistence for SSH connection liveness status.

Reads and writes the connection status subsection of the current-run wiki
page. The connection status is stored as a YAML mapping in the frontmatter
under the ``connection`` key, alongside the existing run-state fields.

This module is additive: it reads the existing current-run wiki file,
injects/updates the ``connection`` subsection in the frontmatter, updates
the markdown body to include a Connection Status section, and writes it
back atomically. Existing run-state fields are never modified.

Wiki file location: {wiki_root}/pages/daemon/current-run.md

Usage:
    from jules_daemon.wiki.connection_status import (
        update_connection_status,
        read_connection_status,
    )

    record = ConnectionStatusRecord.from_probe_result(probe_result)
    update_connection_status(wiki_root, record)

    loaded = read_connection_status(wiki_root)
    if loaded and loaded.health == ConnectionHealth.CONNECTED:
        ...
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.ssh.liveness import ConnectionHealth, ProbeResult
from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument

__all__ = [
    "ConnectionStatusRecord",
    "read_connection_status",
    "update_connection_status",
]

logger = logging.getLogger(__name__)

_CURRENT_RUN_FILENAME = "current-run.md"
_DAEMON_DIR = "pages/daemon"
_CONNECTION_KEY = "connection"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wiki_file_path(wiki_root: Path) -> Path:
    """Resolve the absolute path to the current-run wiki file."""
    return wiki_root / _DAEMON_DIR / _CURRENT_RUN_FILENAME


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
# Model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConnectionStatusRecord:
    """Immutable record of SSH connection liveness status.

    Represents the most recent liveness probe result, suitable for
    persistence in the wiki frontmatter.

    Attributes:
        health: Current connection health classification.
        last_probe_at: UTC timestamp of the most recent probe.
        probe_latency_ms: Round-trip time of the probe in milliseconds.
        probe_command: The command string used for probing.
        probe_output: Stripped stdout from the probe command.
        consecutive_failures: Number of consecutive failed probes.
            Reset to 0 on success.
        error: Error description from the most recent probe. None on success.
        session_id: SSH session identifier (optional, for correlation).
    """

    health: ConnectionHealth
    last_probe_at: datetime
    probe_latency_ms: float
    probe_command: str
    probe_output: str = ""
    consecutive_failures: int = 0
    error: str | None = None
    session_id: str | None = None

    @classmethod
    def from_probe_result(
        cls,
        result: ProbeResult,
        *,
        consecutive_failures: int = 0,
        session_id: str | None = None,
    ) -> ConnectionStatusRecord:
        """Build a ConnectionStatusRecord from a ProbeResult.

        Args:
            result: The probe result to convert.
            consecutive_failures: Running count of consecutive failures.
                Caller is responsible for tracking and incrementing.
            session_id: Optional SSH session identifier for correlation.

        Returns:
            New frozen ConnectionStatusRecord.
        """
        return cls(
            health=result.health,
            last_probe_at=result.timestamp,
            probe_latency_ms=result.latency_ms,
            probe_command=result.probe_command,
            probe_output=result.output,
            consecutive_failures=consecutive_failures,
            error=result.error,
            session_id=session_id,
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _record_to_dict(record: ConnectionStatusRecord) -> dict[str, Any]:
    """Serialize a ConnectionStatusRecord to a YAML-compatible dict."""
    return {
        "health": record.health.value,
        "last_probe_at": _datetime_to_iso(record.last_probe_at),
        "probe_latency_ms": record.probe_latency_ms,
        "probe_command": record.probe_command,
        "probe_output": record.probe_output,
        "consecutive_failures": record.consecutive_failures,
        "error": record.error,
        "session_id": record.session_id,
    }


def _dict_to_record(data: dict[str, Any]) -> ConnectionStatusRecord:
    """Deserialize a ConnectionStatusRecord from a YAML dict."""
    return ConnectionStatusRecord(
        health=ConnectionHealth(data["health"]),
        last_probe_at=_iso_to_datetime(data["last_probe_at"])
        or datetime.now(timezone.utc),
        probe_latency_ms=float(data.get("probe_latency_ms", 0.0)),
        probe_command=data.get("probe_command", ""),
        probe_output=data.get("probe_output", ""),
        consecutive_failures=int(data.get("consecutive_failures", 0)),
        error=data.get("error"),
        session_id=data.get("session_id"),
    )


def _sanitize_for_markdown(text: str) -> str:
    """Sanitize a string for safe inline markdown rendering.

    Replaces newlines with spaces and escapes backticks to prevent
    markdown structure injection (e.g., headings, code blocks).
    """
    return text.replace("\n", " ").replace("`", "\\`")


def _build_connection_body_section(record: ConnectionStatusRecord) -> str:
    """Generate the markdown section for connection status."""
    lines = [
        "## Connection Status",
        "",
        f"- **Health:** {record.health.value}",
        f"- **Last Probe:** {_datetime_to_iso(record.last_probe_at)}",
        f"- **Latency:** {record.probe_latency_ms:.1f}ms",
        f"- **Probe Command:** `{_sanitize_for_markdown(record.probe_command)}`",
    ]
    if record.probe_output:
        sanitized = _sanitize_for_markdown(record.probe_output)
        lines.append(f"- **Probe Output:** `{sanitized}`")
    if record.consecutive_failures > 0:
        lines.append(
            f"- **Consecutive Failures:** {record.consecutive_failures}"
        )
    if record.error:
        sanitized = _sanitize_for_markdown(record.error)
        lines.append(f"- **Error:** `{sanitized}`")
    if record.session_id:
        lines.append(f"- **Session ID:** {record.session_id}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def update_connection_status(
    wiki_root: Path,
    record: ConnectionStatusRecord,
) -> Path:
    """Update the current-run wiki page with connection liveness status.

    Reads the existing current-run wiki file, injects or replaces the
    ``connection`` subsection in the YAML frontmatter, appends or replaces
    the Connection Status markdown section in the body, and writes the
    file back atomically.

    Existing run-state fields (status, run_id, ssh_target, command, etc.)
    are preserved unchanged.

    Args:
        wiki_root: Path to the wiki root directory.
        record: The connection status to persist.

    Returns:
        Path to the updated wiki file.

    Raises:
        FileNotFoundError: If no current-run wiki file exists.
    """
    file_path = _wiki_file_path(wiki_root)
    if not file_path.exists():
        raise FileNotFoundError(
            f"No current-run wiki file at {file_path}. "
            "Cannot update connection status without an active run."
        )

    # Read existing document
    raw = file_path.read_text(encoding="utf-8")
    doc = frontmatter.parse(raw)

    # Update frontmatter with connection status
    updated_fm = dict(doc.frontmatter)
    updated_fm[_CONNECTION_KEY] = _record_to_dict(record)
    updated_fm["updated"] = _datetime_to_iso(datetime.now(timezone.utc))

    # Update body: replace or append Connection Status section
    body = _update_body_section(doc.body, record)

    updated_doc = WikiDocument(frontmatter=updated_fm, body=body)
    content = frontmatter.serialize(updated_doc)

    # Atomic write with cleanup on failure
    tmp_path = file_path.parent / (file_path.name + ".tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, file_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info(
        "Updated connection status in %s: health=%s latency=%.1fms failures=%d",
        file_path,
        record.health.value,
        record.probe_latency_ms,
        record.consecutive_failures,
    )
    return file_path


def read_connection_status(
    wiki_root: Path,
) -> ConnectionStatusRecord | None:
    """Read the connection liveness status from the current-run wiki page.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        The deserialized ConnectionStatusRecord, or None if:
        - The wiki file does not exist
        - The file has no ``connection`` section in frontmatter
    """
    file_path = _wiki_file_path(wiki_root)
    if not file_path.exists():
        return None

    raw = file_path.read_text(encoding="utf-8")
    doc = frontmatter.parse(raw)

    connection_data = doc.frontmatter.get(_CONNECTION_KEY)
    if connection_data is None or not isinstance(connection_data, dict):
        return None

    try:
        return _dict_to_record(connection_data)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Failed to parse connection status from %s: %s",
            file_path,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Body section management
# ---------------------------------------------------------------------------

_SECTION_MARKER = "## Connection Status"


def _is_section_heading(line: str, marker: str) -> bool:
    """Check if a line is exactly the given section heading."""
    return line.strip() == marker


def _update_body_section(
    body: str,
    record: ConnectionStatusRecord,
) -> str:
    """Replace or append the Connection Status section in the markdown body.

    If a '## Connection Status' heading exists at the start of a line,
    it and all lines until the next ``## `` heading are replaced.
    Otherwise, the section is inserted before the Timestamps section
    or appended at the end.
    """
    new_section = _build_connection_body_section(record)

    # Check if the section marker exists on its own line
    lines = body.split("\n")
    has_existing = any(
        _is_section_heading(line, _SECTION_MARKER) for line in lines
    )

    if has_existing:
        # Replace existing section
        result_lines: list[str] = []
        skip = False

        for line in lines:
            if _is_section_heading(line, _SECTION_MARKER):
                skip = True
                # Insert the new section (with trailing blank line)
                for section_line in new_section.split("\n"):
                    result_lines.append(section_line)
                continue

            if skip:
                # Stop skipping when we hit another heading
                if line.startswith("## ") and not _is_section_heading(
                    line, _SECTION_MARKER
                ):
                    skip = False
                    result_lines.append(line)
                # Skip lines in the old section
                continue

            result_lines.append(line)

        return "\n".join(result_lines)

    # No existing section -- insert before "## Timestamps" or append at end
    has_timestamps = any(
        _is_section_heading(line, "## Timestamps") for line in lines
    )
    if has_timestamps:
        return body.replace(
            "## Timestamps",
            new_section + "\n## Timestamps",
        )

    # Append at end
    stripped = body.rstrip()
    if stripped:
        return stripped + "\n\n" + new_section
    return new_section
