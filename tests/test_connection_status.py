"""Tests for wiki connection status persistence.

Covers:
    - Writing connection status to wiki current-run page
    - Reading connection status from wiki current-run page
    - Status roundtrip (write then read)
    - Updating connection status on existing run
    - Connection status serialization to YAML frontmatter
    - Connection status appears in markdown body
    - Handling missing current-run file gracefully
    - Consecutive failure count tracking
    - ConnectionStatusRecord immutability
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.ssh.liveness import ConnectionHealth, ProbeResult
from jules_daemon.wiki import current_run
from jules_daemon.wiki.connection_status import (
    ConnectionStatusRecord,
    read_connection_status,
    update_connection_status,
)
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    RunStatus,
    SSHTarget,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Create a temporary wiki root directory."""
    return tmp_path / "wiki"


@pytest.fixture
def running_wiki(wiki_root: Path) -> Path:
    """Create a wiki with an active running state."""
    target = SSHTarget(host="test.example.com", user="deploy", port=22)
    cmd = Command(natural_language="run the tests")
    run = CurrentRun(
        status=RunStatus.RUNNING,
        ssh_target=target,
        command=cmd,
    )
    run = run.with_running("pytest -v", remote_pid=12345)
    current_run.write(wiki_root, run)
    return wiki_root


def _make_probe_result(
    *,
    success: bool = True,
    health: ConnectionHealth = ConnectionHealth.CONNECTED,
    latency_ms: float = 42.5,
    output: str = "__jules_probe_ok__",
    exit_code: int = 0,
    error: str | None = None,
    probe_command: str = "echo __jules_probe_ok__",
) -> ProbeResult:
    """Build a ProbeResult for testing."""
    return ProbeResult(
        success=success,
        health=health,
        latency_ms=latency_ms,
        output=output,
        exit_code=exit_code,
        error=error,
        probe_command=probe_command,
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# ConnectionStatusRecord tests
# ---------------------------------------------------------------------------


class TestConnectionStatusRecord:
    """Verify ConnectionStatusRecord structure and immutability."""

    def test_default_values(self) -> None:
        record = ConnectionStatusRecord(
            health=ConnectionHealth.CONNECTED,
            last_probe_at=datetime.now(timezone.utc),
            probe_latency_ms=42.5,
            probe_command="echo ok",
        )
        assert record.consecutive_failures == 0
        assert record.probe_output == ""
        assert record.error is None
        assert record.session_id is None

    def test_frozen(self) -> None:
        record = ConnectionStatusRecord(
            health=ConnectionHealth.CONNECTED,
            last_probe_at=datetime.now(timezone.utc),
            probe_latency_ms=42.5,
            probe_command="echo ok",
        )
        with pytest.raises(AttributeError):
            record.health = ConnectionHealth.DISCONNECTED  # type: ignore[misc]

    def test_from_probe_result_connected(self) -> None:
        probe = _make_probe_result(
            success=True,
            health=ConnectionHealth.CONNECTED,
            latency_ms=15.3,
            output="__jules_probe_ok__",
        )
        record = ConnectionStatusRecord.from_probe_result(probe)

        assert record.health == ConnectionHealth.CONNECTED
        assert record.probe_latency_ms == 15.3
        assert record.probe_output == "__jules_probe_ok__"
        assert record.consecutive_failures == 0
        assert record.error is None

    def test_from_probe_result_failed(self) -> None:
        probe = _make_probe_result(
            success=False,
            health=ConnectionHealth.DISCONNECTED,
            error="Connection reset",
        )
        record = ConnectionStatusRecord.from_probe_result(
            probe, consecutive_failures=3
        )

        assert record.health == ConnectionHealth.DISCONNECTED
        assert record.consecutive_failures == 3
        assert record.error == "Connection reset"

    def test_from_probe_result_with_session_id(self) -> None:
        probe = _make_probe_result()
        record = ConnectionStatusRecord.from_probe_result(
            probe, session_id="session-abc-123"
        )

        assert record.session_id == "session-abc-123"


# ---------------------------------------------------------------------------
# Writing connection status
# ---------------------------------------------------------------------------


class TestUpdateConnectionStatus:
    """Verify updating wiki page with connection status."""

    def test_update_on_running_run(self, running_wiki: Path) -> None:
        probe = _make_probe_result(
            success=True,
            health=ConnectionHealth.CONNECTED,
            latency_ms=25.0,
            output="__jules_probe_ok__",
        )
        record = ConnectionStatusRecord.from_probe_result(probe)

        path = update_connection_status(running_wiki, record)
        assert path.exists()

    def test_update_preserves_run_state(self, running_wiki: Path) -> None:
        """Connection status update must not alter run state fields."""
        original = current_run.read(running_wiki)
        assert original is not None

        probe = _make_probe_result()
        record = ConnectionStatusRecord.from_probe_result(probe)
        update_connection_status(running_wiki, record)

        updated = current_run.read(running_wiki)
        assert updated is not None
        assert updated.status == original.status
        assert updated.run_id == original.run_id
        assert updated.ssh_target == original.ssh_target

    def test_update_writes_connection_section_in_frontmatter(
        self, running_wiki: Path
    ) -> None:
        probe = _make_probe_result(
            health=ConnectionHealth.CONNECTED,
            latency_ms=30.0,
        )
        record = ConnectionStatusRecord.from_probe_result(probe)
        update_connection_status(running_wiki, record)

        # Read raw file and check frontmatter
        status = read_connection_status(running_wiki)
        assert status is not None
        assert status.health == ConnectionHealth.CONNECTED
        assert status.probe_latency_ms == 30.0

    def test_update_raises_if_no_current_run(self, wiki_root: Path) -> None:
        probe = _make_probe_result()
        record = ConnectionStatusRecord.from_probe_result(probe)

        with pytest.raises(FileNotFoundError):
            update_connection_status(wiki_root, record)


# ---------------------------------------------------------------------------
# Reading connection status
# ---------------------------------------------------------------------------


class TestReadConnectionStatus:
    """Verify reading connection status from wiki page."""

    def test_read_after_write(self, running_wiki: Path) -> None:
        probe = _make_probe_result(
            health=ConnectionHealth.CONNECTED,
            latency_ms=18.7,
            output="__jules_probe_ok__",
        )
        record = ConnectionStatusRecord.from_probe_result(
            probe, session_id="session-001"
        )
        update_connection_status(running_wiki, record)

        loaded = read_connection_status(running_wiki)
        assert loaded is not None
        assert loaded.health == ConnectionHealth.CONNECTED
        assert loaded.probe_latency_ms == 18.7
        assert loaded.probe_output == "__jules_probe_ok__"
        assert loaded.session_id == "session-001"

    def test_read_returns_none_when_no_file(self, wiki_root: Path) -> None:
        result = read_connection_status(wiki_root)
        assert result is None

    def test_read_returns_none_when_no_connection_status(
        self, running_wiki: Path
    ) -> None:
        """A run without connection status should return None."""
        result = read_connection_status(running_wiki)
        assert result is None

    def test_roundtrip_disconnected(self, running_wiki: Path) -> None:
        probe = _make_probe_result(
            success=False,
            health=ConnectionHealth.DISCONNECTED,
            error="Connection timed out",
            latency_ms=5000.0,
        )
        record = ConnectionStatusRecord.from_probe_result(
            probe, consecutive_failures=5
        )
        update_connection_status(running_wiki, record)

        loaded = read_connection_status(running_wiki)
        assert loaded is not None
        assert loaded.health == ConnectionHealth.DISCONNECTED
        assert loaded.consecutive_failures == 5
        assert loaded.error == "Connection timed out"

    def test_roundtrip_degraded(self, running_wiki: Path) -> None:
        probe = _make_probe_result(
            success=False,
            health=ConnectionHealth.DEGRADED,
            error="output mismatch",
            latency_ms=150.0,
        )
        record = ConnectionStatusRecord.from_probe_result(
            probe, consecutive_failures=1
        )
        update_connection_status(running_wiki, record)

        loaded = read_connection_status(running_wiki)
        assert loaded is not None
        assert loaded.health == ConnectionHealth.DEGRADED
        assert loaded.consecutive_failures == 1


# ---------------------------------------------------------------------------
# Multiple updates (consecutive failures tracking)
# ---------------------------------------------------------------------------


class TestConsecutiveFailureTracking:
    """Verify that consecutive failure counts are persisted correctly."""

    def test_failure_count_increments_on_each_update(
        self, running_wiki: Path
    ) -> None:
        for i in range(3):
            probe = _make_probe_result(
                success=False,
                health=ConnectionHealth.DISCONNECTED,
                error=f"failure {i + 1}",
            )
            record = ConnectionStatusRecord.from_probe_result(
                probe, consecutive_failures=i + 1
            )
            update_connection_status(running_wiki, record)

        loaded = read_connection_status(running_wiki)
        assert loaded is not None
        assert loaded.consecutive_failures == 3

    def test_success_resets_failure_count(self, running_wiki: Path) -> None:
        # First write a failure
        probe_fail = _make_probe_result(
            success=False,
            health=ConnectionHealth.DISCONNECTED,
            error="failed",
        )
        record_fail = ConnectionStatusRecord.from_probe_result(
            probe_fail, consecutive_failures=3
        )
        update_connection_status(running_wiki, record_fail)

        # Then write a success
        probe_ok = _make_probe_result(
            success=True,
            health=ConnectionHealth.CONNECTED,
        )
        record_ok = ConnectionStatusRecord.from_probe_result(probe_ok)
        update_connection_status(running_wiki, record_ok)

        loaded = read_connection_status(running_wiki)
        assert loaded is not None
        assert loaded.consecutive_failures == 0
        assert loaded.health == ConnectionHealth.CONNECTED


# ---------------------------------------------------------------------------
# Body section rendering and replacement
# ---------------------------------------------------------------------------


class TestBodySectionRendering:
    """Verify markdown body contains correct connection status content."""

    def test_body_contains_health_field(self, running_wiki: Path) -> None:
        probe = _make_probe_result(
            health=ConnectionHealth.CONNECTED,
            latency_ms=22.0,
        )
        record = ConnectionStatusRecord.from_probe_result(probe)
        path = update_connection_status(running_wiki, record)

        body = path.read_text(encoding="utf-8")
        assert "## Connection Status" in body
        assert "connected" in body
        assert "22.0ms" in body

    def test_double_update_produces_single_section(
        self, running_wiki: Path
    ) -> None:
        """Two updates must not produce duplicate sections."""
        for latency in (10.0, 20.0):
            probe = _make_probe_result(latency_ms=latency)
            record = ConnectionStatusRecord.from_probe_result(probe)
            update_connection_status(running_wiki, record)

        body = running_wiki.joinpath(
            "pages/daemon/current-run.md"
        ).read_text(encoding="utf-8")
        count = body.count("## Connection Status")
        assert count == 1

    def test_section_inserted_before_timestamps(
        self, running_wiki: Path
    ) -> None:
        probe = _make_probe_result()
        record = ConnectionStatusRecord.from_probe_result(probe)
        update_connection_status(running_wiki, record)

        body = running_wiki.joinpath(
            "pages/daemon/current-run.md"
        ).read_text(encoding="utf-8")
        conn_pos = body.index("## Connection Status")
        ts_pos = body.index("## Timestamps")
        assert conn_pos < ts_pos

    def test_error_with_newlines_is_sanitized(
        self, running_wiki: Path
    ) -> None:
        """Errors with newlines must not inject markdown headings in the body."""
        probe = _make_probe_result(
            success=False,
            health=ConnectionHealth.DISCONNECTED,
            error="line1\n## Injected Heading\nline3",
        )
        record = ConnectionStatusRecord.from_probe_result(probe)
        update_connection_status(running_wiki, record)

        raw = running_wiki.joinpath(
            "pages/daemon/current-run.md"
        ).read_text(encoding="utf-8")

        # Extract the markdown body (after the closing --- of frontmatter)
        parts = raw.split("---", 2)
        assert len(parts) >= 3, "Expected frontmatter-delimited document"
        body = parts[2]

        # The injected heading must not appear at the start of any line
        # (which is what makes it a real markdown heading)
        body_lines = body.split("\n")
        for line in body_lines:
            assert line.strip() != "## Injected Heading", (
                "Newline in error text produced a standalone heading in body"
            )
        # The sanitized error should be present inline (newlines collapsed)
        assert "line1 ## Injected Heading line3" in body
