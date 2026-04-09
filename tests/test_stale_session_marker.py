"""Tests for stale session marker.

Verifies that the stale session marker:
- Given a list of LivenessResult verdicts, identifies non-live sessions
- Updates each non-live session's wiki entry YAML frontmatter to
  status: stale with a staleness reason and detection timestamp
- Uses immutable write-new-file semantics (writes a new file, never
  mutates the original)
- Preserves the original file contents (original is not deleted or modified)
- Skips sessions that are already live (alive=True)
- Handles sessions where the source wiki file does not exist
- Returns structured results for each session processed
- Generates correct staleness reasons based on SessionHealth
- Records detection timestamp in ISO 8601 UTC format
- Is idempotent (marking an already-stale session again produces a new
  versioned file without corrupting state)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki import current_run, frontmatter
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    RunStatus,
    SSHTarget,
)
from jules_daemon.wiki.stale_session_marker import (
    MarkResult,
    MarkOutcome,
    StaleMarkerInput,
    mark_stale_sessions,
    mark_single_session_stale,
    build_staleness_reason,
)
from jules_daemon.monitor.session_liveness import (
    LivenessResult,
    SessionHealth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    return tmp_path / "wiki"


def _make_liveness_result(
    *,
    session_id: str = "run-abc",
    health: SessionHealth = SessionHealth.PROCESS_DEAD,
    alive: bool = False,
    errors: tuple[str, ...] = (),
) -> LivenessResult:
    """Create a minimal LivenessResult for testing."""
    return LivenessResult(
        session_id=session_id,
        health=health,
        alive=alive,
        process_result=None,
        ssh_result=None,
        errors=errors,
        latency_ms=1.0,
        timestamp=datetime.now(timezone.utc),
    )


def _write_running_session(wiki_root: Path) -> CurrentRun:
    """Write a running session to the wiki and return the run model."""
    target = SSHTarget(host="staging.example.com", user="ci")
    cmd = Command(natural_language="run the test suite")
    run = CurrentRun().with_pending_approval(target, cmd, daemon_pid=1234)
    run = run.with_running("pytest -v", remote_pid=5678)
    current_run.write(wiki_root, run)
    return run


# ---------------------------------------------------------------------------
# build_staleness_reason
# ---------------------------------------------------------------------------


class TestBuildStalenessReason:
    """Verify staleness reason generation from SessionHealth."""

    def test_process_dead_reason(self) -> None:
        reason = build_staleness_reason(SessionHealth.PROCESS_DEAD)
        assert "process" in reason.lower()
        assert "dead" in reason.lower()

    def test_connection_lost_reason(self) -> None:
        reason = build_staleness_reason(SessionHealth.CONNECTION_LOST)
        assert "connection" in reason.lower()

    def test_unknown_reason(self) -> None:
        reason = build_staleness_reason(SessionHealth.UNKNOWN)
        assert "unknown" in reason.lower() or "undetermined" in reason.lower()

    def test_healthy_returns_defensive_reason(self) -> None:
        # Should not normally be called for healthy, but should not crash
        reason = build_staleness_reason(SessionHealth.HEALTHY)
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_degraded_returns_reason(self) -> None:
        reason = build_staleness_reason(SessionHealth.DEGRADED)
        assert isinstance(reason, str)
        assert len(reason) > 0


# ---------------------------------------------------------------------------
# StaleMarkerInput model
# ---------------------------------------------------------------------------


class TestStaleMarkerInput:
    """Verify the immutable input model."""

    def test_frozen(self) -> None:
        liveness = _make_liveness_result()
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=Path("/tmp/test.md"),
        )
        with pytest.raises(AttributeError):
            inp.source_path = Path("/other")  # type: ignore[misc]

    def test_required_fields(self) -> None:
        liveness = _make_liveness_result()
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=Path("/tmp/test.md"),
        )
        assert inp.liveness_result is liveness
        assert inp.source_path == Path("/tmp/test.md")


# ---------------------------------------------------------------------------
# MarkResult model
# ---------------------------------------------------------------------------


class TestMarkResult:
    """Verify the immutable result model."""

    def test_frozen(self) -> None:
        result = MarkResult(
            session_id="run-1",
            outcome=MarkOutcome.MARKED_STALE,
            source_path=Path("/tmp/old.md"),
            stale_path=Path("/tmp/new.md"),
            reason="Daemon process dead",
            detected_at=datetime.now(timezone.utc),
            error=None,
        )
        with pytest.raises(AttributeError):
            result.outcome = MarkOutcome.SKIPPED_ALIVE  # type: ignore[misc]

    def test_all_outcomes_exist(self) -> None:
        assert MarkOutcome.MARKED_STALE.value == "marked_stale"
        assert MarkOutcome.SKIPPED_ALIVE.value == "skipped_alive"
        assert MarkOutcome.SOURCE_MISSING.value == "source_missing"
        assert MarkOutcome.ERROR.value == "error"


# ---------------------------------------------------------------------------
# mark_single_session_stale -- immutable write-new-file
# ---------------------------------------------------------------------------


class TestMarkSingleSessionStale:
    """mark_single_session_stale writes a new stale file without mutating
    the original."""

    def test_writes_new_file(self, wiki_root: Path) -> None:
        """A new versioned file is created alongside the original."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)
        original_content = source.read_text(encoding="utf-8")

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
            alive=False,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)

        assert result.outcome == MarkOutcome.MARKED_STALE
        assert result.stale_path is not None
        assert result.stale_path.exists()
        # Original file must be unchanged
        assert source.read_text(encoding="utf-8") == original_content
        # New file is at a different path
        assert result.stale_path != source

    def test_new_file_has_stale_status(self, wiki_root: Path) -> None:
        """The new file's frontmatter has status: stale."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)
        assert result.stale_path is not None

        doc = frontmatter.parse(result.stale_path.read_text(encoding="utf-8"))
        assert doc.frontmatter["status"] == "stale"

    def test_new_file_has_staleness_reason(self, wiki_root: Path) -> None:
        """The new file's frontmatter includes the staleness reason."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.CONNECTION_LOST,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)
        assert result.stale_path is not None

        doc = frontmatter.parse(result.stale_path.read_text(encoding="utf-8"))
        assert "staleness_reason" in doc.frontmatter
        assert "connection" in doc.frontmatter["staleness_reason"].lower()

    def test_new_file_has_detected_at_timestamp(self, wiki_root: Path) -> None:
        """The new file's frontmatter includes detected_at as ISO 8601."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        before = datetime.now(timezone.utc)
        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)
        after = datetime.now(timezone.utc)

        assert result.stale_path is not None
        doc = frontmatter.parse(result.stale_path.read_text(encoding="utf-8"))
        detected_at_str = doc.frontmatter["staleness_detected_at"]
        detected_at = datetime.fromisoformat(detected_at_str)
        assert before <= detected_at <= after

    def test_new_file_preserves_original_frontmatter(self, wiki_root: Path) -> None:
        """The new file preserves run_id, ssh_target, etc. from the original."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)
        assert result.stale_path is not None

        doc = frontmatter.parse(result.stale_path.read_text(encoding="utf-8"))
        assert doc.frontmatter["run_id"] == run.run_id
        assert doc.frontmatter["ssh_target"]["host"] == "staging.example.com"
        assert doc.frontmatter["ssh_target"]["user"] == "ci"

    def test_new_file_preserves_body(self, wiki_root: Path) -> None:
        """The new file preserves the markdown body from the original."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)
        original_doc = frontmatter.parse(source.read_text(encoding="utf-8"))

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)
        assert result.stale_path is not None

        new_doc = frontmatter.parse(result.stale_path.read_text(encoding="utf-8"))
        assert new_doc.body == original_doc.body

    def test_new_file_has_previous_status(self, wiki_root: Path) -> None:
        """The new file records the previous status for audit trail."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)
        assert result.stale_path is not None

        doc = frontmatter.parse(result.stale_path.read_text(encoding="utf-8"))
        assert doc.frontmatter["previous_status"] == "running"

    def test_source_missing_returns_source_missing_outcome(
        self, wiki_root: Path
    ) -> None:
        """When the source file does not exist, outcome is SOURCE_MISSING."""
        missing_path = wiki_root / "pages" / "daemon" / "ghost.md"
        liveness = _make_liveness_result(
            session_id="ghost-run",
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=missing_path,
        )

        result = mark_single_session_stale(inp, wiki_root)
        assert result.outcome == MarkOutcome.SOURCE_MISSING
        assert result.stale_path is None

    def test_new_file_records_session_health(self, wiki_root: Path) -> None:
        """The new file records the SessionHealth that triggered staleness."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.UNKNOWN,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)
        assert result.stale_path is not None

        doc = frontmatter.parse(result.stale_path.read_text(encoding="utf-8"))
        assert doc.frontmatter["staleness_health"] == "unknown"


# ---------------------------------------------------------------------------
# mark_stale_sessions -- batch processing
# ---------------------------------------------------------------------------


class TestMarkStaleSessions:
    """mark_stale_sessions processes a batch of liveness verdicts."""

    def test_marks_non_live_sessions(self, wiki_root: Path) -> None:
        """Non-live sessions get marked stale."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
            alive=False,
        )
        inputs = [
            StaleMarkerInput(liveness_result=liveness, source_path=source)
        ]

        results = mark_stale_sessions(inputs, wiki_root)
        assert len(results) == 1
        assert results[0].outcome == MarkOutcome.MARKED_STALE

    def test_skips_alive_sessions(self, wiki_root: Path) -> None:
        """Alive sessions are skipped."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.HEALTHY,
            alive=True,
        )
        inputs = [
            StaleMarkerInput(liveness_result=liveness, source_path=source)
        ]

        results = mark_stale_sessions(inputs, wiki_root)
        assert len(results) == 1
        assert results[0].outcome == MarkOutcome.SKIPPED_ALIVE

    def test_mixed_alive_and_dead(self, wiki_root: Path) -> None:
        """Batch with both alive and dead sessions."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        alive_liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.HEALTHY,
            alive=True,
        )
        dead_liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
            alive=False,
        )

        inputs = [
            StaleMarkerInput(liveness_result=alive_liveness, source_path=source),
            StaleMarkerInput(liveness_result=dead_liveness, source_path=source),
        ]

        results = mark_stale_sessions(inputs, wiki_root)
        assert len(results) == 2
        outcomes = {r.outcome for r in results}
        assert MarkOutcome.SKIPPED_ALIVE in outcomes
        assert MarkOutcome.MARKED_STALE in outcomes

    def test_empty_input_returns_empty(self, wiki_root: Path) -> None:
        """Empty input list returns empty result list."""
        results = mark_stale_sessions([], wiki_root)
        assert results == ()

    def test_result_session_ids_match(self, wiki_root: Path) -> None:
        """Each result's session_id matches its input's liveness session_id."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
            alive=False,
        )
        inputs = [
            StaleMarkerInput(liveness_result=liveness, source_path=source)
        ]

        results = mark_stale_sessions(inputs, wiki_root)
        assert results[0].session_id == run.run_id


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Marking a session stale multiple times should not corrupt state."""

    def test_second_mark_produces_new_file(self, wiki_root: Path) -> None:
        """Marking the same session stale twice creates two distinct files."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result1 = mark_single_session_stale(inp, wiki_root)
        assert result1.outcome == MarkOutcome.MARKED_STALE

        # Mark again from the same source
        result2 = mark_single_session_stale(inp, wiki_root)
        assert result2.outcome == MarkOutcome.MARKED_STALE
        assert result2.stale_path != result1.stale_path

    def test_original_unchanged_after_multiple_marks(
        self, wiki_root: Path
    ) -> None:
        """The original file is never modified regardless of how many
        times we mark it stale."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)
        original_content = source.read_text(encoding="utf-8")

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        mark_single_session_stale(inp, wiki_root)
        mark_single_session_stale(inp, wiki_root)
        mark_single_session_stale(inp, wiki_root)

        assert source.read_text(encoding="utf-8") == original_content


# ---------------------------------------------------------------------------
# New file naming
# ---------------------------------------------------------------------------


class TestStaleFileNaming:
    """The stale file should have a deterministic, discoverable naming pattern."""

    def test_stale_file_in_same_directory(self, wiki_root: Path) -> None:
        """The new stale file lives in the same directory as the source."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)
        assert result.stale_path is not None
        assert result.stale_path.parent == source.parent

    def test_stale_file_is_markdown(self, wiki_root: Path) -> None:
        """The new stale file has .md extension."""
        run = _write_running_session(wiki_root)
        source = current_run.file_path(wiki_root)

        liveness = _make_liveness_result(
            session_id=run.run_id,
            health=SessionHealth.PROCESS_DEAD,
        )
        inp = StaleMarkerInput(
            liveness_result=liveness,
            source_path=source,
        )

        result = mark_single_session_stale(inp, wiki_root)
        assert result.stale_path is not None
        assert result.stale_path.suffix == ".md"
