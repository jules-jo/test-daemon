"""Integration tests for the daemon-startup crash recovery wiring.

Covers the ``jules_daemon.startup.crash_recovery_wire`` module which
connects the existing crash-recovery machinery
(``wiki.crash_recovery`` + ``wiki.recovery_orchestrator``) to the
daemon's ``__main__._run_daemon()`` startup sequence.

These are integration tests in the sense that they:
- Build real wiki files on disk with ``current_run.write``
- Invoke the real ``try_crash_recovery`` entry point
- Verify the ``RecoveredRunInfo`` shape and the resulting wiki state

SSH is not involved -- the orchestrator is invoked without an
``SSHConnector`` which matches the production call site in
``__main__._run_daemon()``. The orchestrator's graceful FAIL branch
handles that case by marking the run FAILED.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.ipc.request_handler import (
    RequestHandler,
    RequestHandlerConfig,
)
from jules_daemon.startup.crash_recovery_wire import (
    DEFAULT_RECOVERY_DEADLINE_SECONDS,
    RecoveredRunInfo,
    try_crash_recovery,
)
from jules_daemon.wiki import current_run
from jules_daemon.wiki.layout import initialize_wiki
from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    Progress,
    RunStatus,
    SSHTarget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _make_running_run(
    *,
    host: str = "ci.example.com",
    user: str = "runner",
    port: int = 22,
    natural_language: str = "run the integration tests",
    resolved_shell: str = "pytest -v tests/",
    daemon_pid: int = 11111,
    remote_pid: int = 22222,
    percent: float = 40.0,
) -> CurrentRun:
    """Build a realistic in-flight RUNNING state."""
    target = SSHTarget(host=host, user=user, port=port)
    cmd = Command(natural_language=natural_language)
    pending = CurrentRun().with_pending_approval(
        target, cmd, daemon_pid=daemon_pid,
    )
    running = pending.with_running(resolved_shell, remote_pid=remote_pid)
    return running.with_progress(
        Progress(
            percent=percent,
            tests_passed=20,
            tests_failed=0,
            tests_skipped=1,
            tests_total=50,
            last_output_line="tests/test_thing.py::test_x PASSED",
            checkpoint_at=_now_utc(),
        )
    )


def _make_pending_approval_run(
    *,
    host: str = "ci.example.com",
    user: str = "runner",
) -> CurrentRun:
    """Build a PENDING_APPROVAL state (approval prompt in flight)."""
    target = SSHTarget(host=host, user=user)
    cmd = Command(natural_language="run smoke tests")
    return CurrentRun().with_pending_approval(
        target, cmd, daemon_pid=33333,
    )


@pytest.fixture
def wiki_dir(tmp_path: Path) -> Path:
    """Provide an initialized wiki directory."""
    root = tmp_path / "wiki"
    initialize_wiki(root)
    return root


# ---------------------------------------------------------------------------
# try_crash_recovery: no interrupted run scenarios
# ---------------------------------------------------------------------------


class TestTryCrashRecoveryNoRun:
    """When there is nothing to recover, the wire returns None."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_wiki_file(
        self, tmp_path: Path
    ) -> None:
        # No wiki directory at all -- should not raise, should return None.
        missing = tmp_path / "does-not-exist"
        result = await try_crash_recovery(missing)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_wiki_idle(
        self, wiki_dir: Path
    ) -> None:
        current_run.clear(wiki_dir)
        result = await try_crash_recovery(wiki_dir)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_run_already_completed(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        completed = run.with_completed(run.progress)
        current_run.write(wiki_dir, completed)

        result = await try_crash_recovery(wiki_dir)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_run_already_failed(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        failed = run.with_failed("pre-existing failure", run.progress)
        current_run.write(wiki_dir, failed)

        result = await try_crash_recovery(wiki_dir)
        assert result is None


# ---------------------------------------------------------------------------
# try_crash_recovery: RUNNING state recovery
# ---------------------------------------------------------------------------


class TestTryCrashRecoveryRunning:
    """When a RUNNING run is detected, the wire reconciles it."""

    @pytest.mark.asyncio
    async def test_returns_recovered_run_info(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None
        assert isinstance(recovered, RecoveredRunInfo)
        assert recovered.run_id == run.run_id

    @pytest.mark.asyncio
    async def test_result_is_marked_failure(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None
        # A recovered run is always surfaced as FAILED because we
        # cannot know whether the tests actually passed while the
        # daemon was down.
        assert recovered.result.success is False
        assert recovered.status_label == "FAILED"

    @pytest.mark.asyncio
    async def test_result_has_daemon_was_down_marker(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None
        assert recovered.result.error is not None
        assert "daemon was down" in recovered.result.error.lower()

    @pytest.mark.asyncio
    async def test_result_preserves_ssh_target(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run(host="h1.example.com", user="alice")
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None
        assert recovered.result.target_host == "h1.example.com"
        assert recovered.result.target_user == "alice"

    @pytest.mark.asyncio
    async def test_result_preserves_command(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run(
            natural_language="run unit tests",
            resolved_shell="pytest tests/unit",
        )
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None
        # The resolved shell is preferred over the natural language
        # because it is what actually ran (or was about to run).
        assert recovered.result.command == "pytest tests/unit"

    @pytest.mark.asyncio
    async def test_notification_contains_banner(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None
        assert "DAEMON RECOVERED FROM CRASH" in recovered.notification
        assert run.run_id in recovered.notification
        assert "FAILED" in recovered.notification

    @pytest.mark.asyncio
    async def test_wiki_current_run_reset_to_idle(
        self, wiki_dir: Path
    ) -> None:
        """After recovery the wiki is back to idle so new runs can start."""
        run = _make_running_run()
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None

        # The wiki current-run should be reset to IDLE (either by the
        # orchestrator's FAILED path via promote, or by our explicit
        # mark-and-promote fallback).
        after = current_run.read(wiki_dir)
        assert after is not None
        assert after.status == RunStatus.IDLE

    @pytest.mark.asyncio
    async def test_recovery_completes_within_sla(
        self, wiki_dir: Path
    ) -> None:
        """Recovery must complete in under 30 seconds (SLA)."""
        run = _make_running_run()
        current_run.write(wiki_dir, run)

        wall_start = time.monotonic()
        recovered = await try_crash_recovery(wiki_dir)
        wall_elapsed = time.monotonic() - wall_start

        assert recovered is not None
        assert wall_elapsed < DEFAULT_RECOVERY_DEADLINE_SECONDS, (
            f"Recovery took {wall_elapsed:.2f}s, exceeding "
            f"{DEFAULT_RECOVERY_DEADLINE_SECONDS:.0f}s SLA"
        )

    @pytest.mark.asyncio
    async def test_recovery_complete_flag_set(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None
        assert recovered.recovery_complete is True


# ---------------------------------------------------------------------------
# try_crash_recovery: PENDING_APPROVAL state recovery
# ---------------------------------------------------------------------------


class TestTryCrashRecoveryPendingApproval:
    """When a PENDING_APPROVAL run is detected, the wire reconciles it."""

    @pytest.mark.asyncio
    async def test_returns_recovered_run_info(
        self, wiki_dir: Path
    ) -> None:
        run = _make_pending_approval_run()
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None
        assert isinstance(recovered, RecoveredRunInfo)
        assert recovered.run_id == run.run_id

    @pytest.mark.asyncio
    async def test_wiki_reset_to_idle(
        self, wiki_dir: Path
    ) -> None:
        run = _make_pending_approval_run()
        current_run.write(wiki_dir, run)

        await try_crash_recovery(wiki_dir)

        after = current_run.read(wiki_dir)
        assert after is not None
        assert after.status == RunStatus.IDLE

    @pytest.mark.asyncio
    async def test_error_references_pending_approval(
        self, wiki_dir: Path
    ) -> None:
        run = _make_pending_approval_run()
        current_run.write(wiki_dir, run)

        recovered = await try_crash_recovery(wiki_dir)

        assert recovered is not None
        assert recovered.result.error is not None
        err = recovered.result.error.lower()
        # The error should surface that the approval was interrupted
        # and the user should re-submit.
        assert "daemon was down" in err
        assert "approval" in err or "re-submit" in err


# ---------------------------------------------------------------------------
# try_crash_recovery: robustness
# ---------------------------------------------------------------------------


class TestTryCrashRecoveryRobustness:
    """try_crash_recovery never raises."""

    @pytest.mark.asyncio
    async def test_never_raises_on_missing_path(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "completely-missing"
        # Should not raise even though the directory does not exist.
        await try_crash_recovery(path)

    @pytest.mark.asyncio
    async def test_never_raises_on_corrupted_wiki(
        self, wiki_dir: Path
    ) -> None:
        # Write garbage to the current-run file
        file_path = current_run.file_path(wiki_dir)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("this is not valid yaml frontmatter")

        # Should not raise -- either return None (fresh start) or a
        # RecoveredRunInfo describing the degraded recovery.
        result = await try_crash_recovery(wiki_dir)
        # A corrupted file is treated as FRESH_START by detect_crash_recovery
        assert result is None

    @pytest.mark.asyncio
    async def test_deadline_parameter_respected(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        current_run.write(wiki_dir, run)

        # With a tiny deadline the recovery might time out or still
        # complete very fast, but it must not raise and must return
        # a RecoveredRunInfo.
        recovered = await try_crash_recovery(
            wiki_dir, deadline_seconds=2.0,
        )
        assert recovered is not None


# ---------------------------------------------------------------------------
# RequestHandler seeding
# ---------------------------------------------------------------------------


class TestRequestHandlerSeeding:
    """The RecoveredRunInfo can be assigned to RequestHandler fields."""

    @pytest.mark.asyncio
    async def test_last_completed_run_set(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        current_run.write(wiki_dir, run)
        recovered = await try_crash_recovery(wiki_dir)
        assert recovered is not None

        handler = RequestHandler(
            config=RequestHandlerConfig(wiki_root=wiki_dir),
        )
        handler._last_completed_run = recovered.result

        assert handler._last_completed_run is not None
        assert handler._last_completed_run.run_id == run.run_id
        assert handler._last_completed_run.success is False

    @pytest.mark.asyncio
    async def test_last_failure_set(
        self, wiki_dir: Path
    ) -> None:
        run = _make_running_run()
        current_run.write(wiki_dir, run)
        recovered = await try_crash_recovery(wiki_dir)
        assert recovered is not None

        handler = RequestHandler(
            config=RequestHandlerConfig(wiki_root=wiki_dir),
        )
        handler._last_failure = recovered.notification

        assert handler._last_failure is not None
        assert "DAEMON RECOVERED FROM CRASH" in handler._last_failure

    @pytest.mark.asyncio
    async def test_next_handshake_surfaces_failure(
        self, wiki_dir: Path
    ) -> None:
        """Handshake should relay the recovery banner to the CLI."""
        run = _make_running_run()
        current_run.write(wiki_dir, run)
        recovered = await try_crash_recovery(wiki_dir)
        assert recovered is not None

        handler = RequestHandler(
            config=RequestHandlerConfig(wiki_root=wiki_dir),
        )
        handler._last_failure = recovered.notification

        envelope = handler._handle_handshake("msg-1", {})
        assert envelope.payload.get("pending_failure") is not None
        assert "DAEMON RECOVERED FROM CRASH" in envelope.payload[
            "pending_failure"
        ]
        # After delivery, _last_failure is cleared
        assert handler._last_failure is None

    @pytest.mark.asyncio
    async def test_next_status_reports_recovered_run(
        self, wiki_dir: Path
    ) -> None:
        """Status query should surface the recovered run as completed."""
        run = _make_running_run()
        current_run.write(wiki_dir, run)
        recovered = await try_crash_recovery(wiki_dir)
        assert recovered is not None

        handler = RequestHandler(
            config=RequestHandlerConfig(wiki_root=wiki_dir),
        )
        handler._last_completed_run = recovered.result

        envelope = handler._handle_status("msg-2", {})
        assert envelope.payload.get("state") == "completed"
        assert envelope.payload.get("status") == "FAILED"
        assert envelope.payload.get("run_id") == run.run_id
        assert "daemon was down" in (envelope.payload.get("error") or "").lower()
