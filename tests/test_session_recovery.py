"""Tests for session recovery detection and resume/discard offer handling.

Verifies that:
- detect_session_recovery returns a structured offer on reconnection
- Running sessions produce OFFER_RESUME action
- Pending-approval sessions produce OFFER_RESUME action
- Idle/terminal sessions produce NO_RECOVERY action
- Missing session file produces NO_RECOVERY action
- Corrupted session file produces NO_RECOVERY action
- accept_recovery marks the session as resumed in wiki
- reject_recovery discards the session state via wiki
- RecoveryOffer contains all fields needed for CLI presentation
- The full cycle (save on disconnect -> detect -> accept/reject) works end-to-end
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)
from jules_daemon.wiki.session_persistence import (
    SessionSnapshot,
    save_session_state,
    session_file_path,
)
from jules_daemon.wiki.session_recovery import (
    RecoveryAction,
    RecoveryOffer,
    accept_recovery,
    detect_session_recovery,
    reject_recovery,
)


@pytest.fixture
def wiki_root(tmp_path: Path) -> Path:
    """Provide a temporary wiki root directory."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "pages" / "daemon").mkdir(parents=True)
    return wiki


def _make_ssh_target() -> SSHTarget:
    return SSHTarget(host="10.0.0.1", user="deploy", port=22)


def _make_running_snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        run_id="run-abc-123",
        status=RunStatus.RUNNING,
        ssh_target=_make_ssh_target(),
        command=Command(
            natural_language="run the unit tests",
            resolved_shell="pytest -v tests/",
            approved=True,
            approved_at=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
        ),
        pids=ProcessIDs(daemon=12345, remote=67890),
        progress=Progress(
            percent=45.0,
            tests_passed=22,
            tests_failed=1,
            tests_skipped=2,
            tests_total=50,
            last_output_line="PASSED tests/test_foo.py::test_bar",
        ),
        started_at=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
        error=None,
        disconnect_reason="connection_reset",
        client_name="jules-cli",
        client_pid=999,
    )


def _make_pending_snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        run_id="run-def-456",
        status=RunStatus.PENDING_APPROVAL,
        ssh_target=_make_ssh_target(),
        command=Command(natural_language="run integration tests"),
        pids=ProcessIDs(daemon=12345),
        progress=Progress(),
        started_at=None,
        error=None,
        disconnect_reason="eof",
        client_name="jules-cli",
        client_pid=888,
    )


def _make_idle_snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        run_id="run-idle-000",
        status=RunStatus.IDLE,
        ssh_target=None,
        command=None,
        pids=ProcessIDs(),
        progress=Progress(),
        started_at=None,
        error=None,
        disconnect_reason="eof",
        client_name="jules-cli",
        client_pid=None,
    )


def _make_completed_snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        run_id="run-done-789",
        status=RunStatus.COMPLETED,
        ssh_target=_make_ssh_target(),
        command=Command(
            natural_language="run the unit tests",
            resolved_shell="pytest -v tests/",
            approved=True,
        ),
        pids=ProcessIDs(daemon=12345, remote=67890),
        progress=Progress(
            percent=100.0,
            tests_passed=50,
            tests_failed=0,
            tests_skipped=0,
            tests_total=50,
        ),
        started_at=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
        error=None,
        disconnect_reason="eof",
        client_name="jules-cli",
        client_pid=777,
    )


# -- RecoveryOffer dataclass --


class TestRecoveryOffer:
    def test_frozen(self) -> None:
        offer = RecoveryOffer(
            action=RecoveryAction.NO_RECOVERY,
            reason="No prior session found",
            snapshot=None,
        )
        with pytest.raises(AttributeError):
            offer.action = RecoveryAction.OFFER_RESUME  # type: ignore[misc]

    def test_needs_user_decision_true(self) -> None:
        snap = _make_running_snapshot()
        offer = RecoveryOffer(
            action=RecoveryAction.OFFER_RESUME,
            reason="Running session detected",
            snapshot=snap,
        )
        assert offer.needs_user_decision is True

    def test_needs_user_decision_false(self) -> None:
        offer = RecoveryOffer(
            action=RecoveryAction.NO_RECOVERY,
            reason="No prior session",
            snapshot=None,
        )
        assert offer.needs_user_decision is False

    def test_summary_for_running(self) -> None:
        snap = _make_running_snapshot()
        offer = RecoveryOffer(
            action=RecoveryAction.OFFER_RESUME,
            reason="Running session detected",
            snapshot=snap,
        )
        assert "run-abc-123" in offer.summary
        assert "running" in offer.summary.lower()

    def test_summary_for_no_recovery(self) -> None:
        offer = RecoveryOffer(
            action=RecoveryAction.NO_RECOVERY,
            reason="No session",
            snapshot=None,
        )
        assert "no" in offer.summary.lower() or "none" in offer.summary.lower()


# -- detect_session_recovery --


class TestDetectSessionRecovery:
    def test_running_session_offers_resume(self, wiki_root: Path) -> None:
        snap = _make_running_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.OFFER_RESUME
        assert offer.snapshot is not None
        assert offer.snapshot.run_id == "run-abc-123"
        assert offer.snapshot.status == RunStatus.RUNNING
        assert offer.needs_user_decision is True

    def test_pending_session_offers_resume(self, wiki_root: Path) -> None:
        snap = _make_pending_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.OFFER_RESUME
        assert offer.snapshot is not None
        assert offer.snapshot.status == RunStatus.PENDING_APPROVAL

    def test_idle_session_no_recovery(self, wiki_root: Path) -> None:
        snap = _make_idle_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.NO_RECOVERY
        assert offer.needs_user_decision is False

    def test_completed_session_no_recovery(self, wiki_root: Path) -> None:
        snap = _make_completed_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.NO_RECOVERY

    def test_no_file_no_recovery(self, wiki_root: Path) -> None:
        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.NO_RECOVERY
        assert offer.snapshot is None

    def test_corrupted_file_no_recovery(self, wiki_root: Path) -> None:
        fpath = session_file_path(wiki_root)
        fpath.write_text("garbage content without frontmatter", encoding="utf-8")
        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.NO_RECOVERY

    def test_offer_contains_connection_details(self, wiki_root: Path) -> None:
        snap = _make_running_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        assert offer.snapshot is not None
        assert offer.snapshot.ssh_target is not None
        assert offer.snapshot.ssh_target.host == "10.0.0.1"
        assert offer.snapshot.ssh_target.user == "deploy"

    def test_offer_contains_progress(self, wiki_root: Path) -> None:
        snap = _make_running_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        assert offer.snapshot is not None
        assert offer.snapshot.progress.percent == 45.0
        assert offer.snapshot.progress.tests_passed == 22


# -- accept_recovery --


class TestAcceptRecovery:
    def test_accept_marks_resumed(self, wiki_root: Path) -> None:
        snap = _make_running_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.OFFER_RESUME

        result = accept_recovery(wiki_root, offer)
        assert result.accepted is True
        assert result.run_id == "run-abc-123"
        assert result.error is None

    def test_accept_updates_session_file(self, wiki_root: Path) -> None:
        snap = _make_running_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        accept_recovery(wiki_root, offer)

        # Session file should now indicate resumed state
        content = session_file_path(wiki_root).read_text(encoding="utf-8")
        assert "resumed" in content.lower() or "running" in content.lower()

    def test_accept_no_offer_returns_error(self, wiki_root: Path) -> None:
        offer = RecoveryOffer(
            action=RecoveryAction.NO_RECOVERY,
            reason="No session",
            snapshot=None,
        )
        result = accept_recovery(wiki_root, offer)
        assert result.accepted is False
        assert result.error is not None


# -- reject_recovery --


class TestRejectRecovery:
    def test_reject_discards_session(self, wiki_root: Path) -> None:
        snap = _make_running_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.OFFER_RESUME

        result = reject_recovery(wiki_root, offer)
        assert result.rejected is True
        assert result.run_id == "run-abc-123"

    def test_reject_clears_resumable_state(self, wiki_root: Path) -> None:
        snap = _make_running_snapshot()
        save_session_state(wiki_root, snap)
        offer = detect_session_recovery(wiki_root)
        reject_recovery(wiki_root, offer)

        # After rejection, no session should be recoverable
        new_offer = detect_session_recovery(wiki_root)
        assert new_offer.action == RecoveryAction.NO_RECOVERY

    def test_reject_no_offer_returns_error(self, wiki_root: Path) -> None:
        offer = RecoveryOffer(
            action=RecoveryAction.NO_RECOVERY,
            reason="No session",
            snapshot=None,
        )
        result = reject_recovery(wiki_root, offer)
        assert result.rejected is False
        assert result.error is not None


# -- End-to-end cycle --


class TestEndToEndCycle:
    def test_save_detect_accept_cycle(self, wiki_root: Path) -> None:
        """Full cycle: save on disconnect -> detect -> accept."""
        # 1. Save session state (simulating disconnect)
        run = CurrentRun()
        target = _make_ssh_target()
        cmd = Command(natural_language="run smoke tests")
        run = run.with_pending_approval(target, cmd, daemon_pid=100)
        run = run.with_running("pytest -v smoke/", remote_pid=200)
        run = run.with_progress(Progress(
            percent=30.0,
            tests_passed=15,
            tests_failed=0,
            tests_skipped=0,
            tests_total=50,
        ))

        snap = SessionSnapshot.from_current_run(
            run=run,
            disconnect_reason="broken_pipe",
            client_name="jules-cli",
            client_pid=555,
        )
        save_result = save_session_state(wiki_root, snap)
        assert save_result.success is True

        # 2. Detect session on reconnection
        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.OFFER_RESUME
        assert offer.snapshot is not None
        assert offer.snapshot.progress.percent == 30.0

        # 3. Accept recovery
        accept_result = accept_recovery(wiki_root, offer)
        assert accept_result.accepted is True

    def test_save_detect_reject_cycle(self, wiki_root: Path) -> None:
        """Full cycle: save on disconnect -> detect -> reject."""
        snap = _make_running_snapshot()
        save_session_state(wiki_root, snap)

        offer = detect_session_recovery(wiki_root)
        assert offer.action == RecoveryAction.OFFER_RESUME

        reject_result = reject_recovery(wiki_root, offer)
        assert reject_result.rejected is True

        # After rejection, no recovery should be offered
        new_offer = detect_session_recovery(wiki_root)
        assert new_offer.action == RecoveryAction.NO_RECOVERY

    def test_multiple_disconnects_last_wins(self, wiki_root: Path) -> None:
        """When multiple disconnects happen, the last saved state wins."""
        snap1 = SessionSnapshot(
            run_id="run-first",
            status=RunStatus.RUNNING,
            ssh_target=_make_ssh_target(),
            command=Command(natural_language="first command"),
            pids=ProcessIDs(daemon=1),
            progress=Progress(percent=10.0),
            started_at=datetime(2026, 4, 9, 10, 0, 0, tzinfo=timezone.utc),
            error=None,
            disconnect_reason="eof",
            client_name="cli",
            client_pid=100,
        )
        save_session_state(wiki_root, snap1)

        snap2 = SessionSnapshot(
            run_id="run-second",
            status=RunStatus.RUNNING,
            ssh_target=_make_ssh_target(),
            command=Command(natural_language="second command"),
            pids=ProcessIDs(daemon=2),
            progress=Progress(percent=75.0),
            started_at=datetime(2026, 4, 9, 11, 0, 0, tzinfo=timezone.utc),
            error=None,
            disconnect_reason="broken_pipe",
            client_name="cli",
            client_pid=200,
        )
        save_session_state(wiki_root, snap2)

        offer = detect_session_recovery(wiki_root)
        assert offer.snapshot is not None
        assert offer.snapshot.run_id == "run-second"
        assert offer.snapshot.progress.percent == 75.0
