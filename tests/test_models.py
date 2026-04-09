"""Tests for wiki state data models."""

from datetime import datetime, timezone

import pytest

from jules_daemon.wiki.models import (
    Command,
    CurrentRun,
    ProcessIDs,
    Progress,
    RunStatus,
    SSHTarget,
)


# -- SSHTarget --


class TestSSHTarget:
    def test_create_with_defaults(self) -> None:
        target = SSHTarget(host="staging.example.com", user="deploy")
        assert target.host == "staging.example.com"
        assert target.user == "deploy"
        assert target.port == 22

    def test_create_with_custom_port(self) -> None:
        target = SSHTarget(host="10.0.0.5", user="root", port=2222)
        assert target.port == 2222

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host must not be empty"):
            SSHTarget(host="", user="deploy")

    def test_empty_user_raises(self) -> None:
        with pytest.raises(ValueError, match="user must not be empty"):
            SSHTarget(host="example.com", user="")

    def test_invalid_port_raises(self) -> None:
        with pytest.raises(ValueError, match="port must be 1-65535"):
            SSHTarget(host="example.com", user="deploy", port=0)
        with pytest.raises(ValueError, match="port must be 1-65535"):
            SSHTarget(host="example.com", user="deploy", port=70000)

    def test_create_with_key_path(self) -> None:
        target = SSHTarget(
            host="staging.example.com",
            user="deploy",
            key_path="/home/deploy/.ssh/id_ed25519",
        )
        assert target.key_path == "/home/deploy/.ssh/id_ed25519"

    def test_default_key_path_is_none(self) -> None:
        target = SSHTarget(host="example.com", user="deploy")
        assert target.key_path is None

    def test_relative_key_path_raises(self) -> None:
        with pytest.raises(ValueError, match="key_path must be an absolute path"):
            SSHTarget(
                host="example.com",
                user="deploy",
                key_path="relative/path/to/key",
            )

    def test_traversal_key_path_raises(self) -> None:
        with pytest.raises(ValueError, match="key_path must be an absolute path"):
            SSHTarget(
                host="example.com",
                user="deploy",
                key_path="../../etc/passwd",
            )

    def test_frozen(self) -> None:
        target = SSHTarget(host="example.com", user="deploy")
        with pytest.raises(AttributeError):
            target.host = "other.com"  # type: ignore[misc]


# -- Command --


class TestCommand:
    def test_create_minimal(self) -> None:
        cmd = Command(natural_language="run the tests")
        assert cmd.natural_language == "run the tests"
        assert cmd.resolved_shell == ""
        assert cmd.approved is False
        assert cmd.approved_at is None

    def test_empty_natural_language_raises(self) -> None:
        with pytest.raises(ValueError, match="Natural language command"):
            Command(natural_language="")

    def test_with_approval(self) -> None:
        cmd = Command(natural_language="run pytest")
        approved = cmd.with_approval("cd /app && pytest -v")
        assert approved.resolved_shell == "cd /app && pytest -v"
        assert approved.approved is True
        assert approved.approved_at is not None
        # Original is unchanged (immutable)
        assert cmd.approved is False
        assert cmd.resolved_shell == ""

    def test_frozen(self) -> None:
        cmd = Command(natural_language="run tests")
        with pytest.raises(AttributeError):
            cmd.approved = True  # type: ignore[misc]


# -- ProcessIDs --


class TestProcessIDs:
    def test_defaults(self) -> None:
        pids = ProcessIDs()
        assert pids.daemon is None
        assert pids.remote is None

    def test_create_with_values(self) -> None:
        pids = ProcessIDs(daemon=1234, remote=5678)
        assert pids.daemon == 1234
        assert pids.remote == 5678


# -- Progress --


class TestProgress:
    def test_defaults(self) -> None:
        prog = Progress()
        assert prog.percent == 0.0
        assert prog.tests_passed == 0
        assert prog.tests_total == 0

    def test_valid_progress(self) -> None:
        now = datetime.now(timezone.utc)
        prog = Progress(
            percent=50.0,
            tests_passed=5,
            tests_failed=1,
            tests_skipped=2,
            tests_total=10,
            last_output_line="PASSED test_login",
            checkpoint_at=now,
        )
        assert prog.percent == 50.0
        assert prog.last_output_line == "PASSED test_login"

    def test_percent_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="percent must be 0-100"):
            Progress(percent=-1.0)
        with pytest.raises(ValueError, match="percent must be 0-100"):
            Progress(percent=101.0)

    def test_negative_count_raises(self) -> None:
        with pytest.raises(ValueError, match="Test counts must not be negative"):
            Progress(tests_passed=-1)


# -- RunStatus --


class TestRunStatus:
    def test_values(self) -> None:
        assert RunStatus.IDLE.value == "idle"
        assert RunStatus.PENDING_APPROVAL.value == "pending_approval"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.CANCELLED.value == "cancelled"


# -- CurrentRun --


class TestCurrentRun:
    def test_default_idle(self) -> None:
        run = CurrentRun()
        assert run.status == RunStatus.IDLE
        assert run.ssh_target is None
        assert run.command is None
        assert run.is_active is False
        assert run.is_terminal is False

    def test_has_run_id(self) -> None:
        run = CurrentRun()
        assert run.run_id  # non-empty UUID string
        assert len(run.run_id) == 36  # UUID format

    def test_unique_run_ids(self) -> None:
        run1 = CurrentRun()
        run2 = CurrentRun()
        assert run1.run_id != run2.run_id

    def test_transition_to_pending_approval(self) -> None:
        run = CurrentRun()
        target = SSHTarget(host="staging.example.com", user="deploy")
        cmd = Command(natural_language="run pytest on staging")
        pending = run.with_pending_approval(target, cmd, daemon_pid=9999)

        assert pending.status == RunStatus.PENDING_APPROVAL
        assert pending.ssh_target == target
        assert pending.command == cmd
        assert pending.pids.daemon == 9999
        assert pending.is_active is True
        assert pending.is_terminal is False
        # Original unchanged
        assert run.status == RunStatus.IDLE

    def test_transition_to_running(self) -> None:
        run = CurrentRun()
        target = SSHTarget(host="staging.example.com", user="deploy")
        cmd = Command(natural_language="run pytest")
        pending = run.with_pending_approval(target, cmd, daemon_pid=1000)
        running = pending.with_running("pytest -v --tb=short", remote_pid=5555)

        assert running.status == RunStatus.RUNNING
        assert running.command is not None
        assert running.command.approved is True
        assert running.command.resolved_shell == "pytest -v --tb=short"
        assert running.pids.remote == 5555
        assert running.started_at is not None
        assert running.is_active is True

    def test_running_without_command_raises(self) -> None:
        run = CurrentRun()
        with pytest.raises(ValueError, match="Cannot start running without a command"):
            run.with_running("pytest")

    def test_transition_to_completed(self) -> None:
        run = CurrentRun()
        target = SSHTarget(host="prod.example.com", user="ci")
        cmd = Command(natural_language="run full suite")
        running = run.with_pending_approval(target, cmd, daemon_pid=100)
        running = running.with_running("pytest", remote_pid=200)
        final_progress = Progress(
            percent=100.0,
            tests_passed=50,
            tests_failed=0,
            tests_total=50,
        )
        completed = running.with_completed(final_progress)

        assert completed.status == RunStatus.COMPLETED
        assert completed.progress.tests_passed == 50
        assert completed.completed_at is not None
        assert completed.is_terminal is True
        assert completed.is_active is False

    def test_transition_to_failed(self) -> None:
        run = CurrentRun()
        target = SSHTarget(host="dev.example.com", user="tester")
        cmd = Command(natural_language="run smoke tests")
        running = run.with_pending_approval(target, cmd, daemon_pid=100)
        running = running.with_running("pytest -m smoke")
        failed_progress = Progress(tests_passed=3, tests_failed=2, tests_total=5)
        failed = running.with_failed("Connection reset by peer", failed_progress)

        assert failed.status == RunStatus.FAILED
        assert failed.error == "Connection reset by peer"
        assert failed.is_terminal is True

    def test_transition_to_cancelled(self) -> None:
        run = CurrentRun()
        target = SSHTarget(host="dev.example.com", user="ci")
        cmd = Command(natural_language="run tests")
        pending = run.with_pending_approval(target, cmd, daemon_pid=100)
        cancelled = pending.with_cancelled()

        assert cancelled.status == RunStatus.CANCELLED
        assert cancelled.completed_at is not None
        assert cancelled.is_terminal is True

    def test_with_progress(self) -> None:
        run = CurrentRun()
        target = SSHTarget(host="host", user="user")
        cmd = Command(natural_language="run tests")
        running = run.with_pending_approval(target, cmd, daemon_pid=1)
        running = running.with_running("pytest")

        new_progress = Progress(
            percent=45.0,
            tests_passed=9,
            tests_failed=0,
            tests_total=20,
            last_output_line="PASSED test_checkout",
        )
        updated = running.with_progress(new_progress)

        assert updated.progress.percent == 45.0
        assert updated.progress.last_output_line == "PASSED test_checkout"
        # Original unchanged
        assert running.progress.percent == 0.0

    def test_frozen(self) -> None:
        run = CurrentRun()
        with pytest.raises(AttributeError):
            run.status = RunStatus.RUNNING  # type: ignore[misc]
