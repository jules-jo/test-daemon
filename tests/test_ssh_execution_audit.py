"""Tests for SSH execution audit instrumentation.

Verifies that the audit instrumentation wired into the SSH execution stage:
- Records an SSHExecutionRecord with command, host, and outcome
- Advances the AuditRecord to SSH_DISPATCHED stage
- Captures success outcomes with exit code and PID
- Captures failure outcomes with error details
- Uses the StageAudit context manager for chain-based audit trail
- Produces an AuditEntry with "ssh_execution" stage in the chain
- Returns immutable results (AuditRecord is frozen)
- Records timing information (started_at, completed_at, duration)
- Handles SSH errors without raising (captures in audit record)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from jules_daemon.audit.models import (
    AuditRecord,
    ConfirmationDecision,
    ConfirmationRecord,
    NLInputRecord,
    ParsedCommandRecord,
    PipelineStage,
    SSHExecutionRecord,
)
from jules_daemon.audit_models import AuditChain
from jules_daemon.ssh.command import SSHCommand
from jules_daemon.ssh.command_gen import (
    GeneratedCommand,
    RecoveryCommandAction,
    TestFramework,
)
from jules_daemon.ssh.dispatch import DispatchResult, SSHDispatchHandle
from jules_daemon.ssh.execution_audit import (
    AuditedDispatchResult,
    build_ssh_execution_record,
    record_ssh_execution_audit,
)
from jules_daemon.wiki.models import SSHTarget


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeDispatchHandle:
    """Fake SSH dispatch handle for testing."""

    remote_pid: int | None = 42
    error: Exception | None = None
    executed_commands: list[str] | None = None

    def __post_init__(self) -> None:
        if self.executed_commands is None:
            self.executed_commands = []

    async def execute(self, command: str, timeout: int) -> int | None:
        if self.executed_commands is not None:
            self.executed_commands.append(command)
        if self.error is not None:
            raise self.error
        return self.remote_pid

    @property
    def session_id(self) -> str:
        return "fake-session-audit-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audit_record_at_confirmation() -> AuditRecord:
    """Build an AuditRecord advanced to the CONFIRMATION stage."""
    nl = NLInputRecord(
        raw_input="run the full test suite",
        timestamp=_T0,
        source="cli",
    )
    parsed = ParsedCommandRecord(
        natural_language="run the full test suite",
        resolved_shell="pytest -v --tb=short",
        model_id="dataiku-mesh-llm",
        risk_level="low",
        explanation="Run all tests with verbose output",
        affected_paths=("tests/",),
        timestamp=_T0,
    )
    confirmation = ConfirmationRecord(
        decision=ConfirmationDecision.APPROVED,
        original_command="pytest -v --tb=short",
        final_command="pytest -v --tb=short",
        decided_by="operator",
        timestamp=_T0,
    )
    record = AuditRecord.create(run_id="run-audit-test-001", nl_input=nl)
    record = record.with_parsed_command(parsed)
    record = record.with_confirmation(confirmation)
    return record


def _make_ssh_target() -> SSHTarget:
    return SSHTarget(host="prod.example.com", user="ci", port=22)


def _make_dispatch_result_success(
    *,
    command_string: str = "pytest -v --tb=short",
    remote_pid: int = 9999,
    session_id: str = "fake-session-audit-test",
) -> DispatchResult:
    return DispatchResult(
        success=True,
        action=RecoveryCommandAction.RESTART,
        command_string=command_string,
        run_id="run-audit-test-001",
        remote_pid=remote_pid,
        error=None,
        wiki_updated=True,
        session_id=session_id,
    )


def _make_dispatch_result_failure(
    *,
    command_string: str = "pytest -v --tb=short",
    error: str = "SSH dispatch failed: connection refused",
    session_id: str = "fake-session-audit-test",
) -> DispatchResult:
    return DispatchResult(
        success=False,
        action=RecoveryCommandAction.RESTART,
        command_string=command_string,
        run_id="run-audit-test-001",
        remote_pid=None,
        error=error,
        wiki_updated=True,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# build_ssh_execution_record: success
# ---------------------------------------------------------------------------


class TestBuildSSHExecutionRecordSuccess:
    def test_captures_host(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.host == "prod.example.com"

    def test_captures_user(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.user == "ci"

    def test_captures_port(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.port == 22

    def test_captures_command(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_success(command_string="pytest --lf -v")
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.command == "pytest --lf -v"

    def test_captures_session_id(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_success(session_id="session-xyz")
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.session_id == "session-xyz"

    def test_captures_remote_pid(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_success(remote_pid=4242)
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.remote_pid == 4242

    def test_started_at_defaults_to_dispatch_timestamp(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.started_at == result.timestamp

    def test_started_at_override(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        override = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
            started_at=override,
        )
        assert record.started_at == override

    def test_exit_code_is_none_on_dispatch(self) -> None:
        """Exit code is unknown at dispatch time -- set to None."""
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.exit_code is None

    def test_is_frozen(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        with pytest.raises(AttributeError):
            record.host = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# build_ssh_execution_record: failure
# ---------------------------------------------------------------------------


class TestBuildSSHExecutionRecordFailure:
    def test_captures_host_on_failure(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_failure()
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.host == "prod.example.com"

    def test_captures_command_on_failure(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_failure(command_string="pytest -x")
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.command == "pytest -x"

    def test_remote_pid_is_none_on_failure(self) -> None:
        target = _make_ssh_target()
        result = _make_dispatch_result_failure()
        record = build_ssh_execution_record(
            target=target,
            dispatch_result=result,
        )
        assert record.remote_pid is None


# ---------------------------------------------------------------------------
# record_ssh_execution_audit: advances pipeline to SSH_DISPATCHED
# ---------------------------------------------------------------------------


class TestRecordSSHExecutionAudit:
    def test_advances_to_ssh_dispatched_stage(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        assert audited.audit_record.pipeline_stage == PipelineStage.SSH_DISPATCHED

    def test_ssh_execution_record_is_populated(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        assert audited.audit_record.ssh_execution is not None

    def test_preserves_correlation_id(self) -> None:
        audit = _make_audit_record_at_confirmation()
        original_id = audit.correlation_id
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        assert audited.audit_record.correlation_id == original_id

    def test_preserves_earlier_stages(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        assert audited.audit_record.nl_input is not None
        assert audited.audit_record.parsed_command is not None
        assert audited.audit_record.confirmation is not None

    def test_ssh_record_has_correct_host(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        assert audited.audit_record.ssh_execution is not None
        assert audited.audit_record.ssh_execution.host == "prod.example.com"

    def test_ssh_record_has_correct_command(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success(
            command_string="pytest --co -q"
        )

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        assert audited.audit_record.ssh_execution is not None
        assert audited.audit_record.ssh_execution.command == "pytest --co -q"

    def test_records_outcome_success(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success(remote_pid=1234)

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        assert audited.dispatch_result.success is True
        assert audited.audit_record.ssh_execution is not None
        assert audited.audit_record.ssh_execution.remote_pid == 1234

    def test_records_outcome_failure(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_failure(
            error="SSH dispatch failed: timeout"
        )

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        assert audited.dispatch_result.success is False
        assert audited.dispatch_result.error is not None
        assert "timeout" in audited.dispatch_result.error

    def test_original_audit_record_unchanged(self) -> None:
        """Immutability check: original record is not modified."""
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        # Original should still be at CONFIRMATION stage
        assert audit.pipeline_stage == PipelineStage.CONFIRMATION
        assert audit.ssh_execution is None


# ---------------------------------------------------------------------------
# record_ssh_execution_audit: audit chain integration
# ---------------------------------------------------------------------------


class TestRecordSSHExecutionAuditChain:
    def test_appends_entry_to_chain(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        chain = AuditChain.empty()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
            audit_chain=chain,
        )
        assert len(audited.audit_chain) == 1

    def test_chain_entry_has_ssh_execution_stage(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        chain = AuditChain.empty()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
            audit_chain=chain,
        )
        assert audited.audit_chain.latest is not None
        assert audited.audit_chain.latest.stage == "ssh_execution"

    def test_chain_entry_records_success_status(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()
        chain = AuditChain.empty()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
            audit_chain=chain,
        )
        assert audited.audit_chain.latest is not None
        assert audited.audit_chain.latest.status == "success"

    def test_chain_entry_records_error_status(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_failure()
        chain = AuditChain.empty()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
            audit_chain=chain,
        )
        assert audited.audit_chain.latest is not None
        assert audited.audit_chain.latest.status == "error"

    def test_chain_entry_has_error_message_on_failure(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_failure(error="connection refused")
        chain = AuditChain.empty()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
            audit_chain=chain,
        )
        assert audited.audit_chain.latest is not None
        assert audited.audit_chain.latest.error is not None
        assert "connection refused" in audited.audit_chain.latest.error

    def test_chain_preserves_existing_entries(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        # Pre-populate chain with an earlier entry
        from jules_daemon.audit_models import AuditEntry

        earlier = AuditEntry(
            stage="confirmation",
            timestamp=_T0,
            before_snapshot=None,
            after_snapshot=None,
            duration=0.5,
            status="success",
            error=None,
        )
        chain = AuditChain.empty().append(earlier)

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
            audit_chain=chain,
        )
        assert len(audited.audit_chain) == 2
        assert audited.audit_chain.stages == ("confirmation", "ssh_execution")

    def test_default_chain_when_none_provided(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        # Should create a fresh chain with 1 entry
        assert len(audited.audit_chain) == 1


# ---------------------------------------------------------------------------
# AuditedDispatchResult: frozen dataclass
# ---------------------------------------------------------------------------


class TestAuditedDispatchResult:
    def test_is_frozen(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        with pytest.raises(AttributeError):
            audited.dispatch_result = result  # type: ignore[misc]

    def test_has_all_fields(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )
        assert audited.dispatch_result is not None
        assert audited.audit_record is not None
        assert audited.audit_chain is not None
        assert audited.ssh_execution_record is not None


# ---------------------------------------------------------------------------
# Serialization: audit record with SSH execution round-trips
# ---------------------------------------------------------------------------


class TestSSHExecutionAuditSerialization:
    def test_audit_record_round_trips(self) -> None:
        audit = _make_audit_record_at_confirmation()
        target = _make_ssh_target()
        result = _make_dispatch_result_success()

        audited = record_ssh_execution_audit(
            audit_record=audit,
            target=target,
            dispatch_result=result,
        )

        serialized = audited.audit_record.to_dict()
        restored = AuditRecord.from_dict(serialized)

        assert restored.pipeline_stage == PipelineStage.SSH_DISPATCHED
        assert restored.ssh_execution is not None
        assert restored.ssh_execution.host == "prod.example.com"
        assert restored.ssh_execution.command == "pytest -v --tb=short"
        assert restored.ssh_execution.session_id == "fake-session-audit-test"
