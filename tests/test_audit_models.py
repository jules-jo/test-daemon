"""Tests for audit record data model.

Covers every pipeline stage sub-model, the top-level AuditRecord,
immutable update methods, correlation ID linking, serialization,
and validation constraints.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from jules_daemon.audit.models import (
    AuditRecord,
    ConfirmationDecision,
    ConfirmationRecord,
    NLInputRecord,
    ParsedCommandRecord,
    PipelineStage,
    SSHExecutionRecord,
    StructuredResultRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
_LATER = datetime(2026, 4, 9, 12, 5, 0, tzinfo=timezone.utc)
_EVEN_LATER = datetime(2026, 4, 9, 12, 10, 0, tzinfo=timezone.utc)


def _make_nl_input(
    raw_input: str = "run the full test suite on staging",
    timestamp: datetime = _NOW,
    source: str = "cli",
) -> NLInputRecord:
    return NLInputRecord(
        raw_input=raw_input,
        timestamp=timestamp,
        source=source,
    )


def _make_parsed_command(
    natural_language: str = "run the full test suite on staging",
    resolved_shell: str = "cd /opt/app && pytest -v --tb=short",
    model_id: str = "openai:mesh-conn:gpt-4",
    risk_level: str = "medium",
    explanation: str = "Runs the pytest test suite with verbose output",
    affected_paths: tuple[str, ...] = ("/opt/app/tests",),
    timestamp: datetime = _NOW,
) -> ParsedCommandRecord:
    return ParsedCommandRecord(
        natural_language=natural_language,
        resolved_shell=resolved_shell,
        model_id=model_id,
        risk_level=risk_level,
        explanation=explanation,
        affected_paths=affected_paths,
        timestamp=timestamp,
    )


def _make_confirmation(
    decision: ConfirmationDecision = ConfirmationDecision.APPROVED,
    original_command: str = "cd /opt/app && pytest -v --tb=short",
    final_command: str = "cd /opt/app && pytest -v --tb=short",
    decided_by: str = "human",
    timestamp: datetime = _NOW,
) -> ConfirmationRecord:
    return ConfirmationRecord(
        decision=decision,
        original_command=original_command,
        final_command=final_command,
        decided_by=decided_by,
        timestamp=timestamp,
    )


def _make_ssh_execution(
    host: str = "staging.example.com",
    user: str = "deploy",
    port: int = 22,
    command: str = "cd /opt/app && pytest -v --tb=short",
    session_id: str = "sess-001",
    started_at: datetime = _NOW,
    remote_pid: int | None = 12345,
    completed_at: datetime | None = None,
    exit_code: int | None = None,
    duration_seconds: float | None = None,
) -> SSHExecutionRecord:
    return SSHExecutionRecord(
        host=host,
        user=user,
        port=port,
        command=command,
        session_id=session_id,
        started_at=started_at,
        remote_pid=remote_pid,
        completed_at=completed_at,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
    )


def _make_structured_result(
    tests_passed: int = 42,
    tests_failed: int = 3,
    tests_skipped: int = 5,
    tests_total: int = 50,
    exit_code: int = 1,
    success: bool = False,
    error_message: str | None = "3 tests failed",
    summary: str = "42 passed, 3 failed, 5 skipped",
    timestamp: datetime = _EVEN_LATER,
) -> StructuredResultRecord:
    return StructuredResultRecord(
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_skipped=tests_skipped,
        tests_total=tests_total,
        exit_code=exit_code,
        success=success,
        error_message=error_message,
        summary=summary,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# PipelineStage enum
# ---------------------------------------------------------------------------


class TestPipelineStage:
    def test_all_stages_present(self) -> None:
        expected = {
            "nl_input",
            "command_parsed",
            "confirmation",
            "ssh_dispatched",
            "execution_complete",
        }
        actual = {stage.value for stage in PipelineStage}
        assert actual == expected

    def test_stage_ordering(self) -> None:
        stages = list(PipelineStage)
        assert stages[0] == PipelineStage.NL_INPUT
        assert stages[-1] == PipelineStage.EXECUTION_COMPLETE


# ---------------------------------------------------------------------------
# ConfirmationDecision enum
# ---------------------------------------------------------------------------


class TestConfirmationDecision:
    def test_all_decisions_present(self) -> None:
        expected = {"approved", "denied", "edited"}
        actual = {d.value for d in ConfirmationDecision}
        assert actual == expected


# ---------------------------------------------------------------------------
# NLInputRecord
# ---------------------------------------------------------------------------


class TestNLInputRecord:
    def test_create_valid(self) -> None:
        record = _make_nl_input()
        assert record.raw_input == "run the full test suite on staging"
        assert record.timestamp == _NOW
        assert record.source == "cli"

    def test_frozen(self) -> None:
        record = _make_nl_input()
        with pytest.raises(AttributeError):
            record.raw_input = "changed"  # type: ignore[misc]

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="raw_input must not be empty"):
            NLInputRecord(raw_input="", timestamp=_NOW, source="cli")

    def test_whitespace_only_input_raises(self) -> None:
        with pytest.raises(ValueError, match="raw_input must not be empty"):
            NLInputRecord(raw_input="   ", timestamp=_NOW, source="cli")

    def test_empty_source_raises(self) -> None:
        with pytest.raises(ValueError, match="source must not be empty"):
            NLInputRecord(
                raw_input="run tests", timestamp=_NOW, source=""
            )


# ---------------------------------------------------------------------------
# ParsedCommandRecord
# ---------------------------------------------------------------------------


class TestParsedCommandRecord:
    def test_create_valid(self) -> None:
        record = _make_parsed_command()
        assert record.natural_language == "run the full test suite on staging"
        assert record.resolved_shell == "cd /opt/app && pytest -v --tb=short"
        assert record.model_id == "openai:mesh-conn:gpt-4"
        assert record.risk_level == "medium"
        assert record.affected_paths == ("/opt/app/tests",)

    def test_frozen(self) -> None:
        record = _make_parsed_command()
        with pytest.raises(AttributeError):
            record.resolved_shell = "new command"  # type: ignore[misc]

    def test_empty_natural_language_raises(self) -> None:
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            ParsedCommandRecord(
                natural_language="",
                resolved_shell="pytest",
                model_id="m",
                risk_level="low",
                explanation="runs tests",
                affected_paths=(),
                timestamp=_NOW,
            )

    def test_empty_resolved_shell_raises(self) -> None:
        with pytest.raises(ValueError, match="resolved_shell must not be empty"):
            ParsedCommandRecord(
                natural_language="run tests",
                resolved_shell="",
                model_id="m",
                risk_level="low",
                explanation="runs tests",
                affected_paths=(),
                timestamp=_NOW,
            )

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="model_id must not be empty"):
            ParsedCommandRecord(
                natural_language="run tests",
                resolved_shell="pytest",
                model_id="",
                risk_level="low",
                explanation="runs tests",
                affected_paths=(),
                timestamp=_NOW,
            )

    def test_invalid_risk_level_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_level must be one of"):
            ParsedCommandRecord(
                natural_language="run tests",
                resolved_shell="pytest",
                model_id="m",
                risk_level="extreme",
                explanation="runs tests",
                affected_paths=(),
                timestamp=_NOW,
            )


# ---------------------------------------------------------------------------
# ConfirmationRecord
# ---------------------------------------------------------------------------


class TestConfirmationRecord:
    def test_create_approved(self) -> None:
        record = _make_confirmation()
        assert record.decision == ConfirmationDecision.APPROVED
        assert record.original_command == record.final_command

    def test_create_edited(self) -> None:
        record = _make_confirmation(
            decision=ConfirmationDecision.EDITED,
            final_command="cd /opt/app && pytest -v --tb=long",
        )
        assert record.decision == ConfirmationDecision.EDITED
        assert record.original_command != record.final_command

    def test_create_denied(self) -> None:
        record = _make_confirmation(decision=ConfirmationDecision.DENIED)
        assert record.decision == ConfirmationDecision.DENIED

    def test_frozen(self) -> None:
        record = _make_confirmation()
        with pytest.raises(AttributeError):
            record.decision = ConfirmationDecision.DENIED  # type: ignore[misc]

    def test_empty_original_command_raises(self) -> None:
        with pytest.raises(ValueError, match="original_command must not be empty"):
            ConfirmationRecord(
                decision=ConfirmationDecision.APPROVED,
                original_command="",
                final_command="pytest",
                decided_by="human",
                timestamp=_NOW,
            )

    def test_empty_decided_by_raises(self) -> None:
        with pytest.raises(ValueError, match="decided_by must not be empty"):
            ConfirmationRecord(
                decision=ConfirmationDecision.APPROVED,
                original_command="pytest",
                final_command="pytest",
                decided_by="",
                timestamp=_NOW,
            )


# ---------------------------------------------------------------------------
# SSHExecutionRecord
# ---------------------------------------------------------------------------


class TestSSHExecutionRecord:
    def test_create_valid(self) -> None:
        record = _make_ssh_execution()
        assert record.host == "staging.example.com"
        assert record.user == "deploy"
        assert record.port == 22
        assert record.remote_pid == 12345
        assert record.exit_code is None
        assert not record.is_complete

    def test_complete_execution(self) -> None:
        record = _make_ssh_execution(
            completed_at=_LATER,
            exit_code=0,
            duration_seconds=300.5,
        )
        assert record.is_complete
        assert record.is_success
        assert record.duration_seconds == 300.5

    def test_failed_execution(self) -> None:
        record = _make_ssh_execution(
            completed_at=_LATER,
            exit_code=1,
            duration_seconds=120.0,
        )
        assert record.is_complete
        assert not record.is_success

    def test_frozen(self) -> None:
        record = _make_ssh_execution()
        with pytest.raises(AttributeError):
            record.host = "other"  # type: ignore[misc]

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host must not be empty"):
            SSHExecutionRecord(
                host="",
                user="deploy",
                port=22,
                command="pytest",
                session_id="s1",
                started_at=_NOW,
            )

    def test_empty_user_raises(self) -> None:
        with pytest.raises(ValueError, match="user must not be empty"):
            SSHExecutionRecord(
                host="staging",
                user="",
                port=22,
                command="pytest",
                session_id="s1",
                started_at=_NOW,
            )

    def test_invalid_port_raises(self) -> None:
        with pytest.raises(ValueError, match="port must be 1-65535"):
            SSHExecutionRecord(
                host="staging",
                user="deploy",
                port=0,
                command="pytest",
                session_id="s1",
                started_at=_NOW,
            )

    def test_empty_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id must not be empty"):
            SSHExecutionRecord(
                host="staging",
                user="deploy",
                port=22,
                command="pytest",
                session_id="",
                started_at=_NOW,
            )

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_seconds must not be negative"):
            SSHExecutionRecord(
                host="staging",
                user="deploy",
                port=22,
                command="pytest",
                session_id="s1",
                started_at=_NOW,
                duration_seconds=-1.0,
            )


# ---------------------------------------------------------------------------
# StructuredResultRecord
# ---------------------------------------------------------------------------


class TestStructuredResultRecord:
    def test_create_valid(self) -> None:
        record = _make_structured_result()
        assert record.tests_passed == 42
        assert record.tests_failed == 3
        assert record.tests_total == 50
        assert not record.success
        assert record.error_message == "3 tests failed"

    def test_successful_result(self) -> None:
        record = _make_structured_result(
            tests_passed=50,
            tests_failed=0,
            exit_code=0,
            success=True,
            error_message=None,
        )
        assert record.success
        assert record.error_message is None

    def test_frozen(self) -> None:
        record = _make_structured_result()
        with pytest.raises(AttributeError):
            record.tests_passed = 100  # type: ignore[misc]

    def test_negative_tests_passed_raises(self) -> None:
        with pytest.raises(ValueError, match="tests_passed must not be negative"):
            StructuredResultRecord(
                tests_passed=-1,
                tests_failed=0,
                tests_skipped=0,
                tests_total=0,
                exit_code=0,
                success=True,
                error_message=None,
                summary="",
                timestamp=_NOW,
            )

    def test_negative_tests_failed_raises(self) -> None:
        with pytest.raises(ValueError, match="tests_failed must not be negative"):
            StructuredResultRecord(
                tests_passed=0,
                tests_failed=-1,
                tests_skipped=0,
                tests_total=0,
                exit_code=0,
                success=True,
                error_message=None,
                summary="",
                timestamp=_NOW,
            )

    def test_negative_tests_skipped_raises(self) -> None:
        with pytest.raises(ValueError, match="tests_skipped must not be negative"):
            StructuredResultRecord(
                tests_passed=0,
                tests_failed=0,
                tests_skipped=-1,
                tests_total=0,
                exit_code=0,
                success=True,
                error_message=None,
                summary="",
                timestamp=_NOW,
            )

    def test_negative_tests_total_raises(self) -> None:
        with pytest.raises(ValueError, match="tests_total must not be negative"):
            StructuredResultRecord(
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                tests_total=-1,
                exit_code=0,
                success=True,
                error_message=None,
                summary="",
                timestamp=_NOW,
            )


# ---------------------------------------------------------------------------
# AuditRecord -- creation and correlation
# ---------------------------------------------------------------------------


class TestAuditRecordCreation:
    def test_create_at_nl_input_stage(self) -> None:
        nl = _make_nl_input()
        record = AuditRecord.create(
            run_id="run-001",
            nl_input=nl,
        )
        assert record.pipeline_stage == PipelineStage.NL_INPUT
        assert record.nl_input == nl
        assert record.parsed_command is None
        assert record.confirmation is None
        assert record.ssh_execution is None
        assert record.structured_result is None
        assert record.run_id == "run-001"
        # correlation_id is a valid UUID string
        uuid.UUID(record.correlation_id)

    def test_correlation_id_is_unique(self) -> None:
        nl = _make_nl_input()
        r1 = AuditRecord.create(run_id="run-001", nl_input=nl)
        r2 = AuditRecord.create(run_id="run-002", nl_input=nl)
        assert r1.correlation_id != r2.correlation_id

    def test_created_at_populated(self) -> None:
        nl = _make_nl_input()
        record = AuditRecord.create(run_id="run-001", nl_input=nl)
        assert record.created_at is not None
        assert record.created_at.tzinfo is not None

    def test_frozen(self) -> None:
        nl = _make_nl_input()
        record = AuditRecord.create(run_id="run-001", nl_input=nl)
        with pytest.raises(AttributeError):
            record.run_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AuditRecord -- immutable stage transitions
# ---------------------------------------------------------------------------


class TestAuditRecordTransitions:
    def test_with_parsed_command(self) -> None:
        nl = _make_nl_input()
        record = AuditRecord.create(run_id="run-001", nl_input=nl)
        parsed = _make_parsed_command()
        updated = record.with_parsed_command(parsed)

        assert updated.pipeline_stage == PipelineStage.COMMAND_PARSED
        assert updated.parsed_command == parsed
        # Original unchanged
        assert record.pipeline_stage == PipelineStage.NL_INPUT
        assert record.parsed_command is None
        # Correlation preserved
        assert updated.correlation_id == record.correlation_id

    def test_with_confirmation(self) -> None:
        nl = _make_nl_input()
        record = (
            AuditRecord.create(run_id="run-001", nl_input=nl)
            .with_parsed_command(_make_parsed_command())
            .with_confirmation(_make_confirmation())
        )
        assert record.pipeline_stage == PipelineStage.CONFIRMATION
        assert record.confirmation is not None
        assert record.confirmation.decision == ConfirmationDecision.APPROVED

    def test_with_ssh_execution(self) -> None:
        nl = _make_nl_input()
        ssh = _make_ssh_execution()
        record = (
            AuditRecord.create(run_id="run-001", nl_input=nl)
            .with_parsed_command(_make_parsed_command())
            .with_confirmation(_make_confirmation())
            .with_ssh_execution(ssh)
        )
        assert record.pipeline_stage == PipelineStage.SSH_DISPATCHED
        assert record.ssh_execution == ssh

    def test_with_structured_result(self) -> None:
        nl = _make_nl_input()
        result = _make_structured_result()
        ssh = _make_ssh_execution(
            completed_at=_LATER,
            exit_code=1,
            duration_seconds=300.0,
        )
        record = (
            AuditRecord.create(run_id="run-001", nl_input=nl)
            .with_parsed_command(_make_parsed_command())
            .with_confirmation(_make_confirmation())
            .with_ssh_execution(ssh)
            .with_structured_result(result)
        )
        assert record.pipeline_stage == PipelineStage.EXECUTION_COMPLETE
        assert record.structured_result == result
        assert record.completed_at is not None

    def test_full_chain_preserves_all_stages(self) -> None:
        nl = _make_nl_input()
        parsed = _make_parsed_command()
        confirmation = _make_confirmation()
        ssh = _make_ssh_execution()
        result = _make_structured_result()

        record = (
            AuditRecord.create(run_id="run-001", nl_input=nl)
            .with_parsed_command(parsed)
            .with_confirmation(confirmation)
            .with_ssh_execution(ssh)
            .with_structured_result(result)
        )

        assert record.nl_input == nl
        assert record.parsed_command == parsed
        assert record.confirmation == confirmation
        assert record.ssh_execution == ssh
        assert record.structured_result == result
        assert record.correlation_id is not None

    def test_correlation_id_preserved_through_chain(self) -> None:
        nl = _make_nl_input()
        r1 = AuditRecord.create(run_id="run-001", nl_input=nl)
        r2 = r1.with_parsed_command(_make_parsed_command())
        r3 = r2.with_confirmation(_make_confirmation())
        r4 = r3.with_ssh_execution(_make_ssh_execution())
        r5 = r4.with_structured_result(_make_structured_result())

        cid = r1.correlation_id
        assert r2.correlation_id == cid
        assert r3.correlation_id == cid
        assert r4.correlation_id == cid
        assert r5.correlation_id == cid

    def test_immutability_through_chain(self) -> None:
        nl = _make_nl_input()
        r1 = AuditRecord.create(run_id="run-001", nl_input=nl)
        r2 = r1.with_parsed_command(_make_parsed_command())

        # r1 is untouched
        assert r1.pipeline_stage == PipelineStage.NL_INPUT
        assert r1.parsed_command is None
        # r2 advanced
        assert r2.pipeline_stage == PipelineStage.COMMAND_PARSED
        assert r2.parsed_command is not None


# ---------------------------------------------------------------------------
# AuditRecord -- computed properties
# ---------------------------------------------------------------------------


class TestAuditRecordProperties:
    def test_is_complete_false_when_partial(self) -> None:
        nl = _make_nl_input()
        record = AuditRecord.create(run_id="run-001", nl_input=nl)
        assert not record.is_complete

    def test_is_complete_true_when_all_stages(self) -> None:
        record = (
            AuditRecord.create(run_id="run-001", nl_input=_make_nl_input())
            .with_parsed_command(_make_parsed_command())
            .with_confirmation(_make_confirmation())
            .with_ssh_execution(_make_ssh_execution())
            .with_structured_result(_make_structured_result())
        )
        assert record.is_complete

    def test_is_denied_when_confirmation_denied(self) -> None:
        record = (
            AuditRecord.create(run_id="run-001", nl_input=_make_nl_input())
            .with_parsed_command(_make_parsed_command())
            .with_confirmation(
                _make_confirmation(decision=ConfirmationDecision.DENIED)
            )
        )
        assert record.is_denied

    def test_is_denied_false_when_approved(self) -> None:
        record = (
            AuditRecord.create(run_id="run-001", nl_input=_make_nl_input())
            .with_parsed_command(_make_parsed_command())
            .with_confirmation(
                _make_confirmation(decision=ConfirmationDecision.APPROVED)
            )
        )
        assert not record.is_denied

    def test_is_denied_false_before_confirmation(self) -> None:
        record = AuditRecord.create(
            run_id="run-001", nl_input=_make_nl_input()
        )
        assert not record.is_denied


# ---------------------------------------------------------------------------
# AuditRecord -- serialization
# ---------------------------------------------------------------------------


class TestAuditRecordSerialization:
    def test_to_dict_roundtrip(self) -> None:
        nl = _make_nl_input()
        record = (
            AuditRecord.create(run_id="run-001", nl_input=nl)
            .with_parsed_command(_make_parsed_command())
            .with_confirmation(_make_confirmation())
            .with_ssh_execution(_make_ssh_execution())
            .with_structured_result(_make_structured_result())
        )
        data = record.to_dict()

        assert data["correlation_id"] == record.correlation_id
        assert data["run_id"] == "run-001"
        assert data["pipeline_stage"] == "execution_complete"
        assert data["nl_input"]["raw_input"] == nl.raw_input
        assert data["parsed_command"]["resolved_shell"] == "cd /opt/app && pytest -v --tb=short"
        assert data["confirmation"]["decision"] == "approved"
        assert data["ssh_execution"]["host"] == "staging.example.com"
        assert data["structured_result"]["tests_passed"] == 42

    def test_to_dict_partial_record(self) -> None:
        record = AuditRecord.create(
            run_id="run-001", nl_input=_make_nl_input()
        )
        data = record.to_dict()
        assert data["parsed_command"] is None
        assert data["confirmation"] is None
        assert data["ssh_execution"] is None
        assert data["structured_result"] is None

    def test_from_dict_roundtrip(self) -> None:
        original = (
            AuditRecord.create(run_id="run-001", nl_input=_make_nl_input())
            .with_parsed_command(_make_parsed_command())
            .with_confirmation(_make_confirmation())
            .with_ssh_execution(_make_ssh_execution())
            .with_structured_result(_make_structured_result())
        )
        data = original.to_dict()
        restored = AuditRecord.from_dict(data)

        assert restored.correlation_id == original.correlation_id
        assert restored.run_id == original.run_id
        assert restored.pipeline_stage == original.pipeline_stage
        assert restored.nl_input == original.nl_input
        assert restored.parsed_command == original.parsed_command
        assert restored.confirmation == original.confirmation
        assert restored.ssh_execution == original.ssh_execution
        assert restored.structured_result == original.structured_result

    def test_from_dict_with_completed_ssh(self) -> None:
        """Ensures completed_at string deserialization in SSHExecutionRecord."""
        ssh = _make_ssh_execution(
            completed_at=_LATER,
            exit_code=0,
            duration_seconds=300.0,
        )
        original = (
            AuditRecord.create(run_id="run-001", nl_input=_make_nl_input())
            .with_parsed_command(_make_parsed_command())
            .with_confirmation(_make_confirmation())
            .with_ssh_execution(ssh)
            .with_structured_result(_make_structured_result())
        )
        data = original.to_dict()
        restored = AuditRecord.from_dict(data)

        assert restored.ssh_execution is not None
        assert restored.ssh_execution.completed_at == _LATER
        assert restored.ssh_execution.exit_code == 0
        assert restored.ssh_execution.duration_seconds == 300.0

    def test_from_dict_partial_record(self) -> None:
        original = AuditRecord.create(
            run_id="run-001", nl_input=_make_nl_input()
        )
        data = original.to_dict()
        restored = AuditRecord.from_dict(data)

        assert restored.correlation_id == original.correlation_id
        assert restored.parsed_command is None
        assert restored.confirmation is None


# ---------------------------------------------------------------------------
# Validation edge cases
# ---------------------------------------------------------------------------


class TestValidationEdgeCases:
    def test_nl_input_strips_whitespace(self) -> None:
        record = NLInputRecord(
            raw_input="  run tests  ",
            timestamp=_NOW,
            source="cli",
        )
        assert record.raw_input == "run tests"

    def test_parsed_command_valid_risk_levels(self) -> None:
        for level in ("low", "medium", "high", "critical"):
            record = ParsedCommandRecord(
                natural_language="run tests",
                resolved_shell="pytest",
                model_id="m",
                risk_level=level,
                explanation="runs tests",
                affected_paths=(),
                timestamp=_NOW,
            )
            assert record.risk_level == level

    def test_ssh_execution_port_boundaries(self) -> None:
        # Port 1 -- valid
        r1 = SSHExecutionRecord(
            host="h", user="u", port=1, command="c",
            session_id="s", started_at=_NOW,
        )
        assert r1.port == 1

        # Port 65535 -- valid
        r2 = SSHExecutionRecord(
            host="h", user="u", port=65535, command="c",
            session_id="s", started_at=_NOW,
        )
        assert r2.port == 65535

        # Port 65536 -- invalid
        with pytest.raises(ValueError, match="port must be 1-65535"):
            SSHExecutionRecord(
                host="h", user="u", port=65536, command="c",
                session_id="s", started_at=_NOW,
            )
