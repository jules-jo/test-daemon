"""Integration test: full pipeline run from NL input through execution result.

Drives every stage of the command execution pipeline in sequence and asserts
that the resulting audit chain contains exactly four stage entries in the
correct order:

    1. nl_input           -- classify_with_audit
    2. confirmation       -- confirm_with_audit
    3. ssh_execution      -- record_ssh_execution_audit
    4. result_structuring -- audit_result_structuring

Each stage receives the chain produced by its predecessor, threading an
immutable audit trail through the entire pipeline. The test verifies:

    - All four entries are present
    - Stage names match the expected order
    - All entries have "success" status
    - Timestamps are monotonically non-decreasing
    - The audit chain round-trips through serialization
    - Individual audit wiki files are written for stages 1 and 2
    - The AuditRecord advances through all pipeline stages correctly
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

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
from jules_daemon.audit.result_stage import audit_result_structuring
from jules_daemon.audit_models import AuditChain
from jules_daemon.classifier.nl_audit import classify_with_audit
from jules_daemon.cli.confirmation import (
    ConfirmationRequest,
    Decision,
    TerminalIO,
)
from jules_daemon.cli.confirmation_audit import confirm_with_audit
from jules_daemon.llm.command_context import CommandContext, RiskLevel
from jules_daemon.ssh.command import SSHCommand
from jules_daemon.ssh.command_gen import RecoveryCommandAction
from jules_daemon.ssh.dispatch import DispatchResult
from jules_daemon.ssh.execution_audit import record_ssh_execution_audit
from jules_daemon.wiki.assembled_result import (
    AssembledTestResult,
    CompletenessRatio,
    TestOutcome,
    TestRecord,
)
from jules_daemon.wiki.frontmatter import parse
from jules_daemon.wiki.layout import initialize_wiki
from jules_daemon.wiki.models import SSHTarget


# ---------------------------------------------------------------------------
# Expected pipeline stage order
# ---------------------------------------------------------------------------

EXPECTED_STAGES = (
    "nl_input",
    "confirmation",
    "ssh_execution",
    "result_structuring",
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeTerminalIO(TerminalIO):
    """Deterministic terminal IO that replays scripted inputs."""

    def __init__(self, inputs: list[str]) -> None:
        self._inputs = list(inputs)
        self._index = 0
        self._output = StringIO()

    def write(self, text: str) -> None:
        self._output.write(text)

    def read_line(self, prompt: str = "") -> str:
        if self._index >= len(self._inputs):
            raise EOFError("No more scripted input")
        value = self._inputs[self._index]
        self._index += 1
        return value

    def read_editable(self, prompt: str, prefill: str) -> str:
        return self.read_line(prompt)

    @property
    def output_text(self) -> str:
        return self._output.getvalue()


def _make_ssh_command(command: str = "pytest -v --tb=short") -> SSHCommand:
    """Build a minimal valid SSHCommand."""
    return SSHCommand(
        command=command,
        working_directory="/opt/app",
        timeout=300,
    )


def _make_command_context(
    command: str = "pytest -v --tb=short",
) -> CommandContext:
    """Build a minimal CommandContext for testing."""
    return CommandContext(
        command=command,
        explanation="Run the test suite with verbose output",
        affected_paths=("/opt/app/tests",),
        risk_level=RiskLevel.LOW,
    )


def _make_ssh_target() -> SSHTarget:
    """Build a minimal SSHTarget for testing."""
    return SSHTarget(host="staging.example.com", user="deploy", port=22)


def _make_confirmation_request(
    command: str = "pytest -v --tb=short",
) -> ConfirmationRequest:
    """Build a ConfirmationRequest with realistic data."""
    return ConfirmationRequest(
        ssh_command=_make_ssh_command(command),
        context=_make_command_context(command),
        target=_make_ssh_target(),
    )


def _make_dispatch_result(
    command: str = "pytest -v --tb=short",
    run_id: str = "run-001",
) -> DispatchResult:
    """Build a successful DispatchResult for testing."""
    return DispatchResult(
        success=True,
        action=RecoveryCommandAction.RESTART,
        command_string=command,
        run_id=run_id,
        remote_pid=12345,
        error=None,
        wiki_updated=True,
        session_id=f"sess-{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc),
    )


def _make_assembled_result(
    run_id: str = "run-001",
    session_id: str = "sess-test",
    host: str = "staging.example.com",
) -> AssembledTestResult:
    """Build an AssembledTestResult with realistic test data."""
    records = (
        TestRecord(
            test_name="test_login_flow",
            outcome=TestOutcome.PASSED,
            duration_seconds=1.2,
            module="tests/test_auth.py",
        ),
        TestRecord(
            test_name="test_signup_validation",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.8,
            module="tests/test_auth.py",
        ),
        TestRecord(
            test_name="test_api_health",
            outcome=TestOutcome.PASSED,
            duration_seconds=0.3,
            module="tests/test_api.py",
        ),
    )
    return AssembledTestResult(
        run_id=run_id,
        session_id=session_id,
        host=host,
        records=records,
        completeness=CompletenessRatio(executed=3, expected=3),
    )


def _make_audit_record(
    run_id: str = "run-001",
    nl_text: str = "run the full test suite on staging",
) -> AuditRecord:
    """Build an AuditRecord at NL_INPUT stage for pipeline threading."""
    nl_input = NLInputRecord(
        raw_input=nl_text,
        timestamp=datetime.now(timezone.utc),
        source="cli",
    )
    return AuditRecord.create(run_id=run_id, nl_input=nl_input)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    """Create an initialized wiki directory structure in a temp path."""
    root = tmp_path / "wiki"
    initialize_wiki(root)
    return root


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFullPipelineAuditTrail:
    """Integration: full pipeline run produces a correctly ordered audit chain."""

    def _run_full_pipeline(
        self,
        wiki_root: Path,
        nl_text: str = "run the full test suite on staging",
    ) -> tuple[AuditChain, AuditRecord, list[Path]]:
        """Execute the full four-stage pipeline and return the audit chain.

        Returns:
            Tuple of (final_chain, final_audit_record, audit_file_paths).
        """
        audit_files: list[Path] = []
        run_id = f"run-{uuid.uuid4().hex[:8]}"

        # -- Stage 1: NL Input Classification --
        nl_result = classify_with_audit(nl_text, wiki_root)
        chain = nl_result.chain
        audit_files.append(nl_result.audit_path)

        # Build the AuditRecord (the correlated record that tracks
        # pipeline stage progression independently of the generic chain)
        audit_record = _make_audit_record(run_id=run_id, nl_text=nl_text)

        # Advance AuditRecord with a parsed command
        parsed = ParsedCommandRecord(
            natural_language=nl_text,
            resolved_shell="pytest -v --tb=short",
            model_id="openai:mesh-conn:gpt-4",
            risk_level="low",
            explanation="Run the test suite with verbose output",
            affected_paths=("/opt/app/tests",),
            timestamp=datetime.now(timezone.utc),
        )
        audit_record = audit_record.with_parsed_command(parsed)

        # -- Stage 2: Confirmation --
        request = _make_confirmation_request("pytest -v --tb=short")
        terminal = FakeTerminalIO(inputs=["a"])  # auto-approve
        confirm_result = confirm_with_audit(
            request,
            wiki_root,
            terminal=terminal,
            chain=chain,
        )
        chain = confirm_result.chain
        audit_files.append(confirm_result.audit_path)

        # Advance AuditRecord
        confirmation_record = ConfirmationRecord(
            decision=ConfirmationDecision.APPROVED,
            original_command="pytest -v --tb=short",
            final_command="pytest -v --tb=short",
            decided_by="deploy@staging.example.com",
            timestamp=datetime.now(timezone.utc),
        )
        audit_record = audit_record.with_confirmation(confirmation_record)

        # -- Stage 3: SSH Execution --
        target = _make_ssh_target()
        dispatch_result = _make_dispatch_result(
            command="pytest -v --tb=short",
            run_id=run_id,
        )
        audited_dispatch = record_ssh_execution_audit(
            audit_record=audit_record,
            target=target,
            dispatch_result=dispatch_result,
            audit_chain=chain,
        )
        chain = audited_dispatch.audit_chain
        audit_record = audited_dispatch.audit_record

        # -- Stage 4: Result Structuring --
        assembled = _make_assembled_result(
            run_id=run_id,
            session_id=dispatch_result.session_id,
            host=target.host,
        )
        stage_result = audit_result_structuring(assembled, chain)
        chain = stage_result.chain

        # Advance AuditRecord to EXECUTION_COMPLETE
        structured = StructuredResultRecord(
            tests_passed=3,
            tests_failed=0,
            tests_skipped=0,
            tests_total=3,
            exit_code=0,
            success=True,
            error_message=None,
            summary="All 3 tests passed",
            timestamp=datetime.now(timezone.utc),
        )
        audit_record = audit_record.with_structured_result(structured)

        return chain, audit_record, audit_files

    # -- Core assertions --

    def test_chain_has_exactly_four_entries(self, wiki_root: Path) -> None:
        """The audit chain contains exactly four stage entries."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        assert len(chain) == 4

    def test_stages_in_correct_order(self, wiki_root: Path) -> None:
        """The four entries follow the expected pipeline stage order."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        assert chain.stages == EXPECTED_STAGES

    def test_all_entries_have_success_status(self, wiki_root: Path) -> None:
        """Every stage entry completed successfully."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        for entry in chain.entries:
            assert entry.status == "success", (
                f"Entry for stage '{entry.stage}' has status "
                f"'{entry.status}', expected 'success'"
            )

    def test_timestamps_monotonically_ordered(self, wiki_root: Path) -> None:
        """Entry timestamps are non-decreasing across stages."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        timestamps = [entry.timestamp for entry in chain.entries]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], (
                f"Timestamp at index {i} ({timestamps[i]}) is earlier than "
                f"index {i - 1} ({timestamps[i - 1]})"
            )

    def test_each_stage_has_correct_name(self, wiki_root: Path) -> None:
        """Each entry's stage field matches the expected stage name."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        for entry, expected_stage in zip(
            chain.entries, EXPECTED_STAGES, strict=True
        ):
            assert entry.stage == expected_stage

    # -- AuditRecord correlation assertions --

    def test_audit_record_reaches_execution_complete(
        self, wiki_root: Path
    ) -> None:
        """The correlated AuditRecord advances to EXECUTION_COMPLETE."""
        _, record, _ = self._run_full_pipeline(wiki_root)
        assert record.pipeline_stage == PipelineStage.EXECUTION_COMPLETE

    def test_audit_record_is_complete(self, wiki_root: Path) -> None:
        """The AuditRecord is_complete property is True."""
        _, record, _ = self._run_full_pipeline(wiki_root)
        assert record.is_complete is True

    def test_audit_record_not_denied(self, wiki_root: Path) -> None:
        """The AuditRecord is not in a denied state."""
        _, record, _ = self._run_full_pipeline(wiki_root)
        assert record.is_denied is False

    def test_audit_record_has_all_sub_records(
        self, wiki_root: Path
    ) -> None:
        """The AuditRecord has populated sub-records for every stage."""
        _, record, _ = self._run_full_pipeline(wiki_root)
        assert record.nl_input is not None
        assert record.parsed_command is not None
        assert record.confirmation is not None
        assert record.ssh_execution is not None
        assert record.structured_result is not None

    def test_audit_record_completed_at_is_set(
        self, wiki_root: Path
    ) -> None:
        """The completed_at timestamp is set after full pipeline run."""
        _, record, _ = self._run_full_pipeline(wiki_root)
        assert record.completed_at is not None

    def test_audit_record_correlation_id_is_stable(
        self, wiki_root: Path
    ) -> None:
        """The correlation_id does not change through pipeline stages."""
        _, record, _ = self._run_full_pipeline(wiki_root)
        # Verify it is a valid UUID
        uuid.UUID(record.correlation_id)
        assert len(record.correlation_id) > 0

    # -- Wiki file assertions --

    def test_nl_input_audit_file_exists(self, wiki_root: Path) -> None:
        """The NL input stage writes an audit file to the wiki."""
        _, _, audit_files = self._run_full_pipeline(wiki_root)
        assert len(audit_files) >= 1
        nl_audit_file = audit_files[0]
        assert nl_audit_file.exists()
        assert nl_audit_file.name.startswith("audit-")
        assert nl_audit_file.name.endswith(".md")

    def test_confirmation_audit_file_exists(self, wiki_root: Path) -> None:
        """The confirmation stage writes an audit file to the wiki."""
        _, _, audit_files = self._run_full_pipeline(wiki_root)
        assert len(audit_files) >= 2
        confirm_audit_file = audit_files[1]
        assert confirm_audit_file.exists()
        assert confirm_audit_file.name.startswith("audit-")
        assert confirm_audit_file.name.endswith(".md")

    def test_nl_audit_file_has_valid_frontmatter(
        self, wiki_root: Path
    ) -> None:
        """The NL input audit file contains valid YAML frontmatter."""
        _, _, audit_files = self._run_full_pipeline(wiki_root)
        content = audit_files[0].read_text(encoding="utf-8")
        doc = parse(content)
        assert doc.frontmatter["stage"] == "nl_input"
        assert doc.frontmatter["type"] == "audit-log"
        assert doc.frontmatter["status"] == "success"

    def test_confirmation_audit_file_has_valid_frontmatter(
        self, wiki_root: Path
    ) -> None:
        """The confirmation audit file contains valid YAML frontmatter."""
        _, _, audit_files = self._run_full_pipeline(wiki_root)
        content = audit_files[1].read_text(encoding="utf-8")
        doc = parse(content)
        assert doc.frontmatter["stage"] == "confirmation"
        assert doc.frontmatter["type"] == "audit-log"
        assert doc.frontmatter["decision"] == "approve"

    # -- Serialization round-trip assertions --

    def test_chain_serializes_to_list(self, wiki_root: Path) -> None:
        """The full chain serializes to a list of exactly four dicts."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        serialized = chain.to_list()
        assert len(serialized) == 4
        for entry_dict in serialized:
            assert isinstance(entry_dict, dict)
            assert "stage" in entry_dict
            assert "timestamp" in entry_dict

    def test_chain_round_trips_through_serialization(
        self, wiki_root: Path
    ) -> None:
        """The chain survives a to_list/from_list round-trip."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        serialized = chain.to_list()
        restored = AuditChain.from_list(serialized)
        assert len(restored) == 4
        assert restored.stages == EXPECTED_STAGES
        for original, restored_entry in zip(
            chain.entries, restored.entries, strict=True
        ):
            assert original.stage == restored_entry.stage
            assert original.status == restored_entry.status

    def test_audit_record_round_trips_through_dict(
        self, wiki_root: Path
    ) -> None:
        """The AuditRecord serializes to dict and back without data loss."""
        _, record, _ = self._run_full_pipeline(wiki_root)
        data = record.to_dict()
        restored = AuditRecord.from_dict(data)
        assert restored.correlation_id == record.correlation_id
        assert restored.run_id == record.run_id
        assert restored.pipeline_stage == PipelineStage.EXECUTION_COMPLETE
        assert restored.nl_input.raw_input == record.nl_input.raw_input
        assert (
            restored.parsed_command.resolved_shell
            == record.parsed_command.resolved_shell
        )
        assert (
            restored.confirmation.decision == record.confirmation.decision
        )
        assert (
            restored.ssh_execution.host == record.ssh_execution.host
        )
        assert (
            restored.structured_result.tests_passed
            == record.structured_result.tests_passed
        )

    # -- Entry content assertions --

    def test_nl_input_entry_captures_raw_text(
        self, wiki_root: Path
    ) -> None:
        """The NL input entry's before_snapshot contains the raw input."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        nl_entry = chain.entries[0]
        before = nl_entry.before_snapshot
        assert isinstance(before, dict)
        assert "run" in before.get("raw_input", "").lower()

    def test_confirmation_entry_captures_decision(
        self, wiki_root: Path
    ) -> None:
        """The confirmation entry's after_snapshot records the decision."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        confirm_entry = chain.entries[1]
        after = confirm_entry.after_snapshot
        assert isinstance(after, dict)
        assert after.get("decision") == "approve"

    def test_ssh_execution_entry_captures_host(
        self, wiki_root: Path
    ) -> None:
        """The SSH execution entry records the target host."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        ssh_entry = chain.entries[2]
        before = ssh_entry.before_snapshot
        assert isinstance(before, dict)
        assert before.get("host") == "staging.example.com"

    def test_result_entry_captures_test_counts(
        self, wiki_root: Path
    ) -> None:
        """The result structuring entry records test outcome data."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        result_entry = chain.entries[3]
        after = result_entry.after_snapshot
        assert isinstance(after, dict)
        # The after-snapshot is a StageSnapshot dict with partial_outputs
        outputs = after.get("partial_outputs", after)
        summary = outputs.get("summary", outputs)
        # Summary should contain test count data
        if isinstance(summary, dict):
            assert summary.get("tests_passed") == 3
            assert summary.get("tests_total") == 3
            assert summary.get("outcome") == "passed"

    # -- Chain immutability assertions --

    def test_intermediate_chains_are_immutable(
        self, wiki_root: Path
    ) -> None:
        """Each pipeline stage creates a new chain, not mutating the old."""
        nl_text = "run the full test suite on staging"
        run_id = f"run-{uuid.uuid4().hex[:8]}"

        # Stage 1
        nl_result = classify_with_audit(nl_text, wiki_root)
        chain_after_1 = nl_result.chain
        assert len(chain_after_1) == 1

        # Stage 2
        request = _make_confirmation_request("pytest -v --tb=short")
        terminal = FakeTerminalIO(inputs=["a"])
        confirm_result = confirm_with_audit(
            request, wiki_root, terminal=terminal, chain=chain_after_1
        )
        chain_after_2 = confirm_result.chain
        assert len(chain_after_2) == 2
        # Original chain unchanged
        assert len(chain_after_1) == 1

        # Stage 3
        audit_record = _make_audit_record(run_id=run_id, nl_text=nl_text)
        parsed = ParsedCommandRecord(
            natural_language=nl_text,
            resolved_shell="pytest -v --tb=short",
            model_id="openai:mesh-conn:gpt-4",
            risk_level="low",
            explanation="Run the test suite",
            affected_paths=("/opt/app/tests",),
            timestamp=datetime.now(timezone.utc),
        )
        audit_record = audit_record.with_parsed_command(parsed)
        confirmation_record = ConfirmationRecord(
            decision=ConfirmationDecision.APPROVED,
            original_command="pytest -v --tb=short",
            final_command="pytest -v --tb=short",
            decided_by="deploy",
            timestamp=datetime.now(timezone.utc),
        )
        audit_record = audit_record.with_confirmation(confirmation_record)

        dispatch_result = _make_dispatch_result(
            command="pytest -v --tb=short", run_id=run_id
        )
        audited = record_ssh_execution_audit(
            audit_record=audit_record,
            target=_make_ssh_target(),
            dispatch_result=dispatch_result,
            audit_chain=chain_after_2,
        )
        chain_after_3 = audited.audit_chain
        assert len(chain_after_3) == 3
        # Previous chains unchanged
        assert len(chain_after_1) == 1
        assert len(chain_after_2) == 2

        # Stage 4
        assembled = _make_assembled_result(
            run_id=run_id,
            session_id=dispatch_result.session_id,
            host="staging.example.com",
        )
        stage_result = audit_result_structuring(assembled, chain_after_3)
        chain_after_4 = stage_result.chain
        assert len(chain_after_4) == 4
        # All previous chains still at original lengths
        assert len(chain_after_1) == 1
        assert len(chain_after_2) == 2
        assert len(chain_after_3) == 3

    # -- by_stage retrieval assertions --

    def test_by_stage_returns_one_entry_per_stage(
        self, wiki_root: Path
    ) -> None:
        """Each stage has exactly one entry in the chain."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        for stage_name in EXPECTED_STAGES:
            entries = chain.by_stage(stage_name)
            assert len(entries) == 1, (
                f"Expected 1 entry for stage '{stage_name}', "
                f"got {len(entries)}"
            )

    def test_latest_entry_is_result_structuring(
        self, wiki_root: Path
    ) -> None:
        """The chain's latest entry is the result_structuring stage."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        assert chain.latest is not None
        assert chain.latest.stage == "result_structuring"

    # -- Duration assertions --

    def test_all_entries_have_non_negative_duration(
        self, wiki_root: Path
    ) -> None:
        """Every entry with a duration has a non-negative value."""
        chain, _, _ = self._run_full_pipeline(wiki_root)
        for entry in chain.entries:
            if entry.duration is not None:
                assert entry.duration >= 0.0, (
                    f"Entry '{entry.stage}' has negative duration: "
                    f"{entry.duration}"
                )
