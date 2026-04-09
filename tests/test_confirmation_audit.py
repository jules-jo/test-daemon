"""Tests for confirmation stage audit instrumentation.

Verifies that confirm_with_audit():
1. Records an AuditEntry capturing the user's confirm/edit/reject action
2. Writes the audit entry to the wiki filesystem as a markdown file
3. Captures correct stage name, before/after snapshots, and decision
4. Appends to an existing AuditChain when provided
5. Handles all three user actions: approve, reject, edit-then-approve
6. Records the original and final command in the audit trail
7. Handles EOF (reject) correctly
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from jules_daemon.audit_models import AuditChain, AuditEntry
from jules_daemon.cli.confirmation import (
    ConfirmationRequest,
    Decision,
    TerminalIO,
)
from jules_daemon.cli.confirmation_audit import (
    ConfirmationAuditResult,
    confirm_with_audit,
)
from jules_daemon.llm.command_context import CommandContext, RiskLevel
from jules_daemon.ssh.command import SSHCommand
from jules_daemon.wiki.frontmatter import parse as parse_frontmatter
from jules_daemon.wiki.models import SSHTarget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wiki_root(tmp_path: Path) -> Path:
    """Create a minimal wiki directory structure for audit tests."""
    audit_dir = tmp_path / "wiki" / "pages" / "daemon" / "audit"
    audit_dir.mkdir(parents=True)
    return tmp_path / "wiki"


class FakeTerminalIO(TerminalIO):
    """Fake terminal IO for testing confirmation flows."""

    def __init__(self, inputs: list[str]) -> None:
        self._inputs = list(inputs)
        self._input_index = 0
        self._output = StringIO()

    def write(self, text: str) -> None:
        self._output.write(text)

    def read_line(self, prompt: str = "") -> str:
        if self._input_index >= len(self._inputs):
            raise EOFError("No more test input available")
        self._output.write(prompt)
        value = self._inputs[self._input_index]
        self._input_index += 1
        return value

    def read_editable(self, prompt: str, prefill: str) -> str:
        if self._input_index >= len(self._inputs):
            raise EOFError("No more test input available")
        self._output.write(prompt)
        value = self._inputs[self._input_index]
        self._input_index += 1
        if value == "":
            return prefill
        return value

    @property
    def output_text(self) -> str:
        return self._output.getvalue()


def _make_context(
    *,
    command: str = "pytest -v",
    explanation: str = "Run the test suite with verbose output",
    risk_level: RiskLevel = RiskLevel.LOW,
    affected_paths: tuple[str, ...] = ("/opt/app/tests",),
    risk_factors: tuple[str, ...] = (),
    safe_to_execute: bool = True,
) -> CommandContext:
    return CommandContext(
        command=command,
        explanation=explanation,
        risk_level=risk_level,
        affected_paths=affected_paths,
        risk_factors=risk_factors,
        safe_to_execute=safe_to_execute,
    )


def _make_ssh_command(
    *,
    command: str = "pytest -v",
    working_directory: str | None = "/opt/app",
    timeout: int = 300,
) -> SSHCommand:
    return SSHCommand(
        command=command,
        working_directory=working_directory,
        timeout=timeout,
    )


def _make_target(
    *,
    host: str = "test.example.com",
    user: str = "deploy",
    port: int = 22,
) -> SSHTarget:
    return SSHTarget(host=host, user=user, port=port)


def _make_request(
    *,
    ssh_command: SSHCommand | None = None,
    context: CommandContext | None = None,
    target: SSHTarget | None = None,
) -> ConfirmationRequest:
    return ConfirmationRequest(
        ssh_command=ssh_command or _make_ssh_command(),
        context=context or _make_context(),
        target=target if target is not None else _make_target(),
    )


def _fixed_ts() -> datetime:
    """Return a fixed UTC timestamp for test entries."""
    return datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Core: returns ConfirmationAuditResult
# ---------------------------------------------------------------------------


class TestConfirmationAuditResultType:
    """confirm_with_audit returns the correct result type."""

    def test_returns_confirmation_audit_result(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert isinstance(result, ConfirmationAuditResult)

    def test_result_has_confirmation(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert result.confirmation is not None
        assert result.confirmation.decision == Decision.APPROVE

    def test_result_has_entry(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert result.entry is not None

    def test_result_has_chain(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert result.chain is not None

    def test_result_has_audit_path(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert result.audit_path is not None
        assert result.audit_path.exists()


# ---------------------------------------------------------------------------
# Audit entry content correctness
# ---------------------------------------------------------------------------


class TestAuditEntryContent:
    """Verify the AuditEntry has correct stage and snapshot data."""

    def test_entry_stage_is_confirmation(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert result.entry.stage == "confirmation"

    def test_entry_status_is_success(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert result.entry.status == "success"

    def test_entry_before_snapshot_contains_original_command(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        before = result.entry.before_snapshot
        assert isinstance(before, dict)
        assert before["original_command"] == "pytest -v"

    def test_entry_before_snapshot_contains_risk_level(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(
                context=_make_context(risk_level=RiskLevel.HIGH),
            ),
            wiki_root,
            terminal=io,
        )
        before = result.entry.before_snapshot
        assert isinstance(before, dict)
        assert before["risk_level"] == "high"

    def test_entry_after_snapshot_contains_decision_approve(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert isinstance(after, dict)
        assert after["decision"] == "approve"

    def test_entry_after_snapshot_contains_decision_reject(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=["r"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert isinstance(after, dict)
        assert after["decision"] == "reject"

    def test_entry_after_snapshot_contains_decision_edited(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=[
            "e",
            "pytest -v --tb=short",
            "a",
        ])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert isinstance(after, dict)
        assert after["decision"] == "edited"

    def test_entry_after_snapshot_contains_was_edited_flag(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=[
            "e",
            "pytest -v --tb=short",
            "a",
        ])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert after["was_edited"] is True

    def test_entry_after_snapshot_contains_final_command(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=[
            "e",
            "pytest -v --tb=short",
            "a",
        ])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert after["final_command"] == "pytest -v --tb=short"

    def test_entry_has_non_negative_duration(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert result.entry.duration is not None
        assert result.entry.duration >= 0.0

    def test_entry_has_timestamp(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert result.entry.timestamp is not None


# ---------------------------------------------------------------------------
# Decision mapping: approve / reject / edit
# ---------------------------------------------------------------------------


class TestDecisionMapping:
    """Verify the correct decision label is recorded for each user action."""

    def test_approve_records_approve(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert after["decision"] == "approve"
        assert after["was_edited"] is False

    def test_reject_records_reject(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["n"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert after["decision"] == "reject"
        assert after["was_edited"] is False

    def test_edit_then_approve_records_edited(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=[
            "e",
            "pytest -x",
            "a",
        ])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert after["decision"] == "edited"
        assert after["was_edited"] is True
        assert after["final_command"] == "pytest -x"

    def test_edit_then_reject_records_reject(self, wiki_root: Path) -> None:
        """Even if the user edited, a reject decision is 'reject'."""
        io = FakeTerminalIO(inputs=[
            "e",
            "pytest -x",
            "r",
        ])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert after["decision"] == "reject"
        assert after["was_edited"] is True

    def test_eof_records_reject(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=[])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        after = result.entry.after_snapshot
        assert after["decision"] == "reject"


# ---------------------------------------------------------------------------
# Wiki file written
# ---------------------------------------------------------------------------


class TestAuditFileWritten:
    """Confirm that confirm_with_audit writes an audit markdown file."""

    def test_writes_audit_file(self, wiki_root: Path) -> None:
        audit_dir = wiki_root / "pages" / "daemon" / "audit"
        io = FakeTerminalIO(inputs=["a"])

        confirm_with_audit(_make_request(), wiki_root, terminal=io)

        audit_files = list(audit_dir.glob("audit-*.md"))
        assert len(audit_files) == 1

    def test_file_has_yaml_frontmatter(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter is not None
        assert isinstance(doc.frontmatter, dict)

    def test_frontmatter_has_type_audit_log(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter["type"] == "audit-log"

    def test_frontmatter_has_stage_confirmation(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter["stage"] == "confirmation"

    def test_frontmatter_has_decision(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["r"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert doc.frontmatter["decision"] == "reject"

    def test_frontmatter_has_event_id(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert "event_id" in doc.frontmatter
        assert len(doc.frontmatter["event_id"]) > 0

    def test_body_contains_confirmation_heading(
        self, wiki_root: Path
    ) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        content = result.audit_path.read_text(encoding="utf-8")
        doc = parse_frontmatter(content)
        assert "# Confirmation Audit" in doc.body


# ---------------------------------------------------------------------------
# Chain threading
# ---------------------------------------------------------------------------


class TestChainThreading:
    """Verify that an existing AuditChain is correctly extended."""

    def test_default_chain_has_one_entry(self, wiki_root: Path) -> None:
        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
        )
        assert len(result.chain) == 1

    def test_appends_to_provided_chain(self, wiki_root: Path) -> None:
        existing_entry = AuditEntry(
            stage="nl_input",
            timestamp=_fixed_ts(),
            before_snapshot=None,
            after_snapshot=None,
            duration=0.1,
            status="success",
            error=None,
        )
        existing_chain = AuditChain.empty().append(existing_entry)

        io = FakeTerminalIO(inputs=["a"])
        result = confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
            chain=existing_chain,
        )
        assert len(result.chain) == 2
        assert result.chain.entries[0].stage == "nl_input"
        assert result.chain.entries[1].stage == "confirmation"

    def test_original_chain_not_mutated(self, wiki_root: Path) -> None:
        original = AuditChain.empty()

        io = FakeTerminalIO(inputs=["a"])
        confirm_with_audit(
            _make_request(),
            wiki_root,
            terminal=io,
            chain=original,
        )
        assert len(original) == 0
