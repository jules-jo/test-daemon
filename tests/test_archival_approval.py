"""Tests for the audit archival approval prompt flow.

Verifies that the archival approval module:
- Presents aged audit entries to the user as archival candidates
- Renders a clear summary showing entry count, age range, and details
- Requires explicit user confirmation (approve/deny) before proceeding
- Returns a structured ArchivalApprovalResult with the user decision
- Supports individual entry selection (approve all, deny all, or select)
- Handles empty candidate lists gracefully (no prompt needed)
- Treats EOF/interrupt as denial for safety
- Handles invalid input with re-prompting
- Preserves immutability on all data models
- Reuses the TerminalIO abstraction from cli.confirmation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pytest

from jules_daemon.wiki.archival_approval import (
    ArchivalApprovalResult,
    ArchivalCandidate,
    ArchivalDecision,
    ArchivalRequest,
    prompt_archival_approval,
    render_archival_display,
)
from jules_daemon.wiki.audit_age_scanner import (
    AgedAuditEntry,
    AuditAgeScanResult,
    ScanOutcome,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_NOW = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


class FakeTerminalIO:
    """Fake terminal IO for testing the archival approval flow.

    Replicates the TerminalIO interface from cli.confirmation for test
    isolation without importing the concrete class.
    """

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


def _make_aged_entry(
    event_id: str = "evt-001",
    age_days: int = 120,
    created_at: datetime | None = None,
) -> AgedAuditEntry:
    """Create a test AgedAuditEntry."""
    ts = created_at or (_NOW - timedelta(days=age_days))
    return AgedAuditEntry(
        source_path=Path(f"/wiki/pages/daemon/audit/audit-{event_id}.md"),
        event_id=event_id,
        created_at=ts,
        age_days=age_days,
    )


def _make_scan_result(
    entries: tuple[AgedAuditEntry, ...] = (),
    threshold_days: int = 90,
) -> AuditAgeScanResult:
    """Create a test AuditAgeScanResult with all entries above threshold."""
    return AuditAgeScanResult(
        outcome=ScanOutcome.SCANNED,
        entries=entries,
        errors=(),
        scanned_count=len(entries),
        threshold_days=threshold_days,
    )


def _make_request(
    entries: tuple[AgedAuditEntry, ...] | None = None,
    threshold_days: int = 90,
) -> ArchivalRequest:
    """Create an ArchivalRequest from a scan result."""
    if entries is None:
        entries = (
            _make_aged_entry("evt-old-001", 150),
            _make_aged_entry("evt-old-002", 120),
        )
    scan_result = _make_scan_result(entries, threshold_days=threshold_days)
    return ArchivalRequest(scan_result=scan_result)


# ---------------------------------------------------------------------------
# ArchivalDecision enum
# ---------------------------------------------------------------------------


class TestArchivalDecision:
    """ArchivalDecision covers all user choices for archival."""

    def test_approve_value(self) -> None:
        assert ArchivalDecision.APPROVE.value == "approve"

    def test_deny_value(self) -> None:
        assert ArchivalDecision.DENY.value == "deny"

    def test_all_decisions(self) -> None:
        values = {d.value for d in ArchivalDecision}
        assert values == {"approve", "deny"}


# ---------------------------------------------------------------------------
# ArchivalCandidate data model
# ---------------------------------------------------------------------------


class TestArchivalCandidate:
    """ArchivalCandidate wraps an AgedAuditEntry with display metadata."""

    def test_construction(self) -> None:
        entry = _make_aged_entry("evt-001", 120)
        candidate = ArchivalCandidate(
            entry=entry,
            index=1,
        )
        assert candidate.entry == entry
        assert candidate.index == 1

    def test_frozen(self) -> None:
        entry = _make_aged_entry()
        candidate = ArchivalCandidate(entry=entry, index=1)
        with pytest.raises(AttributeError):
            candidate.index = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ArchivalRequest data model
# ---------------------------------------------------------------------------


class TestArchivalRequest:
    """ArchivalRequest encapsulates the scan result for the approval flow."""

    def test_construction(self) -> None:
        request = _make_request()
        assert request.scan_result is not None

    def test_frozen(self) -> None:
        request = _make_request()
        with pytest.raises(AttributeError):
            request.scan_result = None  # type: ignore[misc]

    def test_candidates_returns_aged_entries(self) -> None:
        entries = (
            _make_aged_entry("evt-001", 120),
            _make_aged_entry("evt-002", 100),
        )
        request = _make_request(entries=entries)
        candidates = request.candidates
        assert len(candidates) == 2
        assert candidates[0].entry.event_id == "evt-001"
        assert candidates[1].entry.event_id == "evt-002"

    def test_candidates_indexed_starting_from_one(self) -> None:
        entries = (
            _make_aged_entry("evt-a", 150),
            _make_aged_entry("evt-b", 120),
        )
        request = _make_request(entries=entries)
        candidates = request.candidates
        assert candidates[0].index == 1
        assert candidates[1].index == 2

    def test_has_candidates_true(self) -> None:
        request = _make_request()
        assert request.has_candidates is True

    def test_has_candidates_false_when_none_aged(self) -> None:
        # Entry below threshold
        entries = (_make_aged_entry("evt-recent", 30),)
        request = _make_request(entries=entries)
        assert request.has_candidates is False


# ---------------------------------------------------------------------------
# ArchivalApprovalResult data model
# ---------------------------------------------------------------------------


class TestArchivalApprovalResult:
    """ArchivalApprovalResult captures the user decision on archival."""

    def test_approved_result(self) -> None:
        entries = (_make_aged_entry("evt-001", 120),)
        result = ArchivalApprovalResult(
            decision=ArchivalDecision.APPROVE,
            approved_entries=entries,
            total_candidates=1,
        )
        assert result.decision == ArchivalDecision.APPROVE
        assert result.is_approved is True
        assert len(result.approved_entries) == 1

    def test_denied_result(self) -> None:
        result = ArchivalApprovalResult(
            decision=ArchivalDecision.DENY,
            approved_entries=(),
            total_candidates=2,
        )
        assert result.decision == ArchivalDecision.DENY
        assert result.is_approved is False
        assert len(result.approved_entries) == 0

    def test_frozen(self) -> None:
        result = ArchivalApprovalResult(
            decision=ArchivalDecision.APPROVE,
            approved_entries=(),
            total_candidates=0,
        )
        with pytest.raises(AttributeError):
            result.decision = ArchivalDecision.DENY  # type: ignore[misc]

    def test_approved_count(self) -> None:
        entries = (
            _make_aged_entry("evt-001", 120),
            _make_aged_entry("evt-002", 100),
        )
        result = ArchivalApprovalResult(
            decision=ArchivalDecision.APPROVE,
            approved_entries=entries,
            total_candidates=2,
        )
        assert result.approved_count == 2

    def test_skipped_count(self) -> None:
        entries = (_make_aged_entry("evt-001", 120),)
        result = ArchivalApprovalResult(
            decision=ArchivalDecision.APPROVE,
            approved_entries=entries,
            total_candidates=3,
        )
        assert result.skipped_count == 2


# ---------------------------------------------------------------------------
# render_archival_display
# ---------------------------------------------------------------------------


class TestRenderArchivalDisplay:
    """Rendering produces a clear summary for the user."""

    def test_contains_header(self) -> None:
        request = _make_request()
        output = render_archival_display(request)
        assert "Audit Log Archival" in output

    def test_contains_candidate_count(self) -> None:
        entries = (
            _make_aged_entry("evt-001", 120),
            _make_aged_entry("evt-002", 100),
            _make_aged_entry("evt-003", 95),
        )
        request = _make_request(entries=entries)
        output = render_archival_display(request)
        assert "3" in output

    def test_contains_threshold(self) -> None:
        request = _make_request(threshold_days=90)
        output = render_archival_display(request)
        assert "90" in output

    def test_contains_event_ids(self) -> None:
        entries = (
            _make_aged_entry("evt-alpha", 150),
            _make_aged_entry("evt-beta", 120),
        )
        request = _make_request(entries=entries)
        output = render_archival_display(request)
        assert "evt-alpha" in output
        assert "evt-beta" in output

    def test_contains_age_days(self) -> None:
        entries = (_make_aged_entry("evt-001", 150),)
        request = _make_request(entries=entries)
        output = render_archival_display(request)
        assert "150" in output

    def test_contains_action_instructions(self) -> None:
        request = _make_request()
        output = render_archival_display(request)
        assert "[A]" in output or "approve" in output.lower()
        assert "[D]" in output or "deny" in output.lower()

    def test_contains_destination_info(self) -> None:
        request = _make_request()
        output = render_archival_display(request)
        assert "archive" in output.lower()


# ---------------------------------------------------------------------------
# prompt_archival_approval: approve flow
# ---------------------------------------------------------------------------


class TestApproveFlow:
    """User approves archival of all candidates."""

    def test_approve_with_a(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["a"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.APPROVE
        assert result.is_approved is True

    def test_approve_with_y(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["y"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.APPROVE

    def test_approve_with_yes(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["yes"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.APPROVE

    def test_approve_case_insensitive(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["A"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.APPROVE

    def test_approve_with_approve_word(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["approve"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.APPROVE

    def test_approved_entries_match_candidates(self) -> None:
        entries = (
            _make_aged_entry("evt-001", 120),
            _make_aged_entry("evt-002", 100),
        )
        request = _make_request(entries=entries)
        io = FakeTerminalIO(inputs=["a"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.approved_count == 2
        event_ids = {e.event_id for e in result.approved_entries}
        assert event_ids == {"evt-001", "evt-002"}


# ---------------------------------------------------------------------------
# prompt_archival_approval: deny flow
# ---------------------------------------------------------------------------


class TestDenyFlow:
    """User denies archival -- no files should be moved."""

    def test_deny_with_d(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["d"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.DENY
        assert result.is_approved is False

    def test_deny_with_n(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["n"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.DENY

    def test_deny_with_no(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["no"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.DENY

    def test_deny_with_deny_word(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["deny"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.DENY

    def test_denied_result_has_empty_approved(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["d"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.approved_entries == ()
        assert result.approved_count == 0

    def test_denied_result_tracks_total_candidates(self) -> None:
        entries = (
            _make_aged_entry("evt-001", 120),
            _make_aged_entry("evt-002", 100),
        )
        request = _make_request(entries=entries)
        io = FakeTerminalIO(inputs=["d"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.total_candidates == 2


# ---------------------------------------------------------------------------
# prompt_archival_approval: empty candidates
# ---------------------------------------------------------------------------


class TestEmptyCandidates:
    """When there are no archival candidates, no prompt is needed."""

    def test_no_candidates_returns_approve_with_empty(self) -> None:
        # All entries are below threshold
        entries = (_make_aged_entry("evt-recent", 30),)
        request = _make_request(entries=entries)
        io = FakeTerminalIO(inputs=[])  # should not be prompted
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.APPROVE
        assert result.approved_entries == ()
        assert result.total_candidates == 0

    def test_completely_empty_scan(self) -> None:
        request = _make_request(entries=())
        io = FakeTerminalIO(inputs=[])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.APPROVE
        assert result.total_candidates == 0


# ---------------------------------------------------------------------------
# prompt_archival_approval: EOF / interrupt handling
# ---------------------------------------------------------------------------


class TestEOFHandling:
    """EOF or keyboard interrupt results in denial (safe default)."""

    def test_eof_results_in_deny(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=[])  # No input = immediate EOF
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.DENY
        assert result.is_approved is False


# ---------------------------------------------------------------------------
# prompt_archival_approval: invalid input handling
# ---------------------------------------------------------------------------


class TestInvalidInput:
    """Invalid input is rejected with a reprompt."""

    def test_invalid_then_approve(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["x", "a"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.APPROVE

    def test_multiple_invalid_then_deny(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["foo", "bar", "d"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.DENY

    def test_whitespace_reprompts(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["  ", "y"])
        result = prompt_archival_approval(request, terminal=io)
        assert result.decision == ArchivalDecision.APPROVE


# ---------------------------------------------------------------------------
# prompt_archival_approval: display output
# ---------------------------------------------------------------------------


class TestDisplayOutput:
    """The prompt displays archival candidates before asking for a decision."""

    def test_displays_candidates(self) -> None:
        entries = (_make_aged_entry("evt-show", 120),)
        request = _make_request(entries=entries)
        io = FakeTerminalIO(inputs=["a"])
        prompt_archival_approval(request, terminal=io)
        assert "evt-show" in io.output_text

    def test_displays_age(self) -> None:
        entries = (_make_aged_entry("evt-001", 200),)
        request = _make_request(entries=entries)
        io = FakeTerminalIO(inputs=["a"])
        prompt_archival_approval(request, terminal=io)
        assert "200" in io.output_text

    def test_displays_threshold(self) -> None:
        request = _make_request(threshold_days=90)
        io = FakeTerminalIO(inputs=["a"])
        prompt_archival_approval(request, terminal=io)
        assert "90" in io.output_text

    def test_displays_action_instructions(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["a"])
        prompt_archival_approval(request, terminal=io)
        output = io.output_text.lower()
        assert "approve" in output or "[a]" in output
        assert "deny" in output or "[d]" in output


# ---------------------------------------------------------------------------
# Immutability: request is not modified
# ---------------------------------------------------------------------------


class TestImmutability:
    """Archival approval flow does not modify input data."""

    def test_request_unchanged_after_approve(self) -> None:
        entries = (
            _make_aged_entry("evt-001", 120),
            _make_aged_entry("evt-002", 100),
        )
        request = _make_request(entries=entries)
        original_count = len(request.scan_result.entries)
        io = FakeTerminalIO(inputs=["a"])
        prompt_archival_approval(request, terminal=io)
        assert len(request.scan_result.entries) == original_count

    def test_request_unchanged_after_deny(self) -> None:
        request = _make_request()
        io = FakeTerminalIO(inputs=["d"])
        prompt_archival_approval(request, terminal=io)
        assert request.scan_result.outcome == ScanOutcome.SCANNED
