"""User approval prompt flow for audit log archival.

Presents aged audit entries as archival candidates and requires explicit
human confirmation before any files are moved to the archive directory.

This module enforces the constraint:
    "Audit log archival requires explicit user approval before moving
    to archive."

The flow:
1. Accept an AuditAgeScanResult (from audit_age_scanner) as input.
2. Identify entries exceeding the threshold age as archival candidates.
3. Render a human-readable summary of the candidates.
4. Prompt the user for an explicit approve/deny decision.
5. Return a structured ArchivalApprovalResult recording the decision.

If there are no candidates (no aged entries), the flow returns an
approved result with an empty entry list -- no prompt is shown.

EOF or keyboard interrupts are treated as denial (safe default).

The IO layer is abstracted behind the TerminalIO protocol from
cli.confirmation for testability.

Usage::

    from jules_daemon.wiki.archival_approval import (
        ArchivalRequest,
        prompt_archival_approval,
    )
    from jules_daemon.wiki.audit_age_scanner import scan_aged_audit_entries

    scan = scan_aged_audit_entries(wiki_root)
    request = ArchivalRequest(scan_result=scan)
    result = prompt_archival_approval(request)

    if result.is_approved:
        for entry in result.approved_entries:
            move_to_archive(entry.source_path)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from jules_daemon.wiki.audit_age_scanner import AgedAuditEntry, AuditAgeScanResult

__all__ = [
    "ArchivalApprovalResult",
    "ArchivalCandidate",
    "ArchivalDecision",
    "ArchivalRequest",
    "prompt_archival_approval",
    "render_archival_display",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO protocol (matches cli.confirmation.TerminalIO)
# ---------------------------------------------------------------------------


class _TerminalIO(Protocol):
    """Terminal IO protocol for testability.

    Matches the interface of cli.confirmation.TerminalIO without
    creating a hard coupling to that module.
    """

    def write(self, text: str) -> None: ...

    def read_line(self, prompt: str = "") -> str: ...

    def read_editable(self, prompt: str, prefill: str) -> str: ...


# ---------------------------------------------------------------------------
# Decision enum
# ---------------------------------------------------------------------------


class ArchivalDecision(Enum):
    """User's decision on whether to proceed with audit log archival.

    Values:
        APPROVE: Move the aged entries to the archive directory.
        DENY: Do not move any entries; leave them in place.
    """

    APPROVE = "approve"
    DENY = "deny"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchivalCandidate:
    """An aged audit entry presented as an archival candidate.

    Attributes:
        entry: The aged audit entry from the scanner.
        index: One-based display index for the user.
    """

    entry: AgedAuditEntry
    index: int


@dataclass(frozen=True)
class ArchivalRequest:
    """Input to the archival approval prompt.

    Wraps an AuditAgeScanResult and provides convenience accessors
    for the archival candidates (entries exceeding the threshold).

    Attributes:
        scan_result: The age scan result containing all discovered entries.
    """

    scan_result: AuditAgeScanResult

    @property
    def candidates(self) -> tuple[ArchivalCandidate, ...]:
        """Aged entries wrapped as indexed candidates for display."""
        aged = self.scan_result.aged_entries
        return tuple(
            ArchivalCandidate(entry=entry, index=idx + 1)
            for idx, entry in enumerate(aged)
        )

    @property
    def has_candidates(self) -> bool:
        """True if there are any entries exceeding the threshold age."""
        return self.scan_result.aged_count > 0


@dataclass(frozen=True)
class ArchivalApprovalResult:
    """Output from the archival approval prompt.

    Captures the user's decision and the specific entries approved
    for archival. When denied, approved_entries is empty.

    Attributes:
        decision: Whether the user approved or denied archival.
        approved_entries: The entries approved for archival (empty on deny).
        total_candidates: How many candidates were presented.
    """

    decision: ArchivalDecision
    approved_entries: tuple[AgedAuditEntry, ...]
    total_candidates: int

    @property
    def is_approved(self) -> bool:
        """True when the user approved archival."""
        return self.decision == ArchivalDecision.APPROVE

    @property
    def approved_count(self) -> int:
        """Number of entries approved for archival."""
        return len(self.approved_entries)

    @property
    def skipped_count(self) -> int:
        """Number of candidates not approved (denied or not selected)."""
        return self.total_candidates - self.approved_count


# ---------------------------------------------------------------------------
# Input classification
# ---------------------------------------------------------------------------

_APPROVE_INPUTS = frozenset({"a", "y", "yes", "approve", "allow"})
_DENY_INPUTS = frozenset({"d", "n", "no", "deny", "reject"})


def _classify_input(raw: str) -> str:
    """Classify user input into 'approve', 'deny', or 'unknown'.

    Args:
        raw: Raw user input string.

    Returns:
        One of 'approve', 'deny', 'unknown'.
    """
    normalized = raw.strip().lower()
    if normalized in _APPROVE_INPUTS:
        return "approve"
    if normalized in _DENY_INPUTS:
        return "deny"
    return "unknown"


# ---------------------------------------------------------------------------
# Display rendering
# ---------------------------------------------------------------------------

_SEPARATOR = "-" * 60


def render_archival_display(request: ArchivalRequest) -> str:
    """Render the archival candidates as a formatted display string.

    Produces a human-readable block showing:
    - Header identifying this as an archival prompt
    - Threshold age used for the scan
    - Number of candidates found
    - Per-entry details (index, event ID, age, source path)
    - Destination (archive directory)
    - Instructions for approve/deny

    Args:
        request: The archival request with scan results.

    Returns:
        Formatted multi-line string ready for terminal display.
    """
    candidates = request.candidates
    threshold = request.scan_result.threshold_days
    lines: list[str] = []

    lines.append("")
    lines.append(_SEPARATOR)
    lines.append("  Audit Log Archival")
    lines.append(_SEPARATOR)
    lines.append("")
    lines.append(
        f"  {len(candidates)} audit log(s) older than "
        f"{threshold} days found."
    )
    lines.append(
        "  These entries will be moved to the archive/ subdirectory."
    )

    if candidates:
        lines.append("")
        lines.append("  Candidates:")
        lines.append("")

        for candidate in candidates:
            entry = candidate.entry
            lines.append(
                f"    {candidate.index}. {entry.event_id}"
                f"  ({entry.age_days} days old)"
            )
            lines.append(
                f"       {entry.source_path}"
            )

    # Age range summary
    if len(candidates) > 1:
        ages = [c.entry.age_days for c in candidates]
        lines.append("")
        lines.append(
            f"  Age range: {min(ages)}-{max(ages)} days"
        )

    lines.append("")
    lines.append(_SEPARATOR)
    lines.append("  [A]pprove archival  [D]eny (keep in place)")
    lines.append(_SEPARATOR)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Approval prompt
# ---------------------------------------------------------------------------


def _get_default_terminal() -> _TerminalIO:
    """Import and return the DefaultTerminalIO from cli.confirmation.

    Deferred import avoids circular dependency and keeps this module
    usable in contexts where cli.confirmation may not be available
    (e.g., tests with FakeTerminalIO).
    """
    from jules_daemon.cli.confirmation import DefaultTerminalIO

    return DefaultTerminalIO()


def prompt_archival_approval(
    request: ArchivalRequest,
    *,
    terminal: _TerminalIO | None = None,
) -> ArchivalApprovalResult:
    """Present archival candidates and prompt for explicit user approval.

    If there are no candidates (no entries exceed the threshold), returns
    an approved result with an empty entry list immediately -- the user
    is not prompted.

    For non-empty candidate lists, renders the display and loops until
    the user provides a valid approve/deny decision.

    EOF or keyboard interrupts are treated as denial for safety.

    Args:
        request: ArchivalRequest containing the scan results.
        terminal: IO abstraction. Defaults to DefaultTerminalIO.

    Returns:
        ArchivalApprovalResult with the user's decision and entries.
    """
    # No candidates -- nothing to archive, return early
    if not request.has_candidates:
        logger.info("No archival candidates found -- nothing to prompt")
        return ArchivalApprovalResult(
            decision=ArchivalDecision.APPROVE,
            approved_entries=(),
            total_candidates=0,
        )

    io = terminal or _get_default_terminal()
    candidates = request.candidates
    aged_entries = request.scan_result.aged_entries

    # Render and display the candidates
    display = render_archival_display(request)
    io.write(display)

    while True:
        try:
            raw = io.read_line("  Decision: ")
        except EOFError:
            logger.info(
                "EOF received during archival approval -- treating as deny"
            )
            return ArchivalApprovalResult(
                decision=ArchivalDecision.DENY,
                approved_entries=(),
                total_candidates=len(candidates),
            )

        action = _classify_input(raw)

        if action == "approve":
            logger.info(
                "User approved archival of %d entries", len(aged_entries)
            )
            return ArchivalApprovalResult(
                decision=ArchivalDecision.APPROVE,
                approved_entries=aged_entries,
                total_candidates=len(candidates),
            )

        if action == "deny":
            logger.info("User denied archival of %d entries", len(candidates))
            return ArchivalApprovalResult(
                decision=ArchivalDecision.DENY,
                approved_entries=(),
                total_candidates=len(candidates),
            )

        # Unknown input -- reprompt
        io.write(
            "  Invalid input. Use [A]pprove or [D]eny.\n"
        )
