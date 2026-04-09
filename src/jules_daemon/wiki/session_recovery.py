"""Session recovery detection and resume/discard offer handling.

On CLI reconnection, detects whether a prior session was interrupted and
builds a structured RecoveryOffer that the daemon presents to the user.
The user can then accept (resume) or reject (discard) the prior session.

This module reads session state from the wiki persistence layer and
provides a clean interface for the IPC request handler to:
1. Detect if a prior session exists and is resumable
2. Present the offer to the user with full context
3. Process the user's accept/reject decision

Decision logic:
  - RUNNING session -> OFFER_RESUME (test was in progress)
  - PENDING_APPROVAL session -> OFFER_RESUME (command awaiting approval)
  - IDLE/terminal session -> NO_RECOVERY (nothing to resume)
  - No session file -> NO_RECOVERY (first ever connection)
  - Corrupted file -> NO_RECOVERY (safe degradation)

Usage:
    from pathlib import Path
    from jules_daemon.wiki.session_recovery import (
        detect_session_recovery,
        accept_recovery,
        reject_recovery,
    )

    offer = detect_session_recovery(wiki_root)
    if offer.needs_user_decision:
        # Show offer.summary to user, get their choice
        if user_accepts:
            result = accept_recovery(wiki_root, offer)
        else:
            result = reject_recovery(wiki_root, offer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from jules_daemon.wiki.models import RunStatus
from jules_daemon.wiki.session_persistence import (
    LoadSessionOutcome,
    SessionSnapshot,
    discard_session_state,
    load_session_state,
    save_session_state,
)

__all__ = [
    "AcceptResult",
    "RecoveryAction",
    "RecoveryOffer",
    "RejectResult",
    "accept_recovery",
    "detect_session_recovery",
    "reject_recovery",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recovery action enum
# ---------------------------------------------------------------------------


class RecoveryAction(Enum):
    """Action the daemon should take based on session recovery detection.

    OFFER_RESUME: An interrupted active session was found. The daemon
        should present the offer to the user and let them choose.
    NO_RECOVERY: No resumable session found. The daemon starts fresh.
    """

    OFFER_RESUME = "offer_resume"
    NO_RECOVERY = "no_recovery"


# ---------------------------------------------------------------------------
# Recovery offer model
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RecoveryOffer:
    """Structured offer for session recovery presented to the user.

    Contains the recovery action, human-readable reason, and the full
    session snapshot for context. The CLI uses this to display a
    "Resume or Discard?" prompt with all relevant details.

    Attributes:
        action: Whether to offer resume or skip recovery.
        reason: Human-readable explanation of the decision.
        snapshot: The prior session snapshot (None if no recovery).
    """

    action: RecoveryAction
    reason: str
    snapshot: Optional[SessionSnapshot]

    @property
    def needs_user_decision(self) -> bool:
        """True if the user must choose to resume or discard."""
        return self.action == RecoveryAction.OFFER_RESUME

    @property
    def summary(self) -> str:
        """Human-readable summary for CLI display.

        Provides a one-line description suitable for terminal output.
        """
        if self.snapshot is None or self.action == RecoveryAction.NO_RECOVERY:
            return "No prior session to recover."

        snap = self.snapshot
        parts = [
            f"Prior session found: run_id={snap.run_id}",
            f"status={snap.status.value}",
        ]

        if snap.ssh_target is not None:
            parts.append(f"host={snap.ssh_target.host}")

        if snap.status == RunStatus.RUNNING:
            parts.append(f"progress={snap.progress.percent:.1f}%")

        if snap.command is not None:
            # Truncate long commands for display
            cmd_text = snap.command.natural_language
            if len(cmd_text) > 50:
                cmd_text = cmd_text[:47] + "..."
            parts.append(f'command="{cmd_text}"')

        parts.append(f"disconnect={snap.disconnect_reason}")

        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Accept/reject result models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptResult:
    """Outcome of accepting a recovery offer.

    Attributes:
        accepted: True if the recovery was successfully accepted.
        run_id: The run ID of the resumed session.
        error: Error description if acceptance failed (None on success).
    """

    accepted: bool
    run_id: str
    error: Optional[str]


@dataclass(frozen=True)
class RejectResult:
    """Outcome of rejecting a recovery offer.

    Attributes:
        rejected: True if the session was successfully discarded.
        run_id: The run ID of the discarded session.
        error: Error description if rejection failed (None on success).
    """

    rejected: bool
    run_id: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Public API: detect
# ---------------------------------------------------------------------------


def detect_session_recovery(wiki_root: Path) -> RecoveryOffer:
    """Detect if a prior session exists and offer recovery.

    Reads the session state file from the wiki and determines whether
    the prior session is resumable. Returns a structured RecoveryOffer
    that the daemon presents to the reconnecting user.

    Decision logic:
    1. No session file: NO_RECOVERY
    2. Corrupted file: NO_RECOVERY (safe degradation)
    3. Non-resumable status (IDLE/terminal): NO_RECOVERY
    4. Resumable status (RUNNING/PENDING_APPROVAL): OFFER_RESUME

    This function never raises. All errors are captured in the returned
    offer's reason field.

    Args:
        wiki_root: Path to the wiki root directory.

    Returns:
        RecoveryOffer with action, reason, and snapshot context.
    """
    loaded = load_session_state(wiki_root)

    # Case 1: No file
    if loaded.outcome == LoadSessionOutcome.NO_FILE:
        logger.info("No prior session file -- no recovery to offer")
        return RecoveryOffer(
            action=RecoveryAction.NO_RECOVERY,
            reason="No prior session found -- starting fresh",
            snapshot=None,
        )

    # Case 2: Corrupted file
    if loaded.outcome == LoadSessionOutcome.CORRUPTED:
        logger.warning(
            "Corrupted session file -- cannot offer recovery: %s",
            loaded.error,
        )
        return RecoveryOffer(
            action=RecoveryAction.NO_RECOVERY,
            reason=f"Session file corrupted: {loaded.error}",
            snapshot=None,
        )

    # Case 3: Loaded but snapshot is None (defensive)
    if loaded.snapshot is None:
        logger.warning("Session loaded but snapshot is None")
        return RecoveryOffer(
            action=RecoveryAction.NO_RECOVERY,
            reason="Session file loaded but contained no usable data",
            snapshot=None,
        )

    # Case 4: Check if resumable
    snap = loaded.snapshot

    if not snap.is_resumable:
        logger.info(
            "Prior session is not resumable (status=%s) -- no recovery",
            snap.status.value,
        )
        return RecoveryOffer(
            action=RecoveryAction.NO_RECOVERY,
            reason=(
                f"Prior session was {snap.status.value} -- "
                f"no recovery needed"
            ),
            snapshot=snap,
        )

    # Case 5: Resumable session found
    logger.info(
        "Resumable session detected: run_id=%s status=%s "
        "disconnect_reason=%s",
        snap.run_id,
        snap.status.value,
        snap.disconnect_reason,
    )
    return RecoveryOffer(
        action=RecoveryAction.OFFER_RESUME,
        reason=(
            f"Interrupted {snap.status.value} session detected "
            f"(run_id={snap.run_id}, disconnect={snap.disconnect_reason})"
        ),
        snapshot=snap,
    )


# ---------------------------------------------------------------------------
# Public API: accept
# ---------------------------------------------------------------------------


def accept_recovery(
    wiki_root: Path,
    offer: RecoveryOffer,
) -> AcceptResult:
    """Accept a recovery offer, marking the session as resumed.

    Updates the session state file to indicate that the user chose to
    resume the prior session. The daemon should then proceed with
    reconnection and monitoring based on the snapshot data.

    Args:
        wiki_root: Path to the wiki root directory.
        offer: The recovery offer to accept.

    Returns:
        AcceptResult indicating success or failure.
    """
    if offer.action != RecoveryAction.OFFER_RESUME or offer.snapshot is None:
        return AcceptResult(
            accepted=False,
            run_id="",
            error="Cannot accept: no resumable session in the offer",
        )

    snap = offer.snapshot

    try:
        # Update the snapshot to record that it was accepted/resumed
        # We save a new snapshot with the same data but a note that
        # it was resumed (the disconnect_reason becomes "resumed")
        resumed_snap = SessionSnapshot(
            run_id=snap.run_id,
            status=snap.status,
            ssh_target=snap.ssh_target,
            command=snap.command,
            pids=snap.pids,
            progress=snap.progress,
            started_at=snap.started_at,
            error=snap.error,
            disconnect_reason="resumed",
            client_name=snap.client_name,
            client_pid=snap.client_pid,
            saved_at=_now_utc(),
        )

        write_result = save_session_state(wiki_root, resumed_snap)

        if not write_result.success:
            return AcceptResult(
                accepted=False,
                run_id=snap.run_id,
                error=f"Failed to update session file: {write_result.error}",
            )

        logger.info(
            "Recovery accepted: run_id=%s status=%s",
            snap.run_id,
            snap.status.value,
        )

        return AcceptResult(
            accepted=True,
            run_id=snap.run_id,
            error=None,
        )

    except (TypeError, ValueError) as exc:
        logger.warning(
            "Failed to accept recovery for run_id=%s: %s",
            snap.run_id,
            exc,
        )
        return AcceptResult(
            accepted=False,
            run_id=snap.run_id,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Public API: reject
# ---------------------------------------------------------------------------


def reject_recovery(
    wiki_root: Path,
    offer: RecoveryOffer,
) -> RejectResult:
    """Reject a recovery offer, discarding the prior session state.

    Clears the session state file by writing a non-resumable idle marker.
    The daemon should then start fresh as if no prior session existed.

    Args:
        wiki_root: Path to the wiki root directory.
        offer: The recovery offer to reject.

    Returns:
        RejectResult indicating success or failure.
    """
    if offer.action != RecoveryAction.OFFER_RESUME or offer.snapshot is None:
        return RejectResult(
            rejected=False,
            run_id="",
            error="Cannot reject: no resumable session in the offer",
        )

    run_id = offer.snapshot.run_id

    try:
        success = discard_session_state(wiki_root)

        if not success:
            return RejectResult(
                rejected=False,
                run_id=run_id,
                error="Failed to discard session state file",
            )

        logger.info(
            "Recovery rejected: run_id=%s -- session discarded",
            run_id,
        )

        return RejectResult(
            rejected=True,
            run_id=run_id,
            error=None,
        )

    except (TypeError, ValueError, OSError) as exc:
        logger.warning(
            "Failed to reject recovery for run_id=%s: %s",
            run_id,
            exc,
        )
        return RejectResult(
            rejected=False,
            run_id=run_id,
            error=str(exc),
        )
