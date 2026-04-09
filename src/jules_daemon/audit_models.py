"""AuditEntry and AuditChain -- append-only audit ledger data models.

Provides a stage-agnostic audit trail where each ``AuditEntry`` captures a
single event in the command execution pipeline -- its stage name, timestamp,
before/after snapshots, duration, status, and optional error message.

``AuditChain`` is the append-only ledger: a frozen tuple wrapper that grows
exclusively via ``append``, which returns a *new* chain (the original is
never mutated). The full chain serializes to a list of dicts for wiki YAML
persistence.

Design principles:
    - Frozen dataclasses -- no mutation after construction
    - Immutable append -- every ``append`` returns a new chain
    - Deterministic serialization -- entries preserve insertion order
    - No side effects -- pure data structures with no I/O

Usage::

    from jules_daemon.audit_models import AuditEntry, AuditChain
    from datetime import datetime, timezone

    entry = AuditEntry(
        stage="nl_input",
        timestamp=datetime.now(timezone.utc),
        before_snapshot="run all tests",
        after_snapshot={"parsed": "pytest -v"},
        duration=0.45,
        status="success",
        error=None,
    )
    chain = AuditChain.empty().append(entry)
    chain.to_list()  # -> [{"stage": "nl_input", ...}]
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

__all__ = [
    "AuditEntry",
    "AuditChain",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_non_empty(value: str, field_name: str) -> str:
    """Strip and validate that a string is not empty.

    Returns the stripped value so callers can normalize whitespace.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


# ---------------------------------------------------------------------------
# AuditEntry -- a single audit event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    """Immutable record of a single audit event in the execution pipeline.

    Each entry captures a snapshot of one pipeline stage: what the state
    was before the stage ran (``before_snapshot``), what it became after
    (``after_snapshot``), how long it took, whether it succeeded, and
    any error that occurred.

    Attributes:
        stage: Name of the pipeline stage (e.g. "nl_input", "confirmation").
            Must not be empty; leading/trailing whitespace is stripped.
        timestamp: UTC datetime when the event occurred.
        before_snapshot: Arbitrary data representing the state before the
            stage executed. Can be a string, dict, list, None, or any
            JSON-serializable value.
        after_snapshot: Arbitrary data representing the state after the
            stage executed. Can be a string, dict, list, None, or any
            JSON-serializable value.
        duration: Wall-clock seconds the stage took to execute, or None
            if timing is not applicable. Must not be negative.
        status: Outcome label for the stage (e.g. "success", "failure",
            "error", "skipped"). Must not be empty.
        error: Human-readable error message, or None when the stage
            completed without error.
    """

    stage: str
    timestamp: datetime
    before_snapshot: object
    after_snapshot: object
    duration: float | None
    status: str
    error: str | None

    def __post_init__(self) -> None:
        stripped_stage = _require_non_empty(self.stage, "stage")
        if stripped_stage != self.stage:
            object.__setattr__(self, "stage", stripped_stage)

        stripped_status = _require_non_empty(self.status, "status")
        if stripped_status != self.status:
            object.__setattr__(self, "status", stripped_status)

        if self.duration is not None and self.duration < 0:
            raise ValueError("duration must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki YAML persistence.

        The timestamp is converted to an ISO 8601 string. All other fields
        are passed through as-is (they must already be JSON-serializable).
        """
        return {
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "before_snapshot": self.before_snapshot,
            "after_snapshot": self.after_snapshot,
            "duration": self.duration,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEntry:
        """Deserialize from a plain dict (e.g. parsed from wiki YAML).

        If ``timestamp`` is a string, it is parsed as ISO 8601. If it is
        already a ``datetime``, it is used directly.
        """
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            stage=data["stage"],
            timestamp=ts,
            before_snapshot=data["before_snapshot"],
            after_snapshot=data["after_snapshot"],
            duration=data["duration"],
            status=data["status"],
            error=data["error"],
        )


# ---------------------------------------------------------------------------
# AuditChain -- immutable append-only ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditChain:
    """Immutable append-only ledger of AuditEntry records.

    Grows exclusively via ``append``, which returns a new ``AuditChain``
    with the additional entry appended -- the original chain is never
    modified.

    Entries are stored as a tuple to enforce immutability at the
    collection level.

    Attributes:
        entries: Ordered tuple of audit entries (insertion order preserved).
    """

    entries: tuple[AuditEntry, ...]

    # -- Factory methods --

    @classmethod
    def empty(cls) -> AuditChain:
        """Create a new empty chain with no entries."""
        return cls(entries=())

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> AuditChain:
        """Deserialize a chain from a list of entry dicts.

        Each dict in the list is deserialized via ``AuditEntry.from_dict``.
        """
        entries = tuple(AuditEntry.from_dict(d) for d in data)
        return cls(entries=entries)

    # -- Immutable operations --

    def append(self, entry: AuditEntry) -> AuditChain:
        """Return a new chain with the given entry appended.

        The current chain is not modified.
        """
        return AuditChain(entries=(*self.entries, entry))

    # -- Serialization --

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize the full chain to a list of dicts.

        Each entry is serialized via ``AuditEntry.to_dict``. Returns a
        new list on every call (safe to mutate).
        """
        return [entry.to_dict() for entry in self.entries]

    # -- Retrieval accessors --

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def latest(self) -> AuditEntry | None:
        """The most recently appended entry, or None if the chain is empty."""
        if not self.entries:
            return None
        return self.entries[-1]

    @property
    def stages(self) -> tuple[str, ...]:
        """Ordered tuple of stage names from all entries."""
        return tuple(entry.stage for entry in self.entries)

    def by_stage(self, stage: str) -> tuple[AuditEntry, ...]:
        """Return all entries matching the given stage name."""
        return tuple(entry for entry in self.entries if entry.stage == stage)
