"""Generic audit entry and immutable chain accumulator.

Provides a stage-agnostic audit trail where each ``AuditEntry`` captures a
single event -- its timestamp, the pipeline stage name, an input snapshot,
an output snapshot, and arbitrary metadata.

``AuditChain`` is the accumulator: a frozen tuple wrapper that grows only
via ``append``, which returns a *new* chain (the original is never mutated).
The full chain serializes to a list of dicts for wiki YAML persistence.

Design principles:
    - Frozen dataclasses -- no mutation after construction
    - Immutable append -- every ``append`` returns a new chain
    - Deterministic serialization -- entries preserve insertion order
    - No side effects -- pure data structures with no I/O

Usage::

    from jules_daemon.audit.chain import AuditEntry, AuditChain
    from datetime import datetime, timezone

    entry = AuditEntry(
        timestamp=datetime.now(timezone.utc),
        stage="nl_input",
        input_snapshot="run all tests",
        output_snapshot={"parsed": "pytest -v"},
        metadata={"model_id": "gpt-4"},
    )
    chain = AuditChain.empty().append(entry)
    chain.to_list()  # -> [{"timestamp": "...", "stage": "nl_input", ...}]
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
    """Strip and validate that a string is not empty."""
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

    Each entry captures a snapshot of one pipeline stage: what went in,
    what came out, when it happened, and any extra context.

    Attributes:
        timestamp: UTC datetime when the event occurred.
        stage: Name of the pipeline stage (e.g. "nl_input", "confirmation").
        input_snapshot: Arbitrary data representing the stage input.
            Can be a string, dict, list, None, or any JSON-serializable value.
        output_snapshot: Arbitrary data representing the stage output.
            Can be a string, dict, list, None, or any JSON-serializable value.
        metadata: Key-value pairs with additional context about the event
            (model IDs, latency, tags, etc.). Must not be None.
    """

    timestamp: datetime
    stage: str
    input_snapshot: object
    output_snapshot: object
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        stripped = _require_non_empty(self.stage, "stage")
        if stripped != self.stage:
            object.__setattr__(self, "stage", stripped)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for wiki YAML persistence.

        The timestamp is converted to an ISO 8601 string. All other fields
        are passed through as-is (they must already be JSON-serializable).
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage,
            "input_snapshot": self.input_snapshot,
            "output_snapshot": self.output_snapshot,
            "metadata": dict(self.metadata),
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
            timestamp=ts,
            stage=data["stage"],
            input_snapshot=data["input_snapshot"],
            output_snapshot=data["output_snapshot"],
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# AuditChain -- immutable accumulator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditChain:
    """Immutable accumulator of AuditEntry records.

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

    # -- Accessors --

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
