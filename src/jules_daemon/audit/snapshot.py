"""Snapshot capture utility for pipeline stage state.

Serializes pipeline stage state (inputs, config, partial outputs) into
deeply-frozen immutable structures suitable for storing as before/after
snapshots in ``AuditEntry``.

The core primitive is ``deep_freeze``: a recursive transformation that
converts mutable Python structures into their immutable counterparts:

    - ``dict`` -> ``types.MappingProxyType``
    - ``list`` -> ``tuple``
    - ``set`` -> ``frozenset``
    - ``datetime`` -> ISO 8601 string
    - ``Enum`` -> its ``.value``
    - ``dataclass`` -> frozen dict of its fields

``StageSnapshot`` bundles the three standard pipeline-state categories
(inputs, config, partial_outputs) into a single frozen dataclass, with
``to_dict`` / ``from_dict`` for wiki YAML round-tripping.

``capture_snapshot`` is the convenience factory: pass plain dicts and
get back a fully frozen ``StageSnapshot``.

Design principles:
    - Never mutate input data -- all transformations return new values
    - Deeply immutable output -- no mutable reference reachable from result
    - Deterministic serialization -- frozen dicts preserve insertion order
    - JSON-compatible to_dict -- suitable for YAML frontmatter persistence

Usage::

    from jules_daemon.audit.snapshot import capture_snapshot

    before = capture_snapshot(
        stage="nl_input",
        inputs={"raw_input": "run all tests"},
        config={"model": "gpt-4"},
    )
    # before.inputs["raw_input"] == "run all tests"
    # before.to_dict() -> plain dict for AuditEntry snapshot field
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

__all__ = [
    "StageSnapshot",
    "capture_snapshot",
    "deep_freeze",
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
# deep_freeze -- recursive immutability transform
# ---------------------------------------------------------------------------


def deep_freeze(value: Any) -> Any:
    """Recursively transform *value* into a deeply immutable equivalent.

    Transformation rules (applied recursively to nested structures):

        - ``None``, ``bool``, ``int``, ``float``, ``str``, ``bytes``
          -- returned as-is (already immutable scalars).
        - ``dict`` -- each key and value is deep-frozen, result wrapped
          in ``types.MappingProxyType``.
        - ``MappingProxyType`` -- values are deep-frozen, result rewrapped.
        - ``list`` -- each element is deep-frozen, result converted to ``tuple``.
        - ``tuple`` -- each element is deep-frozen (stays ``tuple``).
        - ``set`` -- each element is deep-frozen, result converted to ``frozenset``.
        - ``frozenset`` -- elements deep-frozen, stays ``frozenset``.
        - ``datetime`` -- converted to ISO 8601 string.
        - ``Enum`` -- converted to its ``.value``.
        - dataclass instances -- fields extracted via ``dataclasses.asdict``-like
          logic and deep-frozen into a ``MappingProxyType``.
        - All other types -- returned as-is (assumed immutable or opaque).

    The input value is never mutated.

    Args:
        value: Any Python value to freeze.

    Returns:
        A deeply immutable equivalent of *value*.
    """
    # Immutable scalars -- pass through
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value

    # datetime -> ISO string (before Enum check, since datetime is not Enum)
    if isinstance(value, datetime):
        return value.isoformat()

    # Enum -> its value
    if isinstance(value, Enum):
        return deep_freeze(value.value)

    # dict -> MappingProxyType with frozen values
    if isinstance(value, dict):
        return MappingProxyType(
            {deep_freeze(k): deep_freeze(v) for k, v in value.items()}
        )

    # MappingProxyType -> re-freeze values
    if isinstance(value, MappingProxyType):
        return MappingProxyType(
            {deep_freeze(k): deep_freeze(v) for k, v in value.items()}
        )

    # list -> tuple with frozen elements
    if isinstance(value, list):
        return tuple(deep_freeze(item) for item in value)

    # tuple -> tuple with frozen elements
    if isinstance(value, tuple):
        return tuple(deep_freeze(item) for item in value)

    # set -> frozenset with frozen elements
    if isinstance(value, set):
        return frozenset(deep_freeze(item) for item in value)

    # frozenset -> frozenset with frozen elements
    if isinstance(value, frozenset):
        return frozenset(deep_freeze(item) for item in value)

    # dataclass -> frozen dict of fields
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return MappingProxyType(
            {f.name: deep_freeze(getattr(value, f.name)) for f in fields}
        )

    # Fallback: return as-is (assumed immutable or opaque)
    return value


# ---------------------------------------------------------------------------
# deep_thaw -- reverse transformation for serialization
# ---------------------------------------------------------------------------


def _deep_thaw(value: Any) -> Any:
    """Recursively convert frozen structures back to plain mutable types.

    Inverse of ``deep_freeze`` for serialization to JSON/YAML:

        - ``MappingProxyType`` -> ``dict``
        - ``tuple`` -> ``list``
        - ``frozenset`` -> sorted ``list`` (deterministic output)
        - All other types -> returned as-is.

    The input value is never mutated.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value

    if isinstance(value, MappingProxyType):
        return {_deep_thaw(k): _deep_thaw(v) for k, v in value.items()}

    if isinstance(value, dict):
        return {_deep_thaw(k): _deep_thaw(v) for k, v in value.items()}

    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]

    if isinstance(value, frozenset):
        return sorted(_deep_thaw(item) for item in value)

    if isinstance(value, list):
        return [_deep_thaw(item) for item in value]

    return value


# ---------------------------------------------------------------------------
# StageSnapshot -- frozen snapshot of a pipeline stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSnapshot:
    """Immutable snapshot of a pipeline stage's state.

    Bundles the three standard state categories that an audit trail
    needs to capture for any pipeline stage:

        - **inputs**: data flowing into the stage
        - **config**: configuration/parameters governing the stage
        - **partial_outputs**: any outputs produced so far (may be empty)

    All three are stored as ``MappingProxyType`` (deeply frozen dicts).

    Attributes:
        stage: Name of the pipeline stage (e.g. "nl_input", "confirmation").
            Must not be empty; leading/trailing whitespace is stripped.
        captured_at: UTC datetime when the snapshot was taken.
        inputs: Frozen dict of stage input data.
        config: Frozen dict of stage configuration.
        partial_outputs: Frozen dict of any partial output data.
    """

    stage: str
    captured_at: datetime
    inputs: MappingProxyType
    config: MappingProxyType
    partial_outputs: MappingProxyType

    def __post_init__(self) -> None:
        stripped = _require_non_empty(self.stage, "stage")
        if stripped != self.stage:
            object.__setattr__(self, "stage", stripped)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for wiki YAML persistence.

        ``MappingProxyType`` values are thawed back to plain dicts,
        tuples are converted to lists, and the timestamp is ISO 8601.
        The result is safe for ``json.dumps`` and YAML serialization.
        """
        return {
            "stage": self.stage,
            "captured_at": self.captured_at.isoformat(),
            "inputs": _deep_thaw(self.inputs),
            "config": _deep_thaw(self.config),
            "partial_outputs": _deep_thaw(self.partial_outputs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageSnapshot:
        """Deserialize from a plain dict (e.g. parsed from wiki YAML).

        Values in ``inputs``, ``config``, and ``partial_outputs`` are
        deep-frozen on construction. The ``captured_at`` field accepts
        both ISO 8601 strings and ``datetime`` objects.
        """
        ts = data["captured_at"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            stage=data["stage"],
            captured_at=ts,
            inputs=deep_freeze(data.get("inputs", {})),
            config=deep_freeze(data.get("config", {})),
            partial_outputs=deep_freeze(data.get("partial_outputs", {})),
        )


# ---------------------------------------------------------------------------
# capture_snapshot -- convenience factory
# ---------------------------------------------------------------------------


def capture_snapshot(
    stage: str,
    *,
    inputs: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    partial_outputs: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> StageSnapshot:
    """Capture a frozen snapshot of a pipeline stage's current state.

    Convenience factory that deep-freezes the provided dicts and wraps
    them in a ``StageSnapshot``. Missing arguments default to empty dicts.

    Args:
        stage: Pipeline stage name (must not be empty).
        inputs: Stage input data (deep-frozen on capture).
        config: Stage configuration (deep-frozen on capture).
        partial_outputs: Partial outputs so far (deep-frozen on capture).
        timestamp: When the snapshot was taken. Defaults to ``datetime.now(UTC)``.

    Returns:
        A fully frozen ``StageSnapshot`` instance.

    Raises:
        ValueError: If *stage* is empty or whitespace-only.
    """
    captured_at = timestamp if timestamp is not None else datetime.now(timezone.utc)
    return StageSnapshot(
        stage=stage,
        captured_at=captured_at,
        inputs=deep_freeze(inputs if inputs is not None else {}),
        config=deep_freeze(config if config is not None else {}),
        partial_outputs=deep_freeze(
            partial_outputs if partial_outputs is not None else {}
        ),
    )
