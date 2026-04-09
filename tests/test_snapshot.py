"""Tests for the snapshot capture utility.

Covers:
- deep_freeze: recursive immutability for dicts, lists, sets, nested structures
- deep_freeze: special-type handling (datetime, enum, dataclass, Pydantic)
- StageSnapshot: construction, frozenness, validation
- StageSnapshot: serialization round-trip (to_dict / from_dict)
- capture_snapshot: convenience factory with defaults
- Edge cases: empty inputs, None values, deeply nested data
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

import pytest

from jules_daemon.audit.snapshot import (
    StageSnapshot,
    capture_snapshot,
    deep_freeze,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 4, 9, 12, 5, 0, tzinfo=timezone.utc)


class _SampleEnum(Enum):
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class _SampleRecord:
    name: str
    value: int


# ---------------------------------------------------------------------------
# deep_freeze -- scalars pass through unchanged
# ---------------------------------------------------------------------------


class TestDeepFreezeScalars:
    def test_none(self) -> None:
        assert deep_freeze(None) is None

    def test_string(self) -> None:
        assert deep_freeze("hello") == "hello"

    def test_int(self) -> None:
        assert deep_freeze(42) == 42

    def test_float(self) -> None:
        assert deep_freeze(3.14) == 3.14

    def test_bool(self) -> None:
        assert deep_freeze(True) is True

    def test_bytes(self) -> None:
        assert deep_freeze(b"data") == b"data"


# ---------------------------------------------------------------------------
# deep_freeze -- collections become immutable
# ---------------------------------------------------------------------------


class TestDeepFreezeCollections:
    def test_dict_becomes_mapping_proxy(self) -> None:
        result = deep_freeze({"key": "value"})
        assert isinstance(result, MappingProxyType)
        assert result["key"] == "value"

    def test_dict_is_not_mutable(self) -> None:
        result = deep_freeze({"a": 1})
        with pytest.raises(TypeError):
            result["a"] = 2  # type: ignore[index]

    def test_list_becomes_tuple(self) -> None:
        result = deep_freeze([1, 2, 3])
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_tuple_stays_tuple(self) -> None:
        result = deep_freeze((1, 2, 3))
        assert isinstance(result, tuple)
        assert result == (1, 2, 3)

    def test_set_becomes_frozenset(self) -> None:
        result = deep_freeze({1, 2, 3})
        assert isinstance(result, frozenset)
        assert result == frozenset({1, 2, 3})

    def test_frozenset_stays_frozenset(self) -> None:
        result = deep_freeze(frozenset({1, 2}))
        assert isinstance(result, frozenset)

    def test_mapping_proxy_passes_through(self) -> None:
        proxy = MappingProxyType({"x": 1})
        result = deep_freeze(proxy)
        assert isinstance(result, MappingProxyType)
        assert result["x"] == 1


# ---------------------------------------------------------------------------
# deep_freeze -- nested structures are frozen recursively
# ---------------------------------------------------------------------------


class TestDeepFreezeNested:
    def test_dict_with_nested_dict(self) -> None:
        data = {"outer": {"inner": "value"}}
        result = deep_freeze(data)
        assert isinstance(result, MappingProxyType)
        assert isinstance(result["outer"], MappingProxyType)
        assert result["outer"]["inner"] == "value"

    def test_dict_with_nested_list(self) -> None:
        data = {"items": [1, 2, {"a": 3}]}
        result = deep_freeze(data)
        assert isinstance(result["items"], tuple)
        assert isinstance(result["items"][2], MappingProxyType)
        assert result["items"][2]["a"] == 3

    def test_list_of_dicts(self) -> None:
        data = [{"a": 1}, {"b": 2}]
        result = deep_freeze(data)
        assert isinstance(result, tuple)
        assert isinstance(result[0], MappingProxyType)
        assert isinstance(result[1], MappingProxyType)

    def test_deeply_nested(self) -> None:
        data = {"l1": {"l2": {"l3": {"l4": [1, 2]}}}}
        result = deep_freeze(data)
        assert result["l1"]["l2"]["l3"]["l4"] == (1, 2)

    def test_nested_dict_not_mutable(self) -> None:
        data = {"outer": {"inner": "value"}}
        result = deep_freeze(data)
        with pytest.raises(TypeError):
            result["outer"]["inner"] = "changed"  # type: ignore[index]


# ---------------------------------------------------------------------------
# deep_freeze -- special types
# ---------------------------------------------------------------------------


class TestDeepFreezeSpecialTypes:
    def test_datetime_becomes_iso_string(self) -> None:
        result = deep_freeze(_T0)
        assert result == _T0.isoformat()
        assert isinstance(result, str)

    def test_enum_becomes_value(self) -> None:
        result = deep_freeze(_SampleEnum.LOW)
        assert result == "low"

    def test_frozen_dataclass_becomes_frozen_dict(self) -> None:
        record = _SampleRecord(name="test", value=42)
        result = deep_freeze(record)
        assert isinstance(result, MappingProxyType)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_dataclass_with_nested_values(self) -> None:
        """Dataclass fields are themselves deep-frozen."""

        @dataclass(frozen=True)
        class _Nested:
            tags: list[str]
            meta: dict[str, int]

        record = _Nested(tags=["a", "b"], meta={"x": 1})
        result = deep_freeze(record)
        assert isinstance(result, MappingProxyType)
        assert isinstance(result["tags"], tuple)
        assert isinstance(result["meta"], MappingProxyType)


# ---------------------------------------------------------------------------
# deep_freeze -- does not mutate input
# ---------------------------------------------------------------------------


class TestDeepFreezeImmutability:
    def test_original_dict_unchanged(self) -> None:
        original = {"key": [1, 2, {"nested": True}]}
        deep_freeze(original)
        # Original must remain a plain mutable dict/list
        assert isinstance(original, dict)
        assert isinstance(original["key"], list)
        assert isinstance(original["key"][2], dict)

    def test_original_list_unchanged(self) -> None:
        original = [{"a": 1}, [2, 3]]
        deep_freeze(original)
        assert isinstance(original, list)
        assert isinstance(original[0], dict)
        assert isinstance(original[1], list)


# ---------------------------------------------------------------------------
# StageSnapshot -- construction and frozenness
# ---------------------------------------------------------------------------


class TestStageSnapshotConstruction:
    def test_create_minimal(self) -> None:
        snap = StageSnapshot(
            stage="nl_input",
            captured_at=_T0,
            inputs=MappingProxyType({}),
            config=MappingProxyType({}),
            partial_outputs=MappingProxyType({}),
        )
        assert snap.stage == "nl_input"
        assert snap.captured_at == _T0
        assert dict(snap.inputs) == {}
        assert dict(snap.config) == {}
        assert dict(snap.partial_outputs) == {}

    def test_create_with_data(self) -> None:
        inputs = MappingProxyType({"raw_input": "run tests"})
        config = MappingProxyType({"model_id": "gpt-4"})
        outputs = MappingProxyType({"resolved_shell": "pytest -v"})
        snap = StageSnapshot(
            stage="command_parsed",
            captured_at=_T0,
            inputs=inputs,
            config=config,
            partial_outputs=outputs,
        )
        assert snap.inputs["raw_input"] == "run tests"
        assert snap.config["model_id"] == "gpt-4"
        assert snap.partial_outputs["resolved_shell"] == "pytest -v"

    def test_is_frozen(self) -> None:
        snap = StageSnapshot(
            stage="nl_input",
            captured_at=_T0,
            inputs=MappingProxyType({}),
            config=MappingProxyType({}),
            partial_outputs=MappingProxyType({}),
        )
        with pytest.raises(AttributeError):
            snap.stage = "other"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            snap.inputs = MappingProxyType({})  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StageSnapshot -- validation
# ---------------------------------------------------------------------------


class TestStageSnapshotValidation:
    def test_empty_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="stage must not be empty"):
            StageSnapshot(
                stage="",
                captured_at=_T0,
                inputs=MappingProxyType({}),
                config=MappingProxyType({}),
                partial_outputs=MappingProxyType({}),
            )

    def test_whitespace_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="stage must not be empty"):
            StageSnapshot(
                stage="   ",
                captured_at=_T0,
                inputs=MappingProxyType({}),
                config=MappingProxyType({}),
                partial_outputs=MappingProxyType({}),
            )

    def test_stage_is_stripped(self) -> None:
        snap = StageSnapshot(
            stage="  nl_input  ",
            captured_at=_T0,
            inputs=MappingProxyType({}),
            config=MappingProxyType({}),
            partial_outputs=MappingProxyType({}),
        )
        assert snap.stage == "nl_input"


# ---------------------------------------------------------------------------
# StageSnapshot -- serialization
# ---------------------------------------------------------------------------


class TestStageSnapshotSerialization:
    def test_to_dict_basic(self) -> None:
        snap = StageSnapshot(
            stage="nl_input",
            captured_at=_T0,
            inputs=MappingProxyType({"raw": "run tests"}),
            config=MappingProxyType({"timeout": 30}),
            partial_outputs=MappingProxyType({}),
        )
        d = snap.to_dict()
        assert d["stage"] == "nl_input"
        assert d["captured_at"] == _T0.isoformat()
        assert d["inputs"] == {"raw": "run tests"}
        assert d["config"] == {"timeout": 30}
        assert d["partial_outputs"] == {}

    def test_to_dict_returns_plain_dicts(self) -> None:
        """to_dict should return plain dicts, not MappingProxyType."""
        snap = StageSnapshot(
            stage="nl_input",
            captured_at=_T0,
            inputs=MappingProxyType({"nested": MappingProxyType({"a": 1})}),
            config=MappingProxyType({}),
            partial_outputs=MappingProxyType({}),
        )
        d = snap.to_dict()
        assert isinstance(d["inputs"], dict)
        assert isinstance(d["inputs"]["nested"], dict)

    def test_to_dict_converts_tuples_to_lists(self) -> None:
        """to_dict should convert frozen tuples back to lists for YAML compat."""
        snap = StageSnapshot(
            stage="nl_input",
            captured_at=_T0,
            inputs=MappingProxyType({"tags": ("a", "b")}),
            config=MappingProxyType({}),
            partial_outputs=MappingProxyType({}),
        )
        d = snap.to_dict()
        assert isinstance(d["inputs"]["tags"], list)
        assert d["inputs"]["tags"] == ["a", "b"]

    def test_from_dict_roundtrip(self) -> None:
        snap = StageSnapshot(
            stage="confirmation",
            captured_at=_T0,
            inputs=MappingProxyType({"command": "pytest -v"}),
            config=MappingProxyType({"host": "staging"}),
            partial_outputs=MappingProxyType({"decision": "approved"}),
        )
        d = snap.to_dict()
        restored = StageSnapshot.from_dict(d)
        assert restored.stage == snap.stage
        assert restored.captured_at == snap.captured_at
        assert dict(restored.inputs) == dict(snap.inputs)
        assert dict(restored.config) == dict(snap.config)
        assert dict(restored.partial_outputs) == dict(snap.partial_outputs)

    def test_from_dict_with_string_timestamp(self) -> None:
        d = {
            "stage": "nl_input",
            "captured_at": "2026-04-09T12:00:00+00:00",
            "inputs": {},
            "config": {},
            "partial_outputs": {},
        }
        snap = StageSnapshot.from_dict(d)
        assert snap.captured_at == _T0

    def test_from_dict_with_datetime_timestamp(self) -> None:
        d = {
            "stage": "nl_input",
            "captured_at": _T0,
            "inputs": {"a": 1},
            "config": {},
            "partial_outputs": {},
        }
        snap = StageSnapshot.from_dict(d)
        assert snap.captured_at == _T0
        assert snap.inputs["a"] == 1

    def test_from_dict_nested_values_are_frozen(self) -> None:
        d = {
            "stage": "nl_input",
            "captured_at": _T0,
            "inputs": {"nested": {"deep": [1, 2]}},
            "config": {},
            "partial_outputs": {},
        }
        snap = StageSnapshot.from_dict(d)
        assert isinstance(snap.inputs, MappingProxyType)
        assert isinstance(snap.inputs["nested"], MappingProxyType)
        assert isinstance(snap.inputs["nested"]["deep"], tuple)


# ---------------------------------------------------------------------------
# capture_snapshot -- convenience factory
# ---------------------------------------------------------------------------


class TestCaptureSnapshot:
    def test_basic_capture(self) -> None:
        snap = capture_snapshot(
            stage="nl_input",
            inputs={"raw_input": "run tests"},
            config={"model": "gpt-4"},
            partial_outputs={},
            timestamp=_T0,
        )
        assert isinstance(snap, StageSnapshot)
        assert snap.stage == "nl_input"
        assert snap.captured_at == _T0
        assert isinstance(snap.inputs, MappingProxyType)
        assert snap.inputs["raw_input"] == "run tests"
        assert isinstance(snap.config, MappingProxyType)
        assert snap.config["model"] == "gpt-4"

    def test_defaults_to_empty_dicts(self) -> None:
        snap = capture_snapshot(stage="nl_input", timestamp=_T0)
        assert dict(snap.inputs) == {}
        assert dict(snap.config) == {}
        assert dict(snap.partial_outputs) == {}

    def test_auto_timestamp(self) -> None:
        """When timestamp is not provided, one is generated automatically."""
        before = datetime.now(timezone.utc)
        snap = capture_snapshot(stage="nl_input")
        after = datetime.now(timezone.utc)
        assert before <= snap.captured_at <= after

    def test_freezes_nested_inputs(self) -> None:
        snap = capture_snapshot(
            stage="command_parsed",
            inputs={"args": [1, 2, {"key": "val"}]},
            timestamp=_T0,
        )
        assert isinstance(snap.inputs["args"], tuple)
        assert isinstance(snap.inputs["args"][2], MappingProxyType)

    def test_does_not_mutate_originals(self) -> None:
        inputs = {"items": [1, 2, 3]}
        config = {"host": "prod"}
        capture_snapshot(
            stage="nl_input",
            inputs=inputs,
            config=config,
            timestamp=_T0,
        )
        # Originals must remain mutable
        assert isinstance(inputs["items"], list)
        assert isinstance(config, dict)

    def test_empty_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="stage must not be empty"):
            capture_snapshot(stage="", timestamp=_T0)


# ---------------------------------------------------------------------------
# StageSnapshot -- as audit-entry snapshot value
# ---------------------------------------------------------------------------


class TestSnapshotAsAuditValue:
    """Verify StageSnapshot.to_dict() produces a value suitable for
    AuditEntry before_snapshot/after_snapshot fields."""

    def test_to_dict_is_json_serializable(self) -> None:
        import json

        snap = capture_snapshot(
            stage="ssh_dispatched",
            inputs={"command": "pytest -v", "host": "staging"},
            config={"port": 22, "timeout": 30},
            partial_outputs={"pid": 12345},
            timestamp=_T0,
        )
        d = snap.to_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_to_dict_has_expected_keys(self) -> None:
        snap = capture_snapshot(stage="nl_input", timestamp=_T0)
        d = snap.to_dict()
        assert set(d.keys()) == {
            "stage",
            "captured_at",
            "inputs",
            "config",
            "partial_outputs",
        }
