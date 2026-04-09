"""Tests for AuditEntry and AuditChain data models.

Covers:
- AuditEntry construction, frozenness, validation, and serialization
- AuditChain immutable append, serialization, and filtering
- Edge cases: empty chains, metadata with nested structures, snapshot types
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jules_daemon.audit.chain import AuditEntry, AuditChain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 4, 9, 12, 5, 0, tzinfo=timezone.utc)
_T2 = datetime(2026, 4, 9, 12, 10, 0, tzinfo=timezone.utc)


def _make_entry(
    timestamp: datetime = _T0,
    stage: str = "nl_input",
    input_snapshot: object = "raw user text",
    output_snapshot: object = "parsed command",
    metadata: dict[str, object] | None = None,
) -> AuditEntry:
    return AuditEntry(
        timestamp=timestamp,
        stage=stage,
        input_snapshot=input_snapshot,
        output_snapshot=output_snapshot,
        metadata=metadata if metadata is not None else {},
    )


# ---------------------------------------------------------------------------
# AuditEntry -- construction
# ---------------------------------------------------------------------------


class TestAuditEntryConstruction:
    def test_create_valid(self) -> None:
        entry = _make_entry()
        assert entry.timestamp == _T0
        assert entry.stage == "nl_input"
        assert entry.input_snapshot == "raw user text"
        assert entry.output_snapshot == "parsed command"
        assert entry.metadata == {}

    def test_create_with_metadata(self) -> None:
        meta = {"model_id": "gpt-4", "latency_ms": 150}
        entry = _make_entry(metadata=meta)
        assert entry.metadata == {"model_id": "gpt-4", "latency_ms": 150}

    def test_create_with_dict_snapshots(self) -> None:
        inp = {"raw_input": "run tests", "source": "cli"}
        out = {"resolved_shell": "pytest -v", "risk": "low"}
        entry = _make_entry(input_snapshot=inp, output_snapshot=out)
        assert entry.input_snapshot == inp
        assert entry.output_snapshot == out

    def test_create_with_none_snapshots(self) -> None:
        entry = _make_entry(input_snapshot=None, output_snapshot=None)
        assert entry.input_snapshot is None
        assert entry.output_snapshot is None


# ---------------------------------------------------------------------------
# AuditEntry -- frozenness
# ---------------------------------------------------------------------------


class TestAuditEntryFrozen:
    def test_cannot_set_timestamp(self) -> None:
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.timestamp = _T1  # type: ignore[misc]

    def test_cannot_set_stage(self) -> None:
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.stage = "other"  # type: ignore[misc]

    def test_cannot_set_input_snapshot(self) -> None:
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.input_snapshot = "changed"  # type: ignore[misc]

    def test_cannot_set_output_snapshot(self) -> None:
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.output_snapshot = "changed"  # type: ignore[misc]

    def test_cannot_set_metadata(self) -> None:
        entry = _make_entry()
        with pytest.raises(AttributeError):
            entry.metadata = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AuditEntry -- validation
# ---------------------------------------------------------------------------


class TestAuditEntryValidation:
    def test_empty_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="stage must not be empty"):
            AuditEntry(
                timestamp=_T0,
                stage="",
                input_snapshot=None,
                output_snapshot=None,
                metadata={},
            )

    def test_whitespace_only_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="stage must not be empty"):
            AuditEntry(
                timestamp=_T0,
                stage="   ",
                input_snapshot=None,
                output_snapshot=None,
                metadata={},
            )

    def test_stage_is_stripped(self) -> None:
        entry = AuditEntry(
            timestamp=_T0,
            stage="  nl_input  ",
            input_snapshot=None,
            output_snapshot=None,
            metadata={},
        )
        assert entry.stage == "nl_input"


# ---------------------------------------------------------------------------
# AuditEntry -- serialization
# ---------------------------------------------------------------------------


class TestAuditEntrySerialization:
    def test_to_dict_basic(self) -> None:
        entry = _make_entry()
        d = entry.to_dict()
        assert d["timestamp"] == _T0.isoformat()
        assert d["stage"] == "nl_input"
        assert d["input_snapshot"] == "raw user text"
        assert d["output_snapshot"] == "parsed command"
        assert d["metadata"] == {}

    def test_to_dict_with_nested_metadata(self) -> None:
        meta = {"model": {"id": "gpt-4", "version": 1}, "tags": ["test"]}
        entry = _make_entry(metadata=meta)
        d = entry.to_dict()
        assert d["metadata"] == meta

    def test_to_dict_with_dict_snapshots(self) -> None:
        inp = {"raw": "run tests"}
        out = {"shell": "pytest"}
        entry = _make_entry(input_snapshot=inp, output_snapshot=out)
        d = entry.to_dict()
        assert d["input_snapshot"] == inp
        assert d["output_snapshot"] == out

    def test_to_dict_with_none_snapshots(self) -> None:
        entry = _make_entry(input_snapshot=None, output_snapshot=None)
        d = entry.to_dict()
        assert d["input_snapshot"] is None
        assert d["output_snapshot"] is None

    def test_from_dict_roundtrip(self) -> None:
        entry = _make_entry(
            metadata={"key": "value"},
            input_snapshot={"data": [1, 2, 3]},
            output_snapshot={"result": True},
        )
        d = entry.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored == entry

    def test_from_dict_with_string_timestamp(self) -> None:
        d = {
            "timestamp": "2026-04-09T12:00:00+00:00",
            "stage": "confirmation",
            "input_snapshot": "cmd",
            "output_snapshot": "approved",
            "metadata": {},
        }
        entry = AuditEntry.from_dict(d)
        assert entry.timestamp == _T0
        assert entry.stage == "confirmation"

    def test_from_dict_with_datetime_timestamp(self) -> None:
        d = {
            "timestamp": _T0,
            "stage": "nl_input",
            "input_snapshot": None,
            "output_snapshot": None,
            "metadata": {},
        }
        entry = AuditEntry.from_dict(d)
        assert entry.timestamp == _T0


# ---------------------------------------------------------------------------
# AuditChain -- construction
# ---------------------------------------------------------------------------


class TestAuditChainConstruction:
    def test_empty_chain(self) -> None:
        chain = AuditChain.empty()
        assert len(chain) == 0
        assert chain.entries == ()

    def test_empty_chain_is_frozen(self) -> None:
        chain = AuditChain.empty()
        with pytest.raises(AttributeError):
            chain.entries = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AuditChain -- immutable append
# ---------------------------------------------------------------------------


class TestAuditChainAppend:
    def test_append_returns_new_chain(self) -> None:
        chain = AuditChain.empty()
        entry = _make_entry()
        new_chain = chain.append(entry)
        assert new_chain is not chain
        assert len(new_chain) == 1
        assert len(chain) == 0  # original unchanged

    def test_append_preserves_order(self) -> None:
        e1 = _make_entry(timestamp=_T0, stage="stage_1")
        e2 = _make_entry(timestamp=_T1, stage="stage_2")
        e3 = _make_entry(timestamp=_T2, stage="stage_3")

        chain = AuditChain.empty().append(e1).append(e2).append(e3)
        assert len(chain) == 3
        assert chain.entries[0].stage == "stage_1"
        assert chain.entries[1].stage == "stage_2"
        assert chain.entries[2].stage == "stage_3"

    def test_append_does_not_mutate_original(self) -> None:
        e1 = _make_entry(stage="first")
        e2 = _make_entry(stage="second")

        chain1 = AuditChain.empty().append(e1)
        chain2 = chain1.append(e2)

        assert len(chain1) == 1
        assert chain1.entries[0].stage == "first"
        assert len(chain2) == 2
        assert chain2.entries[0].stage == "first"
        assert chain2.entries[1].stage == "second"

    def test_multiple_appends_produce_correct_length(self) -> None:
        chain = AuditChain.empty()
        for i in range(10):
            chain = chain.append(_make_entry(stage=f"stage_{i}"))
        assert len(chain) == 10


# ---------------------------------------------------------------------------
# AuditChain -- serialization
# ---------------------------------------------------------------------------


class TestAuditChainSerialization:
    def test_to_list_empty(self) -> None:
        chain = AuditChain.empty()
        result = chain.to_list()
        assert result == []

    def test_to_list_single_entry(self) -> None:
        entry = _make_entry()
        chain = AuditChain.empty().append(entry)
        result = chain.to_list()
        assert len(result) == 1
        assert result[0]["stage"] == "nl_input"
        assert result[0]["timestamp"] == _T0.isoformat()

    def test_to_list_multiple_entries(self) -> None:
        e1 = _make_entry(timestamp=_T0, stage="stage_1")
        e2 = _make_entry(timestamp=_T1, stage="stage_2")
        chain = AuditChain.empty().append(e1).append(e2)
        result = chain.to_list()
        assert len(result) == 2
        assert result[0]["stage"] == "stage_1"
        assert result[1]["stage"] == "stage_2"

    def test_to_list_returns_new_list(self) -> None:
        chain = AuditChain.empty().append(_make_entry())
        list1 = chain.to_list()
        list2 = chain.to_list()
        assert list1 == list2
        assert list1 is not list2

    def test_from_list_roundtrip(self) -> None:
        e1 = _make_entry(timestamp=_T0, stage="a", metadata={"k": "v"})
        e2 = _make_entry(timestamp=_T1, stage="b", input_snapshot={"d": 1})
        chain = AuditChain.empty().append(e1).append(e2)

        data = chain.to_list()
        restored = AuditChain.from_list(data)
        assert len(restored) == 2
        assert restored.entries == chain.entries

    def test_from_list_empty(self) -> None:
        chain = AuditChain.from_list([])
        assert len(chain) == 0


# ---------------------------------------------------------------------------
# AuditChain -- accessors and properties
# ---------------------------------------------------------------------------


class TestAuditChainAccessors:
    def test_len(self) -> None:
        chain = AuditChain.empty().append(_make_entry())
        assert len(chain) == 1

    def test_latest_empty_chain(self) -> None:
        chain = AuditChain.empty()
        assert chain.latest is None

    def test_latest_returns_last_appended(self) -> None:
        e1 = _make_entry(stage="first")
        e2 = _make_entry(stage="last")
        chain = AuditChain.empty().append(e1).append(e2)
        assert chain.latest is not None
        assert chain.latest.stage == "last"

    def test_stages_empty_chain(self) -> None:
        chain = AuditChain.empty()
        assert chain.stages == ()

    def test_stages_returns_ordered_stage_names(self) -> None:
        e1 = _make_entry(stage="nl_input")
        e2 = _make_entry(stage="command_parsed")
        e3 = _make_entry(stage="confirmation")
        chain = AuditChain.empty().append(e1).append(e2).append(e3)
        assert chain.stages == ("nl_input", "command_parsed", "confirmation")

    def test_by_stage_returns_matching_entries(self) -> None:
        e1 = _make_entry(stage="nl_input")
        e2 = _make_entry(stage="confirmation")
        e3 = _make_entry(stage="nl_input")
        chain = AuditChain.empty().append(e1).append(e2).append(e3)
        matches = chain.by_stage("nl_input")
        assert len(matches) == 2
        assert all(e.stage == "nl_input" for e in matches)

    def test_by_stage_no_match(self) -> None:
        chain = AuditChain.empty().append(_make_entry(stage="nl_input"))
        matches = chain.by_stage("nonexistent")
        assert matches == ()


# ---------------------------------------------------------------------------
# AuditChain -- equality
# ---------------------------------------------------------------------------


class TestAuditChainEquality:
    def test_equal_chains(self) -> None:
        e1 = _make_entry(stage="a")
        chain1 = AuditChain.empty().append(e1)
        chain2 = AuditChain.empty().append(e1)
        assert chain1 == chain2

    def test_unequal_chains(self) -> None:
        e1 = _make_entry(stage="a")
        e2 = _make_entry(stage="b")
        chain1 = AuditChain.empty().append(e1)
        chain2 = AuditChain.empty().append(e2)
        assert chain1 != chain2
