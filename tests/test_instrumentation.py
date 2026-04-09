"""Tests for the audit instrumentation module.

Covers:
- StageResult: construction, frozenness, accessors
- StageError: construction, audit chain/entry attachment
- StageAudit: context manager with before/after snapshots, duration, status
- StageAudit: error handling (records entry, does not suppress exception)
- StageAudit: record_output accumulation
- StageAudit: chain/entry properties before, during, and after context
- stage_instrumented: decorator success path returns StageResult
- stage_instrumented: decorator error path raises StageError with audit
- stage_instrumented: snapshot capture of function args
- stage_instrumented: default empty chain when none provided
- stage_instrumented: preserves function metadata via functools.wraps
- Integration: StageAudit -> AuditEntry -> AuditChain append chain
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pytest

from jules_daemon.audit.instrumentation import (
    StageAudit,
    StageError,
    StageResult,
    stage_instrumented,
)
from jules_daemon.audit_models import AuditChain, AuditEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)


def _empty_chain() -> AuditChain:
    return AuditChain.empty()


# ---------------------------------------------------------------------------
# StageResult -- construction and frozenness
# ---------------------------------------------------------------------------


class TestStageResultConstruction:
    def test_create_basic(self) -> None:
        chain = _empty_chain()
        entry = AuditEntry(
            stage="nl_input",
            timestamp=_T0,
            before_snapshot=None,
            after_snapshot=None,
            duration=1.0,
            status="success",
            error=None,
        )
        result = StageResult(value="hello", chain=chain, entry=entry)
        assert result.value == "hello"
        assert result.chain is chain
        assert result.entry is entry

    def test_value_can_be_none(self) -> None:
        chain = _empty_chain()
        entry = AuditEntry(
            stage="test",
            timestamp=_T0,
            before_snapshot=None,
            after_snapshot=None,
            duration=0.0,
            status="success",
            error=None,
        )
        result = StageResult(value=None, chain=chain, entry=entry)
        assert result.value is None

    def test_value_can_be_dict(self) -> None:
        chain = _empty_chain()
        entry = AuditEntry(
            stage="test",
            timestamp=_T0,
            before_snapshot=None,
            after_snapshot=None,
            duration=0.5,
            status="success",
            error=None,
        )
        payload = {"tests_passed": 42, "tests_failed": 0}
        result = StageResult(value=payload, chain=chain, entry=entry)
        assert result.value == payload

    def test_is_frozen(self) -> None:
        chain = _empty_chain()
        entry = AuditEntry(
            stage="test",
            timestamp=_T0,
            before_snapshot=None,
            after_snapshot=None,
            duration=0.0,
            status="success",
            error=None,
        )
        result = StageResult(value="x", chain=chain, entry=entry)
        with pytest.raises(AttributeError):
            result.value = "y"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            result.chain = chain  # type: ignore[misc]
        with pytest.raises(AttributeError):
            result.entry = entry  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StageError -- construction and audit info
# ---------------------------------------------------------------------------


class TestStageError:
    def test_create_with_chain_and_entry(self) -> None:
        chain = _empty_chain()
        entry = AuditEntry(
            stage="ssh_dispatch",
            timestamp=_T0,
            before_snapshot=None,
            after_snapshot=None,
            duration=2.0,
            status="error",
            error="connection refused",
        )
        cause = RuntimeError("connection refused")
        err = StageError(cause=cause, chain=chain, entry=entry)
        assert err.cause is cause
        assert err.chain is chain
        assert err.entry is entry
        assert "connection refused" in str(err)

    def test_is_exception(self) -> None:
        err = StageError(
            cause=ValueError("bad"),
            chain=_empty_chain(),
            entry=None,
        )
        assert isinstance(err, Exception)

    def test_entry_can_be_none(self) -> None:
        err = StageError(
            cause=RuntimeError("fail"),
            chain=_empty_chain(),
            entry=None,
        )
        assert err.entry is None


# ---------------------------------------------------------------------------
# StageAudit -- context manager basics
# ---------------------------------------------------------------------------


class TestStageAuditBasic:
    def test_chain_before_enter_returns_initial(self) -> None:
        chain = _empty_chain()
        audit = StageAudit("nl_input", chain)
        assert audit.chain is chain

    def test_entry_before_enter_is_none(self) -> None:
        audit = StageAudit("nl_input", _empty_chain())
        assert audit.entry is None

    def test_success_path_appends_entry(self) -> None:
        chain = _empty_chain()
        audit = StageAudit("nl_input", chain, inputs={"raw": "run tests"})
        with audit:
            audit.record_output({"parsed": "pytest -v"})
        assert len(audit.chain) == 1
        assert audit.entry is not None
        assert audit.entry.stage == "nl_input"
        assert audit.entry.status == "success"
        assert audit.entry.error is None

    def test_chain_is_new_instance(self) -> None:
        chain = _empty_chain()
        audit = StageAudit("nl_input", chain)
        with audit:
            pass
        assert audit.chain is not chain
        assert len(chain) == 0  # original unchanged
        assert len(audit.chain) == 1

    def test_duration_is_positive(self) -> None:
        audit = StageAudit("nl_input", _empty_chain())
        with audit:
            time.sleep(0.01)
        assert audit.entry is not None
        assert audit.entry.duration is not None
        assert audit.entry.duration >= 0.01

    def test_before_snapshot_captured(self) -> None:
        audit = StageAudit(
            "nl_input",
            _empty_chain(),
            inputs={"raw": "run tests"},
            config={"model": "gpt-4"},
        )
        with audit:
            pass
        assert audit.entry is not None
        before = audit.entry.before_snapshot
        assert isinstance(before, dict)
        assert before["stage"] == "nl_input"
        assert before["inputs"]["raw"] == "run tests"
        assert before["config"]["model"] == "gpt-4"

    def test_after_snapshot_captured(self) -> None:
        audit = StageAudit("nl_input", _empty_chain())
        with audit:
            audit.record_output({"command": "pytest -v"})
        assert audit.entry is not None
        after = audit.entry.after_snapshot
        assert isinstance(after, dict)
        assert after["stage"] == "nl_input"
        assert after["partial_outputs"]["command"] == "pytest -v"

    def test_timestamp_is_utc(self) -> None:
        before = datetime.now(timezone.utc)
        audit = StageAudit("nl_input", _empty_chain())
        with audit:
            pass
        after = datetime.now(timezone.utc)
        assert audit.entry is not None
        ts = datetime.fromisoformat(audit.entry.before_snapshot["captured_at"])
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# StageAudit -- error handling
# ---------------------------------------------------------------------------


class TestStageAuditError:
    def test_error_does_not_suppress_exception(self) -> None:
        audit = StageAudit("ssh_dispatch", _empty_chain())
        with pytest.raises(RuntimeError, match="connection refused"):
            with audit:
                raise RuntimeError("connection refused")

    def test_error_still_records_entry(self) -> None:
        audit = StageAudit("ssh_dispatch", _empty_chain())
        with pytest.raises(RuntimeError):
            with audit:
                raise RuntimeError("connection refused")
        assert audit.entry is not None
        assert audit.entry.status == "error"
        assert audit.entry.error == "connection refused"

    def test_error_chain_has_entry(self) -> None:
        chain = _empty_chain()
        audit = StageAudit("ssh_dispatch", chain)
        with pytest.raises(RuntimeError):
            with audit:
                raise RuntimeError("fail")
        assert len(audit.chain) == 1
        assert audit.chain.latest is not None
        assert audit.chain.latest.status == "error"

    def test_error_duration_recorded(self) -> None:
        audit = StageAudit("ssh_dispatch", _empty_chain())
        with pytest.raises(RuntimeError):
            with audit:
                time.sleep(0.01)
                raise RuntimeError("fail")
        assert audit.entry is not None
        assert audit.entry.duration is not None
        assert audit.entry.duration >= 0.01

    def test_error_after_snapshot_contains_error(self) -> None:
        audit = StageAudit("ssh_dispatch", _empty_chain())
        with pytest.raises(ValueError):
            with audit:
                raise ValueError("invalid command")
        assert audit.entry is not None
        after = audit.entry.after_snapshot
        assert isinstance(after, dict)
        assert "error" in after["partial_outputs"]
        assert "invalid command" in after["partial_outputs"]["error"]


# ---------------------------------------------------------------------------
# StageAudit -- record_output
# ---------------------------------------------------------------------------


class TestStageAuditRecordOutput:
    def test_no_output_recorded_uses_empty(self) -> None:
        audit = StageAudit("nl_input", _empty_chain())
        with audit:
            pass  # no record_output call
        assert audit.entry is not None
        after = audit.entry.after_snapshot
        assert after["partial_outputs"] == {}

    def test_record_output_last_call_wins(self) -> None:
        audit = StageAudit("nl_input", _empty_chain())
        with audit:
            audit.record_output({"first": True})
            audit.record_output({"second": True})
        assert audit.entry is not None
        after = audit.entry.after_snapshot
        assert after["partial_outputs"]["second"] is True
        assert "first" not in after["partial_outputs"]

    def test_record_output_makes_defensive_copy(self) -> None:
        audit = StageAudit("nl_input", _empty_chain())
        outputs: dict[str, Any] = {"key": "value"}
        with audit:
            audit.record_output(outputs)
            outputs["key"] = "mutated"  # mutate after recording
        assert audit.entry is not None
        after = audit.entry.after_snapshot
        assert after["partial_outputs"]["key"] == "value"


# ---------------------------------------------------------------------------
# StageAudit -- chaining multiple stages
# ---------------------------------------------------------------------------


class TestStageAuditChaining:
    def test_chain_two_stages(self) -> None:
        chain = _empty_chain()

        audit1 = StageAudit("nl_input", chain, inputs={"raw": "run tests"})
        with audit1:
            audit1.record_output({"command": "pytest"})

        audit2 = StageAudit(
            "confirmation",
            audit1.chain,
            inputs={"command": "pytest"},
        )
        with audit2:
            audit2.record_output({"decision": "approved"})

        assert len(audit2.chain) == 2
        assert audit2.chain.stages == ("nl_input", "confirmation")

    def test_chain_preserves_earlier_entries(self) -> None:
        chain = _empty_chain()
        audit1 = StageAudit("stage_1", chain)
        with audit1:
            pass
        audit2 = StageAudit("stage_2", audit1.chain)
        with audit2:
            pass
        assert audit2.chain.entries[0].stage == "stage_1"
        assert audit2.chain.entries[1].stage == "stage_2"


# ---------------------------------------------------------------------------
# StageAudit -- validation
# ---------------------------------------------------------------------------


class TestStageAuditValidation:
    def test_empty_stage_name_raises(self) -> None:
        with pytest.raises(ValueError, match="stage must not be empty"):
            StageAudit("", _empty_chain())

    def test_whitespace_stage_name_raises(self) -> None:
        with pytest.raises(ValueError, match="stage must not be empty"):
            StageAudit("   ", _empty_chain())


# ---------------------------------------------------------------------------
# stage_instrumented -- decorator success path
# ---------------------------------------------------------------------------


class TestStageInstrumentedSuccess:
    def test_returns_stage_result(self) -> None:
        @stage_instrumented("nl_input")
        def translate(text: str) -> str:
            return "pytest -v"

        result = translate("run all tests", _audit_chain=_empty_chain())
        assert isinstance(result, StageResult)
        assert result.value == "pytest -v"

    def test_chain_has_one_entry(self) -> None:
        @stage_instrumented("nl_input")
        def translate(text: str) -> str:
            return "pytest -v"

        result = translate("run tests", _audit_chain=_empty_chain())
        assert len(result.chain) == 1
        assert result.entry.stage == "nl_input"
        assert result.entry.status == "success"

    def test_default_empty_chain(self) -> None:
        @stage_instrumented("nl_input")
        def translate(text: str) -> str:
            return "pytest"

        result = translate("run tests")
        assert len(result.chain) == 1

    def test_appends_to_existing_chain(self) -> None:
        @stage_instrumented("confirmation")
        def confirm(command: str) -> str:
            return "approved"

        # Pre-populate a chain with one entry
        entry = AuditEntry(
            stage="nl_input",
            timestamp=_T0,
            before_snapshot=None,
            after_snapshot=None,
            duration=0.5,
            status="success",
            error=None,
        )
        chain = _empty_chain().append(entry)
        result = confirm("pytest -v", _audit_chain=chain)
        assert len(result.chain) == 2
        assert result.chain.stages == ("nl_input", "confirmation")

    def test_captures_args_in_before_snapshot(self) -> None:
        @stage_instrumented("nl_input")
        def translate(text: str, verbose: bool = False) -> str:
            return "pytest -v"

        result = translate("run tests", verbose=True)
        before = result.entry.before_snapshot
        assert isinstance(before, dict)
        inputs = before["inputs"]
        assert "run tests" in str(inputs.values())
        assert True in inputs.values() or "True" in str(inputs.values())

    def test_captures_return_value_in_after_snapshot(self) -> None:
        @stage_instrumented("nl_input")
        def translate(text: str) -> dict[str, str]:
            return {"command": "pytest -v", "risk": "low"}

        result = translate("run tests")
        after = result.entry.after_snapshot
        assert isinstance(after, dict)
        outputs = after["partial_outputs"]
        assert "pytest -v" in str(outputs)

    def test_records_duration(self) -> None:
        @stage_instrumented("slow_stage")
        def slow_fn() -> str:
            time.sleep(0.02)
            return "done"

        result = slow_fn()
        assert result.entry.duration is not None
        assert result.entry.duration >= 0.02

    def test_preserves_function_name(self) -> None:
        @stage_instrumented("nl_input")
        def my_special_function(text: str) -> str:
            """My docstring."""
            return text

        assert my_special_function.__name__ == "my_special_function"
        assert my_special_function.__doc__ == "My docstring."

    def test_config_in_before_snapshot(self) -> None:
        @stage_instrumented("nl_input", config={"model": "gpt-4", "temp": 0.0})
        def translate(text: str) -> str:
            return "pytest"

        result = translate("run tests")
        before = result.entry.before_snapshot
        assert before["config"]["model"] == "gpt-4"
        assert before["config"]["temp"] == 0.0


# ---------------------------------------------------------------------------
# stage_instrumented -- decorator error path
# ---------------------------------------------------------------------------


class TestStageInstrumentedError:
    def test_raises_stage_error(self) -> None:
        @stage_instrumented("ssh_dispatch")
        def dispatch(command: str) -> None:
            raise RuntimeError("connection refused")

        with pytest.raises(StageError) as exc_info:
            dispatch("pytest -v")
        assert exc_info.value.cause is not None
        assert "connection refused" in str(exc_info.value.cause)

    def test_stage_error_has_chain(self) -> None:
        @stage_instrumented("ssh_dispatch")
        def dispatch(command: str) -> None:
            raise RuntimeError("fail")

        with pytest.raises(StageError) as exc_info:
            dispatch("pytest -v")
        assert len(exc_info.value.chain) == 1
        assert exc_info.value.chain.latest.status == "error"

    def test_stage_error_has_entry(self) -> None:
        @stage_instrumented("ssh_dispatch")
        def dispatch(command: str) -> None:
            raise ValueError("bad command")

        with pytest.raises(StageError) as exc_info:
            dispatch("rm -rf /")
        assert exc_info.value.entry is not None
        assert exc_info.value.entry.stage == "ssh_dispatch"
        assert exc_info.value.entry.status == "error"
        assert "bad command" in exc_info.value.entry.error

    def test_stage_error_preserves_original_via_cause(self) -> None:
        original = TypeError("wrong type")

        @stage_instrumented("parse")
        def parse(text: str) -> None:
            raise original

        with pytest.raises(StageError) as exc_info:
            parse("bad input")
        assert exc_info.value.__cause__ is not None

    def test_error_appends_to_existing_chain(self) -> None:
        @stage_instrumented("ssh_dispatch")
        def dispatch(command: str) -> None:
            raise RuntimeError("fail")

        entry = AuditEntry(
            stage="confirmation",
            timestamp=_T0,
            before_snapshot=None,
            after_snapshot=None,
            duration=0.1,
            status="success",
            error=None,
        )
        chain = _empty_chain().append(entry)
        with pytest.raises(StageError) as exc_info:
            dispatch("pytest", _audit_chain=chain)
        assert len(exc_info.value.chain) == 2
        assert exc_info.value.chain.stages == ("confirmation", "ssh_dispatch")


# ---------------------------------------------------------------------------
# Integration: audit chain serialization round-trip
# ---------------------------------------------------------------------------


class TestInstrumentationIntegration:
    def test_chain_serializes_to_list(self) -> None:
        @stage_instrumented("nl_input")
        def translate(text: str) -> str:
            return "pytest -v"

        result = translate("run tests")
        serialized = result.chain.to_list()
        assert len(serialized) == 1
        assert serialized[0]["stage"] == "nl_input"
        assert serialized[0]["status"] == "success"
        assert isinstance(serialized[0]["timestamp"], str)

    def test_chain_round_trips(self) -> None:
        @stage_instrumented("nl_input")
        def translate(text: str) -> str:
            return "pytest -v"

        result = translate("run tests")
        serialized = result.chain.to_list()
        restored = AuditChain.from_list(serialized)
        assert len(restored) == 1
        assert restored.entries[0].stage == "nl_input"
        assert restored.entries[0].status == "success"

    def test_multi_stage_pipeline(self) -> None:
        @stage_instrumented("nl_input")
        def step1(text: str) -> str:
            return "pytest -v"

        @stage_instrumented("confirmation")
        def step2(command: str) -> str:
            return "approved"

        @stage_instrumented("ssh_dispatch")
        def step3(command: str) -> int:
            return 0  # exit code

        r1 = step1("run tests")
        r2 = step2(r1.value, _audit_chain=r1.chain)
        r3 = step3(r2.value, _audit_chain=r2.chain)

        assert len(r3.chain) == 3
        assert r3.chain.stages == ("nl_input", "confirmation", "ssh_dispatch")
        assert all(e.status == "success" for e in r3.chain.entries)

    def test_error_mid_pipeline_preserves_earlier_entries(self) -> None:
        @stage_instrumented("nl_input")
        def step1(text: str) -> str:
            return "pytest -v"

        @stage_instrumented("ssh_dispatch")
        def step2(command: str) -> None:
            raise RuntimeError("connection lost")

        r1 = step1("run tests")
        with pytest.raises(StageError) as exc_info:
            step2(r1.value, _audit_chain=r1.chain)

        chain = exc_info.value.chain
        assert len(chain) == 2
        assert chain.entries[0].stage == "nl_input"
        assert chain.entries[0].status == "success"
        assert chain.entries[1].stage == "ssh_dispatch"
        assert chain.entries[1].status == "error"
