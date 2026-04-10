"""Tests for the full-chain audit wiring in the run pipeline.

Validates that:
- :mod:`jules_daemon.audit.run_audit_builder` constructs the expected
  immutable records from runtime values.
- The :class:`RequestHandler` persists an audit file when a run is
  denied at the confirmation stage (no SSH execution).
- The :class:`RequestHandler` persists a fully populated audit file
  when a run completes successfully or fails.
- Audit failures never crash the run pipeline.
- Sensitive credentials never appear in the audit artifacts.

The tests exercise the real wiki writer against a temporary directory
so the full serialization -> parse round-trip is covered end-to-end.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.audit.models import (
    AuditRecord,
    ConfirmationDecision,
    ConfirmationRecord,
    NLInputRecord,
    ParsedCommandRecord,
    PipelineStage,
    SSHExecutionRecord,
    StructuredResultRecord,
)
from jules_daemon.audit.run_audit_builder import (
    AUDIT_OUTPUT_LIMIT,
    build_confirmation_record,
    build_nl_input_record,
    build_parsed_command_record,
    build_ssh_execution_record,
    build_structured_result_record,
    create_initial_audit,
    safe_write_audit,
    safe_write_audit_async,
    truncate_text,
)
from jules_daemon.execution.run_pipeline import RunResult
from jules_daemon.ipc.framing import (
    MessageEnvelope,
    MessageType,
    encode_frame,
)
from jules_daemon.ipc.request_handler import (
    RequestHandler,
    RequestHandlerConfig,
)
from jules_daemon.ipc.server import ClientConnection
from jules_daemon.wiki.audit_writer import list_audit_files, read_audit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(seconds: int = 0) -> datetime:
    """Return a deterministic UTC timestamp for testing."""
    return datetime(2026, 4, 9, 12, 0, seconds, tzinfo=timezone.utc)


def _make_client() -> ClientConnection:
    """Build a stub ClientConnection with async reader/writer mocks."""
    return ClientConnection(
        client_id="test-client-audit",
        reader=AsyncMock(spec=asyncio.StreamReader),
        writer=AsyncMock(spec=asyncio.StreamWriter),
        connected_at="2026-04-09T12:00:00Z",
    )


def _make_request(
    payload: dict[str, Any],
    msg_id: str = "req-audit-001",
) -> MessageEnvelope:
    """Build a REQUEST-type envelope."""
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload=payload,
    )


def _make_success_result(
    *,
    run_id: str = "run-success-001",
    command: str = "pytest -v",
    host: str = "staging.example.com",
    user: str = "deploy",
    stdout: str = "All tests passed.",
    stderr: str = "",
) -> RunResult:
    return RunResult(
        success=True,
        run_id=run_id,
        command=command,
        target_host=host,
        target_user=user,
        exit_code=0,
        stdout=stdout,
        stderr=stderr,
        error=None,
        duration_seconds=12.5,
        started_at=_ts(),
        completed_at=_ts(12),
    )


def _make_failure_result(
    *,
    run_id: str = "run-failure-001",
    command: str = "pytest -v",
    host: str = "staging.example.com",
    user: str = "deploy",
    stderr: str = "AssertionError: 1 test failed",
) -> RunResult:
    return RunResult(
        success=False,
        run_id=run_id,
        command=command,
        target_host=host,
        target_user=user,
        exit_code=1,
        stdout="",
        stderr=stderr,
        error=f"Command exited with code 1",
        duration_seconds=7.25,
        started_at=_ts(),
        completed_at=_ts(7),
    )


# ---------------------------------------------------------------------------
# Builder tests (unit)
# ---------------------------------------------------------------------------


class TestBuildNLInputRecord:
    """Tests for :func:`build_nl_input_record`."""

    def test_returns_record_with_raw_input(self) -> None:
        record = build_nl_input_record(raw_input="run the suite")
        assert isinstance(record, NLInputRecord)
        assert record.raw_input == "run the suite"

    def test_default_source_is_ipc(self) -> None:
        record = build_nl_input_record(raw_input="run the suite")
        assert record.source == "ipc"

    def test_custom_source_is_preserved(self) -> None:
        record = build_nl_input_record(
            raw_input="run the suite",
            source="queue",
        )
        assert record.source == "queue"

    def test_uses_current_time_when_no_timestamp(self) -> None:
        before = datetime.now(timezone.utc)
        record = build_nl_input_record(raw_input="go")
        after = datetime.now(timezone.utc)
        assert before <= record.timestamp <= after

    def test_empty_input_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            build_nl_input_record(raw_input="")


class TestBuildParsedCommandRecord:
    """Tests for :func:`build_parsed_command_record`."""

    def test_direct_command_uses_direct_model_id(self) -> None:
        record = build_parsed_command_record(
            natural_language="pytest -v",
            resolved_shell="pytest -v",
            is_direct_command=True,
            model_id=None,
        )
        assert record.model_id == "direct"
        assert "Direct shell command" in record.explanation

    def test_llm_command_keeps_model_id(self) -> None:
        record = build_parsed_command_record(
            natural_language="run all tests",
            resolved_shell="pytest -v",
            is_direct_command=False,
            model_id="openai:gpt-4",
        )
        assert record.model_id == "openai:gpt-4"

    def test_llm_with_none_model_defaults(self) -> None:
        record = build_parsed_command_record(
            natural_language="run all tests",
            resolved_shell="pytest -v",
            is_direct_command=False,
            model_id=None,
        )
        assert record.model_id == "llm"

    def test_default_risk_level_is_low(self) -> None:
        record = build_parsed_command_record(
            natural_language="pytest",
            resolved_shell="pytest",
            is_direct_command=True,
            model_id=None,
        )
        assert record.risk_level == "low"

    def test_affected_paths_coerced_to_tuple(self) -> None:
        record = build_parsed_command_record(
            natural_language="ls",
            resolved_shell="ls",
            is_direct_command=True,
            model_id=None,
            affected_paths=("/tmp",),
        )
        assert record.affected_paths == ("/tmp",)


class TestBuildConfirmationRecord:
    """Tests for :func:`build_confirmation_record`."""

    def test_approved_plain(self) -> None:
        record = build_confirmation_record(
            original_command="pytest",
            final_command="pytest",
            approved=True,
            edited=False,
        )
        assert record.decision == ConfirmationDecision.APPROVED

    def test_approved_edited(self) -> None:
        record = build_confirmation_record(
            original_command="pytest",
            final_command="pytest -v",
            approved=True,
            edited=True,
        )
        assert record.decision == ConfirmationDecision.EDITED

    def test_denied(self) -> None:
        record = build_confirmation_record(
            original_command="rm -rf /",
            final_command="",
            approved=False,
            edited=False,
        )
        assert record.decision == ConfirmationDecision.DENIED

    def test_custom_approver(self) -> None:
        record = build_confirmation_record(
            original_command="ls",
            final_command="ls",
            approved=True,
            edited=False,
            decided_by="alice@example.com",
        )
        assert record.decided_by == "alice@example.com"

    def test_default_approver_falls_back_to_user_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USER", "bob")
        monkeypatch.delenv("USERNAME", raising=False)
        record = build_confirmation_record(
            original_command="ls",
            final_command="ls",
            approved=True,
            edited=False,
        )
        assert record.decided_by == "bob"

    def test_fallback_when_no_env_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("USER", raising=False)
        monkeypatch.delenv("USERNAME", raising=False)
        record = build_confirmation_record(
            original_command="ls",
            final_command="ls",
            approved=True,
            edited=False,
        )
        assert record.decided_by == "unknown"


class TestBuildSSHExecutionRecord:
    """Tests for :func:`build_ssh_execution_record`."""

    def test_basic_record(self) -> None:
        record = build_ssh_execution_record(
            host="staging.example.com",
            user="deploy",
            port=22,
            command="pytest",
            session_id="run-abc",
            started_at=_ts(),
            completed_at=_ts(5),
            exit_code=0,
            duration_seconds=5.0,
        )
        assert isinstance(record, SSHExecutionRecord)
        assert record.host == "staging.example.com"
        assert record.exit_code == 0
        assert record.duration_seconds == 5.0

    def test_negative_duration_clamped_to_zero(self) -> None:
        record = build_ssh_execution_record(
            host="host",
            user="user",
            port=22,
            command="cmd",
            session_id="sess",
            started_at=_ts(),
            completed_at=_ts(),
            exit_code=0,
            duration_seconds=-0.001,
        )
        assert record.duration_seconds == 0.0

    def test_none_duration_preserved(self) -> None:
        record = build_ssh_execution_record(
            host="host",
            user="user",
            port=22,
            command="cmd",
            session_id="sess",
            started_at=_ts(),
            completed_at=_ts(),
            exit_code=None,
            duration_seconds=None,
        )
        assert record.duration_seconds is None
        assert record.exit_code is None


class TestBuildStructuredResultRecord:
    """Tests for :func:`build_structured_result_record`."""

    def test_success_defaults(self) -> None:
        record = build_structured_result_record(
            success=True,
            exit_code=0,
            summary="ok",
        )
        assert record.success is True
        assert record.exit_code == 0
        assert record.error_message is None

    def test_none_exit_code_mapped_to_minus_one(self) -> None:
        record = build_structured_result_record(
            success=False,
            exit_code=None,
            summary="connection failed",
        )
        assert record.exit_code == -1

    def test_error_message_preserved(self) -> None:
        record = build_structured_result_record(
            success=False,
            exit_code=2,
            summary="failed",
            error_message="boom",
        )
        assert record.error_message == "boom"


class TestTruncateText:
    """Tests for :func:`truncate_text`."""

    def test_none_becomes_empty_string(self) -> None:
        assert truncate_text(None) == ""

    def test_short_text_passthrough(self) -> None:
        assert truncate_text("hello") == "hello"

    def test_long_text_truncated(self) -> None:
        text = "a" * (AUDIT_OUTPUT_LIMIT + 500)
        result = truncate_text(text)
        assert len(result) <= AUDIT_OUTPUT_LIMIT + 80
        assert "truncated" in result

    def test_custom_limit(self) -> None:
        result = truncate_text("abcdefgh", limit=3)
        assert result.startswith("abc")
        assert "truncated" in result


# ---------------------------------------------------------------------------
# Safe write wrappers
# ---------------------------------------------------------------------------


class TestSafeWriteAudit:
    """Tests for :func:`safe_write_audit` and its async variant."""

    def _complete_record(self) -> AuditRecord:
        nl = build_nl_input_record(raw_input="pytest")
        record = create_initial_audit(run_id="run-001", nl_input=nl)
        record = record.with_parsed_command(
            build_parsed_command_record(
                natural_language="pytest",
                resolved_shell="pytest",
                is_direct_command=True,
                model_id=None,
            )
        )
        record = record.with_confirmation(
            build_confirmation_record(
                original_command="pytest",
                final_command="pytest",
                approved=True,
                edited=False,
            )
        )
        record = record.with_ssh_execution(
            build_ssh_execution_record(
                host="staging.example.com",
                user="deploy",
                port=22,
                command="pytest",
                session_id="run-001",
                started_at=_ts(),
                completed_at=_ts(5),
                exit_code=0,
                duration_seconds=5.0,
            )
        )
        record = record.with_structured_result(
            build_structured_result_record(
                success=True, exit_code=0, summary="ok",
            )
        )
        return record

    def test_sync_write_creates_file(self, tmp_path: Path) -> None:
        record = self._complete_record()
        outcome = safe_write_audit(wiki_root=tmp_path, record=record)
        assert outcome is not None
        assert outcome.file_path.exists()
        assert outcome.correlation_id == record.correlation_id

    @pytest.mark.asyncio
    async def test_async_write_creates_file(self, tmp_path: Path) -> None:
        record = self._complete_record()
        outcome = await safe_write_audit_async(
            wiki_root=tmp_path, record=record,
        )
        assert outcome is not None
        assert outcome.file_path.exists()

    def test_sync_write_swallows_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record = self._complete_record()

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("disk on fire")

        from jules_daemon.wiki import audit_writer as aw_module

        monkeypatch.setattr(aw_module, "write_audit", _boom)
        result = safe_write_audit(wiki_root=tmp_path, record=record)
        assert result is None  # no exception propagated

    @pytest.mark.asyncio
    async def test_async_write_swallows_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        record = self._complete_record()

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("disk on fire")

        from jules_daemon.wiki import audit_writer as aw_module

        monkeypatch.setattr(aw_module, "write_audit", _boom)
        result = await safe_write_audit_async(
            wiki_root=tmp_path, record=record,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Integration: _handle_run denied path writes an audit file
# ---------------------------------------------------------------------------


class TestHandleRunDeniedAudit:
    """End-to-end: a denied run leaves a partial audit file behind."""

    @pytest.mark.asyncio
    async def test_denied_run_writes_audit_file(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        # User denies the confirmation
        deny_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="deny-audit",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": False},
        )
        frame = encode_frame(deny_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[frame[:4], frame[4:]],
        )

        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "pytest -v",
        })

        response = await handler.handle_message(envelope, client)

        assert response.payload["status"] == "denied"

        files = list_audit_files(tmp_path)
        assert len(files) == 1
        record = read_audit(files[0])
        assert record is not None
        assert record.pipeline_stage == PipelineStage.CONFIRMATION
        assert record.confirmation is not None
        assert record.confirmation.decision == ConfirmationDecision.DENIED
        assert record.ssh_execution is None
        assert record.structured_result is None

    @pytest.mark.asyncio
    async def test_denied_audit_has_nl_input_and_parsed_command(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        deny_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="deny-audit-2",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": False},
        )
        frame = encode_frame(deny_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[frame[:4], frame[4:]],
        )

        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "pytest -v",
        })
        await handler.handle_message(envelope, client)

        files = list_audit_files(tmp_path)
        record = read_audit(files[0])
        assert record is not None
        assert record.nl_input.raw_input == "pytest -v"
        assert record.parsed_command is not None
        assert record.parsed_command.resolved_shell == "pytest -v"
        assert record.parsed_command.model_id == "direct"


# ---------------------------------------------------------------------------
# Integration: _finalize_and_write_audit directly
# ---------------------------------------------------------------------------


class TestFinalizeAndWriteAudit:
    """Tests for :meth:`RequestHandler._finalize_and_write_audit`."""

    def _make_partial_audit(self, *, command: str = "pytest") -> AuditRecord:
        nl = build_nl_input_record(raw_input=command)
        audit = create_initial_audit(run_id="run-initial", nl_input=nl)
        audit = audit.with_parsed_command(
            build_parsed_command_record(
                natural_language=command,
                resolved_shell=command,
                is_direct_command=True,
                model_id=None,
            )
        )
        audit = audit.with_confirmation(
            build_confirmation_record(
                original_command=command,
                final_command=command,
                approved=True,
                edited=False,
            )
        )
        return audit

    @pytest.mark.asyncio
    async def test_successful_run_writes_full_chain(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit()
        result = _make_success_result(stdout="42 tests passed")

        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="pytest",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        files = list_audit_files(tmp_path)
        assert len(files) == 1
        record = read_audit(files[0])
        assert record is not None
        assert record.pipeline_stage == PipelineStage.EXECUTION_COMPLETE
        assert record.ssh_execution is not None
        assert record.ssh_execution.exit_code == 0
        assert record.ssh_execution.host == "staging.example.com"
        assert record.structured_result is not None
        assert record.structured_result.success is True
        assert "42 tests passed" in record.structured_result.summary
        # run_id is realigned to the result run_id
        assert record.run_id == result.run_id

    @pytest.mark.asyncio
    async def test_failed_run_writes_error_details(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit()
        result = _make_failure_result(stderr="AssertionError: boom")

        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="pytest",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        files = list_audit_files(tmp_path)
        assert len(files) == 1
        record = read_audit(files[0])
        assert record is not None
        assert record.structured_result is not None
        assert record.structured_result.success is False
        assert record.structured_result.exit_code == 1
        assert record.structured_result.error_message is not None
        assert "AssertionError: boom" in record.structured_result.error_message

    @pytest.mark.asyncio
    async def test_large_output_truncated_in_audit(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit()
        giant_stdout = "x" * (AUDIT_OUTPUT_LIMIT * 3)
        result = _make_success_result(stdout=giant_stdout)

        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="pytest",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        files = list_audit_files(tmp_path)
        record = read_audit(files[0])
        assert record is not None
        assert record.structured_result is not None
        # Summary is bounded: never larger than the limit + a small marker
        assert len(record.structured_result.summary) < AUDIT_OUTPUT_LIMIT + 200
        assert "truncated" in record.structured_result.summary

    @pytest.mark.asyncio
    async def test_audit_failure_does_not_propagate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the audit writer crashes, the run must continue cleanly."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit()
        result = _make_success_result()

        from jules_daemon.wiki import audit_writer as aw_module

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("disk unavailable")

        monkeypatch.setattr(aw_module, "write_audit", _boom)

        # Must not raise
        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="pytest",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

    @pytest.mark.asyncio
    async def test_audit_does_not_contain_passwords(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sensitive environment variables must never leak into audit files."""
        # Set a secret that would leak if the audit writer were careless
        monkeypatch.setenv("JULES_SSH_PASSWORD", "hunter2-should-not-leak")

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit()
        result = _make_success_result()

        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="pytest",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        files = list_audit_files(tmp_path)
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "hunter2-should-not-leak" not in content


# ---------------------------------------------------------------------------
# Integration: queued auto-start builds a minimal audit
# ---------------------------------------------------------------------------


class TestKnowledgeWiringInAudit:
    """Verify the wiki knowledge loop integrates with the audit flow."""

    def _make_partial_audit(self, *, command: str = "pytest") -> AuditRecord:
        nl = build_nl_input_record(raw_input=command)
        audit = create_initial_audit(run_id="run-knowledge", nl_input=nl)
        audit = audit.with_parsed_command(
            build_parsed_command_record(
                natural_language=command,
                resolved_shell=command,
                is_direct_command=True,
                model_id=None,
            )
        )
        audit = audit.with_confirmation(
            build_confirmation_record(
                original_command=command,
                final_command=command,
                approved=True,
                edited=False,
            )
        )
        return audit

    @pytest.mark.asyncio
    async def test_first_run_writes_knowledge_file_when_llm_present(
        self, tmp_path: Path
    ) -> None:
        """When the LLM client returns observations, a knowledge file is created."""
        from jules_daemon.wiki.test_knowledge import (
            derive_test_slug,
            knowledge_file_path,
            load_test_knowledge,
        )

        # Lightweight stub for both the summarizer and the extractor
        # calls. Both modules read the same client; the response is
        # JSON that satisfies both prompts (extra fields are ignored).
        class _Resp:
            def __init__(self, content: str) -> None:
                self.choices = [
                    type(
                        "_C",
                        (),
                        {
                            "message": type(
                                "_M", (), {"content": content}
                            )()
                        },
                    )()
                ]

        class _Comps:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def create(self, **kwargs: Any) -> _Resp:
                self.calls.append(kwargs)
                return _Resp(
                    '{"passed": 1, "failed": 0, "skipped": 0, "total": 1, '
                    '"key_failures": [], "narrative": "All tests passed.", '
                    '"purpose": "runs the agent suite", '
                    '"output_format": "iteration logs", '
                    '"common_failures": [], '
                    '"normal_behavior": "all iterations pass quickly"}'
                )

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Comps()

        class _Client:
            def __init__(self) -> None:
                self.chat = _Chat()

        from jules_daemon.llm.config import LLMConfig

        llm_config = LLMConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            default_model="openai:default:gpt-4o",
        )
        client = _Client()
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=client,  # type: ignore[arg-type]
            llm_config=llm_config,
        )
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit(command="python3 agent_test.py")
        result = _make_success_result(
            command="python3 agent_test.py",
            stdout="iteration 1: PASSED",
        )

        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="python3 agent_test.py",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        slug = derive_test_slug("python3 agent_test.py")
        assert slug == "agent-test-py"
        loaded = load_test_knowledge(tmp_path, slug)
        assert loaded is not None
        assert loaded.runs_observed == 1
        assert loaded.purpose == "runs the agent suite"
        # File path is in the expected location
        path = knowledge_file_path(tmp_path, slug)
        assert path.is_file()

    @pytest.mark.asyncio
    async def test_no_llm_client_does_not_create_knowledge_file(
        self, tmp_path: Path
    ) -> None:
        """Without an LLM client, the audit still writes but no knowledge file is created."""
        from jules_daemon.wiki.test_knowledge import (
            derive_test_slug,
            knowledge_file_path,
        )

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit(command="python3 agent_test.py")
        result = _make_success_result(command="python3 agent_test.py")

        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="python3 agent_test.py",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        slug = derive_test_slug("python3 agent_test.py")
        path = knowledge_file_path(tmp_path, slug)
        # No knowledge file because there is no existing knowledge AND
        # no LLM client to extract from.
        assert not path.exists()
        # Audit still written though
        assert len(list_audit_files(tmp_path)) == 1

    @pytest.mark.asyncio
    async def test_existing_knowledge_passes_context_to_summarizer(
        self, tmp_path: Path
    ) -> None:
        """Pre-existing knowledge is loaded and embedded in the LLM prompt."""
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            save_test_knowledge,
        )

        # Seed the wiki with prior knowledge.
        prior = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 agent_test.py",
            purpose="runs the agent end to end",
            output_format="iteration N: PASSED|FAILED",
            normal_behavior="all 100 iterations pass in <30s",
            common_failures=("timeout",),
            runs_observed=2,
        )
        save_test_knowledge(tmp_path, prior)

        captured_prompts: list[str] = []

        class _Resp:
            def __init__(self, content: str) -> None:
                self.choices = [
                    type(
                        "_C",
                        (),
                        {
                            "message": type(
                                "_M", (), {"content": content}
                            )()
                        },
                    )()
                ]

        class _Comps:
            def create(self, **kwargs: Any) -> _Resp:
                # Capture the user prompt to verify the context block is present
                for msg in kwargs.get("messages", []):
                    if msg.get("role") == "user":
                        captured_prompts.append(msg["content"])
                return _Resp(
                    '{"passed": 1, "failed": 0, "skipped": 0, "total": 1, '
                    '"key_failures": [], "narrative": "All passed.", '
                    '"purpose": "", "output_format": "", '
                    '"common_failures": [], "normal_behavior": ""}'
                )

        class _Chat:
            def __init__(self) -> None:
                self.completions = _Comps()

        class _Client:
            def __init__(self) -> None:
                self.chat = _Chat()

        from jules_daemon.llm.config import LLMConfig

        llm_config = LLMConfig(
            base_url="https://example.test/v1",
            api_key="test-key",
            default_model="openai:default:gpt-4o",
        )
        client = _Client()
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=client,  # type: ignore[arg-type]
            llm_config=llm_config,
        )
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit(command="python3 agent_test.py")
        result = _make_success_result(command="python3 agent_test.py")

        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="python3 agent_test.py",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        # At least one prompt (summarizer) should contain the prior
        # knowledge text. The extractor also includes it in a separate
        # call.
        joined = "\n".join(captured_prompts)
        assert "runs the agent end to end" in joined
        assert "iteration N: PASSED|FAILED" in joined

    @pytest.mark.asyncio
    async def test_knowledge_save_failure_does_not_break_audit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Wiki I/O failures during knowledge save must not break the audit."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit()
        result = _make_success_result()

        # Make save_test_knowledge raise to simulate disk failure.
        from jules_daemon.ipc import request_handler as rh_module

        def _boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("disk failure")

        monkeypatch.setattr(rh_module, "save_test_knowledge", _boom)

        # Must not raise
        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="pytest",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        # The audit file is still written.
        assert len(list_audit_files(tmp_path)) == 1

    @pytest.mark.asyncio
    async def test_existing_knowledge_runs_observed_increments(
        self, tmp_path: Path
    ) -> None:
        """Even without an LLM client, prior knowledge gets a run-count bump."""
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            load_test_knowledge,
            save_test_knowledge,
        )

        prior = TestKnowledge(
            test_slug="agent-test-py",
            command_pattern="python3 agent_test.py",
            purpose="seed",
            runs_observed=2,
        )
        save_test_knowledge(tmp_path, prior)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = self._make_partial_audit(command="python3 agent_test.py")
        result = _make_success_result(command="python3 agent_test.py")

        await handler._finalize_and_write_audit(
            audit=audit,
            result=result,
            command="python3 agent_test.py",
            target_host="staging.example.com",
            target_user="deploy",
            target_port=22,
        )

        loaded = load_test_knowledge(tmp_path, "agent-test-py")
        assert loaded is not None
        assert loaded.runs_observed == 3
        assert loaded.purpose == "seed"


class TestQueuedAutoStartAudit:
    """Tests for :meth:`RequestHandler._build_queued_audit`."""

    def test_builds_audit_with_queue_source(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = handler._build_queued_audit(
            run_id="run-queue-001",
            command="pytest -v",
        )
        assert audit is not None
        assert audit.nl_input.source == "queue"
        assert audit.parsed_command is not None
        assert audit.parsed_command.resolved_shell == "pytest -v"
        assert audit.confirmation is not None
        assert audit.confirmation.decision == ConfirmationDecision.APPROVED
        assert audit.confirmation.decided_by == "queue-autostart"

    def test_empty_command_returns_none(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        audit = handler._build_queued_audit(
            run_id="run-queue-002",
            command="   ",
        )
        assert audit is None
