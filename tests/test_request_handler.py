"""Tests for the IPC request handler.

Validates that the RequestHandler (concrete ClientHandler implementation)
correctly bridges the socket server to the validation + dispatch pipeline:
- Validates incoming requests via the validation layer
- Returns validation error envelopes for invalid requests
- Dispatches valid requests to registered handlers
- Returns response envelopes with handler results
- Returns enqueue confirmation for queue verb
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.cli.verbs import Verb
from jules_daemon.ipc.framing import (
    HEADER_SIZE,
    MessageEnvelope,
    MessageType,
    decode_envelope,
    encode_frame,
    unpack_header,
)
from jules_daemon.ipc.notification_broadcaster import NotificationBroadcaster
from jules_daemon.ipc.request_handler import (
    RequestHandler,
    RequestHandlerConfig,
)
from jules_daemon.protocol.notifications import (
    NotificationEventType,
    NotificationSeverity,
)
from jules_daemon.ipc.server import ClientConnection
from jules_daemon.llm.intent_classifier import (
    ClassifiedIntent,
    IntentConfidence,
)
from jules_daemon.execution.run_pipeline import RunResult
from jules_daemon.workflows.runner import (
    WorkflowExecutionPlan,
    WorkflowExecutionStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> ClientConnection:
    """Build a stub ClientConnection for testing."""
    return ClientConnection(
        client_id="test-client-001",
        reader=AsyncMock(spec=asyncio.StreamReader),
        writer=AsyncMock(spec=asyncio.StreamWriter),
        connected_at="2026-04-09T12:00:00Z",
    )


def _make_request(
    payload: dict[str, Any],
    msg_id: str = "req-001",
) -> MessageEnvelope:
    """Build a REQUEST-type envelope."""
    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload=payload,
    )


def _make_non_request(
    payload: dict[str, Any],
    msg_type: MessageType = MessageType.RESPONSE,
    msg_id: str = "req-001",
) -> MessageEnvelope:
    """Build a non-REQUEST envelope."""
    return MessageEnvelope(
        msg_type=msg_type,
        msg_id=msg_id,
        timestamp="2026-04-09T12:00:00Z",
        payload=payload,
    )


def _make_llm_handler(tmp_path: Path) -> RequestHandler:
    """Build a RequestHandler with mocked LLM config enabled."""
    llm_config = MagicMock()
    llm_config.default_model = "provider:connection:model-v1"
    return RequestHandler(
        config=RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=MagicMock(),
            llm_config=llm_config,
        ),
    )


# ---------------------------------------------------------------------------
# RequestHandlerConfig tests
# ---------------------------------------------------------------------------


class TestRequestHandlerConfig:
    """Tests for the immutable handler configuration."""

    def test_create_with_wiki_root(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        assert config.wiki_root == tmp_path

    def test_frozen(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        with pytest.raises(AttributeError):
            config.wiki_root = Path("/other")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RequestHandler: protocol conformance
# ---------------------------------------------------------------------------


class TestRequestHandlerProtocol:
    """Tests that RequestHandler implements the ClientHandler protocol."""

    def test_has_handle_message_method(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        assert hasattr(handler, "handle_message")
        assert callable(handler.handle_message)

    def test_implements_client_handler(self, tmp_path: Path) -> None:
        from jules_daemon.ipc.server import ClientHandler

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        assert isinstance(handler, ClientHandler)


# ---------------------------------------------------------------------------
# RequestHandler: invalid message type
# ---------------------------------------------------------------------------


class TestRequestHandlerInvalidType:
    """Tests for non-REQUEST message type handling."""

    @pytest.mark.asyncio
    async def test_non_request_returns_error(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_non_request(payload={"verb": "status"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert response.msg_id == envelope.msg_id
        assert "error" in response.payload
        assert "validation_errors" in response.payload

    @pytest.mark.asyncio
    async def test_stream_type_returns_error(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_non_request(
            payload={"verb": "status"},
            msg_type=MessageType.STREAM,
        )

        response = await handler.handle_message(envelope, client)
        assert response.msg_type == MessageType.ERROR


# ---------------------------------------------------------------------------
# RequestHandler: missing / invalid verb
# ---------------------------------------------------------------------------


class TestRequestHandlerVerbErrors:
    """Tests for verb-level validation errors."""

    @pytest.mark.asyncio
    async def test_missing_verb_returns_error(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "validation_errors" in response.payload
        errors = response.payload["validation_errors"]
        assert any(e["field"] == "verb" for e in errors)

    @pytest.mark.asyncio
    async def test_unknown_verb_returns_error(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "teleport"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        errors = response.payload["validation_errors"]
        assert any(e["code"] == "unknown_verb" for e in errors)


# ---------------------------------------------------------------------------
# RequestHandler: field-level validation errors
# ---------------------------------------------------------------------------


class TestRequestHandlerFieldErrors:
    """Tests for verb-specific field validation errors."""

    @pytest.mark.asyncio
    async def test_run_missing_fields_returns_all_errors(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "run"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        errors = response.payload["validation_errors"]
        fields = {e["field"] for e in errors}
        assert "target_host" in fields
        assert "target_user" in fields
        assert "natural_language" in fields

    @pytest.mark.asyncio
    async def test_queue_missing_fields_returns_errors(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "queue"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        errors = response.payload["validation_errors"]
        fields = {e["field"] for e in errors}
        assert "target_host" in fields


# ---------------------------------------------------------------------------
# RequestHandler: valid status request
# ---------------------------------------------------------------------------


class TestRequestHandlerStatusVerb:
    """Tests for valid status verb handling."""

    @pytest.mark.asyncio
    async def test_status_returns_response(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "status"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.msg_id == envelope.msg_id
        assert "verb" in response.payload
        assert response.payload["verb"] == "status"

    @pytest.mark.asyncio
    async def test_status_verbose(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "status",
            "verbose": True,
        })

        response = await handler.handle_message(envelope, client)
        assert response.msg_type == MessageType.RESPONSE

    @pytest.mark.asyncio
    async def test_status_active_includes_test_context_and_parsed_output(
        self, tmp_path: Path
    ) -> None:
        from jules_daemon.wiki import current_run as cr_io
        from jules_daemon.wiki.layout import initialize_wiki
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            ProcessIDs,
            RunStatus,
            SSHTarget,
        )
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            derive_test_slug,
            save_test_knowledge,
        )

        initialize_wiki(tmp_path)
        command = "pytest tests/test_status.py -v"
        slug = derive_test_slug(command)
        save_test_knowledge(
            tmp_path,
            TestKnowledge(
                test_slug=slug,
                command_pattern=command,
                purpose="Status smoke coverage",
                output_format="pytest verbose lines",
                summary_fields=("passed", "failed", "incomplete"),
                normal_behavior="Tests emit PASSED/FAILED markers",
                required_args=("env",),
                workflow_steps=("precheck", "status_smoke"),
                prerequisites=("precheck",),
                artifact_requirements=("env_file",),
                when_missing_artifact_ask="Environment file is missing. Run precheck first?",
                success_criteria="Pytest summary shows no failures.",
                failure_criteria="Any pytest failure or incomplete test.",
                runs_observed=3,
            ),
        )

        run = CurrentRun(
            status=RunStatus.RUNNING,
            run_id="run-status-active",
            ssh_target=SSHTarget(host="host.example.com", user="deploy"),
            command=Command(
                natural_language="run status smoke coverage",
                resolved_shell=command,
                approved=True,
            ),
            pids=ProcessIDs(daemon=1234),
        )
        cr_io.write(tmp_path, run)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        handler._current_run_id = run.run_id
        handler._output_buffer = [
            "tests/test_status.py::test_one PASSED\n",
            "tests/test_status.py::test_two FAILED\n",
            "tests/test_status.py::test_three",
        ]

        async def _long_running() -> None:
            await asyncio.sleep(3600)

        handler._current_task = asyncio.create_task(_long_running())
        envelope = _make_request(payload={"verb": "status"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["state"] == "active"
        assert response.payload["test_context"]["purpose"] == (
            "Status smoke coverage"
        )
        assert response.payload["test_context"]["summary_fields"] == [
            "passed",
            "failed",
            "incomplete",
        ]
        assert response.payload["test_context"]["workflow_steps"] == [
            "precheck",
            "status_smoke",
        ]
        assert response.payload["test_context"]["prerequisites"] == [
            "precheck",
        ]
        assert response.payload["test_context"]["artifact_requirements"] == [
            "env_file",
        ]
        assert response.payload["test_context"]["when_missing_artifact_ask"] == (
            "Environment file is missing. Run precheck first?"
        )
        assert response.payload["test_context"]["success_criteria"] == (
            "Pytest summary shows no failures."
        )
        assert response.payload["test_context"]["failure_criteria"] == (
            "Any pytest failure or incomplete test."
        )
        assert response.payload["parsed_output"]["framework"] == "pytest"
        assert response.payload["parsed_output"]["summary"]["passed"] == 1
        assert response.payload["parsed_output"]["summary"]["failed"] == 1
        assert response.payload["parsed_output"]["summary"]["incomplete"] == 1
        assert response.payload["parsed_output"]["focused_summary"] == {
            "passed": 1,
            "failed": 1,
            "incomplete": 1,
        }
        assert (
            "pytest so far: 1 passed, 1 failed, 1 incomplete"
            in response.payload["status_summary"]
        )

        handler._current_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler._current_task

    @pytest.mark.asyncio
    async def test_status_completed_includes_test_context_and_parsed_output(
        self, tmp_path: Path
    ) -> None:
        from jules_daemon.execution.run_pipeline import RunResult
        from jules_daemon.wiki.layout import initialize_wiki
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            derive_test_slug,
            save_test_knowledge,
        )

        initialize_wiki(tmp_path)
        command = "pytest tests/test_status.py -v"
        slug = derive_test_slug(command)
        save_test_knowledge(
            tmp_path,
            TestKnowledge(
                test_slug=slug,
                command_pattern=command,
                purpose="Completed status smoke coverage",
                output_format="pytest verbose lines",
                summary_fields=("passed", "failed"),
                runs_observed=7,
            ),
        )

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        handler._last_completed_run = RunResult(
            success=False,
            run_id="run-status-complete",
            command=command,
            target_host="host.example.com",
            target_user="deploy",
            exit_code=1,
            stdout=(
                "tests/test_status.py::test_one PASSED\n"
                "tests/test_status.py::test_two FAILED\n"
            ),
            stderr="FAILED tests/test_status.py::test_two - AssertionError\n",
            error="Command exited with code 1",
            duration_seconds=2.5,
        )

        envelope = _make_request(payload={"verb": "status"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["state"] == "completed"
        assert response.payload["test_context"]["purpose"] == (
            "Completed status smoke coverage"
        )
        assert response.payload["test_context"]["summary_fields"] == [
            "passed",
            "failed",
        ]
        assert response.payload["parsed_output"]["framework"] == "pytest"
        assert response.payload["parsed_output"]["summary"]["passed"] == 1
        assert response.payload["parsed_output"]["summary"]["failed"] == 1
        assert response.payload["parsed_output"]["focused_summary"] == {
            "passed": 1,
            "failed": 1,
        }
        assert "pytest summary: 1 passed, 1 failed" in (
            response.payload["status_summary"]
        )

    @pytest.mark.asyncio
    async def test_status_summary_prefers_spec_summary_field_order(
        self, tmp_path: Path
    ) -> None:
        from jules_daemon.execution.run_pipeline import RunResult
        from jules_daemon.wiki.layout import initialize_wiki
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            derive_test_slug,
            save_test_knowledge,
        )

        initialize_wiki(tmp_path)
        command = "pytest tests/test_status.py -v"
        slug = derive_test_slug(command)
        save_test_knowledge(
            tmp_path,
            TestKnowledge(
                test_slug=slug,
                command_pattern=command,
                summary_fields=("failed", "passed", "iterations_done"),
                runs_observed=2,
            ),
        )

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        handler._last_completed_run = RunResult(
            success=False,
            run_id="run-status-ordered",
            command=command,
            target_host="host.example.com",
            target_user="deploy",
            exit_code=1,
            stdout=(
                "tests/test_status.py::test_one PASSED\n"
                "tests/test_status.py::test_two FAILED\n"
            ),
            stderr="",
            error="Command exited with code 1",
            duration_seconds=1.5,
        )

        envelope = _make_request(payload={"verb": "status"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["state"] == "completed"
        assert response.payload["parsed_output"]["focused_summary"] == {
            "failed": 1,
            "passed": 1,
        }
        assert response.payload["parsed_output"]["unmapped_summary_fields"] == [
            "iterations_done",
        ]
        assert "pytest summary: 1 failed, 1 passed" in (
            response.payload["status_summary"]
        )

    @pytest.mark.asyncio
    async def test_status_active_includes_recent_output_tail(
        self, tmp_path: Path
    ) -> None:
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            ProcessIDs,
            RunStatus,
            SSHTarget,
        )
        from jules_daemon.wiki import current_run as cr_io

        run = CurrentRun(
            status=RunStatus.RUNNING,
            run_id="run-live-1",
            ssh_target=SSHTarget(host="host.example.com", user="deploy"),
            command=Command(
                natural_language="run tests",
                resolved_shell="pytest -q",
                approved=True,
            ),
            pids=ProcessIDs(daemon=1234),
        )
        cr_io.write(tmp_path, run)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        handler._current_run_id = "run-live-1"
        handler._output_buffer = [
            "tests/test_a.py::test_one PASSED\n",
            "tests/test_b.py::test_two FAILED\n",
        ]

        async def _long_running() -> None:
            await asyncio.sleep(3600)

        handler._current_task = asyncio.create_task(_long_running())

        envelope = _make_request(payload={"verb": "status"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["state"] == "active"
        assert response.payload["last_output_line"] == (
            "tests/test_b.py::test_two FAILED"
        )
        assert response.payload["recent_output_lines"] == [
            "tests/test_a.py::test_one PASSED\n",
            "tests/test_b.py::test_two FAILED\n",
        ]

        handler._current_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler._current_task

    @pytest.mark.asyncio
    async def test_status_active_prefers_broadcaster_buffer(
        self, tmp_path: Path,
    ) -> None:
        from jules_daemon.wiki import current_run as cr_io
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            ProcessIDs,
            RunStatus,
            SSHTarget,
        )

        run = CurrentRun(
            status=RunStatus.RUNNING,
            run_id="run-live-broadcaster",
            ssh_target=SSHTarget(host="host.example.com", user="deploy"),
            command=Command(
                natural_language="run tests",
                resolved_shell="pytest -q",
                approved=True,
            ),
            pids=ProcessIDs(daemon=1234),
        )
        cr_io.write(tmp_path, run)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        handler._current_run_id = "run-live-broadcaster"
        handler._output_buffer = ["stale buffered line\n"]
        handler._job_output_broadcaster.register_job("run-live-broadcaster")
        handler._job_output_broadcaster.publish(
            "run-live-broadcaster",
            "tests/test_a.py::test_one PASSED\n",
        )
        handler._job_output_broadcaster.publish(
            "run-live-broadcaster",
            "tests/test_b.py::test_two FAILED\n",
        )

        async def _long_running() -> None:
            await asyncio.sleep(3600)

        handler._current_task = asyncio.create_task(_long_running())

        envelope = _make_request(payload={"verb": "status"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["state"] == "active"
        assert response.payload["last_output_line"] == (
            "tests/test_b.py::test_two FAILED"
        )
        assert response.payload["recent_output_lines"] == [
            "tests/test_a.py::test_one PASSED\n",
            "tests/test_b.py::test_two FAILED\n",
        ]

        handler._current_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler._current_task

    @pytest.mark.asyncio
    async def test_status_active_includes_monitor_alert_summary(
        self, tmp_path: Path,
    ) -> None:
        from jules_daemon.wiki import current_run as cr_io
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            ProcessIDs,
            RunStatus,
            SSHTarget,
        )

        run = CurrentRun(
            status=RunStatus.RUNNING,
            run_id="run-live-alerts",
            ssh_target=SSHTarget(host="host.example.com", user="deploy"),
            command=Command(
                natural_language="run tests",
                resolved_shell="pytest -q",
                approved=True,
            ),
            pids=ProcessIDs(daemon=1234),
        )
        cr_io.write(tmp_path, run)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        handler._current_run_id = "run-live-alerts"

        async def _long_running() -> None:
            await asyncio.sleep(3600)

        handler._current_task = asyncio.create_task(_long_running())
        await handler._process_monitor_output_line(
            run_id="run-live-alerts",
            line="SIGSEGV at 0xdeadbeef\n",
        )

        envelope = _make_request(payload={"verb": "status"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["state"] == "active"
        assert response.payload["alert_summary"]["total_alerts"] == 1
        assert response.payload["alert_summary"]["highest_priority_alerts"][
            0
        ]["pattern_name"] == "segfault"

        handler._current_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler._current_task


# ---------------------------------------------------------------------------
# RequestHandler: workflow integration
# ---------------------------------------------------------------------------


class TestRequestHandlerWorkflowIntegration:
    """Tests for workflow persistence wired into run/status flows."""

    @pytest.mark.asyncio
    async def test_spawn_background_run_creates_workflow_records(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from jules_daemon.wiki.layout import initialize_wiki
        from jules_daemon.workflows.store import read_step, read_workflow

        initialize_wiki(tmp_path)
        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))

        async def _fake_background_execute(**_: Any) -> None:
            await asyncio.sleep(3600)

        monkeypatch.setattr(handler, "_background_execute", _fake_background_execute)

        started = handler._spawn_background_run(
            target_host="host.example.com",
            target_user="deploy",
            command="pytest -q",
            workflow_request="run smoke tests",
        )

        workflow = read_workflow(tmp_path, started["workflow_id"])
        step = read_step(tmp_path, started["workflow_id"], "primary-run")

        assert started["workflow_id"] == started["run_id"]
        assert workflow is not None
        assert workflow.request_text == "run smoke tests"
        assert workflow.status.value == "running"
        assert step is not None
        assert step.command == "pytest -q"
        assert step.status.value == "running"

        handler._current_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler._current_task

    @pytest.mark.asyncio
    async def test_spawn_background_run_persists_artifact_states(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from jules_daemon.wiki.layout import initialize_wiki
        from jules_daemon.workflows.models import ArtifactState, ArtifactStatus
        from jules_daemon.workflows.store import read_workflow

        initialize_wiki(tmp_path)
        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))

        async def _fake_background_execute(**_: Any) -> None:
            await asyncio.sleep(3600)

        monkeypatch.setattr(handler, "_background_execute", _fake_background_execute)

        started = handler._spawn_background_run(
            target_host="host.example.com",
            target_user="deploy",
            command="pytest -q",
            workflow_request="run main check",
            artifact_states=(
                ArtifactState(
                    name="/tmp/setup-ready.flag",
                    status=ArtifactStatus.MISSING,
                    details="Verified remote path is missing.",
                ),
            ),
        )

        workflow = read_workflow(tmp_path, started["workflow_id"])

        assert workflow is not None
        assert len(workflow.artifact_states) == 1
        assert workflow.artifact_states[0].name == "/tmp/setup-ready.flag"
        assert workflow.artifact_states[0].status.value == "missing"

        handler._current_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler._current_task

    @pytest.mark.asyncio
    async def test_status_idle_includes_latest_workflow_snapshot(
        self,
        tmp_path: Path,
    ) -> None:
        from jules_daemon.wiki.layout import initialize_wiki
        from jules_daemon.workflows.models import WorkflowRecord, WorkflowStepRecord
        from jules_daemon.workflows.store import save_step, save_workflow

        initialize_wiki(tmp_path)
        save_workflow(
            tmp_path,
            WorkflowRecord(
                workflow_id="run-finished-1",
                request_text="run main check",
            ).with_completed_success(
                summary="Run completed successfully.",
                current_step_id="primary-run",
            ),
        )
        save_step(
            tmp_path,
            WorkflowStepRecord(
                workflow_id="run-finished-1",
                step_id="primary-run",
                name="primary run",
            ).with_completed_success(
                summary="Run completed successfully.",
                exit_code=0,
            ),
        )

        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        client = _make_client()
        envelope = _make_request(payload={"verb": "status"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["state"] == "idle"
        assert response.payload["workflow"]["workflow_id"] == "run-finished-1"
        assert response.payload["workflow"]["status"] == "completed_success"
        assert response.payload["workflow"]["active_step"]["status"] == (
            "completed_success"
        )


# ---------------------------------------------------------------------------
# RequestHandler: valid queue request with enqueue confirmation
# ---------------------------------------------------------------------------


class TestRequestHandlerQueueVerb:
    """Tests for queue verb with enqueue confirmation."""

    @pytest.mark.asyncio
    async def test_queue_returns_enqueue_confirmation(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "queue",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run the smoke tests",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.msg_id == envelope.msg_id
        assert response.payload["verb"] == "queue"
        assert response.payload["status"] == "enqueued"
        assert "queue_id" in response.payload
        assert "position" in response.payload


class TestRequestHandlerWorkflowPreflight:
    """Workflow-aware runs should ask preflight questions before agent routing."""

    @pytest.mark.asyncio
    async def test_run_with_workflow_preflight_passes_context_to_agent_loop(
        self,
        tmp_path: Path,
    ) -> None:
        from jules_daemon.wiki.test_knowledge import TestKnowledge, save_test_knowledge

        handler = _make_llm_handler(tmp_path)
        client = _make_client()
        ask_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="ask-workflow-001",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True, "answer": "no"},
        )
        ask_frame = encode_frame(ask_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[ask_frame[:4], ask_frame[4:]]
        )
        save_test_knowledge(
            tmp_path,
            TestKnowledge(
                test_slug="main-check",
                command_pattern="python3 /root/main_check.py --target {target}",
                workflow_steps=("setup-step", "main-check"),
                prerequisites=("setup-step",),
                artifact_requirements=("setup_ready_file",),
                success_criteria="Main check summary reports zero failures.",
                failure_criteria="Setup step fails or main check reports any failure.",
            ),
        )

        expected = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="req-001",
            timestamp="2026-04-09T12:00:02Z",
            payload={"status": "started", "mode": "agent_loop"},
        )
        handler._handle_run_agent_loop = AsyncMock(return_value=expected)  # type: ignore[method-assign]

        response = await handler.handle_message(
            _make_request(payload={
                "verb": "run",
                "target_host": "10.0.0.10",
                "target_user": "root",
                "natural_language": "run main check",
            }),
            client,
        )

        assert response == expected
        prompt_bytes = b"".join(
            call.args[0] for call in client.writer.write.call_args_list
        )
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(
            prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length]
        )
        assert prompt.payload["type"] == "question"
        assert "could not verify required artifacts automatically" in (
            prompt.payload["question"].lower()
        )

        run_args = handler._handle_run_agent_loop.await_args.args[1]
        workflow_context = run_args["workflow_context"]
        assert workflow_context["matched_test_slug"] == "main-check"
        assert workflow_context["workflow_steps"] == ["setup-step", "main-check"]
        assert workflow_context["preflight"]["user_decision"] == (
            "declined_prerequisites"
        )
        assert workflow_context["preflight"]["should_run_prerequisites"] is False


class TestRequestHandlerWorkflowExecution:
    """Sequential workflow execution should persist step-by-step state."""

    @staticmethod
    def _queue_reply_frames(*replies: MessageEnvelope) -> list[bytes]:
        chunks: list[bytes] = []
        for reply in replies:
            frame = encode_frame(reply)
            chunks.extend([frame[:4], frame[4:]])
        return chunks

    @pytest.mark.asyncio
    async def test_run_executes_prerequisite_then_main_workflow(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from jules_daemon.wiki.test_knowledge import (
            TestKnowledge,
            save_test_knowledge,
        )
        from jules_daemon.workflows.status import build_workflow_status

        handler = _make_llm_handler(tmp_path)
        client = _make_client()

        client.reader.readexactly = AsyncMock(
            side_effect=self._queue_reply_frames(
                MessageEnvelope(
                    msg_type=MessageType.CONFIRM_REPLY,
                    msg_id="preflight-001",
                    timestamp="2026-04-16T12:00:01Z",
                    payload={"approved": True, "answer": "yes"},
                ),
                MessageEnvelope(
                    msg_type=MessageType.CONFIRM_REPLY,
                    msg_id="step-approve-001",
                    timestamp="2026-04-16T12:00:02Z",
                    payload={"approved": True},
                ),
                MessageEnvelope(
                    msg_type=MessageType.CONFIRM_REPLY,
                    msg_id="arg-001",
                    timestamp="2026-04-16T12:00:03Z",
                    payload={"approved": True, "answer": "5"},
                ),
                MessageEnvelope(
                    msg_type=MessageType.CONFIRM_REPLY,
                    msg_id="step-approve-002",
                    timestamp="2026-04-16T12:00:04Z",
                    payload={"approved": True},
                ),
            )
        )

        save_test_knowledge(
            tmp_path,
            TestKnowledge(
                test_slug="main-check",
                command_pattern="python3 /root/main_check.py --target {target}",
                required_args=("target",),
                workflow_steps=("setup-step", "main-check"),
                prerequisites=("setup-step",),
                artifact_requirements=("setup_ready_file",),
                success_criteria="Main check summary reports zero failures.",
                failure_criteria="Setup step fails or main check reports any failure.",
            ),
        )
        save_test_knowledge(
            tmp_path,
            TestKnowledge(
                test_slug="setup-step",
                command_pattern="python3 /root/setup_step.py",
            ),
        )

        executed_commands: list[str] = []

        async def _fake_execute_run_once(
            *,
            run_id: str,
            stream_run_id: str,
            target_host: str,
            target_user: str,
            command: str,
            target_port: int,
            timeout: int = 3600,
        ) -> RunResult:
            del stream_run_id, target_port, timeout
            executed_commands.append(command)
            now = datetime.now(timezone.utc)
            if "setup_step.py" in command:
                stdout = "tests/test_setup.py::test_ready PASSED\n"
            else:
                stdout = "tests/test_main.py::test_target PASSED\n"
            return RunResult(
                success=True,
                run_id=run_id,
                command=command,
                target_host=target_host,
                target_user=target_user,
                exit_code=0,
                stdout=stdout,
                started_at=now,
                completed_at=now,
            )

        monkeypatch.setattr(handler, "_execute_run_once", _fake_execute_run_once)

        response = await handler.handle_message(
            _make_request(payload={
                "verb": "run",
                "target_host": "10.0.0.10",
                "target_user": "root",
                "natural_language": "run main check",
            }),
            client,
        )

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "started"
        assert response.payload["mode"] == "workflow"
        assert response.payload["workflow_id"].startswith("run-")
        assert [step["name"] for step in response.payload["workflow_steps"]] == [
            "setup-step",
            "main-check",
        ]

        assert handler._current_task is not None
        await handler._current_task

        assert executed_commands == [
            "python3 /root/setup_step.py",
            "python3 /root/main_check.py --target 5",
        ]

        workflow_snapshot = build_workflow_status(
            tmp_path,
            response.payload["workflow_id"],
        )
        assert workflow_snapshot is not None
        assert workflow_snapshot["status"] == "completed_success"
        assert workflow_snapshot["step_count"] == 2
        assert [step["status"] for step in workflow_snapshot["steps"]] == [
            "completed_success",
            "completed_success",
        ]
        assert workflow_snapshot["steps"][0]["parsed_status"]["state"] == (
            "completed_success"
        )
        assert workflow_snapshot["steps"][0]["parsed_status"]["summary_fields"][
            "passed"
        ] == 1
        assert workflow_snapshot["steps"][1]["parsed_status"]["state"] == (
            "completed_success"
        )

        status_response = await handler.handle_message(
            _make_request(payload={"verb": "status"}, msg_id="status-001"),
            client,
        )
        assert status_response.payload["workflow"]["workflow_id"] == (
            response.payload["workflow_id"]
        )
        assert status_response.payload["workflow"]["step_count"] == 2
        assert status_response.payload["workflow"]["steps"][1]["parsed_status"][
            "summary_fields"
        ]["passed"] == 1

    @pytest.mark.asyncio
    async def test_status_active_workflow_includes_live_parsed_step_status(
        self,
        tmp_path: Path,
    ) -> None:
        from jules_daemon.wiki import current_run as current_run_io
        from jules_daemon.wiki.layout import initialize_wiki
        from jules_daemon.wiki.models import Command, CurrentRun, RunStatus, SSHTarget
        from jules_daemon.workflows.models import WorkflowRecord, WorkflowStepRecord
        from jules_daemon.workflows.store import save_step, save_workflow

        initialize_wiki(tmp_path)
        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        workflow_id = "run-live-workflow-001"
        now = datetime.now(timezone.utc)

        save_workflow(
            tmp_path,
            WorkflowRecord(
                workflow_id=workflow_id,
                request_text="run main check",
            ).with_running(
                current_step_id="step-01-main-check",
                run_id=workflow_id,
                target_host="10.0.0.10",
                target_user="root",
            ),
        )
        save_step(
            tmp_path,
            WorkflowStepRecord(
                workflow_id=workflow_id,
                step_id="step-01-main-check",
                name="main-check",
                kind="main",
            ).with_running(
                run_id="run-step-001",
                command="pytest -q",
                target_host="10.0.0.10",
                target_user="root",
            ),
        )
        current_run_io.write(
            tmp_path,
            CurrentRun(
                status=RunStatus.RUNNING,
                run_id=workflow_id,
                ssh_target=SSHTarget(host="10.0.0.10", user="root"),
                command=Command(
                    natural_language="run main check",
                    resolved_shell="pytest -q",
                    approved=True,
                    approved_at=now,
                ),
                started_at=now,
            ),
        )

        handler._current_run_id = workflow_id
        handler._current_workflow_id = workflow_id
        handler._prepare_live_run_state(run_id=workflow_id)
        handler._publish_live_output_line(
            workflow_id,
            "tests/test_live.py::test_ok PASSED\n",
        )
        handler._current_task = asyncio.create_task(asyncio.sleep(3600))

        response = await handler.handle_message(
            _make_request(payload={"verb": "status"}, msg_id="status-live-001"),
            _make_client(),
        )

        assert response.payload["state"] == "active"
        assert response.payload["workflow"]["workflow_id"] == workflow_id
        assert response.payload["workflow"]["active_step"]["parsed_status"][
            "state"
        ] == "running"
        assert response.payload["workflow"]["active_step"]["parsed_status"][
            "summary_fields"
        ]["passed"] == 1
        assert "test output so far" in response.payload["workflow"][
            "active_step_summary"
        ]

        handler._job_output_broadcaster.unregister_job(workflow_id)
        handler._current_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler._current_task

    @pytest.mark.asyncio
    async def test_workflow_execution_emits_step_alerts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        broadcaster = NotificationBroadcaster()
        subscription = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.ALERT}),
        )
        handler = RequestHandler(
            config=RequestHandlerConfig(
                wiki_root=tmp_path,
                notification_broadcaster=broadcaster,
            )
        )
        plan = WorkflowExecutionPlan(
            workflow_id="run-workflow-alerts-001",
            request_text="run main check",
            target_host="10.0.0.10",
            target_user="root",
            target_port=22,
            steps=(
                WorkflowExecutionStep(
                    step_id="step-01-setup-step",
                    step_name="setup-step",
                    phase="prerequisite",
                    test_slug="setup-step",
                    command="python3 /root/setup_step.py",
                    command_pattern="python3 /root/setup_step.py",
                    required_args=(),
                ),
                WorkflowExecutionStep(
                    step_id="step-02-main-check",
                    step_name="main-check",
                    phase="main",
                    test_slug="main-check",
                    command="python3 /root/main_check.py --target 5",
                    command_pattern="python3 /root/main_check.py --target {target}",
                    required_args=("target",),
                ),
            ),
        )

        async def _fake_execute_run_once(
            *,
            run_id: str,
            stream_run_id: str,
            target_host: str,
            target_user: str,
            command: str,
            target_port: int,
            timeout: int = 3600,
        ) -> RunResult:
            del stream_run_id, target_port, timeout
            now = datetime.now(timezone.utc)
            stdout = (
                "tests/test_setup.py::test_ready PASSED\n"
                if "setup_step.py" in command
                else "tests/test_main.py::test_target PASSED\n"
            )
            return RunResult(
                success=True,
                run_id=run_id,
                command=command,
                target_host=target_host,
                target_user=target_user,
                exit_code=0,
                stdout=stdout,
                started_at=now,
                completed_at=now,
            )

        monkeypatch.setattr(handler, "_execute_run_once", _fake_execute_run_once)

        started = handler._spawn_background_workflow(plan=plan)
        assert started["workflow_id"] == "run-workflow-alerts-001"
        assert handler._current_task is not None
        await handler._current_task

        envelopes = [
            await broadcaster.receive(subscription.subscription_id, timeout=1.0)
            for _ in range(4)
        ]

        assert [envelope.payload.title for envelope in envelopes if envelope] == [
            "Workflow step started: setup-step",
            "Workflow step completed: setup-step",
            "Workflow step started: main-check",
            "Workflow step completed: main-check",
        ]
        assert [envelope.payload.severity for envelope in envelopes if envelope] == [
            NotificationSeverity.INFO,
            NotificationSeverity.SUCCESS,
            NotificationSeverity.INFO,
            NotificationSeverity.SUCCESS,
        ]

    @pytest.mark.asyncio
    async def test_cancel_marks_active_workflow_by_workflow_id(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from jules_daemon.wiki.layout import initialize_wiki
        from jules_daemon.workflows.store import read_workflow

        initialize_wiki(tmp_path)
        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        plan = WorkflowExecutionPlan(
            workflow_id="run-workflow-123",
            request_text="run main check",
            target_host="10.0.0.10",
            target_user="root",
            target_port=22,
            steps=(
                WorkflowExecutionStep(
                    step_id="step-01-setup-step",
                    step_name="setup-step",
                    phase="prerequisite",
                    test_slug="setup-step",
                    command="python3 /root/setup_step.py",
                    command_pattern="python3 /root/setup_step.py",
                    required_args=(),
                ),
                WorkflowExecutionStep(
                    step_id="step-02-main-check",
                    step_name="main-check",
                    phase="main",
                    test_slug="main-check",
                    command="python3 /root/main_check.py --target 5",
                    command_pattern="python3 /root/main_check.py --target {target}",
                    required_args=("target",),
                ),
            ),
        )

        async def _fake_background_execute_workflow(*, plan: WorkflowExecutionPlan) -> RunResult:
            del plan
            await asyncio.sleep(3600)
            raise AssertionError("workflow task should be cancelled before completion")

        monkeypatch.setattr(
            handler,
            "_background_execute_workflow",
            _fake_background_execute_workflow,
        )

        started = handler._spawn_background_workflow(plan=plan)
        task = handler._current_task
        assert task is not None
        assert started["workflow_id"] == "run-workflow-123"

        response = handler._handle_cancel("cancel-001", {})

        assert response.payload["status"] == "cancelled"
        assert response.payload["workflow_id"] == "run-workflow-123"

        workflow = read_workflow(tmp_path, "run-workflow-123")
        assert workflow is not None
        assert workflow.status.value == "cancelled"

        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestRequestHandlerQueuePersistence:
    """Queue-related persistence and ordering behavior."""

    @pytest.mark.asyncio
    async def test_queue_creates_wiki_file(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "queue",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run integration tests",
        })

        response = await handler.handle_message(envelope, client)

        # Verify the wiki queue directory was created
        queue_dir = tmp_path / "pages" / "daemon" / "queue"
        assert queue_dir.exists()
        queue_files = list(queue_dir.glob("*.md"))
        assert len(queue_files) == 1

    @pytest.mark.asyncio
    async def test_multiple_queues_increment_position(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        for i in range(3):
            envelope = _make_request(
                payload={
                    "verb": "queue",
                    "target_host": f"host-{i}.example.com",
                    "target_user": "deploy",
                    "natural_language": f"run test suite {i}",
                },
                msg_id=f"req-{i}",
            )
            response = await handler.handle_message(envelope, client)
            assert response.payload["status"] == "enqueued"
            assert response.payload["position"] == i + 1


# ---------------------------------------------------------------------------
# RequestHandler: valid cancel request
# ---------------------------------------------------------------------------


class TestRequestHandlerCancelVerb:
    """Tests for cancel verb handling."""

    @pytest.mark.asyncio
    async def test_cancel_no_task_returns_error(self, tmp_path: Path) -> None:
        """When no task is running, cancel returns an error."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "cancel"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "No test is currently running" in response.payload["error"]

    @pytest.mark.asyncio
    async def test_cancel_running_task_returns_success(
        self, tmp_path: Path
    ) -> None:
        """When a task is running, cancel stops it and returns success."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        # Simulate a running task
        async def _long_running() -> None:
            await asyncio.sleep(3600)

        handler._current_task = asyncio.create_task(_long_running())
        handler._current_run_id = "run-test-123"

        envelope = _make_request(payload={"verb": "cancel"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "cancel"
        assert response.payload["status"] == "cancelled"
        assert response.payload["run_id"] == "run-test-123"
        assert handler._current_task is None
        assert handler._current_run_id is None

    @pytest.mark.asyncio
    async def test_cancel_done_task_returns_error(
        self, tmp_path: Path
    ) -> None:
        """When the task is already done, cancel returns an error."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        # Create a task that finishes immediately
        async def _instant() -> None:
            pass

        task = asyncio.create_task(_instant())
        await task  # Let it complete
        handler._current_task = task
        handler._current_run_id = "run-done-456"

        envelope = _make_request(payload={"verb": "cancel"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "No test is currently running" in response.payload["error"]


# ---------------------------------------------------------------------------
# RequestHandler: valid history request
# ---------------------------------------------------------------------------


class TestRequestHandlerHistoryVerb:
    """Tests for history verb handling."""

    @pytest.mark.asyncio
    async def test_history_empty_returns_empty_list(
        self, tmp_path: Path,
    ) -> None:
        """When no history files exist, returns empty records."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "history"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "history"
        assert response.payload["records"] == []
        assert response.payload["total"] == 0

    @pytest.mark.asyncio
    async def test_history_reads_wiki_files(self, tmp_path: Path) -> None:
        """History command reads and parses history wiki files."""
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            ProcessIDs,
            RunStatus,
            SSHTarget,
        )
        from jules_daemon.wiki.run_promotion import promote_run
        from jules_daemon.wiki import current_run as cr_io

        # Create a completed run and promote it to history
        run = CurrentRun(
            status=RunStatus.RUNNING,
            run_id="abc123",
            ssh_target=SSHTarget(host="host1.example.com", user="deploy"),
            command=Command(
                natural_language="run tests",
                resolved_shell="pytest -v",
                approved=True,
            ),
            pids=ProcessIDs(daemon=1234),
        )
        completed = run.with_completed(
            final_progress=run.progress,
        )
        cr_io.write(tmp_path, completed)
        promote_run(tmp_path, completed)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "history"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "history"
        assert response.payload["total"] == 1
        records = response.payload["records"]
        assert len(records) == 1
        assert records[0]["run_id"] == "abc123"
        assert records[0]["host"] == "host1.example.com"
        assert records[0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_history_respects_limit(self, tmp_path: Path) -> None:
        """History command respects the limit parameter."""
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            ProcessIDs,
            RunStatus,
            SSHTarget,
        )
        from jules_daemon.wiki.run_promotion import promote_run
        from jules_daemon.wiki import current_run as cr_io

        # Create and promote 3 runs
        for i in range(3):
            run = CurrentRun(
                status=RunStatus.RUNNING,
                run_id=f"run-{i}",
                ssh_target=SSHTarget(host="host.example.com", user="deploy"),
                command=Command(
                    natural_language=f"test {i}",
                    resolved_shell=f"pytest test_{i}.py",
                    approved=True,
                ),
                pids=ProcessIDs(daemon=1234),
            )
            completed = run.with_completed(final_progress=run.progress)
            cr_io.write(tmp_path, completed)
            promote_run(tmp_path, completed)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "history",
            "limit": 2,
        })

        response = await handler.handle_message(envelope, client)

        assert response.payload["total"] == 3
        assert len(response.payload["records"]) == 2

    @pytest.mark.asyncio
    async def test_history_filters_by_status(self, tmp_path: Path) -> None:
        """History command filters by status_filter."""
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            ProcessIDs,
            Progress,
            RunStatus,
            SSHTarget,
        )
        from jules_daemon.wiki.run_promotion import promote_run
        from jules_daemon.wiki import current_run as cr_io

        # Create a completed run
        run1 = CurrentRun(
            status=RunStatus.RUNNING,
            run_id="completed-run",
            ssh_target=SSHTarget(host="host.example.com", user="deploy"),
            command=Command(
                natural_language="test ok",
                resolved_shell="pytest",
                approved=True,
            ),
            pids=ProcessIDs(daemon=1234),
        )
        completed = run1.with_completed(final_progress=run1.progress)
        cr_io.write(tmp_path, completed)
        promote_run(tmp_path, completed)

        # Create a failed run
        run2 = CurrentRun(
            status=RunStatus.RUNNING,
            run_id="failed-run",
            ssh_target=SSHTarget(host="host.example.com", user="deploy"),
            command=Command(
                natural_language="test fail",
                resolved_shell="pytest",
                approved=True,
            ),
            pids=ProcessIDs(daemon=1234),
        )
        failed = run2.with_failed("exit code 1", Progress())
        cr_io.write(tmp_path, failed)
        promote_run(tmp_path, failed)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        # Filter for completed only
        envelope = _make_request(payload={
            "verb": "history",
            "status_filter": "completed",
        })
        response = await handler.handle_message(envelope, client)

        assert response.payload["total"] == 1
        assert response.payload["records"][0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_history_filters_by_host(self, tmp_path: Path) -> None:
        """History command filters by host_filter."""
        from jules_daemon.wiki.models import (
            Command,
            CurrentRun,
            ProcessIDs,
            RunStatus,
            SSHTarget,
        )
        from jules_daemon.wiki.run_promotion import promote_run
        from jules_daemon.wiki import current_run as cr_io

        for host in ["alpha.example.com", "beta.example.com"]:
            run = CurrentRun(
                status=RunStatus.RUNNING,
                run_id=f"run-{host}",
                ssh_target=SSHTarget(host=host, user="deploy"),
                command=Command(
                    natural_language="test",
                    resolved_shell="pytest",
                    approved=True,
                ),
                pids=ProcessIDs(daemon=1234),
            )
            completed = run.with_completed(final_progress=run.progress)
            cr_io.write(tmp_path, completed)
            promote_run(tmp_path, completed)

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={
            "verb": "history",
            "host_filter": "alpha.example.com",
        })

        response = await handler.handle_message(envelope, client)

        assert response.payload["total"] == 1
        assert response.payload["records"][0]["host"] == "alpha.example.com"


# ---------------------------------------------------------------------------
# RequestHandler: valid watch request
# ---------------------------------------------------------------------------


class TestRequestHandlerWatchVerb:
    """Tests for watch verb handling."""

    @pytest.mark.asyncio
    async def test_watch_no_active_run_returns_no_active_run(
        self, tmp_path: Path,
    ) -> None:
        """When no task is running and no buffer exists, returns no_active_run."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "watch"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "watch"
        assert response.payload["status"] == "no_active_run"

    @pytest.mark.asyncio
    async def test_watch_streams_buffered_output(
        self, tmp_path: Path,
    ) -> None:
        """Watch sends buffered lines and completes when task is done."""
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        # Pre-fill the output buffer with some lines
        handler._output_buffer = ["line 1\n", "line 2\n"]

        # Create a task that is already done
        async def _instant() -> None:
            pass

        task = asyncio.create_task(_instant())
        await task
        handler._current_task = task
        handler._current_run_id = "run-watch-test"

        # Collect frames sent to the mock writer
        sent_frames: list[bytes] = []
        client = _make_client()
        original_write = client.writer.write

        def _capture_write(data: bytes) -> None:
            sent_frames.append(data)

        client.writer.write = _capture_write

        envelope = _make_request(payload={"verb": "watch"})
        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "watch"
        assert response.payload["status"] == "completed"
        # Should have sent at least the 2 buffered lines + end-of-stream
        assert len(sent_frames) >= 3

    @pytest.mark.asyncio
    async def test_watch_streams_active_broadcaster_output(
        self, tmp_path: Path,
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        handler._current_run_id = "run-watch-live"
        handler._job_output_broadcaster.register_job("run-watch-live")
        handler._job_output_broadcaster.publish(
            "run-watch-live",
            "line 1\n",
        )

        async def _long_running() -> None:
            await asyncio.sleep(3600)

        handler._current_task = asyncio.create_task(_long_running())

        sent_frames: list[bytes] = []
        client = _make_client()

        def _capture_write(data: bytes) -> None:
            sent_frames.append(data)

        client.writer.write = _capture_write

        async def _finish_stream() -> None:
            await asyncio.sleep(0.05)
            handler._job_output_broadcaster.publish(
                "run-watch-live",
                "line 2\n",
            )
            handler._job_output_broadcaster.unregister_job("run-watch-live")

        finish_task = asyncio.create_task(_finish_stream())

        envelope = _make_request(payload={"verb": "watch"})
        response = await handler.handle_message(envelope, client)

        stream_payloads = [
            decode_envelope(frame[HEADER_SIZE:]).payload for frame in sent_frames
        ]

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "watch"
        assert response.payload["status"] == "completed"
        assert [payload["line"] for payload in stream_payloads[:-1]] == [
            "line 1\n",
            "line 2\n",
        ]
        assert stream_payloads[-1]["is_end"] is True

        await finish_task
        handler._current_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await handler._current_task


class TestRequestHandlerMonitorAlerts:
    """Tests for monitor-driven alert collection and emission."""

    @pytest.mark.asyncio
    async def test_monitor_output_line_emits_alert_notification(
        self, tmp_path: Path,
    ) -> None:
        broadcaster = NotificationBroadcaster()
        subscription = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.ALERT}),
        )
        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            notification_broadcaster=broadcaster,
        )
        handler = RequestHandler(config=config)

        await handler._process_monitor_output_line(
            run_id="run-alert-1",
            line="SIGSEGV at 0xdeadbeef\n",
        )

        envelope = await broadcaster.receive(
            subscription.subscription_id,
            timeout=1.0,
        )

        assert envelope is not None
        assert envelope.event_type is NotificationEventType.ALERT
        assert envelope.payload.title == "Monitor alert: segfault"
        assert "SIGSEGV" in envelope.payload.message

    @pytest.mark.asyncio
    async def test_failure_rate_alert_is_deduped_per_pattern(
        self, tmp_path: Path,
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)

        for _ in range(12):
            await handler._process_monitor_output_line(
                run_id="run-alert-2",
                line="FAILED tests/test_example.py::test_case\n",
            )

        summary = handler._build_status_alert_enrichment(run_id="run-alert-2")

        assert summary["alert_summary"]["total_alerts"] == 1
        assert summary["alert_summary"]["highest_priority_alerts"][0][
            "pattern_name"
        ] == "failure_rate_spike"


# ---------------------------------------------------------------------------
# RequestHandler: valid run request
# ---------------------------------------------------------------------------


class TestRequestHandlerRunVerb:
    """Tests for run verb handling.

    The run handler now implements the full confirmation flow:
    CONFIRM_PROMPT -> CONFIRM_REPLY -> execute (or deny).
    Tests simulate the multi-message exchange via mocked reader/writer.
    """

    @pytest.mark.asyncio
    async def test_run_denied_returns_denied_status(
        self, tmp_path: Path
    ) -> None:
        """When the user denies the confirmation, status is 'denied'."""
        from jules_daemon.ipc.framing import encode_frame

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        # Build the deny reply that the reader will return
        deny_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="deny-001",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": False},
        )
        deny_frame = encode_frame(deny_reply)
        # readexactly: first call returns 4-byte header,
        # second call returns payload bytes
        header_bytes = deny_frame[:4]
        payload_bytes = deny_frame[4:]
        client.reader.readexactly = AsyncMock(
            side_effect=[header_bytes, payload_bytes]
        )

        envelope = _make_request(payload={
            "verb": "run",
            "target_host": "staging.example.com",
            "target_user": "deploy",
            "natural_language": "run the full regression suite",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "run"
        assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_run_with_system_name_resolves_from_wiki(
        self, tmp_path: Path
    ) -> None:
        from jules_daemon.ipc.framing import encode_frame

        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tuto.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tuto\n"
            "host: 10.0.0.10\n"
            "user: root\n"
            "---\n\n"
            "# Tuto\n",
            encoding="utf-8",
        )

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        deny_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="deny-002",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": False},
        )
        deny_frame = encode_frame(deny_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[deny_frame[:4], deny_frame[4:]]
        )

        envelope = _make_request(payload={
            "verb": "run",
            "system_name": "tuto",
            "natural_language": "run the full regression suite",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "denied"
        prompt_bytes = b"".join(call.args[0] for call in client.writer.write.call_args_list)
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length])
        assert prompt.msg_type == MessageType.CONFIRM_PROMPT
        assert prompt.payload["target_host"] == "10.0.0.10"
        assert prompt.payload["target_user"] == "root"
        assert prompt.payload["system_name"] == "tuto"
        assert prompt.payload["auth_mode"] == "key-based"
        assert "JULES_SSH_PASSWORD" in prompt.payload["credential_guidance"]

    @pytest.mark.asyncio
    async def test_run_with_inferred_system_resolves_from_wiki(
        self, tmp_path: Path
    ) -> None:
        from jules_daemon.ipc.framing import encode_frame

        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tuto.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tuto\n"
            "aliases:\n"
            "  - tutorial\n"
            "host: 10.0.0.10\n"
            "user: root\n"
            "---\n\n"
            "# Tuto\n",
            encoding="utf-8",
        )

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        deny_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="deny-003",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": False},
        )
        deny_frame = encode_frame(deny_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[deny_frame[:4], deny_frame[4:]]
        )

        envelope = _make_request(payload={
            "verb": "run",
            "infer_target": True,
            "natural_language": "run the full regression suite in tuto",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "denied"
        prompt_bytes = b"".join(call.args[0] for call in client.writer.write.call_args_list)
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length])
        assert prompt.msg_type == MessageType.CONFIRM_PROMPT
        assert prompt.payload["target_host"] == "10.0.0.10"
        assert prompt.payload["target_user"] == "root"
        assert prompt.payload["system_name"] == "tuto"

    def test_resolve_named_system_strips_explicit_system_phrase_from_nl(
        self,
        tmp_path: Path,
    ) -> None:
        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tuto.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tuto\n"
            "host: 10.0.0.10\n"
            "user: root\n"
            "---\n\n"
            "# Tuto\n",
            encoding="utf-8",
        )

        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        resolved = handler._resolve_named_system({
            "system_name": "tuto",
            "natural_language": "run smoke tests in system tuto",
        })

        assert isinstance(resolved, dict)
        assert resolved["natural_language"] == "run smoke tests"
        assert resolved["original_natural_language"] == "run smoke tests in system tuto"

    def test_infer_named_system_strips_implicit_system_phrase_from_nl(
        self,
        tmp_path: Path,
    ) -> None:
        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tutorial-box.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tutorial-box\n"
            "aliases:\n"
            "  - tuto\n"
            "host: 10.0.0.10\n"
            "user: root\n"
            "---\n\n"
            "# Tutorial Box\n",
            encoding="utf-8",
        )

        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        resolved = handler._infer_named_system_from_request({
            "infer_target": True,
            "natural_language": "run smoke tests in tuto",
        })

        assert isinstance(resolved, dict)
        assert resolved["resolved_system_name"] == "tutorial-box"
        assert resolved["natural_language"] == "run smoke tests"
        assert resolved["original_natural_language"] == "run smoke tests in tuto"

    def test_infer_named_system_preserves_following_args_after_alias(
        self,
        tmp_path: Path,
    ) -> None:
        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tutorial-box.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tutorial-box\n"
            "aliases:\n"
            "  - tuto\n"
            "host: 10.0.0.10\n"
            "user: root\n"
            "---\n\n"
            "# Tutorial Box\n",
            encoding="utf-8",
        )

        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        resolved = handler._infer_named_system_from_request({
            "infer_target": True,
            "natural_language": "run smoke tests in tuto. 1 iteration",
        })

        assert isinstance(resolved, dict)
        assert resolved["resolved_system_name"] == "tutorial-box"
        assert resolved["natural_language"] == "run smoke tests. 1 iteration"
        assert (
            resolved["original_natural_language"]
            == "run smoke tests in tuto. 1 iteration"
        )

    @pytest.mark.asyncio
    async def test_run_with_named_system_includes_optional_prompt_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tuto.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tuto\n"
            "host: 10.0.0.10\n"
            "hostname: tuto.internal.example\n"
            "ip_address: 10.0.0.10\n"
            "user: root\n"
            "description: Tutorial box for smoke-test runs.\n"
            "---\n\n"
            "# Tuto\n",
            encoding="utf-8",
        )

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        deny_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="deny-optional-001",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": False},
        )
        deny_frame = encode_frame(deny_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[deny_frame[:4], deny_frame[4:]]
        )

        envelope = _make_request(payload={
            "verb": "run",
            "system_name": "tuto",
            "natural_language": "run smoke tests",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        prompt_bytes = b"".join(call.args[0] for call in client.writer.write.call_args_list)
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length])
        assert prompt.payload["system_hostname"] == "tuto.internal.example"
        assert prompt.payload["system_ip_address"] == "10.0.0.10"
        assert prompt.payload["system_description"] == "Tutorial box for smoke-test runs."

    @pytest.mark.asyncio
    async def test_run_with_stored_password_shows_credential_source(
        self,
        tmp_path: Path,
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        deny_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="deny-password-001",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": False},
        )
        deny_frame = encode_frame(deny_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[deny_frame[:4], deny_frame[4:]]
        )

        with patch(
            "jules_daemon.ipc.request_handler.resolve_ssh_credentials",
            return_value=MagicMock(source="credentials_file:/tmp/creds.yaml"),
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "10.0.0.10",
                "target_user": "root",
                "natural_language": "run smoke tests",
            })

            response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        prompt_bytes = b"".join(call.args[0] for call in client.writer.write.call_args_list)
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length])
        assert prompt.payload["auth_mode"] == "password"
        assert prompt.payload["credential_source"] == "credentials_file:/tmp/creds.yaml"

    @pytest.mark.asyncio
    async def test_run_with_unknown_system_returns_error(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        envelope = _make_request(payload={
            "verb": "run",
            "system_name": "missing-box",
            "natural_language": "run smoke tests",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "Unknown system 'missing-box'" in response.payload["error"]

    @pytest.mark.asyncio
    async def test_run_with_infer_target_and_unknown_system_asks_for_target(
        self, tmp_path: Path
    ) -> None:
        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tuto.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tuto\n"
            "host: 10.0.0.10\n"
            "user: root\n"
            "---\n\n"
            "# Tuto\n",
            encoding="utf-8",
        )

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        ask_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="ask-target-001",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True, "answer": "tuto"},
        )
        ask_frame = encode_frame(ask_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[ask_frame[:4], ask_frame[4:]]
        )
        expected = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="req-001",
            timestamp="2026-04-09T12:00:02Z",
            payload={"status": "started"},
        )
        handler._handle_run_oneshot = AsyncMock(return_value=expected)  # type: ignore[method-assign]

        envelope = _make_request(payload={
            "verb": "run",
            "infer_target": True,
            "natural_language": "run smoke tests in qa-box",
        })

        response = await handler.handle_message(envelope, client)

        assert response == expected
        prompt_bytes = b"".join(call.args[0] for call in client.writer.write.call_args_list)
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length])
        assert prompt.msg_type == MessageType.CONFIRM_PROMPT
        assert prompt.payload["type"] == "question"
        run_args = handler._handle_run_oneshot.await_args.args[1]
        assert run_args["target_host"] == "10.0.0.10"
        assert run_args["target_user"] == "root"
        assert run_args["resolved_system_name"] == "tuto"

    @pytest.mark.asyncio
    async def test_run_with_interpret_request_asks_for_target_when_llm_is_unavailable(
        self,
        tmp_path: Path,
    ) -> None:
        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tuto.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tuto\n"
            "host: 10.0.0.10\n"
            "user: root\n"
            "---\n\n"
            "# Tuto\n",
            encoding="utf-8",
        )

        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        client = _make_client()
        ask_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="ask-target-002",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True, "answer": "tuto"},
        )
        ask_frame = encode_frame(ask_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[ask_frame[:4], ask_frame[4:]]
        )
        expected = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="req-001",
            timestamp="2026-04-09T12:00:02Z",
            payload={"status": "started"},
        )
        handler._handle_run_oneshot = AsyncMock(return_value=expected)  # type: ignore[method-assign]

        envelope = _make_request(payload={
            "verb": "run",
            "interpret_request": True,
            "natural_language": "run smoke tests",
        })

        response = await handler.handle_message(envelope, client)

        assert response == expected
        prompt_bytes = b"".join(call.args[0] for call in client.writer.write.call_args_list)
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length])
        assert prompt.msg_type == MessageType.CONFIRM_PROMPT
        assert prompt.payload["type"] == "question"
        assert "Which named system or SSH target" in prompt.payload["question"]
        run_args = handler._handle_run_oneshot.await_args.args[1]
        assert run_args["target_host"] == "10.0.0.10"
        assert run_args["target_user"] == "root"
        assert run_args["natural_language"] == "run smoke tests"

    @pytest.mark.asyncio
    async def test_run_with_interpret_request_uses_llm_resolution_and_cleaned_request(
        self,
        tmp_path: Path,
    ) -> None:
        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tutorial-box.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tutorial-box\n"
            "aliases:\n"
            "  - tuto\n"
            "host: 10.0.0.10\n"
            "user: root\n"
            "---\n\n"
            "# Tutorial Box\n",
            encoding="utf-8",
        )

        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        client = _make_client()
        expected = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="req-001",
            timestamp="2026-04-09T12:00:02Z",
            payload={"status": "started"},
        )
        handler._handle_run_oneshot = AsyncMock(return_value=expected)  # type: ignore[method-assign]

        interpret_mock = AsyncMock(return_value={
            "system_name": "tuto",
            "natural_language": "run smoke tests. 1 iteration",
        })

        with patch.object(
            handler,
            "_interpret_run_request_with_llm",
            interpret_mock,
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "interpret_request": True,
                "natural_language": "run smoke tests in tuto. 1 iteration",
                "agent_original_user_input": "please run smoke tests in tuto. 1 iteration",
            })

            response = await handler.handle_message(envelope, client)

        assert response == expected
        assert interpret_mock.await_args.kwargs["natural_language"] == (
            "please run smoke tests in tuto. 1 iteration"
        )
        run_args = handler._handle_run_oneshot.await_args.args[1]
        assert run_args["target_host"] == "10.0.0.10"
        assert run_args["target_user"] == "root"
        assert run_args["resolved_system_name"] == "tutorial-box"
        assert run_args["natural_language"] == "run smoke tests. 1 iteration"
        assert (
            run_args["original_natural_language"]
            == "run smoke tests in tuto. 1 iteration"
        )

    @pytest.mark.asyncio
    async def test_run_with_interpret_request_uses_llm_followup_question(
        self,
        tmp_path: Path,
    ) -> None:
        systems_dir = tmp_path / "pages" / "systems"
        systems_dir.mkdir(parents=True, exist_ok=True)
        (systems_dir / "tuto.md").write_text(
            "---\n"
            "type: system-info\n"
            "system_name: tuto\n"
            "host: 10.0.0.10\n"
            "user: root\n"
            "---\n\n"
            "# Tuto\n",
            encoding="utf-8",
        )

        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        client = _make_client()
        ask_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="ask-target-003",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True, "answer": "tuto"},
        )
        ask_frame = encode_frame(ask_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[ask_frame[:4], ask_frame[4:]]
        )
        expected = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="req-001",
            timestamp="2026-04-09T12:00:02Z",
            payload={"status": "started"},
        )
        handler._handle_run_oneshot = AsyncMock(return_value=expected)  # type: ignore[method-assign]

        with patch.object(
            handler,
            "_interpret_run_request_with_llm",
            AsyncMock(return_value={
                "natural_language": "run smoke tests. 1 iteration",
                "followup_question": "Which named system should I use for this run?",
            }),
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "interpret_request": True,
                "natural_language": "run smoke tests there. 1 iteration",
            })

            response = await handler.handle_message(envelope, client)

        assert response == expected
        prompt_bytes = b"".join(call.args[0] for call in client.writer.write.call_args_list)
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length])
        assert prompt.payload["question"] == "Which named system should I use for this run?"
        run_args = handler._handle_run_oneshot.await_args.args[1]
        assert run_args["target_host"] == "10.0.0.10"
        assert run_args["target_user"] == "root"
        assert run_args["natural_language"] == "run smoke tests. 1 iteration"


# ---------------------------------------------------------------------------
# RequestHandler: daemon-side interpret verb
# ---------------------------------------------------------------------------


class TestRequestHandlerInterpretVerb:
    """Tests for daemon-side conversational interpretation dispatch."""

    @pytest.mark.asyncio
    async def test_interpret_requires_llm_configuration(
        self,
        tmp_path: Path,
    ) -> None:
        handler = RequestHandler(config=RequestHandlerConfig(wiki_root=tmp_path))
        client = _make_client()

        response = await handler.handle_message(
            _make_request(payload={
                "verb": "interpret",
                "input_text": "give me the current status",
            }),
            client,
        )

        assert response.msg_type == MessageType.ERROR
        assert "Conversational requests require LLM configuration" in (
            response.payload["error"]
        )

    @pytest.mark.asyncio
    async def test_interpret_status_dispatches_to_status(
        self,
        tmp_path: Path,
    ) -> None:
        handler = _make_llm_handler(tmp_path)
        client = _make_client()
        intent = ClassifiedIntent(
            verb=Verb.STATUS,
            confidence=IntentConfidence.HIGH,
            parameters={"verbose": True},
            raw_input="status",
            reasoning="The user wants status.",
        )

        with patch.object(
            handler,
            "_classify_intent_with_llm",
            AsyncMock(return_value=intent),
        ), patch.object(
            handler,
            "_answer_conversational_request_with_llm",
            AsyncMock(return_value=None),
        ):
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "interpret",
                    "input_text": "status",
                }),
                client,
            )

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "status"

    @pytest.mark.asyncio
    async def test_interpret_information_prompt_returns_chat_answer(
        self,
        tmp_path: Path,
    ) -> None:
        handler = _make_llm_handler(tmp_path)
        client = _make_client()

        with patch.object(
            handler,
            "_answer_conversational_request_with_llm",
            AsyncMock(return_value="Yes. I know about test X from the saved spec."),
        ), patch.object(
            handler,
            "_classify_intent_with_llm",
            AsyncMock(),
        ) as classify_mock:
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "interpret",
                    "input_text": "do you know about test X?",
                }),
                client,
            )

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["verb"] == "interpret"
        assert response.payload["status"] == "answered"
        assert response.payload["mode"] == "chat"
        assert "test X" in response.payload["message"]
        classify_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_interpret_run_dispatches_to_run_with_interpret_request(
        self,
        tmp_path: Path,
    ) -> None:
        handler = _make_llm_handler(tmp_path)
        client = _make_client()
        intent = ClassifiedIntent(
            verb=Verb.RUN,
            confidence=IntentConfidence.HIGH,
            parameters={"natural_language": "run the smoke tests"},
            raw_input="run the smoke tests in tuto",
            reasoning="The user wants to start a test run.",
        )
        expected = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="req-001",
            timestamp="2026-04-09T12:00:02Z",
            payload={"verb": "run", "status": "started"},
        )
        run_handler = AsyncMock(return_value=expected)
        handler._async_client_dispatch["run"] = run_handler

        with patch.object(
            handler,
            "_classify_intent_with_llm",
            AsyncMock(return_value=intent),
        ):
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "interpret",
                    "input_text": "run the smoke tests in tuto",
                }),
                client,
            )

        assert response == expected
        parsed_run = run_handler.await_args.args[1]
        assert parsed_run["natural_language"] == "run the smoke tests"
        assert parsed_run["agent_original_user_input"] == "run the smoke tests in tuto"
        assert parsed_run["interpret_request"] is True

    @pytest.mark.asyncio
    async def test_interpret_low_confidence_asks_for_clarification_then_dispatches(
        self,
        tmp_path: Path,
    ) -> None:
        handler = _make_llm_handler(tmp_path)
        client = _make_client()
        ask_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="ask-interpret-001",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True, "answer": "Just show the active run."},
        )
        ask_frame = encode_frame(ask_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[ask_frame[:4], ask_frame[4:]]
        )
        low_intent = ClassifiedIntent(
            verb=Verb.RUN,
            confidence=IntentConfidence.LOW,
            parameters={},
            raw_input="can you run it?",
            reasoning="This looks like a run request, but it is ambiguous.",
        )
        high_intent = ClassifiedIntent(
            verb=Verb.RUN,
            confidence=IntentConfidence.HIGH,
            parameters={"natural_language": "run the active workflow"},
            raw_input="can you run it?",
            reasoning="The clarified request is a run.",
        )
        expected = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="req-001",
            timestamp="2026-04-09T12:00:02Z",
            payload={"verb": "run", "status": "started"},
        )
        run_handler = AsyncMock(return_value=expected)
        handler._async_client_dispatch["run"] = run_handler

        with patch.object(
            handler,
            "_classify_intent_with_llm",
            AsyncMock(side_effect=[low_intent, high_intent]),
        ):
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "interpret",
                    "input_text": "can you run it?",
                }),
                client,
            )

        assert response == expected
        prompt_bytes = b"".join(
            call.args[0] for call in client.writer.write.call_args_list
        )
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(
            prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length]
        )
        assert prompt.msg_type == MessageType.CONFIRM_PROMPT
        assert prompt.payload["type"] == "question"
        assert "Can you clarify" in prompt.payload["question"]

    @pytest.mark.asyncio
    async def test_interpret_validation_error_asks_for_clarification_then_retries(
        self,
        tmp_path: Path,
    ) -> None:
        handler = _make_llm_handler(tmp_path)
        client = _make_client()
        ask_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="ask-interpret-002",
            timestamp="2026-04-09T12:00:01Z",
            payload={
                "approved": True,
                "answer": "Use deploy@staging and command python3 test.py -h.",
            },
        )
        ask_frame = encode_frame(ask_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[ask_frame[:4], ask_frame[4:]]
        )
        invalid_intent = ClassifiedIntent(
            verb=Verb.DISCOVER,
            confidence=IntentConfidence.HIGH,
            parameters={},
            raw_input="discover that test",
            reasoning="The user wants discovery, but fields are missing.",
        )
        valid_intent = ClassifiedIntent(
            verb=Verb.DISCOVER,
            confidence=IntentConfidence.HIGH,
            parameters={
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "command": "python3 test.py -h",
            },
            raw_input="discover that test",
            reasoning="The clarified request has the required SSH target.",
        )
        expected = MessageEnvelope(
            msg_type=MessageType.RESPONSE,
            msg_id="req-001",
            timestamp="2026-04-09T12:00:02Z",
            payload={"verb": "discover", "status": "ok"},
        )
        discover_handler = AsyncMock(return_value=expected)
        handler._async_client_dispatch["discover"] = discover_handler

        with patch.object(
            handler,
            "_classify_intent_with_llm",
            AsyncMock(side_effect=[invalid_intent, valid_intent]),
        ):
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "interpret",
                    "input_text": "discover that test",
                }),
                client,
            )

        assert response == expected
        parsed_discover = discover_handler.await_args.args[1]
        assert parsed_discover["target_host"] == "staging.example.com"
        assert parsed_discover["target_user"] == "deploy"
        assert parsed_discover["command"] == "python3 test.py -h"
        prompt_bytes = b"".join(
            call.args[0] for call in client.writer.write.call_args_list
        )
        prompt_length = unpack_header(prompt_bytes[:HEADER_SIZE])
        prompt = decode_envelope(
            prompt_bytes[HEADER_SIZE:HEADER_SIZE + prompt_length]
        )
        assert prompt.payload["type"] == "question"
        assert "I still need a bit more information" in prompt.payload["context"]


class TestRequestHandlerDiscoverVerb:
    """Tests for discover verb handling."""

    @pytest.mark.asyncio
    async def test_discover_python_script_prompt_uses_python3_first(
        self,
        tmp_path: Path,
    ) -> None:
        from jules_daemon.execution.test_discovery import DiscoveredTestSpec

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        approve_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-pre-reply",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True},
        )
        save_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-save-reply",
            timestamp="2026-04-09T12:00:02Z",
            payload={"approved": True},
        )
        approve_frame = encode_frame(approve_reply)
        save_frame = encode_frame(save_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[
                approve_frame[:4], approve_frame[4:],
                save_frame[:4], save_frame[4:],
            ],
        )

        with patch(
            "jules_daemon.ipc.request_handler.discover_test",
            AsyncMock(return_value=DiscoveredTestSpec(
                command_template="python3 /root/step.py",
                required_args=(),
                optional_args=(),
                arg_descriptions={},
                typical_duration=None,
                raw_help_text="usage: step.py [-h]",
            )),
        ), patch(
            "jules_daemon.ipc.request_handler.save_discovered_spec",
            return_value=tmp_path / "pages" / "daemon" / "knowledge" / "test-step.md",
        ):
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "discover",
                    "target_host": "10.0.0.10",
                    "target_user": "root",
                    "command": "/root/step.py",
                }),
                client,
            )

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "saved"

        prompt_frame = client.writer.write.call_args_list[0].args[0]
        prompt_length = unpack_header(prompt_frame[:HEADER_SIZE])
        prompt = decode_envelope(
            prompt_frame[HEADER_SIZE:HEADER_SIZE + prompt_length]
        )
        assert prompt.msg_type == MessageType.CONFIRM_PROMPT
        assert prompt.payload["proposed_command"] == "python3 /root/step.py -h"
        assert "python3 /root/step.py -h" in prompt.payload["message"]
        assert "python3 /root/step.py --help" in prompt.payload["message"]

    @pytest.mark.asyncio
    async def test_discover_python_script_fallback_requires_second_approval(
        self,
        tmp_path: Path,
    ) -> None:
        from jules_daemon.execution.test_discovery import DiscoveredTestSpec

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        approve_python3_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-pre-python3-reply",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True},
        )
        approve_python_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-pre-python-reply",
            timestamp="2026-04-09T12:00:02Z",
            payload={"approved": True},
        )
        save_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-save-reply",
            timestamp="2026-04-09T12:00:03Z",
            payload={"approved": True},
        )
        frames = [
            encode_frame(approve_python3_reply),
            encode_frame(approve_python_reply),
            encode_frame(save_reply),
        ]
        client.reader.readexactly = AsyncMock(
            side_effect=[
                frames[0][:4], frames[0][4:],
                frames[1][:4], frames[1][4:],
                frames[2][:4], frames[2][4:],
            ],
        )

        sent_frames: list[bytes] = []

        def _capture_write(data: bytes) -> None:
            sent_frames.append(data)

        client.writer.write = _capture_write

        with patch(
            "jules_daemon.ipc.request_handler.discover_test",
            AsyncMock(side_effect=[
                None,
                DiscoveredTestSpec(
                    command_template="python /root/step.py",
                    required_args=(),
                    optional_args=(),
                    arg_descriptions={},
                    typical_duration=None,
                    raw_help_text="usage: step.py [-h]",
                ),
            ]),
        ), patch(
            "jules_daemon.ipc.request_handler.save_discovered_spec",
            return_value=tmp_path / "pages" / "daemon" / "knowledge" / "test-step.md",
        ):
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "discover",
                    "target_host": "10.0.0.10",
                    "target_user": "root",
                    "command": "/root/step.py",
                }),
                client,
            )

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "saved"

        envelopes = [
            decode_envelope(frame[HEADER_SIZE:]) for frame in sent_frames
        ]
        confirm_prompts = [
            env for env in envelopes if env.msg_type == MessageType.CONFIRM_PROMPT
        ]
        assert confirm_prompts[0].payload["proposed_command"] == "python3 /root/step.py -h"
        assert confirm_prompts[1].payload["proposed_command"] == "python /root/step.py -h"
        assert "python /root/step.py --help" in confirm_prompts[1].payload["message"]

    @pytest.mark.asyncio
    async def test_discover_auth_failure_does_not_prompt_python_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        from jules_daemon.ssh.errors import SSHAuthenticationError

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        approve_python3_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-pre-python3-reply",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True},
        )
        frame = encode_frame(approve_python3_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[frame[:4], frame[4:]],
        )

        sent_frames: list[bytes] = []

        def _capture_write(data: bytes) -> None:
            sent_frames.append(data)

        client.writer.write = _capture_write

        with patch(
            "jules_daemon.ipc.request_handler.discover_test",
            AsyncMock(side_effect=SSHAuthenticationError("Authentication failed")),
        ):
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "discover",
                    "target_host": "10.0.0.10",
                    "target_user": "root",
                    "command": "/root/step.py",
                }),
                client,
            )

        assert response.msg_type == MessageType.ERROR
        assert "Authentication failed" in response.payload["error"]

        envelopes = [
            decode_envelope(frame[HEADER_SIZE:]) for frame in sent_frames
        ]
        confirm_prompts = [
            env for env in envelopes if env.msg_type == MessageType.CONFIRM_PROMPT
        ]
        assert len(confirm_prompts) == 1
        assert confirm_prompts[0].payload["proposed_command"] == "python3 /root/step.py -h"

    @pytest.mark.asyncio
    async def test_discover_probe_failure_explains_reason_before_python_fallback(
        self,
        tmp_path: Path,
    ) -> None:
        from jules_daemon.execution.test_discovery import (
            DiscoveryProbeError,
            DiscoveredTestSpec,
        )

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        approve_python3_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-pre-python3-reply",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True},
        )
        approve_python_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-pre-python-reply",
            timestamp="2026-04-09T12:00:02Z",
            payload={"approved": True},
        )
        save_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-save-reply",
            timestamp="2026-04-09T12:00:03Z",
            payload={"approved": True},
        )
        frames = [
            encode_frame(approve_python3_reply),
            encode_frame(approve_python_reply),
            encode_frame(save_reply),
        ]
        client.reader.readexactly = AsyncMock(
            side_effect=[
                frames[0][:4], frames[0][4:],
                frames[1][:4], frames[1][4:],
                frames[2][:4], frames[2][4:],
            ],
        )

        sent_frames: list[bytes] = []

        def _capture_write(data: bytes) -> None:
            sent_frames.append(data)

        client.writer.write = _capture_write

        with patch(
            "jules_daemon.ipc.request_handler.discover_test",
            AsyncMock(side_effect=[
                DiscoveryProbeError(
                    executed_command="python3 /root/step.py",
                    attempted_help_commands=(
                        "python3 /root/step.py -h",
                        "python3 /root/step.py --help",
                    ),
                    exit_code=1,
                    stdout_text="",
                    stderr_text="python3: command not found",
                ),
                DiscoveredTestSpec(
                    command_template="python /root/step.py",
                    required_args=(),
                    optional_args=(),
                    arg_descriptions={},
                    typical_duration=None,
                    raw_help_text="usage: step.py [-h]",
                ),
            ]),
        ), patch(
            "jules_daemon.ipc.request_handler.save_discovered_spec",
            return_value=tmp_path / "pages" / "daemon" / "knowledge" / "test-step.md",
        ):
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "discover",
                    "target_host": "10.0.0.10",
                    "target_user": "root",
                    "command": "/root/step.py",
                }),
                client,
            )

        assert response.msg_type == MessageType.RESPONSE
        assert response.payload["status"] == "saved"

        envelopes = [
            decode_envelope(frame[HEADER_SIZE:]) for frame in sent_frames
        ]
        stream_lines = [
            env.payload["line"]
            for env in envelopes
            if env.msg_type == MessageType.STREAM and "line" in env.payload
        ]
        assert any("python3: command not found" in line for line in stream_lines)
        assert any("python /root/step.py -h" in line for line in stream_lines)

    @pytest.mark.asyncio
    async def test_discover_final_probe_failure_includes_root_cause(
        self,
        tmp_path: Path,
    ) -> None:
        from jules_daemon.execution.test_discovery import DiscoveryProbeError

        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        approve_reply = MessageEnvelope(
            msg_type=MessageType.CONFIRM_REPLY,
            msg_id="discover-pre-reply",
            timestamp="2026-04-09T12:00:01Z",
            payload={"approved": True},
        )
        frame = encode_frame(approve_reply)
        client.reader.readexactly = AsyncMock(
            side_effect=[frame[:4], frame[4:]],
        )

        with patch(
            "jules_daemon.ipc.request_handler.discover_test",
            AsyncMock(side_effect=DiscoveryProbeError(
                executed_command="python3 /root/step.py",
                attempted_help_commands=(
                    "python3 /root/step.py -h",
                    "python3 /root/step.py --help",
                ),
                exit_code=1,
                stdout_text="",
                stderr_text="permission denied",
            )),
        ):
            response = await handler.handle_message(
                _make_request(payload={
                    "verb": "discover",
                    "target_host": "10.0.0.10",
                    "target_user": "root",
                    "command": "python3 /root/step.py",
                }),
                client,
            )

        assert response.msg_type == MessageType.ERROR
        assert "permission denied" in response.payload["error"]
        assert "exited with code 1" in response.payload["error"]


# ---------------------------------------------------------------------------
# RequestHandler: response correlation
# ---------------------------------------------------------------------------


class TestRequestHandlerCorrelation:
    """Tests for request/response ID correlation."""

    @pytest.mark.asyncio
    async def test_response_preserves_msg_id(self, tmp_path: Path) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(
            payload={"verb": "status"},
            msg_id="unique-correlation-id",
        )

        response = await handler.handle_message(envelope, client)
        assert response.msg_id == "unique-correlation-id"

    @pytest.mark.asyncio
    async def test_error_response_preserves_msg_id(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(
            payload={"verb": "teleport"},
            msg_id="err-correlation-id",
        )

        response = await handler.handle_message(envelope, client)
        assert response.msg_id == "err-correlation-id"


# ---------------------------------------------------------------------------
# RequestHandler: response envelope structure
# ---------------------------------------------------------------------------


class TestRequestHandlerResponseStructure:
    """Tests for response envelope structure consistency."""

    @pytest.mark.asyncio
    async def test_error_has_validation_errors_list(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "error" in response.payload
        assert "validation_errors" in response.payload
        assert isinstance(response.payload["validation_errors"], list)
        for err in response.payload["validation_errors"]:
            assert "field" in err
            assert "message" in err
            assert "code" in err

    @pytest.mark.asyncio
    async def test_success_has_verb_and_timestamp(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()
        envelope = _make_request(payload={"verb": "status"})

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.RESPONSE
        assert "verb" in response.payload
        assert response.timestamp  # non-empty
