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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from jules_daemon.protocol.notifications import NotificationEventType
from jules_daemon.ipc.server import ClientConnection


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
    async def test_run_with_infer_target_and_unknown_system_returns_error(
        self, tmp_path: Path
    ) -> None:
        config = RequestHandlerConfig(wiki_root=tmp_path)
        handler = RequestHandler(config=config)
        client = _make_client()

        envelope = _make_request(payload={
            "verb": "run",
            "infer_target": True,
            "natural_language": "run smoke tests in tuto",
        })

        response = await handler.handle_message(envelope, client)

        assert response.msg_type == MessageType.ERROR
        assert "Could not infer a named system" in response.payload["error"]


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
