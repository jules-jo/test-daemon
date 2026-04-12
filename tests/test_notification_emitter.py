"""Tests for the notification emitter wiring.

Covers the three emitter functions that bridge domain results to the
notification broadcaster:

    ``emit_agent_loop_completion``:
        - COMPLETE state emits a CompletionNotification (exit_status=0)
        - ERROR state emits an AlertNotification with error details
        - ERROR with denial emits WARNING severity
        - ERROR with retry_exhausted emits WARNING severity
        - Other ERROR emits ERROR severity
        - Non-terminal state returns None (defensive)
        - None broadcaster returns None (no-op)

    ``emit_run_completion``:
        - Successful run emits CompletionNotification with exit_status=0
        - Failed run emits CompletionNotification with non-zero exit_status
        - None exit_code maps to -1 sentinel
        - None broadcaster returns None

    ``emit_alert``:
        - Emits AlertNotification with correct severity, title, message
        - Optional run_id and details forwarded
        - None broadcaster returns None
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from jules_daemon.agent.agent_loop import AgentLoopState
from jules_daemon.ipc.notification_broadcaster import (
    BroadcastResult,
    NotificationBroadcaster,
)
from jules_daemon.ipc.notification_emitter import (
    emit_agent_loop_completion,
    emit_alert,
    emit_run_completion,
)
from jules_daemon.protocol.notifications import (
    AlertNotification,
    CompletionNotification,
    NotificationEventType,
    NotificationSeverity,
)


@dataclass(frozen=True)
class _FakeAgentLoopResult:
    """Mimics AgentLoopResult for testing.

    Uses the real ``AgentLoopState`` enum so ``is`` comparisons in the
    emitter work correctly.
    """

    final_state: AgentLoopState
    iterations_used: int
    history: tuple[dict[str, Any], ...] = ()
    error_message: str | None = None
    retry_exhausted: bool = False


@dataclass(frozen=True)
class _FakeRunResult:
    """Mimics RunResult for testing."""

    success: bool
    run_id: str
    command: str
    target_host: str
    target_user: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Recording broadcaster
# ---------------------------------------------------------------------------


class _RecordingBroadcaster(NotificationBroadcaster):
    """NotificationBroadcaster subclass that records all broadcast calls.

    Instead of delivering to subscriber queues, captures the envelopes
    for assertion. Still returns a real BroadcastResult.
    """

    def __init__(self) -> None:
        super().__init__()
        self.envelopes: list[Any] = []

    async def broadcast(self, envelope: Any) -> BroadcastResult:
        """Record the envelope and return a synthetic result."""
        self.envelopes.append(envelope)
        return BroadcastResult(delivered_count=1)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def recorder() -> _RecordingBroadcaster:
    """Create a fresh recording broadcaster."""
    return _RecordingBroadcaster()


# ---------------------------------------------------------------------------
# emit_agent_loop_completion tests
# ---------------------------------------------------------------------------


class TestEmitAgentLoopCompletion:
    """Tests for emit_agent_loop_completion."""

    @pytest.mark.asyncio()
    async def test_none_broadcaster_returns_none(self) -> None:
        """No-op when broadcaster is None."""
        result = await emit_agent_loop_completion(
            broadcaster=None,
            loop_result=_FakeAgentLoopResult(
                final_state=AgentLoopState.COMPLETE,
                iterations_used=1,
            ),
            natural_language_command="run tests",
        )
        assert result is None

    @pytest.mark.asyncio()
    async def test_complete_state_emits_completion(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """COMPLETE state emits a CompletionNotification."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=3,
        )
        result = await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="run smoke tests",
            run_id="run-abc",
        )
        assert result is not None
        assert result.delivered_count == 1
        assert len(recorder.envelopes) == 1

        envelope = recorder.envelopes[0]
        assert envelope.event_type is NotificationEventType.COMPLETION
        assert isinstance(envelope.payload, CompletionNotification)
        assert envelope.payload.run_id == "run-abc"
        assert envelope.payload.natural_language_command == "run smoke tests"
        assert envelope.payload.exit_status == 0

    @pytest.mark.asyncio()
    async def test_complete_state_with_resolved_shell(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """COMPLETE state forwards the resolved shell command."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=2,
        )
        await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="run pytest",
            run_id="run-xyz",
            resolved_shell="pytest tests/ -v",
        )
        assert len(recorder.envelopes) == 1
        payload = recorder.envelopes[0].payload
        assert payload.resolved_shell == "pytest tests/ -v"

    @pytest.mark.asyncio()
    async def test_error_state_emits_alert(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """ERROR state emits an AlertNotification."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=5,
            error_message="Agent loop reached max iterations (5)",
        )
        result = await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="run integration tests",
            run_id="run-err",
        )
        assert result is not None
        assert len(recorder.envelopes) == 1

        envelope = recorder.envelopes[0]
        assert envelope.event_type is NotificationEventType.ALERT
        assert isinstance(envelope.payload, AlertNotification)
        assert envelope.payload.run_id == "run-err"
        assert envelope.payload.severity is NotificationSeverity.ERROR
        assert "Agent loop reached max iterations" in envelope.payload.message

    @pytest.mark.asyncio()
    async def test_error_denied_emits_warning_severity(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """User denial error emits WARNING severity."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=2,
            error_message="Tool 'execute_ssh' was denied: User denied the operation",
        )
        await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="run tests",
            run_id="run-denied",
        )
        payload = recorder.envelopes[0].payload
        assert payload.severity is NotificationSeverity.WARNING
        assert "denied" in payload.title.lower()

    @pytest.mark.asyncio()
    async def test_error_retry_exhausted_emits_warning_severity(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Retry-exhausted error emits WARNING severity."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=1,
            error_message="LLM transient error after exhausting retries",
            retry_exhausted=True,
        )
        await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="run tests",
            run_id="run-retry",
        )
        payload = recorder.envelopes[0].payload
        assert payload.severity is NotificationSeverity.WARNING
        assert "retries exhausted" in payload.title.lower()

    @pytest.mark.asyncio()
    async def test_error_details_include_metadata(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Alert details carry loop metadata."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=4,
            error_message="Permanent LLM error",
            retry_exhausted=False,
        )
        await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="run auth tests",
            run_id="run-meta",
        )
        details = recorder.envelopes[0].payload.details
        assert details is not None
        assert details["iterations_used"] == 4
        assert details["retry_exhausted"] is False
        assert details["natural_language_command"] == "run auth tests"

    @pytest.mark.asyncio()
    async def test_error_none_message_uses_default(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """None error_message falls back to default string."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=1,
            error_message=None,
        )
        await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="run tests",
            run_id="run-none-msg",
        )
        payload = recorder.envelopes[0].payload
        assert payload.message == "Agent loop terminated"

    @pytest.mark.asyncio()
    async def test_non_terminal_state_returns_none(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Non-terminal state (THINKING) returns None without emitting."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.THINKING,
            iterations_used=1,
        )
        result = await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="run tests",
        )
        assert result is None
        assert len(recorder.envelopes) == 0

    @pytest.mark.asyncio()
    async def test_auto_generates_run_id_when_none(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """When run_id is None, a synthetic ID is generated."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=1,
        )
        await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="run tests",
            run_id=None,
        )
        payload = recorder.envelopes[0].payload
        assert payload.run_id  # non-empty
        assert payload.run_id.startswith("agent-loop-")


# ---------------------------------------------------------------------------
# emit_run_completion tests
# ---------------------------------------------------------------------------


class TestEmitRunCompletion:
    """Tests for emit_run_completion."""

    @pytest.mark.asyncio()
    async def test_none_broadcaster_returns_none(self) -> None:
        """No-op when broadcaster is None."""
        result = await emit_run_completion(
            broadcaster=None,
            run_result=_FakeRunResult(
                success=True,
                run_id="run-001",
                command="pytest",
                target_host="host",
                target_user="user",
                exit_code=0,
            ),
        )
        assert result is None

    @pytest.mark.asyncio()
    async def test_successful_run_emits_completion(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Successful run emits CompletionNotification with exit_status=0."""
        run_result = _FakeRunResult(
            success=True,
            run_id="run-success",
            command="pytest tests/",
            target_host="staging",
            target_user="deploy",
            exit_code=0,
            duration_seconds=45.2,
        )
        result = await emit_run_completion(
            broadcaster=recorder,
            run_result=run_result,
        )
        assert result is not None
        assert result.delivered_count == 1

        envelope = recorder.envelopes[0]
        assert envelope.event_type is NotificationEventType.COMPLETION
        payload = envelope.payload
        assert isinstance(payload, CompletionNotification)
        assert payload.run_id == "run-success"
        assert payload.exit_status == 0
        assert payload.resolved_shell == "pytest tests/"
        assert payload.duration_seconds == 45.2
        assert payload.error_message is None

    @pytest.mark.asyncio()
    async def test_failed_run_emits_completion_with_error(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Failed run emits CompletionNotification with non-zero exit_status."""
        run_result = _FakeRunResult(
            success=False,
            run_id="run-fail",
            command="pytest tests/auth/",
            target_host="staging",
            target_user="deploy",
            exit_code=1,
            duration_seconds=12.0,
            error="Command exited with code 1",
        )
        result = await emit_run_completion(
            broadcaster=recorder,
            run_result=run_result,
        )
        assert result is not None

        payload = recorder.envelopes[0].payload
        assert payload.exit_status == 1
        assert payload.error_message == "Command exited with code 1"

    @pytest.mark.asyncio()
    async def test_none_exit_code_maps_to_negative_one(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """When exit_code is None (connection failure), uses -1 sentinel."""
        run_result = _FakeRunResult(
            success=False,
            run_id="run-conn-fail",
            command="pytest",
            target_host="host",
            target_user="user",
            exit_code=None,
            error="SSH connection failed",
        )
        await emit_run_completion(
            broadcaster=recorder,
            run_result=run_result,
        )
        payload = recorder.envelopes[0].payload
        assert payload.exit_status == -1

    @pytest.mark.asyncio()
    async def test_custom_natural_language_override(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """natural_language_command override takes precedence."""
        run_result = _FakeRunResult(
            success=True,
            run_id="run-nl",
            command="pytest tests/",
            target_host="host",
            target_user="user",
            exit_code=0,
        )
        await emit_run_completion(
            broadcaster=recorder,
            run_result=run_result,
            natural_language_command="run all unit tests",
        )
        payload = recorder.envelopes[0].payload
        assert payload.natural_language_command == "run all unit tests"
        assert payload.resolved_shell == "pytest tests/"

    @pytest.mark.asyncio()
    async def test_default_natural_language_from_command(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Without override, natural_language_command comes from run_result.command."""
        run_result = _FakeRunResult(
            success=True,
            run_id="run-default-nl",
            command="python3 test.py",
            target_host="host",
            target_user="user",
            exit_code=0,
        )
        await emit_run_completion(
            broadcaster=recorder,
            run_result=run_result,
        )
        payload = recorder.envelopes[0].payload
        assert payload.natural_language_command == "python3 test.py"

    @pytest.mark.asyncio()
    async def test_zero_duration_not_emitted(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Zero duration maps to None (not emitted)."""
        run_result = _FakeRunResult(
            success=True,
            run_id="run-no-dur",
            command="pytest",
            target_host="host",
            target_user="user",
            exit_code=0,
            duration_seconds=0.0,
        )
        await emit_run_completion(
            broadcaster=recorder,
            run_result=run_result,
        )
        payload = recorder.envelopes[0].payload
        # 0.0 is falsy, so `or None` makes it None
        assert payload.duration_seconds is None


# ---------------------------------------------------------------------------
# emit_alert tests
# ---------------------------------------------------------------------------


class TestEmitAlert:
    """Tests for emit_alert."""

    @pytest.mark.asyncio()
    async def test_none_broadcaster_returns_none(self) -> None:
        """No-op when broadcaster is None."""
        result = await emit_alert(
            broadcaster=None,
            severity=NotificationSeverity.ERROR,
            title="Test",
            message="Test message",
        )
        assert result is None

    @pytest.mark.asyncio()
    async def test_emits_alert_with_all_fields(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Alert with all fields is emitted correctly."""
        result = await emit_alert(
            broadcaster=recorder,
            severity=NotificationSeverity.WARNING,
            title="High failure rate",
            message="50% of tests are failing in run-abc",
            run_id="run-abc",
            details={"failure_rate": 0.5, "total_tests": 100},
        )
        assert result is not None
        assert result.delivered_count == 1

        envelope = recorder.envelopes[0]
        assert envelope.event_type is NotificationEventType.ALERT
        payload = envelope.payload
        assert isinstance(payload, AlertNotification)
        assert payload.severity is NotificationSeverity.WARNING
        assert payload.title == "High failure rate"
        assert payload.message == "50% of tests are failing in run-abc"
        assert payload.run_id == "run-abc"
        assert payload.details == {"failure_rate": 0.5, "total_tests": 100}

    @pytest.mark.asyncio()
    async def test_emits_alert_without_optional_fields(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Alert without run_id and details works."""
        await emit_alert(
            broadcaster=recorder,
            severity=NotificationSeverity.INFO,
            title="Daemon started",
            message="SSH daemon is ready for connections",
        )
        payload = recorder.envelopes[0].payload
        assert payload.run_id is None
        assert payload.details is None

    @pytest.mark.asyncio()
    async def test_error_severity(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """ERROR severity is correctly forwarded."""
        await emit_alert(
            broadcaster=recorder,
            severity=NotificationSeverity.ERROR,
            title="SSH auth failure",
            message="Authentication failed for deploy@staging",
        )
        assert recorder.envelopes[0].payload.severity is NotificationSeverity.ERROR

    @pytest.mark.asyncio()
    async def test_success_severity(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """SUCCESS severity is correctly forwarded."""
        await emit_alert(
            broadcaster=recorder,
            severity=NotificationSeverity.SUCCESS,
            title="Tests passed",
            message="All 100 tests passed successfully",
        )
        assert recorder.envelopes[0].payload.severity is NotificationSeverity.SUCCESS


# ---------------------------------------------------------------------------
# Envelope metadata tests
# ---------------------------------------------------------------------------


class TestEnvelopeMetadata:
    """Tests that emitted envelopes have correct metadata."""

    @pytest.mark.asyncio()
    async def test_completion_envelope_has_event_id(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Completion envelopes have auto-generated event IDs."""
        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=1,
        )
        await emit_agent_loop_completion(
            broadcaster=recorder,
            loop_result=loop_result,
            natural_language_command="test",
            run_id="run-id",
        )
        envelope = recorder.envelopes[0]
        assert envelope.event_id  # non-empty
        assert envelope.event_id.startswith("evt-")

    @pytest.mark.asyncio()
    async def test_alert_envelope_has_timestamp(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Alert envelopes have UTC timestamps."""
        await emit_alert(
            broadcaster=recorder,
            severity=NotificationSeverity.INFO,
            title="Test",
            message="Message",
        )
        envelope = recorder.envelopes[0]
        assert envelope.timestamp is not None
        assert envelope.timestamp.tzinfo is not None

    @pytest.mark.asyncio()
    async def test_run_completion_envelope_has_channel_version(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Run completion envelopes carry the channel version."""
        from jules_daemon.protocol.notifications import (
            NOTIFICATION_CHANNEL_VERSION,
        )

        run_result = _FakeRunResult(
            success=True,
            run_id="run-ver",
            command="pytest",
            target_host="host",
            target_user="user",
            exit_code=0,
        )
        await emit_run_completion(
            broadcaster=recorder,
            run_result=run_result,
        )
        envelope = recorder.envelopes[0]
        assert envelope.channel_version == NOTIFICATION_CHANNEL_VERSION

    @pytest.mark.asyncio()
    async def test_unique_event_ids_across_calls(
        self,
        recorder: _RecordingBroadcaster,
    ) -> None:
        """Each emitted envelope gets a unique event ID."""
        for _ in range(3):
            await emit_alert(
                broadcaster=recorder,
                severity=NotificationSeverity.INFO,
                title="Test",
                message="Message",
            )
        ids = [env.event_id for env in recorder.envelopes]
        assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# Integration with real broadcaster
# ---------------------------------------------------------------------------


class TestRealBroadcasterIntegration:
    """Integration tests using a real NotificationBroadcaster."""

    @pytest.mark.asyncio()
    async def test_agent_loop_completion_delivered_to_subscriber(self) -> None:
        """Agent loop completion is delivered to a real subscriber."""
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.COMPLETION}),
        )

        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=2,
        )
        broadcast_result = await emit_agent_loop_completion(
            broadcaster=broadcaster,
            loop_result=loop_result,
            natural_language_command="run smoke tests",
            run_id="run-int-test",
        )
        assert broadcast_result is not None
        assert broadcast_result.delivered_count == 1

        # Receive the event from the subscriber queue
        envelope = await broadcaster.receive(
            handle.subscription_id,
            timeout=1.0,
        )
        assert envelope is not None
        assert envelope.event_type is NotificationEventType.COMPLETION
        assert envelope.payload.run_id == "run-int-test"

        await broadcaster.unsubscribe(handle.subscription_id)

    @pytest.mark.asyncio()
    async def test_agent_loop_error_delivered_as_alert(self) -> None:
        """Agent loop error is delivered as an alert to a subscriber."""
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.ALERT}),
        )

        loop_result = _FakeAgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=5,
            error_message="Max iterations reached",
        )
        broadcast_result = await emit_agent_loop_completion(
            broadcaster=broadcaster,
            loop_result=loop_result,
            natural_language_command="run tests",
            run_id="run-alert-test",
        )
        assert broadcast_result is not None
        assert broadcast_result.delivered_count == 1

        envelope = await broadcaster.receive(
            handle.subscription_id,
            timeout=1.0,
        )
        assert envelope is not None
        assert envelope.event_type is NotificationEventType.ALERT
        assert isinstance(envelope.payload, AlertNotification)
        assert envelope.payload.run_id == "run-alert-test"

        await broadcaster.unsubscribe(handle.subscription_id)

    @pytest.mark.asyncio()
    async def test_run_completion_delivered_to_subscriber(self) -> None:
        """Run pipeline completion is delivered to a subscriber."""
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe()  # all event types

        run_result = _FakeRunResult(
            success=True,
            run_id="run-pipe-test",
            command="pytest tests/ -v",
            target_host="staging",
            target_user="deploy",
            exit_code=0,
            duration_seconds=23.5,
        )
        await emit_run_completion(
            broadcaster=broadcaster,
            run_result=run_result,
            natural_language_command="run the tests",
        )

        envelope = await broadcaster.receive(
            handle.subscription_id,
            timeout=1.0,
        )
        assert envelope is not None
        assert envelope.event_type is NotificationEventType.COMPLETION
        payload = envelope.payload
        assert payload.run_id == "run-pipe-test"
        assert payload.natural_language_command == "run the tests"
        assert payload.resolved_shell == "pytest tests/ -v"
        assert payload.exit_status == 0

        await broadcaster.unsubscribe(handle.subscription_id)

    @pytest.mark.asyncio()
    async def test_alert_filtered_by_subscriber(self) -> None:
        """Subscriber with COMPLETION-only filter does not receive alerts."""
        broadcaster = NotificationBroadcaster()
        handle = await broadcaster.subscribe(
            event_filter=frozenset({NotificationEventType.COMPLETION}),
        )

        result = await emit_alert(
            broadcaster=broadcaster,
            severity=NotificationSeverity.WARNING,
            title="Test alert",
            message="This should be filtered out",
        )
        assert result is not None
        assert result.filtered_count == 1
        assert result.delivered_count == 0

        # Should timeout because the alert was filtered
        envelope = await broadcaster.receive(
            handle.subscription_id,
            timeout=0.1,
        )
        assert envelope is None

        await broadcaster.unsubscribe(handle.subscription_id)

    @pytest.mark.asyncio()
    async def test_no_subscribers_produces_zero_deliveries(self) -> None:
        """Broadcasting with no subscribers returns zero deliveries."""
        broadcaster = NotificationBroadcaster()
        result = await emit_run_completion(
            broadcaster=broadcaster,
            run_result=_FakeRunResult(
                success=True,
                run_id="run-empty",
                command="pytest",
                target_host="host",
                target_user="user",
                exit_code=0,
            ),
        )
        assert result is not None
        assert result.delivered_count == 0
        assert result.total_subscribers == 0
