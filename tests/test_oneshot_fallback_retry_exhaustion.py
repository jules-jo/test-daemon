"""Tests for one-shot fallback path after retry exhaustion (Sub-AC 3).

Validates that when the agent loop exhausts all transient error retries,
the system falls back to the original single-pass LLM translation and
returns its result. This is the specific fallback path for transient
errors that persist through all retry attempts (network blips, LLM
timeouts, etc.).

Coverage:
  - AgentLoopResult.retry_exhausted is True after transient retry exhaustion
  - AgentLoopResult.retry_exhausted is False for permanent errors
  - AgentLoopResult.retry_exhausted is False for successful completion
  - AgentLoopResult.retry_exhausted is False for user denial
  - RetryExhaustedError stores iterations_used metadata
  - _handle_run_agent_loop raises RetryExhaustedError on retry_exhausted
  - _handle_run catches RetryExhaustedError and falls through to one-shot
  - One-shot path returns the LLM translation result after fallback
  - Permanent errors do NOT trigger the one-shot fallback via retry path
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jules_daemon.agent.agent_loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentLoopResult,
    AgentLoopState,
)
from jules_daemon.agent.error_classification import (
    RetryExhaustedError,
)
from jules_daemon.agent.tool_types import (
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FailingLLMClient:
    """Mock LLM client that always raises a given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.call_count = 0

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self.call_count += 1
        raise self._exc


class SuccessLLMClient:
    """Mock LLM client that returns empty tool calls (completion)."""

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        return ()


class SingleToolLLMClient:
    """Mock LLM client that returns one tool call then completes."""

    def __init__(self) -> None:
        self._call_count = 0

    async def get_tool_calls(
        self,
        messages: tuple[dict[str, Any], ...],
    ) -> tuple[ToolCall, ...]:
        self._call_count += 1
        if self._call_count == 1:
            return (
                ToolCall(
                    call_id="call-1",
                    tool_name="read_wiki",
                    arguments={"page": "test"},
                ),
            )
        return ()


class MockToolDispatcher:
    """Mock tool dispatcher that returns success results."""

    async def dispatch(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolResultStatus.SUCCESS,
            output="ok",
        )


class DenialToolDispatcher:
    """Mock tool dispatcher that returns a DENIED result."""

    async def dispatch(self, call: ToolCall) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolResultStatus.DENIED,
            output="",
            error_message="User denied the operation",
        )


# No-op sleep for deterministic tests
_noop_sleep = AsyncMock()


# ---------------------------------------------------------------------------
# AgentLoopResult.retry_exhausted flag
# ---------------------------------------------------------------------------


class TestRetryExhaustedFlag:
    """Tests for the retry_exhausted flag on AgentLoopResult."""

    @pytest.mark.asyncio
    async def test_transient_retries_exhausted_sets_flag(self) -> None:
        """retry_exhausted is True when all transient retries are consumed."""
        llm = FailingLLMClient(ConnectionError("network down"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(
            max_retries=2, retry_base_delay=0.0,
        )
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run smoke tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert "falling back to one-shot" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_permanent_error_does_not_set_retry_exhausted(self) -> None:
        """retry_exhausted is False for permanent errors (no retries attempted)."""
        llm = FailingLLMClient(ValueError("malformed JSON"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(
            max_retries=2, retry_base_delay=0.0,
        )
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is False
        assert "permanent" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_successful_completion_no_retry_exhausted(self) -> None:
        """retry_exhausted is False when the loop completes successfully."""
        llm = SuccessLLMClient()
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False

    @pytest.mark.asyncio
    async def test_user_denial_no_retry_exhausted(self) -> None:
        """retry_exhausted is False when user denies a tool call."""
        llm = SingleToolLLMClient()
        dispatcher = DenialToolDispatcher()

        config = AgentLoopConfig(max_retries=2, retry_base_delay=0.0)
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is False

    @pytest.mark.asyncio
    async def test_max_iterations_no_retry_exhausted(self) -> None:
        """retry_exhausted is False when max iterations cap is reached."""

        class InfiniteToolLLM:
            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                return (
                    ToolCall(
                        call_id="call-inf",
                        tool_name="read_wiki",
                        arguments={"page": "test"},
                    ),
                )

        config = AgentLoopConfig(
            max_iterations=2,
            max_retries=2,
            retry_base_delay=0.0,
        )
        loop = AgentLoop(
            llm_client=InfiniteToolLLM(),
            tool_dispatcher=MockToolDispatcher(),
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is False
        assert "max iterations" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_zero_retries_still_sets_retry_exhausted(self) -> None:
        """With max_retries=0, transient errors set retry_exhausted immediately."""
        llm = FailingLLMClient(ConnectionError("down"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(
            max_retries=0, retry_base_delay=0.0,
        )
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        # Only 1 call (no retries)
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_error_exhausts_retries(self) -> None:
        """TimeoutError (transient) also sets retry_exhausted after exhaustion."""
        llm = FailingLLMClient(TimeoutError("LLM timed out"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(
            max_retries=1, retry_base_delay=0.0,
        )
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert llm.call_count == 2  # 1 original + 1 retry

    @pytest.mark.asyncio
    async def test_retry_exhausted_default_false(self) -> None:
        """AgentLoopResult.retry_exhausted defaults to False."""
        result = AgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=1,
            history=(),
            error_message=None,
        )
        assert result.retry_exhausted is False


# ---------------------------------------------------------------------------
# RetryExhaustedError
# ---------------------------------------------------------------------------


class TestRetryExhaustedError:
    """Tests for the RetryExhaustedError exception class."""

    def test_is_agent_error_subclass(self) -> None:
        """RetryExhaustedError inherits from AgentError."""
        from jules_daemon.agent.error_classification import AgentError

        err = RetryExhaustedError("test failure")
        assert isinstance(err, AgentError)
        assert isinstance(err, Exception)

    def test_stores_message(self) -> None:
        """Message is preserved in the exception."""
        err = RetryExhaustedError("LLM transient error after retries")
        assert str(err) == "LLM transient error after retries"

    def test_stores_iterations_used(self) -> None:
        """iterations_used metadata is stored on the exception."""
        err = RetryExhaustedError("failed", iterations_used=3)
        assert err.iterations_used == 3

    def test_iterations_used_defaults_zero(self) -> None:
        """iterations_used defaults to 0 when not provided."""
        err = RetryExhaustedError("failed")
        assert err.iterations_used == 0

    def test_can_be_caught_as_exception(self) -> None:
        """Can be caught as a generic Exception."""
        with pytest.raises(Exception, match="retry failed"):
            raise RetryExhaustedError("retry failed", iterations_used=2)

    def test_can_be_caught_specifically(self) -> None:
        """Can be caught specifically as RetryExhaustedError."""
        with pytest.raises(RetryExhaustedError) as exc_info:
            raise RetryExhaustedError("test", iterations_used=5)

        assert exc_info.value.iterations_used == 5


# ---------------------------------------------------------------------------
# _handle_run_agent_loop raises RetryExhaustedError
# ---------------------------------------------------------------------------


def _make_client() -> Any:
    """Build a stub ClientConnection for testing."""
    from jules_daemon.ipc.server import ClientConnection

    return ClientConnection(
        client_id="test-client-retry",
        reader=AsyncMock(spec=asyncio.StreamReader),
        writer=AsyncMock(spec=asyncio.StreamWriter),
        connected_at="2026-04-12T12:00:00Z",
    )


def _make_request(
    payload: dict[str, Any],
    msg_id: str = "req-retry-001",
) -> Any:
    """Build a REQUEST-type envelope."""
    from jules_daemon.ipc.framing import MessageEnvelope, MessageType

    return MessageEnvelope(
        msg_type=MessageType.REQUEST,
        msg_id=msg_id,
        timestamp="2026-04-12T12:00:00Z",
        payload=payload,
    )


def _make_llm_client() -> MagicMock:
    """Build a mock OpenAI client."""
    return MagicMock()


def _make_llm_config() -> MagicMock:
    """Build a mock LLMConfig."""
    config = MagicMock()
    config.default_model = "provider:connection:model-v1"
    return config


def _setup_deny_reply(client: Any) -> None:
    """Configure the mock client to return a deny CONFIRM_REPLY."""
    from jules_daemon.ipc.framing import (
        MessageEnvelope,
        MessageType,
        encode_frame,
    )

    deny_reply = MessageEnvelope(
        msg_type=MessageType.CONFIRM_REPLY,
        msg_id="deny-retry-001",
        timestamp="2026-04-12T12:00:01Z",
        payload={"approved": False},
    )
    deny_frame = encode_frame(deny_reply)
    header_bytes = deny_frame[:4]
    payload_bytes = deny_frame[4:]
    client.reader.readexactly = AsyncMock(
        side_effect=[header_bytes, payload_bytes],
    )


def _setup_approve_reply(client: Any) -> None:
    """Configure the mock client to return an approve CONFIRM_REPLY."""
    from jules_daemon.ipc.framing import (
        MessageEnvelope,
        MessageType,
        encode_frame,
    )

    approve_reply = MessageEnvelope(
        msg_type=MessageType.CONFIRM_REPLY,
        msg_id="approve-retry-001",
        timestamp="2026-04-12T12:00:01Z",
        payload={"approved": True},
    )
    approve_frame = encode_frame(approve_reply)
    header_bytes = approve_frame[:4]
    payload_bytes = approve_frame[4:]
    client.reader.readexactly = AsyncMock(
        side_effect=[header_bytes, payload_bytes],
    )


class TestRetryExhaustionTriggersOneShot:
    """Verify that retry exhaustion in the agent loop triggers one-shot fallback."""

    @pytest.mark.asyncio
    async def test_retry_exhaustion_falls_back_to_oneshot(
        self, tmp_path: Path,
    ) -> None:
        """Agent loop retry exhaustion triggers one-shot fallback.

        When the agent loop exhausts all retries (transient errors persist),
        _handle_run_agent_loop raises RetryExhaustedError, and _handle_run
        catches it and falls through to _handle_run_oneshot.
        """
        from jules_daemon.ipc.request_handler import (
            RequestHandler,
            RequestHandlerConfig,
        )
        from jules_daemon.ipc.framing import MessageType

        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        # Mock _handle_run_agent_loop to raise RetryExhaustedError
        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
            side_effect=RetryExhaustedError(
                "Transient error retries exhausted",
                iterations_used=1,
            ),
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the regression suite",
            })

            response = await handler.handle_message(envelope, client)

            # Fell back to one-shot, user denied
            assert response.msg_type == MessageType.RESPONSE
            assert response.payload["status"] == "denied"

    @pytest.mark.asyncio
    async def test_retry_exhaustion_oneshot_uses_llm_translation(
        self, tmp_path: Path,
    ) -> None:
        """After retry exhaustion fallback, one-shot path uses LLM translation.

        The one-shot path should still attempt LLM translation (single-pass)
        even after the agent loop failed with retry exhaustion.
        """
        from jules_daemon.ipc.request_handler import (
            RequestHandler,
            RequestHandlerConfig,
        )

        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
            side_effect=RetryExhaustedError(
                "Retries exhausted", iterations_used=1,
            ),
        ), patch.object(
            handler,
            "_translate_via_llm",
            new_callable=AsyncMock,
            return_value="pytest -v tests/regression/",
        ) as mock_translate:
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run the regression suite",
            })

            await handler.handle_message(envelope, client)

            # Verify the one-shot LLM translation was called
            mock_translate.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_exhaustion_oneshot_approval_starts_run(
        self, tmp_path: Path,
    ) -> None:
        """After retry exhaustion, one-shot approval path starts execution.

        Full path: retry exhaustion -> one-shot fallback -> user approves
        -> SSH credentials resolved -> background task spawned -> started.
        """
        from jules_daemon.ipc.request_handler import (
            RequestHandler,
            RequestHandlerConfig,
        )
        from jules_daemon.ipc.framing import MessageType

        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_approve_reply(client)

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
            side_effect=RetryExhaustedError(
                "Retries exhausted", iterations_used=2,
            ),
        ), patch(
            "jules_daemon.ipc.request_handler.resolve_ssh_credentials",
            return_value=MagicMock(),
        ), patch(
            "jules_daemon.ipc.request_handler.check_remote_processes",
            new_callable=AsyncMock,
            return_value=[],
        ), patch(
            "jules_daemon.ipc.request_handler.execute_run",
            new_callable=AsyncMock,
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "pytest -v tests/smoke/",
            })

            response = await handler.handle_message(envelope, client)

            assert response.msg_type == MessageType.RESPONSE
            assert response.payload["status"] == "started"
            assert "run_id" in response.payload

    @pytest.mark.asyncio
    async def test_permanent_error_not_retry_exhaustion(
        self, tmp_path: Path,
    ) -> None:
        """Permanent errors from agent loop also trigger fallback but not via
        RetryExhaustedError. They are caught as generic Exception.
        """
        from jules_daemon.ipc.request_handler import (
            RequestHandler,
            RequestHandlerConfig,
        )
        from jules_daemon.ipc.framing import MessageType

        config = RequestHandlerConfig(
            wiki_root=tmp_path,
            llm_client=_make_llm_client(),
            llm_config=_make_llm_config(),
        )
        handler = RequestHandler(config=config)
        client = _make_client()
        _setup_deny_reply(client)

        with patch.object(
            handler,
            "_handle_run_agent_loop",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Auth failure"),
        ):
            envelope = _make_request(payload={
                "verb": "run",
                "target_host": "staging.example.com",
                "target_user": "deploy",
                "natural_language": "run all tests",
            })

            response = await handler.handle_message(envelope, client)

            # Still falls back to one-shot (generic exception catch)
            assert response.msg_type == MessageType.RESPONSE
            assert response.payload["status"] == "denied"


# ---------------------------------------------------------------------------
# Integration: AgentLoop retry exhaustion -> RetryExhaustedError -> one-shot
# ---------------------------------------------------------------------------


class TestEndToEndRetryExhaustionFallback:
    """End-to-end test: agent loop retry exhaustion triggers one-shot."""

    @pytest.mark.asyncio
    async def test_agent_loop_produces_retry_exhausted_result(self) -> None:
        """AgentLoop correctly produces retry_exhausted=True result.

        This verifies the full path inside the agent loop: transient error
        -> retries -> exhaustion -> AgentLoopResult with retry_exhausted.
        """
        llm = FailingLLMClient(ConnectionError("persistent network failure"))
        dispatcher = MockToolDispatcher()

        config = AgentLoopConfig(
            max_iterations=3,
            max_retries=2,
            retry_base_delay=0.0,
        )
        loop = AgentLoop(
            llm_client=llm,
            tool_dispatcher=dispatcher,
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run the smoke tests on staging")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert result.iterations_used == 1
        assert "falling back to one-shot" in result.error_message.lower()
        # 1 original + 2 retries = 3 total LLM calls
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_successful_retry_clears_exhaustion_state(self) -> None:
        """If a retry succeeds, retry_exhausted stays False."""

        call_count = 0

        class RetryThenSucceedLLM:
            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("first attempt fails")
                # Second attempt succeeds with no tool calls (completion)
                return ()

        config = AgentLoopConfig(
            max_retries=2, retry_base_delay=0.0,
        )
        loop = AgentLoop(
            llm_client=RetryThenSucceedLLM(),
            tool_dispatcher=MockToolDispatcher(),
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.COMPLETE
        assert result.retry_exhausted is False

    @pytest.mark.asyncio
    async def test_retry_exhausted_in_later_iteration(self) -> None:
        """Retry exhaustion can happen in iteration > 1."""

        call_count = 0

        class FirstSucceedThenFail:
            """Succeeds on iteration 1, then fails with transient error."""

            async def get_tool_calls(
                self, messages: tuple[dict[str, Any], ...],
            ) -> tuple[ToolCall, ...]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First iteration: return a tool call
                    return (
                        ToolCall(
                            call_id="call-1",
                            tool_name="read_wiki",
                            arguments={"page": "test"},
                        ),
                    )
                # Subsequent calls: transient error
                raise ConnectionError("network down")

        config = AgentLoopConfig(
            max_iterations=5,
            max_retries=1,
            retry_base_delay=0.0,
        )
        loop = AgentLoop(
            llm_client=FirstSucceedThenFail(),
            tool_dispatcher=MockToolDispatcher(),
            system_prompt="test",
            config=config,
            sleep_fn=_noop_sleep,
        )
        result = await loop.run("run tests")

        assert result.final_state is AgentLoopState.ERROR
        assert result.retry_exhausted is True
        assert result.iterations_used == 2  # Failed at iteration 2


# ---------------------------------------------------------------------------
# Result immutability
# ---------------------------------------------------------------------------


class TestRetryExhaustedImmutability:
    """AgentLoopResult with retry_exhausted is frozen (immutable)."""

    def test_cannot_modify_retry_exhausted(self) -> None:
        """retry_exhausted cannot be modified after construction."""
        result = AgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=1,
            history=(),
            error_message="test",
            retry_exhausted=True,
        )
        with pytest.raises(AttributeError):
            result.retry_exhausted = False  # type: ignore[misc]

    def test_explicit_true(self) -> None:
        """retry_exhausted=True can be set explicitly."""
        result = AgentLoopResult(
            final_state=AgentLoopState.ERROR,
            iterations_used=1,
            history=(),
            error_message="test",
            retry_exhausted=True,
        )
        assert result.retry_exhausted is True

    def test_explicit_false(self) -> None:
        """retry_exhausted=False can be set explicitly."""
        result = AgentLoopResult(
            final_state=AgentLoopState.COMPLETE,
            iterations_used=1,
            history=(),
            error_message=None,
            retry_exhausted=False,
        )
        assert result.retry_exhausted is False
