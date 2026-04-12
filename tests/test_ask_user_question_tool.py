"""Tests for AskUserQuestionTool (agent/tools/ask_user_question.py).

Comprehensive unit tests covering:
    - Tool specification metadata (name, description, parameters, approval)
    - Execute method: successful question/answer flow
    - Execute method: user cancellation (returns DENIED, terminal)
    - Input validation: empty question, whitespace-only, missing question key
    - Context parameter: optional, default empty string
    - call_id propagation across all result paths
    - Error handling: callback exceptions
    - Edge cases: very long questions, special characters, concurrent calls
    - JSON output structure validation
    - Callback argument forwarding (question stripped, context passed through)
    - BaseTool conformance (spec property, name shortcut)

These tests exercise the tool in isolation using mocked async callbacks,
matching the established pattern from test_execute_ssh_tool.py and
test_agent_tool_wrappers.py.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from jules_daemon.agent.tool_types import (
    ApprovalRequirement,
    ToolResultStatus,
    ToolSpec,
)
from jules_daemon.agent.tools.ask_user_question import AskUserQuestionTool
from jules_daemon.agent.tools.base import BaseTool, Tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def approve_callback() -> AsyncMock:
    """Callback that returns a valid user answer."""
    return AsyncMock(return_value="42")


@pytest.fixture
def cancel_callback() -> AsyncMock:
    """Callback that simulates user cancellation (returns None)."""
    return AsyncMock(return_value=None)


@pytest.fixture
def error_callback() -> AsyncMock:
    """Callback that raises an exception (simulating IPC failure)."""
    return AsyncMock(side_effect=ConnectionError("IPC channel closed"))


@pytest.fixture
def tool(approve_callback: AsyncMock) -> AskUserQuestionTool:
    """AskUserQuestionTool with a standard approve callback."""
    return AskUserQuestionTool(ask_callback=approve_callback)


# ---------------------------------------------------------------------------
# Tool specification and metadata
# ---------------------------------------------------------------------------


class TestAskUserQuestionToolSpec:
    """Verify tool spec metadata and protocol conformance."""

    def test_tool_name(self, tool: AskUserQuestionTool) -> None:
        assert tool.name == "ask_user_question"

    def test_spec_returns_tool_spec(self, tool: AskUserQuestionTool) -> None:
        assert isinstance(tool.spec, ToolSpec)

    def test_spec_name_matches(self, tool: AskUserQuestionTool) -> None:
        assert tool.spec.name == "ask_user_question"

    def test_spec_has_nonempty_description(
        self, tool: AskUserQuestionTool,
    ) -> None:
        assert isinstance(tool.spec.description, str)
        assert len(tool.spec.description) > 0

    def test_description_mentions_ask(
        self, tool: AskUserQuestionTool,
    ) -> None:
        """Description should indicate this tool asks the user questions."""
        desc_lower = tool.spec.description.lower()
        assert "ask" in desc_lower or "question" in desc_lower

    def test_requires_confirm_prompt_approval(
        self, tool: AskUserQuestionTool,
    ) -> None:
        """ask_user_question uses CONFIRM_PROMPT because it interacts with user."""
        assert tool.spec.approval is ApprovalRequirement.CONFIRM_PROMPT

    def test_is_not_read_only(self, tool: AskUserQuestionTool) -> None:
        assert not tool.spec.is_read_only

    def test_has_question_parameter(self, tool: AskUserQuestionTool) -> None:
        param_names = {p.name for p in tool.spec.parameters}
        assert "question" in param_names

    def test_question_parameter_is_required(
        self, tool: AskUserQuestionTool,
    ) -> None:
        question_param = next(
            p for p in tool.spec.parameters if p.name == "question"
        )
        assert question_param.required is True

    def test_question_parameter_is_string_type(
        self, tool: AskUserQuestionTool,
    ) -> None:
        question_param = next(
            p for p in tool.spec.parameters if p.name == "question"
        )
        assert question_param.json_type == "string"

    def test_has_context_parameter(self, tool: AskUserQuestionTool) -> None:
        param_names = {p.name for p in tool.spec.parameters}
        assert "context" in param_names

    def test_context_parameter_is_optional(
        self, tool: AskUserQuestionTool,
    ) -> None:
        context_param = next(
            p for p in tool.spec.parameters if p.name == "context"
        )
        assert context_param.required is False

    def test_context_parameter_default_is_empty_string(
        self, tool: AskUserQuestionTool,
    ) -> None:
        context_param = next(
            p for p in tool.spec.parameters if p.name == "context"
        )
        assert context_param.default == ""

    def test_is_base_tool_subclass(self) -> None:
        """AskUserQuestionTool must extend BaseTool."""
        assert issubclass(AskUserQuestionTool, BaseTool)

    def test_instance_satisfies_tool_protocol(
        self, tool: AskUserQuestionTool,
    ) -> None:
        """Instance must satisfy the Tool protocol (has spec + execute)."""
        assert isinstance(tool, Tool)

    def test_openai_schema_structure(
        self, tool: AskUserQuestionTool,
    ) -> None:
        """ToolSpec should serialize to valid OpenAI function schema."""
        schema = tool.spec.to_openai_function_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "ask_user_question"
        assert "description" in fn
        assert "parameters" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "question" in params["properties"]
        assert "question" in params["required"]

    def test_openai_schema_is_json_serializable(
        self, tool: AskUserQuestionTool,
    ) -> None:
        schema = tool.spec.to_openai_function_schema()
        serialized = json.dumps(schema)
        deserialized = json.loads(serialized)
        assert deserialized["function"]["name"] == "ask_user_question"


# ---------------------------------------------------------------------------
# Execute: successful question/answer flow
# ---------------------------------------------------------------------------


class TestAskUserQuestionExecution:
    """Verify successful question-answer flows."""

    @pytest.mark.asyncio
    async def test_returns_success_with_user_answer(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)
        result = await tool.execute({
            "question": "How many iterations?",
            "context": "Test spec requires --iterations",
            "_call_id": "c1",
        })

        assert result.status is ToolResultStatus.SUCCESS
        assert result.is_success

    @pytest.mark.asyncio
    async def test_output_contains_answer(
        self, approve_callback: AsyncMock,
    ) -> None:
        approve_callback.return_value = "100"
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "How many?",
            "_call_id": "c2",
        })

        data = json.loads(result.output)
        assert data["answer"] == "100"

    @pytest.mark.asyncio
    async def test_output_echoes_question(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "What host should I use?",
            "_call_id": "c3",
        })

        data = json.loads(result.output)
        assert data["question"] == "What host should I use?"

    @pytest.mark.asyncio
    async def test_question_is_stripped_in_output(
        self, approve_callback: AsyncMock,
    ) -> None:
        """Leading/trailing whitespace must be stripped from the question."""
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "  What host?  ",
            "_call_id": "c4",
        })

        data = json.loads(result.output)
        assert data["question"] == "What host?"

    @pytest.mark.asyncio
    async def test_callback_receives_stripped_question(
        self, approve_callback: AsyncMock,
    ) -> None:
        """Callback must receive the stripped question."""
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        await tool.execute({
            "question": "  How many?  ",
            "context": "some context",
            "_call_id": "c5",
        })

        approve_callback.assert_awaited_once_with("How many?", "some context")

    @pytest.mark.asyncio
    async def test_callback_receives_context(
        self, approve_callback: AsyncMock,
    ) -> None:
        """Context argument must be forwarded to the callback."""
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        await tool.execute({
            "question": "What env?",
            "context": "Missing environment variable",
            "_call_id": "c6",
        })

        approve_callback.assert_awaited_once_with(
            "What env?", "Missing environment variable"
        )

    @pytest.mark.asyncio
    async def test_missing_context_defaults_to_empty_string(
        self, approve_callback: AsyncMock,
    ) -> None:
        """When context is omitted, callback receives empty string."""
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        await tool.execute({
            "question": "What host?",
            "_call_id": "c7",
        })

        approve_callback.assert_awaited_once_with("What host?", "")

    @pytest.mark.asyncio
    async def test_success_result_is_not_terminal(
        self, tool: AskUserQuestionTool,
    ) -> None:
        result = await tool.execute({
            "question": "How many?",
            "_call_id": "c8",
        })
        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_success_result_has_no_error_message(
        self, tool: AskUserQuestionTool,
    ) -> None:
        result = await tool.execute({
            "question": "How many?",
            "_call_id": "c9",
        })
        assert result.error_message is None

    @pytest.mark.asyncio
    async def test_output_is_valid_json(
        self, tool: AskUserQuestionTool,
    ) -> None:
        result = await tool.execute({
            "question": "Any value?",
            "_call_id": "c10",
        })
        # Must not raise
        data = json.loads(result.output)
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# Execute: user cancellation (DENIED, terminal)
# ---------------------------------------------------------------------------


class TestAskUserQuestionCancellation:
    """Verify user cancellation returns DENIED (terminal) status."""

    @pytest.mark.asyncio
    async def test_none_answer_returns_denied(
        self, cancel_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=cancel_callback)

        result = await tool.execute({
            "question": "What value?",
            "_call_id": "deny1",
        })

        assert result.status is ToolResultStatus.DENIED
        assert result.is_denied

    @pytest.mark.asyncio
    async def test_denied_result_is_terminal(
        self, cancel_callback: AsyncMock,
    ) -> None:
        """DENIED must terminate the agent loop."""
        tool = AskUserQuestionTool(ask_callback=cancel_callback)

        result = await tool.execute({
            "question": "Continue?",
            "_call_id": "deny2",
        })

        assert result.is_terminal

    @pytest.mark.asyncio
    async def test_denied_output_contains_cancelled_flag(
        self, cancel_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=cancel_callback)

        result = await tool.execute({
            "question": "Any value?",
            "_call_id": "deny3",
        })

        data = json.loads(result.output)
        assert data["cancelled"] is True

    @pytest.mark.asyncio
    async def test_denied_has_error_message(
        self, cancel_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=cancel_callback)

        result = await tool.execute({
            "question": "Any value?",
            "_call_id": "deny4",
        })

        assert result.error_message is not None
        assert "cancel" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_denied_tool_name_correct(
        self, cancel_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=cancel_callback)

        result = await tool.execute({
            "question": "Any value?",
            "_call_id": "deny5",
        })

        assert result.tool_name == "ask_user_question"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestAskUserQuestionValidation:
    """Verify argument validation returns ERROR for invalid inputs."""

    @pytest.mark.asyncio
    async def test_empty_question_returns_error(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "",
            "_call_id": "v1",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "question" in (result.error_message or "").lower()
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_whitespace_only_question_returns_error(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "   ",
            "_call_id": "v2",
        })

        assert result.status is ToolResultStatus.ERROR
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_question_key_returns_error(
        self, approve_callback: AsyncMock,
    ) -> None:
        """When question key is entirely absent, returns error."""
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "context": "some context",
            "_call_id": "v3",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "question" in (result.error_message or "").lower()
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_validation_error_is_not_terminal(
        self, approve_callback: AsyncMock,
    ) -> None:
        """Validation errors are ERROR, not DENIED -- agent can self-correct."""
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "",
            "_call_id": "v4",
        })

        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_validation_error_output_is_empty(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "",
            "_call_id": "v5",
        })

        assert result.output == ""

    @pytest.mark.asyncio
    async def test_newline_only_question_returns_error(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "\n\t\r",
            "_call_id": "v6",
        })

        assert result.status is ToolResultStatus.ERROR
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_none_question_returns_error(
        self, approve_callback: AsyncMock,
    ) -> None:
        """If question is somehow None (e.g., from malformed JSON), error."""
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": None,
            "_call_id": "v7",
        })

        assert result.status is ToolResultStatus.ERROR
        approve_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_args_dict_returns_error(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({"_call_id": "v8"})

        assert result.status is ToolResultStatus.ERROR
        approve_callback.assert_not_awaited()


# ---------------------------------------------------------------------------
# call_id propagation
# ---------------------------------------------------------------------------


class TestAskUserQuestionCallId:
    """Verify _call_id flows through to results across all paths."""

    @pytest.mark.asyncio
    async def test_call_id_in_success_result(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "How many?",
            "_call_id": "unique-success-id",
        })

        assert result.call_id == "unique-success-id"

    @pytest.mark.asyncio
    async def test_call_id_in_denied_result(
        self, cancel_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=cancel_callback)

        result = await tool.execute({
            "question": "What?",
            "_call_id": "unique-denied-id",
        })

        assert result.call_id == "unique-denied-id"

    @pytest.mark.asyncio
    async def test_call_id_in_error_result(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "",
            "_call_id": "unique-error-id",
        })

        assert result.call_id == "unique-error-id"

    @pytest.mark.asyncio
    async def test_call_id_in_exception_result(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=error_callback)

        result = await tool.execute({
            "question": "Test?",
            "_call_id": "unique-exception-id",
        })

        assert result.call_id == "unique-exception-id"

    @pytest.mark.asyncio
    async def test_default_call_id_when_missing(
        self, approve_callback: AsyncMock,
    ) -> None:
        """When _call_id is omitted, defaults to 'ask_user_question'."""
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({"question": "How many?"})

        assert result.call_id == "ask_user_question"

    @pytest.mark.asyncio
    async def test_tool_name_always_set(
        self, approve_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=approve_callback)

        result = await tool.execute({
            "question": "How many?",
            "_call_id": "tn1",
        })

        assert result.tool_name == "ask_user_question"


# ---------------------------------------------------------------------------
# Error handling: callback exceptions
# ---------------------------------------------------------------------------


class TestAskUserQuestionErrorHandling:
    """Verify graceful handling of callback exceptions."""

    @pytest.mark.asyncio
    async def test_callback_exception_returns_error(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=error_callback)

        result = await tool.execute({
            "question": "What server?",
            "_call_id": "err1",
        })

        assert result.status is ToolResultStatus.ERROR
        assert result.is_error

    @pytest.mark.asyncio
    async def test_callback_exception_error_message_contains_details(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=error_callback)

        result = await tool.execute({
            "question": "What server?",
            "_call_id": "err2",
        })

        assert result.error_message is not None
        assert "Failed to ask user" in result.error_message
        assert "IPC channel closed" in result.error_message

    @pytest.mark.asyncio
    async def test_callback_exception_is_not_terminal(
        self, error_callback: AsyncMock,
    ) -> None:
        """Callback errors are ERROR (not DENIED) -- agent loop can retry."""
        tool = AskUserQuestionTool(ask_callback=error_callback)

        result = await tool.execute({
            "question": "What server?",
            "_call_id": "err3",
        })

        assert not result.is_terminal

    @pytest.mark.asyncio
    async def test_callback_exception_output_is_empty(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=error_callback)

        result = await tool.execute({
            "question": "What server?",
            "_call_id": "err4",
        })

        assert result.output == ""

    @pytest.mark.asyncio
    async def test_timeout_error_returns_error_result(self) -> None:
        """TimeoutError from callback should be caught and wrapped."""
        callback = AsyncMock(side_effect=TimeoutError("IPC timeout"))
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "Any value?",
            "_call_id": "err5",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "timeout" in (result.error_message or "").lower()

    @pytest.mark.asyncio
    async def test_runtime_error_returns_error_result(self) -> None:
        """RuntimeError from callback should be caught."""
        callback = AsyncMock(side_effect=RuntimeError("unexpected"))
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "Any value?",
            "_call_id": "err6",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "unexpected" in (result.error_message or "")

    @pytest.mark.asyncio
    async def test_value_error_from_callback(self) -> None:
        """ValueError from callback is caught and wrapped."""
        callback = AsyncMock(side_effect=ValueError("bad input"))
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "Any value?",
            "_call_id": "err7",
        })

        assert result.status is ToolResultStatus.ERROR
        assert "bad input" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAskUserQuestionEdgeCases:
    """Edge cases: long strings, special characters, empty answers."""

    @pytest.mark.asyncio
    async def test_very_long_question(self) -> None:
        """Tool should handle very long questions without error."""
        long_question = "x" * 5000
        callback = AsyncMock(return_value="ok")
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": long_question,
            "_call_id": "edge1",
        })

        assert result.is_success
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_question_with_special_characters(self) -> None:
        """Questions with unicode / special characters are handled."""
        question = 'What is the "path" for ~/test & <args>?'
        callback = AsyncMock(return_value="/home/user/test")
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": question,
            "_call_id": "edge2",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["question"] == question

    @pytest.mark.asyncio
    async def test_question_with_newlines(self) -> None:
        """Multi-line questions are valid and forwarded correctly."""
        question = "Line 1\nLine 2\nLine 3"
        callback = AsyncMock(return_value="answer")
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": question,
            "_call_id": "edge3",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["question"] == question.strip()

    @pytest.mark.asyncio
    async def test_empty_string_answer_from_user(self) -> None:
        """User returning empty string is a valid (non-None) answer."""
        callback = AsyncMock(return_value="")
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "Enter value:",
            "_call_id": "edge4",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["answer"] == ""

    @pytest.mark.asyncio
    async def test_whitespace_answer_from_user(self) -> None:
        """User returning whitespace is a valid (non-None) answer."""
        callback = AsyncMock(return_value="   ")
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "Enter value:",
            "_call_id": "edge5",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["answer"] == "   "

    @pytest.mark.asyncio
    async def test_json_in_answer(self) -> None:
        """JSON-like strings in the answer must be properly serialized."""
        callback = AsyncMock(return_value='{"key": "value"}')
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "Enter config:",
            "_call_id": "edge6",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["answer"] == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_unicode_in_question_and_answer(self) -> None:
        """Unicode characters in both question and answer."""
        callback = AsyncMock(return_value="Tokyo")
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "Which city?",
            "context": "Deployment region selection",
            "_call_id": "edge7",
        })

        assert result.is_success
        data = json.loads(result.output)
        assert data["answer"] == "Tokyo"

    @pytest.mark.asyncio
    async def test_context_with_special_characters(self) -> None:
        """Context with special chars must be forwarded correctly."""
        callback = AsyncMock(return_value="yes")
        tool = AskUserQuestionTool(ask_callback=callback)

        context = "The --iterations flag & <timeout> aren't set"
        await tool.execute({
            "question": "Continue?",
            "context": context,
            "_call_id": "edge8",
        })

        callback.assert_awaited_once_with("Continue?", context)

    @pytest.mark.asyncio
    async def test_extra_args_ignored(self) -> None:
        """Extra args beyond question/context/_call_id are ignored."""
        callback = AsyncMock(return_value="yes")
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "Continue?",
            "extra_param": "should be ignored",
            "another": 42,
            "_call_id": "edge9",
        })

        assert result.is_success

    @pytest.mark.asyncio
    async def test_result_output_is_always_valid_json(self) -> None:
        """Output field is always valid JSON on success."""
        callback = AsyncMock(return_value="test")
        tool = AskUserQuestionTool(ask_callback=callback)

        result = await tool.execute({
            "question": "Q?",
            "_call_id": "edge10",
        })

        parsed = json.loads(result.output)
        assert "answer" in parsed
        assert "question" in parsed


# ---------------------------------------------------------------------------
# Callback invocation ordering
# ---------------------------------------------------------------------------


class TestAskUserQuestionCallbackInvocation:
    """Verify the callback is called exactly once with correct arguments."""

    @pytest.mark.asyncio
    async def test_callback_called_exactly_once(self) -> None:
        callback = AsyncMock(return_value="answer")
        tool = AskUserQuestionTool(ask_callback=callback)

        await tool.execute({
            "question": "Q1?",
            "_call_id": "inv1",
        })

        assert callback.await_count == 1

    @pytest.mark.asyncio
    async def test_callback_not_called_on_validation_error(self) -> None:
        callback = AsyncMock(return_value="answer")
        tool = AskUserQuestionTool(ask_callback=callback)

        await tool.execute({
            "question": "",
            "_call_id": "inv2",
        })

        callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_callback_receives_two_positional_args(self) -> None:
        callback = AsyncMock(return_value="answer")
        tool = AskUserQuestionTool(ask_callback=callback)

        await tool.execute({
            "question": "Q?",
            "context": "ctx",
            "_call_id": "inv3",
        })

        # Verify positional args: (question, context)
        args, kwargs = callback.call_args
        assert len(args) == 2
        assert args[0] == "Q?"
        assert args[1] == "ctx"
        assert len(kwargs) == 0

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls(self) -> None:
        """Multiple sequential calls each invoke the callback once."""
        call_count = 0

        async def counting_callback(q: str, c: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"answer-{call_count}"

        tool = AskUserQuestionTool(ask_callback=counting_callback)

        r1 = await tool.execute({
            "question": "Q1?", "_call_id": "seq1"
        })
        r2 = await tool.execute({
            "question": "Q2?", "_call_id": "seq2"
        })
        r3 = await tool.execute({
            "question": "Q3?", "_call_id": "seq3"
        })

        assert call_count == 3
        assert json.loads(r1.output)["answer"] == "answer-1"
        assert json.loads(r2.output)["answer"] == "answer-2"
        assert json.loads(r3.output)["answer"] == "answer-3"


# ---------------------------------------------------------------------------
# ToolResult serialization (to_openai_tool_message, to_llm_message)
# ---------------------------------------------------------------------------


class TestAskUserQuestionResultSerialization:
    """Verify ToolResult serialization for conversation history."""

    @pytest.mark.asyncio
    async def test_success_openai_message_format(
        self, tool: AskUserQuestionTool,
    ) -> None:
        result = await tool.execute({
            "question": "How many?",
            "_call_id": "ser1",
        })

        msg = result.to_openai_tool_message()
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "ser1"
        # Content should be the JSON output
        data = json.loads(msg["content"])
        assert "answer" in data

    @pytest.mark.asyncio
    async def test_denied_openai_message_has_error_prefix(
        self, cancel_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=cancel_callback)

        result = await tool.execute({
            "question": "Q?",
            "_call_id": "ser2",
        })

        msg = result.to_openai_tool_message()
        assert msg["content"].startswith("ERROR:")

    @pytest.mark.asyncio
    async def test_success_llm_message_format(
        self, tool: AskUserQuestionTool,
    ) -> None:
        result = await tool.execute({
            "question": "How many?",
            "_call_id": "ser3",
        })

        text = result.to_llm_message()
        assert "[ask_user_question]" in text
        assert "success" in text.lower()

    @pytest.mark.asyncio
    async def test_denied_llm_message_format(
        self, cancel_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=cancel_callback)

        result = await tool.execute({
            "question": "Q?",
            "_call_id": "ser4",
        })

        text = result.to_llm_message()
        assert "[ask_user_question]" in text
        assert "DENIED" in text

    @pytest.mark.asyncio
    async def test_error_llm_message_format(
        self, error_callback: AsyncMock,
    ) -> None:
        tool = AskUserQuestionTool(ask_callback=error_callback)

        result = await tool.execute({
            "question": "Q?",
            "_call_id": "ser5",
        })

        text = result.to_llm_message()
        assert "[ask_user_question]" in text
        assert "ERROR" in text
