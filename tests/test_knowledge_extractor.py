"""Tests for the LLM-driven knowledge extractor.

The extractor must be fail-soft: any error (no client, network failure,
malformed JSON, timeout) should produce ``None`` rather than raising,
because the audit/run flow must never break.
"""

from __future__ import annotations

from typing import Any

import pytest

from jules_daemon.execution.knowledge_extractor import (
    KNOWLEDGE_EXTRACTOR_TIMEOUT_SECONDS,
    extract_knowledge,
)
from jules_daemon.wiki.test_knowledge import TestKnowledge


# ---------------------------------------------------------------------------
# Lightweight fake OpenAI client (mirrors the pattern in test_output_summarizer)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [
            type("_Choice", (), {
                "message": type("_Message", (), {"content": content})()
            })()
        ]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


def _make_knowledge(
    *,
    purpose: str = "runs the agent",
    output_format: str = "iteration logs",
    normal_behavior: str = "all iterations PASSED",
    common_failures: tuple[str, ...] = (),
    runs_observed: int = 1,
) -> TestKnowledge:
    return TestKnowledge(
        test_slug="agent-test-py",
        command_pattern="python3 agent_test.py",
        purpose=purpose,
        output_format=output_format,
        normal_behavior=normal_behavior,
        common_failures=common_failures,
        runs_observed=runs_observed,
    )


# ---------------------------------------------------------------------------
# Disabled / no-op paths
# ---------------------------------------------------------------------------


class TestExtractKnowledgeDisabled:
    @pytest.mark.asyncio
    async def test_no_client_returns_none(self) -> None:
        result = await extract_knowledge(
            command="pytest",
            stdout="ok",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=None,
            llm_model=None,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_client_without_model_returns_none(self) -> None:
        client = _FakeClient('{"purpose": "x"}')
        result = await extract_knowledge(
            command="pytest",
            stdout="ok",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=client,
            llm_model=None,
        )
        assert result is None
        # The client was never called
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_empty_command_returns_none(self) -> None:
        client = _FakeClient('{"purpose": "x"}')
        result = await extract_knowledge(
            command="",
            stdout="ok",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=client,
            llm_model="m",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_whitespace_command_returns_none(self) -> None:
        client = _FakeClient('{"purpose": "x"}')
        result = await extract_knowledge(
            command="   ",
            stdout="ok",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=client,
            llm_model="m",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Successful extraction
# ---------------------------------------------------------------------------


class TestExtractKnowledgeSuccess:
    @pytest.mark.asyncio
    async def test_returns_normalized_dict(self) -> None:
        llm_response = (
            '{"purpose": "runs the suite", '
            '"output_format": "iteration logs", '
            '"common_failures": ["timeout", "ConnError"], '
            '"normal_behavior": "all iterations pass"}'
        )
        client = _FakeClient(llm_response)
        result = await extract_knowledge(
            command="python3 agent_test.py",
            stdout="iteration 1: PASSED",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=client,
            llm_model="openai:default:gpt-4o",
        )
        assert result is not None
        assert result["purpose"] == "runs the suite"
        assert result["output_format"] == "iteration logs"
        assert result["common_failures"] == ["timeout", "ConnError"]
        assert result["normal_behavior"] == "all iterations pass"

    @pytest.mark.asyncio
    async def test_includes_existing_knowledge_in_prompt(self) -> None:
        llm_response = (
            '{"purpose": "", "output_format": "", '
            '"common_failures": [], "normal_behavior": ""}'
        )
        client = _FakeClient(llm_response)
        existing = _make_knowledge(purpose="existing")
        await extract_knowledge(
            command="python3 agent_test.py",
            stdout="ok",
            stderr="",
            exit_code=0,
            existing_knowledge=existing,
            llm_client=client,
            llm_model="m",
        )
        # The existing knowledge text must be embedded in the user
        # message so the LLM can avoid restating it.
        assert client.calls
        messages = client.calls[0]["messages"]
        user_msg = next(m for m in messages if m["role"] == "user")
        assert "existing" in user_msg["content"]
        assert "Prior knowledge" in user_msg["content"] or "prior" in user_msg["content"].lower()

    @pytest.mark.asyncio
    async def test_no_existing_knowledge_renders_placeholder(self) -> None:
        llm_response = (
            '{"purpose": "x", "output_format": "y", '
            '"common_failures": [], "normal_behavior": "z"}'
        )
        client = _FakeClient(llm_response)
        await extract_knowledge(
            command="python3 agent_test.py",
            stdout="ok",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=client,
            llm_model="m",
        )
        assert client.calls
        user_msg = next(
            m for m in client.calls[0]["messages"] if m["role"] == "user"
        )
        assert "no prior knowledge" in user_msg["content"].lower()

    @pytest.mark.asyncio
    async def test_fenced_json_response(self) -> None:
        llm_response = (
            "Here is the analysis:\n"
            "```json\n"
            '{"purpose": "p", "output_format": "f", '
            '"common_failures": [], "normal_behavior": "n"}\n'
            "```\n"
        )
        client = _FakeClient(llm_response)
        result = await extract_knowledge(
            command="pytest",
            stdout="ok",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=client,
            llm_model="m",
        )
        assert result is not None
        assert result["purpose"] == "p"
        assert result["output_format"] == "f"


# ---------------------------------------------------------------------------
# Failure / coercion paths
# ---------------------------------------------------------------------------


class TestExtractKnowledgeFailures:
    @pytest.mark.asyncio
    async def test_malformed_json_returns_none(self) -> None:
        client = _FakeClient("not json at all")
        result = await extract_knowledge(
            command="pytest",
            stdout="",
            stderr="",
            exit_code=1,
            existing_knowledge=None,
            llm_client=client,
            llm_model="m",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_llm_exception_returns_none(self) -> None:
        class _BrokenClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**_kwargs: Any) -> None:
                        raise RuntimeError("network failure")

        result = await extract_knowledge(
            command="pytest",
            stdout="",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=_BrokenClient(),
            llm_model="m",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _SlowClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**_kwargs: Any) -> None:
                        import time

                        time.sleep(0.5)

        monkeypatch.setattr(
            "jules_daemon.execution.knowledge_extractor."
            "KNOWLEDGE_EXTRACTOR_TIMEOUT_SECONDS",
            0.05,
        )
        result = await extract_knowledge(
            command="pytest",
            stdout="",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=_SlowClient(),
            llm_model="m",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_long_field_is_truncated(self) -> None:
        import json as _json

        huge = "x" * 10_000
        llm_response = _json.dumps(
            {
                "purpose": huge,
                "output_format": "",
                "common_failures": [],
                "normal_behavior": "",
            }
        )
        client = _FakeClient(llm_response)
        result = await extract_knowledge(
            command="pytest",
            stdout="",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=client,
            llm_model="m",
        )
        assert result is not None
        assert len(result["purpose"]) <= 600

    @pytest.mark.asyncio
    async def test_failures_capped_per_extraction(self) -> None:
        many_failures = [f"failure-{i}" for i in range(20)]
        import json as _json

        llm_response = _json.dumps(
            {
                "purpose": "",
                "output_format": "",
                "normal_behavior": "",
                "common_failures": many_failures,
            }
        )
        client = _FakeClient(llm_response)
        result = await extract_knowledge(
            command="pytest",
            stdout="",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=client,
            llm_model="m",
        )
        assert result is not None
        assert len(result["common_failures"]) <= 5

    @pytest.mark.asyncio
    async def test_non_string_fields_normalized(self) -> None:
        llm_response = (
            '{"purpose": 123, "output_format": null, '
            '"common_failures": "not-a-list", "normal_behavior": [1,2]}'
        )
        client = _FakeClient(llm_response)
        result = await extract_knowledge(
            command="pytest",
            stdout="",
            stderr="",
            exit_code=0,
            existing_knowledge=None,
            llm_client=client,
            llm_model="m",
        )
        assert result is not None
        # Non-string fields are dropped to empty
        assert result["purpose"] == ""
        assert result["output_format"] == ""
        assert result["normal_behavior"] == ""
        assert result["common_failures"] == []


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_timeout_constant_is_positive() -> None:
    assert KNOWLEDGE_EXTRACTOR_TIMEOUT_SECONDS > 0
