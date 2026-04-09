"""Tests for Dataiku Mesh LLM model routing."""

from __future__ import annotations

import pytest

from jules_daemon.llm.models import (
    ModelID,
    ToolCallingMode,
    parse_model_id,
)


class TestModelID:
    """Tests for the frozen ModelID dataclass."""

    def test_create_full_model_id(self) -> None:
        mid = ModelID(
            provider="openai",
            connection="my-openai-conn",
            model="gpt-4",
        )
        assert mid.provider == "openai"
        assert mid.connection == "my-openai-conn"
        assert mid.model == "gpt-4"

    def test_to_string(self) -> None:
        mid = ModelID(
            provider="openai",
            connection="my-openai-conn",
            model="gpt-4",
        )
        assert mid.to_string() == "openai:my-openai-conn:gpt-4"

    def test_frozen(self) -> None:
        mid = ModelID(provider="openai", connection="conn", model="gpt-4")
        with pytest.raises(AttributeError):
            mid.provider = "anthropic"  # type: ignore[misc]

    def test_empty_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="provider must not be empty"):
            ModelID(provider="", connection="conn", model="gpt-4")

    def test_empty_connection_raises(self) -> None:
        with pytest.raises(ValueError, match="connection must not be empty"):
            ModelID(provider="openai", connection="", model="gpt-4")

    def test_empty_model_raises(self) -> None:
        with pytest.raises(ValueError, match="model must not be empty"):
            ModelID(provider="openai", connection="conn", model="")


class TestParseModelID:
    """Tests for the parse_model_id factory function."""

    def test_parse_standard_format(self) -> None:
        mid = parse_model_id("openai:my-openai-conn:gpt-4")
        assert mid.provider == "openai"
        assert mid.connection == "my-openai-conn"
        assert mid.model == "gpt-4"

    def test_parse_anthropic_model(self) -> None:
        mid = parse_model_id("anthropic:my-claude:claude-3-opus")
        assert mid.provider == "anthropic"
        assert mid.connection == "my-claude"
        assert mid.model == "claude-3-opus"

    def test_parse_azure_model(self) -> None:
        mid = parse_model_id("azure-openai:eastus-conn:gpt-4-turbo")
        assert mid.provider == "azure-openai"
        assert mid.connection == "eastus-conn"
        assert mid.model == "gpt-4-turbo"

    def test_parse_model_with_dots(self) -> None:
        mid = parse_model_id("mistral:my-mistral:mistral-large-2402")
        assert mid.model == "mistral-large-2402"

    def test_roundtrip(self) -> None:
        original = "openai:production-conn:gpt-4o-mini"
        mid = parse_model_id(original)
        assert mid.to_string() == original

    def test_parse_too_few_parts_raises(self) -> None:
        with pytest.raises(ValueError, match="provider:connection:model"):
            parse_model_id("openai:gpt-4")

    def test_parse_single_part_raises(self) -> None:
        with pytest.raises(ValueError, match="provider:connection:model"):
            parse_model_id("gpt-4")

    def test_parse_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Model ID string must not be empty"):
            parse_model_id("")

    def test_parse_too_many_colons_uses_rest_as_model(self) -> None:
        """Models with colons in their name should work."""
        mid = parse_model_id("openai:conn:gpt-4:latest")
        assert mid.provider == "openai"
        assert mid.connection == "conn"
        assert mid.model == "gpt-4:latest"

    def test_parse_whitespace_stripped(self) -> None:
        mid = parse_model_id("  openai:conn:gpt-4  ")
        assert mid.provider == "openai"
        assert mid.connection == "conn"
        assert mid.model == "gpt-4"


class TestToolCallingMode:
    """Tests for ToolCallingMode enum."""

    def test_native_mode(self) -> None:
        assert ToolCallingMode.NATIVE.value == "native"

    def test_prompt_based_mode(self) -> None:
        assert ToolCallingMode.PROMPT_BASED.value == "prompt_based"
