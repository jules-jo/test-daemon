"""Tests for prompt construction module.

Tests the system prompt builder (SSH constraints + output schema)
and user prompt builder (natural-language input + host context)
for the LLM call that translates NL requests into shell commands.
"""

from __future__ import annotations

import json

import pytest

from jules_daemon.llm.prompts import (
    HostContext,
    PromptConfig,
    build_messages,
    build_system_prompt,
    build_user_prompt,
)


# -- HostContext tests --


class TestHostContext:
    """Tests for the frozen HostContext dataclass."""

    def test_create_minimal(self) -> None:
        ctx = HostContext(hostname="staging.example.com", user="deploy")
        assert ctx.hostname == "staging.example.com"
        assert ctx.user == "deploy"
        assert ctx.port == 22
        assert ctx.working_directory is None
        assert ctx.os_hint is None
        assert ctx.shell_hint is None
        assert ctx.test_framework_hint is None
        assert ctx.extra_context == ()

    def test_create_full(self) -> None:
        ctx = HostContext(
            hostname="ci.internal.net",
            user="runner",
            port=2222,
            working_directory="/opt/app",
            os_hint="Ubuntu 22.04",
            shell_hint="bash",
            test_framework_hint="pytest",
            extra_context=("Python 3.12", "Django 5.0"),
        )
        assert ctx.port == 2222
        assert ctx.working_directory == "/opt/app"
        assert ctx.os_hint == "Ubuntu 22.04"
        assert ctx.shell_hint == "bash"
        assert ctx.test_framework_hint == "pytest"
        assert ctx.extra_context == ("Python 3.12", "Django 5.0")

    def test_frozen(self) -> None:
        ctx = HostContext(hostname="h", user="u")
        with pytest.raises(AttributeError):
            ctx.hostname = "other"  # type: ignore[misc]

    def test_empty_hostname_raises(self) -> None:
        with pytest.raises(ValueError, match="hostname must not be empty"):
            HostContext(hostname="", user="deploy")

    def test_empty_user_raises(self) -> None:
        with pytest.raises(ValueError, match="user must not be empty"):
            HostContext(hostname="host", user="")

    def test_invalid_port_low(self) -> None:
        with pytest.raises(ValueError, match="port must be 1-65535"):
            HostContext(hostname="host", user="user", port=0)

    def test_invalid_port_high(self) -> None:
        with pytest.raises(ValueError, match="port must be 1-65535"):
            HostContext(hostname="host", user="user", port=70000)

    def test_extra_context_tuple_immutable(self) -> None:
        """extra_context is a tuple, not a list, ensuring immutability."""
        ctx = HostContext(
            hostname="h",
            user="u",
            extra_context=("a", "b"),
        )
        assert isinstance(ctx.extra_context, tuple)


# -- PromptConfig tests --


class TestPromptConfig:
    """Tests for the frozen PromptConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = PromptConfig()
        assert cfg.max_commands == 5
        assert len(cfg.forbidden_patterns) > 0
        assert len(cfg.allowed_actions) > 0
        assert cfg.require_human_approval is True

    def test_custom_values(self) -> None:
        cfg = PromptConfig(
            max_commands=3,
            forbidden_patterns=("rm -rf /",),
            allowed_actions=("run tests",),
            require_human_approval=True,
        )
        assert cfg.max_commands == 3
        assert cfg.forbidden_patterns == ("rm -rf /",)
        assert cfg.allowed_actions == ("run tests",)

    def test_frozen(self) -> None:
        cfg = PromptConfig()
        with pytest.raises(AttributeError):
            cfg.max_commands = 10  # type: ignore[misc]

    def test_max_commands_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_commands must be positive"):
            PromptConfig(max_commands=0)


# -- build_system_prompt tests --


class TestBuildSystemPrompt:
    """Tests for system prompt construction."""

    def test_returns_string(self) -> None:
        prompt = build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_role_definition(self) -> None:
        prompt = build_system_prompt()
        assert "test execution assistant" in prompt.lower()

    def test_contains_ssh_constraints(self) -> None:
        prompt = build_system_prompt()
        lower = prompt.lower()
        assert "forbidden" in lower or "must not" in lower or "never" in lower

    def test_contains_output_schema(self) -> None:
        """The system prompt must define the expected JSON output schema."""
        prompt = build_system_prompt()
        assert "commands" in prompt
        assert "explanation" in prompt

    def test_contains_approval_requirement(self) -> None:
        prompt = build_system_prompt()
        lower = prompt.lower()
        assert "human approval" in lower or "human review" in lower

    def test_default_forbidden_patterns_present(self) -> None:
        prompt = build_system_prompt()
        assert "rm -rf /" in prompt

    def test_custom_config_overrides_forbidden(self) -> None:
        cfg = PromptConfig(
            forbidden_patterns=("custom_bad_cmd",),
        )
        prompt = build_system_prompt(config=cfg)
        assert "custom_bad_cmd" in prompt
        # The default "rm -rf /" should not appear when overridden
        assert "rm -rf /" not in prompt

    def test_custom_max_commands(self) -> None:
        cfg = PromptConfig(max_commands=2)
        prompt = build_system_prompt(config=cfg)
        assert "2" in prompt

    def test_approval_language_absent_when_disabled(self) -> None:
        cfg = PromptConfig(require_human_approval=False)
        prompt = build_system_prompt(config=cfg)
        assert "human approval" not in prompt.lower()
        assert "CRITICAL" not in prompt

    def test_output_schema_is_valid_json(self) -> None:
        """The prompt must contain a parseable JSON schema example."""
        prompt = build_system_prompt()
        # Find the JSON block in the prompt
        start = prompt.find("```json")
        end = prompt.find("```", start + 7)
        assert start != -1, "System prompt must contain a ```json block"
        assert end != -1, "JSON block must be closed"
        json_str = prompt[start + 7 : end].strip()
        parsed = json.loads(json_str)
        assert "commands" in parsed
        assert "explanation" in parsed


# -- build_user_prompt tests --


class TestBuildUserPrompt:
    """Tests for user prompt construction."""

    def test_includes_natural_language(self) -> None:
        ctx = HostContext(hostname="staging", user="deploy")
        prompt = build_user_prompt(
            natural_language="run the smoke tests",
            host_context=ctx,
        )
        assert "run the smoke tests" in prompt

    def test_includes_hostname(self) -> None:
        ctx = HostContext(hostname="staging.example.com", user="deploy")
        prompt = build_user_prompt(
            natural_language="run tests",
            host_context=ctx,
        )
        assert "staging.example.com" in prompt

    def test_includes_user(self) -> None:
        ctx = HostContext(hostname="host", user="testrunner")
        prompt = build_user_prompt(
            natural_language="run tests",
            host_context=ctx,
        )
        assert "testrunner" in prompt

    def test_includes_port_when_non_default(self) -> None:
        ctx = HostContext(hostname="host", user="user", port=2222)
        prompt = build_user_prompt(
            natural_language="run tests",
            host_context=ctx,
        )
        assert "2222" in prompt

    def test_includes_working_directory(self) -> None:
        ctx = HostContext(
            hostname="host",
            user="user",
            working_directory="/opt/myapp",
        )
        prompt = build_user_prompt(
            natural_language="run tests",
            host_context=ctx,
        )
        assert "/opt/myapp" in prompt

    def test_includes_os_hint(self) -> None:
        ctx = HostContext(
            hostname="host",
            user="user",
            os_hint="Ubuntu 22.04",
        )
        prompt = build_user_prompt(
            natural_language="run tests",
            host_context=ctx,
        )
        assert "Ubuntu 22.04" in prompt

    def test_includes_shell_hint(self) -> None:
        ctx = HostContext(
            hostname="host",
            user="user",
            shell_hint="zsh",
        )
        prompt = build_user_prompt(
            natural_language="run tests",
            host_context=ctx,
        )
        assert "zsh" in prompt

    def test_includes_test_framework_hint(self) -> None:
        ctx = HostContext(
            hostname="host",
            user="user",
            test_framework_hint="pytest",
        )
        prompt = build_user_prompt(
            natural_language="run tests",
            host_context=ctx,
        )
        assert "pytest" in prompt

    def test_includes_extra_context(self) -> None:
        ctx = HostContext(
            hostname="host",
            user="user",
            extra_context=("Python 3.12", "Django 5.0"),
        )
        prompt = build_user_prompt(
            natural_language="run tests",
            host_context=ctx,
        )
        assert "Python 3.12" in prompt
        assert "Django 5.0" in prompt

    def test_empty_natural_language_raises(self) -> None:
        ctx = HostContext(hostname="host", user="user")
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            build_user_prompt(natural_language="", host_context=ctx)

    def test_whitespace_only_natural_language_raises(self) -> None:
        ctx = HostContext(hostname="host", user="user")
        with pytest.raises(ValueError, match="natural_language must not be empty"):
            build_user_prompt(natural_language="   ", host_context=ctx)

    def test_minimal_context_still_valid(self) -> None:
        """With only hostname/user, prompt should still be well-formed."""
        ctx = HostContext(hostname="h", user="u")
        prompt = build_user_prompt(
            natural_language="do something",
            host_context=ctx,
        )
        assert "do something" in prompt
        assert len(prompt) > 20


# -- build_messages tests --


class TestBuildMessages:
    """Tests for the convenience function that builds the full messages list."""

    def test_returns_list_of_dicts(self) -> None:
        ctx = HostContext(hostname="host", user="user")
        messages = build_messages(
            natural_language="run tests",
            host_context=ctx,
        )
        assert isinstance(messages, list)
        assert len(messages) == 2

    def test_first_message_is_system(self) -> None:
        ctx = HostContext(hostname="host", user="user")
        messages = build_messages(
            natural_language="run tests",
            host_context=ctx,
        )
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], str)

    def test_second_message_is_user(self) -> None:
        ctx = HostContext(hostname="host", user="user")
        messages = build_messages(
            natural_language="run tests",
            host_context=ctx,
        )
        assert messages[1]["role"] == "user"
        assert "run tests" in messages[1]["content"]

    def test_custom_config_flows_through(self) -> None:
        ctx = HostContext(hostname="host", user="user")
        cfg = PromptConfig(max_commands=7)
        messages = build_messages(
            natural_language="run tests",
            host_context=ctx,
            config=cfg,
        )
        assert "7" in messages[0]["content"]

    def test_messages_are_independent_copies(self) -> None:
        """Returned messages must not share mutable state."""
        ctx = HostContext(hostname="host", user="user")
        m1 = build_messages(natural_language="a", host_context=ctx)
        m2 = build_messages(natural_language="b", host_context=ctx)
        assert m1[1]["content"] != m2[1]["content"]

    def test_system_prompt_contains_schema(self) -> None:
        ctx = HostContext(hostname="host", user="user")
        messages = build_messages(
            natural_language="run tests",
            host_context=ctx,
        )
        system_content = messages[0]["content"]
        assert "commands" in system_content
        assert "explanation" in system_content

    def test_conversation_history_included(self) -> None:
        ctx = HostContext(hostname="host", user="user")
        history = [
            {"role": "assistant", "content": "Previous response"},
            {"role": "user", "content": "Follow-up question"},
        ]
        messages = build_messages(
            natural_language="run tests",
            host_context=ctx,
            conversation_history=history,
        )
        # system + history(2) + user = 4
        assert len(messages) == 4
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "user"
        assert "run tests" in messages[3]["content"]

    def test_conversation_history_not_mutated(self) -> None:
        ctx = HostContext(hostname="host", user="user")
        history = [
            {"role": "assistant", "content": "Previous response"},
        ]
        original_len = len(history)
        build_messages(
            natural_language="run tests",
            host_context=ctx,
            conversation_history=history,
        )
        assert len(history) == original_len
