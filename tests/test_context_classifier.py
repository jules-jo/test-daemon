"""Tests for the command context classifier (LLM-powered risk analysis).

Covers:
    - ContextClassifier construction and configuration
    - build_context_prompt: prompt generation for risk classification
    - classify(): end-to-end pipeline with mocked LLM responses
    - classify_command(): convenience function
    - Error handling: LLM failures, parse failures, empty responses
    - Security invariant: requires_approval is always True in output
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from jules_daemon.llm.command_context import (
    CommandContext,
    RiskLevel,
)
from jules_daemon.llm.config import LLMConfig
from jules_daemon.llm.context_classifier import (
    ContextClassifier,
    build_context_system_prompt,
    build_context_user_prompt,
    classify_command,
)
from jules_daemon.llm.errors import LLMError, LLMParseError
from jules_daemon.ssh.command import SSHCommand


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_config() -> LLMConfig:
    return LLMConfig(
        base_url="https://dss.example.com/public/api/projects/TEST/llms/openai/v1/",
        api_key="test-api-key",
        default_model="openai:conn:gpt-4",
    )


def _make_ssh_command(
    command: str = "pytest -v --tb=short",
    working_directory: str | None = "/opt/app",
    timeout: int = 300,
) -> SSHCommand:
    return SSHCommand(
        command=command,
        working_directory=working_directory,
        timeout=timeout,
    )


def _make_llm_json_response(
    *,
    explanation: str = "Runs the pytest test suite with verbose output",
    affected_paths: list[str] | None = None,
    risk_level: str = "low",
    risk_factors: list[str] | None = None,
    safe_to_execute: bool = True,
) -> str:
    return json.dumps({
        "explanation": explanation,
        "affected_paths": affected_paths or ["/opt/app/tests"],
        "risk_level": risk_level,
        "risk_factors": risk_factors or [],
        "safe_to_execute": safe_to_execute,
    })


def _mock_completion(content: str) -> MagicMock:
    """Create a mock ChatCompletion with the given content."""
    choice = MagicMock()
    choice.message.content = content
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


class TestBuildContextPrompts:
    """Tests for prompt template construction."""

    def test_system_prompt_contains_role(self) -> None:
        prompt = build_context_system_prompt()
        assert "security analyst" in prompt.lower() or "analyze" in prompt.lower()

    def test_system_prompt_contains_risk_levels(self) -> None:
        prompt = build_context_system_prompt()
        assert "low" in prompt.lower()
        assert "medium" in prompt.lower()
        assert "high" in prompt.lower()
        assert "critical" in prompt.lower()

    def test_system_prompt_contains_json_schema(self) -> None:
        prompt = build_context_system_prompt()
        assert "explanation" in prompt
        assert "affected_paths" in prompt
        assert "risk_level" in prompt
        assert "risk_factors" in prompt
        assert "safe_to_execute" in prompt

    def test_user_prompt_contains_command(self) -> None:
        cmd = _make_ssh_command(command="pytest -v")
        prompt = build_context_user_prompt(ssh_command=cmd)
        assert "pytest -v" in prompt

    def test_user_prompt_contains_working_directory(self) -> None:
        cmd = _make_ssh_command(working_directory="/opt/app")
        prompt = build_context_user_prompt(ssh_command=cmd)
        assert "/opt/app" in prompt

    def test_user_prompt_with_no_working_directory(self) -> None:
        cmd = _make_ssh_command(working_directory=None)
        prompt = build_context_user_prompt(ssh_command=cmd)
        assert "pytest -v" in prompt

    def test_user_prompt_contains_timeout(self) -> None:
        cmd = _make_ssh_command(timeout=600)
        prompt = build_context_user_prompt(ssh_command=cmd)
        assert "600" in prompt

    def test_user_prompt_with_environment(self) -> None:
        cmd = SSHCommand(
            command="pytest -v",
            working_directory="/opt/app",
            timeout=300,
            environment={"NODE_ENV": "test", "CI": "true"},
        )
        prompt = build_context_user_prompt(ssh_command=cmd)
        assert "NODE_ENV" in prompt
        assert "CI" in prompt


# ---------------------------------------------------------------------------
# ContextClassifier construction
# ---------------------------------------------------------------------------


class TestContextClassifierConstruction:
    """Tests for ContextClassifier initialization."""

    def test_create_with_defaults(self) -> None:
        client = MagicMock()
        config = _make_config()
        classifier = ContextClassifier(client=client, config=config)
        assert classifier is not None

    def test_create_with_custom_temperature(self) -> None:
        client = MagicMock()
        config = _make_config()
        classifier = ContextClassifier(
            client=client, config=config, temperature=0.2
        )
        assert classifier._temperature == 0.2

    def test_negative_temperature_raises(self) -> None:
        client = MagicMock()
        config = _make_config()
        with pytest.raises(ValueError, match="temperature"):
            ContextClassifier(client=client, config=config, temperature=-0.1)

    def test_temperature_above_two_raises(self) -> None:
        client = MagicMock()
        config = _make_config()
        with pytest.raises(ValueError, match="temperature"):
            ContextClassifier(client=client, config=config, temperature=2.1)


# ---------------------------------------------------------------------------
# ContextClassifier.classify() with mocked LLM
# ---------------------------------------------------------------------------


class TestContextClassifierClassify:
    """Tests for the classify() method with mocked LLM calls."""

    def _make_classifier(self) -> tuple[ContextClassifier, MagicMock]:
        client = MagicMock()
        config = _make_config()
        classifier = ContextClassifier(client=client, config=config)
        return classifier, client

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_low_risk_command(
        self, mock_create: MagicMock
    ) -> None:
        classifier, _ = self._make_classifier()
        mock_create.return_value = _mock_completion(
            _make_llm_json_response(
                explanation="Lists all files in the current directory",
                affected_paths=["/home/user"],
                risk_level="low",
                risk_factors=[],
                safe_to_execute=True,
            )
        )

        cmd = _make_ssh_command(command="ls -la")
        result = classifier.classify(ssh_command=cmd)

        assert isinstance(result, CommandContext)
        assert result.command == "ls -la"
        assert result.risk_level is RiskLevel.LOW
        assert result.safe_to_execute is True
        assert result.requires_approval is True

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_high_risk_command(
        self, mock_create: MagicMock
    ) -> None:
        classifier, _ = self._make_classifier()
        mock_create.return_value = _mock_completion(
            _make_llm_json_response(
                explanation="Removes the test output directory recursively",
                affected_paths=["/tmp/test-output"],
                risk_level="high",
                risk_factors=["Uses rm -rf", "Deletes directory tree"],
                safe_to_execute=False,
            )
        )

        cmd = _make_ssh_command(command="rm -rf /tmp/test-output")
        result = classifier.classify(ssh_command=cmd)

        assert result.risk_level is RiskLevel.HIGH
        assert result.safe_to_execute is False
        assert len(result.risk_factors) == 2
        assert result.requires_approval is True

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_critical_risk_command(
        self, mock_create: MagicMock
    ) -> None:
        classifier, _ = self._make_classifier()
        mock_create.return_value = _mock_completion(
            _make_llm_json_response(
                explanation="Formats the disk partition",
                affected_paths=["/dev/sda1"],
                risk_level="critical",
                risk_factors=["Destructive disk operation", "Data loss risk"],
                safe_to_execute=False,
            )
        )

        cmd = _make_ssh_command(command="mkfs.ext4 /dev/sda1")
        result = classifier.classify(ssh_command=cmd)

        assert result.risk_level is RiskLevel.CRITICAL
        assert result.safe_to_execute is False
        assert result.requires_approval is True

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_preserves_original_command(
        self, mock_create: MagicMock
    ) -> None:
        """The original SSHCommand.command is used, not whatever the LLM returns."""
        classifier, _ = self._make_classifier()
        mock_create.return_value = _mock_completion(
            _make_llm_json_response(
                explanation="Run tests",
            )
        )

        cmd = _make_ssh_command(command="pytest -v --tb=short")
        result = classifier.classify(ssh_command=cmd)
        assert result.command == "pytest -v --tb=short"

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_calls_create_completion(
        self, mock_create: MagicMock
    ) -> None:
        classifier, client = self._make_classifier()
        mock_create.return_value = _mock_completion(
            _make_llm_json_response()
        )

        cmd = _make_ssh_command()
        classifier.classify(ssh_command=cmd)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["client"] is client
        assert len(call_kwargs.kwargs["messages"]) == 2  # system + user

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_uses_temperature_zero(
        self, mock_create: MagicMock
    ) -> None:
        classifier, _ = self._make_classifier()
        mock_create.return_value = _mock_completion(
            _make_llm_json_response()
        )

        cmd = _make_ssh_command()
        classifier.classify(ssh_command=cmd)

        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_with_custom_temperature(
        self, mock_create: MagicMock
    ) -> None:
        client = MagicMock()
        config = _make_config()
        classifier = ContextClassifier(
            client=client, config=config, temperature=0.3
        )
        mock_create.return_value = _mock_completion(
            _make_llm_json_response()
        )

        cmd = _make_ssh_command()
        classifier.classify(ssh_command=cmd)

        call_kwargs = mock_create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.3

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_empty_choices_raises(
        self, mock_create: MagicMock
    ) -> None:
        classifier, _ = self._make_classifier()
        completion = MagicMock()
        completion.choices = []
        mock_create.return_value = completion

        cmd = _make_ssh_command()
        with pytest.raises(LLMParseError, match="empty"):
            classifier.classify(ssh_command=cmd)

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_null_content_raises(
        self, mock_create: MagicMock
    ) -> None:
        classifier, _ = self._make_classifier()
        choice = MagicMock()
        choice.message.content = None
        completion = MagicMock()
        completion.choices = [choice]
        mock_create.return_value = completion

        cmd = _make_ssh_command()
        with pytest.raises(LLMParseError, match="empty"):
            classifier.classify(ssh_command=cmd)

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_llm_error_propagates(
        self, mock_create: MagicMock
    ) -> None:
        classifier, _ = self._make_classifier()
        mock_create.side_effect = LLMError("Connection failed")

        cmd = _make_ssh_command()
        with pytest.raises(LLMError, match="Connection failed"):
            classifier.classify(ssh_command=cmd)

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_malformed_json_raises_parse_error(
        self, mock_create: MagicMock
    ) -> None:
        classifier, _ = self._make_classifier()
        mock_create.return_value = _mock_completion("This is not JSON at all.")

        cmd = _make_ssh_command()
        with pytest.raises(LLMParseError):
            classifier.classify(ssh_command=cmd)

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_code_fenced_response(
        self, mock_create: MagicMock
    ) -> None:
        """LLM responses wrapped in markdown code fences should be parsed."""
        classifier, _ = self._make_classifier()
        inner = _make_llm_json_response(
            explanation="Run pytest in verbose mode",
            risk_level="low",
        )
        content = f"Here is the analysis:\n```json\n{inner}\n```\n"
        mock_create.return_value = _mock_completion(content)

        cmd = _make_ssh_command()
        result = classifier.classify(ssh_command=cmd)
        assert result.risk_level is RiskLevel.LOW
        assert result.explanation == "Run pytest in verbose mode"


# ---------------------------------------------------------------------------
# classify_command convenience function
# ---------------------------------------------------------------------------


class TestClassifyCommandFunction:
    """Tests for the one-shot classify_command() convenience function."""

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_command_returns_context(
        self, mock_create: MagicMock
    ) -> None:
        mock_create.return_value = _mock_completion(
            _make_llm_json_response(
                explanation="Run tests",
                risk_level="low",
            )
        )

        config = _make_config()
        client = MagicMock()
        cmd = _make_ssh_command(command="pytest -v")

        result = classify_command(
            ssh_command=cmd,
            client=client,
            config=config,
        )
        assert isinstance(result, CommandContext)
        assert result.command == "pytest -v"
        assert result.requires_approval is True

    @patch("jules_daemon.llm.context_classifier.create_completion")
    def test_classify_command_requires_approval_always(
        self, mock_create: MagicMock
    ) -> None:
        """Security invariant: every classified command requires approval."""
        mock_create.return_value = _mock_completion(
            _make_llm_json_response(
                risk_level="low",
                safe_to_execute=True,
            )
        )

        config = _make_config()
        client = MagicMock()
        cmd = _make_ssh_command()

        result = classify_command(
            ssh_command=cmd,
            client=client,
            config=config,
        )
        assert result.requires_approval is True
