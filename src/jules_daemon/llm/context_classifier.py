"""LLM-powered command context classifier.

Sends an SSH command to the Dataiku Mesh LLM and receives a structured
risk analysis: what the command does, which paths it affects, and its
risk level. This is the pre-approval step that gives the human operator
the context needed to make an informed Allow/Deny decision.

The classifier is a pure analysis step -- it never executes commands.
Its output (a ``CommandContext``) is presented to the human for review
before any SSH execution occurs.

Design:
    - Temperature 0.0 by default for deterministic classification
    - Single LLM call per command (no multi-turn)
    - Original command string is preserved (never trust the LLM echo)
    - ``requires_approval`` is force-set to True (defense in depth)

Usage::

    from jules_daemon.llm.context_classifier import (
        ContextClassifier,
        classify_command,
    )

    # Reusable classifier (preferred for daemon)
    classifier = ContextClassifier(client=client, config=config)
    context = classifier.classify(ssh_command=cmd)

    # One-shot convenience
    context = classify_command(
        ssh_command=cmd,
        client=client,
        config=config,
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from jules_daemon.llm.client import create_completion
from jules_daemon.llm.command_context import (
    CommandContext,
    parse_context_response,
)
from jules_daemon.llm.config import LLMConfig
from jules_daemon.llm.errors import LLMParseError
from jules_daemon.llm.models import ToolCallingMode
from jules_daemon.ssh.command import SSHCommand

logger = logging.getLogger(__name__)

__all__ = [
    "ContextClassifier",
    "build_context_system_prompt",
    "build_context_user_prompt",
    "classify_command",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TEMPERATURE: float = 0.0
"""Deterministic output for consistent classification."""

_MIN_TEMPERATURE: float = 0.0
_MAX_TEMPERATURE: float = 2.0


# ---------------------------------------------------------------------------
# Output schema example (embedded in system prompt)
# ---------------------------------------------------------------------------

_OUTPUT_EXAMPLE: dict[str, Any] = {
    "explanation": "Runs the pytest test suite with verbose output and short tracebacks in the /opt/app directory",
    "affected_paths": ["/opt/app/tests", "/opt/app/.pytest_cache"],
    "risk_level": "low",
    "risk_factors": [],
    "safe_to_execute": True,
}

_OUTPUT_EXAMPLE_JSON: str = json.dumps(_OUTPUT_EXAMPLE, indent=2)


# ---------------------------------------------------------------------------
# Risk level definitions (embedded in system prompt)
# ---------------------------------------------------------------------------

_RISK_LEVEL_DEFINITIONS: str = """Risk levels (choose exactly one):
  - "low": Read-only operations. Listing files, checking versions, reading logs, viewing configuration. No data is modified.
  - "medium": Operations that run tests or builds. May create temporary output files but do not modify persistent application state or configuration.
  - "high": Operations that modify files, write data, change configuration, or alter directory contents. Reversible but consequential.
  - "critical": Destructive or system-level operations. Disk formatting, package removal, firewall changes, user management, service restarts. Potentially irreversible."""


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_context_system_prompt() -> str:
    """Build the system prompt for command context classification.

    The system prompt instructs the LLM to analyze an SSH command and
    produce a structured JSON response with:
    - What the command does (explanation)
    - Which paths it affects (affected_paths)
    - Risk level classification (risk_level)
    - Specific risk factors (risk_factors)
    - Whether it is safe to execute (safe_to_execute)

    Returns:
        Complete system prompt string.
    """
    sections: list[str] = [
        _section_role(),
        _section_risk_definitions(),
        _section_output_schema(),
        _section_analysis_rules(),
    ]
    return "\n\n".join(sections)


def _section_role() -> str:
    """Role definition for the security analyst."""
    return (
        "You are a security analyst for SSH command execution. "
        "Your job is to analyze shell commands and produce a structured "
        "risk assessment. You do NOT execute commands -- you only analyze them.\n"
        "\n"
        "For each command, you must determine:\n"
        "1. What the command does (clear, non-technical explanation)\n"
        "2. Which filesystem paths are read or modified\n"
        "3. The risk level of execution\n"
        "4. Specific risk factors that justify the risk level\n"
        "5. Whether the command is safe to execute"
    )


def _section_risk_definitions() -> str:
    """Risk level definitions."""
    return f"## Risk Level Definitions\n\n{_RISK_LEVEL_DEFINITIONS}"


def _section_output_schema() -> str:
    """JSON output schema with example."""
    return (
        "## Output Schema\n"
        "\n"
        "You MUST respond with a JSON object in exactly this format:\n"
        "\n"
        "```json\n"
        f"{_OUTPUT_EXAMPLE_JSON}\n"
        "```\n"
        "\n"
        "Field definitions:\n"
        '  - "explanation": Brief, clear description of what the command does\n'
        '  - "affected_paths": Array of filesystem paths the command reads or writes\n'
        '  - "risk_level": One of "low", "medium", "high", "critical"\n'
        '  - "risk_factors": Array of strings explaining why this risk level was assigned. '
        "Empty array for low-risk commands with no concerns.\n"
        '  - "safe_to_execute": Boolean. true if the command appears safe, '
        "false if it could cause harm or data loss"
    )


def _section_analysis_rules() -> str:
    """Behavioral rules for the analysis."""
    return (
        "## Analysis Rules\n"
        "\n"
        "1. Be conservative: when uncertain, assign a higher risk level\n"
        "2. Consider side effects: piped commands, subshells, and redirections\n"
        "3. Consider the working directory: relative paths resolve there\n"
        "4. Flag chained commands (&&, ||, ;) as higher risk than single commands\n"
        "5. Flag wildcard patterns (*, ?) that could match unexpected files\n"
        "6. Flag any command that writes, deletes, or modifies files as at least 'high'\n"
        "7. Flag system administration commands (sudo, service, systemctl) as 'critical'\n"
        "8. Consider environment variables: they may alter command behavior\n"
        "9. If the command is ambiguous or you cannot determine what it does, "
        'set risk_level to "high" and safe_to_execute to false'
    )


def build_context_user_prompt(
    *,
    ssh_command: SSHCommand,
) -> str:
    """Build the user prompt for a specific SSH command analysis.

    Provides the command string and all available context (working
    directory, timeout, environment variables) to the LLM for analysis.

    Args:
        ssh_command: The SSH command to analyze.

    Returns:
        Formatted user prompt string.
    """
    lines: list[str] = [
        "## Command to Analyze",
        "",
        "```",
        ssh_command.command,
        "```",
        "",
    ]

    lines.append("## Execution Context")
    lines.append("")

    if ssh_command.working_directory is not None:
        lines.append(
            f"- **Working directory:** {ssh_command.working_directory}"
        )
    else:
        lines.append("- **Working directory:** (user home directory)")

    lines.append(f"- **Timeout:** {ssh_command.timeout} seconds")

    if ssh_command.environment:
        lines.append("")
        lines.append("### Environment Variables")
        for key, value in ssh_command.environment.items():
            lines.append(f"- `{key}={value}`")

    lines.extend([
        "",
        "Analyze this command and respond with the JSON risk assessment.",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class ContextClassifier:
    """LLM-powered command context and risk-level classifier.

    Sends SSH commands to the Dataiku Mesh LLM for security analysis.
    Each call produces a ``CommandContext`` with explanation, affected
    paths, risk level, and risk factors.

    The classifier is stateless and reusable across multiple commands.
    For the daemon's single-user model, one instance per daemon lifetime.

    Args:
        client: OpenAI client configured for Dataiku Mesh.
        config: LLM configuration.
        tool_calling_mode: How tools are passed to the LLM.
        temperature: LLM temperature (0.0 for deterministic).
    """

    def __init__(
        self,
        *,
        client: OpenAI,
        config: LLMConfig,
        tool_calling_mode: ToolCallingMode = ToolCallingMode.NATIVE,
        temperature: float = _DEFAULT_TEMPERATURE,
    ) -> None:
        if temperature < _MIN_TEMPERATURE or temperature > _MAX_TEMPERATURE:
            raise ValueError(
                f"temperature must be between {_MIN_TEMPERATURE} and "
                f"{_MAX_TEMPERATURE}, got {temperature}"
            )

        self._client = client
        self._config = config
        self._tool_calling_mode = tool_calling_mode
        self._temperature = temperature

        # Cache the system prompt (deterministic, same per classifier)
        self._system_prompt = build_context_system_prompt()

    def classify(
        self,
        *,
        ssh_command: SSHCommand,
    ) -> CommandContext:
        """Classify an SSH command and produce a structured context.

        Pipeline:
        1. Build system + user prompt messages
        2. Call LLM via Dataiku Mesh
        3. Extract and validate JSON response
        4. Override command string with the authoritative original
        5. Force requires_approval=True (defense in depth)

        Args:
            ssh_command: The SSH command to analyze.

        Returns:
            Validated CommandContext with risk classification.

        Raises:
            LLMParseError: If the LLM response cannot be parsed.
            LLMError: For LLM client errors (auth, connection, etc.).
        """
        logger.info(
            "Classifying command: %.100s",
            ssh_command.command,
        )

        messages = self._build_messages(ssh_command=ssh_command)

        raw_content = self._call_llm(messages=messages)

        context = parse_context_response(
            raw_content,
            command=ssh_command.command,
        )

        logger.info(
            "Classification complete: risk=%s, safe=%s, factors=%d",
            context.risk_level.value,
            context.safe_to_execute,
            len(context.risk_factors),
        )

        return context

    def _build_messages(
        self,
        *,
        ssh_command: SSHCommand,
    ) -> list[dict[str, str]]:
        """Build the LLM message list for classification."""
        user_content = build_context_user_prompt(ssh_command=ssh_command)
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _call_llm(
        self,
        *,
        messages: list[dict[str, Any]],
    ) -> str:
        """Call the LLM and extract the response content.

        Args:
            messages: Message list for the LLM.

        Returns:
            Raw content string from the LLM response.

        Raises:
            LLMError: On any LLM client error.
            LLMParseError: If the response has no content.
        """
        response = create_completion(
            client=self._client,
            config=self._config,
            messages=messages,
            tool_calling_mode=self._tool_calling_mode,
            temperature=self._temperature,
        )

        if not response.choices:
            raise LLMParseError(
                "LLM returned empty choices list",
                raw_content="",
            )

        content = response.choices[0].message.content
        if not content:
            raise LLMParseError(
                "LLM returned empty content",
                raw_content="",
            )

        return content


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def classify_command(
    *,
    ssh_command: SSHCommand,
    client: OpenAI,
    config: LLMConfig,
    tool_calling_mode: ToolCallingMode = ToolCallingMode.NATIVE,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> CommandContext:
    """One-shot convenience function for command classification.

    Creates a temporary ContextClassifier and calls classify().
    For repeated classifications, prefer creating a ContextClassifier
    instance directly to benefit from system prompt caching.

    Args:
        ssh_command: The SSH command to analyze.
        client: OpenAI client configured for Dataiku Mesh.
        config: LLM configuration.
        tool_calling_mode: How tools are passed to the LLM.
        temperature: LLM temperature.

    Returns:
        Validated CommandContext with risk classification.

    Raises:
        LLMParseError: If the LLM response cannot be parsed.
        LLMError: For LLM client errors.
    """
    classifier = ContextClassifier(
        client=client,
        config=config,
        tool_calling_mode=tool_calling_mode,
        temperature=temperature,
    )
    return classifier.classify(ssh_command=ssh_command)
