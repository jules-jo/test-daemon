"""Prompt construction for the LLM agent loop.

Builds the system prompt (role definition, SSH command constraints,
forbidden patterns, and structured output schema) and user prompt
(natural-language request with host context) for the Dataiku Mesh
LLM call that translates user intent into executable shell commands.

The system prompt enforces security-first principles:
- Every SSH command requires explicit human approval
- Dangerous shell patterns are explicitly forbidden
- Output is structured JSON so the daemon can parse and present
  commands for confirmation before execution

Usage::

    from jules_daemon.llm.prompts import (
        HostContext,
        PromptConfig,
        build_messages,
    )

    ctx = HostContext(hostname="staging.example.com", user="deploy")
    messages = build_messages(
        natural_language="run the smoke tests",
        host_context=ctx,
    )
    # Pass messages to create_completion()
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any


# -- Default security constraints --

_DEFAULT_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    "> /dev/sd",
    ":(){ :|:& };:",
    "chmod -R 777 /",
    "wget | sh",
    "curl | sh",
    "wget | bash",
    "curl | bash",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "halt",
    "poweroff",
    "passwd",
    "userdel",
    "useradd",
    "groupdel",
    "visudo",
    "iptables -F",
    "> /etc/",
    "mv /etc/",
    "rm /etc/",
)

_DEFAULT_ALLOWED_ACTIONS: tuple[str, ...] = (
    "run test suites",
    "execute test commands",
    "list test files",
    "check test framework version",
    "view test configuration",
    "navigate to project directories",
    "read log files",
    "check process status",
)


# -- Immutable data types --


@dataclass(frozen=True)
class HostContext:
    """Remote host context for prompt construction.

    Contains connection details and optional hints about the remote
    environment to help the LLM generate accurate shell commands.
    """

    hostname: str
    user: str
    port: int = 22
    working_directory: str | None = None
    os_hint: str | None = None
    shell_hint: str | None = None
    test_framework_hint: str | None = None
    extra_context: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.hostname:
            raise ValueError("hostname must not be empty")
        if not self.user:
            raise ValueError("user must not be empty")
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be 1-65535, got {self.port}")


@dataclass(frozen=True)
class PromptConfig:
    """Configuration for prompt construction.

    Controls security constraints and output formatting that are
    embedded into the system prompt.
    """

    max_commands: int = 5
    forbidden_patterns: tuple[str, ...] = _DEFAULT_FORBIDDEN_PATTERNS
    allowed_actions: tuple[str, ...] = _DEFAULT_ALLOWED_ACTIONS
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        if self.max_commands < 1:
            raise ValueError(
                f"max_commands must be positive, got {self.max_commands}"
            )


# -- Output schema definition --

_OUTPUT_SCHEMA: dict[str, Any] = {
    "commands": [
        {
            "command": "cd /opt/app && pytest -v --tb=short",
            "description": "Run the full pytest suite with verbose output",
            "working_directory": "/opt/app",
            "timeout_seconds": 300,
        }
    ],
    "explanation": "Running the full test suite in the application directory using pytest with verbose and short traceback output.",
    "confidence": "high",
    "warnings": [],
}

_OUTPUT_SCHEMA_JSON: str = json.dumps(_OUTPUT_SCHEMA, indent=2)


# -- Prompt builders --


def _build_forbidden_section(patterns: tuple[str, ...]) -> str:
    """Format the forbidden patterns list for the system prompt."""
    lines = ["The following shell patterns are FORBIDDEN. Never generate commands containing:"]
    for pattern in patterns:
        lines.append(f"  - `{pattern}`")
    return "\n".join(lines)


def _build_allowed_section(actions: tuple[str, ...]) -> str:
    """Format the allowed actions list for the system prompt."""
    lines = ["You are ONLY allowed to perform these categories of actions:"]
    for action in actions:
        lines.append(f"  - {action}")
    return "\n".join(lines)


def build_system_prompt(
    *,
    config: PromptConfig | None = None,
) -> str:
    """Build the system prompt for the test execution LLM.

    The system prompt defines:
    - The agent's role as a test execution assistant
    - SSH command safety constraints and forbidden patterns
    - Allowed action categories
    - The structured JSON output schema
    - Human approval requirements

    Args:
        config: Prompt configuration. Uses defaults if None.

    Returns:
        Complete system prompt string.
    """
    cfg = config or PromptConfig()

    sections: list[str] = [
        _section_role(cfg),
        _section_constraints(cfg),
        _section_output_schema(cfg),
        _section_rules(cfg),
    ]

    return "\n\n".join(sections)


def _section_role(cfg: PromptConfig) -> str:
    """Build the role definition section."""
    lines = [
        "You are a test execution assistant that translates natural-language "
        "requests into safe shell commands for execution on remote systems via SSH.",
        "",
        "Your job is to:",
        "1. Understand the user's intent (which tests to run, on which host)",
        "2. Generate the precise shell command(s) needed",
        "3. Return structured JSON so the daemon can present commands for human review",
    ]

    if cfg.require_human_approval:
        lines.extend([
            "",
            "CRITICAL: Every command you generate will be shown to a human for approval "
            "before execution. You must never assume commands will run automatically. "
            "All SSH commands require explicit human approval before execution.",
        ])

    return "\n".join(lines)


def _section_constraints(cfg: PromptConfig) -> str:
    """Build the security constraints section."""
    parts: list[str] = [
        "## Security Constraints",
        "",
        _build_allowed_section(cfg.allowed_actions),
        "",
        _build_forbidden_section(cfg.forbidden_patterns),
        "",
        "Additional rules:",
        f"  - Generate at most {cfg.max_commands} commands per response",
        "  - Never generate commands that modify system configuration",
        "  - Never generate commands that install or remove packages",
        "  - Never generate commands that create, modify, or delete users",
        "  - Never pipe untrusted input to a shell interpreter",
        "  - If the request is ambiguous, ask for clarification instead of guessing",
        "  - If the request asks for something outside allowed actions, refuse clearly",
    ]
    return "\n".join(parts)


def _section_output_schema(cfg: PromptConfig) -> str:
    """Build the output schema section with the JSON example."""
    return (
        "## Output Schema\n"
        "\n"
        "You MUST respond with a JSON object in the following format. "
        f"Include at most {cfg.max_commands} commands.\n"
        "\n"
        "```json\n"
        f"{_OUTPUT_SCHEMA_JSON}\n"
        "```\n"
        "\n"
        "Field definitions:\n"
        "  - `commands`: Array of command objects to execute sequentially\n"
        "    - `command`: The exact shell command string\n"
        "    - `description`: Human-readable description of what the command does\n"
        "    - `working_directory`: Directory to cd into before running (optional)\n"
        "    - `timeout_seconds`: Maximum execution time in seconds (default: 300)\n"
        "  - `explanation`: Brief explanation of the overall plan\n"
        "  - `confidence`: One of 'high', 'medium', 'low' -- your confidence that "
        "these commands correctly fulfill the request\n"
        "  - `warnings`: Array of strings noting any risks or caveats\n"
        "\n"
        "If you cannot fulfill the request safely, respond with:\n"
        "```json\n"
        '{\n'
        '  "commands": [],\n'
        '  "explanation": "Reason why the request cannot be fulfilled",\n'
        '  "confidence": "low",\n'
        '  "warnings": ["Specific concern"]\n'
        '}\n'
        "```"
    )


def _section_rules(cfg: PromptConfig) -> str:
    """Build the behavioral rules section."""
    lines = [
        "## Behavioral Rules",
        "",
        "1. Be precise: use exact paths and flags, do not leave placeholders",
        "2. Be safe: when in doubt, choose the more conservative option",
        "3. Be transparent: explain what each command does and why",
        "4. Be minimal: generate the fewest commands needed to fulfill the request",
    ]

    if cfg.require_human_approval:
        lines.append(
            "5. Respect human review: commands will be presented for human approval, "
            "so make them readable and well-documented"
        )
        lines.append(
            "6. If context is insufficient, set confidence to 'low' and add warnings"
        )
    else:
        lines.append(
            "5. Be readable: make commands well-documented for review"
        )
        lines.append(
            "6. If context is insufficient, set confidence to 'low' and add warnings"
        )

    return "\n".join(lines)


def build_user_prompt(
    *,
    natural_language: str,
    host_context: HostContext,
) -> str:
    """Build the user prompt from a natural-language request and host context.

    Combines the user's plain-English request with structured information
    about the remote host so the LLM can generate appropriate commands.

    Args:
        natural_language: The user's natural-language request (e.g.,
            "run the smoke tests on staging").
        host_context: Connection details and environment hints for
            the target host.

    Returns:
        Formatted user prompt string.

    Raises:
        ValueError: If natural_language is empty or whitespace-only.
    """
    stripped = natural_language.strip()
    if not stripped:
        raise ValueError("natural_language must not be empty")

    sections: list[str] = [
        f"## Request\n\n{stripped}",
        _build_host_section(host_context),
    ]

    return "\n\n".join(sections)


def _build_host_section(ctx: HostContext) -> str:
    """Build the host context section of the user prompt."""
    lines: list[str] = [
        "## Target Host",
        "",
        f"- **Hostname:** {ctx.hostname}",
        f"- **User:** {ctx.user}",
    ]

    if ctx.port != 22:
        lines.append(f"- **SSH Port:** {ctx.port}")

    if ctx.working_directory is not None:
        lines.append(f"- **Working Directory:** {ctx.working_directory}")

    if ctx.os_hint is not None:
        lines.append(f"- **OS:** {ctx.os_hint}")

    if ctx.shell_hint is not None:
        lines.append(f"- **Shell:** {ctx.shell_hint}")

    if ctx.test_framework_hint is not None:
        lines.append(f"- **Test Framework:** {ctx.test_framework_hint}")

    if ctx.extra_context:
        lines.append("")
        lines.append("### Additional Context")
        for item in ctx.extra_context:
            lines.append(f"- {item}")

    return "\n".join(lines)


def build_messages(
    *,
    natural_language: str,
    host_context: HostContext,
    config: PromptConfig | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the complete message list for the LLM call.

    Constructs a list of messages in OpenAI chat format:
    1. System message with role definition, constraints, and schema
    2. Optional conversation history (deep-copied, not mutated)
    3. User message with the natural-language request and host context

    Args:
        natural_language: The user's plain-English request.
        host_context: Target host details.
        config: Prompt configuration (uses defaults if None).
        conversation_history: Optional prior messages to include.
            These are deep-copied to prevent mutation.

    Returns:
        List of message dicts with 'role' and 'content' keys.
        The caller owns the returned list and may mutate it freely.
        Do not share the list across call sites without copying.
    """
    system_content = build_system_prompt(config=config)
    user_content = build_user_prompt(
        natural_language=natural_language,
        host_context=host_context,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
    ]

    if conversation_history:
        for msg in conversation_history:
            messages.append(copy.deepcopy(msg))

    messages.append({"role": "user", "content": user_content})

    return messages
