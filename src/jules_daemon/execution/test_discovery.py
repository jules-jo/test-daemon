"""Auto-discover test command specs by SSHing in and parsing --help output.

Runs ``command -h`` (or ``--help`` as fallback) on a remote host, sends
the captured output to the LLM for structured extraction, and returns an
immutable :class:`DiscoveredTestSpec`.  When the LLM is unavailable, the
raw help text is still returned so the user can parse it manually.

The wiki persistence helper :func:`save_discovered_spec` writes the spec
as a YAML-frontmatter wiki page under ``pages/daemon/knowledge/``,
reusing the existing test-knowledge directory and slug derivation.

Usage::

    from jules_daemon.execution.test_discovery import (
        DiscoveredTestSpec,
        discover_test,
        save_discovered_spec,
    )

    spec = await discover_test(
        host="10.74.30.211",
        user="root",
        command="python3.8 /root/tests/my_test.py",
    )
    if spec is not None:
        path = save_discovered_spec(wiki_root, spec, command, host)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from jules_daemon.ssh.credentials import SSHCredential, resolve_ssh_credentials
from jules_daemon.ssh.errors import SSHAuthenticationError, SSHConnectionError
from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument
from jules_daemon.wiki.test_knowledge import KNOWLEDGE_DIR, derive_test_slug

__all__ = [
    "DiscoveryProbeError",
    "DiscoveredTestSpec",
    "build_discovery_help_command",
    "build_discovery_help_commands",
    "discover_test",
    "format_spec_preview",
    "normalize_discovery_command",
    "resolve_discovery_command_candidates",
    "save_discovered_spec",
]

logger = logging.getLogger(__name__)

# Maximum seconds to wait for the -h / --help command to complete.
_HELP_TIMEOUT: int = 30

# Wiki metadata for discovered test spec pages.
_WIKI_TAGS: tuple[str, ...] = ("daemon", "test-spec", "discovered")
_WIKI_TYPE: str = "test-spec"
_PYTHON_INTERPRETER_NAMES: frozenset[str] = frozenset({
    "python",
    "python2",
    "python3",
    "python3.8",
    "python3.9",
    "python3.10",
    "python3.11",
    "python3.12",
    "python3.13",
    "python3.14",
    "py",
})

# LLM prompt template for parsing help output.
_LLM_PROMPT: str = """\
Parse this command-line help output and extract:
- command_template: the command with {{placeholder}} for each argument
- required_args: list of argument names that are required (no default value)
- optional_args: list of argument names that have defaults
- arg_descriptions: dict mapping arg name to its description from the help text
- typical_duration: estimate in seconds if mentioned, otherwise null

Help output:
{help_text}

Respond with JSON only, no other text:
{{
  "command_template": "...",
  "required_args": [...],
  "optional_args": [...],
  "arg_descriptions": {{...}},
  "typical_duration": null
}}"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredTestSpec:
    """Immutable spec extracted from a command's help output.

    Attributes:
        command_template: Command string with {placeholder} for each arg.
        required_args: Argument names that must be provided.
        optional_args: Argument names that have default values.
        arg_descriptions: Mapping of arg name to human-readable description.
        typical_duration: Estimated run time in seconds, or None.
        raw_help_text: The original stdout+stderr from ``-h`` / ``--help``.
    """

    command_template: str
    required_args: tuple[str, ...]
    optional_args: tuple[str, ...]
    arg_descriptions: dict[str, str]
    typical_duration: int | None
    raw_help_text: str

    # Tell pytest not to collect this as a test class.
    __test__ = False


@dataclass(frozen=True)
class _HelpProbeResult:
    """Captured help text plus the base command that produced it."""

    help_text: str
    executed_command: str


class DiscoveryProbeError(RuntimeError):
    """A remote help probe ran but did not yield usable help output."""

    def __init__(
        self,
        *,
        executed_command: str,
        attempted_help_commands: tuple[str, ...],
        exit_code: int | None,
        stdout_text: str,
        stderr_text: str,
    ) -> None:
        self.executed_command = executed_command
        self.attempted_help_commands = attempted_help_commands
        self.exit_code = exit_code
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text
        super().__init__(self.format_user_message())

    @property
    def last_attempted_command(self) -> str:
        if self.attempted_help_commands:
            return self.attempted_help_commands[-1]
        return self.executed_command

    @property
    def summary_text(self) -> str:
        for raw_text in (self.stderr_text, self.stdout_text):
            for line in raw_text.splitlines():
                stripped = line.strip()
                if stripped:
                    if len(stripped) > 240:
                        return stripped[:237] + "..."
                    return stripped
        return ""

    def format_user_message(self) -> str:
        command_display = self.last_attempted_command
        if self.summary_text:
            if self.exit_code is not None:
                return (
                    f"{command_display!r} exited with code {self.exit_code}: "
                    f"{self.summary_text}"
                )
            return f"{command_display!r} failed: {self.summary_text}"
        if self.exit_code is not None:
            return (
                f"{command_display!r} exited with code {self.exit_code} "
                "and produced no usable output"
            )
        return f"{command_display!r} failed without usable output"


def normalize_discovery_command(command: str) -> str:
    """Normalize a discover command before probing with ``-h``.

    Legacy compatibility helper.

    For bare Python script paths, returns the first preferred interpreter
    candidate. The actual discovery path may try additional candidates.
    """
    candidates = resolve_discovery_command_candidates(command)
    return candidates[0] if candidates else command.strip()


def resolve_discovery_command_candidates(command: str) -> tuple[str, ...]:
    """Return ordered base-command candidates for test discovery.

    Bare ``.py`` script paths try ``python3`` first, then ``python``.
    Commands that already include an interpreter or are not Python scripts
    keep their original form only.
    """
    raw = command.strip()
    if not raw:
        return ()

    try:
        tokens = shlex.split(raw, posix=True)
    except ValueError:
        return (raw,)

    if not tokens:
        return ()

    first = tokens[0]
    first_name = Path(first).name.lower()
    if first_name in _PYTHON_INTERPRETER_NAMES:
        return (raw,)

    if first.lower().endswith((".py", ".pyw")):
        return (f"python3 {raw}", f"python {raw}")

    return (raw,)


def build_discovery_help_command(command: str, *, flag: str = "-h") -> str:
    """Build the first remote help probe command for discovery."""
    commands = build_discovery_help_commands(command, flag=flag)
    return commands[0] if commands else command.strip()


def build_discovery_help_commands(command: str, *, flag: str = "-h") -> tuple[str, ...]:
    """Build all remote help probe commands for discovery."""
    return tuple(
        f"{candidate} {flag}"
        for candidate in resolve_discovery_command_candidates(command)
    )


def _looks_like_help_output(text: str) -> bool:
    """Best-effort check for actual help/usage text."""
    lowered = text.lower()
    return (
        "usage:" in lowered
        or "\nusage:" in lowered
        or "options:" in lowered
        or "--help" in lowered
        or "optional arguments" in lowered
    )


# ---------------------------------------------------------------------------
# SSH help execution (blocking -- runs in thread pool)
# ---------------------------------------------------------------------------


def _run_help_via_paramiko(
    *,
    host: str,
    port: int,
    username: str,
    credential: SSHCredential | None,
    command: str,
    timeout: int = _HELP_TIMEOUT,
) -> tuple[int, str, str]:
    """SSH in and execute ``command``, returning (exit_code, stdout, stderr).

    This is a blocking function intended to be called via
    ``asyncio.to_thread``.

    Raises:
        SSHAuthenticationError: On authentication failure.
        SSHConnectionError: On connection failure.
    """
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    connect_kwargs: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": username,
        "timeout": min(timeout, 30),
        "allow_agent": True,
        "look_for_keys": True,
    }

    if credential is not None:
        connect_kwargs["password"] = credential.password
        connect_kwargs["allow_agent"] = False
        connect_kwargs["look_for_keys"] = False
        if credential.username is not None:
            connect_kwargs["username"] = credential.username

    try:
        client.connect(**connect_kwargs)
    except paramiko.AuthenticationException as exc:
        raise SSHAuthenticationError(
            f"Authentication failed for {username}@{host}:{port}: {exc}"
        ) from exc
    except (
        paramiko.SSHException,
        OSError,
        TimeoutError,
        ConnectionRefusedError,
    ) as exc:
        raise SSHConnectionError(
            f"Connection failed to {host}:{port}: {exc}"
        ) from exc

    try:
        _, stdout_ch, stderr_ch = client.exec_command(command, timeout=timeout)
        stdout_text = stdout_ch.read().decode("utf-8", errors="replace")
        stderr_text = stderr_ch.read().decode("utf-8", errors="replace")
        exit_code = stdout_ch.channel.recv_exit_status()
        return exit_code, stdout_text, stderr_text
    finally:
        client.close()


async def _fetch_help_text(
    *,
    host: str,
    port: int,
    username: str,
    credential: SSHCredential | None,
    command: str,
) -> _HelpProbeResult | None:
    """Run ``command -h``, falling back to ``command --help``.

    Returns the combined stdout+stderr plus the base command that produced it,
    or None when the remote command ran but produced no usable help output.

    Raises:
        SSHAuthenticationError: When SSH authentication fails.
        SSHConnectionError: When the SSH connection itself fails.
    """
    last_probe_error: DiscoveryProbeError | None = None
    candidates = resolve_discovery_command_candidates(command)

    for candidate in candidates:
        candidate_help: _HelpProbeResult | None = None
        attempted_help_commands: list[str] = []
        last_exit_code: int | None = None
        last_stdout = ""
        last_stderr = ""

        for flag in ("-h", "--help"):
            full_cmd = f"{candidate} {flag}"
            attempted_help_commands.append(full_cmd)
            logger.info("Running help command: %s@%s: %s", username, host, full_cmd)
            try:
                exit_code, stdout, stderr = await asyncio.to_thread(
                    _run_help_via_paramiko,
                    host=host,
                    port=port,
                    username=username,
                    credential=credential,
                    command=full_cmd,
                )
            except (SSHAuthenticationError, SSHConnectionError) as exc:
                logger.warning("SSH failed while fetching help: %s", exc)
                raise
            except Exception as exc:
                logger.warning("Unexpected error fetching help: %s", exc)
                return None

            last_exit_code = exit_code
            last_stdout = stdout
            last_stderr = stderr
            combined = (stdout + "\n" + stderr).strip()

            if exit_code == 0 and combined:
                return _HelpProbeResult(
                    help_text=combined,
                    executed_command=candidate,
                )

            if combined and _looks_like_help_output(combined):
                candidate_help = _HelpProbeResult(
                    help_text=combined,
                    executed_command=candidate,
                )
                logger.debug(
                    "Using non-zero help output from %s (flag=%s, exit=%d)",
                    candidate,
                    flag,
                    exit_code,
                )
                break

        if candidate_help is not None:
            return candidate_help

        if attempted_help_commands:
            last_probe_error = DiscoveryProbeError(
                executed_command=candidate,
                attempted_help_commands=tuple(attempted_help_commands),
                exit_code=last_exit_code,
                stdout_text=last_stdout,
                stderr_text=last_stderr,
            )

        if len(candidates) > 1:
            logger.debug(
                "Discovery candidate failed without usable help output, "
                "trying next candidate: %s",
                candidate,
            )

    if last_probe_error is not None:
        raise last_probe_error

    return None


# ---------------------------------------------------------------------------
# LLM parsing
# ---------------------------------------------------------------------------


def _parse_llm_response(raw_content: str) -> dict[str, Any] | None:
    """Extract JSON from the LLM response, tolerating markdown fences."""
    text = raw_content.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON response")
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


async def _parse_help_with_llm(
    help_text: str,
    llm_client: Any,
    llm_model: str,
) -> DiscoveredTestSpec | None:
    """Send help text to the LLM and parse the structured response."""
    prompt = _LLM_PROMPT.format(help_text=help_text)

    try:
        from jules_daemon.llm.client import create_completion
        from jules_daemon.llm.config import LLMConfig

        # The llm_client is an OpenAI instance; llm_model may be a full
        # config or just a model string.  We build a minimal config-like
        # object for create_completion.
        if isinstance(llm_model, LLMConfig):
            config = llm_model
            model = config.default_model
        else:
            # Caller passed a model string -- we need the config too.
            # This path is used when the caller passes both separately.
            config = llm_model  # type: ignore[assignment]
            model = None

        response = await asyncio.to_thread(
            create_completion,
            client=llm_client,
            config=config,
            messages=[{"role": "user", "content": prompt}],
            model=model,
        )
    except Exception as exc:
        logger.warning("LLM call failed during discovery: %s", exc)
        return None

    if not response.choices:
        return None

    content = response.choices[0].message.content or ""
    parsed = _parse_llm_response(content)
    if parsed is None:
        return None

    return DiscoveredTestSpec(
        command_template=str(parsed.get("command_template", "")),
        required_args=tuple(parsed.get("required_args", ())),
        optional_args=tuple(parsed.get("optional_args", ())),
        arg_descriptions={
            str(k): str(v)
            for k, v in (parsed.get("arg_descriptions") or {}).items()
        },
        typical_duration=(
            int(parsed["typical_duration"])
            if parsed.get("typical_duration") is not None
            else None
        ),
        raw_help_text=help_text,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def discover_test(
    *,
    host: str,
    user: str,
    command: str,
    port: int = 22,
    llm_client: Any | None = None,
    llm_config: Any | None = None,
) -> DiscoveredTestSpec | None:
    """SSH in, run -h, parse with LLM, return structured spec.

    When no LLM is configured, returns a minimal spec with just the
    raw help text so the caller can display it for manual review.

    Args:
        host: Remote hostname or IP.
        user: SSH username.
        command: The command to discover (without -h flag).
        port: SSH port (default 22).
        llm_client: Optional OpenAI client for LLM parsing.
        llm_config: Optional LLMConfig for the LLM call.

    Returns:
        DiscoveredTestSpec on success, None when the remote probe produced
        no usable help output.

    Raises:
        SSHAuthenticationError: When SSH authentication fails.
        SSHConnectionError: When the SSH connection itself fails.
        DiscoveryProbeError: When the remote help probe ran but failed.
    """
    credential = resolve_ssh_credentials(host)
    help_result = await _fetch_help_text(
        host=host,
        port=port,
        username=user,
        credential=credential,
        command=command,
    )
    normalized_command = normalize_discovery_command(command)

    if help_result is None:
        logger.warning("Could not fetch help text for %s on %s", command, host)
        return None
    help_text = help_result.help_text
    effective_command = help_result.executed_command or normalized_command

    # Try LLM parsing if available
    if llm_client is not None and llm_config is not None:
        spec = await _parse_help_with_llm(help_text, llm_client, llm_config)
        if spec is not None:
            normalized_template = normalize_discovery_command(
                spec.command_template or effective_command
            )
            if normalized_template != spec.command_template:
                spec = replace(spec, command_template=normalized_template)
            return spec
        logger.warning("LLM parsing failed, returning raw help text only")

    # Fallback: return raw spec without structured extraction
    return DiscoveredTestSpec(
        command_template=effective_command,
        required_args=(),
        optional_args=(),
        arg_descriptions={},
        typical_duration=None,
        raw_help_text=help_text,
    )


def format_spec_preview(spec: DiscoveredTestSpec) -> str:
    """Format a discovered spec as a human-readable preview string.

    Args:
        spec: The spec to format.

    Returns:
        Multi-line string suitable for display in the CLI.
    """
    lines: list[str] = [
        "Discovered test spec:",
        "",
        f"  Command: {spec.command_template}",
    ]
    if spec.required_args:
        lines.append(f"  Required args: {', '.join(spec.required_args)}")
    if spec.optional_args:
        lines.append(f"  Optional args: {', '.join(spec.optional_args)}")
    if spec.arg_descriptions:
        lines.append("  Arg descriptions:")
        for name, desc in spec.arg_descriptions.items():
            lines.append(f"    {name}: {desc}")
    if spec.typical_duration is not None:
        lines.append(f"  Typical duration: {spec.typical_duration}s")
    return "\n".join(lines)


def save_discovered_spec(
    wiki_root: Path,
    spec: DiscoveredTestSpec,
    command: str,
    host: str,
) -> Path:
    """Write the spec to wiki as a test catalog page.

    Uses the existing ``KNOWLEDGE_DIR`` path and ``derive_test_slug``
    from the test_knowledge module for consistent placement.

    Args:
        wiki_root: Path to the wiki root directory.
        spec: The discovered spec to persist.
        command: Original command string (for slug derivation).
        host: Remote host where the command was discovered.

    Returns:
        Absolute path to the written wiki file.
    """
    slug = derive_test_slug(command)
    file_path = wiki_root / KNOWLEDGE_DIR / f"test-{slug}.md"

    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract a human-readable name from the command
    # e.g. "python3.8 /root/tests/ld_test.py" -> "ld_test"
    import os as _os
    name_parts = command.strip().split()
    raw_name = name_parts[-1] if name_parts else slug
    # Strip path and extension
    raw_name = _os.path.splitext(_os.path.basename(raw_name))[0]

    fm: dict[str, Any] = {
        "tags": list(_WIKI_TAGS),
        "type": _WIKI_TYPE,
        "name": raw_name,
        "test_slug": slug,
        "command_pattern": spec.command_template,
        "command_template": spec.command_template,
        "required_args": list(spec.required_args),
        "optional_args": list(spec.optional_args),
        "arg_descriptions": dict(spec.arg_descriptions),
        "typical_duration": spec.typical_duration,
        "discovered_from_host": host,
    }

    body_lines = [
        f"# Test Spec: {slug}",
        "",
        "*Auto-discovered via `discover` command.*",
        "",
        "## Command Template",
        "",
        "```bash",
        spec.command_template,
        "```",
        "",
    ]

    if spec.required_args:
        body_lines.extend(["## Required Arguments", ""])
        for arg in spec.required_args:
            desc = spec.arg_descriptions.get(arg, "")
            suffix = f" -- {desc}" if desc else ""
            body_lines.append(f"- `{arg}`{suffix}")
        body_lines.append("")

    if spec.optional_args:
        body_lines.extend(["## Optional Arguments", ""])
        for arg in spec.optional_args:
            desc = spec.arg_descriptions.get(arg, "")
            suffix = f" -- {desc}" if desc else ""
            body_lines.append(f"- `{arg}`{suffix}")
        body_lines.append("")

    if spec.raw_help_text:
        body_lines.extend([
            "## Raw Help Output",
            "",
            "```",
            spec.raw_help_text,
            "```",
            "",
        ])

    body = "\n".join(body_lines)
    doc = WikiDocument(frontmatter=fm, body=body)
    content = frontmatter.serialize(doc)

    tmp_path = file_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(file_path))

    logger.info("Saved discovered test spec for %s -> %s", slug, file_path)
    return file_path
