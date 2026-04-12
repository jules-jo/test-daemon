"""Direct-command detector for agent loop bypass.

Identifies user inputs that start with known shell executables and
returns a bypass flag so the agent loop can be skipped. When input
is a direct command (e.g., ``pytest -v tests/``), there is no need
for LLM interpretation -- the command can proceed straight to the
SSH approval flow.

The detector distinguishes between:
    - **Direct commands**: Inputs starting with known executables
      (pytest, python3, npm, cargo, make, etc.) or absolute/relative
      paths to executables. These bypass the agent loop.
    - **Daemon verbs**: Inputs starting with daemon verb aliases
      (status, watch, cancel, etc.). These are NOT direct commands --
      they are handled by the daemon's verb pipeline.
    - **Natural language**: Conversational inputs, questions, polite
      requests. These require LLM interpretation via the agent loop.

The detection strips environment variable prefixes (``VAR=value``)
and sudo prefixes before checking the executable. It also handles
``./`` relative paths and ``/absolute/path`` executables.

All results are immutable frozen dataclasses. The function never
raises exceptions for any input and never performs I/O.

Usage::

    from jules_daemon.classifier.direct_command import detect_direct_command

    detection = detect_direct_command("pytest -v tests/")
    if detection.bypass_agent_loop:
        # Skip agent loop, go directly to SSH approval flow
        ...
    else:
        # Enter the agent loop for NL interpretation
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from jules_daemon.classifier.verb_registry import VERB_ALIASES

__all__ = [
    "DEFAULT_KNOWN_EXECUTABLES",
    "DirectCommandDetection",
    "detect_direct_command",
]


# ---------------------------------------------------------------------------
# Known executables registry
# ---------------------------------------------------------------------------

# Test runners and language runtimes
_TEST_RUNNERS: frozenset[str] = frozenset({
    "pytest", "python", "python3", "python3.8", "python3.9",
    "python3.10", "python3.11", "python3.12", "python3.13",
    "npm", "npx", "yarn", "pnpm",
    "cargo",
    "go",
    "make", "cmake",
    "gradle", "gradlew",
    "mvn", "mvnw",
    "ant",
    "dotnet",
    "node", "deno", "bun",
    "java", "javac",
    "ruby", "rake", "rspec", "bundle",
    "perl",
    "php", "phpunit",
    "tox", "nox",
    "bazel",
    "jest", "mocha", "vitest",
})

# Shell utilities commonly used in remote commands
_SHELL_UTILITIES: frozenset[str] = frozenset({
    "bash", "sh", "zsh", "ksh", "dash",
    "ls", "cat", "grep", "find", "tail", "head",
    "wc", "sort", "awk", "sed", "tr", "cut",
    "echo", "printf", "tee",
    "cd", "pwd", "mkdir", "rm", "cp", "mv", "chmod", "chown",
    "ps", "top", "htop", "kill", "pkill",
    "curl", "wget",
    "tar", "gzip", "gunzip", "zip", "unzip",
    "df", "du", "free",
    "env", "export", "source",
    "date", "time", "timeout",
    "xargs",
    "diff",
    "touch",
    "ln",
    "stat",
    "file",
})

# Container and orchestration tools
_CONTAINER_TOOLS: frozenset[str] = frozenset({
    "docker", "podman",
    "kubectl", "helm",
    "docker-compose",
})

# System administration
_SYSADMIN_TOOLS: frozenset[str] = frozenset({
    "systemctl", "service", "journalctl",
    "apt", "apt-get", "yum", "dnf", "pacman",
    "pip", "pip3",
    "git", "svn",
    "ssh", "scp", "rsync",
    "crontab", "at",
})

DEFAULT_KNOWN_EXECUTABLES: frozenset[str] = (
    _TEST_RUNNERS | _SHELL_UTILITIES | _CONTAINER_TOOLS | _SYSADMIN_TOOLS
)
"""Immutable set of known executables for direct-command detection.

This set intentionally excludes daemon verbs (status, watch, cancel,
history, queue) and their aliases. Daemon verbs are handled by the
daemon's own verb pipeline, not by direct SSH execution.
"""


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Matches VAR=value or VAR="value" or VAR='value' tokens at the start
_ENV_PREFIX_RE: re.Pattern[str] = re.compile(
    r"""^[A-Za-z_][A-Za-z0-9_]*=  # key=
    (?:                             # value is one of:
        '[^']*'                     #   single-quoted
        |"[^"]*"                    #   double-quoted
        |\S*                        #   unquoted (no spaces)
    )
    \s+                             # followed by whitespace
    """,
    re.VERBOSE,
)

# Matches sudo with optional flags like -u username, -E, etc.
_SUDO_PREFIX_RE: re.Pattern[str] = re.compile(
    r"""^sudo\s+           # 'sudo' keyword
    (?:-[A-Za-z]\s+        # optional single-char flag
       (?:\S+\s+)?         # optional flag argument
    )*                     # zero or more flags
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectCommandDetection:
    """Immutable result of direct-command detection.

    When ``is_direct_command`` is True, the input starts with a
    recognized executable and can bypass the agent loop, going
    directly to the SSH command approval flow.

    Attributes:
        is_direct_command: True if the input starts with a known
            executable. This is the bypass flag.
        executable: The detected executable name (lowercase),
            or None if not a direct command.
        raw_command: The original input string (unmodified).
        confidence: Detection confidence (1.0 for exact match of
            a known executable, 0.8 for absolute-path match,
            0.0 for non-detection).
    """

    is_direct_command: bool
    executable: str | None
    raw_command: str
    confidence: float

    def __post_init__(self) -> None:
        if self.is_direct_command and self.executable is None:
            raise ValueError(
                "executable must be set when is_direct_command is True"
            )
        if not self.is_direct_command and self.executable is not None:
            raise ValueError(
                "executable must be None when is_direct_command is False"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    @property
    def bypass_agent_loop(self) -> bool:
        """Convenience alias: True when the agent loop should be skipped.

        Equivalent to ``is_direct_command``. Named for clarity at the
        call site where the consumer cares about the routing decision.
        """
        return self.is_direct_command

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for wiki YAML or IPC.

        Returns a new dict each call to preserve immutability.
        """
        return {
            "is_direct_command": self.is_direct_command,
            "executable": self.executable,
            "raw_command": self.raw_command,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Non-detection sentinel (module-level, reused for performance)
# ---------------------------------------------------------------------------

_NO_DETECTION_TEMPLATE = DirectCommandDetection(
    is_direct_command=False,
    executable=None,
    raw_command="",
    confidence=0.0,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_env_prefixes(text: str) -> str:
    """Strip leading VAR=value environment variable prefixes.

    Iteratively removes env prefixes from the start of the string
    so that commands like ``PYTHONPATH=/opt/app LANG=C pytest -v``
    are reduced to ``pytest -v``.

    Args:
        text: Stripped input string.

    Returns:
        Input with leading env prefixes removed.
    """
    result = text
    while True:
        match = _ENV_PREFIX_RE.match(result)
        if match is None:
            break
        result = result[match.end():]
    return result


def _strip_sudo_prefix(text: str) -> str:
    """Strip a leading 'sudo' prefix with optional flags.

    Handles forms like:
        sudo pytest -v
        sudo -u testuser python3 test.py
        sudo -E make test

    Args:
        text: Input after env prefix stripping.

    Returns:
        Input with sudo prefix removed (if present).
    """
    match = _SUDO_PREFIX_RE.match(text)
    if match is not None:
        return text[match.end():]
    return text


def _extract_executable_from_token(token: str) -> str:
    """Extract the executable name from a token.

    Handles:
        - Bare names: ``pytest`` -> ``pytest``
        - Relative paths: ``./gradlew`` -> ``gradlew``
        - Absolute paths: ``/usr/bin/python3`` -> ``python3``

    Args:
        token: First non-prefix token from the input.

    Returns:
        Lowercase executable name.
    """
    # Strip ./ prefix
    if token.startswith("./"):
        token = token[2:]

    # Extract basename from absolute paths
    if "/" in token:
        token = token.rsplit("/", maxsplit=1)[-1]

    return token.lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_direct_command(
    raw: str,
    *,
    known_executables: frozenset[str] | None = None,
) -> DirectCommandDetection:
    """Detect whether input is a direct shell command.

    Analyzes the first token of the input (after stripping environment
    variable prefixes and sudo) to determine if it matches a known
    executable. If it does, returns a detection with ``is_direct_command=True``
    and the ``bypass_agent_loop`` property set.

    Daemon verbs and their aliases are explicitly excluded: they are
    handled by the daemon's verb pipeline, not by direct SSH execution.

    This function never raises exceptions for any input and never
    performs I/O. It is designed to be the first gate before the
    agent loop, running in sub-millisecond time.

    Args:
        raw: Raw user input string.
        known_executables: Optional custom set of executable names to
            recognize. If None, uses ``DEFAULT_KNOWN_EXECUTABLES``.
            All names should be lowercase.

    Returns:
        Immutable DirectCommandDetection. Check ``bypass_agent_loop``
        or ``is_direct_command`` to determine the routing decision.
    """
    executables = (
        known_executables if known_executables is not None
        else DEFAULT_KNOWN_EXECUTABLES
    )

    stripped = raw.strip()
    if not stripped:
        return DirectCommandDetection(
            is_direct_command=False,
            executable=None,
            raw_command=raw,
            confidence=0.0,
        )

    # Step 1: Strip environment variable prefixes
    after_env = _strip_env_prefixes(stripped)

    # Step 2: Strip sudo prefix
    after_sudo = _strip_sudo_prefix(after_env)

    # Step 3: Extract the first token
    first_token = after_sudo.split()[0] if after_sudo.strip() else ""
    if not first_token:
        return DirectCommandDetection(
            is_direct_command=False,
            executable=None,
            raw_command=raw,
            confidence=0.0,
        )

    # Step 4: Extract the executable name
    executable = _extract_executable_from_token(first_token)

    if not executable:
        return DirectCommandDetection(
            is_direct_command=False,
            executable=None,
            raw_command=raw,
            confidence=0.0,
        )

    # Step 5: Check if it is a daemon verb alias (exclude these)
    if executable in VERB_ALIASES:
        return DirectCommandDetection(
            is_direct_command=False,
            executable=None,
            raw_command=raw,
            confidence=0.0,
        )

    # Step 6: Check against known executables
    if executable in executables:
        # Determine confidence based on match type
        is_absolute = first_token.startswith("/")
        confidence = 0.8 if is_absolute else 1.0

        return DirectCommandDetection(
            is_direct_command=True,
            executable=executable,
            raw_command=raw,
            confidence=confidence,
        )

    # Step 7: Check if first token is an absolute path (any executable)
    if first_token.startswith("/"):
        return DirectCommandDetection(
            is_direct_command=True,
            executable=executable,
            raw_command=raw,
            confidence=0.8,
        )

    # Step 8: Check if first token is a relative path with ./
    original_first_token = after_sudo.split()[0] if after_sudo.strip() else ""
    if original_first_token.startswith("./"):
        return DirectCommandDetection(
            is_direct_command=True,
            executable=executable,
            raw_command=raw,
            confidence=0.8,
        )

    # Not a direct command
    return DirectCommandDetection(
        is_direct_command=False,
        executable=None,
        raw_command=raw,
        confidence=0.0,
    )
