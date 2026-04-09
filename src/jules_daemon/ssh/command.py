"""Pydantic model for a validated SSH command.

Represents a single command to be executed on a remote host over SSH,
including working directory, execution timeout, and environment
variable overrides.

The model is frozen (immutable). State changes produce new instances
via ``with_changes()`` -- matching the project-wide immutability
convention.

Serialization helpers (``to_dict``, ``from_dict``, ``to_json``,
``from_json``) bridge this model to the wiki YAML persistence layer.
"""

from __future__ import annotations

import re
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

__all__ = [
    "SSHCommand",
    "DEFAULT_TIMEOUT",
    "MAX_TIMEOUT",
    "MAX_COMMAND_LENGTH",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: int = 300
"""Default command timeout in seconds (5 minutes)."""

MAX_TIMEOUT: int = 86400
"""Maximum allowed timeout in seconds (24 hours)."""

MAX_COMMAND_LENGTH: int = 8192
"""Maximum allowed length for the command string."""

# Pre-compiled pattern for valid environment variable names.
# POSIX: uppercase/lowercase letters, digits, underscores; must not start
# with a digit. We also forbid null bytes, equals signs, and spaces.
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SSHCommand(BaseModel):
    """Validated, immutable representation of a remote SSH command.

    Attributes:
        command: Shell command string to execute on the remote host.
            Stripped of leading/trailing whitespace. Must not be empty
            or exceed ``MAX_COMMAND_LENGTH`` characters.
        working_directory: Absolute path for ``cd`` before execution.
            ``None`` means the remote user's home directory. Must start
            with ``/`` when provided.
        timeout: Maximum execution time in seconds (1 -- ``MAX_TIMEOUT``).
            Defaults to ``DEFAULT_TIMEOUT`` (300 s / 5 min).
        environment: Key-value mapping of environment variables to set
            before execution. Keys must be valid POSIX identifiers.
    """

    model_config = ConfigDict(frozen=True)

    command: str
    working_directory: str | None = None
    timeout: int = DEFAULT_TIMEOUT
    environment: dict[str, str] = {}

    # -- Field validators (run before model_validator) --

    @field_validator("command", mode="before")
    @classmethod
    def _strip_and_validate_command(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("command must be a string")
        stripped = value.strip()
        if not stripped:
            raise ValueError("command must not be empty or whitespace-only")
        if len(stripped) > MAX_COMMAND_LENGTH:
            raise ValueError(
                f"command must not exceed {MAX_COMMAND_LENGTH} characters, "
                f"got {len(stripped)}"
            )
        return stripped

    @field_validator("working_directory", mode="before")
    @classmethod
    def _strip_and_validate_working_directory(
        cls, value: str | None
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("working_directory must be a string or None")
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "working_directory must not be empty or whitespace-only"
            )
        if not stripped.startswith("/"):
            raise ValueError(
                f"working_directory must be an absolute path (start with /), "
                f"got {stripped!r}"
            )
        return stripped

    @field_validator("timeout", mode="before")
    @classmethod
    def _validate_timeout(cls, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("timeout must be an integer")
        if value < 1:
            raise ValueError(
                f"timeout must be at least 1 second, got {value}"
            )
        if value > MAX_TIMEOUT:
            raise ValueError(
                f"timeout must not exceed {MAX_TIMEOUT} seconds, got {value}"
            )
        return value

    @model_validator(mode="after")
    def _validate_environment_keys(self) -> Self:
        """Validate that all environment variable keys are POSIX-safe."""
        for key in self.environment:
            if not _ENV_KEY_RE.match(key):
                raise ValueError(
                    f"environment key {key!r} is not a valid POSIX "
                    f"identifier (letters, digits, underscores; must not "
                    f"start with a digit)"
                )
        return self

    # -- Serialization helpers --

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for YAML/wiki persistence."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SSHCommand:
        """Deserialize from a plain dict (e.g., parsed from wiki YAML)."""
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> SSHCommand:
        """Deserialize from a JSON string."""
        return cls.model_validate_json(payload)

    # -- Immutable copy --

    def with_changes(self, **kwargs: Any) -> SSHCommand:
        """Return a new SSHCommand with the specified fields replaced.

        The original instance is never mutated.  All validation rules
        are re-applied on the resulting instance.
        """
        current = self.model_dump()
        current.update(kwargs)
        return SSHCommand.model_validate(current)
