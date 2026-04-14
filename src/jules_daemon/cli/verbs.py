"""CLI verb grammar data models.

Defines typed, immutable dataclasses for the six CLI verbs that drive
daemon interaction:

    status  -- check current run state
    watch   -- live-stream output from a running test
    run     -- start test execution via natural-language command
    queue   -- queue a command when the daemon is busy
    cancel  -- cancel the current or queued run
    history -- view past test run results

Each verb has a corresponding ``*Args`` frozen dataclass that encodes
the expected argument schema and validation rules. A ``ParsedCommand``
composite binds a ``Verb`` to its validated arguments, enforcing that
the args type matches the verb at construction time.

All models follow the project immutability convention: frozen dataclasses
with ``__post_init__`` validation. State changes require creating new
instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from jules_daemon.wiki.models import SSHTarget

__all__ = [
    "CancelArgs",
    "DiscoverArgs",
    "HistoryArgs",
    "ParsedCommand",
    "QueueArgs",
    "RunArgs",
    "StatusArgs",
    "VALID_OUTPUT_FORMATS",
    "Verb",
    "WatchArgs",
    "parse_verb",
]


# ---------------------------------------------------------------------------
# Verb enum
# ---------------------------------------------------------------------------


class Verb(Enum):
    """The seven CLI verbs supported by the daemon.

    Each verb maps to a specific daemon operation and has a
    corresponding ``*Args`` dataclass that defines its argument schema.

    Values:
        STATUS:   Query the current run state.
        WATCH:    Live-stream output from a running test session.
        RUN:      Start a new test execution via natural-language command.
        QUEUE:    Queue a command for execution when the daemon is busy.
        CANCEL:   Cancel the current or a queued run.
        HISTORY:  Retrieve past test run results from the wiki.
        DISCOVER: Auto-discover a test spec by running command -h via SSH.
    """

    STATUS = "status"
    WATCH = "watch"
    RUN = "run"
    QUEUE = "queue"
    CANCEL = "cancel"
    HISTORY = "history"
    DISCOVER = "discover"


# Lookup table for case-insensitive parsing: lowered value -> Verb
_VERB_LOOKUP: dict[str, Verb] = {v.value: v for v in Verb}


def parse_verb(raw: str) -> Verb:
    """Parse a raw string into a Verb enum member.

    Matching is case-insensitive with leading/trailing whitespace
    stripped.

    Args:
        raw: User-supplied verb string (e.g., "run", "STATUS").

    Returns:
        The matching Verb member.

    Raises:
        ValueError: If the string is empty or does not match any verb.
    """
    normalized = raw.strip().lower()
    if not normalized:
        raise ValueError("Verb must not be empty")

    verb = _VERB_LOOKUP.get(normalized)
    if verb is None:
        valid = ", ".join(sorted(_VERB_LOOKUP))
        raise ValueError(
            f"Unknown verb {raw.strip()!r}. Valid verbs: {valid}"
        )

    return verb


# ---------------------------------------------------------------------------
# Allowed status filter values (must match RunStatus enum values)
# ---------------------------------------------------------------------------

_VALID_STATUS_FILTERS = frozenset({
    "idle",
    "pending_approval",
    "running",
    "completed",
    "failed",
    "cancelled",
})

# Maximum number of history records that can be requested
_MAX_HISTORY_LIMIT = 1000

# Default number of tail lines for the watch verb
_DEFAULT_TAIL_LINES = 50

# Default output format for the watch verb
_DEFAULT_OUTPUT_FORMAT = "text"

# Valid output formats for the watch verb
VALID_OUTPUT_FORMATS = frozenset({"text", "json", "summary"})

# Default number of history records
_DEFAULT_HISTORY_LIMIT = 20


# ---------------------------------------------------------------------------
# StatusArgs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusArgs:
    """Argument schema for the ``status`` verb.

    Attributes:
        verbose: When True, include extended details (PIDs, timestamps,
            progress breakdown). Default is False for a compact summary.
    """

    verbose: bool = False


# ---------------------------------------------------------------------------
# WatchArgs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WatchArgs:
    """Argument schema for the ``watch`` verb.

    Attributes:
        run_id: Target a specific run by ID. When None, watches the
            current active run.
        tail_lines: Number of recent output lines to show on initial
            attach. Must be positive.
        follow: When True, continuously stream new output as it arrives
            (like ``tail -f``). When False, show current output and exit.
        output_format: Presentation format for output. One of ``text``
            (human-readable), ``json`` (machine-parseable), or
            ``summary`` (condensed overview).
    """

    run_id: Optional[str] = None
    tail_lines: int = _DEFAULT_TAIL_LINES
    follow: bool = False
    output_format: str = _DEFAULT_OUTPUT_FORMAT

    def __post_init__(self) -> None:
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("run_id must not be empty when provided")
        if self.tail_lines < 1:
            raise ValueError(
                f"tail_lines must be positive, got {self.tail_lines}"
            )
        if self.output_format not in VALID_OUTPUT_FORMATS:
            valid = ", ".join(sorted(VALID_OUTPUT_FORMATS))
            raise ValueError(
                f"output_format must be one of {valid}, "
                f"got {self.output_format!r}"
            )


# ---------------------------------------------------------------------------
# RunArgs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunArgs:
    """Argument schema for the ``run`` verb.

    Encodes the SSH target and the natural-language command that the
    LLM agent will translate into a shell command. The user confirms
    the translated command via the confirmation flow before execution.

    Attributes:
        target_host: Remote hostname or IP address.
        target_user: SSH username on the remote host.
        natural_language: Free-form description of what tests to run
            (e.g., "run the full regression suite for payments").
        system_name: Optional named system alias defined in the wiki.
            When provided, the daemon resolves host/user/port from
            ``wiki/pages/systems`` and explicit target fields must be empty.
        target_port: SSH port on the remote host. Default is 22.
        key_path: Absolute path to the SSH private key file.
            None means use the SSH agent or default key.
    """

    target_host: str = ""
    target_user: str = ""
    natural_language: str = ""
    system_name: Optional[str] = None
    target_port: int = 22
    key_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.natural_language.strip():
            raise ValueError("natural_language must not be empty")
        has_system_name = self.system_name is not None and self.system_name.strip() != ""
        if has_system_name:
            if self.target_host.strip() or self.target_user.strip():
                raise ValueError(
                    "system_name cannot be combined with target_host or target_user"
                )
            if self.target_port != 22:
                raise ValueError("system_name cannot be combined with target_port")
            if self.key_path is not None:
                raise ValueError("system_name cannot be combined with key_path")
        else:
            if not self.target_host.strip():
                raise ValueError("target_host must not be empty")
            if not self.target_user.strip():
                raise ValueError("target_user must not be empty")
            if not (1 <= self.target_port <= 65535):
                raise ValueError(
                    f"target_port must be 1-65535, got {self.target_port}"
                )
            if self.key_path is not None and not self.key_path.startswith("/"):
                raise ValueError(
                    f"key_path must be an absolute path, got {self.key_path!r}"
                )

    def to_ssh_target(self) -> SSHTarget:
        """Convert to an SSHTarget for use with the wiki persistence layer.

        Returns:
            SSHTarget with matching host, user, port, and key_path.
        """
        if self.system_name is not None and self.system_name.strip():
            raise ValueError("Cannot build SSHTarget until system_name is resolved")
        return SSHTarget(
            host=self.target_host,
            user=self.target_user,
            port=self.target_port,
            key_path=self.key_path,
        )


# ---------------------------------------------------------------------------
# QueueArgs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueArgs:
    """Argument schema for the ``queue`` verb.

    Queues a command for later execution when the daemon is currently
    busy with another run. The daemon uses warn-and-allow collision
    detection, not hard blocking.

    Attributes:
        target_host: Remote hostname or IP address.
        target_user: SSH username on the remote host.
        natural_language: Free-form description of what tests to run.
        target_port: SSH port on the remote host. Default is 22.
        key_path: Absolute path to the SSH private key file.
        priority: Queue priority (higher values run first).
            Default is 0 (normal priority). Must not be negative.
    """

    target_host: str
    target_user: str
    natural_language: str
    target_port: int = 22
    key_path: Optional[str] = None
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.target_host.strip():
            raise ValueError("target_host must not be empty")
        if not self.target_user.strip():
            raise ValueError("target_user must not be empty")
        if not self.natural_language.strip():
            raise ValueError("natural_language must not be empty")
        if not (1 <= self.target_port <= 65535):
            raise ValueError(
                f"target_port must be 1-65535, got {self.target_port}"
            )
        if self.key_path is not None and not self.key_path.startswith("/"):
            raise ValueError(
                f"key_path must be an absolute path, got {self.key_path!r}"
            )
        if self.priority < 0:
            raise ValueError(
                f"priority must not be negative, got {self.priority}"
            )

    def to_ssh_target(self) -> SSHTarget:
        """Convert to an SSHTarget for use with the wiki persistence layer.

        Returns:
            SSHTarget with matching host, user, port, and key_path.
        """
        return SSHTarget(
            host=self.target_host,
            user=self.target_user,
            port=self.target_port,
            key_path=self.key_path,
        )


# ---------------------------------------------------------------------------
# CancelArgs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CancelArgs:
    """Argument schema for the ``cancel`` verb.

    Attributes:
        run_id: Target a specific run by ID. When None, cancels the
            current active run.
        force: When True, send SIGKILL instead of SIGTERM to the
            remote process. Default is False.
        reason: Optional human-readable reason for cancellation.
            Recorded in the audit log.
    """

    run_id: Optional[str] = None
    force: bool = False
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if self.run_id is not None and not self.run_id.strip():
            raise ValueError("run_id must not be empty when provided")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("reason must not be empty when provided")


# ---------------------------------------------------------------------------
# HistoryArgs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryArgs:
    """Argument schema for the ``history`` verb.

    Attributes:
        limit: Maximum number of records to return. Default is 20,
            maximum is 1000.
        status_filter: Filter by run status (must match a RunStatus
            value). None means all statuses.
        host_filter: Filter by SSH target hostname. None means all hosts.
        verbose: When True, include full details per record.
    """

    limit: int = _DEFAULT_HISTORY_LIMIT
    status_filter: Optional[str] = None
    host_filter: Optional[str] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError(f"limit must be positive, got {self.limit}")
        if self.limit > _MAX_HISTORY_LIMIT:
            raise ValueError(
                f"limit must not exceed {_MAX_HISTORY_LIMIT}, "
                f"got {self.limit}"
            )
        if self.status_filter is not None:
            stripped = self.status_filter.strip()
            if not stripped:
                raise ValueError("status_filter must not be empty when provided")
            if stripped not in _VALID_STATUS_FILTERS:
                valid = ", ".join(sorted(_VALID_STATUS_FILTERS))
                raise ValueError(
                    f"Invalid status_filter {stripped!r}. "
                    f"Valid values: {valid}"
                )
        if self.host_filter is not None and not self.host_filter.strip():
            raise ValueError("host_filter must not be empty when provided")


# ---------------------------------------------------------------------------
# DiscoverArgs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoverArgs:
    """Argument schema for the ``discover`` verb.

    Attributes:
        target_host: Remote hostname or IP address.
        target_user: SSH username on the remote host.
        command: The command to discover (run with -h).
        target_port: SSH port on the remote host. Default is 22.
    """

    target_host: str
    target_user: str
    command: str
    target_port: int = 22

    def __post_init__(self) -> None:
        if not self.target_host.strip():
            raise ValueError("target_host must not be empty")
        if not self.target_user.strip():
            raise ValueError("target_user must not be empty")
        if not self.command.strip():
            raise ValueError("command must not be empty")
        if not (1 <= self.target_port <= 65535):
            raise ValueError(
                f"target_port must be 1-65535, got {self.target_port}"
            )


# ---------------------------------------------------------------------------
# Verb -> Args type mapping
# ---------------------------------------------------------------------------

# Maps each Verb to the expected argument dataclass type.
_VERB_ARGS_TYPE: dict[Verb, type] = {
    Verb.STATUS: StatusArgs,
    Verb.WATCH: WatchArgs,
    Verb.RUN: RunArgs,
    Verb.QUEUE: QueueArgs,
    Verb.CANCEL: CancelArgs,
    Verb.HISTORY: HistoryArgs,
    Verb.DISCOVER: DiscoverArgs,
}

# Union type for all argument dataclasses
VerbArgs = Union[
    StatusArgs, WatchArgs, RunArgs, QueueArgs, CancelArgs, HistoryArgs,
    DiscoverArgs,
]


# ---------------------------------------------------------------------------
# ParsedCommand
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedCommand:
    """A validated verb + arguments pair.

    Enforces at construction time that the ``args`` type matches the
    expected type for the given ``verb``. This is the canonical IPC
    message that the CLI sends to the daemon.

    Attributes:
        verb: The CLI verb (status, watch, run, queue, cancel, history, discover).
        args: Validated argument dataclass matching the verb.
    """

    verb: Verb
    args: VerbArgs

    def __post_init__(self) -> None:
        expected_type = _VERB_ARGS_TYPE[self.verb]
        actual_type = type(self.args)
        if actual_type is not expected_type:
            raise ValueError(
                f"Expected {expected_type.__name__} for verb "
                f"{self.verb.value!r}, got {actual_type.__name__}"
            )
