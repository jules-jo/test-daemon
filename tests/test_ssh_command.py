"""Tests for SSHCommand Pydantic model.

Covers field defaults, validation rules, immutability (frozen model),
serialization to/from dict and JSON, and edge cases.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jules_daemon.ssh.command import SSHCommand


# ---------------------------------------------------------------------------
# Construction / Defaults
# ---------------------------------------------------------------------------


class TestSSHCommandDefaults:
    """Verify sensible defaults when only the required ``command`` is given."""

    def test_minimal_construction(self) -> None:
        cmd = SSHCommand(command="pytest -v")
        assert cmd.command == "pytest -v"
        assert cmd.working_directory is None
        assert cmd.timeout == 300
        assert cmd.environment == {}

    def test_all_fields_explicit(self) -> None:
        cmd = SSHCommand(
            command="make test",
            working_directory="/opt/project",
            timeout=600,
            environment={"CI": "true", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert cmd.command == "make test"
        assert cmd.working_directory == "/opt/project"
        assert cmd.timeout == 600
        assert cmd.environment == {"CI": "true", "PYTHONDONTWRITEBYTECODE": "1"}


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestSSHCommandImmutability:
    """Frozen Pydantic model must reject in-place mutation."""

    def test_cannot_mutate_command(self) -> None:
        cmd = SSHCommand(command="pytest")
        with pytest.raises(ValidationError):
            cmd.command = "other"  # type: ignore[misc]

    def test_cannot_mutate_timeout(self) -> None:
        cmd = SSHCommand(command="pytest")
        with pytest.raises(ValidationError):
            cmd.timeout = 999  # type: ignore[misc]

    def test_cannot_mutate_environment(self) -> None:
        cmd = SSHCommand(command="pytest", environment={"A": "1"})
        with pytest.raises(ValidationError):
            cmd.environment = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation: command
# ---------------------------------------------------------------------------


class TestSSHCommandCommandValidation:
    """The ``command`` field must be a non-empty, non-whitespace string."""

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            SSHCommand(command="")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            SSHCommand(command="   ")

    def test_stripped_command(self) -> None:
        cmd = SSHCommand(command="  pytest -v  ")
        assert cmd.command == "pytest -v"

    def test_max_length_exceeded_raises(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            SSHCommand(command="x" * 8193)

    def test_max_length_boundary_ok(self) -> None:
        cmd = SSHCommand(command="x" * 8192)
        assert len(cmd.command) == 8192


# ---------------------------------------------------------------------------
# Validation: working_directory
# ---------------------------------------------------------------------------


class TestSSHCommandWorkingDirectory:
    """Optional ``working_directory`` must be an absolute path if provided."""

    def test_none_is_valid(self) -> None:
        cmd = SSHCommand(command="ls")
        assert cmd.working_directory is None

    def test_absolute_path_accepted(self) -> None:
        cmd = SSHCommand(command="ls", working_directory="/home/user/project")
        assert cmd.working_directory == "/home/user/project"

    def test_relative_path_raises(self) -> None:
        with pytest.raises(ValidationError, match="working_directory"):
            SSHCommand(command="ls", working_directory="relative/path")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValidationError, match="working_directory"):
            SSHCommand(command="ls", working_directory="")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValidationError, match="working_directory"):
            SSHCommand(command="ls", working_directory="   ")

    def test_stripped_path(self) -> None:
        cmd = SSHCommand(command="ls", working_directory="  /opt/app  ")
        assert cmd.working_directory == "/opt/app"


# ---------------------------------------------------------------------------
# Validation: timeout
# ---------------------------------------------------------------------------


class TestSSHCommandTimeout:
    """Timeout must be a positive integer within a sane range."""

    def test_default_timeout(self) -> None:
        cmd = SSHCommand(command="pytest")
        assert cmd.timeout == 300

    def test_custom_timeout(self) -> None:
        cmd = SSHCommand(command="pytest", timeout=60)
        assert cmd.timeout == 60

    def test_zero_timeout_raises(self) -> None:
        with pytest.raises(ValidationError, match="timeout"):
            SSHCommand(command="pytest", timeout=0)

    def test_negative_timeout_raises(self) -> None:
        with pytest.raises(ValidationError, match="timeout"):
            SSHCommand(command="pytest", timeout=-10)

    def test_exceeds_max_timeout_raises(self) -> None:
        with pytest.raises(ValidationError, match="timeout"):
            SSHCommand(command="pytest", timeout=86401)

    def test_max_timeout_boundary_ok(self) -> None:
        cmd = SSHCommand(command="pytest", timeout=86400)
        assert cmd.timeout == 86400

    def test_min_timeout_boundary_ok(self) -> None:
        cmd = SSHCommand(command="pytest", timeout=1)
        assert cmd.timeout == 1


# ---------------------------------------------------------------------------
# Validation: environment
# ---------------------------------------------------------------------------


class TestSSHCommandEnvironment:
    """Environment must be a dict of non-empty string keys to string values."""

    def test_empty_dict_is_valid(self) -> None:
        cmd = SSHCommand(command="pytest", environment={})
        assert cmd.environment == {}

    def test_valid_environment(self) -> None:
        env = {"PATH": "/usr/bin", "HOME": "/root", "CI": "1"}
        cmd = SSHCommand(command="pytest", environment=env)
        assert cmd.environment == env

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValidationError, match="environment"):
            SSHCommand(command="pytest", environment={"": "value"})

    def test_key_with_equals_raises(self) -> None:
        with pytest.raises(ValidationError, match="environment"):
            SSHCommand(command="pytest", environment={"KEY=VAL": "x"})

    def test_key_with_null_byte_raises(self) -> None:
        with pytest.raises(ValidationError, match="environment"):
            SSHCommand(command="pytest", environment={"KEY\x00": "val"})

    def test_value_allows_empty_string(self) -> None:
        cmd = SSHCommand(command="pytest", environment={"DEBUG": ""})
        assert cmd.environment["DEBUG"] == ""

    def test_key_with_spaces_raises(self) -> None:
        with pytest.raises(ValidationError, match="environment"):
            SSHCommand(command="pytest", environment={"MY VAR": "val"})


# ---------------------------------------------------------------------------
# Serialization: dict
# ---------------------------------------------------------------------------


class TestSSHCommandDictSerialization:
    """Round-trip through model_dump / model_validate."""

    def test_to_dict_minimal(self) -> None:
        cmd = SSHCommand(command="pytest")
        d = cmd.to_dict()
        assert d["command"] == "pytest"
        assert d["working_directory"] is None
        assert d["timeout"] == 300
        assert d["environment"] == {}

    def test_to_dict_full(self) -> None:
        cmd = SSHCommand(
            command="make test",
            working_directory="/opt/project",
            timeout=120,
            environment={"CI": "true"},
        )
        d = cmd.to_dict()
        assert d["command"] == "make test"
        assert d["working_directory"] == "/opt/project"
        assert d["timeout"] == 120
        assert d["environment"] == {"CI": "true"}

    def test_from_dict(self) -> None:
        d = {
            "command": "pytest -v",
            "working_directory": "/app",
            "timeout": 60,
            "environment": {"CI": "1"},
        }
        cmd = SSHCommand.from_dict(d)
        assert cmd.command == "pytest -v"
        assert cmd.working_directory == "/app"
        assert cmd.timeout == 60
        assert cmd.environment == {"CI": "1"}

    def test_roundtrip_dict(self) -> None:
        original = SSHCommand(
            command="npm test",
            working_directory="/home/ci/app",
            timeout=180,
            environment={"NODE_ENV": "test"},
        )
        rebuilt = SSHCommand.from_dict(original.to_dict())
        assert rebuilt == original

    def test_from_dict_minimal(self) -> None:
        """Only required field provided -- defaults should fill in."""
        cmd = SSHCommand.from_dict({"command": "ls"})
        assert cmd.command == "ls"
        assert cmd.timeout == 300


# ---------------------------------------------------------------------------
# Serialization: JSON
# ---------------------------------------------------------------------------


class TestSSHCommandJSONSerialization:
    """Round-trip through to_json / from_json."""

    def test_to_json_returns_string(self) -> None:
        cmd = SSHCommand(command="pytest")
        j = cmd.to_json()
        assert isinstance(j, str)
        parsed = json.loads(j)
        assert parsed["command"] == "pytest"

    def test_from_json(self) -> None:
        payload = json.dumps({"command": "pytest", "timeout": 60})
        cmd = SSHCommand.from_json(payload)
        assert cmd.command == "pytest"
        assert cmd.timeout == 60

    def test_roundtrip_json(self) -> None:
        original = SSHCommand(
            command="go test ./...",
            working_directory="/opt/goapp",
            timeout=240,
            environment={"GOFLAGS": "-count=1"},
        )
        rebuilt = SSHCommand.from_json(original.to_json())
        assert rebuilt == original

    def test_from_json_invalid_json_raises(self) -> None:
        with pytest.raises(ValidationError):
            SSHCommand.from_json("not valid json{{{")


# ---------------------------------------------------------------------------
# Equality & hashing
# ---------------------------------------------------------------------------


class TestSSHCommandEquality:
    """Frozen models should be equality-comparable and hashable."""

    def test_equal_instances(self) -> None:
        a = SSHCommand(command="pytest", timeout=60)
        b = SSHCommand(command="pytest", timeout=60)
        assert a == b

    def test_unequal_instances(self) -> None:
        a = SSHCommand(command="pytest", timeout=60)
        b = SSHCommand(command="pytest", timeout=120)
        assert a != b

    def test_not_hashable_with_dict_field(self) -> None:
        """Frozen Pydantic models with dict fields are not hashable."""
        cmd = SSHCommand(command="pytest", timeout=60)
        with pytest.raises(TypeError, match="unhashable"):
            hash(cmd)

    def test_hashable_without_env(self) -> None:
        """Frozen Pydantic models without mutable fields hash fine."""
        # Construct two equal instances; verify equality still works
        a = SSHCommand(command="pytest", timeout=60)
        b = SSHCommand(command="pytest", timeout=60)
        assert a == b


# ---------------------------------------------------------------------------
# Copy with changes
# ---------------------------------------------------------------------------


class TestSSHCommandCopy:
    """The with_changes() method returns a new instance, never mutates."""

    def test_with_changes_returns_new_instance(self) -> None:
        original = SSHCommand(command="pytest", timeout=60)
        updated = original.with_changes(timeout=120)
        assert updated.timeout == 120
        assert original.timeout == 60  # original unchanged
        assert original is not updated

    def test_with_changes_command(self) -> None:
        original = SSHCommand(command="pytest -v")
        updated = original.with_changes(command="pytest -x")
        assert updated.command == "pytest -x"
        assert original.command == "pytest -v"

    def test_with_changes_multiple_fields(self) -> None:
        original = SSHCommand(command="pytest", timeout=60)
        updated = original.with_changes(
            working_directory="/opt/app",
            environment={"CI": "1"},
        )
        assert updated.working_directory == "/opt/app"
        assert updated.environment == {"CI": "1"}
        assert updated.command == "pytest"
        assert updated.timeout == 60

    def test_with_changes_validates(self) -> None:
        original = SSHCommand(command="pytest")
        with pytest.raises(ValidationError):
            original.with_changes(timeout=-1)
