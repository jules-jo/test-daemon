"""Tests for SSH error hierarchy and error classification.

Covers:
    - Error class inheritance chain
    - SSHReconnectionExhaustedError attributes
    - is_transient() classification for all known transient error types
    - is_permanent() classification for all known permanent error types
    - Unclassified errors return False for both
"""

from __future__ import annotations

import pytest

from jules_daemon.ssh.errors import (
    SSHAuthenticationError,
    SSHConnectionError,
    SSHError,
    SSHHostKeyError,
    SSHReconnectionExhaustedError,
    is_permanent,
    is_transient,
)


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestSSHErrorInheritance:
    """All SSH errors must inherit from SSHError."""

    def test_base_error_is_exception(self) -> None:
        assert issubclass(SSHError, Exception)

    def test_connection_error_inherits_from_ssh_error(self) -> None:
        assert issubclass(SSHConnectionError, SSHError)

    def test_authentication_error_inherits_from_ssh_error(self) -> None:
        assert issubclass(SSHAuthenticationError, SSHError)

    def test_host_key_error_inherits_from_ssh_error(self) -> None:
        assert issubclass(SSHHostKeyError, SSHError)

    def test_reconnection_exhausted_inherits_from_ssh_error(self) -> None:
        assert issubclass(SSHReconnectionExhaustedError, SSHError)


# ---------------------------------------------------------------------------
# SSHReconnectionExhaustedError attributes
# ---------------------------------------------------------------------------


class TestSSHReconnectionExhaustedError:
    """Verify extra attributes on the exhaustion error."""

    def test_message_accessible(self) -> None:
        exc = SSHReconnectionExhaustedError("all retries used")
        assert str(exc) == "all retries used"

    def test_default_attributes(self) -> None:
        exc = SSHReconnectionExhaustedError("fail")
        assert exc.attempts == 0
        assert exc.last_error is None

    def test_custom_attributes(self) -> None:
        exc = SSHReconnectionExhaustedError(
            "exhausted",
            attempts=5,
            last_error="Connection refused",
        )
        assert exc.attempts == 5
        assert exc.last_error == "Connection refused"

    def test_catchable_as_ssh_error(self) -> None:
        with pytest.raises(SSHError):
            raise SSHReconnectionExhaustedError("boom")


# ---------------------------------------------------------------------------
# is_transient() classification
# ---------------------------------------------------------------------------


class TestIsTransient:
    """Verify transient error classification."""

    def test_ssh_connection_error(self) -> None:
        assert is_transient(SSHConnectionError("conn refused")) is True

    def test_os_error(self) -> None:
        assert is_transient(OSError("Network unreachable")) is True

    def test_connection_refused(self) -> None:
        assert is_transient(ConnectionRefusedError()) is True

    def test_connection_reset(self) -> None:
        assert is_transient(ConnectionResetError()) is True

    def test_connection_aborted(self) -> None:
        assert is_transient(ConnectionAbortedError()) is True

    def test_timeout_error(self) -> None:
        assert is_transient(TimeoutError("timed out")) is True

    def test_broken_pipe(self) -> None:
        assert is_transient(BrokenPipeError()) is True

    def test_eof_error(self) -> None:
        assert is_transient(EOFError()) is True

    def test_connection_error_base(self) -> None:
        assert is_transient(ConnectionError("generic")) is True

    # Negative cases
    def test_ssh_auth_error_is_not_transient(self) -> None:
        assert is_transient(SSHAuthenticationError("bad key")) is False

    def test_ssh_host_key_error_is_not_transient(self) -> None:
        assert is_transient(SSHHostKeyError("mismatch")) is False

    def test_value_error_is_not_transient(self) -> None:
        assert is_transient(ValueError("bad value")) is False

    def test_runtime_error_is_not_transient(self) -> None:
        assert is_transient(RuntimeError("something")) is False

    def test_keyboard_interrupt_is_not_transient(self) -> None:
        assert is_transient(KeyboardInterrupt()) is False


# ---------------------------------------------------------------------------
# is_permanent() classification
# ---------------------------------------------------------------------------


class TestIsPermanent:
    """Verify permanent error classification."""

    def test_ssh_auth_error(self) -> None:
        assert is_permanent(SSHAuthenticationError("invalid key")) is True

    def test_ssh_host_key_error(self) -> None:
        assert is_permanent(SSHHostKeyError("unexpected key")) is True

    def test_permission_error(self) -> None:
        assert is_permanent(PermissionError("access denied")) is True

    # Negative cases
    def test_ssh_connection_error_is_not_permanent(self) -> None:
        assert is_permanent(SSHConnectionError("refused")) is False

    def test_os_error_is_not_permanent(self) -> None:
        assert is_permanent(OSError("unreachable")) is False

    def test_timeout_is_not_permanent(self) -> None:
        assert is_permanent(TimeoutError()) is False

    def test_value_error_is_not_permanent(self) -> None:
        assert is_permanent(ValueError("bad")) is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestErrorClassificationEdgeCases:
    """Edge cases for error classification."""

    def test_unclassified_error_is_neither(self) -> None:
        """Errors that are not in either category."""
        exc = RuntimeError("unknown")
        assert is_transient(exc) is False
        assert is_permanent(exc) is False

    def test_reconnection_exhausted_is_neither(self) -> None:
        """SSHReconnectionExhaustedError is neither transient nor permanent.

        It is a meta-error about the retry process itself, not a
        classification of the underlying failure.
        """
        exc = SSHReconnectionExhaustedError("done")
        # It inherits from SSHError but not from the classified types
        assert is_transient(exc) is False
        assert is_permanent(exc) is False
