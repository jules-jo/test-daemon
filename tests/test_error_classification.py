"""Tests for the agent error classification module.

Validates that errors from LLM calls, tool executions, SSH operations,
and user actions are correctly classified as transient (retryable) or
permanent (terminal).

Test structure mirrors the classification categories:
- Transient: network, timeout, rate-limit, server errors
- Permanent: auth, user denial, user cancel, malformed response, validation
- Edge cases: chained exceptions, unknown types, boundary cases
"""

from __future__ import annotations

import asyncio

import pytest

from jules_daemon.agent.error_classification import (
    ClassifiedError,
    ErrorCategory,
    ErrorKind,
    classify_error,
    is_permanent,
    is_transient,
)


# ---------------------------------------------------------------------------
# ErrorKind enum
# ---------------------------------------------------------------------------


class TestErrorKind:
    """ErrorKind enum has exactly two members."""

    def test_transient_value(self) -> None:
        assert ErrorKind.TRANSIENT.value == "transient"

    def test_permanent_value(self) -> None:
        assert ErrorKind.PERMANENT.value == "permanent"

    def test_only_two_members(self) -> None:
        assert len(ErrorKind) == 2


# ---------------------------------------------------------------------------
# ErrorCategory enum
# ---------------------------------------------------------------------------


class TestErrorCategory:
    """ErrorCategory covers all documented error categories."""

    def test_has_llm_categories(self) -> None:
        assert ErrorCategory.LLM_TIMEOUT is not None
        assert ErrorCategory.LLM_RATE_LIMIT is not None
        assert ErrorCategory.LLM_CONNECTION is not None
        assert ErrorCategory.LLM_AUTH is not None
        assert ErrorCategory.LLM_MALFORMED_RESPONSE is not None
        assert ErrorCategory.LLM_SERVER_ERROR is not None
        assert ErrorCategory.LLM_UNSUPPORTED is not None

    def test_has_tool_categories(self) -> None:
        assert ErrorCategory.TOOL_NOT_FOUND is not None
        assert ErrorCategory.TOOL_VALIDATION is not None
        assert ErrorCategory.TOOL_TIMEOUT is not None

    def test_has_user_categories(self) -> None:
        assert ErrorCategory.USER_DENIAL is not None
        assert ErrorCategory.USER_CANCEL is not None

    def test_has_ssh_categories(self) -> None:
        assert ErrorCategory.SSH_AUTH is not None
        assert ErrorCategory.SSH_CONNECTION is not None

    def test_has_network_category(self) -> None:
        assert ErrorCategory.NETWORK is not None

    def test_has_unknown_category(self) -> None:
        assert ErrorCategory.UNKNOWN is not None

    @pytest.mark.parametrize(
        "category,expected_kind",
        [
            (ErrorCategory.LLM_TIMEOUT, ErrorKind.TRANSIENT),
            (ErrorCategory.LLM_RATE_LIMIT, ErrorKind.TRANSIENT),
            (ErrorCategory.LLM_CONNECTION, ErrorKind.TRANSIENT),
            (ErrorCategory.LLM_SERVER_ERROR, ErrorKind.TRANSIENT),
            (ErrorCategory.SSH_CONNECTION, ErrorKind.TRANSIENT),
            (ErrorCategory.NETWORK, ErrorKind.TRANSIENT),
            (ErrorCategory.TOOL_TIMEOUT, ErrorKind.TRANSIENT),
            (ErrorCategory.LLM_AUTH, ErrorKind.PERMANENT),
            (ErrorCategory.LLM_MALFORMED_RESPONSE, ErrorKind.PERMANENT),
            (ErrorCategory.LLM_UNSUPPORTED, ErrorKind.PERMANENT),
            (ErrorCategory.SSH_AUTH, ErrorKind.PERMANENT),
            (ErrorCategory.USER_DENIAL, ErrorKind.PERMANENT),
            (ErrorCategory.USER_CANCEL, ErrorKind.PERMANENT),
            (ErrorCategory.TOOL_NOT_FOUND, ErrorKind.PERMANENT),
            (ErrorCategory.TOOL_VALIDATION, ErrorKind.PERMANENT),
            (ErrorCategory.UNKNOWN, ErrorKind.PERMANENT),
        ],
    )
    def test_category_kind_mapping(
        self, category: ErrorCategory, expected_kind: ErrorKind
    ) -> None:
        """Every ErrorCategory maps to the correct ErrorKind."""
        assert category.kind is expected_kind

    def test_all_categories_have_kind_mapping(self) -> None:
        """Every ErrorCategory member appears in the kind mapping."""
        for cat in ErrorCategory:
            # Should not raise -- .kind accesses _CATEGORY_TO_KIND
            assert cat.kind in (ErrorKind.TRANSIENT, ErrorKind.PERMANENT)


# ---------------------------------------------------------------------------
# ClassifiedError frozen dataclass
# ---------------------------------------------------------------------------


class TestClassifiedError:
    """ClassifiedError is immutable and provides convenience properties."""

    def test_transient_error_is_retryable(self) -> None:
        err = ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.NETWORK,
            message="connection reset",
            original=ConnectionError("reset"),
        )
        assert err.is_retryable is True

    def test_permanent_error_is_not_retryable(self) -> None:
        err = ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.USER_DENIAL,
            message="user said no",
            original=None,
        )
        assert err.is_retryable is False

    def test_frozen(self) -> None:
        err = ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.NETWORK,
            message="test",
        )
        with pytest.raises(AttributeError):
            err.kind = ErrorKind.PERMANENT  # type: ignore[misc]

    def test_original_defaults_to_none(self) -> None:
        err = ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.UNKNOWN,
            message="test",
        )
        assert err.original is None

    def test_str_representation(self) -> None:
        err = ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.LLM_TIMEOUT,
            message="LLM call timed out",
        )
        text = str(err)
        assert "transient" in text.lower() or "TRANSIENT" in text
        assert "LLM call timed out" in text


# ---------------------------------------------------------------------------
# Transient error classification
# ---------------------------------------------------------------------------


class TestTransientErrors:
    """Errors that should be classified as TRANSIENT (retryable)."""

    def test_connection_error(self) -> None:
        result = classify_error(ConnectionError("refused"))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.NETWORK

    def test_connection_refused_error(self) -> None:
        result = classify_error(ConnectionRefusedError("refused"))
        assert result.kind is ErrorKind.TRANSIENT

    def test_connection_reset_error(self) -> None:
        result = classify_error(ConnectionResetError("reset"))
        assert result.kind is ErrorKind.TRANSIENT

    def test_connection_aborted_error(self) -> None:
        result = classify_error(ConnectionAbortedError("aborted"))
        assert result.kind is ErrorKind.TRANSIENT

    def test_timeout_error(self) -> None:
        result = classify_error(TimeoutError("timed out"))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_TIMEOUT

    def test_asyncio_timeout_error(self) -> None:
        result = classify_error(asyncio.TimeoutError())
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_TIMEOUT

    def test_os_error(self) -> None:
        result = classify_error(OSError("network unreachable"))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.NETWORK

    def test_broken_pipe_error(self) -> None:
        result = classify_error(BrokenPipeError("broken pipe"))
        assert result.kind is ErrorKind.TRANSIENT

    def test_eof_error(self) -> None:
        result = classify_error(EOFError("unexpected eof"))
        assert result.kind is ErrorKind.TRANSIENT

    # -- OpenAI SDK transient errors -----------------------------------------

    def test_openai_api_connection_error(self) -> None:
        """OpenAI APIConnectionError is transient."""
        try:
            import openai

            exc = openai.APIConnectionError(request=None)  # type: ignore[arg-type]
            result = classify_error(exc)
            assert result.kind is ErrorKind.TRANSIENT
            assert result.category is ErrorCategory.LLM_CONNECTION
        except ImportError:
            pytest.skip("openai not installed")

    def test_openai_rate_limit_error(self) -> None:
        """OpenAI RateLimitError is transient."""
        try:
            import openai

            exc = openai.RateLimitError(
                message="rate limited",
                response=_make_mock_httpx_response(429),
                body=None,
            )
            result = classify_error(exc)
            assert result.kind is ErrorKind.TRANSIENT
            assert result.category is ErrorCategory.LLM_RATE_LIMIT
        except ImportError:
            pytest.skip("openai not installed")

    def test_openai_api_timeout_error(self) -> None:
        """OpenAI APITimeoutError is transient."""
        try:
            import openai

            exc = openai.APITimeoutError(request=None)  # type: ignore[arg-type]
            result = classify_error(exc)
            assert result.kind is ErrorKind.TRANSIENT
            assert result.category is ErrorCategory.LLM_TIMEOUT
        except ImportError:
            pytest.skip("openai not installed")

    # -- HTTP status codes ---------------------------------------------------

    def test_http_429_rate_limit(self) -> None:
        result = classify_error(_make_http_status_error(429))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_RATE_LIMIT

    def test_http_408_request_timeout(self) -> None:
        result = classify_error(_make_http_status_error(408))
        assert result.kind is ErrorKind.TRANSIENT

    def test_http_500_server_error(self) -> None:
        result = classify_error(_make_http_status_error(500))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_SERVER_ERROR

    def test_http_502_bad_gateway(self) -> None:
        result = classify_error(_make_http_status_error(502))
        assert result.kind is ErrorKind.TRANSIENT

    def test_http_503_service_unavailable(self) -> None:
        result = classify_error(_make_http_status_error(503))
        assert result.kind is ErrorKind.TRANSIENT

    def test_http_504_gateway_timeout(self) -> None:
        result = classify_error(_make_http_status_error(504))
        assert result.kind is ErrorKind.TRANSIENT

    # -- LLM error hierarchy (transient) -------------------------------------

    def test_llm_connection_error(self) -> None:
        from jules_daemon.llm.errors import LLMConnectionError

        result = classify_error(LLMConnectionError("unreachable"))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_CONNECTION

    # -- SSH error hierarchy (transient) -------------------------------------

    def test_ssh_connection_error(self) -> None:
        from jules_daemon.ssh.errors import SSHConnectionError

        result = classify_error(SSHConnectionError("refused"))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.SSH_CONNECTION


# ---------------------------------------------------------------------------
# Permanent error classification
# ---------------------------------------------------------------------------


class TestPermanentErrors:
    """Errors that should be classified as PERMANENT (not retryable)."""

    def test_permission_error(self) -> None:
        result = classify_error(PermissionError("denied"))
        assert result.kind is ErrorKind.PERMANENT

    def test_value_error(self) -> None:
        result = classify_error(ValueError("malformed"))
        assert result.kind is ErrorKind.PERMANENT

    def test_type_error(self) -> None:
        result = classify_error(TypeError("wrong type"))
        assert result.kind is ErrorKind.PERMANENT

    def test_key_error(self) -> None:
        result = classify_error(KeyError("missing"))
        assert result.kind is ErrorKind.PERMANENT

    # -- OpenAI SDK permanent errors -----------------------------------------

    def test_openai_authentication_error(self) -> None:
        """OpenAI AuthenticationError is permanent."""
        try:
            import openai

            exc = openai.AuthenticationError(
                message="invalid key",
                response=_make_mock_httpx_response(401),
                body=None,
            )
            result = classify_error(exc)
            assert result.kind is ErrorKind.PERMANENT
            assert result.category is ErrorCategory.LLM_AUTH
        except ImportError:
            pytest.skip("openai not installed")

    # -- HTTP status codes (permanent) ---------------------------------------

    def test_http_400_bad_request(self) -> None:
        result = classify_error(_make_http_status_error(400))
        assert result.kind is ErrorKind.PERMANENT

    def test_http_401_unauthorized(self) -> None:
        result = classify_error(_make_http_status_error(401))
        assert result.kind is ErrorKind.PERMANENT

    def test_http_403_forbidden(self) -> None:
        result = classify_error(_make_http_status_error(403))
        assert result.kind is ErrorKind.PERMANENT

    def test_http_404_not_found(self) -> None:
        result = classify_error(_make_http_status_error(404))
        assert result.kind is ErrorKind.PERMANENT

    # -- LLM error hierarchy (permanent) -------------------------------------

    def test_llm_authentication_error(self) -> None:
        from jules_daemon.llm.errors import LLMAuthenticationError

        result = classify_error(LLMAuthenticationError("bad key"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.LLM_AUTH

    def test_llm_parse_error(self) -> None:
        from jules_daemon.llm.errors import LLMParseError

        result = classify_error(LLMParseError("bad json"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.LLM_MALFORMED_RESPONSE

    def test_llm_tool_calling_unsupported(self) -> None:
        from jules_daemon.llm.errors import LLMToolCallingUnsupportedError

        result = classify_error(LLMToolCallingUnsupportedError("no tools"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.LLM_UNSUPPORTED

    # -- SSH error hierarchy (permanent) -------------------------------------

    def test_ssh_authentication_error(self) -> None:
        from jules_daemon.ssh.errors import SSHAuthenticationError

        result = classify_error(SSHAuthenticationError("bad key"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.SSH_AUTH

    def test_ssh_host_key_error(self) -> None:
        from jules_daemon.ssh.errors import SSHHostKeyError

        result = classify_error(SSHHostKeyError("mismatch"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.SSH_AUTH

    # -- Agent-specific permanent errors -------------------------------------

    def test_user_denial_error(self) -> None:
        from jules_daemon.agent.error_classification import UserDenialError

        result = classify_error(UserDenialError("user said no"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.USER_DENIAL

    def test_user_cancel_error(self) -> None:
        from jules_daemon.agent.error_classification import UserCancelError

        result = classify_error(UserCancelError("user pressed ctrl-c"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.USER_CANCEL

    def test_tool_not_found_error(self) -> None:
        from jules_daemon.agent.error_classification import ToolNotFoundError

        result = classify_error(ToolNotFoundError("no_such_tool"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.TOOL_NOT_FOUND

    def test_tool_validation_error(self) -> None:
        from jules_daemon.agent.error_classification import ToolValidationError

        result = classify_error(ToolValidationError("missing required arg"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.TOOL_VALIDATION


# ---------------------------------------------------------------------------
# Convenience functions: is_transient / is_permanent
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """is_transient() and is_permanent() return correct booleans."""

    def test_is_transient_for_connection_error(self) -> None:
        assert is_transient(ConnectionError("test")) is True

    def test_is_transient_for_timeout_error(self) -> None:
        assert is_transient(TimeoutError("test")) is True

    def test_is_transient_for_value_error(self) -> None:
        assert is_transient(ValueError("test")) is False

    def test_is_permanent_for_permission_error(self) -> None:
        assert is_permanent(PermissionError("test")) is True

    def test_is_permanent_for_connection_error(self) -> None:
        assert is_permanent(ConnectionError("test")) is False

    def test_is_permanent_for_value_error(self) -> None:
        assert is_permanent(ValueError("test")) is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_unknown_error_is_permanent(self) -> None:
        """Unrecognized exceptions default to PERMANENT for safety."""
        result = classify_error(RuntimeError("unknown"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN

    def test_original_exception_preserved(self) -> None:
        """The original exception is stored in ClassifiedError.original."""
        exc = ConnectionError("original")
        result = classify_error(exc)
        assert result.original is exc

    def test_message_includes_original_str(self) -> None:
        """The message field contains useful context from the original."""
        exc = TimeoutError("request took too long")
        result = classify_error(exc)
        assert "request took too long" in result.message

    def test_keyboard_interrupt_is_permanent(self) -> None:
        """KeyboardInterrupt should be treated as user cancel."""
        result = classify_error(KeyboardInterrupt())
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.USER_CANCEL

    def test_llm_response_error_with_transient_status(self) -> None:
        """LLMResponseError with 503 status code is transient."""
        from jules_daemon.llm.errors import LLMResponseError

        result = classify_error(LLMResponseError("service down", status_code=503))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_SERVER_ERROR

    def test_llm_response_error_with_permanent_status(self) -> None:
        """LLMResponseError with 400 status code is permanent."""
        from jules_daemon.llm.errors import LLMResponseError

        result = classify_error(LLMResponseError("bad request", status_code=400))
        assert result.kind is ErrorKind.PERMANENT

    def test_llm_response_error_with_rate_limit_status(self) -> None:
        """LLMResponseError with 429 status code is transient."""
        from jules_daemon.llm.errors import LLMResponseError

        result = classify_error(LLMResponseError("rate limited", status_code=429))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_RATE_LIMIT

    def test_llm_response_error_no_status_code(self) -> None:
        """LLMResponseError without status_code defaults to permanent."""
        from jules_daemon.llm.errors import LLMResponseError

        result = classify_error(LLMResponseError("mystery"))
        assert result.kind is ErrorKind.PERMANENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockHTTPStatusError(Exception):
    """Simulates an error with an HTTP status_code attribute."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _make_http_status_error(status_code: int) -> _MockHTTPStatusError:
    """Create a mock exception with an HTTP status_code attribute."""
    return _MockHTTPStatusError(status_code)


def _make_mock_httpx_response(status_code: int) -> object:
    """Create a minimal mock httpx response for OpenAI SDK errors."""
    try:
        import httpx

        return httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://test.example.com"),
        )
    except ImportError:

        class _FakeResponse:
            def __init__(self, code: int) -> None:
                self.status_code = code
                self.headers = {}

        return _FakeResponse(status_code)
