"""Comprehensive coverage tests for jules_daemon.agent.error_classification.

Targets every uncovered line identified by coverage analysis. Each test
documents the specific line(s) it exercises. Organized by classification
layer in the same order as the module:

1. ErrorCategory.kind property (line 124)
2. RetryExhaustedError.__init__ (lines 201-202)
3. ClassifiedError.__str__ (line 247)
4. _classify_http_status for 429/408 (lines 304-310)
5. _classify_agent_error branches (lines 319-340)
6. _classify_llm_error branches (lines 364-421)
7. _classify_ssh_error branches (lines 442-462)
8. _classify_openai_sdk_error branches (lines 480-527)
9. _classify_builtin_error branches (lines 541-575)
10. _classify_by_status_code (lines 589-590)
11. KeyboardInterrupt handler (line 629)
12. classify_error default fallback (line 652)
13. is_transient / is_permanent convenience (lines 672, 687)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jules_daemon.agent.error_classification import (
    ClassifiedError,
    ErrorCategory,
    ErrorKind,
    RetryExhaustedError,
    ToolNotFoundError,
    ToolValidationError,
    UserCancelError,
    UserDenialError,
    classify_error,
    is_permanent,
    is_transient,
)


# ---------------------------------------------------------------------------
# 1. ErrorCategory.kind property (line 124)
# ---------------------------------------------------------------------------


class TestErrorCategoryKindProperty:
    """Exercise ErrorCategory.kind for every member to hit line 124."""

    @pytest.mark.parametrize(
        "category,expected",
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
    def test_kind_returns_correct_error_kind(
        self, category: ErrorCategory, expected: ErrorKind
    ) -> None:
        assert category.kind is expected


# ---------------------------------------------------------------------------
# 2. RetryExhaustedError.__init__ (lines 201-202)
# ---------------------------------------------------------------------------


class TestRetryExhaustedError:
    """RetryExhaustedError stores message and iterations_used."""

    def test_default_iterations_used(self) -> None:
        err = RetryExhaustedError("all retries exhausted")
        assert str(err) == "all retries exhausted"
        assert err.iterations_used == 0

    def test_custom_iterations_used(self) -> None:
        err = RetryExhaustedError("failed after 5 tries", iterations_used=5)
        assert err.iterations_used == 5
        assert "failed after 5 tries" in str(err)

    def test_is_exception(self) -> None:
        err = RetryExhaustedError("boom", iterations_used=3)
        assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# 3. ClassifiedError.__str__ (line 247)
# ---------------------------------------------------------------------------


class TestClassifiedErrorStr:
    """ClassifiedError __str__ produces [KIND:category] message format."""

    def test_transient_str(self) -> None:
        err = ClassifiedError(
            kind=ErrorKind.TRANSIENT,
            category=ErrorCategory.LLM_TIMEOUT,
            message="request timed out",
        )
        result = str(err)
        assert result == "[TRANSIENT:llm_timeout] request timed out"

    def test_permanent_str(self) -> None:
        err = ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.USER_DENIAL,
            message="user refused",
        )
        result = str(err)
        assert result == "[PERMANENT:user_denial] user refused"

    def test_unknown_category_str(self) -> None:
        err = ClassifiedError(
            kind=ErrorKind.PERMANENT,
            category=ErrorCategory.UNKNOWN,
            message="mystery error",
        )
        result = str(err)
        assert "[PERMANENT:unknown]" in result
        assert "mystery error" in result


# ---------------------------------------------------------------------------
# 4. _classify_http_status for 429 and 408 (lines 304-310)
# ---------------------------------------------------------------------------


class TestClassifyHttpStatus:
    """HTTP status code classification via _classify_http_status."""

    def test_429_rate_limit(self) -> None:
        from jules_daemon.agent.error_classification import _classify_http_status

        kind, category = _classify_http_status(429)
        assert kind is ErrorKind.TRANSIENT
        assert category is ErrorCategory.LLM_RATE_LIMIT

    def test_408_timeout(self) -> None:
        from jules_daemon.agent.error_classification import _classify_http_status

        kind, category = _classify_http_status(408)
        assert kind is ErrorKind.TRANSIENT
        assert category is ErrorCategory.LLM_TIMEOUT

    def test_500_server_error(self) -> None:
        from jules_daemon.agent.error_classification import _classify_http_status

        kind, category = _classify_http_status(500)
        assert kind is ErrorKind.TRANSIENT
        assert category is ErrorCategory.LLM_SERVER_ERROR

    def test_503_service_unavailable(self) -> None:
        from jules_daemon.agent.error_classification import _classify_http_status

        kind, category = _classify_http_status(503)
        assert kind is ErrorKind.TRANSIENT
        assert category is ErrorCategory.LLM_SERVER_ERROR

    def test_400_bad_request_is_permanent(self) -> None:
        from jules_daemon.agent.error_classification import _classify_http_status

        kind, category = _classify_http_status(400)
        assert kind is ErrorKind.PERMANENT
        assert category is ErrorCategory.UNKNOWN

    def test_404_not_found_is_permanent(self) -> None:
        from jules_daemon.agent.error_classification import _classify_http_status

        kind, category = _classify_http_status(404)
        assert kind is ErrorKind.PERMANENT
        assert category is ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# 5. _classify_agent_error branches (lines 319-340)
# ---------------------------------------------------------------------------


class TestClassifyAgentError:
    """Agent-specific error types classified via _classify_agent_error."""

    def test_user_denial_with_message(self) -> None:
        exc = UserDenialError("user said no to ssh command")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.USER_DENIAL
        assert result.message == "user said no to ssh command"
        assert result.original is exc

    def test_user_denial_without_args(self) -> None:
        exc = UserDenialError()
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.USER_DENIAL
        assert "denied" in result.message.lower() or "User denied" in result.message

    def test_user_cancel_with_message(self) -> None:
        exc = UserCancelError("ctrl-c pressed")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.USER_CANCEL
        assert result.message == "ctrl-c pressed"
        assert result.original is exc

    def test_user_cancel_without_args(self) -> None:
        exc = UserCancelError()
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.USER_CANCEL
        assert "cancel" in result.message.lower()

    def test_tool_not_found_with_message(self) -> None:
        exc = ToolNotFoundError("no_such_tool")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.TOOL_NOT_FOUND
        assert result.message == "no_such_tool"
        assert result.original is exc

    def test_tool_not_found_without_args(self) -> None:
        exc = ToolNotFoundError()
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.TOOL_NOT_FOUND
        assert "not found" in result.message.lower() or "Tool not found" in result.message

    def test_tool_validation_with_message(self) -> None:
        exc = ToolValidationError("missing required field 'path'")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.TOOL_VALIDATION
        assert result.message == "missing required field 'path'"
        assert result.original is exc

    def test_tool_validation_without_args(self) -> None:
        exc = ToolValidationError()
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.TOOL_VALIDATION
        assert "validation" in result.message.lower() or "failed" in result.message.lower()


# ---------------------------------------------------------------------------
# 6. _classify_llm_error branches (lines 364-421)
# ---------------------------------------------------------------------------


class TestClassifyLlmError:
    """LLM error hierarchy classification via _classify_llm_error."""

    def test_llm_authentication_error(self) -> None:
        from jules_daemon.llm.errors import LLMAuthenticationError

        exc = LLMAuthenticationError("invalid API key")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.LLM_AUTH
        assert result.message == "invalid API key"
        assert result.original is exc

    def test_llm_connection_error(self) -> None:
        from jules_daemon.llm.errors import LLMConnectionError

        exc = LLMConnectionError("endpoint unreachable")
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_CONNECTION
        assert result.message == "endpoint unreachable"
        assert result.original is exc

    def test_llm_parse_error(self) -> None:
        from jules_daemon.llm.errors import LLMParseError

        exc = LLMParseError("cannot parse JSON", raw_content="{bad")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.LLM_MALFORMED_RESPONSE
        assert result.message == "cannot parse JSON"
        assert result.original is exc

    def test_llm_tool_calling_unsupported(self) -> None:
        from jules_daemon.llm.errors import LLMToolCallingUnsupportedError

        exc = LLMToolCallingUnsupportedError("backend does not support tools")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.LLM_UNSUPPORTED
        assert result.message == "backend does not support tools"
        assert result.original is exc

    def test_llm_response_error_with_429_status(self) -> None:
        from jules_daemon.llm.errors import LLMResponseError

        exc = LLMResponseError("rate limited", status_code=429)
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_RATE_LIMIT
        assert result.original is exc

    def test_llm_response_error_with_500_status(self) -> None:
        from jules_daemon.llm.errors import LLMResponseError

        exc = LLMResponseError("internal error", status_code=500)
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_SERVER_ERROR

    def test_llm_response_error_with_408_status(self) -> None:
        from jules_daemon.llm.errors import LLMResponseError

        exc = LLMResponseError("request timeout", status_code=408)
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_TIMEOUT

    def test_llm_response_error_with_400_status(self) -> None:
        from jules_daemon.llm.errors import LLMResponseError

        exc = LLMResponseError("bad request", status_code=400)
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT

    def test_llm_response_error_without_status_code(self) -> None:
        from jules_daemon.llm.errors import LLMResponseError

        exc = LLMResponseError("unknown failure")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN
        assert result.original is exc

    def test_generic_llm_error_is_permanent(self) -> None:
        """A plain LLMError (not a subclass) maps to PERMANENT/UNKNOWN (line 421)."""
        from jules_daemon.llm.errors import LLMError

        exc = LLMError("generic LLM failure")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN
        assert result.message == "generic LLM failure"
        assert result.original is exc

    def test_llm_import_error_returns_none(self) -> None:
        """When jules_daemon.llm.errors cannot be imported, _classify_llm_error
        returns None (lines 364-365).
        """
        from jules_daemon.agent.error_classification import _classify_llm_error

        with patch.dict("sys.modules", {"jules_daemon.llm.errors": None}):
            # Passing a plain Exception should hit the ImportError branch
            result = _classify_llm_error(RuntimeError("test"))
            assert result is None


# ---------------------------------------------------------------------------
# 7. _classify_ssh_error branches (lines 442-462)
# ---------------------------------------------------------------------------


class TestClassifySshError:
    """SSH error hierarchy classification via _classify_ssh_error."""

    def test_ssh_authentication_error(self) -> None:
        from jules_daemon.ssh.errors import SSHAuthenticationError

        exc = SSHAuthenticationError("invalid key")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.SSH_AUTH
        assert result.message == "invalid key"
        assert result.original is exc

    def test_ssh_host_key_error(self) -> None:
        from jules_daemon.ssh.errors import SSHHostKeyError

        exc = SSHHostKeyError("host key mismatch")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.SSH_AUTH
        assert result.message == "host key mismatch"
        assert result.original is exc

    def test_ssh_connection_error(self) -> None:
        from jules_daemon.ssh.errors import SSHConnectionError

        exc = SSHConnectionError("connection refused")
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.SSH_CONNECTION
        assert result.message == "connection refused"
        assert result.original is exc

    def test_ssh_import_error_returns_none(self) -> None:
        """When jules_daemon.ssh.errors cannot be imported, _classify_ssh_error
        returns None (lines 442-443).
        """
        from jules_daemon.agent.error_classification import _classify_ssh_error

        with patch.dict("sys.modules", {"jules_daemon.ssh.errors": None}):
            result = _classify_ssh_error(RuntimeError("test"))
            assert result is None


# ---------------------------------------------------------------------------
# 8. _classify_openai_sdk_error branches (lines 480-527)
# ---------------------------------------------------------------------------


class TestClassifyOpenaiSdkError:
    """OpenAI SDK error classification via _classify_openai_sdk_error."""

    @pytest.fixture()
    def mock_response(self) -> object:
        """Create a mock httpx response for OpenAI SDK exceptions."""
        import httpx

        return httpx.Response(
            status_code=500,
            request=httpx.Request("POST", "https://api.openai.com/v1/test"),
        )

    def _make_response(self, status_code: int) -> object:
        import httpx

        return httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.openai.com/v1/test"),
        )

    def test_api_timeout_error(self) -> None:
        import openai

        exc = openai.APITimeoutError(request=None)  # type: ignore[arg-type]
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_TIMEOUT
        assert result.original is exc

    def test_api_connection_error(self) -> None:
        import openai

        exc = openai.APIConnectionError(request=None)  # type: ignore[arg-type]
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_CONNECTION
        assert result.original is exc

    def test_rate_limit_error(self) -> None:
        import openai

        exc = openai.RateLimitError(
            message="rate limited",
            response=self._make_response(429),
            body=None,
        )
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_RATE_LIMIT
        assert result.original is exc

    def test_authentication_error(self) -> None:
        import openai

        exc = openai.AuthenticationError(
            message="invalid api key",
            response=self._make_response(401),
            body=None,
        )
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.LLM_AUTH
        assert result.original is exc

    def test_api_status_error_500(self) -> None:
        """APIStatusError with 500 falls through to the generic branch (lines 519-520)."""
        import openai

        exc = openai.InternalServerError(
            message="internal server error",
            response=self._make_response(500),
            body=None,
        )
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_SERVER_ERROR
        assert result.original is exc

    def test_api_status_error_404(self) -> None:
        """APIStatusError with 404 is classified as permanent via generic branch."""
        import openai

        exc = openai.NotFoundError(
            message="not found",
            response=self._make_response(404),
            body=None,
        )
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN
        assert result.original is exc

    def test_api_status_error_422(self) -> None:
        """APIStatusError with 422 (Unprocessable Entity) is permanent."""
        import openai

        exc = openai.UnprocessableEntityError(
            message="unprocessable",
            response=self._make_response(422),
            body=None,
        )
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.original is exc

    def test_openai_import_error_returns_none(self) -> None:
        """When openai cannot be imported, _classify_openai_sdk_error
        returns None (lines 480-481).
        """
        from jules_daemon.agent.error_classification import (
            _classify_openai_sdk_error,
        )

        with patch.dict("sys.modules", {"openai": None}):
            result = _classify_openai_sdk_error(RuntimeError("test"))
            assert result is None


# ---------------------------------------------------------------------------
# 9. _classify_builtin_error branches (lines 541-575)
# ---------------------------------------------------------------------------


class TestClassifyBuiltinError:
    """Python built-in exception classification via _classify_builtin_error."""

    def test_permission_error_is_permanent(self) -> None:
        """PermissionError (subclass of OSError) is permanent, not transient (line 541)."""
        result = classify_error(PermissionError("access denied"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN
        assert result.original is not None

    def test_value_error_is_permanent(self) -> None:
        result = classify_error(ValueError("bad value"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN

    def test_type_error_is_permanent(self) -> None:
        result = classify_error(TypeError("wrong type"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN

    def test_key_error_is_permanent(self) -> None:
        result = classify_error(KeyError("missing_key"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN

    def test_attribute_error_is_permanent(self) -> None:
        result = classify_error(AttributeError("no such attr"))
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN

    def test_timeout_error_is_transient(self) -> None:
        exc = TimeoutError("operation timed out")
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_TIMEOUT
        assert "operation timed out" in result.message

    def test_timeout_error_without_args(self) -> None:
        """TimeoutError with no args uses fallback message (line 562)."""
        exc = TimeoutError()
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_TIMEOUT
        assert "timed out" in result.message.lower()

    def test_connection_error_is_transient(self) -> None:
        result = classify_error(ConnectionError("refused"))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.NETWORK

    def test_os_error_is_transient(self) -> None:
        result = classify_error(OSError("network unreachable"))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.NETWORK

    def test_broken_pipe_is_transient(self) -> None:
        result = classify_error(BrokenPipeError("pipe broken"))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.NETWORK

    def test_eof_error_is_transient(self) -> None:
        result = classify_error(EOFError("unexpected eof"))
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.NETWORK

    def test_unrecognized_builtin_returns_none(self) -> None:
        """_classify_builtin_error returns None for unrecognized types (line 575)."""
        from jules_daemon.agent.error_classification import _classify_builtin_error

        result = _classify_builtin_error(RuntimeError("unknown"))
        assert result is None


# ---------------------------------------------------------------------------
# 10. _classify_by_status_code (lines 589-590)
# ---------------------------------------------------------------------------


class TestClassifyByStatusCode:
    """Classification of exceptions with a status_code attribute."""

    def test_exception_with_status_code_500(self) -> None:
        exc = _ExceptionWithStatusCode(500)
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_SERVER_ERROR

    def test_exception_with_status_code_429(self) -> None:
        exc = _ExceptionWithStatusCode(429)
        result = classify_error(exc)
        assert result.kind is ErrorKind.TRANSIENT
        assert result.category is ErrorCategory.LLM_RATE_LIMIT

    def test_exception_with_status_code_400(self) -> None:
        exc = _ExceptionWithStatusCode(400)
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN

    def test_exception_without_status_code_skipped(self) -> None:
        """Exceptions without status_code are not matched by _classify_by_status_code."""
        from jules_daemon.agent.error_classification import _classify_by_status_code

        result = _classify_by_status_code(RuntimeError("no status"))
        assert result is None

    def test_exception_with_non_int_status_code_skipped(self) -> None:
        """Non-integer status_code attribute is ignored."""
        from jules_daemon.agent.error_classification import _classify_by_status_code

        exc = RuntimeError("bad status")
        exc.status_code = "not_an_int"  # type: ignore[attr-defined]
        result = _classify_by_status_code(exc)
        assert result is None


# ---------------------------------------------------------------------------
# 11. KeyboardInterrupt handler (line 629)
# ---------------------------------------------------------------------------


class TestKeyboardInterrupt:
    """KeyboardInterrupt is classified as USER_CANCEL / PERMANENT."""

    def test_keyboard_interrupt(self) -> None:
        exc = KeyboardInterrupt()
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.USER_CANCEL
        assert "keyboard interrupt" in result.message.lower()
        assert result.original is exc


# ---------------------------------------------------------------------------
# 12. classify_error default fallback (line 652)
# ---------------------------------------------------------------------------


class TestDefaultFallback:
    """Unknown exceptions fall through all classifiers to the default."""

    def test_runtime_error_defaults_to_permanent(self) -> None:
        exc = RuntimeError("totally unknown")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN
        assert result.message == "totally unknown"
        assert result.original is exc

    def test_custom_exception_defaults_to_permanent(self) -> None:
        class CustomError(Exception):
            pass

        exc = CustomError("custom failure")
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN
        assert result.original is exc

    def test_exception_with_empty_message(self) -> None:
        """Empty-message exception uses fallback format (line 655)."""
        exc = Exception()
        result = classify_error(exc)
        assert result.kind is ErrorKind.PERMANENT
        assert result.category is ErrorCategory.UNKNOWN
        # str(Exception()) is empty, so the fallback format kicks in
        assert "Unclassified error" in result.message or result.message == ""


# ---------------------------------------------------------------------------
# 13. is_transient / is_permanent convenience functions (lines 672, 687)
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """is_transient() and is_permanent() wrap classify_error correctly."""

    def test_is_transient_true_for_timeout(self) -> None:
        assert is_transient(TimeoutError("timeout")) is True

    def test_is_transient_false_for_value_error(self) -> None:
        assert is_transient(ValueError("bad")) is False

    def test_is_transient_true_for_connection_error(self) -> None:
        assert is_transient(ConnectionError("reset")) is True

    def test_is_permanent_true_for_permission_error(self) -> None:
        assert is_permanent(PermissionError("denied")) is True

    def test_is_permanent_false_for_timeout(self) -> None:
        assert is_permanent(TimeoutError("timeout")) is False

    def test_is_permanent_true_for_unknown(self) -> None:
        assert is_permanent(RuntimeError("unknown")) is True

    def test_is_transient_for_user_denial(self) -> None:
        assert is_transient(UserDenialError("no")) is False

    def test_is_permanent_for_user_cancel(self) -> None:
        assert is_permanent(UserCancelError("cancelled")) is True

    def test_is_transient_for_keyboard_interrupt(self) -> None:
        assert is_transient(KeyboardInterrupt()) is False

    def test_is_permanent_for_keyboard_interrupt(self) -> None:
        assert is_permanent(KeyboardInterrupt()) is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ExceptionWithStatusCode(Exception):
    """Test exception that carries an HTTP status_code attribute."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
