"""Tests for LLM error hierarchy."""

from __future__ import annotations

import pytest

from jules_daemon.llm.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMParseError,
    LLMResponseError,
    LLMToolCallingUnsupportedError,
)


class TestLLMErrorHierarchy:
    """All LLM errors inherit from LLMError."""

    def test_base_error(self) -> None:
        err = LLMError("something went wrong")
        assert str(err) == "something went wrong"
        assert isinstance(err, Exception)

    def test_auth_error_inherits(self) -> None:
        err = LLMAuthenticationError("bad key")
        assert isinstance(err, LLMError)
        assert str(err) == "bad key"

    def test_connection_error_inherits(self) -> None:
        err = LLMConnectionError("unreachable")
        assert isinstance(err, LLMError)

    def test_response_error_inherits(self) -> None:
        err = LLMResponseError("500 error", status_code=500)
        assert isinstance(err, LLMError)
        assert err.status_code == 500

    def test_response_error_default_status(self) -> None:
        err = LLMResponseError("unknown error")
        assert err.status_code is None

    def test_tool_calling_unsupported_inherits(self) -> None:
        err = LLMToolCallingUnsupportedError("no tools")
        assert isinstance(err, LLMError)

    def test_parse_error_inherits(self) -> None:
        err = LLMParseError("bad json", raw_content='{"broken":')
        assert isinstance(err, LLMError)
        assert str(err) == "bad json"
        assert err.raw_content == '{"broken":'

    def test_parse_error_default_raw_content(self) -> None:
        err = LLMParseError("no json")
        assert err.raw_content is None

    def test_catch_all_with_base(self) -> None:
        """All specific errors can be caught with LLMError."""
        errors: list[LLMError] = [
            LLMAuthenticationError("auth"),
            LLMConnectionError("conn"),
            LLMResponseError("resp"),
            LLMToolCallingUnsupportedError("tools"),
            LLMParseError("parse"),
        ]
        for err in errors:
            with pytest.raises(LLMError):
                raise err
