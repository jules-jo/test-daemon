"""Custom error types for LLM operations.

All errors inherit from LLMError, enabling callers to catch the base
class for any LLM-related failure or catch specific subclasses for
targeted handling.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base error for all LLM client operations."""


class LLMAuthenticationError(LLMError):
    """Raised when the Dataiku API key is invalid or expired."""


class LLMConnectionError(LLMError):
    """Raised when the Dataiku Mesh endpoint is unreachable."""


class LLMResponseError(LLMError):
    """Raised when the API returns a non-success status code."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMToolCallingUnsupportedError(LLMError):
    """Raised when native tool calling is not supported by the backend."""


class LLMParseError(LLMError):
    """Raised when the LLM response cannot be parsed into structured output.

    Covers JSON extraction failures, schema validation errors, and any
    other parse-related issues when converting raw LLM text into a
    validated SSHCommand or LLMCommandResponse.
    """

    def __init__(
        self,
        message: str,
        raw_content: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_content = raw_content
