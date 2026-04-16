"""Registry for workflow step output interpreters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .generic import interpret_generic_step_output

InterpreterFn = Callable[..., dict[str, Any] | None]


class StepInterpreterRegistry:
    """Resolve the appropriate interpreter for one workflow step."""

    def __init__(self) -> None:
        self._generic: InterpreterFn = interpret_generic_step_output

    def interpret(
        self,
        *,
        step_name: str,
        command: str,
        raw_output: str,
        success: bool | None,
        active: bool,
    ) -> dict[str, Any] | None:
        """Interpret one workflow step using the best available interpreter."""
        del step_name  # reserved for future family-specific selection
        return self._generic(
            raw_output=raw_output,
            command=command,
            success=success,
            active=active,
        )
