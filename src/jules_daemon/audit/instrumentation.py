"""Pipeline stage instrumentation with snapshot-based audit tracking.

Provides two complementary mechanisms for wrapping pipeline stage execution
with automatic audit trail generation:

``StageAudit``
    A context manager for explicit, fine-grained control. The caller creates
    the audit context, executes stage logic inside a ``with`` block, and
    optionally calls ``record_output`` to capture stage outputs. On exit
    (success or error), a ``StageSnapshot`` is taken, an ``AuditEntry`` is
    constructed, and it is appended to the immutable ``AuditChain``.

``stage_instrumented``
    A decorator factory for concise, automatic instrumentation. Wraps a
    callable so that its arguments are captured as the before-snapshot
    inputs and its return value is captured as the after-snapshot output.
    On success, returns a ``StageResult`` containing the return value and
    updated chain. On error, raises ``StageError`` carrying the chain and
    entry so the caller can inspect the audit trail.

Both mechanisms:
    - Invoke ``capture_snapshot`` before and after execution
    - Measure wall-clock duration via ``time.monotonic``
    - Construct an ``AuditEntry`` (from ``audit_models``) with
      before/after snapshots, duration, status, and optional error
    - Append the entry to an immutable ``AuditChain``
    - Never suppress exceptions

Design principles:
    - Immutable outputs: ``StageResult`` and ``AuditEntry`` are frozen
    - Defensive copies: ``record_output`` copies the dict before storing
    - No side effects beyond timing: all I/O is the caller's responsibility
    - Composable: chain multiple stages by threading the chain through

Usage (context manager)::

    from jules_daemon.audit.instrumentation import StageAudit
    from jules_daemon.audit_models import AuditChain

    chain = AuditChain.empty()
    audit = StageAudit("nl_input", chain, inputs={"raw": "run tests"})
    with audit:
        result = translate(text)
        audit.record_output({"command": result})
    updated_chain = audit.chain  # chain with new entry appended

Usage (decorator)::

    from jules_daemon.audit.instrumentation import stage_instrumented

    @stage_instrumented("nl_input")
    def translate(text: str) -> str:
        return "pytest -v"

    result = translate("run tests", _audit_chain=chain)
    result.value   # "pytest -v"
    result.chain   # updated chain
    result.entry   # the appended AuditEntry
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import Any, Callable

from jules_daemon.audit.snapshot import capture_snapshot
from jules_daemon.audit_models import AuditChain, AuditEntry

__all__ = [
    "StageAudit",
    "StageError",
    "StageResult",
    "stage_instrumented",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_non_empty(value: str, field_name: str) -> str:
    """Strip and validate that a string is not empty."""
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


# ---------------------------------------------------------------------------
# StageResult -- frozen output from an instrumented stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageResult:
    """Immutable result from an instrumented pipeline stage.

    Bundles the stage's return value with the updated audit chain and the
    specific entry that was appended.

    Attributes:
        value: The return value produced by the stage callable.
            Can be any type including None.
        chain: The ``AuditChain`` with the new entry appended.
            This is a new chain instance (the input chain is not modified).
        entry: The ``AuditEntry`` that was constructed and appended.
    """

    value: object
    chain: AuditChain
    entry: AuditEntry


# ---------------------------------------------------------------------------
# StageError -- exception wrapper carrying audit info
# ---------------------------------------------------------------------------


class StageError(Exception):
    """Raised when an instrumented stage fails during execution.

    Wraps the original exception and carries the audit chain and entry
    so callers can inspect the audit trail even after a failure.

    Attributes:
        cause: The original exception that was raised by the stage.
        chain: The ``AuditChain`` with the error entry appended.
        entry: The ``AuditEntry`` recording the failure, or None if the
            entry could not be constructed.
    """

    def __init__(
        self,
        *,
        cause: Exception,
        chain: AuditChain,
        entry: AuditEntry | None,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.chain = chain
        self.entry = entry


# ---------------------------------------------------------------------------
# StageAudit -- context manager for explicit instrumentation
# ---------------------------------------------------------------------------


class StageAudit:
    """Context manager that wraps a pipeline stage with audit tracking.

    Captures a ``StageSnapshot`` on entry (before-snapshot) and on exit
    (after-snapshot), measures wall-clock duration, constructs an
    ``AuditEntry``, and appends it to the immutable ``AuditChain``.

    The ``chain`` and ``entry`` properties are available after the ``with``
    block completes -- even if an exception occurred inside the block.
    Exceptions are never suppressed.

    Args:
        stage: Pipeline stage name (must not be empty).
        chain: The current audit chain to append to.
        inputs: Optional dict of stage input data for the before-snapshot.
        config: Optional dict of stage configuration for the before-snapshot.

    Raises:
        ValueError: If *stage* is empty or whitespace-only.

    Example::

        audit = StageAudit("nl_input", chain, inputs={"raw": text})
        with audit:
            result = do_work()
            audit.record_output({"parsed": result})
        new_chain = audit.chain
    """

    def __init__(
        self,
        stage: str,
        chain: AuditChain,
        *,
        inputs: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._stage = _require_non_empty(stage, "stage")
        self._initial_chain = chain
        self._inputs = dict(inputs) if inputs is not None else {}
        self._config = dict(config) if config is not None else {}
        self._outputs: dict[str, Any] = {}
        self._result_chain: AuditChain | None = None
        self._result_entry: AuditEntry | None = None
        self._start_time: float = 0.0

    def __enter__(self) -> StageAudit:
        self._before_snapshot = capture_snapshot(
            self._stage,
            inputs=self._inputs,
            config=self._config,
        )
        self._start_time = time.monotonic()
        return self

    def record_output(self, outputs: dict[str, Any]) -> None:
        """Record stage outputs for the after-snapshot.

        Can be called multiple times inside the ``with`` block; the last
        call wins. A defensive copy is made so subsequent mutation of the
        dict does not affect the recorded snapshot.

        Args:
            outputs: Dict of output data to include in the after-snapshot.
        """
        self._outputs = dict(outputs)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        duration = time.monotonic() - self._start_time

        is_error = exc_type is not None
        status = "error" if is_error else "success"
        error_msg = str(exc_val) if exc_val is not None else None

        after_outputs = (
            {"error": str(exc_val)} if is_error else dict(self._outputs)
        )
        after_snapshot = capture_snapshot(
            self._stage,
            partial_outputs=after_outputs,
        )

        entry = AuditEntry(
            stage=self._stage,
            timestamp=self._before_snapshot.captured_at,
            before_snapshot=self._before_snapshot.to_dict(),
            after_snapshot=after_snapshot.to_dict(),
            duration=duration,
            status=status,
            error=error_msg,
        )

        self._result_entry = entry
        self._result_chain = self._initial_chain.append(entry)

        return False  # never suppress exceptions

    @property
    def chain(self) -> AuditChain:
        """The audit chain after the context block completes.

        Before entering or during execution, returns the initial chain.
        After exit (success or error), returns the chain with the new
        entry appended.
        """
        if self._result_chain is not None:
            return self._result_chain
        return self._initial_chain

    @property
    def entry(self) -> AuditEntry | None:
        """The audit entry created on exit, or None if not yet complete."""
        return self._result_entry


# ---------------------------------------------------------------------------
# stage_instrumented -- decorator factory
# ---------------------------------------------------------------------------


def _build_inputs_dict(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build an inputs dict from positional and keyword arguments.

    Positional arguments are stored under ``"args"`` as a list.
    Keyword arguments (excluding ``_audit_chain``) are stored under
    ``"kwargs"`` as a dict.
    """
    filtered_kwargs = {
        k: v for k, v in kwargs.items() if k != "_audit_chain"
    }
    return {
        "args": list(args),
        "kwargs": filtered_kwargs,
    }


def stage_instrumented(
    stage: str,
    *,
    config: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Decorator factory that wraps a pipeline stage with audit instrumentation.

    The decorated function gains an optional ``_audit_chain`` keyword
    argument. When provided, the new entry is appended to that chain;
    otherwise a fresh empty chain is used.

    On success, returns a ``StageResult`` containing the function's return
    value, the updated chain, and the appended entry.

    On error, raises ``StageError`` carrying the chain and entry so the
    caller can inspect the audit trail.

    Args:
        stage: Pipeline stage name (must not be empty).
        config: Optional static configuration dict for the before-snapshot.

    Returns:
        A decorator that wraps a callable with audit instrumentation.

    Example::

        @stage_instrumented("nl_input", config={"model": "gpt-4"})
        def translate(text: str) -> str:
            return "pytest -v"

        result = translate("run tests", _audit_chain=chain)
        result.value  # "pytest -v"
        result.chain  # updated chain with entry
    """
    _require_non_empty(stage, "stage")
    stage_config = dict(config) if config is not None else None

    def decorator(fn: Callable[..., Any]) -> Callable[..., StageResult]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> StageResult:
            chain = kwargs.pop("_audit_chain", None)
            if chain is None:
                chain = AuditChain.empty()

            inputs = _build_inputs_dict(args, kwargs)
            audit_ctx = StageAudit(
                stage,
                chain,
                inputs=inputs,
                config=stage_config,
            )
            result: Any = None
            try:
                with audit_ctx:
                    result = fn(*args, **kwargs)
                    audit_ctx.record_output({"return_value": result})
            except Exception as exc:
                raise StageError(
                    cause=exc,
                    chain=audit_ctx.chain,
                    entry=audit_ctx.entry,
                ) from exc

            entry = audit_ctx.entry
            assert entry is not None, "StageAudit.entry must be set after context block"
            return StageResult(
                value=result,
                chain=audit_ctx.chain,
                entry=entry,
            )

        return wrapper

    return decorator
