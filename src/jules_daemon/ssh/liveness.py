"""SSH connection liveness validator.

Executes a lightweight probe command over an established SSH session to
confirm the remote shell is responsive. Returns an immutable ProbeResult
with health classification, latency measurement, and diagnostic details.

The probe executor is Protocol-based, decoupling the liveness check from
any specific SSH library. The caller provides an implementation that
maps to their SSH backend (paramiko, asyncssh, subprocess, etc.).

Health classification:
    - CONNECTED: Probe succeeded, output matched, exit code 0
    - DEGRADED: Transport works but output mismatch or non-zero exit
    - DISCONNECTED: Probe timed out or raised a transport-level error

Usage:
    config = ProbeConfig(timeout_seconds=5.0)
    result = await validate_liveness(executor, config)
    if result.health == ConnectionHealth.CONNECTED:
        # Shell is responsive
        ...
    elif result.health == ConnectionHealth.DEGRADED:
        # Shell responds but something is off
        ...
    else:
        # Shell is unreachable
        ...
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable

__all__ = [
    "ConnectionHealth",
    "ProbeConfig",
    "ProbeExecutor",
    "ProbeResult",
    "validate_liveness",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PROBE_COMMAND = "echo __jules_probe_ok__"
_DEFAULT_EXPECTED_OUTPUT = "__jules_probe_ok__"
_DEFAULT_TIMEOUT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConnectionHealth(Enum):
    """Health classification of an SSH connection after a liveness probe.

    Values:
        CONNECTED: Shell is responsive, probe passed all checks.
        DEGRADED: Transport works but output or exit code is unexpected.
        DISCONNECTED: Shell is unreachable (timeout or transport error).
    """

    CONNECTED = "connected"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    """Immutable configuration for a liveness probe.

    Attributes:
        command: Shell command to execute on the remote host. Should be
            lightweight and side-effect-free. Defaults to an echo command
            that produces a known sentinel string.
        timeout_seconds: Maximum time to wait for the probe command to
            complete. Must be positive.
        expected_output: Expected stdout content (stripped). When non-empty,
            the probe checks that actual output contains this string.
            When empty, any output is accepted (only exit code matters).
    """

    command: str = _DEFAULT_PROBE_COMMAND
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    expected_output: str = _DEFAULT_EXPECTED_OUTPUT

    def __post_init__(self) -> None:
        if not self.command or not self.command.strip():
            raise ValueError("command must not be empty or whitespace-only")
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ProbeExecutor(Protocol):
    """Protocol for executing a probe command over an SSH session.

    Implementations wrap a specific SSH library and execute the given
    command string on the remote host. The method must return the
    stdout output and the exit code.

    The timeout parameter is advisory -- the executor should attempt to
    honor it, but the caller also wraps the call in asyncio.wait_for()
    as a safety net.
    """

    async def execute_probe(
        self, command: str, timeout: float
    ) -> tuple[str, int]:
        """Execute a probe command on the remote host.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds to wait for completion.

        Returns:
            Tuple of (stdout_output, exit_code).

        Raises:
            OSError: Transport-level failure (connection lost, etc.).
            EOFError: Channel closed unexpectedly.
            TimeoutError: Command did not complete within timeout.
        """
        ...


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """Immutable result of a connection liveness probe.

    Attributes:
        success: True if the probe passed all checks (exit code 0 and
            output matched, if expected_output was configured).
        health: Classification of connection health.
        latency_ms: Time from probe start to completion (or timeout)
            in milliseconds.
        output: Stripped stdout from the probe command. Empty string
            on timeout or transport error.
        exit_code: Remote process exit code. None on timeout or error.
        error: Human-readable error description. None on success.
        probe_command: The command string that was executed.
        timestamp: UTC datetime when the probe completed.
    """

    success: bool
    health: ConnectionHealth
    latency_ms: float
    output: str
    exit_code: int | None
    error: str | None
    probe_command: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _build_success(
    *,
    output: str,
    exit_code: int,
    latency_ms: float,
    probe_command: str,
) -> ProbeResult:
    """Build a successful probe result."""
    return ProbeResult(
        success=True,
        health=ConnectionHealth.CONNECTED,
        latency_ms=latency_ms,
        output=output,
        exit_code=exit_code,
        error=None,
        probe_command=probe_command,
        timestamp=_now_utc(),
    )


def _build_degraded(
    *,
    output: str,
    exit_code: int,
    latency_ms: float,
    probe_command: str,
    error: str,
) -> ProbeResult:
    """Build a degraded probe result (transport works but checks failed)."""
    return ProbeResult(
        success=False,
        health=ConnectionHealth.DEGRADED,
        latency_ms=latency_ms,
        output=output,
        exit_code=exit_code,
        error=error,
        probe_command=probe_command,
        timestamp=_now_utc(),
    )


def _build_disconnected(
    *,
    latency_ms: float,
    probe_command: str,
    error: str,
) -> ProbeResult:
    """Build a disconnected probe result (transport-level failure)."""
    return ProbeResult(
        success=False,
        health=ConnectionHealth.DISCONNECTED,
        latency_ms=latency_ms,
        output="",
        exit_code=None,
        error=error,
        probe_command=probe_command,
        timestamp=_now_utc(),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def validate_liveness(
    executor: ProbeExecutor,
    config: ProbeConfig | None = None,
) -> ProbeResult:
    """Execute a liveness probe and classify the connection health.

    Runs the configured probe command via the executor, measures latency,
    and classifies the result:

    1. CONNECTED: Exit code 0 and output matches (if expected).
    2. DEGRADED: Command ran but exit code or output is unexpected.
    3. DISCONNECTED: Timeout or transport-level exception.

    The probe is wrapped in asyncio.wait_for() with the configured
    timeout as a safety net, even if the executor also honors timeouts.

    Args:
        executor: Implementation of ProbeExecutor that runs commands
            over the SSH session.
        config: Probe configuration (command, timeout, expected output).
            When None, uses default ProbeConfig.

    Returns:
        Immutable ProbeResult with health classification and diagnostics.
        Never raises -- all errors are captured in the result.
    """
    if config is None:
        config = ProbeConfig()
    probe_command = config.command
    start_ns = time.monotonic_ns()

    try:
        raw_output, exit_code = await asyncio.wait_for(
            executor.execute_probe(probe_command, config.timeout_seconds),
            timeout=config.timeout_seconds,
        )
    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        logger.warning(
            "Liveness probe timed out after %.1fms (command: %s)",
            elapsed_ms,
            probe_command,
        )
        return _build_disconnected(
            latency_ms=elapsed_ms,
            probe_command=probe_command,
            error=f"Probe timeout after {elapsed_ms:.1f}ms",
        )
    except Exception as exc:
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Liveness probe failed after %.1fms: %s (command: %s)",
            elapsed_ms,
            error_msg,
            probe_command,
        )
        return _build_disconnected(
            latency_ms=elapsed_ms,
            probe_command=probe_command,
            error=error_msg,
        )

    elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
    output = raw_output.strip()

    # Check exit code
    if exit_code != 0:
        logger.info(
            "Liveness probe returned non-zero exit code %d (%.1fms, command: %s)",
            exit_code,
            elapsed_ms,
            probe_command,
        )
        return _build_degraded(
            output=output,
            exit_code=exit_code,
            latency_ms=elapsed_ms,
            probe_command=probe_command,
            error=f"Non-zero exit code: {exit_code}",
        )

    # Check expected output (only when non-empty)
    if config.expected_output and config.expected_output not in output:
        logger.info(
            "Liveness probe output mismatch: expected %r in %r (%.1fms)",
            config.expected_output,
            output,
            elapsed_ms,
        )
        return _build_degraded(
            output=output,
            exit_code=exit_code,
            latency_ms=elapsed_ms,
            probe_command=probe_command,
            error=(
                f"Output mismatch: expected {config.expected_output!r} "
                f"in {output!r}"
            ),
        )

    # All checks passed
    logger.debug(
        "Liveness probe succeeded (%.1fms, command: %s)",
        elapsed_ms,
        probe_command,
    )
    return _build_success(
        output=output,
        exit_code=exit_code,
        latency_ms=elapsed_ms,
        probe_command=probe_command,
    )
