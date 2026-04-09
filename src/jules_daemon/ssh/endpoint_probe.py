"""SSH endpoint reachability probing via TCP connect and banner capture.

Performs lightweight, connectionless health checks against remote host/port
pairs without requiring an established SSH session. Useful for pre-flight
validation before attempting full SSH handshake.

Each probe opens a raw TCP socket and optionally reads the SSH server
banner (the first line the server sends per RFC 4253). Probes for all
endpoints run concurrently via asyncio.gather().

Usage:
    endpoints = (
        Endpoint(host="web1.example.com", port=22),
        Endpoint(host="web2.example.com", port=2222),
    )
    settings = ProbeSettings(timeout_seconds=3.0)
    verdicts = await check_endpoints(endpoints, settings)

    for v in verdicts:
        status = "reachable" if v.reachable else "unreachable"
        print(f"{v.endpoint.host}:{v.endpoint.port} -> {status}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

__all__ = [
    "Endpoint",
    "EndpointVerdict",
    "ProbeSettings",
    "check_endpoints",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT_SECONDS = 5.0
_BANNER_READ_TIMEOUT_SECONDS = 2.0
_MAX_BANNER_BYTES = 512
_MIN_PORT = 1
_MAX_PORT = 65535


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """Immutable host/port pair representing a remote SSH endpoint.

    Attributes:
        host: Hostname or IP address of the remote system.
        port: TCP port number (1-65535). Defaults to 22 (standard SSH).
    """

    host: str
    port: int = 22

    def __post_init__(self) -> None:
        if not self.host or not self.host.strip():
            raise ValueError("host must not be empty or whitespace-only")
        if not (_MIN_PORT <= self.port <= _MAX_PORT):
            raise ValueError(
                f"port must be between {_MIN_PORT} and {_MAX_PORT}, "
                f"got {self.port}"
            )


@dataclass(frozen=True)
class ProbeSettings:
    """Immutable configuration for endpoint reachability probes.

    Attributes:
        timeout_seconds: Maximum time for TCP connect + optional banner
            read. Must be positive.
        capture_banner: When True, attempt to read the SSH banner after
            establishing the TCP connection. When False, only verify TCP
            connectivity.
    """

    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    capture_banner: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )


@dataclass(frozen=True)
class EndpointVerdict:
    """Immutable result of a single endpoint reachability probe.

    Attributes:
        endpoint: The endpoint that was probed.
        reachable: True if TCP connection succeeded within the timeout.
        banner: SSH server banner string (stripped), or None if banner
            capture was disabled, timed out, or the server did not send one.
        latency_ms: Time from probe start to completion in milliseconds.
        error: Human-readable error description, or None on success.
        timestamp: UTC datetime when the probe completed.
    """

    endpoint: Endpoint
    reachable: bool
    banner: str | None
    latency_ms: float
    error: str | None
    timestamp: datetime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _build_reachable(
    *,
    endpoint: Endpoint,
    banner: str | None,
    latency_ms: float,
) -> EndpointVerdict:
    """Build a reachable verdict."""
    return EndpointVerdict(
        endpoint=endpoint,
        reachable=True,
        banner=banner,
        latency_ms=latency_ms,
        error=None,
        timestamp=_now_utc(),
    )


def _build_unreachable(
    *,
    endpoint: Endpoint,
    latency_ms: float,
    error: str,
) -> EndpointVerdict:
    """Build an unreachable verdict."""
    return EndpointVerdict(
        endpoint=endpoint,
        reachable=False,
        banner=None,
        latency_ms=latency_ms,
        error=error,
        timestamp=_now_utc(),
    )


async def _read_banner(
    reader: asyncio.StreamReader,
    timeout: float,
) -> str | None:
    """Attempt to read an SSH banner from the stream.

    SSH servers send a version identification string (banner) as the first
    line upon connection per RFC 4253. This reads up to _MAX_BANNER_BYTES
    within the given timeout.

    Returns:
        Stripped banner string, or None if read times out or fails.
    """
    try:
        raw = await asyncio.wait_for(
            reader.read(_MAX_BANNER_BYTES),
            timeout=timeout,
        )
        if raw:
            return raw.decode("utf-8", errors="replace").strip()
    except asyncio.TimeoutError:
        logger.debug("Banner read timed out for stream")
    except OSError as exc:
        logger.debug("Banner read failed: %s", exc)
    return None


async def _probe_single(
    endpoint: Endpoint,
    settings: ProbeSettings,
) -> EndpointVerdict:
    """Probe a single endpoint for TCP reachability and optional banner.

    Opens a TCP connection to the endpoint within the configured timeout.
    If capture_banner is enabled and connection succeeds, attempts to read
    the SSH banner with a sub-timeout.

    Args:
        endpoint: Target host/port pair.
        settings: Probe configuration.

    Returns:
        EndpointVerdict with reachable/unreachable classification.
        Never raises -- all errors are captured in the verdict.
    """
    start_ns = time.monotonic_ns()

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(endpoint.host, endpoint.port),
            timeout=settings.timeout_seconds,
        )
    except asyncio.TimeoutError:
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        logger.info(
            "Endpoint probe timeout: %s:%d after %.1fms",
            endpoint.host,
            endpoint.port,
            elapsed_ms,
        )
        return _build_unreachable(
            endpoint=endpoint,
            latency_ms=elapsed_ms,
            error=f"Connection timeout after {elapsed_ms:.1f}ms",
        )
    except OSError as exc:
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.info(
            "Endpoint probe failed: %s:%d -> %s (%.1fms)",
            endpoint.host,
            endpoint.port,
            error_msg,
            elapsed_ms,
        )
        return _build_unreachable(
            endpoint=endpoint,
            latency_ms=elapsed_ms,
            error=error_msg,
        )

    # TCP connection succeeded -- ensure writer is always closed
    banner: str | None = None
    try:
        if settings.capture_banner:
            # Use a sub-timeout for banner read, capped to remaining time
            elapsed_so_far = (time.monotonic_ns() - start_ns) / 1_000_000_000
            remaining = max(0.1, settings.timeout_seconds - elapsed_so_far)
            banner_timeout = min(_BANNER_READ_TIMEOUT_SECONDS, remaining)
            banner = await _read_banner(reader, banner_timeout)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except OSError:
            pass  # Connection may have already been closed by server

    elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
    logger.debug(
        "Endpoint probe succeeded: %s:%d (%.1fms, banner=%s)",
        endpoint.host,
        endpoint.port,
        elapsed_ms,
        repr(banner),
    )
    return _build_reachable(
        endpoint=endpoint,
        banner=banner,
        latency_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def check_endpoints(
    endpoints: Sequence[Endpoint],
    settings: ProbeSettings | None = None,
) -> tuple[EndpointVerdict, ...]:
    """Probe multiple SSH endpoints for TCP reachability concurrently.

    Performs a lightweight TCP connect (and optional SSH banner read) to
    each endpoint in the list. All probes run concurrently via
    asyncio.gather(), so the total wall-clock time is bounded by the
    slowest probe rather than the sum.

    Results are returned in the same order as the input endpoints.

    Args:
        endpoints: Sequence of Endpoint objects to probe. May be empty.
        settings: Probe configuration (timeout, banner capture). When
            None, uses default ProbeSettings.

    Returns:
        Tuple of EndpointVerdict objects, one per input endpoint, in the
        same order. Never raises -- all errors are captured per-verdict.
    """
    if not endpoints:
        return ()

    if settings is None:
        settings = ProbeSettings()

    verdicts = await asyncio.gather(
        *(_probe_single(ep, settings) for ep in endpoints)
    )
    return tuple(verdicts)
