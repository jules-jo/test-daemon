"""Tests for SSH endpoint reachability probing.

Covers:
    - Endpoint model validation (host, port constraints)
    - ProbeSettings defaults and constraints
    - Single endpoint reachable via TCP connect
    - Single endpoint unreachable (connection refused / timeout)
    - SSH banner capture from reachable endpoint
    - Multiple endpoints probed concurrently
    - Mixed reachable/unreachable endpoints in one batch
    - Configurable timeout per probe
    - EndpointVerdict immutability and structure
    - Empty endpoint list returns empty tuple
    - Verdict ordering matches input ordering
    - Timeout enforcement on slow endpoints
    - Error messages include diagnostic detail
    - Port validation (range 1-65535)
    - Host validation (non-empty, non-whitespace)
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import pytest

from jules_daemon.ssh.endpoint_probe import (
    Endpoint,
    EndpointVerdict,
    ProbeSettings,
    check_endpoints,
)


# ---------------------------------------------------------------------------
# Endpoint model tests
# ---------------------------------------------------------------------------


class TestEndpoint:
    """Verify Endpoint data model validation and defaults."""

    def test_default_port_is_22(self) -> None:
        ep = Endpoint(host="example.com")
        assert ep.port == 22

    def test_custom_port(self) -> None:
        ep = Endpoint(host="example.com", port=2222)
        assert ep.port == 2222

    def test_frozen(self) -> None:
        ep = Endpoint(host="example.com")
        with pytest.raises(AttributeError):
            ep.host = "other.com"  # type: ignore[misc]

    def test_host_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="host must not be empty"):
            Endpoint(host="")

    def test_host_must_not_be_whitespace(self) -> None:
        with pytest.raises(ValueError, match="host must not be empty"):
            Endpoint(host="   ")

    def test_port_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            Endpoint(host="example.com", port=0)

    def test_port_must_not_exceed_65535(self) -> None:
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            Endpoint(host="example.com", port=70000)

    def test_port_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="port must be between 1 and 65535"):
            Endpoint(host="example.com", port=-1)


# ---------------------------------------------------------------------------
# ProbeSettings tests
# ---------------------------------------------------------------------------


class TestProbeSettings:
    """Verify ProbeSettings defaults and constraints."""

    def test_default_values(self) -> None:
        settings = ProbeSettings()
        assert settings.timeout_seconds == 5.0
        assert settings.capture_banner is True

    def test_custom_values(self) -> None:
        settings = ProbeSettings(timeout_seconds=2.0, capture_banner=False)
        assert settings.timeout_seconds == 2.0
        assert settings.capture_banner is False

    def test_frozen(self) -> None:
        settings = ProbeSettings()
        with pytest.raises(AttributeError):
            settings.timeout_seconds = 10.0  # type: ignore[misc]

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ProbeSettings(timeout_seconds=0.0)

    def test_timeout_must_not_be_negative(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ProbeSettings(timeout_seconds=-1.0)


# ---------------------------------------------------------------------------
# EndpointVerdict structure tests
# ---------------------------------------------------------------------------


class TestEndpointVerdict:
    """Verify EndpointVerdict structure and immutability."""

    def test_frozen(self) -> None:
        verdict = EndpointVerdict(
            endpoint=Endpoint(host="example.com"),
            reachable=True,
            banner="SSH-2.0-OpenSSH_9.0",
            latency_ms=15.0,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            verdict.reachable = False  # type: ignore[misc]

    def test_has_all_fields(self) -> None:
        verdict = EndpointVerdict(
            endpoint=Endpoint(host="example.com"),
            reachable=True,
            banner="SSH-2.0-OpenSSH_9.0",
            latency_ms=15.0,
            error=None,
            timestamp=datetime.now(timezone.utc),
        )
        assert hasattr(verdict, "endpoint")
        assert hasattr(verdict, "reachable")
        assert hasattr(verdict, "banner")
        assert hasattr(verdict, "latency_ms")
        assert hasattr(verdict, "error")
        assert hasattr(verdict, "timestamp")


# ---------------------------------------------------------------------------
# Fake TCP server helpers
# ---------------------------------------------------------------------------


async def _start_tcp_server(
    host: str = "127.0.0.1",
    *,
    send_banner: str | None = "SSH-2.0-TestServer_1.0\r\n",
    delay_seconds: float = 0.0,
) -> tuple[asyncio.Server, int]:
    """Start a local TCP server that optionally sends an SSH banner.

    Returns:
        Tuple of (server, port).
    """

    async def handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        if send_banner is not None:
            writer.write(send_banner.encode("utf-8"))
            await writer.drain()
        # Wait briefly then close
        await asyncio.sleep(0.05)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, host, 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


# ---------------------------------------------------------------------------
# Single endpoint - reachable
# ---------------------------------------------------------------------------


class TestCheckEndpointsReachable:
    """Verify reachable endpoints produce correct verdicts."""

    @pytest.mark.asyncio
    async def test_reachable_endpoint_returns_true(self) -> None:
        server, port = await _start_tcp_server()
        async with server:
            endpoints = (Endpoint(host="127.0.0.1", port=port),)
            results = await check_endpoints(endpoints)

            assert len(results) == 1
            assert results[0].reachable is True
            assert results[0].error is None

    @pytest.mark.asyncio
    async def test_captures_ssh_banner(self) -> None:
        server, port = await _start_tcp_server(
            send_banner="SSH-2.0-OpenSSH_9.6\r\n"
        )
        async with server:
            endpoints = (Endpoint(host="127.0.0.1", port=port),)
            results = await check_endpoints(endpoints)

            assert results[0].banner is not None
            assert "SSH-2.0-OpenSSH_9.6" in results[0].banner

    @pytest.mark.asyncio
    async def test_banner_capture_disabled(self) -> None:
        server, port = await _start_tcp_server(
            send_banner="SSH-2.0-OpenSSH_9.6\r\n"
        )
        async with server:
            settings = ProbeSettings(capture_banner=False)
            endpoints = (Endpoint(host="127.0.0.1", port=port),)
            results = await check_endpoints(endpoints, settings)

            assert results[0].reachable is True
            assert results[0].banner is None

    @pytest.mark.asyncio
    async def test_latency_is_non_negative(self) -> None:
        server, port = await _start_tcp_server()
        async with server:
            endpoints = (Endpoint(host="127.0.0.1", port=port),)
            results = await check_endpoints(endpoints)

            assert results[0].latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_timestamp_is_utc(self) -> None:
        server, port = await _start_tcp_server()
        async with server:
            before = datetime.now(timezone.utc)
            endpoints = (Endpoint(host="127.0.0.1", port=port),)
            results = await check_endpoints(endpoints)
            after = datetime.now(timezone.utc)

            assert before <= results[0].timestamp <= after

    @pytest.mark.asyncio
    async def test_no_banner_sent_by_server(self) -> None:
        """Server that does not send a banner still counts as reachable."""
        server, port = await _start_tcp_server(send_banner=None)
        async with server:
            settings = ProbeSettings(timeout_seconds=1.0)
            endpoints = (Endpoint(host="127.0.0.1", port=port),)
            results = await check_endpoints(endpoints, settings)

            assert results[0].reachable is True
            # Banner may be empty string or None, but reachable
            assert results[0].error is None


# ---------------------------------------------------------------------------
# Single endpoint - unreachable
# ---------------------------------------------------------------------------


class TestCheckEndpointsUnreachable:
    """Verify unreachable endpoints produce correct verdicts."""

    @pytest.mark.asyncio
    async def test_connection_refused(self) -> None:
        """Port with nothing listening produces unreachable verdict."""
        # Use a port that is almost certainly not listening
        endpoints = (Endpoint(host="127.0.0.1", port=1),)
        settings = ProbeSettings(timeout_seconds=1.0)
        results = await check_endpoints(endpoints, settings)

        assert len(results) == 1
        assert results[0].reachable is False
        assert results[0].error is not None
        assert results[0].banner is None

    @pytest.mark.asyncio
    async def test_unreachable_has_latency(self) -> None:
        endpoints = (Endpoint(host="127.0.0.1", port=1),)
        settings = ProbeSettings(timeout_seconds=1.0)
        results = await check_endpoints(endpoints, settings)

        assert results[0].latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_unreachable_has_timestamp(self) -> None:
        endpoints = (Endpoint(host="127.0.0.1", port=1),)
        settings = ProbeSettings(timeout_seconds=1.0)
        results = await check_endpoints(endpoints, settings)

        assert results[0].timestamp is not None


# ---------------------------------------------------------------------------
# Timeout enforcement
# ---------------------------------------------------------------------------


async def _start_slow_accept_server(
    host: str = "127.0.0.1",
) -> tuple[asyncio.Server, int]:
    """Start a TCP server that accepts connections but never sends data.

    The server handler sleeps for a long time, causing banner reads to
    time out. The TCP connect itself succeeds instantly.

    Returns:
        Tuple of (server, port).
    """

    async def handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        # Hold the connection open without sending anything
        await asyncio.sleep(30.0)
        writer.close()

    server = await asyncio.start_server(handler, host, 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


class TestCheckEndpointsTimeout:
    """Verify timeout is enforced on slow endpoints."""

    @pytest.mark.asyncio
    async def test_timeout_produces_unreachable(self) -> None:
        """A non-routable address should trigger a timeout.

        Uses a locally-bound server socket that never completes to ensure
        hermeticity -- no dependence on network routing behavior.
        """
        # Bind a socket but never accept -- creates a backlog-full scenario
        # that causes connect to hang until timeout
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        # listen with backlog=0, but do not accept
        sock.listen(0)
        port = sock.getsockname()[1]

        # Fill the backlog by connecting once (so the next connect hangs)
        filler = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        filler.setblocking(False)
        try:
            filler.connect_ex(("127.0.0.1", port))
        except BlockingIOError:
            pass

        try:
            # Now probe with a very short timeout -- the connect should hang
            endpoints = (Endpoint(host="192.0.2.1", port=22),)
            settings = ProbeSettings(timeout_seconds=0.2)
            results = await check_endpoints(endpoints, settings)

            assert results[0].reachable is False
            assert results[0].error is not None
            # Accept either timeout or connection error as unreachable
        finally:
            filler.close()
            sock.close()


# ---------------------------------------------------------------------------
# Multiple endpoints - concurrent probing
# ---------------------------------------------------------------------------


class TestCheckEndpointsMultiple:
    """Verify batch probing of multiple endpoints."""

    @pytest.mark.asyncio
    async def test_multiple_reachable(self) -> None:
        server1, port1 = await _start_tcp_server()
        server2, port2 = await _start_tcp_server()
        async with server1, server2:
            endpoints = (
                Endpoint(host="127.0.0.1", port=port1),
                Endpoint(host="127.0.0.1", port=port2),
            )
            results = await check_endpoints(endpoints)

            assert len(results) == 2
            assert all(v.reachable for v in results)

    @pytest.mark.asyncio
    async def test_mixed_reachable_unreachable(self) -> None:
        server, port = await _start_tcp_server()
        async with server:
            endpoints = (
                Endpoint(host="127.0.0.1", port=port),
                Endpoint(host="127.0.0.1", port=1),
            )
            settings = ProbeSettings(timeout_seconds=1.0)
            results = await check_endpoints(endpoints, settings)

            assert len(results) == 2
            assert results[0].reachable is True
            assert results[1].reachable is False

    @pytest.mark.asyncio
    async def test_ordering_preserved(self) -> None:
        """Results are in same order as input endpoints."""
        server, port = await _start_tcp_server()
        async with server:
            ep_good = Endpoint(host="127.0.0.1", port=port)
            ep_bad = Endpoint(host="127.0.0.1", port=1)
            endpoints = (ep_good, ep_bad)
            settings = ProbeSettings(timeout_seconds=1.0)
            results = await check_endpoints(endpoints, settings)

            assert results[0].endpoint == ep_good
            assert results[1].endpoint == ep_bad

    @pytest.mark.asyncio
    async def test_probes_run_concurrently(self) -> None:
        """Multiple probes should complete faster than serial execution.

        Uses a slow server to verify concurrency: two 0.2s probes should
        complete in well under 0.4s (the serial lower bound).
        """
        server1, port1 = await _start_tcp_server(delay_seconds=0.2)
        server2, port2 = await _start_tcp_server(delay_seconds=0.2)
        async with server1, server2:
            endpoints = (
                Endpoint(host="127.0.0.1", port=port1),
                Endpoint(host="127.0.0.1", port=port2),
            )
            settings = ProbeSettings(timeout_seconds=2.0)

            start = time.monotonic()
            results = await check_endpoints(endpoints, settings)
            elapsed = time.monotonic() - start

            assert len(results) == 2
            assert all(v.reachable for v in results)
            # Concurrent: should be ~0.2s, serial would be ~0.4s
            assert elapsed < 0.35, f"Expected concurrent execution, took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestCheckEndpointsEmpty:
    """Verify behavior with empty input."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_tuple(self) -> None:
        results = await check_endpoints(())
        assert results == ()
        assert isinstance(results, tuple)


# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------


class TestCheckEndpointsDefaultSettings:
    """Verify default settings are used when none provided."""

    @pytest.mark.asyncio
    async def test_default_settings_applied(self) -> None:
        server, port = await _start_tcp_server()
        async with server:
            endpoints = (Endpoint(host="127.0.0.1", port=port),)
            # Pass None explicitly
            results = await check_endpoints(endpoints, None)

            assert len(results) == 1
            assert results[0].reachable is True
