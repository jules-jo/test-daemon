"""Tests for the server-side SSH job output broadcaster."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from jules_daemon.monitor.output_broadcaster import (
    BroadcasterConfig,
    JobOutputBroadcaster,
    OutputLine,
    SubscriberHandle,
)


# ---------------------------------------------------------------------------
# OutputLine
# ---------------------------------------------------------------------------


class TestOutputLine:
    def test_create(self) -> None:
        line = OutputLine(
            job_id="job-1",
            line="PASSED test_login",
            sequence=0,
            timestamp="2026-04-09T12:00:00+00:00",
        )
        assert line.job_id == "job-1"
        assert line.line == "PASSED test_login"
        assert line.sequence == 0
        assert line.timestamp == "2026-04-09T12:00:00+00:00"

    def test_frozen(self) -> None:
        line = OutputLine(
            job_id="job-1",
            line="x",
            sequence=0,
            timestamp="2026-04-09T12:00:00+00:00",
        )
        with pytest.raises(AttributeError):
            line.line = "mutated"  # type: ignore[misc]

    def test_empty_job_id_raises(self) -> None:
        with pytest.raises(ValueError, match="job_id must not be empty"):
            OutputLine(job_id="", line="x", sequence=0, timestamp="t")

    def test_negative_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="sequence must not be negative"):
            OutputLine(job_id="j", line="x", sequence=-1, timestamp="t")


# ---------------------------------------------------------------------------
# BroadcasterConfig
# ---------------------------------------------------------------------------


class TestBroadcasterConfig:
    def test_defaults(self) -> None:
        config = BroadcasterConfig()
        assert config.buffer_size == 1000
        assert config.subscriber_queue_size == 500

    def test_custom_values(self) -> None:
        config = BroadcasterConfig(buffer_size=50, subscriber_queue_size=25)
        assert config.buffer_size == 50
        assert config.subscriber_queue_size == 25

    def test_frozen(self) -> None:
        config = BroadcasterConfig()
        with pytest.raises(AttributeError):
            config.buffer_size = 42  # type: ignore[misc]

    def test_zero_buffer_raises(self) -> None:
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            BroadcasterConfig(buffer_size=0)

    def test_negative_buffer_raises(self) -> None:
        with pytest.raises(ValueError, match="buffer_size must be positive"):
            BroadcasterConfig(buffer_size=-1)

    def test_zero_queue_size_raises(self) -> None:
        with pytest.raises(ValueError, match="subscriber_queue_size must be positive"):
            BroadcasterConfig(subscriber_queue_size=0)


# ---------------------------------------------------------------------------
# SubscriberHandle
# ---------------------------------------------------------------------------


class TestSubscriberHandle:
    def test_create(self) -> None:
        handle = SubscriberHandle(subscriber_id="sub-abc", job_id="job-1")
        assert handle.subscriber_id == "sub-abc"
        assert handle.job_id == "job-1"

    def test_frozen(self) -> None:
        handle = SubscriberHandle(subscriber_id="sub-abc", job_id="job-1")
        with pytest.raises(AttributeError):
            handle.job_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# JobOutputBroadcaster -- job registration
# ---------------------------------------------------------------------------


class TestJobRegistration:
    def test_register_job(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        assert broadcaster.is_registered("job-1")

    def test_register_duplicate_is_idempotent(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        broadcaster.register_job("job-1")  # should not raise
        assert broadcaster.is_registered("job-1")

    def test_unregister_job(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        broadcaster.unregister_job("job-1")
        assert not broadcaster.is_registered("job-1")

    def test_unregister_unknown_job_is_safe(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.unregister_job("nonexistent")  # no-op

    def test_registered_job_ids(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-a")
        broadcaster.register_job("job-b")
        ids = broadcaster.registered_job_ids()
        assert ids == frozenset({"job-a", "job-b"})


# ---------------------------------------------------------------------------
# JobOutputBroadcaster -- publish
# ---------------------------------------------------------------------------


class TestPublish:
    def test_publish_to_registered_job(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        broadcaster.publish("job-1", "line one")
        buffer = broadcaster.get_buffer("job-1")
        assert len(buffer) == 1
        assert buffer[0].line == "line one"
        assert buffer[0].job_id == "job-1"
        assert buffer[0].sequence == 0

    def test_publish_increments_sequence(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        broadcaster.publish("job-1", "line 0")
        broadcaster.publish("job-1", "line 1")
        broadcaster.publish("job-1", "line 2")
        buffer = broadcaster.get_buffer("job-1")
        assert [ol.sequence for ol in buffer] == [0, 1, 2]

    def test_publish_to_unregistered_job_raises(self) -> None:
        broadcaster = JobOutputBroadcaster()
        with pytest.raises(ValueError, match="not registered"):
            broadcaster.publish("unknown", "line")

    def test_buffer_respects_max_size(self) -> None:
        config = BroadcasterConfig(buffer_size=3)
        broadcaster = JobOutputBroadcaster(config=config)
        broadcaster.register_job("job-1")
        for i in range(5):
            broadcaster.publish("job-1", f"line {i}")
        buffer = broadcaster.get_buffer("job-1")
        assert len(buffer) == 3
        # Oldest lines evicted, newest retained
        assert [ol.line for ol in buffer] == ["line 2", "line 3", "line 4"]

    def test_get_buffer_for_unregistered_returns_empty(self) -> None:
        broadcaster = JobOutputBroadcaster()
        assert broadcaster.get_buffer("nonexistent") == ()

    def test_publish_returns_output_line(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        result = broadcaster.publish("job-1", "hello")
        assert isinstance(result, OutputLine)
        assert result.line == "hello"
        assert result.job_id == "job-1"


# ---------------------------------------------------------------------------
# JobOutputBroadcaster -- subscribe / unsubscribe
# ---------------------------------------------------------------------------


class TestSubscription:
    def test_subscribe_returns_handle(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        handle = broadcaster.subscribe("job-1")
        assert isinstance(handle, SubscriberHandle)
        assert handle.job_id == "job-1"
        assert handle.subscriber_id.startswith("sub-")

    def test_subscribe_to_unregistered_job_raises(self) -> None:
        broadcaster = JobOutputBroadcaster()
        with pytest.raises(ValueError, match="not registered"):
            broadcaster.subscribe("unknown")

    def test_subscriber_count(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        assert broadcaster.subscriber_count("job-1") == 0
        h1 = broadcaster.subscribe("job-1")
        assert broadcaster.subscriber_count("job-1") == 1
        h2 = broadcaster.subscribe("job-1")
        assert broadcaster.subscriber_count("job-1") == 2
        broadcaster.unsubscribe(h1)
        assert broadcaster.subscriber_count("job-1") == 1
        broadcaster.unsubscribe(h2)
        assert broadcaster.subscriber_count("job-1") == 0

    def test_unsubscribe_idempotent(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        handle = broadcaster.subscribe("job-1")
        broadcaster.unsubscribe(handle)
        broadcaster.unsubscribe(handle)  # should not raise
        assert broadcaster.subscriber_count("job-1") == 0

    def test_subscriber_count_unregistered_returns_zero(self) -> None:
        broadcaster = JobOutputBroadcaster()
        assert broadcaster.subscriber_count("nonexistent") == 0


# ---------------------------------------------------------------------------
# JobOutputBroadcaster -- fan-out delivery
# ---------------------------------------------------------------------------


class TestFanOut:
    @pytest.mark.asyncio
    async def test_publish_delivers_to_single_subscriber(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        handle = broadcaster.subscribe("job-1")
        broadcaster.publish("job-1", "test output")
        line = await broadcaster.receive(handle, timeout=1.0)
        assert line is not None
        assert line.line == "test output"
        assert line.job_id == "job-1"

    @pytest.mark.asyncio
    async def test_publish_delivers_to_multiple_subscribers(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        h1 = broadcaster.subscribe("job-1")
        h2 = broadcaster.subscribe("job-1")
        h3 = broadcaster.subscribe("job-1")
        broadcaster.publish("job-1", "shared line")

        for handle in (h1, h2, h3):
            line = await broadcaster.receive(handle, timeout=1.0)
            assert line is not None
            assert line.line == "shared line"

    @pytest.mark.asyncio
    async def test_subscribers_are_independent(self) -> None:
        """Each subscriber gets its own copy; consuming from one does
        not affect the others."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        h1 = broadcaster.subscribe("job-1")
        h2 = broadcaster.subscribe("job-1")
        broadcaster.publish("job-1", "line A")
        broadcaster.publish("job-1", "line B")

        # Consume from h1 only
        a1 = await broadcaster.receive(h1, timeout=1.0)
        b1 = await broadcaster.receive(h1, timeout=1.0)
        assert a1 is not None and a1.line == "line A"
        assert b1 is not None and b1.line == "line B"

        # h2 still has both lines queued
        a2 = await broadcaster.receive(h2, timeout=1.0)
        b2 = await broadcaster.receive(h2, timeout=1.0)
        assert a2 is not None and a2.line == "line A"
        assert b2 is not None and b2.line == "line B"

    @pytest.mark.asyncio
    async def test_receive_returns_none_on_timeout(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        handle = broadcaster.subscribe("job-1")
        result = await broadcaster.receive(handle, timeout=0.05)
        assert result is None

    @pytest.mark.asyncio
    async def test_receive_invalid_handle_raises(self) -> None:
        broadcaster = JobOutputBroadcaster()
        bad_handle = SubscriberHandle(subscriber_id="bogus", job_id="none")
        with pytest.raises(ValueError, match="not found"):
            await broadcaster.receive(bad_handle, timeout=0.05)

    @pytest.mark.asyncio
    async def test_multiple_jobs_are_isolated(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-a")
        broadcaster.register_job("job-b")
        ha = broadcaster.subscribe("job-a")
        hb = broadcaster.subscribe("job-b")
        broadcaster.publish("job-a", "from A")
        broadcaster.publish("job-b", "from B")

        line_a = await broadcaster.receive(ha, timeout=1.0)
        line_b = await broadcaster.receive(hb, timeout=1.0)
        assert line_a is not None and line_a.line == "from A"
        assert line_a.job_id == "job-a"
        assert line_b is not None and line_b.line == "from B"
        assert line_b.job_id == "job-b"

    @pytest.mark.asyncio
    async def test_late_subscriber_does_not_get_old_lines(self) -> None:
        """A subscriber joining after publish should not receive
        previously published lines (buffer replay is explicit)."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        broadcaster.publish("job-1", "before subscribe")
        handle = broadcaster.subscribe("job-1")
        result = await broadcaster.receive(handle, timeout=0.05)
        assert result is None


# ---------------------------------------------------------------------------
# JobOutputBroadcaster -- buffer replay
# ---------------------------------------------------------------------------


class TestBufferReplay:
    @pytest.mark.asyncio
    async def test_replay_buffer_to_subscriber(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        broadcaster.publish("job-1", "line 0")
        broadcaster.publish("job-1", "line 1")
        broadcaster.publish("job-1", "line 2")

        handle = broadcaster.subscribe("job-1")
        replayed = await broadcaster.replay_buffer(handle)
        assert len(replayed) == 3
        assert [r.line for r in replayed] == ["line 0", "line 1", "line 2"]

    @pytest.mark.asyncio
    async def test_replay_empty_buffer(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        handle = broadcaster.subscribe("job-1")
        replayed = await broadcaster.replay_buffer(handle)
        assert replayed == ()

    @pytest.mark.asyncio
    async def test_replay_enqueues_to_subscriber(self) -> None:
        """After replay, lines should be receivable from the subscriber queue."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        broadcaster.publish("job-1", "buffered line")

        handle = broadcaster.subscribe("job-1")
        await broadcaster.replay_buffer(handle)

        line = await broadcaster.receive(handle, timeout=1.0)
        assert line is not None
        assert line.line == "buffered line"

    @pytest.mark.asyncio
    async def test_replay_invalid_handle_raises(self) -> None:
        broadcaster = JobOutputBroadcaster()
        bad_handle = SubscriberHandle(subscriber_id="bogus", job_id="none")
        with pytest.raises(ValueError, match="not found"):
            await broadcaster.replay_buffer(bad_handle)


# ---------------------------------------------------------------------------
# JobOutputBroadcaster -- unregister with active subscribers
# ---------------------------------------------------------------------------


class TestUnregisterWithSubscribers:
    @pytest.mark.asyncio
    async def test_unregister_signals_end_to_subscribers(self) -> None:
        """When a job is unregistered, active subscribers should receive
        a sentinel (None line) so they know the stream ended."""
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        handle = broadcaster.subscribe("job-1")
        broadcaster.unregister_job("job-1")

        # The sentinel should be an OutputLine with empty line and is_end=True
        line = await broadcaster.receive(handle, timeout=1.0)
        assert line is not None
        assert line.is_end is True

    @pytest.mark.asyncio
    async def test_unregister_clears_subscriber_count(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        broadcaster.subscribe("job-1")
        broadcaster.subscribe("job-1")
        broadcaster.unregister_job("job-1")
        assert broadcaster.subscriber_count("job-1") == 0


# ---------------------------------------------------------------------------
# JobOutputBroadcaster -- slow subscriber backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_slow_subscriber_drops_oldest_on_overflow(self) -> None:
        """When a subscriber queue is full, the oldest entry is dropped
        to make room, so the producer never blocks."""
        config = BroadcasterConfig(buffer_size=100, subscriber_queue_size=3)
        broadcaster = JobOutputBroadcaster(config=config)
        broadcaster.register_job("job-1")
        handle = broadcaster.subscribe("job-1")

        # Publish more lines than the subscriber queue can hold
        for i in range(6):
            broadcaster.publish("job-1", f"line {i}")

        # Only the 3 most recent should remain
        received = []
        for _ in range(3):
            line = await broadcaster.receive(handle, timeout=0.1)
            if line is not None:
                received.append(line.line)

        assert received == ["line 3", "line 4", "line 5"]


# ---------------------------------------------------------------------------
# JobOutputBroadcaster -- concurrent async iteration
# ---------------------------------------------------------------------------


class TestAsyncIteration:
    @pytest.mark.asyncio
    async def test_iter_lines_yields_published_lines(self) -> None:
        broadcaster = JobOutputBroadcaster()
        broadcaster.register_job("job-1")
        handle = broadcaster.subscribe("job-1")

        broadcaster.publish("job-1", "alpha")
        broadcaster.publish("job-1", "beta")

        # Unregister to send end sentinel after a short delay
        async def unregister_later() -> None:
            await asyncio.sleep(0.05)
            broadcaster.unregister_job("job-1")

        task = asyncio.create_task(unregister_later())

        collected: list[str] = []
        async for line in broadcaster.iter_lines(handle):
            collected.append(line.line)

        await task
        assert collected == ["alpha", "beta"]
