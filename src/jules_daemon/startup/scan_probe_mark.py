"""Scan-probe-mark pipeline for daemon startup session cleanup.

Orchestrates the three-phase session cleanup pipeline that runs before
the daemon transitions to its ready/accepting state:

1. SCAN: Discover all active sessions in the wiki via session_scanner
2. PROBE: Check liveness of each active session:
   - Local daemon PID alive? (via process_state.check_pid)
   - SSH endpoint reachable? (via endpoint_probe.check_endpoints)
3. MARK: Mark non-live sessions as stale in the wiki

This pipeline ensures that orphaned sessions from previous daemon
crashes are cleaned up before the daemon accepts new commands. It
supports the crash_resilience invariant: daemon recovers from crash
within 30s by reading wiki state.

The pipeline never raises. All errors from sub-components are captured
in the returned PipelineResult.

Usage:
    from pathlib import Path
    from jules_daemon.startup.scan_probe_mark import run_pipeline

    result = await run_pipeline(Path("wiki"))
    for verdict in result.verdicts:
        print(f"Session {verdict.session_entry.run_id}: "
              f"health={verdict.health.value}, alive={verdict.alive}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from jules_daemon.monitor.process_state import (
    ProcessCheckResult,
    ProcessVerdict,
    check_pid,
)
from jules_daemon.monitor.session_liveness import (
    LivenessResult,
    SessionHealth,
)
from jules_daemon.ssh.endpoint_probe import (
    Endpoint,
    EndpointVerdict,
    ProbeSettings,
    check_endpoints,
)
from jules_daemon.wiki.session_scanner import (
    ScanResult,
    SessionEntry,
    scan_all_sessions,
)
from jules_daemon.wiki.stale_session_marker import (
    MarkOutcome,
    MarkResult,
    StaleMarkerInput,
    mark_stale_sessions,
)

__all__ = [
    "PipelineConfig",
    "PipelinePhase",
    "PipelineResult",
    "SessionVerdict",
    "run_pipeline",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_PROBE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the scan-probe-mark pipeline.

    Attributes:
        probe_timeout_seconds: TCP connect timeout for endpoint probes.
            Must be positive. Default: 5.0 seconds.
        capture_banner: Whether to read the SSH server banner during
            endpoint probing. Default: False (not needed for liveness).
    """

    probe_timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS
    capture_banner: bool = False

    def __post_init__(self) -> None:
        if self.probe_timeout_seconds <= 0:
            raise ValueError(
                f"probe_timeout_seconds must be positive, "
                f"got {self.probe_timeout_seconds}"
            )


# ---------------------------------------------------------------------------
# Pipeline phase enumeration
# ---------------------------------------------------------------------------


class PipelinePhase(Enum):
    """Named phases of the scan-probe-mark pipeline."""

    SCAN = "scan"
    PROBE = "probe"
    MARK = "mark"


# ---------------------------------------------------------------------------
# Session verdict model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionVerdict:
    """Immutable liveness verdict for a single session from the probe phase.

    Combines the results of PID checking and endpoint probing into a
    composite health classification.

    Attributes:
        session_entry: The SessionEntry that was evaluated.
        process_alive: True if daemon PID is alive, False if dead, None
            if no daemon PID was available to check.
        endpoint_reachable: True if SSH endpoint is reachable, False if
            unreachable, None if no SSH target or probe was skipped.
        health: Composite SessionHealth classification.
        alive: True when the session is considered operational (health
            is HEALTHY or DEGRADED).
    """

    session_entry: SessionEntry
    process_alive: bool | None
    endpoint_reachable: bool | None
    health: SessionHealth
    alive: bool


# ---------------------------------------------------------------------------
# Pipeline result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """Immutable result of the scan-probe-mark pipeline.

    Contains structured output from each phase for inspection and audit.

    Attributes:
        scan_result: Full scan result from the wiki session scanner.
        verdicts: Tuple of SessionVerdict objects, one per active session.
        mark_results: Tuple of MarkResult objects from stale marking.
        duration_seconds: Wall-clock time for the entire pipeline.
        timestamp: UTC datetime when the pipeline completed.
    """

    scan_result: ScanResult
    verdicts: tuple[SessionVerdict, ...]
    mark_results: tuple[MarkResult, ...]
    duration_seconds: float
    timestamp: datetime


# ---------------------------------------------------------------------------
# Internal: health classification
# ---------------------------------------------------------------------------


def _classify_health(
    process_alive: bool | None,
    endpoint_reachable: bool | None,
) -> tuple[SessionHealth, bool]:
    """Derive composite health and alive flag from probe results.

    Classification matrix:
        Process  | Endpoint     | Health          | Alive
        ---------|--------------|-----------------|------
        Dead     | (any/skip)   | PROCESS_DEAD    | False
        Alive    | Reachable    | HEALTHY         | True
        Alive    | Unreachable  | CONNECTION_LOST | False
        Alive    | None (skip)  | DEGRADED        | True
        None     | Reachable    | DEGRADED        | True
        None     | Unreachable  | CONNECTION_LOST | False
        None     | None         | UNKNOWN         | False

    Args:
        process_alive: Result of PID check (True/False/None).
        endpoint_reachable: Result of endpoint probe (True/False/None).

    Returns:
        Tuple of (SessionHealth, alive_flag).
    """
    # Dead process dominates all other signals
    if process_alive is False:
        return (SessionHealth.PROCESS_DEAD, False)

    # Alive process + endpoint reachable = healthy
    if process_alive is True and endpoint_reachable is True:
        return (SessionHealth.HEALTHY, True)

    # Alive process + endpoint unreachable = connection lost
    if process_alive is True and endpoint_reachable is False:
        return (SessionHealth.CONNECTION_LOST, False)

    # Alive process + no endpoint check = degraded (partial info)
    if process_alive is True and endpoint_reachable is None:
        return (SessionHealth.DEGRADED, True)

    # No PID to check + endpoint reachable = degraded
    if process_alive is None and endpoint_reachable is True:
        return (SessionHealth.DEGRADED, True)

    # No PID to check + endpoint unreachable = connection lost
    if process_alive is None and endpoint_reachable is False:
        return (SessionHealth.CONNECTION_LOST, False)

    # Both unknown
    return (SessionHealth.UNKNOWN, False)


# ---------------------------------------------------------------------------
# Internal: build LivenessResult for stale marker
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _build_liveness_result(
    session_entry: SessionEntry,
    health: SessionHealth,
    alive: bool,
) -> LivenessResult:
    """Build a LivenessResult from a session verdict for the stale marker.

    The stale marker expects LivenessResult objects. This builds one
    from our probe-phase results without requiring a full SSH session
    liveness check.

    Args:
        session_entry: The session being evaluated.
        health: Composite health classification.
        alive: Whether the session is considered operational.

    Returns:
        A LivenessResult suitable for the stale session marker.
    """
    return LivenessResult(
        session_id=session_entry.run_id,
        health=health,
        alive=alive,
        process_result=None,
        ssh_result=None,
        errors=(),
        latency_ms=0.0,
        timestamp=_now_utc(),
    )


# ---------------------------------------------------------------------------
# Internal: probe a single session
# ---------------------------------------------------------------------------


def _probe_process(
    entry: SessionEntry,
) -> tuple[bool | None, ProcessCheckResult | None]:
    """Check if the daemon PID for a session is alive.

    Args:
        entry: Session entry with optional daemon_pid.

    Returns:
        Tuple of (alive_flag, process_result). alive_flag is None when
        no daemon PID is available. process_result is None when PID
        check was not performed.

    Note:
        The broad ``except Exception`` is intentional. ``check_pid``
        can raise ``ValueError`` (invalid PID) or propagate unexpected
        ``OSError`` variants. The pipeline contract requires never-raises
        semantics, so all failures are logged and treated as unknown.
    """
    if entry.daemon_pid is None:
        return (None, None)

    try:
        result = check_pid(entry.daemon_pid)
        alive = result.verdict == ProcessVerdict.ALIVE
        return (alive, result)
    except Exception as exc:
        logger.warning(
            "PID check failed for session %s (PID %d): %s",
            entry.run_id,
            entry.daemon_pid,
            exc,
        )
        return (None, None)


async def _probe_endpoints(
    entries: tuple[SessionEntry, ...],
    settings: ProbeSettings,
) -> dict[str, EndpointVerdict]:
    """Probe SSH endpoints for all sessions that have SSH targets.

    Only probes sessions where the SSH host is available. Runs all
    probes concurrently.

    Args:
        entries: Active session entries to probe.
        settings: Probe timeout and banner settings.

    Returns:
        Mapping from run_id to EndpointVerdict for sessions that were
        probed. Sessions without SSH targets are omitted.
    """
    # Build endpoint list and track which entry each maps to
    endpoints: list[Endpoint] = []
    entry_run_ids: list[str] = []

    for entry in entries:
        if entry.ssh_host is not None:
            port = entry.ssh_port if entry.ssh_port is not None else 22
            endpoints.append(Endpoint(host=entry.ssh_host, port=port))
            entry_run_ids.append(entry.run_id)

    if not endpoints:
        return {}

    try:
        verdicts = await check_endpoints(endpoints, settings)
    except Exception as exc:
        logger.warning("Endpoint probing failed: %s", exc)
        return {}

    result: dict[str, EndpointVerdict] = {}
    for run_id, verdict in zip(entry_run_ids, verdicts, strict=True):
        result[run_id] = verdict

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_pipeline(
    wiki_root: Path,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the scan-probe-mark pipeline for daemon startup cleanup.

    Orchestrates the three-phase pipeline:

    1. **SCAN**: Discovers all session entries in the wiki. Identifies
       which sessions are active (RUNNING or PENDING_APPROVAL).

    2. **PROBE**: For each active session:
       - Checks if the daemon PID is alive (synchronous os.kill check).
         If the PID is dead, the endpoint probe is skipped.
       - For sessions with alive/unknown PIDs, checks if the SSH
         endpoint is TCP-reachable (async concurrent probes).
       - Classifies each session's composite health.

    3. **MARK**: Feeds the health verdicts into the stale session marker.
       Non-live sessions get a new stale wiki file written alongside
       the original. Live sessions are skipped.

    This function never raises. All errors from sub-components are
    captured in the returned PipelineResult.

    Args:
        wiki_root: Path to the wiki root directory.
        config: Optional pipeline configuration. Uses defaults if None.

    Returns:
        PipelineResult with structured output from each phase.
    """
    effective_config = config if config is not None else PipelineConfig()
    start_ns = time.monotonic_ns()

    # -- Phase 1: SCAN --
    logger.info("Startup pipeline: SCAN phase -- scanning wiki sessions")
    scan_result = scan_all_sessions(wiki_root)
    active_entries = scan_result.active_entries

    logger.info(
        "Startup pipeline: SCAN complete -- %d sessions found, %d active",
        scan_result.total_count,
        len(active_entries),
    )

    # Short-circuit if no active sessions
    if not active_entries:
        elapsed = (time.monotonic_ns() - start_ns) / 1_000_000_000
        return PipelineResult(
            scan_result=scan_result,
            verdicts=(),
            mark_results=(),
            duration_seconds=elapsed,
            timestamp=_now_utc(),
        )

    # -- Phase 2: PROBE --
    logger.info(
        "Startup pipeline: PROBE phase -- checking %d active sessions",
        len(active_entries),
    )

    probe_settings = ProbeSettings(
        timeout_seconds=effective_config.probe_timeout_seconds,
        capture_banner=effective_config.capture_banner,
    )

    # Step 2a: Check daemon PIDs (synchronous, fast)
    pid_results: dict[str, tuple[bool | None, ProcessCheckResult | None]] = {}
    for entry in active_entries:
        pid_results[entry.run_id] = _probe_process(entry)

    # Step 2b: Probe SSH endpoints for sessions with alive/unknown PIDs
    # Skip endpoint probing for dead-PID sessions (no point checking SSH)
    entries_to_probe = tuple(
        e for e in active_entries
        if pid_results.get(e.run_id, (None, None))[0] is not False
    )

    endpoint_verdicts: dict[str, EndpointVerdict] = {}
    if entries_to_probe:
        endpoint_verdicts = await _probe_endpoints(entries_to_probe, probe_settings)

    # Step 2c: Build composite verdicts
    verdicts: list[SessionVerdict] = []
    for entry in active_entries:
        process_alive, _ = pid_results.get(entry.run_id, (None, None))

        # Only look up endpoint verdict if we probed it
        ep_verdict = endpoint_verdicts.get(entry.run_id)
        endpoint_reachable: bool | None = None
        if ep_verdict is not None:
            endpoint_reachable = ep_verdict.reachable

        health, alive = _classify_health(process_alive, endpoint_reachable)

        verdicts.append(SessionVerdict(
            session_entry=entry,
            process_alive=process_alive,
            endpoint_reachable=endpoint_reachable,
            health=health,
            alive=alive,
        ))

        logger.info(
            "Startup probe: session %s -- process=%s endpoint=%s "
            "health=%s alive=%s",
            entry.run_id,
            process_alive,
            endpoint_reachable,
            health.value,
            alive,
        )

    # -- Phase 3: MARK --
    logger.info("Startup pipeline: MARK phase -- marking stale sessions")

    marker_inputs: list[StaleMarkerInput] = []
    for verdict in verdicts:
        liveness = _build_liveness_result(
            session_entry=verdict.session_entry,
            health=verdict.health,
            alive=verdict.alive,
        )
        marker_inputs.append(StaleMarkerInput(
            liveness_result=liveness,
            source_path=verdict.session_entry.source_path,
        ))

    mark_results = mark_stale_sessions(marker_inputs, wiki_root)

    stale_count = sum(
        1 for r in mark_results if r.outcome == MarkOutcome.MARKED_STALE
    )
    skipped_count = sum(
        1 for r in mark_results if r.outcome == MarkOutcome.SKIPPED_ALIVE
    )

    logger.info(
        "Startup pipeline: MARK complete -- %d marked stale, %d skipped alive",
        stale_count,
        skipped_count,
    )

    # Emit structured log records for each stale session (audit_completeness).
    # Deferred import to avoid circular dependency: stale_session_logger
    # imports SessionVerdict from this module.
    from jules_daemon.startup.stale_session_logger import (
        log_stale_sessions_from_verdicts,
    )

    log_stale_sessions_from_verdicts(
        verdicts=tuple(verdicts),
        mark_results=mark_results,
    )

    elapsed = (time.monotonic_ns() - start_ns) / 1_000_000_000
    logger.info(
        "Startup pipeline: complete in %.3fs", elapsed,
    )

    return PipelineResult(
        scan_result=scan_result,
        verdicts=tuple(verdicts),
        mark_results=mark_results,
        duration_seconds=elapsed,
        timestamp=_now_utc(),
    )
