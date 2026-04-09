"""Daemon startup lifecycle management.

Manages the daemon's transition from startup to ready state, running
required initialization hooks (including the scan-probe-mark pipeline)
before the daemon can accept commands.

Lifecycle phases:
    STARTING  -> initial state, daemon process just launched
    SCANNING  -> scan-probe-mark pipeline is running
    READY     -> daemon is accepting commands

The lifecycle guarantees:
- The scan-probe-mark pipeline runs BEFORE the daemon transitions to READY
- Pipeline errors are captured but do NOT prevent the daemon from becoming
  READY (warn-and-continue, not hard-block)
- A startup event wiki page is written for audit completeness
- The startup result captures timing and error details

Usage:
    from pathlib import Path
    from jules_daemon.startup.lifecycle import run_startup

    result = await run_startup(Path("wiki"))
    if result.is_ready:
        # Daemon can start accepting commands
        ...
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from jules_daemon.startup.scan_probe_mark import (
    PipelineConfig,
    PipelineResult,
    run_pipeline,
)
from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.frontmatter import WikiDocument

__all__ = [
    "DaemonPhase",
    "StartupHookConfig",
    "StartupResult",
    "run_startup",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0
_STARTUP_EVENT_FILENAME = "startup-event.md"
_DAEMON_DIR = "pages/daemon"
_WIKI_TAGS: tuple[str, ...] = ("daemon", "startup", "audit")
_WIKI_TYPE = "daemon-startup-event"


# ---------------------------------------------------------------------------
# Phase enumeration
# ---------------------------------------------------------------------------


class DaemonPhase(Enum):
    """Daemon lifecycle phase during startup.

    Values:
        STARTING: Initial state. Daemon process just launched.
        SCANNING: Scan-probe-mark pipeline is running.
        READY: Pipeline complete. Daemon is accepting commands.
    """

    STARTING = "starting"
    SCANNING = "scanning"
    READY = "ready"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartupHookConfig:
    """Immutable configuration for the startup lifecycle.

    Attributes:
        run_scan_probe_mark: Whether to run the scan-probe-mark pipeline
            during startup. Default: True. Set to False to skip.
        pipeline_config: Optional custom configuration for the
            scan-probe-mark pipeline. When None, uses pipeline defaults.
    """

    run_scan_probe_mark: bool = True
    pipeline_config: PipelineConfig | None = None


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartupResult:
    """Immutable result of the daemon startup lifecycle.

    Attributes:
        final_phase: The phase the daemon reached at the end of startup.
            Should be READY on success.
        pipeline_result: Result of the scan-probe-mark pipeline, or None
            if the pipeline was not run or was skipped.
        duration_seconds: Wall-clock time for the entire startup.
        error: Human-readable error description if something went wrong
            during startup. None on clean startup. Note: errors do NOT
            prevent reaching READY -- they are informational.
        timestamp: UTC datetime when startup completed.
    """

    final_phase: DaemonPhase
    pipeline_result: PipelineResult | None
    duration_seconds: float
    error: str | None
    timestamp: datetime

    @property
    def is_ready(self) -> bool:
        """True if the daemon successfully reached the READY phase."""
        return self.final_phase == DaemonPhase.READY


# ---------------------------------------------------------------------------
# Internal: wiki audit event
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _write_startup_event(
    wiki_root: Path,
    result: StartupResult,
) -> bool:
    """Write a startup event wiki page for audit completeness.

    Creates or replaces the startup-event.md file in the daemon wiki
    directory. Each startup produces a complete snapshot.

    Args:
        wiki_root: Path to the wiki root directory.
        result: The startup result to record.

    Returns:
        True if the write succeeded, False on any error.
    """
    file_path = wiki_root / _DAEMON_DIR / _STARTUP_EVENT_FILENAME

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)

        now = _now_utc()
        fm: dict[str, object] = {
            "tags": list(_WIKI_TAGS),
            "type": _WIKI_TYPE,
            "created": now.isoformat(),
            "updated": now.isoformat(),
            "final_phase": result.final_phase.value,
            "duration_seconds": round(result.duration_seconds, 3),
            "is_ready": result.is_ready,
            "error": result.error,
            "pipeline_ran": result.pipeline_result is not None,
        }

        # Add pipeline summary if available
        if result.pipeline_result is not None:
            pr = result.pipeline_result
            fm["pipeline_summary"] = {
                "scan_outcome": pr.scan_result.outcome.value,
                "sessions_scanned": pr.scan_result.scanned_count,
                "active_sessions": pr.scan_result.active_count,
                "verdicts_count": len(pr.verdicts),
                "marks_count": len(pr.mark_results),
                "pipeline_duration_seconds": round(pr.duration_seconds, 3),
            }

        body = _build_startup_event_body(result)
        doc = WikiDocument(frontmatter=fm, body=body)
        content = frontmatter.serialize(doc)

        # Atomic write
        tmp_path = file_path.with_suffix(".md.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, file_path)

        logger.info("Startup event written to %s", file_path)
        return True

    except Exception as exc:
        logger.warning("Failed to write startup event: %s", exc)
        return False


def _build_startup_event_body(result: StartupResult) -> str:
    """Generate human-readable markdown body for the startup event."""
    status_label = "Ready" if result.is_ready else "Failed"

    lines = [
        "# Daemon Startup Event",
        "",
        f"*Startup lifecycle -- result: {status_label}*",
        "",
        "## Summary",
        "",
        f"- **Final Phase:** {result.final_phase.value}",
        f"- **Duration:** {result.duration_seconds:.3f}s",
        f"- **Ready:** {'yes' if result.is_ready else 'no'}",
        f"- **Pipeline Ran:** {'yes' if result.pipeline_result is not None else 'no'}",
        "",
    ]

    if result.error:
        lines.extend([
            "## Error",
            "",
            "```",
            result.error,
            "```",
            "",
        ])

    if result.pipeline_result is not None:
        pr = result.pipeline_result
        lines.extend([
            "## Scan-Probe-Mark Pipeline",
            "",
            f"- **Scan Outcome:** {pr.scan_result.outcome.value}",
            f"- **Sessions Scanned:** {pr.scan_result.scanned_count}",
            f"- **Active Sessions:** {pr.scan_result.active_count}",
            f"- **Verdicts:** {len(pr.verdicts)}",
            f"- **Stale Marks:** {len(pr.mark_results)}",
            f"- **Pipeline Duration:** {pr.duration_seconds:.3f}s",
            "",
        ])

        if pr.verdicts:
            lines.extend([
                "### Session Verdicts",
                "",
                "| Run ID | Health | Alive | Process | Endpoint |",
                "|--------|--------|-------|---------|----------|",
            ])
            for v in pr.verdicts:
                lines.append(
                    f"| {v.session_entry.run_id[:12]}... | "
                    f"{v.health.value} | "
                    f"{'yes' if v.alive else 'no'} | "
                    f"{v.process_alive} | "
                    f"{v.endpoint_reachable} |"
                )
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_startup(
    wiki_root: Path,
    config: StartupHookConfig | None = None,
    *,
    startup_timeout_seconds: float | None = None,
) -> StartupResult:
    """Run the daemon startup lifecycle with initialization hooks.

    Transitions the daemon through its startup phases:

    1. **STARTING**: Entry point. Validates configuration.
    2. **SCANNING**: Runs the scan-probe-mark pipeline to clean up
       orphaned sessions from previous daemon crashes.
    3. **READY**: Pipeline complete. Daemon can accept commands.

    The pipeline is run within a timeout to prevent startup from
    blocking indefinitely. If the pipeline times out or raises an
    exception, the daemon still transitions to READY (warn-and-continue)
    because blocking startup would be worse than having stale sessions.

    This function never raises. All errors are captured in the returned
    StartupResult.

    Args:
        wiki_root: Path to the wiki root directory.
        config: Optional startup configuration. Uses defaults if None.
        startup_timeout_seconds: Maximum seconds to wait for the pipeline.
            Default: 30.0 seconds. Override for testing.

    Returns:
        StartupResult with the final phase, pipeline output, timing,
        and any error details.
    """
    effective_config = config if config is not None else StartupHookConfig()
    timeout = (
        startup_timeout_seconds
        if startup_timeout_seconds is not None
        else _DEFAULT_STARTUP_TIMEOUT_SECONDS
    )
    start_ns = time.monotonic_ns()
    error: str | None = None
    pipeline_result: PipelineResult | None = None

    # -- Phase: STARTING --
    logger.info("Daemon startup: phase=STARTING")

    # -- Phase: SCANNING --
    if effective_config.run_scan_probe_mark:
        logger.info("Daemon startup: phase=SCANNING (scan-probe-mark pipeline)")

        try:
            pipeline_result = await asyncio.wait_for(
                run_pipeline(wiki_root, config=effective_config.pipeline_config),
                timeout=timeout,
            )
            logger.info(
                "Daemon startup: scan-probe-mark pipeline completed in %.3fs",
                pipeline_result.duration_seconds,
            )
        except asyncio.TimeoutError:
            error = (
                f"Scan-probe-mark pipeline timed out after {timeout:.1f}s "
                f"-- continuing to READY"
            )
            logger.warning("Daemon startup: %s", error)
        except Exception as exc:
            error = (
                f"Scan-probe-mark pipeline failed: {exc} "
                f"-- continuing to READY"
            )
            logger.warning("Daemon startup: %s", error, exc_info=True)
    else:
        logger.info(
            "Daemon startup: scan-probe-mark pipeline skipped (disabled)"
        )

    # -- Phase: READY --
    elapsed = (time.monotonic_ns() - start_ns) / 1_000_000_000
    logger.info("Daemon startup: phase=READY (%.3fs)", elapsed)

    result = StartupResult(
        final_phase=DaemonPhase.READY,
        pipeline_result=pipeline_result,
        duration_seconds=elapsed,
        error=error,
        timestamp=_now_utc(),
    )

    # Write audit event (best-effort, does not affect result)
    _write_startup_event(wiki_root, result)

    return result
