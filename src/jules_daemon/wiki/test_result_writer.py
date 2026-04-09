"""Persist AssembledTestResult as Karpathy-style wiki entries.

Converts an AssembledTestResult into a wiki document with:
  - YAML frontmatter containing structured metadata for machine consumption
  - Markdown body with human-readable formatted results

Wiki file location: {wiki_root}/pages/daemon/results/result-{run_id}.md

Each test result is a standalone wiki file that preserves:
  - Full per-test records with outcome, timing, and error detail
  - Completeness ratio (executed vs expected)
  - Coverage gap analysis
  - Interruption metadata (if the run was cut short)
  - Run identifiers and timestamps for auditability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jules_daemon.wiki import frontmatter
from jules_daemon.wiki.assembled_result import (
    AssembledTestResult,
    CompletenessRatio,
    CoverageGap,
    DaemonDowntime,
    GapSeverity,
    InterruptionPoint,
    TestOutcome,
    TestRecord,
)
from jules_daemon.wiki.frontmatter import WikiDocument

__all__ = [
    "ResultWriteOutcome",
    "read_result",
    "result_to_document",
    "write_result",
]

logger = logging.getLogger(__name__)

_RESULTS_DIR = "pages/daemon/results"
_WIKI_TAGS = ["daemon", "test-result"]
_WIKI_TYPE = "test-result"


# ---------------------------------------------------------------------------
# Immutable result type for write operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultWriteOutcome:
    """Outcome of writing an AssembledTestResult to the wiki.

    Carries the path to the written file, the run_id, and
    the timestamp when the write occurred.
    """

    file_path: Path
    run_id: str
    written_at: datetime


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------


def _results_dir(wiki_root: Path) -> Path:
    """Resolve the results directory path."""
    return wiki_root / _RESULTS_DIR


def _result_file_path(wiki_root: Path, run_id: str) -> Path:
    """Resolve the path for a specific result entry."""
    return _results_dir(wiki_root) / f"result-{run_id}.md"


def _ensure_directory(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------


def _datetime_to_iso(dt: datetime | None) -> str | None:
    """Convert datetime to ISO 8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _iso_to_datetime(value: str | None) -> datetime | None:
    """Parse ISO 8601 string to datetime, or None."""
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Outcome classification
# ---------------------------------------------------------------------------


def _classify_outcome(result: AssembledTestResult) -> str:
    """Classify the overall result into a human-readable outcome string.

    Returns one of: 'passed', 'failed', 'interrupted', 'empty'.
    """
    if result.total_tests == 0:
        return "empty"
    if result.was_interrupted:
        return "interrupted"
    if result.has_failures:
        return "failed"
    return "passed"


# ---------------------------------------------------------------------------
# Serialization: AssembledTestResult -> frontmatter dict
# ---------------------------------------------------------------------------


def _record_to_dict(record: TestRecord) -> dict[str, Any]:
    """Serialize a TestRecord to a YAML-friendly dict."""
    return {
        "test_name": record.test_name,
        "outcome": record.outcome.value,
        "duration_seconds": record.duration_seconds,
        "error_message": record.error_message,
        "module": record.module,
        "line_number": record.line_number,
    }


def _completeness_to_dict(comp: CompletenessRatio) -> dict[str, Any]:
    """Serialize a CompletenessRatio to a YAML-friendly dict."""
    return {
        "executed": comp.executed,
        "expected": comp.expected,
        "ratio": round(comp.ratio, 4),
        "is_complete": comp.is_complete,
    }


def _coverage_gap_to_dict(gap: CoverageGap) -> dict[str, Any]:
    """Serialize a CoverageGap to a YAML-friendly dict."""
    return {
        "module": gap.module,
        "reason": gap.reason,
        "severity": gap.severity.value,
        "expected_tests": gap.expected_tests,
        "actual_tests": gap.actual_tests,
    }


def _interruption_to_dict(intr: InterruptionPoint) -> dict[str, Any]:
    """Serialize an InterruptionPoint to a YAML-friendly dict."""
    return {
        "interrupted": intr.interrupted,
        "at_test": intr.at_test,
        "at_timestamp": _datetime_to_iso(intr.at_timestamp),
        "reason": intr.reason,
        "exit_code": intr.exit_code,
    }


def _daemon_downtime_to_dict(dd: DaemonDowntime) -> dict[str, Any]:
    """Serialize a DaemonDowntime to a YAML-friendly dict.

    The daemon_was_down boolean is always present as a top-level flag
    so downstream consumers can quickly filter without parsing nested
    structures.
    """
    return {
        "daemon_was_down": dd.daemon_was_down,
        "down_started_at": _datetime_to_iso(dd.down_started_at),
        "down_ended_at": _datetime_to_iso(dd.down_ended_at),
        "estimated_down_seconds": dd.estimated_down_seconds,
        "recovery_method": dd.recovery_method,
    }


def _result_to_frontmatter(result: AssembledTestResult) -> dict[str, Any]:
    """Convert an AssembledTestResult to a YAML-serializable frontmatter dict.

    All nested objects are converted to plain dicts/lists. Enums are
    serialized as their string values. Datetimes are ISO 8601 strings.
    """
    return {
        "tags": list(_WIKI_TAGS),
        "type": _WIKI_TYPE,
        "run_id": result.run_id,
        "session_id": result.session_id,
        "host": result.host,
        "assembled_at": _datetime_to_iso(result.assembled_at),
        "outcome": _classify_outcome(result),
        "daemon_downtime": _daemon_downtime_to_dict(result.daemon_downtime),
        "pass_rate": round(result.pass_rate, 4),
        "total_duration_seconds": round(result.total_duration_seconds, 4),
        "counts": {
            "total": result.total_tests,
            "passed": result.passed_count,
            "failed": result.failed_count,
            "skipped": result.skipped_count,
            "errors": result.error_count,
        },
        "completeness": _completeness_to_dict(result.completeness),
        "interruption": _interruption_to_dict(result.interruption),
        "coverage_gaps": [
            _coverage_gap_to_dict(gap) for gap in result.coverage_gaps
        ],
        "records": [_record_to_dict(rec) for rec in result.records],
    }


# ---------------------------------------------------------------------------
# Deserialization: frontmatter dict -> AssembledTestResult
# ---------------------------------------------------------------------------


def _dict_to_record(data: dict[str, Any]) -> TestRecord:
    """Deserialize a TestRecord from a plain dict."""
    return TestRecord(
        test_name=data["test_name"],
        outcome=TestOutcome(data["outcome"]),
        duration_seconds=data.get("duration_seconds"),
        error_message=data.get("error_message", ""),
        module=data.get("module", ""),
        line_number=data.get("line_number"),
    )


def _dict_to_completeness(data: dict[str, Any]) -> CompletenessRatio:
    """Deserialize a CompletenessRatio from a plain dict."""
    return CompletenessRatio(
        executed=int(data.get("executed", 0)),
        expected=int(data.get("expected", 0)),
    )


def _dict_to_coverage_gap(data: dict[str, Any]) -> CoverageGap:
    """Deserialize a CoverageGap from a plain dict."""
    return CoverageGap(
        module=data["module"],
        reason=data["reason"],
        severity=GapSeverity(data.get("severity", "medium")),
        expected_tests=int(data.get("expected_tests", 0)),
        actual_tests=int(data.get("actual_tests", 0)),
    )


def _dict_to_interruption(data: dict[str, Any]) -> InterruptionPoint:
    """Deserialize an InterruptionPoint from a plain dict."""
    return InterruptionPoint(
        interrupted=data.get("interrupted", False),
        at_test=data.get("at_test", ""),
        at_timestamp=_iso_to_datetime(data.get("at_timestamp")),
        reason=data.get("reason", ""),
        exit_code=data.get("exit_code"),
    )


def _dict_to_daemon_downtime(data: dict[str, Any]) -> DaemonDowntime:
    """Deserialize a DaemonDowntime from a plain dict.

    Handles missing keys gracefully for forward compatibility with
    wiki entries written before daemon_downtime was added.
    """
    daemon_was_down = data.get("daemon_was_down", False)
    estimated_down_seconds_raw = data.get("estimated_down_seconds")
    estimated_down_seconds = (
        float(estimated_down_seconds_raw)
        if estimated_down_seconds_raw is not None
        else None
    )
    return DaemonDowntime(
        daemon_was_down=daemon_was_down,
        down_started_at=_iso_to_datetime(data.get("down_started_at")),
        down_ended_at=_iso_to_datetime(data.get("down_ended_at")),
        estimated_down_seconds=estimated_down_seconds,
        recovery_method=data.get("recovery_method", ""),
    )


def _frontmatter_to_result(fm: dict[str, Any]) -> AssembledTestResult:
    """Reconstruct an AssembledTestResult from parsed frontmatter.

    Handles backward compatibility: wiki entries written before
    daemon_downtime was added will get a default DaemonDowntime().
    """
    records_raw = fm.get("records", [])
    records = tuple(_dict_to_record(r) for r in records_raw)

    gaps_raw = fm.get("coverage_gaps", [])
    coverage_gaps = tuple(_dict_to_coverage_gap(g) for g in gaps_raw)

    completeness = _dict_to_completeness(fm.get("completeness", {}))
    interruption = _dict_to_interruption(fm.get("interruption", {}))
    daemon_downtime = _dict_to_daemon_downtime(fm.get("daemon_downtime", {}))

    assembled_at_str = fm.get("assembled_at")
    assembled_at = (
        _iso_to_datetime(assembled_at_str) or _now_utc()
        if assembled_at_str
        else _now_utc()
    )

    return AssembledTestResult(
        run_id=fm.get("run_id", ""),
        session_id=fm.get("session_id", ""),
        host=fm.get("host", ""),
        records=records,
        completeness=completeness,
        coverage_gaps=coverage_gaps,
        interruption=interruption,
        daemon_downtime=daemon_downtime,
        assembled_at=assembled_at,
    )


# ---------------------------------------------------------------------------
# Markdown body generation
# ---------------------------------------------------------------------------


def _format_duration(seconds: float | None) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _build_body(result: AssembledTestResult) -> str:
    """Generate the human-readable markdown body for a test result wiki entry.

    Sections are conditionally included based on the result content:
    - Summary (always)
    - Failed Tests (only when there are failures)
    - Coverage Gaps (only when gaps exist)
    - Interruption (only when the run was interrupted)
    - Completeness (always, shows ratio)
    - Metadata (always, run identifiers and timestamps)
    """
    outcome = _classify_outcome(result)
    lines: list[str] = [
        f"# Test Results: {result.host}",
        "",
        f"*Test execution results -- outcome: {outcome}*",
        "",
    ]

    # -- Summary section --
    lines.extend([
        "## Summary",
        "",
    ])

    if result.total_tests == 0:
        lines.extend([
            "No test records were produced for this run.",
            "",
        ])
    else:
        lines.extend([
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {result.total_tests} |",
            f"| Passed | {result.passed_count} |",
            f"| Failed | {result.failed_count} |",
            f"| Errors | {result.error_count} |",
            f"| Skipped | {result.skipped_count} |",
            f"| Pass Rate | {result.pass_rate:.1%} |",
            f"| Total Duration | {_format_duration(result.total_duration_seconds)} |",
            "",
        ])

    # -- Failed Tests section --
    failed = result.failed_records
    if failed:
        lines.extend([
            "## Failed Tests",
            "",
        ])
        for record in failed:
            lines.extend([
                f"### {record.test_name}",
                "",
            ])
            if record.module:
                lines.append(f"- **Module:** {record.module}")
            lines.append(f"- **Outcome:** {record.outcome.value}")
            if record.duration_seconds is not None:
                lines.append(
                    f"- **Duration:** {_format_duration(record.duration_seconds)}"
                )
            if record.line_number is not None:
                lines.append(f"- **Line:** {record.line_number}")
            if record.error_message:
                lines.extend([
                    "",
                    "```",
                    record.error_message,
                    "```",
                ])
            lines.append("")

    # -- Coverage Gaps section --
    if result.coverage_gaps:
        lines.extend([
            "## Coverage Gaps",
            "",
        ])
        for gap in result.coverage_gaps:
            lines.extend([
                f"### {gap.module}",
                "",
                f"- **Severity:** {gap.severity.value}",
                f"- **Reason:** {gap.reason}",
            ])
            if gap.expected_tests > 0 or gap.actual_tests > 0:
                lines.append(
                    f"- **Tests:** {gap.actual_tests}/{gap.expected_tests} executed"
                )
            lines.append("")

    # -- Interruption section --
    if result.was_interrupted:
        intr = result.interruption
        lines.extend([
            "## Interruption",
            "",
            f"- **Reason:** {intr.reason}",
        ])
        if intr.at_test:
            lines.append(f"- **At Test:** {intr.at_test}")
        if intr.at_timestamp:
            lines.append(
                f"- **At:** {_datetime_to_iso(intr.at_timestamp)}"
            )
        if intr.exit_code is not None:
            lines.append(f"- **Exit Code:** {intr.exit_code}")
        lines.append("")

    # -- Daemon Downtime section --
    if result.daemon_was_down:
        dd = result.daemon_downtime
        lines.extend([
            "## Daemon Downtime",
            "",
            "This result is partial because the daemon was down during execution.",
            "Downstream consumers should treat this as partial-due-to-crash,",
            "not partial-due-to-timeout.",
            "",
        ])
        if dd.estimated_down_seconds is not None:
            lines.append(
                f"- **Estimated Downtime:** {dd.estimated_down_seconds:.1f}s"
            )
        if dd.recovery_method:
            lines.append(f"- **Recovery Method:** {dd.recovery_method}")
        if dd.down_started_at:
            lines.append(
                f"- **Down Started:** {_datetime_to_iso(dd.down_started_at)}"
            )
        if dd.down_ended_at:
            lines.append(
                f"- **Down Ended:** {_datetime_to_iso(dd.down_ended_at)}"
            )
        lines.append("")

    # -- Completeness section --
    comp = result.completeness
    lines.extend([
        "## Completeness",
        "",
        f"- **Executed:** {comp.executed}",
        f"- **Expected:** {comp.expected}",
        f"- **Ratio:** {comp.ratio:.1%}",
        f"- **Complete:** {'yes' if comp.is_complete else 'no'}",
        "",
    ])

    # -- Metadata section --
    lines.extend([
        "## Metadata",
        "",
        f"- **Run ID:** {result.run_id}",
        f"- **Session ID:** {result.session_id}",
        f"- **Host:** {result.host}",
        f"- **Assembled At:** {_datetime_to_iso(result.assembled_at)}",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def result_to_document(result: AssembledTestResult) -> WikiDocument:
    """Convert an AssembledTestResult to a WikiDocument.

    Produces a Karpathy-style wiki document with YAML frontmatter
    containing structured metadata and a markdown body with
    human-readable formatted results.

    This is the pure-conversion step. Use ``write_result`` to persist
    the document to disk.

    Args:
        result: The assembled test result to convert.

    Returns:
        WikiDocument with frontmatter and body.
    """
    return WikiDocument(
        frontmatter=_result_to_frontmatter(result),
        body=_build_body(result),
    )


def write_result(
    wiki_root: Path,
    result: AssembledTestResult,
) -> ResultWriteOutcome:
    """Write an AssembledTestResult as a wiki entry.

    Creates the file and parent directories if needed. Uses atomic
    write (tmp file + rename) to prevent partial files. Overwrites
    any existing file for the same run_id.

    Wiki file location: {wiki_root}/pages/daemon/results/result-{run_id}.md

    Args:
        wiki_root: Path to the wiki root directory.
        result: The assembled test result to persist.

    Returns:
        ResultWriteOutcome with the file path and metadata.
    """
    file_path = _result_file_path(wiki_root, result.run_id)
    _ensure_directory(file_path)

    doc = result_to_document(result)
    content = frontmatter.serialize(doc)

    # Atomic write: write to temp file then rename (Path.replace is atomic on POSIX)
    tmp_path = file_path.with_suffix(".md.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(file_path)

    written_at = _now_utc()

    logger.info(
        "Wrote test result for run %s (%s) to %s",
        result.run_id,
        _classify_outcome(result),
        file_path,
    )

    return ResultWriteOutcome(
        file_path=file_path,
        run_id=result.run_id,
        written_at=written_at,
    )


def read_result(file_path: Path) -> AssembledTestResult | None:
    """Read an AssembledTestResult from a wiki entry.

    Args:
        file_path: Path to the result wiki file.

    Returns:
        The deserialized AssembledTestResult, or None if the file
        does not exist.
    """
    if not file_path.exists():
        return None

    raw = file_path.read_text(encoding="utf-8")
    doc = frontmatter.parse(raw)
    return _frontmatter_to_result(doc.frontmatter)
