"""Anomaly pattern configuration schema and base detector interface.

Defines the data models and protocol for real-time anomaly detection on
SSH test output. The daemon performs background regex pattern matching on
output lines as they arrive, detecting three categories of anomalies:

1. **Error keywords**: Regex-based string matching against individual
   output lines. Detects known error patterns (OOM, segfault, assertion
   failures, stack traces) without invoking the LLM.

2. **Failure-rate thresholds**: Count-based detection over a sliding
   time window. Triggers when the number of matching events within the
   window exceeds a configured threshold.

3. **Stall timeouts**: Elapsed-time-based detection. Triggers when no
   new output has been received for longer than the configured timeout,
   indicating a hung or stalled test process.

All data structures are immutable (frozen dataclasses). Configuration
instances are constructed once at startup (or when patterns are reloaded
from wiki files) and shared read-only across async tasks.

The ``AnomalyDetector`` protocol defines the common interface that all
concrete detector implementations must satisfy. It uses structural
subtyping (``runtime_checkable``) so detectors do not need to inherit
from a base class.

Usage::

    from jules_daemon.monitor.anomaly_models import (
        AnomalyPatternConfig,
        AnomalySeverity,
        ErrorKeywordPattern,
        FailureRatePattern,
        StallTimeoutPattern,
    )

    config = AnomalyPatternConfig(
        error_keywords=(
            ErrorKeywordPattern(name="oom", regex=r"Out of memory|OOM"),
            ErrorKeywordPattern(name="segfault", regex=r"SIGSEGV|Segmentation fault"),
        ),
        failure_rates=(
            FailureRatePattern(name="high_fail", threshold_count=5, window_seconds=60.0),
        ),
        stall_timeouts=(
            StallTimeoutPattern(name="output_stall", timeout_seconds=300.0),
        ),
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "AnomalyDetector",
    "AnomalyPatternConfig",
    "AnomalyReport",
    "AnomalySeverity",
    "ErrorKeywordPattern",
    "FailureRatePattern",
    "PatternType",
    "StallTimeoutPattern",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PatternType(Enum):
    """Classification of anomaly pattern types.

    Values:
        ERROR_KEYWORD: Regex-based string matching against output lines.
        FAILURE_RATE: Count-based detection over a sliding time window.
        STALL_TIMEOUT: Elapsed-time-based detection for hung processes.
    """

    ERROR_KEYWORD = "error_keyword"
    FAILURE_RATE = "failure_rate"
    STALL_TIMEOUT = "stall_timeout"


class AnomalySeverity(Enum):
    """Severity classification for detected anomalies.

    Each severity level has a numeric_level property for ordering and
    threshold comparisons. Higher numeric values indicate more severe
    anomalies.

    Values:
        INFO: Informational anomaly. Logged but does not trigger alerts.
        WARNING: Noteworthy anomaly. May trigger a notification to the user.
        CRITICAL: Severe anomaly. Triggers immediate notification and may
            warrant automatic intervention (e.g., stopping a runaway test).
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def numeric_level(self) -> int:
        """Return a numeric level for ordering comparisons.

        Returns:
            0 for INFO, 1 for WARNING, 2 for CRITICAL.
        """
        return _SEVERITY_LEVELS[self]


# Pre-computed mapping from severity to numeric level for O(1) lookup.
_SEVERITY_LEVELS: dict[AnomalySeverity, int] = {
    AnomalySeverity.INFO: 0,
    AnomalySeverity.WARNING: 1,
    AnomalySeverity.CRITICAL: 2,
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_non_empty(value: str, field_name: str) -> str:
    """Validate that a string field is non-empty after stripping whitespace.

    Args:
        value: The string value to validate.
        field_name: Name of the field for the error message.

    Returns:
        The stripped string value.

    Raises:
        ValueError: If the value is empty or whitespace-only.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


def _require_positive(value: float, field_name: str) -> None:
    """Validate that a numeric field is strictly positive.

    Args:
        value: The numeric value to validate.
        field_name: Name of the field for the error message.

    Raises:
        ValueError: If the value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


def _require_positive_int(value: int, field_name: str) -> None:
    """Validate that an integer field is strictly positive.

    Args:
        value: The integer value to validate.
        field_name: Name of the field for the error message.

    Raises:
        ValueError: If the value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}")


# ---------------------------------------------------------------------------
# Pattern configuration schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorKeywordPattern:
    """Immutable configuration for a regex-based error keyword pattern.

    Matches individual output lines against a compiled regular expression.
    Used to detect known error signatures (OOM, segfault, assertion
    failures, etc.) in real-time as SSH output arrives.

    Attributes:
        name: Unique identifier for this pattern. Used in reports and
            for pattern lookup. Must not be empty.
        regex: Regular expression to match against output lines. Must be
            a valid Python regex. Compiled at construction time for
            performance.
        severity: Severity level assigned to matches from this pattern.
            Defaults to WARNING.
        pattern_type: Always PatternType.ERROR_KEYWORD. Set automatically.
        case_sensitive: Whether the regex match is case-sensitive.
            Defaults to False (case-insensitive matching).
        description: Human-readable description of what this pattern
            detects. Optional, defaults to empty string.
    """

    name: str
    regex: str
    severity: AnomalySeverity = AnomalySeverity.WARNING
    pattern_type: PatternType = field(
        default=PatternType.ERROR_KEYWORD, init=False
    )
    case_sensitive: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        stripped_name = _require_non_empty(self.name, "name")
        if stripped_name != self.name:
            object.__setattr__(self, "name", stripped_name)

        stripped_regex = self.regex.strip()
        if not stripped_regex:
            raise ValueError("regex must not be empty")
        if stripped_regex != self.regex:
            object.__setattr__(self, "regex", stripped_regex)

        # Validate and pre-compile the regex
        try:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            compiled = re.compile(self.regex, flags)
        except re.error as exc:
            raise ValueError(
                f"regex is not a valid regular expression: {exc}"
            ) from exc

        # Store the compiled pattern as a private attribute
        object.__setattr__(self, "_compiled_regex", compiled)

    @property
    def compiled_regex(self) -> re.Pattern[str]:
        """Return the pre-compiled regex pattern.

        The pattern is compiled once at construction time with the
        appropriate flags (IGNORECASE when case_sensitive is False).

        Returns:
            Compiled re.Pattern instance ready for matching.
        """
        return self._compiled_regex  # type: ignore[attr-defined]


@dataclass(frozen=True)
class FailureRatePattern:
    """Immutable configuration for a failure-rate threshold pattern.

    Triggers when the number of matching failure events within a sliding
    time window exceeds the configured threshold. The actual event
    counting and window management are handled by the concrete detector
    implementation, not by this configuration object.

    Attributes:
        name: Unique identifier for this pattern. Must not be empty.
        threshold_count: Number of failures within the window that
            triggers the anomaly. Must be positive.
        window_seconds: Size of the sliding time window in seconds.
            Must be positive.
        severity: Severity level assigned when the threshold is exceeded.
            Defaults to WARNING.
        pattern_type: Always PatternType.FAILURE_RATE. Set automatically.
        description: Human-readable description. Optional.
    """

    name: str
    threshold_count: int
    window_seconds: float
    severity: AnomalySeverity = AnomalySeverity.WARNING
    pattern_type: PatternType = field(
        default=PatternType.FAILURE_RATE, init=False
    )
    description: str = ""

    def __post_init__(self) -> None:
        stripped_name = _require_non_empty(self.name, "name")
        if stripped_name != self.name:
            object.__setattr__(self, "name", stripped_name)

        _require_positive_int(self.threshold_count, "threshold_count")
        _require_positive(self.window_seconds, "window_seconds")


@dataclass(frozen=True)
class StallTimeoutPattern:
    """Immutable configuration for a stall timeout pattern.

    Triggers when no new output has been received from the SSH session
    for longer than the configured timeout. Indicates a potentially
    hung or stalled test process.

    Attributes:
        name: Unique identifier for this pattern. Must not be empty.
        timeout_seconds: Maximum seconds of silence before triggering.
            Must be positive.
        severity: Severity level assigned when a stall is detected.
            Defaults to WARNING.
        pattern_type: Always PatternType.STALL_TIMEOUT. Set automatically.
        description: Human-readable description. Optional.
    """

    name: str
    timeout_seconds: float
    severity: AnomalySeverity = AnomalySeverity.WARNING
    pattern_type: PatternType = field(
        default=PatternType.STALL_TIMEOUT, init=False
    )
    description: str = ""

    def __post_init__(self) -> None:
        stripped_name = _require_non_empty(self.name, "name")
        if stripped_name != self.name:
            object.__setattr__(self, "name", stripped_name)

        _require_positive(self.timeout_seconds, "timeout_seconds")


# ---------------------------------------------------------------------------
# Union type for any pattern
# ---------------------------------------------------------------------------

AnyPattern = ErrorKeywordPattern | FailureRatePattern | StallTimeoutPattern
"""Type alias for any anomaly pattern configuration."""


# ---------------------------------------------------------------------------
# Anomaly report (detection result)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyReport:
    """Immutable record of a detected anomaly.

    Produced by ``AnomalyDetector.report()`` when a pattern matches
    output. Contains all metadata needed to notify the user and log
    the detection for audit purposes.

    Attributes:
        pattern_name: Name of the pattern that triggered the detection.
        pattern_type: Type classification of the triggering pattern.
        severity: Severity level of the detected anomaly.
        message: Human-readable description of what was detected.
        detected_at: UTC timestamp when the anomaly was detected.
        session_id: Identifier of the SSH session being monitored.
        context: Optional dictionary with additional context about the
            detection (e.g., line number, matched text, failure count).
            None when no additional context is available.
    """

    pattern_name: str
    pattern_type: PatternType
    severity: AnomalySeverity
    message: str
    detected_at: datetime
    session_id: str
    context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        stripped_name = _require_non_empty(self.pattern_name, "pattern_name")
        if stripped_name != self.pattern_name:
            object.__setattr__(self, "pattern_name", stripped_name)

        stripped_msg = _require_non_empty(self.message, "message")
        if stripped_msg != self.message:
            object.__setattr__(self, "message", stripped_msg)

        stripped_sid = _require_non_empty(self.session_id, "session_id")
        if stripped_sid != self.session_id:
            object.__setattr__(self, "session_id", stripped_sid)


# ---------------------------------------------------------------------------
# Composite pattern configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyPatternConfig:
    """Immutable container holding all anomaly pattern configurations.

    Groups patterns by type for efficient dispatch while providing
    convenience methods for cross-type operations (lookup by name,
    iteration over all patterns).

    Attributes:
        error_keywords: Tuple of ErrorKeywordPattern instances.
        failure_rates: Tuple of FailureRatePattern instances.
        stall_timeouts: Tuple of StallTimeoutPattern instances.
    """

    error_keywords: tuple[ErrorKeywordPattern, ...] = ()
    failure_rates: tuple[FailureRatePattern, ...] = ()
    stall_timeouts: tuple[StallTimeoutPattern, ...] = ()

    def __post_init__(self) -> None:
        # Validate uniqueness of pattern names across all types
        all_names: list[str] = []
        for pattern in (
            *self.error_keywords,
            *self.failure_rates,
            *self.stall_timeouts,
        ):
            if pattern.name in all_names:
                raise ValueError(
                    f"Duplicate pattern name {pattern.name!r}. "
                    f"All pattern names must be unique within an "
                    f"AnomalyPatternConfig."
                )
            all_names.append(pattern.name)

        # Build the name-to-pattern index for O(1) lookup
        index: dict[str, AnyPattern] = {}
        for pattern in (
            *self.error_keywords,
            *self.failure_rates,
            *self.stall_timeouts,
        ):
            index[pattern.name] = pattern
        object.__setattr__(self, "_name_index", index)

    @property
    def total_patterns(self) -> int:
        """Total number of configured patterns across all types."""
        return (
            len(self.error_keywords)
            + len(self.failure_rates)
            + len(self.stall_timeouts)
        )

    @property
    def all_patterns(self) -> tuple[AnyPattern, ...]:
        """Return all patterns as a single tuple.

        Ordering: error_keywords first, then failure_rates, then
        stall_timeouts. Stable ordering within each group.
        """
        return (
            *self.error_keywords,
            *self.failure_rates,
            *self.stall_timeouts,
        )

    def get_pattern(self, name: str) -> AnyPattern | None:
        """Look up a pattern by name.

        Args:
            name: The unique pattern name to look up.

        Returns:
            The matching pattern, or None if not found.
        """
        index: dict[str, AnyPattern] = self._name_index  # type: ignore[attr-defined]
        return index.get(name)


# ---------------------------------------------------------------------------
# AnomalyDetector protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AnomalyDetector(Protocol):
    """Protocol defining the interface for anomaly detectors.

    Concrete detector implementations match output against a specific
    pattern and produce anomaly reports when a match is found. Detectors
    are stateless with respect to the protocol -- any state tracking
    (e.g., failure counts for rate patterns, last-output timestamps for
    stall patterns) is internal to the implementation.

    This protocol uses structural subtyping (``runtime_checkable``) so
    implementations do not need to explicitly inherit from this class.
    Any object with the correct method signatures satisfies the protocol.

    Required properties:
        pattern_name: Unique identifier of the pattern this detector
            evaluates.
        pattern_type: The PatternType classification.

    Required methods:
        match(output_line): Test whether an output line triggers this
            detector. Returns True if the line matches, False otherwise.
        report(output_line, *, session_id, detected_at): Produce an
            AnomalyReport for the matched line. Should only be called
            when match() returned True.
    """

    @property
    def pattern_name(self) -> str:
        """Unique identifier of the pattern this detector evaluates."""
        ...

    @property
    def pattern_type(self) -> PatternType:
        """The PatternType classification for this detector."""
        ...

    def match(self, output_line: str) -> bool:
        """Test whether an output line triggers this detector.

        This method should be fast (O(1) or O(n) in line length) since
        it is called for every output line received from SSH.

        Args:
            output_line: A single line of SSH output to test.

        Returns:
            True if the line matches this detector's pattern.
        """
        ...

    def report(
        self,
        output_line: str,
        *,
        session_id: str,
        detected_at: datetime,
    ) -> AnomalyReport:
        """Produce an anomaly report for a matched output line.

        Should only be called after ``match()`` returned True for the
        same output line. The report includes all metadata needed for
        notification and audit logging.

        Args:
            output_line: The matched output line.
            session_id: Identifier of the SSH session being monitored.
            detected_at: UTC timestamp of the detection.

        Returns:
            Immutable AnomalyReport with detection metadata.
        """
        ...
