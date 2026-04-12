"""Error keyword regex matcher anomaly detector.

Concrete implementation of the ``AnomalyDetector`` protocol that matches
SSH output lines against a configurable set of ``ErrorKeywordPattern``
regex patterns. Each pattern has its own severity and compiled regex,
enabling fast per-line scanning without invoking the LLM.

Designed for real-time background matching: the daemon feeds every
output line through ``match()`` and, on a positive result, calls
``report()`` to produce an immutable ``AnomalyReport`` for notification
and audit logging.

Key properties:

- **Configurable patterns**: Accepts one or more ``ErrorKeywordPattern``
  instances at construction time. Patterns are stored as an immutable
  tuple and never modified.

- **First-match semantics**: ``match()`` returns True on the first
  pattern that matches. ``report()`` uses the first matching pattern's
  metadata (name, severity) for the report. Use ``match_all()`` when
  you need to know every pattern that matched.

- **Protocol conformance**: Satisfies the ``AnomalyDetector`` protocol
  via structural subtyping. No inheritance required.

- **Thread-safe reads**: All state is immutable after construction.
  Multiple async tasks can call ``match()`` concurrently.

Usage::

    from jules_daemon.monitor.anomaly_models import (
        AnomalySeverity,
        ErrorKeywordPattern,
    )
    from jules_daemon.monitor.error_keyword_detector import ErrorKeywordDetector

    detector = ErrorKeywordDetector(
        patterns=(
            ErrorKeywordPattern(name="oom", regex=r"Out of memory|OOM",
                                severity=AnomalySeverity.CRITICAL),
            ErrorKeywordPattern(name="segfault", regex=r"SIGSEGV|Segmentation fault",
                                severity=AnomalySeverity.CRITICAL),
        ),
    )

    if detector.match(line):
        report = detector.report(line, session_id="run-1", detected_at=now)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Final

from jules_daemon.monitor.anomaly_models import (
    AnomalyReport,
    AnomalySeverity,
    ErrorKeywordPattern,
    PatternType,
)

__all__ = ["ErrorKeywordDetector"]

logger = logging.getLogger(__name__)

_DEFAULT_NAME: Final[str] = "error_keyword_detector"


class ErrorKeywordDetector:
    """Regex-based anomaly detector that scans output for error keywords.

    Accepts one or more ``ErrorKeywordPattern`` instances and tests each
    output line against them in order. Conforms to the ``AnomalyDetector``
    protocol via structural subtyping.

    Attributes:
        patterns: Immutable tuple of ``ErrorKeywordPattern`` instances
            checked in order during matching.
    """

    __slots__ = ("_name", "_patterns")

    def __init__(
        self,
        *,
        patterns: tuple[ErrorKeywordPattern, ...],
        name: str = _DEFAULT_NAME,
    ) -> None:
        """Initialize with one or more error keyword patterns.

        Args:
            patterns: Tuple of ``ErrorKeywordPattern`` instances to match
                against. Must contain at least one pattern.
            name: Identifier for this detector instance. Defaults to
                ``"error_keyword_detector"``.

        Raises:
            ValueError: If patterns is empty or name is blank.
        """
        stripped_name = name.strip()
        if not stripped_name:
            raise ValueError("name must not be empty")

        if not patterns:
            raise ValueError(
                "patterns must contain at least one ErrorKeywordPattern"
            )

        self._name: Final[str] = stripped_name
        self._patterns: Final[tuple[ErrorKeywordPattern, ...]] = patterns

    # ------------------------------------------------------------------
    # AnomalyDetector protocol properties
    # ------------------------------------------------------------------

    @property
    def pattern_name(self) -> str:
        """Unique identifier of this detector instance."""
        return self._name

    @property
    def pattern_type(self) -> PatternType:
        """Always ``PatternType.ERROR_KEYWORD`` for this detector."""
        return PatternType.ERROR_KEYWORD

    @property
    def patterns(self) -> tuple[ErrorKeywordPattern, ...]:
        """Immutable tuple of configured patterns."""
        return self._patterns

    # ------------------------------------------------------------------
    # AnomalyDetector protocol methods
    # ------------------------------------------------------------------

    def match(self, output_line: str) -> bool:
        """Test whether any configured pattern matches the output line.

        Scans patterns in order and short-circuits on the first match.
        Uses the pre-compiled regex from each ``ErrorKeywordPattern``
        for performance.

        Args:
            output_line: A single line of SSH output to test.

        Returns:
            True if at least one pattern matches the line.
        """
        for pattern in self._patterns:
            if pattern.compiled_regex.search(output_line):
                return True
        return False

    def report(
        self,
        output_line: str,
        *,
        session_id: str,
        detected_at: datetime,
    ) -> AnomalyReport:
        """Produce an anomaly report for a matched output line.

        Finds the first matching pattern and uses its metadata for the
        report. If no pattern matches (caller error), produces a
        fallback report using the detector's own name with INFO severity.

        Args:
            output_line: The matched output line.
            session_id: Identifier of the SSH session being monitored.
            detected_at: UTC timestamp of the detection.

        Returns:
            Immutable ``AnomalyReport`` with detection metadata.
        """
        matched_pattern = self.find_match(output_line)

        if matched_pattern is not None:
            # Extract the specific matched text for context
            regex_match = matched_pattern.compiled_regex.search(output_line)
            matched_text = regex_match.group(0) if regex_match else output_line

            return AnomalyReport(
                pattern_name=matched_pattern.name,
                pattern_type=PatternType.ERROR_KEYWORD,
                severity=matched_pattern.severity,
                message=(
                    f"Error keyword '{matched_pattern.name}' detected: "
                    f"{matched_text}"
                ),
                detected_at=detected_at,
                session_id=session_id,
                context={
                    "pattern_name": matched_pattern.name,
                    "matched_text": matched_text,
                    "output_line": output_line,
                },
            )

        # Fallback: no pattern matched (caller should have checked match() first)
        logger.warning(
            "report() called on non-matching line for detector '%s'",
            self._name,
        )
        return AnomalyReport(
            pattern_name=self._name,
            pattern_type=PatternType.ERROR_KEYWORD,
            severity=AnomalySeverity.INFO,
            message=f"No pattern matched in detector '{self._name}'",
            detected_at=detected_at,
            session_id=session_id,
            context={
                "pattern_name": self._name,
                "matched_text": "",
                "output_line": output_line,
            },
        )

    # ------------------------------------------------------------------
    # Extended matching helpers
    # ------------------------------------------------------------------

    def find_match(self, output_line: str) -> ErrorKeywordPattern | None:
        """Find the first pattern that matches the output line.

        Scans patterns in order and returns the first match. Useful when
        callers need to know which specific pattern matched, not just
        whether any pattern matched.

        Args:
            output_line: A single line of SSH output to test.

        Returns:
            The first matching ``ErrorKeywordPattern``, or None if no
            pattern matches.
        """
        for pattern in self._patterns:
            if pattern.compiled_regex.search(output_line):
                return pattern
        return None

    def match_all(self, output_line: str) -> tuple[ErrorKeywordPattern, ...]:
        """Find all patterns that match the output line.

        Scans every pattern (no short-circuit) and returns all matches
        in their original configuration order. Useful for audit logging
        when you want to record every triggered pattern.

        Args:
            output_line: A single line of SSH output to test.

        Returns:
            Tuple of all matching ``ErrorKeywordPattern`` instances.
            Empty tuple if no pattern matches.
        """
        matched: list[ErrorKeywordPattern] = []
        for pattern in self._patterns:
            if pattern.compiled_regex.search(output_line):
                matched.append(pattern)
        return tuple(matched)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        pattern_count = len(self._patterns)
        pattern_names = ", ".join(p.name for p in self._patterns)
        return (
            f"ErrorKeywordDetector(name={self._name!r}, "
            f"{pattern_count} pattern{'s' if pattern_count != 1 else ''}: "
            f"[{pattern_names}])"
        )
