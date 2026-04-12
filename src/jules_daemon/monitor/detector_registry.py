"""Detector registry with register/unregister pattern for anomaly detectors.

Provides a centralized registry that manages ``AnomalyDetector`` instances
by their unique ``pattern_name``. Detectors can be registered, looked up,
listed, and unregistered at runtime.

The registry is the foundation for the fan-out dispatcher: when a new
output line arrives, the dispatcher iterates over all registered detectors
and sends the line to each one concurrently.

Key properties:

- **Name uniqueness**: Each detector is keyed by its ``pattern_name``.
  Attempting to register a second detector with the same name raises
  ``ValueError``. This prevents silent shadowing of detectors.

- **Protocol-based acceptance**: Accepts any object satisfying the
  ``AnomalyDetector`` protocol (structural subtyping). No inheritance
  required from concrete implementations.

- **Immutable snapshots**: ``list_detectors()`` and ``detector_names``
  return new immutable collections (tuple, frozenset) on each call,
  safe to use across async boundaries.

- **Thread-safe**: All mutable state is guarded by a ``threading.Lock``,
  safe for concurrent access from multiple asyncio tasks or OS threads.

Usage::

    from jules_daemon.monitor.detector_registry import DetectorRegistry
    from jules_daemon.monitor.error_keyword_detector import ErrorKeywordDetector

    registry = DetectorRegistry()

    detector = ErrorKeywordDetector(
        patterns=(ErrorKeywordPattern(name="oom", regex=r"OOM"),),
        name="oom_detector",
    )
    registry.register(detector)

    # Look up by name
    found = registry.get("oom_detector")

    # List all registered detectors
    all_detectors = registry.list_detectors()

    # Unregister when no longer needed
    removed = registry.unregister("oom_detector")
"""

from __future__ import annotations

import logging
import threading
from typing import Final

from jules_daemon.monitor.anomaly_models import AnomalyDetector

__all__ = ["DetectorRegistry"]

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """Registry for ``AnomalyDetector`` instances with register/unregister.

    Manages detectors by their unique ``pattern_name``. Provides O(1)
    lookup by name and snapshot-based listing methods that return immutable
    collections.

    Thread safety:
        All methods are guarded by a ``threading.Lock`` and are safe for
        concurrent access from multiple asyncio tasks or OS threads.

    Attributes (read-only via properties):
        count: Number of currently registered detectors.
        detector_names: Frozenset of registered detector names.
    """

    __slots__ = ("_detectors", "_lock")

    def __init__(self) -> None:
        """Initialize an empty detector registry."""
        self._detectors: dict[str, AnomalyDetector] = {}
        self._lock: Final[threading.Lock] = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, detector: AnomalyDetector) -> None:
        """Register an anomaly detector by its pattern_name.

        The detector must satisfy the ``AnomalyDetector`` protocol
        (structural subtyping). Its ``pattern_name`` property is used
        as the unique key in the registry.

        Args:
            detector: The detector to register. Must have a unique
                ``pattern_name`` that is not already registered.

        Raises:
            ValueError: If a detector with the same ``pattern_name``
                is already registered.
        """
        name = detector.pattern_name
        with self._lock:
            if name in self._detectors:
                raise ValueError(
                    f"Detector {name!r} is already registered. "
                    f"Unregister it first before re-registering."
                )
            self._detectors[name] = detector
        logger.debug("Registered detector %r", name)

    def unregister(self, name: str) -> AnomalyDetector:
        """Unregister a detector by name and return it.

        Args:
            name: The ``pattern_name`` of the detector to remove.

        Returns:
            The removed detector instance.

        Raises:
            KeyError: If no detector with the given name is registered.
        """
        with self._lock:
            try:
                detector = self._detectors.pop(name)
            except KeyError:
                raise KeyError(
                    f"No detector named {name!r} is registered"
                ) from None
        logger.debug("Unregistered detector %r", name)
        return detector

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> AnomalyDetector | None:
        """Look up a registered detector by name.

        Args:
            name: The ``pattern_name`` to look up.

        Returns:
            The detector if found, or None if not registered.
        """
        with self._lock:
            return self._detectors.get(name)

    def list_detectors(self) -> tuple[AnomalyDetector, ...]:
        """Return all registered detectors as an immutable tuple.

        Each call produces a new tuple snapshot, safe to use across
        async boundaries without synchronization concerns.

        Returns:
            Tuple of all registered ``AnomalyDetector`` instances.
            Order is insertion order (Python dict ordering).
        """
        with self._lock:
            return tuple(self._detectors.values())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of currently registered detectors."""
        with self._lock:
            return len(self._detectors)

    @property
    def detector_names(self) -> frozenset[str]:
        """Frozenset of registered detector names.

        Each call produces a new frozenset snapshot.

        Returns:
            Frozenset of detector ``pattern_name`` values.
        """
        with self._lock:
            return frozenset(self._detectors.keys())

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._detectors)
            names = ", ".join(sorted(self._detectors.keys()))
        if names:
            return f"DetectorRegistry({count} detectors: [{names}])"
        return f"DetectorRegistry({count} detectors)"
