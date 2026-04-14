"""Markdown-backed system definitions for named SSH target aliases.

System pages live under ``{wiki_root}/pages/systems/`` and let users bind a
friendly name such as ``tuto`` to an SSH target. The CLI and daemon can then
resolve prompts like ``run the smoke tests in system tuto`` without requiring
the user to type ``root@<ip>`` each time.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jules_daemon.wiki import frontmatter

__all__ = [
    "SYSTEMS_DIR",
    "SystemInfo",
    "find_system",
    "find_system_mention",
    "list_systems",
    "strip_system_mention",
]

logger = logging.getLogger(__name__)

SYSTEMS_DIR = "pages/systems"
_WIKI_TYPE = "system-info"
_NAME_RE = re.compile(r"[^a-z0-9_.-]+")
_SYSTEM_MENTION_PREFIXES: tuple[str, ...] = ("in", "on", "at")


def _normalize_name(value: str) -> str:
    """Normalize system identifiers for matching."""
    return _NAME_RE.sub("-", value.strip().lower()).strip("-")


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _coerce_port(value: Any) -> int:
    if value in (None, ""):
        return 22
    try:
        port = int(value)
    except (TypeError, ValueError):
        return 22
    return port if 1 <= port <= 65535 else 22


def _coerce_aliases(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raw_items = [value]
    normalized: list[str] = []
    for item in raw_items:
        alias = _normalize_name(_coerce_string(item))
        if alias and alias not in normalized:
            normalized.append(alias)
    return tuple(normalized)


@dataclass(frozen=True)
class SystemInfo:
    """Resolved SSH target metadata loaded from a wiki page."""

    system_name: str
    host: str
    user: str
    port: int = 22
    aliases: tuple[str, ...] = ()
    key_path: str | None = None
    description: str = ""
    hostname: str = ""
    ip_address: str = ""

    def __post_init__(self) -> None:
        if not self.system_name.strip():
            raise ValueError("system_name must not be empty")
        if not self.host.strip():
            raise ValueError("host must not be empty")
        if not self.user.strip():
            raise ValueError("user must not be empty")
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be 1-65535, got {self.port}")

    @property
    def normalized_name(self) -> str:
        return _normalize_name(self.system_name)

    def matches(self, query: str) -> bool:
        normalized = _normalize_name(query)
        if not normalized:
            return False
        return normalized == self.normalized_name or normalized in self.aliases

    @property
    def display_hostname(self) -> str:
        """Human-readable hostname for prompts, if available."""
        if self.hostname:
            return self.hostname
        try:
            ipaddress.ip_address(self.host)
        except ValueError:
            return self.host
        return ""

    @property
    def display_ip_address(self) -> str:
        """Human-readable IP address for prompts, if available."""
        if self.ip_address:
            return self.ip_address
        try:
            ipaddress.ip_address(self.host)
        except ValueError:
            return ""
        return self.host


def _systems_path(wiki_root: Path) -> Path:
    return wiki_root / SYSTEMS_DIR


def _from_frontmatter(file_path: Path, fm: dict[str, Any], body: str) -> SystemInfo:
    name = (
        _coerce_string(fm.get("system_name"))
        or _coerce_string(fm.get("name"))
        or file_path.stem
    )
    host = _coerce_string(fm.get("host"))
    user = _coerce_string(fm.get("user"))
    port = _coerce_port(fm.get("port"))
    aliases = _coerce_aliases(fm.get("aliases"))
    key_path = _coerce_string(fm.get("key_path")) or None
    description = _coerce_string(fm.get("description"))
    hostname = _coerce_string(fm.get("hostname"))
    ip_address = (
        _coerce_string(fm.get("ip_address"))
        or _coerce_string(fm.get("ip"))
        or _coerce_string(fm.get("address"))
    )
    if not description and body.strip():
        description = body.splitlines()[0].strip()
    return SystemInfo(
        system_name=name,
        host=host,
        user=user,
        port=port,
        aliases=aliases,
        key_path=key_path,
        description=description,
        hostname=hostname,
        ip_address=ip_address,
    )


def list_systems(wiki_root: Path) -> tuple[SystemInfo, ...]:
    """Load all valid system definitions from the wiki."""
    systems_dir = _systems_path(wiki_root)
    if not systems_dir.exists():
        return ()

    systems: list[SystemInfo] = []
    for file_path in sorted(systems_dir.glob("*.md")):
        if file_path.name.lower() == "readme.md":
            continue
        try:
            raw = file_path.read_text(encoding="utf-8")
            doc = frontmatter.parse(raw)
            doc_type = _coerce_string(doc.frontmatter.get("type"))
            if doc_type and doc_type != _WIKI_TYPE:
                continue
            systems.append(_from_frontmatter(file_path, doc.frontmatter, doc.body))
        except Exception as exc:
            logger.warning("Skipping invalid system page %s: %s", file_path, exc)
    return tuple(systems)


def find_system(wiki_root: Path, query: str) -> SystemInfo | None:
    """Return the first system matching *query* by name or alias."""
    normalized = _normalize_name(query)
    if not normalized:
        return None
    for system in list_systems(wiki_root):
        if system.matches(normalized):
            return system
    return None


def find_system_mention(wiki_root: Path, text: str) -> SystemInfo | None:
    """Infer a system from free text like ``run tests in tuto``.

    Matches only known system names/aliases from the wiki, which keeps
    phrases like ``in parallel`` from becoming false positives unless a
    real system named ``parallel`` exists.
    """
    raw = text.strip()
    if not raw:
        return None

    candidates: list[tuple[str, SystemInfo]] = []
    for system in list_systems(wiki_root):
        names = [system.normalized_name, *system.aliases]
        for candidate in names:
            if candidate:
                candidates.append((candidate, system))

    # Prefer longer aliases first when one system name is a prefix of another.
    for candidate, system in sorted(
        candidates,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        escaped = re.escape(candidate)
        prefix = "|".join(re.escape(p) for p in _SYSTEM_MENTION_PREFIXES)
        pattern = re.compile(
            rf"\b(?:{prefix})\s+(?:system\s+)?{escaped}(?=$|[\s?.!,;:])",
            re.IGNORECASE,
        )
        if pattern.search(raw):
            return system
    return None


def strip_system_mention(text: str, system: SystemInfo) -> str:
    """Remove a resolved system alias phrase from free text.

    Examples:
        ``run the smoke tests in tuto`` -> ``run the smoke tests``
        ``run the smoke tests in system tuto`` -> ``run the smoke tests``
    """
    raw = text.strip()
    if not raw:
        return raw

    result = raw
    prefix = "|".join(re.escape(p) for p in _SYSTEM_MENTION_PREFIXES)
    candidates = [system.normalized_name, *system.aliases]
    for candidate in sorted(
        (item for item in candidates if item),
        key=len,
        reverse=True,
    ):
        escaped = re.escape(candidate)
        pattern = re.compile(
            rf"\b(?:{prefix})\s+(?:system\s+)?{escaped}(?=$|[\s?.!,;:])",
            re.IGNORECASE,
        )
        result = pattern.sub("", result)

    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r"\s+([?.!,;:])", r"\1", result)
    cleaned = result.strip()
    return cleaned or raw
