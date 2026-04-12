"""Load Nemoclaw SKILL.md for runtime LLM prompts (strip Cursor YAML frontmatter)."""

from __future__ import annotations

from pathlib import Path

_SKILL_PATH = Path(__file__).resolve().parent / "SKILL.md"
_MAX_CHARS = 10_000

_cached: str | None = None


def load_skill_markdown_body() -> str:
    """Return SKILL.md body text without frontmatter, capped for prompt size."""
    global _cached
    if _cached is not None:
        return _cached
    if not _SKILL_PATH.is_file():
        _cached = ""
        return _cached
    raw = _SKILL_PATH.read_text(encoding="utf-8")
    if raw.startswith("---"):
        segments = raw.split("---", 2)
        if len(segments) >= 3:
            raw = segments[2].lstrip("\n")
    _cached = raw.strip()[:_MAX_CHARS]
    return _cached


def clear_skill_cache() -> None:
    """For tests: reload SKILL.md from disk on next call."""
    global _cached
    _cached = None
