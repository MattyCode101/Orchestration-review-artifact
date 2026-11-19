"""Lessons for configuration F2: number-free, design-free cases from the training best-known paths.

Every lesson remembers which training design it came from (metadata only; the prompt
never shows it). In a leave-one-design-out fold, `lessons_block(..., exclude_design=d)`
drops every lesson derived from d, and also any lesson with identical wording, because
that wording was derived from d as well. Identical wording from several designs is shown
once.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Lesson:
    id: str
    brief: str
    source_design: str
    text: str


def load_lessons(path: Path | str) -> list[Lesson]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != "lessons-v1":
        raise ValueError("unknown lessons file schema")
    return [Lesson(x["id"], x["brief"], x["source_design"], x["text"]) for x in data["lessons"]]


def lessons_block(lessons: list[Lesson], exclude_design: str | None = None) -> str:
    designs = {lesson.source_design for lesson in lessons}
    if exclude_design is not None and exclude_design not in designs:
        raise ValueError(f"no lesson comes from {exclude_design!r}; check the design name")
    banned = {lesson.text for lesson in lessons if lesson.source_design == exclude_design}
    seen: set[str] = set()
    lines = []
    for lesson in lessons:
        if lesson.text in banned or lesson.text in seen:
            continue
        seen.add(lesson.text)
        lines.append(f"- {lesson.text}")
    return "\n".join(lines)
