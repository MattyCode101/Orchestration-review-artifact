"""Prompt rendering for the LLM orchestrator (final specification, sections 6 and 8).

The template file holds only worker-agnostic principles; everything specific to a
worker arrives through the cards and their generated profiles. Placeholders are
replaced by exact string substitution (no format strings), so braces in JSON are
safe. A prompt is a pure function of its inputs; its SHA-256 is the replay key.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orch.contract.models import WorkerCard
from orch.policies.llm_schemas import (
    MAX_CONFLICT_WORDS, MAX_DIAGNOSIS_WORDS, MAX_RATIONALE_CHARS, STRATEGIES,
)
from orch.policies.llm_view import HIGHER_IS_BETTER, METRIC_NAMES

PLACEHOLDERS = (
    "BRIEF", "WORKER_CARDS", "LESSONS_SECTION", "VIEW", "SCHEMA", "STRATEGIES",
    "MAX_DIAGNOSIS_WORDS", "MAX_CONFLICT_WORDS", "MAX_RATIONALE_CHARS", "MAX_ATTEMPTS", "ATTEMPTS_RULE",
)
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three"}


@dataclass(frozen=True)
class Template:
    path: str
    text: str
    sha256: str


def load_template(path: Path | str) -> Template:
    raw = Path(path).read_bytes()
    return Template(str(path), raw.decode("utf-8"), hashlib.sha256(raw).hexdigest())


def render_cards(cards: Sequence[WorkerCard], profiles: Mapping[str, str] | None) -> str:
    """One block per worker: objective and settings, then its generated profile when given."""

    blocks = []
    for card in cards:
        metric = METRIC_NAMES.get(card.objective.metric, card.objective.metric.value)
        better = "higher" if card.objective.metric in HIGHER_IS_BETTER or card.objective.direction == "max" else "lower"
        default = _NUMBER_WORDS.get(card.default_attempts, str(card.default_attempts))
        lines = [
            f"Worker {card.id}: improves {metric} ({better} is better). Settings: preserve_placement "
            f"true or false; attempts one to three (default {default})."
        ]
        text = (profiles or {}).get(card.id)
        if text:
            lines.append("Observed behaviour on past designs (frequencies in words):")
            lines.append(text.rstrip())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_prompt(
    template: Template, *, brief_text: str, cards_text: str, lessons: str | None,
    view_text: str, schema: Mapping[str, Any], max_attempts: int,
) -> str:
    lessons_section = ""
    if lessons and lessons.strip():
        lessons_section = (
            "\nLessons from earlier episodes (general, not tied to any design):\n" + lessons.strip() + "\n"
        )
    values = {
        "BRIEF": brief_text.rstrip(),
        "WORKER_CARDS": cards_text.rstrip(),
        "LESSONS_SECTION": lessons_section,
        "VIEW": view_text.rstrip("\n"),
        "SCHEMA": json.dumps(schema, indent=1, sort_keys=False),
        "STRATEGIES": ", ".join(STRATEGIES),
        "MAX_DIAGNOSIS_WORDS": str(MAX_DIAGNOSIS_WORDS),
        "MAX_CONFLICT_WORDS": str(MAX_CONFLICT_WORDS),
        "MAX_RATIONALE_CHARS": str(MAX_RATIONALE_CHARS),
        "MAX_ATTEMPTS": _NUMBER_WORDS.get(max_attempts, str(max_attempts)),
        "ATTEMPTS_RULE": (
            "Give exactly one attempt." if max_attempts <= 1 else
            f"Give attempts between one and {_NUMBER_WORDS.get(max_attempts, str(max_attempts))}."
        ),
    }
    out = template.text
    for name in PLACEHOLDERS:
        out = out.replace("{{" + name + "}}", values[name])
    return out


def repair_prompt(prompt: str, previous_reply: str, error: str) -> str:
    """The original prompt, the rejected reply and the exact reason; the model must answer again."""

    return (
        f"{prompt}\n\nYour previous reply was rejected.\nPrevious reply:\n{previous_reply.strip()[:1500]}\n"
        f"Reason: {error}\nReply again with one corrected JSON object and nothing else.\n"
    )
