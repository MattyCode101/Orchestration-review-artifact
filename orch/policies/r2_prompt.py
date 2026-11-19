"""Prompt rendering for the round-2 evidence sections.

The round-2 template (configs/prompts/orchestrator2_v1.txt) is the R1 template with two extra
placeholders between the situation and the reply format, so the decision request stays last. Everything
else is rendered by the round-1 renderer, so with both sections empty the prompt is byte-identical to
the R1 prompt.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from orch.policies.llm_prompt import Template, render_prompt

R2_PLACEHOLDERS = ("PHYSICAL_SECTION", "EXPERIENCE_SECTION")


def section_block(text: str) -> str:
    """A section sits between blank lines; an absent section leaves no trace."""

    return "" if not text else "\n" + text.rstrip("\n") + "\n"


def render_prompt_r2(
    template: Template, *, brief_text: str, cards_text: str, view_text: str, schema: Mapping[str, Any],
    max_attempts: int, physical_text: str = "", experience_text: str = "",
) -> str:
    prompt = render_prompt(
        template, brief_text=brief_text, cards_text=cards_text, lessons=None, view_text=view_text,
        schema=schema, max_attempts=max_attempts,
    )
    values = {"PHYSICAL_SECTION": section_block(physical_text), "EXPERIENCE_SECTION": section_block(experience_text)}
    for name in R2_PLACEHOLDERS:
        prompt = prompt.replace("{{" + name + "}}", values[name])
    return prompt
