"""Lessons block for configuration F2: provenance, leave-one-design-out exclusion, hygiene."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from orch.analysis.lessons import Lesson, lessons_block, load_lessons

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = ("vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter", "rosetta_3d-rendering", "finn_radioml")


def _lessons() -> list[Lesson]:
    return [
        Lesson("T0-1", "T0", "vexriscv_re-place", "First case text. Takeaway: use both kinds of worker."),
        Lesson("T0-2", "T0", "logicnets_jscl", "Second case text. Takeaway: keep the intermediate."),
        Lesson("A0-1", "A0", "vexriscv_re-place", "Third case text. Takeaway: call the same worker again."),
        Lesson("A0-2", "A0", "finn_radioml", "Third case text. Takeaway: call the same worker again."),   # same wording, other design
    ]


def test_without_a_held_out_design_every_distinct_lesson_appears_once() -> None:
    block = lessons_block(_lessons())
    assert block.count("\n") == 2 and block.startswith("- First case text")
    assert block.count("Third case text") == 1                      # identical wording is not repeated


def test_a_fold_excludes_every_lesson_derived_from_the_held_out_design() -> None:
    block = lessons_block(_lessons(), exclude_design="vexriscv_re-place")
    assert "First case" not in block and "Second case" in block
    # the same wording was also derived from the held-out design, so it is excluded as well
    assert "Third case" not in block
    assert lessons_block(_lessons()[:1], exclude_design="vexriscv_re-place") == ""
    assert "Third case" in lessons_block(_lessons(), exclude_design="logicnets_jscl")


def test_the_block_is_deterministic() -> None:
    assert lessons_block(_lessons(), exclude_design="finn_radioml") == lessons_block(list(_lessons()), exclude_design="finn_radioml")


def test_an_unknown_excluded_design_is_an_error_not_a_silent_no_op() -> None:
    with pytest.raises(ValueError):
        lessons_block(_lessons(), exclude_design="not_a_design_in_the_file")


@pytest.mark.skipif(not (ROOT / "configs/lessons/lessons_training.yaml").exists(), reason="lessons file not generated yet")
def test_the_generated_lessons_are_free_of_numbers_and_design_names_and_folds_are_clean() -> None:
    path = ROOT / "configs/lessons/lessons_training.yaml"
    lessons = load_lessons(path)
    assert len(lessons) == 12 and {x.brief for x in lessons} == {"T0", "A0", "T", "A"}
    for lesson in lessons:
        assert lesson.source_design in DESIGNS
        assert not re.search(r"\d", lesson.text), lesson.text
        assert not any(name in lesson.text for name in DESIGNS)
        assert not re.search(r"[0-9a-f]{12,}", lesson.text)
    for design in DESIGNS:
        block = lessons_block(lessons, exclude_design=design)
        assert block, "every fold keeps some lessons"
        for lesson in lessons:
            if lesson.source_design == design:
                assert lesson.text not in block, f"a lesson from {design} survives its own fold"
    assert yaml.safe_load(path.read_text())["schema_version"] == "lessons-v1"
