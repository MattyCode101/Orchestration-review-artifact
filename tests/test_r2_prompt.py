"""Round-2 template: neutral, identical to R1 apart from the two section placeholders, decision request last."""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import yaml

import llm_support as ls
from orch.contract.brief_text import render_brief_text
from orch.policies.llm_prompt import PLACEHOLDERS, load_template, render_cards, render_prompt
from orch.policies.llm_schemas import decision_json_schema
from orch.policies.llm_view import build_fuzzy_view
from orch.policies.r2_prompt import R2_PLACEHOLDERS, render_prompt_r2, section_block

ROOT = Path(__file__).resolve().parents[1]
R1 = ROOT / "configs/prompts/orchestrator_v2.txt"
R2 = ROOT / "configs/prompts/orchestrator2_v1.txt"
PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")
WORKER_IDS = ("timing", "area", "power")
DESIGN_NAMES = ("vexriscv", "logicnets", "rosetta", "finn", "radioml", "corescore", "fir_", "digit", "spam", "rendering", "jscl", "mcml", "vtr")


def test_the_round_2_template_is_neutral_like_every_other_revision() -> None:
    raw = R2.read_text(encoding="utf-8")
    assert set(PLACEHOLDER.findall(raw)) <= set(PLACEHOLDERS) | set(R2_PLACEHOLDERS)
    text = PLACEHOLDER.sub("", raw)
    assert not re.search(r"\d", text), "digits must come only from placeholders"
    lowered = text.lower()
    for word in WORKER_IDS:
        assert not re.search(rf"\b{word}\b", lowered)
    recipes = [e["id"] for e in yaml.safe_load((ROOT / "recipes/registry.yaml").read_text(encoding="utf-8"))["recipes"]]
    for recipe in recipes:
        assert recipe.lower() not in lowered
    for name in (*DESIGN_NAMES, "latch"):
        assert name not in lowered, name
    for needed in ("Never repeat a combination", "satisfies every limit", "saved state"):
        assert needed in raw


def test_the_only_change_from_r1_is_the_line_holding_the_two_section_placeholders() -> None:
    a, b = R1.read_text(encoding="utf-8").splitlines(), R2.read_text(encoding="utf-8").splitlines()
    changes = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    assert changes == ["-", "+{{PHYSICAL_SECTION}}{{EXPERIENCE_SECTION}}"]


def test_the_eight_principles_are_the_round_1_principles() -> None:
    def principles(path: Path) -> list[str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        start = lines.index("Principles") + 1
        end = lines.index("The brief")
        return [l for l in lines[start:end] if l.startswith("- ")]
    assert len(principles(R2)) == 8 and principles(R2) == principles(R1)


def _inputs():
    view = ls.view(visited=[ls.state(10, -0.347, 1337), ls.state(11, -0.147, 1352, brief_valid=False)],
                   calls=(ls.call(1, "timing", 10, 11, reason="push the goal"),), steps=(ls.step("timing", 10, 11),), budget_left=4)
    built = build_fuzzy_view(view)
    cards = render_cards(view.worker_cards, {"timing": "  profile text", "area": "  other profile text"})
    schema = decision_json_schema(["timing", "area"], ["S0", "S1"], max_attempts=3)
    return view, built, cards, schema


def test_with_both_sections_empty_the_prompt_is_the_round_1_prompt_byte_for_byte() -> None:
    view, built, cards, schema = _inputs()
    old = render_prompt(load_template(R1), brief_text=render_brief_text(view.brief), cards_text=cards, lessons=None,
                        view_text=built.text, schema=schema, max_attempts=3)
    new = render_prompt_r2(load_template(R2), brief_text=render_brief_text(view.brief), cards_text=cards,
                           view_text=built.text, schema=schema, max_attempts=3)
    assert new == old and not PLACEHOLDER.search(new)


def test_the_evidence_sections_sit_between_the_situation_and_the_decision_request() -> None:
    view, built, cards, schema = _inputs()
    prompt = render_prompt_r2(
        load_template(R2), brief_text=render_brief_text(view.brief), cards_text=cards, view_text=built.text,
        schema=schema, max_attempts=3, physical_text="Physical description of the saved states\nS0 (start)",
        experience_text="Past experience with the workers (profiles from earlier designs)\ntiming",
    )
    order = [prompt.index(s) for s in ("Current situation", "trajectory_log:", "Physical description of the saved states",
                                       "Past experience with the workers", "Reply format")]
    assert order == sorted(order)
    assert "\n\nPhysical description of the saved states\nS0 (start)\n\nPast experience with the workers" in prompt
    assert "timing\n\nReply format\n" in prompt
    assert prompt.rstrip().splitlines()[-1].startswith("Choose the strategy from")      # the request is last
    assert not PLACEHOLDER.search(prompt)


def test_one_section_alone_leaves_single_blank_lines() -> None:
    view, built, cards, schema = _inputs()
    common = dict(brief_text=render_brief_text(view.brief), cards_text=cards, view_text=built.text, schema=schema, max_attempts=3)
    only_experience = render_prompt_r2(load_template(R2), experience_text="Past experience\nline", **common)
    assert "\n\n\n" not in only_experience and "trajectory_log:" in only_experience
    assert "\n\nPast experience\nline\n\nReply format" in only_experience
    assert section_block("") == "" and section_block("x\n\n") == "\nx\n"
