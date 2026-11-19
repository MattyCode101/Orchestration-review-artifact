"""Template v2 packages: assembly, neutrality (the round-1 rule still applies), and what each package removes."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from orch.policies.llm_prompt import PLACEHOLDERS
from orch.policies.v2_prompt import AssemblyError, apply_edits, assemble, combo_name

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "configs/prompts/template_v2"
BASE = (ROOT / "configs/prompts/orchestrator2_v1.txt").read_text(encoding="utf-8")
PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")
V2_PLACEHOLDERS = set(PLACEHOLDERS) | {"PHYSICAL_SECTION", "EXPERIENCE_SECTION", "VOCABULARY", "START_ALIAS",
                                       "REPLY_FIELDS"}
COMBOS = [("P1",), ("P1", "P2"), ("P1", "P2", "P3"), ("P1", "P2", "P3", "P4"), ("P2", "P3", "P4"),
          ("P1", "P3", "P4"), ("P1", "P2", "P4"), ("P1", "P2", "P3", "P4r")]
WORKER_IDS = ("timing", "area", "power")
DESIGN_NAMES = ("vexriscv", "logicnets", "rosetta", "finn", "radioml", "corescore", "fir_", "digit", "spam",
                "rendering", "jscl", "mcml", "vtr", "cordic", "mini-isp", "or1200")


def _recipe_ids() -> list[str]:
    registry = yaml.safe_load((ROOT / "recipes/registry.yaml").read_text(encoding="utf-8"))
    return [entry["id"] for entry in registry["recipes"]]


def test_no_package_is_the_round1_wording_byte_for_byte() -> None:
    assert assemble(BASE, (), FOLDER) == BASE


@pytest.mark.parametrize("combo", COMBOS, ids=lambda c: combo_name(c))
def test_every_combination_assembles_and_stays_neutral(combo) -> None:
    text = assemble(BASE, combo, FOLDER)
    assert set(PLACEHOLDER.findall(text)) <= V2_PLACEHOLDERS
    outside = PLACEHOLDER.sub("", text)
    assert not re.search(r"\d", outside), "digits must come only from placeholders"
    lowered = outside.lower()
    for word in WORKER_IDS:
        assert not re.search(rf"\b{word}\b", lowered), f"worker id {word!r}"
    for recipe in _recipe_ids():
        assert recipe.lower() not in lowered
    for name in (*DESIGN_NAMES, "latch", "b2"):
        assert not re.search(rf"\b{re.escape(name)}", lowered), name
    for needed in ("satisfies every limit", "saved state"):
        assert needed in text


@pytest.mark.parametrize("combo", COMBOS, ids=lambda c: combo_name(c))
def test_each_package_removes_what_it_should(combo) -> None:
    text = assemble(BASE, combo, FOLDER)
    if "P1" in combo:
        assert "Never repeat a combination" not in text and "raw measurements" not in text
        assert "profiles show high variance" not in text and "{{VOCABULARY}}" in text
    if "P2" in combo:
        assert "{{STRATEGIES}}" not in text and "conflict_check" not in text
        assert "safe" not in text and "undo" not in text and "Considerations" in text
        assert "placement may change" in text.lower() and "placement must not change" in text.lower()
    if "P4" in combo or "P4r" in combo:
        assert "{{SCHEMA}}" not in text and "{{REPLY_FIELDS}}" in text
    elif "P3" in combo:
        assert "physical_facts" in text


def test_p4_and_its_revision_cannot_be_combined_and_have_distinct_names() -> None:
    with pytest.raises(AssemblyError):
        assemble(BASE, ("P1", "P4", "P4r"), FOLDER)
    assert combo_name(("P1", "P2", "P3", "P4r")) == "p1234r" and combo_name(("P4", "P1")) == "p14"


def test_the_revision_keeps_physical_facts_and_anchors_the_forecast() -> None:
    from orch.policies.v2_schemas import SchemaFlags, reply_schema

    texts = yaml.safe_load((FOLDER / "p4_reply_rev1.yaml").read_text(encoding="utf-8"))["texts"]
    flags = SchemaFlags(p2=True, p3=True, p4=True, vocab_v2=True, p4r=True)
    schema = reply_schema(flags, workers=["timing", "area"], aliases=["S0"], max_attempts=3, limit_names=["wns"],
                          texts=texts)
    order = schema["propertyOrdering"]
    assert order.index("evidence") < order.index("physical_facts") < order.index("options")
    assert "experience line" in schema["properties"]["forecast"]["description"]
    plain = reply_schema(SchemaFlags(p2=True, p3=True, p4=True, vocab_v2=True), workers=["timing"], aliases=["S0"],
                         max_attempts=3, limit_names=["wns"], texts=texts)
    assert "physical_facts" not in plain["properties"]


def test_an_edit_that_does_not_apply_exactly_once_is_refused() -> None:
    with pytest.raises(AssemblyError):
        apply_edits("abc abc", [{"old": "abc", "new": "x"}], package="T")
    with pytest.raises(AssemblyError):
        apply_edits("abc", [{"old": "zzz", "new": "x"}], package="T")


def test_stored_assembled_templates_equal_a_fresh_assembly() -> None:
    for combo in COMBOS:
        path = FOLDER / "assembled" / f"orchestrator3_{combo_name(combo)}.txt"
        assert path.read_text(encoding="utf-8") == assemble(BASE, combo, FOLDER), path.name


def test_worker_descriptions_name_no_recipe_or_design() -> None:
    data = yaml.safe_load((FOLDER / "worker_descriptions.yaml").read_text(encoding="utf-8"))
    assert set(data) == {"W1", "W2", "W3"}
    for wording in data.values():
        for text in wording.values():
            lowered = text.lower()
            assert not re.search(r"\d", text)
            for recipe in _recipe_ids():
                assert recipe.lower() not in lowered
            for name in (*DESIGN_NAMES, "latch"):
                assert name not in lowered
