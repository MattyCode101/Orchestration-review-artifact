"""Prompt template neutrality, anonymisation, schema and card rendering (acceptance tests 2 and 3)."""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import llm_support as ls
from orch.contract.brief_text import render_brief_text
from orch.policies.llm_prompt import (
    PLACEHOLDERS, load_template, render_cards, render_prompt, repair_prompt,
)
from orch.policies.llm_schemas import (
    LLMDecisionOutput, MandatePayload, STRATEGIES, decision_json_schema,
)
from orch.policies.llm_view import build_fuzzy_view

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/prompts/orchestrator_v1.txt"
PLACEHOLDER = re.compile(r"\{\{([A-Z_]+)\}\}")
WORKER_IDS = ("timing", "area", "power")
DESIGN_NAMES = (
    "vexriscv", "logicnets", "rosetta", "finn", "radioml", "corescore", "fir_", "digit",
    "spam", "rendering", "jscl", "mcml", "vtr",
)


def _recipe_ids() -> list[str]:
    registry = yaml.safe_load((ROOT / "recipes/registry.yaml").read_text(encoding="utf-8"))
    return [entry["id"] for entry in registry["recipes"]]


def _outside_placeholders(text: str) -> str:
    return PLACEHOLDER.sub("", text)


def test_the_template_hash_is_stable_and_every_placeholder_is_known() -> None:
    template = load_template(TEMPLATE)
    assert len(template.sha256) == 64
    assert set(PLACEHOLDER.findall(template.text)) <= set(PLACEHOLDERS)
    assert {"BRIEF", "WORKER_CARDS", "VIEW", "SCHEMA"} <= set(PLACEHOLDER.findall(template.text))


def test_template_neutrality_no_digits_worker_ids_recipes_or_design_names() -> None:
    text = _outside_placeholders(TEMPLATE.read_text(encoding="utf-8"))
    assert not re.search(r"\d", text), "digits must come only from placeholders"
    lowered = text.lower()
    for word in WORKER_IDS:
        assert not re.search(rf"\b{word}\b", lowered), f"worker id {word!r} in the template"
    for recipe in _recipe_ids():
        assert recipe.lower() not in lowered, f"recipe {recipe!r} in the template"
    assert "latch" not in lowered
    for name in DESIGN_NAMES:
        assert name not in lowered, f"design name fragment {name!r} in the template"


def _prompt(*, profiles_on: bool = True, lessons: str | None = None, mode: str = "fuzzy"):
    view = ls.view(
        visited=[ls.state(10, -0.347, 1337), ls.state(11, -0.147, 1352, brief_valid=False)],
        calls=(ls.call(1, "timing", 10, 11, reason="push the goal"),),
        steps=(ls.step("timing", 10, 11),), budget_left=4,
    )
    built = build_fuzzy_view(view, mode=mode)
    profiles = {"timing": "  when placement is preserved: goal within_noise usually.",
                "area": "  when placement is preserved: goal small_gain often."}
    cards_text = render_cards(view.worker_cards, profiles if profiles_on else None)
    template = load_template(TEMPLATE)
    schema = decision_json_schema(["timing", "area"], ["S0", "S1"], max_attempts=3)
    return render_prompt(
        template, brief_text=render_brief_text(view.brief), cards_text=cards_text,
        lessons=lessons, view_text=built.text, schema=schema, max_attempts=3,
    ), view, built


def test_a_rendered_prompt_is_anonymous() -> None:
    prompt, view, built = _prompt()
    assert ls.DESIGN_SECRET not in prompt and view.episode_id not in prompt
    assert not re.search(r"[0-9a-f]{12,}", prompt)
    assert "/home" not in prompt and "data/" not in prompt and ".dcp" not in prompt.lower()
    for state_hash in (ls.sid(10), ls.sid(11)):
        assert state_hash not in prompt
    for recipe in _recipe_ids():
        assert recipe not in prompt, f"recipe {recipe!r} leaked into an orchestrator prompt"
    for name in DESIGN_NAMES:
        assert name not in prompt.lower(), name


def test_every_placeholder_is_replaced() -> None:
    prompt, _, _ = _prompt()
    assert not PLACEHOLDER.search(prompt)
    assert "{{" not in prompt and "}}" not in prompt


def test_profiles_off_keeps_the_cards_but_drops_the_profile_lines() -> None:
    on, _, _ = _prompt(profiles_on=True)
    off, _, _ = _prompt(profiles_on=False)
    assert "goal within_noise usually" in on and "goal within_noise usually" not in off
    for text in (on, off):
        assert "Worker timing" in text and "Worker area" in text
    assert on != off


def test_cards_state_objective_direction_and_settings_without_recipes() -> None:
    cards = ls.cards()
    text = render_cards(cards, None)
    assert "Worker timing: improves wns (higher is better)" in text
    assert "Worker area: improves packed_luts (lower is better)" in text
    assert "preserve_placement" in text and "attempts" in text


def test_the_lessons_block_exists_only_when_supplied() -> None:
    without, _, _ = _prompt(lessons=None)
    blank, _, _ = _prompt(lessons="")
    with_lessons, _, _ = _prompt(lessons="- A result that breaks a limit can often be repaired by the owner of that limit.")
    assert without == blank and "Lessons" not in without
    assert "Lessons" in with_lessons and "owner of that limit" in with_lessons


def test_exact_and_fuzzy_prompts_differ_only_in_the_view() -> None:
    fuzzy, _, built_f = _prompt(mode="fuzzy")
    exact, _, built_e = _prompt(mode="exact")
    assert fuzzy.replace(built_f.text, "") == exact.replace(built_e.text, "")


def test_the_repair_prompt_carries_the_exact_error_and_the_rejected_reply() -> None:
    prompt, _, _ = _prompt()
    again = repair_prompt(prompt, '{"action": "INVOKE_WORKER"}', "mandate is required when the action is INVOKE_WORKER")
    assert again.startswith(prompt)
    assert "mandate is required when the action is INVOKE_WORKER" in again
    assert '{"action": "INVOKE_WORKER"}' in again


def test_the_schema_follows_the_episode() -> None:
    schema = decision_json_schema(["timing", "area"], ["S0", "S1", "S2"], max_attempts=2)
    mandate = schema["properties"]["mandate"]["anyOf"][1]
    assert mandate["properties"]["worker_id"]["enum"] == ["timing", "area"]
    assert mandate["properties"]["base_state"]["enum"] == ["S0", "S1", "S2"]
    assert mandate["properties"]["attempts"]["minimum"] == 1 and mandate["properties"]["attempts"]["maximum"] == 2
    assert schema["properties"]["strategy"]["enum"] == list(STRATEGIES)
    assert schema["properties"]["action"]["enum"] == ["INVOKE_WORKER", "STOP"]
    assert schema["propertyOrdering"][-1] == "mandate" and schema["propertyOrdering"][0] == "state_diagnosis"


def test_the_schema_uses_only_keywords_the_sdk_supports() -> None:
    allowed = {
        "$id", "$defs", "$ref", "$anchor", "type", "format", "title", "description", "enum", "items",
        "prefixItems", "minItems", "maxItems", "minimum", "maximum", "anyOf", "oneOf", "properties",
        "additionalProperties", "required", "propertyOrdering",
    }
    seen: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key not in ("properties",):
                    seen.add(key)
                if key == "properties":
                    for sub in value.values():
                        walk(sub)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(decision_json_schema(["timing", "area"], ["S0"], max_attempts=3))
    assert seen <= allowed, seen - allowed


def test_the_schema_and_the_pydantic_model_name_the_same_fields() -> None:
    schema = decision_json_schema(["timing"], ["S0"], max_attempts=3)
    assert set(schema["properties"]) == set(LLMDecisionOutput.model_fields)
    inner = schema["properties"]["mandate"]["anyOf"][1]
    assert set(inner["properties"]) == set(MandatePayload.model_fields)
    assert set(inner["required"]) == set(MandatePayload.model_fields)


def _decision(**changes):
    base = {
        "state_diagnosis": "d", "strategy": "PUSH", "conflict_check": "c", "rationale_summary": "r",
        "action": "INVOKE_WORKER",
        "mandate": {"worker_id": "timing", "base_state": "S0", "preserve_placement": True,
                    "attempts": 2, "respect_brief_limits": False},
    }
    base.update(changes)
    return base


def test_a_mandate_is_required_exactly_when_the_action_invokes_a_worker() -> None:
    assert LLMDecisionOutput.model_validate(_decision()).mandate.attempts == 2
    assert LLMDecisionOutput.model_validate(_decision(action="STOP", strategy="STOP", mandate=None)).mandate is None
    with pytest.raises(ValidationError):
        LLMDecisionOutput.model_validate(_decision(mandate=None))
    with pytest.raises(ValidationError):
        LLMDecisionOutput.model_validate(_decision(action="STOP"))


@pytest.mark.parametrize("attempts", [0, 4, -1])
def test_attempts_outside_one_to_three_are_rejected(attempts) -> None:
    bad = _decision()
    bad["mandate"] = {**bad["mandate"], "attempts": attempts}
    with pytest.raises(ValidationError):
        LLMDecisionOutput.model_validate(bad)


def test_an_unknown_strategy_or_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LLMDecisionOutput.model_validate(_decision(strategy="WANDER"))
    with pytest.raises(ValidationError):
        LLMDecisionOutput.model_validate(_decision(extra_field=1))


def test_policy_modules_import_nothing_from_the_engine_runner_apply_or_recipes() -> None:
    allowed_prefixes = (
        "orch.contract", "orch.policies", "orch.config", "__future__", "json", "hashlib", "re",
        "dataclasses", "typing", "collections", "pathlib", "subprocess", "os", "time", "math",
        "tempfile", "datetime", "enum", "pydantic", "yaml", "sys", "threading", "random", "contextlib",
        "itertools", "functools", "textwrap", "fcntl", "uuid", "shutil",
    )
    modules = sorted((ROOT / "orch/policies").glob("llm*.py"))
    assert {p.name for p in modules} >= {
        "llm.py", "llm_client.py", "llm_prompt.py", "llm_schemas.py", "llm_view.py",
    }
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                assert name.startswith(allowed_prefixes), f"{path.name} imports {name}"
                assert not name.startswith(("orch.engine", "orch.recipes", "orch.workers", "orch.episode")), (
                    f"{path.name} imports {name}"
                )


def test_the_attempt_rule_reads_naturally_for_one_attempt_and_for_several() -> None:
    template = load_template(TEMPLATE)
    common = dict(brief_text="b", cards_text="c", lessons=None, view_text="v", schema={"type": "object"})
    one = render_prompt(template, max_attempts=1, **common)
    three = render_prompt(template, max_attempts=3, **common)
    assert "Give exactly one attempt." in one and "between one and one" not in one
    assert "Give attempts between one and three." in three
    assert not PLACEHOLDER.search(one) and not PLACEHOLDER.search(three)


@pytest.mark.parametrize("path", sorted((ROOT / "configs/prompts").glob("orchestrator_v*.txt")), ids=lambda p: p.name)
def test_every_orchestrator_template_revision_is_neutral(path) -> None:
    raw = path.read_text(encoding="utf-8")
    assert set(PLACEHOLDER.findall(raw)) <= set(PLACEHOLDERS)
    text = _outside_placeholders(raw)
    assert not re.search(r"\d", text), "digits must come only from placeholders"
    lowered = text.lower()
    for word in WORKER_IDS:
        assert not re.search(rf"\b{word}\b", lowered), f"worker id {word!r} in {path.name}"
    for recipe in _recipe_ids():
        assert recipe.lower() not in lowered
    for name in (*DESIGN_NAMES, "latch"):
        assert name not in lowered, name
    # every revision keeps the contract sentences the validator and the view rely on
    for needed in ("Never repeat a combination", "satisfies every limit", "saved state"):
        assert needed in raw
