"""Tests of the round-2 ablation row V11 (decision of 2026-11-14 18:00, item 1): V11 is V9 with P on and nothing else changed."""
from __future__ import annotations

from pathlib import Path

import yaml
from orch.policies.v22_ablation import ABLATION_VARIANTS, definition_text_ablation
from orch.policies.v22_prompt import V21_VARIANT, V22Context, render_v22
from orch.policies.v22_situation import load_rows
from orch.policies.v22_switches import BASE_DEFINITION, VARIANTS, Switches, load_variant, sha256_file
from orch.policies.v22_tier3 import SITUATION_TABLE_LLM, T3_VARIANTS, situation_texts_llm
from test_v22_prompt import ASSETS, SIT_ROWS, TEXTS, situation, view

ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_HEADER = "Physical description of the saved states"
POOLED = "Past experience with the workers on other designs"
SITUATION_HEADER = "Past experience in the situation of this decision"


def test_v11_is_v9_with_p_on_and_the_other_registries_are_unchanged() -> None:
    assert ABLATION_VARIANTS == {"V11": Switches(E=True, P=True, H="full", R=False, C=False, S=False, S2=True, U=True)}
    assert {**T3_VARIANTS["V9"].as_json(), "P": True} == ABLATION_VARIANTS["V11"].as_json()
    assert set(T3_VARIANTS) == {"V9", "V10"} and set(VARIANTS) == {f"V{i}" for i in range(1, 9)}


def test_the_variant_file_is_the_generated_text_and_differs_from_v9_in_p_name_and_note_only() -> None:
    base = sha256_file(ROOT / BASE_DEFINITION)
    path = ROOT / "configs/policies/template_v2.2_r2/V11.yaml"
    assert path.read_text(encoding="utf-8") == definition_text_ablation("V11", base)
    name, sw, data = load_variant(path)
    assert (name, sw) == ("V11", ABLATION_VARIANTS["V11"])
    v9 = yaml.safe_load((ROOT / "configs/policies/template_v2.2_r2/V9.yaml").read_text(encoding="utf-8"))
    changed = {k for k in set(v9) | set(data) if v9.get(k) != data.get(k)}
    assert changed == {"name", "variant", "switches", "note"}
    assert {k for k in v9["switches"] if v9["switches"][k] != data["switches"][k]} == {"P"}
    assert data["situation_table"] == SITUATION_TABLE_LLM and data["seeds"] == [3, 4, 5]


def test_the_prompt_of_v11_is_v9s_plus_the_physical_paragraph_and_field() -> None:
    rows, _sha = load_rows(ROOT / SITUATION_TABLE_LLM)
    ctx = V22Context(situation_texts_llm(TEXTS, ROOT), [], candidate_seed=0, situations=SIT_ROWS)
    v = view(broken=True, calls=True)
    p9, schema9, _ = render_v22(situation(v), V21_VARIANT, ASSETS, T3_VARIANTS["V9"], ctx)
    p11, schema11, _ = render_v22(situation(v), V21_VARIANT, ASSETS, ABLATION_VARIANTS["V11"], ctx)
    assert PHYSICAL_HEADER not in p9 and "physical_facts" not in p9 and "physical_facts" not in schema9["properties"]
    assert PHYSICAL_HEADER in p11 and "physical_facts" in p11 and "physical_facts" in schema11["properties"]
    for prompt in (p9, p11):                                                  # everything else of V9 is in V11
        assert POOLED in prompt and SITUATION_HEADER in prompt
        assert "- Any worker can move the goal:" in prompt and "- Push:" not in prompt
    assert p9.count("LLM-orchestrated") == p11.count("LLM-orchestrated") == 1
    assert rows                                                                # the S'2 table loads (V9 and V11 read the same one)
