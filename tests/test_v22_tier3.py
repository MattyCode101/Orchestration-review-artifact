"""Tests of the tier 3 additions (decisions of 2026-11-14 11:30): variants V9 and V10, the S'2 table and its population line, the plan."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
from orch.policies.v22_prompt import V21_VARIANT, V22Context, render_v22
from orch.policies.v22_situation import load_rows
from orch.policies.v22_switches import BASE_DEFINITION, VARIANTS, Switches, load_variant, sha256_file
from orch.policies.v22_tier3 import (
    POPULATION_TEXTS, SITUATION_TABLE_LLM, T3_VARIANTS, definition_text_t3, llm_rows, situation_texts_llm,
)
from test_v22_prompt import ASSETS, SIT_ROWS, TEXTS, situation, view

ROOT = Path(__file__).resolve().parents[1]
DESIGNS = {"finn_radioml", "logicnets_jscl", "rosetta_3d-rendering", "rosetta_spam-filter", "vexriscv_re-place",
           "t2_cordic_s", "vtr_mcml_v2", "amd_mini-isp"}
POP = yaml.safe_load((ROOT / POPULATION_TEXTS).read_text(encoding="utf-8"))["population"]
POOLED = "Past experience with the workers on other designs"
SITUATION_HEADER = "Past experience in the situation of this decision"


def test_v9_and_v10_are_the_declared_switch_sets_and_the_grid_registry_is_unchanged() -> None:
    assert T3_VARIANTS["V9"] == Switches(E=True, P=False, S2=True, U=True)
    assert T3_VARIANTS["V10"] == Switches(E=False, P=False, S2=True, U=True)
    assert set(T3_VARIANTS) == {"V9", "V10"} and set(VARIANTS) == {f"V{i}" for i in range(1, 9)}


def test_variant_files_are_the_generated_text_and_load() -> None:
    base = sha256_file(ROOT / BASE_DEFINITION)
    for name, sw in T3_VARIANTS.items():
        path = ROOT / "configs/policies/template_v2.2_r2" / f"{name}.yaml"
        assert path.read_text(encoding="utf-8") == definition_text_t3(name, base)
        got_name, got_sw, data = load_variant(path)
        assert (got_name, got_sw) == (name, sw)
        assert data["situation_table"] == SITUATION_TABLE_LLM and data["seeds"] == [3, 4, 5]


def test_the_population_line_opens_the_situation_section_of_v9_and_v10_only() -> None:
    texts2 = situation_texts_llm(TEXTS, ROOT)
    assert texts2["situation"]["intro"].splitlines()[0] == POP
    assert "LLM-orchestrated" not in TEXTS["situation"]["intro"]                    # the old texts are not changed
    ctx2 = V22Context(texts2, [], candidate_seed=0, situations=SIT_ROWS)
    ctx1 = V22Context(TEXTS, [], candidate_seed=0, situations=SIT_ROWS)
    v = view(broken=True, calls=True)
    p9, *_ = render_v22(situation(v), V21_VARIANT, ASSETS, T3_VARIANTS["V9"], ctx2)
    p10, *_ = render_v22(situation(v), V21_VARIANT, ASSETS, T3_VARIANTS["V10"], ctx2)
    p6, *_ = render_v22(situation(v), V21_VARIANT, ASSETS, VARIANTS["V6"], ctx1)
    head = SITUATION_HEADER + "\n" + POP + "\n"
    assert head in p9 and head in p10 and p9.count(POP) == 1 and p10.count(POP) == 1
    assert POP not in p6 and SITUATION_HEADER in p6
    assert POOLED in p9 and POOLED not in p10                                      # E on beside the lines, E off instead of them
    for prompt, schema in ((p9, None), (p10, None)):
        assert "Physical description of the saved states" not in prompt and "physical_facts" not in prompt
        assert "- Any worker can move the goal:" in prompt and "- Push:" not in prompt


def test_the_s_prime2_table_holds_llm_calls_only_and_matches_the_launch_record() -> None:
    rows, sha = load_rows(ROOT / SITUATION_TABLE_LLM)
    assert rows and {r["source"] for r in rows} == {"d6v2", "grid"}
    assert {r["variant"] for r in rows} <= {f"V{i}" for i in range(9)}
    assert {r["brief"] for r in rows} <= {"T0", "A0"} and {r["design"] for r in rows} <= DESIGNS
    assert sha != sha256_file(ROOT / "configs/experience/situations_v22.json")
    record = (ROOT / "notes/template_v22_tier3.sha256").read_text(encoding="utf-8")
    assert f"{sha}  configs/experience/situations_v22_llm.json" in record


def _episode(variant: str) -> dict:
    return {"variant": variant, "brief": "A0", "group": "TRAINING", "design": "d1",
            "states": {"s0": {"gain_b": 0.0, "margin_b": 0.0, "valid": True, "call": -1}},
            "calls": [{"base": "s0", "worker": "timing", "preserve": True,
                       "steps": [{"out": "s1", "gain_b": 1.5, "margin_b": 0.2, "valid": True, "phys": True}]}]}


def test_llm_rows_counts_every_call_and_skips_other_policies() -> None:
    eps = {"e1": _episode("V2"), "e2": _episode("V2"), "e3": _episode("RULE"), "e4": _episode("FIXED")}
    rows = llm_rows(eps, source="grid", allowed=("V2",))
    assert len(rows) == 2 and {r["episode"] for r in rows} == {"e1", "e2"}          # two identical calls are two rows
    assert all(r["source"] == "grid" and r["variant"] == "V2" and r["situation"] == "start" for r in rows)


def test_the_plan_and_the_work_folder_in_a_fresh_process() -> None:
    code = ("import sys, json; sys.path.insert(0, %r); import v22_t3_plan as p; t = p.make_tasks(); "
            "print(json.dumps({'n': len(t), 'seeds': sorted({x['seed'] for x in t}), 'variants': sorted({x['approach'] for x in t}), "
            "'designs': sorted({x['design'] for x in t}), 'work': str(p.dc.WORK), 'caps': p.dc.CAPS, "
            "'per_cell': sorted({(x['design'], x['brief']) for x in t}).__len__(), 'first_tag': t[0]['tag'], 'last_tag': t[-1]['tag']}))"
            ) % str(ROOT / "scripts/v22_tier3_2026-11-14")
    res = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, cwd=ROOT, check=True)
    d = json.loads(res.stdout.strip().splitlines()[-1])
    assert d["n"] == 192 and d["seeds"] == [3, 4, 5] and d["variants"] == ["V0", "V1", "V10", "V9"] and d["per_cell"] == 16
    assert set(d["designs"]) == DESIGNS and d["work"].endswith("data/v22_2026-11-14/tier3")
    assert d["caps"] == {"new_real_runs": 150, "job_hours": 60, "tokens": 30000000, "usd": 3.0}
    assert d["first_tag"] == "t1" and d["last_tag"] == "t3"
