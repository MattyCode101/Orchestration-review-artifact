"""Template v2.2, round-2 ablation row V11 (decision of 2026-11-14 18:00, item 1; new file).

V11 is V9 with P on: the physical paragraph and the physical_facts reply field are restored; everything else is V9 (the pooled
experience lines, the S'2 situation lines from LLM-orchestrated calls only with their one-line population statement, the neutral
worker line U). It is an ablation row for TEST, not a candidate: it asks what the physical description adds to V9.

  V9   v2.1 without P, with U and S'2        (orch/policies/v22_tier3.py, frozen with the round-2 LLM package)
  V11  V9 with P on                          (this file)

Nothing launched or frozen is edited: orch/policies/v22_switches.py and v22_tier3.py are as they were. The S'2 table, its population
text and the rendering are those of V9 (v22_tier3.SITUATION_TABLE_LLM, POPULATION_TEXTS, situation_texts_llm); a driver that builds V9
builds V11 the same way, with this registry added to the one it checks the variant file against.
"""
from __future__ import annotations

import yaml

from orch.policies.v22_switches import BASE_DEFINITION, SCHEMA, Switches
from orch.policies.v22_tier3 import POPULATION_TEXTS, SITUATION_RULE_LLM, SITUATION_TABLE_LLM, T3_VARIANTS

ABLATION_VARIANTS: dict[str, Switches] = {
    "V11": Switches(P=True, S2=True, U=True),          # V9 with P on (E on, H full; R, C, S off)
}
ABLATION_NOTES = {
    "V11": "V9 with P on (the physical paragraph restored); an ablation row for TEST, not a candidate",
}
if ABLATION_VARIANTS["V11"] != Switches(**{**T3_VARIANTS["V9"].as_json(), "P": True}):
    raise ImportError("V11 must be V9 with P on and nothing else changed")


def definition_text_ablation(name: str, base_sha256: str) -> str:
    """The text of configs/policies/template_v2.2_r2/<name>.yaml for V11 (generated, so a test can compare the stored file).
    It has the fields and layout of V9.yaml; only the header, name, variant, P and the note differ."""

    sw = ABLATION_VARIANTS[name]
    body = {
        "name": f"template_v2.2_{name}",
        "variant": name,
        "schema": SCHEMA,
        "base": "template_v2.1",
        "base_definition": BASE_DEFINITION,
        "base_definition_sha256": base_sha256,
        "switches": sw.as_json(),
        "switch_texts": "configs/prompts/template_v22/switch_texts.yaml",
        "situation_population_text": POPULATION_TEXTS,
        "situation_table": SITUATION_TABLE_LLM,
        "situation_rule": SITUATION_RULE_LLM,
        "model": "gemini-3.1-flash-lite", "temperature": 0.3, "seeds": [3, 4, 5],
        "noise": "configs/noise_round2.yaml",
        "loop": "orch.episode_r2.run_episode_r2 with ROUND2 (both options on)",
        "wording_schema_truncation": "as template v2.1 (orch.policies.round2_harness TEXT_LIMITS); no power-specific wording",
        "note": ABLATION_NOTES[name],
    }
    header = (f"# Template v2.2 variant {name} (round-2 ablation row): {ABLATION_NOTES[name]}.\n"
              "# Independent switches on top of the frozen template v2.1 (orch/policies/v22_*.py). Nothing frozen is edited.\n")
    return header + yaml.safe_dump(body, sort_keys=False, width=120)


__all__ = ["ABLATION_NOTES", "ABLATION_VARIANTS", "definition_text_ablation"]
