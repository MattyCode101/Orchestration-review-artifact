"""Template v2.2, tier 3 (decisions of 2026-11-14 11:30): variants V9 and V10 and the S'2 situation lines (new file).

S'2 is switch S2 (orch/policies/v22_situation.py) with one change of evidence: the rows are the calls of LLM-orchestrated
episodes only (D6-v2 F1-R1 and v2.1, grid V2 to V8; briefs T0 and A0), one row per call, and the prompt says so in one line
(configs/prompts/template_v22/situation_llm_texts.yaml). The situations, the rendering, "n calls from k designs", the
frequency words, the minimum of three other designs and the leave-one-design-out rule are the code of S2, unchanged.

Nothing launched is edited: the registry of orch/policies/v22_switches.py stays as it was; V9 and V10 are registered here.
  V9   v2.1 without P, with U and S'2    (E stays on: the pooled lines and the situation lines are both in the prompt)
  V10  V9 with E off                      (the situation lines take the place of the pooled lines)
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from orch.policies.v22_situation import rows_from_diagnosis_episode
from orch.policies.v22_switches import BASE_DEFINITION, SCHEMA, Switches

SITUATION_TABLE_LLM = "configs/experience/situations_v22_llm.json"
POPULATION_TEXTS = "configs/prompts/template_v22/situation_llm_texts.yaml"
T3_VARIANTS: dict[str, Switches] = {
    "V9": Switches(P=False, S2=True, U=True),                  # v2.1 without P, with U and S'2
    "V10": Switches(E=False, P=False, S2=True, U=True),        # V9 with E off
}
T3_NOTES = {
    "V9": "v2.1 without P, with U and S'2 (situation lines from LLM-orchestrated calls only, beside the pooled lines)",
    "V10": "V9 with E off (the S'2 situation lines instead of the pooled experience lines)",
}
SITUATION_RULE_LLM = ("S'2: rows are the calls of LLM-orchestrated episodes only (D6-v2 F1-R1 and v2.1, grid V2 to V8; T0 and "
                      "A0), one row per call, no de-duplication; leave one design out; a line needs at least 3 other designs, "
                      "else 'no evidence for this situation'; no design is named; the prompt states the population in one line")
TIER3_SEEDS = {"t1": 3, "t2": 4, "t3": 5}                      # D6-v2 and the grid used 0, 1, 2
LLM_VARIANTS_D6 = ("V0", "V1")                                  # the D6-v2 LLM rows, labelled as in the grid scorer
LLM_VARIANTS_GRID = ("V2", "V3", "V4", "V5", "V6", "V7", "V8")


def llm_rows(episodes: Mapping[str, Mapping[str, Any]], *, source: str, allowed: Sequence[str]) -> list[dict[str, Any]]:
    """One row per call of the given parsed episodes (the scorer's episode dicts), in episode id then call order. Episodes of a
    variant outside `allowed` are skipped, so a deterministic policy can never enter the table."""

    rows: list[dict[str, Any]] = []
    for eid in sorted(episodes):
        ep = episodes[eid]
        if ep["variant"] not in allowed:
            continue
        for row in rows_from_diagnosis_episode(ep):
            rows.append({**row, "source": source, "episode": eid, "variant": ep["variant"]})
    return rows


def situation_texts_llm(texts: Mapping[str, Any], project: Path | str) -> dict[str, Any]:
    """The switch texts with the population line added to the situation section (one line after its header)."""

    pop = yaml.safe_load((Path(project) / POPULATION_TEXTS).read_text(encoding="utf-8"))["population"]
    situation = dict(texts["situation"])
    situation["intro"] = pop + "\n" + situation["intro"]
    return {**texts, "situation": situation}


def definition_text_t3(name: str, base_sha256: str) -> str:
    """The text of configs/policies/template_v2.2_r2/<name>.yaml for V9 and V10 (generated, so a test can compare the file)."""

    sw = T3_VARIANTS[name]
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
        "note": T3_NOTES[name],
    }
    header = (f"# Template v2.2 variant {name} (tier 3): {T3_NOTES[name]}.\n"
              "# Independent switches on top of the frozen template v2.1 (orch/policies/v22_*.py). Nothing frozen is edited.\n")
    return header + yaml.safe_dump(body, sort_keys=False, width=120)
