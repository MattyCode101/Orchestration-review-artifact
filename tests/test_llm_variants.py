"""The named, hashed LLM cells of the test campaign (decision of 2026-11-07, decisions (a) and (e))."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from orch.policies.llm_client import CachedLLM, LLMParams, ReplayCache, TelemetrySink
from orch.baselines.llm_variants import (
    VariantDropped, VariantSet, build_policy, load_variants,
)

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("F1-R1", "F1-R0", "E0-R1", "GA-R1")


def _project(tmp_path: Path) -> Path:
    """A small copy of the files a variant reads, so a test can change one of them."""

    for relative in ("configs/policies", "configs/prompts", "configs/profiles"):
        shutil.copytree(ROOT / relative, tmp_path / relative)
    return tmp_path


def test_the_four_cells_exist_with_the_templates_of_their_wording() -> None:
    variants = load_variants(ROOT)
    assert tuple(variants.names()) == NAMES
    assert variants.get("F1-R1").template_path == "configs/prompts/orchestrator_v2.txt"
    assert variants.get("F1-R0").template_path == "configs/prompts/orchestrator_v1.txt"
    assert variants.get("E0-R1").template_path == "configs/prompts/orchestrator_v2.txt"
    assert variants.get("GA-R1").template_path == "configs/prompts/general_agent_v2.txt"


def test_each_cell_states_its_configuration() -> None:
    variants = load_variants(ROOT)
    f1, e0, ga = variants.get("F1-R1"), variants.get("E0-R1"), variants.get("GA-R1")
    assert (f1.kind, f1.mode, f1.profiles_on, f1.lessons) == ("orchestrator", "fuzzy", True, False)
    assert (e0.kind, e0.mode, e0.profiles_on, e0.lessons) == ("orchestrator", "exact", False, False)
    assert ga.kind == "general" and ga.lessons is False


def test_f2_is_dropped_and_cannot_be_selected() -> None:
    variants = load_variants(ROOT)
    assert "F2" not in variants.names()
    with pytest.raises(VariantDropped, match="F2"):
        variants.get("F2")
    with pytest.raises(KeyError):
        variants.get("F9")


def test_a_cell_hash_changes_with_its_template_and_the_other_cells_do_not(tmp_path) -> None:
    project = _project(tmp_path)
    before = load_variants(project)
    template = project / "configs/prompts/orchestrator_v1.txt"
    template.write_text(template.read_text(encoding="utf-8") + "\nA changed sentence.\n", encoding="utf-8")
    after = load_variants(project)
    assert before.get("F1-R0").sha256 != after.get("F1-R0").sha256
    for name in ("F1-R1", "E0-R1", "GA-R1"):
        assert before.get(name).sha256 == after.get(name).sha256


def test_a_cell_hash_changes_with_the_profile_manifest_only_when_profiles_are_on(tmp_path) -> None:
    project = _project(tmp_path)
    before = load_variants(project)
    manifest = project / "configs/profiles/manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace("{", '{"note": "changed",', 1), encoding="utf-8")
    after = load_variants(project)
    assert before.get("F1-R1").sha256 != after.get("F1-R1").sha256
    assert before.get("E0-R1").sha256 == after.get("E0-R1").sha256


def test_the_model_is_not_part_of_a_cell_hash(tmp_path) -> None:
    project = _project(tmp_path)
    before = load_variants(project)
    path = project / "configs/policies/llm.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("gemini-3.1-flash-lite", "gemini-4-argon"), encoding="utf-8")
    after = load_variants(project)
    assert [before.get(n).sha256 for n in NAMES] == [after.get(n).sha256 for n in NAMES]


def test_a_variant_file_that_names_a_dropped_configuration_is_refused(tmp_path) -> None:
    project = _project(tmp_path)
    path = project / "configs/policies/llm_variants.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["variants"]["F2-R1"] = {"kind": "orchestrator", "configuration": "F2", "revision": "R1", "role": "x"}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="dropped"):
        load_variants(project)


def test_the_set_hash_covers_every_cell() -> None:
    variants = load_variants(ROOT)
    assert isinstance(variants, VariantSet) and len(variants.sha256) == 64
    assert variants.sha256 == load_variants(ROOT).sha256


def _llm(tmp_path: Path) -> CachedLLM:
    params = LLMParams(model="gemini-3.1-flash-lite", temperature=0.0, max_output_tokens=2048, seed=0)
    return CachedLLM(None, ReplayCache(tmp_path / "cache", mode="replay"), params)


def test_f1_r1_builds_the_policy_the_saturday_runs_were_made_with(tmp_path) -> None:
    variants = load_variants(ROOT)
    model_cfg = yaml.safe_load((ROOT / "configs/policies/llm.yaml").read_text(encoding="utf-8"))
    policy = build_policy(variants.get("F1-R1"), ROOT, model_cfg, _llm(tmp_path), TelemetrySink(tmp_path / "t.jsonl"),
                          ["area", "timing"])
    assert policy.version == "F1-fuzzy-c94f81ac34-8eadb4619c-nolessons-gemini-3.1-flash-lite"
    r0 = build_policy(variants.get("F1-R0"), ROOT, model_cfg, _llm(tmp_path), TelemetrySink(tmp_path / "t.jsonl"),
                      ["area", "timing"])
    assert r0.version == "F1-fuzzy-9b60f0d06b-8eadb4619c-nolessons-gemini-3.1-flash-lite"


def test_the_general_agent_cell_uses_the_r1_general_template(tmp_path) -> None:
    variants = load_variants(ROOT)
    model_cfg = yaml.safe_load((ROOT / "configs/policies/llm.yaml").read_text(encoding="utf-8"))
    agent = build_policy(variants.get("GA-R1"), ROOT, model_cfg, _llm(tmp_path), TelemetrySink(tmp_path / "t.jsonl"),
                         ["area", "timing"])
    assert agent.version.startswith("GA-fuzzy-f4a6ad9132-")
