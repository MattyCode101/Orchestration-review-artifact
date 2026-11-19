"""General-agent baseline: schema, validation, mandates, a whole fake-backend episode and parity
with the orchestrator (acceptance test 10)."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import llm_support as ls
from orch.baselines.direct_recipe_worker import DirectRecipeWorker
from orch.baselines.general_agent import (
    GeneralAgentPolicy, GeneralDecisionOutput, RecipeInfo, general_json_schema, parse_general_decision,
)
from orch.contract.brief_text import load_frozen_brief, render_brief_text
from orch.engine.journal import Journal
from orch.episode import run_episode
from orch.policies.llm import LLMPolicy, LLMPolicyConfig
from orch.policies.llm_client import CachedLLM, LLMParams, ReplayCache, TelemetrySink
from orch.policies.llm_prompt import PLACEHOLDERS, load_template
from orch.policies.llm_schemas import DecisionFormatError

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/prompts/general_agent_v1.txt"
NOISE = {ls.MetricName.WNS_NS: 0.05, ls.MetricName.PACKED_LUTS: 5.0}
RECIPES = [
    RecipeInfo(r, w, f"{r}: targets {t}; keeps the placement. Observed: {t} within_noise usually; no change often; invalid rarely.")
    for r, w, t in (("t1", "timing", "wns"), ("t2", "timing", "wns"), ("t3", "timing", "wns"),
                    ("a1", "area", "packed_luts"), ("a2", "area", "packed_luts"), ("a3", "area", "packed_luts"))
]


def ga_decision(recipe: str = "t1", base: str = "S0", *, strategy: str = "PUSH", reason: str = "push the goal",
                action: str = "RUN_RECIPE") -> str:
    return json.dumps({
        "state_diagnosis": "diagnosis", "strategy": "STOP" if action == "STOP" else strategy,
        "conflict_check": "no gains to undo", "rationale_summary": reason, "action": action,
        "recipe_action": None if action == "STOP" else {"recipe_id": recipe, "base_state": base},
    })


def make_agent(tmp_path, *texts: str, cache_mode: str = "record", cache_dir: str = "ga_cache", telemetry: str = "ga.jsonl"):
    transport = ls.ScriptedTransport(*texts)
    llm = CachedLLM(None if cache_mode == "replay" else transport, ReplayCache(tmp_path / cache_dir, mode=cache_mode),
                    LLMParams(model="scripted-model"))
    config = LLMPolicyConfig(name="GA", mode="fuzzy", profiles_on=True, template=load_template(TEMPLATE),
                             profile_texts={}, profile_sha256={"recipes": "3" * 64}, lessons=None, max_repairs=2)
    return GeneralAgentPolicy(config, llm, TelemetrySink(tmp_path / telemetry), RECIPES), transport


def _view():
    s0, s1 = ls.state(10, -0.347, 1337), ls.state(11, -0.147, 1352, brief_valid=False)
    calls = (ls.call(1, "timing", 10, 11, attempts=1, reason="push the goal", recipe="t1"),)
    return ls.view(visited=[s0, s1], calls=calls, steps=(ls.step("timing", 10, 11, attempts=1),), budget_left=5,
                   current=s1, best=s0)


# ------------------------------------------------------------------ template and schema
def test_the_general_agent_template_is_neutral_too() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    assert set(re.findall(r"\{\{([A-Z_]+)\}\}", template)) <= set(PLACEHOLDERS)
    text = re.sub(r"\{\{[A-Z_]+\}\}", "", template)
    assert not re.search(r"\d", text)
    lowered = text.lower()
    for word in ("timing", "area", "power"):
        assert not re.search(rf"\b{word}\b", lowered)
    registry = yaml.safe_load((ROOT / "recipes/registry.yaml").read_text(encoding="utf-8"))
    for recipe in registry["recipes"]:
        assert recipe["id"].lower() not in lowered
    for fragment in ("vexriscv", "logicnets", "rosetta", "finn", "latch", "corescore"):
        assert fragment not in lowered


def test_the_schema_lists_this_episodes_recipes_and_states() -> None:
    schema = general_json_schema(["t1", "a1"], ["S0", "S1"])
    action = schema["properties"]["recipe_action"]["anyOf"][1]
    assert action["properties"]["recipe_id"]["enum"] == ["t1", "a1"] and action["properties"]["base_state"]["enum"] == ["S0", "S1"]
    assert schema["properties"]["action"]["enum"] == ["RUN_RECIPE", "STOP"]
    assert set(schema["properties"]) == set(GeneralDecisionOutput.model_fields)


def test_an_action_is_required_exactly_when_the_agent_runs_a_recipe() -> None:
    assert parse_general_decision(ga_decision()).recipe_action.recipe_id == "t1"
    assert parse_general_decision(ga_decision(action="STOP")).recipe_action is None
    for bad in (json.loads(ga_decision()) | {"recipe_action": None}, json.loads(ga_decision(action="STOP")) | {"action": "RUN_RECIPE"}):
        with pytest.raises((DecisionFormatError, ValidationError)):
            parse_general_decision(json.dumps(bad))
    with pytest.raises(DecisionFormatError):
        parse_general_decision("nope")


# ------------------------------------------------------------------ mapping and validation
def test_a_valid_action_becomes_a_one_attempt_mandate_to_the_recipes_own_worker(tmp_path) -> None:
    agent, transport = make_agent(tmp_path, ga_decision("a2", "S1", strategy="REPAIR", reason="repair the broken limit"))
    mandate = agent.decide(_view())
    assert mandate.worker_id == "area" and mandate.base_state_id == ls.sid(11)
    assert mandate.settings.max_runs == 1 and mandate.settings.preserve_placement is False and mandate.guards == ()
    ref = json.loads(mandate.context_ref)
    assert ref["x"] == "a2" and ref["s"] == "REPAIR" and ref["r"] == "repair the broken limit"
    assert "a2: targets packed_luts" in transport.prompts[0] and "Recipes" in transport.prompts[0]


@pytest.mark.parametrize(("reply", "expected"), [
    (ga_decision("zzz", "S0"), "recipe_id 'zzz' is not one of the listed recipes"),
    (ga_decision("t2", "S9"), "base_state 'S9' is not one of the listed states"),
    (ga_decision("t1", "S0"), "already run on that state"),
])
def test_a_rejected_action_is_repaired_with_the_exact_error(tmp_path, reply, expected) -> None:
    agent, transport = make_agent(tmp_path, reply, ga_decision("t2", "S0"))
    mandate = agent.decide(_view())
    assert mandate is not None and json.loads(mandate.context_ref)["x"] == "t2"
    assert expected in transport.prompts[1] and transport.prompts[1].startswith(transport.prompts[0])


def test_the_same_recipe_may_run_on_a_different_state_and_stop_is_honoured(tmp_path) -> None:
    agent, _ = make_agent(tmp_path, ga_decision("t1", "S1"))
    assert agent.decide(_view()).base_state_id == ls.sid(11)
    stopper, _ = make_agent(tmp_path, ga_decision(action="STOP", reason="nothing left"), cache_dir="stop", telemetry="stop.jsonl")
    assert stopper.decide(_view()) is None


def test_the_catalog_is_limited_to_the_workers_of_the_episode(tmp_path) -> None:
    only_timing = _view().model_copy(update={"worker_cards": (ls.cards()[0],)})
    agent, transport = make_agent(tmp_path, ga_decision("t2", "S0"))
    agent.decide(only_timing)
    assert "t1: targets wns" in transport.prompts[0] and "a1: targets packed_luts" not in transport.prompts[0]


def test_the_general_agent_prompt_is_anonymous_but_names_recipes(tmp_path) -> None:
    agent, transport = make_agent(tmp_path, ga_decision("t2", "S0"))
    view = _view()
    agent.decide(view)
    prompt = transport.prompts[0]
    assert ls.DESIGN_SECRET not in prompt and view.episode_id not in prompt and not re.search(r"[0-9a-f]{12,}", prompt)
    assert ls.sid(10) not in prompt and "1337" not in prompt and "mandate_" not in prompt


# ------------------------------------------------------------------ the one-recipe worker
def test_the_direct_worker_plans_only_the_requested_untried_recipe() -> None:
    from orch.contract.models import Mandate, WorkerSettings

    worker = DirectRecipeWorker(ls.cards()[0], ("t1", "t2", "t3"))

    class State:
        id = ls.sid(10)
        stage = "routed"

    def mandate(recipe):
        ref = json.dumps({"x": recipe}) if recipe else "not json"
        return Mandate(id="m", worker_id="timing", base_state_id=ls.sid(10), settings=WorkerSettings(max_runs=1),
                       brief_hash="h", context_ref=ref)

    assert worker.plan(State, None, mandate("t2"), ()) == ("t2",)
    assert worker.plan(State, None, mandate("a1"), ()) == ()           # not this worker's recipe
    assert worker.plan(State, None, mandate(None), ()) == ()           # no recipe named


# ------------------------------------------------------------------ whole episode on the fake backend
def test_a_recipe_level_episode_runs_through_the_unchanged_engine(tmp_path) -> None:
    engine, registry, _, backend, start = ls.episode_setup(tmp_path)
    workers = {c.id: DirectRecipeWorker(c, ids) for c, ids in zip(ls.cards(), (("t1", "t2", "t3"), ("a1", "a2", "a3")))}
    agent, _ = make_agent(tmp_path, ga_decision("t1", "S0"), ga_decision("a1", "S1", strategy="REPAIR"),
                          ga_decision("t2", "S0", strategy="BACKTRACK"), ga_decision(action="STOP", reason="done"))
    result = run_episode(engine=engine, registry=registry, workers=workers, policy=agent, brief=load_frozen_brief("T0"),
                         start_state=start, episode_id="ga_unit", noise=NOISE)
    events = Journal(engine.store.root).read("ga_unit")
    mandates = [e["mandate"] for e in events if e["type"] == "mandate"]
    steps = [e["step"] for e in events if e["type"] == "step"]
    assert [m["worker_id"] for m in mandates] == ["timing", "area", "timing"]
    assert [json.loads(m["context_ref"])["x"] for m in mandates] == ["t1", "a1", "t2"]
    assert backend.calls == ["t1", "a1", "t2"] and result.recipe_runs == 3
    assert mandates[1]["base_state_id"] == steps[0]["output_state_id"]       # repaired from the limit-breaking state
    assert mandates[2]["base_state_id"] == start.id                           # branched back to the start
    assert result.stop_reason == "policy_stop" and result.final_verdict.brief_valid
    assert result.final_state.id == steps[1]["output_state_id"]               # best brief-valid state visited


# ------------------------------------------------------------------ parity with the orchestrator (acceptance test 10)
def test_the_general_agent_has_parity_with_the_orchestrator(tmp_path) -> None:
    view = _view()
    orchestrator, o_transport, _ = ls.make_policy(tmp_path, ls.decision("area", "S1", preserve=True, attempts=1), cache_dir="o")
    agent, g_transport = make_agent(tmp_path, ga_decision("t2", "S0"), cache_dir="g")
    orchestrator.decide(view)
    agent.decide(view)
    brief_text = render_brief_text(view.brief)
    assert brief_text in o_transport.prompts[0] and brief_text in g_transport.prompts[0]      # same brief text
    keep = ("budget:", "best_valid_state:", "current_state:", "  brief_valid:", "  limits:", "  vs_start:", "  goal_vs_best:")
    pick = lambda prompt: [x for x in prompt.splitlines() if x.startswith(keep)]             # noqa: E731
    assert pick(o_transport.prompts[0]) == pick(g_transport.prompts[0])                      # same budget and fuzzy view
    assert orchestrator.llm.params == agent.llm.params                                       # same model parameters
    assert GeneralAgentPolicy.decide is LLMPolicy.decide                                     # one decision loop
    assert GeneralAgentPolicy.recipe_level is True and LLMPolicy.recipe_level is False
    assert orchestrator.last_record["temperature"] == agent.last_record["temperature"] == 0.0


def test_both_agents_are_built_from_the_same_model_configuration() -> None:
    from scripts import saturday_llm as driver

    orchestrator = driver.make_llm(driver.Cell("F1", "d", "T0", "r0", "record", None, "orchestrator", "ab"))
    general = driver.make_llm(driver.Cell("GA", "d", "T0", "r0", "record", None, "general", "ab"))
    assert orchestrator.params == general.params and orchestrator.params.model == yaml.safe_load(
        (ROOT / "configs/policies/llm.yaml").read_text())["model"]


def test_the_general_agent_module_imports_no_engine_runner_recipe_or_worker_code() -> None:
    path = ROOT / "orch/baselines/general_agent.py"
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
        for name in names:
            assert not name.startswith(("orch.engine", "orch.recipes", "orch.workers", "orch.episode")), name


R1_PHRASE = "nothing has been gained yet there is nothing to protect"


def test_the_general_agent_template_received_the_same_r1_clause_as_the_orchestrator_template() -> None:
    """R1 added one placement clause to the orchestrator wording; the general agent's R1 wording must match it."""

    orchestrator_r0 = (ROOT / "configs/prompts/orchestrator_v1.txt").read_text(encoding="utf-8")
    orchestrator_r1 = (ROOT / "configs/prompts/orchestrator_v2.txt").read_text(encoding="utf-8")
    agent_r0 = (ROOT / "configs/prompts/general_agent_v1.txt").read_text(encoding="utf-8")
    agent_r1 = (ROOT / "configs/prompts/general_agent_v2.txt").read_text(encoding="utf-8")
    assert R1_PHRASE not in orchestrator_r0 and R1_PHRASE not in agent_r0
    assert R1_PHRASE in orchestrator_r1 and R1_PHRASE in agent_r1


def test_general_agent_r1_differs_from_r0_by_that_one_sentence_only() -> None:
    r0 = (ROOT / "configs/prompts/general_agent_v1.txt").read_text(encoding="utf-8")
    r1 = (ROOT / "configs/prompts/general_agent_v2.txt").read_text(encoding="utf-8")
    sentence = " On a state where nothing has been gained yet there is nothing to protect."
    anchor = "check this before choosing it."
    assert r0.count(anchor) == 1
    assert r1 == r0.replace(anchor, anchor + sentence)


@pytest.mark.parametrize("path", sorted((ROOT / "configs/prompts").glob("general_agent_v*.txt")), ids=lambda p: p.name)
def test_every_general_agent_template_revision_is_neutral(path) -> None:
    raw = path.read_text(encoding="utf-8")
    assert set(re.findall(r"\{\{([A-Z_]+)\}\}", raw)) <= set(PLACEHOLDERS)
    lowered = re.sub(r"\{\{[A-Z_]+\}\}", "", raw).lower()
    assert not re.search(r"\d", lowered)
    for word in ("timing", "area", "power"):
        assert not re.search(rf"\b{word}\b", lowered)
    registry = yaml.safe_load((ROOT / "recipes/registry.yaml").read_text(encoding="utf-8"))
    for recipe in registry["recipes"]:
        assert recipe["id"].lower() not in lowered
    for fragment in ("vexriscv", "logicnets", "rosetta", "finn", "latch", "corescore"):
        assert fragment not in lowered
    for needed in ("Never repeat a recipe", "satisfies every limit", "only the final pick must be valid"):
        assert needed in raw or needed.lower() in lowered
