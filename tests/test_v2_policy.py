"""V2Policy: harness rules stay in code (legality, repeats, guards) whatever the template variant says."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import llm_support as ls
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import MetricName
from orch.policies.llm import LLMPolicyConfig
from orch.policies.llm_client import CachedLLM, LLMParams, ReplayCache, TelemetrySink
from orch.policies.r2_experience import Call, build_table
from orch.policies.v2_experience import CallV2
from orch.policies.v2_policy import DescriptionUnavailableError, V2Policy, exhausted_combinations
from orch.policies.v2_prompt import Variant, load_assets

ROOT = Path(__file__).resolve().parents[1]
FULL = Variant("P1234", ("P1", "P2", "P3", "P4"))
BASE = Variant("R1")


def v2_reply(worker="timing", base="S0", placement="must not change", attempts=1, respect=True, action="INVOKE_WORKER"):
    mandate = None if action == "STOP" else {"worker_id": worker, "base_state": base, "placement": placement,
                                             "attempts": attempts, "respect_brief_limits": respect}
    forecast = None if action == "STOP" else {"goal": "small gain", "packed_luts": "within noise"}
    return json.dumps({"reasoning": "r", "evidence": [], "options": [{"option": "a", "expected_effect": "b"}] * 2,
                       "rationale": "because", "action": action, "mandate": mandate, "forecast": forecast})


def make(tmp_path, variant, *texts, describe=None):
    transport = ls.ScriptedTransport(*texts)
    llm = CachedLLM(transport, ReplayCache(tmp_path / "cache", mode="record"), LLMParams(model="scripted"))
    assets = load_assets(ROOT)
    config = LLMPolicyConfig(name="v2", mode="fuzzy", profiles_on=False, template=assets.templates["base"])
    table = build_table([Call(d, w, s, True, {"wns": "within_noise"}) for d in ("a", "b") for w in ("timing", "area")
                         for s in ("preserve_placement_true", "preserve_placement_false")], "x")
    calls_v2 = [CallV2(d, "T0", "timing", "preserve_placement_true", "selected", {"wns_ns": 0.0, "packed_luts": 10.0},
                       {"wns_ns": 0.2, "packed_luts": 10.0}, {"wns_ns": 0.1, "packed_luts": 1.0}) for d in "abcd"]
    policy = V2Policy(config, llm, TelemetrySink(tmp_path / "t.jsonl"), variant=variant, assets=assets, design="z",
                      describe=describe or (lambda sid: {}), r2_experience=table, calls_v2=calls_v2)
    return policy, transport


def view(calls=(), steps=(), budget_left=6, visited=None):
    visited = visited or [ls.state(10, -0.3, 1000), ls.state(11, -0.2, 1001, brief_valid=False)]
    return ls.view(visited=visited, calls=calls, steps=steps, budget_left=budget_left, brief=load_frozen_brief("T0"),
                   noise={MetricName.WNS_NS: 0.1, MetricName.PACKED_LUTS: 10.0})


def test_full_variant_builds_a_guarded_mandate_with_the_placement_word(tmp_path) -> None:
    policy, transport = make(tmp_path, FULL, v2_reply(placement="must not change", respect=True))
    mandate = policy.decide(view())
    assert mandate is not None and mandate.settings.preserve_placement is True
    assert mandate.guards and mandate.guards[0].metric == MetricName.PACKED_LUTS
    assert "{{" not in transport.prompts[0] and "Reply in the structured output format" in transport.prompts[0]
    assert policy.last_record["forecast"] == {"goal": "small gain", "packed_luts": "within noise"}


def test_with_p1_a_repeat_is_legal_until_the_view_shows_no_options_left(tmp_path) -> None:
    tried = (ls.call(1, "timing", 10, 11, preserve=True, attempts=1, runs=1),)
    policy, _ = make(tmp_path, FULL, v2_reply(placement="must not change"))
    assert policy.decide(view(calls=tried, steps=(ls.step("timing", 10, 11, preserve=True, attempts=1),),
                              budget_left=5)) is not None
    exhausted = (ls.call(1, "timing", 10, 11, preserve=True, attempts=2, runs=1),)
    assert exhausted_combinations(view(calls=exhausted)) == {("timing", ls.sid(10), True)}
    policy2, transport2 = make(tmp_path / "b", FULL, v2_reply(placement="must not change"),
                               v2_reply(worker="area", placement="may change"))
    mandate = policy2.decide(view(calls=exhausted, steps=(ls.step("timing", 10, 11, preserve=True),), budget_left=5))
    assert mandate is not None and mandate.worker_id == "area" and len(transport2.prompts) == 2
    assert "no options left" in transport2.prompts[1]


def test_without_p1_the_round1_never_repeat_rule_stays(tmp_path) -> None:
    tried = (ls.call(1, "timing", 10, 11, preserve=True, attempts=1, runs=1),)
    reply = ls.decision("timing", "S0", preserve=True, attempts=1)
    policy, transport = make(tmp_path, BASE, reply, reply, reply)
    assert policy.decide(view(calls=tried, steps=(ls.step("timing", 10, 11, preserve=True, attempts=1),),
                              budget_left=5)) is None
    # three turns; the third repeats the second prompt exactly, so the replay cache answers it
    assert len(policy.last_record["calls"]) == 3 and policy.last_record["fallback_triggered"] is True
    assert len(transport.prompts) == 2 and policy.last_record["calls"][2]["cache_hit"] is True


def test_attempts_beyond_the_budget_are_refused_in_code(tmp_path) -> None:
    policy, transport = make(tmp_path, FULL, v2_reply(attempts=3), v2_reply(attempts=1))
    mandate = policy.decide(view(budget_left=2))
    assert mandate is not None and mandate.settings.max_runs == 1 and len(transport.prompts) == 2


def test_a_missing_description_is_an_infrastructure_failure(tmp_path) -> None:
    def describe(sid):
        raise KeyError(sid)

    policy, _ = make(tmp_path, FULL, v2_reply(), describe=describe)
    with pytest.raises(DescriptionUnavailableError):
        policy.decide(view())
