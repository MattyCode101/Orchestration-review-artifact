"""R2Policy: round-1 behaviour with every feature off, evidence sections when on, infrastructure failure, logging."""
from __future__ import annotations

from pathlib import Path

import pytest

import llm_support as ls
from orch.policies.llm import LLMPolicy, LLMPolicyConfig
from orch.policies.llm_client import CachedLLM, LLMParams, LLMUnavailableError, ReplayCache, TelemetrySink
from orch.policies.llm_prompt import load_template
from orch.policies.r2_experience import Call, SCOPES, build_table
from orch.policies.r2_policy import DescriptionUnavailableError, R2Policy
from orch.policies.r2_view import Features

ROOT = Path(__file__).resolve().parents[1]
R1, R2 = ROOT / "configs/prompts/orchestrator_v2.txt", ROOT / "configs/prompts/orchestrator2_v1.txt"
KEEP, FREE = SCOPES


def _calls() -> list[Call]:
    out: list[Call] = []
    for design, wns_free in (("d1", "small_loss"), ("d2", "small_loss"), ("d3", "large_loss")):
        for _ in range(3):
            out.append(Call(design, "area", FREE, True, {"packed_luts": "large_gain", "wns": wns_free}))
            out.append(Call(design, "area", KEEP, True, {"packed_luts": "within_noise", "wns": "within_noise"}))
            out.append(Call(design, "timing", KEEP, True, {"packed_luts": "within_noise", "wns": "small_gain"}))
            out.append(Call(design, "timing", FREE, True, {"packed_luts": "small_loss", "wns": "small_gain"}))
    return out


def record(levels: int = 11) -> dict:
    return {
        "metrics": {"tns_ns": {"value": -100.0}},
        "probe": {
            "timing": {"failing_endpoints": 500}, "critical_path_features": {"dominant_path_type": "routing_dominated"},
            "utilization_summary": {k: {"used": 1, "utilization_pct": 0.1} for k in ("CLB LUTs", "CLB Registers", "Block RAM Tile", "DSPs", "URAM")},
            "lut_primitive_counts": {f"LUT{i}": 10 for i in range(1, 7)}, "lut_as_shift_register": 5,
            "fanout": {"max_fanout": 100, "num_high_fanout_nets": 4},
        },
        "wpath": {"logic_levels": levels, "data_path_delay_ns": 3.0, "net_delay_ns": 2.0, "tile_span": {"dx": 5, "dy": 5}},
    }


def describe(state_id: str) -> dict:
    return record(levels={10: 11, 11: 8, 12: 11}.get(int(state_id, 16), 11))


def _view(budget_left: int = 3):
    s0, s1, s2 = ls.state(10, -0.347, 1337), ls.state(11, -0.147, 1352, brief_valid=False), ls.state(12, -0.357, 1301)
    calls = (ls.call(1, "timing", 10, 11, reason="largest goal move first"), ls.call(2, "area", 10, 12, preserve=True, attempts=1))
    steps = (ls.step("timing", 10, 11), ls.step("area", 10, 12, preserve=True, attempts=1))
    return ls.view(visited=[s0, s1, s2], calls=calls, steps=steps, budget_left=budget_left, current=s2, best=s2,
                   brief=ls.brief_a(), noise={ls.MetricName.WNS_NS: 0.05, ls.MetricName.PACKED_LUTS: 5.0})


def _config(template: Path, *, profiles_on: bool, name: str = "L3", mode: str = "fuzzy") -> LLMPolicyConfig:
    return LLMPolicyConfig(
        name=name, mode=mode, profiles_on=profiles_on, template=load_template(template),
        profile_texts=dict(ls.PROFILE_TEXT) if profiles_on else {}, profile_sha256=dict(ls.PROFILE_SHA) if profiles_on else {},
        lessons=None, max_repairs=2,
    )


def make_r2(tmp_path: Path, *texts: str, features: Features, with_tables: bool = True, source: str = "lodo", described=describe,
            profiles_on: bool = False):
    transport = ls.ScriptedTransport(*texts)
    llm = CachedLLM(transport, ReplayCache(tmp_path / "llm_cache", mode="record"), LLMParams(model="scripted-model"))
    sink = TelemetrySink(tmp_path / "telemetry.jsonl")
    calls = _calls()
    table = build_table(calls, "src", exclude_design="d3" if source == "lodo" else None) if with_tables else None
    policy = R2Policy(
        _config(R2, profiles_on=profiles_on, mode="exact" if features.history else "fuzzy"), llm, sink, features=features,
        experience=table if features.experience != "off" else None, greedy=table, describe=described,
    )
    return policy, transport, sink


def test_with_every_feature_off_the_policy_asks_the_round_1_question(tmp_path) -> None:
    reply = ls.decision("area", "S1", preserve=True, attempts=1, reason="repair the limit")
    r2, t2, _ = make_r2(tmp_path / "a", reply, features=Features(), profiles_on=True, with_tables=False)
    r1_transport = ls.ScriptedTransport(reply)
    r1 = LLMPolicy(_config(R1, profiles_on=True, name="F1"),
                   CachedLLM(r1_transport, ReplayCache(tmp_path / "b", mode="record"), LLMParams(model="scripted-model")),
                   TelemetrySink(tmp_path / "b.jsonl"))
    view = _view()
    m2, m1 = r2.decide(view), r1.decide(view)
    assert t2.prompts == r1_transport.prompts
    assert m2.id == m1.id and m2.worker_id == m1.worker_id and m2.settings == m1.settings and m2.base_state_id == m1.base_state_id


def test_the_evidence_sections_are_in_the_prompt_in_order_with_one_description_per_state(tmp_path) -> None:
    seen: list[str] = []

    def spy(state_id: str) -> dict:
        seen.append(state_id)
        return describe(state_id)

    features = Features(history=True, physical=("timing", "utilisation", "power"), experience="fuzzy")
    policy, transport, sink = make_r2(tmp_path, ls.decision("area", "S0", preserve=False, attempts=1), features=features, described=spy)
    mandate = policy.decide(_view())
    assert mandate.worker_id == "area" and mandate.base_state_id == ls.sid(10)
    prompt = transport.prompts[0]
    order = [prompt.index(s) for s in ("Current situation", "  made_by: timing, re-placing allowed", "Physical description of the saved states",
                                       "Past experience with the workers", "Reply format")]
    assert order == sorted(order)
    assert sorted(seen) == [ls.sid(10), ls.sid(11), ls.sid(12)]
    assert "S1\n  timing: worst-path logic levels 11 -> 8" in prompt
    assert "S2\n  no meaningful change from the start in these fields" in prompt
    assert "brief goal packed_luts: large_gain usually" in prompt
    record_ = TelemetrySink.read(sink.path)[0]
    assert record_["r2"]["features"] == features.key and record_["r2"]["physical"]["states"] == 3
    assert record_["r2"]["experience"]["form"] == "fuzzy" and record_["r2"]["experience"]["excluded_design_in_table"] is True


def test_a_fully_featured_prompt_is_anonymous(tmp_path) -> None:
    import re

    features = Features(history=True, physical=("timing", "utilisation", "power"), experience="count")
    policy, transport, _ = make_r2(tmp_path, ls.decision("area", "S0", preserve=False, attempts=1), features=features)
    policy.decide(_view())
    prompt = transport.prompts[0]
    assert not re.search(r"[0-9a-f]{64}", prompt), "a state hash reached the prompt"
    for secret in (ls.DESIGN_SECRET, "vexriscv", "data/", ".dcp", "mandate_", "episode"):
        assert secret not in prompt, secret
    assert "A_R_" not in prompt and "LATCH" not in prompt and "phys_opt" not in prompt


def test_one_feature_alone_adds_only_its_own_section(tmp_path) -> None:
    only_p, tp, _ = make_r2(tmp_path / "p", ls.decision("area", "S0", preserve=False, attempts=1), features=Features(physical=("timing",)))
    only_p.decide(_view())
    assert "Physical description" in tp.prompts[0] and "Past experience" not in tp.prompts[0] and "(call 1" not in tp.prompts[0]
    only_e, te, _ = make_r2(tmp_path / "e", ls.decision("area", "S0", preserve=False, attempts=1), features=Features(experience="count"))
    only_e.decide(_view())
    assert "Past experience" in te.prompts[0] and "Physical description" not in te.prompts[0]
    assert "re-placing allowed (" in te.prompts[0] and "calls on 2 designs" in te.prompts[0]


def test_leave_one_design_out_hides_the_held_out_design_and_all_designs_shows_it(tmp_path) -> None:
    reply = ls.decision("area", "S0", preserve=False, attempts=1)
    lodo, t1, _ = make_r2(tmp_path / "l", reply, features=Features(experience="count"), source="lodo")
    full, t2, _ = make_r2(tmp_path / "f", reply, features=Features(experience="count", experience_source="all"), source="all")
    lodo.decide(_view()); full.decide(_view())
    assert "large_loss" not in t1.prompts[0].split("Past experience", 1)[1]
    assert "large_loss" in t2.prompts[0].split("Past experience", 1)[1]
    assert lodo.version != full.version


def test_a_state_without_a_description_is_an_infrastructure_failure_with_no_model_call(tmp_path) -> None:
    def refuse(state_id: str) -> dict:
        raise DescriptionUnavailableError("extraction failed")

    policy, transport, sink = make_r2(tmp_path, features=Features(physical=("timing",)), described=refuse)
    with pytest.raises(LLMUnavailableError):
        policy.decide(_view())
    assert transport.prompts == [] and TelemetrySink.read(sink.path) == []


def test_the_greedy_comparison_is_logged_on_the_same_view(tmp_path) -> None:
    agree, _, sink = make_r2(tmp_path / "a", ls.decision("area", "S0", preserve=False, attempts=1), features=Features())
    agree.decide(_view())
    g = TelemetrySink.read(sink.path)[0]["r2"]["greedy"]
    assert g["base"] == "S0" and g["tried_at_base"] == ["area|keep", "timing|free"]
    assert g["pick"]["worker"] == "area" and g["pick"]["preserve_placement"] is False
    assert g["agree"] is True and g["agree_worker"] is True and g["table_excludes_design"] is True
    differ, _, sink2 = make_r2(tmp_path / "b", ls.decision("timing", "S2", preserve=True, attempts=1), features=Features())
    differ.decide(_view())
    g2 = TelemetrySink.read(sink2.path)[0]["r2"]["greedy"]
    assert g2["agree"] is False and g2["agree_worker"] is False and g2["base"] == "S2"
    stop, _, sink3 = make_r2(tmp_path / "c", ls.decision(action="STOP", reason="nothing left"), features=Features())
    assert stop.decide(_view()) is None
    g3 = TelemetrySink.read(sink3.path)[0]["r2"]["greedy"]
    assert g3["agree"] is None and g3["base"] == "S2"


def test_the_policy_version_binds_features_and_tables_so_a_resume_under_another_setting_is_refused(tmp_path) -> None:
    a, _, _ = make_r2(tmp_path / "1", features=Features(experience="fuzzy"))
    b, _, _ = make_r2(tmp_path / "2", features=Features(experience="fuzzy"))
    c, _, _ = make_r2(tmp_path / "3", features=Features(experience="count"))
    d, _, _ = make_r2(tmp_path / "4", features=Features(experience="fuzzy"), source="all")
    assert a.version == b.version and len({a.version, c.version, d.version}) == 3
    assert a.id == "llm_orchestrator_r2" and a.version.startswith("L3-fuzzy-")


def test_misconfiguration_is_refused_early(tmp_path) -> None:
    with pytest.raises(ValueError):
        make_r2(tmp_path / "x", features=Features(physical=("timing",)), described=None)
    with pytest.raises(ValueError):
        make_r2(tmp_path / "y", features=Features(experience="fuzzy"), with_tables=False)


def test_bad_replies_are_repaired_and_fall_back_exactly_as_in_round_1(tmp_path) -> None:
    good = ls.decision("area", "S1", preserve=True, attempts=1)
    policy, transport, sink = make_r2(tmp_path / "r", "not json", good, features=Features(history=True))
    assert policy.decide(_view()).base_state_id == ls.sid(11)
    assert len(transport.prompts) == 2 and "not valid JSON" in transport.prompts[1]
    lost, t3, sink3 = make_r2(tmp_path / "s", "x", "y", "z", features=Features())
    assert lost.decide(_view()) is None
    rec = TelemetrySink.read(sink3.path)[0]
    assert rec["fallback_triggered"] is True and rec["r2"]["greedy"]["agree"] is None
