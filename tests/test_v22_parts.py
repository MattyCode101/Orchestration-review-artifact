"""Template v2.2: switches, compact view, relative context, candidates and sequence experience (new tests)."""
from __future__ import annotations

from pathlib import Path

import llm_support as ls
import pytest
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import Brief, BriefTerm, MetricName, Objective, WorkerCard
from orch.policies.v2_prompt import load_assets
from orch.policies.v2_view import ViewFlags, render_view_v2
from orch.policies.v22_candidates import (
    GOAL_REPAIR_V2, generate_candidates, render_candidates, shuffled, which_candidate,
)
from orch.policies.v22_relative import Anchors, anchors_from_record, render_relative
from orch.policies.v22_sequence import build_records, render_sequences
from orch.policies.v22_switches import (
    GRID_VARIANTS, VARIANTS, Switches, definition_text, load_switch_texts, load_variant, reply_texts_for,
    sha256_file, template_text_for,
)
from orch.policies.v22_view import render_view_compact

ROOT = Path(__file__).resolve().parents[1]
FULL = ViewFlags(labels=True, vocab=True, placement=True, exact_only=False)


# ------------------------------------------------------------------------------------------------ switches
def test_registry_is_the_declared_grid() -> None:
    """The authors' decisions of 2026-11-13 21:45: V0 F1-R1, V1 v2.1, V2 no E, V3 no P, V4 +R, V5 +U, V6 +S' and U, V7 +S', U, R,
    V8 compact (H compact, no P, +S', U, R). Candidate options (C) and the old sequences (S) are in no variant."""

    assert GRID_VARIANTS == ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")
    assert VARIANTS["V1"].is_v21 and not any(v.is_v21 for k, v in VARIANTS.items() if k != "V1")
    assert VARIANTS["V2"] == Switches(E=False) and VARIANTS["V3"] == Switches(P=False)
    assert VARIANTS["V4"] == Switches(R=True) and VARIANTS["V5"] == Switches(U=True)
    assert VARIANTS["V6"] == Switches(S2=True, U=True) and VARIANTS["V7"] == Switches(S2=True, U=True, R=True)
    assert VARIANTS["V8"] == Switches(H="compact", P=False, S2=True, U=True, R=True)
    assert not any(v.C or v.S for v in VARIANTS.values())


def test_h_mode_is_checked() -> None:
    with pytest.raises(ValueError):
        Switches(H="short")


def test_u_needs_experience_content_and_the_key_names_every_switch() -> None:
    with pytest.raises(ValueError):
        Switches(E=False, U=True)
    assert Switches(E=False, S2=True, U=True).experience_shown
    assert Switches().key == "EPHrcstu" and VARIANTS["V8"].key == "EphRcsTU" and VARIANTS["V5"].key == "EPHrcstU"
    assert len({v.key for v in VARIANTS.values()}) == len(VARIANTS)
    assert Switches().as_json() == {"E": True, "P": True, "H": "full", "R": False, "C": False, "S": False, "S2": False, "U": False}


def test_variant_files_equal_the_generated_text() -> None:
    base = sha256_file(ROOT / "configs/policies/template_v2.1.yaml")
    for name, sw in VARIANTS.items():
        path = ROOT / f"configs/policies/template_v2.2_r2/{name}.yaml"
        assert path.read_text(encoding="utf-8") == definition_text(name, base)
        loaded_name, loaded, data = load_variant(path)
        assert loaded_name == name and loaded == sw and data["base_definition_sha256"] == base
        assert data["situation_table"] == "configs/experience/situations_v22.json" and "candidate_rules" not in data


def test_u_replaces_the_push_bullet_with_the_neutral_line_and_nothing_else() -> None:
    assets, texts = load_assets(ROOT), load_switch_texts(ROOT)
    base = assets.templates["p1234r"].text
    push = "- Push: the worker whose objective matches the goal usually moves the goal most; check its experience lines."
    neutral = "- Any worker can move the goal: check what each worker and setting did to the goal in this situation."
    assert base.count(push) == 1
    u = template_text_for(base, Switches(U=True), texts)
    assert u == base.replace(push, neutral) and u.count(neutral) == 1 and "- Push:" not in u
    assert template_text_for(base, Switches(S2=True), texts) == base                      # S' alone leaves the template alone
    assert template_text_for(base, Switches(E=False, S2=True, U=True), texts) == base.replace(push, neutral)
    no_p = template_text_for(base, Switches(P=False, U=True), texts)                     # edits compose
    assert neutral in no_p and "physical" not in no_p


def test_template_edits_cut_exactly_the_switched_off_parts() -> None:
    assets, texts = load_assets(ROOT), load_switch_texts(ROOT)
    base = assets.templates["p1234r"].text
    assert template_text_for(base, Switches(), texts) == base
    assert template_text_for(base, Switches(E=False, S=True), texts) == base          # experience content stays
    no_e = template_text_for(base, Switches(E=False), texts)
    assert "experience" not in no_e and "physical facts about each saved state." in no_e
    no_p = template_text_for(base, Switches(P=False), texts)
    assert "physical" not in no_p and "experience with the workers on other designs." in no_p
    neither = template_text_for(base, Switches(E=False, P=False), texts)
    assert "experience" not in neither and "physical" not in neither
    assert "and every saved state with its exact goal and limit values and how they changed." in neither


def test_reply_texts_drop_the_experience_citations_only_when_e_is_off() -> None:
    assets, texts = load_assets(ROOT), load_switch_texts(ROOT)
    p4r = assets.texts["P4r"]
    assert reply_texts_for(p4r, Switches(), texts) == p4r
    off = reply_texts_for(p4r, Switches(E=False), texts)
    assert "experience" not in off["evidence"] and "experience" not in off["forecast"]
    assert off["forecast"].endswith("null when you stop.")


# ------------------------------------------------------------------------------------------------ compact view
def _view():
    visited = [ls.state(10, -0.30, 1000), ls.state(11, -0.10, 1030, brief_valid=False), ls.state(12, -0.28, 990)]
    calls = (ls.call(1, "timing", 10, 11, attempts=2, runs=2, reason="push"),
             ls.call(2, "area", 11, None, preserve=True, attempts=2, runs=1, reason="fix"))
    steps = (ls.step("timing", 10, 11), ls.step("timing", 10, 12), ls.step("area", 11, None, preserve=True))
    return ls.view(visited=visited, calls=calls, steps=steps, budget_left=2, best=visited[2])


def test_compact_view_keeps_latest_values_and_drops_history() -> None:
    view = _view()
    full, compact = render_view_v2(view, FULL), render_view_compact(view)
    assert compact.alias_to_state == full.alias_to_state and compact.state_to_alias == full.state_to_alias
    text = compact.text
    for gone in ("parent:", "made by", "vs parent", "trajectory log", "selected by the worker", "Reason:"):
        assert gone not in text
    for kept in ("budget: used 4, remaining 2", "best state within limits so far: S2", "saved states:", "meets every limit: no",
                 "wns: -0.100 ns (", "vs start)", "limits: packed_luts broken by more",
                 "no options left", "tried: nothing yet"):
        assert kept in text
    assert len(text) < len(full.text)
    assert "trajectory log" in full.text           # the full form is untouched


# ------------------------------------------------------------------------------------------------ relative context
def test_anchors_come_from_the_stored_record_fields() -> None:
    record = {"wpath": {"requirement_ns": 1.57}, "probe": {"utilization_summary": {"CLB LUTs": {"available": 394080}}}}
    assert anchors_from_record(record) == Anchors(1.57, 394080.0)
    assert anchors_from_record({}) == Anchors(None, None)
    assert anchors_from_record({"wpath": {"requirement_ns": 0}}).period_ns is None


def test_relative_context_states_each_metric_against_its_anchor() -> None:
    view, texts = _view(), load_switch_texts(ROOT)["relative"]
    aliases = {s.state_id: f"S{i}" for i, s in enumerate([view.start, *view.visited[1:]])}
    text = render_relative(view, aliases, Anchors(1.5, 100000.0), header=texts["header"], intro_all=texts["intro_all"],
                           intro_no_period=texts["intro_no_period"])
    assert text.startswith("Relative context\n") and "(1.500 ns)" in text
    assert "S0: wns -0.300 ns (-20.0% of the clock requirement); packed_luts 1000 (1.0% of the device, 100.0% of the start)" in text
    assert "S1: wns -0.100 ns (-6.7% of the clock requirement); packed_luts 1030 (1.0% of the device, 103.0% of the start)" in text
    assert "power 1.500 W (100.0% of the start)" in text
    bare = render_relative(view, aliases, Anchors(None, None), header=texts["header"], intro_all=texts["intro_all"],
                           intro_no_period=texts["intro_no_period"])
    assert "of the clock requirement" not in bare and "wns -0.300 ns;" in bare


# ------------------------------------------------------------------------------------------------ candidates
def test_candidates_at_decision_zero_are_push_and_gentle() -> None:
    view = ls.view(visited=[ls.state(10, -0.3, 1000)])
    cands = generate_candidates(view)
    assert [(c.kind, c.worker_id, c.base_alias, c.preserve_placement, c.attempts, c.respect) for c in cands] == [
        ("push", "timing", "S0", False, 2, False), ("gentle", "timing", "S0", True, 1, False)]


def test_a_broken_limit_gets_a_repair_from_the_best_breaking_state_with_a_goal_gain() -> None:
    visited = [ls.state(10, -0.30, 1000), ls.state(11, -0.10, 1030, brief_valid=False),
               ls.state(12, -0.20, 1040, brief_valid=False), ls.state(13, -0.40, 1060, brief_valid=False)]
    cands = generate_candidates(ls.view(visited=visited))
    repair = [c for c in cands if c.kind == "repair"]
    assert len(repair) == 1 and (repair[0].worker_id, repair[0].base_alias) == ("area", "S1")
    assert (repair[0].preserve_placement, repair[0].attempts) == (GOAL_REPAIR_V2["repair_preserve_placement"], 1)
    assert [c.kind for c in cands] == ["push", "repair", "gentle"]


def test_repair_base_on_equal_goal_values_is_the_state_with_the_smaller_limit_break() -> None:
    """Decision of 2026-11-13, item 4 (the candidate code stays; no variant uses it)."""

    visited = [ls.state(10, -0.30, 1000), ls.state(11, -0.10, 1060, brief_valid=False),
               ls.state(12, -0.10, 1030, brief_valid=False), ls.state(13, -0.10, 1045, brief_valid=False)]
    repair = [c for c in generate_candidates(ls.view(visited=visited)) if c.kind == "repair"]
    assert [c.base_alias for c in repair] == ["S2"]                                  # LUTs 1030 is the smallest break
    flipped = [ls.state(10, -0.30, 1000), ls.state(12, -0.10, 1030, brief_valid=False), ls.state(11, -0.10, 1060, brief_valid=False)]
    assert [c.base_alias for c in generate_candidates(ls.view(visited=flipped)) if c.kind == "repair"] == ["S1"]
    better_goal = [ls.state(10, -0.30, 1000), ls.state(11, -0.05, 1090, brief_valid=False), ls.state(12, -0.10, 1030, brief_valid=False)]
    assert [c.base_alias for c in generate_candidates(ls.view(visited=better_goal)) if c.kind == "repair"] == ["S1"]   # the goal first


def test_exhausted_combinations_and_the_budget_limit_the_list() -> None:
    visited = [ls.state(10, -0.3, 1000)]
    spent = (ls.call(1, "timing", 10, None, preserve=True, attempts=2, runs=1),)        # gentle's combination has no options left
    cands = generate_candidates(ls.view(visited=visited, calls=spent, budget_left=1))
    assert [c.kind for c in cands] == ["push"] and cands[0].attempts == 1                # attempts capped by the budget
    assert generate_candidates(ls.view(visited=visited, budget_left=0)) == []


def test_two_broken_limits_give_two_repairs_for_three_workers() -> None:
    power = WorkerCard(id="power", version="v", objective=Objective(metric=MetricName.DYNAMIC_POWER_W, direction="min"),
                       recipe_set_id="power", may_replace=True)
    brief = Brief(id="P0", objective=Objective(metric=MetricName.DYNAMIC_POWER_W, direction="min"), budget_runs=6, terms=(
        BriefTerm(id="w", metric=MetricName.WNS_NS, comparison=">="), BriefTerm(id="l", metric=MetricName.PACKED_LUTS, comparison="<=")))
    visited = [ls.state(10, -0.3, 1000, power=2.0), ls.state(11, -0.5, 1100, brief_valid=False, power=1.0)]
    view = ls.view(visited=visited, brief=brief).model_copy(update={"worker_cards": (*ls.cards(), power)})
    cands = generate_candidates(view)
    assert [(c.kind, c.worker_id) for c in cands] == [("push", "power"), ("repair", "timing"), ("repair", "area"),
                                                      ("gentle", "power")]


def test_shuffle_is_seeded_labels_are_neutral_and_a_pick_is_recognised() -> None:
    visited = [ls.state(10, -0.30, 1000), ls.state(11, -0.10, 1030, brief_valid=False)]
    view = ls.view(visited=visited)
    cands = generate_candidates(view)
    a = shuffled(cands, seed=0, episode_id="e", decision_index=1)
    assert a == shuffled(cands, seed=0, episode_id="e", decision_index=1)
    assert [c.label for c in a] == ["Option A", "Option B", "Option C"]
    orders = {tuple(c.kind for c in shuffled(cands, seed=s, episode_id="e", decision_index=i))
              for s in range(4) for i in range(6)}
    assert len(orders) > 1
    text = render_candidates(a, view, header="H", intro="I", no_line="none", line_for=lambda w, p: f"{w} line")
    for word in ("push", "repair", "gentle", "recommend"):
        assert word not in text
    assert "Experience for timing, timing line" in text
    aliases = {"S0": ls.sid(10), "S1": ls.sid(11)}
    repair = next(c for c in a if c.kind == "repair")
    pick = which_candidate(a, {"worker": "area", "base": "S1", "preserve": False, "attempts": 1, "respect": False}, aliases)
    assert pick == {"pick": repair.label, "exact": True, "kind": "repair"}
    assert which_candidate(a, {"worker": "area", "base": "S1", "preserve": False, "attempts": 2, "respect": True}, aliases)["exact"] is False
    assert which_candidate(a, {"worker": "area", "base": "S0", "preserve": True, "attempts": 1, "respect": True}, aliases)["pick"] == "custom"
    assert which_candidate(a, None, aliases)["pick"] == "stop"


# ------------------------------------------------------------------------------------------------ sequences
def _episode(design: str, *, second_worker: str = "timing", first_status: str = "BRIEF_VIOLATION", ok: bool = True) -> dict:
    steps1 = [{"out_id": f"{design}x", "physically_valid": True, "status": first_status,
               "metrics": {"packed_luts": 800.0, "wns_ns": -0.9}}]
    steps2 = [{"out_id": f"{design}y", "physically_valid": True, "status": "ACCEPTABLE" if ok else "BRIEF_VIOLATION",
               "metrics": {"packed_luts": 850.0, "wns_ns": -0.5}}]
    return {"design": design, "brief": "A0", "episode_id": f"e_{design}", "start_metrics": {"packed_luts": 1000.0, "wns_ns": -0.5},
            "calls": [{"completed": True, "worker": "area", "base_id": f"{design}s", "steps": steps1},
                      {"completed": True, "worker": second_worker, "base_id": f"{design}x", "steps": steps2}]}


BANDS = {d: {"packed_luts": 10.0, "wns_ns": 0.1, "dynamic_power_w": 0.01} for d in ("d1", "d2", "d3", "d4")}


def test_sequence_records_classify_break_and_clean_pairs() -> None:
    briefs = {"A0": load_frozen_brief("A0")}
    rec = build_records([_episode("d1")], BANDS, briefs)
    assert len(rec) == 1 and (rec[0]["pattern"], rec[0]["second"], rec[0]["restored"]) == ("break", "repair", True)
    assert rec[0]["final_bands"] == 15.0
    other = build_records([_episode("d1", second_worker="area")], BANDS, briefs)
    assert other[0]["second"] == "other"
    unrestored = build_records([_episode("d1", ok=False)], BANDS, briefs)
    assert unrestored[0]["restored"] is False and unrestored[0]["final_bands"] is None
    clean = build_records([_episode("d1", first_status="ACCEPTABLE")], BANDS, briefs)
    assert (clean[0]["pattern"], clean[0]["second"]) == ("clean", "any") and clean[0]["further"] is False
    assert build_records([_episode("zz")], BANDS, briefs) == []                            # no band row for the design


def test_sequence_lines_need_three_designs_and_leave_the_evaluated_design_out() -> None:
    briefs = {"A0": load_frozen_brief("A0")}
    records = build_records([_episode(d) for d in ("d1", "d2", "d3", "d4")], BANDS, briefs)
    t = load_switch_texts(ROOT)["sequence"]
    kw = dict(brief="A0", header=t["header"], intro=t["intro"], empty=t["empty"])
    text, meta = render_sequences(records, exclude_design="d4", **kw)
    assert "(3 sequences from 3 designs), a repair call" in text and "restored every limit with a goal gain usually" in text
    assert meta["records_in_scope"] == 3 and meta["excluded_design"] == "d4"
    short, meta2 = render_sequences(records, exclude_design="d3", **kw)
    assert "3 sequences from 3 designs" in short
    few, meta3 = render_sequences(records[:2], exclude_design=None, **kw)
    assert few.endswith(t["empty"]) and meta3["omitted"] and not meta3["shown"]
    other, _ = render_sequences(records, exclude_design=None, **{**kw, "brief": "T0"})
    assert other.endswith(t["empty"])
