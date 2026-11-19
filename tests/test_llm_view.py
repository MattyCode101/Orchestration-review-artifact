"""The fuzzy view: binning edges, limit status, aliases, tried actions, anonymisation."""
from __future__ import annotations

import re

import pytest

import llm_support as ls
from orch.contract.models import MetricName
from orch.policies.llm_view import (
    build_fuzzy_view, categorise, effective_bands, limit_status,
)


@pytest.mark.parametrize(("delta", "expected"), [
    (0.0, "within_noise"), (0.0999, "within_noise"), (0.1, "small_gain"), (0.3, "small_gain"),
    (0.1 * 3, "small_gain"), (0.3001, "large_gain"), (-0.0999, "within_noise"),
    (-0.1, "small_loss"), (-0.3, "small_loss"), (-0.3001, "large_loss"),
])
def test_categories_at_the_exact_band_edges_for_a_higher_is_better_metric(delta, expected) -> None:
    assert categorise(delta, 0.1, higher_is_better=True) == expected


@pytest.mark.parametrize(("delta", "expected"), [
    (-10, "small_gain"), (-30, "small_gain"), (-31, "large_gain"), (-9, "within_noise"),
    (10, "small_loss"), (30, "small_loss"), (31, "large_loss"),
])
def test_a_lower_is_better_metric_flips_the_sign(delta, expected) -> None:
    assert categorise(delta, 10.0, higher_is_better=False) == expected


def test_a_zero_band_treats_every_change_as_large_and_no_change_as_noise() -> None:
    assert categorise(0.0, 0.0, higher_is_better=True) == "within_noise"
    assert categorise(1e-9, 0.0, higher_is_better=True) == "large_gain"
    assert categorise(-1e-9, 0.0, higher_is_better=True) == "large_loss"


@pytest.mark.parametrize(("margin", "expected"), [
    (0.2, "satisfied_margin"), (0.1001, "satisfied_margin"), (0.1, "satisfied_tight"),
    (0.05, "satisfied_tight"), (0.0, "satisfied_tight"), (-0.0001, "broken_small"),
    (-0.3, "broken_small"), (-0.3001, "broken_large"),
])
def test_limit_status_edges(margin, expected) -> None:
    assert limit_status(margin, 0.1) == expected


def test_limit_status_with_a_zero_band() -> None:
    assert limit_status(0.5, 0.0) == "satisfied_margin"
    assert limit_status(0.0, 0.0) == "satisfied_tight"
    assert limit_status(-0.5, 0.0) == "broken_large"


def test_netlist_and_ff_bands_borrow_the_packed_lut_fraction_and_power_is_never_invented() -> None:
    start = ls.metrics(-0.5, 1000, ffs=500, netlist=2000)
    bands = effective_bands({MetricName.WNS_NS: 0.1, MetricName.PACKED_LUTS: 10.0}, start)
    assert bands[MetricName.PACKED_LUTS] == 10.0 and bands[MetricName.WNS_NS] == 0.1
    assert bands[MetricName.NETLIST_LUTS] == pytest.approx(20.0)
    assert bands[MetricName.FFS] == pytest.approx(5.0)
    assert MetricName.DYNAMIC_POWER_W not in bands
    given = effective_bands({MetricName.PACKED_LUTS: 10.0, MetricName.DYNAMIC_POWER_W: 0.05}, start)
    assert given[MetricName.DYNAMIC_POWER_W] == 0.05
    assert MetricName.NETLIST_LUTS not in effective_bands({MetricName.WNS_NS: 0.1}, start)


def _tree():
    """S0 start; S1 and S2 both from S0; S3 from S2. Values are unusual so leaks are detectable."""

    s0 = ls.state(10, -0.347, 1337)
    s1 = ls.state(11, -0.147, 1352, brief_valid=False)      # wns +0.2 (small_gain); luts +15 beyond the limit
    s2 = ls.state(12, -0.357, 1301)                          # wns -0.01 noise; luts -36 (large_gain? 3b=30)
    s3 = ls.state(13, -0.037, 1330, brief_valid=True)
    calls = (
        ls.call(1, "timing", 10, 11, reason="largest goal move first"),
        ls.call(2, "area", 10, 12, preserve=True, attempts=1),
        ls.call(3, "timing", 12, 13, reason="build on the smaller design"),
    )
    steps = (ls.step("timing", 10, 11), ls.step("area", 10, 12, preserve=True, attempts=1),
             ls.step("timing", 12, 13))
    return ls.view(visited=[s0, s1, s2, s3], calls=calls, steps=steps, budget_left=2,
                   current=s3, best=s3)


def test_aliases_follow_discovery_order_and_branching_resolves_to_the_right_states() -> None:
    built = build_fuzzy_view(_tree(), mode="fuzzy")
    assert [s.id for s in built.fuzzy.states] == ["S0", "S1", "S2", "S3"]
    assert built.alias_to_state["S2"] == ls.sid(12) and built.alias_to_state["S0"] == ls.sid(10)
    assert built.state_to_alias[ls.sid(13)] == "S3"
    parents = {s.id: s.parent for s in built.fuzzy.states}
    assert parents == {"S0": "none", "S1": "S0", "S2": "S0", "S3": "S2"}
    assert built.fuzzy.best_valid_state == "S3" and built.fuzzy.current_state == "S3"


def test_each_state_lists_the_actions_already_tried_from_it() -> None:
    built = build_fuzzy_view(_tree(), mode="fuzzy")
    tried = {s.id: s.tried for s in built.fuzzy.states}
    assert len(tried["S0"]) == 2 and len(tried["S2"]) == 1 and tried["S1"] == () and tried["S3"] == ()
    assert any("timing" in line and "re-placing allowed" in line and "S1" in line for line in tried["S0"])
    assert any("area" in line and "placement preserved" in line and "S2" in line for line in tried["S0"])


def test_made_by_describes_the_producing_call_without_recipes() -> None:
    built = build_fuzzy_view(_tree(), mode="fuzzy")
    made = {s.id: s.made_by for s in built.fuzzy.states}
    assert made["S0"] == "start"
    assert "timing" in made["S1"] and "re-placing allowed" in made["S1"]
    assert "area" in made["S2"] and "placement preserved" in made["S2"] and "1 attempt" in made["S2"]


def test_changes_are_categories_against_the_start_and_the_parent() -> None:
    built = build_fuzzy_view(_tree(), mode="fuzzy")
    s1 = built.fuzzy.states[1]
    assert s1.vs_start["wns"] == "small_gain" and s1.vs_start["packed_luts"] == "small_loss"
    s3 = built.fuzzy.states[3]
    assert s3.vs_parent["wns"] == "large_gain"            # -0.357 -> -0.037 = +0.32 (> 3b)
    assert s3.vs_start["wns"] == "large_gain"
    assert built.fuzzy.states[0].vs_start == {} and built.fuzzy.states[0].vs_parent == {}


def test_limit_status_uses_the_brief_threshold() -> None:
    built = build_fuzzy_view(_tree(), mode="fuzzy")
    limits = {s.id: s.limits for s in built.fuzzy.states}
    assert limits["S0"] == {"packed_luts": "satisfied_tight"}           # exactly at the start value
    assert limits["S1"] == {"packed_luts": "broken_small"}              # 15 over, band 10
    assert limits["S2"] == {"packed_luts": "satisfied_margin"}


def test_a_control_brief_limit_with_an_offset_is_judged_against_start_plus_offset() -> None:
    s0 = ls.state(20, -0.5, 1000)
    s1 = ls.state(21, -0.65, 900)                  # wns 0.15 below start; limit is 0.1 below -> beyond by 0.05
    built = build_fuzzy_view(ls.view(visited=[s0, s1], brief=ls.brief_a(), current=s1, best=s0,
                                     calls=(ls.call(1, "area", 20, 21),), steps=(ls.step("area", 20, 21),)),
                             mode="fuzzy")
    assert built.fuzzy.states[1].limits == {"wns": "broken_small"}
    assert built.fuzzy.states[0].limits == {"wns": "satisfied_tight"}   # one band inside the offset limit


def test_the_goal_metric_is_categorised_against_the_best_valid_state() -> None:
    built = build_fuzzy_view(_tree(), mode="fuzzy")
    goal = {s.id: s.goal_vs_best for s in built.fuzzy.states}
    assert goal["S3"] == "within_noise" and goal["S0"] == "large_loss"     # best is S3 at -0.037
    assert goal["S1"] == "small_loss"                                      # 0.11 below the best, just over b


def test_a_duplicate_output_keeps_one_alias_and_an_unchanged_result_is_named() -> None:
    s0, s1 = ls.state(30, -0.4, 1000), ls.state(31, -0.2, 1000)
    calls = (ls.call(1, "timing", 30, 31), ls.call(2, "area", 30, 30), ls.call(3, "area", 31, None, runs=2))
    steps = (ls.step("timing", 30, 31), ls.step("area", 30, 30), ls.step("area", 31, None))
    built = build_fuzzy_view(ls.view(visited=[s0, s1, s0], calls=calls, steps=steps, budget_left=1, best=s1,
                                     current=s1), mode="fuzzy")
    assert [s.id for s in built.fuzzy.states] == ["S0", "S1"]
    tried_s0 = built.fuzzy.states[0].tried
    assert any("unchanged" in line for line in tried_s0)
    assert any("no usable state" in line for line in built.fuzzy.states[1].tried)


def test_the_trajectory_log_carries_the_models_own_reasons_and_binned_outcomes() -> None:
    built = build_fuzzy_view(_tree(), mode="fuzzy")
    log = built.fuzzy.trajectory_log
    assert len(log) == 3
    assert log[0].startswith("Step 1: timing from S0")
    assert "largest goal move first" in log[0] and "S1" in log[0]
    assert "limit packed_luts broken_small" in log[0]
    assert "Reason: not recorded" in log[1]
    assert "build on the smaller design" in log[2]


def test_a_fuzzy_prompt_has_no_hashes_design_names_ids_or_raw_metric_values() -> None:
    built = build_fuzzy_view(_tree(), mode="fuzzy")
    text = built.text
    assert ls.DESIGN_SECRET not in text and "mandate_" not in text and "episode" not in text.lower()
    assert not re.search(r"[0-9a-f]{12,}", text)
    for leaked in ("1337", "1352", "1301", "-0.347", "-0.147", "-0.357", "0.347"):
        assert leaked not in text
    assert len(built.sha256) == 64


def test_exact_mode_adds_absolute_values_and_exact_deltas_but_stays_anonymous() -> None:
    built = build_fuzzy_view(_tree(), mode="exact")
    text = built.text
    assert "1352" in text and "-0.147" in text and "+15" in text
    assert ls.DESIGN_SECRET not in text and not re.search(r"[0-9a-f]{12,}", text)
    assert build_fuzzy_view(_tree(), mode="fuzzy").sha256 != built.sha256


def test_the_view_is_a_pure_function_of_its_input() -> None:
    first = build_fuzzy_view(_tree(), mode="fuzzy")
    second = build_fuzzy_view(_tree(), mode="fuzzy")
    assert first.text == second.text and first.sha256 == second.sha256


def test_a_metric_without_a_band_is_left_out_instead_of_guessed() -> None:
    s0 = ls.state(40, -0.4, 1000)
    s1 = ls.state(41, -0.3, 990)
    built = build_fuzzy_view(ls.view(visited=[s0, s1], calls=(ls.call(1, "timing", 40, 41),),
                                     steps=(ls.step("timing", 40, 41),), current=s1, best=s1,
                                     noise={MetricName.WNS_NS: 0.1}), mode="fuzzy")
    assert set(built.fuzzy.states[1].vs_start) == {"wns"}


def test_budget_is_reported_as_used_remaining_and_total() -> None:
    built = build_fuzzy_view(_tree(), mode="fuzzy")
    assert built.fuzzy.budget == {"used": 4, "remaining": 2, "total": 6}
    assert "used 4, remaining 2, total 6" in built.text


# ------------------------------------------------------------------ recipe-level view (general agent)
def test_the_recipe_level_view_describes_actions_as_recipes() -> None:
    s0, s1 = ls.state(60, -0.4, 1000), ls.state(61, -0.2, 1030, brief_valid=False)
    view = ls.view(visited=[s0, s1], calls=(ls.call(1, "timing", 60, 61, attempts=1, reason="push the goal", recipe="recipe_one"),),
                   steps=(ls.step("timing", 60, 61, attempts=1),), current=s1, best=s0, budget_left=5)
    built = build_fuzzy_view(view, mode="fuzzy", recipe_level=True)
    state1 = built.fuzzy.states[1]
    assert state1.made_by == "recipe recipe_one" and state1.picked_by_worker == "n/a"
    assert state1.tried == () and built.fuzzy.states[0].tried == ("recipe recipe_one -> S1",)
    assert built.fuzzy.trajectory_log[0].startswith("Step 1: recipe recipe_one from S0. Reason: push the goal.")
    assert "picked by the worker" not in built.text and "placement" not in built.text
    # the worker-level view of the same data is unchanged
    assert "picked by the worker: yes" in build_fuzzy_view(view, mode="fuzzy").text


def test_the_recipe_level_view_shares_every_non_action_line_with_the_worker_level_view() -> None:
    view = _tree()
    worker, recipe = build_fuzzy_view(view).text.splitlines(), build_fuzzy_view(view, recipe_level=True).text.splitlines()
    keep = ("budget:", "best_valid_state:", "current_state:", "  brief_valid:", "  limits:", "  vs_start:", "  goal_vs_best:")
    assert [x for x in worker if x.startswith(keep)] == [x for x in recipe if x.startswith(keep)]


def test_a_call_that_ran_fewer_attempts_than_requested_says_the_worker_ran_out_of_options() -> None:
    s0, s1 = ls.state(70, -0.4, 1000), ls.state(71, -0.3, 1000)
    view = ls.view(visited=[s0, s1], calls=(ls.call(1, "timing", 70, 71, attempts=3, runs=1, reason="push"),),
                   steps=(ls.step("timing", 70, 71, attempts=3),), current=s1, best=s1, budget_left=5)
    built = build_fuzzy_view(view)
    assert "only 1 of 3 attempts could run: nothing more to try there" in built.fuzzy.states[0].tried[0]
    assert "only 1 of 3 attempts could run" in built.fuzzy.trajectory_log[0]
    full = build_fuzzy_view(_tree())
    assert "attempts could run" not in full.text                        # calls that used their attempts say nothing
    assert "attempts could run" not in build_fuzzy_view(view, recipe_level=True).text


def test_a_reason_that_ends_with_a_full_stop_does_not_double_it_in_the_log() -> None:
    s0, s1 = ls.state(80, -0.4, 1000), ls.state(81, -0.3, 1000)
    view = ls.view(visited=[s0, s1], calls=(ls.call(1, "timing", 80, 81, reason="Push the goal first."),),
                   steps=(ls.step("timing", 80, 81),), current=s1, best=s1, budget_left=4)
    log = build_fuzzy_view(view).fuzzy.trajectory_log[0]
    assert "Reason: Push the goal first. Outcome:" in log and ".." not in log
