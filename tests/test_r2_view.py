"""Round-2 view features: history is additive over the round-1 fuzzy view; features validate and switch alone."""
from __future__ import annotations

import pytest

import llm_support as ls
from orch.contract.models import MetricName
from orch.policies.llm_view import build_fuzzy_view
from orch.policies.r2_view import (
    Features, brief_metrics, build_r2_view, goal_and_limits, producing_calls, render_history_view,
)


def _view(noise=None, brief=None):
    s0 = ls.state(10, -0.347, 1337)
    s1 = ls.state(11, -0.147, 1352, brief_valid=False)
    s2 = ls.state(12, -0.357, 1301)
    calls = (
        ls.call(1, "timing", 10, 11, reason="largest goal move first"),
        ls.call(2, "area", 10, 12, preserve=True, attempts=1),
    )
    steps = (ls.step("timing", 10, 11), ls.step("area", 10, 12, preserve=True, attempts=1))
    return ls.view(visited=[s0, s1, s2], calls=calls, steps=steps, budget_left=3, current=s2, best=s2, noise=noise, brief=brief)


def test_with_every_feature_off_the_view_is_the_round_1_fuzzy_view() -> None:
    view = _view()
    assert build_r2_view(view, Features()).text == build_fuzzy_view(view, mode="fuzzy").text
    assert build_r2_view(view, Features(physical=("timing",), experience="fuzzy")).text == build_fuzzy_view(view).text


def test_history_keeps_every_round_1_line_and_adds_provenance_and_exact_values() -> None:
    view = _view()
    base = build_fuzzy_view(view, mode="fuzzy")
    rich = build_r2_view(view, Features(history=True))
    lines = rich.text.splitlines()
    for state in base.fuzzy.states:                       # the structure and aliases are the round-1 ones
        assert state.id in rich.text
    assert rich.alias_to_state == base.alias_to_state and rich.sha256 != base.sha256
    for fixed in ("budget: used 3, remaining 3, total 6", "best_valid_state: S2", "current_state: S2", "states:", "trajectory_log:"):
        assert fixed in lines
    assert sum(l.startswith("  limits:") for l in lines) == 3 and sum(l.startswith("  goal_vs_best:") for l in lines) == 3
    assert any(l.startswith("  Step 1:") or l.startswith("  Step 1") for l in lines)
    s1 = lines[lines.index("S1"):lines.index("S2")]
    assert "  parent: S0" in s1
    assert "  made_by: timing, re-placing allowed, 2 attempts (call 1, attempt 1 of 2); picked by the worker: yes" in s1
    assert "  wns: -0.147 ns; vs start +0.200 small_gain; vs parent same as vs start" in s1
    assert "  packed_luts: 1352; vs start +15 small_loss; vs parent same as vs start" in s1
    assert any(l.startswith("  other metrics vs start:") and "wns" not in l and "packed_luts" not in l for l in s1)
    s2 = lines[lines.index("S2"):lines.index("trajectory_log:")]
    assert "  made_by: area, placement preserved, 1 attempt (call 2, attempt 1 of 1); picked by the worker: yes" in s2
    assert "  wns: -0.357 ns; vs start -0.010 within_noise; vs parent same as vs start" in s2
    assert "  packed_luts: 1301; vs start -36 large_gain; vs parent same as vs start" in s2
    s0 = lines[lines.index("S0 (start)"):lines.index("S1")]
    assert "  wns: -0.347 ns" in s0 and "  packed_luts: 1337" in s0 and not any("vs start" in l for l in s0 if l.startswith("  wns"))


def test_a_state_built_on_another_state_shows_its_own_change_against_the_parent() -> None:
    s0, s1, s2 = ls.state(10, -0.5, 1000), ls.state(11, -0.4, 1100), ls.state(12, -0.3, 1040)
    calls = (ls.call(1, "timing", 10, 11), ls.call(2, "area", 11, 12, attempts=1))
    steps = (ls.step("timing", 10, 11), ls.step("area", 11, 12, attempts=1))
    view = ls.view(visited=[s0, s1, s2], calls=calls, steps=steps, budget_left=2, current=s2, best=s2)
    lines = build_r2_view(view, Features(history=True)).text.splitlines()
    s2_lines = lines[lines.index("S2"):lines.index("trajectory_log:")]
    assert "  parent: S1" in s2_lines
    assert "  wns: -0.300 ns; vs start +0.200 small_gain; vs parent +0.100 small_gain" in s2_lines
    assert "  packed_luts: 1040; vs start +40 large_loss; vs parent -60 large_gain" in s2_lines   # band 10: 4b is large


def test_a_design_without_a_noise_band_still_shows_exact_values_and_says_so() -> None:
    view = _view(noise={})
    text = build_r2_view(view, Features(history=True)).text
    assert "vs start +0.200 (no noise band recorded)" in text and "vs start +15 (no noise band recorded)" in text


def test_producing_calls_matches_a_step_to_its_call_by_worker_base_and_placement() -> None:
    assert producing_calls(_view()) == {ls.sid(11): (1, 1, 2), ls.sid(12): (2, 1, 1)}
    s0, s1, s2 = ls.state(10, -0.5, 1000), ls.state(11, -0.4, 1010), ls.state(12, -0.45, 990)
    calls = (ls.call(1, "timing", 10, 11, attempts=2, runs=2),)
    steps = (ls.step("timing", 10, 11), ls.step("timing", 10, 12))
    view = ls.view(visited=[s0, s1, s2], calls=calls, steps=steps, budget_left=4, current=s1, best=s1)
    assert producing_calls(view) == {ls.sid(11): (1, 1, 2), ls.sid(12): (1, 2, 2)}


def test_the_brief_metrics_are_the_goal_then_each_limit() -> None:
    assert brief_metrics(_view()) == [MetricName.WNS_NS, MetricName.PACKED_LUTS]
    area = _view(brief=ls.brief_a())
    assert brief_metrics(area) == [MetricName.PACKED_LUTS, MetricName.WNS_NS]
    assert goal_and_limits(area) == ("packed_luts", ["wns"])


def test_features_validate_normalise_and_name_themselves() -> None:
    f = Features.from_config({"history": True, "physical": ["power", "timing"], "experience": "count", "experience_source": "all"})
    assert f.physical == ("timing", "power") and f.key == "history|physical=timing+power|experience=count:all"
    assert Features().key == "nohistory|physical=none|experience=off:-"
    for bad in ({"physical": ["congestion"]}, {"experience": "words"}, {"experience_source": "some"}):
        with pytest.raises(ValueError):
            Features.from_config(bad)


def test_render_history_view_needs_no_telemetry() -> None:
    view = _view()
    base = build_fuzzy_view(view, mode="fuzzy")
    assert render_history_view(view, base).startswith("budget: used 3")
