"""Template v2 experience: one test per build-review defect it fixes (1 to 5, and the evidence word of 9).

Where the round-2 code (orch/policies/r2_experience.py, kept unchanged) shows the defect, a test also asserts
that it does, so each fix is shown against the behaviour it replaces.
"""
from __future__ import annotations

import json

import pytest

from orch.policies import r2_experience as r2
from orch.policies.v2_experience import (
    CALLS_SCHEMA_V2, CallV2, line_stats, load_calls_v2, render_experience_v2, render_line,
)
from orch.policies.v2_vocab import categorise_v2, limit_status_v2

LUT, WNS = "packed_luts", "wns_ns"
BANDS = {LUT: 10.0, WNS: 0.05}


def area_call(design: str, lut_delta: float, *, wns_delta: float = 0.0, scope: str = "preserve_placement_true",
              outcome: str = "selected", brief: str = "A0", bands: dict | None = None) -> CallV2:
    base = {LUT: 1000.0, WNS: -0.2}
    result = None if outcome != "selected" else {LUT: 1000.0 + lut_delta, WNS: -0.2 + wns_delta}
    return CallV2(design, brief, "area", scope, outcome, base, result, dict(BANDS if bands is None else bands))


def render(calls, **kw):
    args = dict(brief="A0", goal=LUT, limits=[WNS], workers=["area"], exclude_design=None, header="H", intro="I",
                empty_text="none")
    args.update(kw)
    return render_experience_v2(calls, **args)


def skewed_calls() -> list[CallV2]:
    """One design (d1) gains large every time; three others never do."""

    calls = [area_call("d1", -50.0) for _ in range(6)]
    for design in ("d2", "d3", "d4"):
        calls += [area_call(design, 0.0) for _ in range(14)]
    return calls


def test_defect1_lines_pool_calls_so_one_design_cannot_set_the_word() -> None:
    text, _ = render(skewed_calls())
    assert "large gain rarely" in text                  # 6 of 48 calls = 12.5%
    # the round-2 design-equal average gives the one design a quarter of the weight: "sometimes"
    old = [r2.Call(c.design, "area", c.scope, True, {"packed_luts": ("large_gain" if c.result_values[LUT] < 990 else
                                                                      "within_noise")}) for c in skewed_calls()]
    old_text = r2.render_experience_section(r2.build_table(old, "x"), goal="packed_luts", limits=[], workers=["area"],
                                            form="fuzzy")
    assert "large_gain sometimes" in old_text


def test_defect1_and_9_every_line_says_n_calls_from_k_designs() -> None:
    text, meta = render(skewed_calls())
    assert "(48 calls from 4 designs)" in text
    assert meta["shown"] == [{"worker": "area", "scope": "preserve_placement_true", "calls": 48, "designs": 4}]
    # round 2 said "some observations" from the call count alone
    assert r2.strength_word(48) == "many observations" and r2.strength_word(20) == "some observations"


def test_defect1_a_line_from_fewer_than_four_designs_is_left_out() -> None:
    calls = [c for c in skewed_calls() if c.design != "d4"]
    text, meta = render(calls)
    assert "calls from 3 designs" not in text and text.endswith("none")
    assert meta["omitted"][0]["designs"] == 3 and meta["shown"] == []


def test_defect1_the_evaluated_design_is_left_out_before_counting() -> None:
    calls = skewed_calls() + [area_call("d5", 0.0) for _ in range(3)]
    text, meta = render(calls, exclude_design="d1")
    assert "(45 calls from 4 designs)" in text and "large gain" not in text
    assert meta["excluded_design"] == "d1" and "d1" not in meta["designs_in_scope"]


def test_defect2_very_large_tier_separates_three_bands_from_eighty() -> None:
    assert categorise_v2(-30.0, 10.0, higher_is_better=False) == "small gain"       # exactly 3 bands
    assert categorise_v2(-31.0, 10.0, higher_is_better=False) == "large gain"
    assert categorise_v2(-300.0, 10.0, higher_is_better=False) == "large gain"      # exactly 30 bands
    assert categorise_v2(-800.0, 10.0, higher_is_better=False) == "very large gain"
    assert categorise_v2(0.4, 0.05, higher_is_better=True) == "large gain"
    assert categorise_v2(-0.049, 0.05, higher_is_better=True) == "within noise"
    calls = [area_call(d, -800.0) for d in ("d1", "d2", "d3", "d4")] + [area_call(d, -40.0) for d in ("d1", "d2")]
    text, _ = render(calls)
    assert "very large gain often" in text and "large gain sometimes" in text


def test_defect3_no_change_counts_only_calls_with_a_goal_category() -> None:
    banded = [area_call(d, -50.0) for d in ("d1", "d2", "d3", "d4")]
    unbanded = [area_call("d5", 0.0, bands={}) for _ in range(4)]     # a design with no band: no category
    stats = line_stats(banded + unbanded, worker="area", scope="preserve_placement_true", goal=LUT, limits=[WNS], v2=True)
    assert stats.no_change_base == 4 and stats.no_change == 0 and stats.calls == 8 and stats.designs == 5
    text, _ = render(banded + unbanded)
    assert "no change rarely" in text
    # round 2 counted the band-less calls as "no change" with a full design weight
    old = [r2.Call(c.design, "area", c.scope, True, {} if c.design == "d5" else {"packed_luts": "large_gain"})
           for c in banded + unbanded]
    table = r2.build_table(old, "x")
    assert table.get("area", "preserve_placement_true").no_change_rate == 0.0      # empty changes: not quiet
    old_quiet = [r2.Call("d5", "area", "preserve_placement_true", True, {"power": "within_noise"})]
    assert r2.build_table(old_quiet, "x").get("area", "preserve_placement_true").no_change_rate == 1.0


def test_defect4_nothing_selectable_is_its_own_outcome() -> None:
    calls = [area_call(d, -50.0) for d in ("d1", "d2", "d3", "d4")]
    calls += [area_call(d, 0.0, outcome="nothing_selectable") for d in ("d1", "d2", "d3", "d4")]
    stats = line_stats(calls, worker="area", scope="preserve_placement_true", goal=LUT, limits=[WNS], v2=True)
    assert (stats.usable, stats.nothing_selectable, stats.no_usable) == (4, 4, 0)
    text, _ = render(calls)
    assert "nothing selectable often" in text and "large gain usually" in text     # changes: shares of results


def test_defect4_calls_are_per_brief() -> None:
    calls = [area_call(d, -50.0) for d in ("d1", "d2", "d3", "d4")]
    calls += [area_call(d, 0.0, brief="A") for d in ("d1", "d2", "d3", "d4")]
    text, meta = render(calls, brief="A")
    assert meta["calls_in_scope"] == 4 and "within noise usually" in text


def test_defect5_calls_that_could_not_run_count_as_no_usable_result() -> None:
    calls = [area_call(d, -50.0) for d in ("d1", "d2", "d3", "d4")]
    calls += [area_call(d, 0.0, outcome="no_options") for d in ("d1", "d2")]
    calls += [area_call("d3", 0.0, outcome="not_stored")]
    stats = line_stats(calls, worker="area", scope="preserve_placement_true", goal=LUT, limits=[WNS], v2=True)
    assert stats.no_usable == 3 and stats.calls == 7
    text, _ = render(calls)
    assert "no usable result often" in text                                        # 3 of 7 = 43%


def test_round1_vocabulary_and_placement_words_are_available_for_the_ladder() -> None:
    calls = [area_call(d, -800.0) for d in ("d1", "d2", "d3", "d4")]
    text, _ = render(calls, v2=False, placement_v2=False)
    assert "large_gain usually" in text and "placement preserved (4 calls from 4 designs)" in text
    text2, _ = render(calls)
    assert "very large gain usually" in text2 and "placement must not change (4 calls from 4 designs)" in text2


def test_exact_form_shows_medians_and_no_category_words() -> None:
    calls = [area_call(d, -10.0 * (i + 1), wns_delta=0.01) for i, d in enumerate(("d1", "d2", "d3", "d4"))]
    stats = line_stats(calls, worker="area", scope="preserve_placement_true", goal=LUT, limits=[WNS], v2=True)
    line = render_line(stats, goal=LUT, limits=[WNS], form="exact", v2=True, placement_v2=True)
    assert "median -2.5%" in line and "limit wns: median +0.010 ns" in line
    assert "gain" not in line and "noise" not in line


def test_the_record_loader_checks_schema_and_outcomes(tmp_path) -> None:
    good = {"schema_version": CALLS_SCHEMA_V2, "calls": [
        {"design": "d1", "brief": "A0", "worker": "area", "scope": "preserve_placement_true", "outcome": "selected",
         "base_values": {LUT: 1.0}, "result_values": {LUT: 0.0}, "bands": {LUT: 1.0}}]}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(good))
    calls, sha = load_calls_v2(path)
    assert len(calls) == 1 and len(sha) == 64
    bad = dict(good, calls=[dict(good["calls"][0], outcome="nothing_selectable")])
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_calls_v2(path)


def test_limit_statuses_v2_edges() -> None:
    assert limit_status_v2(0.0, 10.0, start_relative=True) == "at limit (equal to the start)"
    assert limit_status_v2(0.0, 10.0, start_relative=False) == "at limit"
    assert limit_status_v2(10.0, 10.0, start_relative=True) == "within limit by under one band"
    assert limit_status_v2(10.5, 10.0, start_relative=True) == "within limit with margin"
    assert limit_status_v2(-10.0, 10.0, start_relative=True) == "broken by under one band"
    assert limit_status_v2(-10.5, 10.0, start_relative=True) == "broken by more"
