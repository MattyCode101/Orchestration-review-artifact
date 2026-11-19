"""The stronger-model rule (decision of 2026-11-07, item 4) and the determinism comparison, on synthetic numbers."""
from __future__ import annotations

import pytest

from scripts.satnight_report import (
    apply_rule, basis_a0_t0, cell_means, compare_replies, four_brief_mean, spend_by_stage,
)

CELLS_A0T0 = [(d, b) for d in ("d1", "d2", "d3", "d4", "d5") for b in ("A0", "T0")]
CELLS_ALL = [(d, b) for d in ("d1", "d2", "d3", "d4", "d5") for b in ("A0", "T0", "T", "A")]


def _run(value: float, cells=CELLS_ALL) -> dict:
    return {cell: value for cell in cells}


def test_cell_means_average_each_cell_over_the_runs_that_cover_it() -> None:
    partial = {("d1", "A0"): 0.2}
    full = {("d1", "A0"): 0.4, ("d1", "T0"): 0.6}
    means = cell_means([full, partial])
    assert means[("d1", "A0")] == pytest.approx(0.3) and means[("d1", "T0")] == pytest.approx(0.6)


def test_the_basis_averages_r0_and_r1_after_averaging_r1_over_its_runs() -> None:
    r0 = _run(0.40)
    r1_runs = [_run(0.60), _run(0.50), _run(0.40)]          # R1 first: mean 0.50
    assert basis_a0_t0(r0, r1_runs) == pytest.approx((0.40 + 0.50) / 2)


def test_only_the_criterion_briefs_count_in_the_basis() -> None:
    r0 = {**_run(0.40), ("d1", "T"): 0.99, ("d1", "A"): 0.99}
    assert basis_a0_t0(r0, [_run(0.40)]) == pytest.approx(0.40)


def test_the_four_brief_mean_uses_every_cell_of_f1_r1() -> None:
    runs = [{**_run(0.50), ("d1", "T"): 0.10}, _run(0.50)]
    expected = (sum(cell_means(runs).values())) / len(CELLS_ALL)
    assert four_brief_mean(runs) == pytest.approx(expected) and expected < 0.50


def test_the_rule_switches_only_when_both_conditions_hold() -> None:
    flash_r0, flash_r1 = _run(0.455), [_run(0.526)]          # basis 0.4905
    both = apply_rule(flash_r0, flash_r1, _run(0.50), [_run(0.60)])
    assert both["gain_points"] == pytest.approx((0.55 - 0.4905) * 100) and both["condition_i"] is True
    assert both["condition_ii"] is True and both["switch"] is True
    only_i = apply_rule(flash_r0, flash_r1, _run(0.50), [_run(0.60, CELLS_A0T0) | {c: 0.40 for c in CELLS_ALL if c[1] in ("T", "A")}])
    assert only_i["condition_i"] is True and only_i["condition_ii"] is False and only_i["switch"] is False
    neither = apply_rule(flash_r0, flash_r1, _run(0.45), [_run(0.45)])
    assert neither["switch"] is False


def test_the_gain_threshold_is_five_points_exactly() -> None:
    flash = apply_rule(_run(0.40), [_run(0.40)], _run(0.45), [_run(0.45)])
    assert flash["gain_points"] == pytest.approx(5.0) and flash["condition_i"] is True
    below = apply_rule(_run(0.40), [_run(0.40)], _run(0.449), [_run(0.449)])
    assert below["condition_i"] is False


def test_replies_to_the_same_prompt_are_compared_by_text() -> None:
    recorded = {"k1": '{"a": 1}', "k2": '{"a": 2}', "k3": '{"a": 3}'}
    fresh = {"k1": '{"a": 1}', "k2": '{"a": 99}', "k4": '{"a": 4}'}
    result = compare_replies(recorded, fresh)
    assert result["prompts_in_both"] == 2 and result["identical"] == 1 and result["differing"] == 1
    assert result["any_differs"] is True and result["fresh_only"] == 1
    assert compare_replies({"k": "x"}, {"k": "x"})["any_differs"] is False


def test_spend_is_summed_by_stage_from_the_ledger() -> None:
    records = [{"source": "llm_ar1f_F1_d_A0_r0", "usd": 0.5}, {"source": "llm_ar1f_F1_d_T0_r0", "usd": 0.25},
               {"source": "llm_ar0f_F1_d_A0_r0", "usd": 1.0}, {"source": "smoke", "usd": 0.02}]
    totals = spend_by_stage(records)
    assert totals["ar1f"] == pytest.approx(0.75) and totals["ar0f"] == pytest.approx(1.0) and totals["smoke"] == pytest.approx(0.02)
