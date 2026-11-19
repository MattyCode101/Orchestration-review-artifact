"""Selection logic of the Saturday LLM report, on synthetic result rows."""
from __future__ import annotations

import math
from collections import Counter

import pytest

from scripts import friday_analysis as fa
from scripts import saturday_llm_report as rep

D = fa.DESIGNS


class FakeScorer(rep.Scorer):
    def __init__(self) -> None:          # no stores needed
        pass

    def share(self, row):
        return row["_share"]

    def gain(self, row):
        return row["_gain"]

    def band(self, design, brief):
        return 0.1


def row(config, design, brief, share, *, tag="r0", fold=None, gain=None, choices=(), final="s"):
    return {
        "policy": config, "design": design, "brief": brief, "_share": share, "_gain": share if gain is None else gain,
        "cell": {"tag": tag, "fold": fold}, "runs_used": 4, "new_real_runs_reserved": 1, "stop_reason": "policy_stop",
        "final_state_id": final,
        "llm": {"tokens": {"total": 100}, "decisions": len(choices), "repairs": 0, "fallbacks": 0, "choices": list(choices)},
    }


def grid(config, per_design, *, fold_mode=False):
    """per_design: share for every A0/T0 cell of that design."""
    rows = []
    for design in D:
        for brief in ("A0", "T0"):
            if config == "F2" and fold_mode:                 # one cell per (fold, design)
                for fold in D:
                    rows.append(row(config, design, brief, per_design(design, fold), fold=fold))
            else:
                rows.append(row(config, design, brief, per_design(design, None)))
    return rows


def test_entropy_of_a_uniform_choice_distribution_is_log_two_of_the_count() -> None:
    assert rep.entropy(Counter({"a": 5, "b": 5, "c": 5, "d": 5})) == pytest.approx(2.0)
    assert rep.entropy(Counter({"a": 9})) == 0.0 and rep.entropy(Counter()) == 0.0


def test_choice_keys_cover_stops_workers_and_recipes() -> None:
    assert rep.choice_key({"action": "STOP", "strategy": "STOP", "mandate": None}) == ("STOP",)
    assert rep.choice_key({"action": "INVOKE_WORKER", "strategy": "PUSH",
                           "mandate": {"worker": "area", "preserve_placement": True}}) == ("area", True, "PUSH")
    assert rep.choice_key({"action": "RUN_RECIPE", "strategy": "REPAIR", "mandate": {"recipe": "r1"}}) == ("r1", "REPAIR")


def test_the_lodo_choice_uses_the_four_other_designs_and_reports_the_held_out_design() -> None:
    # F1 is flat; F0 is much better everywhere except on the held-out design vexriscv, where it is worse
    rows = grid("F1", lambda d, f: 0.5) + grid("F0", lambda d, f: 0.2 if d == D[0] else 0.9)
    result = rep.lodo(rows, FakeScorer(), ["F1", "F0"])
    fold = {f["held_out"]: f for f in result["folds"]}
    assert fold[D[0]]["chosen"] == "F0" and fold[D[0]]["held_out_result"] == pytest.approx(0.2)
    assert fold[D[1]]["chosen"] == "F0"                    # F0 averages (0.2 + 3 * 0.9) / 4 = 0.725 on the other four
    assert result["held_out_mean_by_config"]["F0"] == pytest.approx((0.2 + 4 * 0.9) / 5)
    assert result["held_out_mean_by_config"]["F1"] == pytest.approx(0.5)


def test_ties_go_to_f1_first() -> None:
    rows = grid("F1", lambda d, f: 0.5) + grid("F0", lambda d, f: 0.5)
    assert all(f["chosen"] == "F1" for f in rep.lodo(rows, FakeScorer(), ["F0", "F1"])["folds"])


def test_f2_is_scored_with_the_lessons_of_the_fold_that_excludes_the_held_out_design() -> None:
    # F2 does well only when its fold equals the design being scored (an honest held-out cell);
    # when a fold's lessons are used on another design it does badly. The selection stage must use the fold.
    def f2(design, fold):
        return 0.9 if fold == design else 0.1

    rows = grid("F1", lambda d, f: 0.5) + grid("F2", f2, fold_mode=True)
    result = rep.lodo(rows, FakeScorer(), ["F1", "F2"])
    fold = {f["held_out"]: f for f in result["folds"]}
    # on the four training designs F2 (fold = held-out) scores 0.1, so F1 is chosen, and F1 is scored on the held-out design
    assert fold[D[0]]["chosen"] == "F1" and fold[D[0]]["held_out_result"] == pytest.approx(0.5)
    # as a configuration, F2's own-fold cells are its honest held-out cells
    assert result["held_out_mean_by_config"]["F2"] == pytest.approx(0.9)


def _tables(f1_bits, other_bits):
    return {"F1": {"diversity_bits": f1_bits}, "F0": {"diversity_bits": other_bits}, "F2": {"diversity_bits": other_bits}}


def test_f1_stays_main_unless_another_beats_it_by_five_points_without_lower_diversity() -> None:
    held = {"F1": 0.40, "F0": 0.46, "F2": 0.443}
    assert rep.decide_main(_tables(2.0, 2.5), held)["main"] == "F0"            # +6 points, more diverse
    assert rep.decide_main(_tables(2.0, 1.5), held)["main"] == "F1"            # +6 points but less diverse
    assert rep.decide_main(_tables(2.0, 2.5), {"F1": 0.40, "F0": 0.449})["main"] == "F1"   # +4.9 points
    best = rep.decide_main(_tables(2.0, 2.5), {"F1": 0.40, "F0": 0.46, "F2": 0.52})
    assert best["main"] == "F2"                                                 # the largest qualifying gain wins


def test_win_tie_loss_uses_the_noise_band() -> None:
    sc = FakeScorer()
    me, other = row("F1", D[0], "T0", 0.5, gain=1.00), row("v2", D[0], "T0", 0.5, gain=0.95)
    assert sc.outcome_vs(me, other) == "tie"                                    # inside the band of 0.1
    assert sc.outcome_vs(row("F1", D[0], "T0", 0.5, gain=1.2), other) == "win"
    assert sc.outcome_vs(row("F1", D[0], "T0", 0.5, gain=0.7), other) == "loss"


def test_variance_is_the_sample_standard_deviation_of_the_share_across_repeats() -> None:
    rows = [row("F1", D[0], "A0", s, tag=t, final=f) for s, t, f in ((0.5, "r0", "x"), (0.6, "v1", "x"), (0.7, "v2", "y"))]
    rows += [row("F1", D[1], "A0", 0.4, tag=t, final="z") for t in ("r0", "v1", "v2")]
    result = rep.variance(rows, "F1", FakeScorer())
    assert result["cells_with_repeats"] == 2 and result["identical_final_cells"] == 1
    assert result["mean_share_std_points"] == pytest.approx((0.1 + 0.0) / 2 * 100.0)
