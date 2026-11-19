"""The pre-declared v2 selection rule, on synthetic share tables."""
from __future__ import annotations

from scripts import friday_analysis as fa
from scripts import friday_sweep as sweep
from scripts import friday_sweep_select as sel


def _table(shares: dict[tuple[str, str], float], default: float = 0.5):
    """shares[(grid_id, design)] applies to both A0 and T0; default elsewhere."""
    table = {}
    for g in sweep.GRID_IDS:
        for d in fa.DESIGNS:
            for b in sweep.PHASE1_BRIEFS:
                table[(g, d, b)] = {"share": shares.get((g, d), default)}
    return table


def test_knob_changes_count_differences_from_v1() -> None:
    assert sel.knob_changes("g01") == 0
    assert sel.knob_changes("g02") == 1            # continue instead of rollback
    assert sel.knob_changes("g16") == 4            # all four knobs differ


def test_ties_go_to_the_fewest_knob_changes_then_the_lowest_id() -> None:
    pick, tied = sel.choose({"g16": 0.7, "g05": 0.7, "g03": 0.7, "g09": 0.6})
    assert pick == "g03" and tied == ["g03", "g05", "g16"]
    pick, _ = sel.choose({"g01": 0.7, "g02": 0.7})
    assert pick == "g01"
    assert sel.choose({"g01": None}) == (None, [])


def test_leave_one_design_out_trains_on_four_and_scores_the_fifth() -> None:
    # g09 is great everywhere except on the held-out design vexriscv, g01 is flat.
    shares = {("g09", d): 0.9 for d in fa.DESIGNS}
    shares[("g09", fa.DESIGNS[0])] = 0.2
    table = _table(shares, default=0.5)
    result = sel.select(table)
    fold = {row["held_out"]: row for row in result["lodo"]}
    held = fold[fa.DESIGNS[0]]
    assert held["chosen"] == "g09"
    assert abs(held["train_mean"] - 0.9) < 1e-12
    assert abs(held["held_out_result"] - 0.2) < 1e-12 and abs(held["v1_held_out"] - 0.5) < 1e-12
    assert result["v2"]["grid_id"] == "g09"          # (0.9*4 + 0.2) / 5 = 0.76 beats 0.5
    summary = result["lodo_summary"]
    assert summary["folds_complete"] == 5
    assert abs(summary["v1_mean_held_out"] - 0.5) < 1e-12
    expected = (0.2 + 4 * 0.9) / 5
    assert abs(summary["procedure_mean_held_out"] - expected) < 1e-12


def test_a_variant_with_a_missing_cell_is_not_ranked() -> None:
    table = _table({("g09", d): 0.99 for d in fa.DESIGNS})
    del table[("g09", fa.DESIGNS[1], "T0")]
    assert sel.mean_share(table, "g09", fa.DESIGNS) is None
    assert sel.select(table)["v2"]["grid_id"] == "g01"   # all-equal tie -> v1
