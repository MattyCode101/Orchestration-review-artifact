"""The Saturday extension of the repair-loop sweep and the authors' revised v2 rule."""
from __future__ import annotations

from scripts import friday_analysis as fa
from scripts import friday_sweep as sweep
from scripts import friday_sweep_select as old_select
from scripts import saturday_sweep_ta as ext
from scripts import saturday_v2_selection as sel


def test_extension_runs_every_other_variant_on_T_and_A_once() -> None:
    cells = ext.rows()
    assert len(cells) == 14 * 5 * 2 == len(set(cells))
    assert {brief for _, brief, _ in cells} == {"T", "A"}
    assert {grid for _, _, grid in cells} == set(sweep.GRID_IDS) - {"g01", "g14"}


def test_candidates_come_first_in_criterion_order() -> None:
    scores = {g: 0.1 for g in sweep.GRID_IDS}
    scores.update({"g06": 0.9, "g13": 0.8, "g05": 0.7, "g14": 0.99, "g01": 0.5})
    assert ext.variant_order(scores)[:3] == ["g06", "g13", "g05"]
    assert "g14" not in ext.variant_order(scores) and "g01" not in ext.variant_order(scores)


def test_ties_in_the_order_go_to_the_fewest_knob_changes() -> None:
    scores = {g: 0.1 for g in sweep.GRID_IDS}
    scores.update({"g16": 0.5, "g05": 0.5, "g03": 0.5})
    order = ext.variant_order(scores)
    assert order[:3] == ["g03", "g05", "g16"]


def test_episode_ids_never_collide_with_the_friday_cells() -> None:
    friday = {sweep.episode_id(*row) for row in sweep.phase1_rows()}
    mine = {sweep.episode_id(*row) for row in ext.rows()}
    assert not friday & mine
    assert sweep.episode_id("vexriscv_re-place", "T", "g14") not in mine


def test_rebinding_is_scoped_and_restored() -> None:
    before = (sweep.ROOT, sweep.CAP, sweep.SCRIPT, sweep.PHASE1_BRIEFS, dict(sweep.SHARDS))
    with ext.rebound():
        assert sweep.ROOT == ext.ROOT and sweep.CAP == ext.CAP
        assert sweep.PHASE1_BRIEFS == ("T", "A")
        assert sweep.phase1_rows() == ext.rows()
        assert "scripts/saturday_sweep_ta.py" in sweep.SOURCE_FILES.values()
        assert "scripts/friday_sweep.py" in sweep.SOURCE_FILES.values()
    assert (sweep.ROOT, sweep.CAP, sweep.SCRIPT, sweep.PHASE1_BRIEFS, dict(sweep.SHARDS)) == before


def test_the_friday_campaign_directory_is_never_the_extension_root() -> None:
    assert ext.ROOT != sweep.ROOT and ext.ROOT.parent == sweep.ROOT.parent


def _shares(default: float = 0.5, **overrides: float) -> dict[tuple[str, str, str], float]:
    """overrides: name 'g06_T=0.3' sets that variant's brief share on every design."""
    table = {}
    for g in sweep.GRID_IDS:
        for d in fa.DESIGNS:
            for b in ("A0", "T0", "T", "A"):
                table[(g, d, b)] = overrides.get(f"{g}_{b}", default)
    return table


def test_a_variant_that_fails_a_control_is_skipped_for_the_next_best() -> None:
    shares = _shares(g14_A0=0.9, g14_T0=0.9, g14_A=0.2, g06_A0=0.8, g06_T0=0.8, g06_T=0.5, g06_A=0.5)
    pick, tied, _ = sel.choose_v2(shares, fa.DESIGNS)
    assert pick == "g06"          # g14 scores higher but loses 30 points on A


def test_exactly_five_points_below_v1_still_qualifies() -> None:
    shares = _shares(g05_A0=0.9, g05_T0=0.9, g05_T=0.45, g05_A=0.45)
    assert sel.choose_v2(shares, fa.DESIGNS)[0] == "g05"
    shares = _shares(g05_A0=0.9, g05_T0=0.9, g05_T=0.449, g05_A=0.5)
    assert sel.choose_v2(shares, fa.DESIGNS)[0] == "g01"


def test_if_nothing_qualifies_v2_is_v1() -> None:
    shares = {}
    for g in sweep.GRID_IDS:
        for d in fa.DESIGNS:
            bad = g != "g01"
            shares.update({(g, d, "A0"): 0.9 if bad else 0.5, (g, d, "T0"): 0.9 if bad else 0.5,
                           (g, d, "T"): 0.0 if bad else 0.5, (g, d, "A"): 0.5})
    pick, tied, _ = sel.choose_v2(shares, fa.DESIGNS)
    assert pick == "g01" and tied == ["g01"]


def test_a_gain_on_a_control_never_disqualifies_but_the_two_sided_reading_does() -> None:
    shares = _shares(g05_A0=0.9, g05_T0=0.9, g05_T=0.9, g05_A=0.5)
    assert sel.choose_v2(shares, fa.DESIGNS)[0] == "g05"
    assert sel.choose_v2(shares, fa.DESIGNS, one_sided=False)[0] == "g01"


def test_ties_on_the_criterion_go_to_the_fewest_knob_changes() -> None:
    shares = _shares(g16_A0=0.8, g16_T0=0.8, g05_A0=0.8, g05_T0=0.8)
    pick, tied, _ = sel.choose_v2(shares, fa.DESIGNS)
    assert pick == "g05" and tied == ["g05", "g16"]
    assert old_select.knob_changes("g05") < old_select.knob_changes("g16")


def test_a_missing_cell_removes_the_variant_instead_of_guessing() -> None:
    shares = _shares(g06_A0=0.9, g06_T0=0.9)
    del shares[("g06", fa.DESIGNS[0], "A")]
    assert sel.choose_v2(shares, fa.DESIGNS)[0] == "g01"


def test_leave_one_design_out_applies_the_filter_on_the_four_training_designs() -> None:
    # g06 is great and safe everywhere, except that on the held-out design its A share collapses:
    # the qualification must look only at the other four, so g06 is still chosen and then scored.
    shares = _shares(g06_A0=0.9, g06_T0=0.9)
    shares[("g06", fa.DESIGNS[0], "A")] = 0.0
    shares[("g06", fa.DESIGNS[0], "A0")] = 0.1
    result = sel.select(shares)
    fold = {row["held_out"]: row for row in result["lodo"]}
    held = fold[fa.DESIGNS[0]]
    assert held["chosen"] == "g06"
    assert abs(held["held_out_result"] - (0.1 + 0.9) / 2) < 1e-12
    assert abs(held["v1_held_out"] - 0.5) < 1e-12
    # on all five designs g06's A share is 4/5 * 0.5 = 0.4: 10 points below v1, so it does not qualify
    assert result["v2"]["grid_id"] == "g01"
    assert result["v2_two_sided"]["grid_id"] == "g01"
    assert result["lodo_summary"]["folds_complete"] == 5
