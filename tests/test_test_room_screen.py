"""Synthetic checks for the frozen held-out room screen; no test DCP is read."""
from __future__ import annotations

import pytest

from orch.baselines.test_room_screen import noise_bands, screen


ORDER = tuple(f"design_{index}" for index in range(7))
RECIPES = tuple(f"recipe_{index}" for index in range(10))
VARIANTS = ("n_repeat", "n_thr4", "n_unc0.001", "n_unc0.002", "n_unc0.003")


def inputs():
    baselines = {
        design: {"wns_ns": -1.0, "packed_luts": 100.0}
        for design in ORDER
    }
    bands = {
        design: {"wns_ns": 0.02, "packed_luts": 2.0}
        for design in ORDER
    }
    one_steps = {
        design: {
            recipe: {"wns_ns": -1.0, "packed_luts": 100.0}
            for recipe in RECIPES
        }
        for design in ORDER
    }
    return baselines, bands, one_steps


def test_noise_band_is_max_absolute_difference_per_metric() -> None:
    baseline = {
        "wns_ns": -1.0, "tns_ns": -100.0,
        "packed_luts": 100.0, "netlist_luts": 104.0,
    }
    controls = {variant: dict(baseline) for variant in VARIANTS}
    controls["n_repeat"]["wns_ns"] = -1.03
    controls["n_thr4"]["tns_ns"] = -105.0
    controls["n_unc0.001"]["packed_luts"] = 97.0
    controls["n_unc0.003"]["netlist_luts"] = 110.0
    assert noise_bands(baseline, controls, VARIANTS) == {
        "wns_ns": pytest.approx(0.03), "tns_ns": 5.0,
        "packed_luts": 3.0, "netlist_luts": 6.0,
    }


def test_strict_band_and_ordered_five_places() -> None:
    baselines, bands, one_steps = inputs()
    one_steps[ORDER[0]][RECIPES[0]]["wns_ns"] = -0.98
    one_steps[ORDER[1]][RECIPES[0]]["packed_luts"] = 97.0
    for design in ORDER[2:]:
        one_steps[design][RECIPES[1]]["wns_ns"] = -0.95
    flags = screen(
        candidate_order=ORDER, selected_recipe_ids=RECIPES,
        baselines=baselines, bands=bands, one_steps=one_steps,
    )
    assert flags == {
        ORDER[0]: False, ORDER[1]: True, ORDER[2]: True,
        ORDER[3]: True, ORDER[4]: True, ORDER[5]: True, ORDER[6]: False,
    }


def test_incomplete_or_nonfinite_inputs_fail_closed() -> None:
    baselines, bands, one_steps = inputs()
    one_steps[ORDER[0]].pop(RECIPES[0])
    with pytest.raises(ValueError, match="lacks one of ten"):
        screen(candidate_order=ORDER, selected_recipe_ids=RECIPES,
               baselines=baselines, bands=bands, one_steps=one_steps)
    controls = {variant: {
        "wns_ns": 0.0, "tns_ns": 0.0,
        "packed_luts": 0.0, "netlist_luts": 0.0,
    } for variant in VARIANTS}
    controls.pop(VARIANTS[0])
    with pytest.raises(ValueError, match="all five"):
        noise_bands(controls[VARIANTS[1]], controls, VARIANTS)


def test_failed_design_is_not_admitted_but_other_designs_can_fill_room() -> None:
    baselines, bands, one_steps = inputs()
    for design in ORDER:
        one_steps[design][RECIPES[0]]["packed_luts"] = 95.0
    one_steps[ORDER[0]] = {}
    flags = screen(
        candidate_order=ORDER, selected_recipe_ids=RECIPES,
        baselines=baselines, bands=bands, one_steps=one_steps,
        ineligible_designs=(ORDER[0],),
    )
    assert flags[ORDER[0]] is False
    assert [design for design in ORDER if flags[design]] == list(ORDER[1:6])
