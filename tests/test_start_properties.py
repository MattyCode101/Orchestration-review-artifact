"""One definition of the placement-property validity rule (decision of 2026-11-07).

An output state may not have more DONT_TOUCH, fixed LOC or fixed BEL cells than the
start state of its episode. The rule replaces "zero", which the training designs
satisfy only because every training start has zero.
"""

from __future__ import annotations

import itertools

import pytest

from orch.contract.models import Evidence, MetricName, MetricValue
from orch.engine.start_properties import (
    PROPERTY_KEYS, PropertyRuleError, check_not_above_start, check_start_counts,
)
from scripts import friday_test_power as power_driver


def _evidence(props, *, state: str = "a" * 64) -> Evidence:
    """Built without validation: the model coerces True and "1" to int, but the rule must not rely on that."""

    metrics = {name: MetricValue(value=1.0, unit="unit", method="meter") for name in MetricName}
    return Evidence.model_construct(
        id="e" + state[:4], state_id=state, metrics=metrics, routed=True, route_errors=0,
        drc_errors=0, constraint_fingerprint="constraint", clock_digest="clock",
        v_props=props, meter_version="meter", fake=False,
    )


def _props(dont_touch: int, loc: int, bel: int) -> dict[str, int]:
    return {"n_DONT_TOUCH": dont_touch, "n_IS_LOC_FIXED": loc, "n_IS_BEL_FIXED": bel}


def test_the_three_property_keys_are_the_meter_keys() -> None:
    assert PROPERTY_KEYS == ("n_DONT_TOUCH", "n_IS_LOC_FIXED", "n_IS_BEL_FIXED")


@pytest.mark.parametrize("n", [0, 1, 7])
def test_output_with_exactly_the_start_counts_passes(n: int) -> None:
    check_not_above_start(_evidence(_props(n, n, n), state="b" * 64), _evidence(_props(n, n, n)))


@pytest.mark.parametrize("n", [0, 4])
@pytest.mark.parametrize("key", PROPERTY_KEYS)
def test_output_with_one_more_cell_than_the_start_fails_for_each_property(key: str, n: int) -> None:
    start = _evidence(_props(n, n, n))
    output = _evidence({**_props(n, n, n), key: n + 1}, state="b" * 64)
    with pytest.raises(PropertyRuleError, match="more DONT_TOUCH/fixed LOC/BEL cells than the start state"):
        check_not_above_start(output, start)


def test_output_with_fewer_cells_than_the_start_passes() -> None:
    check_not_above_start(_evidence(_props(1, 0, 2), state="b" * 64), _evidence(_props(3, 3, 3)))


def test_a_deficit_in_one_property_does_not_offset_a_surplus_in_another() -> None:
    start = _evidence(_props(5, 0, 2))
    with pytest.raises(PropertyRuleError, match="more DONT_TOUCH"):
        check_not_above_start(_evidence(_props(0, 1, 0), state="b" * 64), start)


@pytest.mark.parametrize(
    "props",
    [
        None,
        {"n_DONT_TOUCH": 0, "n_IS_LOC_FIXED": 0},
        {**_props(0, 0, 0), "n_IS_BEL_FIXED": True},
        {**_props(0, 0, 0), "n_IS_LOC_FIXED": 1.0},
        {**_props(0, 0, 0), "n_DONT_TOUCH": "0"},
        {**_props(0, 0, 0), "n_DONT_TOUCH": -1},
    ],
)
def test_output_counts_must_be_present_nonnegative_integers(props) -> None:
    with pytest.raises(PropertyRuleError, match="missing DONT_TOUCH/fixed LOC/BEL counts"):
        check_not_above_start(_evidence(props, state="b" * 64), _evidence(_props(9, 9, 9)))


@pytest.mark.parametrize(
    "props",
    [None, {"n_DONT_TOUCH": 1}, {**_props(0, 0, 0), "n_IS_BEL_FIXED": False}, {**_props(0, 0, 0), "n_DONT_TOUCH": -2}],
)
def test_start_counts_must_be_present_nonnegative_integers(props) -> None:
    with pytest.raises(PropertyRuleError, match="start state lacks DONT_TOUCH/fixed LOC/BEL counts"):
        check_not_above_start(_evidence(_props(0, 0, 0), state="b" * 64), _evidence(props))
    with pytest.raises(PropertyRuleError, match="start state lacks DONT_TOUCH/fixed LOC/BEL counts"):
        check_start_counts(_evidence(props))


def test_a_start_may_carry_any_number_of_fixed_cells() -> None:
    assert check_start_counts(_evidence(_props(12, 40, 3))) == _props(12, 40, 3)


def test_rule_agrees_with_the_test_power_baseline_driver_on_a_grid() -> None:
    """Same accept and reject verdicts as friday_test_power.check_evidence (its inline rule)."""

    plan = {"meter_version": "meter"}
    values = [0, 1, 2, 3, 4]
    starts = [(0, 0, 0), (1, 0, 3), (2, 2, 2)]
    for start_counts in starts:
        start = _evidence(_props(*start_counts))
        inputs = {
            "b2_clock_digest": "clock", "b2_constraint_fingerprint": "constraint",
            "b2_v_props": _props(*start_counts),
        }
        candidates = [_props(*combo) for combo in itertools.product(values, repeat=3)]
        candidates += [
            None, {"n_DONT_TOUCH": 0}, {**_props(0, 0, 0), "n_IS_LOC_FIXED": True},
            {**_props(0, 0, 0), "n_IS_BEL_FIXED": 1.0}, {**_props(0, 0, 0), "n_DONT_TOUCH": "1"},
        ]
        for props in candidates:
            output = _evidence(props, state="b" * 64)
            try:
                power_driver.check_evidence(output, "b" * 64, plan, inputs)
                driver_accepts = True
            except ValueError:
                driver_accepts = False
            try:
                check_not_above_start(output, start)
                rule_accepts = True
            except PropertyRuleError:
                rule_accepts = False
            assert rule_accepts == driver_accepts, (start_counts, props)
