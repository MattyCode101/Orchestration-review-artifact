"""Fixed-order and random baselines over a brief's relevant workers (P0 has three)."""
from __future__ import annotations

import hashlib

import pytest

from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import MetricName, Objective, VerdictStatus, WorkerCard
from orch.contract.views import CallSummary, PolicyView, StateSummary
from orch.policies.relevant_workers import (
    FixedOrderPolicy, RelevantRandomPolicy, relevant_cards,
)
from orch.policies.training_baselines import SingleSpecialistPolicy

WNS, LUTS, POWER = MetricName.WNS_NS, MetricName.PACKED_LUTS, MetricName.DYNAMIC_POWER_W


def _state(label: str, wns: float, luts: float, power: float, *, valid=True) -> StateSummary:
    return StateSummary(
        state_id=hashlib.sha256(label.encode()).hexdigest(),
        metrics={WNS: wns, LUTS: luts, POWER: power},
        physically_valid=True, brief_valid=valid,
    )


def _card(worker_id: str, metric: MetricName, direction: str) -> WorkerCard:
    return WorkerCard(
        id=worker_id, version="test", recipe_set_id=worker_id,
        objective=Objective(metric=metric, direction=direction),
    )


TIMING, AREA, POWER_CARD = (
    _card("timing", WNS, "max"), _card("area", LUTS, "min"), _card("power", POWER, "min"),
)


def _p0_view(*, visited, calls=(), budget=6, cards=(TIMING, AREA, POWER_CARD)) -> PolicyView:
    return PolicyView(
        brief=load_frozen_brief("P0"), budget_left=budget, start=visited[0],
        current=visited[-1], best_valid=visited[0], visited=tuple(visited),
        calls=tuple(calls), worker_cards=cards, episode_id="relevant-synthetic", fake=True,
    )


def _call(mandate, selected: StateSummary, charged: int) -> CallSummary:
    return CallSummary(
        mandate_id=mandate.id, worker_id=mandate.worker_id,
        base_state_id=mandate.base_state_id, settings=mandate.settings,
        selected_state_id=selected.state_id,
        selected_verdict_status=VerdictStatus.ACCEPTABLE, recipe_runs=charged,
        context_ref=mandate.context_ref,
    )


def test_relevant_cards_are_the_primary_then_one_worker_per_limit() -> None:
    start = _state("s", -1.0, 100, 1.0)
    assert [c.id for c in relevant_cards(_p0_view(visited=(start,)))] == [
        "power", "timing", "area",
    ]
    t0 = PolicyView(
        brief=load_frozen_brief("T0"), budget_left=6, start=start, current=start,
        best_valid=start, visited=(start,), worker_cards=(TIMING, AREA, POWER_CARD),
        episode_id="x", fake=True,
    )
    assert [c.id for c in relevant_cards(t0)] == ["timing", "area"]


def test_fixed_order_cycles_the_three_workers_from_the_latest_state() -> None:
    policy = FixedOrderPolicy(order="CAB")
    assert policy.id == "fixed_CAB"
    start = _state("s", -1.0, 100, 1.0)
    first = policy.decide(_p0_view(visited=(start,)))
    assert (first.worker_id, first.base_state_id) == ("power", start.state_id)
    assert first.settings.max_runs == 2 and first.settings.preserve_placement is False
    second_state = _state("c1", -1.2, 100, 0.95, valid=False)
    view = _p0_view(visited=(start, second_state), budget=4,
                    calls=(_call(first, second_state, 2),))
    second = policy.decide(view)
    assert (second.worker_id, second.base_state_id) == ("timing", second_state.state_id)
    third_state = _state("c2", -1.0, 110, 0.95, valid=False)
    view = _p0_view(visited=(start, second_state, third_state), budget=2,
                    calls=(_call(first, second_state, 2), _call(second, third_state, 2)))
    assert policy.decide(view).worker_id == "area"
    done = _p0_view(visited=(start, second_state, third_state), budget=0,
                    calls=(_call(first, second_state, 2), _call(second, third_state, 2)))
    assert policy.decide(done) is None


@pytest.mark.parametrize("order", ["CAB", "CBA", "ACB", "ABC", "BCA", "BAC"])
def test_every_permutation_starts_with_its_own_first_worker(order: str) -> None:
    start = _state("s", -1.0, 100, 1.0)
    first = FixedOrderPolicy(order=order).decide(_p0_view(visited=(start,)))
    assert first.worker_id == {"C": "power", "A": "timing", "B": "area"}[order[0]]


def test_invalid_orders_are_refused() -> None:
    for bad in ("CA", "CAA", "XYZ", "CABB"):
        with pytest.raises(ValueError):
            FixedOrderPolicy(order=bad)


def test_relevant_random_is_replayable_and_uses_all_three_workers() -> None:
    start = _state("s", -1.0, 100, 1.0)
    latest = _state("l", -0.9, 105, 0.97, valid=False)
    view = _p0_view(visited=(start, latest))
    first = RelevantRandomPolicy(seed=7).decide(view)
    assert first == RelevantRandomPolicy(seed=7).decide(view)
    choices = [RelevantRandomPolicy(seed=s).decide(view) for s in range(150)]
    assert {c.worker_id for c in choices} == {"power", "timing", "area"}
    assert {c.settings.preserve_placement for c in choices} == {False, True}
    assert {c.base_state_id for c in choices} == {start.state_id, latest.state_id}
    assert RelevantRandomPolicy(seed=1).version != RelevantRandomPolicy(seed=2).version


def test_single_specialist_uses_the_power_worker_on_p0() -> None:
    start = _state("s", -1.0, 100, 1.0)
    mandate = SingleSpecialistPolicy().decide(_p0_view(visited=(start,)))
    assert mandate.worker_id == "power" and mandate.settings.max_runs == 2
