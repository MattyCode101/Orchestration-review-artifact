"""Pure-view behavior tests for fixed repair and training baselines."""
from __future__ import annotations

import hashlib

from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import MetricName, Objective, VerdictStatus, WorkerCard
from orch.contract.views import CallSummary, PolicyView, StateSummary
from orch.policies.repair_loop import RepairLoopPolicy
from orch.policies.training_baselines import (
    FixedAlternationPolicy, SingleSpecialistPolicy, UniformRandomPolicy,
)


def _state(
    label: str, wns: float, luts: float, *, valid: bool = True,
) -> StateSummary:
    return StateSummary(
        state_id=hashlib.sha256(label.encode()).hexdigest(),
        metrics={MetricName.WNS_NS: wns, MetricName.PACKED_LUTS: luts},
        physically_valid=True, brief_valid=valid,
    )


def _cards() -> tuple[WorkerCard, WorkerCard]:
    return (
        WorkerCard(
            id="timing", version="test", recipe_set_id="timing",
            objective=Objective(metric=MetricName.WNS_NS, direction="max"),
        ),
        WorkerCard(
            id="area", version="test", recipe_set_id="area",
            objective=Objective(metric=MetricName.PACKED_LUTS, direction="min"),
        ),
    )


def _view(
    *, current: StateSummary, best: StateSummary,
    visited: tuple[StateSummary, ...], calls: tuple[CallSummary, ...] = (),
    budget: int = 6,
) -> PolicyView:
    return PolicyView(
        brief=load_frozen_brief("T0"), budget_left=budget,
        start=visited[0], current=current, best_valid=best,
        visited=visited, calls=calls, worker_cards=_cards(),
        episode_id="repair-synthetic", fake=True,
    )


def _call(
    mandate, selected: StateSummary | None, status: VerdictStatus | None,
    charged: int,
) -> CallSummary:
    return CallSummary(
        mandate_id=mandate.id, worker_id=mandate.worker_id,
        base_state_id=mandate.base_state_id, settings=mandate.settings,
        selected_state_id=selected.state_id if selected else None,
        selected_verdict_status=status, recipe_runs=charged,
        context_ref=mandate.context_ref,
    )


def test_repair_loop_bold_violation_then_repair_then_new_cycle() -> None:
    start = _state("start", -1.0, 100)
    policy = RepairLoopPolicy()
    first = policy.decide(_view(current=start, best=start, visited=(start,)))
    assert first.worker_id == "timing"
    assert first.base_state_id == start.state_id
    assert not first.settings.preserve_placement
    assert first.settings.max_runs == 2 and first.guards == ()

    invalid = _state("invalid", -0.5, 110, valid=False)
    first_call = _call(first, invalid, VerdictStatus.BRIEF_VIOLATION, 2)
    view = _view(
        current=invalid, best=start, visited=(start, invalid),
        calls=(first_call,), budget=4,
    )
    repair = policy.decide(view)
    assert repair.worker_id == "area"
    assert repair.base_state_id == invalid.state_id
    assert repair.settings.preserve_placement
    assert repair.settings.max_runs == 1 and repair.guards == ()

    recovered = _state("recovered", -0.6, 100)
    repaired = _call(repair, recovered, VerdictStatus.ACCEPTABLE, 1)
    view = _view(
        current=recovered, best=recovered,
        visited=(start, invalid, recovered),
        calls=(first_call, repaired), budget=3,
    )
    next_bold = policy.decide(view)
    assert next_bold.worker_id == "timing"
    assert next_bold.base_state_id == recovered.state_id
    assert next_bold.settings.max_runs == 2
    assert next_bold == policy.decide(view)


def test_repair_loop_gentle_duplicate_skip_and_two_failed_cycles_stop() -> None:
    start = _state("start", -1.0, 100)
    policy = RepairLoopPolicy()
    bold = policy.decide(_view(current=start, best=start, visited=(start,)))
    worse = _state("worse", -1.1, 100)
    bold_call = _call(bold, worse, VerdictStatus.ACCEPTABLE, 2)
    view = _view(
        current=worse, best=start, visited=(start, worse),
        calls=(bold_call,), budget=4,
    )
    gentle = policy.decide(view)
    assert gentle.worker_id == "timing"
    assert gentle.base_state_id == start.state_id
    assert gentle.settings.preserve_placement and gentle.settings.max_runs == 1
    still_worse = _state("still-worse", -1.2, 100)
    gentle_call = _call(gentle, still_worse, VerdictStatus.ACCEPTABLE, 1)
    view = _view(
        current=still_worse, best=start,
        visited=(start, worse, still_worse),
        calls=(bold_call, gentle_call), budget=3,
    )
    assert policy.decide(view) is None
    assert policy.decide(_view(current=start, best=start, visited=(start,), budget=1)) is None


def test_single_specialist_and_fixed_orders_use_latest_result_without_rollback() -> None:
    start = _state("start", -1.0, 100)
    invalid = _state("invalid", -0.5, 110, valid=False)
    first_view = _view(current=start, best=start, visited=(start,))
    specialist = SingleSpecialistPolicy()
    first = specialist.decide(first_view)
    assert (first.worker_id, first.base_state_id, first.settings.max_runs) == (
        "timing", start.state_id, 2,
    )
    call = _call(first, invalid, VerdictStatus.BRIEF_VIOLATION, 2)
    next_view = _view(
        current=invalid, best=start, visited=(start, invalid),
        calls=(call,), budget=4,
    )
    second = specialist.decide(next_view)
    assert second.base_state_id == invalid.state_id

    ab = FixedAlternationPolicy(first="A")
    ba = FixedAlternationPolicy(first="B")
    assert ab.decide(first_view).worker_id == "timing"
    assert ba.decide(first_view).worker_id == "area"
    assert ab.decide(next_view).worker_id == "area"
    assert ba.decide(next_view).worker_id == "timing"
    assert ab.decide(next_view).base_state_id == invalid.state_id


def test_seeded_random_is_replayable_and_uses_all_four_uniform_choices() -> None:
    start = _state("start", -1.0, 100)
    latest = _state("latest", -0.9, 105, valid=False)
    view = _view(current=latest, best=start, visited=(start, latest))
    first = UniformRandomPolicy(seed=7).decide(view)
    assert first == UniformRandomPolicy(seed=7).decide(view)
    choices = [UniformRandomPolicy(seed=seed).decide(view) for seed in range(100)]
    assert {choice.worker_id for choice in choices} == {"timing", "area"}
    assert {choice.settings.preserve_placement for choice in choices} == {False, True}
    assert {choice.settings.max_runs for choice in choices} == {1, 2}
    assert {choice.base_state_id for choice in choices} == {
        start.state_id, latest.state_id,
    }
    low = _view(current=latest, best=start, visited=(start, latest), budget=1)
    assert UniformRandomPolicy(seed=7).decide(low).settings.max_runs == 1
