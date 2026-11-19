"""Pre-declared repair-loop variant knobs and the three-worker repair rule.

The authors' four knobs (2026-11-06): repair_attempts 1|2, repair_preserve_placement
true|false, gentle_step on|off, after_failed_repair rollback|continue. The
defaults are the frozen v1 behaviour, which keeps version 1.0-frozen.
"""
from __future__ import annotations

import hashlib

import pytest

from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import (
    Brief, BriefTerm, MetricName, Objective, VerdictStatus, WorkerCard,
)
from orch.contract.views import CallSummary, PolicyView, StateSummary
from orch.policies.repair_loop import RepairLoopPolicy

WNS = MetricName.WNS_NS
LUTS = MetricName.PACKED_LUTS
POWER = MetricName.DYNAMIC_POWER_W


def _state(
    label: str, wns: float, luts: float, *, valid: bool = True,
    physical: bool = True, power: float | None = None,
) -> StateSummary:
    metrics = {WNS: wns, LUTS: luts}
    if power is not None:
        metrics[POWER] = power
    return StateSummary(
        state_id=hashlib.sha256(label.encode()).hexdigest(),
        metrics=metrics, physically_valid=physical, brief_valid=valid,
    )


def _card(worker_id: str, metric: MetricName, direction: str) -> WorkerCard:
    return WorkerCard(
        id=worker_id, version="test", recipe_set_id=worker_id,
        objective=Objective(metric=metric, direction=direction),
    )


TIMING = _card("timing", WNS, "max")
AREA = _card("area", LUTS, "min")
POWER_CARD = _card("power", POWER, "min")

P0_LIKE = Brief(
    id="P0-like",
    objective=Objective(metric=POWER, direction="min"),
    terms=(
        BriefTerm(id="p0_limit_wns_ns", metric=WNS, comparison=">="),
        BriefTerm(id="p0_limit_packed_luts", metric=LUTS, comparison="<="),
    ),
)


def _view(
    *, brief: Brief, cards: tuple[WorkerCard, ...],
    current: StateSummary, best: StateSummary, visited: tuple[StateSummary, ...],
    calls: tuple[CallSummary, ...] = (), budget: int = 6,
    start: StateSummary | None = None,
) -> PolicyView:
    return PolicyView(
        brief=brief, budget_left=budget, start=start or visited[0], current=current,
        best_valid=best, visited=visited, calls=calls, worker_cards=cards,
        episode_id="variant-synthetic", fake=True,
    )


def _t0_view(**kwargs) -> PolicyView:
    return _view(brief=load_frozen_brief("T0"), cards=(TIMING, AREA), **kwargs)


def _call(mandate, selected: StateSummary | None, status, charged: int) -> CallSummary:
    return CallSummary(
        mandate_id=mandate.id, worker_id=mandate.worker_id,
        base_state_id=mandate.base_state_id, settings=mandate.settings,
        selected_state_id=selected.state_id if selected else None,
        selected_verdict_status=status, recipe_runs=charged,
        context_ref=mandate.context_ref,
    )


def _after_bold_violation(policy: RepairLoopPolicy):
    """Start, a bold that violates the LUT limit, and the repair mandate."""
    start = _state("start", -1.0, 100)
    bold = policy.decide(_t0_view(current=start, best=start, visited=(start,)))
    violating = _state("violating", -0.5, 110, valid=False)
    bold_call = _call(bold, violating, VerdictStatus.BRIEF_VIOLATION, 2)
    repair = policy.decide(_t0_view(
        current=violating, best=start, visited=(start, violating),
        calls=(bold_call,), budget=4,
    ))
    return start, violating, bold_call, repair


def test_default_policy_is_the_frozen_v1_identity() -> None:
    policy = RepairLoopPolicy()
    assert policy.version == "1.0-frozen"
    explicit = RepairLoopPolicy(
        repair_attempts=1, repair_preserve_placement=True,
        gentle_step=True, after_failed_repair="rollback",
    )
    assert explicit.version == "1.0-frozen"
    start = _state("start", -1.0, 100)
    view = _t0_view(current=start, best=start, visited=(start,))
    assert policy.decide(view) == explicit.decide(view)


def test_every_non_default_knob_changes_the_version_and_bad_values_are_refused() -> None:
    versions = {
        RepairLoopPolicy().version,
        RepairLoopPolicy(repair_attempts=2).version,
        RepairLoopPolicy(repair_preserve_placement=False).version,
        RepairLoopPolicy(gentle_step=False).version,
        RepairLoopPolicy(after_failed_repair="continue").version,
    }
    assert len(versions) == 5
    with pytest.raises(ValueError):
        RepairLoopPolicy(repair_attempts=3)
    with pytest.raises(ValueError):
        RepairLoopPolicy(after_failed_repair="loop")


def test_repair_attempts_two_gives_the_repair_call_two_runs() -> None:
    _, violating, _, repair = _after_bold_violation(RepairLoopPolicy(repair_attempts=2))
    assert repair.worker_id == "area" and repair.base_state_id == violating.state_id
    assert repair.settings.max_runs == 2 and repair.settings.preserve_placement


def test_repair_may_replace_placement_when_the_knob_is_off() -> None:
    _, _, _, repair = _after_bold_violation(
        RepairLoopPolicy(repair_preserve_placement=False)
    )
    assert repair.worker_id == "area"
    assert repair.settings.preserve_placement is False
    assert repair.settings.max_runs == 1


def test_gentle_off_stops_where_v1_would_try_the_gentle_step() -> None:
    start = _state("start", -1.0, 100)
    v1 = RepairLoopPolicy()
    bold = v1.decide(_t0_view(current=start, best=start, visited=(start,)))
    worse = _state("worse", -1.1, 100)
    view = _t0_view(
        current=worse, best=start, visited=(start, worse),
        calls=(_call(bold, worse, VerdictStatus.ACCEPTABLE, 2),), budget=4,
    )
    gentle = v1.decide(view)
    assert gentle.settings.preserve_placement and gentle.settings.max_runs == 1
    no_gentle = RepairLoopPolicy(gentle_step=False)
    bold_off = no_gentle.decide(_t0_view(current=start, best=start, visited=(start,)))
    view_off = _t0_view(
        current=worse, best=start, visited=(start, worse),
        calls=(_call(bold_off, worse, VerdictStatus.ACCEPTABLE, 2),), budget=4,
    )
    assert no_gentle.decide(view_off) is None


def test_gentle_off_stops_after_a_repair_that_did_not_help() -> None:
    policy = RepairLoopPolicy(gentle_step=False)
    start, violating, bold_call, repair = _after_bold_violation(policy)
    unhelpful = _state("unhelpful", -1.2, 105, valid=False)
    view = _t0_view(
        current=unhelpful, best=start, visited=(start, violating, unhelpful),
        calls=(bold_call, _call(repair, unhelpful, VerdictStatus.BRIEF_VIOLATION, 1)),
        budget=3,
    )
    assert policy.decide(view) is None


def _after_failed_repair_view(policy, repaired: StateSummary):
    start, violating, bold_call, repair = _after_bold_violation(policy)
    view = _t0_view(
        current=repaired, best=start, visited=(start, violating, repaired),
        calls=(bold_call, _call(repair, repaired, VerdictStatus.BRIEF_VIOLATION, 1)),
        budget=3,
    )
    return start, view


def test_rollback_tries_the_gentle_step_from_the_best_valid_state() -> None:
    policy = RepairLoopPolicy()
    repaired = _state("repaired", -0.6, 105, valid=False)
    start, view = _after_failed_repair_view(policy, repaired)
    nxt = policy.decide(view)
    assert nxt.worker_id == "timing" and nxt.base_state_id == start.state_id
    assert nxt.settings.preserve_placement and nxt.settings.max_runs == 1


def test_continue_runs_the_next_bold_step_from_the_improving_repaired_state() -> None:
    policy = RepairLoopPolicy(after_failed_repair="continue")
    repaired = _state("repaired", -0.6, 105, valid=False)
    _, view = _after_failed_repair_view(policy, repaired)
    nxt = policy.decide(view)
    assert nxt.worker_id == "timing" and nxt.base_state_id == repaired.state_id
    assert nxt.settings.preserve_placement is False and nxt.settings.max_runs == 2
    assert '"phase":"bold"' in nxt.context_ref
    assert nxt == policy.decide(view)


def test_continue_rolls_back_when_the_repaired_state_did_not_improve_the_objective() -> None:
    policy = RepairLoopPolicy(after_failed_repair="continue")
    repaired = _state("repaired", -1.2, 105, valid=False)
    start, view = _after_failed_repair_view(policy, repaired)
    nxt = policy.decide(view)
    assert nxt.base_state_id == start.state_id
    assert nxt.settings.preserve_placement and nxt.settings.max_runs == 1


def test_continue_rolls_back_when_the_repaired_state_is_not_physically_valid() -> None:
    policy = RepairLoopPolicy(after_failed_repair="continue")
    repaired = _state("repaired", -0.6, 105, valid=False, physical=False)
    start, view = _after_failed_repair_view(policy, repaired)
    assert policy.decide(view).base_state_id == start.state_id


def test_a_variant_refuses_a_journaled_context_from_another_version() -> None:
    v1 = RepairLoopPolicy()
    start = _state("start", -1.0, 100)
    bold = v1.decide(_t0_view(current=start, best=start, visited=(start,)))
    worse = _state("worse", -1.1, 100)
    view = _t0_view(
        current=worse, best=start, visited=(start, worse),
        calls=(_call(bold, worse, VerdictStatus.ACCEPTABLE, 2),), budget=4,
    )
    with pytest.raises(ValueError, match="invalid cycle context"):
        RepairLoopPolicy(repair_attempts=2).decide(view)


def _p0_view(*, calls=(), visited, current=None, best=None, budget=6,
             cards=(POWER_CARD, TIMING, AREA)) -> PolicyView:
    return _view(
        brief=P0_LIKE, cards=cards, start=visited[0],
        current=current or visited[-1], best=best or visited[0],
        visited=visited, calls=calls, budget=budget,
    )


def test_three_worker_primary_is_the_worker_with_the_brief_objective() -> None:
    start = _state("p0-start", -1.0, 100, power=1.0)
    first = RepairLoopPolicy().decide(_p0_view(visited=(start,)))
    assert first.worker_id == "power" and first.base_state_id == start.state_id
    assert first.settings.preserve_placement is False and first.settings.max_runs == 2


@pytest.mark.parametrize(
    ("violation", "wns", "luts", "expected"),
    [
        ("wns", -1.3, 100, "timing"),
        ("luts", -1.0, 110, "area"),
        ("both", -1.3, 110, "timing"),
    ],
)
def test_three_worker_repair_goes_to_the_worker_of_the_violated_limit(
    violation: str, wns: float, luts: float, expected: str,
) -> None:
    policy = RepairLoopPolicy()
    start = _state("p0-start", -1.0, 100, power=1.0)
    bold = policy.decide(_p0_view(visited=(start,)))
    violating = _state("p0-" + violation, wns, luts, valid=False, power=0.9)
    view = _p0_view(
        visited=(start, violating), current=violating, budget=4,
        calls=(_call(bold, violating, VerdictStatus.BRIEF_VIOLATION, 2),),
    )
    repair = policy.decide(view)
    assert repair.worker_id == expected
    assert repair.base_state_id == violating.state_id
    assert repair.settings.preserve_placement and repair.settings.max_runs == 1


def test_three_worker_view_without_a_worker_for_a_limit_is_refused() -> None:
    start = _state("p0-start", -1.0, 100, power=1.0)
    view = _p0_view(visited=(start,), cards=(POWER_CARD, TIMING))
    with pytest.raises(ValueError, match="distinct primary and repair worker cards"):
        RepairLoopPolicy().decide(view)
