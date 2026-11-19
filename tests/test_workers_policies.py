"""Behavioral tests for fixed-order workers and view-only policies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from orch.contract.models import (
    AppliedStep,
    Brief,
    BriefTerm,
    DesignState,
    Evidence,
    Guard,
    Mandate,
    MetricName,
    MetricValue,
    Objective,
    RunOutcome,
    RunRecord,
    RunStatus,
    WorkerCard,
    WorkerSettings,
)
from orch.contract.views import PolicyView, StateSummary
from orch.engine.gate import judge
from orch.interfaces import RecipeSpec
from orch.policies.random_policy import RandomPolicy
from orch.policies.trivial import TrivialPolicy
from orch.workers.base import GenericWorker, run_worker


START_ID = "a" * 64


def evidence(state_id: str, *, wns: float, luts: float) -> Evidence:
    return Evidence(
        id=f"evidence_{state_id[:12]}",
        state_id=state_id,
        metrics={
            MetricName.WNS_NS: MetricValue(
                value=wns, unit="ns", method="fake-test", basis="routed"
            ),
            MetricName.PACKED_LUTS: MetricValue(
                value=luts, unit="count", method="fake-test", basis="routed"
            ),
        },
        routed=True,
        route_errors=0,
        drc_errors=0,
        constraint_fingerprint="xdc",
        clock_digest="clock",
        meter_version="fake-test-v1",
        fake=True,
    )


def start_state() -> DesignState:
    return DesignState(id=START_ID, design="test-design", path="fake://start", fake=True)


def card(worker_id: str, metric: MetricName, direction: str) -> WorkerCard:
    return WorkerCard(
        id=worker_id,
        version="test-v1",
        objective=Objective(metric=metric, direction=direction),
        recipe_set_id=worker_id,
        may_replace=True,
    )


def area_brief() -> Brief:
    return Brief(
        id="test-area",
        objective=Objective(metric=MetricName.PACKED_LUTS, direction="min"),
        terms=(
            BriefTerm(
                id="wns_floor",
                metric=MetricName.WNS_NS,
                comparison=">=",
                factor=1.0,
                offset=-0.1,
            ),
        ),
        budget_runs=6,
    )


def timing_brief() -> Brief:
    return Brief(
        id="test-timing",
        objective=Objective(metric=MetricName.WNS_NS, direction="max"),
        terms=(
            BriefTerm(
                id="lut_cap",
                metric=MetricName.PACKED_LUTS,
                comparison="<=",
                factor=1.01,
            ),
        ),
        budget_runs=6,
    )


def recipe(recipe_id: str, *, worker: str, replaces: bool = False) -> RecipeSpec:
    return RecipeSpec(
        id=recipe_id,
        worker=worker,
        kind="tcl",
        script=f"fake/{recipe_id}",
        replaces=replaces,
        cause="fake test",
        source="fake test",
        source_sha256="fake",
        timeout_s=10,
        version="test-recipe-v1",
    )


def mandate(
    brief: Brief,
    *,
    worker: str,
    preserve_placement: bool = False,
    max_runs: int = 2,
    guards: tuple[Guard, ...] = (),
) -> Mandate:
    return Mandate(
        id="mandate-test",
        worker_id=worker,
        base_state_id=START_ID,
        settings=WorkerSettings(
            preserve_placement=preserve_placement,
            max_runs=max_runs,
        ),
        guards=guards,
        brief_hash=brief.fingerprint(),
    )


@dataclass
class StubRegistry:
    entries: dict[str, RecipeSpec]

    def get(self, recipe_id: str) -> RecipeSpec:
        return self.entries[recipe_id]


class StubEngine:
    """Return measured fake outcomes; record only the recipe calls."""

    def __init__(self, outputs: dict[str, tuple[float, float] | RunStatus], *, cache_hit: str = ""):
        self.outputs = outputs
        self.cache_hit = cache_hit
        self.calls: list[str] = []

    def apply(self, state, recipe, *, brief, start, parent, guards=(), noise=None):
        self.calls.append(recipe.id)
        specification = self.outputs[recipe.id]
        status = specification if isinstance(specification, RunStatus) else RunStatus.OK
        out_state = None
        out_evidence = None
        if status == RunStatus.OK:
            new_id = hashlib.sha256(recipe.id.encode("utf-8")).hexdigest()
            out_state = DesignState(
                id=new_id,
                design=state.design,
                parent_id=state.id,
                path=f"fake://{recipe.id}",
                fake=True,
            )
            wns, luts = specification
            out_evidence = evidence(new_id, wns=wns, luts=luts)
        run = RunRecord(
            id=f"run_{recipe.id}",
            recipe_id=recipe.id,
            recipe_version=recipe.version,
            status=status,
            wall_s=3.0,
            input_sha256_before=state.id,
            input_sha256_after=state.id,
            cache_hit=recipe.id == self.cache_hit,
            fake=True,
        )
        outcome = RunOutcome(run=run, state=out_state, evidence=out_evidence)
        verdict = judge(
            brief,
            start,
            parent,
            out_evidence,
            guards=guards,
            noise=noise,
            run_status=status,
        )
        return AppliedStep(outcome=outcome, verdict=verdict)


def run(
    worker: GenericWorker,
    *,
    brief: Brief,
    registry: StubRegistry,
    engine: StubEngine,
    mandate_: Mandate,
    remaining: int = 6,
    history: tuple[AppliedStep, ...] = (),
):
    base = start_state()
    base_evidence = evidence(START_ID, wns=-0.785, luts=100)
    return run_worker(
        worker,
        mandate=mandate_,
        engine=engine,
        registry=registry,
        base_state=base,
        base_evidence=base_evidence,
        start_evidence=base_evidence,
        brief=brief,
        remaining_budget=remaining,
        history_on_state=history,
    )


def test_worker_filters_replacing_and_charges_cache_hit() -> None:
    brief = area_brief()
    entries = {
        "replace": recipe("replace", worker="area", replaces=True),
        "post1": recipe("post1", worker="area"),
        "post2": recipe("post2", worker="area"),
    }
    worker = GenericWorker(
        card("area", MetricName.PACKED_LUTS, "min"), tuple(entries)
    )
    engine = StubEngine(
        {"replace": (-0.78, 85), "post1": (-0.78, 90), "post2": (-0.78, 80)},
        cache_hit="post1",
    )
    result = run(
        worker,
        brief=brief,
        registry=StubRegistry(entries),
        engine=engine,
        mandate_=mandate(
            brief, worker="area", preserve_placement=True, max_runs=1
        ),
    )
    assert engine.calls == ["post1"]
    assert result.planned_recipe_ids == ("post1", "post2")
    assert result.excluded_replacing == ("replace",)
    assert result.recipe_runs == 1
    assert result.charged_wall_s == 3.0
    assert result.attempts[0].outcome.run.cache_hit
    assert result.selected is result.attempts[0]


def test_worker_can_return_brief_invalid_intermediate_but_respects_guard() -> None:
    brief = timing_brief()
    entries = {
        "fast_over_cap": recipe("fast_over_cap", worker="timing"),
        "slower_under_cap": recipe("slower_under_cap", worker="timing"),
    }
    worker = GenericWorker(card("timing", MetricName.WNS_NS, "max"), tuple(entries))
    output = {
        "fast_over_cap": (-0.70, 103),
        "slower_under_cap": (-0.72, 100),
    }
    unguarded = run(
        worker,
        brief=brief,
        registry=StubRegistry(entries),
        engine=StubEngine(output),
        mandate_=mandate(brief, worker="timing"),
    )
    assert unguarded.selected is unguarded.attempts[0]
    assert not unguarded.selected.verdict.brief_valid
    assert unguarded.selected.verdict.physically_valid

    guarded = run(
        worker,
        brief=brief,
        registry=StubRegistry(entries),
        engine=StubEngine(output),
        mandate_=mandate(
            brief,
            worker="timing",
            guards=(
                Guard(
                    id="cap",
                    metric=MetricName.PACKED_LUTS,
                    comparison="<=",
                    threshold=101,
                ),
            ),
        ),
    )
    assert not guarded.attempts[0].verdict.guard_pass
    assert guarded.selected is guarded.attempts[1]


def test_worker_uses_history_and_does_not_charge_transient() -> None:
    brief = area_brief()
    entries = {
        "old": recipe("old", worker="area"),
        "transient": recipe("transient", worker="area"),
        "next": recipe("next", worker="area"),
    }
    worker = GenericWorker(card("area", MetricName.PACKED_LUTS, "min"), tuple(entries))
    prior = StubEngine({"old": (-0.78, 98)}).apply(
        start_state(),
        entries["old"],
        brief=brief,
        start=evidence(START_ID, wns=-0.785, luts=100),
        parent=evidence(START_ID, wns=-0.785, luts=100),
    )
    engine = StubEngine({"transient": RunStatus.TRANSIENT, "next": (-0.78, 93)})
    result = run(
        worker,
        brief=brief,
        registry=StubRegistry(entries),
        engine=engine,
        mandate_=mandate(brief, worker="area", max_runs=1),
        remaining=1,
        history=(prior,),
    )
    assert engine.calls == ["transient", "next"]
    assert result.planned_recipe_ids == ("transient", "next")
    assert result.recipe_runs == 1
    assert result.charged_wall_s == 3.0
    assert len(result.attempts) == 2
    assert result.selected is result.attempts[1]
    assert worker.plan(start_state(), None, mandate(brief, worker="area"), (prior, result.attempts[0])) == ("transient", "next")


def test_worker_rejects_wrong_recipe_owner_before_running() -> None:
    brief = area_brief()
    worker = GenericWorker(
        card("area", MetricName.PACKED_LUTS, "min"), ("foreign",)
    )
    engine = StubEngine({"foreign": (-0.78, 90)})
    with pytest.raises(ValueError, match="belongs"):
        run(
            worker,
            brief=brief,
            registry=StubRegistry(
                {"foreign": recipe("foreign", worker="timing")}
            ),
            engine=engine,
            mandate_=mandate(brief, worker="area"),
        )
    assert engine.calls == []


def policy_view(*, budget_left: int = 4) -> PolicyView:
    brief = area_brief()
    start = StateSummary(
        state_id=START_ID,
        metrics={MetricName.WNS_NS: -0.785, MetricName.PACKED_LUTS: 100},
        physically_valid=True,
        brief_valid=True,
    )
    current = StateSummary(
        state_id="b" * 64,
        metrics={MetricName.WNS_NS: -0.70, MetricName.PACKED_LUTS: 104},
        physically_valid=True,
        brief_valid=False,
    )
    return PolicyView(
        brief=brief,
        budget_left=budget_left,
        start=start,
        current=current,
        best_valid=start,
        visited=(start, current),
        worker_cards=(
            card("timing", MetricName.WNS_NS, "max"),
            card("area", MetricName.PACKED_LUTS, "min"),
        ),
        episode_id="episode_test",
        fake=True,
    )


def test_trivial_policy_uses_brief_objective_and_latest_physical_state() -> None:
    view = policy_view()
    policy = TrivialPolicy()
    decision = policy.decide(view)
    assert decision is not None
    assert decision.worker_id == "area"
    assert decision.base_state_id == view.current.state_id
    assert decision.settings.max_runs == 2
    decision.validate_against(view.brief, evidence(START_ID, wns=-0.785, luts=100))
    assert policy.decide(view) == decision


def test_random_policy_replays_same_choice_from_same_seed_and_view() -> None:
    view = policy_view()
    first = RandomPolicy(seed=42).decide(view)
    second = RandomPolicy(seed=42).decide(view)
    assert first == second
    assert first is not None
    assert first.base_state_id == view.current.state_id
    assert first.worker_id in {"area", "timing"}
    assert 1 <= first.settings.max_runs <= 2
    assert first.guards == ()


@pytest.mark.parametrize("policy", [TrivialPolicy(), RandomPolicy(seed=7)])
def test_policies_stop_at_zero_budget(policy) -> None:
    assert policy.decide(policy_view(budget_left=0)) is None



def test_worker_does_not_select_incomplete_brief_evidence() -> None:
    """A physically routed state with an unknown brief metric is unusable."""
    brief = area_brief()
    chosen = recipe("partial", worker="timing")
    worker = GenericWorker(
        card("timing", MetricName.WNS_NS, "max"), (chosen.id,)
    )
    start = evidence(START_ID, wns=-0.785, luts=100)
    applied = StubEngine({"partial": (-0.70, 99)}).apply(
        start_state(),
        chosen,
        brief=brief,
        start=start,
        parent=start,
    )
    assert applied.outcome.evidence is not None
    partial = applied.outcome.evidence.model_copy(
        update={"metrics": {MetricName.WNS_NS: applied.outcome.evidence.metrics[MetricName.WNS_NS]}}
    )
    verdict = judge(brief, start, start, partial)
    assert verdict.physically_valid
    assert verdict.status.value == "INVALID"
    candidate = AppliedStep(
        outcome=RunOutcome(
            run=applied.outcome.run,
            state=applied.outcome.state,
            evidence=partial,
        ),
        verdict=verdict,
    )
    assert worker.select((candidate,), mandate(brief, worker="timing")) is None
