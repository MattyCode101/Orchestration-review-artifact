"""End-to-end software tests for episodes, resume and final-only selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from orch.contract.models import (
    Brief,
    BriefTerm,
    DesignState,
    EquivalenceCheck,
    Evidence,
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
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.journal import Journal
from orch.engine.store import StateStore
from orch.episode import FinalValidation, run_episode
from orch.interfaces import RecipeSpec
from orch.policies.random_policy import RandomPolicy
from orch.policies.trivial import TrivialPolicy
from orch.workers.base import GenericWorker


@dataclass
class StubRegistry:
    entries: dict[str, RecipeSpec]

    def get(self, recipe_id: str) -> RecipeSpec:
        return self.entries[recipe_id]


class MiniBackend:
    """Deterministic test-only backend; never invokes a hardware tool."""

    meter_version = "mini-backend-v1"
    effects = {
        "timing": (0.05, 2.0),
        "area": (-0.01, -3.0),
        "a1": (-0.01, -3.0),
        "a2": (-0.02, -5.0),
    }

    def __init__(self, store: StateStore, *, fake: bool = True, fail_on_call: int | None = None):
        self.store = store
        self.fake = fake
        self.fail_on_call = fail_on_call
        self.metrics: dict[str, tuple[float, float]] = {}
        self.calls: list[str] = []
        self.start_measure_calls = 0

    def create_start(self, folder: Path) -> DesignState:
        source = folder / "start.dcp"
        source.write_bytes(b"unit-test-start-state")
        state = self.store.import_state(source, design="unit-test", fake=self.fake)
        self.metrics[state.id] = (-1.0, 100.0)
        return state

    def _evidence(self, state: DesignState) -> Evidence:
        wns, luts = self.metrics[state.id]
        digest = hashlib.sha256(f"{state.id}:{wns}:{luts}".encode()).hexdigest()
        return Evidence(
            id=digest,
            state_id=state.id,
            metrics={
                MetricName.WNS_NS: MetricValue(
                    value=wns, unit="ns", method="unit-test-backend", basis="routed"
                ),
                MetricName.PACKED_LUTS: MetricValue(
                    value=luts, unit="count", method="unit-test-backend", basis="routed"
                ),
            },
            routed=True,
            route_errors=0,
            drc_errors=0,
            constraint_fingerprint="unit-xdc",
            clock_digest="unit-clocks",
            meter_version=self.meter_version,
            fake=self.fake,
        )

    def measure_start(self, state: DesignState, *, run_dir: Path) -> Evidence:
        self.start_measure_calls += 1
        return self._evidence(state)

    def execute(
        self, state: DesignState, recipe: RecipeSpec, *, run_id: str, run_dir: Path
    ) -> RunOutcome:
        self.calls.append(recipe.id)
        if self.fail_on_call == len(self.calls):
            self.fail_on_call = None
            raise RuntimeError("simulated crash between worker attempts")
        wns, luts = self.metrics[state.id]
        delta_wns, delta_luts = self.effects[recipe.id]
        output = run_dir / "out.dcp"
        output.write_bytes(f"{state.id}:{recipe.id}".encode())
        successor = self.store.import_state(
            output,
            design=state.design,
            parent_id=state.id,
            made_by_run_id=run_id,
            fake=self.fake,
        )
        self.metrics[successor.id] = (wns + delta_wns, luts + delta_luts)
        return RunOutcome(
            run=RunRecord(
                id=run_id,
                recipe_id=recipe.id,
                recipe_version=recipe.version,
                status=RunStatus.OK,
                exit_code=0,
                wall_s=1.25,
                input_sha256_before=state.id,
                input_sha256_after=state.id,
                fake=self.fake,
            ),
            state=successor,
            evidence=self._evidence(successor),
        )


def _recipe(recipe_id: str, worker: str) -> RecipeSpec:
    return RecipeSpec(
        id=recipe_id,
        worker=worker,
        kind="tcl",
        script=f"unit-test/{recipe_id}",
        replaces=False,
        cause="unit test",
        source="unit test",
        source_sha256="unit-test",
        timeout_s=10,
        version="unit-recipe-v1",
    )


def _worker(worker_id: str, metric: MetricName, direction: str, ids: tuple[str, ...]) -> GenericWorker:
    return GenericWorker(
        WorkerCard(
            id=worker_id,
            version="unit-worker-v1",
            objective=Objective(metric=metric, direction=direction),
            recipe_set_id="unit-test-explicit-set",
        ),
        ids,
    )


def _brief(kind: str, *, budget: int = 3) -> Brief:
    if kind == "timing":
        return Brief(
            id="unit-timing",
            objective=Objective(metric=MetricName.WNS_NS, direction="max"),
            terms=(
                BriefTerm(
                    id="lut_cap",
                    metric=MetricName.PACKED_LUTS,
                    comparison="<=",
                    factor=1.01,
                ),
            ),
            budget_runs=budget,
        )
    return Brief(
        id="unit-area",
        objective=Objective(metric=MetricName.PACKED_LUTS, direction="min"),
        terms=(
            BriefTerm(
                id="wns_floor",
                metric=MetricName.WNS_NS,
                comparison=">=",
                offset=-0.1,
            ),
        ),
        budget_runs=budget,
    )


def _setup(tmp_path: Path, *, fake: bool = True, fail_on_call: int | None = None):
    store = StateStore(tmp_path / "data")
    backend = MiniBackend(store, fake=fake, fail_on_call=fail_on_call)
    start = backend.create_start(tmp_path)
    engine = StepEngine(
        store=store,
        cache=RunCache(store.root),
        backend=backend,
        meter_version=backend.meter_version,
        cache_enabled=True,
    )
    entries = {
        recipe_id: _recipe(recipe_id, worker)
        for recipe_id, worker in (
            ("timing", "timing"),
            ("area", "area"),
            ("a1", "area"),
            ("a2", "area"),
        )
    }
    workers = {
        "timing": _worker("timing", MetricName.WNS_NS, "max", ("timing",)),
        "area": _worker("area", MetricName.PACKED_LUTS, "min", ("area",)),
    }
    return engine, StubRegistry(entries), workers, backend, start


@pytest.mark.parametrize("brief_kind", ["timing", "area"])
@pytest.mark.parametrize("policy_kind", ["trivial", "random"])
def test_complete_fake_episode_for_both_policies_and_briefs(
    tmp_path: Path, brief_kind: str, policy_kind: str
) -> None:
    engine, registry, workers, backend, start = _setup(tmp_path)
    brief = _brief(brief_kind)
    policy = (
        TrivialPolicy(max_runs=1)
        if policy_kind == "trivial"
        else RandomPolicy(seed=119, max_runs=2)
    )
    result = run_episode(
        engine=engine,
        registry=registry,
        workers=workers,
        policy=policy,
        brief=brief,
        start_state=start,
        episode_id=f"unit_{brief_kind}_{policy_kind}",
    )
    assert result.fake
    assert result.recipe_runs == brief.budget_runs
    assert result.charged_wall_s == 1.25 * brief.budget_runs
    assert result.stop_reason == "budget_exhausted"
    assert result.validation_status == "skipped_fake"
    assert result.final_verdict is not None and result.final_verdict.brief_valid
    assert result.final_state is not None
    assert result.final_evidence is not None
    assert len(result.steps) == brief.budget_runs
    assert backend.start_measure_calls == 1
    events = Journal(engine.store.root).read(result.episode_id)
    assert sum(event["type"] == "step" for event in events) == brief.budget_runs
    assert events[-1]["type"] == "final"


class RecoveryPolicy:
    id = "unit-recovery"
    version = "1"

    def __init__(self) -> None:
        self.views = []

    def decide(self, view):
        self.views.append(view)
        if len(view.steps) >= 2:
            return None
        worker_id = "timing" if len(view.steps) == 0 else "area"
        return Mandate(
            id=f"mandate_{len(view.steps)}",
            worker_id=worker_id,
            base_state_id=view.current.state_id,
            settings=WorkerSettings(max_runs=1, preserve_placement=True),
            brief_hash=view.brief.fingerprint(),
        )


def test_brief_invalid_timing_state_can_be_recovered_by_area_step(tmp_path: Path) -> None:
    engine, registry, workers, _, start = _setup(tmp_path)
    policy = RecoveryPolicy()
    result = run_episode(
        engine=engine,
        registry=registry,
        workers=workers,
        policy=policy,
        brief=_brief("timing", budget=2),
        start_state=start,
        episode_id="unit_recovery",
    )
    assert len(result.steps) == 2
    assert policy.views[1].current.physically_valid
    assert not policy.views[1].current.brief_valid
    assert policy.views[1].current.state_id == result.steps[0].output_state_id
    assert result.steps[1].parent_state_id == result.steps[0].output_state_id
    assert result.final_rank == 0
    assert result.final_evidence is not None
    assert result.final_evidence.metric(MetricName.WNS_NS).value > -1.0
    assert result.final_verdict is not None and result.final_verdict.brief_valid


class CountingTrivial:
    id = "unit-counting-trivial"
    version = "1"

    def __init__(self, *, max_runs: int) -> None:
        self.delegate = TrivialPolicy(max_runs=max_runs)
        self.calls = 0

    def decide(self, view):
        self.calls += 1
        return self.delegate.decide(view)


def test_resume_mid_worker_uses_journalled_first_attempt(tmp_path: Path) -> None:
    engine, registry, workers, backend, start = _setup(tmp_path, fail_on_call=2)
    workers = {
        "area": _worker("area", MetricName.PACKED_LUTS, "min", ("a1", "a2"))
    }
    policy = CountingTrivial(max_runs=2)
    kwargs = dict(
        engine=engine,
        registry=registry,
        workers=workers,
        policy=policy,
        brief=_brief("area", budget=2),
        start_state=start,
        episode_id="unit_resume",
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_episode(**kwargs)
    events = Journal(engine.store.root).read("unit_resume")
    assert [event["type"] for event in events] == ["start", "mandate", "step"]
    assert policy.calls == 1

    result = run_episode(**kwargs)
    assert result.resumed
    assert result.recipe_runs == 2
    assert len(result.steps) == 2
    assert backend.calls == ["a1", "a2", "a2"]
    assert backend.start_measure_calls == 1
    assert policy.calls == 1


def test_empty_worker_stops_without_looping(tmp_path: Path) -> None:
    engine, registry, _, _, start = _setup(tmp_path)
    workers = {"area": _worker("area", MetricName.PACKED_LUTS, "min", ())}
    result = run_episode(
        engine=engine,
        registry=registry,
        workers=workers,
        policy=TrivialPolicy(max_runs=2),
        brief=_brief("area"),
        start_state=start,
        episode_id="unit_empty",
    )
    assert result.recipe_runs == 0
    assert result.steps == ()
    assert result.stop_reason == "no_recipes_available"
    assert result.final_state is not None and result.final_state.id == start.id
    assert any(
        event["type"] == "stop" and event["reason"] == "no_recipes_available"
        for event in Journal(engine.store.root).read("unit_empty")
    )


def test_validator_failure_falls_back_to_next_brief_valid_state(tmp_path: Path) -> None:
    """A test stub exercises real-branch control flow without hardware tools."""

    engine, registry, _, _, start = _setup(tmp_path, fake=False)
    workers = {"area": _worker("area", MetricName.PACKED_LUTS, "min", ("a1", "a2"))}
    seen = []

    def validate(reference: DesignState, candidate: DesignState) -> FinalValidation:
        seen.append(candidate.id)
        result = "fail" if len(seen) == 1 else "pass"
        return FinalValidation(
            check=EquivalenceCheck(
                result=result,
                method="unit-test-validator",
                vectors=200,
                reference_state_id=reference.id,
                strength="weak",
            ),
            run_id=f"unit_validator_{len(seen)}",
            artifact_dir=str(tmp_path),
        )

    result = run_episode(
        engine=engine,
        registry=registry,
        workers=workers,
        policy=TrivialPolicy(max_runs=2),
        brief=_brief("area", budget=2),
        start_state=start,
        episode_id="unit_validator_fallback",
        validator=validate,
    )
    assert result.validation_status == "pass"
    assert result.final_rank == 1
    assert len(result.validation_attempts) == 2
    assert [item.outcome.check.result for item in result.validation_attempts] == [
        "fail",
        "pass",
    ]
    assert result.final_state is not None
    assert result.final_state.id == seen[1]
    assert len(seen) == 2


def test_two_drivers_of_same_episode_share_one_journalled_run(
    tmp_path: Path,
) -> None:
    """The episode flock prevents two live drivers from repeating decisions."""

    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from time import sleep

    engine, registry, workers, backend, start = _setup(tmp_path)
    barrier = Barrier(2)
    original_execute = backend.execute

    def slow_execute(state, recipe, *, run_id, run_dir):
        sleep(0.05)
        return original_execute(state, recipe, run_id=run_id, run_dir=run_dir)

    backend.execute = slow_execute
    kwargs = dict(
        engine=engine,
        registry=registry,
        workers=workers,
        policy=TrivialPolicy(max_runs=1),
        brief=_brief("area", budget=2),
        start_state=start,
        episode_id="unit_concurrent",
    )

    def drive():
        barrier.wait()
        return run_episode(**kwargs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(drive) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]

    assert sorted(result.resumed for result in results) == [False, True]
    assert len(backend.calls) == 2
    assert backend.start_measure_calls == 1
    assert results[0].steps == results[1].steps
    assert len(Journal(engine.store.root).read("unit_concurrent")) == 9


def test_replay_rejects_recipe_out_of_worker_order(tmp_path: Path) -> None:
    """Even a checksummed forged journal cannot rearrange a worker's plan."""

    from copy import deepcopy

    from orch.engine.store import _json_bytes

    engine, registry, _, _, start = _setup(tmp_path)
    workers = {"area": _worker("area", MetricName.PACKED_LUTS, "min", ("a1", "a2"))}
    kwargs = dict(
        engine=engine,
        registry=registry,
        workers=workers,
        policy=TrivialPolicy(max_runs=2),
        brief=_brief("area", budget=2),
        start_state=start,
        episode_id="unit_replay_order",
    )
    run_episode(**kwargs)
    journal = Journal(engine.store.root)
    events = journal.read("unit_replay_order")
    assert [event["type"] for event in events[:4]] == [
        "start", "mandate", "step", "step"
    ]
    assert events[2]["step"]["index"] == 0
    assert events[3]["step"]["index"] == 1
    forged = deepcopy(events[3])
    forged["step"]["index"] = 0
    forged_id = "step_" + hashlib.sha256(
        _json_bytes(
            [
                "unit_replay_order",
                0,
                forged["step"]["run_id"],
                forged["step"]["verdict_id"],
            ]
        )
    ).hexdigest()
    forged["step"]["id"] = forged_id
    forged["id"] = forged_id

    def journal_line(event):
        return _json_bytes(
            {
                "event": event,
                "sha256": hashlib.sha256(_json_bytes(event)).hexdigest(),
            }
        )

    journal.path("unit_replay_order").write_bytes(
        b"".join(journal_line(event) for event in (events[0], events[1], forged))
    )
    with pytest.raises(ValueError, match="step identity or provenance"):
        run_episode(**kwargs)


def test_old_allow_replace_journal_replays_without_new_tool_calls(tmp_path: Path) -> None:
    """A checksummed pre-rename mandate is accepted on full episode replay."""

    from orch.engine.store import _json_bytes

    engine, registry, workers, backend, start = _setup(tmp_path)
    kwargs = dict(
        engine=engine, registry=registry, workers=workers,
        policy=TrivialPolicy(max_runs=1, preserve_placement=False),
        brief=_brief("timing", budget=1), start_state=start,
        episode_id="unit_legacy_placement",
    )
    first = run_episode(**kwargs)
    before_calls = tuple(backend.calls)
    journal = Journal(engine.store.root)
    events = journal.read("unit_legacy_placement")
    mandate_events = [event for event in events if event["type"] == "mandate"]
    assert len(mandate_events) == 1
    settings = mandate_events[0]["mandate"]["settings"]
    assert settings["preserve_placement"] is False
    settings["allow_replace"] = not settings.pop("preserve_placement")

    def line(event):
        return _json_bytes({
            "event": event,
            "sha256": hashlib.sha256(_json_bytes(event)).hexdigest(),
        })

    journal.path("unit_legacy_placement").write_bytes(
        b"".join(line(event) for event in events)
    )
    replayed = run_episode(**kwargs)
    assert replayed.resumed
    assert replayed.steps == first.steps
    assert tuple(backend.calls) == before_calls


@pytest.mark.parametrize("policy_name", [
    "repair_loop", "single_specialist", "fixed_AB", "fixed_BA", "random",
])
def test_frozen_training_policies_run_and_replay_fake_episodes(
    tmp_path: Path, policy_name: str,
) -> None:
    """All new view-only policies complete under the shared fake harness."""

    import json

    from orch.contract.brief_text import load_frozen_brief
    from orch.policies.repair_loop import RepairLoopPolicy
    from orch.policies.training_baselines import (
        FixedAlternationPolicy, SingleSpecialistPolicy, UniformRandomPolicy,
    )

    policies = {
        "repair_loop": RepairLoopPolicy(),
        "single_specialist": SingleSpecialistPolicy(),
        "fixed_AB": FixedAlternationPolicy(first="A"),
        "fixed_BA": FixedAlternationPolicy(first="B"),
        "random": UniformRandomPolicy(seed=17),
    }
    engine, registry, workers, backend, start = _setup(tmp_path)
    args = dict(
        engine=engine, registry=registry, workers=workers,
        policy=policies[policy_name], brief=load_frozen_brief("T0"),
        start_state=start, episode_id="night_" + policy_name,
    )
    first = run_episode(**args)
    assert first.fake and first.recipe_runs <= 6
    assert first.final_verdict is not None and first.final_verdict.brief_valid
    events = Journal(engine.store.root).read(first.episode_id)
    mandates = [
        event["mandate"] for event in events if event["type"] == "mandate"
    ]
    assert mandates
    if policy_name == "repair_loop":
        stages = [json.loads(row["context_ref"])["phase"] for row in mandates]
        assert stages[:2] == ["bold", "repair"]
        keys = [
            (
                row["worker_id"], row["base_state_id"],
                row["settings"]["preserve_placement"], row["settings"]["max_runs"],
            )
            for row in mandates
        ]
        assert len(keys) == len(set(keys))
    before = tuple(backend.calls)
    replayed = run_episode(**args)
    assert replayed.resumed and replayed.steps == first.steps
    assert tuple(backend.calls) == before
