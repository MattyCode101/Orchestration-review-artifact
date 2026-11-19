"""Offline fitted episode and saved training telemetry checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orch.config import load_brief, load_recipe_sets, load_worker_card
from orch.contract.models import DesignState, Evidence
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.journal import Journal
from orch.engine.store import StateStore
from orch.episode import run_episode
from orch.fake.backend import FakeBackend
from orch.policies.trivial import TrivialPolicy
from orch.recipes.registry import load_registry
from orch.workers.fitted import FittedWorker
from orch.workers.telemetry import SavedTelemetryProvider

PROJECT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT / "data/sweeps/worker_fitting_train"


class FakeInputTelemetry:
    """Synthetic, state-bound feature values for software integration only."""

    def __init__(self, shift: int = 0) -> None:
        self.shift = shift

    def identity(self) -> dict[str, str]:
        return {"kind": "fake-test-telemetry-v1"}

    def __call__(self, state, evidence, producer_recipe_id, run_id):
        assert state.id == evidence.state_id
        record = {
            "state_sha256": state.id,
            "fake": True,
            "metrics": evidence.model_dump(mode="json")["metrics"],
            "wpath": {
                "requirement_ns": 1.57,
                "logic_levels": 2 + self.shift,
                "tile_span": {"x_min": 1, "x_max": 4, "y_min": 2, "y_max": 7},
                "net_delay_ns": 1.0,
                "data_path_delay_ns": 2.0,
            },
            "probe": {
                "pblock_sizing_resources": {"luts": 1000},
                "utilization_summary": {
                    "Block RAM Tile": {"used": 2},
                    "DSPs": {"used": 1},
                    "URAM": {"used": 0},
                    "CLB Registers": {"used": 500},
                },
                "lut_as_shift_register": 10,
                "lut_primitive_counts": {
                    "LUT1": 10, "LUT2": 20, "LUT3": 30,
                    "LUT4": 40, "LUT5": 50, "LUT6": 60,
                },
            },
        }
        return {
            "telemetry": record,
            "produced_by_recipe_id": producer_recipe_id,
            "telemetry_source": {"kind": "fake-test", "state_sha256": state.id},
        }


def _setup(tmp_path: Path):
    store = StateStore(tmp_path / "data")
    backend = FakeBackend(store=store)
    engine = StepEngine(
        store=store, cache=RunCache(store.root), backend=backend,
        meter_version=backend.meter_version, cache_enabled=True,
    )
    registry = load_registry(PROJECT)
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    workers = {
        worker_id: FittedWorker(
            load_worker_card(PROJECT / "configs/workers" / f"{worker_id}.yaml"),
            sets.worker_sets[worker_id], registry,
        )
        for worker_id in ("timing", "area")
    }
    return engine, registry, workers, backend


@pytest.mark.parametrize("worker_id,brief_name", [
    ("timing", "timing"), ("area", "area"),
])
def test_fitted_episode_journals_scores_and_producer(
    tmp_path: Path, worker_id: str, brief_name: str,
) -> None:
    engine, registry, workers, backend = _setup(tmp_path)
    start = backend.create_start("vexriscv_re-place")
    brief = load_brief(PROJECT / "configs/briefs" / f"{brief_name}.yaml").model_copy(
        update={"budget_runs": 2}
    )
    result = run_episode(
        engine=engine, registry=registry, workers=workers,
        policy=TrivialPolicy(worker_id=worker_id, max_runs=1, preserve_placement=False),
        brief=brief, start_state=start, episode_id=f"fitted_{worker_id}",
        telemetry_provider=FakeInputTelemetry(),
    )
    assert result.fake and result.recipe_runs == 2
    events = Journal(engine.store.root).read(result.episode_id)
    plans = [event["worker_plan"] for event in events if event["type"] == "mandate"]
    assert len(plans) == 2
    assert all(plan["score_plan"]["model_label"] == "FINAL" for plan in plans)
    assert all(plan["score_plan"]["model_sha256"] == workers[worker_id].model_sha256 for plan in plans)
    assert all(plan["score_plan"]["ranked"] for plan in plans)
    assert all(plan["telemetry_source"]["state_sha256"] == plan["input_state_id"] for plan in plans)
    if worker_id == "area":
        assert plans[0]["score_plan"]["incomplete_measurement"]
    first_recipe = result.steps[0].run_id
    first_outcome = engine.store.load_outcome(first_recipe)
    if plans[1]["input_state_id"] == first_outcome.state.id:
        assert plans[1]["producer_recipe_id"] == first_outcome.run.recipe_id
        assert first_outcome.run.recipe_id in plans[1]["score_plan"]["excluded_producer"]


def test_resume_rejects_changed_telemetry_with_same_provider_identity(tmp_path: Path) -> None:
    engine, registry, workers, backend = _setup(tmp_path)
    start = backend.create_start("vexriscv_re-place")
    brief = load_brief(PROJECT / "configs/briefs/timing.yaml").model_copy(
        update={"budget_runs": 1}
    )
    args = dict(
        engine=engine, registry=registry, workers=workers,
        policy=TrivialPolicy(worker_id="timing", max_runs=1),
        brief=brief, start_state=start, episode_id="fitted_replay_guard",
    )
    run_episode(**args, telemetry_provider=FakeInputTelemetry())
    with pytest.raises(ValueError, match="score plan"):
        run_episode(**args, telemetry_provider=FakeInputTelemetry(shift=1))


def test_saved_provider_refuses_non_training_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-training"):
        SavedTelemetryProvider(tmp_path)


@pytest.mark.skipif(not TRAIN.is_dir(), reason="training campaign is not present")
def test_saved_provider_prefers_saved_utilization_and_checks_state() -> None:
    provider = SavedTelemetryProvider(TRAIN)
    store = StateStore(TRAIN)
    start_id = "8f20f4036a9be0800a0c1fb3072e7a1155ea2f32f630a66a0f96917d2c4e3efc"
    start = DesignState.model_validate_json((TRAIN / "states" / start_id / "state.json").read_bytes())
    evidence = Evidence.model_validate_json(
        next((TRAIN / "measurements" / start_id).glob("*/evidence.json")).read_bytes()
    )
    context = provider(start, evidence, None, None)
    assert context["telemetry"]["state_sha256"] == start.id
    assert context["telemetry_source"]["utilization_report_sha256"] is None

    outcome = store.load_outcome("run_1f37c8eaa5ab42df824bfd818a16ccc2")
    child = provider(outcome.state, outcome.evidence, outcome.run.recipe_id, outcome.run.id)
    assert child["telemetry"]["state_sha256"] == outcome.state.id
    assert child["telemetry_source"]["utilization_report_sha256"]
    assert child["telemetry_source"]["utilization_report"].startswith("runs/")
    with pytest.raises(ValueError, match="producer identity"):
        provider(outcome.state, outcome.evidence, "LATCH.retime", outcome.run.id)
    with pytest.raises(ValueError, match="producer identity"):
        provider(
            outcome.state, outcome.evidence.model_copy(update={"id": "wrong"}),
            outcome.run.recipe_id, outcome.run.id,
        )
    wrong = outcome.state.model_copy(update={"id": "0" * 64})
    with pytest.raises(ValueError):
        provider(wrong, outcome.evidence, outcome.run.recipe_id, outcome.run.id)


@pytest.mark.skipif(not TRAIN.is_dir(), reason="training campaign is not present")
def test_post_plan_d3_state_uses_saved_report_and_flags_missing_path() -> None:
    """A derived D3 output absent from v3 still has checked area rows."""

    from orch.analysis.fit_workers import extract_feature, load_feature_spec

    store = StateStore(TRAIN)
    outcome = store.load_outcome("run_307d9f32ee124a299acb09d68cd5c41e")
    provider = SavedTelemetryProvider(TRAIN)
    context = provider(
        outcome.state, outcome.evidence, outcome.run.recipe_id, outcome.run.id
    )
    assert context["telemetry_source"]["v3_record_sha256"] is None
    assert context["telemetry_source"]["utilization_report_sha256"]
    spec = load_feature_spec()
    assert extract_feature(context["telemetry"], "A_R_EA", spec) is not None
    assert extract_feature(context["telemetry"], "LATCH.retime", spec) is None

def test_real_episode_cli_selects_published_fitted_workers() -> None:
    """Construction alone must not create a Vivado or validator job."""

    from orch.cli import _Context, _episode_workers

    context = _Context(
        engine=None, store=None, registry=load_registry(PROJECT),
        fake_backend=None, real_runtime=object(),
    )
    workers, label = _episode_workers(context)
    assert label == "Matt_signed"
    assert set(workers) == {"timing", "area"}
    assert all(isinstance(worker, FittedWorker) for worker in workers.values())
    assert all(worker.model_label == "FINAL" for worker in workers.values())

def test_fitted_resume_runs_remaining_scored_recipe_once(tmp_path: Path) -> None:
    """A persisted first step must not truncate the top-two fitted plan."""

    engine, registry, workers, backend = _setup(tmp_path)
    start = backend.create_start("vexriscv_re-place")
    brief = load_brief(PROJECT / "configs/briefs/area.yaml").model_copy(
        update={"budget_runs": 2}
    )
    policy = TrivialPolicy(worker_id="area", max_runs=2, preserve_placement=False)
    provider = FakeInputTelemetry()
    execute_original = backend.execute
    calls = 0

    def interrupt_on_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption before second saved result")
        return execute_original(*args, **kwargs)

    backend.execute = interrupt_on_second
    arguments = dict(
        engine=engine, registry=registry, workers=workers,
        policy=policy, brief=brief, start_state=start,
        episode_id="fitted_interrupted", telemetry_provider=provider,
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_episode(**arguments)
    initial = Journal(engine.store.root).read("fitted_interrupted")
    plan = next(event["worker_plan"]["score_plan"]["selected_ids"]
                for event in initial if event["type"] == "mandate")
    assert len(plan) == 2
    assert sum(event["type"] == "step" for event in initial) == 1

    backend.execute = execute_original
    completed = run_episode(**arguments)
    assert completed.resumed and completed.recipe_runs == 2
    attempted = [
        engine.store.load_outcome(step.run_id).run.recipe_id
        for step in completed.steps
    ]
    assert attempted == plan
