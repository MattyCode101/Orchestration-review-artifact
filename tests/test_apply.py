"""Integration checks for the cacheable run versus fresh brief judgment split."""

from __future__ import annotations

from pathlib import Path

from orch.config import load_brief
from orch.contract.models import (
    Claims,
    DesignState,
    Evidence,
    MetricName,
    MetricValue,
    RunOutcome,
    RunRecord,
    RunStatus,
    VerdictStatus,
)
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.store import StateStore
from orch.interfaces import RecipeSpec


ROOT = Path(__file__).resolve().parents[1]


def evidence(state_id: str, version: str, *, wns: float, luts: int) -> Evidence:
    """Label all fixture evidence as fake while retaining real contract units."""

    return Evidence(
        id=f"fake-{state_id[:8]}-{version}",
        state_id=state_id,
        metrics={
            MetricName.WNS_NS: MetricValue(
                value=wns, unit="ns", method="fake_fixture", basis="routed"
            ),
            MetricName.PACKED_LUTS: MetricValue(
                value=luts, unit="count", method="fake_fixture", basis="routed"
            ),
        },
        routed=True,
        route_errors=0,
        drc_errors=0,
        constraint_fingerprint="fake-xdc",
        clock_digest="fake-clock",
        meter_version=version,
        fake=True,
    )


class DummyBackend:
    """A tiny deterministic backend; it never invokes a hardware tool."""

    def __init__(self, store: StateStore, version: str, *, failure: bool = False) -> None:
        self.store = store
        self.version = version
        self.failure = failure
        self.calls = 0
        self.measure_calls = 0

    def measure_start(self, state: DesignState, *, run_dir: Path) -> Evidence:
        self.measure_calls += 1
        return evidence(state.id, self.version, wns=-0.785, luts=1895)

    def execute(
        self, state: DesignState, recipe: RecipeSpec, *, run_id: str, run_dir: Path
    ) -> RunOutcome:
        self.calls += 1
        if self.failure:
            return RunOutcome(
                run=RunRecord(
                    id=run_id,
                    recipe_id=recipe.id,
                    recipe_version=recipe.version,
                    status=RunStatus.TOOL_ERROR,
                    wall_s=3.0,
                    input_sha256_before=state.id,
                    input_sha256_after=state.id,
                    fake=True,
                )
            )
        output = run_dir / "out.dcp"
        output.write_bytes(b"fake-output-for-cache-integration")
        child = self.store.import_state(
            output,
            design=state.design,
            parent_id=state.id,
            made_by_run_id=run_id,
            fake=True,
        )
        run = RunRecord(
            id=run_id,
            recipe_id=recipe.id,
            recipe_version=recipe.version,
            status=RunStatus.OK,
            wall_s=3.0,
            cpu_user_s=2.25,
            cpu_sys_s=0.75,
            peak_memory_bytes=123456,
            load_1m_start=6.0,
            load_1m_end=7.0,
            busy_slots_start=4,
            input_sha256_before=state.id,
            input_sha256_after=state.id,
            fake=True,
        )
        return RunOutcome(
            run=run,
            state=child,
            evidence=evidence(child.id, self.version, wns=-0.70, luts=1920),
            claims=Claims(id=f"{run_id}:claims", run_id=run_id, fake=True),
        )


def setup(tmp_path: Path, version: str = "meter-v1"):
    data = tmp_path / "data"
    store = StateStore(data)
    cache = RunCache(data)
    input_file = tmp_path / "start.dcp"
    input_file.write_bytes(b"fake-start")
    state = store.import_state(input_file, design="fake-vex", fake=True)
    backend = DummyBackend(store, version)
    engine = StepEngine(
        store=store,
        cache=cache,
        backend=backend,
        meter_version=version,
        cache_enabled=True,
    )
    recipe = RecipeSpec(
        id="FAKE.AREA",
        worker="area",
        kind="tcl",
        script="fake.tcl",
        replaces=True,
        cause="test",
        source="fake",
        source_sha256="fake",
        timeout_s=5,
        version="recipe-v1",
    )
    return store, cache, backend, engine, state, recipe


def test_same_measured_run_gets_fresh_verdict_for_each_brief(tmp_path: Path) -> None:
    store, _, backend, engine, state, recipe = setup(tmp_path)
    start = engine.measure_start(state)
    timing = load_brief(ROOT / "configs" / "briefs" / "timing.yaml")
    area = load_brief(ROOT / "configs" / "briefs" / "area.yaml")

    first = engine.apply(
        state, recipe, brief=timing, start=start, parent=start
    )
    second = engine.apply(
        state, recipe, brief=area, start=start, parent=start
    )

    assert backend.calls == 1
    assert first.verdict.status == VerdictStatus.BRIEF_VIOLATION
    assert first.verdict.physically_valid
    assert second.verdict.status == VerdictStatus.ACCEPTABLE
    assert second.verdict.brief_valid
    assert second.outcome.run.cache_hit
    assert second.outcome.run.original_run_id == first.outcome.run.id
    assert second.outcome.run.wall_s == 3.0
    assert second.outcome.run.cpu_user_s == first.outcome.run.cpu_user_s == 2.25
    assert second.outcome.run.cpu_sys_s == first.outcome.run.cpu_sys_s == 0.75
    assert second.outcome.run.peak_memory_bytes == first.outcome.run.peak_memory_bytes == 123456
    assert second.outcome.run.load_1m_start == first.outcome.run.load_1m_start == 6.0
    assert second.outcome.run.busy_slots_start == first.outcome.run.busy_slots_start == 4
    assert store.load_outcome(second.outcome.run.id).claims.run_id == second.outcome.run.id
    assert len(list((store.root / "runs" / first.outcome.run.id / "verdicts").glob("*.json"))) == 1
    assert len(list((store.root / "runs" / second.outcome.run.id / "verdicts").glob("*.json"))) == 1


def test_meter_version_change_invalidates_cached_evidence(tmp_path: Path) -> None:
    store, cache, backend, engine, state, recipe = setup(tmp_path)
    start_v1 = engine.measure_start(state)
    brief = load_brief(ROOT / "configs" / "briefs" / "area.yaml")
    first = engine.apply(state, recipe, brief=brief, start=start_v1, parent=start_v1)
    assert backend.calls == 1

    backend_v2 = DummyBackend(store, "meter-v2")
    engine_v2 = StepEngine(
        store=store,
        cache=cache,
        backend=backend_v2,
        meter_version="meter-v2",
        cache_enabled=True,
    )
    start_v2 = engine_v2.measure_start(state)
    second = engine_v2.apply(
        state, recipe, brief=brief, start=start_v2, parent=start_v2
    )
    assert backend_v2.calls == 1
    assert not second.outcome.run.cache_hit
    assert second.outcome.evidence.meter_version == "meter-v2"
    assert second.outcome.run.id != first.outcome.run.id
    assert cache.get(cache.key(state.id, recipe.id, recipe.version)) == second.outcome.run.id


def test_start_measurement_is_cached_by_state_and_meter(tmp_path: Path) -> None:
    _, _, backend, engine, state, _ = setup(tmp_path)
    first = engine.measure_start(state)
    second = engine.measure_start(state)
    assert first == second
    assert backend.measure_calls == 1


def test_deterministic_failure_is_cached_but_judged_as_run_failed(tmp_path: Path) -> None:
    store, cache, _, _, state, recipe = setup(tmp_path)
    backend = DummyBackend(store, "meter-v1", failure=True)
    engine = StepEngine(
        store=store,
        cache=cache,
        backend=backend,
        meter_version="meter-v1",
        cache_enabled=True,
    )
    start = engine.measure_start(state)
    brief = load_brief(ROOT / "configs" / "briefs" / "area.yaml")
    first = engine.apply(state, recipe, brief=brief, start=start, parent=start)
    second = engine.apply(state, recipe, brief=brief, start=start, parent=start)
    assert backend.calls == 1
    assert first.verdict.status == second.verdict.status == VerdictStatus.RUN_FAILED
    assert second.outcome.run.cache_hit

