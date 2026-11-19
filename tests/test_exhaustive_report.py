"""Software-only checks for exhaustive search, resume and share-of-best."""

from __future__ import annotations

from pathlib import Path

import pytest

from orch.analysis.report import (
    Comparison,
    best_valid,
    build_share_of_best_table,
    render_share_of_best_markdown,
    share_of_best,
)
from orch.baselines.exhaustive import run_exhaustive
from orch.contract.models import (
    Brief,
    BriefTerm,
    Claims,
    DesignState,
    Evidence,
    MetricName,
    MetricValue,
    Objective,
    RunOutcome,
    RunRecord,
    RunStatus,
    VerdictStatus,
)
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.store import StateStore
from orch.interfaces import RecipeSpec


def _evidence(
    state_id: str, *, wns: float, luts: float, routed: bool = True
) -> Evidence:
    return Evidence(
        id=f"fake-meter-{state_id}",
        state_id=state_id,
        metrics={
            MetricName.WNS_NS: MetricValue(
                value=wns, unit="ns", method="fake_fixture", basis="routed"
            ),
            MetricName.PACKED_LUTS: MetricValue(
                value=luts, unit="count", method="fake_fixture", basis="routed"
            ),
        },
        routed=routed,
        route_errors=0 if routed else 1,
        drc_errors=0,
        constraint_fingerprint="fake-xdc",
        clock_digest="fake-clocks",
        meter_version="fake-meter-v1",
        fake=True,
    )


class SequenceBackend:
    """A deterministic sequence model; every checkpoint byte is synthetic."""

    def __init__(
        self,
        store: StateStore,
        *,
        bad_first: str | None = None,
        fail_on_call: int | None = None,
        timeout_once: bool = False,
    ) -> None:
        self.store = store
        self.meter_version = "fake-meter-v1"
        self.bad_first = bad_first
        self.fail_on_call = fail_on_call
        self.timeout_once = timeout_once
        self.calls = 0
        self.sequences: list[tuple[str, ...]] = []

    def measure_start(self, state: DesignState, *, run_dir: Path) -> Evidence:
        return _evidence(state.id, wns=-1.0, luts=100)

    def execute(
        self, state: DesignState, recipe: RecipeSpec, *, run_id: str, run_dir: Path
    ) -> RunOutcome:
        raw = self.store.state_path(state.id).read_text(encoding="ascii")
        prior = tuple(raw.split("|")[1:])
        sequence = prior + (recipe.id,)
        self.calls += 1
        self.sequences.append(sequence)
        if self.fail_on_call == self.calls:
            raise RuntimeError("simulated crash before outcome commit")
        if self.timeout_once:
            self.timeout_once = False
            return RunOutcome(
                run=RunRecord(
                    id=run_id,
                    recipe_id=recipe.id,
                    recipe_version=recipe.version,
                    status=RunStatus.TIMEOUT,
                    wall_s=2.0,
                    input_sha256_before=state.id,
                    input_sha256_after=state.id,
                    fake=True,
                )
            )

        output = run_dir / "out.dcp"
        output.write_text("START|" + "|".join(sequence), encoding="ascii")
        child = self.store.import_state(
            output,
            design=state.design,
            parent_id=state.id,
            made_by_run_id=run_id,
            fake=True,
        )
        wns = -1.0
        luts = 100.0
        for recipe_id in sequence:
            if recipe_id == "T":
                wns += 0.2
                luts += 4.0
            elif recipe_id == "A":
                wns -= 0.02
                luts -= 5.0
            elif recipe_id == "BAD":
                wns += 0.01
        routed = not (self.bad_first is not None and sequence[0] == self.bad_first)
        return RunOutcome(
            run=RunRecord(
                id=run_id,
                recipe_id=recipe.id,
                recipe_version=recipe.version,
                status=RunStatus.OK,
                wall_s=2.0,
                input_sha256_before=state.id,
                input_sha256_after=state.id,
                fake=True,
            ),
            state=child,
            evidence=_evidence(child.id, wns=wns, luts=luts, routed=routed),
            claims=Claims(id=f"{run_id}:claims", run_id=run_id, fake=True),
        )


def _recipe(recipe_id: str, worker: str, version: str = "v1") -> RecipeSpec:
    return RecipeSpec(
        id=recipe_id,
        worker=worker,
        kind="tcl",
        script="fake.tcl",
        replaces=False,
        cause="software-only fixture",
        source="test",
        source_sha256="fake",
        timeout_s=5,
        version=version,
    )


def _briefs() -> tuple[Brief, Brief]:
    timing = Brief(
        id="T",
        objective=Objective(metric=MetricName.WNS_NS, direction="max"),
        terms=(
            BriefTerm(
                id="lut_cap",
                metric=MetricName.PACKED_LUTS,
                comparison="<=",
                factor=1.01,
            ),
        ),
    )
    area = Brief(
        id="A",
        objective=Objective(metric=MetricName.PACKED_LUTS, direction="min"),
        terms=(
            BriefTerm(
                id="wns_floor",
                metric=MetricName.WNS_NS,
                comparison=">=",
                offset=-0.1,
            ),
        ),
    )
    return timing, area


def _setup(
    tmp_path: Path,
    *,
    bad_first: str | None = None,
    fail_on_call: int | None = None,
    timeout_once: bool = False,
) -> tuple[StepEngine, SequenceBackend, DesignState]:
    store = StateStore(tmp_path / "data")
    start_path = tmp_path / "start.dcp"
    start_path.write_text("START", encoding="ascii")
    start_state = store.import_state(start_path, design="fake-design", fake=True)
    backend = SequenceBackend(
        store,
        bad_first=bad_first,
        fail_on_call=fail_on_call,
        timeout_once=timeout_once,
    )
    engine = StepEngine(
        store=store,
        cache=RunCache(store.root),
        backend=backend,
        meter_version=backend.meter_version,
        cache_enabled=True,
    )
    return engine, backend, start_state


def test_all_ordered_sequences_share_one_run_across_both_briefs(
    tmp_path: Path,
) -> None:
    engine, backend, start = _setup(tmp_path)
    timing, area = _briefs()
    recipes = (_recipe("T", "timing"), _recipe("A", "area"))
    sweep = run_exhaustive(
        engine=engine, start_state=start, recipes=recipes,
        briefs=(timing, area), sweep_id="all", max_depth=2,
    )

    assert len(sweep.entries) == 6
    assert {entry.sequence for entry in sweep.entries} == {
        ("T",), ("A",), ("T", "T"), ("T", "A"), ("A", "T"), ("A", "A")
    }
    assert backend.calls == 6
    timing_first = next(entry for entry in sweep.entries if entry.sequence == ("T",))
    recovered = next(entry for entry in sweep.entries if entry.sequence == ("T", "A"))
    assert timing_first.verdicts["T"].status == VerdictStatus.BRIEF_VIOLATION
    assert timing_first.verdicts["T"].physically_valid
    assert recovered.verdicts["T"].brief_valid
    assert all(entry.fake and entry.evidence.fake for entry in sweep.entries)
    assert all(
        set(entry.verdicts) == {"T", "A"} for entry in sweep.entries
    )
    assert sweep.total_recipe_runs == 6
    assert sweep.total_charged_wall_s == 12.0
    assert sweep.actual_backend_runs == 6
    assert len(list((engine.store.root / "runs" / timing_first.run_id / "verdicts").glob("*.json"))) == 2


def test_physically_invalid_first_output_skips_second_step(
    tmp_path: Path,
) -> None:
    engine, backend, start = _setup(tmp_path, bad_first="BAD")
    recipes = (_recipe("BAD", "timing"), _recipe("A", "area"))
    sweep = run_exhaustive(
        engine=engine, start_state=start, recipes=recipes,
        briefs=_briefs(), sweep_id="invalid", max_depth=2,
    )
    skipped = [entry for entry in sweep.entries if entry.skipped_reason]
    assert {entry.sequence for entry in skipped} == {
        ("BAD", "BAD"), ("BAD", "A")
    }
    assert backend.calls == 4
    assert all(entry.run_id is None and entry.fake for entry in skipped)
    assert len(sweep.entries) == 6


def test_depth_one_extends_and_resume_skips_committed_runs(tmp_path: Path) -> None:
    engine, backend, start = _setup(tmp_path)
    recipes = (_recipe("T", "timing"), _recipe("A", "area"))
    briefs = _briefs()
    first = run_exhaustive(
        engine=engine, start_state=start, recipes=recipes,
        briefs=briefs, sweep_id="resume", max_depth=1,
    )
    first_ids = [entry.run_id for entry in first.entries]
    assert backend.calls == 2
    extended = run_exhaustive(
        engine=engine, start_state=start, recipes=recipes,
        briefs=briefs, sweep_id="resume", max_depth=2,
    )
    assert backend.calls == 6
    assert [entry.run_id for entry in extended.entries[:2]] == first_ids
    resumed = run_exhaustive(
        engine=engine, start_state=start, recipes=recipes,
        briefs=briefs, sweep_id="resume", max_depth=2,
    )
    assert backend.calls == 6
    assert [entry.run_id for entry in resumed.entries] == [
        entry.run_id for entry in extended.entries
    ]
    assert resumed.total_recipe_runs == 6
    shallow_again = run_exhaustive(
        engine=engine, start_state=start, recipes=recipes,
        briefs=briefs, sweep_id="resume", max_depth=1,
    )
    assert shallow_again.total_recipe_runs == 2
    assert backend.calls == 6


def test_crash_mid_sweep_resumes_without_repeating_committed_edges(
    tmp_path: Path,
) -> None:
    engine, backend, start = _setup(tmp_path, fail_on_call=3)
    recipes = (_recipe("T", "timing"), _recipe("A", "area"))
    kwargs = {
        "engine": engine, "start_state": start, "recipes": recipes,
        "briefs": _briefs(), "sweep_id": "crash", "max_depth": 2,
    }
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_exhaustive(**kwargs)
    assert backend.sequences[:3] == [("T",), ("A",), ("T", "T")]
    backend.fail_on_call = None
    sweep = run_exhaustive(**kwargs)
    assert len(sweep.entries) == 6
    assert backend.sequences.count(("T",)) == 1
    assert backend.sequences.count(("A",)) == 1
    assert backend.sequences.count(("T", "T")) == 2
    assert backend.calls == 7


def test_retryable_edge_is_retried_on_resume(tmp_path: Path) -> None:
    engine, backend, start = _setup(tmp_path, timeout_once=True)
    recipe = _recipe("T", "timing")
    kwargs = {
        "engine": engine, "start_state": start, "recipes": (recipe,),
        "briefs": _briefs(), "sweep_id": "retry", "max_depth": 1,
    }
    first = run_exhaustive(**kwargs)
    assert first.entries[0].run_status == RunStatus.TIMEOUT
    second = run_exhaustive(**kwargs)
    assert second.entries[0].run_status == RunStatus.OK
    assert backend.calls == 2
    assert second.total_recipe_runs == 2


def test_changed_configuration_cannot_resume_same_sweep_id(tmp_path: Path) -> None:
    engine, _, start = _setup(tmp_path)
    recipes = (_recipe("T", "timing"),)
    run_exhaustive(
        engine=engine, start_state=start, recipes=recipes,
        briefs=_briefs(), sweep_id="identity", max_depth=1,
    )
    with pytest.raises(ValueError, match="configuration differs"):
        run_exhaustive(
            engine=engine, start_state=start,
            recipes=(_recipe("T", "timing", version="v2"),),
            briefs=_briefs(), sweep_id="identity", max_depth=1,
        )


def test_share_of_best_uses_improvement_from_negative_wns() -> None:
    assert share_of_best(-1.0, -0.8, -0.9, "max") == pytest.approx(0.5)
    assert share_of_best(100.0, 90.0, 95.0, "min") == pytest.approx(0.5)
    assert share_of_best(-1.0, -1.0, -0.9, "max") is None
    assert share_of_best(100.0, 100.0, 95.0, "min") is None


def test_report_table_includes_fake_marker_and_undefined_zero_gain(
    tmp_path: Path,
) -> None:
    engine, _, start = _setup(tmp_path)
    timing, area = _briefs()
    sweep = run_exhaustive(
        engine=engine, start_state=start,
        recipes=(_recipe("T", "timing"), _recipe("A", "area")),
        briefs=(timing, area), sweep_id="report", max_depth=2,
    )
    area_first = next(entry for entry in sweep.entries if entry.sequence == ("A",))
    area_best, _ = best_valid(sweep, area)
    assert area_best is not None
    assert area_best.metric(MetricName.PACKED_LUTS).value == 90
    rows = build_share_of_best_table(
        sweep,
        comparisons=(
            Comparison(
                approach="trivial",
                brief_id="A",
                evidence=area_first.evidence,
                verdict=area_first.verdicts["A"],
                run_count=1,
                wall_s=2.0,
                actual_backend_runs=1,
                fake=True,
            ),
        ),
    )
    assert len(rows) == 3
    area_row = next(row for row in rows if row.approach == "trivial")
    assert area_row.share_of_best == pytest.approx(0.5)
    assert area_row.fake is True
    assert all(row.fake for row in rows)
    rendered = render_share_of_best_markdown(rows)
    assert "| fake-design | A | trivial | 0.500 |" in rendered

    no_gain = run_exhaustive(
        engine=engine, start_state=start,
        recipes=(_recipe("T", "timing"),),
        briefs=(area,), sweep_id="no_gain", max_depth=1,
    )
    no_gain_row = build_share_of_best_table(no_gain)[0]
    assert no_gain_row.share_of_best is None
    assert no_gain_row.share_reason == "zero_best_improvement"




def test_report_rejects_a_verdict_that_promotes_invalid_meter_evidence(
    tmp_path: Path,
) -> None:
    engine, _, start = _setup(tmp_path)
    timing, area = _briefs()
    sweep = run_exhaustive(
        engine=engine, start_state=start,
        recipes=(_recipe("T", "timing"),),
        briefs=(timing, area), sweep_id="adversarial", max_depth=1,
    )
    timing_first = sweep.entries[0]
    assert not timing_first.verdicts["T"].brief_valid
    forged = timing_first.verdicts["T"].model_copy(
        update={"brief_valid": True}
    )
    with pytest.raises(ValueError, match="contradicts measured evidence"):
        build_share_of_best_table(
            sweep,
            comparisons=(
                Comparison(
                    approach="forged",
                    brief_id="T",
                    evidence=timing_first.evidence,
                    verdict=forged,
                    run_count=1,
                    wall_s=2.0,
                    fake=True,
                ),
            ),
        )
