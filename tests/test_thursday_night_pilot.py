"""Safety tests for the isolated, bounded training policy pilot."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from orch.contract.models import (
    AppliedStep, DesignState, Evidence, RunOutcome, RunRecord, RunStatus,
    Verdict, VerdictStatus,
)
from scripts import thursday_night_pilot as pilot
from scripts.thursday_night_pilot import (
    BRIEFS, DESIGNS, POLICIES, CountedBackend, PilotRunCap, PilotSafetyStop,
    VerifiedEpisodeEngine, _audit_events, episode_schedule, require_clean_properties,
    require_start_properties,
)

ZERO = {"n_DONT_TOUCH": 0, "n_IS_LOC_FIXED": 0, "n_IS_BEL_FIXED": 0}


def _evidence(props: dict[str, int] | None, *, state: str = "a" * 64) -> Evidence:
    return Evidence(
        id="evidence", state_id=state, metrics={}, routed=True,
        route_errors=0, drc_errors=0, constraint_fingerprint="x",
        clock_digest="c", v_props=props, meter_version="meter", fake=True,
    )


def test_schedule_covers_every_training_cell_once() -> None:
    rows = episode_schedule()
    assert len(rows) == 140 == len(set(rows))
    assert {design for design, _, _ in rows} == set(DESIGNS)
    assert {brief for _, brief, _ in rows} == set(BRIEFS)
    assert {policy for _, _, policy in rows} == set(POLICIES)
    for design in DESIGNS:
        for brief in BRIEFS:
            assert {policy for d, b, policy in rows if (d, b) == (design, brief)} == set(POLICIES)


@pytest.mark.parametrize(
    "props",
    [
        None,
        {"n_DONT_TOUCH": 0, "n_IS_LOC_FIXED": 0},
        {"n_DONT_TOUCH": 1, "n_IS_LOC_FIXED": 0, "n_IS_BEL_FIXED": 0},
        {"n_DONT_TOUCH": 0, "n_IS_LOC_FIXED": 0, "n_IS_BEL_FIXED": 2},
    ],
)
def test_placement_validity_fails_closed(props: dict[str, int] | None) -> None:
    start = _evidence(ZERO)
    with pytest.raises(PilotSafetyStop, match="DONT_TOUCH"):
        require_clean_properties(_evidence(props, state="b" * 64), start)
    require_clean_properties(_evidence(ZERO, state="b" * 64), start)


def test_placement_validity_without_start_evidence_fails_closed() -> None:
    with pytest.raises(PilotSafetyStop, match="start"):
        require_clean_properties(_evidence(ZERO, state="b" * 64), None)


def test_a_start_with_fixed_cells_is_accepted_but_a_start_without_counts_is_not() -> None:
    require_start_properties(_evidence({"n_DONT_TOUCH": 5, "n_IS_LOC_FIXED": 3, "n_IS_BEL_FIXED": 2}))
    with pytest.raises(PilotSafetyStop, match="DONT_TOUCH") as no_counts:
        require_start_properties(_evidence(None))
    # A start without counts is a setup fault that must stop a stage, not an invalid-output cell.
    assert str(no_counts.value).startswith("start state lacks DONT_TOUCH")
    with pytest.raises(PilotSafetyStop, match="DONT_TOUCH"):
        require_start_properties(_evidence({"n_DONT_TOUCH": 1}))


def test_exception_messages_keep_the_prefixes_the_stage_driver_matches() -> None:
    """scripts/saturday_llm.py treats these prefixes as an invalid output, not a crash."""

    with pytest.raises(PilotSafetyStop) as missing:
        require_clean_properties(_evidence(None, state="b" * 64), _evidence(ZERO))
    assert str(missing.value).startswith("missing DONT_TOUCH")
    with pytest.raises(PilotSafetyStop) as more:
        require_clean_properties(_evidence({**ZERO, "n_IS_LOC_FIXED": 1}, state="b" * 64), _evidence(ZERO))
    assert str(more.value).startswith("more DONT_TOUCH")


def test_backend_cap_is_persistent_and_checked_before_launch(tmp_path: Path) -> None:
    class Backend:
        calls = 0

        def execute(self, *args, **kwargs):
            self.calls += 1
            return self.calls

    backend = Backend()
    checked = []
    counter = CountedBackend(backend, tmp_path, "f" * 64, lambda: checked.append(True), cap=2)
    state = SimpleNamespace(id="a" * 64)
    recipe = SimpleNamespace(id="recipe")
    with pytest.raises(PilotSafetyStop, match="episode"):
        counter.execute(state, recipe, run_id="run_0", run_dir=tmp_path)
    counter.active_episode = "pilot_test"
    assert counter.execute(state, recipe, run_id="run_1", run_dir=tmp_path) == 1
    assert counter.execute(state, recipe, run_id="run_2", run_dir=tmp_path) == 2
    with pytest.raises(PilotRunCap, match="2"):
        counter.execute(state, recipe, run_id="run_3", run_dir=tmp_path)
    resumed = CountedBackend(backend, tmp_path, "f" * 64, lambda: checked.append(True), cap=2)
    resumed.active_episode = "pilot_test"
    with pytest.raises(PilotRunCap):
        resumed.execute(state, recipe, run_id="run_4", run_dir=tmp_path)
    assert backend.calls == 2
    assert resumed.count() == 2
    assert len(checked) == 5


class _Harness:
    """A fake inner engine and AMD ledger around the real VerifiedEpisodeEngine."""

    def __init__(self, tmp_path: Path) -> None:
        log = tmp_path / "runs" / "validator" / "validate.log"
        log.parent.mkdir(parents=True)
        log.write_text("Outputs (2): A B\n", encoding="utf-8")
        self.log = log
        canonical = tmp_path / "records" / "pass.json"
        canonical.parent.mkdir(parents=True)
        canonical.write_text('{"result":"pass"}\n', encoding="utf-8")
        record = {"result": "pass", "attempts": [{"run_dir": str(log.parent)}]}

        class Ledger:
            root = tmp_path
            calls = 0

            def verify(self, *args):
                self.calls += 1
                return record

            def _record_path(self, *args):
                return canonical

        class Engine:
            store = object()
            meter_version = "meter"
            cache_enabled = True

            def __init__(self) -> None:
                self.applied = None
                self.start_evidence = None

            def apply(self, *args, **kwargs):
                return self.applied

            def measure_start(self, state):
                return self.start_evidence

        self.engine = Engine()
        self.ledger = Ledger()
        self.wrapped = VerifiedEpisodeEngine(self.engine, self.ledger)
        self.tmp_path = tmp_path
        self.state = DesignState(id="a" * 64, design=DESIGNS[0], path="/unused", fake=True)
        self.output = DesignState(
            id="b" * 64, design=DESIGNS[0], path="/unused", fake=True, parent_id="a" * 64,
        )
        self.run = RunRecord(
            id="run_1", recipe_id="recipe", recipe_version="v",
            status=RunStatus.OK, fake=True,
        )
        self.verdict = Verdict(
            id="verdict", evidence_id="evidence", status=VerdictStatus.ACCEPTABLE,
            physically_valid=True, brief_valid=True, guard_pass=True, fake=True,
        )

    def returns(self, props: dict[str, int] | None) -> None:
        self.engine.applied = AppliedStep(
            outcome=RunOutcome(
                run=self.run, state=self.output, evidence=_evidence(props, state="b" * 64),
            ),
            verdict=self.verdict,
        )

    def apply(self, start: Evidence | None):
        return self.wrapped.apply(self.state, None, brief=None, start=start, parent=start)


def test_verified_engine_blocks_dirty_output_before_amd_or_journal(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    start = _evidence(ZERO)
    harness.returns({**ZERO, "n_DONT_TOUCH": 1})
    with pytest.raises(PilotSafetyStop, match="DONT_TOUCH"):
        harness.apply(start)
    assert harness.ledger.calls == 0
    harness.returns(ZERO)
    assert harness.apply(start) == harness.engine.applied
    assert harness.ledger.calls == 1
    proof = tmp_path / "output_census" / DESIGNS[0] / f"{harness.output.id}.json"
    assert proof.is_file()
    harness.log.write_text("Outputs (0):\n", encoding="utf-8")
    with pytest.raises(PilotSafetyStop, match="compared outputs"):
        harness.apply(start)


@pytest.mark.parametrize("key", list(ZERO))
@pytest.mark.parametrize("n", [1, 3])
def test_synthetic_start_with_n_fixed_cells_passes_at_n_and_fails_at_n_plus_one(
    tmp_path: Path, key: str, n: int,
) -> None:
    harness = _Harness(tmp_path)
    start = _evidence({name: n for name in ZERO})
    harness.returns({name: n for name in ZERO})
    assert harness.apply(start) == harness.engine.applied
    assert harness.ledger.calls == 1
    harness.returns({**{name: n for name in ZERO}, key: n + 1})
    with pytest.raises(PilotSafetyStop, match="more DONT_TOUCH/fixed LOC/BEL cells than the start state"):
        harness.apply(start)
    assert harness.ledger.calls == 1, "the AMD simulation must not run for a rejected output"


def test_verified_engine_without_start_evidence_fails_closed(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    harness.returns(ZERO)
    with pytest.raises(PilotSafetyStop, match="start"):
        harness.apply(None)
    assert harness.ledger.calls == 0


def test_measure_start_accepts_fixed_cells_in_the_start_but_not_missing_counts(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    carrying = _evidence({"n_DONT_TOUCH": 2, "n_IS_LOC_FIXED": 6, "n_IS_BEL_FIXED": 1})
    harness.engine.start_evidence = carrying
    assert harness.wrapped.measure_start(harness.state) is carrying
    harness.engine.start_evidence = _evidence(None)
    with pytest.raises(PilotSafetyStop, match="DONT_TOUCH"):
        harness.wrapped.measure_start(harness.state)


class _Store:
    """Outcome store for journal audits: one OK output per step run id."""

    def __init__(self, outputs: dict[str, tuple[str, dict[str, int]]]) -> None:
        self.outputs = outputs

    def load_outcome(self, run_id: str) -> RunOutcome:
        state_id, props = self.outputs[run_id]
        return RunOutcome(
            run=RunRecord(id=run_id, recipe_id="recipe", recipe_version="v", status=RunStatus.OK, fake=True),
            state=DesignState(id=state_id, design=DESIGNS[0], path="/unused", fake=True),
            evidence=_evidence(props, state=state_id),
        )


def _journal(start_props: dict[str, int], run_ids: list[str]) -> list[dict]:
    events: list[dict] = [{
        "id": "start", "type": "start", "envelope": {},
        "start_evidence": _evidence(start_props).model_dump(mode="json"),
    }]
    for run_id in run_ids:
        events.append({
            "type": "step", "step": {"run_id": run_id}, "verdict": {"physically_valid": True},
        })
    return events


def test_journal_audit_compares_every_step_with_the_start_recorded_in_the_journal(monkeypatch) -> None:
    amd_checked: list[str] = []
    monkeypatch.setattr(pilot, "require_amd_pass", lambda ledger, design, state_id, **kw: amd_checked.append(state_id))
    fixed = {"n_DONT_TOUCH": 2, "n_IS_LOC_FIXED": 2, "n_IS_BEL_FIXED": 2}
    store = _Store({"run_ok": ("b" * 64, fixed), "run_more": ("c" * 64, {**fixed, "n_IS_BEL_FIXED": 3})})
    _audit_events(_journal(fixed, ["run_ok"]), store, object())
    assert amd_checked == ["b" * 64]
    with pytest.raises(PilotSafetyStop, match="more DONT_TOUCH"):
        _audit_events(_journal(fixed, ["run_ok", "run_more"]), store, object())


def test_journal_audit_refuses_a_journal_without_a_start_event(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "require_amd_pass", lambda *a, **kw: None)
    store = _Store({"run_ok": ("b" * 64, ZERO)})
    with pytest.raises(PilotSafetyStop, match="start"):
        _audit_events(_journal(ZERO, ["run_ok"])[1:], store, object())


def test_journal_audit_refuses_a_start_without_counts(monkeypatch) -> None:
    monkeypatch.setattr(pilot, "require_amd_pass", lambda *a, **kw: None)
    with pytest.raises(PilotSafetyStop, match="DONT_TOUCH"):
        _audit_events(_journal({"n_DONT_TOUCH": 0}, []), _Store({}), object())


def test_journal_audit_of_an_episode_that_has_not_started_is_a_no_op(monkeypatch) -> None:
    """Every driver audits an episode before running it; a new episode has no journal yet (found by the first
    smoke stage of the stronger-model test, which stopped every cell with 'journal lacks its start event')."""

    monkeypatch.setattr(pilot, "require_amd_pass", lambda *a, **kw: pytest.fail("nothing to check"))
    assert _audit_events([], _Store({}), object()) is None
