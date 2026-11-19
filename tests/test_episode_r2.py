"""Round-2 loop options (orch/episode_r2.py): off = the frozen loop; on = empty-call observation and stop rule."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from orch import episode as frozen_module
from orch.contract.models import Mandate, MetricName, WorkerSettings
from orch.engine.journal import Journal
from orch.episode import run_episode
from orch.episode_r2 import (
    MAX_CONSECUTIVE_EMPTY, ROUND2, STOP_NOTE, LoopOptions, call_gains, loop_facts, run_episode_r2,
)
from orch.policies.random_policy import RandomPolicy
from orch.policies.trivial import TrivialPolicy

from test_episode import _brief, _setup, _worker  # the frozen loop's own fixtures (imported, not copied)

RUN = re.compile(r"run_[0-9a-f]{32}")
STEP = re.compile(r"step_[0-9a-f]{64}")


def _journal_bytes(engine, episode_id: str) -> bytes:
    return Journal(engine.store.root).path(episode_id).read_bytes()


def _normalised(events: list[dict]) -> list[dict]:
    import json

    text = STEP.sub("<step>", RUN.sub("<run>", json.dumps(events, sort_keys=True)))
    value = json.loads(text)
    for event in value:
        if event.get("type") == "start":
            event["envelope"].pop("loop_options", None)
    return value


class Scripted:
    """Plays a fixed list of (worker, base) choices; base 'start' or 'current'; None = STOP. Records notes seen."""

    id = "unit-scripted"
    version = "1"

    def __init__(self, script, *, max_runs: int = 1, preserve: bool = False) -> None:
        self.script = list(script)
        self.max_runs, self.preserve = max_runs, preserve
        self.harness_note = None
        self.seen_notes: list[str | None] = []
        self.asked = 0

    def decide(self, view):
        self.seen_notes.append(self.harness_note)
        index = self.asked
        self.asked += 1
        item = self.script[index] if index < len(self.script) else None
        if item is None:
            return None
        worker, base = item
        base_id = view.start.state_id if base == "start" else view.current.state_id
        return Mandate(
            id=f"mandate_{index}", worker_id=worker, base_state_id=base_id,
            settings=WorkerSettings(max_runs=self.max_runs, preserve_placement=self.preserve),
            brief_hash=view.brief.fingerprint(),
        )


# ---------------------------------------------------------------------------------------------------- options off
def test_options_off_calls_the_frozen_loop(tmp_path: Path, monkeypatch) -> None:
    engine, registry, workers, _, start = _setup(tmp_path)
    seen = []
    original = frozen_module.run_episode

    def spy(**kwargs):
        seen.append(kwargs["episode_id"])
        return original(**kwargs)

    monkeypatch.setattr(frozen_module, "run_episode", spy)
    result = run_episode_r2(engine=engine, registry=registry, workers=workers, policy=TrivialPolicy(max_runs=1),
                            brief=_brief("timing"), start_state=start, episode_id="off_spy")
    assert seen == ["off_spy"]
    assert result.stop_reason == "budget_exhausted"
    assert "loop_options" not in Journal(engine.store.root).read("off_spy")[0]["envelope"]


@pytest.mark.parametrize("policy_kind", ["trivial", "random"])
def test_frozen_journal_replays_byte_for_byte_with_options_off(tmp_path: Path, policy_kind: str) -> None:
    engine, registry, workers, backend, start = _setup(tmp_path)

    def policy():
        return TrivialPolicy(max_runs=1) if policy_kind == "trivial" else RandomPolicy(seed=7, max_runs=2)

    kwargs = dict(engine=engine, registry=registry, workers=workers, brief=_brief("area"), start_state=start,
                  episode_id=f"r1_{policy_kind}")
    first = run_episode(policy=policy(), **kwargs)
    before = _journal_bytes(engine, kwargs["episode_id"])
    calls = list(backend.calls)
    again = run_episode_r2(policy=policy(), **kwargs)
    assert _journal_bytes(engine, kwargs["episode_id"]) == before
    assert backend.calls == calls
    assert again.model_copy(update={"resumed": False}) == first
    assert again.resumed


def test_interrupted_frozen_episode_resumes_identically_with_options_off(tmp_path: Path) -> None:
    engine, registry, _, backend, start = _setup(tmp_path, fail_on_call=2)
    workers = {"area": _worker("area", MetricName.PACKED_LUTS, "min", ("a1", "a2"))}
    kwargs = dict(engine=engine, registry=registry, workers=workers, brief=_brief("area", budget=2),
                  start_state=start, episode_id="r1_resume")
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_episode(policy=TrivialPolicy(max_runs=2), **kwargs)
    result = run_episode_r2(policy=TrivialPolicy(max_runs=2), **kwargs)
    assert result.resumed and result.recipe_runs == 2
    final_bytes = _journal_bytes(engine, "r1_resume")
    replay = run_episode(policy=TrivialPolicy(max_runs=2), **kwargs)          # the frozen loop accepts it
    assert _journal_bytes(engine, "r1_resume") == final_bytes
    assert replay.model_copy(update={"resumed": False}) == result.model_copy(update={"resumed": False})


def test_options_on_change_nothing_when_no_option_applies(tmp_path: Path) -> None:
    engine, registry, workers, _, start = _setup(tmp_path / "a")
    frozen_result = run_episode(engine=engine, registry=registry, workers=workers,
                                policy=TrivialPolicy(max_runs=1), brief=_brief("timing"), start_state=start,
                                episode_id="same")
    engine2, registry2, workers2, _, start2 = _setup(tmp_path / "b")
    r2_result = run_episode_r2(engine=engine2, registry=registry2, workers=workers2,
                               policy=TrivialPolicy(max_runs=1), brief=_brief("timing"), start_state=start2,
                               episode_id="same", options=ROUND2)
    a = _normalised(Journal(engine.store.root).read("same"))
    b = _normalised(Journal(engine2.store.root).read("same"))
    for side, root in ((a, engine.store.root), (b, engine2.store.root)):
        text = __import__("json").dumps(side)
        assert str(root) in text
    import json

    a_text = json.dumps(a).replace(str(tmp_path / "a"), "<tmp>")
    b_text = json.dumps(b).replace(str(tmp_path / "b"), "<tmp>")
    assert a_text == b_text
    assert frozen_result.stop_reason == r2_result.stop_reason == "budget_exhausted"
    assert Journal(engine2.store.root).read("same")[0]["envelope"]["loop_options"] == ROUND2.as_json()


def test_journal_cannot_resume_under_other_options(tmp_path: Path) -> None:
    engine, registry, workers, _, start = _setup(tmp_path)
    kwargs = dict(engine=engine, registry=registry, workers=workers, brief=_brief("timing"), start_state=start,
                  episode_id="opts")
    run_episode_r2(policy=TrivialPolicy(max_runs=1), options=ROUND2, **kwargs)
    with pytest.raises(ValueError, match="resume configuration differs"):
        run_episode_r2(policy=TrivialPolicy(max_runs=1), **kwargs)
    with pytest.raises(ValueError, match="resume configuration differs"):
        run_episode_r2(policy=TrivialPolicy(max_runs=1), options=LoopOptions(empty_call_observation=True), **kwargs)


# ---------------------------------------------------------------------------------------------------- empty calls
def test_frozen_loop_ends_at_an_empty_call(tmp_path: Path) -> None:
    engine, registry, workers, _, start = _setup(tmp_path)
    policy = Scripted([("area", "start"), ("area", "start"), ("timing", "start")])
    result = run_episode(engine=engine, registry=registry, workers=workers, policy=policy,
                         brief=_brief("timing", budget=3), start_state=start, episode_id="frozen_empty")
    assert result.stop_reason == "no_recipes_available" and result.recipe_runs == 1


def test_empty_call_is_an_observation_and_costs_no_budget(tmp_path: Path) -> None:
    engine, registry, workers, backend, start = _setup(tmp_path)
    policy = Scripted([("area", "start"), ("area", "start"), ("timing", "start"), ("timing", "current")])
    result = run_episode_r2(engine=engine, registry=registry, workers=workers, policy=policy,
                            brief=_brief("timing", budget=3), start_state=start, episode_id="obs",
                            options=LoopOptions(empty_call_observation=True))
    events = Journal(engine.store.root).read("obs")
    selections = [e for e in events if e["type"] == "selection"]
    assert [len(e["attempt_run_ids"]) for e in selections] == [1, 0, 1, 1]
    assert selections[1]["recipe_runs"] == 0
    assert backend.calls == ["area", "timing", "timing"]
    assert result.recipe_runs == 3 and result.stop_reason == "budget_exhausted"
    facts = loop_facts(events)
    assert facts["empty_calls"] == 1 and facts["calls"] == 4


def test_consecutive_empty_calls_are_bounded(tmp_path: Path) -> None:
    engine, registry, workers, _, start = _setup(tmp_path)
    policy = Scripted([("area", "start")] * 10)
    result = run_episode_r2(engine=engine, registry=registry, workers=workers, policy=policy,
                            brief=_brief("timing", budget=5), start_state=start, episode_id="bound",
                            options=LoopOptions(empty_call_observation=True))
    events = Journal(engine.store.root).read("bound")
    empties = [e for e in events if e["type"] == "selection" and not e["attempt_run_ids"]]
    assert len(empties) == MAX_CONSECUTIVE_EMPTY
    assert result.stop_reason == "no_recipes_available" and result.recipe_runs == 1
    assert policy.asked == 1 + MAX_CONSECUTIVE_EMPTY


# ---------------------------------------------------------------------------------------------------- stop rule
def test_early_stop_is_refused_once_and_asked_again_with_the_note(tmp_path: Path) -> None:
    engine, registry, workers, _, start = _setup(tmp_path)
    policy = Scripted([None, None])
    result = run_episode_r2(engine=engine, registry=registry, workers=workers, policy=policy,
                            brief=_brief("timing"), start_state=start, episode_id="stop_twice",
                            options=LoopOptions(stop_rule=True))
    events = Journal(engine.store.root).read("stop_twice")
    assert [e["type"] for e in events][1:] == ["stop_attempt", "stop", "final"]
    assert policy.seen_notes == [None, STOP_NOTE]
    assert result.stop_reason == "policy_stop" and result.recipe_runs == 0
    facts = loop_facts(events)
    assert facts["stop_rule_firings"] == 1 and facts["stop_after_second_ask"]


def test_reasked_policy_may_continue(tmp_path: Path) -> None:
    # Timing brief (LUT limit 101): timing on start is brief-invalid (no gain), area on that output gains 0.04 WNS,
    # area on start is worse (no gain), area on start again is an empty call (no gain): the next stop is accepted.
    engine, registry, workers, _, start = _setup(tmp_path)
    policy = Scripted([None, ("timing", "start"), ("area", "current"), ("area", "start"), ("area", "start"), None])
    result = run_episode_r2(engine=engine, registry=registry, workers=workers, policy=policy,
                            brief=_brief("timing", budget=6), start_state=start, episode_id="continue",
                            noise={MetricName.WNS_NS: 0.01}, options=ROUND2)
    events = Journal(engine.store.root).read("continue")
    kinds = [e["type"] for e in events if e["type"] in ("stop_attempt", "mandate", "stop")]
    assert kinds == ["stop_attempt", "mandate", "mandate", "mandate", "mandate", "stop"]
    assert policy.seen_notes == [None, STOP_NOTE, None, None, None, None]
    assert result.stop_reason == "policy_stop" and result.recipe_runs == 3
    facts = loop_facts(events)
    assert facts["stop_after_second_ask"] is False and facts["stop_rule_firings"] == 1 and facts["empty_calls"] == 1


def test_stop_after_a_gain_is_refused(tmp_path: Path) -> None:
    # Area brief (WNS floor -1.1): timing on start gains nothing on LUTs, area on start gains 3 LUTs > band 1.
    engine, registry, workers, _, start = _setup(tmp_path)
    policy = Scripted([("timing", "start"), ("area", "start"), None, None])
    result = run_episode_r2(engine=engine, registry=registry, workers=workers, policy=policy,
                            brief=_brief("area", budget=6), start_state=start, episode_id="gain",
                            noise={MetricName.PACKED_LUTS: 1.0}, options=ROUND2)
    events = Journal(engine.store.root).read("gain")
    attempt = next(e for e in events if e["type"] == "stop_attempt")
    assert attempt["completed_calls"] == 2 and attempt["last_two_gains"] == [0.0, pytest.approx(3.0)]
    assert attempt["goal_band"] == 1.0 and attempt["budget_left"] == 4
    assert policy.seen_notes == [None, None, None, STOP_NOTE]
    assert result.stop_reason == "policy_stop"


def test_gain_within_noise_does_not_block_stop(tmp_path: Path) -> None:
    engine, registry, workers, _, start = _setup(tmp_path)
    policy = Scripted([("area", "start"), ("area", "current"), None])
    result = run_episode_r2(engine=engine, registry=registry, workers=workers, policy=policy,
                            brief=_brief("area", budget=6), start_state=start, episode_id="band",
                            noise={MetricName.PACKED_LUTS: 5.0}, options=ROUND2)
    events = Journal(engine.store.root).read("band")
    assert not any(e["type"] == "stop_attempt" for e in events)
    assert result.stop_reason == "policy_stop" and result.recipe_runs == 2


def test_empty_calls_count_as_calls_without_gain(tmp_path: Path) -> None:
    # area gains 3 LUTs, timing gains nothing, the repeated area call on start is empty: two calls without gain.
    engine, registry, workers, _, start = _setup(tmp_path)
    policy = Scripted([("area", "start"), ("timing", "start"), ("area", "start"), None])
    run_episode_r2(engine=engine, registry=registry, workers=workers, policy=policy,
                   brief=_brief("area", budget=6), start_state=start, episode_id="emptygain",
                   noise={MetricName.PACKED_LUTS: 1.0}, options=ROUND2)
    events = Journal(engine.store.root).read("emptygain")
    assert not any(e["type"] == "stop_attempt" for e in events)
    assert loop_facts(events)["empty_calls"] == 1


class Crashing(Scripted):
    """Raises once, on the re-ask after a refused stop (a crash between the two asks)."""

    def __init__(self, script) -> None:
        super().__init__(script)
        self.crashed = False

    def decide(self, view):
        if self.harness_note is not None and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated crash before the second ask")
        return super().decide(view)


def test_resume_after_a_refused_stop_asks_again_with_the_note(tmp_path: Path) -> None:
    engine, registry, workers, _, start = _setup(tmp_path)
    kwargs = dict(engine=engine, registry=registry, workers=workers, brief=_brief("timing"), start_state=start,
                  episode_id="resume_stop", options=ROUND2)
    policy = Crashing([None, None, None])
    with pytest.raises(RuntimeError, match="second ask"):
        run_episode_r2(policy=policy, **kwargs)
    assert [e["type"] for e in Journal(engine.store.root).read("resume_stop")] == ["start", "stop_attempt"]
    fresh = Scripted([None])
    result = run_episode_r2(policy=fresh, **kwargs)
    assert fresh.seen_notes == [STOP_NOTE]
    assert result.resumed and result.stop_reason == "policy_stop"
    assert [e["type"] for e in Journal(engine.store.root).read("resume_stop")] == [
        "start", "stop_attempt", "stop", "final"]


def test_call_gains_use_the_best_brief_valid_goal(tmp_path: Path) -> None:
    engine, registry, workers, _, start = _setup(tmp_path)
    policy = Scripted([("timing", "start"), ("area", "current")])
    run_episode_r2(engine=engine, registry=registry, workers=workers, policy=policy, brief=_brief("timing", budget=2),
                   start_state=start, episode_id="gains", options=LoopOptions(empty_call_observation=True))
    from orch.episode_r2 import _MachineR2  # noqa: F401  (the machine is rebuilt below from the journal)

    events = Journal(engine.store.root).read("gains")
    # timing call: WNS -1.0 -> -0.95 but LUTs 100 -> 102 > 101 (limit 1.01x): brief-invalid, so no gain;
    # area call on that state: LUTs 99 and WNS -0.96: brief-valid, gain 0.04 over the start
    machine = _rebuild(engine, registry, workers, policy, "gains", events)
    gains = call_gains(machine)
    assert gains[0] == pytest.approx(0.0) and gains[1] == pytest.approx(0.04)


def _rebuild(engine, registry, workers, policy, episode_id, events):
    from orch.contract.models import Evidence
    from orch.episode_r2 import _MachineR2

    first = events[0]
    start_state = __import__("orch.contract.models", fromlist=["DesignState"]).DesignState.model_validate(
        first["envelope"]["start_state"])
    machine = _MachineR2(
        engine=engine, journal=Journal(engine.store.root), episode_id=episode_id, workers=workers,
        registry=registry, policy=policy, brief=__import__("orch.contract.models", fromlist=["Brief"]).Brief
        .model_validate(first["envelope"]["brief"]), start_state=start_state,
        start_evidence=Evidence.model_validate(first["start_evidence"]), noise={}, telemetry_provider=None,
    )
    for event in events[1:]:
        if event["type"] in ("validation", "final"):
            break
        machine.accept(event)
    return machine
