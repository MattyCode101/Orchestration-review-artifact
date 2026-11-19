"""Round-2 episode loop options (D6-v2; the authors' decisions 3 and 4, 2026-11-12), kept out of the frozen loop.

orch/episode.py is frozen and is not edited. This module imports its machine, worker completion, final pick and
envelope, and differs from its loop only where an option is on:

  empty_call_observation
      A worker call that had no recipe to run ("no options left") no longer ends the episode. Its selection event
      (no attempt, nothing selected, zero runs) is the observation, the call stays in the view, and the policy is
      asked again. It costs no budget. A call that ran recipes and selected nothing ("nothing selectable") already
      continued in the frozen loop, and its runs still count. MAX_CONSECUTIVE_EMPTY empty calls in a row end the
      episode with the frozen reason "no_recipes_available" (a bound, so a policy cannot loop without cost).
      A call whose every attempt was an infrastructure transient still ends the episode, as before.

  stop_rule
      A STOP is accepted only when the budget is spent, or when the last two completed calls produced no gain
      beyond noise on the goal (the best brief-valid goal value improved by at most the goal's noise band; an
      empty call has no gain). Otherwise the attempted stop is journalled ("stop_attempt") and the policy is asked
      again once, with STOP_NOTE as one line in its view (policies that render a view expose `harness_note`);
      a second STOP is accepted. This mirrors goal-repair's two-stall stop.

With both options off (the default) run_episode_r2 calls the frozen run_episode itself, so round-1 journals replay
byte for byte. With an option on, the envelope records the options (a journal cannot resume under other options)
and the machine accepts the one new event type, stop_attempt.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from orch import episode as frozen
from orch.contract.models import Brief, DesignState, Evidence, MetricName
from orch.engine.apply import StepEngine
from orch.engine.journal import Journal
from orch.engine.store import _file_lock
from orch.episode import (
    EpisodeResult,
    FinalValidation,
    _complete_mandate,
    _envelope,
    _finish,
    _Machine,
)
from orch.interfaces import Policy
from orch.recipes.registry import Registry
from orch.workers.base import GenericWorker
from orch.workers.fitted import FittedWorker

STOP_NOTE = "Stop is available after two calls without gain beyond noise."
MAX_CONSECUTIVE_EMPTY = 3
OPTIONS_SCHEMA = "loop-options-r2-v1"


@dataclass(frozen=True)
class LoopOptions:
    empty_call_observation: bool = False
    stop_rule: bool = False

    @property
    def any_on(self) -> bool:
        return self.empty_call_observation or self.stop_rule

    def as_json(self) -> dict[str, Any]:
        return {
            "schema": OPTIONS_SCHEMA,
            "empty_call_observation": self.empty_call_observation,
            "stop_rule": self.stop_rule,
            "max_consecutive_empty": MAX_CONSECUTIVE_EMPTY,
            "stop_note": STOP_NOTE,
        }


ROUND2 = LoopOptions(empty_call_observation=True, stop_rule=True)


class _MachineR2(_Machine):
    """The frozen machine plus the stop_attempt event."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reask_pending = False
        self.stop_attempts: list[dict[str, Any]] = []

    def accept(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "stop_attempt":
            if self.stop_reason is not None or self.pending_index() is not None:
                raise ValueError("stop attempt follows stop or unfinished mandate")
            if self.reask_pending:
                raise ValueError("two refused stops without a decision between them")
            if event.get("decision_index") != len(self.mandates):
                raise ValueError("stop attempt index differs from journal replay")
            if event.get("id") != f"stop_attempt_{len(self.stop_attempts)}":
                raise ValueError("stop attempt ids are not contiguous")
            self.reask_pending = True
            self.stop_attempts.append(dict(event))
            return
        super().accept(event)
        if kind in ("mandate", "stop"):
            self.reask_pending = False


def trailing_empty_calls(machine: _Machine) -> int:
    """Completed calls at the end of the episode that had no recipe to run."""

    count = 0
    for index in range(len(machine.mandates) - 1, -1, -1):
        if index not in machine.completed or machine.attempts.get(index):
            break
        count += 1
    return count


def call_gains(machine: _Machine) -> list[float]:
    """Per completed call: the change of the best brief-valid goal value it caused (goal direction, >= 0)."""

    goal = machine.brief.objective.metric
    sign = 1.0 if machine.brief.objective.direction == "max" else -1.0

    def best(pool: list) -> float:
        values = [
            sign * visit.evidence.metric(goal).value
            for visit in pool
            if visit.verdict.brief_valid and visit.evidence.metric(goal) is not None
        ]
        return max(values)

    pool = [machine.start]
    gains: list[float] = []
    for index in sorted(machine.completed):
        before = best(pool)
        for applied in machine.attempts[index]:
            visit = machine.visit_by_run.get(applied.outcome.run.id)
            if visit is not None:
                pool.append(visit)
        gains.append(best(pool) - before)
    return gains


def stop_allowed(machine: _Machine, budget_left: int) -> tuple[bool, dict[str, Any]]:
    gains = call_gains(machine)
    band = float(machine.noise.get(MetricName(machine.brief.objective.metric), 0.0))
    last_two = gains[-2:]
    stalled = len(last_two) == 2 and all(gain <= band for gain in last_two)
    facts = {
        "budget_left": budget_left,
        "completed_calls": len(gains),
        "last_two_gains": [round(g, 9) for g in last_two],
        "goal_band": band,
    }
    return budget_left <= 0 or stalled, facts


def _decide(policy: Policy, view, note: str | None):
    if note is None or not hasattr(policy, "harness_note"):
        return policy.decide(view)
    policy.harness_note = note
    try:
        return policy.decide(view)
    finally:
        policy.harness_note = None


def _run_episode_locked_r2(
    *,
    engine: StepEngine,
    registry: Registry,
    workers: Mapping[str, GenericWorker],
    policy: Policy,
    brief: Brief,
    start_state: DesignState,
    episode_id: str,
    noise: Mapping[MetricName, float] | None,
    validator: Callable[[DesignState, DesignState], FinalValidation] | None,
    telemetry_provider: Callable[..., Mapping[str, object]] | None,
    options: LoopOptions,
) -> EpisodeResult:
    """The frozen _run_episode_locked with the options' branches; the preamble is the frozen one."""

    if not start_state.fake and brief.equivalence_policy == "final_only" and validator is None:
        raise ValueError("real final_only episode requires a validator callback")
    if any(isinstance(worker, FittedWorker) for worker in workers.values()):
        if telemetry_provider is None or not callable(telemetry_provider):
            raise ValueError("fitted episode needs a callable telemetry provider")
    normalized_noise = {MetricName(name): float(value) for name, value in (noise or {}).items()}
    if any(value < 0 for value in normalized_noise.values()):
        raise ValueError("noise bands must be nonnegative")
    journal = Journal(engine.store.root)
    journal.path(episode_id)
    envelope = _envelope(
        engine=engine, registry=registry, workers=workers, policy=policy, brief=brief,
        start_state=start_state, noise=normalized_noise, telemetry_provider=telemetry_provider,
    )
    envelope["episode_id"] = episode_id
    envelope["loop_options"] = options.as_json()
    events = journal.read(episode_id)
    resumed = bool(events)
    if events:
        first = events[0]
        if first.get("id") != "start" or first.get("type") != "start":
            raise ValueError("episode journal lacks a start event")
        if first.get("envelope") != envelope:
            raise ValueError("episode resume configuration differs from its journal")
        start_evidence = Evidence.model_validate(first["start_evidence"])
        if (
            start_evidence.state_id != start_state.id
            or start_evidence.meter_version != engine.meter_version
            or start_evidence.fake != start_state.fake
        ):
            raise ValueError("journal start evidence differs from current start")
        engine.store.state_path(start_state.id)
        import hashlib

        version_key = hashlib.sha256(engine.meter_version.encode("utf-8")).hexdigest()
        saved_evidence_path = (
            engine.store.root / "measurements" / start_state.id / version_key / "evidence.json"
        )
        saved_evidence = Evidence.model_validate_json(saved_evidence_path.read_bytes())
        if saved_evidence != start_evidence:
            raise ValueError("journal start evidence differs from the persisted meter")
    else:
        start_evidence = engine.measure_start(start_state)
        journal.append(
            episode_id,
            {
                "id": "start",
                "type": "start",
                "envelope": envelope,
                "start_evidence": start_evidence.model_dump(mode="json"),
            },
        )

    machine = _MachineR2(
        engine=engine, journal=journal, episode_id=episode_id, workers=workers, registry=registry,
        policy=policy, brief=brief, start_state=start_state, start_evidence=start_evidence,
        noise=normalized_noise, telemetry_provider=telemetry_provider,
    )
    for event in events[1:]:
        machine.accept(event)
    cards = tuple(worker.card for worker in workers.values())
    while True:
        if machine.final is not None or machine.stop_reason is not None:
            return _finish(machine=machine, policy=policy, validator=validator, resumed=resumed)
        pending = machine.pending_index()
        if pending is not None:
            stall = _complete_mandate(
                machine=machine, registry=registry, policy=policy, decision_index=pending,
            )
            if stall == "no_recipes_available" and options.empty_call_observation:
                continue                         # the observation; the bound is checked below
            if stall is not None:
                machine.record({"id": "stop", "type": "stop", "reason": stall})
            continue
        if options.empty_call_observation and trailing_empty_calls(machine) >= MAX_CONSECUTIVE_EMPTY:
            machine.record({"id": "stop", "type": "stop", "reason": "no_recipes_available"})
            continue
        if machine.recipe_runs >= brief.budget_runs:
            machine.record({"id": "stop", "type": "stop", "reason": "budget_exhausted"})
            continue
        view = machine.view(cards)
        reask = options.stop_rule and machine.reask_pending
        mandate = _decide(policy, view, STOP_NOTE if reask else None)
        if mandate is None:
            if options.stop_rule and not reask:
                allowed, facts = stop_allowed(machine, view.budget_left)
                if not allowed:
                    machine.record({
                        "id": f"stop_attempt_{len(machine.stop_attempts)}",
                        "type": "stop_attempt",
                        "decision_index": len(machine.mandates),
                        **facts,
                    })
                    continue
            machine.record({"id": "stop", "type": "stop", "reason": "policy_stop"})
            continue
        mandate.validate_against(brief, start_evidence)
        if mandate.worker_id not in workers:
            raise ValueError("policy named an unregistered worker")
        base = machine.latest_visit(mandate.base_state_id)
        _, trace, _ = machine.planning(mandate, base)
        decision_index = len(machine.mandates)
        event = {
            "id": f"mandate_{decision_index}",
            "type": "mandate",
            "index": decision_index,
            "budget_left_before": view.budget_left,
            "mandate": mandate.model_dump(mode="json"),
        }
        if trace is not None:
            event["worker_plan"] = trace
        machine.record(event)


def run_episode_r2(
    *,
    engine: StepEngine,
    registry: Registry,
    workers: Mapping[str, GenericWorker],
    policy: Policy,
    brief: Brief,
    start_state: DesignState,
    episode_id: str,
    noise: Mapping[MetricName, float] | None = None,
    validator: Callable[[DesignState, DesignState], FinalValidation] | None = None,
    telemetry_provider: Callable[..., Mapping[str, object]] | None = None,
    options: LoopOptions = LoopOptions(),
) -> EpisodeResult:
    """run_episode with round-2 options; with every option off it is the frozen run_episode itself."""

    if not options.any_on:
        return frozen.run_episode(
            engine=engine, registry=registry, workers=workers, policy=policy, brief=brief,
            start_state=start_state, episode_id=episode_id, noise=noise, validator=validator,
            telemetry_provider=telemetry_provider,
        )
    journal = Journal(engine.store.root)
    lock_path = journal.path(episode_id).parent / "episode.lock"
    with _file_lock(lock_path):
        return _run_episode_locked_r2(
            engine=engine, registry=registry, workers=workers, policy=policy, brief=brief,
            start_state=start_state, episode_id=episode_id, noise=noise, validator=validator,
            telemetry_provider=telemetry_provider, options=options,
        )


def loop_facts(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts a report needs from one journal: empty calls, refused stops, stops after the second ask."""

    selections = [e for e in events if e.get("type") == "selection"]
    attempts = [e for e in events if e.get("type") == "stop_attempt"]
    stop = next((e for e in events if e.get("type") == "stop"), None)
    last_kind = None
    for e in events:
        if e.get("type") == "stop":
            break
        if e.get("type") in ("stop_attempt", "mandate"):
            last_kind = e["type"]
    return {
        "calls": len(selections),
        "empty_calls": sum(1 for e in selections if not e.get("attempt_run_ids")),
        "nothing_selected_calls": sum(
            1 for e in selections if e.get("attempt_run_ids") and e.get("selected_run_id") is None
        ),
        "stop_rule_firings": len(attempts),
        "stop_after_second_ask": bool(stop and stop.get("reason") == "policy_stop" and last_kind == "stop_attempt"),
        "stop_reason": stop.get("reason") if stop else None,
    }


__all__ = [
    "LoopOptions", "MAX_CONSECUTIVE_EMPTY", "ROUND2", "STOP_NOTE", "call_gains", "loop_facts", "run_episode_r2",
    "stop_allowed", "trailing_empty_calls",
]
