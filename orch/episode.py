"""Journalled episode loop and final-pick rule shared by view-only policies.

The loop owns decisions and state navigation. StepEngine alone executes tools.
A physically valid brief violation remains available as a later base; final
selection always uses the episode's fixed brief and its start evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from orch.contract.models import (
    AppliedStep,
    Brief,
    DesignState,
    EquivalenceCheck,
    Evidence,
    FrozenModel,
    Mandate,
    MetricName,
    Record,
    RunStatus,
    Step,
    Verdict,
)
from orch.contract.views import CallSummary, PolicyView, StateSummary, StepSummary
from orch.engine.apply import StepEngine
from orch.engine.gate import GATE_VERSION, judge
from orch.engine.journal import Journal
from orch.engine.store import _file_lock, _json_bytes
from orch.interfaces import Policy
from orch.recipes.registry import Registry
from orch.workers.base import GenericWorker, run_worker
from orch.workers.fitted import FittedWorker


class FinalValidation(FrozenModel):
    """One final-only weak validator result with its process artifacts."""

    check: EquivalenceCheck
    run_id: str | None = None
    artifact_dir: str | None = None
    note: str | None = None


class ValidationRecord(FrozenModel):
    """The ranked candidate checked by a final-only validator."""

    rank: int
    state_id: str
    evidence_id: str
    candidate_run_id: str | None = None
    outcome: FinalValidation


class EpisodeResult(Record):
    """Serializable final episode result; fake quality is always labelled."""

    episode_id: str
    brief_id: str
    brief_hash: str
    policy_id: str
    policy_version: str
    start_state_id: str
    steps: tuple[Step, ...]
    final_state: DesignState | None
    final_evidence: Evidence | None
    final_verdict: Verdict | None
    final_rank: int | None = None
    validation_status: str
    validation_attempts: tuple[ValidationRecord, ...] = ()
    stop_reason: str
    recipe_runs: int
    charged_wall_s: float
    fake: bool
    resumed: bool = False
    journal_path: str


@dataclass(frozen=True)
class _Visit:
    """One measured state occurrence; equal bytes may have different lineage."""

    state: DesignState
    evidence: Evidence
    verdict: Verdict
    run_id: str | None
    producer_recipe_id: str | None
    lineage_runs: int


def _summary(visit: _Visit) -> StateSummary:
    """Expose only policy-safe metric values, never raw reports or scripts."""

    return StateSummary(
        state_id=visit.state.id,
        metrics={name: value.value for name, value in visit.evidence.metrics.items()},
        physically_valid=visit.verdict.physically_valid,
        brief_valid=visit.verdict.brief_valid,
    )


def _rank(brief: Brief, visits: list[_Visit]) -> list[_Visit]:
    """Rank every brief-valid occurrence, including the start, deterministically."""

    other = (
        MetricName.PACKED_LUTS
        if brief.objective.metric == MetricName.WNS_NS
        else MetricName.WNS_NS
    )

    def key(visit: _Visit) -> tuple[float, bool, float, int, str, str]:
        primary = visit.evidence.metric(brief.objective.metric)
        if primary is None:
            raise ValueError("brief-valid state lacks its objective metric")
        secondary = visit.evidence.metric(other)
        first = -primary.value if brief.objective.direction == "max" else primary.value
        if secondary is None:
            second = 0.0
        else:
            second = secondary.value if other == MetricName.PACKED_LUTS else -secondary.value
        return (
            first,
            secondary is None,
            second,
            visit.lineage_runs,
            visit.state.id,
            visit.run_id or "",
        )

    return sorted((visit for visit in visits if visit.verdict.brief_valid), key=key)


def _worker_snapshot(
    workers: Mapping[str, GenericWorker], registry: Registry
) -> list[dict[str, object]]:
    """Freeze the order and exact recipe versions for a resumable episode."""

    snapshot: list[dict[str, object]] = []
    for worker_id, worker in workers.items():
        if worker_id != worker.card.id:
            raise ValueError("worker mapping key differs from card id")
        recipes = []
        for recipe_id in worker.recipe_ids:
            recipe = registry.get(recipe_id)
            if recipe.worker != worker_id:
                raise ValueError(f"recipe {recipe_id} belongs to {recipe.worker}")
            recipes.append({"id": recipe.id, "version": recipe.version})
        row: dict[str, object] = {
            "worker_id": worker_id,
            "card": worker.card.model_dump(mode="json"),
            "recipes": recipes,
        }
        if isinstance(worker, FittedWorker):
            row["model"] = {
                "label": worker.model_label,
                "sha256": worker.model_sha256,
            }
        snapshot.append(row)
    return snapshot


def _envelope(
    *,
    engine: StepEngine,
    registry: Registry,
    workers: Mapping[str, GenericWorker],
    policy: Policy,
    brief: Brief,
    start_state: DesignState,
    noise: Mapping[MetricName, float],
    telemetry_provider: Callable[..., Mapping[str, object]] | None,
) -> dict[str, object]:
    """Bind journal resume to the actual brief, code and recipe selection."""

    gate_source = Path(__file__).with_name("engine") / "gate.py"
    return {
        "episode_id": "",
        "brief": brief.model_dump(mode="json"),
        "brief_hash": brief.fingerprint(),
        "start_state": start_state.model_dump(mode="json"),
        "policy": {"id": policy.id, "version": policy.version},
        "workers": _worker_snapshot(workers, registry),
        "noise": {name.value: value for name, value in noise.items()},
        "meter_version": engine.meter_version,
        "gate_version": GATE_VERSION,
        "gate_sha256": hashlib.sha256(gate_source.read_bytes()).hexdigest(),
        "episode_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(
            (Path(__file__).with_name("workers") / "base.py").read_bytes()
        ).hexdigest(),
        "fitted_worker_sha256": hashlib.sha256(
            (Path(__file__).with_name("workers") / "fitted.py").read_bytes()
        ).hexdigest(),
        "telemetry_provider": (
            telemetry_provider.identity() if telemetry_provider is not None else None
        ),
        "cache_enabled": engine.cache_enabled,
        "fake": start_state.fake,
    }


def _step_id(episode_id: str, index: int, applied: AppliedStep) -> str:
    """Bind a step identity to its run and fresh gate verdict."""

    payload = [episode_id, index, applied.outcome.run.id, applied.verdict.id]
    return "step_" + hashlib.sha256(_json_bytes(payload)).hexdigest()


def _charged(attempts: list[AppliedStep]) -> tuple[int, float]:
    """Count recipe choices; exhausted infrastructure transients do not count."""

    billable = [
        attempt.outcome.run
        for attempt in attempts
        if attempt.outcome.run.status != RunStatus.TRANSIENT
    ]
    return len(billable), sum(run.wall_s for run in billable)


class _Machine:
    """Rebuild one episode from checksummed journal events and persisted runs."""

    def __init__(
        self,
        *,
        engine: StepEngine,
        journal: Journal,
        episode_id: str,
        workers: Mapping[str, GenericWorker],
        registry: Registry,
        policy: Policy,
        brief: Brief,
        start_state: DesignState,
        start_evidence: Evidence,
        noise: Mapping[MetricName, float],
        telemetry_provider: Callable[..., Mapping[str, object]] | None,
    ) -> None:
        self.engine = engine
        self.journal = journal
        self.episode_id = episode_id
        self.workers = workers
        self.registry = registry
        self.policy = policy
        self.recipe_versions = {
            recipe_id: registry.get(recipe_id).version
            for worker in workers.values()
            for recipe_id in worker.recipe_ids
        }
        self.brief = brief
        self.noise = noise
        self.telemetry_provider = telemetry_provider
        self.start_evidence = start_evidence
        start_verdict = judge(brief, start_evidence, start_evidence, start_evidence)
        if not start_verdict.brief_valid:
            raise ValueError("episode start is not independently measured and brief-valid")
        self.start = _Visit(start_state, start_evidence, start_verdict, None, None, 0)
        self.visits: list[_Visit] = [self.start]
        self.visit_by_run: dict[str, _Visit] = {}
        self.current = self.start
        self.steps: list[Step] = []
        self.step_summaries: list[StepSummary] = []
        self.call_summaries: list[CallSummary] = []
        self.history: dict[str, list[AppliedStep]] = defaultdict(list)
        self.mandates: dict[int, Mandate] = {}
        self.bases: dict[int, _Visit] = {}
        self.budget_before: dict[int, int] = {}
        self.allowed_order: dict[int, tuple[str, ...]] = {}
        self.plan_features: dict[int, Mapping[str, object] | None] = {}
        self.attempts: dict[int, list[AppliedStep]] = {}
        self.completed: set[int] = set()
        self.recipe_runs = 0
        self.charged_wall_s = 0.0
        self.stop_reason: str | None = None
        self.validations: list[ValidationRecord] = []
        self.final: EpisodeResult | None = None

    def latest_visit(self, state_id: str) -> _Visit:
        """Resolve a policy-selected base to a previously measured safe state."""

        for visit in reversed(self.visits):
            if visit.state.id == state_id and visit.verdict.physically_valid:
                return visit
        raise ValueError("mandate base state is not a visited physically valid state")

    def pending_index(self) -> int | None:
        pending = sorted(set(self.mandates) - self.completed)
        if len(pending) > 1:
            raise ValueError("journal contains multiple unfinished mandates")
        return pending[0] if pending else None

    def view(self, worker_cards: tuple) -> PolicyView:
        """Build the compact data-only view exposed to every policy."""

        ranked = _rank(self.brief, self.visits)
        return PolicyView(
            brief=self.brief,
            budget_left=self.brief.budget_runs - self.recipe_runs,
            start=_summary(self.start),
            current=_summary(self.current),
            best_valid=_summary(ranked[0]) if ranked else None,
            visited=tuple(_summary(visit) for visit in self.visits),
            steps=tuple(self.step_summaries),
            calls=tuple(self.call_summaries),
            worker_cards=worker_cards,
            noise_bands=dict(self.noise),
            episode_id=self.episode_id,
            fake=self.start.state.fake,
        )

    def planning(
        self, mandate: Mandate, base: _Visit,
    ) -> tuple[tuple[str, ...], dict[str, object] | None, Mapping[str, object] | None]:
        """Bind a fitted choice to checked telemetry and an exact model hash."""

        worker = self.workers[mandate.worker_id]
        history = tuple(self.history[base.state.id])
        if not isinstance(worker, FittedWorker):
            return worker.plan(base.state, None, mandate, history), None, None
        if self.telemetry_provider is None:
            raise ValueError("fitted episode needs an input telemetry provider")
        features = self.telemetry_provider(
            base.state, base.evidence, base.producer_recipe_id, base.run_id
        )
        if not isinstance(features, Mapping):
            raise TypeError("telemetry provider must return a mapping")
        plan = worker.explain_plan(base.state, features, mandate, history)
        trace = json.loads(_json_bytes({
            "score_plan": asdict(plan),
            "telemetry_source": features.get("telemetry_source"),
            "producer_recipe_id": base.producer_recipe_id,
            "input_state_id": base.state.id,
        }))
        return plan.selected_ids, trace, features

    def record(self, event: dict[str, Any]) -> None:
        """Fsync an event before accepting its effect in memory."""

        self.journal.append(self.episode_id, event)
        self.accept(event)

    def accept(self, event: dict[str, Any]) -> None:
        """Validate and replay one committed event in strict order."""

        kind = event.get("type")
        if kind == "mandate":
            if self.stop_reason is not None or self.pending_index() is not None:
                raise ValueError("mandate follows stop or unfinished mandate")
            index = event.get("index")
            if index != len(self.mandates):
                raise ValueError("mandate index is not contiguous")
            mandate = Mandate.model_validate(event["mandate"])
            mandate.validate_against(self.brief, self.start_evidence)
            if mandate.worker_id not in self.workers:
                raise ValueError("mandate names an unknown worker")
            base = self.latest_visit(mandate.base_state_id)
            left = self.brief.budget_runs - self.recipe_runs
            if left <= 0 or event.get("budget_left_before") != left:
                raise ValueError("mandate budget differs from journal replay")
            self.mandates[index] = mandate
            self.bases[index] = base
            self.budget_before[index] = left
            worker = self.workers[mandate.worker_id]
            ordered, trace, features = self.planning(mandate, base)
            if event.get("worker_plan") != trace:
                raise ValueError("journal fitted score plan differs from current model or telemetry")
            if len(set(ordered)) != len(ordered):
                raise ValueError("worker plan contains duplicate recipe IDs")
            allowed: list[str] = []
            for recipe_id in ordered:
                if recipe_id not in worker.recipe_ids:
                    raise ValueError("worker plan names a recipe outside its set")
                recipe = self.registry.get(recipe_id)
                if recipe.worker != mandate.worker_id:
                    raise ValueError("worker plan names a foreign recipe")
                if recipe.replaces and mandate.settings.preserve_placement:
                    continue
                allowed.append(recipe_id)
            self.allowed_order[index] = tuple(allowed)
            self.plan_features[index] = features
            self.attempts[index] = []
            return

        if kind == "step":
            index = event.get("decision_index")
            if index != self.pending_index():
                raise ValueError("step has no matching pending mandate")
            mandate = self.mandates[index]
            step = Step.model_validate(event["step"])
            verdict = Verdict.model_validate(event["verdict"])
            outcome = self.engine.store.load_outcome(step.run_id)
            attempt_index = len(self.attempts[index])
            expected_recipe = (
                self.allowed_order[index][attempt_index]
                if attempt_index < len(self.allowed_order[index])
                else None
            )
            if (
                step.episode_id != self.episode_id
                or step.index != len(self.steps)
                or step.mandate_id != mandate.id
                or step.parent_state_id != mandate.base_state_id
                or step.output_state_id != (outcome.state.id if outcome.state else None)
                or step.evidence_id != (outcome.evidence.id if outcome.evidence else None)
                or step.verdict_id != verdict.id
                or step.meter_version != self.engine.meter_version
                or step.gate_version != GATE_VERSION
                or step.decider_id != self.policy.id
                or step.decider_version != self.policy.version
                or outcome.run.recipe_version
                != self.recipe_versions.get(outcome.run.recipe_id)
                or outcome.run.input_sha256_before not in (None, mandate.base_state_id)
                or (
                    outcome.run.status != RunStatus.INPUT_MUTATED
                    and outcome.run.input_sha256_after not in (None, mandate.base_state_id)
                )
                or (
                    outcome.state is not None
                    and outcome.state.parent_id != mandate.base_state_id
                )
                or step.fake != self.start.state.fake
                or outcome.run.fake != self.start.state.fake
                or outcome.run.recipe_id != expected_recipe
                or (
                    self.registry.get(outcome.run.recipe_id).replaces
                    and mandate.settings.preserve_placement
                )
                or any(old.run_id == step.run_id for old in self.steps)
            ):
                raise ValueError("step identity or provenance differs from persisted run")
            expected = judge(
                self.brief,
                self.start_evidence,
                self.bases[index].evidence,
                outcome.evidence,
                guards=mandate.guards,
                noise=self.noise,
                run_status=outcome.run.status,
            )
            if verdict != expected:
                raise ValueError("journal verdict differs from the current pure gate")
            applied = AppliedStep(outcome=outcome, verdict=verdict)
            if step.id != _step_id(self.episode_id, step.index, applied):
                raise ValueError("step ID does not bind this run and verdict")
            self.steps.append(step)
            self.attempts[index].append(applied)
            self.history[mandate.base_state_id].append(applied)
            self.step_summaries.append(
                StepSummary(
                    worker_id=mandate.worker_id,
                    settings=mandate.settings,
                    base_state_id=mandate.base_state_id,
                    output_state_id=step.output_state_id,
                    delta_from_parent=verdict.delta_from_parent,
                    delta_from_start=verdict.delta_from_start,
                    verdict_status=verdict.status,
                    within_noise=verdict.within_noise,
                )
            )
            if outcome.run.status != RunStatus.TRANSIENT:
                self.recipe_runs += 1
                self.charged_wall_s += outcome.run.wall_s
            if verdict.physically_valid and outcome.state and outcome.evidence:
                visit = _Visit(
                    outcome.state, outcome.evidence, verdict, outcome.run.id,
                    outcome.run.recipe_id,
                    self.bases[index].lineage_runs + 1,
                )
                self.visits.append(visit)
                self.visit_by_run[outcome.run.id] = visit
            if self.recipe_runs > self.brief.budget_runs:
                raise ValueError("journal exceeded the fixed brief run budget")
            return

        if kind == "selection":
            index = event.get("index")
            if index != self.pending_index():
                raise ValueError("selection has no matching pending mandate")
            mandate = self.mandates[index]
            attempts = self.attempts[index]
            ids = [attempt.outcome.run.id for attempt in attempts]
            if event.get("attempt_run_ids") != ids:
                raise ValueError("selection attempt list differs from journal")
            worker = self.workers[mandate.worker_id]
            chosen = worker.select(attempts, mandate)
            selected_id = chosen.outcome.run.id if chosen else None
            if event.get("selected_run_id") != selected_id:
                raise ValueError("selection differs from worker's current rule")
            charged_runs, charged_wall = _charged(attempts)
            if (
                event.get("recipe_runs") != charged_runs
                or abs(float(event.get("charged_wall_s", -1)) - charged_wall) > 1e-6
                or charged_runs > min(
                    mandate.settings.resolved_max_runs(worker.card),
                    self.budget_before[index],
                )
            ):
                raise ValueError("selection charged count differs from attempts")
            if selected_id is not None:
                self.current = self.visit_by_run[selected_id]
            self.call_summaries.append(CallSummary(
                mandate_id=mandate.id,
                worker_id=mandate.worker_id,
                base_state_id=mandate.base_state_id,
                settings=mandate.settings,
                selected_state_id=(
                    self.visit_by_run[selected_id].state.id if selected_id is not None else None
                ),
                selected_verdict_status=(
                    self.visit_by_run[selected_id].verdict.status
                    if selected_id is not None else None
                ),
                recipe_runs=charged_runs,
                context_ref=mandate.context_ref,
            ))
            self.completed.add(index)
            return

        if kind == "stop":
            if self.stop_reason is not None or self.pending_index() is not None:
                raise ValueError("stop follows another stop or pending mandate")
            reason = event.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError("stop event needs a reason")
            self.stop_reason = reason
            return

        if kind == "validation":
            if self.stop_reason is None or self.final is not None:
                raise ValueError("validation occurred before stop or after final")
            record = ValidationRecord.model_validate(event["record"])
            if record.rank != len(self.validations):
                raise ValueError("validation ranks are not contiguous")
            if (
                record.outcome.check.reference_state_id != self.start.state.id
                or record.outcome.check.vectors != self.brief.equivalence_vectors
            ):
                raise ValueError("validation used the wrong reference or vector count")
            self.validations.append(record)
            return

        if kind == "final":
            if self.stop_reason is None or self.final is not None:
                raise ValueError("final occurred before stop or after final")
            result = EpisodeResult.model_validate(event["result"])
            if (
                result.episode_id != self.episode_id
                or result.brief_hash != self.brief.fingerprint()
                or result.start_state_id != self.start.state.id
                or result.policy_id != self.policy.id
                or result.policy_version != self.policy.version
                or result.steps != tuple(self.steps)
                or result.recipe_runs != self.recipe_runs
                or abs(result.charged_wall_s - self.charged_wall_s) > 1e-6
                or result.stop_reason != self.stop_reason
                or result.validation_attempts != tuple(self.validations)
                or result.fake != self.start.state.fake
                or result.journal_path != str(self.journal.path(self.episode_id))
            ):
                raise ValueError("final result differs from replayed episode")
            if self.start.state.fake and (
                result.validation_status != "skipped_fake" or result.validation_attempts
            ):
                raise ValueError("fake final result has an invalid validation label")
            if result.final_rank is None and result.validation_status == "pass":
                raise ValueError("validation cannot pass without a final pick")
            if result.final_rank is None and any(
                item is not None
                for item in (result.final_state, result.final_evidence, result.final_verdict)
            ):
                raise ValueError("unranked final result contains a selected state")
            if result.final_rank is not None:
                ranked = _rank(self.brief, self.visits)
                if (
                    result.final_rank >= len(ranked)
                    or result.final_state != ranked[result.final_rank].state
                    or result.final_evidence != ranked[result.final_rank].evidence
                    or result.final_verdict != ranked[result.final_rank].verdict
                ):
                    raise ValueError("final pick differs from ranked visited states")
            self.final = result
            return

        raise ValueError(f"unknown episode journal event type: {kind!r}")


class _JournaledEngine:
    """Record every returned apply result before the worker tries another recipe."""

    def __init__(
        self,
        machine: _Machine,
        engine: StepEngine,
        *,
        decision_index: int,
        policy: Policy,
    ) -> None:
        self.machine = machine
        self.engine = engine
        self.decision_index = decision_index
        self.policy = policy

    def apply(self, state, recipe, *, brief, start, parent, guards=(), noise=None):
        applied = self.engine.apply(
            state,
            recipe,
            brief=brief,
            start=start,
            parent=parent,
            guards=guards,
            noise=noise,
        )
        outcome = applied.outcome
        index = len(self.machine.steps)
        step = Step(
            id=_step_id(self.machine.episode_id, index, applied),
            episode_id=self.machine.episode_id,
            index=index,
            mandate_id=self.machine.mandates[self.decision_index].id,
            run_id=outcome.run.id,
            parent_state_id=state.id,
            output_state_id=outcome.state.id if outcome.state else None,
            evidence_id=outcome.evidence.id if outcome.evidence else None,
            verdict_id=applied.verdict.id,
            decider_id=self.policy.id,
            decider_version=self.policy.version,
            tokens=0,
            meter_version=self.engine.meter_version,
            gate_version=GATE_VERSION,
            fake=outcome.run.fake,
        )
        self.machine.record(
            {
                "id": step.id,
                "type": "step",
                "decision_index": self.decision_index,
                "step": step.model_dump(mode="json"),
                "verdict": applied.verdict.model_dump(mode="json"),
            }
        )
        return applied


class _RemainingWorker:
    """Execute the exact remaining order committed with the mandate."""

    def __init__(self, worker: GenericWorker, remaining_ids: tuple[str, ...]) -> None:
        self.worker = worker
        self.card = worker.card
        self.remaining_ids = remaining_ids

    def plan(self, state, features, mandate, history_on_this_state):
        del features, history_on_this_state
        if state.id != mandate.base_state_id or mandate.worker_id != self.card.id:
            raise ValueError("remaining worker plan differs from mandate")
        return self.remaining_ids

    def select(self, candidates, mandate):
        return self.worker.select(candidates, mandate)


def _complete_mandate(
    *,
    machine: _Machine,
    registry: Registry,
    policy: Policy,
    decision_index: int,
) -> str | None:
    """Finish a new or interrupted worker call without repeating journalled runs."""

    mandate = machine.mandates[decision_index]
    worker = machine.workers[mandate.worker_id]
    prior_attempts = list(machine.attempts[decision_index])
    prior_charged, _ = _charged(prior_attempts)
    allowance = min(
        mandate.settings.resolved_max_runs(worker.card),
        machine.budget_before[decision_index],
    )
    remaining = allowance - prior_charged
    if remaining < 0:
        raise ValueError("pending mandate already exceeded its allowance")
    if remaining:
        base = machine.bases[decision_index]
        reduced = mandate.model_copy(
            update={
                "settings": mandate.settings.model_copy(
                    update={"max_runs": remaining}
                )
            }
        )
        run_worker(
            _RemainingWorker(
                worker, machine.allowed_order[decision_index][len(prior_attempts):]
            ),
            mandate=reduced,
            engine=_JournaledEngine(
                machine, machine.engine, decision_index=decision_index, policy=policy
            ),
            registry=registry,
            base_state=base.state,
            base_evidence=base.evidence,
            start_evidence=machine.start_evidence,
            brief=machine.brief,
            remaining_budget=machine.brief.budget_runs - machine.recipe_runs,
            history_on_state=tuple(machine.history[base.state.id]),
            features=machine.plan_features[decision_index],
            noise=machine.noise,
        )
    attempts = machine.attempts[decision_index]
    selected = worker.select(attempts, mandate)
    count, wall = _charged(attempts)
    machine.record(
        {
            "id": f"selection_{decision_index}",
            "type": "selection",
            "index": decision_index,
            "attempt_run_ids": [attempt.outcome.run.id for attempt in attempts],
            "selected_run_id": selected.outcome.run.id if selected else None,
            "recipe_runs": count,
            "charged_wall_s": wall,
        }
    )
    if not attempts:
        return "no_recipes_available"
    if count == 0:
        return "transient_unresolved"
    return None


def _finish(
    *,
    machine: _Machine,
    policy: Policy,
    validator: Callable[[DesignState, DesignState], FinalValidation] | None,
    resumed: bool,
) -> EpisodeResult:
    """Validate ranked final candidates once each, then commit the final pick."""

    if machine.final is not None:
        return machine.final.model_copy(update={"resumed": resumed})
    if machine.stop_reason is None:
        raise ValueError("cannot final-pick before a committed stop event")

    ranked = _rank(machine.brief, machine.visits)
    chosen: _Visit | None = None
    final_rank: int | None = None
    if machine.start.state.fake:
        chosen = ranked[0] if ranked else None
        final_rank = 0 if chosen else None
        validation_status = "skipped_fake"
    elif machine.brief.equivalence_policy == "none":
        chosen = ranked[0] if ranked else None
        final_rank = 0 if chosen else None
        validation_status = "skipped_by_brief"
    else:
        if validator is None:
            raise ValueError("real final_only episode requires a validator callback")
        validation_status = "all_failed"
        for rank, visit in enumerate(ranked):
            if rank < len(machine.validations):
                record = machine.validations[rank]
                if (
                    record.state_id != visit.state.id
                    or record.evidence_id != visit.evidence.id
                    or record.candidate_run_id != visit.run_id
                ):
                    raise ValueError("resumed validation refers to a different ranked state")
            else:
                outcome = validator(machine.start.state, visit.state)
                if isinstance(outcome, EquivalenceCheck):
                    outcome = FinalValidation(check=outcome)
                if not isinstance(outcome, FinalValidation):
                    raise TypeError("validator must return FinalValidation")
                if (
                    outcome.check.reference_state_id != machine.start.state.id
                    or outcome.check.vectors != machine.brief.equivalence_vectors
                ):
                    raise ValueError("validator used the wrong reference or vector count")
                record = ValidationRecord(
                    rank=rank,
                    state_id=visit.state.id,
                    evidence_id=visit.evidence.id,
                    candidate_run_id=visit.run_id,
                    outcome=outcome,
                )
                machine.record(
                    {
                        "id": f"validation_{rank}",
                        "type": "validation",
                        "record": record.model_dump(mode="json"),
                    }
                )
            if record.outcome.check.result == "pass":
                chosen = visit
                final_rank = rank
                validation_status = "pass"
                break

    result = EpisodeResult(
        episode_id=machine.episode_id,
        brief_id=machine.brief.id,
        brief_hash=machine.brief.fingerprint(),
        policy_id=policy.id,
        policy_version=policy.version,
        start_state_id=machine.start.state.id,
        steps=tuple(machine.steps),
        final_state=chosen.state if chosen else None,
        final_evidence=chosen.evidence if chosen else None,
        final_verdict=chosen.verdict if chosen else None,
        final_rank=final_rank,
        validation_status=validation_status,
        validation_attempts=tuple(machine.validations),
        stop_reason=machine.stop_reason,
        recipe_runs=machine.recipe_runs,
        charged_wall_s=machine.charged_wall_s,
        fake=machine.start.state.fake,
        resumed=False,
        journal_path=str(machine.journal.path(machine.episode_id)),
    )
    machine.record(
        {
            "id": "final",
            "type": "final",
            "result": result.model_dump(mode="json"),
        }
    )
    return result.model_copy(update={"resumed": resumed})


def _run_episode_locked(
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
) -> EpisodeResult:
    """Run or resume one bounded episode, with no tool access granted to policy.

    Every completed recipe attempt is fsynced as a Step before the worker tries
    its next recipe. A crash after backend persistence but before this append may
    still require a repeat execution when caching is disabled.
    """

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
        engine=engine,
        registry=registry,
        workers=workers,
        policy=policy,
        brief=brief,
        start_state=start_state,
        noise=normalized_noise,
        telemetry_provider=telemetry_provider,
    )
    envelope["episode_id"] = episode_id
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
        version_key = hashlib.sha256(engine.meter_version.encode("utf-8")).hexdigest()
        saved_evidence_path = (
            engine.store.root / "measurements" / start_state.id
            / version_key / "evidence.json"
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

    machine = _Machine(
        engine=engine,
        journal=journal,
        episode_id=episode_id,
        workers=workers,
        registry=registry,
        policy=policy,
        brief=brief,
        start_state=start_state,
        start_evidence=start_evidence,
        noise=normalized_noise,
        telemetry_provider=telemetry_provider,
    )
    for event in events[1:]:
        machine.accept(event)
    cards = tuple(worker.card for worker in workers.values())
    while True:
        if machine.final is not None or machine.stop_reason is not None:
            return _finish(
                machine=machine,
                policy=policy,
                validator=validator,
                resumed=resumed,
            )
        pending = machine.pending_index()
        if pending is not None:
            stall = _complete_mandate(
                machine=machine,
                registry=registry,
                policy=policy,
                decision_index=pending,
            )
            if stall is not None:
                machine.record({"id": "stop", "type": "stop", "reason": stall})
            continue
        if machine.recipe_runs >= brief.budget_runs:
            machine.record(
                {"id": "stop", "type": "stop", "reason": "budget_exhausted"}
            )
            continue
        view = machine.view(cards)
        mandate = policy.decide(view)
        if mandate is None:
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


def run_episode(
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
) -> EpisodeResult:
    """Serialize one episode across processes, then run or resume its journal.

    The flock spans the entire read, decision, tool-run and final-pick cycle.
    A second driver of the same episode waits and then reads the completed
    journal, while different episodes may proceed concurrently.
    """

    journal = Journal(engine.store.root)
    lock_path = journal.path(episode_id).parent / "episode.lock"
    with _file_lock(lock_path):
        return _run_episode_locked(
            engine=engine,
            registry=registry,
            workers=workers,
            policy=policy,
            brief=brief,
            start_state=start_state,
            episode_id=episode_id,
            noise=noise,
            validator=validator,
            telemetry_provider=telemetry_provider,
        )
