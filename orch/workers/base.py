"""Fixed-order worker selection and the shared worker-call loop.

A worker knows its own objective and configured recipe order. Tool execution
remains in StepEngine; the worker never certifies an output or changes a brief.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from orch.contract.models import (
    AppliedStep,
    Brief,
    DesignState,
    Evidence,
    Mandate,
    MetricName,
    RunStatus,
    VerdictStatus,
    WorkerCard,
)
from orch.engine.apply import StepEngine
from orch.recipes.registry import Registry


@dataclass(frozen=True)
class WorkerRunResult:
    """All outcomes of one mandate, including a possibly invalid intermediate.

    recipe_runs is the number charged to the episode budget. Exhausted
    infrastructure-transient outcomes remain in attempts but are not charged.
    charged_wall_s includes original wall time on cache hits.
    """

    mandate_id: str
    planned_recipe_ids: tuple[str, ...]
    excluded_replacing: tuple[str, ...]
    attempts: tuple[AppliedStep, ...]
    selected: AppliedStep | None
    recipe_runs: int
    charged_wall_s: float


class GenericWorker:
    """A deterministic ordered-list worker, pending the specific order rules."""

    def __init__(self, card: WorkerCard, recipe_ids: Sequence[str]) -> None:
        if len(set(recipe_ids)) != len(recipe_ids):
            raise ValueError("worker recipe order contains duplicates")
        self.card = card
        self.recipe_ids = tuple(recipe_ids)

    def plan(
        self,
        state: DesignState,
        features: Mapping[str, object] | None,
        mandate: Mandate,
        history_on_this_state: Sequence[AppliedStep],
    ) -> tuple[str, ...]:
        """Return the fixed order, omitting recipes already tried on this state.

        The status of each earlier outcome is available in history for later
        worker-specific rules. This provisional worker does not use features.
        """
        if mandate.worker_id != self.card.id:
            raise ValueError("mandate names a different worker")
        if state.id != mandate.base_state_id:
            raise ValueError("mandate names a different base state")
        if state.stage != self.card.accepts_stage:
            raise ValueError("worker does not accept this state stage")
        tried = {
            step.outcome.run.recipe_id
            for step in history_on_this_state
            if step.outcome.run.status != RunStatus.TRANSIENT
        }
        return tuple(recipe_id for recipe_id in self.recipe_ids if recipe_id not in tried)

    def select(
        self,
        candidates: Sequence[AppliedStep],
        mandate: Mandate,
    ) -> AppliedStep | None:
        """Pick this worker's best physically valid, guard-passing result.

        A brief term may fail: such a state is still a possible intermediate
        for the episode policy. The fixed brief is enforced by the gate and
        final pick, never silently by this selector.
        """
        if mandate.worker_id != self.card.id:
            raise ValueError("mandate names a different worker")
        best: AppliedStep | None = None
        best_value: float | None = None
        metric = self.card.objective.metric
        for candidate in candidates:
            outcome = candidate.outcome
            if (
                outcome.run.status != RunStatus.OK
                or outcome.state is None
                or outcome.evidence is None
                or outcome.state.stage != self.card.returns_stage
                or not candidate.verdict.physically_valid
                or not candidate.verdict.guard_pass
                or candidate.verdict.status in (VerdictStatus.INVALID, VerdictStatus.RUN_FAILED)
                or metric not in candidate.verdict.delta_from_start
            ):
                continue
            measured = outcome.evidence.metric(metric)
            if measured is None:
                continue
            value = measured.value
            better = (
                best_value is None
                or (self.card.objective.direction == "max" and value > best_value)
                or (self.card.objective.direction == "min" and value < best_value)
            )
            if better:
                best = candidate
                best_value = value
        return best


def run_worker(
    worker: GenericWorker,
    *,
    mandate: Mandate,
    engine: StepEngine,
    registry: Registry,
    base_state: DesignState,
    base_evidence: Evidence,
    start_evidence: Evidence,
    brief: Brief,
    remaining_budget: int,
    history_on_state: Sequence[AppliedStep] = (),
    features: Mapping[str, object] | None = None,
    noise: Mapping[MetricName, float] | None = None,
) -> WorkerRunResult:
    """Apply this worker's ordered recipes within a validated mandate.

    Cache hits consume one recipe-run allowance and their recorded original
    wall time. The runner handles transient retries inside one recipe choice;
    an exhausted TRANSIENT outcome is recorded but does not use that allowance.
    """
    mandate.validate_against(brief, start_evidence)
    if mandate.worker_id != worker.card.id:
        raise ValueError("mandate names a different worker")
    if mandate.base_state_id != base_state.id:
        raise ValueError("mandate names a different base state")
    if base_evidence.state_id != base_state.id:
        raise ValueError("base evidence names a different state")
    if remaining_budget < 0 or remaining_budget > brief.budget_runs:
        raise ValueError("remaining budget is outside the brief's recipe-run budget")

    ordered = worker.plan(base_state, features, mandate, history_on_state)
    included: list[str] = []
    excluded: list[str] = []
    for recipe_id in ordered:
        recipe = registry.get(recipe_id)
        if recipe.worker != worker.card.id:
            raise ValueError(f"recipe {recipe_id} belongs to {recipe.worker}")
        if recipe.replaces and mandate.settings.preserve_placement:
            excluded.append(recipe_id)
        else:
            included.append(recipe_id)

    allowance = min(mandate.settings.resolved_max_runs(worker.card), remaining_budget)
    attempts: list[AppliedStep] = []
    recipe_runs = 0
    charged_wall_s = 0.0
    for recipe_id in included:
        if recipe_runs >= allowance:
            break
        applied = engine.apply(
            base_state,
            registry.get(recipe_id),
            brief=brief,
            start=start_evidence,
            parent=base_evidence,
            guards=mandate.guards,
            noise=noise,
        )
        attempts.append(applied)
        if applied.outcome.run.status != RunStatus.TRANSIENT:
            recipe_runs += 1
            charged_wall_s += applied.outcome.run.wall_s

    return WorkerRunResult(
        mandate_id=mandate.id,
        planned_recipe_ids=tuple(included),
        excluded_replacing=tuple(excluded),
        attempts=tuple(attempts),
        selected=worker.select(attempts, mandate),
        recipe_runs=recipe_runs,
        charged_wall_s=charged_wall_s,
    )

