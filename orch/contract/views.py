"""The only data a policy may inspect when choosing its next mandate."""

from __future__ import annotations

from pydantic import Field

from orch.contract.models import (
    Brief,
    FrozenModel,
    MetricName,
    VerdictStatus,
    WorkerCard,
    WorkerSettings,
)


class StateSummary(FrozenModel):
    state_id: str
    metrics: dict[MetricName, float]
    physically_valid: bool
    brief_valid: bool


class StepSummary(FrozenModel):
    worker_id: str
    settings: WorkerSettings
    base_state_id: str
    output_state_id: str | None
    delta_from_parent: dict[MetricName, float] = Field(default_factory=dict)
    delta_from_start: dict[MetricName, float] = Field(default_factory=dict)
    verdict_status: VerdictStatus
    within_noise: dict[MetricName, bool] = Field(default_factory=dict)


class CallSummary(FrozenModel):
    """One completed worker call, with only policy-safe decision facts."""

    mandate_id: str
    worker_id: str
    base_state_id: str
    settings: WorkerSettings
    selected_state_id: str | None
    selected_verdict_status: VerdictStatus | None = None
    recipe_runs: int = Field(ge=0)
    context_ref: str = ""


class PolicyView(FrozenModel):
    """Serializable evidence summary; no runner, script, log or adapter object."""

    brief: Brief
    budget_left: int = Field(ge=0)
    start: StateSummary
    current: StateSummary
    best_valid: StateSummary | None
    visited: tuple[StateSummary, ...] = ()
    steps: tuple[StepSummary, ...] = ()
    calls: tuple[CallSummary, ...] = ()
    worker_cards: tuple[WorkerCard, ...] = ()
    noise_bands: dict[MetricName, float] = Field(default_factory=dict)
    episode_id: str
    fake: bool = False

