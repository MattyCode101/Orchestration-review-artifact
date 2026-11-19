"""Share-of-best analysis for exhaustive sweeps and comparable final picks.

Only brief-valid meter evidence enters the quality ratio. Costs are recorded
separately as charged recipe runs, charged wall time and actual backend runs;
cache hits count toward the first two, matching the contract budget rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from orch.baselines.exhaustive import SweepEntry, SweepResult
from orch.contract.models import (
    Brief,
    Evidence,
    FrozenModel,
    MetricName,
    Verdict,
)
from orch.engine.gate import judge


class Comparison(FrozenModel):
    """One approach's final pick, already judged against the same brief/start."""

    approach: str
    brief_id: str
    evidence: Evidence | None = None
    verdict: Verdict | None = None
    run_count: int = Field(ge=0)
    wall_s: float = Field(ge=0)
    actual_backend_runs: int | None = Field(default=None, ge=0)
    fake: bool

    @model_validator(mode="after")
    def coherent(self) -> "Comparison":
        if self.evidence is not None and self.evidence.fake != self.fake:
            raise ValueError("comparison evidence origin differs")
        if self.verdict is not None and self.verdict.fake != self.fake:
            raise ValueError("comparison verdict origin differs")
        if (
            self.evidence is not None
            and self.verdict is not None
            and self.verdict.evidence_id != self.evidence.id
        ):
            raise ValueError("comparison verdict cites another evidence record")
        return self


class ShareOfBestRow(FrozenModel):
    """One auditable table row for a design, brief and approach."""

    design: str
    brief_id: str
    approach: str
    objective_metric: MetricName
    direction: Literal["max", "min"]
    state_id: str | None
    best_state_id: str | None
    start_value: float | None
    best_value: float | None
    candidate_value: float | None
    share_of_best: float | None
    share_reason: str | None
    run_count: int = Field(ge=0)
    wall_s: float = Field(ge=0)
    actual_backend_runs: int | None = Field(default=None, ge=0)
    fake: bool


def share_of_best(
    start_value: float | None,
    best_value: float | None,
    candidate_value: float | None,
    direction: Literal["max", "min"],
) -> float | None:
    """Return the fraction of the best improvement captured from the start.

    WNS can be negative, so ratios of raw objective values are misleading.
    A zero or adverse exhaustive improvement leaves this ratio undefined.
    """

    if start_value is None or best_value is None or candidate_value is None:
        return None
    values = (start_value, best_value, candidate_value)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("share-of-best inputs must be finite")
    if direction == "max":
        denominator = best_value - start_value
        numerator = candidate_value - start_value
    elif direction == "min":
        denominator = start_value - best_value
        numerator = start_value - candidate_value
    else:
        raise ValueError("direction must be max or min")
    return numerator / denominator if denominator > 0 else None


def _comparable_value(
    start: Evidence, candidate: Evidence, metric: MetricName
) -> float | None:
    first = start.metric(metric)
    second = candidate.metric(metric)
    if first is None or second is None:
        return None
    if start.meter_version != candidate.meter_version:
        return None
    if (first.unit, first.method, first.basis) != (
        second.unit,
        second.method,
        second.basis,
    ):
        return None
    return second.value


def _rank(
    start: Evidence, evidence: Evidence, brief: Brief
) -> tuple[float, float, str] | None:
    """Lower tuple wins; the other brief's metric breaks objective ties."""

    primary = _comparable_value(start, evidence, brief.objective.metric)
    if primary is None:
        return None
    primary_rank = -primary if brief.objective.direction == "max" else primary
    if brief.objective.metric == MetricName.WNS_NS:
        secondary = _comparable_value(start, evidence, MetricName.PACKED_LUTS)
        secondary_rank = secondary if secondary is not None else math.inf
    else:
        secondary = _comparable_value(start, evidence, MetricName.WNS_NS)
        secondary_rank = -secondary if secondary is not None else math.inf
    return primary_rank, secondary_rank, evidence.state_id


def best_valid(
    sweep: SweepResult, brief: Brief
) -> tuple[Evidence | None, SweepEntry | None]:
    """Find the best brief-valid visited state, including the start."""

    start = sweep.start_evidence
    options: list[tuple[Evidence, SweepEntry | None]] = []
    start_verdict = judge(brief, start, start, start)
    if start_verdict.brief_valid:
        options.append((start, None))
    for entry in sweep.entries:
        verdict = entry.verdicts.get(brief.id)
        if verdict is not None and verdict.brief_valid and entry.evidence is not None:
            options.append((entry.evidence, entry))
    ranked = [
        (_rank(start, evidence, brief), evidence, entry)
        for evidence, entry in options
    ]
    ranked = [row for row in ranked if row[0] is not None]
    if not ranked:
        return None, None
    _, evidence, entry = min(ranked, key=lambda row: row[0])
    return evidence, entry


def _share_reason(
    start_value: float | None,
    best_value: float | None,
    candidate_value: float | None,
    *,
    candidate_status: str,
    direction: Literal["max", "min"],
) -> str | None:
    if candidate_status != "valid":
        return candidate_status
    if start_value is None or best_value is None or candidate_value is None:
        return "missing_objective_metric"
    best_gain = (
        best_value - start_value
        if direction == "max"
        else start_value - best_value
    )
    if best_gain <= 0:
        return "zero_best_improvement"
    return None


def build_share_of_best_table(
    sweep: SweepResult,
    comparisons: Sequence[Comparison] = (),
) -> tuple[ShareOfBestRow, ...]:
    """Report the sweep best and supplied approach final picks for both briefs."""

    brief_by_id = {brief.id: brief for brief in sweep.briefs}
    for comparison in comparisons:
        if comparison.brief_id not in brief_by_id:
            raise ValueError(f"unknown comparison brief: {comparison.brief_id}")
        if comparison.fake != sweep.fake:
            raise ValueError("cannot mix fake and real comparison rows")

    rows: list[ShareOfBestRow] = []
    for brief in sweep.briefs:
        objective = brief.objective.metric
        direction = brief.objective.direction
        start_value = _comparable_value(
            sweep.start_evidence, sweep.start_evidence, objective
        )
        best_evidence, _ = best_valid(sweep, brief)
        best_value = (
            _comparable_value(sweep.start_evidence, best_evidence, objective)
            if best_evidence is not None
            else None
        )
        status = "valid" if best_evidence is not None else "no_valid_state"
        rows.append(
            ShareOfBestRow(
                design=sweep.design,
                brief_id=brief.id,
                approach="exhaustive",
                objective_metric=objective,
                direction=direction,
                state_id=best_evidence.state_id if best_evidence else None,
                best_state_id=best_evidence.state_id if best_evidence else None,
                start_value=start_value,
                best_value=best_value,
                candidate_value=best_value,
                share_of_best=share_of_best(
                    start_value, best_value, best_value, direction
                ),
                share_reason=_share_reason(
                    start_value, best_value, best_value,
                    candidate_status=status, direction=direction
                ),
                run_count=sweep.total_recipe_runs,
                wall_s=sweep.total_charged_wall_s,
                actual_backend_runs=sweep.actual_backend_runs,
                fake=sweep.fake,
            )
        )
        for comparison in comparisons:
            if comparison.brief_id != brief.id:
                continue
            if comparison.evidence is None or comparison.verdict is None:
                candidate_status = "no_result"
                candidate_value = None
                state_id = None
            else:
                # Recheck the fixed brief from meter evidence. A supplied
                # verdict can carry a run failure, but cannot promote evidence
                # that the shared gate would reject.
                fresh = judge(
                    brief,
                    sweep.start_evidence,
                    sweep.start_evidence,
                    comparison.evidence,
                )
                if comparison.verdict.brief_valid and not fresh.brief_valid:
                    raise ValueError("comparison verdict contradicts measured evidence")
                if not comparison.verdict.brief_valid or not fresh.brief_valid:
                    candidate_status = "brief_invalid"
                    candidate_value = None
                else:
                    candidate_value = _comparable_value(
                        sweep.start_evidence, comparison.evidence, objective
                    )
                    candidate_status = (
                        "valid"
                        if candidate_value is not None
                        else "metric_incomparable"
                    )
                state_id = comparison.evidence.state_id
            rows.append(
                ShareOfBestRow(
                    design=sweep.design,
                    brief_id=brief.id,
                    approach=comparison.approach,
                    objective_metric=objective,
                    direction=direction,
                    state_id=state_id,
                    best_state_id=best_evidence.state_id if best_evidence else None,
                    start_value=start_value,
                    best_value=best_value,
                    candidate_value=candidate_value,
                    share_of_best=share_of_best(
                        start_value, best_value, candidate_value, direction
                    ),
                    share_reason=_share_reason(
                        start_value, best_value, candidate_value,
                        candidate_status=candidate_status, direction=direction
                    ),
                    run_count=comparison.run_count,
                    wall_s=comparison.wall_s,
                    actual_backend_runs=comparison.actual_backend_runs,
                    fake=comparison.fake,
                )
            )
    return tuple(rows)


def render_share_of_best_markdown(rows: Sequence[ShareOfBestRow]) -> str:
    """Render a compact table without hiding undefined ratios or fake data."""

    header = (
        "| Design | Brief | Approach | Share of best | Runs | Charged wall (s) | Fake |\n"
        "|---|---|---|---:|---:|---:|:---:|"
    )
    lines = [header]
    for row in rows:
        share = (
            f"{row.share_of_best:.3f}"
            if row.share_of_best is not None
            else f"undefined ({row.share_reason})"
        )
        lines.append(
            f"| {row.design} | {row.brief_id} | {row.approach} | {share} "
            f"| {row.run_count} | {row.wall_s:.1f} | {str(row.fake).lower()} |"
        )
    return "\n".join(lines) + "\n"

