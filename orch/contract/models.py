"""Provisional v0 contract records shared by every decider and backend.

The step engine depends on state identity, recipe identity/version, run status,
measured metrics and measured validity. The remaining fields make decisions
auditable without forcing a later contract design now.
"""

from __future__ import annotations

import hashlib
import json
import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "0.1.0-provisional"


class FrozenModel(BaseModel):
    """Reject undeclared fields and prevent mutation after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Record(FrozenModel):
    """A persisted contract record with an explicit provisional schema."""

    schema_version: str = SCHEMA_VERSION


class MetricName(StrEnum):
    """Metrics used by the two briefs and the shared meter."""

    WNS_NS = "wns_ns"
    TNS_NS = "tns_ns"
    PACKED_LUTS = "packed_luts"
    NETLIST_LUTS = "netlist_luts"
    FFS = "ffs"
    DYNAMIC_POWER_W = "dynamic_power_w"


class MetricValue(FrozenModel):
    """A measured number with the method and physical basis needed to compare it."""

    value: float = Field(allow_inf_nan=False)
    unit: str
    method: str
    basis: str = "routed"


class Objective(FrozenModel):
    metric: MetricName
    direction: Literal["max", "min"]


class BriefTerm(FrozenModel):
    """Threshold computed from the start value: factor * start + offset."""

    id: str
    metric: MetricName
    comparison: Literal["<=", ">="]
    factor: float = Field(default=1.0, allow_inf_nan=False)
    offset: float = Field(default=0.0, allow_inf_nan=False)

    def threshold(self, start: "Evidence") -> float:
        """Compute the absolute limit from this episode's start evidence."""

        value = start.metric(self.metric)
        if value is None:
            raise ValueError(f"start evidence lacks {self.metric}")
        limit = self.factor * value.value + self.offset
        if not math.isfinite(limit):
            raise ValueError("brief term threshold is not finite")
        return limit


class Guard(FrozenModel):
    """An additional absolute threshold requested by the decider."""

    id: str
    metric: MetricName
    comparison: Literal["<=", ">="]
    threshold: float = Field(allow_inf_nan=False)


class Brief(Record):
    """The authors' immutable objective, hard terms, checks and recipe-run budget."""

    id: str
    objective: Objective
    terms: tuple[BriefTerm, ...] = ()
    budget_runs: int = Field(default=6, gt=0)
    require_routed: bool = True
    max_drc_errors: int = 0
    require_constraint_match: bool = True
    require_clock_match: bool = True
    equivalence_policy: Literal["final_only", "none"] = "final_only"
    equivalence_vectors: int = Field(default=200, ge=0)
    equivalence_strength: Literal["weak"] = "weak"

    @model_validator(mode="after")
    def unique_terms(self) -> "Brief":
        if len({term.id for term in self.terms}) != len(self.terms):
            raise ValueError("brief term ids must be unique")
        return self

    def fingerprint(self) -> str:
        """Hash the full brief so a mandate cannot silently change its terms."""

        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class WorkerCard(Record):
    """Static capability declaration; recipe selection remains in config."""

    id: str
    version: str
    objective: Objective
    recipe_set_id: str
    default_attempts: int = Field(default=2, gt=0)
    accepted_settings: tuple[str, ...] = ("preserve_placement", "max_runs")
    guard_modes: dict[MetricName, Literal["selects_within", "gate_only"]] = Field(
        default_factory=dict
    )
    may_replace: bool = False
    accepts_stage: str = "routed"
    returns_stage: str = "routed"


class DesignState(Record):
    """One exact checkpoint, named by the SHA-256 of its bytes."""

    id: str = Field(pattern=r"^[0-9a-f]{64}$")
    design: str
    stage: str = "routed"
    parent_id: str | None = None
    made_by_run_id: str | None = None
    path: str
    fake: bool = False


class WorkerSettings(FrozenModel):
    """Per-call controls; a missing run count uses the worker card default."""

    preserve_placement: bool = False
    max_runs: int | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def legacy_allow_replace(cls, value: object) -> object:
        """Read old journals using the inverse allow_replace spelling."""
        if isinstance(value, dict) and "allow_replace" in value:
            if "preserve_placement" in value:
                raise ValueError("both placement settings were supplied")
            if type(value["allow_replace"]) is not bool:
                raise ValueError("legacy allow_replace must be a boolean")
            upgraded = dict(value)
            upgraded["preserve_placement"] = not upgraded.pop("allow_replace")
            return upgraded
        return value

    def resolved_max_runs(self, card: WorkerCard) -> int:
        """Resolve an omitted attempt count against the named worker card."""
        return self.max_runs if self.max_runs is not None else card.default_attempts


class Mandate(Record):
    """One worker call; it names no recipe and cannot change the brief."""

    id: str
    worker_id: str
    base_state_id: str
    settings: WorkerSettings = Field(default_factory=WorkerSettings)
    guards: tuple[Guard, ...] = ()
    brief_hash: str
    context_ref: str = ""

    @model_validator(mode="after")
    def unique_guards(self) -> "Mandate":
        if len({guard.id for guard in self.guards}) != len(self.guards):
            raise ValueError("guard ids must be unique")
        return self

    def validate_against(self, brief: Brief, start: "Evidence") -> None:
        """Reject a stale brief or a same-direction guard looser than a hard term."""

        if self.brief_hash != brief.fingerprint():
            raise ValueError("mandate brief hash does not match the fixed brief")
        for guard in self.guards:
            for term in brief.terms:
                if guard.metric != term.metric or guard.comparison != term.comparison:
                    continue
                limit = term.threshold(start)
                if guard.comparison == ">=" and guard.threshold < limit - 1e-12:
                    raise ValueError(f"guard {guard.id} loosens brief term {term.id}")
                if guard.comparison == "<=" and guard.threshold > limit + 1e-12:
                    raise ValueError(f"guard {guard.id} loosens brief term {term.id}")


class RunStatus(StrEnum):
    OK = "ok"
    INVALID = "invalid"
    NO_OUTPUT = "no_output"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    METER_ERROR = "meter_error"
    INPUT_MUTATED = "input_mutated"


class RunAttempt(FrozenModel):
    index: int = Field(ge=1)
    status: RunStatus
    exit_code: int | None = None
    wall_s: float = Field(ge=0)
    cpu_user_s: float | None = Field(default=None, ge=0)
    cpu_sys_s: float | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    load_1m_start: float | None = Field(default=None, ge=0)
    load_1m_end: float | None = Field(default=None, ge=0)
    busy_slots_start: int | None = Field(default=None, ge=0)
    backpressure_wait_s: float = Field(default=0, ge=0)
    backpressure_events: int = Field(default=0, ge=0)
    stdout_path: str | None = None
    stderr_path: str | None = None
    failure_reason: str | None = None


class RunRecord(Record):
    """Runner facts and provenance; claims are stored separately."""

    id: str
    recipe_id: str
    recipe_version: str
    status: RunStatus
    failure_reason: str | None = None
    exit_code: int | None = None
    wall_s: float = Field(default=0, ge=0)
    cpu_user_s: float | None = Field(default=None, ge=0)
    cpu_sys_s: float | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    load_1m_start: float | None = Field(default=None, ge=0)
    load_1m_end: float | None = Field(default=None, ge=0)
    busy_slots_start: int | None = Field(default=None, ge=0)
    backpressure_wait_s: float = Field(default=0, ge=0)
    backpressure_events: int = Field(default=0, ge=0)
    process_group_kills: int = Field(default=0, ge=0)
    tag_sweep_kills: int = Field(default=0, ge=0)
    input_sha256_before: str | None = None
    input_sha256_after: str | None = None
    command: tuple[str, ...] = ()
    environment_hash: str | None = None
    script_sha256: str | None = None
    tool_versions: dict[str, str] = Field(default_factory=dict)
    java_version: str | None = None
    attempts: tuple[RunAttempt, ...] = ()
    cache_hit: bool = False
    original_run_id: str | None = None
    fake: bool = False


class Claims(Record):
    """Worker-reported data, never accepted as meter Evidence."""

    id: str
    run_id: str
    raw_log_path: str | None = None
    values: dict[str, str | int | float | bool] = Field(default_factory=dict)
    fake: bool = False


class EquivalenceCheck(FrozenModel):
    result: Literal["pass", "fail", "unknown"]
    method: str
    vectors: int = Field(ge=0)
    reference_state_id: str
    strength: Literal["weak"] = "weak"


class Evidence(Record):
    """Independent meter output for a registered state."""

    id: str
    state_id: str
    metrics: dict[MetricName, MetricValue]
    routed: bool | None
    route_errors: int | None = Field(default=None, ge=0)
    drc_errors: int | None = Field(default=None, ge=0)
    constraint_fingerprint: str | None = None
    clock_digest: str | None = None
    v_props: dict[str, int] | None = None
    meter_version: str
    equivalence: EquivalenceCheck | None = None
    raw_report_path: str | None = None
    fake: bool = False

    def metric(self, name: MetricName | str) -> MetricValue | None:
        """Return one metric, or None when the meter could not measure it."""

        return self.metrics.get(MetricName(name))


class CheckResult(FrozenModel):
    status: Literal["pass", "fail", "unknown"]
    observed: float | str | int | bool | None = None
    threshold: float | str | int | bool | None = None
    evidence_id: str | None = None
    reason: str | None = None


class VerdictStatus(StrEnum):
    ACCEPTABLE = "ACCEPTABLE"
    GUARD_VIOLATION = "GUARD_VIOLATION"
    BRIEF_VIOLATION = "BRIEF_VIOLATION"
    INVALID = "INVALID"
    RUN_FAILED = "RUN_FAILED"


class Verdict(Record):
    """Pure gate decision; brief validity is independent of guard results."""

    id: str
    evidence_id: str | None
    status: VerdictStatus
    physically_valid: bool
    brief_valid: bool
    guard_pass: bool
    validity: dict[str, CheckResult] = Field(default_factory=dict)
    terms: dict[str, CheckResult] = Field(default_factory=dict)
    guards: dict[str, CheckResult] = Field(default_factory=dict)
    delta_from_start: dict[MetricName, float] = Field(default_factory=dict)
    delta_from_parent: dict[MetricName, float] = Field(default_factory=dict)
    within_noise: dict[MetricName, bool] = Field(default_factory=dict)
    fake: bool = False


class Step(Record):
    """One attempted recipe in an episode, including the decision provenance."""

    id: str
    episode_id: str
    index: int = Field(ge=0)
    mandate_id: str
    run_id: str
    parent_state_id: str
    output_state_id: str | None
    evidence_id: str | None
    verdict_id: str
    decider_id: str
    decider_version: str
    tokens: int = Field(default=0, ge=0)
    meter_version: str
    gate_version: str
    fake: bool = False


class RunOutcome(FrozenModel):
    """Cached tool result and measurement, independent of any brief or guard."""

    run: RunRecord
    state: DesignState | None = None
    evidence: Evidence | None = None
    claims: Claims | None = None

    @model_validator(mode="after")
    def consistent(self) -> "RunOutcome":
        if self.evidence is not None:
            if self.state is None or self.evidence.state_id != self.state.id:
                raise ValueError("evidence must name the output state")
        if self.run.status == RunStatus.OK and (
            self.state is None or self.evidence is None
        ):
            raise ValueError("ok outcome requires state and evidence")
        records = (self.state, self.evidence, self.claims)
        if any(
            record is not None and record.fake != self.run.fake
            for record in records
        ):
            raise ValueError("outcome fake origin label mismatch")
        return self


class AppliedStep(FrozenModel):
    """Fresh judgment of a possibly cached run outcome."""

    outcome: RunOutcome
    verdict: Verdict
