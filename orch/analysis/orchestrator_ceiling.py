"""Offline, training-only mandate ceiling over the pre-pilot promotion snapshot.

No backend, meter, validator, or DCP parser is invoked here. Missing cache
outcomes prune a complete mandate; the searched ceiling is a beam lower bound
when exact enumeration is intractable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from orch.analysis.fit_workers import PROJECT, TRAINING_DESIGNS, TRAINING_ROOT
from orch.analysis.landscape_e1 import load_training
from orch.config import load_noise, load_recipe_sets, load_worker_card
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import (
    AppliedStep, Brief, DesignState, Evidence, Mandate, MetricName, RunOutcome,
    RunStatus, WorkerSettings,
)
from orch.contract.views import CallSummary, PolicyView, StepSummary
from orch.engine.cache import RunCache
from orch.engine.gate import judge
from orch.engine.journal import Journal
from orch.engine.start_properties import PROPERTY_KEYS, PropertyRuleError, check_not_above_start
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes, _sha256_file
from orch.episode import _Visit, _rank, _summary
from orch.policies.repair_loop import RepairLoopPolicy
from orch.policies.training_baselines import FixedAlternationPolicy, SingleSpecialistPolicy
from orch.recipes.registry import Registry, load_registry
from orch.workers.base import GenericWorker
from orch.workers.fitted import FittedWorker
from orch.workers.telemetry import PILOT_TRAINING_ROOT_V3, SavedTelemetryProvider

PILOT_ROOT = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"
PRIOR_PILOT_ROOT = PILOT_TRAINING_ROOT_V3
OLD_DEPTH3 = PROJECT / "data/sweeps/overnight_priority2_2026-11-04/depth3"
NEW_DEPTH3 = PROJECT / "data/sweeps/thursday_crossworker_depth3_2026-11-05"
SELECTED_STATES = NEW_DEPTH3 / "selected_states.json"
EQUIVALENCE_AUDIT = PROJECT / "data/scratch_thursday_night_2026-11-05/training_meter_equivalence/audit.json"
PROMOTION_AUDIT = PROJECT / "data/scratch_thursday_night_2026-11-05/training_cache_promotion/audit.json"
PROMOTION_PLAN = PROMOTION_AUDIT.parent / "plan.json"
REPORT = PROJECT / "notes/orchestrator_ceiling_training.md"
BUDGETS = (4, 6, 8, 10)
BRIEFS = ("T0", "A0", "T", "A")
PROPERTIES = PROPERTY_KEYS
_HASH64 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_ID = re.compile(r"run_[0-9a-f]{32}\Z")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"training source path escaped its root: {path}")
    return resolved


def _physical(evidence: Evidence, start: Evidence) -> None:
    """Apply the pilot's additional property and start-basis checks offline.

    The property rule is the one shared definition (no more fixed cells than the start).
    """
    if (
        evidence.fake or evidence.routed is not True
        or evidence.route_errors != 0 or evidence.drc_errors != 0
        or evidence.constraint_fingerprint != start.constraint_fingerprint
        or evidence.clock_digest != start.clock_digest
    ):
        raise ValueError(f"training Evidence fails physical or property gate: {evidence.state_id}")
    try:
        check_not_above_start(evidence, start)
    except PropertyRuleError as exc:
        raise ValueError(
            f"training Evidence fails physical or property gate: {evidence.state_id}"
        ) from exc
    for name in MetricName:
        item = evidence.metric(name)
        reference = start.metric(name)
        if item is None or reference is None or not math.isfinite(item.value):
            raise ValueError(f"training Evidence lacks finite {name.value}: {evidence.state_id}")
        if (item.unit, item.method, item.basis) != (
            reference.unit, reference.method, reference.basis
        ):
            raise ValueError(f"training Evidence method differs: {name.value}")


def _best_key(brief: Brief, visit: _Visit) -> tuple[float, float, int, str]:
    primary = visit.evidence.metric(brief.objective.metric)
    other_name = (
        MetricName.PACKED_LUTS if brief.objective.metric == MetricName.WNS_NS
        else MetricName.WNS_NS
    )
    other = visit.evidence.metric(other_name)
    if primary is None or other is None:
        raise ValueError("ranked visit lacks objective or non-objective metric")
    first = -primary.value if brief.objective.direction == "max" else primary.value
    second = other.value if other_name == MetricName.PACKED_LUTS else -other.value
    return first, second, visit.lineage_runs, visit.state.id


def _improvement(brief: Brief, start: Evidence, candidate: Evidence) -> float:
    first = start.metric(brief.objective.metric)
    last = candidate.metric(brief.objective.metric)
    if first is None or last is None:
        raise ValueError("objective metric is missing")
    return (
        last.value - first.value if brief.objective.direction == "max"
        else first.value - last.value
    )


def _share(brief: Brief, start: Evidence, candidate: Evidence, best: Evidence) -> float | None:
    denominator = _improvement(brief, start, best)
    if denominator <= 1e-12:
        return None
    return 100.0 * _improvement(brief, start, candidate) / denominator


@dataclass(frozen=True)
class CallTrace:
    """One deterministic mandate and the cached recipes it attempted."""
    worker: str
    base_id: str
    preserve_placement: bool
    attempts: int
    recipes: tuple[str, ...]
    selected_id: str | None

    def label(self) -> str:
        placement = "keep" if self.preserve_placement else "free"
        return f"{self.worker}@{self.base_id[:12]}:{placement}:{self.attempts}[{','.join(self.recipes)}]"


@dataclass(frozen=True)
class SimNode:
    """A complete state of a simulated episode, including per-base history."""
    visits: tuple[_Visit, ...]
    current: _Visit
    history: Mapping[str, tuple[AppliedStep, ...]]
    calls: tuple[CallSummary, ...]
    steps: tuple[StepSummary, ...]
    path: tuple[CallTrace, ...]
    used: int

    def latest(self, state_id: str) -> _Visit:
        for visit in reversed(self.visits):
            if visit.state.id == state_id and visit.verdict.physically_valid:
                return visit
        raise ValueError(f"mandate base was not visited: {state_id}")

    def final(self, brief: Brief) -> _Visit:
        ranked = _rank(brief, list(self.visits))
        if not ranked:
            raise ValueError("B2 start was not brief-valid")
        return ranked[0]


class CachedOracle(Protocol):
    """Only a checked, current-meter snapshot RunOutcome may be returned."""
    def get(self, base: DesignState, recipe_id: str) -> RunOutcome | None: ...


class PromotionSnapshotOracle:
    """Read only approved pre-pilot aliases, ignoring the live v4 cache index."""

    def __init__(
        self, root: Path, registry: Registry, meter_version: str,
        approved_outputs: set[tuple[str, str]],
        *, approved_audit_sha256: str | None = None,
    ) -> None:
        self.root = root.resolve(strict=True)
        if self.root != PILOT_ROOT.resolve(strict=True):
            raise ValueError("ceiling oracle refuses a non-training pilot root")
        self.plan_path = PROMOTION_PLAN.resolve(strict=True)
        self.audit_path = PROMOTION_AUDIT.resolve(strict=True)
        self.plan_sha256 = _sha256_file(self.plan_path)
        self.audit_sha256 = _sha256_file(self.audit_path)
        if approved_audit_sha256 is not None and self.audit_sha256 != approved_audit_sha256:
            raise ValueError("promotion audit changed after the preflight")
        plan = _read_object(self.plan_path)
        audit = _read_object(self.audit_path)
        rows = plan.get("rows")
        omitted = plan.get("omitted_ambiguous_keys")
        if (
            plan.get("schema_version") != "training-cache-promotion-plan-v1"
            or audit.get("plan_sha256") != self.plan_sha256
            or plan.get("audit_sha256") != _sha256_file(EQUIVALENCE_AUDIT)
            or plan.get("pilot_plan_sha256") != _sha256_file(self.root / "plan.json")
            or Path(str(plan.get("destination"))).resolve(strict=True) != self.root
            or not isinstance(rows, list)
            or len(rows) != audit.get("promoted_pointers")
            or not isinstance(omitted, list)
            or len(omitted) != 94 or len(set(omitted)) != 94
            or plan.get("ambiguous_keys_excluded") != 94
            or audit.get("ambiguous_keys_excluded") != 94
            or plan.get("tool_jobs_launched") != 0
            or audit.get("tool_jobs_launched") != 0
        ):
            raise ValueError("pre-pilot promotion plan differs from its approval audit")
        self.alias_rows: dict[str, dict[str, Any]] = {}
        self.complete_hashes: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("promotion alias row is not an object")
            key = row.get("key")
            run_id = row.get("alias_run_id")
            output_id = row.get("output_id")
            if (
                not isinstance(key, str) or not _HASH64.fullmatch(key)
                or not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id)
                or not isinstance(output_id, str) or not _HASH64.fullmatch(output_id)
                or row.get("design") not in TRAINING_DESIGNS
                or not isinstance(row.get("new_evidence_id"), str)
                or key in self.alias_rows or key in omitted
                or run_id in self.complete_hashes
                or RunCache.key(
                    row["input_id"], row["recipe_id"], row["recipe_version"]
                ) != key
            ):
                raise ValueError("promotion alias identity or key differs")
            complete = self.root / "runs" / run_id / "complete.json"
            self.complete_hashes[run_id] = _sha256_file(complete)
            self.alias_rows[key] = row
        self.complete_manifest_sha256 = hashlib.sha256(
            _json_bytes(sorted(self.complete_hashes.items()))
        ).hexdigest()
        self.registry = registry
        self.meter_version = meter_version
        self.approved_outputs = approved_outputs
        self.store = StateStore(self.root)
        self.memo: dict[tuple[str, str], RunOutcome | None] = {}
        self.miss_reasons: dict[tuple[str, str], str] = {}

    def require_stable_snapshot(self) -> None:
        """Recheck every pinned alias completion before publishing results."""
        if (
            _sha256_file(self.plan_path) != self.plan_sha256
            or _sha256_file(self.audit_path) != self.audit_sha256
        ):
            raise ValueError("pre-pilot promotion plan or audit changed")
        for run_id, digest in self.complete_hashes.items():
            complete = self.root / "runs" / run_id / "complete.json"
            if _sha256_file(complete) != digest:
                raise ValueError(f"promotion alias completion changed: {run_id}")

    def get(self, base: DesignState, recipe_id: str) -> RunOutcome | None:
        ident = base.id, recipe_id
        if ident in self.memo:
            return self.memo[ident]
        recipe = self.registry.get(recipe_id)
        key = RunCache.key(base.id, recipe.id, recipe.version)
        row = self.alias_rows.get(key)
        if row is None:
            self.memo[ident] = None
            self.miss_reasons[ident] = "outside_pre_pilot_promotion_snapshot"
            return None
        run_id = row["alias_run_id"]
        complete = self.root / "runs" / run_id / "complete.json"
        if _sha256_file(complete) != self.complete_hashes[run_id]:
            raise ValueError(f"promotion alias completion changed: {run_id}")
        outcome = self.store.load_outcome(run_id)
        run = outcome.run
        if (
            run.id != run_id or run.recipe_id != recipe.id
            or run.recipe_version != recipe.version
            or run.input_sha256_before != base.id
            or run.input_sha256_after != base.id
            or run.status != RunStatus.OK or run.fake
            or outcome.state is None or outcome.evidence is None
            or outcome.state.parent_id != base.id
            or outcome.state.id != row["output_id"]
            or outcome.state.design != row["design"]
            or outcome.evidence.id != row["new_evidence_id"]
            or outcome.evidence.state_id != outcome.state.id
            or outcome.evidence.meter_version != self.meter_version
        ):
            raise ValueError(f"pre-pilot alias outcome differs: {run_id}")
        if (base.design, outcome.state.id) not in self.approved_outputs:
            self.memo[ident] = None
            self.miss_reasons[ident] = "no_checked_amd_pass"
            return None
        self.memo[ident] = outcome
        return outcome


class Simulator:
    """Apply the actual worker planner, gate, selector and policy view offline."""

    def __init__(
        self, *, brief: Brief, start_state: DesignState, start_evidence: Evidence,
        workers: Mapping[str, GenericWorker], registry: Registry,
        oracle: CachedOracle, provider: SavedTelemetryProvider | None,
        noise: Mapping[MetricName, float] | None = None,
    ) -> None:
        self.brief = brief
        self.start_state = start_state
        self.start_evidence = start_evidence
        self.workers = workers
        self.registry = registry
        self.oracle = oracle
        self.provider = provider
        self.noise = dict(noise or {})
        self.contexts: dict[tuple[str, str | None, str | None], Mapping[str, object]] = {}
        start_verdict = judge(brief, start_evidence, start_evidence, start_evidence)
        if not start_verdict.brief_valid:
            raise ValueError("start failed the frozen brief")
        self.start_visit = _Visit(start_state, start_evidence, start_verdict, None, None, 0)

    def initial(self) -> SimNode:
        return SimNode((self.start_visit,), self.start_visit, {}, (), (), (), 0)

    def _features(self, visit: _Visit) -> Mapping[str, object] | None:
        if self.provider is None:
            return None
        key = visit.state.id, visit.producer_recipe_id, visit.run_id
        if key not in self.contexts:
            self.contexts[key] = self.provider(
                visit.state, visit.evidence, visit.producer_recipe_id, visit.run_id
            )
        return self.contexts[key]

    def apply(self, node: SimNode, mandate: Mandate) -> tuple[SimNode | None, str]:
        """Prune the whole mandate if any selected recipe lacks a cached outcome."""
        if mandate.worker_id not in self.workers:
            raise ValueError("mandate names an unknown worker")
        if mandate.settings.max_runs is None:
            raise ValueError("ceiling requires an explicit attempt count")
        if node.used + mandate.settings.max_runs > self.brief.budget_runs:
            return None, "budget"
        mandate.validate_against(self.brief, self.start_evidence)
        base = node.latest(mandate.base_state_id)
        worker = self.workers[mandate.worker_id]
        history = node.history.get(base.state.id, ())
        features = self._features(base) if isinstance(worker, FittedWorker) else None
        ordered = worker.plan(base.state, features, mandate, history)
        planned = tuple(
            recipe_id for recipe_id in ordered
            if not (
                self.registry.get(recipe_id).replaces
                and mandate.settings.preserve_placement
            )
        )[:mandate.settings.max_runs]
        if not planned:
            return None, "no_recipe"
        outcomes: list[RunOutcome] = []
        for recipe_id in planned:
            outcome = self.oracle.get(base.state, recipe_id)
            if outcome is None:
                return None, "uncached"
            outcomes.append(outcome)
        attempts: list[AppliedStep] = []
        new_visits = list(node.visits)
        new_steps = list(node.steps)
        for outcome in outcomes:
            verdict = judge(
                self.brief, self.start_evidence, base.evidence,
                outcome.evidence, guards=mandate.guards, noise=self.noise,
                run_status=outcome.run.status,
            )
            applied = AppliedStep(outcome=outcome, verdict=verdict)
            attempts.append(applied)
            new_steps.append(StepSummary(
                worker_id=mandate.worker_id, settings=mandate.settings,
                base_state_id=base.state.id,
                output_state_id=outcome.state.id if outcome.state else None,
                delta_from_parent=verdict.delta_from_parent,
                delta_from_start=verdict.delta_from_start,
                verdict_status=verdict.status, within_noise=verdict.within_noise,
            ))
            if verdict.physically_valid and outcome.state and outcome.evidence:
                new_visits.append(_Visit(
                    outcome.state, outcome.evidence, verdict, outcome.run.id,
                    outcome.run.recipe_id, base.lineage_runs + 1,
                ))
        charged = sum(item.outcome.run.status != RunStatus.TRANSIENT for item in attempts)
        if charged != len(attempts):
            raise ValueError("cached transient attempt cannot be simulated")
        selected = worker.select(attempts, mandate)
        current = node.current
        selected_id = None
        if selected is not None:
            selected_id = selected.outcome.state.id if selected.outcome.state else None
            current = next(
                visit for visit in reversed(new_visits)
                if visit.run_id == selected.outcome.run.id
            )
        new_history = dict(node.history)
        new_history[base.state.id] = (*history, *attempts)
        call = CallSummary(
            mandate_id=mandate.id, worker_id=mandate.worker_id,
            base_state_id=base.state.id, settings=mandate.settings,
            selected_state_id=selected_id,
            selected_verdict_status=selected.verdict.status if selected else None,
            recipe_runs=charged, context_ref=mandate.context_ref,
        )
        trace = CallTrace(
            mandate.worker_id, base.state.id, mandate.settings.preserve_placement,
            mandate.settings.max_runs, tuple(planned), selected_id,
        )
        return SimNode(
            tuple(new_visits), current, new_history, (*node.calls, call),
            tuple(new_steps), (*node.path, trace), node.used + charged,
        ), "ok"

    def view(self, node: SimNode, episode_id: str) -> PolicyView:
        ranked = _rank(self.brief, list(node.visits))
        return PolicyView(
            brief=self.brief, budget_left=self.brief.budget_runs - node.used,
            start=_summary(self.start_visit), current=_summary(node.current),
            best_valid=_summary(ranked[0]) if ranked else None,
            visited=tuple(_summary(v) for v in node.visits),
            steps=node.steps, calls=node.calls,
            worker_cards=tuple(w.card for w in self.workers.values()),
            noise_bands=self.noise, episode_id=episode_id, fake=False,
        )



@dataclass(frozen=True)
class SearchStats:
    """Counts include all proposed mandates before cache pruning."""
    proposed: int
    cache_pruned: int
    no_recipe: int
    explored: int
    beam_discarded: int

    @property
    def prune_share(self) -> float:
        return self.cache_pruned / self.proposed if self.proposed else 0.0


@dataclass(frozen=True)
class SearchResult:
    best: SimNode
    stats: SearchStats
    exact: bool


@dataclass(frozen=True)
class PolicyResult:
    final: _Visit
    calls: tuple[CallTrace, ...]
    used: int
    censored: bool
    reason: str
    node: SimNode


def _visit_best(brief: Brief, node: SimNode) -> _Visit:
    return min(
        (visit for visit in node.visits if visit.verdict.brief_valid),
        key=lambda visit: _best_key(brief, visit),
    )


def _beam_priority(brief: Brief, start: Evidence, node: SimNode) -> tuple:
    best = _visit_best(brief, node)
    best_any = max(
        _improvement(brief, start, visit.evidence)
        for visit in node.visits if visit.verdict.physically_valid
    )
    unique = len({visit.state.id for visit in node.visits})
    return (
        -_improvement(brief, start, best.evidence), -best_any,
        -unique, node.used,
        tuple(trace.label() for trace in node.path),
    )


def _optional_identity(value: str | None) -> tuple[int, str]:
    """Tag absent identities so revisited DCP hashes have sortable visits."""
    return (0, "") if value is None else (1, value)


def _signature(node: SimNode) -> tuple:
    """Conservatively merge only paths with the same accessible state occurrences."""
    latest = {
        visit.state.id: (
            _optional_identity(visit.producer_recipe_id),
            _optional_identity(visit.run_id),
            visit.lineage_runs,
        )
        for visit in node.visits
    }
    history = tuple(sorted(
        (state_id, tuple(item.outcome.run.recipe_id for item in attempts))
        for state_id, attempts in node.history.items()
    ))
    visited = tuple(sorted(
        (
            visit.state.id,
            _optional_identity(visit.producer_recipe_id),
            _optional_identity(visit.run_id),
            visit.lineage_runs,
        )
        for visit in node.visits
    ))
    return (
        node.current.state.id, _optional_identity(node.current.run_id),
        tuple(sorted(latest.items())), history, visited,
    )


def beam_search(sim: Simulator, *, width: int = 128) -> SearchResult:
    """Search deterministic calls with a bounded frontier at each charged cost.

    The width bound makes this a lower bound on the true reachable optimum.
    All alternatives of worker, visited base, placement and 1-3 attempts are
    proposed. Snapshot misses prune a mandate before any state is visited.
    """
    if width < 1:
        raise ValueError("beam width must be positive")
    budget = sim.brief.budget_runs
    buckets: dict[int, list[SimNode]] = defaultdict(list)
    initial = sim.initial()
    buckets[0].append(initial)
    best = initial
    proposed = pruned = no_recipe = explored = discarded = 0
    for cost in range(budget + 1):
        candidates = buckets.pop(cost, [])
        if not candidates:
            continue
        unique: dict[tuple, SimNode] = {}
        for node in candidates:
            signature = _signature(node)
            prior = unique.get(signature)
            if prior is None or _beam_priority(sim.brief, sim.start_evidence, node) < _beam_priority(
                sim.brief, sim.start_evidence, prior
            ):
                unique[signature] = node
        ranked = sorted(
            unique.values(),
            key=lambda node: _beam_priority(sim.brief, sim.start_evidence, node),
        )
        # Four representatives per policy-relevant best/current pair retain
        # distinct histories while preventing one score plateau from filling the beam.
        chosen: list[SimNode] = []
        groups: dict[tuple[str, str], int] = defaultdict(int)
        for node in ranked:
            group = (_visit_best(sim.brief, node).state.id, node.current.state.id)
            if groups[group] >= 4:
                continue
            chosen.append(node)
            groups[group] += 1
            if len(chosen) >= width:
                break
        discarded += len(candidates) - len(chosen)
        for node in chosen:
            explored += 1
            if _best_key(sim.brief, _visit_best(sim.brief, node)) < _best_key(
                sim.brief, _visit_best(sim.brief, best)
            ):
                best = node
            if cost >= budget:
                continue
            base_ids = tuple(dict.fromkeys(visit.state.id for visit in node.visits))
            for base_id in base_ids:
                for worker_id in ("timing", "area"):
                    for preserve in (False, True):
                        for attempts in (1, 2, 3):
                            if cost + attempts > budget:
                                continue
                            mandate = Mandate(
                                id=f"ceiling_{cost}_{len(node.path)}_{worker_id}_{base_id[:12]}_{int(preserve)}_{attempts}",
                                worker_id=worker_id, base_state_id=base_id,
                                settings=WorkerSettings(
                                    preserve_placement=preserve, max_runs=attempts,
                                ),
                                brief_hash=sim.brief.fingerprint(),
                            )
                            proposed += 1
                            successor, reason = sim.apply(node, mandate)
                            if successor is None:
                                if reason == "uncached":
                                    pruned += 1
                                elif reason == "no_recipe":
                                    no_recipe += 1
                                else:
                                    raise ValueError(f"unexpected mandate result: {reason}")
                                continue
                            if successor.used <= cost or successor.used > budget:
                                raise ValueError("simulated mandate did not consume valid budget")
                            buckets[successor.used].append(successor)
    return SearchResult(
        best, SearchStats(proposed, pruned, no_recipe, explored, discarded),
        exact=(discarded == 0 and pruned == 0),
    )


def simulate_policy(sim: Simulator, policy: object, *, episode_id: str) -> PolicyResult:
    """Run one frozen policy against the same cache-only mandate application."""
    node = sim.initial()
    for _ in range(sim.brief.budget_runs + 1):
        mandate = policy.decide(sim.view(node, episode_id))
        if mandate is None:
            return PolicyResult(node.final(sim.brief), node.path, node.used, False, "stop", node)
        successor, reason = sim.apply(node, mandate)
        if successor is None:
            return PolicyResult(
                node.final(sim.brief), node.path, node.used,
                reason == "uncached", reason, node,
            )
        node = successor
    raise ValueError("policy exceeded the maximum number of charged calls")


def _policy(name: str) -> object:
    if name == "repair_loop":
        return RepairLoopPolicy()
    if name == "single_specialist":
        return SingleSpecialistPolicy()
    if name == "fixed_AB":
        return FixedAlternationPolicy(first="A")
    if name == "fixed_BA":
        return FixedAlternationPolicy(first="B")
    raise ValueError(f"unknown policy {name}")


@dataclass(frozen=True)
class Known:
    """One checked training state with one observed lineage."""
    design: str
    state_id: str
    sequence: tuple[str, ...]
    evidence: Evidence
    source: str


def _checked_amd_record(
    root: Path, design: str, start_id: str, output_id: str, path: Path,
) -> None:
    """Check canonical source PASS and the retained raw validator report."""
    proof = _inside(root / "verification", path)
    plan = _read_object(root / "verification/plan.json")
    fingerprint = plan.get("fingerprint")
    canonical = root / "verification/records" / str(fingerprint) / design / f"{output_id}.json"
    latest = root / "verification/latest" / design / f"{output_id}.json"
    if proof != canonical.resolve(strict=True) or latest.read_bytes() != canonical.read_bytes():
        raise ValueError("AMD PASS pointer differs from canonical record")
    record = _read_object(canonical)
    if (
        record.get("design") != design or record.get("reference_sha256") != start_id
        or record.get("output_sha256") != output_id
        or record.get("validator_fingerprint") != fingerprint
        or record.get("result") != "pass"
    ):
        raise ValueError("AMD PASS record has wrong design, state or fingerprint")
    raw_path = _inside(root / "verification/runs", Path(record["raw_report_path"]))
    if _sha256_file(raw_path) != record.get("raw_report_sha256"):
        raise ValueError("AMD raw report checksum changed")
    raw = _read_object(raw_path)
    precheck = (raw.get("simulation_report") or {}).get("precheck_report") or {}
    full = (raw.get("simulation_report") or {}).get("full_report") or {}
    if not (
        raw.get("overall_passed") is True
        and raw.get("phase1_passed") is True
        and raw.get("phase2_passed") is True
        and raw.get("infrastructure_failure") is False
        and raw.get("num_vectors") == plan.get("vectors")
        and raw.get("precheck_vectors") == plan.get("precheck_vectors")
        and precheck.get("passed") is True
        and precheck.get("cycles_simulated") == plan.get("precheck_vectors")
        and precheck.get("mismatch_count") == 0
        and precheck.get("protocol_mismatch_count") == 0
        and full.get("passed") is True
        and full.get("cycles_simulated") == plan.get("vectors")
        and full.get("mismatch_count") == 0
        and full.get("protocol_mismatch_count") == 0
    ):
        raise ValueError("AMD PASS raw report no longer passes")



def _canonical_amd_path(root: Path, design: str, output_id: str) -> Path:
    """Locate the verifier's canonical record; latest is checked separately."""
    plan = _read_object(root / "verification/plan.json")
    fingerprint = plan.get("fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("AMD verification plan lacks a valid fingerprint")
    return root / "verification/records" / fingerprint / design / f"{output_id}.json"


def _require_v4_plan_pins(plan: Mapping[str, Any], project: Path = PROJECT) -> None:
    """Bind the offline simulator to the registry and FINAL models frozen in v4."""
    pins = plan.get("source_hashes")
    if not isinstance(pins, dict):
        raise ValueError("v4 pilot plan lacks frozen source hashes")
    required = {
        "registry": project / "recipes/registry.yaml",
        "A_fit": project / "configs/workers/A_fit.json",
        "B_fit": project / "configs/workers/B_fit.json",
    }
    for name, path in required.items():
        if pins.get(name) != _sha256_file(path):
            raise ValueError(f"v4 pilot {name} hash differs from its frozen plan")


def _selected_parent_sequences(
    selection: Mapping[str, Any], design: str, plan: Mapping[str, Any],
    fit_edges: Mapping[tuple[str, tuple[str, ...]], Any],
    known_by_hash: Mapping[tuple[str, str], list[Known]],
    fit_store: StateStore,
) -> dict[str, tuple[str, ...]]:
    """Resolve each selected input from its recorded fitting edge and source run."""
    by_design = selection.get("by_design")
    block = by_design.get(design) if isinstance(by_design, dict) else None
    records = block.get("selected_states") if isinstance(block, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"{design}: frozen parent selection is absent")
    selected: dict[str, tuple[str, ...]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{design}: selected parent record is not an object")
        state_id = record.get("hash")
        depth = record.get("source_depth")
        sequence_text = record.get("source_sequence")
        run_id = record.get("source_run_id")
        if (
            not isinstance(state_id, str) or len(state_id) != 64
            or record.get("dcp_sha256") != state_id
            or type(depth) is not int or depth not in (1, 2)
            or not isinstance(sequence_text, str) or not sequence_text
            or not isinstance(run_id, str) or not run_id
            or state_id in selected
        ):
            raise ValueError(f"{design}: selected parent identity or depth differs")
        sequence = tuple(sequence_text.split(" > "))
        if len(sequence) != depth or any(not recipe_id for recipe_id in sequence):
            raise ValueError(f"{design}: selected parent sequence differs from its depth")
        # Identical DCP bytes occur at several depths, so require the exact
        # frozen source run and sequence instead of choosing a path by hash.
        edge = fit_edges.get((design, sequence))
        if (
            edge is None or edge.run_id != run_id or edge.output_id != state_id
            or not edge.eligible or edge.amd_status != "pass"
        ):
            raise ValueError(f"{design}: selected parent lacks its exact AMD-PASS fitting edge")
        outcome = fit_store.load_outcome(run_id)
        if (
            outcome.run.id != run_id or outcome.run.recipe_id != sequence[-1]
            or outcome.state is None or outcome.state.id != state_id
            or not any(
                item.sequence == sequence
                for item in known_by_hash.get((design, state_id), ())
            )
        ):
            raise ValueError(f"{design}: selected parent run or checked lineage differs")
        selected[state_id] = sequence
    planned = plan.get("parent_hashes")
    if (
        not isinstance(planned, list) or len(planned) != len(set(planned))
        or set(planned) != set(selected)
    ):
        raise ValueError(f"{design}: selected parents differ from the campaign plan")
    return selected


def _selected_prefix(
    design: str, input_id: str, selected: Mapping[str, tuple[str, ...]],
    known_by_hash: Mapping[tuple[str, str], list[Known]],
) -> tuple[str, ...]:
    """Use the frozen selected lineage even when the DCP hash has other paths."""
    prefix = selected.get(input_id)
    if prefix is None:
        raise ValueError(f"{design}: campaign edge input lacks a selected parent")
    if not any(
        item.sequence == prefix for item in known_by_hash.get((design, input_id), ())
    ):
        raise ValueError(f"{design}: selected parent sequence lacks a checked source state")
    return prefix


def _known_from_campaign(
    root: Path, episode: str, *, design: str, starts: Mapping[str, tuple[DesignState, Evidence]],
    known_by_hash: Mapping[tuple[str, str], list[Known]],
    approved_versions: set[str], source_name: str,
    explicit_sequence: bool,
    selected_parents: Mapping[str, tuple[str, ...]] | None = None,
) -> list[Known]:
    """Load checksummed training edges and independently checked AMD reports."""
    root = root.resolve(strict=True)
    events = Journal(root).read(episode)
    if not events or events[0].get("type") != "configuration":
        raise ValueError(f"training campaign lacks a checksummed configuration: {root}")
    plan_path = root / "plan.json"
    plan = _read_object(plan_path)
    if events[0].get("plan_sha256") != _sha256_file(plan_path):
        raise ValueError(f"training campaign plan differs from journal: {root}")
    store = StateStore(root)
    found: list[Known] = []
    for event in events[1:]:
        if event.get("type") != "edge":
            continue
        row = event["edge"]
        if (
            row.get("design") != design or row.get("status") != "ok"
            or row.get("run_status") != "ok" or row.get("amd_result") != "pass"
            or type(row.get("amd_outputs_compared")) is not int
            or row["amd_outputs_compared"] <= 0
        ):
            raise ValueError(f"training edge lacks checked AMD PASS: {root}")
        run_id = row["run_id"]
        outcome = store.load_outcome(run_id)
        if (
            outcome.run.id != run_id or outcome.run.status != RunStatus.OK
            or (not explicit_sequence and outcome.run.recipe_id != row["recipe_id"])
        ):
            raise ValueError("training edge run identity differs")
        if outcome.state is None or outcome.evidence is None:
            raise ValueError("checked training edge lacks output or Evidence")
        input_id = row["input_hash"]
        output_id = row["output_hash"]
        recipe_id = outcome.run.recipe_id
        if (
            outcome.run.input_sha256_before != input_id
            or outcome.run.input_sha256_after != input_id
            or outcome.state.id != output_id
            or outcome.state.parent_id != input_id
            or outcome.evidence.id != row["evidence_id"]
            or outcome.evidence.state_id != output_id
            or outcome.evidence.meter_version not in approved_versions
            or outcome.state.design != design
        ):
            raise ValueError("training edge differs from checked run artifacts")
        if explicit_sequence:
            sequence = tuple(row["sequence"])
            if len(sequence) != 3 or sequence[-1] != recipe_id:
                raise ValueError("older depth-three sequence differs from recipe")
            parents = known_by_hash.get((design, input_id), [])
            if not any(parent.sequence == sequence[:-1] for parent in parents):
                raise ValueError("older depth-three prefix differs from checked D2")
        else:
            if selected_parents is None:
                raise ValueError("cross-worker campaign lacks frozen parent selection")
            sequence = (
                *_selected_prefix(design, input_id, selected_parents, known_by_hash),
                recipe_id,
            )
        _physical(outcome.evidence, starts[design][1])
        _checked_amd_record(
            root, design, starts[design][0].id, output_id, Path(row["amd_record"]),
        )
        found.append(Known(design, output_id, sequence, outcome.evidence, source_name))
    return found


def _pilot_starts(current_version: str) -> dict[str, tuple[DesignState, Evidence]]:
    """Read the five pinned B2 starts from the active train-only pilot root."""
    root = PILOT_ROOT.resolve(strict=True)
    plan = _read_object(root / "plan.json")
    if (
        plan.get("training_only") is not True
        or set(plan.get("training_inputs", {})) != set(TRAINING_DESIGNS)
        or plan.get("meter_version") != current_version
    ):
        raise ValueError("pilot plan is not the five-design current-meter training plan")
    _require_v4_plan_pins(plan)
    store = StateStore(root)
    version_key = hashlib.sha256(current_version.encode("utf-8")).hexdigest()
    starts: dict[str, tuple[DesignState, Evidence]] = {}
    for design, state_id in plan["training_inputs"].items():
        state = DesignState.model_validate_json(
            (root / "states" / state_id / "state.json").read_bytes()
        )
        if (
            state.id != state_id or state.design != design
            or state.parent_id is not None or state.fake
            or Path(state.path).resolve(strict=True) != store.state_path(state_id)
        ):
            raise ValueError(f"pilot B2 identity differs: {design}")
        evidence = Evidence.model_validate_json(
            (root / "measurements" / state_id / version_key / "evidence.json").read_bytes()
        )
        if evidence.state_id != state_id or evidence.meter_version != current_version:
            raise ValueError(f"pilot B2 Evidence differs: {design}")
        _physical(evidence, evidence)
        starts[design] = state, evidence
    return starts


def _pre_pilot_episode_journals(root: Path) -> list[Path]:
    """Admit only the stopped v3 pilot; live v4 outcomes are outside this snapshot."""
    resolved = root.resolve(strict=True)
    if resolved != PRIOR_PILOT_ROOT.resolve(strict=True):
        raise ValueError("pre-pilot best-known pool refuses live v4 journals")
    return sorted((resolved / "episodes").glob("*/journal.jsonl"))


def load_known(
    *, current_version: str, approved_old_versions: set[str],
) -> tuple[dict[str, tuple[DesignState, Evidence]], list[Known], dict[str, str]]:
    """Take a checked, immutable snapshot of every named training source."""
    starts = _pilot_starts(current_version)
    allowed = {current_version, *approved_old_versions}
    snapshot = load_training(TRAINING_ROOT, final=True)
    known: list[Known] = []
    by_hash: dict[tuple[str, str], list[Known]] = defaultdict(list)
    source_hashes: dict[str, str] = {
        "worker_fitting_journal_prefix": snapshot.journal_prefix_sha256,
        "v4_pilot_plan": _sha256_file(PILOT_ROOT / "plan.json"),
    }
    for design, (state, evidence) in starts.items():
        item = Known(design, state.id, (), evidence, "B2")
        known.append(item)
        by_hash[(design, state.id)].append(item)
        start_metrics = snapshot.starts[design]
        if (
            start_metrics.meter_version not in allowed
            or start_metrics.state_id != state.id
            or start_metrics.wns_ns != evidence.metric("wns_ns").value
            or start_metrics.packed_luts != evidence.metric("packed_luts").value
            or start_metrics.constraint_fingerprint != evidence.constraint_fingerprint
            or start_metrics.clock_digest != evidence.clock_digest
        ):
            raise ValueError(f"training B2 current/old Evidence differs: {design}")
    main_store = StateStore(TRAINING_ROOT)
    for (design, sequence), edge in sorted(snapshot.edges.items()):
        if (
            not edge.eligible or edge.amd_status != "pass"
            or edge.run_id is None or edge.output_id is None
        ):
            raise ValueError("D1-D3 training prefix lacks checked AMD PASS")
        outcome = main_store.load_outcome(edge.run_id)
        if outcome.state is None or outcome.evidence is None or outcome.state.id != edge.output_id:
            raise ValueError("D1-D3 output differs from checked journal")
        evidence = outcome.evidence
        if (
            evidence.meter_version not in allowed
            or evidence.metric("wns_ns").value != edge.metrics.wns_ns
            or evidence.metric("packed_luts").value != edge.metrics.packed_luts
        ):
            raise ValueError("D1-D3 source Evidence differs or has unapproved meter")
        _physical(evidence, starts[design][1])
        item = Known(design, edge.output_id, sequence, evidence, "D1-D3")
        known.append(item)
        by_hash[(design, edge.output_id)].append(item)
    old_root = OLD_DEPTH3.resolve(strict=True)
    old_plan = _read_object(old_root / "plan.json")
    if old_plan.get("design") != "vexriscv_re-place":
        raise ValueError("older depth-three source is not training VexRiscv")
    source_hashes["old_depth3_plan"] = _sha256_file(old_root / "plan.json")
    source_hashes["old_depth3_journal"] = _sha256_file(
        Journal(old_root).path("overnight_depth3_vexriscv")
    )
    extra = _known_from_campaign(
        old_root, "overnight_depth3_vexriscv", design="vexriscv_re-place",
        starts=starts, known_by_hash=by_hash, approved_versions=allowed,
        source_name="overnight_depth3", explicit_sequence=True,
    )
    for item in extra:
        known.append(item)
        by_hash[(item.design, item.state_id)].append(item)
    selection_raw = SELECTED_STATES.read_bytes()
    selection = json.loads(selection_raw)
    if (
        not isinstance(selection, dict)
        or set(selection.get("training_designs", ())) != set(TRAINING_DESIGNS)
    ):
        raise ValueError("Thursday selected states are not the five training designs")
    selection_hash = hashlib.sha256(selection_raw).hexdigest()
    source_hashes["thursday_selected_states"] = selection_hash
    for design in TRAINING_DESIGNS:
        root = (NEW_DEPTH3 / design).resolve(strict=True)
        plan = _read_object(root / "plan.json")
        if (
            plan.get("design") != design
            or plan.get("meter_version") != current_version
            or plan.get("selection_sha256") != selection_hash
        ):
            raise ValueError("Thursday follow-up plan differs from training or selection")
        selected_parents = _selected_parent_sequences(
            selection, design, plan, snapshot.edges, by_hash, main_store,
        )
        source_hashes[f"thursday_followup_{design}_plan"] = _sha256_file(root / "plan.json")
        source_hashes[f"thursday_followup_{design}_journal"] = _sha256_file(
            Journal(root).path(f"thursday_crossworker_{design}")
        )
        extra = _known_from_campaign(
            root, f"thursday_crossworker_{design}", design=design,
            starts=starts, known_by_hash=by_hash, approved_versions=allowed,
            source_name="thursday_crossworker_followup", explicit_sequence=False,
            selected_parents=selected_parents,
        )
        for item in extra:
            known.append(item)
            by_hash[(item.design, item.state_id)].append(item)
    for pilot_source in (PRIOR_PILOT_ROOT,):
        pilot_root = pilot_source.resolve(strict=True)
        store = StateStore(pilot_root)
        episodes = _pre_pilot_episode_journals(pilot_root)
        for journal_path in episodes:
            episode_id = journal_path.parent.name
            if not episode_id.startswith("pilot_"):
                raise ValueError("pilot root contains a non-pilot episode journal")
            source_hashes[f"{pilot_root.name}_{episode_id}"] = _sha256_file(journal_path)
            for event in Journal(pilot_root).read(episode_id):
                if event.get("type") != "step":
                    continue
                step = event["step"]
                if event["verdict"].get("physically_valid") is not True:
                    continue
                outcome = store.load_outcome(step["run_id"])
                if outcome.state is None or outcome.evidence is None:
                    raise ValueError("pilot physical step lacks output Evidence")
                design = outcome.state.design
                if design not in starts or outcome.evidence.meter_version != current_version:
                    raise ValueError("pilot step is outside training or has stale meter")
                parent_id = step["parent_state_id"]
                if (
                    outcome.run.input_sha256_before != parent_id
                    or outcome.state.parent_id != parent_id
                    or outcome.state.id != step["output_state_id"]
                    or outcome.evidence.id != step["evidence_id"]
                ):
                    raise ValueError("pilot step differs from its run outcome")
                _physical(outcome.evidence, starts[design][1])
                path = _canonical_amd_path(pilot_root, design, outcome.state.id)
                _checked_amd_record(
                    pilot_root, design, starts[design][0].id, outcome.state.id, path,
                )
                parents = by_hash.get((design, parent_id), [])
                if not parents:
                    raise ValueError("pilot step parent lacks a checked source path")
                parent = min(parents, key=lambda item: (len(item.sequence), item.sequence))
                item = Known(
                    design, outcome.state.id, (*parent.sequence, outcome.run.recipe_id),
                    outcome.evidence, pilot_root.name,
                )
                known.append(item)
                by_hash[(design, item.state_id)].append(item)
    prior_cached, pointer_digest, _ = _prior_pilot_cache_known(
        starts, by_hash, current_version
    )
    source_hashes["v3_terminal_cache_pointer_manifest"] = pointer_digest
    for item in prior_cached:
        known.append(item)
        by_hash[(item.design, item.state_id)].append(item)
    return starts, known, source_hashes



def _prior_pilot_cache_known(
    starts: Mapping[str, tuple[DesignState, Evidence]],
    by_hash: Mapping[tuple[str, str], list[Known]],
    current_version: str,
) -> tuple[list[Known], str, int]:
    """Include AMD-PASS v3 terminal pointers even if a stop preceded Step journaling."""
    root = PRIOR_PILOT_ROOT.resolve(strict=True)
    store = StateStore(root)
    found: list[Known] = []
    pointers: list[tuple[str, str]] = []
    omitted_unverified = 0
    for path in sorted((root / "cache").glob("*.json")):
        pointer = _read_object(path)
        if pointer.get("key") != path.stem:
            raise ValueError("prior pilot cache key differs from its filename")
        pointers.append((path.name, _sha256_file(path)))
        if pointer.get("retryable") is True:
            continue
        run_id = pointer.get("run_id")
        if not isinstance(run_id, str):
            raise ValueError("prior pilot cache lacks a run ID")
        outcome = store.load_outcome(run_id)
        run = outcome.run
        if RunCache.key(
            run.input_sha256_before, run.recipe_id, run.recipe_version
        ) != path.stem:
            raise ValueError("prior pilot cache pointer differs from its run")
        if (
            run.status != RunStatus.OK or run.fake
            or run.input_sha256_after != run.input_sha256_before
        ):
            continue
        if outcome.state is None or outcome.evidence is None:
            continue
        design = outcome.state.design
        if design not in starts or outcome.evidence.meter_version != current_version:
            raise ValueError("prior pilot cached outcome is outside current-meter training")
        _physical(outcome.evidence, starts[design][1])
        latest = root / "verification/latest" / design / f"{outcome.state.id}.json"
        if not latest.is_file():
            omitted_unverified += 1
            continue
        proof = _canonical_amd_path(root, design, outcome.state.id)
        _checked_amd_record(
            root, design, starts[design][0].id, outcome.state.id, proof,
        )
        parents = by_hash.get((design, run.input_sha256_before), [])
        if not parents:
            raise ValueError("prior pilot cache parent lacks a checked path")
        parent = min(parents, key=lambda item: (len(item.sequence), item.sequence))
        found.append(Known(
            design, outcome.state.id, (*parent.sequence, run.recipe_id),
            outcome.evidence, "v3_terminal_cache",
        ))
    digest = hashlib.sha256(_json_bytes(pointers)).hexdigest()
    return found, digest, omitted_unverified

def _known_key(brief: Brief, item: Known) -> tuple[float, float, int, str, tuple[str, ...]]:
    objective = item.evidence.metric(brief.objective.metric)
    other_name = (
        MetricName.PACKED_LUTS if brief.objective.metric == MetricName.WNS_NS
        else MetricName.WNS_NS
    )
    other = item.evidence.metric(other_name)
    if objective is None or other is None:
        raise ValueError("known training state lacks WNS or packed LUTs")
    first = -objective.value if brief.objective.direction == "max" else objective.value
    second = other.value if other_name == MetricName.PACKED_LUTS else -other.value
    return first, second, len(item.sequence), item.state_id, item.sequence


def _known_valid(brief: Brief, start: Evidence, item: Known) -> bool:
    for term in brief.terms:
        value = item.evidence.metric(term.metric)
        if value is None:
            raise ValueError("known training state lacks a brief limit metric")
        threshold = term.threshold(start)
        if term.comparison == "<=" and value.value > threshold:
            return False
        if term.comparison == ">=" and value.value < threshold:
            return False
    return True


def best_known(
    brief: Brief, design: str, start: Evidence, known: Sequence[Known],
) -> Known:
    eligible = [
        item for item in known
        if item.design == design and _known_valid(brief, start, item)
    ]
    if not eligible:
        raise ValueError("known training set lacks its brief-valid B2 state")
    return min(eligible, key=lambda item: _known_key(brief, item))


def _check_duplicate_evidence(known: Sequence[Known]) -> None:
    """The same exact DCP bytes must not have conflicting measured metrics."""
    signatures: dict[tuple[str, str], tuple] = {}
    for item in known:
        evidence = item.evidence
        metric_values = tuple(sorted(
            (name.value, value.value, value.unit, value.method, value.basis)
            for name, value in evidence.metrics.items()
        ))
        signature = (
            metric_values, evidence.routed, evidence.route_errors,
            evidence.drc_errors, evidence.constraint_fingerprint,
            evidence.clock_digest, tuple(sorted((evidence.v_props or {}).items())),
        )
        key = item.design, item.state_id
        prior = signatures.get(key)
        if prior is not None and prior != signature:
            raise ValueError(f"same training DCP hash has conflicting Evidence: {key}")
        signatures[key] = signature


def _fmt_value(brief: Brief, evidence: Evidence) -> str:
    metric = evidence.metric(brief.objective.metric)
    if metric is None:
        raise ValueError("objective metric is absent")
    unit = "ns" if brief.objective.metric == MetricName.WNS_NS else "LUTs"
    return f"{metric.value:.3f} {unit}"


def _fmt_sequence(sequence: Sequence[str]) -> str:
    return "B2" if not sequence else "B2 -> " + " -> ".join(sequence)


def _fmt_calls(calls: Sequence[CallTrace]) -> str:
    return "stop at B2" if not calls else " ; ".join(call.label() for call in calls)


def _fmt_reached(
    brief: Brief, start: Evidence, final: Evidence, known: Evidence,
    *, used: int, censored: bool = False,
) -> str:
    share = _share(brief, start, final, known)
    ratio = "n/a" if share is None else f"{share:.1f}%"
    suffix = " censored" if censored else ""
    return f"{_fmt_value(brief, final)} ({ratio}; {used} runs{suffix})"


def render_report(
    *, starts: Mapping[str, tuple[DesignState, Evidence]], known: Sequence[Known],
    source_hashes: Mapping[str, str], equivalence_hash: str, promotion_hash: str,
    model_hashes: Mapping[str, str], cache_root: Path,
    ambiguous_keys_excluded: int,
    results: Mapping[tuple[str, str, int], tuple[Known, SearchResult, Mapping[str, PolicyResult]]],
    width: int, oracle: PromotionSnapshotOracle, summary_hash: str,
) -> str:
    """Render summary, all per-design paths, baselines, coverage and provenance."""
    followup_depths = Counter(
        len(item.sequence) for item in known
        if item.source == "thursday_crossworker_followup"
    )
    if set(followup_depths) != {2, 3}:
        raise ValueError("Thursday follow-up source depths differ from checked selection")
    lines = [
        "# Orchestrator ceiling: pre-pilot promotion snapshot", "",
        "## Summary", "",
        "**Verified, training only, pre-pilot promotion snapshot.** Every mandate "
        f"outcome below came from one of {len(oracle.alias_rows)} approved alias run IDs "
        "in the immutable promotion plan. The live v4 cache pointers and all new v4 "
        "pilot journal results are excluded. Every reachable output had a checked "
        "source AMD simulation PASS and passed saved physical-property checks. "
        "No hardware-tool call was made by this analysis.",
        "",
        "**Search limit.** Beam width " + str(width) +
        " bounds each charged-cost frontier. A result is an exact reachable ceiling "
        "only where no frontier state was discarded and no promotion-snapshot mandate "
        "was pruned; otherwise it is a reachable lower bound. Missing snapshot aliases "
        "prune whole mandates. The known-result benchmark includes observed D1-D3, older "
        "VexRiscv depth-three, Thursday cross-worker follow-ups "
        f"({followup_depths[2]} depth-two and {followup_depths[3]} depth-three), "
        "and stopped v3 pilot outputs; the deeper experiments are selected, not exhaustive.",
        f"The promotion omitted {ambiguous_keys_excluded} cache aliases with distinct "
        "output DCP hashes for the same input/recipe key. Those mandates remain "
        "pruned in this snapshot even if the live pilot later measures them.",
        "",
        "| Brief | Mean B4 share | Mean B6 share | Mean B8 share | Mean B10 share | "
        "Worker-limited gap at B6 | Policy-limited gap at B6 | Repair censored at B6 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for brief_id in BRIEFS:
        shares = {}
        for budget in BUDGETS:
            values = []
            for design in TRAINING_DESIGNS:
                best, search, _ = results[(design, brief_id, budget)]
                start = starts[design][1]
                value = _share(
                    load_frozen_brief(brief_id), start,
                    search.best.final(load_frozen_brief(brief_id)).evidence, best.evidence,
                )
                values.append(100.0 if value is None else value)
            shares[budget] = statistics.mean(values)
        worker_gap = []
        policy_gap = []
        repair_censored = 0
        for design in TRAINING_DESIGNS:
            best, search, policies = results[(design, brief_id, 6)]
            brief = load_frozen_brief(brief_id)
            start = starts[design][1]
            ceiling = search.best.final(brief).evidence
            repair = policies["repair_loop"].final.evidence
            worker_gap.append(
                _improvement(brief, start, best.evidence)
                - _improvement(brief, start, ceiling)
            )
            policy_gap.append(
                _improvement(brief, start, ceiling)
                - _improvement(brief, start, repair)
            )
            repair_censored += policies["repair_loop"].censored
        unit = "ns" if brief_id.startswith("T") else "LUTs"
        lines.append(
            f"| {brief_id} | {shares[4]:.1f}% | {shares[6]:.1f}% | "
            f"{shares[8]:.1f}% | {shares[10]:.1f}% | "
            f"{statistics.mean(worker_gap):.3f} {unit} | "
            f"{statistics.mean(policy_gap):.3f} {unit} | "
            f"{repair_censored}/5 |"
        )
    lines.extend([
        "",
        "A share is the improvement from B2 divided by the pre-pilot best observed "
        "improvement for the same design and brief. A later pilot report can include "
        "new v4 outputs in its denominator, so those shares are not directly "
        "comparable. When the pre-pilot best observed state is B2, a matching "
        "B2 result counts as 100% in the summary. Positive gaps mean the later stage "
        "left a measured improvement unreached. A policy-limited gap with a censored "
        "repair result is an upper bound on that policy's eventual gap.",
        "",
        "## Best known and beam-reachable state by design and budget", "",
        "| Design | Brief | Budget | Best known WNS / LUT / path | Reachable final WNS / LUT / share | "
        "Used | Search status | Snapshot-pruned mandates | Beam-discarded states |",
        "|---|---|---:|---|---|---:|---|---:|---:|",
    ])
    for design in TRAINING_DESIGNS:
        for brief_id in BRIEFS:
            brief = load_frozen_brief(brief_id)
            start = starts[design][1]
            for budget in BUDGETS:
                best, search, _ = results[(design, brief_id, budget)]
                final = search.best.final(brief)
                share = _share(brief, start, final.evidence, best.evidence)
                ratio = "n/a" if share is None else f"{share:.1f}%"
                bwns = best.evidence.metric("wns_ns").value
                blut = best.evidence.metric("packed_luts").value
                fwns = final.evidence.metric("wns_ns").value
                flut = final.evidence.metric("packed_luts").value
                lines.append(
                    f"| {design} | {brief_id} | {budget} | "
                    f"{bwns:.3f} / {blut:.0f} / {_fmt_sequence(best.sequence)} "
                    f"({best.source}) | {fwns:.3f} / {flut:.0f} / {ratio} | "
                    f"{search.best.used} | {'complete' if search.exact else 'reachable lower bound'} | "
                    f"{search.stats.cache_pruned}/{search.stats.proposed} "
                    f"({100 * search.stats.prune_share:.1f}%) | "
                    f"{search.stats.beam_discarded} |"
                )
    lines.extend(["", "## Mandate sequences reaching each beam result", ""])
    for design in TRAINING_DESIGNS:
        for brief_id in BRIEFS:
            for budget in BUDGETS:
                _, search, _ = results[(design, brief_id, budget)]
                lines.append(
                    f"- {design} / {brief_id} / B{budget}: {_fmt_calls(search.best.path)}."
                )
    lines.extend([
        "", "## Frozen policy values from the same cache-only simulator", "",
        "A censored value is the best brief-valid state visited before the first "
        "uncached mandate; it is a lower bound, not a completed policy result.",
        "",
        "| Design | Brief | Budget | Repair loop | Single specialist | Fixed A then B | Fixed B then A |",
        "|---|---|---:|---|---|---|---|",
    ])
    for design in TRAINING_DESIGNS:
        for brief_id in BRIEFS:
            brief = load_frozen_brief(brief_id)
            start = starts[design][1]
            for budget in BUDGETS:
                best, _, policies = results[(design, brief_id, budget)]
                cells = [
                    _fmt_reached(
                        brief, start, policies[name].final.evidence, best.evidence,
                        used=policies[name].used, censored=policies[name].censored,
                    )
                    for name in (
                        "repair_loop", "single_specialist", "fixed_AB", "fixed_BA"
                    )
                ]
                lines.append(
                    f"| {design} | {brief_id} | {budget} | " + " | ".join(cells) + " |"
                )
    lines.extend([
        "", "## Source checks and limits", "",
        f"- Alias store root: {cache_root}; frozen aliases: {len(oracle.alias_rows)}; "
        f"snapshot lookups: {len(oracle.memo)}; "
        f"unavailable lookups: {len(oracle.miss_reasons)}.",
        f"- Training equivalence audit SHA-256: {equivalence_hash}.",
        f"- Training cache-promotion audit SHA-256: {promotion_hash}.",
        f"- Promotion plan SHA-256: {oracle.plan_sha256}; alias completion manifest "
        f"SHA-256: {oracle.complete_manifest_sha256}.",
        f"- Machine-readable summary SHA-256: {summary_hash}.",
        f"- FINAL worker model SHA-256: timing {model_hashes['timing']}; "
        f"area {model_hashes['area']}.",
        f"- Analysis source SHA-256: {_sha256_file(Path(__file__))}.",
    ])
    for name, digest in sorted(source_hashes.items()):
        lines.append(f"- {name}: SHA-256 {digest}.")
    lines.extend([
        "- **Inferred limit:** a beam-reachable value may be below the true "
        "orchestrator optimum; missing cache keys and selected deeper training "
        "coverage also limit the measured comparison. Live v4 pilot results "
        "are deliberately outside the pre-pilot snapshot.",
        "- **Verified limit:** source AMD PASS records certify the saved DCPs. "
        "A restarted pilot may still run its own validator on a cache hit because "
        "its ledger fingerprint differs; this offline analysis does not claim "
        "zero validator cost.",
        "",
    ])
    return "\n".join(lines)



def write_summary_json(
    *, starts: Mapping[str, tuple[DesignState, Evidence]], known: Sequence[Known],
    source_hashes: Mapping[str, str], equivalence_hash: str, promotion_hash: str,
    model_hashes: Mapping[str, str], cache_root: Path,
    ambiguous_keys_excluded: int,
    snapshot_alias_count: int,
    results: Mapping[tuple[str, str, int], tuple[Known, SearchResult, Mapping[str, PolicyResult]]],
    width: int,
) -> tuple[Path, str]:
    """Publish compact checked numbers for the morning report without parsing Markdown."""
    per_brief: dict[str, Any] = {}
    per_design: dict[str, Any] = {}
    for design in TRAINING_DESIGNS:
        per_design[design] = {}
        start = starts[design][1]
        for brief_id in BRIEFS:
            brief = load_frozen_brief(brief_id)
            best, _, _ = results[(design, brief_id, 6)]
            budget_rows: dict[str, Any] = {}
            for budget in BUDGETS:
                _, search, policies = results[(design, brief_id, budget)]
                final = search.best.final(brief)
                budget_rows[str(budget)] = {
                    "ceiling": {
                        "state_sha256": final.state.id,
                        "wns_ns": final.evidence.metric("wns_ns").value,
                        "packed_luts": final.evidence.metric("packed_luts").value,
                        "share_of_best_pct": _share(
                            brief, start, final.evidence, best.evidence,
                        ),
                        "recipe_runs_used": search.best.used,
                        "mandates": [call.label() for call in search.best.path],
                        "complete_search": search.exact,
                        "reachable_lower_bound": not search.exact,
                    },
                    "policies": {
                        name: {
                            "state_sha256": value.final.state.id,
                            "wns_ns": value.final.evidence.metric("wns_ns").value,
                            "packed_luts": value.final.evidence.metric("packed_luts").value,
                            "share_of_best_pct": _share(
                                brief, start, value.final.evidence, best.evidence,
                            ),
                            "recipe_runs_used": value.used,
                            "censored": value.censored,
                            "stop_reason": value.reason,
                        }
                        for name, value in policies.items()
                    },
                    "coverage": {
                        "proposed_mandates": search.stats.proposed,
                        "cache_pruned_mandates": search.stats.cache_pruned,
                        "cache_pruned_share": search.stats.prune_share,
                        "no_recipe_mandates": search.stats.no_recipe,
                        "explored_states": search.stats.explored,
                        "beam_discarded_states": search.stats.beam_discarded,
                    },
                }
            per_design[design][brief_id] = {
                "best_known": {
                    "state_sha256": best.state_id,
                    "sequence": list(best.sequence),
                    "source": best.source,
                    "wns_ns": best.evidence.metric("wns_ns").value,
                    "packed_luts": best.evidence.metric("packed_luts").value,
                },
                "budgets": budget_rows,
            }
    for brief_id in BRIEFS:
        brief = load_frozen_brief(brief_id)
        mean_shares: dict[str, float] = {}
        mean_values: dict[str, float] = {}
        mean_prunes: dict[str, float] = {}
        for budget in BUDGETS:
            shares = []
            values = []
            prunes = []
            for design in TRAINING_DESIGNS:
                entry = per_design[design][brief_id]["budgets"][str(budget)]
                share = entry["ceiling"]["share_of_best_pct"]
                shares.append(100.0 if share is None else share)
                objective = (
                    entry["ceiling"]["wns_ns"] if brief.objective.metric == MetricName.WNS_NS
                    else entry["ceiling"]["packed_luts"]
                )
                values.append(objective)
                prunes.append(entry["coverage"]["cache_pruned_share"])
            mean_shares[str(budget)] = statistics.mean(shares)
            mean_values[str(budget)] = statistics.mean(values)
            mean_prunes[str(budget)] = statistics.mean(prunes)
        worker_gaps = []
        policy_gaps = []
        censored = 0
        for design in TRAINING_DESIGNS:
            best, search, policies = results[(design, brief_id, 6)]
            start = starts[design][1]
            worker_gaps.append(
                _improvement(brief, start, best.evidence)
                - _improvement(brief, start, search.best.final(brief).evidence)
            )
            policy_gaps.append(
                _improvement(brief, start, search.best.final(brief).evidence)
                - _improvement(brief, start, policies["repair_loop"].final.evidence)
            )
            censored += policies["repair_loop"].censored
        per_brief[brief_id] = {
            "unit": "ns" if brief.objective.metric == MetricName.WNS_NS else "packed_luts",
            "mean_ceiling_objective_by_budget": mean_values,
            "mean_ceiling_share_of_best_pct_by_budget": mean_shares,
            "mean_cache_pruned_share_by_budget": mean_prunes,
            "mean_worker_limited_gap_budget_6": statistics.mean(worker_gaps),
            "mean_policy_limited_gap_budget_6": statistics.mean(policy_gaps),
            "repair_loop_censored_designs_budget_6": censored,
            "policy_gap_has_censored_upper_bound": bool(censored),
        }
    payload = {
        "schema_version": "orchestrator-ceiling-training-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_only": True,
        "snapshot_label": "PRE_PILOT_PROMOTION_SNAPSHOT",
        "live_v4_pilot_results_included": False,
        "share_denominator_excludes_live_v4": True,
        "cache_root": str(cache_root.resolve(strict=True)),
        "snapshot_alias_count": snapshot_alias_count,
        "beam_width": width,
        "ambiguous_cache_aliases_excluded": ambiguous_keys_excluded,
        "known_records_by_source": dict(Counter(item.source for item in known)),
        "source_sha256": dict(sorted(source_hashes.items())),
        "equivalence_audit_sha256": equivalence_hash,
        "promotion_audit_sha256": promotion_hash,
        "final_model_sha256": dict(model_hashes),
        "per_brief": per_brief,
        "per_design": per_design,
    }
    path = PROJECT / "data/scratch_thursday_night_2026-11-05/orchestrator_ceiling_summary.json"
    _atomic_bytes(path, _json_bytes(payload))
    return path, _sha256_file(path)


def load_approvals() -> tuple[str, set[str], str, str, int]:
    """Require both completed, hash-linked training-only approval audits."""
    if not EQUIVALENCE_AUDIT.is_file() or not PROMOTION_AUDIT.is_file():
        raise FileNotFoundError("training equivalence and promotion audits must both exist")
    equivalence = _read_object(EQUIVALENCE_AUDIT)
    equivalence_hash = _sha256_file(EQUIVALENCE_AUDIT)
    plan_path = EQUIVALENCE_AUDIT.parent / "plan.json"
    plan = _read_object(plan_path)
    current = plan.get("current_version")
    old = plan.get("old_versions")
    versions = equivalence.get("versions")
    if (
        plan.get("schema_version") != "training-meter-equivalence-plan-v1"
        or equivalence.get("schema_version") != "training-meter-equivalence-audit-v1"
        or equivalence.get("plan_sha256") != _sha256_file(plan_path)
        or not isinstance(current, str) or len(current) != 64
        or not isinstance(old, list) or len(old) != 2
        or any(not isinstance(value, str) or len(value) != 64 for value in old)
        or not isinstance(versions, dict) or set(versions) != set(old)
        or any(
            versions[value] != {"samples": 5, "identical": 5, "different": 0}
            for value in old
        )
        or equivalence.get("identical") != 10
        or equivalence.get("different") != 0
        or equivalence.get("normalized_differences") != []
        or set(equivalence.get("expected_raw_provenance_differences", []))
        != {"id", "meter_version", "raw_report_path"}
        or equivalence.get("tagged_runner_jobs") != 20
        or equivalence.get("unique_tags") != 20
        or equivalence.get("live_exact_tags") != []
    ):
        raise ValueError("training meter equivalence approval is incomplete")
    promotion = _read_object(PROMOTION_AUDIT)
    promotion_hash = _sha256_file(PROMOTION_AUDIT)
    promotion_plan_path = PROMOTION_PLAN
    if (
        promotion.get("schema_version") != "training-cache-promotion-audit-v1"
        or promotion.get("plan_sha256") != _sha256_file(promotion_plan_path)
        or promotion.get("approval_audit_sha256") != equivalence_hash
        or set(promotion.get("approved_versions", ())) != set(old)
        or Path(str(promotion.get("destination_root"))).resolve(strict=True)
        != PILOT_ROOT.resolve(strict=True)
        or promotion.get("pilot_plan_sha256") != _sha256_file(PILOT_ROOT / "plan.json")
        or promotion.get("cost_fields_equal_to_source") is not True
        or promotion.get("preexisting_v4_pointers_unchanged") is not True
        or promotion.get("tool_jobs_launched") != 0
        or type(promotion.get("ambiguous_keys_excluded")) is not int
        or promotion["ambiguous_keys_excluded"] < 94
        or promotion.get("promoted_pointers") != 1166
    ):
        raise ValueError("training cache-promotion approval is incomplete")
    return (
        current, set(old), equivalence_hash, promotion_hash,
        promotion["ambiguous_keys_excluded"],
    )


@dataclass(frozen=True)
class SourceSnapshot:
    """Training-only source records admitted by both completed approval gates."""

    current_version: str
    approved_old_versions: frozenset[str]
    equivalence_hash: str
    promotion_hash: str
    ambiguous_keys_excluded: int
    starts: Mapping[str, tuple[DesignState, Evidence]]
    known: tuple[Known, ...]
    source_hashes: Mapping[str, str]


def preflight() -> SourceSnapshot:
    """Check approval schema, all six metrics and duplicate-DCP consistency."""
    (
        current, approved_old, equivalence_hash, promotion_hash,
        ambiguous_keys_excluded,
    ) = load_approvals()
    starts, known, source_hashes = load_known(
        current_version=current, approved_old_versions=approved_old
    )
    _check_duplicate_evidence(known)
    required_metrics = set(MetricName)
    for item in known:
        if set(item.evidence.metrics) != required_metrics:
            raise ValueError(
                f"training Evidence lacks one of the six meter metrics: {item.state_id}"
            )
    return SourceSnapshot(
        current, frozenset(approved_old), equivalence_hash, promotion_hash,
        ambiguous_keys_excluded,
        starts, tuple(known), source_hashes,
    )


def analyze(*, width: int = 128) -> Path:
    """Run the read-only training analysis after both cache audits are approved."""
    source_started = time.monotonic()
    checked = preflight()
    print(
        f"ceiling preflight: {len(checked.known)} checked training records "
        f"in {time.monotonic() - source_started:.1f}s", flush=True,
    )
    current_version = checked.current_version
    equivalence_hash = checked.equivalence_hash
    promotion_hash = checked.promotion_hash
    ambiguous_keys_excluded = checked.ambiguous_keys_excluded
    starts, known = checked.starts, checked.known
    source_hashes = dict(checked.source_hashes)
    registry = load_registry(PROJECT)
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    workers = {
        worker_id: FittedWorker(
            load_worker_card(PROJECT / "configs/workers" / f"{worker_id}.yaml"),
            sets.worker_sets[worker_id], registry,
        )
        for worker_id in ("timing", "area")
    }
    if any(worker.model_label != "FINAL" for worker in workers.values()):
        raise ValueError("ceiling requires both published FINAL workers")
    provider = SavedTelemetryProvider(PILOT_ROOT)
    oracle = PromotionSnapshotOracle(
        PILOT_ROOT, registry, current_version,
        {(item.design, item.state_id) for item in known},
        approved_audit_sha256=promotion_hash,
    )
    if len(oracle.alias_rows) != 1166:
        raise ValueError("pre-pilot promotion snapshot lacks its 1,166 aliases")
    results: dict[
        tuple[str, str, int],
        tuple[Known, SearchResult, Mapping[str, PolicyResult]],
    ] = {}
    policies = ("repair_loop", "single_specialist", "fixed_AB", "fixed_BA")
    for design in TRAINING_DESIGNS:
        start_state, start_evidence = starts[design]
        noise = load_noise(PROJECT / "configs/noise.yaml").for_start(
            design, start_evidence
        )
        for brief_id in BRIEFS:
            fixed = load_frozen_brief(brief_id)
            best = best_known(fixed, design, start_evidence, known)
            for budget in BUDGETS:
                cell_started = time.monotonic()
                brief = fixed.model_copy(update={"budget_runs": budget})
                sim = Simulator(
                    brief=brief, start_state=start_state,
                    start_evidence=start_evidence, workers=workers,
                    registry=registry, oracle=oracle, provider=provider, noise=noise,
                )
                search = beam_search(sim, width=width)
                baseline = {
                    name: simulate_policy(
                        sim, _policy(name),
                        episode_id=f"ceiling_{design}_{brief_id}_{budget}_{name}",
                    )
                    for name in policies
                }
                best_node = min(
                    (search.best, *(value.node for value in baseline.values())),
                    key=lambda node: _best_key(brief, node.final(brief)),
                )
                search = SearchResult(best_node, search.stats, search.exact)
                if _improvement(
                    fixed, start_evidence, search.best.final(brief).evidence
                ) > _improvement(fixed, start_evidence, best.evidence) + 1e-9:
                    raise ValueError("simulator reached a state missing from exhaustive known results")
                results[(design, brief_id, budget)] = best, search, baseline
                print(
                    f"ceiling cell {design}/{brief_id}/B{budget}: "
                    f"{time.monotonic() - cell_started:.1f}s; "
                    f"explored={search.stats.explored}; "
                    f"cache_pruned={search.stats.cache_pruned}; "
                    f"beam_discarded={search.stats.beam_discarded}",
                    flush=True,
                )
    oracle.require_stable_snapshot()
    source_hashes["pre_pilot_promotion_plan"] = oracle.plan_sha256
    source_hashes["pre_pilot_alias_complete_manifest"] = oracle.complete_manifest_sha256
    source_hashes["pilot_cache_promotion_audit"] = promotion_hash
    model_hashes = {
        key: worker.model_sha256 for key, worker in workers.items()
    }
    _, summary_hash = write_summary_json(
        starts=starts, known=known, source_hashes=source_hashes,
        equivalence_hash=equivalence_hash, promotion_hash=promotion_hash,
        model_hashes=model_hashes, cache_root=PILOT_ROOT,
        ambiguous_keys_excluded=ambiguous_keys_excluded,
        snapshot_alias_count=len(oracle.alias_rows),
        results=results, width=width,
    )
    text = render_report(
        starts=starts, known=known, source_hashes=source_hashes,
        equivalence_hash=equivalence_hash, promotion_hash=promotion_hash,
        model_hashes=model_hashes, cache_root=PILOT_ROOT,
        ambiguous_keys_excluded=ambiguous_keys_excluded,
        results=results, width=width, oracle=oracle,
        summary_hash=summary_hash,
    )
    REPORT.write_text(text, encoding="utf-8")
    return REPORT


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        started = time.monotonic()
        checked = preflight()
        print(json.dumps({
            "training_designs": len(checked.starts),
            "known_records_by_source": dict(Counter(item.source for item in checked.known)),
            "equivalence_audit_sha256": checked.equivalence_hash,
            "promotion_audit_sha256": checked.promotion_hash,
            "ambiguous_keys_excluded": checked.ambiguous_keys_excluded,
            "all_six_metrics_present": True,
            "duplicate_state_evidence_consistent": True,
            "elapsed_s": round(time.monotonic() - started, 3),
        }, sort_keys=True))
        return 0
    path = analyze(width=args.beam_width)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

