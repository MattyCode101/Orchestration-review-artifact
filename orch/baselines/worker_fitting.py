"""Resumable train-only data generation through the shared step engine.

D1 measures each selected recipe once on each training design. D2 executes
every ordered pair across workers. D3 extends the best two distinct states
per worker and depth. D4 exhausts each worker separately on VexRiscv.
No hardware command or Tcl is constructed here.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from pydantic import Field, model_validator

from orch.baselines.exhaustive import _persist_judgments
from orch.contract.models import (
    Brief, DesignState, Evidence, FrozenModel, MetricName, RunOutcome, RunStatus,
)
from orch.engine.apply import StepEngine
from orch.engine.gate import GATE_VERSION, judge
from orch.engine.journal import Journal
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes
from orch.interfaces import RecipeSpec

TRAIN_DESIGNS = (
    "vexriscv_re-place", "rosetta_spam-filter", "rosetta_3d-rendering",
    "logicnets_jscl", "finn_radioml",
)
TIMING_IDS = (
    "LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock",
    "LATCH.hardblock_relocate", "LATCH.memory_path_opt",
)
AREA_IDS = (
    "A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED",
    "A_E_DUPL_Q1000",
)
DEPTH_STUDY_DESIGN = "vexriscv_re-place"
CAMPAIGN_ID = "worker_fitting_train"
SCHEMA = "worker-fitting-train-v1"
RETRYABLE = {RunStatus.TIMEOUT, RunStatus.TRANSIENT, RunStatus.METER_ERROR}
MAX_EDGE_ATTEMPTS = 2


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _edge_id(design: str, sequence: tuple[str, ...], attempt: int) -> str:
    digest = hashlib.sha256(_json_bytes([design, sequence])).hexdigest()
    return f"edge-{digest}-{attempt}"


class EdgeRecord(FrozenModel):
    """One persisted attempt or explicit invalid-parent skip."""

    design: str
    sequence: tuple[str, ...]
    input_state_id: str | None
    run_id: str | None = None
    output_state_id: str | None = None
    run_status: RunStatus | None = None
    cache_hit: bool = False
    wall_s: float = Field(default=0, ge=0)
    physically_valid: bool = False
    brief_valid: dict[str, bool] = Field(default_factory=dict)
    verdict_status: dict[str, str] = Field(default_factory=dict)
    skipped_reason: str | None = None
    first_phase: str
    attempt: int = Field(ge=1)
    fake: bool

    @model_validator(mode="after")
    def coherent(self) -> "EdgeRecord":
        if not 1 <= len(self.sequence) <= 4:
            raise ValueError("edge depth must be one through four")
        if self.skipped_reason is None:
            if self.run_id is None or self.run_status is None or self.input_state_id is None:
                raise ValueError("attempted edge lacks run identity")
        elif (
            self.run_id is not None or self.run_status is not None
            or self.output_state_id is not None or self.cache_hit or self.wall_s
            or self.physically_valid or self.brief_valid or self.verdict_status
        ):
            raise ValueError("skipped edge cannot claim a run or verdict")
        return self


@dataclass(frozen=True)
class _Task:
    design: str
    sequence: tuple[str, ...]
    input_state: DesignState
    parent_evidence: Evidence
    start_evidence: Evidence
    attempt: int


class WorkerFittingCampaign:
    """Execute phase layers with exact-sequence journals and at most 48 tasks."""

    def __init__(
        self, *,
        engine: StepEngine,
        starts: Mapping[str, DesignState],
        recipe_sets: Mapping[str, Sequence[RecipeSpec]],
        briefs: Mapping[str, Brief],
        noise: Mapping[str, Mapping[MetricName, float]] | None = None,
        max_workers: int = 48,
        campaign_id: str = CAMPAIGN_ID,
        depth_study_design: str = DEPTH_STUDY_DESIGN,
        phase_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        if not 1 <= max_workers <= 48:
            raise ValueError("worker-fitting concurrency must be between 1 and 48")
        if set(recipe_sets) != {"timing", "area"} or set(briefs) != {"timing", "area"}:
            raise ValueError("timing and area sets and briefs are required")
        self.recipe_sets = {name: tuple(recipe_sets[name]) for name in ("timing", "area")}
        if any(not values for values in self.recipe_sets.values()):
            raise ValueError("both worker recipe sets must be nonempty")
        self.all_recipes = self.recipe_sets["timing"] + self.recipe_sets["area"]
        self.recipes = {recipe.id: recipe for recipe in self.all_recipes}
        if len(self.recipes) != len(self.all_recipes):
            raise ValueError("recipe IDs must be unique")
        if any(recipe.worker != worker for worker, values in self.recipe_sets.items()
               for recipe in values):
            raise ValueError("recipe worker differs from selected set")
        self.starts = dict(starts)
        if not self.starts or depth_study_design not in self.starts:
            raise ValueError("depth-study design is missing")
        if any(state.design != design for design, state in self.starts.items()):
            raise ValueError("start design differs from campaign key")
        if len({state.fake for state in self.starts.values()}) != 1:
            raise ValueError("campaign mixes fake and real starts")
        self.fake = next(iter(self.starts.values())).fake
        self.briefs = dict(briefs)
        self.noise = {
            design: {MetricName(name): float(value)
                     for name, value in (noise or {}).get(design, {}).items()}
            for design in self.starts
        }
        if any(not math.isfinite(value) or value < 0
               for bands in self.noise.values() for value in bands.values()):
            raise ValueError("noise bands must be finite and nonnegative")
        self.engine = engine
        self.max_workers = max_workers
        self.campaign_id = campaign_id
        self.depth_study_design = depth_study_design
        self.phase_callback = phase_callback
        self.journal = Journal(engine.store.root)
        self.status_path = engine.store.root / "status.json"
        self.history: dict[tuple[str, tuple[str, ...]], list[EdgeRecord]] = {}
        self.index: dict[tuple[str, tuple[str, ...]], EdgeRecord] = {}
        self.selections: dict[str, dict[str, object]] = {}
        self.phase_events: dict[str, dict[str, object]] = {}
        self.start_evidence: dict[str, Evidence] = {}
        self.outcomes: dict[str, RunOutcome] = {}
        self.current_phase = "preparing"
        self._last_status_write = 0.0

    def _configuration(self) -> dict[str, object]:
        """Pin all scientific inputs and driver code for safe resume."""

        return {
            "schema_version": SCHEMA,
            "designs": {
                design: {
                    "start_state_id": state.id,
                    "start_evidence_id": self.start_evidence[design].id,
                    "noise": {
                        name.value: value for name, value in
                        sorted(self.noise[design].items(), key=lambda item: item[0].value)
                    },
                }
                for design, state in sorted(self.starts.items())
            },
            "recipe_sets": {
                worker: [{"id": recipe.id, "version": recipe.version} for recipe in values]
                for worker, values in self.recipe_sets.items()
            },
            "briefs": {
                worker: {"id": brief.id, "fingerprint": brief.fingerprint()}
                for worker, brief in self.briefs.items()
            },
            "meter_version": self.engine.meter_version,
            "gate_version": GATE_VERSION,
            "depth_study_design": self.depth_study_design,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fake": self.fake,
        }

    def _load_outcome(self, run_id: str) -> RunOutcome:
        if run_id not in self.outcomes:
            self.outcomes[run_id] = self.engine.store.load_outcome(run_id)
        return self.outcomes[run_id]

    def _load_journal(self) -> None:
        events = self.journal.read(self.campaign_id)
        configuration = self._configuration()
        if not events:
            self.journal.append(self.campaign_id, {
                "id": "configuration", "type": "configuration",
                "configuration": configuration, "at_utc": _utc(),
            })
            return
        if events[0].get("type") != "configuration" or events[0].get("configuration") != configuration:
            raise ValueError("journal configuration differs; refuse mixed evidence")
        for event in events[1:]:
            kind = event.get("type")
            if kind == "edge":
                edge = EdgeRecord.model_validate(event["edge"])
                if edge.design not in self.starts or any(r not in self.recipes for r in edge.sequence):
                    raise ValueError("journal edge names an unselected design or recipe")
                key = edge.design, edge.sequence
                prior = self.history.setdefault(key, [])
                if edge.attempt != len(prior) + 1:
                    raise ValueError("edge attempt number is not consecutive")
                if edge.run_id is not None:
                    outcome = self._load_outcome(edge.run_id)
                    recipe = self.recipes[edge.sequence[-1]]
                    if (
                        outcome.run.recipe_id != recipe.id
                        or outcome.run.recipe_version != recipe.version
                        or outcome.run.status != edge.run_status
                        or outcome.run.cache_hit != edge.cache_hit
                        or outcome.run.fake != edge.fake
                        or outcome.run.input_sha256_before != edge.input_state_id
                        or (
                            outcome.run.status != RunStatus.INPUT_MUTATED
                            and outcome.run.input_sha256_after != edge.input_state_id
                        )
                        or (outcome.state.id if outcome.state else None) != edge.output_state_id
                    ):
                        raise ValueError("edge differs from its completed run")
                prior.append(edge)
                self.index[key] = edge
            elif kind == "selection":
                selection_id = str(event["id"])
                if selection_id in self.selections:
                    raise ValueError("duplicate D3 selection")
                self.selections[selection_id] = event
            elif kind == "phase_complete":
                phase = str(event["phase"])
                if phase in self.phase_events:
                    raise ValueError("duplicate phase completion")
                self.phase_events[phase] = event
            else:
                raise ValueError(f"unknown journal event: {kind}")

    def _write_status(self, *, force: bool = False, error: str | None = None) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_write < 10:
            return
        _atomic_bytes(self.status_path, _json_bytes({
            "schema_version": SCHEMA,
            "campaign_id": self.campaign_id,
            "phase": self.current_phase,
            "phase_completed": list(self.phase_events),
            "committed_unique_edges": len(self.index),
            "committed_attempt_events": sum(len(rows) for rows in self.history.values()),
            "max_parallel_tasks": self.max_workers,
            "driver_pid": os.getpid(),
            "updated_at_utc": _utc(),
            "error": error,
            "fake": self.fake,
        }))
        self._last_status_write = now

    def _parent(self, design: str, sequence: tuple[str, ...]) -> tuple[DesignState, Evidence] | None:
        if len(sequence) == 1:
            return self.starts[design], self.start_evidence[design]
        prefix = self.index.get((design, sequence[:-1]))
        if prefix is None:
            raise ValueError("sequence lacks its parent edge")
        if not prefix.physically_valid or prefix.run_id is None:
            return None
        outcome = self._load_outcome(prefix.run_id)
        if outcome.state is None or outcome.evidence is None:
            raise ValueError("physically valid parent has no state or evidence")
        return outcome.state, outcome.evidence

    def _append_edge(self, edge: EdgeRecord) -> None:
        key = edge.design, edge.sequence
        if edge.attempt != len(self.history.get(key, ())) + 1:
            raise ValueError("edge attempt differs from journal history")
        self.journal.append(self.campaign_id, {
            "id": _edge_id(edge.design, edge.sequence, edge.attempt),
            "type": "edge", "edge": edge.model_dump(mode="json"),
            "at_utc": _utc(),
        })
        self.history.setdefault(key, []).append(edge)
        self.index[key] = edge
        self._write_status()


    def _record_outcome(self, task: _Task, outcome: RunOutcome, phase: str) -> EdgeRecord:
        recipe = self.recipes[task.sequence[-1]]
        if (
            outcome.run.recipe_id != recipe.id
            or outcome.run.recipe_version != recipe.version
            or outcome.run.input_sha256_before != task.input_state.id
            or (
                outcome.run.status != RunStatus.INPUT_MUTATED
                and outcome.run.input_sha256_after != task.input_state.id
            )
            or outcome.run.fake != self.fake
            or (outcome.state is not None and outcome.state.parent_id != task.input_state.id)
        ):
            raise ValueError("backend outcome differs from scheduled edge")
        judgments = {
            brief.id: judge(
                brief, task.start_evidence, task.parent_evidence, outcome.evidence,
                noise=self.noise[task.design], run_status=outcome.run.status,
            )
            for brief in self.briefs.values()
        }
        _persist_judgments(
            self.engine, outcome, tuple(self.briefs.values()), judgments,
            start=task.start_evidence, parent=task.parent_evidence,
        )
        self.outcomes[outcome.run.id] = outcome
        return EdgeRecord(
            design=task.design, sequence=task.sequence,
            input_state_id=task.input_state.id, run_id=outcome.run.id,
            output_state_id=outcome.state.id if outcome.state else None,
            run_status=outcome.run.status, cache_hit=outcome.run.cache_hit,
            wall_s=outcome.run.wall_s,
            physically_valid=all(v.physically_valid for v in judgments.values()),
            brief_valid={key: value.brief_valid for key, value in judgments.items()},
            verdict_status={key: value.status.value for key, value in judgments.items()},
            first_phase=phase, attempt=task.attempt, fake=self.fake,
        )

    def _run_layer(self, tasks: Sequence[tuple[str, tuple[str, ...]]], phase: str) -> None:
        """Execute independent edges concurrently, committing each completion."""

        if len(set(tasks)) != len(tasks):
            raise ValueError("layer contains a duplicate sequence")
        for _wave in range(MAX_EDGE_ATTEMPTS):
            ready: list[_Task] = []
            for design, sequence in tasks:
                key = design, sequence
                prior = self.history.get(key, ())
                latest = self.index.get(key)
                if latest is not None and (
                    latest.skipped_reason is not None
                    or latest.run_status not in RETRYABLE
                    or len(prior) >= MAX_EDGE_ATTEMPTS
                ):
                    continue
                parent = self._parent(design, sequence)
                attempt = len(prior) + 1
                if parent is None:
                    if latest is not None:
                        raise ValueError("retryable edge lost its parent state")
                    self._append_edge(EdgeRecord(
                        design=design, sequence=sequence, input_state_id=None,
                        skipped_reason="parent output is not physically valid",
                        first_phase=phase, attempt=attempt, fake=self.fake,
                    ))
                    continue
                state, parent_evidence = parent
                ready.append(_Task(
                    design=design, sequence=sequence, input_state=state,
                    parent_evidence=parent_evidence,
                    start_evidence=self.start_evidence[design], attempt=attempt,
                ))
            if not ready:
                break
            errors: list[str] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {
                    pool.submit(
                        self.engine.execute, task.input_state,
                        self.recipes[task.sequence[-1]],
                    ): task
                    for task in ready
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        outcome = future.result()
                        record = self._record_outcome(task, outcome, phase)
                        self._append_edge(record)
                        if (
                            outcome.run.status == RunStatus.TOOL_ERROR
                            and outcome.run.failure_reason
                            and "tagged processes survived cleanup" in outcome.run.failure_reason
                        ):
                            errors.append("exact-tag cleanup left a live process")
                    except Exception as exc:
                        errors.append(
                            f"{task.design} {task.sequence}: {type(exc).__name__}: {exc}"
                        )
            if errors:
                raise RuntimeError("worker-fitting layer failed: " + "; ".join(errors[:5]))
        missing = [key for key in tasks if key not in self.index]
        if missing:
            raise RuntimeError(f"layer has {len(missing)} uncommitted edges")

    def _summary(self, phase: str, keys: Sequence[tuple[str, tuple[str, ...]]]) -> dict[str, object]:
        unique_keys = tuple(dict.fromkeys(keys))
        records = [self.index[key] for key in unique_keys]
        attempts = [attempt for key in unique_keys for attempt in self.history[key]]
        statuses = Counter(
            "skipped" if row.skipped_reason else row.run_status.value
            for row in records
        )
        return {
            "phase": phase,
            "planned_sequences": len(unique_keys),
            "attempted_edges": sum(row.run_id is not None for row in records),
            "physical_valid_edges": sum(row.physically_valid for row in records),
            "cache_hits": sum(row.cache_hit for row in records),
            "new_backend_edges": sum(
                row.first_phase == phase and row.run_id is not None and not row.cache_hit
                for row in records
            ),
            "first_committed_in_phase": sum(row.first_phase == phase for row in records),
            "reused_prior_phase": sum(row.first_phase != phase for row in records),
            "attempt_events": len(attempts),
            "backend_attempts": sum(
                row.run_id is not None and not row.cache_hit for row in attempts
            ),
            "new_backend_attempts_in_phase": sum(
                row.first_phase == phase and row.run_id is not None and not row.cache_hit
                for row in attempts
            ),
            "charged_wall_s_all_attempts": round(
                sum(row.wall_s for row in attempts), 3
            ),
            "statuses": dict(sorted(statuses.items())),
        }

    def _complete_phase(
        self, phase: str, keys: Sequence[tuple[str, tuple[str, ...]]]
    ) -> dict[str, object]:
        if any(key not in self.index for key in keys):
            raise RuntimeError(f"{phase} cannot complete with uncommitted edges")
        summary = self._summary(phase, keys)
        existing = self.phase_events.get(phase)
        if existing is not None:
            if existing.get("summary") != summary:
                raise ValueError(f"{phase} summary changed after completion")
            return summary
        event = {
            "id": f"phase-{phase}", "type": "phase_complete", "phase": phase,
            "summary": summary, "at_utc": _utc(),
        }
        self.journal.append(self.campaign_id, event)
        self.phase_events[phase] = event
        self._write_status(force=True)
        if self.phase_callback is not None:
            self.phase_callback(event)
        return summary

    def _select_two(
        self, design: str, worker: str, depth: int
    ) -> tuple[tuple[str, ...], ...]:
        """Prefer brief-valid states, own objective, then path; dedupe DCPs."""

        recipes = {recipe.id for recipe in self.recipe_sets[worker]}
        brief = self.briefs[worker]
        selection_id = f"select-{design}-{worker}-depth{depth}"
        prior = self.selections.get(selection_id)
        if prior is not None:
            # D4 later adds more paths at the same depth. The D3 choice is a
            # dated training decision, so resume validates and reuses it.
            frozen: list[tuple[str, ...]] = []
            for row in prior["selected"]:
                sequence = tuple(row["sequence"])
                edge = self.index.get((design, sequence))
                if (
                    len(sequence) != depth
                    or any(recipe_id not in recipes for recipe_id in sequence)
                    or edge is None
                    or edge.output_state_id != row["output_state_id"]
                    or edge.brief_valid.get(brief.id, False) != row["brief_valid"]
                    or edge.run_id is None
                ):
                    raise ValueError("frozen D3 selection differs from its edge")
                outcome = self._load_outcome(edge.run_id)
                metric = outcome.evidence.metric(brief.objective.metric) if outcome.evidence else None
                if metric is None or metric.value != row["objective_value"]:
                    raise ValueError("frozen D3 selection objective differs")
                frozen.append(sequence)
            return tuple(frozen)
        options: list[tuple[tuple[object, ...], EdgeRecord, float]] = []
        for (candidate_design, sequence), edge in self.index.items():
            if candidate_design != design or len(sequence) != depth:
                continue
            if any(recipe_id not in recipes for recipe_id in sequence):
                continue
            if not edge.physically_valid or edge.run_id is None or edge.output_state_id is None:
                continue
            outcome = self._load_outcome(edge.run_id)
            if outcome.evidence is None:
                continue
            metric = outcome.evidence.metric(brief.objective.metric)
            if metric is None:
                continue
            objective = metric.value
            rank_value = -objective if brief.objective.direction == "max" else objective
            rank = (
                not edge.brief_valid.get(brief.id, False),
                rank_value, sequence,
            )
            options.append((rank, edge, objective))
        selected: list[tuple[str, ...]] = []
        selected_states: set[str] = set()
        selected_rows: list[dict[str, object]] = []
        for _, edge, objective in sorted(options, key=lambda item: item[0]):
            assert edge.output_state_id is not None
            if edge.output_state_id in selected_states:
                continue
            selected_states.add(edge.output_state_id)
            selected.append(edge.sequence)
            selected_rows.append({
                "sequence": list(edge.sequence),
                "output_state_id": edge.output_state_id,
                "objective_value": objective,
                "brief_valid": edge.brief_valid.get(brief.id, False),
            })
            if len(selected) == 2:
                break
        selection = {
            "id": selection_id, "type": "selection", "phase": "D3",
            "design": design, "worker": worker, "depth": depth,
            "brief_id": brief.id,
            "objective_metric": brief.objective.metric.value,
            "selected": selected_rows,
        }
        self.journal.append(self.campaign_id, selection)
        self.selections[selection_id] = selection
        return tuple(selected)

    def _d1(self) -> dict[str, object]:
        phase = "D1"
        self.current_phase = phase
        self._write_status(force=True)
        tasks = [
            (design, (recipe.id,))
            for design in self.starts for recipe in self.all_recipes
        ]
        self._run_layer(tasks, phase)
        summary = self._complete_phase(phase, tasks)
        if not self.fake:
            for recipe in self.all_recipes:
                rows = [self.index[(design, (recipe.id,))] for design in self.starts]
                if not any(row.physically_valid for row in rows):
                    raise RuntimeError(
                        f"D1 major issue: {recipe.id} has no physically valid training output"
                    )
        return summary

    def _d2(self) -> dict[str, object]:
        phase = "D2"
        self.current_phase = phase
        self._write_status(force=True)
        ids = tuple(recipe.id for recipe in self.all_recipes)
        tasks = [
            (design, sequence)
            for design in self.starts
            for sequence in itertools.product(ids, repeat=2)
        ]
        self._run_layer(tasks, phase)
        summary = self._complete_phase(phase, tasks)
        if not self.fake and summary["attempted_edges"]:
            if summary["physical_valid_edges"] < summary["attempted_edges"] / 2:
                raise RuntimeError(
                    "D2 major issue: fewer than half the attempted outputs are physically valid"
                )
        return summary

    def _d3(self) -> dict[str, object]:
        phase = "D3"
        self.current_phase = phase
        self._write_status(force=True)
        keys: list[tuple[str, tuple[str, ...]]] = []
        for depth in (2, 3):
            tasks: list[tuple[str, tuple[str, ...]]] = []
            for design in self.starts:
                for worker, recipes in self.recipe_sets.items():
                    parents = self._select_two(design, worker, depth)
                    tasks.extend(
                        (design, parent + (recipe.id,))
                        for parent in parents for recipe in recipes
                    )
            self._run_layer(tasks, phase)
            keys.extend(tasks)
        return self._complete_phase(phase, keys)

    def _d4(self) -> dict[str, object]:
        phase = "D4"
        self.current_phase = phase
        self._write_status(force=True)
        keys: list[tuple[str, tuple[str, ...]]] = []
        for depth in range(1, 5):
            tasks = [
                (self.depth_study_design, sequence)
                for recipes in self.recipe_sets.values()
                for sequence in itertools.product(
                    (recipe.id for recipe in recipes), repeat=depth
                )
            ]
            self._run_layer(tasks, phase)
            keys.extend(tasks)
        return self._complete_phase(phase, keys)

    def run(self) -> dict[str, dict[str, object]]:
        """Measure starts, resume exact edges and complete D1 through D4."""

        lock = self.engine.store.root / "campaigns" / self.campaign_id / "driver.lock"
        with _file_lock(lock):
            self.start_evidence = {
                design: self.engine.measure_start(state)
                for design, state in self.starts.items()
            }
            self._load_journal()
            summaries: dict[str, dict[str, object]] = {}
            try:
                for phase, operation in (
                    ("D1", self._d1), ("D2", self._d2),
                    ("D3", self._d3), ("D4", self._d4),
                ):
                    summaries[phase] = operation()
            except Exception as exc:
                self.current_phase = "failed"
                self._write_status(force=True, error=f"{type(exc).__name__}: {exc}")
                raise
            self.current_phase = "complete"
            self._write_status(force=True)
            return summaries

