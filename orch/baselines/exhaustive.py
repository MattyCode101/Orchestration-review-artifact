"""Resumable one- and two-step exhaustive search on the shared step engine.

The driver chooses sequences but never invokes a hardware tool. It executes
one cacheable run per sequence edge, judges that outcome for every brief, and
records each committed edge in a checksummed journal. A failed first edge is
still represented in the result, while only physically valid outputs can be
second-step inputs.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import Field, model_validator

from orch.contract.models import (
    Brief,
    DesignState,
    Evidence,
    FrozenModel,
    MetricName,
    RunOutcome,
    RunStatus,
    Verdict,
)
from orch.engine.apply import StepEngine
from orch.engine.gate import GATE_VERSION, judge
from orch.engine.journal import Journal
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _safe_name
from orch.interfaces import RecipeSpec


SWEEP_SCHEMA_VERSION = "sweep-v0.1.0"
_RETRYABLE = {RunStatus.TIMEOUT, RunStatus.TRANSIENT, RunStatus.METER_ERROR}


class SweepEntry(FrozenModel):
    """One ordered recipe sequence, or a skipped second step.

    wall_s and run_count are cumulative along this sequence. edge_wall_s is
    the charge for this edge alone, so campaign costs do not double-count
    first steps shared by several two-step sequences.
    """

    sequence: tuple[str, ...]
    input_state_id: str | None
    run_id: str | None = None
    run_ids: tuple[str, ...] = ()
    output_state_id: str | None = None
    evidence: Evidence | None = None
    verdicts: dict[str, Verdict] = Field(default_factory=dict)
    run_status: RunStatus | None = None
    edge_wall_s: float = Field(default=0.0, ge=0)
    wall_s: float = Field(default=0.0, ge=0)
    run_count: int = Field(default=0, ge=0)
    cache_hit: bool = False
    skipped_reason: str | None = None
    fake: bool

    @model_validator(mode="after")
    def coherent(self) -> "SweepEntry":
        if len(self.sequence) not in (1, 2):
            raise ValueError("exhaustive sequence depth must be one or two")
        if self.skipped_reason is not None:
            if self.run_id is not None or self.run_status is not None:
                raise ValueError("skipped sequence cannot have a current run")
            if self.evidence is not None or self.verdicts or self.edge_wall_s:
                raise ValueError("skipped sequence cannot have new evidence")
        elif (
            self.run_id is None
            or self.run_status is None
            or not self.run_ids
            or self.run_ids[-1] != self.run_id
        ):
            raise ValueError("completed sequence must identify its current run")
        if self.evidence is not None:
            if self.output_state_id != self.evidence.state_id:
                raise ValueError("evidence must cite the sequence output")
            if self.evidence.fake != self.fake:
                raise ValueError("sweep evidence origin differs from entry")
        if any(verdict.fake != self.fake for verdict in self.verdicts.values()):
            raise ValueError("sweep verdict origin differs from entry")
        if self.run_count > len(self.run_ids):
            raise ValueError("charged runs exceed attempted recipe edges")
        return self


class SweepResult(FrozenModel):
    """A complete in-memory view of a durable sweep journal."""

    sweep_id: str
    design: str
    start_state_id: str
    start_evidence: Evidence
    briefs: tuple[Brief, ...]
    recipes: tuple[RecipeSpec, ...]
    entries: tuple[SweepEntry, ...]
    max_depth: int = Field(ge=1, le=2)
    journal_path: str
    start_measure_wall_s: float = Field(ge=0)
    total_recipe_runs: int = Field(ge=0)
    total_charged_wall_s: float = Field(ge=0)
    actual_backend_runs: int = Field(ge=0)
    fake: bool

    @model_validator(mode="after")
    def origins_match(self) -> "SweepResult":
        if self.start_evidence.state_id != self.start_state_id:
            raise ValueError("sweep start evidence cites another checkpoint")
        if self.start_evidence.fake != self.fake:
            raise ValueError("sweep start evidence origin differs")
        if any(entry.fake != self.fake for entry in self.entries):
            raise ValueError("sweep contains mixed fake and real entries")
        return self


def _normalized_noise(
    noise: Mapping[MetricName, float] | None,
) -> dict[MetricName, float]:
    result = {MetricName(name): float(value) for name, value in (noise or {}).items()}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError("noise bands must be finite and nonnegative")
    return result


def _configuration(
    engine: StepEngine,
    start_state: DesignState,
    recipes: tuple[RecipeSpec, ...],
    briefs: tuple[Brief, ...],
    noise: Mapping[MetricName, float],
) -> dict[str, object]:
    """Bind a journal to exact inputs without blocking depth-one extension."""

    return {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "start_state_id": start_state.id,
        "design": start_state.design,
        "fake": start_state.fake,
        "meter_version": engine.meter_version,
        "gate_version": GATE_VERSION,
        "recipes": [
            {
                "id": recipe.id,
                "version": recipe.version,
                "worker": recipe.worker,
                "replaces": recipe.replaces,
            }
            for recipe in recipes
        ],
        "briefs": [
            {"id": brief.id, "fingerprint": brief.fingerprint()} for brief in briefs
        ],
        "noise": {
            name.value: value for name, value in sorted(noise.items(), key=lambda item: item[0].value)
        },
    }


def _load_or_measure_start(
    *,
    journal: Journal,
    journal_id: str,
    engine: StepEngine,
    start_state: DesignState,
    configuration: dict[str, object],
) -> tuple[Evidence, float, list[dict[str, object]]]:
    """A resumed sweep uses its committed start evidence unchanged."""

    events = journal.read(journal_id)
    if events:
        header = events[0]
        if header.get("type") != "sweep_config" or header.get("config") != configuration:
            raise ValueError("sweep journal configuration differs; choose a new sweep id")
        evidence = Evidence.model_validate(header["start_evidence"])
        if (
            evidence.state_id != start_state.id
            or evidence.meter_version != engine.meter_version
            or evidence.fake != start_state.fake
        ):
            raise ValueError("sweep journal start evidence differs from this engine")
        return evidence, float(header["start_measure_wall_s"]), events

    started = time.monotonic()
    evidence = engine.measure_start(start_state)
    elapsed = time.monotonic() - started
    header = {
        "id": "config",
        "type": "sweep_config",
        "config": configuration,
        "start_evidence": evidence.model_dump(mode="json"),
        "start_measure_wall_s": elapsed,
    }
    journal.append(journal_id, header)
    return evidence, elapsed, [header]


def _history(
    events: Sequence[Mapping[str, object]],
    *,
    fake: bool,
    recipe_ids: set[str],
) -> dict[tuple[str, ...], list[SweepEntry]]:
    """Validate journal syntax before deciding which edges can be reused."""

    result: dict[tuple[str, ...], list[SweepEntry]] = {}
    for event in events[1:]:
        if event.get("type") != "sweep_edge":
            raise ValueError("unknown sweep journal event")
        entry = SweepEntry.model_validate(event["entry"])
        if entry.fake != fake:
            raise ValueError("sweep journal mixes real and fake records")
        if any(recipe_id not in recipe_ids for recipe_id in entry.sequence):
            raise ValueError("sweep journal names a recipe outside its configuration")
        if len(entry.run_ids) != (
            len(entry.sequence) - (1 if entry.skipped_reason else 0)
        ):
            raise ValueError("sweep journal has an inconsistent run lineage")
        result.setdefault(entry.sequence, []).append(entry)
    return result


def _latest_for_parent(
    history: Mapping[tuple[str, ...], list[SweepEntry]],
    sequence: tuple[str, ...],
    prefix_run_ids: tuple[str, ...],
) -> SweepEntry | None:
    for entry in reversed(history.get(sequence, ())):
        if entry.run_ids[: len(prefix_run_ids)] == prefix_run_ids:
            return entry
    return None


def _append_entry(
    journal: Journal,
    journal_id: str,
    history: dict[tuple[str, ...], list[SweepEntry]],
    entry: SweepEntry,
) -> None:
    # A retryable infrastructure result can be followed by a later attempt.
    prior = history.get(entry.sequence, [])
    sequence_hash = hashlib.sha256(_json_bytes(entry.sequence)).hexdigest()
    journal.append(
        journal_id,
        {
            "id": f"edge-{sequence_hash}-{len(prior) + 1}",
            "type": "sweep_edge",
            "entry": entry.model_dump(mode="json"),
        },
    )
    history.setdefault(entry.sequence, []).append(entry)


def _judgments(
    briefs: tuple[Brief, ...],
    *,
    start: Evidence,
    parent: Evidence,
    outcome: RunOutcome,
    noise: Mapping[MetricName, float],
) -> dict[str, Verdict]:
    return {
        brief.id: judge(
            brief,
            start,
            parent,
            outcome.evidence,
            noise=noise,
            run_status=outcome.run.status,
        )
        for brief in briefs
    }


def _persist_judgments(
    engine: StepEngine,
    outcome: RunOutcome,
    briefs: tuple[Brief, ...],
    judgments: Mapping[str, Verdict],
    *,
    start: Evidence,
    parent: Evidence,
) -> None:
    """Keep every cross-brief verdict beside the one measured run."""

    for brief in briefs:
        verdict = judgments[brief.id]
        file_key = hashlib.sha256(verdict.id.encode("utf-8")).hexdigest()
        path = (
            engine.store.root
            / "runs"
            / outcome.run.id
            / "verdicts"
            / f"{file_key}.json"
        )
        payload = _json_bytes(
            {
                "brief_hash": brief.fingerprint(),
                "start_evidence_id": start.id,
                "parent_evidence_id": parent.id,
                "gate_version": GATE_VERSION,
                "guards": [],
                "verdict": verdict.model_dump(mode="json"),
            }
        )
        if path.is_file():
            if path.read_bytes() != payload:
                raise ValueError("persisted sweep verdict differs from current judgment")
        else:
            _atomic_bytes(path, payload)


def _verify_reused(
    entry: SweepEntry,
    *,
    engine: StepEngine,
    recipe: RecipeSpec,
    input_state: DesignState,
    prefix: SweepEntry | None,
    start: Evidence,
    parent: Evidence,
    briefs: tuple[Brief, ...],
    noise: Mapping[MetricName, float],
) -> None:
    """A journal marker is usable only with its committed, matching run."""

    if entry.skipped_reason is not None or entry.run_id is None:
        raise ValueError("cannot load a skipped entry as a completed run")
    expected_prefix = prefix.run_ids if prefix is not None else ()
    if (
        entry.input_state_id != input_state.id
        or entry.run_ids[:-1] != expected_prefix
        or entry.fake != input_state.fake
    ):
        raise ValueError("resumed sweep edge has a different input lineage")
    outcome = engine.store.load_outcome(entry.run_id)
    if (
        outcome.run.recipe_id != recipe.id
        or outcome.run.recipe_version != recipe.version
        or outcome.run.status != entry.run_status
        or outcome.run.fake != entry.fake
        or outcome.run.input_sha256_before not in (None, input_state.id)
        or (
            outcome.run.status != RunStatus.INPUT_MUTATED
            and outcome.run.input_sha256_after not in (None, input_state.id)
        )
        or outcome.run.cache_hit != entry.cache_hit
        or (outcome.state.id if outcome.state else None) != entry.output_state_id
        or outcome.evidence != entry.evidence
    ):
        raise ValueError("resumed sweep edge differs from its committed outcome")
    fresh = _judgments(
        briefs, start=start, parent=parent, outcome=outcome, noise=noise
    )
    if fresh != entry.verdicts:
        raise ValueError("resumed sweep verdict differs from the current gate")
    prefix_wall = prefix.wall_s if prefix is not None else 0.0
    prefix_runs = prefix.run_count if prefix is not None else 0
    charge = 0 if outcome.run.status == RunStatus.TRANSIENT else 1
    if (
        not math.isclose(entry.edge_wall_s, outcome.run.wall_s, abs_tol=1e-9)
        or not math.isclose(
            entry.wall_s, prefix_wall + outcome.run.wall_s, abs_tol=1e-9
        )
        or entry.run_count != prefix_runs + charge
    ):
        raise ValueError("resumed sweep cost differs from the committed run")


def _new_edge(
    *,
    engine: StepEngine,
    recipe: RecipeSpec,
    sequence: tuple[str, ...],
    input_state: DesignState,
    prefix: SweepEntry | None,
    start: Evidence,
    parent: Evidence,
    briefs: tuple[Brief, ...],
    noise: Mapping[MetricName, float],
) -> SweepEntry:
    outcome = engine.execute(input_state, recipe)
    if outcome.run.fake != input_state.fake:
        raise ValueError("backend changed fake/real origin")
    judgments = _judgments(
        briefs, start=start, parent=parent, outcome=outcome, noise=noise
    )
    _persist_judgments(
        engine, outcome, briefs, judgments, start=start, parent=parent
    )
    prefix_ids = prefix.run_ids if prefix is not None else ()
    prefix_runs = prefix.run_count if prefix is not None else 0
    prefix_wall = prefix.wall_s if prefix is not None else 0.0
    charge = 0 if outcome.run.status == RunStatus.TRANSIENT else 1
    return SweepEntry(
        sequence=sequence,
        input_state_id=input_state.id,
        run_id=outcome.run.id,
        run_ids=prefix_ids + (outcome.run.id,),
        output_state_id=outcome.state.id if outcome.state else None,
        evidence=outcome.evidence,
        verdicts=judgments,
        run_status=outcome.run.status,
        edge_wall_s=outcome.run.wall_s,
        wall_s=prefix_wall + outcome.run.wall_s,
        run_count=prefix_runs + charge,
        cache_hit=outcome.run.cache_hit,
        fake=input_state.fake,
    )


def _can_continue(entry: SweepEntry, briefs: tuple[Brief, ...]) -> bool:
    """Brief-invalid states remain eligible when their physical checks pass."""

    return (
        entry.output_state_id is not None
        and entry.evidence is not None
        and all(entry.verdicts[brief.id].physically_valid for brief in briefs)
    )


def run_exhaustive(
    *,
    engine: StepEngine,
    start_state: DesignState,
    recipes: Sequence[RecipeSpec],
    briefs: Sequence[Brief],
    sweep_id: str,
    max_depth: int = 2,
    noise: Mapping[MetricName, float] | None = None,
) -> SweepResult:
    """Run every ordered sequence once; resume committed edges by journal ID.

    A depth-one call may later be extended to depth two with the same sweep ID.
    Configuration changes are rejected rather than silently mixing evidence.
    A process lock prevents two drivers from racing on the same sweep journal.
    """

    if max_depth not in (1, 2):
        raise ValueError("max_depth must be one or two")
    recipe_list = tuple(recipes)
    brief_list = tuple(briefs)
    if not recipe_list or len({recipe.id for recipe in recipe_list}) != len(recipe_list):
        raise ValueError("recipes must be a nonempty set of unique IDs")
    if not brief_list or len({brief.id for brief in brief_list}) != len(brief_list):
        raise ValueError("briefs must be a nonempty set of unique IDs")
    noise_map = _normalized_noise(noise)
    journal_id = _safe_name(f"sweep-{sweep_id}", what="sweep id")
    engine.store.state_path(start_state.id)
    journal = Journal(engine.store.root)
    lock_path = engine.store.root / "campaigns" / journal_id / "driver.lock"
    configuration = _configuration(
        engine, start_state, recipe_list, brief_list, noise_map
    )
    recipe_by_id = {recipe.id: recipe for recipe in recipe_list}

    with _file_lock(lock_path):
        start_evidence, start_measure_wall_s, events = _load_or_measure_start(
            journal=journal,
            journal_id=journal_id,
            engine=engine,
            start_state=start_state,
            configuration=configuration,
        )
        history = _history(
            events,
            fake=start_state.fake,
            recipe_ids=set(recipe_by_id),
        )
        entries: list[SweepEntry] = []
        first_by_id: dict[str, SweepEntry] = {}
        for recipe in recipe_list:
            sequence = (recipe.id,)
            reusable = _latest_for_parent(history, sequence, ())
            if reusable is not None and reusable.run_status not in _RETRYABLE:
                _verify_reused(
                    reusable,
                    engine=engine,
                    recipe=recipe,
                    input_state=start_state,
                    prefix=None,
                    start=start_evidence,
                    parent=start_evidence,
                    briefs=brief_list,
                    noise=noise_map,
                )
                entry = reusable
            else:
                entry = _new_edge(
                    engine=engine,
                    recipe=recipe,
                    sequence=sequence,
                    input_state=start_state,
                    prefix=None,
                    start=start_evidence,
                    parent=start_evidence,
                    briefs=brief_list,
                    noise=noise_map,
                )
                _append_entry(journal, journal_id, history, entry)
            entries.append(entry)
            first_by_id[recipe.id] = entry

        if max_depth == 2:
            for first_recipe, second_recipe in itertools.product(recipe_list, repeat=2):
                sequence = (first_recipe.id, second_recipe.id)
                prefix = first_by_id[first_recipe.id]
                reusable = _latest_for_parent(history, sequence, prefix.run_ids)
                if not _can_continue(prefix, brief_list):
                    if reusable is not None and reusable.skipped_reason is not None:
                        if (
                            reusable.run_ids != prefix.run_ids
                            or reusable.input_state_id != prefix.output_state_id
                            or reusable.run_count != prefix.run_count
                            or not math.isclose(
                                reusable.wall_s, prefix.wall_s, abs_tol=1e-9
                            )
                        ):
                            raise ValueError("resumed skip differs from its first edge")
                        entry = reusable
                    else:
                        entry = SweepEntry(
                            sequence=sequence,
                            input_state_id=prefix.output_state_id,
                            run_ids=prefix.run_ids,
                            wall_s=prefix.wall_s,
                            run_count=prefix.run_count,
                            skipped_reason="first output is not physically valid",
                            fake=start_state.fake,
                        )
                        _append_entry(journal, journal_id, history, entry)
                    entries.append(entry)
                    continue

                if reusable is not None and reusable.skipped_reason is None and reusable.run_status not in _RETRYABLE:
                    first_outcome = engine.store.load_outcome(prefix.run_id)
                    assert first_outcome.state is not None
                    assert first_outcome.evidence is not None
                    _verify_reused(
                        reusable,
                        engine=engine,
                        recipe=second_recipe,
                        input_state=first_outcome.state,
                        prefix=prefix,
                        start=start_evidence,
                        parent=first_outcome.evidence,
                        briefs=brief_list,
                        noise=noise_map,
                    )
                    entry = reusable
                else:
                    first_outcome = engine.store.load_outcome(prefix.run_id)
                    assert first_outcome.state is not None
                    assert first_outcome.evidence is not None
                    entry = _new_edge(
                        engine=engine,
                        recipe=second_recipe,
                        sequence=sequence,
                        input_state=first_outcome.state,
                        prefix=prefix,
                        start=start_evidence,
                        parent=first_outcome.evidence,
                        briefs=brief_list,
                        noise=noise_map,
                    )
                    _append_entry(journal, journal_id, history, entry)
                entries.append(entry)

        all_attempts = [
            entry
            for sequence, attempts in history.items()
            if len(sequence) <= max_depth
            for entry in attempts
            if entry.run_id is not None
        ]
        return SweepResult(
            sweep_id=sweep_id,
            design=start_state.design,
            start_state_id=start_state.id,
            start_evidence=start_evidence,
            briefs=brief_list,
            recipes=recipe_list,
            entries=tuple(entries),
            max_depth=max_depth,
            journal_path=str(journal.path(journal_id)),
            start_measure_wall_s=start_measure_wall_s,
            total_recipe_runs=sum(
                entry.run_status != RunStatus.TRANSIENT for entry in all_attempts
            ),
            total_charged_wall_s=sum(entry.edge_wall_s for entry in all_attempts),
            actual_backend_runs=sum(not entry.cache_hit for entry in all_attempts),
            fake=start_state.fake,
        )

