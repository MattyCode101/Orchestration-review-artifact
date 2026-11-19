"""Detached, train-only pilot of the frozen four briefs and seven policies.

The pilot uses the existing StepEngine, tagged runner, copied recipes and AMD
validator. Its wrapper requires placement properties no higher than the start
state's (orch/engine/start_properties.py; decision of 2026-11-07, replacing "zero")
and AMD PASS before an output can enter an episode journal. A backend counter
caps actual recipe cache misses at 300. No test-design outcome path is opened
by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from orch.analysis.fit_workers import TRAINING_DESIGNS, TRAINING_ROOT
from orch.baselines.verification import VerificationLedger
from orch.config import load_noise, load_recipe_sets, load_worker_card
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import (
    AppliedStep, DesignState, EquivalenceCheck, Evidence, RunOutcome, RunStatus,
)
from orch.engine.apply import StepEngine
from orch.engine.journal import Journal
from orch.engine.jobtag import _own_pids, _stat, _tag_of, group_members, sweep_tag, tagged
from orch.engine.runner import Runner
from orch.engine.start_properties import (
    PROPERTY_KEYS, PropertyRuleError, check_not_above_start, check_start_counts,
)
from orch.engine.store import StateStore, _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from orch.episode import EpisodeResult, FinalValidation, run_episode
from orch.policies.repair_loop import RepairLoopPolicy
from orch.policies.training_baselines import (
    FixedAlternationPolicy, SingleSpecialistPolicy, UniformRandomPolicy,
)
from orch.runtime import make_real_runtime
from orch.workers.fitted import FittedWorker
from orch.workers.telemetry import PILOT_TRAINING_ROOT_V3, SavedTelemetryProvider

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
ROOT = PILOT_TRAINING_ROOT_V3
DEPTH3 = PROJECT / "data/sweeps/thursday_crossworker_depth3_2026-11-05"
SMOKE = PROJECT / "data/scratch_thursday_2026-11-05/cache_gate_smoke/status.json"
BRIEFS = ("T0", "A0", "T", "A")
POLICIES = (
    "repair_loop", "single_specialist", "fixed_AB", "fixed_BA",
    "random_seed_0", "random_seed_1", "random_seed_2",
)
DESIGNS = (
    "vexriscv_re-place", "logicnets_jscl", "rosetta_spam-filter",
    "rosetta_3d-rendering", "finn_radioml",
)
CAP = 300
MAX_SLOTS = 96
LOAD_CEILING = 180.0
MEMORY_FLOOR = 200_000_000_000
SOURCE_FILES = {
    "pilot": "scripts/thursday_night_pilot.py",
    "episode": "orch/episode.py",
    "repair_loop": "orch/policies/repair_loop.py",
    "baselines": "orch/policies/training_baselines.py",
    "views": "orch/contract/views.py",
    "brief_text": "orch/contract/brief_text.py",
    "briefs": "configs/briefs.yaml",
    "worker_features": "configs/worker_features.yaml",
    "A_fit": "configs/workers/A_fit.json",
    "B_fit": "configs/workers/B_fit.json",
    "timing_card": "configs/workers/timing.yaml",
    "area_card": "configs/workers/area.yaml",
    "telemetry": "orch/workers/telemetry.py",
    "fitted_worker": "orch/workers/fitted.py",
    "worker_base": "orch/workers/base.py",
    "models": "orch/contract/models.py",
    "gate": "orch/engine/gate.py",
    "apply": "orch/engine/apply.py",
    "cache": "orch/engine/cache.py",
    "store": "orch/engine/store.py",
    "runner": "orch/engine/runner.py",
    "jobtag": "orch/engine/jobtag.py",
    "runtime": "orch/runtime.py",
    "real_backend": "orch/engine/real_backend.py",
    "meter": "orch/engine/meter.py",
    "meter_parsers": "orch/engine/meter_parsers.py",
    "verification": "orch/baselines/verification.py",
    "validator": "orch/recipes/adapters/validator.py",
    "tcl_adapter": "orch/recipes/adapters/tcl.py",
    "latch_adapter": "orch/recipes/adapters/latch.py",
    "registry": "recipes/registry.yaml",
    "recipe_sets": "configs/recipe_sets.yaml",
    "env": "recipes/support/env.sh",
}


class PilotSafetyStop(RuntimeError):
    """A frozen validity or provenance check failed before journal commit."""


class PilotRunCap(RuntimeError):
    """The shared cap was reached before another real recipe launch."""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def source_hashes() -> dict[str, str]:
    """Bind every code, configuration and tool-facing byte used by the pilot."""

    return {name: _sha256_file(PROJECT / relative) for name, relative in SOURCE_FILES.items()}


def episode_schedule() -> tuple[tuple[str, str, str], ...]:
    """Spread policies across every design and brief before later rounds."""

    rows: list[tuple[str, str, str]] = []
    for round_index in range(len(POLICIES)):
        for brief_index, brief_id in enumerate(BRIEFS):
            for design_index, design in enumerate(DESIGNS):
                policy = POLICIES[(round_index + brief_index * len(DESIGNS) + design_index) % len(POLICIES)]
                rows.append((design, brief_id, policy))
    if len(rows) != 140 or len(set(rows)) != 140:
        raise ValueError("pilot schedule lost a policy/brief/design cell")
    return tuple(rows)


def episode_id(design: str, brief_id: str, policy_key: str) -> str:
    if design not in DESIGNS or brief_id not in BRIEFS or policy_key not in POLICIES:
        raise ValueError("episode is outside the training pilot")
    return f"pilot_{design}_{brief_id}_{policy_key}"


def policy_for(key: str):
    if key == "repair_loop":
        return RepairLoopPolicy()
    if key == "single_specialist":
        return SingleSpecialistPolicy()
    if key == "fixed_AB":
        return FixedAlternationPolicy(first="A")
    if key == "fixed_BA":
        return FixedAlternationPolicy(first="B")
    if key.startswith("random_seed_") and key in POLICIES:
        return UniformRandomPolicy(seed=int(key.rsplit("_", 1)[1]))
    raise ValueError("unknown frozen pilot policy")


def require_start_properties(evidence: Evidence) -> None:
    """A start state needs usable counts, whatever their size: it is the reference."""

    try:
        check_start_counts(evidence)
    except PropertyRuleError as exc:
        raise PilotSafetyStop(str(exc)) from exc


def require_clean_properties(evidence: Evidence, start: Evidence | None) -> None:
    """No more DONT_TOUCH, fixed LOC or fixed BEL cells than the start state.

    The authors' rule of 2026-11-07 replaces the earlier requirement of zero. The pinned pure
    gate leaves these measured counts diagnostic, so this layer enforces them; the rule
    itself is defined once in orch/engine/start_properties.py.
    """

    try:
        check_not_above_start(evidence, start)
    except PropertyRuleError as exc:
        raise PilotSafetyStop(str(exc)) from exc


def _journal_start(events: list[dict[str, Any]]) -> Evidence:
    """The start Evidence an episode journal opens with."""

    if not events or events[0].get("type") != "start":
        raise PilotSafetyStop("episode journal lacks its start event")
    return Evidence.model_validate(events[0]["start_evidence"])


def outputs_compared(record: dict[str, Any], runs_root: Path) -> int:
    """Require the same nonzero AMD output census as the verified depth-three driver.

    Source: scripts/thursday_crossworker_depth3.py:178-184.
    """

    attempts = record.get("attempts") or []
    if not attempts:
        return 0
    run_dir = Path(str(attempts[-1]["run_dir"])).resolve(strict=True)
    if not run_dir.is_relative_to(runs_root.resolve(strict=True)):
        raise PilotSafetyStop("AMD validation log escaped its verification runs")
    log_path = run_dir / "validate.log"
    content = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return max((int(x) for x in re.findall(r"Outputs\s*\((\d+)\):", content)), default=0)


def require_amd_pass(ledger: VerificationLedger, design: str, state_id: str, *,
                     allow_new: bool) -> dict[str, Any]:
    record = ledger.verify(design, state_id) if allow_new else ledger.require_pass(design, state_id)
    count = outputs_compared(record, ledger.root / "runs")
    if record["result"] != "pass" or count <= 0:
        raise PilotSafetyStop(f"AMD simulation did not PASS with compared outputs: {design} {state_id}")
    log_path = Path(record["attempts"][-1]["run_dir"]) / "validate.log"
    proof = {
        "schema_version": "thursday-night-amd-census-v1",
        "design": design, "state_sha256": state_id,
        "validator_record_sha256": _sha256_file(ledger._record_path(design, state_id)),
        "validate_log_sha256": _sha256_file(log_path),
        "outputs_compared": count,
    }
    path = ledger.root / "output_census" / design / f"{state_id}.json"
    with _file_lock(path.with_suffix(".lock")):
        if path.is_file():
            if path.read_bytes() != _json_bytes(proof):
                raise PilotSafetyStop("AMD output census proof changed")
        elif allow_new:
            _atomic_bytes(path, _json_bytes(proof))
        else:
            raise PilotSafetyStop("AMD output census proof is missing")
    return record


def _training_inputs() -> dict[str, str]:
    manifest = read(TRAINING_ROOT / "launch_manifest.json")
    rows = manifest.get("train_inputs", [])
    values = {row["design"]: row["sha256"] for row in rows}
    if set(values) != set(DESIGNS) or len(rows) != len(DESIGNS) or set(DESIGNS) != set(TRAINING_DESIGNS):
        raise ValueError("pilot input manifest is not exactly five training designs")
    return values


def _start_source_hashes(meter_version: str, references: dict[str, str]) -> dict[str, dict[str, str]]:
    """Pin the already completed current-meter B2 input checks."""

    version_key = hashlib.sha256(meter_version.encode("utf-8")).hexdigest()
    return {
        design: {
            "depth3_plan_sha256": _sha256_file(DEPTH3 / design / "plan.json"),
            "input_gate_sha256": _sha256_file(DEPTH3 / design / "input_remeasurement.json"),
            "b2_evidence_sha256": _sha256_file(
                DEPTH3 / design / "measurements" / references[design]
                / version_key / "evidence.json"
            ),
        }
        for design in DESIGNS
    }


def make_plan(runtime) -> dict[str, Any]:
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    selected = {worker: tuple(sets.worker_sets[worker]) for worker in ("timing", "area")}
    if any(len(ids) != 5 for ids in selected.values()):
        raise ValueError("pilot requires both signed five-recipe workers")
    references = _training_inputs()
    return {
        "schema_version": "thursday-night-pilot-v1",
        "training_only": True,
        "data_root": str(ROOT),
        "training_inputs": references,
        "training_launch_manifest_sha256": _sha256_file(TRAINING_ROOT / "launch_manifest.json"),
        "current_b2_sources": _start_source_hashes(runtime.meter.version, references),
        "schedule": [list(row) for row in episode_schedule()],
        "brief_hashes": {brief_id: load_frozen_brief(brief_id).fingerprint() for brief_id in BRIEFS},
        "policy_versions": {key: policy_for(key).version for key in POLICIES},
        "recipe_versions": {
            rid: runtime.registry.get(rid).version for ids in selected.values() for rid in ids
        },
        "meter_version": runtime.meter.version,
        "smoke_sha256": _sha256_file(SMOKE),
        "source_hashes": source_hashes(),
        "max_new_real_recipe_runs": CAP,
        "shared_slots": MAX_SLOTS,
        "load_ceiling_1m": LOAD_CEILING,
        "available_memory_floor_bytes": MEMORY_FLOOR,
    }


def require_plan(runtime) -> dict[str, Any]:
    """Create once, then fail closed on any changed source or input identity."""

    frozen = make_plan(runtime)
    path = ROOT / "plan.json"
    expected = _json_bytes(frozen)
    with _file_lock(ROOT / "plan.lock"):
        if path.is_file():
            if path.read_bytes() != expected:
                raise PilotSafetyStop("pilot plan differs from frozen source or input")
        else:
            _atomic_bytes(path, expected)
    return frozen


class CountedBackend:
    """Reserve one of 300 slots before a real recipe cache miss can launch."""

    def __init__(self, backend, root: Path, plan_sha256: str,
                 check_pins: Callable[[], None], cap: int = CAP) -> None:
        self.backend = backend
        self.root = root
        self.plan_sha256 = plan_sha256
        self.check_pins = check_pins
        self.cap = cap
        self.active_episode: str | None = None
        self.path = root / "new_real_runs.json"
        self.lock = root / "new_real_runs.lock"
        with _file_lock(self.lock):
            if self.path.is_file():
                self._load()
            else:
                _atomic_bytes(self.path, _json_bytes({
                    "schema_version": "thursday-night-new-runs-v1",
                    "plan_sha256": plan_sha256, "cap": cap, "reservations": [],
                }))

    def _load(self) -> dict[str, Any]:
        value = read(self.path)
        rows = value.get("reservations")
        if (
            value.get("schema_version") != "thursday-night-new-runs-v1"
            or value.get("plan_sha256") != self.plan_sha256
            or value.get("cap") != self.cap
            or not isinstance(rows, list) or len(rows) > self.cap
        ):
            raise PilotSafetyStop("new-run counter is corrupt or belongs to another plan")
        return value

    def count(self) -> int:
        return len(self._load()["reservations"])

    def measure_start(self, *args, **kwargs):
        raise PilotSafetyStop("B2 start was not seeded with current-meter Evidence")

    def execute(self, state, recipe, *, run_id, run_dir) -> RunOutcome:
        self.check_pins()
        if self.active_episode is None:
            raise PilotSafetyStop("real recipe launch lacks a pilot episode identity")
        with _file_lock(self.lock):
            value = self._load()
            if len(value["reservations"]) >= self.cap:
                raise PilotRunCap(f"pilot reached {self.cap} new real recipe runs")
            value["reservations"].append({
                "episode_id": self.active_episode, "run_id": run_id,
                "input_state_id": state.id, "recipe_id": recipe.id, "reserved_at_utc": utc(),
            })
            _atomic_bytes(self.path, _json_bytes(value))
        return self.backend.execute(state, recipe, run_id=run_id, run_dir=run_dir)


class VerifiedEpisodeEngine:
    """Permit a gate verdict into the episode only after extra brief validity."""

    def __init__(self, engine: StepEngine, ledger: VerificationLedger) -> None:
        self.engine = engine
        self.ledger = ledger
        self.store = engine.store
        self.meter_version = engine.meter_version
        self.cache_enabled = engine.cache_enabled

    def measure_start(self, state: DesignState) -> Evidence:
        evidence = self.engine.measure_start(state)
        require_start_properties(evidence)
        return evidence

    def apply(self, state, recipe, *, brief, start, parent, guards=(), noise=None) -> AppliedStep:
        applied = self.engine.apply(
            state, recipe, brief=brief, start=start, parent=parent,
            guards=guards, noise=noise,
        )
        outcome = applied.outcome
        if outcome.state is not None:
            if outcome.evidence is None:
                raise PilotSafetyStop("saved output lacks independent meter Evidence")
            require_clean_properties(outcome.evidence, start)
            require_amd_pass(self.ledger, outcome.state.design, outcome.state.id, allow_new=True)
        return applied


def _seed_starts(runtime, references: dict[str, str]) -> dict[str, DesignState]:
    """Copy only pinned training B2 bytes and current independent Evidence."""

    source_store = StateStore(TRAINING_ROOT)
    states: dict[str, DesignState] = {}
    version_key = hashlib.sha256(runtime.meter.version.encode("utf-8")).hexdigest()
    for design in DESIGNS:
        state_id = references[design]
        source = source_store.state_path(state_id)
        state = runtime.store.import_state(source, design=design, fake=False)
        if state.id != state_id or state.parent_id is not None:
            raise PilotSafetyStop(f"training B2 state differs: {design}")
        campaign = DEPTH3 / design
        frozen = read(campaign / "plan.json")
        check = read(campaign / "input_remeasurement.json")
        if (
            frozen.get("b2_hash") != state_id
            or frozen.get("meter_version") != runtime.meter.version
            or check.get("meter_version") != runtime.meter.version
            or state_id not in check.get("checked_state_hashes", ())
        ):
            raise PilotSafetyStop(f"current-meter training input gate is absent: {design}")
        source_evidence = (
            campaign / "measurements" / state_id / version_key / "evidence.json"
        )
        evidence = Evidence.model_validate_json(source_evidence.read_bytes())
        if (
            evidence.state_id != state_id or evidence.fake
            or evidence.meter_version != runtime.meter.version
            or evidence.routed is not True or evidence.route_errors != 0
            or evidence.drc_errors != 0
        ):
            raise PilotSafetyStop(f"current B2 Evidence is invalid: {design}")
        require_start_properties(evidence)
        target = ROOT / "measurements" / state_id / version_key / "evidence.json"
        content = _json_bytes(evidence.model_dump(mode="json"))
        if target.is_file() and target.read_bytes() != content:
            raise PilotSafetyStop(f"pilot B2 Evidence changed: {design}")
        if not target.is_file():
            _atomic_bytes(target, content)
        states[design] = state
    return states


def _workers(runtime) -> dict[str, FittedWorker]:
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    workers = {
        worker_id: FittedWorker(
            load_worker_card(PROJECT / "configs/workers" / f"{worker_id}.yaml"),
            sets.worker_sets[worker_id], runtime.registry,
        )
        for worker_id in ("timing", "area")
    }
    if any(worker.model_label != "FINAL" for worker in workers.values()):
        raise PilotSafetyStop("training pilot requires both frozen FINAL workers")
    return workers


def _validator(ledger: VerificationLedger, brief):
    def saved(golden: DesignState, revised: DesignState) -> FinalValidation:
        if revised.id == golden.id:
            return FinalValidation(check=EquivalenceCheck(
                result="pass", method="identical B2 checkpoint",
                vectors=brief.equivalence_vectors, reference_state_id=golden.id,
            ), note="No-change start is byte-identical to the reference B2")
        record = require_amd_pass(ledger, revised.design, revised.id, allow_new=False)
        return FinalValidation(check=EquivalenceCheck(
            result="pass", method="saved AMD 100+200-vector simulation",
            vectors=brief.equivalence_vectors, reference_state_id=golden.id,
        ), artifact_dir=str(record["raw_report_path"]),
            note="Exact-DCP verified output; no final-pick tool rerun")
    return saved


def _audit_episode(episode: str, store: StateStore, ledger: VerificationLedger) -> None:
    """Recheck every replayed physical state against the frozen extra validity."""

    _audit_events(Journal(ROOT).read(episode), store, ledger)


def _audit_events(events: list[dict[str, Any]], store: Any, ledger: Any) -> None:
    """The audit proper: each step output against the start Evidence the journal opens with.

    An episode that has not started has no journal and nothing to audit; every driver audits an
    episode before it runs it."""

    if not events:
        return
    start = _journal_start(events)
    require_start_properties(start)
    for event in events[1:]:
        if event.get("type") != "step" or not event["verdict"]["physically_valid"]:
            continue
        outcome = store.load_outcome(event["step"]["run_id"])
        if outcome.run.status != RunStatus.OK or outcome.state is None or outcome.evidence is None:
            raise PilotSafetyStop("physically valid journal step lacks an OK output")
        require_clean_properties(outcome.evidence, start)
        require_amd_pass(ledger, outcome.state.design, outcome.state.id, allow_new=False)


def _repair_final_pick(result: EpisodeResult) -> str | None:
    """Identify the selected occurrence, including duplicate output hashes."""

    if result.final_rank is None or not result.validation_attempts:
        return None
    selected_run = result.validation_attempts[result.final_rank].candidate_run_id
    if selected_run is None:
        return None
    events = Journal(ROOT).read(result.episode_id)
    steps = {event["step"]["run_id"]: event["step"]["mandate_id"]
             for event in events if event.get("type") == "step"}
    mandates = {event["mandate"]["id"]: event["mandate"]
                for event in events if event.get("type") == "mandate"}
    if selected_run not in steps or steps[selected_run] not in mandates:
        raise PilotSafetyStop("final run lacks its mandate context")
    context = json.loads(mandates[steps[selected_run]]["context_ref"])
    return context.get("phase") if context.get("policy") == "repair_loop" else None


def _result_row(result: EpisodeResult, store: StateStore, counter: CountedBackend,
                design: str, brief_id: str, policy_key: str) -> dict[str, Any]:
    hits = sum(store.load_outcome(step.run_id).run.cache_hit for step in result.steps)
    reserved = [
        row for row in counter._load()["reservations"]
        if row["episode_id"] == result.episode_id
    ]
    if (
        result.validation_status != "pass" or result.final_state is None
        or result.final_evidence is None or result.final_verdict is None
        or not result.final_verdict.brief_valid
    ):
        raise PilotSafetyStop("completed episode lacks a verified brief-valid final pick")
    require_clean_properties(
        result.final_evidence, _journal_start(Journal(ROOT).read(result.episode_id)),
    )
    return {
        "schema_version": "thursday-night-pilot-result-v1",
        "episode_id": result.episode_id, "design": design, "brief": brief_id,
        "policy": policy_key, "final_state_id": result.final_state.id,
        "final_metrics": {
            name.value: metric.value for name, metric in result.final_evidence.metrics.items()
        },
        "final_brief_valid": True, "validation_status": result.validation_status,
        "runs_used": result.recipe_runs, "cache_hits": hits,
        "new_real_runs_reserved": len(reserved),
        "repair_final_phase": _repair_final_pick(result),
        "stop_reason": result.stop_reason, "journal_path": result.journal_path,
        "completed_at_utc": utc(),
    }


def _status(phase: str, *, error: str | None = None) -> dict[str, Any]:
    rows = [read(path) for path in (ROOT / "results").glob("*.json")] if (ROOT / "results").is_dir() else []
    counter = read(ROOT / "new_real_runs.json") if (ROOT / "new_real_runs.json").is_file() else {"reservations": []}
    value = {
        "schema_version": "thursday-night-pilot-status-v1",
        "phase": phase, "updated_at_utc": utc(),
        "planned_episodes": len(episode_schedule()), "completed_episodes": len(rows),
        "new_real_runs_reserved": len(counter["reservations"]),
        "cache_hits": sum(row["cache_hits"] for row in rows),
        "recipe_runs_used": sum(row["runs_used"] for row in rows),
        "error": error,
    }
    _atomic_bytes(ROOT / "status.json", _json_bytes(value))
    return value


def run() -> None:
    runtime = make_real_runtime(
        ROOT, cache_enabled=True, max_slots=MAX_SLOTS,
        load_ceiling_1m=LOAD_CEILING,
        min_available_memory_bytes=MEMORY_FLOOR,
        latch_smoke_status=SMOKE,
    )
    frozen = require_plan(runtime)
    plan_sha = _sha256_file(ROOT / "plan.json")
    references = frozen["training_inputs"]
    starts = _seed_starts(runtime, references)
    workers = _workers(runtime)
    provider = SavedTelemetryProvider(ROOT)
    ledger = VerificationLedger(
        store=runtime.store, runner=runtime.runner,
        validator=runtime.validator, references=references,
    )
    backend = CountedBackend(runtime.backend, ROOT, plan_sha, lambda: require_plan(runtime))
    engine = VerifiedEpisodeEngine(StepEngine(
        store=runtime.store, cache=runtime.cache, backend=backend,
        meter_version=runtime.meter.version, cache_enabled=True,
    ), ledger)
    (ROOT / "results").mkdir(exist_ok=True)
    _status("running")
    for design, brief_id, key in episode_schedule():
        require_plan(runtime)
        identity = episode_id(design, brief_id, key)
        path = ROOT / "results" / f"{identity}.json"
        if path.is_file():
            row = read(path)
            if (
                row.get("episode_id") != identity
                or row.get("design") != design or row.get("brief") != brief_id
                or row.get("policy") != key or not row.get("final_brief_valid")
            ):
                raise PilotSafetyStop(f"saved pilot result differs: {identity}")
            _audit_episode(identity, runtime.store, ledger)
            continue
        _audit_episode(identity, runtime.store, ledger)
        brief = load_frozen_brief(brief_id)
        start = starts[design]
        start_evidence = engine.measure_start(start)
        noise = load_noise(PROJECT / "configs/noise.yaml").for_start(design, start_evidence)
        backend.active_episode = identity
        try:
            result = run_episode(
                engine=engine, registry=runtime.registry, workers=workers,
                policy=policy_for(key), brief=brief, start_state=start,
                episode_id=identity, noise=noise, telemetry_provider=provider,
                validator=_validator(ledger, brief),
            )
        except PilotRunCap as exc:
            _status("capped", error=str(exc))
            return
        finally:
            backend.active_episode = None
        _audit_episode(identity, runtime.store, ledger)
        row = _result_row(result, runtime.store, backend, design, brief_id, key)
        _atomic_bytes(path, _json_bytes(row))
        _status("running")
    _status("complete")


def driver_alive(pid: int) -> bool:
    try:
        info = _stat(pid)
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return bool(info and info.state != "Z" and str(SCRIPT).encode() in args and b"--run" in args)
    except OSError:
        return False


def _targets() -> tuple[set[str], set[int]]:
    """Find only exact tags of processes working inside this pilot data root."""

    prefix = str(ROOT).encode()
    tags: set[str] = set()
    groups: set[int] = set()
    for pid in _own_pids():
        info = _stat(pid)
        if info is None or info.state == "Z":
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd").encode()
        except OSError:
            continue
        if cwd != prefix and not cwd.startswith(prefix + b"/"):
            continue
        tag = _tag_of(pid)
        if tag and re.fullmatch(r"[0-9a-f]{32}", tag):
            tags.add(tag)
            groups.add(info.pgrp)
    groups.discard(os.getpgrp())
    return tags, groups


def cleanup() -> dict[str, Any]:
    tags, groups = _targets()
    result: dict[str, Any] = {
        "schema_version": "thursday-night-pilot-cleanup-v1",
        "exact_tags": sorted(tags), "groups": sorted(groups), "started_at_utc": utc(),
    }
    _atomic_bytes(ROOT / "watchdog_cleanup_audit.json", _json_bytes(result))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in sorted(groups):
            if any(_tag_of(member.pid) in tags for member in group_members(group)):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    result["tag_sweeps"] = {tag: list(sweep_tag(tag, grace_s=5).alive) for tag in sorted(tags)}
    result["survivors"] = {tag: [p.pid for p in tagged(tag)] for tag in sorted(tags) if tagged(tag)}
    result["finished_at_utc"] = utc()
    _atomic_bytes(ROOT / "watchdog_cleanup_audit.json", _json_bytes(result))
    return result


def watch() -> None:
    pid = read(ROOT / "launch.json")["driver_pid"]
    if not driver_alive(pid):
        raise RuntimeError("pilot driver absent at watchdog start")
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        if not driver_alive(pid):
            state = read(ROOT / "status.json") if (ROOT / "status.json").is_file() else {}
            tags, _ = _targets()
            if state.get("phase") not in ("complete", "capped") or tags:
                cleanup()
            return
        time.sleep(15)
    raise RuntimeError("pilot watchdog lifetime ended with driver alive")


def launch() -> dict[str, Any]:
    runtime = make_real_runtime(
        ROOT, cache_enabled=True, max_slots=MAX_SLOTS,
        load_ceiling_1m=LOAD_CEILING,
        min_available_memory_bytes=MEMORY_FLOOR,
        latch_smoke_status=SMOKE,
    )
    require_plan(runtime)
    _seed_starts(runtime, _training_inputs())
    _workers(runtime)
    SavedTelemetryProvider(ROOT)
    load, available = Runner._resource_snapshot()
    if load > LOAD_CEILING or available < MEMORY_FLOOR:
        raise PilotSafetyStop("load or available memory blocks pilot launch")
    with _file_lock(ROOT / "launch.lock"):
        if (ROOT / "launch.json").exists():
            raise FileExistsError("training pilot already launched")
        with (ROOT / "driver.stdout").open("wb") as out, (ROOT / "driver.stderr").open("wb") as err:
            driver = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--run"], cwd=PROJECT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        result = {
            "schema_version": "thursday-night-pilot-launch-v1",
            "driver_pid": driver.pid, "started_at_utc": utc(),
            "plan_sha256": _sha256_file(ROOT / "plan.json"),
            "load_1m_at_launch": load, "available_memory_bytes_at_launch": available,
        }
        _atomic_bytes(ROOT / "launch.json", _json_bytes(result))
        try:
            with (ROOT / "watchdog.stdout").open("wb") as out, (ROOT / "watchdog.stderr").open("wb") as err:
                watcher = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "--watch"], cwd=PROJECT,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    start_new_session=True, close_fds=True,
                )
            result["watchdog_pid"] = watcher.pid
            _atomic_bytes(ROOT / "launch.json", _json_bytes(result))
            return result
        except Exception:
            if driver_alive(driver.pid):
                group = os.getpgid(driver.pid)
                if group == driver.pid and group != os.getpgrp():
                    os.killpg(group, signal.SIGTERM)
            time.sleep(3)
            cleanup()
            raise


def status() -> dict[str, Any]:
    state = read(ROOT / "status.json") if (ROOT / "status.json").is_file() else {"phase": "not_launched"}
    manifest = read(ROOT / "launch.json") if (ROOT / "launch.json").is_file() else {}
    pid = manifest.get("driver_pid")
    load, available = Runner._resource_snapshot()
    return {
        **state, "driver_pid": pid,
        "driver_alive": driver_alive(pid) if type(pid) is int else False,
        "driver_stderr_bytes": (ROOT / "driver.stderr").stat().st_size if manifest else 0,
        "busy_shared_slots": Runner(PROJECT / "data/slots", max_slots=MAX_SLOTS)._busy_slots(),
        "load_1m": load, "available_memory_bytes": available,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ("launch", "run", "watch", "status"):
        group.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    if args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.run:
        try:
            run()
        except Exception as exc:
            _status("failed", error=f"{type(exc).__name__}: {exc}")
            raise
    elif args.watch:
        watch()
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
