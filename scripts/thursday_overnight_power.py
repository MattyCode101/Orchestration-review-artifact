"""Detached, resumable, AMD-checked power campaign on five training designs.

Phase a applies four copied run-4 power recipes to each distinct B2/D1 state.
After every phase-a queue is terminal, phase b applies ten signed worker
recipes to each of the four B2 power lineages. All tool work goes through the
existing engine, meter, runner and validator. No tool-facing script lives here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.baselines.verification import VerificationLedger
from orch.config import load_recipe_sets
from orch.contract.models import Evidence, MetricName, RunStatus
from orch.engine.jobtag import _own_pids, _stat, _tag_of, group_members, sweep_tag, tagged
from orch.engine.journal import Journal
from orch.engine.runner import Runner
from orch.engine.store import StateStore, _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from orch.recipes.registry import load_registry
from orch.runtime import make_real_runtime

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
TOP = PROJECT / "data/sweeps/thursday_overnight_power_2026-11-05"
SELECTION = TOP / "selection.json"
SELECTION_SHA = "574b0b7b094fd921b511d9db37cfe364a52fb31c97e36834a505d67f5ce19185"
SMOKE = PROJECT / "data/scratch_thursday_2026-11-05/cache_gate_smoke/status.json"
DESIGNS = (
    "vexriscv_re-place", "rosetta_spam-filter", "rosetta_3d-rendering",
    "logicnets_jscl", "finn_radioml",
)
POWER = ("P_POST_ALL", "P_POST_BRAM", "P_PRE_ALL", "P_ROUTE_HIGHDELAY")
WORKER = (
    "LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock",
    "LATCH.hardblock_relocate", "LATCH.memory_path_opt",
    "A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED",
    "A_E_DUPL_Q1000",
)
PROPS = ("n_DONT_TOUCH", "n_IS_LOC_FIXED", "n_IS_BEL_FIXED")
METRICS = tuple(MetricName)
MAX_ACTIVE = 12
MAX_SLOTS = 96
LOAD_CEILING = 180.0
MEMORY_FLOOR = 200_000_000_000
TERMINAL = frozenset({"complete", "stopped", "preflight_failed", "failed"})
SOURCE_FILES = {
    "driver": "scripts/thursday_overnight_power.py",
    "runner": "orch/engine/runner.py",
    "runtime": "orch/runtime.py",
    "real_backend": "orch/engine/real_backend.py",
    "apply": "orch/engine/apply.py",
    "meter": "orch/engine/meter.py",
    "meter_parsers": "orch/engine/meter_parsers.py",
    "models": "orch/contract/models.py",
    "store": "orch/engine/store.py",
    "cache": "orch/engine/cache.py",
    "journal": "orch/engine/journal.py",
    "jobtag": "orch/engine/jobtag.py",
    "verification": "orch/baselines/verification.py",
    "validator": "orch/recipes/adapters/validator.py",
    "tcl_adapter": "orch/recipes/adapters/tcl.py",
    "latch_adapter": "orch/recipes/adapters/latch.py",
    "registry": "recipes/registry.yaml",
    "recipe_sets": "configs/recipe_sets.yaml",
    "env": "recipes/support/env.sh",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, Any]:
    item = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(item, dict):
        raise ValueError(f"expected JSON object: {path}")
    return item


def selected() -> dict[str, Any]:
    """Fail closed if the frozen training inputs or signed worker set change."""
    if _sha256_file(SELECTION) != SELECTION_SHA:
        raise ValueError("frozen power selection changed")
    item = read(SELECTION)
    if (
        item.get("schema_version") != "thursday-overnight-power-selection-v1"
        or tuple(item.get("training_designs", ())) != DESIGNS
        or set(item.get("by_design", {})) != set(DESIGNS)
        or tuple(item.get("power_recipe_ids", ())) != POWER
        or tuple(item.get("worker_recipe_ids", ())) != WORKER
        or item.get("distinct_inputs") != 44
        or item.get("phase_a_edges") != 176
        or item.get("phase_b_edges") != 200
    ):
        raise ValueError("frozen power scope differs")
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    if (
        tuple(sets.worker_sets.get("timing", ())) != WORKER[:5]
        or tuple(sets.worker_sets.get("area", ())) != WORKER[5:]
    ):
        raise ValueError("signed worker recipe set changed")
    return item


def root_for(phase: str, design: str) -> Path:
    if phase not in ("a", "b") or design not in DESIGNS:
        raise ValueError("phase or design is outside the frozen training campaign")
    return TOP / f"phase_{phase}" / design


def a_tasks(inputs: dict[str, Any]) -> tuple[dict[str, str], ...]:
    states = inputs["states"]
    hashes = [state["hash"] for state in states]
    if len(hashes) != len(set(hashes)) or inputs["b2_hash"] not in hashes:
        raise ValueError("phase-a parents are not distinct or lack B2")
    return tuple(
        {
            "key": f"a:{state_hash}:{recipe}",
            "input_hash": state_hash,
            "recipe_id": recipe,
            "power_recipe_id": recipe,
            "source_run_id": "",
        }
        for recipe in POWER for state_hash in sorted(hashes)
    )


def b_tasks(a_rows: dict[str, dict[str, Any]], b2_hash: str) -> tuple[dict[str, str], ...]:
    """Keep four recipe lineages even when their DCP hashes coincide."""
    tasks = []
    for power in POWER:
        row = a_rows.get(f"a:{b2_hash}:{power}")
        if (
            row is None or row.get("status") != "ok"
            or row.get("recipe_id") != power
            or row.get("input_hash") != b2_hash
            or not row.get("output_hash") or not row.get("run_id")
            or row.get("amd_result") != "pass"
            or row.get("amd_outputs_compared", 0) <= 0
        ):
            raise ValueError(f"missing AMD-PASS B2 power lineage: {power}")
        for worker in WORKER:
            tasks.append({
                "key": f"b:{power}:{worker}",
                "input_hash": row["output_hash"],
                "recipe_id": worker,
                "power_recipe_id": power,
                "source_run_id": row["run_id"],
            })
    return tuple(tasks)


def source_hashes() -> dict[str, str]:
    return {name: _sha256_file(PROJECT / path) for name, path in SOURCE_FILES.items()}


def a_rows_for_b(design: str, inputs: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Require a complete journal and sealed index before phase b is planned."""
    root = root_for("a", design)
    status = read(root / "status.json")
    summary = read(root / "summary.json")
    plan_path = root / "plan.json"
    index_path = root / "by_design/index.csv"
    expected = len(a_tasks(inputs))
    if (
        status.get("phase") != "complete" or status.get("failed") != 0
        or status.get("amd_pass") != expected
        or summary.get("phase") != "complete"
        or summary.get("planned") != expected
        or summary.get("attempted") != expected
        or summary.get("amd_pass") != expected
        or summary.get("failed") != 0
        or summary.get("plan_sha256") != _sha256_file(plan_path)
        or summary.get("index_sha256") != _sha256_file(index_path)
    ):
        raise ValueError(f"phase a is not fully sealed: {design}")
    events = Journal(root).read(f"overnight_power_a_{design}")
    if not events or events[0].get("plan_sha256") != _sha256_file(plan_path):
        raise ValueError("phase-a journal plan differs")
    rows = {}
    for event in events[1:]:
        if event.get("type") != "edge":
            raise ValueError("phase-a journal contains unknown event")
        row = event["edge"]
        key = row["task_key"]
        if key in rows or row.get("status") != "ok":
            raise ValueError("phase-a journal has duplicate or failed edge")
        rows[key] = row
    if set(rows) != {task["key"] for task in a_tasks(inputs)}:
        raise ValueError("phase-a journal task coverage differs")
    return rows, {
        "phase_a_summary_sha256": _sha256_file(root / "summary.json"),
        "phase_a_index_sha256": _sha256_file(index_path),
        "phase_a_plan_sha256": _sha256_file(plan_path),
    }


def phase_tasks(phase: str, design: str, inputs: dict[str, Any]) -> tuple[tuple[dict[str, str], ...], dict[str, str]]:
    if phase == "a":
        return a_tasks(inputs), {}
    barrier_path = TOP / "phase_a_barrier.json"
    barrier = read(barrier_path)
    if (
        barrier.get("schema_version") != "overnight-power-phase-a-barrier-v1"
        or barrier.get("selection_sha256") != SELECTION_SHA
        or set(barrier.get("statuses", {})) != set(DESIGNS)
        or any(status not in TERMINAL for status in barrier["statuses"].values())
        or barrier["statuses"].get(design) != "complete"
        or any(
            _sha256_file(root_for("a", name) / "status.json") != digest
            for name, digest in barrier.get("status_sha256", {}).items()
        )
        or set(barrier.get("status_sha256", {})) != set(DESIGNS)
    ):
        raise ValueError("global phase-a terminal barrier is absent or changed")
    rows, pins = a_rows_for_b(design, inputs)
    pins["phase_a_barrier_sha256"] = _sha256_file(barrier_path)
    return b_tasks(rows, inputs["b2_hash"]), pins


def frozen_plan(runtime: Any, phase: str, design: str, inputs: dict[str, Any]) -> dict[str, Any]:
    tasks, upstream = phase_tasks(phase, design, inputs)
    recipe_ids = POWER if phase == "a" else WORKER
    return {
        "schema_version": "overnight-power-plan-v1",
        "phase": phase,
        "design": design,
        "selection_sha256": SELECTION_SHA,
        "b2_hash": inputs["b2_hash"],
        "tasks": list(tasks),
        "recipe_versions": {rid: runtime.registry.get(rid).version for rid in recipe_ids},
        "meter_version": runtime.meter.version,
        "smoke_sha256": _sha256_file(SMOKE),
        "source_hashes": source_hashes(),
        "upstream": upstream,
        "max_active": MAX_ACTIVE,
        "shared_slots": MAX_SLOTS,
        "load_ceiling_1m": LOAD_CEILING,
        "available_memory_floor_bytes": MEMORY_FLOOR,
    }


def check_pins(plan: dict[str, Any], runtime: Any) -> None:
    if (
        _sha256_file(SELECTION) != plan["selection_sha256"]
        or _sha256_file(SMOKE) != plan["smoke_sha256"]
        or source_hashes() != plan["source_hashes"]
        or runtime.meter.version != plan["meter_version"]
    ):
        raise ValueError("frozen source, selection, smoke or meter changed")
    registry = load_registry(PROJECT)
    for rid, version in plan["recipe_versions"].items():
        if registry.get(rid).version != version:
            raise ValueError(f"recipe version changed: {rid}")
    if plan["phase"] == "b":
        aroot = root_for("a", plan["design"])
        for key, path in (
            ("phase_a_summary_sha256", aroot / "summary.json"),
            ("phase_a_index_sha256", aroot / "by_design/index.csv"),
            ("phase_a_plan_sha256", aroot / "plan.json"),
            ("phase_a_barrier_sha256", TOP / "phase_a_barrier.json"),
        ):
            if _sha256_file(path) != plan["upstream"][key]:
                raise ValueError(f"sealed phase-a input changed: {key}")


def metric_signature(evidence: Evidence) -> tuple[tuple[str, float], ...]:
    values = []
    for name in METRICS:
        metric = evidence.metric(name)
        if metric is None or not math.isfinite(metric.value):
            raise ValueError(f"Evidence lacks finite {name.value}")
        values.append((name.value, metric.value))
    return tuple(values)


def check_evidence(evidence: Evidence, state_hash: str, plan: dict[str, Any],
                   inputs: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    if (
        evidence.state_id != state_hash
        or evidence.meter_version != plan["meter_version"]
        or evidence.fake or evidence.routed is not True
        or evidence.route_errors != 0 or evidence.drc_errors != 0
        or evidence.clock_digest != inputs["b2_clock_digest"]
        or evidence.constraint_fingerprint != inputs["b2_constraint_fingerprint"]
        or not isinstance(evidence.v_props, dict)
        or any(type(evidence.v_props.get(key)) is not int or evidence.v_props[key] != 0
               for key in PROPS)
    ):
        raise ValueError(f"routed validity, placement properties or B2 digest changed: {state_hash}")
    return metric_signature(evidence)


def raw_cost(run: Any) -> dict[str, Any]:
    return {key: getattr(run, key) for key in (
        "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
        "load_1m_start", "load_1m_end", "busy_slots_start",
        "backpressure_wait_s", "backpressure_events", "cache_hit",
        "original_run_id",
    )}


def complete_cost(run: Any) -> dict[str, Any]:
    cost = raw_cost(run)
    for key in (
        "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
        "load_1m_start", "load_1m_end", "busy_slots_start",
    ):
        value = cost[key]
        if value is None or not math.isfinite(value):
            raise ValueError(f"run lacks finite {key}")
    if run.cache_hit and not run.original_run_id:
        raise ValueError("cache hit lacks original run ID")
    return cost


def outputs_compared(record: dict[str, Any]) -> int:
    attempts = record.get("attempts") or []
    if not attempts:
        return 0
    log_path = Path(attempts[-1]["run_dir"]) / "validate.log"
    content = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return max((int(x) for x in re.findall(r"Outputs\s*\((\d+)\):", content)), default=0)


def hardlink_alias(link: Path, target: Path) -> None:
    """Publish a readable DCP link atomically after its journal event."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists():
        if _sha256_file(link) != _sha256_file(target):
            raise ValueError("existing alias differs from canonical DCP")
        return
    temporary = link.with_name(f".{link.name}.{uuid.uuid4().hex}.link")
    try:
        os.link(target, temporary)
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)


class Study:
    """One design and one phase; a failed check stops that design's queue."""

    def __init__(self, phase: str, design: str, runtime: Any,
                 plan: dict[str, Any], inputs: dict[str, Any]) -> None:
        self.phase_id = phase
        self.design = design
        self.root = root_for(phase, design)
        self.runtime = runtime
        self.plan = plan
        self.inputs = inputs
        self.tasks = tuple(plan["tasks"])
        self.task_by_key = {task["key"]: task for task in self.tasks}
        if len(self.tasks) != len(self.task_by_key):
            raise ValueError("duplicate planned task key")
        self.specs = {
            rid: runtime.registry.get(rid) for rid in plan["recipe_versions"]
        }
        self.start_source = next(
            state for state in inputs["states"] if state["hash"] == inputs["b2_hash"]
        )
        self.start = runtime.store.import_state(
            Path(self.start_source["dcp"]), design=design, fake=False
        )
        if self.start.id != inputs["b2_hash"]:
            raise ValueError("imported B2 differs from frozen selection")
        self.parents: dict[str, Any] = {}
        self.source_checks: list[tuple[Any, Path, str, Path]] = [
            (self.start, Path(self.start_source["evidence"]),
             self.start_source["evidence_sha256"], Path(self.start_source["dcp"]))
        ]
        self.state_info = {state["hash"]: state for state in inputs["states"]}
        if phase == "a":
            by_hash = {self.start.id: self.start}
            for source in inputs["states"]:
                if source["hash"] == self.start.id:
                    continue
                state = runtime.store.import_state(
                    Path(source["dcp"]), design=design, fake=False
                )
                if state.id != source["hash"]:
                    raise ValueError("imported D1 differs from frozen selection")
                by_hash[state.id] = state
                self.source_checks.append((
                    state, Path(source["evidence"]), source["evidence_sha256"],
                    Path(source["dcp"]),
                ))
            for task in self.tasks:
                self.parents[task["key"]] = by_hash[task["input_hash"]]
        else:
            a_rows, _ = a_rows_for_b(design, inputs)
            aroot = root_for("a", design)
            astore = StateStore(aroot)
            a_ledger = VerificationLedger(
                store=astore, runner=runtime.runner, validator=runtime.validator,
                references={design: self.start.id},
            )
            for power in POWER:
                row = a_rows[f"a:{self.start.id}:{power}"]
                amd = a_ledger.require_pass(design, row["output_hash"])
                if amd.get("result") != "pass" or outputs_compared(amd) <= 0:
                    raise ValueError(f"phase-a parent lost AMD PASS: {power}")
                outcome = astore.load_outcome(row["run_id"])
                if (
                    outcome.run.status != RunStatus.OK
                    or outcome.run.recipe_id != power
                    or outcome.run.input_sha256_before != self.start.id
                    or outcome.run.input_sha256_after != self.start.id
                    or outcome.state is None or outcome.evidence is None
                    or outcome.state.id != row["output_hash"]
                    or outcome.evidence.state_id != row["output_hash"]
                ):
                    raise ValueError(f"phase-a B2 power lineage changed: {power}")
                source_path = astore.state_path(row["output_hash"])
                state = runtime.store.import_state(
                    source_path, design=design, parent_id=self.start.id,
                    made_by_run_id=row["run_id"], fake=False,
                )
                if state.id != row["output_hash"]:
                    raise ValueError("phase-b imported power parent differs")
                evidence_path = aroot / "runs" / row["run_id"] / "evidence.json"
                self.source_checks.append((
                    state, evidence_path, _sha256_file(evidence_path), source_path,
                ))
                for task in self.tasks:
                    if task["power_recipe_id"] == power:
                        if (
                            task["input_hash"] != state.id
                            or task["source_run_id"] != row["run_id"]
                        ):
                            raise ValueError("phase-b task lineage changed")
                        self.parents[task["key"]] = state
        if set(self.parents) != set(self.task_by_key):
            raise ValueError("planned tasks lack parent states")
        self.ledger = VerificationLedger(
            store=runtime.store, runner=runtime.runner, validator=runtime.validator,
            references={design: self.start.id},
        )
        self.journal = Journal(self.root)
        self.journal_id = f"overnight_power_{phase}_{design}"
        self.rows: dict[str, dict[str, Any]] = {}
        self.signatures: dict[str, tuple[tuple[str, float], ...]] = {}
        self.stopped = False
        self.stop_reason: str | None = None
        self.stage = "preparing"
        self.load_journal()
        self.repair_aliases()
        self.write_index()

    def load_journal(self) -> None:
        events = self.journal.read(self.journal_id)
        plan_sha = _sha256_file(self.root / "plan.json")
        if not events:
            self.journal.append(self.journal_id, {
                "id": "configuration", "type": "configuration",
                "plan_sha256": plan_sha, "at_utc": utc(),
            })
            return
        if (
            events[0].get("type") != "configuration"
            or events[0].get("plan_sha256") != plan_sha
        ):
            raise ValueError("saved journal configuration differs")
        for event in events[1:]:
            if event.get("type") != "edge":
                raise ValueError("unknown journal event")
            row = event["edge"]
            key = row["task_key"]
            if key not in self.task_by_key or key in self.rows:
                raise ValueError("duplicate or unknown saved edge")
            task = self.task_by_key[key]
            if (
                row.get("input_hash") != task["input_hash"]
                or row.get("recipe_id") != task["recipe_id"]
                or row.get("power_recipe_id") != task["power_recipe_id"]
            ):
                raise ValueError("saved edge differs from frozen task")
            self.rows[key] = row
            if row["status"] != "ok":
                self.stopped = True
                self.stop_reason = row.get("error") or "saved failed edge"
                continue
            outcome = self.runtime.store.load_outcome(row["run_id"])
            if (
                outcome.run.status != RunStatus.OK
                or outcome.run.recipe_id != task["recipe_id"]
                or outcome.run.recipe_version != self.specs[task["recipe_id"]].version
                or outcome.run.input_sha256_before != task["input_hash"]
                or outcome.run.input_sha256_after != task["input_hash"]
                or outcome.state is None or outcome.evidence is None
                or outcome.state.id != row["output_hash"]
                or outcome.evidence.state_id != row["output_hash"]
                or row["cost"] != complete_cost(outcome.run)
            ):
                raise ValueError("saved successful edge differs from run")
            signature = check_evidence(
                outcome.evidence, row["output_hash"], self.plan, self.inputs
            )
            record = self.ledger.require_pass(self.design, row["output_hash"])
            if (
                row["amd_result"] != "pass"
                or row["amd_outputs_compared"] != outputs_compared(record)
                or row["amd_outputs_compared"] <= 0
            ):
                raise ValueError("saved edge lost its AMD output comparison")
            old = self.signatures.get(row["output_hash"])
            if old is not None and old != signature:
                raise ValueError("one saved DCP has different meter metrics")
            self.signatures[row["output_hash"]] = signature

    def alias_path(self, row: dict[str, Any]) -> Path | None:
        if not row.get("output_hash"):
            return None
        task = self.task_by_key[row["task_key"]]
        if self.phase_id == "a":
            source = self.state_info[task["input_hash"]]
            names = source["source_sequences"]
            label = "B2" if "B2" in names else sorted(names)[0]
            label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
            name = (
                f"{label}__then__{task['recipe_id']}"
                f"__{task['input_hash'][:12]}.dcp"
            )
        else:
            name = (
                f"{task['power_recipe_id']}__then__{task['recipe_id']}"
                f"__{task['input_hash'][:12]}.dcp"
            )
        folder = "successful" if row["status"] == "ok" else "failed"
        return self.root / "by_design" / self.design / folder / name

    def repair_aliases(self) -> None:
        for row in self.rows.values():
            path = self.alias_path(row)
            if row.get("dcp_alias") != (str(path) if path else None):
                raise ValueError("saved alias differs from naming rule")
            if path is not None:
                hardlink_alias(path, self.runtime.store.state_path(row["output_hash"]))

    def write_status(self) -> None:
        _atomic_bytes(self.root / "status.json", _json_bytes({
            "schema_version": "overnight-power-status-v1",
            "phase_id": self.phase_id, "design": self.design,
            "phase": self.stage, "updated_at_utc": utc(),
            "driver_pid": os.getpid(), "planned": len(self.tasks),
            "completed": len(self.rows),
            "amd_pass": sum(row["status"] == "ok" for row in self.rows.values()),
            "failed": sum(row["status"] != "ok" for row in self.rows.values()),
            "stopped": self.stopped, "error": self.stop_reason,
        }))

    def preflight(self) -> None:
        """Remeasure every exact parent under the current frozen meter."""
        self.stage = "input_remeasuring"
        self.write_status()
        checked = []
        for state, evidence_path, evidence_sha, source_path in self.source_checks:
            if _sha256_file(source_path) != state.id:
                raise ValueError(f"source DCP changed: {state.id}")
            if _sha256_file(evidence_path) != evidence_sha:
                raise ValueError(f"source Evidence changed: {state.id}")
            old = Evidence.model_validate_json(evidence_path.read_bytes())
            if (
                old.state_id != state.id or old.fake or old.routed is not True
                or old.route_errors != 0 or old.drc_errors != 0
                or old.clock_digest != self.inputs["b2_clock_digest"]
                or old.constraint_fingerprint != self.inputs["b2_constraint_fingerprint"]
                or not isinstance(old.v_props, dict)
                or any(type(old.v_props.get(key)) is not int or old.v_props[key] != 0
                       for key in PROPS)
            ):
                raise ValueError(f"source Evidence is invalid: {state.id}")
            fresh = self.runtime.engine.measure_start(state)
            signature = check_evidence(fresh, state.id, self.plan, self.inputs)
            if signature != metric_signature(old):
                raise ValueError(f"input meter differs from stored Evidence: {state.id}")
            previous = self.signatures.get(state.id)
            if previous is not None and previous != signature:
                raise ValueError("same source DCP has inconsistent metrics")
            self.signatures[state.id] = signature
            checked.append(state.id)
        _atomic_bytes(self.root / "input_remeasurement.json", _json_bytes({
            "schema_version": "overnight-power-input-check-v1",
            "phase_id": self.phase_id, "design": self.design,
            "meter_version": self.plan["meter_version"],
            "checked_state_hashes": checked, "checked_at_utc": utc(),
        }))

    def one(self, task: dict[str, str]) -> dict[str, Any]:
        state = self.parents[task["key"]]
        outcome = None
        amd = None
        compared = 0
        error = None
        try:
            check_pins(self.plan, self.runtime)
            outcome = self.runtime.engine.execute(state, self.specs[task["recipe_id"]])
            if outcome.state is not None:
                amd = self.ledger.verify(self.design, outcome.state.id)
                compared = outputs_compared(amd)
            if (
                outcome.run.status != RunStatus.OK
                or outcome.state is None or outcome.evidence is None
            ):
                raise ValueError(f"recipe or meter failed: {outcome.run.status.value}")
            if amd is None or amd.get("result") != "pass" or compared <= 0:
                raise ValueError(
                    "AMD check did not PASS with compared outputs: "
                    f"{amd.get('result') if amd else 'missing'}"
                )
            check_evidence(
                outcome.evidence, outcome.state.id, self.plan, self.inputs
            )
            complete_cost(outcome.run)
            if (
                outcome.run.input_sha256_before != state.id
                or outcome.run.input_sha256_after != state.id
            ):
                raise ValueError("recipe input DCP changed")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        run = outcome.run if outcome else None
        output = outcome.state if outcome else None
        return {
            "schema_version": "overnight-power-edge-v1",
            "phase_id": self.phase_id, "design": self.design,
            "task_key": task["key"], "input_hash": task["input_hash"],
            "recipe_id": task["recipe_id"],
            "power_recipe_id": task["power_recipe_id"],
            "source_run_id": task["source_run_id"],
            "status": "failed" if error else "ok", "error": error,
            "run_id": run.id if run else None,
            "run_status": run.status.value if run else None,
            "output_hash": output.id if output else None,
            "evidence_id": outcome.evidence.id if outcome and outcome.evidence else None,
            "amd_result": amd.get("result") if amd else None,
            "amd_outputs_compared": compared,
            "amd_record": (
                str(self.ledger._record_path(self.design, output.id))
                if amd and output else None
            ),
            "cost": raw_cost(run) if run else None,
            "checked_at_utc": utc(),
        }

    def commit(self, row: dict[str, Any]) -> None:
        key = row["task_key"]
        if key in self.rows:
            raise ValueError("edge already committed")
        if row["status"] == "ok":
            outcome = self.runtime.store.load_outcome(row["run_id"])
            signature = check_evidence(
                outcome.evidence, row["output_hash"], self.plan, self.inputs
            )
            old = self.signatures.get(row["output_hash"])
            if old is not None and old != signature:
                row["status"] = "failed"
                row["error"] = "same DCP hash has mismatched independent meter metrics"
            else:
                self.signatures[row["output_hash"]] = signature
        path = self.alias_path(row)
        row["dcp_alias"] = str(path) if path else None
        event_id = hashlib.sha256(_json_bytes(key)).hexdigest()
        self.journal.append(self.journal_id, {
            "id": f"edge-{event_id}", "type": "edge",
            "edge": row, "at_utc": utc(),
        })
        self.rows[key] = row
        if path is not None:
            hardlink_alias(path, self.runtime.store.state_path(row["output_hash"]))
        if row["status"] != "ok":
            self.stopped = True
            self.stop_reason = row.get("error") or "edge check failed"
        self.write_index()
        self.write_status()

    def write_index(self) -> None:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow((
            "design", "phase", "task_key", "parent_sequence", "input_hash",
            "recipe_id", "power_recipe_id", "source_run_id", "status", "run_id",
            "output_hash", "dcp_alias", "canonical_dcp", "amd_result",
            "amd_outputs_compared", "cache_hit", "original_run_id",
            "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
            "load_1m_start", "load_1m_end", "busy_slots_start",
            "backpressure_wait_s", "backpressure_events",
        ))
        for task in self.tasks:
            row = self.rows.get(task["key"])
            if row is None:
                continue
            output_hash = row.get("output_hash")
            cost = row.get("cost") or {}
            parent = (
                "+".join(self.state_info[task["input_hash"]]["source_sequences"])
                if self.phase_id == "a" else task["power_recipe_id"]
            )
            writer.writerow((
                self.design, self.phase_id, task["key"], parent,
                task["input_hash"], task["recipe_id"], task["power_recipe_id"],
                task["source_run_id"], row["status"], row.get("run_id") or "",
                output_hash or "", row.get("dcp_alias") or "",
                str(self.root / "states" / output_hash / "design.dcp") if output_hash else "",
                row.get("amd_result") or "", row.get("amd_outputs_compared") or "",
                cost.get("cache_hit", ""), cost.get("original_run_id") or "",
                cost.get("wall_s", ""), cost.get("cpu_user_s", ""),
                cost.get("cpu_sys_s", ""), cost.get("peak_memory_bytes", ""),
                cost.get("load_1m_start", ""), cost.get("load_1m_end", ""),
                cost.get("busy_slots_start", ""), cost.get("backpressure_wait_s", ""),
                cost.get("backpressure_events", ""),
            ))
        _atomic_bytes(
            self.root / "by_design/index.csv",
            output.getvalue().encode("utf-8"),
        )

    def run(self) -> None:
        if self.stopped:
            self.stage = "stopped"
            self.write_status()
            return
        try:
            self.preflight()
        except Exception as exc:
            self.stopped = True
            self.stop_reason = f"input preflight: {type(exc).__name__}: {exc}"
            self.stage = "preflight_failed"
            self.write_status()
            return
        self.stage = "power_one_step" if self.phase_id == "a" else "worker_after_power"
        self.write_status()
        pending = deque(task for task in self.tasks if task["key"] not in self.rows)
        with ThreadPoolExecutor(max_workers=MAX_ACTIVE) as pool:
            active: dict[Any, dict[str, str]] = {}

            def fill() -> None:
                while pending and not self.stopped and len(active) < MAX_ACTIVE:
                    try:
                        check_pins(self.plan, self.runtime)
                    except Exception as exc:
                        self.stopped = True
                        self.stop_reason = f"source pin: {type(exc).__name__}: {exc}"
                        self.write_status()
                        break
                    task = pending.popleft()
                    active[pool.submit(self.one, task)] = task

            fill()
            while active:
                finished, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in finished:
                    task = active.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "schema_version": "overnight-power-edge-v1",
                            "phase_id": self.phase_id, "design": self.design,
                            "task_key": task["key"],
                            "input_hash": task["input_hash"],
                            "recipe_id": task["recipe_id"],
                            "power_recipe_id": task["power_recipe_id"],
                            "source_run_id": task["source_run_id"],
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "run_id": None, "run_status": None,
                            "output_hash": None, "evidence_id": None,
                            "amd_result": None, "amd_outputs_compared": 0,
                            "amd_record": None, "cost": None,
                            "checked_at_utc": utc(),
                        }
                    self.commit(row)
                fill()
        self.stage = (
            "complete"
            if len(self.rows) == len(self.tasks) and not self.stopped
            else "stopped"
        )
        self.write_status()
        _atomic_bytes(self.root / "summary.json", _json_bytes({
            "schema_version": "overnight-power-summary-v1",
            "phase_id": self.phase_id, "design": self.design,
            "phase": self.stage, "planned": len(self.tasks),
            "attempted": len(self.rows),
            "amd_pass": sum(row["status"] == "ok" for row in self.rows.values()),
            "failed": sum(row["status"] != "ok" for row in self.rows.values()),
            "plan_sha256": _sha256_file(self.root / "plan.json"),
            "index_sha256": _sha256_file(self.root / "by_design/index.csv"),
            "completed_at_utc": utc(),
        }))


def prepare(phase: str, design: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    inputs = selected()["by_design"][design]
    runtime = make_real_runtime(
        root_for(phase, design), cache_enabled=True, max_slots=MAX_SLOTS,
        load_ceiling_1m=LOAD_CEILING,
        min_available_memory_bytes=MEMORY_FLOOR,
        latch_smoke_status=SMOKE,
    )
    return runtime, frozen_plan(runtime, phase, design, inputs), inputs


def run(phase: str, design: str) -> None:
    runtime, plan, inputs = prepare(phase, design)
    root = root_for(phase, design)
    plan_path = root / "plan.json"
    if not plan_path.is_file() or plan_path.read_bytes() != _json_bytes(plan):
        raise ValueError("per-design phase plan changed after launch")
    with _file_lock(root / "driver.lock"):
        study = Study(phase, design, runtime, plan, inputs)
        try:
            study.run()
        except Exception as exc:
            study.stage = "failed"
            study.stop_reason = f"{type(exc).__name__}: {exc}"
            study.stopped = True
            study.write_status()
            raise


def driver_alive(phase: str, design: str, pid: int) -> bool:
    try:
        info = _stat(pid)
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return bool(
            info and info.state != "Z"
            and str(SCRIPT).encode() in args
            and b"--run" in args and b"--phase" in args
            and phase.encode() in args and design.encode() in args
        )
    except OSError:
        return False


def targets(phase: str, design: str) -> tuple[set[str], set[int]]:
    """Find only exact tagged descendants working in this phase/design root."""
    root = root_for(phase, design)
    prefix = str(root).encode()
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


def cleanup(phase: str, design: str) -> dict[str, Any]:
    """Terminate only process groups carrying this root's exact run tags."""
    root = root_for(phase, design)
    tags, groups = targets(phase, design)
    result: dict[str, Any] = {
        "schema_version": "overnight-power-cleanup-v1",
        "phase_id": phase, "design": design,
        "exact_tags": sorted(tags), "groups": sorted(groups),
        "started_at_utc": utc(),
    }
    _atomic_bytes(root / "watchdog_cleanup_audit.json", _json_bytes(result))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in sorted(groups):
            if any(_tag_of(member.pid) in tags for member in group_members(group)):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    result["tag_sweeps"] = {
        tag: list(sweep_tag(tag, grace_s=5).alive) for tag in sorted(tags)
    }
    result["survivors"] = {
        tag: [process.pid for process in tagged(tag)]
        for tag in sorted(tags) if tagged(tag)
    }
    result["finished_at_utc"] = utc()
    _atomic_bytes(root / "watchdog_cleanup_audit.json", _json_bytes(result))
    return result


def watch_phase(phase: str, design: str) -> None:
    root = root_for(phase, design)
    pid = read(root / "launch.json")["driver_pid"]
    if not driver_alive(phase, design, pid):
        raise RuntimeError("driver absent at watchdog start")
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        if not driver_alive(phase, design, pid):
            state = read(root / "status.json") if (root / "status.json").is_file() else {}
            live_tags, _groups = targets(phase, design)
            if state.get("phase") not in TERMINAL or live_tags:
                cleanup(phase, design)
            return
        time.sleep(15)
    raise RuntimeError("watchdog lifetime ended with driver alive")


def launch(phase: str, design: str) -> dict[str, Any]:
    runtime, plan, _inputs = prepare(phase, design)
    check_pins(plan, runtime)
    root = root_for(phase, design)
    plan_path = root / "plan.json"
    manifest_path = root / "launch.json"
    load, available = Runner._resource_snapshot()
    if load > LOAD_CEILING or available < MEMORY_FLOOR:
        raise RuntimeError("resource gate blocks launch")
    if manifest_path.exists():
        raise FileExistsError("phase/design already launched")
    if plan_path.exists():
        if plan_path.read_bytes() != _json_bytes(plan):
            raise ValueError("existing phase plan differs")
    else:
        _atomic_bytes(plan_path, _json_bytes(plan))
    with _file_lock(root / "launch.lock"):
        if manifest_path.exists():
            raise FileExistsError("phase/design already launched")
        with (root / "driver.stdout").open("wb") as out, (root / "driver.stderr").open("wb") as err:
            driver = subprocess.Popen(
                [
                    sys.executable, str(SCRIPT), "--phase", phase,
                    "--design", design, "--run",
                ],
                cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        result = {
            "schema_version": "overnight-power-launch-v1",
            "phase_id": phase, "design": design,
            "driver_pid": driver.pid, "started_at_utc": utc(),
            "plan_sha256": _sha256_file(plan_path),
            "planned": len(plan["tasks"]), "max_active": MAX_ACTIVE,
            "load_1m_at_launch": load,
            "available_memory_bytes_at_launch": available,
        }
        _atomic_bytes(manifest_path, _json_bytes(result))
        try:
            with (root / "watchdog.stdout").open("wb") as out, (root / "watchdog.stderr").open("wb") as err:
                watcher = subprocess.Popen(
                    [
                        sys.executable, str(SCRIPT), "--phase", phase,
                        "--design", design, "--watch",
                    ],
                    cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    start_new_session=True, close_fds=True,
                )
            result["watchdog_pid"] = watcher.pid
            _atomic_bytes(manifest_path, _json_bytes(result))
            return result
        except Exception:
            if driver_alive(phase, design, driver.pid):
                group = os.getpgid(driver.pid)
                if group == driver.pid and group != os.getpgrp():
                    os.killpg(group, signal.SIGTERM)
            time.sleep(3)
            cleanup(phase, design)
            raise


def phase_status(phase: str, design: str) -> dict[str, Any]:
    root = root_for(phase, design)
    manifest = read(root / "launch.json") if (root / "launch.json").is_file() else {}
    state = read(root / "status.json") if (root / "status.json").is_file() else {}
    pid = manifest.get("driver_pid")
    return {
        "phase": state.get("phase", "not_launched"),
        "planned": state.get("planned", manifest.get("planned", 0)),
        "completed": state.get("completed", 0),
        "amd_pass": state.get("amd_pass", 0),
        "failed": state.get("failed", 0),
        "error": state.get("error"),
        "driver_pid": pid,
        "driver_alive": driver_alive(phase, design, pid) if type(pid) is int else False,
        "driver_stderr_bytes": (
            (root / "driver.stderr").stat().st_size if manifest else 0
        ),
    }


def write_controller_status(stage: str, error: str | None = None) -> None:
    _atomic_bytes(TOP / "controller_status.json", _json_bytes({
        "schema_version": "overnight-power-controller-status-v1",
        "phase": stage, "error": error, "updated_at_utc": utc(),
        "controller_pid": os.getpid(),
        "phase_a": {name: phase_status("a", name) for name in DESIGNS},
        "phase_b": {name: phase_status("b", name) for name in DESIGNS},
    }))


def phase_a_barrier() -> dict[str, Any]:
    """Seal a global terminal barrier before any phase-b plan can exist."""
    statuses = {name: phase_status("a", name) for name in DESIGNS}
    if any(row["phase"] not in TERMINAL or row["driver_alive"] for row in statuses.values()):
        raise ValueError("phase-a queues have not all reached a terminal state")
    record = {
        "schema_version": "overnight-power-phase-a-barrier-v1",
        "selection_sha256": SELECTION_SHA,
        "statuses": {name: row["phase"] for name, row in statuses.items()},
        "status_sha256": {
            name: _sha256_file(root_for("a", name) / "status.json")
            for name in DESIGNS
        },
        "sealed_at_utc": utc(),
    }
    path = TOP / "phase_a_barrier.json"
    if path.exists():
        old = read(path)
        if any(old.get(key) != record[key] for key in (
            "schema_version", "selection_sha256", "statuses", "status_sha256"
        )):
            raise ValueError("existing global phase-a barrier differs")
        return old
    _atomic_bytes(path, _json_bytes(record))
    return record


def coordinate() -> None:
    """Launch phase a, wait for its global barrier, then launch phase b."""
    for _ in range(100):
        if (TOP / "controller_launch.json").is_file():
            break
        time.sleep(0.1)
    manifest = read(TOP / "controller_launch.json")
    if (
        manifest.get("selection_sha256") != SELECTION_SHA
        or manifest.get("driver_sha256") != _sha256_file(SCRIPT)
    ):
        raise ValueError("controller launch fingerprint changed")
    deadline = time.monotonic() + 24 * 3600
    try:
        write_controller_status("launching_phase_a")
        for design in DESIGNS:
            while not (root_for("a", design) / "launch.json").is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("phase-a launch deadline")
                try:
                    launch("a", design)
                except RuntimeError as exc:
                    if str(exc) != "resource gate blocks launch":
                        raise
                    time.sleep(20)
            write_controller_status("launching_phase_a")
        write_controller_status("waiting_for_phase_a")
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("phase-a terminal barrier deadline")
            statuses = {name: phase_status("a", name) for name in DESIGNS}
            if all(row["phase"] in TERMINAL and not row["driver_alive"] for row in statuses.values()):
                break
            if any(
                row["phase"] not in TERMINAL
                and not row["driver_alive"]
                and (root_for("a", name) / "launch.json").is_file()
                and (datetime.now(timezone.utc).timestamp()
                     - datetime.fromisoformat(read(root_for("a", name) / "launch.json")["started_at_utc"]).timestamp()) > 120
                for name, row in statuses.items()
            ):
                raise RuntimeError("phase-a driver exited without terminal status")
            write_controller_status("waiting_for_phase_a")
            time.sleep(30)
        barrier = phase_a_barrier()
        write_controller_status("phase_a_barrier_sealed")
        for design in DESIGNS:
            if barrier["statuses"][design] != "complete":
                continue
            while not (root_for("b", design) / "launch.json").is_file():
                if time.monotonic() >= deadline:
                    raise TimeoutError("phase-b launch deadline")
                try:
                    launch("b", design)
                except RuntimeError as exc:
                    if str(exc) != "resource gate blocks launch":
                        raise
                    time.sleep(20)
            write_controller_status("launching_phase_b")
        write_controller_status("waiting_for_phase_b")
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("phase-b terminal deadline")
            statuses = {
                name: phase_status("b", name)
                for name in DESIGNS if barrier["statuses"][name] == "complete"
            }
            if all(row["phase"] in TERMINAL and not row["driver_alive"] for row in statuses.values()):
                break
            if any(
                row["phase"] not in TERMINAL
                and not row["driver_alive"]
                and (datetime.now(timezone.utc).timestamp()
                     - datetime.fromisoformat(read(root_for("b", name) / "launch.json")["started_at_utc"]).timestamp()) > 120
                for name, row in statuses.items()
            ):
                raise RuntimeError("phase-b driver exited without terminal status")
            write_controller_status("waiting_for_phase_b")
            time.sleep(30)
        final = "complete" if all(
            phase_status("a", name)["phase"] == "complete"
            and phase_status("b", name)["phase"] == "complete"
            for name in DESIGNS
        ) else "partial"
        write_controller_status(final)
    except Exception as exc:
        write_controller_status("failed", f"{type(exc).__name__}: {exc}")
        raise


def launch_campaign() -> dict[str, Any]:
    selected()
    manifest_path = TOP / "controller_launch.json"
    if manifest_path.exists():
        raise FileExistsError("power campaign controller already launched")
    load, available = Runner._resource_snapshot()
    if load > LOAD_CEILING or available < MEMORY_FLOOR:
        raise RuntimeError("resource gate blocks campaign launch")
    with _file_lock(TOP / "controller_launch.lock"):
        if manifest_path.exists():
            raise FileExistsError("power campaign controller already launched")
        with (TOP / "controller.stdout").open("wb") as out, (TOP / "controller.stderr").open("wb") as err:
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--coordinate"],
                cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        record = {
            "schema_version": "overnight-power-controller-launch-v1",
            "controller_pid": process.pid, "started_at_utc": utc(),
            "driver_sha256": _sha256_file(SCRIPT),
            "selection_sha256": SELECTION_SHA,
            "phase_a_planned": 176, "phase_b_planned": 200,
            "shared_slots": MAX_SLOTS,
            "max_active_per_design": MAX_ACTIVE,
            "load_1m_at_launch": load,
            "available_memory_bytes_at_launch": available,
        }
        _atomic_bytes(manifest_path, _json_bytes(record))
        return record


def status_all() -> dict[str, Any]:
    load, available = Runner._resource_snapshot()
    controller = (
        read(TOP / "controller_status.json")
        if (TOP / "controller_status.json").is_file()
        else {"phase": "not_launched"}
    )
    return {
        "controller_phase": controller["phase"],
        "controller_error": controller.get("error"),
        "phase_a": {name: phase_status("a", name) for name in DESIGNS},
        "phase_b": {name: phase_status("b", name) for name in DESIGNS},
        "load_1m": load, "available_memory_bytes": available,
        "busy_shared_slots": Runner(
            PROJECT / "data/slots", max_slots=MAX_SLOTS
        )._busy_slots(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"))
    parser.add_argument("--design", choices=DESIGNS)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--watch", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--coordinate", action="store_true")
    modes.add_argument("--launch-campaign", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        item = selected()
        print(json.dumps({
            "training_designs": list(DESIGNS),
            "distinct_inputs": sum(len(item["by_design"][name]["states"]) for name in DESIGNS),
            "phase_a_planned": sum(
                len(a_tasks(item["by_design"][name])) for name in DESIGNS
            ),
            "phase_b_planned": len(DESIGNS) * len(POWER) * len(WORKER),
            "tool_jobs_launched": 0,
        }, sort_keys=True))
    elif args.launch_campaign:
        print(json.dumps(launch_campaign(), sort_keys=True))
    elif args.coordinate:
        coordinate()
    elif args.status:
        print(json.dumps(
            phase_status(args.phase, args.design)
            if args.phase and args.design else status_all(),
            sort_keys=True,
        ))
    else:
        if not args.phase or not args.design:
            parser.error("--phase and --design are required for --launch, --run and --watch")
        if args.launch:
            print(json.dumps(launch(args.phase, args.design), sort_keys=True))
        elif args.run:
            run(args.phase, args.design)
        elif args.watch:
            watch_phase(args.phase, args.design)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
