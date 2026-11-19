"""Detached, per-design, AMD-checked cross-worker depth-three training campaign.

Every selected parent and B2 is remeasured under the current meter before a
recipe starts. Each design has its own queue, journal, store, cache and watchdog,
so one failed design does not stop the other four. Tool commands remain in the
existing runner, meter, adapters and validator. No Tcl is authored here.
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
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from orch.recipes.registry import load_registry
from orch.runtime import make_real_runtime

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
TOP = PROJECT / "data/sweeps/thursday_crossworker_depth3_2026-11-05"
SELECTION = TOP / "selected_states.json"
SELECTION_SHA = "c8a10c677abdb3dd881cc3ad70810e8cf08d5a72c83852c3c816f1c931b6dbae"
SMOKE = PROJECT / "data/scratch_thursday_2026-11-05/cache_gate_smoke/status.json"
DESIGNS = (
    "vexriscv_re-place", "rosetta_spam-filter", "rosetta_3d-rendering",
    "logicnets_jscl", "finn_radioml",
)
TIMING = (
    "LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock",
    "LATCH.hardblock_relocate", "LATCH.memory_path_opt",
)
AREA = ("A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED", "A_E_DUPL_Q1000")
RECIPES = TIMING + AREA
METRICS = tuple(MetricName)
MAX_ACTIVE = 12
MAX_SLOTS = 96
LOAD_CEILING = 180.0
MEMORY_FLOOR = 200_000_000_000
SOURCE_FILES = {
    "driver": "scripts/thursday_crossworker_depth3.py",
    "runner": "orch/engine/runner.py",
    "runtime": "orch/runtime.py",
    "real_backend": "orch/engine/real_backend.py",
    "apply": "orch/engine/apply.py",
    "meter": "orch/engine/meter.py",
    "meter_parsers": "orch/engine/meter_parsers.py",
    "models": "orch/contract/models.py",
    "jobtag": "orch/engine/jobtag.py",
    "verification": "orch/baselines/verification.py",
    "validator": "orch/recipes/adapters/validator.py",
    "registry": "recipes/registry.yaml",
    "recipe_sets": "configs/recipe_sets.yaml",
    "env": "recipes/support/env.sh",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def selection() -> dict[str, Any]:
    if _sha256_file(SELECTION) != SELECTION_SHA:
        raise ValueError("frozen selected-state list changed")
    item = read(SELECTION)
    if (item.get("schema_version") != "thursday-crossworker-selection-v1"
            or tuple(item.get("training_designs", ())) != DESIGNS
            or tuple(item.get("selected_recipe_ids", ())) != RECIPES):
        raise ValueError("selected-state scope differs")
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    if (tuple(sets.worker_sets.get("timing", ())) != TIMING
            or tuple(sets.worker_sets.get("area", ())) != AREA):
        raise ValueError("current signed worker recipes differ")
    return item


def paths(design: str) -> tuple[Path, Path, Path, Path]:
    if design not in DESIGNS:
        raise ValueError(f"not a training design: {design}")
    root = TOP / design
    return root, root / "plan.json", root / "launch.json", root / "status.json"


def tasks(inputs: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    hashes = [row["hash"] for row in inputs["selected_states"]]
    if len(hashes) != len(set(hashes)) or len(hashes) * len(RECIPES) > 200:
        raise ValueError("duplicate selected parent or cap exceeded")
    return tuple((parent_hash, recipe_id) for recipe_id in RECIPES
                 for parent_hash in sorted(hashes))


def source_hashes() -> dict[str, str]:
    return {name: _sha256_file(PROJECT / rel) for name, rel in SOURCE_FILES.items()}


def plan(runtime: Any, design: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "thursday-crossworker-depth3-plan-v1",
        "design": design,
        "selection_sha256": SELECTION_SHA,
        "parent_hashes": sorted(row["hash"] for row in inputs["selected_states"]),
        "b2_hash": inputs["start"]["hash"],
        "recipe_versions": {
            rid: runtime.registry.get(rid).version for rid in RECIPES
        },
        "meter_version": runtime.meter.version,
        "smoke_sha256": _sha256_file(SMOKE),
        "source_hashes": source_hashes(),
        "planned_edges": len(tasks(inputs)),
        "max_active": MAX_ACTIVE, "shared_slots": MAX_SLOTS,
        "load_ceiling_1m": LOAD_CEILING,
        "available_memory_floor_bytes": MEMORY_FLOOR,
    }


def check_pins(frozen: dict[str, Any], runtime: Any) -> None:
    """Stop new jobs if code, selected recipes or the frozen meter changed."""
    if (_sha256_file(SELECTION) != frozen["selection_sha256"]
            or _sha256_file(SMOKE) != frozen["smoke_sha256"]
            or source_hashes() != frozen["source_hashes"]
            or runtime.meter.version != frozen["meter_version"]):
        raise ValueError("frozen source, selection, smoke or meter changed")
    current = load_registry(PROJECT)
    for rid, version in frozen["recipe_versions"].items():
        if current.get(rid).version != version:
            raise ValueError(f"selected recipe version changed: {rid}")


def raw_cost(run: Any) -> dict[str, Any]:
    return {key: getattr(run, key) for key in (
        "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
        "load_1m_start", "load_1m_end", "busy_slots_start",
        "backpressure_wait_s", "backpressure_events", "cache_hit",
        "original_run_id",
    )}


def complete_cost(run: Any) -> dict[str, Any]:
    fields = raw_cost(run)
    for key in ("wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
                "load_1m_start", "load_1m_end", "busy_slots_start"):
        value = fields[key]
        if value is None or not math.isfinite(value):
            raise ValueError(f"run lacks finite {key}")
    if run.cache_hit and not run.original_run_id:
        raise ValueError("cache hit lacks original run ID")
    return fields


def outputs_compared(record: dict[str, Any]) -> int:
    attempts = record.get("attempts") or []
    if not attempts:
        return 0
    log_path = Path(attempts[-1]["run_dir"]) / "validate.log"
    content = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return max((int(x) for x in re.findall(r"Outputs\s*\((\d+)\):", content)), default=0)


def metric_signature(evidence: Evidence) -> tuple[tuple[str, float], ...]:
    values = []
    for name in METRICS:
        item = evidence.metric(name)
        if item is None or not math.isfinite(item.value):
            raise ValueError(f"Evidence lacks finite {name.value}")
        values.append((name.value, item.value))
    return tuple(values)


def alias(link: Path, target: Path) -> None:
    """Publish a readable DCP hard link only after an edge is journaled."""
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
    """Run one design; stop only its pending queue on any failed check."""

    def __init__(self, design: str, runtime: Any, frozen: dict[str, Any],
                 inputs: dict[str, Any]) -> None:
        self.design = design
        self.root, self.plan_path, _, self.status_path = paths(design)
        self.runtime = runtime
        self.frozen = frozen
        self.inputs = inputs
        self.specs = {rid: runtime.registry.get(rid) for rid in RECIPES}
        self.tasks = tasks(inputs)
        self.journal = Journal(self.root)
        self.journal_id = f"thursday_crossworker_{design}"
        self.start = runtime.store.import_state(Path(inputs["start"]["dcp"]), design=design, fake=False)
        if self.start.id != inputs["start"]["hash"]:
            raise ValueError("imported B2 differs")
        self.parents: dict[str, Any] = {}
        self.parent_sources: dict[str, dict[str, Any]] = {}
        for source in inputs["selected_states"]:
            state = runtime.store.import_state(Path(source["dcp"]), design=design, fake=False)
            if state.id != source["hash"]:
                raise ValueError("imported selected parent differs")
            self.parents[state.id] = state
            self.parent_sources[state.id] = source
        self.ledger = VerificationLedger(
            store=runtime.store, runner=runtime.runner, validator=runtime.validator,
            references={design: self.start.id},
        )
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.signatures: dict[str, tuple[tuple[str, float], ...]] = {}
        self.stopped = False
        self.stop_reason: str | None = None
        self.phase = "preparing"
        self.load_journal()
        self.repair_aliases()
        self.write_index()

    def load_journal(self) -> None:
        events = self.journal.read(self.journal_id)
        if not events:
            self.journal.append(self.journal_id, {
                "id": "configuration", "type": "configuration",
                "plan_sha256": _sha256_file(self.plan_path), "at_utc": utc(),
            })
            return
        if events[0].get("type") != "configuration" or events[0].get("plan_sha256") != _sha256_file(self.plan_path):
            raise ValueError("journal configuration differs from frozen plan")
        for event in events[1:]:
            if event.get("type") != "edge":
                raise ValueError("unknown journal event")
            row = event["edge"]
            key = (row["input_hash"], row["recipe_id"])
            if key not in self.tasks or key in self.rows:
                raise ValueError("duplicate or unknown saved edge")
            self.rows[key] = row
            if row["status"] != "ok":
                self.stopped = True
                self.stop_reason = row.get("error") or "saved failure"
                continue
            outcome = self.runtime.store.load_outcome(row["run_id"])
            if (outcome.run.status != RunStatus.OK
                    or outcome.run.recipe_id != key[1]
                    or outcome.run.recipe_version != self.specs[key[1]].version
                    or outcome.run.input_sha256_before != key[0]
                    or outcome.run.input_sha256_after != key[0]
                    or outcome.state is None or outcome.evidence is None
                    or outcome.state.id != row["output_hash"]
                    or outcome.evidence.state_id != outcome.state.id
                    or outcome.evidence.meter_version != self.frozen["meter_version"]
                    or row["cost"] != complete_cost(outcome.run)):
                raise ValueError("saved successful edge differs from run")
            record = self.ledger.require_pass(self.design, outcome.state.id)
            if row["amd_outputs_compared"] != outputs_compared(record) or row["amd_outputs_compared"] <= 0:
                raise ValueError("saved successful edge lost AMD output comparison")
            signature = metric_signature(outcome.evidence)
            old = self.signatures.get(outcome.state.id)
            if old is not None and old != signature:
                raise ValueError("same saved DCP has different meter metrics")
            self.signatures[outcome.state.id] = signature

    def alias_path(self, row: dict[str, Any]) -> Path | None:
        output = row.get("output_hash")
        if not output:
            return None
        parent = self.parent_sources[row["input_hash"]]
        lineage = parent["source_sequence"].replace(" > ", "__")
        name = f"{lineage}__{row['recipe_id']}__{row['input_hash'][:12]}.dcp"
        folder = "depth3" if row["status"] == "ok" else "failed"
        return self.root / "by_design" / self.design / folder / name

    def repair_aliases(self) -> None:
        for row in self.rows.values():
            path = self.alias_path(row)
            if row.get("dcp_alias") != (str(path) if path else None):
                raise ValueError("saved alias differs from naming rule")
            if path is not None:
                alias(path, self.runtime.store.state_path(row["output_hash"]))

    def write_status(self) -> None:
        _atomic_bytes(self.status_path, _json_bytes({
            "schema_version": "thursday-crossworker-depth3-status-v1",
            "design": self.design, "phase": self.phase,
            "updated_at_utc": utc(), "driver_pid": os.getpid(),
            "planned": len(self.tasks), "completed": len(self.rows),
            "amd_pass": sum(row["status"] == "ok" for row in self.rows.values()),
            "failed": sum(row["status"] != "ok" for row in self.rows.values()),
            "stopped": self.stopped, "error": self.stop_reason,
        }))

    def preflight(self) -> None:
        """Remeasure B2 and every selected input under the new meter version."""
        self.phase = "input_remeasuring"
        self.write_status()
        all_sources = [(self.start, self.inputs["start"])] + [
            (self.parents[source["hash"]], source) for source in self.inputs["selected_states"]
        ]
        checked = []
        baseline_clock = self.inputs["start"]["clock_digest"]
        baseline_constraint = self.inputs["start"]["constraint_fingerprint"]
        for state, source in all_sources:
            if _sha256_file(Path(source["dcp"])) != state.id:
                raise ValueError(f"input DCP changed: {state.id}")
            if _sha256_file(Path(source["evidence"])) != source["evidence_sha256"]:
                raise ValueError(f"input Evidence changed: {state.id}")
            old = Evidence.model_validate_json(Path(source["evidence"]).read_bytes())
            fresh = self.runtime.engine.measure_start(state)
            if (fresh.state_id != state.id or fresh.meter_version != self.frozen["meter_version"]
                    or fresh.fake or fresh.routed is not True or fresh.route_errors != 0
                    or fresh.drc_errors != 0 or fresh.clock_digest != baseline_clock
                    or fresh.constraint_fingerprint != baseline_constraint
                    or metric_signature(fresh) != metric_signature(old)):
                raise ValueError(f"input meter or digest mismatch: {state.id}")
            self.signatures[state.id] = metric_signature(fresh)
            checked.append(state.id)
        _atomic_bytes(self.root / "input_remeasurement.json", _json_bytes({
            "schema_version": "thursday-crossworker-input-check-v1", "design": self.design,
            "meter_version": self.frozen["meter_version"], "checked_state_hashes": checked,
            "checked_count": len(checked), "checked_at_utc": utc(),
        }))

    def one(self, parent_hash: str, recipe_id: str) -> dict[str, Any]:
        state = self.parents[parent_hash]
        outcome = None
        amd = None
        compared = 0
        error = None
        try:
            check_pins(self.frozen, self.runtime)
            outcome = self.runtime.engine.execute(state, self.specs[recipe_id])
            if outcome.state is not None:
                amd = self.ledger.verify(self.design, outcome.state.id)
                compared = outputs_compared(amd)
            if outcome.run.status != RunStatus.OK or outcome.state is None or outcome.evidence is None:
                raise ValueError(f"recipe or meter failed: {outcome.run.status.value}")
            if amd is None or amd.get("result") != "pass" or compared <= 0:
                raise ValueError(f"AMD check did not PASS with compared outputs: {amd.get('result') if amd else 'missing'}")
            evidence = outcome.evidence
            if (evidence.state_id != outcome.state.id or evidence.meter_version != self.frozen["meter_version"]
                    or evidence.fake or evidence.routed is not True or evidence.route_errors != 0
                    or evidence.drc_errors != 0 or evidence.clock_digest != self.inputs["start"]["clock_digest"]
                    or evidence.constraint_fingerprint != self.inputs["start"]["constraint_fingerprint"]):
                raise ValueError("output meter or B2 digest mismatch")
            metric_signature(evidence)
            complete_cost(outcome.run)
            if outcome.run.input_sha256_before != state.id or outcome.run.input_sha256_after != state.id:
                raise ValueError("recipe input DCP changed")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        run = outcome.run if outcome else None
        output = outcome.state if outcome else None
        return {
            "schema_version": "thursday-crossworker-depth3-edge-v1",
            "design": self.design, "input_hash": parent_hash, "recipe_id": recipe_id,
            "status": "failed" if error else "ok", "error": error,
            "run_id": run.id if run else None, "run_status": run.status.value if run else None,
            "output_hash": output.id if output else None,
            "evidence_id": outcome.evidence.id if outcome and outcome.evidence else None,
            "amd_result": amd.get("result") if amd else None,
            "amd_outputs_compared": compared,
            "amd_record": str(self.ledger._record_path(self.design, output.id)) if amd and output else None,
            "cost": raw_cost(run) if run else None, "checked_at_utc": utc(),
        }

    def commit(self, row: dict[str, Any]) -> None:
        key = (row["input_hash"], row["recipe_id"])
        if key in self.rows:
            raise ValueError("edge already committed")
        if row["status"] == "ok":
            outcome = self.runtime.store.load_outcome(row["run_id"])
            signature = metric_signature(outcome.evidence)
            old = self.signatures.get(row["output_hash"])
            if old is not None and old != signature:
                row["status"] = "failed"
                row["error"] = "same DCP hash has mismatched independent meter metrics"
            else:
                self.signatures[row["output_hash"]] = signature
        path = self.alias_path(row)
        row["dcp_alias"] = str(path) if path else None
        event_id = hashlib.sha256(_json_bytes(list(key))).hexdigest()
        self.journal.append(self.journal_id, {
            "id": f"edge-{event_id}", "type": "edge", "edge": row, "at_utc": utc(),
        })
        self.rows[key] = row
        if path is not None:
            alias(path, self.runtime.store.state_path(row["output_hash"]))
        if row["status"] != "ok":
            self.stopped = True
            self.stop_reason = row.get("error") or "edge check failed"
        self.write_index()
        self.write_status()

    def write_index(self) -> None:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow((
            "design", "depth", "parent_source_depth", "parent_sequence",
            "parent_hash", "recipe_id", "status", "run_id", "output_hash",
            "dcp_alias", "canonical_dcp", "amd_result", "amd_outputs_compared",
            "cache_hit", "original_run_id", "wall_s", "cpu_user_s", "cpu_sys_s",
        ))
        for parent_hash, recipe_id in self.tasks:
            row = self.rows.get((parent_hash, recipe_id))
            if row is None:
                continue
            parent = self.parent_sources[parent_hash]
            output_hash = row.get("output_hash")
            cost = row.get("cost") or {}
            writer.writerow((
                self.design, parent["source_depth"] + 1, parent["source_depth"],
                parent["source_sequence"], parent_hash, recipe_id, row["status"],
                row.get("run_id") or "", output_hash or "", row.get("dcp_alias") or "",
                str(self.root / "states" / output_hash / "design.dcp") if output_hash else "",
                row.get("amd_result") or "", row.get("amd_outputs_compared") or "",
                cost.get("cache_hit", ""), cost.get("original_run_id") or "",
                cost.get("wall_s", ""), cost.get("cpu_user_s", ""), cost.get("cpu_sys_s", ""),
            ))
        _atomic_bytes(self.root / "by_design/index.csv", output.getvalue().encode("utf-8"))

    def run(self) -> None:
        try:
            self.preflight()
        except Exception as exc:
            self.stopped = True
            self.stop_reason = f"input preflight: {type(exc).__name__}: {exc}"
            self.phase = "preflight_failed"
            self.write_status()
            return
        self.phase = "depth3"
        self.write_status()
        pending = deque(task for task in self.tasks if task not in self.rows)
        with ThreadPoolExecutor(max_workers=MAX_ACTIVE) as pool:
            active: dict[Any, tuple[str, str]] = {}

            def fill() -> None:
                while pending and not self.stopped and len(active) < MAX_ACTIVE:
                    try:
                        check_pins(self.frozen, self.runtime)
                    except Exception as exc:
                        self.stopped = True
                        self.stop_reason = f"source pin: {type(exc).__name__}: {exc}"
                        self.write_status()
                        break
                    parent_hash, recipe_id = pending.popleft()
                    active[pool.submit(self.one, parent_hash, recipe_id)] = (parent_hash, recipe_id)

            fill()
            while active:
                finished, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in finished:
                    parent_hash, recipe_id = active.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "schema_version": "thursday-crossworker-depth3-edge-v1",
                            "design": self.design, "input_hash": parent_hash,
                            "recipe_id": recipe_id, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "run_id": None, "run_status": None, "output_hash": None,
                            "evidence_id": None, "amd_result": None,
                            "amd_outputs_compared": 0, "amd_record": None,
                            "cost": None, "checked_at_utc": utc(),
                        }
                    self.commit(row)
                fill()
        self.phase = "complete" if len(self.rows) == len(self.tasks) and not self.stopped else "stopped"
        self.write_status()
        _atomic_bytes(self.root / "summary.json", _json_bytes({
            "schema_version": "thursday-crossworker-depth3-summary-v1",
            "design": self.design, "phase": self.phase,
            "planned": len(self.tasks), "attempted": len(self.rows),
            "amd_pass": sum(row["status"] == "ok" for row in self.rows.values()),
            "failed": sum(row["status"] != "ok" for row in self.rows.values()),
            "plan_sha256": _sha256_file(self.plan_path),
            "index_sha256": _sha256_file(self.root / "by_design/index.csv"),
            "completed_at_utc": utc(),
        }))


def prepare(design: str) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    selected = selection()
    inputs = selected["by_design"][design]
    root, _, _, _ = paths(design)
    runtime = make_real_runtime(
        root, cache_enabled=True, max_slots=MAX_SLOTS,
        load_ceiling_1m=LOAD_CEILING,
        min_available_memory_bytes=MEMORY_FLOOR,
        latch_smoke_status=SMOKE,
    )
    return runtime, plan(runtime, design, inputs), inputs


def run(design: str) -> None:
    runtime, frozen, inputs = prepare(design)
    root, plan_path, _, _ = paths(design)
    if not plan_path.is_file() or plan_path.read_bytes() != _json_bytes(frozen):
        raise ValueError("per-design plan changed after launch")
    with _file_lock(root / "driver.lock"):
        study = Study(design, runtime, frozen, inputs)
        try:
            study.run()
        except Exception as exc:
            study.phase = "failed"
            study.stop_reason = f"{type(exc).__name__}: {exc}"
            study.stopped = True
            study.write_status()
            raise


def driver_alive(design: str, pid: int) -> bool:
    try:
        info = _stat(pid)
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return bool(info and info.state != "Z" and str(SCRIPT).encode() in args
                    and b"--run" in args and design.encode() in args)
    except OSError:
        return False


def targets(design: str) -> tuple[set[str], set[int]]:
    """Find only exact tagged descendants working in this design's data root."""
    root, _, _, _ = paths(design)
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


def cleanup(design: str) -> dict[str, Any]:
    root, _, _, _ = paths(design)
    tags, groups = targets(design)
    result: dict[str, Any] = {
        "schema_version": "thursday-crossworker-depth3-cleanup-v1",
        "design": design, "exact_tags": sorted(tags), "groups": sorted(groups),
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
    result["tag_sweeps"] = {tag: list(sweep_tag(tag, grace_s=5).alive) for tag in sorted(tags)}
    result["survivors"] = {tag: [p.pid for p in tagged(tag)] for tag in sorted(tags) if tagged(tag)}
    result["finished_at_utc"] = utc()
    _atomic_bytes(root / "watchdog_cleanup_audit.json", _json_bytes(result))
    return result


def watch(design: str) -> None:
    root, _, manifest_path, status_path = paths(design)
    pid = read(manifest_path)["driver_pid"]
    if not driver_alive(design, pid):
        raise RuntimeError("driver absent at watchdog start")
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        if not driver_alive(design, pid):
            state = read(status_path) if status_path.is_file() else {}
            live_tags, _groups = targets(design)
            if state.get("phase") not in ("complete", "stopped", "preflight_failed") or live_tags:
                cleanup(design)
            return
        time.sleep(15)
    raise RuntimeError("watchdog lifetime ended with driver alive")


def launch(design: str) -> dict[str, Any]:
    runtime, frozen, inputs = prepare(design)
    check_pins(frozen, runtime)
    root, plan_path, manifest_path, _ = paths(design)
    load, available = Runner._resource_snapshot()
    if load > LOAD_CEILING or available < MEMORY_FLOOR:
        raise RuntimeError("load or available memory blocks launch")
    if manifest_path.exists():
        raise FileExistsError("design already launched")
    if plan_path.exists():
        if plan_path.read_bytes() != _json_bytes(frozen):
            raise ValueError("existing per-design plan differs")
    else:
        _atomic_bytes(plan_path, _json_bytes(frozen))
    with _file_lock(root / "launch.lock"):
        if manifest_path.exists():
            raise FileExistsError("design already launched")
        with (root / "driver.stdout").open("wb") as out, (root / "driver.stderr").open("wb") as err:
            driver = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--design", design, "--run"],
                cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        result = {
            "schema_version": "thursday-crossworker-depth3-launch-v1",
            "design": design, "driver_pid": driver.pid, "started_at_utc": utc(),
            "plan_sha256": _sha256_file(plan_path), "planned": len(tasks(inputs)),
            "max_active": MAX_ACTIVE, "load_1m_at_launch": load,
            "available_memory_bytes_at_launch": available,
        }
        _atomic_bytes(manifest_path, _json_bytes(result))
        try:
            with (root / "watchdog.stdout").open("wb") as out, (root / "watchdog.stderr").open("wb") as err:
                watcher = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "--design", design, "--watch"],
                    cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    start_new_session=True, close_fds=True,
                )
            result["watchdog_pid"] = watcher.pid
            _atomic_bytes(manifest_path, _json_bytes(result))
            return result
        except Exception:
            if driver_alive(design, driver.pid):
                group = os.getpgid(driver.pid)
                if group == driver.pid and group != os.getpgrp():
                    os.killpg(group, signal.SIGTERM)
            time.sleep(3)
            cleanup(design)
            raise


def status(design: str) -> dict[str, Any]:
    root, _, manifest_path, status_path = paths(design)
    state = read(status_path) if status_path.is_file() else {"phase": "not_launched"}
    manifest = read(manifest_path) if manifest_path.is_file() else {}
    pid = manifest.get("driver_pid")
    load, available = Runner._resource_snapshot()
    return {
        "design": design, "phase": state.get("phase"), "completed": state.get("completed", 0),
        "planned": len(tasks(selection()["by_design"][design])),
        "amd_pass": state.get("amd_pass", 0), "failed": state.get("failed", 0),
        "error": state.get("error"), "driver_pid": pid,
        "driver_alive": driver_alive(design, pid) if type(pid) is int else False,
        "load_1m": load, "available_memory_bytes": available,
        "busy_shared_slots": Runner(PROJECT / "data/slots", max_slots=MAX_SLOTS)._busy_slots(),
        "driver_stderr_bytes": (root / "driver.stderr").stat().st_size if manifest else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, choices=DESIGNS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        inputs = selection()["by_design"][args.design]
        print(json.dumps({"design": args.design, "parents": len(inputs["selected_states"]),
                          "planned": len(tasks(inputs)), "tool_jobs_launched": 0}, sort_keys=True))
    elif args.launch:
        print(json.dumps(launch(args.design), sort_keys=True))
    elif args.run:
        run(args.design)
    elif args.watch:
        watch(args.design)
    else:
        print(json.dumps(status(args.design), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
