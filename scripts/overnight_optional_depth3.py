"""Isolated AMD-checked VexRiscv within-worker depth-three study.

Read the 50 frozen training D2 parents; write only a new Development data root.
The launch gate requires the held-out baseline to finish successfully first.
No Tcl or hardware command is constructed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
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
from orch.runtime import make_real_runtime

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
TRAIN = PROJECT / "data/sweeps/worker_fitting_train"
TEST = PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04"
ROOT = PROJECT / "data/sweeps/overnight_priority2_2026-11-04/depth3"
SMOKE = PROJECT / "data/scratch_overnight_2026-11-04/cache_gate_smoke_20261104/status.json"
PLAN, LAUNCH, STATUS = (ROOT / name for name in ("plan.json", "launch.json", "status.json"))
JOURNAL_ID = "overnight_depth3_vexriscv"
DESIGN = "vexriscv_re-place"
TEST_PLAN_SHA = "060303a7b5fd4a8c46c174e32e35834d4aaa87abfbc966f97b62f165d5550128"
TIMING = ("LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock", "LATCH.hardblock_relocate", "LATCH.memory_path_opt")
AREA = ("A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED", "A_E_DUPL_Q1000")
TASKS = tuple(seq for recipes in (TIMING, AREA) for seq in itertools.product(recipes, repeat=3))
PAIRS = tuple(seq for recipes in (TIMING, AREA) for seq in itertools.product(recipes, repeat=2))
METRICS = (MetricName.WNS_NS, MetricName.TNS_NS, MetricName.PACKED_LUTS, MetricName.NETLIST_LUTS)
MAX_ACTIVE, MAX_SLOTS, LOAD_CEILING, MEMORY_FLOOR = 24, 96, 180.0, 200_000_000_000


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def under(value: str, root: Path) -> Path:
    path = Path(value).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError(f"source escaped its data root: {path}")
    return path


def outputs_compared(record: dict[str, Any]) -> int:
    """Use the observed AMD log output count, as the test baseline does."""
    attempts = record.get("attempts") or []
    if not attempts:
        return 0
    log_path = Path(attempts[-1]["run_dir"]) / "validate.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return max((int(x) for x in re.findall(r"Outputs\s*\((\d+)\):", text)), default=0)


def require_metrics(evidence: Evidence) -> None:
    for name in METRICS:
        item = evidence.metric(name)
        if item is None or not math.isfinite(item.value):
            raise ValueError(f"Evidence lacks finite {name.value}")


def selected(runtime: Any) -> dict[str, Any]:
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    if (tuple(sets.worker_sets.get("timing", ())) != TIMING
            or tuple(sets.worker_sets.get("area", ())) != AREA
            or tuple(sets.worker_sets.get("power", ())) != ()):
        raise ValueError("signed worker recipe sets changed")
    return {name: runtime.registry.get(name) for name in TIMING + AREA}


def source() -> dict[str, Any]:
    """Pin all 50 D2 parents, their meter records and retained AMD PASS logs."""
    index = TRAIN / "by_design/index.csv"
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    starts = [r for r in rows if r["design"] == DESIGN and r["depth"] == "0"]
    if len(starts) != 1 or starts[0]["sequence"] != "B2":
        raise ValueError("training index lacks one VexRiscv B2")
    b2_hash = starts[0]["output_sha256"]
    b2_path = under(starts[0]["canonical_dcp"], TRAIN)
    if _sha256_file(b2_path) != b2_hash:
        raise ValueError("training B2 DCP hash changed")
    if read(TRAIN / "verification/plan.json").get("references", {}).get(DESIGN) != b2_hash:
        raise ValueError("training AMD reference differs from B2")
    starts_evidence = list((TRAIN / "measurements" / b2_hash).glob("*/evidence.json"))
    if len(starts_evidence) != 1:
        raise ValueError("training B2 lacks unique Evidence")
    baseline = Evidence.model_validate_json(starts_evidence[0].read_bytes())
    if (baseline.state_id != b2_hash or baseline.fake or baseline.routed is not True
            or baseline.route_errors != 0 or baseline.drc_errors != 0):
        raise ValueError("training B2 is not physically valid")
    require_metrics(baseline)
    parents: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["design"] != DESIGN or row["depth"] != "2":
            continue
        pair = tuple(row["sequence"].split(" > "))
        if pair not in PAIRS:
            continue
        key = "__".join(pair)
        if key in parents:
            raise ValueError("duplicate training D2 parent")
        output_hash = row["output_sha256"]
        dcp = under(row["canonical_dcp"], TRAIN)
        if (row["run_status"] != "ok" or row["physically_valid"] != "True"
                or row["equivalence_status"] != "pass" or _sha256_file(dcp) != output_hash):
            raise ValueError(f"invalid training D2 parent: {pair}")
        evidence_path = TRAIN / "runs" / row["run_id"] / "evidence.json"
        evidence = Evidence.model_validate_json(evidence_path.read_bytes())
        if (evidence.state_id != output_hash or evidence.fake or evidence.routed is not True
                or evidence.route_errors != 0 or evidence.drc_errors != 0
                or evidence.constraint_fingerprint != baseline.constraint_fingerprint
                or evidence.clock_digest != baseline.clock_digest):
            raise ValueError(f"D2 Evidence differs from B2: {pair}")
        require_metrics(evidence)
        amd_path = under(row["verification_record"], TRAIN / "verification")
        amd = read(amd_path)
        raw = under(amd["raw_report_path"], TRAIN / "verification")
        if (amd.get("result") != "pass" or amd.get("output_sha256") != output_hash
                or amd.get("reference_sha256") != b2_hash
                or _sha256_file(raw) != amd.get("raw_report_sha256")
                or outputs_compared(amd) <= 0):
            raise ValueError(f"D2 parent lacks retained AMD PASS: {pair}")
        parents[key] = {
            "pair": list(pair), "output_hash": output_hash, "dcp": str(dcp),
            "run_id": row["run_id"], "evidence": str(evidence_path),
            "evidence_hash": _sha256_file(evidence_path), "amd": str(amd_path),
            "amd_hash": _sha256_file(amd_path),
        }
    if set(tuple(p["pair"]) for p in parents.values()) != set(PAIRS) or len(parents) != 50:
        raise ValueError("training index lacks all 50 within-worker D2 parents")
    return {
        "index_hash": _sha256_file(index), "b2_hash": b2_hash, "b2_dcp": str(b2_path),
        "b2_evidence": str(starts_evidence[0]), "b2_evidence_hash": _sha256_file(starts_evidence[0]),
        "constraint": baseline.constraint_fingerprint, "clock": baseline.clock_digest,
        "parents": parents,
    }


def test_baseline_ready() -> str:
    """Require all priority-one work to be complete and valid before launch."""
    if _sha256_file(TEST / "plan.json") != TEST_PLAN_SHA:
        raise ValueError("held-out baseline plan changed")
    result = read(TEST / "summary.json")
    status = read(TEST / "status.json")
    if (result.get("phase") != "complete" or status.get("phase") != "complete"
            or result.get("successful_edges_by_phase") != {"C0": 7, "D1": 70, "D2": 500}
            or result.get("starts_amd_source_pass_and_telemetry_ok") != 7
            or result.get("failed_designs") != []
            or len(result.get("admitted_test_designs") or []) != 5):
        raise ValueError("held-out baseline is not complete and valid")
    return _sha256_file(TEST / "summary.json")


def plan(runtime: Any, inputs: dict[str, Any], test_summary_hash: str) -> dict[str, Any]:
    specs = selected(runtime)
    return {
        "schema_version": "overnight-depth3-plan-v1", "design": DESIGN,
        "training_inputs": inputs, "test_summary_hash": test_summary_hash,
        "timing_ids": list(TIMING), "area_ids": list(AREA),
        "recipe_versions": {name: specs[name].version for name in specs},
        "planned_edges": len(TASKS), "active_limit": MAX_ACTIVE,
        "shared_slots": MAX_SLOTS, "load_ceiling": LOAD_CEILING,
        "available_memory_floor": MEMORY_FLOOR,
        "meter_version": runtime.meter.version, "smoke_hash": _sha256_file(SMOKE),
        "source_hashes": {
            "driver": _sha256_file(SCRIPT),
            "runner": _sha256_file(PROJECT / "orch/engine/runner.py"),
            "runtime": _sha256_file(PROJECT / "orch/runtime.py"),
            "real_backend": _sha256_file(PROJECT / "orch/engine/real_backend.py"),
            "apply": _sha256_file(PROJECT / "orch/engine/apply.py"),
            "models": _sha256_file(PROJECT / "orch/contract/models.py"),
            "jobtag": _sha256_file(PROJECT / "orch/engine/jobtag.py"),
            "verification": _sha256_file(PROJECT / "orch/baselines/verification.py"),
            "validator": _sha256_file(PROJECT / "orch/recipes/adapters/validator.py"),
            "env": _sha256_file(PROJECT / "recipes/support/env.sh"),
        },
    }


def cost(run: Any) -> dict[str, Any]:
    fields = raw_cost(run)
    if any(fields[key] is None or not math.isfinite(fields[key]) for key in (
        "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
        "load_1m_start", "load_1m_end", "busy_slots_start",
    )):
        raise ValueError("depth-three run lacks cost telemetry")
    if run.cache_hit and not run.original_run_id:
        raise ValueError("cache hit lacks original run cost")
    return fields


def raw_cost(run: Any) -> dict[str, Any]:
    """Keep observed run costs even if an acceptance check fails."""
    return {key: getattr(run, key) for key in (
        "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
        "load_1m_start", "load_1m_end", "busy_slots_start", "cache_hit",
        "original_run_id", "backpressure_events", "backpressure_wait_s",
    )}


def alias(path: Path, target: Path) -> None:
    """Publish a readable hard link after the journal commit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _sha256_file(path) != _sha256_file(target):
            raise ValueError("DCP alias differs from its canonical state")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.link")
    try:
        os.link(target, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Study:
    """Commit each output only after independent meter and AMD checks."""

    def __init__(self, runtime: Any, frozen: dict[str, Any]) -> None:
        self.runtime = runtime
        self.inputs = frozen["training_inputs"]
        self.specs = selected(runtime)
        self.journal = Journal(ROOT)
        self.start = runtime.store.import_state(Path(self.inputs["b2_dcp"]), design=DESIGN, fake=False)
        if self.start.id != self.inputs["b2_hash"]:
            raise ValueError("imported training B2 differs")
        self.parents = {}
        for parent in self.inputs["parents"].values():
            state = runtime.store.import_state(Path(parent["dcp"]), design=DESIGN, fake=False)
            if state.id != parent["output_hash"]:
                raise ValueError("imported training D2 parent differs")
            self.parents[tuple(parent["pair"])] = state
        self.ledger = VerificationLedger(
            store=runtime.store, runner=runtime.runner, validator=runtime.validator,
            references={DESIGN: self.start.id},
        )
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.stopped = False
        self.phase = "preparing"
        self.load_journal()
        self.repair_aliases()
        self.index()

    def load_journal(self) -> None:
        events = self.journal.read(JOURNAL_ID)
        if not events:
            self.journal.append(JOURNAL_ID, {
                "id": "configuration", "type": "configuration",
                "plan_sha256": _sha256_file(PLAN), "at_utc": utc(),
            })
            return
        if events[0].get("type") != "configuration" or events[0].get("plan_sha256") != _sha256_file(PLAN):
            raise ValueError("depth-three journal plan differs")
        for event in events[1:]:
            if event.get("type") != "edge":
                raise ValueError("unknown depth-three journal event")
            row = event["edge"]
            seq = tuple(row["sequence"])
            if seq not in TASKS or seq in self.rows:
                raise ValueError("duplicate or unknown depth-three sequence")
            self.rows[seq] = row
            if row["status"] != "ok":
                self.stopped = True
                continue
            outcome = self.runtime.store.load_outcome(row["run_id"])
            if (outcome.run.recipe_id != seq[-1]
                    or outcome.run.recipe_version != self.specs[seq[-1]].version
                    or outcome.run.input_sha256_before != row["input_hash"]
                    or outcome.run.input_sha256_after != row["input_hash"]
                    or outcome.run.status != RunStatus.OK
                    or outcome.state is None or outcome.evidence is None
                    or outcome.state.id != row["output_hash"]
                    or outcome.evidence.state_id != outcome.state.id
                    or outcome.evidence.fake
                    or outcome.evidence.routed is not True
                    or outcome.evidence.route_errors != 0
                    or outcome.evidence.drc_errors != 0
                    or outcome.evidence.constraint_fingerprint != self.inputs["constraint"]
                    or outcome.evidence.clock_digest != self.inputs["clock"]
                    or row["cost"] != cost(outcome.run)):
                raise ValueError("saved depth-three edge differs from run")
            require_metrics(outcome.evidence)
            amd = self.ledger.require_pass(DESIGN, outcome.state.id)
            if (row.get("amd_result") != "pass"
                    or row.get("amd_outputs_compared", 0) <= 0
                    or row["amd_outputs_compared"] != outputs_compared(amd)
                    or row.get("amd_record") != str(self.ledger._record_path(DESIGN, outcome.state.id))):
                raise ValueError("saved depth-three edge lost AMD compared outputs")

    def repair_aliases(self) -> None:
        """Recover aliases after a journaled edge and interrupted publication."""
        for row in self.rows.values():
            expected = self.alias_path(row)
            if row.get("dcp_alias") != (str(expected) if expected else None):
                raise ValueError("saved depth-three alias differs from naming rule")
            if expected is not None:
                alias(expected, self.runtime.store.state_path(row["output_hash"]))

    def write_status(self, error: str | None = None) -> None:
        _atomic_bytes(STATUS, _json_bytes({
            "schema_version": "overnight-depth3-status-v1", "phase": self.phase,
            "updated_at_utc": utc(), "driver_pid": os.getpid(),
            "planned": len(TASKS), "completed": len(self.rows),
            "amd_pass": sum(row["status"] == "ok" for row in self.rows.values()),
            "failed": sum(row["status"] != "ok" for row in self.rows.values()),
            "stopped": self.stopped, "error": error,
        }))

    def one(self, seq: tuple[str, str, str]) -> dict[str, Any]:
        state = self.parents[seq[:2]]
        source_parent = self.inputs["parents"]["__".join(seq[:2])]
        outcome = None
        amd = None
        compared = 0
        error = None
        try:
            if (_sha256_file(Path(source_parent["dcp"])) != state.id
                    or _sha256_file(Path(source_parent["evidence"])) != source_parent["evidence_hash"]
                    or _sha256_file(Path(source_parent["amd"])) != source_parent["amd_hash"]):
                raise ValueError("training D2 parent evidence changed")
            outcome = self.runtime.engine.execute(state, self.specs[seq[-1]])
            if outcome.state is not None:
                amd = self.ledger.verify(DESIGN, outcome.state.id)
                compared = outputs_compared(amd)
            if outcome.run.status != RunStatus.OK or outcome.state is None or outcome.evidence is None:
                raise ValueError(f"recipe or meter failed: {outcome.run.status.value}")
            if amd is None or amd.get("result") != "pass" or compared <= 0:
                raise ValueError(f"AMD check did not PASS with compared outputs: {amd.get('result') if amd else 'missing'}")
            evidence = outcome.evidence
            if (evidence.state_id != outcome.state.id or evidence.fake
                    or evidence.routed is not True or evidence.route_errors != 0
                    or evidence.drc_errors != 0
                    or evidence.constraint_fingerprint != self.inputs["constraint"]
                    or evidence.clock_digest != self.inputs["clock"]):
                raise ValueError("output lacks B2-matching physically valid Evidence")
            require_metrics(evidence)
            cost(outcome.run)
            if outcome.run.input_sha256_before != state.id or outcome.run.input_sha256_after != state.id:
                raise ValueError("recipe input DCP changed")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        run = outcome.run if outcome else None
        output = outcome.state if outcome else None
        return {
            "schema_version": "overnight-depth3-edge-v1", "design": DESIGN,
            "sequence": list(seq), "status": "failed" if error else "ok", "error": error,
            "input_hash": state.id, "run_id": run.id if run else None,
            "run_status": run.status.value if run else None,
            "output_hash": output.id if output else None,
            "evidence_id": outcome.evidence.id if outcome and outcome.evidence else None,
            "amd_result": amd.get("result") if amd else None,
            "amd_outputs_compared": compared,
            "amd_record": str(self.ledger._record_path(DESIGN, output.id)) if amd and output else None,
            "cost": raw_cost(run) if run else None, "checked_at_utc": utc(),
        }

    def alias_path(self, row: dict[str, Any]) -> Path | None:
        if not row.get("output_hash"):
            return None
        name = "__".join(row["sequence"])
        if row["status"] == "ok":
            return ROOT / "by_design" / DESIGN / "depth3" / f"{name}.dcp"
        return ROOT / "by_design" / DESIGN / "failed" / f"{name}__{row['output_hash']}.dcp"

    def index(self) -> None:
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(("design", "depth", "sequence", "status", "run_id", "input_hash",
                         "output_hash", "dcp_alias", "canonical_dcp", "amd_result",
                         "amd_outputs_compared", "cache_hit", "original_run_id"))
        for seq in TASKS:
            row = self.rows.get(seq)
            if row is None:
                continue
            output_hash = row.get("output_hash")
            run_cost = row.get("cost") or {}
            writer.writerow((
                DESIGN, 3, " > ".join(seq), row["status"], row.get("run_id") or "",
                row["input_hash"], output_hash or "", row.get("dcp_alias") or "",
                str(ROOT / "states" / output_hash / "design.dcp") if output_hash else "",
                row.get("amd_result") or "", row.get("amd_outputs_compared") or "",
                run_cost.get("cache_hit", ""), run_cost.get("original_run_id") or "",
            ))
        _atomic_bytes(ROOT / "by_design/index.csv", text.getvalue().encode("utf-8"))

    def commit(self, row: dict[str, Any]) -> None:
        seq = tuple(row["sequence"])
        if seq in self.rows:
            raise ValueError("depth-three sequence already committed")
        name = self.alias_path(row)
        row["dcp_alias"] = str(name) if name else None
        event_id = hashlib.sha256(_json_bytes(list(seq))).hexdigest()
        self.journal.append(JOURNAL_ID, {
            "id": f"edge-{event_id}", "type": "edge", "edge": row, "at_utc": utc(),
        })
        self.rows[seq] = row
        if name is not None:
            alias(name, self.runtime.store.state_path(row["output_hash"]))
        if row["status"] != "ok":
            self.stopped = True
        self.index()
        self.write_status()

    def run(self) -> None:
        self.phase = "depth3"
        self.write_status()
        pending = deque(seq for seq in TASKS if seq not in self.rows)
        with ThreadPoolExecutor(max_workers=MAX_ACTIVE) as pool:
            active = {}

            def fill() -> None:
                while pending and not self.stopped and len(active) < MAX_ACTIVE:
                    seq = pending.popleft()
                    active[pool.submit(self.one, seq)] = seq

            fill()
            while active:
                done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in done:
                    seq = active.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "schema_version": "overnight-depth3-edge-v1", "design": DESIGN,
                            "sequence": list(seq), "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "input_hash": self.parents[seq[:2]].id, "run_id": None,
                            "run_status": None, "output_hash": None, "evidence_id": None,
                            "amd_result": None, "amd_outputs_compared": 0, "amd_record": None,
                            "cost": None, "checked_at_utc": utc(),
                        }
                    self.commit(row)
                fill()
        self.phase = "complete" if len(self.rows) == len(TASKS) and not self.stopped else "complete_with_failures"
        self.write_status()
        _atomic_bytes(ROOT / "summary.json", _json_bytes({
            "schema_version": "overnight-depth3-summary-v1", "phase": self.phase,
            "planned": len(TASKS), "attempted": len(self.rows),
            "amd_pass": sum(row["status"] == "ok" for row in self.rows.values()),
            "failed": sum(row["status"] != "ok" for row in self.rows.values()),
            "plan_sha256": _sha256_file(PLAN), "index_sha256": _sha256_file(ROOT / "by_design/index.csv"),
            "completed_at_utc": utc(),
        }))


def prepare() -> tuple[Any, dict[str, Any]]:
    test_hash = test_baseline_ready()
    inputs = source()
    runtime = make_real_runtime(
        ROOT, cache_enabled=True, max_slots=MAX_SLOTS,
        load_ceiling_1m=LOAD_CEILING, min_available_memory_bytes=MEMORY_FLOOR,
        latch_smoke_status=SMOKE,
    )
    return runtime, plan(runtime, inputs, test_hash)


def run() -> None:
    runtime, frozen = prepare()
    if not PLAN.is_file() or PLAN.read_bytes() != _json_bytes(frozen):
        raise ValueError("depth-three plan changed after launch")
    with _file_lock(ROOT / "driver.lock"):
        study = Study(runtime, frozen)
        try:
            study.run()
        except Exception as exc:
            study.phase = "failed"
            study.write_status(error=f"{type(exc).__name__}: {exc}")
            raise


def driver_alive(pid: int) -> bool:
    try:
        info = _stat(pid)
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return bool(info and info.state != "Z" and str(SCRIPT).encode() in args and b"--run" in args)
    except OSError:
        return False


def targets() -> tuple[set[str], set[int]]:
    """Scope orphan cleanup to exact tags working inside this data root."""
    prefix = str(ROOT).encode()
    tags, groups = set(), set()
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
    tags, groups = targets()
    result = {"schema_version": "overnight-depth3-cleanup-v1", "exact_tags": sorted(tags),
              "groups": sorted(groups), "started_at_utc": utc()}
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
    pid = read(LAUNCH)["driver_pid"]
    if not driver_alive(pid):
        raise RuntimeError("depth-three driver absent at watchdog start")
    deadline = time.monotonic() + 36 * 3600
    while time.monotonic() < deadline:
        if not driver_alive(pid):
            state = read(STATUS) if STATUS.is_file() else {}
            if state.get("phase") not in ("complete", "complete_with_failures"):
                cleanup()
            return
        time.sleep(15)
    raise RuntimeError("depth-three watchdog lifetime ended with driver alive")


def launch() -> dict[str, Any]:
    runtime, frozen = prepare()
    load, available = Runner._resource_snapshot()
    if load > LOAD_CEILING or available < MEMORY_FLOOR:
        raise RuntimeError("load or available memory blocks optional launch")
    if LAUNCH.exists():
        raise FileExistsError("depth-three study already launched")
    if PLAN.exists():
        if PLAN.read_bytes() != _json_bytes(frozen):
            raise ValueError("existing depth-three plan differs")
    else:
        _atomic_bytes(PLAN, _json_bytes(frozen))
    with _file_lock(ROOT / "launch.lock"):
        if LAUNCH.exists():
            raise FileExistsError("depth-three study already launched")
        with (ROOT / "driver.stdout").open("wb") as out, (ROOT / "driver.stderr").open("wb") as err:
            driver = subprocess.Popen([sys.executable, str(SCRIPT), "--run"],
                                      cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out,
                                      stderr=err, start_new_session=True, close_fds=True)
        result = {"schema_version": "overnight-depth3-launch-v1", "driver_pid": driver.pid,
                  "started_at_utc": utc(), "plan_sha256": _sha256_file(PLAN),
                  "planned": len(TASKS), "active_limit": MAX_ACTIVE,
                  "load_1m_at_launch": load, "available_memory_bytes_at_launch": available}
        _atomic_bytes(LAUNCH, _json_bytes(result))
        try:
            with (ROOT / "watchdog.stdout").open("wb") as out, (ROOT / "watchdog.stderr").open("wb") as err:
                watcher = subprocess.Popen([sys.executable, str(SCRIPT), "--watch"],
                                           cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out,
                                           stderr=err, start_new_session=True, close_fds=True)
            result["watchdog_pid"] = watcher.pid
            _atomic_bytes(LAUNCH, _json_bytes(result))
            return result
        except Exception:
            if driver_alive(driver.pid):
                group = os.getpgid(driver.pid)
                if group == driver.pid and group != os.getpgrp():
                    os.killpg(group, signal.SIGTERM)
            time.sleep(3)
            cleanup()
            raise


def short_status() -> dict[str, Any]:
    state = read(STATUS) if STATUS.is_file() else {"phase": "not_launched"}
    manifest = read(LAUNCH) if LAUNCH.is_file() else {}
    pid = manifest.get("driver_pid")
    load, available = Runner._resource_snapshot()
    return {"phase": state.get("phase"), "completed": state.get("completed", 0),
            "planned": len(TASKS), "amd_pass": state.get("amd_pass", 0),
            "failed": state.get("failed", 0), "error": state.get("error"),
            "driver_pid": pid, "driver_alive": driver_alive(pid) if type(pid) is int else False,
            "load_1m": load, "available_memory_bytes": available,
            "busy_shared_tool_slots": Runner(PROJECT / "data/slots", max_slots=MAX_SLOTS)._busy_slots(),
            "driver_stderr_bytes": (ROOT / "driver.stderr").stat().st_size if manifest else 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        inputs = source()
        print(json.dumps({"design": DESIGN, "d2_parents": len(inputs["parents"]),
                          "planned_three_step_sequences": len(TASKS),
                          "timing_sequences": len(TIMING) ** 3,
                          "area_sequences": len(AREA) ** 3,
                          "baseline_ready": (TEST / "summary.json").is_file(),
                          "tool_jobs_launched": 0}, sort_keys=True))
    elif args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.run:
        run()
    elif args.watch:
        watch()
    else:
        print(json.dumps(short_status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
