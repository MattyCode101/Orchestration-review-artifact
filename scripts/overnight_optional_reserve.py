"""Conditional verified reserve study on rosetta_3d-rendering_v2.

This is a separate training-data campaign. It starts only after the held-out
baseline is complete, both FIR designs are admitted, and this design is not
admitted. It reads ten AMD-checked D1 parents and writes 100 ordered D2 edges.
No Vivado/Tcl command is created here; the existing engine owns tool execution.
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
from orch.engine.dcp_parser import DcpTelemetryExtractor
from orch.engine.jobtag import _own_pids, _stat, _tag_of, group_members, sweep_tag, tagged
from orch.engine.journal import Journal
from orch.engine.runner import Runner
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from orch.runtime import make_real_runtime

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
ROOT = PROJECT / "data/sweeps/overnight_priority2_2026-11-04/reserve"
TEST = PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04"
NOISE = PROJECT / "data/exp_evening_2026-11-04/test_noise_family"
SMOKE = PROJECT / "data/scratch_overnight_2026-11-04/cache_gate_smoke_20261104/status.json"
PLAN, LAUNCH, STATUS = (ROOT / name for name in ("plan.json", "launch.json", "status.json"))
CONTROLLER_PLAN = ROOT / "controller_plan.json"
CONTROLLER_STATUS = ROOT / "controller_status.json"
JOURNAL_ID = "overnight_reserve_3d_rendering_v2"
TEST_PLAN_SHA = "060303a7b5fd4a8c46c174e32e35834d4aaa87abfbc966f97b62f165d5550128"
DESIGN = "rosetta_3d-rendering_v2"
FIRS = frozenset(("fir_symmetric_systolic", "fir_systolic_transposed"))
TIMING = ("LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock",
          "LATCH.hardblock_relocate", "LATCH.memory_path_opt")
AREA = ("A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED", "A_E_DUPL_Q1000")
RECIPES = TIMING + AREA
TASKS = tuple(itertools.product(RECIPES, repeat=2))
MAX_ACTIVE, MAX_SLOTS = 24, 96
LOAD_CEILING, MEMORY_FLOOR = 180.0, 200_000_000_000
DEADLINE = datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc)
METRICS = (MetricName.WNS_NS, MetricName.TNS_NS, MetricName.PACKED_LUTS,
           MetricName.NETLIST_LUTS, MetricName.FFS, MetricName.DYNAMIC_POWER_W)


def utc() -> str:
    """Return UTC for durable records."""
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, Any]:
    """Read one saved JSON object."""
    item = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(item, dict):
        raise ValueError(f"expected JSON object: {path}")
    return item


def sha(path: Path) -> str:
    """Hash an existing artifact."""
    return _sha256_file(path)


def contained(value: str, root: Path) -> Path:
    """Refuse a source path outside the recorded campaign."""
    path = Path(value).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"source path escaped its campaign: {path}")
    return path


def compared_outputs(record: dict[str, Any]) -> int:
    """Use the output count in the retained AMD validator log."""
    attempts = record.get("attempts") or []
    if not attempts:
        return 0
    log = Path(attempts[-1]["run_dir"]) / "validate.log"
    content = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    return max((int(x) for x in re.findall(r"Outputs\s*\((\d+)\):", content)), default=0)


def require_metrics(evidence: Evidence) -> None:
    """Require finite independent meter readings for all six metrics."""
    for name in METRICS:
        value = evidence.metric(name)
        if value is None or not math.isfinite(value.value):
            raise ValueError(f"missing finite meter metric: {name.value}")


def eligibility(
    status: dict[str, Any], summary: dict[str, Any] | None,
    admission: dict[str, Any] | None,
) -> str:
    """Decide reserve eligibility from counts and admission labels only."""
    if status.get("phase") in ("failed", "complete_with_failures", "stopped_before_d2"):
        return "blocked_by_priority_one"
    if status.get("stopped_designs") or status.get("error"):
        return "blocked_by_priority_one"
    if summary is None or admission is None:
        return "waiting_for_priority_one"
    if (status.get("phase") != "complete"
            or summary.get("phase") != "complete"
            or summary.get("successful_edges_by_phase") != {"C0": 7, "D1": 70, "D2": 500}
            or summary.get("starts_amd_source_pass_and_telemetry_ok") != 7
            or summary.get("failed_designs") != []):
        return "blocked_by_incomplete_priority_one"
    admitted = admission.get("admitted_in_order")
    if (not isinstance(admitted, list) or len(admitted) != 5
            or len(set(admitted)) != 5 or not all(isinstance(x, str) for x in admitted)):
        return "blocked_by_invalid_admission"
    if DESIGN in admitted:
        return "skipped_reserve_is_admitted_test"
    if not FIRS.issubset(admitted):
        return "skipped_fir_condition_not_met"
    return "eligible"


def gate() -> str:
    """Validate the frozen priority-one plan before reading admission labels."""
    if sha(TEST / "plan.json") != TEST_PLAN_SHA:
        raise ValueError("test baseline plan changed")
    status = read(TEST / "status.json")
    summary_path = TEST / "summary.json"
    admission_path = TEST / "admitted_test_designs_2026-11-04.json"
    summary = read(summary_path) if summary_path.is_file() else None
    admission = read(admission_path) if admission_path.is_file() else None
    if admission is not None:
        checksum = admission_path.with_suffix(".sha256")
        if (not checksum.is_file()
                or not checksum.read_text(encoding="utf-8").startswith(sha(admission_path))):
            raise ValueError("test admission checksum differs")
        if summary is not None and summary.get("admission_sha256") != sha(admission_path):
            raise ValueError("test summary admission hash differs")
    return eligibility(status, summary, admission)


def selected(runtime: Any) -> dict[str, Any]:
    """Require exactly the signed ten recipe IDs."""
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    if (tuple(sets.worker_sets.get("timing", ())) != TIMING
            or tuple(sets.worker_sets.get("area", ())) != AREA
            or tuple(sets.worker_sets.get("power", ())) != ()):
        raise ValueError("signed selected recipes changed")
    return {name: runtime.registry.get(name) for name in RECIPES}


def source() -> dict[str, Any]:
    """Pin ten verified test D1 states without reading test outcome rankings."""
    if gate() != "eligible":
        raise ValueError("reserve source is not eligible")
    summary = read(TEST / "summary.json")
    admission = TEST / "admitted_test_designs_2026-11-04.json"
    test_plan = read(TEST / "plan.json")
    reference = read(NOISE / "plan.json")["references"][DESIGN]
    b2 = Path(reference["b2"]).resolve(strict=True)
    original = Path(reference["original"]).resolve(strict=True)
    b2_evidence_path = Path(reference["b2_evidence_path"]).resolve(strict=True)
    if (sha(b2) != reference["b2_sha256"]
            or sha(original) != reference["original_sha256"]
            or sha(b2_evidence_path) != reference["b2_evidence_sha256"]
            or not b2.is_relative_to(PROJECT / "data")
            or not original.is_relative_to(
                (Path(os.environ.get("ORCH_BENCHMARK_DIR", "benchmarks")) / "fpl26_contest_v1.3.0"))
            or read(TEST / "start_checks" / f"{DESIGN}.json").get("status") != "ok"):
        raise ValueError("reserve B2/original/start proof differs")
    baseline = Evidence.model_validate_json(b2_evidence_path.read_bytes())
    if (baseline.state_id != reference["b2_sha256"] or baseline.fake
            or baseline.routed is not True or baseline.route_errors != 0
            or baseline.drc_errors != 0
            or baseline.constraint_fingerprint != reference["b2_constraint_fingerprint"]
            or baseline.clock_digest != reference["b2_clock_digest"]):
        raise ValueError("reserve B2 meter proof is invalid")
    require_metrics(baseline)
    if baseline.meter_version != test_plan["meter_version"]:
        raise ValueError("reserve B2 meter version differs from frozen test plan")
    index = TEST / "by_design/index.csv"
    with index.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parents: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("design") != DESIGN or row.get("depth") != "1":
            continue
        recipe = row.get("sequence")
        if recipe not in RECIPES or recipe in parents:
            raise ValueError("reserve D1 source has duplicate or unknown recipe")
        if (row.get("kind") != "recipe" or row.get("status") != "ok"
                or row.get("amd_result") != "pass"
                or row.get("telemetry_status") != "ok"
                or int(row.get("amd_outputs_compared") or 0) <= 0):
            raise ValueError(f"reserve D1 source is not verified: {recipe}")
        output_hash = row["output_sha256"]
        dcp = contained(row["canonical_dcp"], TEST)
        alias_path = contained(row["dcp_alias"], TEST)
        if sha(dcp) != output_hash or sha(alias_path) != output_hash:
            raise ValueError(f"reserve D1 DCP hash differs: {recipe}")
        evidence_path = TEST / "runs" / row["run_id"] / "evidence.json"
        evidence = Evidence.model_validate_json(evidence_path.read_bytes())
        if (evidence.state_id != output_hash or evidence.fake
                or evidence.routed is not True or evidence.route_errors != 0
                or evidence.drc_errors != 0
                or evidence.constraint_fingerprint != baseline.constraint_fingerprint
                or evidence.clock_digest != baseline.clock_digest):
            raise ValueError(f"reserve D1 meter proof differs: {recipe}")
        require_metrics(evidence)
        if evidence.meter_version != baseline.meter_version:
            raise ValueError(f"reserve D1 meter version differs: {recipe}")
        amd_path = contained(row["amd_record"], TEST / "verification")
        amd = read(amd_path)
        raw = contained(amd["raw_report_path"], TEST / "verification")
        if (amd.get("result") != "pass"
                or amd.get("output_sha256") != output_hash
                or amd.get("reference_sha256") != reference["original_sha256"]
                or sha(raw) != amd.get("raw_report_sha256")
                or compared_outputs(amd) <= 0):
            raise ValueError(f"reserve D1 AMD proof differs: {recipe}")
        telemetry = contained(row["telemetry_json"], TEST / "telemetry")
        telemetry_record = telemetry.parent / "result.json"
        tele = read(telemetry_record)
        if (tele.get("status") != "ok" or tele.get("state_sha256") != output_hash
                or tele.get("telemetry_sha256") != sha(telemetry)
                or tele.get("parser_sha256")
                != test_plan["source_hashes"]["dcp_parser_v3"]):
            raise ValueError(f"reserve D1 telemetry proof differs: {recipe}")
        parents[recipe] = {
            "run_id": row["run_id"], "output_hash": output_hash,
            "dcp": str(dcp), "evidence": str(evidence_path),
            "evidence_sha256": sha(evidence_path), "amd": str(amd_path),
            "amd_sha256": sha(amd_path), "telemetry": str(telemetry),
            "telemetry_sha256": sha(telemetry),
        }
    if set(parents) != set(RECIPES):
        raise ValueError("reserve source lacks all ten D1 parents")
    if summary.get("index_sha256") != sha(index):
        raise ValueError("test source index changed after completion")
    return {
        "test_summary_sha256": sha(TEST / "summary.json"),
        "admission_sha256": sha(admission),
        "test_index_sha256": sha(index),
        "test_parser_sha256": test_plan["source_hashes"]["dcp_parser_v3"],
        "b2_dcp": str(b2), "b2_hash": reference["b2_sha256"],
        "b2_evidence": str(b2_evidence_path),
        "b2_evidence_sha256": sha(b2_evidence_path),
        "original_dcp": str(original),
        "original_hash": reference["original_sha256"],
        "constraint": baseline.constraint_fingerprint,
        "clock": baseline.clock_digest,
        "parents": parents,
    }


def plan(runtime: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    """Freeze sources, code, recipe versions and resource limits."""
    specs = selected(runtime)
    if (runtime.meter.version != read(TEST / "plan.json")["meter_version"]
            or inputs["test_parser_sha256"] != sha(PROJECT / "orch/engine/dcp_parser.py")):
        raise ValueError("reserve meter or v3 parser differs from frozen test baseline")
    return {
        "schema_version": "overnight-reserve-plan-v1",
        "design": DESIGN, "source": inputs, "tasks": len(TASKS),
        "recipe_ids": list(RECIPES),
        "recipe_versions": {name: specs[name].version for name in RECIPES},
        "max_active": MAX_ACTIVE, "max_shared_slots": MAX_SLOTS,
        "load_ceiling": LOAD_CEILING, "memory_floor": MEMORY_FLOOR,
        "meter_version": runtime.meter.version, "smoke_sha256": sha(SMOKE),
        "source_hashes": {
            "driver": sha(SCRIPT), "runner": sha(PROJECT / "orch/engine/runner.py"),
            "runtime": sha(PROJECT / "orch/runtime.py"),
            "real_backend": sha(PROJECT / "orch/engine/real_backend.py"),
            "apply": sha(PROJECT / "orch/engine/apply.py"),
            "models": sha(PROJECT / "orch/contract/models.py"),
            "jobtag": sha(PROJECT / "orch/engine/jobtag.py"),
            "verification": sha(PROJECT / "orch/baselines/verification.py"),
            "validator": sha(PROJECT / "orch/recipes/adapters/validator.py"),
            "parser": sha(PROJECT / "orch/engine/dcp_parser.py"),
            "environment": sha(PROJECT / "recipes/support/env.sh"),
        },
    }


def run_cost(run: Any) -> dict[str, Any]:
    """Require original wall, CPU, RSS, load and occupied-slot cost fields."""
    values = {name: getattr(run, name) for name in (
        "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
        "load_1m_start", "load_1m_end", "busy_slots_start", "cache_hit",
        "original_run_id", "backpressure_events", "backpressure_wait_s",
    )}
    required = ("wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
                "load_1m_start", "load_1m_end", "busy_slots_start")
    if any(values[name] is None or not math.isfinite(values[name]) for name in required):
        raise ValueError("reserve run lacks cost telemetry")
    if values["cache_hit"] and not values["original_run_id"]:
        raise ValueError("reserve cache hit lacks original run identity")
    return values


def alias(path: Path, target: Path) -> None:
    """Publish a readable hard link after the journal has committed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha(path) != sha(target):
            raise ValueError("reserve DCP alias differs from canonical state")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.link")
    try:
        os.link(target, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Telemetry:
    """Check each distinct reserve output against its independent meter Evidence."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.root = ROOT / "telemetry/states"
        self.parser_sha256 = sha(PROJECT / "orch/engine/dcp_parser.py")
        self.extractor = DcpTelemetryExtractor(
            runner=runtime.runner, env_script=PROJECT / "recipes/support/env.sh",
            tool_version="2025.1", meter=runtime.meter,
        )

    def ensure(self, state: Any, evidence: Evidence) -> Path:
        folder = self.root / state.id
        result = folder / "result.json"
        telemetry = folder / "dcp_telemetry.json"
        with _file_lock(folder / "state.lock"):
            if result.is_file():
                record = read(result)
                if (record.get("status") != "ok"
                        or record.get("state_sha256") != state.id
                        or record.get("parser_sha256") != self.parser_sha256
                        or record.get("evidence_id") != evidence.id
                        or record.get("telemetry_sha256") != sha(telemetry)):
                    raise ValueError("saved reserve telemetry differs")
                return telemetry
            attempt = folder / "attempts" / uuid.uuid4().hex
            try:
                payload = self.extractor.extract(
                    state.path, run_dir=attempt, evidence=evidence,
                    design=DESIGN,
                    run_id=f"reserve-telemetry-{state.id[:12]}-{uuid.uuid4().hex[:8]}",
                )
                measured_wns = evidence.metric(MetricName.WNS_NS).value
                measured_luts = evidence.metric(MetricName.PACKED_LUTS).value
                status = ("ok" if (
                    payload["probe"]["timing"]["wns_ns"] == measured_wns
                    and payload["probe"]["utilization_summary"]["CLB LUTs"]["used"]
                    == measured_luts
                ) else "meter_mismatch")
                output = attempt / "dcp_telemetry.json"
                _atomic_bytes(telemetry, output.read_bytes())
                _atomic_bytes(result, _json_bytes({
                    "schema_version": "overnight-reserve-telemetry-v1",
                    "status": status, "state_sha256": state.id,
                    "parser_sha256": self.parser_sha256,
                    "evidence_id": evidence.id,
                    "telemetry_sha256": sha(telemetry),
                    "attempt_dir": str(attempt), "checked_at_utc": utc(),
                }))
                if status != "ok":
                    raise ValueError(f"reserve telemetry {status}: {state.id}")
                return telemetry
            except Exception as exc:
                if not result.is_file():
                    _atomic_bytes(result, _json_bytes({
                        "schema_version": "overnight-reserve-telemetry-v1",
                        "status": "error", "state_sha256": state.id,
                        "parser_sha256": self.parser_sha256,
                        "evidence_id": evidence.id, "attempt_dir": str(attempt),
                        "error": f"{type(exc).__name__}: {exc}",
                        "checked_at_utc": utc(),
                    }))
                raise

    def require(self, state: Any, evidence: Evidence) -> Path:
        folder = self.root / state.id
        result = read(folder / "result.json")
        telemetry = folder / "dcp_telemetry.json"
        if (result.get("status") != "ok" or result.get("state_sha256") != state.id
                or result.get("parser_sha256") != self.parser_sha256
                or result.get("evidence_id") != evidence.id
                or result.get("telemetry_sha256") != sha(telemetry)):
            raise ValueError("committed reserve output lacks matching telemetry")
        return telemetry


def raw_cost(run: Any) -> dict[str, Any] | None:
    """Keep partial run cost on a failed edge without claiming completeness."""
    if run is None:
        return None
    return {name: getattr(run, name) for name in (
        "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
        "load_1m_start", "load_1m_end", "busy_slots_start", "cache_hit",
        "original_run_id", "backpressure_events", "backpressure_wait_s",
    )}


class Reserve:
    """Journal each checked D2 edge and stop this design on the first failure."""

    def __init__(self, runtime: Any, frozen: dict[str, Any]) -> None:
        self.runtime = runtime
        self.inputs = frozen["source"]
        self.specs = selected(runtime)
        self.journal = Journal(ROOT)
        self.start = runtime.store.import_state(
            Path(self.inputs["b2_dcp"]), design=DESIGN, fake=False,
        )
        self.original = runtime.store.import_state(
            Path(self.inputs["original_dcp"]), design=DESIGN, fake=False,
        )
        if (self.start.id != self.inputs["b2_hash"]
                or self.original.id != self.inputs["original_hash"]):
            raise ValueError("reserve B2/original import differs")
        self.parents = {}
        for recipe, details in self.inputs["parents"].items():
            state = runtime.store.import_state(
                Path(details["dcp"]), design=DESIGN, fake=False,
            )
            if state.id != details["output_hash"]:
                raise ValueError(f"reserve parent import differs: {recipe}")
            self.parents[recipe] = state
        self.ledger = VerificationLedger(
            store=runtime.store, runner=runtime.runner, validator=runtime.validator,
            references={DESIGN: self.original.id},
        )
        self.telemetry = Telemetry(runtime)
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.stopped = False
        self.phase = "preparing"
        self.load_journal()
        self.alias_sources()
        self.repair_aliases()
        self.index()

    def alias_sources(self) -> None:
        """Expose the pinned B2 and ten D1 parents beside reserve outputs."""
        base = ROOT / "by_design" / DESIGN
        alias(base / "start_B2.dcp", self.runtime.store.state_path(self.start.id))
        for recipe in RECIPES:
            alias(base / "depth1" / f"{recipe}.dcp",
                  self.runtime.store.state_path(self.parents[recipe].id))

    def alias_path(self, row: dict[str, Any]) -> Path | None:
        """Use one stable filename for each ordered pair."""
        output_hash = row.get("output_hash")
        if not output_hash:
            return None
        first, second = row["sequence"]
        folder = "depth2" if row["status"] == "ok" else "failed"
        name = (f"{first}__{second}.dcp" if row["status"] == "ok"
                else f"{first}__{second}__{output_hash}.dcp")
        return ROOT / "by_design" / DESIGN / folder / name

    def load_journal(self) -> None:
        """Resume only exact source and checked edge identities."""
        events = self.journal.read(JOURNAL_ID)
        if not events:
            self.journal.append(JOURNAL_ID, {
                "id": "configuration", "type": "configuration",
                "plan_sha256": sha(PLAN), "at_utc": utc(),
            })
            return
        if (events[0].get("type") != "configuration"
                or events[0].get("plan_sha256") != sha(PLAN)):
            raise ValueError("reserve journal plan differs")
        for event in events[1:]:
            if event.get("type") != "edge":
                raise ValueError("unknown reserve journal event")
            row = event["edge"]
            sequence = tuple(row["sequence"])
            if sequence not in TASKS or sequence in self.rows:
                raise ValueError("duplicate or unknown reserve pair")
            self.rows[sequence] = row
            if row["status"] != "ok":
                self.stopped = True
                continue
            outcome = self.runtime.store.load_outcome(row["run_id"])
            parent = self.parents[sequence[0]]
            if (outcome.run.recipe_id != sequence[-1]
                    or outcome.run.recipe_version != self.specs[sequence[-1]].version
                    or outcome.run.input_sha256_before != parent.id
                    or outcome.run.input_sha256_after != parent.id
                    or outcome.run.status != RunStatus.OK
                    or outcome.state is None or outcome.evidence is None
                    or outcome.state.id != row["output_hash"]
                    or outcome.evidence.id != row["evidence_id"]
                    or outcome.evidence.fake
                    or outcome.evidence.routed is not True
                    or outcome.evidence.route_errors != 0
                    or outcome.evidence.drc_errors != 0
                    or outcome.evidence.constraint_fingerprint != self.inputs["constraint"]
                    or outcome.evidence.clock_digest != self.inputs["clock"]
                    or row.get("cost") != run_cost(outcome.run)):
                raise ValueError("committed reserve edge differs from stored run")
            require_metrics(outcome.evidence)
            amd = self.ledger.require_pass(DESIGN, outcome.state.id)
            if (row.get("amd_result") != "pass"
                    or row.get("amd_outputs_compared", 0) <= 0
                    or row["amd_outputs_compared"] != compared_outputs(amd)):
                raise ValueError("committed reserve edge lost AMD comparison proof")
            self.telemetry.require(outcome.state, outcome.evidence)

    def repair_aliases(self) -> None:
        """Recover a hard link after a journal commit interrupted publication."""
        for row in self.rows.values():
            expected = self.alias_path(row)
            if row.get("dcp_alias") != (str(expected) if expected else None):
                raise ValueError("saved reserve alias differs from naming rule")
            if expected is not None:
                alias(expected, self.runtime.store.state_path(row["output_hash"]))

    def write_status(self, error: str | None = None) -> None:
        """Publish counts and validity only."""
        telemetry_ok = sum(
            read(path).get("status") == "ok"
            for path in (ROOT / "telemetry/states").glob("*/result.json")
        )
        _atomic_bytes(STATUS, _json_bytes({
            "schema_version": "overnight-reserve-status-v1",
            "phase": self.phase, "updated_at_utc": utc(),
            "driver_pid": os.getpid(), "planned": len(TASKS),
            "completed": len(self.rows),
            "amd_pass": sum(row["status"] == "ok" for row in self.rows.values()),
            "telemetry_unique_ok": telemetry_ok,
            "failed": sum(row["status"] != "ok" for row in self.rows.values()),
            "stopped": self.stopped, "error": error,
        }))

    def one(self, sequence: tuple[str, str]) -> dict[str, Any]:
        """Run one second step; retain its DCP even if a gate rejects it."""
        first, second = sequence
        state = self.parents[first]
        proof = self.inputs["parents"][first]
        outcome = None
        amd = None
        telemetry = None
        outputs = 0
        error = None
        try:
            if (sha(Path(proof["dcp"])) != state.id
                    or sha(Path(proof["evidence"])) != proof["evidence_sha256"]
                    or sha(Path(proof["amd"])) != proof["amd_sha256"]
                    or sha(Path(proof["telemetry"])) != proof["telemetry_sha256"]):
                raise ValueError("reserve D1 parent proof changed")
            outcome = self.runtime.engine.execute(state, self.specs[second])
            if outcome.state is not None:
                amd = self.ledger.verify(DESIGN, outcome.state.id)
                outputs = compared_outputs(amd)
            if outcome.run.status != RunStatus.OK or outcome.state is None or outcome.evidence is None:
                raise ValueError(f"recipe or meter failed: {outcome.run.status.value}")
            if amd is None or amd.get("result") != "pass" or outputs <= 0:
                raise ValueError("AMD check did not PASS with compared outputs")
            evidence = outcome.evidence
            if (evidence.state_id != outcome.state.id or evidence.fake
                    or evidence.routed is not True or evidence.route_errors != 0
                    or evidence.drc_errors != 0
                    or evidence.constraint_fingerprint != self.inputs["constraint"]
                    or evidence.clock_digest != self.inputs["clock"]):
                raise ValueError("reserve output lacks B2-matching routed/DRC evidence")
            require_metrics(evidence)
            run_cost(outcome.run)
            if (outcome.run.input_sha256_before != state.id
                    or outcome.run.input_sha256_after != state.id):
                raise ValueError("reserve recipe mutated its input DCP")
            telemetry = self.telemetry.ensure(outcome.state, evidence)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        run = outcome.run if outcome else None
        output = outcome.state if outcome else None
        evidence = outcome.evidence if outcome else None
        return {
            "schema_version": "overnight-reserve-edge-v1",
            "design": DESIGN, "sequence": list(sequence),
            "status": "failed" if error else "ok", "error": error,
            "input_hash": state.id, "run_id": run.id if run else None,
            "run_status": run.status.value if run else None,
            "output_hash": output.id if output else None,
            "evidence_id": evidence.id if evidence else None,
            "amd_result": amd.get("result") if amd else None,
            "amd_outputs_compared": outputs,
            "amd_record": str(self.ledger._record_path(DESIGN, output.id))
            if amd and output else None,
            "telemetry_status": "ok" if telemetry else None,
            "telemetry_path": str(telemetry) if telemetry else None,
            "cost": raw_cost(run), "checked_at_utc": utc(),
        }

    def index(self) -> None:
        """Locate every source and output DCP without publishing outcomes."""
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow((
            "design", "depth", "sequence", "status", "run_id",
            "input_sha256", "output_sha256", "dcp_alias", "canonical_dcp",
            "amd_result", "amd_outputs_compared", "telemetry_status",
            "telemetry_json", "cache_hit", "original_run_id",
            "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
            "load_1m_start", "load_1m_end", "busy_slots_start",
        ))
        base = ROOT / "by_design" / DESIGN
        source_telemetry = TEST / "telemetry/states"
        writer.writerow((
            DESIGN, 0, "B2", "ok_source", "", "", self.start.id,
            str(base / "start_B2.dcp"),
            str(self.runtime.store.state_path(self.start.id)),
            "pass_source", "", "ok_source",
            str(source_telemetry / self.start.id / "dcp_telemetry.json"),
            "", "", "", "", "", "", "", "", "",
        ))
        for first in RECIPES:
            proof = self.inputs["parents"][first]
            state = self.parents[first]
            writer.writerow((
                DESIGN, 1, first, "ok_source", proof["run_id"], self.start.id,
                state.id, str(base / "depth1" / f"{first}.dcp"),
                str(self.runtime.store.state_path(state.id)),
                "pass_source", "", "ok_source", proof["telemetry"],
                "", "", "", "", "", "", "", "", "",
            ))
        for sequence in TASKS:
            row = self.rows.get(sequence)
            if row is None:
                continue
            cost = row.get("cost") or {}
            output_hash = row.get("output_hash")
            writer.writerow((
                DESIGN, 2, " > ".join(sequence), row["status"],
                row.get("run_id") or "", row["input_hash"], output_hash or "",
                row.get("dcp_alias") or "",
                str(self.runtime.store.state_path(output_hash)) if output_hash else "",
                row.get("amd_result") or "", row.get("amd_outputs_compared") or "",
                row.get("telemetry_status") or "", row.get("telemetry_path") or "",
                cost.get("cache_hit", ""), cost.get("original_run_id") or "",
                cost.get("wall_s", ""), cost.get("cpu_user_s", ""),
                cost.get("cpu_sys_s", ""), cost.get("peak_memory_bytes", ""),
                cost.get("load_1m_start", ""), cost.get("load_1m_end", ""),
                cost.get("busy_slots_start", ""),
            ))
        _atomic_bytes(ROOT / "by_design/index.csv", out.getvalue().encode("utf-8"))

    def commit(self, row: dict[str, Any]) -> None:
        sequence = tuple(row["sequence"])
        if sequence in self.rows:
            raise ValueError("reserve pair already committed")
        name = self.alias_path(row)
        row["dcp_alias"] = str(name) if name else None
        digest = hashlib.sha256(_json_bytes(list(sequence))).hexdigest()
        self.journal.append(JOURNAL_ID, {
            "id": f"edge-{digest}", "type": "edge",
            "edge": row, "at_utc": utc(),
        })
        self.rows[sequence] = row
        if name is not None:
            alias(name, self.runtime.store.state_path(row["output_hash"]))
        if row["status"] != "ok":
            self.stopped = True
        self.index()
        self.write_status()

    def finish(self) -> None:
        """Audit all successful rows and summarize counts and actual cost."""
        self.repair_aliases()
        for row in self.rows.values():
            if row["status"] != "ok":
                continue
            outcome = self.runtime.store.load_outcome(row["run_id"])
            if outcome.state is None or outcome.evidence is None:
                raise ValueError("reserve successful edge lost DCP or Evidence")
            self.ledger.require_pass(DESIGN, outcome.state.id)
            self.telemetry.require(outcome.state, outcome.evidence)
            if (row["amd_outputs_compared"] <= 0
                    or row["cost"] != run_cost(outcome.run)
                    or sha(Path(row["dcp_alias"])) != outcome.state.id):
                raise ValueError("reserve successful edge lost acceptance evidence")
        self.phase = ("complete" if len(self.rows) == len(TASKS) and not self.stopped
                      else "complete_with_failures")
        self.write_status()
        actual = list(ROOT.rglob("runner.json"))
        cpu = 0.0
        wall = 0.0
        missing_cpu = 0
        backpressure_events = 0
        for path in actual:
            process = read(path).get("process") or {}
            user, system = process.get("cpu_user_s"), process.get("cpu_sys_s")
            if user is None or system is None:
                missing_cpu += 1
            else:
                cpu += user + system
            wall += process.get("wall_s") or 0.0
            backpressure_events += process.get("backpressure_events") or 0
        _atomic_bytes(ROOT / "summary.json", _json_bytes({
            "schema_version": "overnight-reserve-summary-v1",
            "phase": self.phase, "planned": len(TASKS),
            "completed": len(self.rows),
            "amd_pass": sum(row["status"] == "ok" for row in self.rows.values()),
            "telemetry_unique_ok": len(list(
                (ROOT / "telemetry/states").glob("*/result.json"))),
            "source_states_with_test_telemetry": 11,
            "actual_tagged_tool_jobs": len(actual),
            "actual_tool_cpu_hours": round(cpu / 3600, 6),
            "actual_tool_wall_hours_summed": round(wall / 3600, 6),
            "tool_jobs_missing_cpu": missing_cpu,
            "backpressure_events": backpressure_events,
            "plan_sha256": sha(PLAN),
            "index_sha256": sha(ROOT / "by_design/index.csv"),
            "completed_at_utc": utc(),
        }))

    def run(self) -> None:
        """Run at most 24 edges; a failure blocks new work on this design."""
        self.phase = "depth2"
        self.write_status()
        pending = deque(sequence for sequence in TASKS if sequence not in self.rows)
        with ThreadPoolExecutor(max_workers=MAX_ACTIVE) as pool:
            active: dict[Any, tuple[str, str]] = {}

            def fill() -> None:
                while pending and not self.stopped and len(active) < MAX_ACTIVE:
                    sequence = pending.popleft()
                    active[pool.submit(self.one, sequence)] = sequence

            fill()
            while active:
                done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in done:
                    sequence = active.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "schema_version": "overnight-reserve-edge-v1",
                            "design": DESIGN, "sequence": list(sequence),
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "input_hash": self.parents[sequence[0]].id,
                            "run_id": None, "run_status": None,
                            "output_hash": None, "evidence_id": None,
                            "amd_result": None, "amd_outputs_compared": 0,
                            "amd_record": None, "telemetry_status": None,
                            "telemetry_path": None, "cost": None,
                            "checked_at_utc": utc(),
                        }
                    self.commit(row)
                fill()
        self.finish()


def prepare() -> tuple[Any, dict[str, Any]]:
    """Do all no-tool source checks before constructing a frozen plan."""
    inputs = source()
    runtime = make_real_runtime(
        ROOT, cache_enabled=True, max_slots=MAX_SLOTS,
        load_ceiling_1m=LOAD_CEILING,
        min_available_memory_bytes=MEMORY_FLOOR,
        latch_smoke_status=SMOKE,
    )
    return runtime, plan(runtime, inputs)


def run() -> None:
    """Require the exact plan and record even an early preflight failure."""
    try:
        runtime, frozen = prepare()
        if not PLAN.is_file() or PLAN.read_bytes() != _json_bytes(frozen):
            raise ValueError("reserve plan changed between launch and driver start")
        with _file_lock(ROOT / "driver.lock"):
            study = Reserve(runtime, frozen)
            study.run()
    except Exception as exc:
        previous = read(STATUS) if STATUS.is_file() else {}
        _atomic_bytes(STATUS, _json_bytes({
            "schema_version": "overnight-reserve-status-v1",
            "phase": "failed", "updated_at_utc": utc(),
            "driver_pid": os.getpid(), "planned": len(TASKS),
            "completed": previous.get("completed", 0),
            "amd_pass": previous.get("amd_pass", 0),
            "telemetry_unique_ok": previous.get("telemetry_unique_ok", 0),
            "failed": previous.get("failed", 0),
            "stopped": True, "error": f"{type(exc).__name__}: {exc}",
        }))
        raise


def driver_alive(pid: int) -> bool:
    """Match the recorded exact driver argv."""
    try:
        info = _stat(pid)
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return bool(info and info.state != "Z"
                    and str(SCRIPT).encode() in args and b"--run" in args)
    except OSError:
        return False


def targets() -> tuple[set[str], set[int]]:
    """Find only exact tags with a cwd inside this reserve data root."""
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
    """Signal only identified groups, then sweep their exact run tags."""
    tags, groups = targets()
    result = {
        "schema_version": "overnight-reserve-cleanup-v1",
        "exact_tags": sorted(tags), "groups": sorted(groups),
        "started_at_utc": utc(),
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
    result["tag_sweeps"] = {
        tag: list(sweep_tag(tag, grace_s=5).alive) for tag in sorted(tags)
    }
    result["survivors"] = {
        tag: [item.pid for item in tagged(tag)]
        for tag in sorted(tags) if tagged(tag)
    }
    result["finished_at_utc"] = utc()
    _atomic_bytes(ROOT / "watchdog_cleanup_audit.json", _json_bytes(result))
    return result


def watch() -> None:
    """Clean only this study's exact tags on abnormal driver exit."""
    pid = read(LAUNCH)["driver_pid"]
    if not driver_alive(pid):
        state = read(STATUS) if STATUS.is_file() else {}
        if state.get("phase") not in ("complete", "complete_with_failures"):
            cleanup()
        return
    deadline = time.monotonic() + 36 * 3600
    while time.monotonic() < deadline:
        if not driver_alive(pid):
            state = read(STATUS) if STATUS.is_file() else {}
            if state.get("phase") not in ("complete", "complete_with_failures"):
                cleanup()
            return
        time.sleep(15)
    raise RuntimeError("reserve watchdog lifetime ended with driver alive")


def launch() -> dict[str, Any]:
    """Launch detached only after the complete, frozen priority-one gate."""
    runtime, frozen = prepare()
    load, available = Runner._resource_snapshot()
    if load > LOAD_CEILING or available < MEMORY_FLOOR:
        raise RuntimeError("load or available memory blocks reserve launch")
    if LAUNCH.exists():
        raise FileExistsError("reserve study already launched")
    if PLAN.exists():
        if PLAN.read_bytes() != _json_bytes(frozen):
            raise ValueError("existing reserve plan differs")
    else:
        _atomic_bytes(PLAN, _json_bytes(frozen))
    with _file_lock(ROOT / "launch.lock"):
        if LAUNCH.exists():
            raise FileExistsError("reserve study already launched")
        with (ROOT / "driver.stdout").open("wb") as out, (ROOT / "driver.stderr").open("wb") as err:
            driver = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--run"], cwd=PROJECT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        manifest = {
            "schema_version": "overnight-reserve-launch-v1",
            "driver_pid": driver.pid, "started_at_utc": utc(),
            "plan_sha256": sha(PLAN), "planned": len(TASKS),
            "active_limit": MAX_ACTIVE, "load_1m_at_launch": load,
            "available_memory_bytes_at_launch": available,
        }
        _atomic_bytes(LAUNCH, _json_bytes(manifest))
        try:
            with (ROOT / "watchdog.stdout").open("wb") as out, (ROOT / "watchdog.stderr").open("wb") as err:
                watcher = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "--watch"], cwd=PROJECT,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    start_new_session=True, close_fds=True,
                )
            manifest["watchdog_pid"] = watcher.pid
            _atomic_bytes(LAUNCH, _json_bytes(manifest))
            return manifest
        except Exception:
            if driver_alive(driver.pid):
                group = os.getpgid(driver.pid)
                if group == driver.pid and group != os.getpgrp():
                    os.killpg(group, signal.SIGTERM)
            time.sleep(3)
            cleanup()
            raise


def controller_status(phase: str, **details: Any) -> None:
    """Record why the conditional controller waited, skipped or launched."""
    ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(CONTROLLER_STATUS, _json_bytes({
        "schema_version": "overnight-reserve-controller-status-v1",
        "phase": phase, "updated_at_utc": utc(),
        "controller_pid": os.getpid(), **details,
    }))


def controller() -> None:
    """Wait without tool jobs, then act only on the frozen eligibility rule."""
    frozen = read(CONTROLLER_PLAN)
    if (frozen.get("script_sha256") != sha(SCRIPT)
            or frozen.get("test_plan_sha256") != TEST_PLAN_SHA
            or frozen.get("deadline_utc") != DEADLINE.isoformat()):
        raise ValueError("reserve controller source or gate changed")
    with _file_lock(ROOT / "controller.lock"):
        while datetime.now(timezone.utc) < DEADLINE:
            decision = gate()
            if decision.startswith("blocked"):
                controller_status(decision)
                _atomic_bytes(STATUS, _json_bytes({
                    "schema_version": "overnight-reserve-status-v1",
                    "phase": decision, "planned": len(TASKS),
                    "completed": 0, "updated_at_utc": utc(),
                }))
                return
            if decision.startswith("skipped"):
                controller_status(decision)
                _atomic_bytes(STATUS, _json_bytes({
                    "schema_version": "overnight-reserve-status-v1",
                    "phase": decision, "planned": len(TASKS),
                    "completed": 0, "updated_at_utc": utc(),
                }))
                return
            if decision == "eligible":
                load, available = Runner._resource_snapshot()
                if load <= LOAD_CEILING and available >= MEMORY_FLOOR:
                    result = launch()
                    controller_status("launched", driver_pid=result["driver_pid"],
                                      plan_sha256=result["plan_sha256"])
                    return
                controller_status("waiting_for_resources", load_1m=load,
                                  available_memory_bytes=available)
            else:
                controller_status(decision)
            time.sleep(60)
        controller_status("expired_at_07_without_launch")


def arm() -> dict[str, Any]:
    """Detach a zero-tool-job controller before priority one has finished."""
    ROOT.mkdir(parents=True, exist_ok=True)
    if CONTROLLER_PLAN.exists() or (ROOT / "controller_launch.json").exists():
        raise FileExistsError("reserve controller is already armed")
    frozen = {
        "schema_version": "overnight-reserve-controller-plan-v1",
        "script_sha256": sha(SCRIPT), "test_plan_sha256": TEST_PLAN_SHA,
        "deadline_utc": DEADLINE.isoformat(), "planned": len(TASKS),
        "max_shared_slots": MAX_SLOTS,
    }
    _atomic_bytes(CONTROLLER_PLAN, _json_bytes(frozen))
    with (ROOT / "controller.stdout").open("wb") as out, (ROOT / "controller.stderr").open("wb") as err:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--wait"], cwd=PROJECT,
            stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            start_new_session=True, close_fds=True,
        )
    manifest = {
        "schema_version": "overnight-reserve-controller-launch-v1",
        "controller_pid": process.pid, "started_at_utc": utc(),
        "script_sha256": sha(SCRIPT),
    }
    _atomic_bytes(ROOT / "controller_launch.json", _json_bytes(manifest))
    return manifest


def short_status() -> dict[str, Any]:
    """Return only counts, validity and resource status."""
    state = read(STATUS) if STATUS.is_file() else {"phase": "not_launched"}
    ctl = read(CONTROLLER_STATUS) if CONTROLLER_STATUS.is_file() else {}
    manifest = read(LAUNCH) if LAUNCH.is_file() else {}
    pid = manifest.get("driver_pid")
    load, available = Runner._resource_snapshot()
    return {
        "phase": state.get("phase"), "controller_phase": ctl.get("phase"),
        "completed": state.get("completed", 0), "planned": len(TASKS),
        "amd_pass": state.get("amd_pass", 0),
        "telemetry_unique_ok": state.get("telemetry_unique_ok", 0),
        "failed": state.get("failed", 0), "error": state.get("error"),
        "driver_pid": pid,
        "driver_alive": driver_alive(pid) if type(pid) is int else False,
        "load_1m": load, "available_memory_bytes": available,
        "busy_shared_tool_slots": Runner(PROJECT / "data/slots", max_slots=MAX_SLOTS)._busy_slots(),
        "driver_stderr_bytes": (ROOT / "driver.stderr").stat().st_size if manifest else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--arm", action="store_true")
    mode.add_argument("--wait", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({
            "design": DESIGN, "planned_pairs": len(TASKS),
            "eligibility": gate(), "tool_jobs_launched": 0,
        }, sort_keys=True))
    elif args.arm:
        print(json.dumps(arm(), sort_keys=True))
    elif args.wait:
        try:
            controller()
        except Exception as exc:
            controller_status("failed", error=f"{type(exc).__name__}: {exc}")
            if not LAUNCH.is_file():
                _atomic_bytes(STATUS, _json_bytes({
                    "schema_version": "overnight-reserve-status-v1",
                    "phase": "controller_failed", "planned": len(TASKS),
                    "completed": 0, "updated_at_utc": utc(),
                    "error": f"{type(exc).__name__}: {exc}",
                }))
            raise
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
