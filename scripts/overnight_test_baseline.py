"""Detached, verified held-out D1/C0/D2 baseline with per-state DCP telemetry.

Only the seven named test B2 starts enter this campaign. The room screen is
pre-registered and alone compares test outcome metrics. All actual tool
commands remain in the existing runner, adapters, meter and copied LATCH Tcl.
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
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from orch.baselines.test_room_screen import METRICS, noise_bands, screen
from orch.baselines.verification import VerificationLedger
from orch.config import load_recipe_sets
from orch.contract.models import DesignState, Evidence, MetricName, RunOutcome, RunStatus
from orch.engine.dcp_parser import DcpTelemetryExtractor
from orch.engine.jobtag import _own_pids, _stat, _tag_of, group_members, sweep_tag, tagged
from orch.engine.journal import Journal
from orch.engine.runner import Runner
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from orch.runtime import make_real_runtime

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/test_baseline_overnight_2026-11-04.yaml"
CONFIG_SHA256 = "d8f62333efe9e5fd3a865d94806faf7fd4ea00340d37bd4b1f4f01e2d2ee66d6"
ROOT = PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04"
NOISE_ROOT = PROJECT / "data/exp_evening_2026-11-04/test_noise_family"
B2_ROOT = PROJECT / "data/exp_evening_2026-11-04/test_b2"
LATCH_SMOKE = PROJECT / "data/scratch_overnight_2026-11-04/cache_gate_smoke_20261104/status.json"
SCRIPT = Path(__file__).resolve()
JOURNAL_ID = "overnight_test_baseline_2026-11-04"
PLAN = ROOT / "plan.json"
LAUNCH = ROOT / "launch.json"
STATUS = ROOT / "status.json"
ADMISSION = ROOT / "admitted_test_designs_2026-11-04.json"
NOISE_BANDS = ROOT / "noise_bands_2026-11-04.json"
ORDER = (
    "rosetta_digit-recognition", "corescore_500_mod",
    "vexriscv_re-place_v2", "fir_symmetric_systolic",
    "fir_systolic_transposed", "rosetta_3d-rendering_v2",
    "amd_mini-isp_v2",
)
TIMING = (
    "LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock",
    "LATCH.hardblock_relocate", "LATCH.memory_path_opt",
)
AREA = ("A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED", "A_E_DUPL_Q1000")
SELECTED = TIMING + AREA
C0 = "C0_reroute"
CONTROLS = ("n_repeat", "n_thr4", "n_unc0.001", "n_unc0.002", "n_unc0.003")
REQUIRED_METRICS = (
    MetricName.WNS_NS, MetricName.TNS_NS, MetricName.PACKED_LUTS,
    MetricName.NETLIST_LUTS, MetricName.FFS, MetricName.DYNAMIC_POWER_W,
)
SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


def utc() -> str:
    """UTC time for immutable plan and progress records."""
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    """Read one JSON object from an existing saved record."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metric(evidence: Evidence, name: MetricName) -> float:
    """Require one finite independent meter value."""
    item = evidence.metric(name)
    if item is None or not math.isfinite(item.value):
        raise ValueError(f"meter lacks finite {name.value}")
    return item.value


def _four(evidence: Evidence) -> dict[str, float]:
    """Timing/area values used only by the frozen noise and room screen."""
    return {name: _metric(evidence, MetricName(name)) for name in METRICS}


def _config() -> dict[str, Any]:
    """Require the exact room rule and recipe allowlist frozen before test runs."""
    if _sha256_file(CONFIG) != CONFIG_SHA256:
        raise ValueError("test campaign config changed after the pre-recipe freeze")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("test campaign config is not a mapping")
    if (
        config.get("schema_version") != "test-baseline-overnight-v1"
        or tuple(config.get("candidate_order", ())) != ORDER
        or tuple(config.get("selected_workers", ())) != ("timing", "area")
        or config.get("c0_recipe_id") != C0
        or config.get("room_screen", {}).get("max_admitted") != 5
        or tuple(config["room_screen"].get("control_variants", ())) != CONTROLS
        or config["room_screen"].get("admit_if")
        != "any_selected_one_step_strictly_exceeds_either_band"
        or config["room_screen"].get("band_method") != "max_abs_control_minus_b2"
        or config.get("resource_limits", {}).get("total_shared_tool_slots") != 96
        or config["resource_limits"].get("pause_new_tool_launch_if_load_1m_above") != 180
        or config["resource_limits"].get("pause_new_tool_launch_if_available_memory_bytes_below")
        != 200_000_000_000
        or config.get("reporting", {}).get("allow_test_outcome_tables") is not False
    ):
        raise ValueError("frozen test scope, screen or resource rule differs")
    selected = load_recipe_sets(PROJECT / config["recipe_sets_source"])
    selected.require_signed_off()
    if (
        tuple(selected.worker_sets["timing"]) != TIMING
        or tuple(selected.worker_sets["area"]) != AREA
        or tuple(selected.worker_sets["power"]) != ()
        or set(selected.worker_sets) != {"timing", "area", "power"}
    ):
        raise ValueError("signed worker recipe sets differ from the ten selected IDs")
    declared = (PROJECT / config["test_data_root"]).resolve()
    if declared != ROOT or not ROOT.is_relative_to(PROJECT / "data"):
        raise ValueError("test campaign data root escaped its frozen Development path")
    if (PROJECT / config["noise_source"]).resolve() != NOISE_ROOT:
        raise ValueError("noise source differs from frozen plan")
    if (PROJECT / config["b2_source"]).resolve() != B2_ROOT:
        raise ValueError("test B2 source differs from frozen plan")
    return config


def _sources() -> dict[str, dict[str, Any]]:
    """Recheck seven exact B2/contest inputs and prior B2 AMD PASS records."""
    noise_plan = _json(NOISE_ROOT / "plan.json")
    b2_plan = _json(B2_ROOT / "plan.json")
    b2_status = _json(B2_ROOT / "status.json")
    if (
        set(noise_plan.get("test_designs", ())) != set(ORDER)
        or tuple(noise_plan.get("variants", ())) != CONTROLS
        or b2_status.get("phase") != "complete"
        or set(b2_plan.get("new_designs", ())) | set(b2_plan.get("existing_designs", ()))
        != set(ORDER)
    ):
        raise ValueError("recorded B2/noise sources differ from seven test candidates")
    rows: dict[str, dict[str, Any]] = {}
    for design in ORDER:
        row = noise_plan["references"][design]
        b2 = Path(row["b2"]).resolve(strict=True)
        original = Path(row["original"]).resolve(strict=True)
        evidence_path = Path(row["b2_evidence_path"]).resolve(strict=True)
        if (
            _sha256_file(b2) != row["b2_sha256"]
            or _sha256_file(original) != row["original_sha256"]
            or _sha256_file(evidence_path) != row["b2_evidence_sha256"]
            or not original.is_relative_to((Path(os.environ.get("ORCH_BENCHMARK_DIR", "benchmarks")) / "fpl26_contest_v1.3.0"))
        ):
            raise ValueError(f"recorded test B2, original or Evidence changed: {design}")
        old = Evidence.model_validate_json(evidence_path.read_bytes())
        if (
            old.state_id != row["b2_sha256"]
            or old.fake or old.routed is not True
            or old.route_errors != 0 or old.drc_errors != 0
            or old.constraint_fingerprint != row["b2_constraint_fingerprint"]
            or old.clock_digest != row["b2_clock_digest"]
        ):
            raise ValueError(f"recorded test B2 Evidence is invalid: {design}")
        _four(old)
        checks = b2_status["results"]
        check = checks["new_validation"].get(design) or checks["existing_validation"].get(design)
        if (
            not isinstance(check, dict)
            or check.get("status") != "pass"
            or check.get("original_sha256") != row["original_sha256"]
            or check.get("revised_sha256") != row["b2_sha256"]
            or check.get("outputs_compared", 0) <= 0
            or check.get("precheck_cycles") != 100
            or check.get("full_cycles") != 200
            or _sha256_file(Path(check["validator_report_path"]))
            != check["validator_report_sha256"]
        ):
            raise ValueError(f"recorded B2 lacks its prior AMD PASS: {design}")
        rows[design] = {
            "design": design,
            "original": str(original), "original_sha256": row["original_sha256"],
            "b2": str(b2), "b2_sha256": row["b2_sha256"],
            "b2_evidence_path": str(evidence_path),
            "b2_evidence_sha256": row["b2_evidence_sha256"],
            "b2_constraint_fingerprint": row["b2_constraint_fingerprint"],
            "b2_clock_digest": row["b2_clock_digest"],
            "b2_amd_report": check["validator_report_path"],
            "b2_amd_report_sha256": check["validator_report_sha256"],
        }
    return rows


def _plan(runtime: Any, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pin code, inputs, recipes and cache gate before detached execution."""
    return {
        "schema_version": "overnight-test-baseline-plan-v1",
        "config_sha256": CONFIG_SHA256,
        "candidate_order": list(ORDER),
        "source_b2_plan_sha256": _sha256_file(B2_ROOT / "plan.json"),
        "source_b2_status_sha256": _sha256_file(B2_ROOT / "status.json"),
        "source_noise_plan_sha256": _sha256_file(NOISE_ROOT / "plan.json"),
        "source_noise_plan_max_jobs": _json(NOISE_ROOT / "plan.json")["max_concurrent_extra_tool_jobs"],
        "inputs": sources,
        "selected_recipe_ids": list(SELECTED),
        "selected_recipe_versions": {
            recipe_id: runtime.registry.get(recipe_id).version
            for recipe_id in (*SELECTED, C0)
        },
        "c0_is_control": True,
        "planned_c0_edges": 7,
        "planned_d1_edges": 70,
        "planned_d2_edges_if_five_admitted": 500,
        "max_shared_tool_slots": 96,
        "load_ceiling_1m": 180.0,
        "available_memory_floor_bytes": 200_000_000_000,
        "meter_version": runtime.meter.version,
        "latch_cache_smoke_sha256": _sha256_file(LATCH_SMOKE),
        "latch_cache_smoke_status": str(LATCH_SMOKE),
        "source_hashes": {
            "driver": _sha256_file(SCRIPT),
            "room_screen": _sha256_file(PROJECT / "orch/baselines/test_room_screen.py"),
            "runtime": _sha256_file(PROJECT / "orch/runtime.py"),
            "runner": _sha256_file(PROJECT / "orch/engine/runner.py"),
            "real_backend": _sha256_file(PROJECT / "orch/engine/real_backend.py"),
            "apply": _sha256_file(PROJECT / "orch/engine/apply.py"),
            "models": _sha256_file(PROJECT / "orch/contract/models.py"),
            "validator_adapter": _sha256_file(PROJECT / "orch/recipes/adapters/validator.py"),
            "verification": _sha256_file(PROJECT / "orch/baselines/verification.py"),
            "jobtag": _sha256_file(PROJECT / "orch/engine/jobtag.py"),
            "dcp_parser_v3": _sha256_file(PROJECT / "orch/engine/dcp_parser.py"),
            "environment": _sha256_file(PROJECT / "recipes/support/env.sh"),
        },
    }


def _evidence_signature(evidence: Evidence) -> str:
    """Same DCP bytes must have the same independent meter measurements."""
    payload = {
        "meter_version": evidence.meter_version,
        "routed": evidence.routed,
        "route_errors": evidence.route_errors,
        "drc_errors": evidence.drc_errors,
        "constraint_fingerprint": evidence.constraint_fingerprint,
        "clock_digest": evidence.clock_digest,
        "metrics": {
            name.value: evidence.metrics[name].model_dump(mode="json")
            for name in sorted(evidence.metrics, key=lambda value: value.value)
        },
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


class TelemetryLedger:
    """Probe each distinct test DCP once, retaining mismatches and raw reports."""

    def __init__(self, runtime: Any) -> None:
        self.root = ROOT / "telemetry"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime
        self.parser_hash = _sha256_file(PROJECT / "orch/engine/dcp_parser.py")
        self.extractor = DcpTelemetryExtractor(
            runner=runtime.runner,
            env_script=PROJECT / "recipes/support/env.sh",
            tool_version="2025.1", meter=runtime.meter,
        )

    def ensure(self, state: DesignState, evidence: Evidence) -> Path:
        """Require exact WNS and packed LUT agreement before an edge commits."""
        if evidence.state_id != state.id or evidence.meter_version != self.runtime.meter.version:
            raise ValueError("telemetry input Evidence differs from state or current meter")
        signature = _evidence_signature(evidence)
        folder = self.root / "states" / state.id
        record_path = folder / "result.json"
        final_path = folder / "dcp_telemetry.json"
        with _file_lock(folder / "state.lock"):
            if record_path.is_file():
                record = _json(record_path)
                if (
                    record.get("state_sha256") != state.id
                    or record.get("design") != state.design
                    or record.get("meter_signature_sha256") != signature
                    or record.get("parser_sha256") != self.parser_hash
                    or record.get("status") != "ok"
                    or not final_path.is_file()
                    or _sha256_file(final_path) != record.get("telemetry_sha256")
                ):
                    raise ValueError(f"saved telemetry is incomplete or differs: {state.id}")
                return final_path
            attempt = folder / "attempts" / uuid.uuid4().hex
            try:
                payload = self.extractor.extract(
                    state.path, run_dir=attempt, evidence=evidence,
                    design=state.design,
                    run_id=f"test-telemetry-{state.id[:12]}-{uuid.uuid4().hex[:8]}",
                )
                observed_wns = payload["probe"]["timing"]["wns_ns"]
                observed_luts = payload["probe"]["utilization_summary"]["CLB LUTs"]["used"]
                if observed_wns != _metric(evidence, MetricName.WNS_NS):
                    status = "meter_mismatch_wns"
                elif observed_luts != _metric(evidence, MetricName.PACKED_LUTS):
                    status = "meter_mismatch_packed_luts"
                else:
                    status = "ok"
                source_json = attempt / "dcp_telemetry.json"
                _atomic_bytes(final_path, source_json.read_bytes())
                record = {
                    "schema_version": "test-dcp-telemetry-result-v1",
                    "state_sha256": state.id, "design": state.design,
                    "meter_signature_sha256": signature,
                    "parser_sha256": self.parser_hash,
                    "status": status, "attempt_dir": str(attempt),
                    "telemetry_path": str(final_path),
                    "telemetry_sha256": _sha256_file(final_path),
                    "checked_at_utc": utc(),
                }
                _atomic_bytes(record_path, _json_bytes(record))
                if status != "ok":
                    raise ValueError(f"telemetry {status}: {state.design} {state.id}")
                return final_path
            except Exception as exc:
                if not record_path.is_file():
                    _atomic_bytes(record_path, _json_bytes({
                        "schema_version": "test-dcp-telemetry-result-v1",
                        "state_sha256": state.id, "design": state.design,
                        "meter_signature_sha256": signature,
                        "parser_sha256": self.parser_hash,
                        "status": "error", "attempt_dir": str(attempt),
                        "error": f"{type(exc).__name__}: {exc}",
                        "checked_at_utc": utc(),
                    }))
                raise


    def require(self, state: DesignState, evidence: Evidence) -> Path:
        """Read an already checked telemetry result; never launch during resume audit."""
        folder = self.root / "states" / state.id
        record_path = folder / "result.json"
        final_path = folder / "dcp_telemetry.json"
        if not record_path.is_file() or not final_path.is_file():
            raise ValueError(f"committed state lacks telemetry: {state.id}")
        record = _json(record_path)
        if (
            record.get("state_sha256") != state.id
            or record.get("design") != state.design
            or record.get("meter_signature_sha256") != _evidence_signature(evidence)
            or record.get("parser_sha256") != self.parser_hash
            or record.get("status") != "ok"
            or record.get("telemetry_sha256") != _sha256_file(final_path)
        ):
            raise ValueError(f"committed telemetry differs: {state.id}")
        return final_path


def _component(value: str) -> str:
    """Constrain readable alias names to one path component."""
    if value in (".", "..") or not SAFE.fullmatch(value):
        raise ValueError(f"unsafe design or recipe alias component: {value!r}")
    return value


def _alias(path: Path, target: Path) -> None:
    """Atomically expose a canonical private DCP under a readable relative name."""
    if not target.is_relative_to(ROOT / "states"):
        raise ValueError("DCP alias target escaped the test campaign state store")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not path.is_symlink():
        raise ValueError(f"existing DCP alias is not a symlink: {path}")
    if path.is_symlink() and path.resolve(strict=True) == target:
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.symlink_to(os.path.relpath(target, path.parent))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _outputs_compared(record: dict[str, Any]) -> int:
    """Require the observed AMD log to compare at least one output."""
    attempts = record.get("attempts") or []
    if not attempts:
        return 0
    log_path = Path(attempts[-1]["run_dir"]) / "validate.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return max((int(item) for item in re.findall(r"Outputs\s*\((\d+)\):", log)), default=0)


def _cost_fields(run: Any) -> dict[str, Any]:
    """Retain raw cost fields even when an edge fails its acceptance checks."""
    return {
        "wall_s": run.wall_s, "cpu_user_s": run.cpu_user_s,
        "cpu_sys_s": run.cpu_sys_s,
        "peak_memory_bytes": run.peak_memory_bytes,
        "load_1m_start": run.load_1m_start,
        "load_1m_end": run.load_1m_end,
        "busy_slots_start": run.busy_slots_start,
        "backpressure_wait_s": run.backpressure_wait_s,
        "backpressure_events": run.backpressure_events,
        "cache_hit": run.cache_hit,
        "original_run_id": run.original_run_id,
    }


def _cost(run: Any) -> dict[str, Any]:
    """Record charged original cost even when this logical edge hits the cache."""
    fields = (
        run.wall_s, run.cpu_user_s, run.cpu_sys_s,
        run.peak_memory_bytes, run.load_1m_start,
        run.load_1m_end, run.busy_slots_start,
    )
    if any(value is None or not math.isfinite(value) for value in fields):
        raise ValueError("run cost lacks wall, CPU, peak RSS, load or busy slots")
    if run.cache_hit and not run.original_run_id:
        raise ValueError("cache hit lacks its original run ID and charged cost")
    return _cost_fields(run)


class Campaign:
    """Journal verified edges only after AMD, meter, digest and telemetry gates."""

    def __init__(
        self, config: dict[str, Any], plan: dict[str, Any],
        runtime: Any, sources: dict[str, dict[str, Any]],
    ) -> None:
        self.config = config
        self.plan = plan
        self.runtime = runtime
        self.sources = sources
        self.journal = Journal(ROOT)
        self.telemetry = TelemetryLedger(runtime)
        self.starts = {
            design: runtime.store.import_state(
                Path(sources[design]["b2"]), design=design, fake=False
            )
            for design in ORDER
        }
        for design, state in self.starts.items():
            if state.id != sources[design]["b2_sha256"]:
                raise ValueError(f"test start import hash differs: {design}")
        original_states = {
            design: runtime.store.import_state(
                Path(sources[design]["original"]), design=design, fake=False
            )
            for design in ORDER
        }
        if any(
            original_states[design].id != sources[design]["original_sha256"]
            for design in ORDER
        ):
            raise ValueError("contest original changed while importing private reference")
        self.ledger = VerificationLedger(
            store=runtime.store, runner=runtime.runner,
            validator=runtime.validator,
            references={
                design: sources[design]["original_sha256"]
                for design in ORDER
            },
        )
        self.recipes = {
            recipe_id: runtime.registry.get(recipe_id)
            for recipe_id in (*SELECTED, C0)
        }
        self.start_evidence: dict[str, Evidence] = {}
        self.edges: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
        self.stopped: dict[str, str] = {}
        self.phase = "preparing"
        self._load_journal()

    def _load_journal(self) -> None:
        """Resume only an exact plan and previously checked edge identities."""
        events = self.journal.read(JOURNAL_ID)
        plan_sha = _sha256_file(PLAN)
        if not events:
            self.journal.append(JOURNAL_ID, {
                "id": "configuration", "type": "configuration",
                "plan_sha256": plan_sha, "at_utc": utc(),
            })
            return
        if events[0].get("type") != "configuration" or events[0].get("plan_sha256") != plan_sha:
            raise ValueError("test journal configuration differs from immutable plan")
        for event in events[1:]:
            kind = event.get("type")
            if kind == "start_failure":
                design = event["design"]
                if design not in ORDER or design in self.stopped:
                    raise ValueError("invalid or duplicate saved start failure")
                self.stopped[design] = event["reason"]
            elif kind == "edge":
                row = event["edge"]
                design = row["design"]
                sequence = tuple(row["sequence"])
                if (
                    design not in ORDER
                    or not 1 <= len(sequence) <= 2
                    or sequence[-1] not in self.recipes
                    or (len(sequence) == 2 and any(item not in SELECTED for item in sequence))
                ):
                    raise ValueError("journal edge escaped test design or recipe scope")
                key = design, sequence
                if key in self.edges:
                    raise ValueError("duplicate test sequence in journal")
                self.edges[key] = row
                if row["status"] != "ok":
                    self.stopped[design] = row.get("error", "saved failed edge")
                    continue
                outcome = self.runtime.store.load_outcome(row["run_id"])
                if (
                    outcome.run.recipe_id != sequence[-1]
                    or outcome.run.recipe_version != self.recipes[sequence[-1]].version
                    or outcome.run.input_sha256_before != row["input_sha256"]
                    or outcome.run.input_sha256_after != row["input_sha256"]
                    or (outcome.state.id if outcome.state else None) != row["output_sha256"]
                    or (outcome.evidence.id if outcome.evidence else None) != row["evidence_id"]
                    or outcome.run.status != RunStatus.OK
                    or outcome.evidence is None
                    or outcome.state is None
                    or outcome.evidence.fake
                    or outcome.evidence.routed is not True
                    or outcome.evidence.route_errors != 0
                    or outcome.evidence.drc_errors != 0
                    or outcome.evidence.constraint_fingerprint
                    != self.sources[design]["b2_constraint_fingerprint"]
                    or outcome.evidence.clock_digest
                    != self.sources[design]["b2_clock_digest"]
                ):
                    raise ValueError("saved test edge differs from its completed run")
                amd = self.ledger.require_pass(design, outcome.state.id)
                if (
                    row.get("amd_result") != "pass"
                    or row.get("amd_outputs_compared", 0) <= 0
                    or _outputs_compared(amd) != row["amd_outputs_compared"]
                    or row.get("cost") != _cost(outcome.run)
                ):
                    raise ValueError("saved test edge lost its AMD output or cost evidence")
                self.telemetry.require(outcome.state, outcome.evidence)
            else:
                raise ValueError(f"unknown test journal event: {kind}")

    def _write_status(self, *, error: str | None = None) -> None:
        """Expose counts and validity only, without test objective outcomes."""
        by_design = {}
        for design in ORDER:
            rows = [row for (name, _), row in self.edges.items() if name == design]
            by_design[design] = {
                "c0_pass": sum(row["phase"] == "C0" and row["status"] == "ok" for row in rows),
                "d1_pass": sum(row["phase"] == "D1" and row["status"] == "ok" for row in rows),
                "d2_pass": sum(row["phase"] == "D2" and row["status"] == "ok" for row in rows),
                "failed_edges": sum(row["status"] != "ok" for row in rows),
                "stopped": design in self.stopped,
            }
        telemetry_ok = sum(
            _json(path).get("status") == "ok"
            for path in (ROOT / "telemetry/states").glob("*/result.json")
        )
        payload = {
            "schema_version": "overnight-test-baseline-status-v1",
            "phase": self.phase,
            "driver_pid": os.getpid(),
            "updated_at_utc": utc(),
            "planned": {"C0": 7, "D1": 70, "D2_after_screen": 500},
            "by_design": by_design,
            "completed_edges": len(self.edges),
            "amd_pass_edges": sum(row["status"] == "ok" for row in self.edges.values()),
            "telemetry_unique_ok": telemetry_ok,
            "stopped_designs": sorted(self.stopped),
            "errors": {
                design: reason for design, reason in sorted(self.stopped.items())
            },
            "error": error,
        }
        _atomic_bytes(STATUS, _json_bytes(payload))

    def _phase(self, phase: str) -> None:
        self.phase = phase
        self._write_status()

    def _start_one(self, design: str) -> Evidence:
        """Re-measure B2 with the current meter and probe it before recipes."""
        state = self.starts[design]
        prior_path = Path(self.sources[design]["b2_evidence_path"])
        if _sha256_file(prior_path) != self.sources[design]["b2_evidence_sha256"]:
            raise ValueError("recorded B2 meter Evidence changed")
        prior = Evidence.model_validate_json(prior_path.read_bytes())
        fresh = self.runtime.engine.measure_start(state)
        if (
            fresh.state_id != state.id or fresh.fake
            or fresh.routed is not True or fresh.route_errors != 0
            or fresh.drc_errors != 0
            or fresh.constraint_fingerprint != self.sources[design]["b2_constraint_fingerprint"]
            or fresh.clock_digest != self.sources[design]["b2_clock_digest"]
            or _four(fresh) != _four(prior)
        ):
            raise ValueError("current B2 meter disagrees with recorded B2 or digests")
        for metric in REQUIRED_METRICS:
            _metric(fresh, metric)
        telemetry_path = self.telemetry.ensure(state, fresh)
        result = {
            "schema_version": "overnight-test-start-check-v1",
            "design": design, "status": "ok", "b2_sha256": state.id,
            "old_evidence_sha256": self.sources[design]["b2_evidence_sha256"],
            "new_evidence_id": fresh.id,
            "new_meter_version": fresh.meter_version,
            "old_new_four_metrics_equal": True,
            "constraint_and_clock_equal": True,
            "b2_amd_source_report": self.sources[design]["b2_amd_report"],
            "b2_amd_source_report_sha256": self.sources[design]["b2_amd_report_sha256"],
            "telemetry_path": str(telemetry_path),
            "telemetry_sha256": _sha256_file(telemetry_path),
            "checked_at_utc": utc(),
        }
        _atomic_bytes(ROOT / "start_checks" / f"{design}.json", _json_bytes(result))
        return fresh

    def prepare_starts(self) -> None:
        """A failed start stops only that candidate; other starts continue."""
        self._phase("start_checks")
        with ThreadPoolExecutor(max_workers=7) as pool:
            futures = {
                pool.submit(self._start_one, design): design
                for design in ORDER if design not in self.stopped
            }
            for future in futures:
                design = futures[future]
                try:
                    evidence = future.result()
                    self.start_evidence[design] = evidence
                    _alias(
                        ROOT / "by_design" / _component(design) / "start_B2.dcp",
                        self.runtime.store.state_path(self.starts[design].id),
                    )
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    self.stopped[design] = reason
                    self.journal.append(JOURNAL_ID, {
                        "id": f"start-failure-{design}", "type": "start_failure",
                        "design": design, "reason": reason, "at_utc": utc(),
                    })
                    _atomic_bytes(
                        ROOT / "start_checks" / f"{design}.json",
                        _json_bytes({
                            "schema_version": "overnight-test-start-check-v1",
                            "design": design, "status": "failed",
                            "error": reason, "checked_at_utc": utc(),
                        }),
                    )
                self._write_status()

    def _input_for(
        self, design: str, sequence: tuple[str, ...]
    ) -> tuple[DesignState, Evidence, Evidence]:
        """Use only a verified, telemetry-checked first step as D2 parent."""
        start = self.start_evidence[design]
        if len(sequence) == 1:
            return self.starts[design], start, start
        prefix = self.edges.get((design, (sequence[0],)))
        if prefix is None or prefix["status"] != "ok":
            raise ValueError("D2 parent lacks a verified D1 edge")
        outcome = self.runtime.store.load_outcome(prefix["run_id"])
        if outcome.state is None or outcome.evidence is None:
            raise ValueError("D2 parent lacks a saved DCP or Evidence")
        self.ledger.require_pass(design, outcome.state.id)
        self.telemetry.require(outcome.state, outcome.evidence)
        return outcome.state, outcome.evidence, start

    def _run_one(
        self, design: str, sequence: tuple[str, ...], phase: str,
        input_state: DesignState, parent: Evidence, start: Evidence,
    ) -> dict[str, Any]:
        """Keep all raw artifacts; report failure without committing a parent."""
        began = time.monotonic()
        outcome: RunOutcome | None = None
        amd: dict[str, Any] | None = None
        telemetry_path: Path | None = None
        outputs_compared = 0
        error: str | None = None
        try:
            if input_state.design != design or parent.state_id != input_state.id:
                raise ValueError("scheduled input differs from parent Evidence")
            if start.state_id != self.starts[design].id:
                raise ValueError("scheduled start differs from test B2")
            outcome = self.runtime.engine.execute(
                input_state, self.recipes[sequence[-1]]
            )
            if outcome.state is not None:
                amd = self.ledger.verify(design, outcome.state.id)
                outputs_compared = _outputs_compared(amd)
            if outcome.run.status != RunStatus.OK or outcome.state is None or outcome.evidence is None:
                raise ValueError(f"recipe or meter failed: {outcome.run.status.value}")
            if amd is None or amd["result"] != "pass" or outputs_compared <= 0:
                raise ValueError(f"AMD check did not compare outputs with PASS: {amd['result'] if amd else 'missing'}")
            evidence = outcome.evidence
            if (
                evidence.state_id != outcome.state.id
                or evidence.fake
                or evidence.routed is not True
                or evidence.route_errors != 0 or evidence.drc_errors != 0
            ):
                raise ValueError("output lacks valid independent routed/DRC evidence")
            if (
                evidence.constraint_fingerprint != start.constraint_fingerprint
                or evidence.clock_digest != start.clock_digest
            ):
                raise ValueError("output constraint or clock digest changed from B2")
            for metric in REQUIRED_METRICS:
                _metric(evidence, metric)
            cost = _cost(outcome.run)
            telemetry_path = self.telemetry.ensure(outcome.state, evidence)
            if outcome.run.input_sha256_before != input_state.id or outcome.run.input_sha256_after != input_state.id:
                raise ValueError("recipe input digest changed")
            if outcome.run.cache_hit and outcome.run.original_run_id == outcome.run.id:
                raise ValueError("cache hit points at itself")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        run = outcome.run if outcome else None
        state = outcome.state if outcome else None
        evidence = outcome.evidence if outcome else None
        return {
            "schema_version": "overnight-test-edge-v1",
            "design": design, "phase": phase, "sequence": list(sequence),
            "status": "failed" if error else "ok", "error": error,
            "input_sha256": input_state.id,
            "run_id": run.id if run else None,
            "run_status": run.status.value if run else None,
            "output_sha256": state.id if state else None,
            "evidence_id": evidence.id if evidence else None,
            "amd_result": amd["result"] if amd else None,
            "amd_record": str(self.ledger._record_path(design, state.id)) if amd and state else None,
            "amd_outputs_compared": outputs_compared,
            "telemetry_status": "ok" if telemetry_path else None,
            "telemetry_path": str(telemetry_path) if telemetry_path else None,
            "cost": _cost_fields(run) if run else None,
            "edge_elapsed_s": time.monotonic() - began,
            "checked_at_utc": utc(),
        }

    def _edge_alias_path(self, row: dict[str, Any]) -> Path | None:
        """Choose a readable location before the journal commit."""
        output_id = row.get("output_sha256")
        if not output_id:
            return None
        design = _component(row["design"])
        sequence = tuple(_component(item) for item in row["sequence"])
        if row["status"] == "ok":
            if row["phase"] == "C0":
                relative = Path("control") / "C0_reroute.dcp"
            else:
                relative = Path(f"depth{len(sequence)}") / ("__".join(sequence) + ".dcp")
        else:
            relative = Path("failed") / (
                row["phase"] + "__" + "__".join(sequence) + "__" + output_id + ".dcp"
            )
        return ROOT / "by_design" / design / relative

    def _repair_aliases(self) -> None:
        """Recreate only aliases named by committed journal events on resume."""
        for row in self.edges.values():
            alias_name = row.get("alias_path")
            output_id = row.get("output_sha256")
            if not output_id:
                continue
            expected = self._edge_alias_path(row)
            if expected is None or str(expected) != alias_name:
                raise ValueError("journaled alias differs from frozen naming rule")
            target = self.runtime.store.state_path(output_id)
            _alias(expected, target)

    def _write_index(self) -> None:
        """Publish a raw DCP/cost locator without any test outcome ranking."""
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow((
            "kind", "design", "depth", "sequence", "status", "run_id",
            "input_sha256", "output_sha256", "dcp_alias", "canonical_dcp",
            "amd_result", "amd_record", "amd_outputs_compared",
            "telemetry_status", "telemetry_json", "cache_hit", "original_run_id",
            "charged_wall_s", "charged_cpu_user_s", "charged_cpu_sys_s",
            "recipe_peak_rss_bytes", "load_1m_start", "load_1m_end",
            "busy_slots_start",
        ))
        for design in ORDER:
            start_file = ROOT / "start_checks" / f"{design}.json"
            if not start_file.is_file() or _json(start_file).get("status") != "ok":
                continue
            state = self.starts[design]
            alias = ROOT / "by_design" / design / "start_B2.dcp"
            writer.writerow((
                "start", design, 0, "B2", "ok", "", "", state.id,
                str(alias), str(self.runtime.store.state_path(state.id)),
                "pass_prior_B2", self.sources[design]["b2_amd_report"], "",
                "ok", str(self.telemetry.root / "states" / state.id / "dcp_telemetry.json"),
                "", "", "", "", "", "", "", "", "",
            ))
        def order_key(item: tuple[tuple[str, tuple[str, ...]], dict[str, Any]]) -> tuple:
            (design, sequence), row = item
            return (ORDER.index(design), len(sequence), sequence)
        for (_design, sequence), row in sorted(self.edges.items(), key=order_key):
            cost = row.get("cost") or {}
            output_id = row.get("output_sha256")
            canonical = (
                str(ROOT / "states" / output_id / "design.dcp") if output_id else ""
            )
            writer.writerow((
                "control" if row["phase"] == "C0" else "recipe",
                row["design"], len(sequence), "__".join(sequence),
                row["status"], row.get("run_id") or "",
                row["input_sha256"], output_id or "", row.get("alias_path") or "",
                canonical, row.get("amd_result") or "", row.get("amd_record") or "",
                row.get("amd_outputs_compared") or "",
                row.get("telemetry_status") or "", row.get("telemetry_path") or "",
                cost.get("cache_hit", ""), cost.get("original_run_id") or "",
                cost.get("wall_s", ""), cost.get("cpu_user_s", ""),
                cost.get("cpu_sys_s", ""), cost.get("peak_memory_bytes", ""),
                cost.get("load_1m_start", ""), cost.get("load_1m_end", ""),
                cost.get("busy_slots_start", ""),
            ))
        _atomic_bytes(ROOT / "by_design" / "index.csv", buffer.getvalue().encode("utf-8"))

    def _commit_edge(self, row: dict[str, Any]) -> None:
        """Fsync one checked result and make its retained DCP accessible."""
        design = row["design"]
        sequence = tuple(row["sequence"])
        key = design, sequence
        if key in self.edges:
            raise ValueError("test edge was already committed")
        alias = self._edge_alias_path(row)
        row["alias_path"] = str(alias) if alias else None
        event_id = hashlib.sha256(_json_bytes([design, sequence])).hexdigest()
        self.journal.append(JOURNAL_ID, {
            "id": f"edge-{event_id}", "type": "edge",
            "edge": row, "at_utc": utc(),
        })
        self.edges[key] = row
        if alias is not None:
            target = self.runtime.store.state_path(row["output_sha256"])
            _alias(alias, target)
        if row["status"] != "ok":
            self.stopped[design] = row.get("error") or "failed test edge"
        self._write_index()
        self._write_status()

    def _run_layer(
        self, tasks: list[tuple[str, tuple[str, ...]]], *,
        phase: str, per_design: int, max_active: int,
    ) -> None:
        """Bound in-flight spill after a failure while serving other designs."""
        self._phase(phase)
        if len(set(tasks)) != len(tasks):
            raise ValueError("test layer contains a duplicate sequence")
        queues = {
            design: deque(
                (seq for name, seq in tasks
                 if name == design and (name, seq) not in self.edges)
            )
            for design in ORDER
        }
        active_per_design: Counter[str] = Counter()
        with ThreadPoolExecutor(max_workers=max_active) as pool:
            active: dict[Future, tuple[str, tuple[str, ...], str, str]] = {}

            def fill() -> None:
                for design in ORDER:
                    while (
                        design not in self.stopped
                        and queues[design]
                        and active_per_design[design] < per_design
                        and len(active) < max_active
                    ):
                        sequence = queues[design].popleft()
                        edge_phase = "C0" if sequence == (C0,) else (
                            "D1" if len(sequence) == 1 else "D2"
                        )
                        try:
                            input_state, parent, start = self._input_for(design, sequence)
                        except Exception as exc:
                            row = {
                                "schema_version": "overnight-test-edge-v1",
                                "design": design, "phase": edge_phase,
                                "sequence": list(sequence), "status": "failed",
                                "error": f"{type(exc).__name__}: {exc}",
                                "input_sha256": self.starts[design].id,
                                "run_id": None, "run_status": None,
                                "output_sha256": None, "evidence_id": None,
                                "amd_result": None, "amd_record": None,
                                "amd_outputs_compared": 0,
                                "telemetry_status": None, "telemetry_path": None,
                                "cost": None, "edge_elapsed_s": 0,
                                "checked_at_utc": utc(),
                            }
                            self._commit_edge(row)
                            queues[design].clear()
                            break
                        future = pool.submit(
                            self._run_one, design, sequence, edge_phase,
                            input_state, parent, start,
                        )
                        active[future] = (design, sequence, edge_phase, input_state.id)
                        active_per_design[design] += 1

            fill()
            while active:
                done, _pending = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in done:
                    design, sequence, edge_phase, input_id = active.pop(future)
                    active_per_design[design] -= 1
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = {
                            "schema_version": "overnight-test-edge-v1",
                            "design": design, "phase": edge_phase,
                            "sequence": list(sequence), "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "input_sha256": input_id,
                            "run_id": None, "run_status": None,
                            "output_sha256": None, "evidence_id": None,
                            "amd_result": None, "amd_record": None,
                            "amd_outputs_compared": 0,
                            "telemetry_status": None, "telemetry_path": None,
                            "cost": None, "edge_elapsed_s": 0,
                            "checked_at_utc": utc(),
                        }
                    self._commit_edge(row)
                    if row["status"] != "ok":
                        queues[design].clear()
                fill()

    def run_c0_and_d1(self) -> None:
        """Run the control and every selected one-step recipe on seven B2s."""
        tasks = [
            (design, (recipe_id,))
            for design in ORDER
            for recipe_id in (C0, *SELECTED)
        ]
        self._run_layer(
            tasks, phase="one_step_and_c0", per_design=4, max_active=28,
        )

    def _wait_noise(self) -> None:
        """Let the pinned four-worker noise family finish without altering it."""
        self._phase("waiting_for_noise")
        while True:
            state = _json(NOISE_ROOT / "status.json")
            if (
                state.get("phase") == "complete"
                and state.get("completed") == 35
                and state.get("statuses", {}).get("pass") == 35
                and state.get("errors") == []
            ):
                return
            if state.get("phase") == "failed" or state.get("errors"):
                raise RuntimeError("recorded test noise family failed; no room screen or D2")
            time.sleep(30)
            self._write_status()

    def _noise_bands(self) -> dict[str, dict[str, float]]:
        """Revalidate all 35 PASS controls and publish four per-metric bands."""
        source_plan_sha = _sha256_file(NOISE_ROOT / "plan.json")
        bands: dict[str, dict[str, float]] = {}
        sources: dict[str, dict[str, str]] = {}
        for design in ORDER:
            prior = Evidence.model_validate_json(
                Path(self.sources[design]["b2_evidence_path"]).read_bytes()
            )
            controls: dict[str, dict[str, float]] = {}
            sources[design] = {}
            for variant in CONTROLS:
                path = NOISE_ROOT / "results" / f"{design}__{variant}.json"
                row = _json(path)
                meter_path = Path(row["meter_evidence_path"])
                report_path = Path(row["validator_report_path"])
                if (
                    row.get("status") != "pass"
                    or row.get("design") != design
                    or row.get("variant") != variant
                    or row.get("plan_sha256") != source_plan_sha
                    or row.get("b2_sha256") != self.sources[design]["b2_sha256"]
                    or row.get("original_sha256") != self.sources[design]["original_sha256"]
                    or row.get("outputs_compared", 0) <= 0
                    or row.get("precheck_cycles") != 100
                    or row.get("full_cycles") != 200
                    or row.get("constraint_fingerprint")
                    != self.sources[design]["b2_constraint_fingerprint"]
                    or row.get("clock_digest") != self.sources[design]["b2_clock_digest"]
                    or _sha256_file(Path(row["output_path"])) != row["output_sha256"]
                    or _sha256_file(Path(row["alias_path"])) != row["output_sha256"]
                    or _sha256_file(meter_path) != row["meter_evidence_sha256"]
                    or _sha256_file(report_path) != row["validator_report_sha256"]
                ):
                    raise ValueError(f"noise control lacks retained AMD/meter PASS: {design} {variant}")
                measured = Evidence.model_validate_json(meter_path.read_bytes())
                if measured.state_id != row["output_sha256"] or _four(measured) != row["metrics"]:
                    raise ValueError(f"noise result differs from independent Evidence: {design} {variant}")
                controls[variant] = _four(measured)
                sources[design][variant] = _sha256_file(path)
            bands[design] = noise_bands(_four(prior), controls, CONTROLS)
        payload = {
            "schema_version": "overnight-test-noise-bands-v1",
            "method": "max_abs_control_minus_b2",
            "candidate_order": list(ORDER),
            "variants": list(CONTROLS),
            "source_noise_plan_sha256": source_plan_sha,
            "source_result_sha256": sources,
            "bands": bands,
        }
        expected = _json_bytes(payload)
        if NOISE_BANDS.is_file():
            if NOISE_BANDS.read_bytes() != expected:
                raise ValueError("frozen test noise bands differ on resume")
        else:
            _atomic_bytes(NOISE_BANDS, expected)
        return bands

    def room_screen(self, bands: dict[str, dict[str, float]]) -> list[str]:
        """Apply only the frozen strict WNS-or-packed-LUT admission rule."""
        self._phase("room_screen")
        baselines = {
            design: _four(Evidence.model_validate_json(
                Path(self.sources[design]["b2_evidence_path"]).read_bytes()
            ))
            for design in ORDER
        }
        one_steps: dict[str, dict[str, dict[str, float]]] = {}
        for design in ORDER:
            if design in self.stopped:
                one_steps[design] = {}
                continue
            rows = {}
            for recipe_id in SELECTED:
                edge = self.edges.get((design, (recipe_id,)))
                if edge is None or edge["status"] != "ok":
                    raise ValueError(f"screen lacks a complete AMD/telemetry checked D1: {design}")
                outcome = self.runtime.store.load_outcome(edge["run_id"])
                assert outcome.evidence is not None
                rows[recipe_id] = _four(outcome.evidence)
            one_steps[design] = rows
        flags = screen(
            candidate_order=ORDER, selected_recipe_ids=SELECTED,
            baselines=baselines, bands=bands, one_steps=one_steps,
            ineligible_designs=tuple(self.stopped),
        )
        admitted = [design for design in ORDER if flags[design]]
        d1_ids = [
            [design, recipe_id, self.edges[(design, (recipe_id,))]["run_id"]]
            for design in ORDER if design not in self.stopped
            for recipe_id in SELECTED
        ]
        payload = {
            "schema_version": "overnight-test-room-screen-v1",
            "plan_sha256": _sha256_file(PLAN),
            "noise_bands_sha256": _sha256_file(NOISE_BANDS),
            "candidate_order": list(ORDER),
            "selected_recipe_ids": list(SELECTED),
            "d1_run_ids_sha256": hashlib.sha256(_json_bytes(d1_ids)).hexdigest(),
            "admitted_flags": flags,
            "admitted_in_order": admitted,
        }
        expected = _json_bytes(payload)
        if ADMISSION.is_file():
            if ADMISSION.read_bytes() != expected:
                raise ValueError("frozen test admissions differ on resume")
        else:
            _atomic_bytes(ADMISSION, expected)
            _atomic_bytes(
                ADMISSION.with_suffix(".sha256"),
                (_sha256_file(ADMISSION) + "  " + ADMISSION.name + "\n").encode(),
            )
        labels = "".join(
            f"{design}: {'admitted' if flags[design] else 'not admitted'}\n"
            for design in ORDER
        )
        _atomic_bytes(ROOT / "room_screen.stdout", labels.encode())
        print(labels, end="", flush=True)
        return admitted

    def run_d2(self, admitted: list[str]) -> None:
        """Evaluate every ordered selected recipe pair for the frozen five."""
        if len(admitted) != 5 or not ADMISSION.is_file():
            raise ValueError("D2 requires five frozen room admissions")
        tasks = [
            (design, (first, second))
            for design in admitted
            for first in SELECTED
            for second in SELECTED
        ]
        if len(tasks) != 500:
            raise ValueError("D2 plan is not 100 ordered pairs per admitted design")
        self._run_layer(
            tasks, phase="two_step", per_design=12, max_active=60,
        )

    def _cost_summary(self) -> dict[str, Any]:
        """Sum actual tagged tool processes separately from cached cost charges."""
        files = sorted(ROOT.rglob("runner.json"))
        actual_cpu_s = 0.0
        actual_wall_s = 0.0
        missing_cpu = []
        backpressure_events = 0
        backpressure_wait_s = 0.0
        peak_busy = 0
        for path in files:
            process = _json(path).get("process", {})
            user = process.get("cpu_user_s")
            system = process.get("cpu_sys_s")
            if user is None or system is None:
                missing_cpu.append(str(path))
            else:
                actual_cpu_s += user + system
            actual_wall_s += process.get("wall_s") or 0.0
            backpressure_events += process.get("backpressure_events") or 0
            backpressure_wait_s += process.get("backpressure_wait_s") or 0.0
            peak_busy = max(peak_busy, process.get("busy_slots_start") or 0)
        logical = [
            row["cost"] for row in self.edges.values()
            if isinstance(row.get("cost"), dict)
        ]
        return {
            "schema_version": "overnight-test-cost-summary-v1",
            "actual_tagged_tool_jobs": len(files),
            "actual_tool_cpu_hours": round(actual_cpu_s / 3600, 6),
            "actual_tool_wall_hours_summed": round(actual_wall_s / 3600, 6),
            "tool_jobs_missing_cpu": len(missing_cpu),
            "tool_jobs_missing_cpu_paths": missing_cpu,
            "logical_edges_with_run_cost": len(logical),
            "charged_original_recipe_wall_hours": round(
                sum(row.get("wall_s") or 0 for row in logical) / 3600, 6
            ),
            "charged_original_recipe_cpu_hours": round(
                sum((row.get("cpu_user_s") or 0) + (row.get("cpu_sys_s") or 0)
                    for row in logical) / 3600, 6
            ),
            "cache_hit_edges": sum(bool(row.get("cache_hit")) for row in logical),
            "backpressure_events": backpressure_events,
            "backpressure_wait_hours": round(backpressure_wait_s / 3600, 6),
            "peak_busy_slots_at_tool_start": peak_busy,
        }

    def finish(self, admitted: list[str], *, screen_stopped: bool = False) -> dict[str, Any]:
        """Audit all committed outputs and write counts/costs, without rankings."""
        self._repair_aliases()
        self._write_index()
        starts_ok = 0
        distinct_states: set[tuple[str, str]] = set()
        for design in ORDER:
            if design not in self.start_evidence:
                continue
            state, evidence = self.starts[design], self.start_evidence[design]
            self.telemetry.require(state, evidence)
            if _sha256_file(Path(self.sources[design]["b2_amd_report"])) != self.sources[design]["b2_amd_report_sha256"]:
                raise ValueError("prior B2 AMD source report changed")
            starts_ok += 1
            distinct_states.add((design, state.id))
        for row in self.edges.values():
            if row["status"] != "ok":
                continue
            outcome = self.runtime.store.load_outcome(row["run_id"])
            if outcome.state is None or outcome.evidence is None:
                raise ValueError("committed successful edge lost its state or Evidence")
            self.ledger.require_pass(row["design"], outcome.state.id)
            self.telemetry.require(outcome.state, outcome.evidence)
            if (
                row.get("amd_outputs_compared", 0) <= 0
                or row.get("amd_result") != "pass"
                or _sha256_file(Path(row["alias_path"])) != outcome.state.id
            ):
                raise ValueError("successful edge lost AMD compared outputs or DCP alias")
            _cost(outcome.run)
            distinct_states.add((row["design"], outcome.state.id))
        cost = self._cost_summary()
        _atomic_bytes(ROOT / "cost_summary.json", _json_bytes(cost))
        phases = {
            name: sum(
                row["phase"] == name and row["status"] == "ok"
                for row in self.edges.values()
            )
            for name in ("C0", "D1", "D2")
        }
        complete = (
            not self.stopped and not screen_stopped
            and starts_ok == 7 and len(admitted) == 5
            and phases == {"C0": 7, "D1": 70, "D2": 500}
        )
        self.phase = (
            "complete" if complete else
            "stopped_before_d2" if screen_stopped else
            "complete_with_failures"
        )
        self._write_status()
        summary = {
            "schema_version": "overnight-test-baseline-summary-v1",
            "phase": self.phase,
            "completed_at_utc": utc(),
            "admitted_test_designs": admitted,
            "successful_edges_by_phase": phases,
            "planned_edges_by_phase": {"C0": 7, "D1": 70, "D2": 500},
            "starts_amd_source_pass_and_telemetry_ok": starts_ok,
            "distinct_test_states_telemetry_ok": len(distinct_states),
            "all_saved_success_outputs_amd_pass_and_telemetry_ok": True,
            "failed_designs": sorted(self.stopped),
            "cost_summary_path": str(ROOT / "cost_summary.json"),
            "admission_path": str(ADMISSION) if ADMISSION.is_file() else None,
            "admission_sha256": _sha256_file(ADMISSION) if ADMISSION.is_file() else None,
            "index_path": str(ROOT / "by_design/index.csv"),
            "index_sha256": _sha256_file(ROOT / "by_design/index.csv"),
        }
        _atomic_bytes(ROOT / "summary.json", _json_bytes(summary))
        return summary


def _prepare() -> tuple[dict[str, Any], dict[str, dict[str, Any]], Any, dict[str, Any]]:
    """No-tool preflight before launch and exact plan comparison on resume."""
    config = _config()
    sources = _sources()
    runtime = make_real_runtime(
        ROOT, cache_enabled=True, max_slots=96,
        load_ceiling_1m=180.0,
        min_available_memory_bytes=200_000_000_000,
        latch_smoke_status=LATCH_SMOKE,
    )
    plan = _plan(runtime, sources)
    return config, sources, runtime, plan


def _run() -> dict[str, Any]:
    """Run D1/C0 while noise finishes, then freeze admission and run D2."""
    config, sources, runtime, current = _prepare()
    if not PLAN.is_file() or PLAN.read_bytes() != _json_bytes(current):
        raise ValueError("test campaign plan changed between launch and driver start")
    with _file_lock(ROOT / "driver.lock"):
        campaign = Campaign(config, current, runtime, sources)
        try:
            campaign.prepare_starts()
            campaign._repair_aliases()
            campaign._write_index()
            campaign.run_c0_and_d1()
            campaign._wait_noise()
            bands = campaign._noise_bands()
            admitted = campaign.room_screen(bands)
            if len(admitted) != 5:
                return campaign.finish(admitted, screen_stopped=True)
            campaign.run_d2(admitted)
            return campaign.finish(admitted)
        except Exception as exc:
            campaign.phase = "failed"
            campaign._write_status(error=f"{type(exc).__name__}: {exc}")
            raise


def _cwd_under_root(cwd: bytes, prefix: bytes) -> bool:
    """Match a path component boundary so sibling campaigns never overlap."""
    return cwd == prefix or cwd.startswith(prefix + b"/")


def _driver_alive(pid: int) -> bool:
    """Identify only the recorded detached baseline process."""
    try:
        info = _stat(pid)
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return bool(
            info and info.state != "Z"
            and str(SCRIPT).encode() in args and b"--run" in args
        )
    except OSError:
        return False


def _targets() -> tuple[set[str], set[int]]:
    """Find only same-user tool groups rooted in this private test campaign."""
    prefix = str(ROOT).encode()
    tags: set[str] = set()
    groups: set[int] = set()
    for pid in _own_pids():
        info = _stat(pid)
        if info is None or info.state == "Z":
            continue
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            cwd = os.readlink(f"/proc/{pid}/cwd").encode()
        except OSError:
            continue
        # Scope to this directory, not another campaign sharing its prefix.
        if not _cwd_under_root(cwd, prefix):
            continue
        tag = _tag_of(pid)
        if tag and re.fullmatch(r"[0-9a-f]{32}", tag):
            tags.add(tag)
            groups.add(info.pgrp)
    groups.discard(os.getpgrp())
    return tags, groups


def _cleanup() -> dict[str, Any]:
    """Signal only identified tool groups, then sweep their exact run tags."""
    tags, groups = _targets()
    audit: dict[str, Any] = {
        "schema_version": "overnight-test-baseline-watchdog-v1",
        "started_at_utc": utc(), "exact_tags": sorted(tags),
        "process_groups": sorted(groups),
    }
    _atomic_bytes(ROOT / "watchdog_cleanup_audit.json", _json_bytes(audit))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in sorted(groups):
            if any(_tag_of(member.pid) in tags for member in group_members(group)):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    audit["tag_sweeps"] = {
        tag: list(sweep_tag(tag, grace_s=5).alive) for tag in sorted(tags)
    }
    audit["survivors"] = {
        tag: [item.pid for item in tagged(tag)]
        for tag in sorted(tags) if tagged(tag)
    }
    audit["finished_at_utc"] = utc()
    _atomic_bytes(ROOT / "watchdog_cleanup_audit.json", _json_bytes(audit))
    return audit


def _watch() -> None:
    """Watch the detached driver and clean only this campaign on abnormal exit."""
    manifest = _json(LAUNCH)
    pid = manifest["driver_pid"]
    if not _driver_alive(pid):
        raise RuntimeError("recorded test baseline driver absent at watchdog start")
    deadline = time.monotonic() + 36 * 3600
    while time.monotonic() < deadline:
        if not _driver_alive(pid):
            state = _json(STATUS) if STATUS.is_file() else {}
            if state.get("phase") not in (
                "complete", "complete_with_failures", "stopped_before_d2",
            ):
                _cleanup()
            return
        time.sleep(15)
    raise RuntimeError("test baseline watchdog lifetime ended while driver stayed alive")


def _launch() -> dict[str, Any]:
    """Publish immutable plan and detach driver plus scoped exact-tag watchdog."""
    config, sources, runtime, plan = _prepare()
    load, available = Runner._resource_snapshot()
    if load > 180 or available < 200_000_000_000:
        raise RuntimeError("server load or available memory blocks new test launch")
    if LAUNCH.exists():
        raise FileExistsError("test baseline already launched; use --status")
    if PLAN.exists():
        if PLAN.read_bytes() != _json_bytes(plan):
            raise ValueError("existing test plan differs from current source hashes")
    else:
        _atomic_bytes(PLAN, _json_bytes(plan))
    with _file_lock(ROOT / "launch.lock"):
        if LAUNCH.exists():
            raise FileExistsError("test baseline already launched")
        stdout_path = ROOT / "driver.stdout"
        stderr_path = ROOT / "driver.stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--run"],
                cwd=PROJECT, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr,
                start_new_session=True, close_fds=True,
            )
        manifest = {
            "schema_version": "overnight-test-baseline-launch-v1",
            "driver_pid": process.pid, "started_at_utc": utc(),
            "plan_sha256": _sha256_file(PLAN),
            "config_sha256": CONFIG_SHA256,
            "source_designs": list(ORDER),
            "planned_jobs": {"C0": 7, "D1": 70, "D2": 500},
            "max_shared_tool_slots": 96,
            "load_1m_at_launch": load,
            "available_memory_bytes_at_launch": available,
            "stdout": str(stdout_path), "stderr": str(stderr_path),
        }
        _atomic_bytes(LAUNCH, _json_bytes(manifest))
        try:
            watch_out = ROOT / "watchdog.stdout"
            watch_err = ROOT / "watchdog.stderr"
            with watch_out.open("wb") as stdout, watch_err.open("wb") as stderr:
                watcher = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "--watch"],
                    cwd=PROJECT, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr,
                    start_new_session=True, close_fds=True,
                )
            manifest["watchdog_pid"] = watcher.pid
            _atomic_bytes(LAUNCH, _json_bytes(manifest))
            return manifest
        except Exception:
            if process.poll() is None:
                group = os.getpgid(process.pid)
                if group == process.pid and group != os.getpgrp():
                    os.killpg(group, signal.SIGTERM)
            time.sleep(3)
            _cleanup()
            raise


def _short_status() -> dict[str, Any]:
    """Read counts and capacity without inspecting test objective outcomes."""
    state = _json(STATUS) if STATUS.is_file() else {"phase": "not_launched"}
    manifest = _json(LAUNCH) if LAUNCH.is_file() else {}
    pid = manifest.get("driver_pid")
    load, available = Runner._resource_snapshot()
    runner = Runner(PROJECT / "data/slots", max_slots=96)
    noise = _json(NOISE_ROOT / "status.json")
    return {
        "phase": state.get("phase"), "updated_at_utc": state.get("updated_at_utc"),
        "driver_pid": pid,
        "driver_alive": _driver_alive(pid) if isinstance(pid, int) else False,
        "by_design": state.get("by_design"),
        "completed_edges": state.get("completed_edges", 0),
        "amd_pass_edges": state.get("amd_pass_edges", 0),
        "telemetry_unique_ok": state.get("telemetry_unique_ok", 0),
        "stopped_designs": state.get("stopped_designs", []),
        "error": state.get("error"),
        "noise_completed": noise.get("completed"),
        "noise_total": noise.get("total_controls"),
        "noise_errors": len(noise.get("errors", [])),
        "load_1m": load,
        "available_memory_bytes": available,
        "busy_shared_tool_slots": runner._busy_slots(),
        "max_shared_tool_slots": 96,
        "driver_stderr_bytes": Path(manifest["stderr"]).stat().st_size if manifest else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--watch", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--screen", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(_short_status(), sort_keys=True))
        return 0
    if args.screen:
        flags = _json(ADMISSION)["admitted_flags"]
        for design in ORDER:
            print(f"{design}: {'admitted' if flags[design] else 'not admitted'}")
        return 0
    if args.watch:
        _watch()
        return 0
    if args.dry_run:
        config, sources, runtime, plan = _prepare()
        print(json.dumps({
            "designs": list(ORDER),
            "recipe_ids": list(SELECTED),
            "c0_is_control": True,
            "planned": {"C0": 7, "D1": 70, "D2_if_five_admitted": 500},
            "data_root": str(ROOT), "max_shared_tool_slots": 96,
            "meter_version": runtime.meter.version,
            "config_sha256": CONFIG_SHA256,
        }, sort_keys=True))
        return 0
    if args.launch:
        print(json.dumps(_launch(), sort_keys=True))
        return 0
    summary = _run()
    print(json.dumps({
        "phase": summary["phase"],
        "successful_edges_by_phase": summary["successful_edges_by_phase"],
        "distinct_test_states_telemetry_ok": summary["distinct_test_states_telemetry_ok"],
        "failed_designs": summary["failed_designs"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
