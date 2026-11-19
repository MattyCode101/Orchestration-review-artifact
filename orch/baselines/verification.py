"""Functional verification records for the train-only worker fitting campaign.

The copied AMD validator is a weak 100+200-vector check. A PASS is required
before a new output can be committed as a parent state; no formal proof is
claimed. Every result cites the exact B2 and output DCP byte hashes.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from orch.contract.models import DesignState, Evidence, RunOutcome, RunStatus
from orch.engine.store import StateStore, _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from orch.interfaces import RecipeSpec, Runner
from orch.recipes.adapters.validator import SOURCE_HASHES, ValidatorAdapter

SCHEMA = "worker-fitting-equivalence-v1"
VECTORS = 200
PRECHECK_VECTORS = 100
MAX_ATTEMPTS = 2


class VerificationFailure(RuntimeError):
    """A DCP lacks a recorded functional validator PASS."""


class VerificationLedger:
    """Run or reuse a pinned, exact-DCP validator check under a file lock."""

    def __init__(
        self,
        *,
        store: StateStore,
        runner: Runner,
        validator: ValidatorAdapter,
        references: Mapping[str, str],
        vectors: int = VECTORS,
    ) -> None:
        if vectors != VECTORS:
            raise ValueError("this campaign uses the observed run-3 200-vector check")
        if not references or any(len(value) != 64 for value in references.values()):
            raise ValueError("verification requires exact B2 hashes")
        self.store = store
        self.runner = runner
        self.validator = validator
        self.references = dict(references)
        self.root = store.root / "verification"
        self.root.mkdir(parents=True, exist_ok=True)
        self.start_paths = {
            design: store.state_path(state_id)
            for design, state_id in self.references.items()
        }
        project = Path(__file__).resolve().parents[2]
        inputs = {
            "schema_version": SCHEMA,
            "references": dict(sorted(self.references.items())),
            "vectors": VECTORS,
            "precheck_vectors": PRECHECK_VECTORS,
            "adapter_sha256": _sha256_file(
                project / "orch/recipes/adapters/validator.py"
            ),
            "verification_code_sha256": _sha256_file(Path(__file__)),
            "runner_sha256": _sha256_file(project / "orch/engine/runner.py"),
            "jobtag_sha256": _sha256_file(project / "orch/engine/jobtag.py"),
            "env_script_sha256": _sha256_file(validator.env_script),
            "validator_sources": SOURCE_HASHES,
            "validator_python": str(validator.python),
            "rapidwright_root": str(validator.rapidwright_root),
            "validator_adapter_version": validator.version,
        }
        self.fingerprint = hashlib.sha256(_json_bytes(inputs)).hexdigest()
        self.plan = {**inputs, "fingerprint": self.fingerprint}
        path = self.root / "plan.json"
        expected = _json_bytes(self.plan)
        with _file_lock(self.root / "plan.lock"):
            if path.is_file():
                if path.read_bytes() != expected:
                    raise ValueError("verification plan changed after launch")
            else:
                _atomic_bytes(path, expected)

    def _record_path(self, design: str, output_id: str) -> Path:
        return self.root / "records" / self.fingerprint / design / f"{output_id}.json"

    def _latest_path(self, design: str, output_id: str) -> Path:
        return self.root / "latest" / design / f"{output_id}.json"

    def _load_record(self, path: Path, design: str, output_id: str) -> dict:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("schema_version") != SCHEMA
            or record.get("design") != design
            or record.get("reference_sha256") != self.references[design]
            or record.get("output_sha256") != output_id
            or record.get("validator_fingerprint") != self.fingerprint
            or record.get("result") not in ("pass", "fail", "unknown")
        ):
            raise ValueError(f"verification record identity differs: {path}")
        report_path = record.get("raw_report_path")
        report_sha = record.get("raw_report_sha256")
        if report_path and report_sha:
            actual = _sha256_file(Path(report_path))
            if actual != report_sha:
                raise ValueError(f"validator raw report changed: {report_path}")
        elif record["result"] == "pass":
            raise ValueError("PASS record lacks a retained validator report")
        return record

    def _classify(self, process, check, report: dict | None) -> str:
        if report is None:
            return "unknown"
        if (
            process.status == RunStatus.OK
            and check.result == "pass"
            and report.get("overall_passed") is True
            and report.get("phase1_passed") is True
            and report.get("phase2_passed") is True
            and report.get("infrastructure_failure") is False
            and report.get("num_vectors") == VECTORS
            and report.get("precheck_vectors") == PRECHECK_VECTORS
        ):
            simulation = report.get("simulation_report") or {}
            precheck = simulation.get("precheck_report") or {}
            full = simulation.get("full_report") or {}
            if (
                precheck.get("passed") is True
                and precheck.get("cycles_simulated") == PRECHECK_VECTORS
                and precheck.get("mismatch_count") == 0
                and precheck.get("protocol_mismatch_count") == 0
                and full.get("passed") is True
                and full.get("cycles_simulated") == VECTORS
                and full.get("mismatch_count") == 0
                and full.get("protocol_mismatch_count") == 0
            ):
                return "pass"
        if (
            check.result == "fail"
            and report.get("overall_passed") is False
            and report.get("infrastructure_failure") is False
        ):
            return "fail"
        return "unknown"

    def verify(self, design: str, output_id: str) -> dict:
        """Validate a distinct saved output once; retry infrastructure unknowns."""
        if design not in self.references:
            raise ValueError(f"design is outside verification references: {design}")
        revised = self.store.state_path(output_id)
        golden = self.start_paths[design]
        path = self._record_path(design, output_id)
        with _file_lock(path.with_suffix(".lock")):
            if path.is_file():
                record = self._load_record(path, design, output_id)
                latest = self._latest_path(design, output_id)
                expected = _json_bytes(record)
                if latest.is_file():
                    if latest.read_bytes() != expected:
                        raise ValueError("latest verification pointer differs")
                else:
                    _atomic_bytes(latest, expected)
                return record
            attempts: list[dict[str, object]] = []
            result = "unknown"
            for _ in range(MAX_ATTEMPTS):
                run_id = "verify_" + uuid.uuid4().hex
                run_dir = self.root / "runs" / run_id
                began = time.monotonic()
                process, check = self.validator.validate(
                    runner=self.runner,
                    golden_path=golden,
                    revised_path=revised,
                    reference_state_id=self.references[design],
                    run_id=run_id,
                    run_dir=run_dir,
                    vectors=VECTORS,
                    timeout_s=3600,
                )
                raw_path = run_dir / "validator_work" / "validation_report.json"
                report = (
                    json.loads(raw_path.read_text(encoding="utf-8"))
                    if raw_path.is_file() else None
                )
                result = self._classify(process, check, report)
                attempts.append({
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "process_status": process.status.value,
                    "check_result": check.result,
                    "result": result,
                    "wall_s": round(time.monotonic() - began, 3),
                    "raw_report_path": str(raw_path) if report is not None else "",
                    "raw_report_sha256": _sha256_file(raw_path) if report is not None else "",
                    "process": process.model_dump(mode="json"),
                })
                if result != "unknown":
                    break
            if self.store.state_path(output_id) != revised:
                raise ValueError("revised DCP changed during verification")
            if self.store.state_path(self.references[design]) != golden:
                raise ValueError("reference B2 changed during verification")
            last = attempts[-1]
            record = {
                "schema_version": SCHEMA,
                "design": design,
                "reference_sha256": self.references[design],
                "output_sha256": output_id,
                "validator_fingerprint": self.fingerprint,
                "result": result,
                "strength": "weak",
                "vectors": VECTORS,
                "precheck_vectors": PRECHECK_VECTORS,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
                "attempts": attempts,
                "raw_report_path": last["raw_report_path"],
                "raw_report_sha256": last["raw_report_sha256"],
            }
            _atomic_bytes(path, _json_bytes(record))
            _atomic_bytes(self._latest_path(design, output_id), _json_bytes(record))
            return record


    def require_pass(self, design: str, output_id: str) -> dict:
        """Read a prior PASS without starting a new validator job."""
        if design not in self.references:
            raise ValueError(f"design is outside verification references: {design}")
        self.store.state_path(output_id)
        path = self._record_path(design, output_id)
        if not path.is_file():
            raise VerificationFailure(f"missing validator record: {design} {output_id}")
        record = self._load_record(path, design, output_id)
        latest = self._latest_path(design, output_id)
        if not latest.is_file() or latest.read_bytes() != _json_bytes(record):
            raise ValueError("latest verification pointer missing or differs")
        if record["result"] != "pass":
            raise VerificationFailure(
                f"validator {record['result']}: {design} {output_id}"
            )
        return record


def audit_committed(journal, ledger: VerificationLedger) -> dict[str, int]:
    """Require a PASS for every distinct output in the checked campaign journal."""
    latest: dict[tuple[str, tuple[str, ...]], dict] = {}
    for event in journal:
        if event.get("type") != "edge":
            continue
        edge = event["edge"]
        design = edge["design"]
        if design not in ledger.references:
            raise ValueError("journal contains an unselected design")
        latest[(design, tuple(edge["sequence"]))] = edge
    distinct = {
        (edge["design"], edge["output_state_id"])
        for edge in latest.values() if edge.get("output_state_id")
    }
    for design, output_id in sorted(distinct):
        ledger.require_pass(design, output_id)
    return {
        "committed_edges": len(latest),
        "saved_outputs": sum(bool(edge.get("output_state_id")) for edge in latest.values()),
        "distinct_verified_dcps": len(distinct),
    }


class VerifiedStepEngine:
    """Permit a new edge to reach the campaign journal only after verification."""

    def __init__(self, engine, ledger: VerificationLedger) -> None:
        self.engine = engine
        self.ledger = ledger
        self.store = engine.store
        self.meter_version = engine.meter_version

    def measure_start(self, state: DesignState) -> Evidence:
        return self.engine.measure_start(state)

    def execute(self, state: DesignState, recipe: RecipeSpec) -> RunOutcome:
        outcome = self.engine.execute(state, recipe)
        if outcome.state is not None:
            record = self.ledger.verify(outcome.state.design, outcome.state.id)
            if outcome.run.status == RunStatus.OK and record["result"] != "pass":
                raise VerificationFailure(
                    f"{outcome.state.design} {outcome.state.id}: "
                    f"AMD validator {record['result']}; see "
                    f"{self.ledger._record_path(outcome.state.design, outcome.state.id)}"
                )
        return outcome
