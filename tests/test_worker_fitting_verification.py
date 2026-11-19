"""Adversarial software tests for mandatory per-DCP equivalence records."""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orch.baselines.verification import (
    VerificationFailure, VerificationLedger, VerifiedStepEngine, audit_committed,
)
from orch.contract.models import (
    EquivalenceCheck, Evidence, RunOutcome, RunRecord, RunStatus,
)
from orch.engine.store import StateStore
from orch.interfaces import ProcessResult


class FakeValidator:
    """Produce a validator-shaped report without invoking any hardware tool."""

    version = "fake-validator-v1"

    def __init__(self, tmp_path: Path, mode: str = "pass") -> None:
        self.mode = mode
        self.calls = 0
        self.env_script = tmp_path / "env.sh"
        self.env_script.write_text("test\n", encoding="utf-8")
        self.python = Path(sys.executable)
        self.rapidwright_root = tmp_path
        self.delay_s = 0.0

    def validate(self, **kwargs):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        run_dir = kwargs["run_dir"]
        run_id = kwargs["run_id"]
        report_path = run_dir / "validator_work" / "validation_report.json"
        if self.mode != "unknown":
            report_path.parent.mkdir(parents=True)
            passed = self.mode == "pass"
            simulation = {
                "precheck_report": {
                    "passed": passed, "cycles_simulated": 100,
                    "mismatch_count": 0 if passed else 1,
                    "protocol_mismatch_count": 0,
                },
                "full_report": {
                    "passed": passed, "cycles_simulated": 200,
                    "mismatch_count": 0 if passed else 1,
                    "protocol_mismatch_count": 0,
                },
            }
            report_path.write_text(json.dumps({
                "num_vectors": 200, "precheck_vectors": 100,
                "phase1_passed": passed, "phase2_passed": passed,
                "overall_passed": passed, "infrastructure_failure": False,
                "simulation_report": simulation,
            }), encoding="utf-8")
        status = RunStatus.OK if self.mode == "pass" else RunStatus.TOOL_ERROR
        process = ProcessResult(
            run_id=run_id, status=status, exit_code=0 if status == RunStatus.OK else 1,
            wall_s=0.01, process_group_kills=0, tag_sweep_kills=0,
            tag="fake-tag", stdout_path=str(run_dir / "stdout.txt"),
            stderr_path=str(run_dir / "stderr.txt"),
        )
        check = EquivalenceCheck(
            result=self.mode, method="fake-amd-validator", vectors=200,
            reference_state_id=kwargs["reference_state_id"], strength="weak",
        )
        return process, check


def setup_ledger(tmp_path: Path, mode: str = "pass"):
    root = tmp_path / "campaign"
    store = StateStore(root)
    source = tmp_path / "source"
    source.mkdir()
    golden_file = source / "B2.dcp"
    revised_file = source / "output.dcp"
    golden_file.write_bytes(b"golden DCP bytes")
    revised_file.write_bytes(b"revised DCP bytes")
    golden = store.import_state(golden_file, design="train")
    revised = store.import_state(
        revised_file, design="train", parent_id=golden.id,
        made_by_run_id="run_test",
    )
    validator = FakeValidator(tmp_path, mode=mode)
    ledger = VerificationLedger(
        store=store, runner=object(), validator=validator,
        references={"train": golden.id},
    )
    return ledger, validator, golden, revised


def test_pass_is_recorded_once_and_reused_by_exact_dcp(tmp_path: Path) -> None:
    ledger, validator, _, revised = setup_ledger(tmp_path)
    first = ledger.verify("train", revised.id)
    second = ledger.verify("train", revised.id)
    assert first == second
    assert first["result"] == "pass"
    assert validator.calls == 1
    assert Path(first["raw_report_path"]).is_file()


def test_two_callers_share_one_validator_run(tmp_path: Path) -> None:
    ledger, validator, _, revised = setup_ledger(tmp_path)
    validator.delay_s = 0.05
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: ledger.verify("train", revised.id), range(2)
        ))
    assert results[0] == results[1]
    assert validator.calls == 1


def test_unknown_retries_twice_and_never_becomes_pass(tmp_path: Path) -> None:
    ledger, validator, _, revised = setup_ledger(tmp_path, mode="unknown")
    record = ledger.verify("train", revised.id)
    assert record["result"] == "unknown"
    assert len(record["attempts"]) == 2
    assert validator.calls == 2


def test_stale_latest_pointer_is_rejected(tmp_path: Path) -> None:
    ledger, _, _, revised = setup_ledger(tmp_path)
    ledger.verify("train", revised.id)
    latest = ledger._latest_path("train", revised.id)
    record = json.loads(latest.read_text(encoding="utf-8"))
    record["result"] = "fail"
    latest.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="latest verification pointer differs"):
        ledger.verify("train", revised.id)


def test_verified_engine_refuses_functional_fail(tmp_path: Path) -> None:
    ledger, _, golden, revised = setup_ledger(tmp_path, mode="fail")
    evidence = Evidence(
        id="evidence", state_id=revised.id, metrics={},
        routed=True, route_errors=0, drc_errors=0, meter_version="m",
    )
    outcome = RunOutcome(
        run=RunRecord(
            id="run_test", recipe_id="recipe", recipe_version="v1",
            status=RunStatus.OK, input_sha256_before=golden.id,
            input_sha256_after=golden.id,
        ),
        state=revised, evidence=evidence,
    )

    class Engine:
        store = ledger.store
        meter_version = "m"

        def execute(self, state, recipe):
            return outcome

    wrapped = VerifiedStepEngine(Engine(), ledger)
    with pytest.raises(VerificationFailure, match="AMD validator fail"):
        wrapped.execute(golden, object())


def test_resume_audit_requires_pass_for_each_saved_output(tmp_path: Path) -> None:
    ledger, _, _, revised = setup_ledger(tmp_path)
    events = [{
        "type": "edge",
        "edge": {
            "design": "train",
            "sequence": ["recipe"],
            "output_state_id": revised.id,
        },
    }]
    with pytest.raises(VerificationFailure, match="missing validator record"):
        audit_committed(events, ledger)
    ledger.verify("train", revised.id)
    result = audit_committed(events, ledger)
    assert result == {
        "committed_edges": 1, "saved_outputs": 1,
        "distinct_verified_dcps": 1,
    }
