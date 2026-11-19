"""Failure-path checks for the shared real-backend lifecycle, with no EDA tool."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from orch.contract.models import (
    Claims,
    Evidence,
    MetricName,
    MetricValue,
    RunStatus,
)
from orch.engine.meter_parsers import MeterError
from orch.engine.real_backend import RealBackend
from orch.engine.store import StateStore
from orch.interfaces import LaunchSpec, ProcessResult, RecipeSpec


class StubAdapter:
    kind = "tcl"

    def __init__(self, *, native_status: str = "OK", output: bool = True) -> None:
        self.native_status = native_status
        self.output = output
        self.input_copy: Path | None = None

    def build_launch(self, recipe, *, input_path, run_dir, run_id):
        folder = run_dir / "input"
        folder.mkdir()
        self.input_copy = folder / "design.dcp"
        shutil.copyfile(input_path, self.input_copy)
        (run_dir / "status.txt").write_text(self.native_status, encoding="utf-8")
        if self.output:
            (run_dir / "out.dcp").write_bytes(b"real-backend-stub-output")
        return LaunchSpec(
            argv=("/bin/true",),
            cwd=str(run_dir),
            input_copy_path=str(self.input_copy),
            timeout_s=1,
            tool_versions={"stub": "v1"},
            script_sha256="stub-script",
        )

    def classify(self, recipe, *, run_dir, process):
        return (
            RunStatus.OK
            if self.native_status == "OK"
            else RunStatus.TOOL_ERROR
        )

    def find_output(self, recipe, *, run_dir):
        path = run_dir / "out.dcp"
        return path if path.is_file() else None

    def parse_claims(self, recipe, *, run_dir, run_id):
        return Claims(id=f"{run_id}:claims", run_id=run_id)


class StubRunner:
    def __init__(self, *, mutate_input: bool = False) -> None:
        self.mutate_input = mutate_input

    def run(self, spec, *, run_id, run_dir):
        if self.mutate_input:
            Path(spec.input_copy_path).write_bytes(b"mutated")
        return ProcessResult(
            run_id=run_id,
            status=RunStatus.OK,
            exit_code=0,
            wall_s=0.01,
            process_group_kills=0,
            tag_sweep_kills=0,
            tag="CONTRACT_RUN_ID=stub",
            stdout_path=str(run_dir / "stdout.txt"),
            stderr_path=str(run_dir / "stderr.txt"),
        )


class StubMeter:
    version = "stub-meter"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def measure(self, state, *, run_dir, start=None):
        self.calls += 1
        if self.fail:
            raise MeterError("missing required timing report")
        return Evidence(
            id=f"meter-{state.id[:12]}",
            state_id=state.id,
            metrics={
                MetricName.WNS_NS: MetricValue(
                    value=-0.7, unit="ns", method="stub", basis="routed"
                ),
                MetricName.PACKED_LUTS: MetricValue(
                    value=1900, unit="count", method="stub", basis="routed"
                ),
            },
            routed=True,
            route_errors=0,
            drc_errors=0,
            constraint_fingerprint="xdc",
            clock_digest="clock",
            meter_version=self.version,
        )


def fixture(tmp_path: Path, *, adapter=None, runner=None, meter=None):
    store = StateStore(tmp_path / "data")
    source = tmp_path / "start.dcp"
    source.write_bytes(b"real-backend-stub-input")
    state = store.import_state(source, design="stub-design")
    adapter = adapter or StubAdapter()
    runner = runner or StubRunner()
    meter = meter or StubMeter()
    backend = RealBackend(
        store=store, runner=runner, meter=meter, adapters={"tcl": adapter}
    )
    recipe = RecipeSpec(
        id="A_STUB",
        worker="area",
        kind="tcl",
        script="stub.tcl",
        replaces=True,
        cause="test failure paths",
        source="stub",
        source_sha256="stub",
        timeout_s=1,
        version="v1",
    )
    return store, state, recipe, backend, adapter, meter


def test_zero_exit_with_failed_native_status_is_tool_error(tmp_path: Path) -> None:
    store, state, recipe, backend, _, meter = fixture(
        tmp_path, adapter=StubAdapter(native_status="FAILED_AT incr_read")
    )
    outcome = backend.execute(
        state, recipe, run_id="failed_native", run_dir=store.root / "runs" / "failed_native"
    )
    assert outcome.run.status == RunStatus.TOOL_ERROR
    assert outcome.run.exit_code == 0
    assert outcome.run.failure_reason == "FAILED_AT incr_read"
    assert outcome.state is None and outcome.evidence is None
    assert meter.calls == 0


def test_zero_exit_without_checkpoint_is_no_output(tmp_path: Path) -> None:
    store, state, recipe, backend, _, meter = fixture(
        tmp_path, adapter=StubAdapter(output=False)
    )
    outcome = backend.execute(
        state, recipe, run_id="missing_output", run_dir=store.root / "runs" / "missing_output"
    )
    assert outcome.run.status == RunStatus.NO_OUTPUT
    assert outcome.state is None
    assert meter.calls == 0


def test_private_input_mutation_is_detected_before_meter(tmp_path: Path) -> None:
    store, state, recipe, backend, _, meter = fixture(
        tmp_path, runner=StubRunner(mutate_input=True)
    )
    outcome = backend.execute(
        state, recipe, run_id="mutated_input", run_dir=store.root / "runs" / "mutated_input"
    )
    assert outcome.run.status == RunStatus.INPUT_MUTATED
    assert outcome.run.input_sha256_before == state.id
    assert outcome.run.input_sha256_after != state.id
    assert outcome.state is None and meter.calls == 0
    assert store.state_path(state.id).read_bytes() == b"real-backend-stub-input"


def test_meter_failure_keeps_output_but_not_evidence(tmp_path: Path) -> None:
    meter = StubMeter(fail=True)
    store, state, recipe, backend, _, _ = fixture(tmp_path, meter=meter)
    outcome = backend.execute(
        state, recipe, run_id="meter_error", run_dir=store.root / "runs" / "meter_error"
    )
    assert outcome.run.status == RunStatus.METER_ERROR
    assert outcome.state is not None
    assert outcome.evidence is None
    assert "missing required timing report" in outcome.run.failure_reason
    assert meter.calls == 1

