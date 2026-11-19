"""Real recipe backend composed from a registry adapter, runner and meter.

UNTESTED REAL TOOL PATH until the VexRiscv smoke. Tool-facing commands remain
inside adapters and copied Tcl; this module only enforces the shared lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Mapping

from orch.contract.models import (
    Claims,
    DesignState,
    Evidence,
    RunOutcome,
    RunRecord,
    RunStatus,
)
from orch.engine.meter_parsers import MeterError
from orch.engine.store import StateStore
from orch.interfaces import Meter, RecipeAdapter, RecipeSpec, Runner


def _sha256(path: Path) -> str | None:
    """Hash a private input copy, returning None if a worker removed it."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RealBackend:
    """Execute one recipe with no brief-dependent logic or state selection."""

    def __init__(
        self,
        *,
        store: StateStore,
        runner: Runner,
        meter: Meter,
        adapters: Mapping[str, RecipeAdapter],
    ) -> None:
        self.store = store
        self.runner = runner
        self.meter = meter
        self.adapters = dict(adapters)
        self.meter_version = meter.version

    def measure_start(self, state: DesignState, *, run_dir: Path) -> Evidence:
        """Measure the start through the same independent meter as later states."""

        if state.fake:
            raise ValueError("real backend cannot measure a fake state")
        self.store.state_path(state.id)
        return self.meter.measure(state, run_dir=run_dir)

    def execute(
        self,
        state: DesignState,
        recipe: RecipeSpec,
        *,
        run_id: str,
        run_dir: Path,
    ) -> RunOutcome:
        """Run, discover, import and independently measure one output."""

        if state.fake:
            raise ValueError("real backend cannot execute on a fake state")
        adapter = self.adapters.get(recipe.kind)
        if adapter is None or adapter.kind != recipe.kind:
            raise ValueError(f"no adapter for recipe kind {recipe.kind}")
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        source = self.store.state_path(state.id)
        if _sha256(source) != state.id:
            raise ValueError("registered input checkpoint changed")
        started = time.monotonic()
        launch = adapter.build_launch(
            recipe, input_path=source, run_dir=run_dir, run_id=run_id
        )
        if launch.input_copy_path is None:
            raise ValueError("recipe adapter did not identify its private input")
        private_input = Path(launch.input_copy_path).resolve(strict=True)
        if not private_input.is_relative_to(run_dir):
            raise ValueError("private input path escapes its run directory")
        input_before = _sha256(private_input)
        if input_before != state.id:
            raise ValueError("private input differs from the registered state")
        process = self.runner.run(launch, run_id=run_id, run_dir=run_dir)
        input_after = _sha256(private_input)
        status = process.status
        failure_reason: str | None = None
        if status == RunStatus.OK:
            status = adapter.classify(
                recipe, run_dir=run_dir, process=process
            )
        if process.attempts and process.attempts[-1].failure_reason:
            failure_reason = process.attempts[-1].failure_reason
        status_file = run_dir / "status.txt"
        if status_file.is_file():
            native_status = status_file.read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            if native_status and native_status != "OK":
                failure_reason = native_status
        if input_after != input_before:
            status = RunStatus.INPUT_MUTATED
            failure_reason = "private input checkpoint changed during recipe execution"

        claims: Claims | None = adapter.parse_claims(
            recipe, run_dir=run_dir, run_id=run_id
        )
        output_state: DesignState | None = None
        evidence: Evidence | None = None
        if status == RunStatus.OK:
            output = adapter.find_output(recipe, run_dir=run_dir)
            if output is None:
                status = RunStatus.NO_OUTPUT
                failure_reason = "adapter found no completed output checkpoint"
            else:
                resolved = Path(output).resolve(strict=True)
                if not resolved.is_relative_to(run_dir):
                    raise ValueError("recipe output escapes its private run directory")
                output_state = self.store.import_state(
                    resolved,
                    design=state.design,
                    parent_id=state.id,
                    made_by_run_id=run_id,
                    fake=False,
                )
                try:
                    evidence = self.meter.measure(
                        output_state, run_dir=run_dir
                    )
                except MeterError as exc:
                    status = RunStatus.METER_ERROR
                    failure_reason = str(exc)
                else:
                    if (
                        evidence.routed is not True
                        or evidence.route_errors != 0
                        or evidence.drc_errors != 0
                    ):
                        status = RunStatus.INVALID
                        failure_reason = "meter found unrouted nets or DRC errors"

        # The project-local environment script and its hash enter the recipe
        # version. Hash only explicit overrides and tool selectors here; never
        # record inherited environment values or secret-bearing variables.
        environment_hash = hashlib.sha256(
            json.dumps(
                {
                    "env_overrides": launch.env_overrides,
                    "tool_versions": launch.tool_versions,
                    "java_version": launch.java_version,
                    "recipe_version": recipe.version,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        record = RunRecord(
            id=run_id,
            recipe_id=recipe.id,
            recipe_version=recipe.version,
            status=status,
            failure_reason=failure_reason,
            exit_code=process.exit_code,
            wall_s=time.monotonic() - started,
            cpu_user_s=process.cpu_user_s,
            cpu_sys_s=process.cpu_sys_s,
            peak_memory_bytes=process.peak_memory_bytes,
            load_1m_start=process.load_1m_start,
            load_1m_end=process.load_1m_end,
            busy_slots_start=process.busy_slots_start,
            backpressure_wait_s=process.backpressure_wait_s,
            backpressure_events=process.backpressure_events,
            process_group_kills=process.process_group_kills,
            tag_sweep_kills=process.tag_sweep_kills,
            input_sha256_before=input_before,
            input_sha256_after=input_after,
            command=launch.argv,
            environment_hash=environment_hash,
            script_sha256=launch.script_sha256,
            tool_versions=launch.tool_versions,
            java_version=launch.java_version,
            attempts=process.attempts,
            fake=False,
        )
        return RunOutcome(
            run=record,
            state=output_state,
            evidence=evidence,
            claims=claims,
        )

