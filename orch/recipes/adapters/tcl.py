"""Adapter for verbatim run-3/run-4 generated Tcl recipes.

UNTESTED REAL VIVADO PATH. The new launch, native-status and output discovery
paths have unit tests, but no design job has yet exercised this adapter.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

from orch.contract.models import Claims, RunStatus
from orch.interfaces import LaunchSpec, ProcessResult, RecipeSpec
from orch.recipes.registry import ADAPTER_VERSIONS, sha256_file


class TclAdapter:
    """Prepare one private run directory for a generated Tcl recipe."""

    kind = "tcl"
    version = ADAPTER_VERSIONS["tcl"]

    def __init__(self, project_root: Path | None = None) -> None:
        self.root = (project_root or Path(__file__).resolve().parents[3]).resolve()
        self.support = self.root / "recipes/support"
        self.env_script = self.support / "env.sh"

    def _workdir(self, run_dir: Path) -> Path:
        work = run_dir.resolve()
        if not work.is_relative_to(self.root):
            raise ValueError("run directory must stay inside the Development project")
        work.mkdir(parents=True, exist_ok=True)
        return work

    def build_launch(
        self, recipe: RecipeSpec, *, input_path: Path, run_dir: Path, run_id: str
    ) -> LaunchSpec:
        """Copy the checkpoint and sourced Tcl helpers; build run-3's batch call."""

        if recipe.kind != self.kind or not recipe.script:
            raise ValueError("TclAdapter requires a Tcl recipe")
        work = self._workdir(run_dir)
        private = work / "input"
        private.mkdir(exist_ok=True)
        local_input = private / "input.dcp"
        before = sha256_file(input_path)
        shutil.copy2(input_path, local_input)
        if sha256_file(input_path) != before or sha256_file(local_input) != before:
            raise ValueError("input checkpoint changed while being copied")
        os.chmod(local_input, 0o444)

        source = (self.root / recipe.script).resolve()
        copied = work / source.name
        shutil.copy2(source, copied)
        if sha256_file(copied) != recipe.source_sha256:
            raise ValueError("copied Tcl hash differs from registry")
        for helper in ("lib_measure.tcl", "lib_run3.tcl"):
            original = self.support / helper
            destination = work / helper
            shutil.copy2(original, destination)
            if sha256_file(destination) != sha256_file(original):
                raise ValueError(f"copied helper hash mismatch: {helper}")

        # R3/code/runner3.py:115-117 and generated Tcl usage line 5.
        # UNTESTED real Vivado invocation through the new runner.
        cache = self.root / ".uv_cache"
        mpl = self.root / ".mplconfig"
        command = (
            f". {shlex.quote(str(self.env_script))}"
            f" && export UV_CACHE_DIR={shlex.quote(str(cache))}"
            f" MPLCONFIGDIR={shlex.quote(str(mpl))}"
            " && vivado -mode batch -nojournal -log vivado.log"
            f" -source {shlex.quote(copied.name)}"
            f" -tclargs {shlex.quote(str(local_input))}"
        )
        return LaunchSpec(
            argv=("bash", "-c", command),
            cwd=str(work),
            env_overrides={"UV_CACHE_DIR": str(cache), "MPLCONFIGDIR": str(mpl)},
            timeout_s=recipe.timeout_s,
            output_hint=str(work / "out.dcp"),
            input_copy_path=str(local_input),
            script_sha256=recipe.source_sha256,
            tool_versions={"vivado": "vivado v2025.1 build 6140274"},
            java_version="11.0.16.1",
        )

    def find_output(self, recipe: RecipeSpec, *, run_dir: Path) -> Path | None:
        """Find the generated script's final checkpoint, never an intermediate."""

        if recipe.kind != self.kind:
            raise ValueError("wrong recipe kind")
        output = run_dir / "out.dcp"
        return output if output.is_file() and output.stat().st_size > 0 else None

    def classify(
        self, recipe: RecipeSpec, *, run_dir: Path, process: ProcessResult
    ) -> RunStatus:
        """Require native status.txt, since FAILED_AT can accompany exit code zero."""

        if recipe.kind != self.kind:
            raise ValueError("wrong recipe kind")
        if process.status != RunStatus.OK:
            return process.status
        status_path = run_dir / "status.txt"
        if not status_path.is_file():
            return RunStatus.TOOL_ERROR
        native = status_path.read_text(encoding="utf-8", errors="replace").strip()
        if native != "OK":
            return RunStatus.TOOL_ERROR
        if self.find_output(recipe, run_dir=run_dir) is None:
            return RunStatus.NO_OUTPUT
        return RunStatus.OK

    def parse_claims(
        self, recipe: RecipeSpec, *, run_dir: Path, run_id: str
    ) -> Claims:
        """Store native status and log path as claims, never as meter Evidence."""

        if recipe.kind != self.kind:
            raise ValueError("wrong recipe kind")
        status_path = run_dir / "status.txt"
        native = (
            status_path.read_text(encoding="utf-8", errors="replace").strip()
            if status_path.is_file()
            else "missing"
        )
        stdout = run_dir / "stdout.txt"
        return Claims(
            id=f"{run_id}:claims",
            run_id=run_id,
            raw_log_path=str(stdout) if stdout.is_file() else None,
            values={"native_status": native},
        )

