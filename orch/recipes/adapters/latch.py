"""Adapter for one unmodified, fixed-plan LATCH flow.

UNTESTED REAL LATCH PATH. The new command and selected-output publication must
be checked in the VexRiscv smoke before this path is considered verified.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from orch.contract.models import Claims, RunStatus
from orch.interfaces import LaunchSpec, ProcessResult, RecipeSpec
from orch.recipes.registry import ADAPTER_VERSIONS, sha256_file


R3_CODE = Path(os.environ.get("ORCH_RECIPE_TOOLS_DIR", "external/recipe_tools"))
PINNED_COMMIT = "c53a8bf5f24159d527eb4508dc3cdda6144ca2aa"
JRE21 = (Path(os.environ.get("ORCH_VIVADO_ROOT", "/opt/Xilinx/2025.1")) / "Vivado/tps/lnx64/jre21.0.1_12")


class LatchAdapter:
    """Prepare a neutral private input and the exact fixed-plan LATCH call."""

    kind = "latch"
    version = ADAPTER_VERSIONS["latch"]

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        latch_root: Path = R3_CODE / "LATCH",
        latch_python: Path = R3_CODE / "venv_latch/bin/python",
        java_home: Path = JRE21,
        strict_pinning: bool = True,
        stability_interval_s: float = 0.5,
        publication_timeout_s: float = 10.0,
    ) -> None:
        self.root = (project_root or Path(__file__).resolve().parents[3]).resolve()
        self.env_script = self.root / "recipes/support/env.sh"
        self.latch_root = latch_root.resolve()
        # Keep the venv interpreter symlink: Python finds pyvenv.cfg beside it.
        # Resolving this path switches the command to the system interpreter.
        self.latch_python = Path(latch_python).expanduser().absolute()
        self.java_home = java_home.resolve()
        self.strict_pinning = strict_pinning
        self.stability_interval_s = stability_interval_s
        self.publication_timeout_s = publication_timeout_s

    def _preflight(self) -> None:
        if not self.latch_python.is_file():
            raise FileNotFoundError(self.latch_python)
        if not (self.latch_root / "dcp_optimizer.py").is_file():
            raise FileNotFoundError(self.latch_root / "dcp_optimizer.py")
        if not (self.java_home / "bin/java").is_file():
            raise FileNotFoundError(self.java_home / "bin/java")
        if self.strict_pinning:
            commit = subprocess.check_output(
                ["git", "-C", str(self.latch_root), "rev-parse", "HEAD"], text=True
            ).strip()
            if commit != PINNED_COMMIT:
                raise ValueError(f"LATCH checkout changed: {commit}")
            changes = subprocess.check_output(
                ["git", "-C", str(self.latch_root), "status", "--porcelain"], text=True
            )
            if changes.strip():
                raise ValueError("pinned LATCH checkout is dirty")

    def build_launch(
        self, recipe: RecipeSpec, *, input_path: Path, run_dir: Path, run_id: str
    ) -> LaunchSpec:
        """Use the run-3 flags with a neutral filename and local output paths."""

        if recipe.kind != self.kind or not recipe.flow:
            raise ValueError("LatchAdapter requires a LATCH flow")
        work = run_dir.resolve()
        if not work.is_relative_to(self.root):
            raise ValueError("run directory must stay inside the Development project")
        work.mkdir(parents=True, exist_ok=True)
        self._preflight()
        private = work / "input"
        private.mkdir(exist_ok=True)
        local_input = private / "design.dcp"
        before = sha256_file(input_path)
        shutil.copy2(input_path, local_input)
        if sha256_file(input_path) != before or sha256_file(local_input) != before:
            raise ValueError("input checkpoint changed while being copied")
        os.chmod(local_input, 0o444)

        # R3/code/latch_job.sh:21-22 is the exact flag source. The run-3 wrapper
        # measures a transition; this adapter publishes LATCH's selected out.dcp.
        # UNTESTED real LATCH invocation through the new runner.
        out = work / "out.dcp"
        latch_run = work / "latch_run"
        cache = self.root / ".uv_cache"
        mpl = self.root / ".mplconfig"
        command = (
            f". {shlex.quote(str(self.env_script))}"
            f" && export UV_CACHE_DIR={shlex.quote(str(cache))}"
            f" MPLCONFIGDIR={shlex.quote(str(mpl))}"
            f" JAVA_HOME={shlex.quote(str(self.java_home))}"
            f" PATH={shlex.quote(str(self.java_home / 'bin'))}:\"$PATH\""
            f" && {shlex.quote(str(self.latch_python))}"
            f" {shlex.quote(str(self.latch_root / 'dcp_optimizer.py'))}"
            f" {shlex.quote(str(local_input))}"
            " --no-dotenv --pipeline fixed"
            f" --plan {shlex.quote(recipe.flow)} --budget-runs 1"
            f" --output {shlex.quote(str(out))}"
            f" --run-dir {shlex.quote(str(latch_run))}"
        )
        return LaunchSpec(
            argv=("bash", "-c", command),
            cwd=str(work),
            env_overrides={"UV_CACHE_DIR": str(cache), "MPLCONFIGDIR": str(mpl), "JAVA_HOME": str(self.java_home)},
            timeout_s=recipe.timeout_s,
            output_hint=str(out),
            input_copy_path=str(local_input),
            script_sha256=sha256_file(self.latch_root / "dcp_optimizer.py"),
            tool_versions={
                "latch_commit": PINNED_COMMIT,
                "vivado": "vivado v2025.1 build 6140274",
            },
            java_version="21.0.1",
        )

    def find_output(self, recipe: RecipeSpec, *, run_dir: Path) -> Path | None:
        """Wait for the selected out.dcp to exist and remain byte-stable."""

        if recipe.kind != self.kind:
            raise ValueError("wrong recipe kind")
        output = run_dir / "out.dcp"
        deadline = time.monotonic() + self.publication_timeout_s
        # R3/code/latch_job.sh:21-31 distinguishes selected and transition files.
        # UNTESTED publication-stability check for the new adapter.
        while True:
            try:
                if output.is_file() and output.stat().st_size > 0:
                    first = (output.stat().st_size, sha256_file(output))
                    time.sleep(self.stability_interval_s)
                    if output.is_file():
                        second = (output.stat().st_size, sha256_file(output))
                        if first == second and second[0] > 0:
                            return output
            except OSError:
                # A concurrent rename or unfinished publication can remove the
                # path between stat and hash; retry until the bounded deadline.
                pass
            if time.monotonic() >= deadline:
                return None
            time.sleep(self.stability_interval_s)

    def classify(
        self, recipe: RecipeSpec, *, run_dir: Path, process: ProcessResult
    ) -> RunStatus:
        if recipe.kind != self.kind:
            raise ValueError("wrong recipe kind")
        if process.status != RunStatus.OK:
            return process.status
        return (
            RunStatus.OK
            if self.find_output(recipe, run_dir=run_dir) is not None
            else RunStatus.NO_OUTPUT
        )

    def parse_claims(
        self, recipe: RecipeSpec, *, run_dir: Path, run_id: str
    ) -> Claims:
        """Read LATCH's reported WNS only as a claim, never as Evidence."""

        if recipe.kind != self.kind:
            raise ValueError("wrong recipe kind")
        stdout = run_dir / "stdout.txt"
        values: dict[str, str | int | float | bool] = {
            "selected_output": "out.dcp",
            "transition_count": len(
                tuple((run_dir / "latch_run").glob("transition_000_*/output.dcp"))
            ),
        }
        if stdout.is_file():
            text = stdout.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(
                r"Final Fmax:\s*[-+0-9.]+\s*MHz\s*\(WNS:\s*([-+]?[0-9.]+)\s*ns\)",
                text,
            )
            if matches:
                values["latch_reported_wns_ns"] = float(matches[-1])
        return Claims(
            id=f"{run_id}:claims",
            run_id=run_id,
            raw_log_path=str(stdout) if stdout.is_file() else None,
            values=values,
        )

