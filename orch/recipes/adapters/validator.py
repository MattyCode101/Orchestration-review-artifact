"""Final-pick AMD validator adapter, with a private copy of its script tree.

UNTESTED REAL VALIDATOR PATH. Unit tests cover safe preparation and report
parsing; no new validator invocation has yet exercised this adapter.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path

from orch.contract.models import EquivalenceCheck, RunStatus
from orch.interfaces import LaunchSpec, ProcessResult, Runner
from orch.recipes.registry import sha256_file


R3_VALIDATOR = Path(os.environ.get("ORCH_RECIPE_TOOLS_DIR", "external/recipe_tools")) / "validator"
VALIDATOR_PYTHON = Path(os.environ.get("ORCH_VALIDATOR_PYTHON", "external/contest/.venv/bin/python"))
RAPIDWRIGHT_ROOT = Path(os.environ.get("ORCH_RAPIDWRIGHT_ROOT", "external/RapidWright"))
SOURCE_HASHES = {
    "validate_dcps.py": "0f5060215048205e995206cbe0e7a22392192dd0283ebbaf77e1b9107b5369f5",
    "RapidWrightMCP/server.py": "095072895cb2ddcd2efdb03ffda0ced4ab52fdcd811213accb293720fdaf3bac",
    "RapidWrightMCP/rapidwright_tools.py": "c6fd4170bdde1575efde89a46b737305deac541bc3eff00e35141e4b1e7a984b",
    "VivadoMCP/vivado_mcp_server.py": "16bd6fdc53e10e538e92a548bc859b4f8891b8af0e3b46351fec3ce376e512a3",
}


class ValidatorAdapter:
    """Run only after final-pick ranking, never as a per-step validity gate."""

    kind = "validator"
    version = "validator-adapter-v0.1.0"

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        source_root: Path = R3_VALIDATOR,
        python: Path = VALIDATOR_PYTHON,
        rapidwright_root: Path = RAPIDWRIGHT_ROOT,
    ) -> None:
        self.root = (project_root or Path(__file__).resolve().parents[3]).resolve()
        self.source_root = source_root.resolve()
        # Keep the virtualenv entry point: resolving its symlink to /usr/bin
        # discards the contest venv site-packages (including mcp).
        self.python = Path(python).expanduser().absolute()
        self.rapidwright_root = rapidwright_root.resolve()
        self.env_script = self.root / "recipes/support/env.sh"

    def _private_dir(self, run_dir: Path) -> Path:
        work = run_dir.resolve()
        if not work.is_relative_to(self.root):
            raise ValueError("validator run directory must stay inside the project")
        work.mkdir(parents=True, exist_ok=True)
        return work

    def build_launch(
        self,
        *,
        golden_path: Path,
        revised_path: Path,
        run_dir: Path,
        vectors: int = 200,
        timeout_s: float = 3600,
    ) -> LaunchSpec:
        """Copy allowlisted validator modules and inputs into one private run."""

        if vectors <= 0:
            raise ValueError("validator vectors must be positive")
        work = self._private_dir(run_dir)
        tool = work / "validator"
        tool.mkdir(exist_ok=True)
        for relative, expected in SOURCE_HASHES.items():
            source = self.source_root / relative
            if sha256_file(source) != expected:
                raise ValueError(f"validator source changed: {relative}")
            target = tool / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha256_file(target) != expected:
                raise ValueError(f"validator copy failed: {relative}")
        rw_link = tool / "RapidWright"
        if not rw_link.exists():
            rw_link.symlink_to(self.rapidwright_root, target_is_directory=True)
        if rw_link.resolve() != self.rapidwright_root:
            raise ValueError("validator RapidWright sibling points elsewhere")
        if not self.python.is_file():
            raise FileNotFoundError(self.python)

        private = work / "input"
        private.mkdir(exist_ok=True)
        golden = private / "golden.dcp"
        revised = private / "revised.dcp"
        for source, target in ((golden_path, golden), (revised_path, revised)):
            before = sha256_file(source)
            shutil.copy2(source, target)
            if sha256_file(source) != before or sha256_file(target) != before:
                raise ValueError("validator input changed during copy")

        # R3/code/val_job.sh:8-11 is the invocation and log/exit-code source.
        # UNTESTED real validator invocation through the new runner.
        cache = self.root / ".uv_cache"
        mpl = self.root / ".mplconfig"
        script = tool / "validate_dcps.py"
        command = (
            f". {shlex.quote(str(self.env_script))}"
            f" && export UV_CACHE_DIR={shlex.quote(str(cache))}"
            f" MPLCONFIGDIR={shlex.quote(str(mpl))}"
            f" VALIDATOR={shlex.quote(str(script))}"
            f" && {shlex.quote(str(self.python))} -B"
            f" {shlex.quote(str(script))}"
            f" {shlex.quote(str(golden))} {shlex.quote(str(revised))}"
            f" --vectors {vectors} > validate.log 2>&1"
            "; rc=$?; echo \"VALIDATOR_RC=$rc\" >> validate.log; exit $rc"
        )
        return LaunchSpec(
            argv=("bash", "-c", command),
            cwd=str(work),
            env_overrides={"UV_CACHE_DIR": str(cache), "MPLCONFIGDIR": str(mpl), "VALIDATOR": str(script)},
            timeout_s=timeout_s,
            output_hint=str(work / "validator_work/validation_report.json"),
            script_sha256=SOURCE_HASHES["validate_dcps.py"],
            tool_versions={
                "validator_sha256": SOURCE_HASHES["validate_dcps.py"],
                "vivado": "vivado v2025.1 build 6140274",
            },
            java_version="11.0.16.1",
        )

    def collect_report(self, run_dir: Path) -> dict[str, object] | None:
        """Copy the observed JSON/log outputs, then clean only this private temp."""

        work = self._private_dir(run_dir)
        tool = work / "validator"
        candidates = sorted(
            tool.glob("dcp_validation_*/validation_report.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            return None
        report_path = candidates[-1]
        temp_dir = report_path.parent.resolve()
        if not temp_dir.is_relative_to(tool.resolve()):
            raise ValueError("validator temp directory escaped private tool copy")
        if not temp_dir.name.startswith("dcp_validation_"):
            raise ValueError("unexpected validator temp directory")
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        retained = work / "validator_work"
        retained.mkdir(exist_ok=True)
        for suffix in ("*.json", "*.log"):
            for source in temp_dir.glob(suffix):
                shutil.copy2(source, retained / source.name)
        shutil.rmtree(temp_dir)
        return raw

    def result(
        self,
        *,
        run_dir: Path,
        process: ProcessResult,
        reference_state_id: str,
        vectors: int = 200,
    ) -> EquivalenceCheck:
        """Label the 200-vector result as weak, even when it passes."""

        report = self.collect_report(run_dir)
        if process.status != RunStatus.OK or report is None:
            status = "unknown"
        elif report.get("infrastructure_failure"):
            status = "unknown"
        else:
            status = "pass" if report.get("overall_passed") is True else "fail"
        return EquivalenceCheck(
            result=status,
            method="amd_validate_dcps",
            vectors=vectors,
            reference_state_id=reference_state_id,
            strength="weak",
        )

    def validate(
        self,
        *,
        runner: Runner,
        golden_path: Path,
        revised_path: Path,
        reference_state_id: str,
        run_id: str,
        run_dir: Path,
        vectors: int = 200,
        timeout_s: float = 3600,
    ) -> tuple[ProcessResult, EquivalenceCheck]:
        """Convenience wrapper for the final-pick fallback loop."""

        spec = self.build_launch(
            golden_path=golden_path,
            revised_path=revised_path,
            run_dir=run_dir,
            vectors=vectors,
            timeout_s=timeout_s,
        )
        process = runner.run(spec, run_id=run_id, run_dir=run_dir)
        check = self.result(
            run_dir=run_dir,
            process=process,
            reference_state_id=reference_state_id,
            vectors=vectors,
        )
        return process, check

