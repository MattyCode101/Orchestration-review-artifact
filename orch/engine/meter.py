"""Vivado meter using only the verbatim, copied run-3 Tcl scripts.

UNTESTED REAL TOOL PATH until the Phase 5 smoke run. The report parser has been
tested on copied run-3 VexRiscv reports. This class deliberately makes two
separate tagged jobs: measure2.tcl for reports, diag_lut.tcl for logical LUTs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import uuid
from pathlib import Path

from orch.contract.models import DesignState, Evidence, RunStatus
from orch.engine.meter_parsers import MeterError, has_measure_error, parse_reports
from orch.interfaces import LaunchSpec, Runner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPPORT = PROJECT_ROOT / "recipes" / "support"
SOURCE_HASHES = {
    "measure2.tcl": "84093a6d36af492ad556ece3760430412e5a9a1462d0bde863195b56ac2f9ce1",
    "lib_measure.tcl": "959248322ae20ac5bb163fd0faafa026e0db7c18e8cb08af0b15e49b19b10950",
    "lib_run3.tcl": "d4bfe1c0ec0a32f7e87150ad86e3879c9c8c8fb67bc3080027efa0ceebed8d2b",
    "diag_lut.tcl": "81934e8c9d6b081d9b0d0e37c1c2a25ef29f500dbc65d7e6e65848000b08f7af",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_log(path: str, cwd: Path) -> str:
    file = Path(path)
    if not file.is_absolute():
        file = cwd / file
    try:
        return file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise MeterError(f"missing runner log: {file}") from exc


def _atomic_json(path: Path, payload: dict) -> None:
    """Keep auxiliary meter provenance intact across a process crash."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class VivadoMeter:
    """Wrap copied run-3 measurement Tcl under the shared tagged runner.

    The caller supplies a copied, project-local environment script and the
    observed Vivado version. No run-3 path is sourced directly at runtime.
    """

    def __init__(
        self,
        *,
        runner: Runner,
        env_script: Path,
        tool_version: str,
        support_dir: Path = DEFAULT_SUPPORT,
        timeout_s: float = 3600,
        diag_timeout_s: float = 1800,
    ) -> None:
        self.runner = runner
        self.env_script = Path(env_script).resolve()
        self.tool_version = tool_version
        self.support_dir = Path(support_dir).resolve()
        self.timeout_s = timeout_s
        self.diag_timeout_s = diag_timeout_s
        if not self.env_script.is_file():
            raise MeterError(f"missing project environment script: {self.env_script}")
        if not self.env_script.is_relative_to(PROJECT_ROOT):
            raise MeterError("meter environment script must be inside this build folder")
        for name, expected in SOURCE_HASHES.items():
            source = self.support_dir / name
            if not source.is_file() or _sha256_file(source) != expected:
                raise MeterError(f"copied Tcl hash mismatch or missing file: {source}")
        self.version = self._version()

    def _version(self) -> str:
        """Hash parser, wrapper, Tcl, environment and tool version for cache safety."""

        inputs = {
            "meter_py": _sha256_file(Path(__file__)),
            "parser_py": _sha256_file(
                Path(__file__).with_name("meter_parsers.py")
            ),
            "contract_models_py": _sha256_file(
                PROJECT_ROOT / "orch" / "contract" / "models.py"
            ),
            "env_script": _sha256_file(self.env_script),
            "scripts": SOURCE_HASHES,
            "tool_version": self.tool_version,
        }
        payload = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _copy_tcl(self, names: tuple[str, ...], destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            copied = destination / name
            shutil.copyfile(self.support_dir / name, copied)
            if _sha256_file(copied) != SOURCE_HASHES[name]:
                raise MeterError(f"private Tcl copy changed: {copied}")

    def _launch(
        self,
        *,
        script: str,
        args: tuple[str, ...],
        cwd: Path,
        timeout_s: float,
    ) -> LaunchSpec:
        """Use R3/code/runner3.py:115-117 and the script's cited -tclargs."""

        # R3/code/measure2.tcl:5 lists exactly five arguments; diag_lut.tcl:7
        # lists one. The copied environment script must make vivado 2025.1 live.
        words = " ".join(shlex.quote(arg) for arg in args)
        local_uv_cache = cwd / ".uv_cache"
        local_mpl_config = cwd / ".mplconfig"
        local_uv_cache.mkdir(parents=True, exist_ok=True)
        local_mpl_config.mkdir(parents=True, exist_ok=True)
        # R3/code/env.sh:30-32 points these caches at the old run folder.
        # Override them after sourcing so this job writes only in its own run.
        inner = (
            f". {shlex.quote(str(self.env_script))} && "
            f"export UV_CACHE_DIR={shlex.quote(str(local_uv_cache))} && "
            f"export MPLCONFIGDIR={shlex.quote(str(local_mpl_config))} && "
            "vivado -mode batch -nojournal -log vivado.log "
            f"-source {shlex.quote(script)} -tclargs {words}"
        )
        return LaunchSpec(
            argv=("/bin/bash", "-c", inner),
            cwd=str(cwd),
            env_overrides={
                "UV_CACHE_DIR": str(local_uv_cache),
                "MPLCONFIGDIR": str(local_mpl_config),
            },
            timeout_s=timeout_s,
            script_sha256=SOURCE_HASHES[script],
            tool_versions={"vivado": self.tool_version},
        )

    def measure(
        self, state: DesignState, *, run_dir: Path, start: Evidence | None = None
    ) -> Evidence:
        """Re-open and measure one exact checkpoint without judging the brief.

        UNTESTED REAL TOOL PATH: runner invocation, report publication and
        tagged cleanup need the requested real smoke. The start parameter is
        accepted for the Meter interface; constraint comparison stays in gate.
        """

        del start
        checkpoint = Path(state.path)
        if not checkpoint.is_file():
            raise MeterError(f"missing state checkpoint: {checkpoint}")
        if _sha256_file(checkpoint) != state.id:
            raise MeterError("state file hash differs from DesignState.id")

        run_dir = Path(run_dir)
        measure_dir = run_dir / "meter" / "measure"
        diag_dir = run_dir / "meter" / "diag"
        self._copy_tcl(
            ("measure2.tcl", "lib_measure.tcl", "lib_run3.tcl"), measure_dir
        )
        self._copy_tcl(("diag_lut.tcl",), diag_dir)

        measurement = self.runner.run(
            self._launch(
                script="measure2.tcl",
                args=(str(checkpoint), str(measure_dir), "final", "0", "0"),
                cwd=measure_dir,
                timeout_s=self.timeout_s,
            ),
            run_id=f"meter-{state.id[:12]}-{uuid.uuid4().hex[:10]}",
            run_dir=measure_dir,
        )
        if measurement.status != RunStatus.OK:
            raise MeterError(
                f"measure2.tcl job {measurement.run_id} ended {measurement.status}"
            )
        measure_parts = [
            _read_log(measurement.stdout_path, measure_dir),
            _read_log(measurement.stderr_path, measure_dir),
        ]
        vivado_log = measure_dir / "vivado.log"
        if vivado_log.exists():
            measure_parts.append(
                vivado_log.read_text(encoding="utf-8", errors="replace")
            )
        # Preserve line boundaries when a runner log ends without a newline.
        measure_log = "\n".join(measure_parts)
        if has_measure_error(measure_log):
            raise MeterError("measure2.tcl reported MEASURE_ERROR")

        # Observed R3 report header names the actual Vivado version. Validate
        # the configured version before the result can enter a content cache.
        summary = measure_dir / "reports" / "final_timing_summary.rpt"
        if summary.is_file():
            header = summary.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"\| Tool Version\s*:\s*Vivado v\.([^\s]+)", header)
            if match is None or match.group(1) != self.tool_version:
                raise MeterError(
                    f"Vivado report version does not match {self.tool_version}"
                )

        diagnostic = self.runner.run(
            self._launch(
                script="diag_lut.tcl",
                args=(str(checkpoint),),
                cwd=diag_dir,
                timeout_s=self.diag_timeout_s,
            ),
            run_id=f"lut-census-{state.id[:12]}-{uuid.uuid4().hex[:10]}",
            run_dir=diag_dir,
        )
        if diagnostic.status != RunStatus.OK:
            raise MeterError(
                f"diag_lut.tcl job {diagnostic.run_id} ended {diagnostic.status}"
            )
        diag_log = "\n".join(
            (
                _read_log(diagnostic.stdout_path, diag_dir),
                _read_log(diagnostic.stderr_path, diag_dir),
            )
        )
        evidence = parse_reports(
            measure_dir / "reports",
            tag="final",
            state_id=state.id,
            meter_version=self.version,
            diag_stdout=diag_log,
            measurement_log=measure_log,
            fake=False,
        )
        _atomic_json(
            run_dir / "meter" / "provenance.json",
            {
                "state_id": state.id,
                "meter_version": self.version,
                "support_tcl_sha256": SOURCE_HASHES,
                "measurement": measurement.model_dump(mode="json"),
                "diagnostic": diagnostic.model_dump(mode="json"),
            },
        )
        return evidence
