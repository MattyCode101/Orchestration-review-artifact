"""Create and check held-out B2 starts with run-1's recorded default flow.

Only the source Tcl's two comment lines and open_checkpoint path are rendered.
Every tool subprocess uses the shared tagged Runner. Four worker threads cap
this campaign at four concurrent tool jobs; the shared slot lock caps the server.
No recipe is applied to any held-out checkpoint.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from orch.contract.models import DesignState, MetricName, RunStatus
from orch.engine.jobtag import (
    LEGACY_TAG_KEY, TAG_KEY, _own_pids, _stat, _tag_of,
    group_members, sweep_tag, tagged,
)
from orch.engine.meter import VivadoMeter
from orch.engine.runner import Runner
from orch.engine.store import _atomic_bytes, _json_bytes
from orch.interfaces import LaunchSpec
from orch.recipes.adapters.validator import ValidatorAdapter

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data/exp_evening_2026-11-04/test_b2"
NOTE = PROJECT / "notes/exp_test_b2.md"
CONTEST = (Path(os.environ.get("ORCH_BENCHMARK_DIR", "benchmarks")) / "fpl26_contest_v1.3.0")
OLD_B2 = (Path(os.environ.get("ORCH_BENCHMARK_DIR", "benchmarks")) / "fpl26_B2_default-reimpl_2026-10-24")
PROVENANCE = OLD_B2 / "provenance/vexriscv_re-place"
SOURCE_TCL = PROVENANCE / "c2_vexriscv_re-place.tcl"
SOURCE_LIB = PROVENANCE / "lib_measure.tcl"
SOURCE_TCL_SHA = "5329b350f5a8a2377ad49b0fef0948ef406020da6d5ae2d022d038597852128f"
SOURCE_LIB_SHA = "959248322ae20ac5bb163fd0faafa026e0db7c18e8cb08af0b15e49b19b10950"
SETTINGS = (Path(os.environ.get("ORCH_VIVADO_ROOT", "/opt/Xilinx/2025.1")) / "Vivado/settings64.sh")
SAFE_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")
NEW = {
    "fir_symmetric_systolic": "fir_symmetric_systolic_routed_2025.1.dcp",
    "fir_systolic_transposed": "fir_systolic_transposed_routed_2025.1.dcp",
    "rosetta_3d-rendering_v2": "rosetta_3d-rendering_2025.1_v2.dcp",
    "amd_mini-isp_v2": "amd_mini-isp_2025.1_v2.dcp",
}
EXISTING = {
    "rosetta_digit-recognition": "rosetta_digit-recognition_2025.1.dcp",
    "corescore_500_mod": "corescore_500_mod_2025.1.dcp",
    "vexriscv_re-place_v2": "vexriscv_re-place_v2_2025.1.dcp",
}
MAX_JOBS = 4
_STATUS_LOCK = threading.Lock()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _original(design: str) -> Path:
    return CONTEST / (NEW | EXISTING)[design]


def _existing_b2(design: str) -> Path:
    return OLD_B2 / f"{design}_B2.dcp"


def _render(design: str, original: Path) -> tuple[str, str]:
    """Prove all Tcl body lines are byte-identical to recorded run-1 c2."""
    source = SOURCE_TCL.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    if len(lines) < 37 or "run_id=c2_vexriscv_re-place" not in lines[0]:
        raise RuntimeError("recorded B2 source has unexpected layout")
    if not lines[1].startswith("# design=vexriscv_re-place input="):
        raise RuntimeError("recorded B2 design comment changed")
    if not lines[6].startswith("open_checkpoint "):
        raise RuntimeError("recorded B2 input command changed")
    if not SAFE_PATH.fullmatch(str(original)):
        raise RuntimeError("checkpoint path is unsafe for the copied unquoted Tcl")
    rendered = lines.copy()
    rendered[0] = lines[0].replace("c2_vexriscv_re-place", f"c2_{design}", 1)
    rendered[1] = f"# design={design} input={original}\n"
    rendered[6] = f"open_checkpoint {original}\n"
    if any(rendered[index] != line for index, line in enumerate(lines) if index not in {0, 1, 6}):
        raise RuntimeError("B2 Tcl body changed")
    body = "".join(line for index, line in enumerate(rendered) if index not in {0, 1, 6})
    source_body = "".join(line for index, line in enumerate(lines) if index not in {0, 1, 6})
    if body != source_body:
        raise RuntimeError("B2 Tcl command body differs from run 1")
    return "".join(rendered), hashlib.sha256(body.encode("utf-8")).hexdigest()


def _preflight() -> dict:
    """Verify exact source bytes and seven allowlisted contest paths offline."""
    if _sha(SOURCE_TCL) != SOURCE_TCL_SHA or _sha(SOURCE_LIB) != SOURCE_LIB_SHA:
        raise RuntimeError("recorded run-1 B2 source Tcl or library changed")
    if not SETTINGS.is_file():
        raise RuntimeError("recorded Vivado settings script is missing")
    rows = {}
    body_hashes = set()
    for design in NEW | EXISTING:
        original = _original(design)
        if not original.is_file():
            raise FileNotFoundError(original)
        row = {"original": str(original), "original_sha256": _sha(original)}
        if design in NEW:
            rendered, body_sha = _render(design, original)
            row["rendered_tcl_sha256"] = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            row["unchanged_command_body_sha256"] = body_sha
            body_hashes.add(body_sha)
        else:
            b2 = _existing_b2(design)
            if not b2.is_file():
                raise FileNotFoundError(b2)
            row["existing_b2"] = str(b2)
            row["existing_b2_sha256"] = _sha(b2)
        rows[design] = row
    if len(body_hashes) != 1:
        raise RuntimeError("rendered B2 bodies differ across four designs")
    return {
        "schema_version": "exp-evening-test-b2-plan-v1",
        "source_tcl": str(SOURCE_TCL), "source_tcl_sha256": SOURCE_TCL_SHA,
        "source_lib": str(SOURCE_LIB), "source_lib_sha256": SOURCE_LIB_SHA,
        "settings_script": str(SETTINGS), "body_sha256": body_hashes.pop(),
        "max_concurrent_tool_jobs": MAX_JOBS,
        "new_designs": list(NEW), "existing_designs": list(EXISTING),
        "designs": rows,
    }


def _note(state: dict) -> None:
    """Keep one short human-readable status beside the machine record."""
    lines = [
        "# Held-out B2 start preparation (4 November 2026)", "",
        f"**Verified status:** `{state['phase']}` at {state['updated_at_utc']}. ",
        f"Data root: `{ROOT}`. Maximum four concurrent tagged tool jobs.", "",
        f"Recorded run-1 source Tcl SHA-256: `{SOURCE_TCL_SHA}`; copied `lib_measure.tcl` SHA-256: `{SOURCE_LIB_SHA}`. ",
        "Rendered Tcl changes only the run/design comments and `open_checkpoint` path; all other command bytes match the run-1 source.", "",
    ]
    for category, label in (("existing_validation", "Existing B2 checks"),
                            ("new_b2", "New B2 creation and measurement"),
                            ("new_validation", "New B2 checks")):
        lines.extend((f"## {label}", ""))
        for design, row in sorted(state["results"][category].items()):
            lines.append(f"- `{design}`: {row.get('status', 'unknown')}"
                         f"; output `{row.get('output_path', row.get('revised_path', ''))}`"
                         f"; WNS {row.get('wns_ns', 'n/a')} ns; packed LUTs {row.get('packed_luts', 'n/a')}"
                         f"; compared outputs {row.get('outputs_compared', 'n/a')}"
                         f"; cycles {row.get('precheck_cycles', 'n/a')}/{row.get('full_cycles', 'n/a')}.")
        lines.append("")
    if state.get("errors"):
        lines.extend(("## Errors", ""))
        lines.extend(f"- {error}" for error in state["errors"])
    _atomic_bytes(NOTE, ("\n".join(lines) + "\n").encode("utf-8"))


def _update(state: dict, *, phase: str | None = None,
            category: str | None = None, design: str | None = None,
            record: dict | None = None, error: str | None = None) -> None:
    with _STATUS_LOCK:
        if phase is not None:
            state["phase"] = phase
        if category is not None and design is not None and record is not None:
            state["results"][category][design] = record
        if error is not None:
            state["errors"].append(error)
        state["updated_at_utc"] = _utc()
        _write_json(ROOT / "status.json", state)
        _note(state)


def _make_b2(design: str, runner: Runner, meter: VivadoMeter, plan: dict) -> dict:
    """Run the copied c2 flow, reopen its output and save independent Evidence."""
    design_root = ROOT / "starts" / design
    latest = design_root / "latest.json"
    if latest.is_file():
        prior = _read_json(latest)
        output = Path(prior.get("output_path", ""))
        if prior.get("status") == "ok" and output.is_file() and _sha(output) == prior.get("output_sha256"):
            return {**prior, "reused": True}
    original = _original(design)
    original_sha = plan["designs"][design]["original_sha256"]
    if _sha(original) != original_sha:
        raise RuntimeError("contest original changed before B2 creation")
    attempt = design_root / f"attempt-{uuid.uuid4().hex[:12]}"
    attempt.mkdir(parents=True)
    rendered, body_sha = _render(design, original)
    script = attempt / f"c2_{design}.tcl"
    with script.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
    shutil.copyfile(SOURCE_LIB, attempt / "lib_measure.tcl")
    if _sha(attempt / "lib_measure.tcl") != SOURCE_LIB_SHA:
        raise RuntimeError("copied run-1 measurement library changed")
    script_sha = _sha(script)
    if script_sha != plan["designs"][design]["rendered_tcl_sha256"] or body_sha != plan["body_sha256"]:
        raise RuntimeError("rendered B2 script differs from preflight")
    _write_json(attempt / "script_provenance.json", {
        "source_tcl": str(SOURCE_TCL), "source_tcl_sha256": SOURCE_TCL_SHA,
        "rendered_tcl": str(script), "rendered_tcl_sha256": script_sha,
        "source_lib": str(SOURCE_LIB), "copied_lib_sha256": SOURCE_LIB_SHA,
        "unchanged_command_body_sha256": body_sha,
        "original": str(original), "original_sha256_before": original_sha,
    })
    # The command and settings script are copied from run-1 runinfo.json.
    command = (
        f"source {shlex.quote(str(SETTINGS))} && "
        "vivado -mode batch -nojournal -log vivado.log "
        f"-source {shlex.quote(script.name)} -tclargs "
    )
    process = runner.run(
        LaunchSpec(argv=("/bin/bash", "-c", command), cwd=str(attempt),
                   timeout_s=3600, script_sha256=script_sha,
                   tool_versions={"vivado": "2025.1"}),
        run_id=f"test-b2-c2-{design}-{uuid.uuid4().hex[:8]}",
        run_dir=attempt,
    )
    _write_json(attempt / "process.json", process.model_dump(mode="json"))
    if process.status != RunStatus.OK or process.exit_code != 0:
        raise RuntimeError(f"B2 tool job failed: {design}, {process.status}, rc={process.exit_code}")
    if (attempt / "status.txt").read_text(encoding="utf-8").strip() != "OK":
        raise RuntimeError(f"B2 native status is not OK: {design}")
    output = attempt / "out.dcp"
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"B2 produced no nonempty checkpoint: {design}")
    output_sha = _sha(output)
    state = DesignState(id=output_sha, design=design, path=str(output), fake=False)
    evidence = meter.measure(state, run_dir=attempt / "meter_run")
    _write_json(attempt / "meter_evidence.json", evidence.model_dump(mode="json"))
    metrics = {}
    for name in (MetricName.WNS_NS, MetricName.TNS_NS,
                 MetricName.PACKED_LUTS, MetricName.NETLIST_LUTS):
        value = evidence.metric(name)
        if value is None or not math.isfinite(value.value):
            raise RuntimeError(f"B2 meter lacks {name}: {design}")
        metrics[name.value] = value.value
    if evidence.routed is not True or evidence.route_errors != 0 or evidence.drc_errors != 0:
        raise RuntimeError(f"B2 physical validity failed: {design}")
    if _sha(original) != original_sha or _sha(output) != output_sha:
        raise RuntimeError(f"B2 input or output changed during measurement: {design}")
    record = {
        "status": "ok", "design": design, "output_path": str(output),
        "output_sha256": output_sha, "output_bytes": output.stat().st_size,
        "original_sha256": original_sha, "rendered_tcl_sha256": script_sha,
        "source_tcl_sha256": SOURCE_TCL_SHA, "copied_lib_sha256": SOURCE_LIB_SHA,
        "unchanged_command_body_sha256": body_sha,
        "wns_ns": metrics["wns_ns"], "tns_ns": metrics["tns_ns"],
        "packed_luts": metrics["packed_luts"], "netlist_luts": metrics["netlist_luts"],
        "constraint_fingerprint": evidence.constraint_fingerprint,
        "clock_digest": evidence.clock_digest,
        "route_errors": evidence.route_errors, "drc_errors": evidence.drc_errors,
        "meter_version": evidence.meter_version,
        "meter_evidence_path": str(attempt / "meter_evidence.json"),
        "process_tag": process.tag,
    }
    _write_json(latest, record)
    alias_dir = ROOT / "by_design"
    alias_dir.mkdir(parents=True, exist_ok=True)
    alias = alias_dir / f"{design}_B2.dcp"
    temporary = alias.with_name(f".{alias.name}.{uuid.uuid4().hex}.tmp")
    temporary.symlink_to(os.path.relpath(output, alias_dir))
    os.replace(temporary, alias)
    return record


def _validate(design: str, revised: Path, runner: Runner,
              validator: ValidatorAdapter, plan: dict) -> dict:
    """Require an AMD PASS with nonzero compared outputs and 100/200 cycles."""
    original = _original(design)
    original_sha = plan["designs"][design]["original_sha256"]
    revised_sha = _sha(revised)
    design_root = ROOT / "validations" / design
    latest = design_root / "latest.json"
    if latest.is_file():
        prior = _read_json(latest)
        if (prior.get("status") == "pass" and prior.get("original_sha256") == original_sha
                and prior.get("revised_sha256") == revised_sha):
            return {**prior, "reused": True}
    attempt = design_root / f"attempt-{uuid.uuid4().hex[:12]}"
    attempt.mkdir(parents=True)
    process, check = validator.validate(
        runner=runner, golden_path=original, revised_path=revised,
        reference_state_id=original_sha,
        run_id=f"test-b2-val-{design}-{uuid.uuid4().hex[:8]}",
        run_dir=attempt, vectors=200, timeout_s=3600,
    )
    log_path = attempt / "validate.log"
    report_path = attempt / "validator_work/validation_report.json"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    raw = _read_json(report_path) if report_path.is_file() else {}
    output_counts = [int(value) for value in re.findall(r"Outputs\s*\((\d+)\):", log)]
    compared = max(output_counts, default=0)
    simulation = raw.get("simulation_report") or {}
    precheck = (simulation.get("precheck_report") or {}).get("cycles_simulated")
    full = (simulation.get("full_report") or {}).get("cycles_simulated")
    passed = (check.result == "pass" and raw.get("overall_passed") is True
              and compared > 0 and precheck == 100 and full == 200
              and _sha(original) == original_sha and _sha(revised) == revised_sha)
    record = {
        "status": "pass" if passed else ("fail" if raw.get("overall_passed") is False else "unknown"),
        "design": design, "original": str(original), "revised_path": str(revised),
        "original_sha256": original_sha, "revised_sha256": revised_sha,
        "validator_result": check.result, "raw_overall_passed": raw.get("overall_passed"),
        "outputs_compared": compared, "precheck_cycles": precheck, "full_cycles": full,
        "process_status": process.status.value, "process_exit_code": process.exit_code,
        "process_tag": process.tag, "validator_report_path": str(report_path),
        "validator_report_sha256": _sha(report_path) if report_path.is_file() else None,
        "validate_log_path": str(log_path),
    }
    _write_json(latest, record)
    return record


def _parallel(state: dict, category: str, tasks: dict[str, object], operation) -> dict[str, dict]:
    """Execute one bounded phase, recording each completion before the next."""
    _update(state, phase=category)
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_JOBS) as pool:
        futures = {pool.submit(operation, design, arg): design for design, arg in tasks.items()}
        for future in as_completed(futures):
            design = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"status": "error", "design": design,
                          "error": f"{type(exc).__name__}: {exc}"}
            results[design] = record
            _update(state, category=category, design=design, record=record)
    failed = {design: row for design, row in results.items() if row.get("status") not in {"ok", "pass"}}
    if failed:
        raise RuntimeError(f"{category} failed: {failed}")
    return results


def _driver() -> None:
    """Run the three stages; no test recipe or outcome analysis is performed."""
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "driver.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = _preflight()
        _write_json(ROOT / "plan.json", plan)
        state = {
            "schema_version": "exp-evening-test-b2-status-v1", "phase": "starting",
            "started_at_utc": _utc(), "updated_at_utc": _utc(),
            "driver_pid": os.getpid(), "max_concurrent_tool_jobs": MAX_JOBS,
            "results": {"existing_validation": {}, "new_b2": {}, "new_validation": {}},
            "errors": [],
        }
        _update(state)
        runner = Runner(PROJECT / "data/slots", max_slots=48)
        meter = VivadoMeter(runner=runner, env_script=PROJECT / "recipes/support/env.sh",
                            tool_version="2025.1")
        validator = ValidatorAdapter(PROJECT)
        try:
            _parallel(
                state, "existing_validation",
                {design: _existing_b2(design) for design in EXISTING},
                lambda design, path: _validate(design, path, runner, validator, plan),
            )
            new = _parallel(
                state, "new_b2", {design: None for design in NEW},
                lambda design, _: _make_b2(design, runner, meter, plan),
            )
            _parallel(
                state, "new_validation",
                {design: Path(new[design]["output_path"]) for design in NEW},
                lambda design, path: _validate(design, path, runner, validator, plan),
            )
        except Exception as exc:
            _update(state, phase="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        _update(state, phase="complete")


def _driver_alive(pid: int) -> bool:
    """Match this exact script and --run in /proc before trusting a PID."""
    try:
        args = [item.decode("utf-8", errors="replace")
                for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if item]
        info = _stat(pid)
    except OSError:
        return False
    return bool(info and info.state != "Z" and str(Path(__file__).resolve()) in args and "--run" in args)


def _scope_targets() -> tuple[set[str], set[int]]:
    """Find exact tags whose current cwd or argv is inside only this data root."""
    prefix = str(ROOT).encode("utf-8")
    tags: set[str] = set()
    groups: set[int] = set()
    for pid in _own_pids():
        info = _stat(pid)
        if info is None or info.state == "Z":
            continue
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            cwd = os.readlink(f"/proc/{pid}/cwd").encode("utf-8")
        except OSError:
            continue
        if prefix not in command and not cwd.startswith(prefix):
            continue
        tag = _tag_of(pid)
        if tag and re.fullmatch(r"[0-9a-f]{32}", tag):
            tags.add(tag)
            groups.add(info.pgrp)
    groups.discard(os.getpgrp())
    return tags, groups


def _cleanup_scoped() -> dict:
    """On driver loss, signal only groups containing this campaign's exact tags."""
    tags, groups = _scope_targets()
    audit = {"at_utc": _utc(), "exact_tags": sorted(tags), "groups": sorted(groups)}
    _write_json(ROOT / "watchdog_audit.json", audit)
    for group in sorted(groups):
        if any(_tag_of(member.pid) in tags for member in group_members(group)):
            try:
                os.killpg(group, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(3)
    for group in sorted(groups):
        if any(_tag_of(member.pid) in tags for member in group_members(group)):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    audit["tag_sweeps"] = {tag: list(sweep_tag(tag, grace_s=5).alive) for tag in sorted(tags)}
    audit["survivors"] = {tag: [item.pid for item in tagged(tag)]
                          for tag in sorted(tags) if tagged(tag)}
    audit["finished_at_utc"] = _utc()
    _write_json(ROOT / "watchdog_audit.json", audit)
    return audit


def _watch(pid: int) -> None:
    """Watch a detached driver and audit exact-tag descendants on failure."""
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        status = _read_json(ROOT / "status.json") if (ROOT / "status.json").is_file() else {}
        if status.get("phase") == "complete":
            _write_json(ROOT / "watchdog_audit.json", {"finished_at_utc": _utc(),
                       "reason": "normal completion", "survivors": {}})
            return
        if not _driver_alive(pid):
            _cleanup_scoped()
            return
        time.sleep(10)
    raise RuntimeError("B2 watchdog lifetime ended while driver remained active")


def _launch() -> dict:
    """Detach one driver and one scoped watchdog after read-only preflight."""
    plan = _preflight()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "launch.json"
    if manifest.is_file():
        old = _read_json(manifest)
        if _driver_alive(old.get("driver_pid", -1)):
            raise RuntimeError("B2 driver already running")
        raise RuntimeError("prior B2 launch exists; inspect it before any relaunch")
    _write_json(ROOT / "plan.json", plan)
    tag = uuid.uuid4().hex
    env = os.environ.copy()
    env[TAG_KEY] = env[LEGACY_TAG_KEY] = tag
    command = [sys.executable, str(Path(__file__).resolve()), "--run"]
    with (ROOT / "driver.stdout").open("wb") as stdout, (ROOT / "driver.stderr").open("wb") as stderr:
        driver = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                  stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
    record = {"schema_version": "exp-evening-test-b2-launch-v1",
              "started_at_utc": _utc(), "driver_pid": driver.pid, "driver_tag": tag,
              "command": command, "script_sha256": _sha(Path(__file__)),
              "max_concurrent_tool_jobs": MAX_JOBS, "watchdog_pid": None}
    _write_json(manifest, record)
    with (ROOT / "watchdog.stdout").open("wb") as stdout, (ROOT / "watchdog.stderr").open("wb") as stderr:
        watcher = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--watch", str(driver.pid)],
                                   cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                   start_new_session=True, close_fds=True)
    record["watchdog_pid"] = watcher.pid
    _write_json(manifest, record)
    return record


def _status() -> dict:
    state = _read_json(ROOT / "status.json") if (ROOT / "status.json").is_file() else {"phase": "not_launched"}
    launch = _read_json(ROOT / "launch.json") if (ROOT / "launch.json").is_file() else {}
    state["driver_alive"] = _driver_alive(launch.get("driver_pid", -1))
    state["watchdog_pid"] = launch.get("watchdog_pid")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--launch", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--watch", type=int, metavar="DRIVER_PID")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(_preflight(), sort_keys=True))
    elif args.launch:
        row = _launch()
        print(json.dumps({"phase": "launched", "driver_pid": row["driver_pid"],
                          "watchdog_pid": row["watchdog_pid"], "data_root": str(ROOT)}, sort_keys=True))
    elif args.run:
        _driver()
    elif args.status:
        print(json.dumps(_status(), sort_keys=True))
    else:
        _watch(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())