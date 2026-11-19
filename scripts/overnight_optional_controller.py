"""Launch optional VexRiscv depth three only after priority one completes.

This detached controller runs no hardware tool itself. It expires at 07:00 UTC
and stops on any failed or incomplete held-out baseline check.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.engine.runner import Runner
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
BASELINE = PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04"
DEPTH3 = PROJECT / "scripts/overnight_optional_depth3.py"
ROOT = PROJECT / "data/sweeps/overnight_priority2_2026-11-04/controller"
PLAN, LAUNCH, STATUS = (ROOT / name for name in ("plan.json", "launch.json", "status.json"))
BASELINE_PLAN_SHA = "060303a7b5fd4a8c46c174e32e35834d4aaa87abfbc966f97b62f165d5550128"
DEADLINE = datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def frozen_plan() -> dict[str, Any]:
    if _sha256_file(BASELINE / "plan.json") != BASELINE_PLAN_SHA:
        raise ValueError("held-out baseline immutable plan changed")
    return {
        "schema_version": "overnight-optional-controller-plan-v1",
        "baseline_plan_sha256": BASELINE_PLAN_SHA,
        "depth3_driver_sha256": _sha256_file(DEPTH3),
        "controller_sha256": _sha256_file(SCRIPT),
        "deadline_utc": DEADLINE.isoformat(),
        "load_ceiling_1m": 180.0,
        "available_memory_floor_bytes": 200_000_000_000,
    }


def driver_alive(pid: int, script: Path, flag: bytes) -> bool:
    """Check exact argv so an unrelated reused PID cannot authorize work."""
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
    except (OSError, IndexError):
        return False
    return state != "Z" and str(script).encode() in args and flag in args


def gate(status: dict[str, Any], summary: dict[str, Any] | None) -> str:
    """Accept only a complete, failure-free priority-one summary."""
    phase = status.get("phase")
    if phase in ("failed", "complete_with_failures", "stopped_before_d2"):
        return "stop"
    if status.get("stopped_designs") or status.get("error"):
        return "stop"
    if summary is None:
        return "wait"
    if (phase == "complete" and summary.get("phase") == "complete"
            and summary.get("successful_edges_by_phase") == {"C0": 7, "D1": 70, "D2": 500}
            and summary.get("starts_amd_source_pass_and_telemetry_ok") == 7
            and summary.get("failed_designs") == []
            and len(summary.get("admitted_test_designs") or []) == 5):
        return "launch"
    return "stop"


def write_status(phase: str, **details: Any) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(STATUS, _json_bytes({
        "schema_version": "overnight-optional-controller-status-v1",
        "phase": phase, "updated_at_utc": utc(), "controller_pid": os.getpid(),
        **details,
    }))


def run() -> None:
    if PLAN.read_bytes() != _json_bytes(frozen_plan()):
        raise ValueError("optional controller plan changed after launch")
    with _file_lock(ROOT / "controller.lock"):
        write_status("waiting_for_priority_one")
        while datetime.now(timezone.utc) < DEADLINE:
            if (ROOT.parent / "depth3/launch.json").is_file():
                write_status("depth3_already_launched")
                return
            status = read(BASELINE / "status.json")
            summary_path = BASELINE / "summary.json"
            summary = read(summary_path) if summary_path.is_file() else None
            decision = gate(status, summary)
            if decision == "stop":
                write_status("blocked_by_priority_one", baseline_phase=status.get("phase"),
                             stopped_designs=status.get("stopped_designs"))
                return
            if decision == "launch":
                load, available = Runner._resource_snapshot()
                if load <= 180.0 and available >= 200_000_000_000:
                    command = [sys.executable, str(DEPTH3), "--launch"]
                    result = subprocess.run(command, cwd=PROJECT, capture_output=True,
                                            text=True, timeout=900, check=False)
                    if result.returncode != 0:
                        if "load or available memory blocks optional launch" in result.stderr:
                            write_status("waiting_for_resources_after_race")
                            time.sleep(60)
                            continue
                        write_status("launch_failed", returncode=result.returncode,
                                     stderr_tail=result.stderr[-2000:])
                        return
                    manifest = json.loads(result.stdout)
                    write_status("depth3_launched", depth3_driver_pid=manifest["driver_pid"],
                                 depth3_plan_sha256=manifest["plan_sha256"])
                    return
                write_status("waiting_for_resources", load_1m=load,
                             available_memory_bytes=available)
            else:
                if status.get("phase") == "complete" and summary is None:
                    # The campaign writes status before its final summary.
                    write_status("waiting_for_priority_one_summary")
                    time.sleep(10)
                    continue
                manifest = read(BASELINE / "launch.json")
                pid = manifest["driver_pid"]
                if not driver_alive(pid, PROJECT / "scripts/overnight_test_baseline.py", b"--run"):
                    write_status("baseline_driver_absent_before_completion", driver_pid=pid)
                    return
                write_status("waiting_for_priority_one", baseline_phase=status.get("phase"),
                             completed_edges=status.get("completed_edges"))
            time.sleep(60)
        write_status("expired_at_07_without_launch")


def launch() -> dict[str, Any]:
    load, available = Runner._resource_snapshot()
    if load > 180.0 or available < 200_000_000_000:
        raise RuntimeError("resources block optional controller launch")
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = frozen_plan()
    if LAUNCH.exists():
        raise FileExistsError("optional controller already launched")
    _atomic_bytes(PLAN, _json_bytes(frozen))
    with _file_lock(ROOT / "launch.lock"):
        if LAUNCH.exists():
            raise FileExistsError("optional controller already launched")
        with (ROOT / "controller.stdout").open("wb") as out, (ROOT / "controller.stderr").open("wb") as err:
            child = subprocess.Popen([sys.executable, str(SCRIPT), "--run"],
                                     cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out,
                                     stderr=err, start_new_session=True, close_fds=True)
        manifest = {"schema_version": "overnight-optional-controller-launch-v1",
                    "controller_pid": child.pid, "controller_pgid": child.pid,
                    "launched_at_utc": utc(), "plan_sha256": _sha256_file(PLAN),
                    "deadline_utc": DEADLINE.isoformat()}
        _atomic_bytes(LAUNCH, _json_bytes(manifest))
        return manifest


def short_status() -> dict[str, Any]:
    status = read(STATUS) if STATUS.is_file() else {"phase": "not_launched"}
    manifest = read(LAUNCH) if LAUNCH.is_file() else {}
    pid = manifest.get("controller_pid")
    return {"phase": status.get("phase"), "updated_at_utc": status.get("updated_at_utc"),
            "controller_pid": pid,
            "controller_alive": driver_alive(pid, SCRIPT, b"--run") if type(pid) is int else False,
            "depth3_driver_pid": status.get("depth3_driver_pid"),
            "controller_stderr_bytes": (ROOT / "controller.stderr").stat().st_size if manifest else 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.run:
        run()
    else:
        print(json.dumps(short_status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
