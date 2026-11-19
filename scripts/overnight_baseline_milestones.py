"""Counts-only unattended milestone log for the held-out test baseline.

Reads status and the frozen room-screen labels; never reads objective values or
launches tools. This monitor writes notes/progress.md after phase milestones.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.engine.store import _atomic_bytes, _json_bytes

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
ROOT = PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04"
PROGRESS = PROJECT / "notes/progress.md"
LOCK = PROJECT / "notes/overnight_report_progress.lock"
PLAN_SHA = "060303a7b5fd4a8c46c174e32e35834d4aaa87abfbc966f97b62f165d5550128"
STATUS = ROOT / "milestone_monitor_status.json"
LAUNCH = ROOT / "milestone_monitor_launch.json"
ADMISSION = ROOT / "admitted_test_designs_2026-11-04.json"
DEADLINE = datetime(2026, 11, 5, 7, 0, tzinfo=timezone.utc)
ORDER = (
    "rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2",
    "fir_symmetric_systolic", "fir_systolic_transposed",
    "rosetta_3d-rendering_v2", "amd_mini-isp_v2",
)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(state: dict[str, Any], admission: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Build milestone lines from counts and registered admission labels only."""
    rows = state.get("by_design") or {}
    if set(rows) != set(ORDER):
        raise ValueError("status design set differs from seven frozen candidates")
    c0 = sum(int(rows[name].get("c0_pass", 0)) for name in ORDER)
    d1 = sum(int(rows[name].get("d1_pass", 0)) for name in ORDER)
    d2 = sum(int(rows[name].get("d2_pass", 0)) for name in ORDER)
    stopped = state.get("stopped_designs") or []
    result: list[tuple[str, str]] = []
    if c0 == 7:
        result.append(("c0_complete", "C0 is 7/7 AMD PASS with independent meter and telemetry checks."))
    if d1 == 70:
        result.append(("d1_complete",
                       f"D1 is 70/70 AMD PASS; telemetry has "
                       f"{state.get('telemetry_unique_ok', 0)} distinct checked DCPs."))
    if admission is not None:
        # A stopped design can have fewer than ten D1 edges; the frozen
        # screen makes it ineligible while complete designs may fill the room.
        flags = admission.get("admitted_flags")
        if not isinstance(flags, dict) or set(flags) != set(ORDER):
            raise ValueError("admission flags differ from seven frozen candidates")
        labels = ", ".join(
            f"{name}={'admitted' if flags[name] is True else 'not admitted'}"
            for name in ORDER
        )
        if any(type(flags[name]) is not bool for name in ORDER):
            raise ValueError("admission flag is not boolean")
        result.append(("room_screen", f"Pre-registered room screen frozen at SHA-256 "
                       f"{sha(ADMISSION)}; labels only: {labels}."))
    for threshold in (100, 200, 300, 400, 500):
        if d2 >= threshold:
            result.append((f"d2_{threshold}",
                           f"D2 reached {threshold}/500 committed AMD PASS edges; "
                           f"stopped designs: {', '.join(stopped) if stopped else 'none'}."))
    phase = state.get("phase")
    if phase in ("complete", "complete_with_failures", "stopped_before_d2", "failed"):
        result.append(("terminal", f"Baseline phase {phase}: C0 {c0}/7, D1 {d1}/70, "
                       f"D2 {d2}/500, stopped designs "
                       f"{', '.join(stopped) if stopped else 'none'}."))
    return result


def append_once(key: str, message: str) -> None:
    """Use a shared report lock and a unique marker across monitor restarts."""
    marker = f"[overnight-baseline-{key}]"
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = PROGRESS.read_text(encoding="utf-8") if PROGRESS.is_file() else ""
        if marker in previous:
            return
        with PROGRESS.open("a", encoding="utf-8") as output:
            output.write(f"\n- {utc()} {marker} {message}\n")
            output.flush()
            os.fsync(output.fileno())


def monitor() -> None:
    """Poll atomic status files until the campaign ends or the report deadline."""
    for _ in range(100):
        if LAUNCH.is_file():
            break
        time.sleep(0.1)
    manifest = read(LAUNCH)
    if manifest.get("script_sha256") != sha(SCRIPT):
        raise ValueError("milestone monitor source changed after launch")
    if sha(ROOT / "plan.json") != PLAN_SHA:
        raise ValueError("held-out baseline plan changed")
    try:
        while datetime.now(timezone.utc) < DEADLINE:
            state = read(ROOT / "status.json")
            admission = read(ADMISSION) if ADMISSION.is_file() else None
            if admission is not None:
                checksum = ADMISSION.with_suffix(".sha256")
                if (not checksum.is_file()
                        or not checksum.read_text(encoding="utf-8").startswith(sha(ADMISSION))):
                    raise ValueError("frozen admission checksum differs")
            for key, message in events(state, admission):
                append_once(key, message)
            _atomic_bytes(STATUS, _json_bytes({
                "schema_version": "overnight-baseline-milestone-monitor-v1",
                "phase": "watching", "updated_at_utc": utc(),
                "baseline_phase": state.get("phase"),
                "completed_edges": state.get("completed_edges"),
                "admission_frozen": admission is not None,
            }))
            if state.get("phase") in (
                "complete", "complete_with_failures", "stopped_before_d2", "failed",
            ):
                break
            time.sleep(30)
        _atomic_bytes(STATUS, _json_bytes({
            "schema_version": "overnight-baseline-milestone-monitor-v1",
            "phase": "finished", "updated_at_utc": utc(),
        }))
    except Exception as exc:
        _atomic_bytes(STATUS, _json_bytes({
            "schema_version": "overnight-baseline-milestone-monitor-v1",
            "phase": "failed", "updated_at_utc": utc(),
            "error": f"{type(exc).__name__}: {exc}",
        }))
        raise


def launch() -> dict[str, Any]:
    """Detach one zero-tool monitor; leave the test driver untouched."""
    if LAUNCH.exists():
        raise FileExistsError("milestone monitor already launched")
    if sha(ROOT / "plan.json") != PLAN_SHA:
        raise ValueError("held-out baseline plan changed")
    with (ROOT / "milestone_monitor.stdout").open("wb") as stdout, (
        ROOT / "milestone_monitor.stderr"
    ).open("wb") as stderr:
        child = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--run"], cwd=PROJECT,
            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            start_new_session=True, close_fds=True,
        )
    manifest = {
        "schema_version": "overnight-baseline-milestone-monitor-launch-v1",
        "pid": child.pid, "started_at_utc": utc(),
        "script_sha256": sha(SCRIPT), "plan_sha256": PLAN_SHA,
    }
    _atomic_bytes(LAUNCH, _json_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        state = read(ROOT / "status.json")
        admission = read(ADMISSION) if ADMISSION.is_file() else None
        print(json.dumps({"events": [key for key, _ in events(state, admission)],
                          "tool_jobs_launched": 0}, sort_keys=True))
    elif args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.run:
        monitor()
    else:
        print(json.dumps(read(STATUS) if STATUS.is_file() else
                         {"phase": "not_launched"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
