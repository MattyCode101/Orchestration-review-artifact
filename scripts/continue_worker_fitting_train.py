"""Detached fail-closed continuation of the five-design training sweep.

Wait for every AMD validator PASS, audit every saved attempt, refresh DCP
aliases, run the software suite, then call the existing verified resume once.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PYTHON = PROJECT / ".venv/bin/python"
ROOT = PROJECT / "data/sweeps/worker_fitting_train"
VERIFY = ROOT / "verification"
WORK = VERIFY / "continuation"
FINGERPRINT = "54b00e8ca68aa336aaea8caa1ab98db86743d067d5ddbc03a64f5c80691373af"
DCPS = 460
EDGES = 708


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gate(state: dict) -> str:
    """Fail on changed scope, any non-PASS, or inconsistent progress."""
    if (state.get("mode") != "backfill"
            or state.get("validator_fingerprint") != FINGERPRINT
            or state.get("total_distinct_dcps") != DCPS):
        raise ValueError("backfill scope or validator fingerprint changed")
    statuses = state.get("statuses") or {}
    done = state.get("completed")
    if state.get("errors") or any(k != "pass" and v for k, v in statuses.items()):
        raise RuntimeError(f"backfill recorded non-PASS: {statuses}")
    if type(done) is not int or not 0 <= done <= DCPS or statuses.get("pass", 0) != done:
        raise ValueError("backfill PASS count differs from completed count")
    if state.get("phase") == "running":
        return "wait"
    if state.get("phase") == "complete" and done == DCPS and statuses == {"pass": DCPS}:
        return "complete"
    raise RuntimeError(f"backfill is incomplete or failed: {state.get('phase')} {done}/{DCPS}")


def _driver_alive(pid: int) -> bool:
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return False
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
    except (OSError, IndexError):
        return False
    return (state != "Z" and
            str(PROJECT / "scripts/verify_worker_fitting_train.py").encode() in args
            and b"--run-backfill" in args)


def _signal_backfill(pid: int, sig: signal.Signals = signal.SIGTERM) -> bool:
    """Stop only the exact recorded validator driver's private group."""
    if not _driver_alive(pid):
        return False
    group = os.getpgid(pid)
    if group != pid or group == os.getpgrp():
        raise RuntimeError("unsafe backfill process group")
    os.killpg(group, sig)
    return True


def _stop_backfill_and_audit(pid: int) -> dict:
    """Stop the exact group, then require watchdog or direct exact-tag cleanup."""
    term = _signal_backfill(pid, signal.SIGTERM)
    for _ in range(15):
        if not _driver_alive(pid):
            break
        time.sleep(1)
    kill = _signal_backfill(pid, signal.SIGKILL) if _driver_alive(pid) else False
    if _driver_alive(pid):
        raise RuntimeError("exact backfill driver survived SIGKILL")
    audit_path = VERIFY / "backfill_watchdog_cleanup_audit.json"
    # The existing watchdog polls every 30 seconds. Leave it time to sweep
    # detached descendants before falling back to the same scoped routine.
    for _ in range(12):
        if audit_path.is_file() and _read(audit_path).get("finished_at_utc"):
            break
        time.sleep(10)
    from scripts import watch_worker_fitting_verification as watcher
    if watcher.ROOT != VERIFY:
        raise RuntimeError("validator watchdog selected another verification root")
    if not audit_path.is_file() or not _read(audit_path).get("finished_at_utc"):
        audit = watcher._cleanup("backfill")
    else:
        audit = _read(audit_path)
    if audit.get("survivors") or watcher._targets()[0]:
        raise RuntimeError(f"validator exact tags survived cleanup: {audit.get('survivors')}")
    return {"term_group": term, "kill_group": kill,
            "cleanup_audit": str(audit_path), "survivors": audit.get("survivors")}


def _exact_script_alive(pid: object, script: str, flag: str) -> bool:
    """Identify one same-user monitor or recipe driver by exact argv."""
    if type(pid) is not int:
        return False
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return False
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
    except (OSError, IndexError):
        return False
    return (state != "Z" and str(PROJECT / "scripts" / script).encode() in args
            and flag.encode() in args)


def _resume_monitors() -> dict[str, bool]:
    record = _read(ROOT / "resume_manifest.json")
    return {
        "driver": _exact_script_alive(record.get("driver_pid"),
                                      "launch_worker_fitting_train.py", "--run"),
        "watchdog": _exact_script_alive(record.get("watchdog_pid"),
                                        "watch_worker_fitting_train.py", "--watch"),
        "indexer": _exact_script_alive(record.get("indexer_pid"),
                                       "index_worker_fitting_dcps.py", "--watch"),
    }


def _stop_unmonitored_resume() -> dict:
    """Stop an exact resumed driver; sweep tags if its watchdog did not."""
    manifest = ROOT / "resume_manifest.json"
    if not manifest.is_file():
        return {"resume_manifest": False}
    record = _read(manifest)
    monitors = _resume_monitors()
    pid = record.get("driver_pid")
    signalled = False
    if monitors["driver"]:
        group = os.getpgid(pid)
        if group != pid or group == os.getpgrp():
            raise RuntimeError("unsafe resumed driver process group")
        os.killpg(group, signal.SIGTERM)
        signalled = True
    audit_path = ROOT / "watchdog_verified_cleanup_audit.json"
    if monitors["watchdog"] and signalled:
        # Give the already-running scoped watchdog its normal cleanup window.
        for _ in range(9):
            if audit_path.is_file() and _read(audit_path).get("finished_at_utc"):
                break
            time.sleep(10)
    if not audit_path.is_file() or not _read(audit_path).get("finished_at_utc"):
        from scripts import watch_worker_fitting_train as watcher
        if watcher.MANIFEST != manifest:
            raise RuntimeError("watchdog selected a different resume manifest")
        audit = watcher._cleanup()
    else:
        audit = _read(audit_path)
    if audit.get("survivors"):
        raise RuntimeError(f"resumed recipe tags survived cleanup: {audit['survivors']}")
    return {"resume_manifest": True, "driver_signalled": signalled,
            "monitor_state": monitors, "cleanup_audit": str(audit_path)}


def _status(state: dict) -> None:
    state["updated_at_utc"] = _utc()
    _atomic_bytes(WORK / "status.json", _json_bytes(state))


def _command(name: str, args: list[str], timeout: int = 3600) -> str:
    """Retain exact command output and stop on any nonzero exit."""
    stdout = WORK / f"{name}.stdout"
    stderr = WORK / f"{name}.stderr"
    with stdout.open("wb") as out, stderr.open("wb") as err:
        result = subprocess.run(args, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{name} exited {result.returncode}; inspect {stdout} and {stderr}")
    return stdout.read_text(encoding="utf-8")


def _json_command(name: str, args: list[str], timeout: int = 3600) -> dict:
    lines = [line for line in _command(name, args, timeout).splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{name} produced no JSON")
    return json.loads(lines[-1])


def _append(path: Path, line: str) -> None:
    with _file_lock(path.with_suffix(".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def run() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": "worker-fitting-continuation-v1",
             "phase": "waiting_for_backfill", "pid": os.getpid(),
             "started_at_utc": _utc(), "validator_fingerprint": FINGERPRINT}
    _status(state)
    try:
        if _read(VERIFY / "plan.json")["fingerprint"] != FINGERPRINT:
            raise ValueError("frozen validator plan changed")
        pid = _read(VERIFY / "backfill_launch.json")["pid"]
        if type(pid) is not int:
            raise ValueError("backfill has no exact driver PID")
        deadline = time.monotonic() + 35 * 3600
        while time.monotonic() < deadline:
            current = _read(VERIFY / "backfill_status.json")
            try:
                decision = gate(current)
            except Exception:
                state["backfill_cleanup"] = _stop_backfill_and_audit(pid)
                _status(state)
                raise
            state["backfill_completed"] = current["completed"]
            _status(state)
            if decision == "wait":
                if not _driver_alive(pid):
                    raise RuntimeError("backfill driver exited before complete PASS")
                time.sleep(30)
                continue
            if _driver_alive(pid):
                time.sleep(10)
                continue
            break
        else:
            state["backfill_cleanup"] = _stop_backfill_and_audit(pid)
            raise TimeoutError("backfill exceeded 35 hours; exact group and tags cleaned")
        if gate(_read(VERIFY / "backfill_summary.json")) != "complete":
            raise RuntimeError("backfill summary is incomplete")
        pilot = _read(VERIFY / "pilot_summary.json")
        if (pilot.get("phase") != "complete" or pilot.get("statuses") != {"pass": 5}
                or pilot.get("total_distinct_dcps") != 5
                or pilot.get("validator_fingerprint") != FINGERPRINT):
            raise RuntimeError("five-design pilot did not pass")
        tag_deadline = time.monotonic() + 300
        while True:
            tags = _json_command("exact_tags", [str(PYTHON),
                str(PROJECT / "scripts/watch_worker_fitting_verification.py"),
                "--mode", "backfill"], 60)
            if tags.get("active_exact_tags") == 0:
                break
            if time.monotonic() >= tag_deadline:
                raise RuntimeError(f"validator exact tags remain: {tags}")
            time.sleep(15)
        state["phase"] = "auditing"
        _status(state)
        audit = _json_command("audit", [str(PYTHON),
            str(PROJECT / "scripts/verify_worker_fitting_train.py"), "--audit"])
        if audit.get("latest") != {"committed_edges": EDGES,
                                   "saved_outputs": EDGES,
                                   "distinct_verified_dcps": DCPS}:
            raise ValueError(f"latest audit differs: {audit}")
        if audit.get("all_attempts") != {"journal_edge_attempts": EDGES,
                                          "distinct_sequences": EDGES,
                                          "saved_output_attempts": EDGES,
                                          "distinct_verified_dcps": DCPS}:
            raise ValueError(f"all-attempt audit differs: {audit}")
        state["coverage"] = audit
        index = _json_command("index", [str(PYTHON),
            str(PROJECT / "scripts/index_worker_fitting_dcps.py"), "--once"])
        if (index.get("committed_edges") != EDGES
                or index.get("saved_attempt_outputs") != EDGES
                or index.get("unique_output_states") != DCPS
                or index.get("unverified_aliases") != 0):
            raise ValueError(f"DCP index is incomplete: {index}")
        state["index"] = index
        state["phase"] = "testing"
        _status(state)
        _command("pytest", [str(PYTHON), "-m", "pytest", "-q"], 900)
        state["pytest"] = "passed"
        state["phase"] = "resuming"
        _status(state)
        resumed = _json_command("resume", [str(PYTHON),
            str(PROJECT / "scripts/launch_worker_fitting_train.py"), "--resume"], 300)
        if (resumed.get("phase") != "resumed"
                or resumed.get("validator_fingerprint") != FINGERPRINT
                or resumed.get("coverage") != audit["latest"]
                or resumed.get("all_attempts_coverage") != audit["all_attempts"]):
            raise ValueError(f"resume differs from audit: {resumed}")
        state["resume"] = resumed
        time.sleep(20)
        campaign = _json_command("campaign_status", [str(PYTHON),
            str(PROJECT / "scripts/launch_worker_fitting_train.py"), "--status"], 120)
        if not campaign.get("driver_alive") and campaign.get("phase") != "complete":
            raise RuntimeError(f"resumed driver exited early: {campaign}")
        monitors = _resume_monitors()
        if campaign.get("phase") != "complete" and not all(monitors.values()):
            raise RuntimeError(f"resumed driver lacks a live watchdog or indexer: {monitors}")
        state["monitor_state"] = monitors
        state["campaign_status"] = campaign
        state["phase"] = "resumed"
        _status(state)
        line = (f"\n- Verified automatic recovery at {_utc()}: AMD backfill "
                f"{DCPS}/{DCPS} PASS, all {EDGES} saved attempts audited, "
                f"DCP index all PASS, software suite passed, and verified "
                f"D3/D4 driver PID {resumed['driver_pid']} launched.\n")
        try:
            for path in (PROJECT / "notes/progress.md",
                         PROJECT / "notes/worker_sweep_launch_report.md",
                         PROJECT / "notes/worker_sweep_progress.md"):
                _append(path, line)
        except OSError as exc:
            state["note_error"] = f"{type(exc).__name__}: {exc}"
            _status(state)
        return state
    except Exception as exc:
        if state.get("phase") == "resuming":
            try:
                state["resume_cleanup"] = _stop_unmonitored_resume()
            except Exception as cleanup_exc:
                state["resume_cleanup_error"] = (
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
        state["phase"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        _status(state)
        _append(PROJECT / "notes/progress.md",
                f"\n- Continuation stopped at {_utc()}: {state['error']}.\n")
        raise


def launch() -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    with _file_lock(WORK / "launch.lock"):
        if (WORK / "launch.json").exists():
            raise FileExistsError("continuation already launched")
        with (WORK / "driver.stdout").open("wb") as out, \
             (WORK / "driver.stderr").open("wb") as err:
            process = subprocess.Popen([str(PYTHON), str(SCRIPT), "--run"],
                cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True)
        record = {"pid": process.pid, "launched_at_utc": _utc(),
                  "stdout": str(WORK / "driver.stdout"),
                  "stderr": str(WORK / "driver.stderr")}
        _atomic_bytes(WORK / "launch.json", _json_bytes(record))
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("launch", "run", "status", "check"):
        modes.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    if args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.run:
        print(json.dumps(run(), sort_keys=True))
    elif args.status:
        result = _read(WORK / "status.json") if (WORK / "status.json").is_file() else {"phase": "not_started"}
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps({"decision": gate(_read(VERIFY / "backfill_status.json"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
