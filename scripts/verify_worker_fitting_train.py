"""Detached pilot and backfill of AMD checks on saved training DCPs.

The pilot validates one netlist-changing D1 output per design. Backfill checks
every distinct checkpoint named by the checked campaign journal. A failed or
unknown result blocks campaign resume; all outputs remain saved.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from orch.baselines.verification import VerificationLedger, audit_committed
from orch.baselines.verification_guard import audit_all_saved
from orch.baselines.worker_fitting import CAMPAIGN_ID, TRAIN_DESIGNS
from orch.engine.journal import Journal
from orch.engine.store import _atomic_bytes, _json_bytes
from orch.runtime import make_real_runtime

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data" / "sweeps" / "worker_fitting_train"
VERIFY = ROOT / "verification"
SCRIPT = Path(__file__).resolve()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _references() -> dict[str, str]:
    manifest = json.loads((ROOT / "launch_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["train_inputs"]
    if tuple(row["design"] for row in rows) != TRAIN_DESIGNS:
        raise ValueError("verification manifest is not exactly five training designs")
    return {row["design"]: row["sha256"] for row in rows}


def _tasks(mode: str) -> list[tuple[str, str]]:
    """Select exact, journaled outputs without reading held-out data."""
    latest: dict[tuple[str, tuple[str, ...]], dict] = {}
    for event in Journal(ROOT).read(CAMPAIGN_ID):
        if event.get("type") != "edge":
            continue
        edge = event["edge"]
        if edge["design"] not in TRAIN_DESIGNS:
            raise ValueError("journal edge names a held-out design")
        latest[(edge["design"], tuple(edge["sequence"]))] = edge
    if mode == "pilot":
        tasks = []
        for design in TRAIN_DESIGNS:
            edge = latest.get((design, ("A_R_EA",)))
            if edge is None or not edge["physically_valid"] or not edge["output_state_id"]:
                raise ValueError(f"pilot lacks valid EA output on {design}")
            tasks.append((design, edge["output_state_id"]))
        return tasks
    return sorted({
        (edge["design"], edge["output_state_id"])
        for edge in latest.values() if edge.get("output_state_id")
    })


def _write_status(mode: str, payload: dict[str, object]) -> None:
    _atomic_bytes(VERIFY / f"{mode}_status.json", _json_bytes(payload))


def run(mode: str) -> dict[str, object]:
    """Validate selected DCPs in tagged jobs, recording every completion."""
    refs = _references()
    runtime = make_real_runtime(ROOT, cache_enabled=True, max_slots=48)
    ledger = VerificationLedger(
        store=runtime.store, runner=runtime.runner,
        validator=runtime.validator, references=refs,
    )
    tasks = _tasks(mode)
    expected = 5 if mode == "pilot" else None
    if expected is not None and len(tasks) != expected:
        raise ValueError("pilot must contain five unique training DCPs")
    concurrency = 5 if mode == "pilot" else 24
    state: dict[str, object] = {
        "schema_version": "worker-fitting-verification-batch-v1",
        "mode": mode, "phase": "running", "pid": os.getpid(),
        "total_distinct_dcps": len(tasks), "completed": 0,
        "statuses": {}, "errors": [],
        "concurrency": concurrency, "started_at_utc": _utc(),
        "updated_at_utc": _utc(), "validator_fingerprint": ledger.fingerprint,
    }
    _write_status(mode, state)
    counts: Counter[str] = Counter()
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(ledger.verify, design, output_id): (design, output_id)
            for design, output_id in tasks
        }
        for future in as_completed(futures):
            design, output_id = futures[future]
            try:
                record = future.result()
                counts[str(record["result"])] += 1
            except Exception as exc:
                counts["error"] += 1
                errors.append(f"{design} {output_id}: {type(exc).__name__}: {exc}")
            state["completed"] = sum(counts.values())
            state["statuses"] = dict(sorted(counts.items()))
            state["errors"] = errors[:10]
            state["updated_at_utc"] = _utc()
            _write_status(mode, state)
    if mode == "backfill" and not errors and counts.get("pass", 0) == len(tasks):
        coverage = audit_committed(Journal(ROOT).read(CAMPAIGN_ID), ledger)
    else:
        coverage = None
    result = {
        **state, "phase": "complete" if not errors and counts.get("pass", 0) == len(tasks)
        else "failed",
        "finished_at_utc": _utc(), "coverage": coverage,
    }
    _write_status(mode, result)
    _atomic_bytes(VERIFY / f"{mode}_summary.json", _json_bytes(result))
    if mode == "backfill":
        subprocess.run(
            [sys.executable, str(PROJECT / "scripts/index_worker_fitting_dcps.py"), "--once"],
            cwd=PROJECT, check=True,
        )
    return result


def _alive(pid: int, mode: str) -> bool:
    try:
        args = [
            item.decode("utf-8", errors="replace")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
            if item
        ]
        return str(SCRIPT) in args and f"--run-{mode}" in args
    except OSError:
        return False


def launch(mode: str) -> dict[str, object]:
    """Launch one detached process; retain old attempts and logs on resume."""
    VERIFY.mkdir(parents=True, exist_ok=True)
    latest = VERIFY / f"{mode}_launch.json"
    if latest.is_file():
        prior = json.loads(latest.read_text(encoding="utf-8"))
        if isinstance(prior.get("pid"), int) and _alive(prior["pid"], mode):
            raise RuntimeError(f"{mode} validator batch is already running")
    token = uuid.uuid4().hex
    stdout_path = VERIFY / f"{mode}_{token}.stdout"
    stderr_path = VERIFY / f"{mode}_{token}.stderr"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), f"--run-{mode}"],
            cwd=PROJECT, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr,
            start_new_session=True, close_fds=True,
        )
    watchdog = PROJECT / "scripts/watch_worker_fitting_verification.py"
    watchdog_stdout = VERIFY / f"{mode}_{token}.watchdog.stdout"
    watchdog_stderr = VERIFY / f"{mode}_{token}.watchdog.stderr"
    with watchdog_stdout.open("wb") as stdout, watchdog_stderr.open("wb") as stderr:
        watcher = subprocess.Popen(
            [sys.executable, str(watchdog), "--watch", "--mode", mode,
             "--pid", str(process.pid)],
            cwd=PROJECT, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr,
            start_new_session=True, close_fds=True,
        )
    result = {
        "mode": mode, "pid": process.pid, "watchdog_pid": watcher.pid,
        "launched_at_utc": _utc(), "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "watchdog_stdout": str(watchdog_stdout),
        "watchdog_stderr": str(watchdog_stderr),
    }
    _atomic_bytes(latest, _json_bytes(result))
    return result


def status() -> dict[str, object]:
    result = {}
    for mode in ("pilot", "backfill"):
        path = VERIFY / f"{mode}_status.json"
        result[mode] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file() else {"phase": "not_started"}
        )
        launch_path = VERIFY / f"{mode}_launch.json"
        if launch_path.is_file():
            launch_info = json.loads(launch_path.read_text(encoding="utf-8"))
            pid = launch_info.get("pid")
            result[mode]["driver_alive"] = isinstance(pid, int) and _alive(pid, mode)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("pilot", "backfill"):
        mode.add_argument(f"--launch-{name}", action="store_true")
        mode.add_argument(f"--run-{name}", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(status(), sort_keys=True))
        return 0
    if args.audit:
        runtime = make_real_runtime(ROOT, cache_enabled=True, max_slots=48)
        ledger = VerificationLedger(
            store=runtime.store, runner=runtime.runner,
            validator=runtime.validator, references=_references(),
        )
        events = Journal(ROOT).read(CAMPAIGN_ID)
        print(json.dumps({
            "latest": audit_committed(events, ledger),
            "all_attempts": audit_all_saved(events, ledger),
        }, sort_keys=True))
        return 0
    for name in ("pilot", "backfill"):
        if getattr(args, f"launch_{name}"):
            print(json.dumps(launch(name), sort_keys=True))
            return 0
        if getattr(args, f"run_{name}"):
            result = run(name)
            print(json.dumps(result, sort_keys=True))
            return 0 if result["phase"] == "complete" else 1
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
