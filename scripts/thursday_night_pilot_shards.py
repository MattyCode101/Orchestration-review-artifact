"""Run the frozen Thursday training pilot in disjoint process shards.

This wrapper leaves the pinned pilot and engine unchanged. Each process owns a
slice of the 140-cell schedule, while the existing per-key cache locks and
persistent 300-run reservation counter remain shared. The wrapper's bytes are
pinned in sharded_launch.json. No held-out outcome path is opened here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from orch.baselines.verification import VerificationLedger
from orch.config import load_noise
from orch.contract.brief_text import load_frozen_brief
from orch.engine.apply import StepEngine
from orch.engine.runner import Runner
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from orch.episode import run_episode
from orch.runtime import make_real_runtime
from orch.workers.telemetry import SavedTelemetryProvider
from scripts import thursday_night_pilot as pilot

SCRIPT = Path(__file__).resolve()
ROOT = pilot.ROOT
PROJECT = pilot.PROJECT
SHARDS = 8
LAUNCH = ROOT / "sharded_launch.json"
STATUS = ROOT / "sharded_status.json"


def rows_for(shard: int) -> tuple[tuple[str, str, str], ...]:
    """Partition the frozen schedule without changing its order or coverage."""
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid pilot shard")
    return pilot.episode_schedule()[shard::SHARDS]


def _write_shard(shard: int, phase: str, completed: int, error: str | None = None) -> None:
    _atomic_bytes(ROOT / f"shard_{shard}.json", _json_bytes({
        "schema_version": "thursday-night-pilot-shard-v1",
        "shard": shard, "planned": len(rows_for(shard)),
        "processed": completed, "phase": phase, "error": error,
        "updated_at_utc": pilot.utc(),
    }))


def _manifest() -> dict[str, Any]:
    value = pilot.read(LAUNCH)
    if (
        value.get("schema_version") != "thursday-night-pilot-shards-v1"
        or value.get("shards") != SHARDS
        or value.get("wrapper_sha256") != _sha256_file(SCRIPT)
        or value.get("pilot_plan_sha256") != _sha256_file(ROOT / "plan.json")
        or value.get("schedule_sha256") != hashlib.sha256(
            _json_bytes(pilot.episode_schedule())
        ).hexdigest()
    ):
        raise pilot.PilotSafetyStop("sharded pilot source or schedule differs from launch")
    return value


def run_shard(shard: int) -> None:
    """Resume only this shard's episodes, preserving the original pilot gates."""
    runtime = make_real_runtime(
        ROOT, cache_enabled=True, max_slots=pilot.MAX_SLOTS,
        load_ceiling_1m=pilot.LOAD_CEILING,
        min_available_memory_bytes=pilot.MEMORY_FLOOR,
        latch_smoke_status=pilot.SMOKE,
    )
    frozen = pilot.require_plan(runtime)
    _manifest()
    starts = pilot._seed_starts(runtime, frozen["training_inputs"])
    workers = pilot._workers(runtime)
    provider = SavedTelemetryProvider(ROOT)
    ledger = VerificationLedger(
        store=runtime.store, runner=runtime.runner,
        validator=runtime.validator, references=frozen["training_inputs"],
    )
    backend = pilot.CountedBackend(
        runtime.backend, ROOT, _sha256_file(ROOT / "plan.json"),
        lambda: pilot.require_plan(runtime),
    )
    engine = pilot.VerifiedEpisodeEngine(
        StepEngine(
            store=runtime.store, cache=runtime.cache, backend=backend,
            meter_version=runtime.meter.version, cache_enabled=True,
        ), ledger,
    )
    (ROOT / "results").mkdir(exist_ok=True)
    processed = 0
    _write_shard(shard, "running", processed)
    for design, brief_id, key in rows_for(shard):
        pilot.require_plan(runtime)
        _manifest()
        identity = pilot.episode_id(design, brief_id, key)
        path = ROOT / "results" / f"{identity}.json"
        if path.is_file():
            saved = pilot.read(path)
            if (
                saved.get("episode_id") != identity
                or saved.get("design") != design
                or saved.get("brief") != brief_id
                or saved.get("policy") != key
                or not saved.get("final_brief_valid")
            ):
                raise pilot.PilotSafetyStop(f"saved pilot result differs: {identity}")
            pilot._audit_episode(identity, runtime.store, ledger)
            processed += 1
            _write_shard(shard, "running", processed)
            continue
        pilot._audit_episode(identity, runtime.store, ledger)
        brief = load_frozen_brief(brief_id)
        start = starts[design]
        start_evidence = engine.measure_start(start)
        noise = load_noise(PROJECT / "configs/noise.yaml").for_start(
            design, start_evidence
        )
        backend.active_episode = identity
        try:
            result = run_episode(
                engine=engine, registry=runtime.registry, workers=workers,
                policy=pilot.policy_for(key), brief=brief, start_state=start,
                episode_id=identity, noise=noise, telemetry_provider=provider,
                validator=pilot._validator(ledger, brief),
            )
        except pilot.PilotRunCap as exc:
            _write_shard(shard, "capped", processed, str(exc))
            return
        finally:
            backend.active_episode = None
        pilot._audit_episode(identity, runtime.store, ledger)
        row = pilot._result_row(result, runtime.store, backend, design, brief_id, key)
        _atomic_bytes(path, _json_bytes(row))
        processed += 1
        _write_shard(shard, "running", processed)
        pilot._status("running")
    _write_shard(shard, "complete", processed)


def _alive(pid: int, shard: int) -> bool:
    """Check exact recorded PID, script path and shard argument."""
    info = pilot._stat(pid)
    if info is None or info.state == "Z" or info.pgrp != pid:
        return False
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return (
        str(SCRIPT).encode() in args
        and b"--shard" in args
        and str(shard).encode() in args
    )


def _stop_recorded(groups: dict[int, int]) -> None:
    """Stop only verified wrapper process groups, then sweep exact pilot tags."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for shard, pid in groups.items():
            if _alive(pid, shard):
                try:
                    os.killpg(pid, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    pilot.cleanup()


def watch() -> None:
    """Fail closed if any shard exits unexpectedly; preserve exact-tag cleanup."""
    manifest = _manifest()
    groups = {int(key): int(value) for key, value in manifest["driver_pids"].items()}
    if set(groups) != set(range(SHARDS)):
        raise pilot.PilotSafetyStop("shard launch lacks an exact driver PID set")
    deadline = time.monotonic() + 36 * 3600
    while time.monotonic() < deadline:
        states = {
            shard: pilot.read(ROOT / f"shard_{shard}.json")
            if (ROOT / f"shard_{shard}.json").is_file() else {}
            for shard in groups
        }
        live = {shard: _alive(pid, shard) for shard, pid in groups.items()}
        failed = [
            shard for shard in groups
            if states[shard].get("phase") == "failed"
            or (not live[shard] and states[shard].get("phase") not in ("complete", "capped"))
        ]
        if failed:
            _stop_recorded(groups)
            state = pilot._status("failed", error=f"shards failed: {failed}")
            _atomic_bytes(STATUS, _json_bytes(state))
            return
        if not any(live.values()):
            phase = "capped" if any(
                states[shard].get("phase") == "capped" for shard in groups
            ) else "complete"
            state = pilot._status(phase)
            _atomic_bytes(STATUS, _json_bytes(state))
            return
        time.sleep(15)
    _stop_recorded(groups)
    state = pilot._status("failed", error="shard watchdog lifetime exceeded")
    _atomic_bytes(STATUS, _json_bytes(state))


def launch() -> dict[str, Any]:
    """Start eight isolated drivers only after the old serial pair has exited."""
    old = pilot.read(ROOT / "launch.json")
    old_pid = old.get("driver_pid")
    old_watch = old.get("watchdog_pid")
    if type(old_pid) is not int or type(old_watch) is not int:
        raise pilot.PilotSafetyStop("serial pilot launch is missing recorded PIDs")
    if pilot.driver_alive(old_pid):
        raise pilot.PilotSafetyStop("serial pilot driver is still running")
    old_info = pilot._stat(old_watch)
    if old_info is not None and old_info.state != "Z":
        raise pilot.PilotSafetyStop("serial pilot watchdog is still running")
    runtime = make_real_runtime(
        ROOT, cache_enabled=True, max_slots=pilot.MAX_SLOTS,
        load_ceiling_1m=pilot.LOAD_CEILING,
        min_available_memory_bytes=pilot.MEMORY_FLOOR,
        latch_smoke_status=pilot.SMOKE,
    )
    pilot.require_plan(runtime)
    pilot._seed_starts(runtime, pilot._training_inputs())
    pilot._workers(runtime)
    SavedTelemetryProvider(ROOT)
    load, available = Runner._resource_snapshot()
    if load > pilot.LOAD_CEILING or available < pilot.MEMORY_FLOOR:
        raise pilot.PilotSafetyStop("load or available memory blocks sharded launch")
    with _file_lock(ROOT / "sharded_launch.lock"):
        if LAUNCH.exists():
            raise FileExistsError("sharded training pilot already launched")
        frozen = {
            "schema_version": "thursday-night-pilot-shards-v1",
            "pilot_plan_sha256": _sha256_file(ROOT / "plan.json"),
            "wrapper_sha256": _sha256_file(SCRIPT),
            "schedule_sha256": hashlib.sha256(
                _json_bytes(pilot.episode_schedule())
            ).hexdigest(),
            "shards": SHARDS, "driver_pids": {},
            "started_at_utc": pilot.utc(),
            "load_1m_at_launch": load,
            "available_memory_bytes_at_launch": available,
        }
        _atomic_bytes(LAUNCH, _json_bytes(frozen))
        spawned: dict[int, int] = {}
        try:
            for shard in range(SHARDS):
                with (ROOT / f"shard_{shard}.stdout").open("wb") as out, (
                    ROOT / f"shard_{shard}.stderr"
                ).open("wb") as err:
                    child = subprocess.Popen(
                        [sys.executable, str(SCRIPT), "--shard", str(shard)],
                        cwd=PROJECT, stdin=subprocess.DEVNULL,
                        stdout=out, stderr=err,
                        start_new_session=True, close_fds=True,
                    )
                spawned[shard] = child.pid
            frozen["driver_pids"] = {str(key): pid for key, pid in spawned.items()}
            _atomic_bytes(LAUNCH, _json_bytes(frozen))
            with (ROOT / "shard_watchdog.stdout").open("wb") as out, (
                ROOT / "shard_watchdog.stderr"
            ).open("wb") as err:
                watcher = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "--watch"],
                    cwd=PROJECT, stdin=subprocess.DEVNULL,
                    stdout=out, stderr=err,
                    start_new_session=True, close_fds=True,
                )
            frozen["watchdog_pid"] = watcher.pid
            _atomic_bytes(LAUNCH, _json_bytes(frozen))
            pilot._status("running")
            return frozen
        except Exception:
            _stop_recorded(spawned)
            raise


def status() -> dict[str, Any]:
    manifest = _manifest()
    groups = {int(key): int(value) for key, value in manifest["driver_pids"].items()}
    return {
        "pilot": pilot.status(),
        "shards": {
            str(shard): {
                **(pilot.read(ROOT / f"shard_{shard}.json")
                   if (ROOT / f"shard_{shard}.json").is_file() else {}),
                "alive": _alive(pid, shard),
            }
            for shard, pid in groups.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--launch", action="store_true")
    options.add_argument("--watch", action="store_true")
    options.add_argument("--status", action="store_true")
    options.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.watch:
        watch()
    elif args.status:
        print(json.dumps(status(), sort_keys=True))
    else:
        shard = args.shard
        try:
            run_shard(shard)
        except Exception as exc:
            _write_shard(shard, "failed", 0, f"{type(exc).__name__}: {exc}")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

