"""Restart the stopped training pilot under a fresh, provenance-pinned v4 root.

The v3 registry hash changed after its clean stop, so v3 episode results are
preserved but never replayed here. This wrapper reuses the frozen episode,
policy, worker, AMD, property and tagged-runner gates. Only the operational
counter rises to 600 cumulative recipe-launch reservations.
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
from datetime import datetime, timedelta, timezone
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
from orch.workers.telemetry import PILOT_TRAINING_ROOT_V4, SavedTelemetryProvider
from scripts import thursday_night_pilot as pilot

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
ROOT = PILOT_TRAINING_ROOT_V4
V3_ROOT = PROJECT / "data/pilots/thursday_night_train_v3_2026-11-05"
STOP = PROJECT / "notes/pilot_stopped.done"
REGISTRY_MARKER = PROJECT / "notes/registry_power_update.done"
PROMOTION_AUDIT = PROJECT / "data/scratch_thursday_night_2026-11-05/training_cache_promotion/audit.json"
EXPECTED_TELEMETRY_SHA256 = "a526c71c58f679af8f74b9c8a1262ac353de8bbdf9398b823365f925d7d7bfba"
CAP = 600
SHARDS = 8
LAUNCH = ROOT / "v4_launch.json"
STATUS = ROOT / "v4_sharded_status.json"
ORIGINAL_ROOT = pilot.ROOT
ORIGINAL_CAP = pilot.CAP
ORIGINAL_SOURCE_FILES = dict(pilot.SOURCE_FILES)
ORIGINAL_MAKE_PLAN = pilot.make_plan
_BOUND = False


def _read(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _hash(path: Path) -> str:
    return _sha256_file(path)


def _exact_old_driver_alive(pid: int, script: bytes, mode: bytes) -> bool:
    info = pilot._stat(pid)
    if info is None or info.state == "Z" or info.pgrp != pid:
        return False
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return script in args and mode in args


def _stop_and_marker() -> dict[str, str]:
    """Require the authors' stop and the first of the registry marker or 60 minutes."""
    stopped = datetime.fromisoformat(STOP.read_text(encoding="utf-8").strip())
    if stopped.tzinfo is None:
        raise pilot.PilotSafetyStop("pilot stop marker lacks UTC timezone")
    audit_path = V3_ROOT / "pilot_requested_stop_audit.json"
    audit = _read(audit_path)
    if (audit.get("stopped_at_utc") != stopped.isoformat()
            or audit.get("survivors") != {}
            or audit.get("remaining_recorded_groups") != {}
            or audit.get("cleanup", {}).get("survivors") != {}):
        raise pilot.PilotSafetyStop("v3 pilot stop or exact-tag cleanup differs")
    launch = _read(V3_ROOT / "sharded_launch.json")
    old_script = str(PROJECT / "scripts/thursday_night_pilot_shards.py").encode()
    for raw_pid in launch.get("driver_pids", {}).values():
        if _exact_old_driver_alive(int(raw_pid), old_script, b"--shard"):
            raise pilot.PilotSafetyStop("a recorded v3 shard is still running")
    watchdog = launch.get("watchdog_pid")
    if type(watchdog) is int and _exact_old_driver_alive(
        watchdog, old_script, b"--watch"
    ):
        raise pilot.PilotSafetyStop("the recorded v3 watchdog is still running")
    marker_sha = ""
    if REGISTRY_MARKER.is_file():
        content = REGISTRY_MARKER.read_text(encoding="utf-8").splitlines()
        if len(content) != 2 or content[0] != "updated" or not content[1].startswith("registry_sha256="):
            raise pilot.PilotSafetyStop("registry update marker is malformed")
        expected_registry = content[1].split("=", 1)[1]
        if _hash(PROJECT / "recipes/registry.yaml") != expected_registry:
            raise pilot.PilotSafetyStop("registry marker does not match current registry")
        marker_sha = _hash(REGISTRY_MARKER)
    elif datetime.now(timezone.utc) < stopped + timedelta(minutes=60):
        raise pilot.PilotSafetyStop("the 60-minute pilot restart hold has not elapsed")
    return {
        "stopped_at_utc": stopped.isoformat(),
        "stop_marker_sha256": _hash(STOP),
        "stop_audit_sha256": _hash(audit_path),
        "registry_marker_sha256": marker_sha,
    }


def _require_expected_source_changes(
    old_hashes: dict[str, str], current_hashes: dict[str, str]
) -> None:
    """Only the approved registry append and v4 root allowlist may differ."""
    changed = {
        key for key, old_hash in old_hashes.items()
        if current_hashes.get(key) != old_hash
    }
    if changed != {"registry", "telemetry"}:
        raise pilot.PilotSafetyStop(
            f"unexpected frozen v3 source changes: {sorted(changed)}"
        )


def _v3_provenance() -> dict[str, Any]:
    """Pin old evidence and reject any unrelated code or worker-model drift."""
    release = _stop_and_marker()
    old_plan_path = V3_ROOT / "plan.json"
    old = _read(old_plan_path)
    if old.get("schema_version") != "thursday-night-pilot-v1":
        raise pilot.PilotSafetyStop("v3 pilot plan schema differs")
    if old.get("data_root") != str(V3_ROOT):
        raise pilot.PilotSafetyStop("v3 plan root differs")
    current_hashes = {
        key: _hash(PROJECT / relative)
        for key, relative in ORIGINAL_SOURCE_FILES.items()
    }
    _require_expected_source_changes(old["source_hashes"], current_hashes)
    if current_hashes["telemetry"] != EXPECTED_TELEMETRY_SHA256:
        raise pilot.PilotSafetyStop("v4 telemetry allowlist source changed")
    if REGISTRY_MARKER.is_file() and (
        current_hashes["registry"] ==
        old["source_hashes"]["registry"]
    ):
        raise pilot.PilotSafetyStop("new registry marker did not change v3 registry")
    old_shards = _read(V3_ROOT / "sharded_launch.json")
    if old_shards.get("wrapper_sha256") != _hash(
        PROJECT / "scripts/thursday_night_pilot_shards.py"
    ):
        raise pilot.PilotSafetyStop("v3 shard wrapper changed")
    results = sorted((V3_ROOT / "results").glob("*.json"))
    if len(results) != 9:
        raise pilot.PilotSafetyStop("v3 completed-episode count changed")
    ledger_path = V3_ROOT / "new_real_runs.json"
    ledger = _read(ledger_path)
    rows = ledger.get("reservations")
    if (ledger.get("schema_version") != "thursday-night-new-runs-v1"
            or ledger.get("cap") != 300
            or ledger.get("plan_sha256") != _hash(old_plan_path)
            or not isinstance(rows, list) or len(rows) != 62):
        raise pilot.PilotSafetyStop("v3 cumulative recipe reservation ledger changed")
    run_ids = [row.get("run_id") for row in rows if isinstance(row, dict)]
    if (len(run_ids) != len(rows)
            or any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
            or len(set(run_ids)) != len(run_ids)):
        raise pilot.PilotSafetyStop("v3 reservation IDs are missing or duplicated")
    stop_audit = _read(V3_ROOT / "pilot_requested_stop_audit.json")
    if (stop_audit["status_before"]["completed_episodes"] != len(results)
            or stop_audit["status_before"]["new_real_runs_reserved"] != len(rows)):
        raise pilot.PilotSafetyStop("v3 stop snapshot and preserved results differ")
    return {
        **release,
        "v3_plan_sha256": _hash(old_plan_path),
        "v3_registry_sha256": old["source_hashes"]["registry"],
        "new_registry_sha256": current_hashes["registry"],
        "v3_reservation_ledger_sha256": _hash(ledger_path),
        "v3_reservations": len(rows),
        "v3_completed_results": {
            path.name: _hash(path) for path in results
        },
        "episode_reuse": False,
        "episode_reuse_reason": "registry hash changed after v3 stop",
    }


def _new_plan(runtime) -> dict[str, Any]:
    frozen = ORIGINAL_MAKE_PLAN(runtime)
    prior = _v3_provenance()
    if frozen["data_root"] != str(ROOT) or frozen["max_new_real_recipe_runs"] != CAP:
        raise pilot.PilotSafetyStop("v4 root or recipe cap differs")
    old = _read(V3_ROOT / "plan.json")
    if frozen["recipe_versions"] != old["recipe_versions"]:
        raise pilot.PilotSafetyStop("one of ten worker recipe versions changed")
    for key in ("A_fit", "B_fit", "timing_card", "area_card", "recipe_sets"):
        if frozen["source_hashes"][key] != old["source_hashes"][key]:
            raise pilot.PilotSafetyStop(f"frozen worker or recipe-set hash changed: {key}")
    return {
        **frozen,
        "schema_version": "thursday-overnight-pilot-v4",
        "prior_v3": prior,
        "episode_reuse": False,
    }


def _bind_v4() -> None:
    """Bind the unchanged v3 helpers to an explicit new root in this process."""
    global _BOUND
    if _BOUND:
        return
    if ORIGINAL_ROOT != V3_ROOT or ORIGINAL_CAP != 300:
        raise pilot.PilotSafetyStop("original pilot root or cap differs")
    pilot.ROOT = ROOT
    pilot.CAP = CAP
    pilot.SOURCE_FILES = {
        **ORIGINAL_SOURCE_FILES,
        "v4_driver": "scripts/thursday_overnight_pilot_v4.py",
    }
    pilot.make_plan = _new_plan
    _BOUND = True


def _runtime():
    return make_real_runtime(
        ROOT, cache_enabled=True, max_slots=pilot.MAX_SLOTS,
        load_ceiling_1m=pilot.LOAD_CEILING,
        min_available_memory_bytes=pilot.MEMORY_FLOOR,
        latch_smoke_status=pilot.SMOKE,
    )


def _carryover_rows(old_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count prior reservations against 600 without charging a v4 episode."""
    run_ids = [row.get("run_id") for row in old_rows]
    if (any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
            or len(set(run_ids)) != len(run_ids)):
        raise pilot.PilotSafetyStop("historical reservation IDs are invalid")
    return [
        {
            **row,
            "original_episode_id": row["episode_id"],
            "episode_id": f"carryover:v3:{index}",
            "source_root": str(V3_ROOT),
        }
        for index, row in enumerate(old_rows)
    ]


def _seed_counter(plan_sha: str) -> None:
    old = _read(V3_ROOT / "new_real_runs.json")
    expected_old_sha = _read(ROOT / "plan.json")["prior_v3"]["v3_reservation_ledger_sha256"]
    if _hash(V3_ROOT / "new_real_runs.json") != expected_old_sha:
        raise pilot.PilotSafetyStop("v3 reservation ledger differs from frozen v4 plan")
    expected_rows = _carryover_rows(old["reservations"])
    expected = {
        "schema_version": "thursday-night-new-runs-v1",
        "plan_sha256": plan_sha, "cap": CAP,
        "reservations": expected_rows,
    }
    path = ROOT / "new_real_runs.json"
    with _file_lock(ROOT / "new_real_runs.lock"):
        if path.is_file():
            actual = _read(path)
            if (actual.get("schema_version") != expected["schema_version"]
                    or actual.get("plan_sha256") != plan_sha
                    or actual.get("cap") != CAP
                    or actual.get("reservations", [])[:len(expected_rows)] != expected_rows):
                raise pilot.PilotSafetyStop("v4 counter or historical carryover differs")
        else:
            _atomic_bytes(path, _json_bytes(expected))
    audit_path = ROOT / "carryover_audit.json"
    audit = {
        "schema_version": "thursday-overnight-carryover-v1",
        "source_ledger_sha256": _hash(V3_ROOT / "new_real_runs.json"),
        "destination_plan_sha256": plan_sha,
        "carryover_reservations": len(expected_rows),
        "cap": CAP,
    }
    if audit_path.is_file() and _read(audit_path) != audit:
        raise pilot.PilotSafetyStop("v4 carryover audit changed")
    _atomic_bytes(audit_path, _json_bytes(audit))


def prepare() -> dict[str, Any]:
    """Freeze the new plan, five B2 checkpoints and cumulative counter offline."""
    _bind_v4()
    _v3_provenance()
    runtime = _runtime()
    frozen = pilot.require_plan(runtime)
    starts = pilot._seed_starts(runtime, frozen["training_inputs"])
    pilot._workers(runtime)
    SavedTelemetryProvider(ROOT)
    plan_sha = _hash(ROOT / "plan.json")
    _seed_counter(plan_sha)
    result = {
        "schema_version": "thursday-overnight-pilot-v4-prepared",
        "prepared_at_utc": pilot.utc(),
        "plan_sha256": plan_sha,
        "meter_version": runtime.meter.version,
        "registry_sha256": frozen["source_hashes"]["registry"],
        "worker_model_sha256": {
            key: frozen["source_hashes"][key] for key in ("A_fit", "B_fit")
        },
        "start_states": len(starts),
        "carryover_reservations": 62,
        "new_real_cap": CAP,
        "episode_reuse": False,
        "tool_jobs_launched": 0,
    }
    _atomic_bytes(ROOT / "v4_prepared.json", _json_bytes(result))
    return result


def _promotion(expected_sha256: str) -> dict[str, Any]:
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise pilot.PilotSafetyStop("promotion audit SHA-256 must be exact lowercase hex")
    if not PROMOTION_AUDIT.is_file() or _hash(PROMOTION_AUDIT) != expected_sha256:
        raise pilot.PilotSafetyStop("training cache promotion audit is absent or changed")
    result = _read(PROMOTION_AUDIT)
    if (result.get("destination_root") != str(ROOT)
            or result.get("pilot_plan_sha256") != _hash(ROOT / "plan.json")
            or result.get("tool_jobs_launched") != 0
            or result.get("promoted_pointers", 0) < 1):
        raise pilot.PilotSafetyStop("training cache promotion audit fails launch gate")
    return result


def _launch_manifest(audit_sha256: str) -> dict[str, Any]:
    record = _read(LAUNCH)
    if (record.get("schema_version") != "thursday-overnight-pilot-v4-launch"
            or record.get("wrapper_sha256") != _hash(SCRIPT)
            or record.get("pilot_plan_sha256") != _hash(ROOT / "plan.json")
            or record.get("promotion_audit_sha256") != audit_sha256
            or record.get("shards") != SHARDS):
        raise pilot.PilotSafetyStop("v4 launch source, plan or approval changed")
    return record


def rows_for(shard: int) -> tuple[tuple[str, str, str], ...]:
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("invalid v4 shard")
    return pilot.episode_schedule()[shard::SHARDS]


def _write_shard(shard: int, phase: str, processed: int,
                 error: str | None = None) -> None:
    _atomic_bytes(ROOT / f"v4_shard_{shard}.json", _json_bytes({
        "schema_version": "thursday-overnight-pilot-v4-shard",
        "shard": shard, "planned": len(rows_for(shard)),
        "processed": processed, "phase": phase, "error": error,
        "updated_at_utc": pilot.utc(),
    }))


def run_shard(shard: int, audit_sha256: str) -> None:
    """Run frozen policies with the unchanged AMD/property checks and 600 cap."""
    _bind_v4()
    runtime = _runtime()
    frozen = pilot.require_plan(runtime)
    _promotion(audit_sha256)
    _launch_manifest(audit_sha256)
    starts = pilot._seed_starts(runtime, frozen["training_inputs"])
    workers = pilot._workers(runtime)
    provider = SavedTelemetryProvider(ROOT)
    ledger = VerificationLedger(
        store=runtime.store, runner=runtime.runner,
        validator=runtime.validator, references=frozen["training_inputs"],
    )
    backend = pilot.CountedBackend(
        runtime.backend, ROOT, _hash(ROOT / "plan.json"),
        lambda: pilot.require_plan(runtime), cap=CAP,
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
    for design, brief_id, policy_key in rows_for(shard):
        pilot.require_plan(runtime)
        _promotion(audit_sha256)
        _launch_manifest(audit_sha256)
        episode = pilot.episode_id(design, brief_id, policy_key)
        result_path = ROOT / "results" / f"{episode}.json"
        if result_path.is_file():
            saved = _read(result_path)
            if (saved.get("episode_id") != episode
                    or saved.get("design") != design
                    or saved.get("brief") != brief_id
                    or saved.get("policy") != policy_key
                    or saved.get("final_brief_valid") is not True):
                raise pilot.PilotSafetyStop("saved v4 result identity differs")
            pilot._audit_episode(episode, runtime.store, ledger)
            processed += 1
            _write_shard(shard, "running", processed)
            continue
        pilot._audit_episode(episode, runtime.store, ledger)
        brief = load_frozen_brief(brief_id)
        start = starts[design]
        start_evidence = engine.measure_start(start)
        noise = load_noise(PROJECT / "configs/noise.yaml").for_start(
            design, start_evidence
        )
        backend.active_episode = episode
        try:
            result = run_episode(
                engine=engine, registry=runtime.registry, workers=workers,
                policy=pilot.policy_for(policy_key), brief=brief,
                start_state=start, episode_id=episode, noise=noise,
                telemetry_provider=provider,
                validator=pilot._validator(ledger, brief),
            )
        except pilot.PilotRunCap as exc:
            _write_shard(shard, "capped", processed, str(exc))
            return
        finally:
            backend.active_episode = None
        pilot._audit_episode(episode, runtime.store, ledger)
        row = pilot._result_row(
            result, runtime.store, backend, design, brief_id, policy_key
        )
        _atomic_bytes(result_path, _json_bytes(row))
        processed += 1
        _write_shard(shard, "running", processed)
        pilot._status("running")
    _write_shard(shard, "complete", processed)


def _alive(pid: int, shard: int | None) -> bool:
    info = pilot._stat(pid)
    if info is None or info.state == "Z" or info.pgrp != pid:
        return False
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    mode = b"--watch" if shard is None else b"--shard"
    return (
        str(SCRIPT).encode() in args and mode in args
        and (shard is None or str(shard).encode() in args)
    )


def _stop_recorded(groups: dict[int, int]) -> dict[str, Any]:
    """Signal only verified v4 process groups and their exact runner tags."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for shard, pid in groups.items():
            if _alive(pid, shard):
                try:
                    os.killpg(pid, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    return pilot.cleanup()


def watch(audit_sha256: str) -> None:
    _bind_v4()
    _promotion(audit_sha256)
    manifest = _launch_manifest(audit_sha256)
    groups = {int(k): int(v) for k, v in manifest["driver_pids"].items()}
    if set(groups) != set(range(SHARDS)):
        raise pilot.PilotSafetyStop("v4 launch lacks all exact shard PIDs")
    deadline = time.monotonic() + 36 * 3600
    while time.monotonic() < deadline:
        states = {
            shard: _read(ROOT / f"v4_shard_{shard}.json")
            if (ROOT / f"v4_shard_{shard}.json").is_file() else {}
            for shard in groups
        }
        live = {shard: _alive(pid, shard) for shard, pid in groups.items()}
        failed = [
            shard for shard in groups
            if states[shard].get("phase") == "failed"
            or (not live[shard] and states[shard].get("phase") not in ("complete", "capped"))
        ]
        if failed:
            cleanup = _stop_recorded(groups)
            _atomic_bytes(ROOT / "v4_watchdog_cleanup.json", _json_bytes(cleanup))
            _atomic_bytes(STATUS, _json_bytes(
                pilot._status("failed", error=f"v4 shards failed: {failed}")
            ))
            return
        if not any(live.values()):
            phase = "capped" if any(
                states[shard].get("phase") == "capped" for shard in groups
            ) else "complete"
            _atomic_bytes(STATUS, _json_bytes(pilot._status(phase)))
            return
        time.sleep(15)
    cleanup = _stop_recorded(groups)
    _atomic_bytes(ROOT / "v4_watchdog_cleanup.json", _json_bytes(cleanup))
    _atomic_bytes(STATUS, _json_bytes(
        pilot._status("failed", error="v4 watchdog lifetime expired")
    ))


def launch(audit_sha256: str) -> dict[str, Any]:
    _bind_v4()
    prepared = prepare()
    _promotion(audit_sha256)
    runtime = _runtime()
    frozen = pilot.require_plan(runtime)
    if frozen["source_hashes"]["v4_driver"] != _hash(SCRIPT):
        raise pilot.PilotSafetyStop("v4 driver source hash differs")
    results_dir = ROOT / "results"
    if results_dir.exists() and list(results_dir.glob("*.json")):
        raise pilot.PilotSafetyStop("fresh v4 root already has completed episodes")
    counter = _read(ROOT / "new_real_runs.json")
    if len(counter["reservations"]) != prepared["carryover_reservations"]:
        raise pilot.PilotSafetyStop("v4 had recipe reservations before first launch")
    load, available = Runner._resource_snapshot()
    if load > pilot.LOAD_CEILING or available < pilot.MEMORY_FLOOR:
        raise pilot.PilotSafetyStop("load or available memory blocks v4 launch")
    with _file_lock(ROOT / "v4_launch.lock"):
        if LAUNCH.exists():
            raise FileExistsError("v4 pilot already launched")
        manifest: dict[str, Any] = {
            "schema_version": "thursday-overnight-pilot-v4-launch",
            "started_at_utc": pilot.utc(),
            "pilot_plan_sha256": _hash(ROOT / "plan.json"),
            "wrapper_sha256": _hash(SCRIPT),
            "promotion_audit_sha256": audit_sha256,
            "registry_sha256": frozen["source_hashes"]["registry"],
            "worker_model_sha256": {
                key: frozen["source_hashes"][key] for key in ("A_fit", "B_fit")
            },
            "shards": SHARDS, "driver_pids": {},
            "load_1m_at_launch": load,
            "available_memory_bytes_at_launch": available,
            "new_real_cap": CAP,
            "carryover_reservations": prepared["carryover_reservations"],
            "episode_reuse": False,
        }
        _atomic_bytes(LAUNCH, _json_bytes(manifest))
        spawned: dict[int, int] = {}
        try:
            for shard in range(SHARDS):
                with (ROOT / f"v4_shard_{shard}.stdout").open("wb") as out, (
                    ROOT / f"v4_shard_{shard}.stderr"
                ).open("wb") as err:
                    child = subprocess.Popen(
                        [sys.executable, str(SCRIPT), "--shard", str(shard),
                         "--promotion-audit-sha256", audit_sha256],
                        cwd=PROJECT, stdin=subprocess.DEVNULL,
                        stdout=out, stderr=err,
                        start_new_session=True, close_fds=True,
                    )
                spawned[shard] = child.pid
            manifest["driver_pids"] = {
                str(key): pid for key, pid in spawned.items()
            }
            _atomic_bytes(LAUNCH, _json_bytes(manifest))
            with (ROOT / "v4_watchdog.stdout").open("wb") as out, (
                ROOT / "v4_watchdog.stderr"
            ).open("wb") as err:
                watcher = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "--watch",
                     "--promotion-audit-sha256", audit_sha256],
                    cwd=PROJECT, stdin=subprocess.DEVNULL,
                    stdout=out, stderr=err,
                    start_new_session=True, close_fds=True,
                )
            manifest["watchdog_pid"] = watcher.pid
            _atomic_bytes(LAUNCH, _json_bytes(manifest))
            pilot._status("running")
            return manifest
        except Exception:
            cleanup = _stop_recorded(spawned)
            _atomic_bytes(ROOT / "v4_launch_cleanup.json", _json_bytes(cleanup))
            raise


def status() -> dict[str, Any]:
    _bind_v4()
    state = _read(ROOT / "status.json") if (ROOT / "status.json").is_file() else {
        "phase": "not_launched"
    }
    manifest = _read(LAUNCH) if LAUNCH.is_file() else {}
    groups = {int(k): int(v) for k, v in manifest.get("driver_pids", {}).items()}
    return {
        "pilot": state,
        "driver_pids": manifest.get("driver_pids", {}),
        "watchdog_pid": manifest.get("watchdog_pid"),
        "shards": {
            str(shard): {
                **(_read(ROOT / f"v4_shard_{shard}.json")
                   if (ROOT / f"v4_shard_{shard}.json").is_file() else {}),
                "alive": _alive(pid, shard),
            }
            for shard, pid in groups.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--watch", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--shard", type=int)
    parser.add_argument("--promotion-audit-sha256")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), sort_keys=True))
    elif args.launch:
        if not args.promotion_audit_sha256:
            raise pilot.PilotSafetyStop("launch needs exact promotion audit hash")
        print(json.dumps(launch(args.promotion_audit_sha256), sort_keys=True))
    elif args.watch:
        if not args.promotion_audit_sha256:
            raise pilot.PilotSafetyStop("watchdog needs exact promotion audit hash")
        watch(args.promotion_audit_sha256)
    elif args.status:
        print(json.dumps(status(), sort_keys=True))
    else:
        if not args.promotion_audit_sha256:
            raise pilot.PilotSafetyStop("shard needs exact promotion audit hash")
        try:
            run_shard(args.shard, args.promotion_audit_sha256)
        except Exception as exc:
            _bind_v4()
            _write_shard(args.shard, "failed", 0, f"{type(exc).__name__}: {exc}")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

