"""Migrate the frozen test cache through an audited meter-version equivalence.

No Vivado, recipe, validator, or test outcome table is read here. Original run
records remain untouched. New current-version Evidence aliases and run records
retain the original run ID, report path, cost, and approval checksum. Cache
pointers are replaced atomically after each alias is complete, with backups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.contract.models import Claims, Evidence, RunOutcome, RunStatus
from orch.engine.cache import RunCache
from orch.engine.jobtag import _stat
from orch.engine.meter_equivalence import (
    CURRENT_METER, MeterEquivalence, approved_test_equivalence,
    promote_evidence,
)
from orch.engine.runner import Runner
from orch.engine.store import StateStore, _atomic_bytes, _file_lock, _json_bytes, _sha256_file

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
TEST_ROOT = PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04"
ROOT = PROJECT / "data/scratch_thursday_night_2026-11-05/test_cache_promotion"
SOURCE_FILES = (
    "scripts/promote_test_cache_meter.py", "orch/engine/meter_equivalence.py",
    "orch/engine/apply.py", "orch/runtime.py", "orch/engine/cache.py",
    "orch/engine/store.py", "orch/contract/models.py",
)
TABLES = ("summary.json", "by_design/index.csv")
LOAD_CEILING = 180.0
MEMORY_FLOOR = 200_000_000_000


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def source_pins() -> dict[str, str]:
    return {rel: _sha256_file(PROJECT / rel) for rel in SOURCE_FILES}


def alias_id(key: str, approval: MeterEquivalence) -> str:
    digest = hashlib.sha256((key + approval.audit_sha256).encode("ascii")).hexdigest()
    return f"metereq-{digest[:32]}"


def version_folder(version: str) -> str:
    return hashlib.sha256(version.encode("utf-8")).hexdigest()


def _campaign_stopped(root: Path) -> None:
    """Require the frozen test driver to be absent before cache pointer edits."""

    status = read(root / "status.json")
    launch = read(root / "launch.json")
    if status.get("phase") != "complete":
        raise ValueError("frozen test campaign is not complete")
    pid = launch.get("driver_pid")
    if type(pid) is int:
        info = _stat(pid)
        if info is not None and info.state != "Z":
            try:
                args = Path(f"/proc/{pid}/cmdline").read_bytes().split(bytes([0]))
            except OSError:
                args = []
            if b"--run" in args and any(b"overnight_test_baseline" in arg for arg in args):
                raise ValueError("frozen test campaign driver is still running")


def prepare(root: Path, approval: MeterEquivalence, *,
            require_complete: bool = False) -> dict[str, Any]:
    """Validate every terminal pointer before any mutation; disclose no metrics."""

    root = root.resolve()
    if require_complete:
        if root != TEST_ROOT.resolve():
            raise ValueError("real cache migration is restricted to frozen test root")
        _campaign_stopped(root)
    store = StateStore(root)
    cache = RunCache(root)
    rows: list[dict[str, str]] = []
    for pointer in sorted(cache.cache_dir.glob("*.json")):
        key = pointer.stem
        entry = read(pointer)
        if entry.get("key") != key or entry.get("retryable") is not False:
            raise ValueError("unexpected retryable or malformed test cache pointer")
        run_id = cache.get(key)
        if run_id != entry.get("run_id"):
            raise ValueError("test cache pointer changed during preflight")
        outcome = store.load_outcome(run_id)
        run, state, evidence = outcome.run, outcome.state, outcome.evidence
        if (
            run.status != RunStatus.OK or state is None or evidence is None
            or run.fake or evidence.fake
            or run.input_sha256_before != run.input_sha256_after
            or state.parent_id != run.input_sha256_before
            or RunCache.key(run.input_sha256_before, run.recipe_id,
                            run.recipe_version) != key
            or evidence.meter_version != approval.old_version
        ):
            raise ValueError("test cache outcome identity or version differs")
        promoted = promote_evidence(evidence, approval)
        rows.append({
            "key": key, "old_run_id": run_id,
            "old_pointer_sha256": _sha256_file(pointer),
            "old_complete_sha256": _sha256_file(
                root / "runs" / run_id / "complete.json"
            ),
            "old_evidence_id": evidence.id,
            "new_evidence_id": promoted.id,
            "alias_run_id": alias_id(key, approval),
            "state_id": state.id,
        })
    starts: list[dict[str, str]] = []
    for old_path in sorted((root / "measurements").glob("*/*/evidence.json")):
        old = Evidence.model_validate_json(old_path.read_bytes())
        if old.meter_version != approval.old_version or old.fake:
            raise ValueError("test start Evidence version or origin differs")
        if old_path.parents[1].name != old.state_id:
            raise ValueError("test start Evidence names a different state")
        store.state_path(old.state_id)
        promoted = promote_evidence(old, approval)
        starts.append({
            "state_id": old.state_id,
            "old_path": str(old_path),
            "old_sha256": _sha256_file(old_path),
            "new_path": str(
                root / "measurements" / old.state_id
                / version_folder(approval.new_version) / "evidence.json"
            ),
            "new_evidence_id": promoted.id,
        })
    if require_complete and (len(rows) != 437 or len(starts) != 7):
        raise ValueError("frozen test cache or start count changed")
    return {
        "schema_version": "test-meter-cache-promotion-plan-v1",
        "created_at_utc": utc(),
        "root": str(root),
        "approval": {
            "old_version": approval.old_version,
            "new_version": approval.new_version,
            "audit_sha256": approval.audit_sha256,
        },
        "source_pins": source_pins(),
        "table_hashes": {
            rel: _sha256_file(root / rel)
            for rel in TABLES if (root / rel).is_file()
        },
        "pointers": rows, "starts": starts,
    }


def check_plan(plan: dict[str, Any], root: Path,
               approval: MeterEquivalence) -> None:
    if (
        plan.get("schema_version") != "test-meter-cache-promotion-plan-v1"
        or plan.get("root") != str(root.resolve())
        or plan.get("approval") != {
            "old_version": approval.old_version,
            "new_version": approval.new_version,
            "audit_sha256": approval.audit_sha256,
        }
        or plan.get("source_pins") != source_pins()
    ):
        raise ValueError("test cache promotion source or approval changed")


def _sidecar(approval: MeterEquivalence, *,
             old_run_id: str | None, old_evidence_id: str,
             promoted_evidence_id: str) -> dict[str, Any]:
    return {
        "schema_version": "meter-equivalent-cache-alias-v1",
        "old_version": approval.old_version,
        "new_version": approval.new_version,
        "audit_sha256": approval.audit_sha256,
        "old_run_id": old_run_id,
        "old_evidence_id": old_evidence_id,
        "promoted_evidence_id": promoted_evidence_id,
    }


def promote_start(root: Path, row: dict[str, str],
                  approval: MeterEquivalence) -> None:
    old_path = Path(row["old_path"])
    target = Path(row["new_path"])
    if _sha256_file(old_path) != row["old_sha256"]:
        raise ValueError("saved start Evidence changed")
    old = Evidence.model_validate_json(old_path.read_bytes())
    promoted = promote_evidence(old, approval)
    if promoted.id != row["new_evidence_id"]:
        raise ValueError("start promotion ID changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(target.parent / "evidence.lock"):
        if target.is_file():
            current = Evidence.model_validate_json(target.read_bytes())
            if current != promoted:
                raise ValueError("existing current start Evidence differs")
            return
        _atomic_bytes(
            target.parent / "meter_equivalence.json",
            _json_bytes(_sidecar(
                approval, old_run_id=None, old_evidence_id=old.id,
                promoted_evidence_id=promoted.id,
            )),
        )
        _atomic_bytes(target, _json_bytes(promoted.model_dump(mode="json")))


def _alias_outcome(original: RunOutcome, approval: MeterEquivalence,
                   new_run_id: str) -> RunOutcome:
    if original.evidence is None:
        raise ValueError("old cached run has no Evidence")
    promoted = promote_evidence(original.evidence, approval)
    run = original.run.model_copy(update={
        "id": new_run_id, "cache_hit": True,
        "original_run_id": original.run.id, "attempts": (),
        "process_group_kills": 0, "tag_sweep_kills": 0,
    })
    claims = original.claims
    if claims is not None:
        claims = claims.model_copy(update={
            "id": f"{new_run_id}:claims", "run_id": new_run_id,
        })
    return RunOutcome(
        run=run, state=original.state, evidence=promoted, claims=claims
    )


def promote_pointer(root: Path, row: dict[str, str],
                    approval: MeterEquivalence,
                    backup_root: Path) -> None:
    store = StateStore(root)
    cache = RunCache(root)
    key = row["key"]
    pointer = cache.cache_dir / f"{key}.json"
    with cache.locked(key):
        current_run_id = cache.get(key)
        if current_run_id == row["alias_run_id"]:
            alias = store.load_outcome(current_run_id)
            if (
                alias.evidence is None
                or alias.evidence.id != row["new_evidence_id"]
                or alias.run.original_run_id != row["old_run_id"]
                or alias.evidence.meter_version != approval.new_version
            ):
                raise ValueError("existing promoted cache alias differs")
            return
        if (current_run_id != row["old_run_id"]
                or _sha256_file(pointer) != row["old_pointer_sha256"]):
            raise ValueError("test cache pointer changed during promotion")
        original = store.load_outcome(row["old_run_id"])
        if (
            _sha256_file(root / "runs" / row["old_run_id"] / "complete.json")
            != row["old_complete_sha256"]
            or original.evidence is None
            or original.evidence.id != row["old_evidence_id"]
            or original.state is None
            or original.state.id != row["state_id"]
        ):
            raise ValueError("old cached outcome changed")
        alias = _alias_outcome(original, approval, row["alias_run_id"])
        if alias.evidence is None or alias.evidence.id != row["new_evidence_id"]:
            raise ValueError("promoted cached Evidence ID changed")
        alias_dir = root / "runs" / row["alias_run_id"]
        alias_dir.mkdir(parents=True, exist_ok=True)
        sidecar = _sidecar(
            approval, old_run_id=original.run.id,
            old_evidence_id=original.evidence.id,
            promoted_evidence_id=alias.evidence.id,
        )
        sidecar_path = alias_dir / "meter_equivalence.json"
        if sidecar_path.is_file() and read(sidecar_path) != sidecar:
            raise ValueError("existing alias approval sidecar differs")
        _atomic_bytes(sidecar_path, _json_bytes(sidecar))
        store.save_outcome(alias)
        backup = backup_root / f"{key}.json"
        original_pointer = pointer.read_bytes()
        if backup.is_file():
            if backup.read_bytes() != original_pointer:
                raise ValueError("cache pointer backup differs")
        else:
            _atomic_bytes(backup, original_pointer)
        cache.put(key, alias.run.id, retryable=False)


def _status(phase: str, *, migrated: int, starts: int,
            error: str | None = None) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(ROOT / "status.json", _json_bytes({
        "schema_version": "test-meter-cache-promotion-status-v1",
        "phase": phase, "planned_pointers": 437, "migrated_pointers": migrated,
        "planned_starts": 7, "promoted_starts": starts,
        "error": error, "updated_at_utc": utc(),
        "tool_jobs_launched": 0,
    }))


def migrate(root: Path, plan: dict[str, Any],
            approval: MeterEquivalence,
            output: Path) -> dict[str, Any]:
    """Idempotent migration; each pointer is switched only after alias commit."""

    check_plan(plan, root, approval)
    backup_root = output / "cache_backups"
    backup_root.mkdir(parents=True, exist_ok=True)
    start_count = 0
    for row in plan["starts"]:
        check_plan(plan, root, approval)
        promote_start(root, row, approval)
        start_count += 1
    migrated = 0
    for row in plan["pointers"]:
        check_plan(plan, root, approval)
        promote_pointer(root, row, approval, backup_root)
        migrated += 1
        if migrated % 10 == 0 or migrated == len(plan["pointers"]):
            _status("running", migrated=migrated, starts=start_count)
    if {
        rel: _sha256_file(root / rel) for rel in plan["table_hashes"]
    } != plan["table_hashes"]:
        raise ValueError("frozen test outcome tables changed")
    cache = RunCache(root)
    store = StateStore(root)
    for row in plan["pointers"]:
        if cache.get(row["key"]) != row["alias_run_id"]:
            raise ValueError("promoted cache pointer missing")
        alias = store.load_outcome(row["alias_run_id"])
        original = store.load_outcome(row["old_run_id"])
        if alias.evidence is None or alias.evidence.meter_version != approval.new_version:
            raise ValueError("promoted Evidence is not current")
        for field in (
            "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
            "load_1m_start", "load_1m_end", "busy_slots_start",
        ):
            if getattr(alias.run, field) != getattr(original.run, field):
                raise ValueError("promoted cache hit changed original cost")
    result = {
        "schema_version": "test-meter-cache-promotion-audit-v1",
        "checked_at_utc": utc(),
        "plan_sha256": _sha256_file(output / "plan.json") if (output / "plan.json").is_file() else None,
        "approval_audit_sha256": approval.audit_sha256,
        "promoted_pointers": migrated,
        "promoted_starts": start_count,
        "unchanged_table_sha256": plan["table_hashes"],
        "cost_fields_equal_to_original": True,
        "tool_jobs_launched": 0,
    }
    _atomic_bytes(output / "audit.json", _json_bytes(result))
    return result


def _alive(pid: int) -> bool:
    try:
        info = _stat(pid)
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(bytes([0]))
        return bool(info and info.state != "Z" and str(SCRIPT).encode() in args
                    and b"--run" in args)
    except OSError:
        return False


def run() -> None:
    with _file_lock(ROOT / "driver.lock"):
        approval = approved_test_equivalence()
        plan = read(ROOT / "plan.json")
        check_plan(plan, TEST_ROOT, approval)
        _status("running", migrated=0, starts=0)
        try:
            migrate(TEST_ROOT, plan, approval, ROOT)
        except Exception as exc:
            _status("stopped", migrated=read(ROOT / "status.json").get("migrated_pointers", 0),
                    starts=read(ROOT / "status.json").get("promoted_starts", 0),
                    error=type(exc).__name__)
            raise
        _status("complete", migrated=len(plan["pointers"]),
                starts=len(plan["starts"]))


def watch() -> None:
    manifest = read(ROOT / "launch.json")
    pid = manifest["driver_pid"]
    if not _alive(pid):
        raise RuntimeError("promotion driver absent at watchdog start")
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        if not _alive(pid):
            state = read(ROOT / "status.json") if (ROOT / "status.json").is_file() else {}
            _atomic_bytes(ROOT / "watchdog_status.json", _json_bytes({
                "phase": state.get("phase"), "driver_exited_at_utc": utc(),
                "tool_jobs_launched": 0,
            }))
            return
        time.sleep(15)
    raise RuntimeError("promotion watchdog expired while driver remained alive")


def launch() -> dict[str, Any]:
    approval = approved_test_equivalence()
    load, available = Runner._resource_snapshot()
    if load > LOAD_CEILING or available < MEMORY_FLOOR:
        raise RuntimeError("load or memory gate blocks offline cache promotion")
    plan = prepare(TEST_ROOT, approval, require_complete=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with _file_lock(ROOT / "launch.lock"):
        if (ROOT / "launch.json").exists():
            raise FileExistsError("test cache promotion already launched")
        _atomic_bytes(ROOT / "plan.json", _json_bytes(plan))
        with (ROOT / "driver.stdout").open("wb") as out, (ROOT / "driver.stderr").open("wb") as err:
            driver = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--run"], cwd=PROJECT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        result: dict[str, Any] = {
            "schema_version": "test-meter-cache-promotion-launch-v1",
            "driver_pid": driver.pid, "started_at_utc": utc(),
            "plan_sha256": _sha256_file(ROOT / "plan.json"),
            "load_1m": load, "available_memory_bytes": available,
        }
        _atomic_bytes(ROOT / "launch.json", _json_bytes(result))
        try:
            with (ROOT / "watchdog.stdout").open("wb") as out, (ROOT / "watchdog.stderr").open("wb") as err:
                watcher = subprocess.Popen(
                    [sys.executable, str(SCRIPT), "--watch"], cwd=PROJECT,
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    start_new_session=True, close_fds=True,
                )
            result["watchdog_pid"] = watcher.pid
            _atomic_bytes(ROOT / "launch.json", _json_bytes(result))
        except Exception:
            raise
        return result


def status() -> dict[str, Any]:
    state = read(ROOT / "status.json") if (ROOT / "status.json").is_file() else {"phase": "not_started"}
    launch_record = read(ROOT / "launch.json") if (ROOT / "launch.json").is_file() else {}
    state.update({
        "driver_alive": _alive(launch_record["driver_pid"]) if launch_record else False,
        "driver_stderr_bytes": (ROOT / "driver.stderr").stat().st_size if launch_record else 0,
        "audit_exists": (ROOT / "audit.json").is_file(),
    })
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "launch", "run", "watch", "status"):
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        approval = approved_test_equivalence()
        plan = prepare(TEST_ROOT, approval, require_complete=True)
        print(json.dumps({
            "terminal_pointers": len(plan["pointers"]),
            "start_measurements": len(plan["starts"]),
            "tool_jobs_launched": 0,
            "table_hashes_recorded": len(plan["table_hashes"]),
        }, sort_keys=True))
    elif args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.run:
        run()
    elif args.watch:
        watch()
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
