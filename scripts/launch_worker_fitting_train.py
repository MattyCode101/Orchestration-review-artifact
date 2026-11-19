"""Launch and monitor the authors' five-design worker-fitting data campaign.

Dry-run is the default. --launch seeds only exact-version selected first steps
and starts one detached D1-D4 driver. --status reads short local status files.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from orch.baselines.cache_seed import seed_first_steps
from orch.baselines.verification import VerificationLedger, audit_committed
from orch.baselines.verification_guard import (
    FailClosedVerifiedStepEngine, audit_all_saved,
)
from orch.baselines.worker_fitting import (
    AREA_IDS, CAMPAIGN_ID, DEPTH_STUDY_DESIGN, TIMING_IDS,
    TRAIN_DESIGNS, WorkerFittingCampaign,
)
from orch.config import load_brief, load_noise, load_recipe_sets
from orch.engine.journal import Journal
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes
from orch.recipes.registry import load_registry, sha256_file
from orch.runtime import make_real_runtime

PROJECT = Path(__file__).resolve().parents[1]
B2 = (Path(os.environ.get("ORCH_BENCHMARK_DIR", "benchmarks")) / "fpl26_B2_default-reimpl_2026-10-24")
DATA_ROOT = PROJECT / "data" / "sweeps" / "worker_fitting_train"
SOURCE_ROOT = PROJECT / "data" / "real_sweep_b2_venv_fixed"
MANIFEST = DATA_ROOT / "launch_manifest.json"
RESUME_MANIFEST = DATA_ROOT / "resume_manifest.json"
SEED_MANIFEST = DATA_ROOT / "cache_seed_manifest.json"
PROGRESS_NOTE = PROJECT / "notes" / "worker_sweep_progress.md"
SLOTS = 48
EXPECTED_SHA = {
    "vexriscv_re-place": "8f20f4036a9be0800a0c1fb3072e7a1155ea2f32f630a66a0f96917d2c4e3efc",
    "rosetta_spam-filter": "64d87ad3e212fd066e2c3034c6cc56738fe89751602cf3ef4724ab3021b68105",
    "rosetta_3d-rendering": "feed3365596b7d0b83a77156237f9d5ffc39c1642c245c87077d091ab1680d90",
    "logicnets_jscl": "d7671ae14d9a3800dc416a2eb29494a2378e4119efdcac4ecde9eb28c58e8fa4",
    "finn_radioml": "9cb9f3799ee08b83ce452f8525a52ab4a9c513ff6349abf41a36428bcd1ca233",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plan() -> dict[str, object]:
    """Reject any drift from the exact signed ten-recipe, five-design plan."""

    selected = load_recipe_sets(PROJECT / "configs" / "recipe_sets.yaml")
    selected.require_signed_off()
    if (
        tuple(selected.worker_sets.get("timing", ())) != TIMING_IDS
        or tuple(selected.worker_sets.get("area", ())) != AREA_IDS
        or tuple(selected.worker_sets.get("power", ())) != ()
        or set(selected.worker_sets) != {"timing", "area", "power"}
        or tuple(EXPECTED_SHA) != TRAIN_DESIGNS
    ):
        raise ValueError("selected recipe sets or train allowlist differ from the authors' directive")
    registry = load_registry(PROJECT)
    recipe_sets = {
        worker: tuple(registry.get(recipe_id) for recipe_id in ids)
        for worker, ids in (("timing", TIMING_IDS), ("area", AREA_IDS))
    }
    if any(recipe.worker != worker for worker, values in recipe_sets.items()
           for recipe in values):
        raise ValueError("registry worker label differs from selected set")
    inputs = []
    for design in TRAIN_DESIGNS:
        path = B2 / f"{design}_B2.dcp"
        if not path.is_file() or sha256_file(path) != EXPECTED_SHA[design]:
            raise ValueError(f"training B2 checkpoint missing or changed: {path}")
        inputs.append({
            "design": design, "path": str(path), "sha256": EXPECTED_SHA[design],
            "bytes": path.stat().st_size,
        })
    return {
        "schema_version": "worker-fitting-launch-v1",
        "train_inputs": inputs,
        "recipe_sets": {
            worker: [{"id": recipe.id, "version": recipe.version} for recipe in values]
            for worker, values in recipe_sets.items()
        },
        "depth_study_design": DEPTH_STUDY_DESIGN,
        "phase_sequence": ["D1", "D2", "D3", "D4"],
        "D1_planned_edges": len(TRAIN_DESIGNS) * 10,
        "D2_planned_edges": len(TRAIN_DESIGNS) * 100,
        "D3_max_new_edges": len(TRAIN_DESIGNS) * 2 * 2 * 2 * 5,
        "D4_total_within_worker_edges": 2 * sum(5 ** depth for depth in range(1, 5)),
        "max_shared_runner_slots": SLOTS,
        "source_cache_root": str(SOURCE_ROOT),
        "data_root": str(DATA_ROOT),
        "runner_sha256": sha256_file(PROJECT / "orch/engine/runner.py"),
        "jobtag_sha256": sha256_file(PROJECT / "orch/engine/jobtag.py"),
        "driver_sha256": sha256_file(PROJECT / "orch/baselines/worker_fitting.py"),
    }


def _resources() -> dict[str, object]:
    """Read only public load, memory and disk facts before 48-way launch."""

    logical = os.cpu_count() or 0
    load_1m, load_5m, load_15m = os.getloadavg()
    available_kib = None
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
            break
    if available_kib is None:
        raise RuntimeError("cannot read available RAM")
    free_disk = shutil.disk_usage(PROJECT).free
    result = {
        "logical_cpus": logical,
        "load_1m": load_1m,
        "load_5m": load_5m,
        "load_15m": load_15m,
        "available_ram_bytes": available_kib * 1024,
        "free_disk_bytes": free_disk,
        "checked_at_utc": _utc(),
    }
    if logical < SLOTS or load_1m > logical * 0.75:
        raise RuntimeError("server load does not permit 48-slot launch")
    if available_kib * 1024 < 256 * 2**30 or free_disk < 1 * 2**40:
        raise RuntimeError("server memory or project disk headroom is too low")
    return result


def _old_driver_alive(job: dict[str, object]) -> bool:
    pid = job.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        args = [
            item.decode("utf-8", errors="replace")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
            if item
        ]
        return (
            "sweep" in args
            and args[args.index("--data-root") + 1] == str(SOURCE_ROOT)
            and args[args.index("--id") + 1] == job["sweep_id"]
        )
    except (OSError, ValueError, IndexError):
        return False


def _require_old_stopped() -> None:
    audit_path = SOURCE_ROOT / "train_only_stop_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("driver_survivors") != [] or audit.get("tag_survivors") != {}:
        raise RuntimeError("superseded eight-design sweep has recorded survivors")
    old = json.loads((SOURCE_ROOT / "launch_manifest.json").read_text(encoding="utf-8"))
    if any(_old_driver_alive(job) for job in old["jobs"]):
        raise RuntimeError("superseded eight-design driver is still running")


def _busy_slots() -> int:
    busy = 0
    for index in range(SLOTS):
        path = PROJECT / "data" / "slots" / f"slot{index}"
        if not path.exists():
            continue
        with path.open("r") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy += 1
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
    return busy


def _starts(runtime, plan: dict[str, object]):
    return {
        row["design"]: runtime.store.import_state(
            Path(row["path"]), design=row["design"], fake=False
        )
        for row in plan["train_inputs"]
    }


def _phase_callback(event: dict[str, object]) -> None:
    """Publish a phase result locally before the driver enters the next phase."""

    phase = event["phase"]
    summary = event["summary"]
    report = DATA_ROOT / "phase_reports" / f"{phase}.json"
    _atomic_bytes(report, _json_bytes(event))
    paragraph = (
        f"\n## {phase} complete ? {event['at_utc']}\n\n"
        f"- Verified journal: {summary['planned_sequences']} planned sequences, "
        f"{summary['attempted_edges']} attempted, {summary['physical_valid_edges']} "
        f"physically valid, {summary['cache_hits']} cache hits, "
        f"{summary['new_backend_edges']} new backend edges.\n"
        f"- Verified status counts: {json.dumps(summary['statuses'], sort_keys=True)}.\n"
        f"- Next phase starts automatically after this record.\n"
    )
    if "verification_coverage" in event:
        verified = event["verification_coverage"]
        paragraph += (
            f"- AMD validator PASS coverage: {verified['distinct_verified_dcps']} "
            f"distinct saved DCPs across {verified['saved_output_attempts']} output attempts.\n"
        )
    with _file_lock(PROGRESS_NOTE.with_suffix(".lock")):
        with PROGRESS_NOTE.open("a", encoding="utf-8") as handle:
            handle.write(paragraph)
            handle.flush()
            os.fsync(handle.fileno())
    print(json.dumps({"phase_complete": phase, "summary": summary}, sort_keys=True), flush=True)


def launch() -> dict[str, object]:
    """Validate resources, seed authorized cache entries, then detach once."""

    plan = _plan()
    resources = _resources()
    _require_old_stopped()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    with _file_lock(DATA_ROOT / "launch.lock"):
        if MANIFEST.exists():
            raise FileExistsError("launch manifest exists; use --status and resume explicitly")
        runtime = make_real_runtime(DATA_ROOT, cache_enabled=True, max_slots=SLOTS)
        starts = _starts(runtime, plan)
        recipes = tuple(
            runtime.registry.get(recipe_id) for recipe_id in TIMING_IDS + AREA_IDS
        )
        seeded = seed_first_steps(
            engine=runtime.engine, source_root=SOURCE_ROOT,
            starts=starts, recipes=recipes,
        )
        _atomic_bytes(SEED_MANIFEST, _json_bytes(seeded))
        initial = (
            "# Worker-fitting training sweep progress\n\n"
            f"- Verified launch preparation at {_utc()}: exactly five training B2 "
            f"checkpoints, ten selected recipes, {seeded['seeded_run_count']} "
            f"completed first-step cache entries imported.\n"
            "- D1, D2, D3 and the VexRiscv-only D4 run in order. "
            "Each phase appends its journal summary before the next starts.\n"
        )
        _atomic_bytes(PROGRESS_NOTE, initial.encode("utf-8"))
        manifest = {
            **plan, "phase": "launching", "started_at_utc": _utc(),
            "resource_check": resources, "busy_slots_before_launch": _busy_slots(),
            "seeded_run_count": seeded["seeded_run_count"],
            "seeded_start_designs": seeded["seeded_start_designs"],
            "driver_pid": None,
            "stdout": str(DATA_ROOT / "driver.stdout"),
            "stderr": str(DATA_ROOT / "driver.stderr"),
        }
        _atomic_bytes(MANIFEST, _json_bytes(manifest))
        command = [
            str(PROJECT / ".venv" / "bin" / "python"),
            str(Path(__file__).resolve()), "--run",
        ]
        with Path(manifest["stdout"]).open("wb") as stdout, Path(manifest["stderr"]).open("wb") as stderr:
            process = subprocess.Popen(
                command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr,
                start_new_session=True, close_fds=True,
            )
        manifest["driver_pid"] = process.pid
        manifest["phase"] = "launched"
        manifest["finished_launch_at_utc"] = _utc()
        _atomic_bytes(MANIFEST, _json_bytes(manifest))
        return manifest


def run() -> dict[str, object]:
    """Detached driver: enforce the frozen plan and execute all four phases."""

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    plan = _plan()
    if any(manifest.get(key) != value for key, value in plan.items()):
        raise ValueError("frozen launch plan differs from current code, recipes or inputs")
    runtime = make_real_runtime(DATA_ROOT, cache_enabled=True, max_slots=SLOTS)
    references = {
        row["design"]: row["sha256"] for row in plan["train_inputs"]
    }
    ledger = VerificationLedger(
        store=runtime.store, runner=runtime.runner,
        validator=runtime.validator, references=references,
    )
    prior_events = Journal(DATA_ROOT).read(CAMPAIGN_ID)
    preflight = audit_committed(prior_events, ledger)
    all_attempts_preflight = audit_all_saved(prior_events, ledger)
    print(json.dumps({
        "equivalence_preflight": preflight,
        "all_attempts_preflight": all_attempts_preflight,
        "validator_fingerprint": ledger.fingerprint,
    }, sort_keys=True), flush=True)
    verified_engine = FailClosedVerifiedStepEngine(runtime.engine, ledger)
    starts = _starts(runtime, plan)
    config = load_noise(PROJECT / "configs/noise.yaml")
    measured = {
        design: runtime.engine.measure_start(state) for design, state in starts.items()
    }
    noise = {
        design: config.for_start(design, measured[design])
        for design in starts
    }
    recipe_sets = {
        "timing": tuple(runtime.registry.get(recipe_id) for recipe_id in TIMING_IDS),
        "area": tuple(runtime.registry.get(recipe_id) for recipe_id in AREA_IDS),
    }
    briefs = {
        name: load_brief(PROJECT / "configs" / "briefs" / f"{name}.yaml")
        for name in ("timing", "area")
    }
    def checked_phase_callback(event):
        coverage = audit_all_saved(Journal(DATA_ROOT).read(CAMPAIGN_ID), ledger)
        _phase_callback({**event, "verification_coverage": coverage})

    campaign = WorkerFittingCampaign(
        engine=verified_engine, starts=starts, recipe_sets=recipe_sets,
        briefs=briefs, noise=noise, max_workers=SLOTS,
        campaign_id=CAMPAIGN_ID, depth_study_design=DEPTH_STUDY_DESIGN,
        phase_callback=checked_phase_callback,
    )
    summaries = campaign.run()
    verification_coverage = audit_all_saved(Journal(DATA_ROOT).read(CAMPAIGN_ID), ledger)
    final = {
        "phase": "complete", "completed_at_utc": _utc(),
        "summaries": summaries, "committed_unique_edges": len(campaign.index),
        "equivalence_preflight": preflight,
        "all_attempts_preflight": all_attempts_preflight,
        "verification_coverage": verification_coverage,
        "validator_fingerprint": ledger.fingerprint, "fake": False,
        "verification_guard_sha256": sha256_file(PROJECT / "orch/baselines/verification_guard.py"),
    }
    _atomic_bytes(DATA_ROOT / "final_summary.json", _json_bytes(final))
    print(json.dumps(final, sort_keys=True), flush=True)
    return final




def resume() -> dict[str, object]:
    """Resume the stopped journal only after complete functional backfill."""
    plan = _plan()
    if not MANIFEST.is_file():
        raise FileNotFoundError("original launch manifest is missing")
    original = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if any(original.get(key) != value for key, value in plan.items()):
        raise ValueError("frozen original plan differs before verified resume")
    stopped = DATA_ROOT / "watchdog_cleanup_audit.json"
    audit = json.loads(stopped.read_text(encoding="utf-8"))
    if audit.get("survivors") != {}:
        raise RuntimeError("stopped campaign has tagged process survivors")
    if status()["driver_alive"]:
        raise RuntimeError("old recipe driver remains active")
    if _busy_slots():
        raise RuntimeError("shared runner slots remain active; wait for validator backfill")
    verification = DATA_ROOT / "verification"
    for name in ("pilot", "backfill"):
        report = json.loads(
            (verification / f"{name}_summary.json").read_text(encoding="utf-8")
        )
        if (
            report.get("phase") != "complete"
            or report.get("statuses", {}).get("pass") != report.get("total_distinct_dcps")
        ):
            raise RuntimeError(f"{name} validator coverage is incomplete")
    runtime = make_real_runtime(DATA_ROOT, cache_enabled=True, max_slots=SLOTS)
    ledger = VerificationLedger(
        store=runtime.store, runner=runtime.runner,
        validator=runtime.validator,
        references={row["design"]: row["sha256"] for row in plan["train_inputs"]},
    )
    prior_events = Journal(DATA_ROOT).read(CAMPAIGN_ID)
    coverage = audit_committed(prior_events, ledger)
    all_attempts_coverage = audit_all_saved(prior_events, ledger)
    resources = _resources()
    with _file_lock(DATA_ROOT / "launch.lock"):
        if RESUME_MANIFEST.exists():
            raise FileExistsError("verified resume manifest already exists")
        stdout_path = DATA_ROOT / "driver_verified.stdout"
        stderr_path = DATA_ROOT / "driver_verified.stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [str(PROJECT / ".venv/bin/python"), str(Path(__file__).resolve()), "--run"],
                cwd=PROJECT, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr,
                start_new_session=True, close_fds=True,
            )
        watcher = None
        try:
            record = {
                "schema_version": "worker-fitting-verified-resume-v1",
                "phase": "resumed", "driver_pid": process.pid,
                "started_at_utc": _utc(), "stdout": str(stdout_path),
                "stderr": str(stderr_path), "coverage": coverage,
                "all_attempts_coverage": all_attempts_coverage,
                "validator_fingerprint": ledger.fingerprint,
                "verification_plan_sha256": sha256_file(verification / "plan.json"),
                "verification_guard_sha256": sha256_file(PROJECT / "orch/baselines/verification_guard.py"),
                "dcp_indexer_sha256": sha256_file(PROJECT / "scripts/index_worker_fitting_dcps.py"),
                "launcher_sha256": sha256_file(Path(__file__)),
                "original_launch_sha256": sha256_file(MANIFEST),
                "paused_cleanup_sha256": sha256_file(stopped),
                "resource_check": resources,
            }
            _atomic_bytes(RESUME_MANIFEST, _json_bytes(record))
            watchdog_script = PROJECT / "scripts/watch_worker_fitting_train.py"
            watchdog_stdout = DATA_ROOT / "watchdog_verified.stdout"
            watchdog_stderr = DATA_ROOT / "watchdog_verified.stderr"
            with watchdog_stdout.open("wb") as stdout, watchdog_stderr.open("wb") as stderr:
                watcher = subprocess.Popen(
                    [str(PROJECT / ".venv/bin/python"), str(watchdog_script), "--watch"],
                    cwd=PROJECT, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr,
                    start_new_session=True, close_fds=True,
                )
            indexer_script = PROJECT / "scripts/index_worker_fitting_dcps.py"
            indexer_stdout = DATA_ROOT / "indexer_verified.stdout"
            indexer_stderr = DATA_ROOT / "indexer_verified.stderr"
            with indexer_stdout.open("wb") as stdout, indexer_stderr.open("wb") as stderr:
                indexer = subprocess.Popen(
                    [str(PROJECT / ".venv/bin/python"), str(indexer_script),
                     "--watch", "--pid", str(process.pid)],
                    cwd=PROJECT, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr,
                    start_new_session=True, close_fds=True,
                )
            record["watchdog_pid"] = watcher.pid
            record["indexer_pid"] = indexer.pid
            _atomic_bytes(RESUME_MANIFEST, _json_bytes(record))
            return record
        except Exception as exc:
            # A detached driver can outlive a failed monitor spawn. Signal only
            # its recorded private group, then let the scoped watchdog sweep
            # CONTRACT_RUN_ID descendants (or do that sweep here if it died).
            failure = {
                "schema_version": "worker-fitting-verified-resume-v1",
                "phase": "launch_failed", "driver_pid": process.pid,
                "error": f"{type(exc).__name__}: {exc}",
            }
            try:
                if RESUME_MANIFEST.is_file():
                    failure = {**json.loads(RESUME_MANIFEST.read_text(encoding="utf-8")),
                               **failure}
                _atomic_bytes(RESUME_MANIFEST, _json_bytes(failure))
            except OSError:
                pass
            if process.poll() is None:
                try:
                    group = os.getpgid(process.pid)
                    if group != process.pid or group == os.getpgrp():
                        raise RuntimeError("unsafe resumed driver process group")
                    os.killpg(group, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if RESUME_MANIFEST.is_file() and (watcher is None or watcher.poll() is not None):
                from scripts import watch_worker_fitting_train as scoped_watchdog
                if scoped_watchdog.MANIFEST != RESUME_MANIFEST:
                    raise RuntimeError("scoped watchdog selected a different manifest") from exc
                scoped_watchdog._cleanup()
            raise


def status() -> dict[str, object]:
    """Short status without reading any test-design result data."""

    if not MANIFEST.is_file():
        return {"phase": "not_launched", "data_root": str(DATA_ROOT)}
    active_manifest = RESUME_MANIFEST if RESUME_MANIFEST.is_file() else MANIFEST
    manifest = json.loads(active_manifest.read_text(encoding="utf-8"))
    state_file = DATA_ROOT / "status.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
    events = Journal(DATA_ROOT).read(CAMPAIGN_ID)
    latest_edges = {
        (event["edge"]["design"], tuple(event["edge"]["sequence"])): event["edge"]
        for event in events if event.get("type") == "edge"
    }
    completed = [
        event["phase"] for event in events if event.get("type") == "phase_complete"
    ]
    phase_files = sorted((DATA_ROOT / "phase_reports").glob("D*.json"))
    pid = manifest.get("driver_pid")
    alive = False
    if isinstance(pid, int):
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
            alive = b"--run" in args and str(Path(__file__).resolve()).encode() in args
        except OSError:
            pass
    return {
        "data_root": str(DATA_ROOT),
        "launch_phase": manifest["phase"],
        "driver_pid": pid, "driver_alive": alive,
        "phase": state.get("phase", "starting"),
        "phase_completed": completed,
        "committed_unique_edges": len(latest_edges),
        "committed_cache_hits": sum(edge["cache_hit"] for edge in latest_edges.values()),
        "busy_shared_runner_slots": _busy_slots(),
        "max_shared_runner_slots": SLOTS,
        "phase_report_files": [path.name for path in phase_files],
        "error": state.get("error"),
        "updated_at_utc": state.get("updated_at_utc"),
        "stderr_bytes": Path(manifest["stderr"]).stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--launch", action="store_true")
    group.add_argument("--resume", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.launch:
        result = launch()
        print(json.dumps({
            "phase": result["phase"], "driver_pid": result["driver_pid"],
            "train_designs": [row["design"] for row in result["train_inputs"]],
            "seeded_run_count": result["seeded_run_count"],
            "max_shared_runner_slots": SLOTS,
            "data_root": str(DATA_ROOT),
        }, sort_keys=True))
    elif args.resume:
        result = resume()
        print(json.dumps({
            "phase": result["phase"], "driver_pid": result["driver_pid"],
            "watchdog_pid": result["watchdog_pid"],
            "coverage": result["coverage"],
            "all_attempts_coverage": result["all_attempts_coverage"],
            "validator_fingerprint": result["validator_fingerprint"],
        }, sort_keys=True))
    elif args.run:
        run()
    elif args.status:
        print(json.dumps(status(), sort_keys=True))
    else:
        plan = _plan()
        print(json.dumps({
            "phase": "dry_run", "train_designs": [row["design"] for row in plan["train_inputs"]],
            "timing_recipes": [row["id"] for row in plan["recipe_sets"]["timing"]],
            "area_recipes": [row["id"] for row in plan["recipe_sets"]["area"]],
            "D1": plan["D1_planned_edges"], "D2": plan["D2_planned_edges"],
            "D3_max": plan["D3_max_new_edges"],
            "D4_total": plan["D4_total_within_worker_edges"],
            "slots": SLOTS,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

