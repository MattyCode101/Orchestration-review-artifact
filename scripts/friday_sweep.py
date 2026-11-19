"""Friday repair-loop variant sweep (variants pre-declared by decision of 2026-11-06).

Phase 1 runs the 16 pre-declared repair-loop variants (g01 = v1) on briefs A0
and T0 for the five training designs. Phase 2 runs the selected variant (v2) on
briefs T and A. Everything goes through the unchanged verified engine: the
tagged runner, AMD simulation of every new output, clean-property checks and
the counted real-run backend. Training designs only.

The engine root is the sealed v4 pilot root itself (the approved training
cache): cached outcomes are reused exactly, and new runs, states, verification
records and journals are only ever added to it; no existing file is changed.
A clone cannot work because every stored outcome embeds its root's absolute
state path under a checksum. All bookkeeping (plan, run counter, results,
status, shard logs) lives in this sweep's own directory, never in the v4
root. Briefs are snapshotted at process start and the plan pins their
fingerprints rather than the YAML bytes, so the later dated P0 addition cannot
disturb a running sweep.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orch.baselines.verification import VerificationLedger
from orch.config import load_noise, load_recipe_sets
from orch.contract.brief_text import load_frozen_brief
from orch.engine.apply import StepEngine
from orch.engine.runner import Runner
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from orch.episode import run_episode
from orch.policies.repair_loop import RepairLoopPolicy
from orch.runtime import make_real_runtime
from orch.workers.telemetry import SavedTelemetryProvider
from scripts import thursday_night_pilot as pilot

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
ROOT = PROJECT / "data/pilots/friday_repair_sweep_2026-11-06"
SOURCE_ROOT = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"
ENGINE_ROOT = SOURCE_ROOT
SELECT_SCRIPT = PROJECT / "scripts/friday_sweep_select.py"
REPORT_SCRIPT = PROJECT / "scripts/friday_sweep_report.py"
LAUNCH = ROOT / "sweep_launch.json"
CONTROL = ROOT / "sweep_control.json"
CAP = 500
PHASE1_SOFT_LIMIT = 440
SHARDS = {1: 12, 2: 10}
DESIGNS = tuple(pilot.DESIGNS)
PHASE1_BRIEFS = ("A0", "T0")
PHASE2_BRIEFS = ("T", "A")
GRID = tuple(itertools.product((1, 2), (True, False), (True, False), ("rollback", "continue")))
GRID_IDS = tuple(f"g{index:02d}" for index in range(1, len(GRID) + 1))
SOURCE_FILES = {
    "sweep_driver": "scripts/friday_sweep.py",
    "pilot": "scripts/thursday_night_pilot.py",
    "episode": "orch/episode.py",
    "repair_loop": "orch/policies/repair_loop.py",
    "views": "orch/contract/views.py",
    "worker_features": "configs/worker_features.yaml",
    "A_fit": "configs/workers/A_fit.json",
    "B_fit": "configs/workers/B_fit.json",
    "timing_card": "configs/workers/timing.yaml",
    "area_card": "configs/workers/area.yaml",
    "telemetry": "orch/workers/telemetry.py",
    "fitted_worker": "orch/workers/fitted.py",
    "worker_base": "orch/workers/base.py",
    "models": "orch/contract/models.py",
    "gate": "orch/engine/gate.py",
    "apply": "orch/engine/apply.py",
    "cache": "orch/engine/cache.py",
    "store": "orch/engine/store.py",
    "runner": "orch/engine/runner.py",
    "jobtag": "orch/engine/jobtag.py",
    "runtime": "orch/runtime.py",
    "real_backend": "orch/engine/real_backend.py",
    "meter": "orch/engine/meter.py",
    "meter_parsers": "orch/engine/meter_parsers.py",
    "verification": "orch/baselines/verification.py",
    "validator": "orch/recipes/adapters/validator.py",
    "tcl_adapter": "orch/recipes/adapters/tcl.py",
    "latch_adapter": "orch/recipes/adapters/latch.py",
    "registry": "recipes/registry.yaml",
    "env": "recipes/support/env.sh",
}
_hash = _sha256_file
_BRIEFS: dict[str, Any] = {}
_BOUND = False


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with _file_lock(path.with_suffix(path.suffix + ".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def policy_for_grid(grid_id: str) -> RepairLoopPolicy:
    attempts, preserve, gentle, after = GRID[GRID_IDS.index(grid_id)]
    return RepairLoopPolicy(
        repair_attempts=attempts, repair_preserve_placement=preserve,
        gentle_step=gentle, after_failed_repair=after,
    )


def episode_id(design: str, brief_id: str, grid_id: str) -> str:
    return f"sweep_{design}_{brief_id}_{grid_id}"


def phase1_rows() -> list[tuple[str, str, str]]:
    """Design-major so designs complete one at a time if the cap ever binds."""

    return [
        (design, brief, grid)
        for design in DESIGNS for brief in PHASE1_BRIEFS for grid in GRID_IDS
    ]


def phase2_rows(path: Path | None = None) -> list[tuple[str, str, str]]:
    work = pilot.read(path or ROOT / "phase2_work.json")
    grid_id = work.get("selected_grid_id")
    if grid_id not in GRID_IDS:
        raise ValueError("phase 2 names an unknown variant")
    rows = [tuple(row) for row in work.get("rows", [])]
    for row in rows:
        if (
            len(row) != 3 or row[0] not in DESIGNS
            or row[1] not in PHASE2_BRIEFS or row[2] != grid_id
        ):
            raise ValueError(f"phase 2 row is outside the pre-declared plan: {row}")
    if len(set(rows)) != len(rows):
        raise ValueError("phase 2 rows are duplicated")
    return rows  # type: ignore[return-value]


def _brief(brief_id: str):
    if brief_id not in _BRIEFS:
        _BRIEFS[brief_id] = load_frozen_brief(brief_id)
    return _BRIEFS[brief_id]


class PhasedBackend(pilot.CountedBackend):
    """The counted backend with an optional lower per-phase limit."""

    def __init__(self, *args: Any, soft_limit: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.soft_limit = soft_limit

    def execute(self, state, recipe, *, run_id, run_dir):
        if self.soft_limit is not None and self.count() >= self.soft_limit:
            raise pilot.PilotRunCap(f"sweep phase limit of {self.soft_limit} new runs reached")
        return super().execute(state, recipe, run_id=run_id, run_dir=run_dir)


def _make_plan(runtime) -> dict[str, Any]:
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    selected = {worker: tuple(sets.worker_sets[worker]) for worker in ("timing", "area")}
    if any(len(ids) != 5 for ids in selected.values()):
        raise pilot.PilotSafetyStop("sweep requires both signed five-recipe workers")
    references = pilot._training_inputs()
    briefs = {b: _brief(b).fingerprint() for b in (*PHASE1_BRIEFS, *PHASE2_BRIEFS)}
    source_plan = pilot.read(SOURCE_ROOT / "plan.json")
    if briefs != {b: source_plan["brief_hashes"][b] for b in briefs}:
        raise pilot.PilotSafetyStop("a sweep brief differs from the pilot's frozen brief")
    return {
        "schema_version": "friday-repair-sweep-v2",
        "training_only": True,
        "bookkeeping_root": str(ROOT),
        "engine_root": str(ENGINE_ROOT),
        "engine_root_plan_sha256": _hash(SOURCE_ROOT / "plan.json"),
        "training_inputs": references,
        "training_launch_manifest_sha256": _hash(pilot.TRAINING_ROOT / "launch_manifest.json"),
        "current_b2_sources": pilot._start_source_hashes(runtime.meter.version, references),
        "grid": {gid: list(knobs) for gid, knobs in zip(GRID_IDS, GRID)},
        "grid_versions": {gid: policy_for_grid(gid).version for gid in GRID_IDS},
        "phase1_schedule": [list(row) for row in phase1_rows()],
        "phase1_briefs": list(PHASE1_BRIEFS),
        "phase2_briefs": list(PHASE2_BRIEFS),
        "brief_hashes": briefs,
        "worker_sets": {worker: list(ids) for worker, ids in selected.items()},
        "recipe_versions": {
            rid: runtime.registry.get(rid).version
            for ids in selected.values() for rid in ids
        },
        "meter_version": runtime.meter.version,
        "smoke_sha256": _hash(pilot.SMOKE),
        "source_hashes": pilot.source_hashes(),
        "max_new_real_recipe_runs": CAP,
        "phase1_soft_limit": PHASE1_SOFT_LIMIT,
        "shared_slots": pilot.MAX_SLOTS,
        "load_ceiling_1m": pilot.LOAD_CEILING,
        "available_memory_floor_bytes": pilot.MEMORY_FLOOR,
    }


def _require_plan(runtime) -> dict[str, Any]:
    """Create once in the sweep directory, then fail closed on any change."""

    frozen = _make_plan(runtime)
    path = ROOT / "plan.json"
    expected = _json_bytes(frozen)
    ROOT.mkdir(parents=True, exist_ok=True)
    with _file_lock(ROOT / "plan.lock"):
        if path.is_file():
            if path.read_bytes() != expected:
                raise pilot.PilotSafetyStop("sweep plan differs from frozen source or input")
        else:
            _atomic_bytes(path, expected)
    return frozen


def _status(phase: str, *, error: str | None = None) -> dict[str, Any]:
    rows = [pilot.read(p) for p in (ROOT / "results").glob("*.json")] if (ROOT / "results").is_dir() else []
    counter = pilot.read(ROOT / "new_real_runs.json") if (ROOT / "new_real_runs.json").is_file() else {"reservations": []}
    value = {
        "schema_version": "friday-repair-sweep-status-v1", "phase": phase,
        "updated_at_utc": pilot.utc(), "completed_cells": len(rows),
        "new_real_runs_reserved": len(counter["reservations"]),
        "cache_hits": sum(row["cache_hits"] for row in rows),
        "recipe_runs_used": sum(row["runs_used"] for row in rows), "error": error,
    }
    _write_json(ROOT / "status.json", value)
    return value


def _targets() -> tuple[set[str], set[int]]:
    """Exact tags of tool jobs whose run IDs this sweep reserved; never other campaigns."""

    counter = ROOT / "new_real_runs.json"
    mine = {row["run_id"] for row in pilot.read(counter)["reservations"]} if counter.is_file() else set()
    prefix = str(ENGINE_ROOT / "runs").encode() + b"/"
    tags: set[str] = set()
    groups: set[int] = set()
    for pid in pilot._own_pids():
        info = pilot._stat(pid)
        if info is None or info.state == "Z":
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd").encode()
        except OSError:
            continue
        if not cwd.startswith(prefix):
            continue
        run_id = cwd[len(prefix):].split(b"/", 1)[0].decode()
        tag = pilot._tag_of(pid)
        if run_id in mine and tag and re.fullmatch(r"[0-9a-f]{32}", tag):
            tags.add(tag)
            groups.add(info.pgrp)
    groups.discard(os.getpgrp())
    return tags, groups


def _cleanup() -> dict[str, Any]:
    tags, groups = _targets()
    result: dict[str, Any] = {
        "schema_version": "friday-repair-sweep-cleanup-v1",
        "exact_tags": sorted(tags), "groups": sorted(groups), "started_at_utc": pilot.utc(),
    }
    _write_json(ROOT / "watchdog_cleanup_audit.json", result)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in sorted(groups):
            if any(pilot._tag_of(member.pid) in tags for member in pilot.group_members(group)):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    result["tag_sweeps"] = {tag: list(pilot.sweep_tag(tag, grace_s=5).alive) for tag in sorted(tags)}
    result["survivors"] = {t: [p.pid for p in pilot.tagged(t)] for t in sorted(tags) if pilot.tagged(t)}
    result["finished_at_utc"] = pilot.utc()
    _write_json(ROOT / "watchdog_cleanup_audit.json", result)
    return result


def _bind() -> None:
    """Point the unchanged pilot helpers at the engine root, plan and pins."""

    global _BOUND
    if _BOUND:
        return
    for brief_id in (*PHASE1_BRIEFS, *PHASE2_BRIEFS):
        _brief(brief_id)
    status = pilot.read(SOURCE_ROOT / "status.json")
    if status.get("phase") != "complete":
        raise pilot.PilotSafetyStop("the approved cache root is not sealed complete")
    pilot.ROOT = ENGINE_ROOT
    pilot.CAP = CAP
    pilot.SOURCE_FILES = dict(SOURCE_FILES)
    pilot.make_plan = _make_plan
    pilot.require_plan = _require_plan
    pilot._status = _status
    pilot.cleanup = _cleanup
    pilot.episode_schedule = lambda: tuple(phase1_rows())
    _BOUND = True


def _runtime():
    return make_real_runtime(
        ENGINE_ROOT, cache_enabled=True, max_slots=pilot.MAX_SLOTS,
        load_ceiling_1m=pilot.LOAD_CEILING,
        min_available_memory_bytes=pilot.MEMORY_FLOOR,
        latch_smoke_status=pilot.SMOKE,
    )


@dataclass
class _Context:
    runtime: Any
    starts: dict[str, Any]
    workers: dict[str, Any]
    provider: SavedTelemetryProvider
    ledger: VerificationLedger
    backend: PhasedBackend
    engine: pilot.VerifiedEpisodeEngine


def _context(soft_limit: int | None) -> _Context:
    runtime = _runtime()
    frozen = _require_plan(runtime)
    starts = pilot._seed_starts(runtime, frozen["training_inputs"])
    workers = pilot._workers(runtime)
    provider = SavedTelemetryProvider(ENGINE_ROOT)
    ledger = VerificationLedger(
        store=runtime.store, runner=runtime.runner,
        validator=runtime.validator, references=frozen["training_inputs"],
    )
    backend = PhasedBackend(
        runtime.backend, ROOT, _hash(ROOT / "plan.json"),
        lambda: _require_plan(runtime), cap=CAP, soft_limit=soft_limit,
    )
    engine = pilot.VerifiedEpisodeEngine(
        StepEngine(
            store=runtime.store, cache=runtime.cache, backend=backend,
            meter_version=runtime.meter.version, cache_enabled=True,
        ), ledger,
    )
    (ROOT / "results").mkdir(exist_ok=True)
    return _Context(runtime, starts, workers, provider, ledger, backend, engine)


def _run_cell(ctx: _Context, phase: int, cell: tuple[str, str, str]) -> str:
    design, brief_id, grid_id = cell
    episode = episode_id(design, brief_id, grid_id)
    result_path = ROOT / "results" / f"{episode}.json"
    if result_path.is_file():
        saved = pilot.read(result_path)
        if (
            saved.get("episode_id") != episode or saved.get("design") != design
            or saved.get("brief") != brief_id or saved.get("policy") != grid_id
            or saved.get("final_brief_valid") is not True
        ):
            raise pilot.PilotSafetyStop("saved sweep result identity differs")
        pilot._audit_episode(episode, ctx.runtime.store, ctx.ledger)
        return "skipped"
    pilot._audit_episode(episode, ctx.runtime.store, ctx.ledger)
    brief = _brief(brief_id)
    start = ctx.starts[design]
    start_evidence = ctx.engine.measure_start(start)
    noise = load_noise(PROJECT / "configs/noise.yaml").for_start(design, start_evidence)
    ctx.backend.active_episode = episode
    try:
        result = run_episode(
            engine=ctx.engine, registry=ctx.runtime.registry, workers=ctx.workers,
            policy=policy_for_grid(grid_id), brief=brief, start_state=start,
            episode_id=episode, noise=noise, telemetry_provider=ctx.provider,
            validator=pilot._validator(ctx.ledger, brief),
        )
    except pilot.PilotRunCap as exc:
        _append_jsonl(ROOT / "capped_cells.jsonl", {
            "episode_id": episode, "phase": phase, "reason": str(exc), "at_utc": pilot.utc(),
        })
        return "capped"
    finally:
        ctx.backend.active_episode = None
    pilot._audit_episode(episode, ctx.runtime.store, ctx.ledger)
    row = pilot._result_row(result, ctx.runtime.store, ctx.backend, design, brief_id, grid_id)
    row.update({
        "campaign": "friday-repair-sweep", "phase": phase, "grid_id": grid_id,
        "knobs": list(GRID[GRID_IDS.index(grid_id)]),
    })
    _atomic_bytes(result_path, _json_bytes(row))
    _status("running")
    return "done"


def prepare() -> dict[str, Any]:
    """Freeze the plan and seed the five starts; no tool job."""

    _bind()
    ROOT.mkdir(parents=True, exist_ok=True)
    runtime = _runtime()
    frozen = _require_plan(runtime)
    starts = pilot._seed_starts(runtime, frozen["training_inputs"])
    pilot._workers(runtime)
    SavedTelemetryProvider(ENGINE_ROOT)
    plan_sha = _hash(ROOT / "plan.json")
    PhasedBackend(runtime.backend, ROOT, plan_sha, lambda: _require_plan(runtime), cap=CAP)
    result = {
        "schema_version": "friday-repair-sweep-prepared-v1",
        "prepared_at_utc": pilot.utc(), "plan_sha256": plan_sha,
        "meter_version": runtime.meter.version,
        "start_states": len(starts), "new_real_cap": CAP,
        "phase1_cells": len(phase1_rows()), "engine_root": str(ENGINE_ROOT),
        "tool_jobs_launched": 0,
    }
    _write_json(ROOT / "prepared.json", result)
    return result


def smoke() -> dict[str, Any]:
    """Run the first cell with zero new real runs allowed and compare to the pilot."""

    _bind()
    prepare()
    cell = phase1_rows()[0]
    ctx = _context(0)
    outcome = _run_cell(ctx, 1, cell)
    row = pilot.read(ROOT / "results" / f"{episode_id(*cell)}.json")
    reference = pilot.read(
        SOURCE_ROOT / "results" / f"pilot_{cell[0]}_{cell[1]}_repair_loop.json"
    )
    keys = ("final_state_id", "final_metrics", "runs_used", "repair_final_phase")
    match = {key: row[key] == reference[key] for key in keys}
    return {
        "cell": list(cell), "outcome": outcome, "matches_pilot": match,
        "all_match": all(match.values()), "new_real_runs": row["new_real_runs_reserved"],
    }


def _write_shard(phase: int, shard: int, state: str, counts: dict[str, int],
                 planned: int, error: str | None = None) -> None:
    _write_json(ROOT / f"sweep_shard_p{phase}_{shard}.json", {
        "phase": phase, "shard": shard, "state": state, "planned": planned,
        "counts": counts, "error": error, "updated_at_utc": pilot.utc(),
    })


def run_shard(phase: int, shard: int) -> None:
    _bind()
    ctx = _context(PHASE1_SOFT_LIMIT if phase == 1 else None)
    rows = (phase1_rows() if phase == 1 else phase2_rows())[shard::SHARDS[phase]]
    counts = {"done": 0, "skipped": 0, "capped": 0}
    _write_shard(phase, shard, "running", counts, len(rows))
    try:
        for cell in rows:
            _require_plan(ctx.runtime)
            counts[_run_cell(ctx, phase, cell)] += 1
            _write_shard(phase, shard, "running", counts, len(rows))
    except BaseException as exc:
        _write_shard(phase, shard, "failed", counts, len(rows), f"{type(exc).__name__}: {exc}")
        raise
    _write_shard(phase, shard, "complete", counts, len(rows))


def _ctl(state: str, **extra: Any) -> None:
    _write_json(CONTROL, {
        "state": state, "controller_pid": os.getpid(), "updated_at_utc": pilot.utc(), **extra,
    })


def _spawn_shard(phase: int, shard: int) -> subprocess.Popen:
    tag = f"sweep_p{phase}_s{shard}"
    with (ROOT / f"{tag}.stdout").open("ab") as out, (ROOT / f"{tag}.stderr").open("ab") as err:
        return subprocess.Popen(
            [sys.executable, str(SCRIPT), "--shard", str(shard), "--phase", str(phase)],
            cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            start_new_session=True, close_fds=True,
        )


def _run_phase(phase: int) -> dict[str, Any]:
    """Run all shards of one phase; restart a failed shard at most twice."""

    attempts = {shard: 1 for shard in range(SHARDS[phase])}
    live = {shard: _spawn_shard(phase, shard) for shard in attempts}
    final: dict[int, int] = {}
    _ctl(f"running_phase{phase}", pids={str(s): p.pid for s, p in live.items()})
    while live:
        time.sleep(15)
        for shard, proc in list(live.items()):
            code = proc.poll()
            if code is None:
                continue
            del live[shard]
            if code != 0 and attempts[shard] < 3:
                time.sleep(30)
                attempts[shard] += 1
                live[shard] = _spawn_shard(phase, shard)
            else:
                final[shard] = code
        _ctl(f"running_phase{phase}", live=sorted(live), finished={str(s): c for s, c in final.items()})
    summary = {"phase": phase, "exit_codes": {str(s): c for s, c in final.items()},
               "attempts": {str(s): a for s, a in attempts.items()}}
    _write_json(ROOT / f"sweep_phase{phase}_summary.json", summary)
    return summary


def _run_until_success(script: Path, marker: Path, label: str, max_attempts: int) -> bool:
    for attempt in range(1, max_attempts + 1):
        proc = subprocess.run(
            [sys.executable, str(script)], cwd=PROJECT, capture_output=True, text=True,
        )
        _append_jsonl(ROOT / f"{label}_attempts.jsonl", {
            "attempt": attempt, "returncode": proc.returncode, "at_utc": pilot.utc(),
            "stderr_tail": proc.stderr[-1500:], "stdout_tail": proc.stdout[-500:],
        })
        if proc.returncode == 0 and marker.is_file():
            return True
        _ctl(f"{label}_retrying", attempt=attempt)
        time.sleep(300)
    return False


def control() -> None:
    _bind()
    _ctl("starting")
    _run_phase(1)
    _ctl("selecting")
    if not _run_until_success(SELECT_SCRIPT, ROOT / "phase2_work.json", "select", 144):
        _ctl("select_failed")
        return
    _run_phase(2)
    _ctl("reporting")
    ok = _run_until_success(REPORT_SCRIPT, PROJECT / "notes/repair_loop_v2_selection.md", "report", 24)
    _ctl("complete" if ok else "report_failed")


def _alive(pid: int, mode: bytes) -> bool:
    info = pilot._stat(pid)
    if info is None or info.state == "Z":
        return False
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return str(SCRIPT).encode() in args and mode in args


def watch() -> None:
    _bind()
    pid = int(pilot.read(LAUNCH)["controller_pid"])
    deadline = time.monotonic() + 72 * 3600
    while time.monotonic() < deadline:
        if not _alive(pid, b"--control"):
            state = pilot.read(CONTROL) if CONTROL.is_file() else {}
            terminal = state.get("state") in ("complete", "report_failed", "select_failed")
            tags, _ = _targets()
            if not terminal or tags:
                _write_json(ROOT / "sweep_watchdog_cleanup.json", _cleanup())
                if not terminal:
                    _ctl("failed", error="controller exited early; watchdog swept exact tags")
            return
        time.sleep(15)
    _write_json(ROOT / "sweep_watchdog_cleanup.json", _cleanup())
    _ctl("failed", error="watchdog lifetime expired")


def launch() -> dict[str, Any]:
    _bind()
    prepared = prepare()
    load, available = Runner._resource_snapshot()
    if load > pilot.LOAD_CEILING or available < pilot.MEMORY_FLOOR:
        raise pilot.PilotSafetyStop("load or available memory blocks the sweep launch")
    with _file_lock(ROOT / "sweep_launch.lock"):
        if LAUNCH.exists():
            raise FileExistsError("the sweep was already launched")
        manifest: dict[str, Any] = {
            "schema_version": "friday-repair-sweep-launch-v1",
            "started_at_utc": pilot.utc(), "plan_sha256": prepared["plan_sha256"],
            "driver_sha256": _hash(SCRIPT), "new_real_cap": CAP,
            "phase1_soft_limit": PHASE1_SOFT_LIMIT, "shards": SHARDS,
            "load_1m_at_launch": load, "available_memory_bytes_at_launch": available,
        }
        _write_json(LAUNCH, manifest)
        with (ROOT / "sweep_control.stdout").open("wb") as out, (
            ROOT / "sweep_control.stderr"
        ).open("wb") as err:
            controller = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--control"], cwd=PROJECT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        manifest["controller_pid"] = controller.pid
        _write_json(LAUNCH, manifest)
        with (ROOT / "sweep_watchdog.stdout").open("wb") as out, (
            ROOT / "sweep_watchdog.stderr"
        ).open("wb") as err:
            watcher = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--watch"], cwd=PROJECT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        manifest["watchdog_pid"] = watcher.pid
        _write_json(LAUNCH, manifest)
        return manifest


def status() -> dict[str, Any]:
    _bind()
    shards = {}
    for path in sorted(ROOT.glob("sweep_shard_p*_*.json")):
        shards[path.stem] = pilot.read(path)
    counter = pilot.read(ROOT / "new_real_runs.json") if (ROOT / "new_real_runs.json").is_file() else {"reservations": []}
    results = list((ROOT / "results").glob("*.json")) if (ROOT / "results").is_dir() else []
    return {
        "control": pilot.read(CONTROL) if CONTROL.is_file() else None,
        "launch": pilot.read(LAUNCH) if LAUNCH.is_file() else None,
        "completed_cells": len(results),
        "new_real_runs_reserved": len(counter["reservations"]),
        "shards": {k: {"state": v["state"], "counts": v["counts"], "planned": v["planned"],
                       "error": v["error"]} for k, v in shards.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "smoke", "launch", "control", "watch", "status"):
        modes.add_argument(f"--{name}", action="store_true")
    modes.add_argument("--shard", type=int)
    parser.add_argument("--phase", type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), sort_keys=True))
    elif args.smoke:
        print(json.dumps(smoke(), sort_keys=True))
    elif args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.control:
        control()
    elif args.watch:
        watch()
    elif args.status:
        print(json.dumps(status(), sort_keys=True))
    else:
        if args.phase is None:
            parser.error("--shard needs --phase")
        run_shard(args.phase, args.shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
