"""Friday P0 pilot: every existing policy on the power brief (training only).

Brief P0 minimises dynamic power with WNS >= start and packed LUTs <= start.
Three workers take part: timing A and area B (the frozen FINAL fitted workers)
and power C (a ranked list, no features: validity-weighted mean power saving
over the training power runs, equal weight per design; two attempts by
default). The 11 policies are v1 repair loop (three-worker repair), single
specialist, the six fixed orders of the three workers, and random with three
seeds over the relevant workers.

As in the sweep, the engine root is the sealed v4 training cache itself (only
additions), and all bookkeeping lives in this pilot's own directory. Hard cap
of 300 new real recipe runs, shared tagged runner, AMD simulation of every new
output, clean-property checks. Training designs only.
"""
from __future__ import annotations

import argparse
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
from orch.config import load_noise, load_recipe_sets, load_worker_card
from orch.contract.brief_text import load_frozen_brief
from orch.engine.apply import StepEngine
from orch.engine.runner import Runner
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from orch.episode import run_episode
from orch.policies.relevant_workers import FixedOrderPolicy, RelevantRandomPolicy
from orch.policies.repair_loop import RepairLoopPolicy
from orch.policies.training_baselines import SingleSpecialistPolicy
from orch.runtime import make_real_runtime
from orch.workers.base import GenericWorker
from orch.workers.telemetry import SavedTelemetryProvider
from scripts import thursday_night_pilot as pilot

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
ROOT = PROJECT / "data/pilots/friday_p0_pilot_2026-11-06"
SOURCE_ROOT = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"
ENGINE_ROOT = SOURCE_ROOT
RANKING_FILE = PROJECT / "data/scratch_friday_2026-11-06/worker_c_ranking.json"
REPORT_SCRIPT = PROJECT / "scripts/friday_p0_report.py"
REPORT_OUTPUT = PROJECT / "notes/p0_pilot_training.md"
LAUNCH = ROOT / "p0_launch.json"
CONTROL = ROOT / "p0_control.json"
CAP = 300
SHARD_COUNT = 10
BRIEF = "P0"
DESIGNS = tuple(pilot.DESIGNS)
POLICIES = (
    "repair_loop", "single_specialist", "fixed_CAB", "fixed_CBA",
    "random_seed_0", "random_seed_1", "random_seed_2",
    "fixed_ACB", "fixed_ABC", "fixed_BCA", "fixed_BAC",
)
SELECTED_WORKERS = {
    "timing": 5, "area": 5, "power": 4,
}
SOURCE_FILES = {
    "p0_driver": "scripts/friday_p0.py",
    "pilot": "scripts/thursday_night_pilot.py",
    "episode": "orch/episode.py",
    "repair_loop": "orch/policies/repair_loop.py",
    "relevant_workers": "orch/policies/relevant_workers.py",
    "baselines": "orch/policies/training_baselines.py",
    "views": "orch/contract/views.py",
    "brief_text": "orch/contract/brief_text.py",
    "briefs": "configs/briefs.yaml",
    "recipe_sets": "configs/recipe_sets.yaml",
    "power_recipe_set": "configs/power_recipe_set.yaml",
    "power_card": "configs/workers/power.yaml",
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


def episode_id(design: str, brief_id: str, policy_key: str) -> str:
    return f"p0_{design}_{brief_id}_{policy_key}"


def rows() -> list[tuple[str, str, str]]:
    """Policy-major, so a binding cap would leave the last policies incomplete."""

    return [(design, BRIEF, policy) for policy in POLICIES for design in DESIGNS]


def policy_for(key: str):
    if key == "repair_loop":
        return RepairLoopPolicy()
    if key == "single_specialist":
        return SingleSpecialistPolicy()
    if key.startswith("fixed_") and key in POLICIES:
        return FixedOrderPolicy(order=key.split("_", 1)[1])
    if key.startswith("random_seed_") and key in POLICIES:
        return RelevantRandomPolicy(seed=int(key.rsplit("_", 1)[1]))
    raise ValueError("unknown P0 pilot policy")


def _brief(brief_id: str):
    if brief_id not in _BRIEFS:
        _BRIEFS[brief_id] = load_frozen_brief(brief_id)
    return _BRIEFS[brief_id]


def _ranking() -> list[str]:
    data = pilot.read(RANKING_FILE)
    if data.get("schema_version") != "friday-worker-c-ranking-v1":
        raise pilot.PilotSafetyStop("worker C ranking file has an unknown schema")
    return list(data["ranking"])


class PhasedBackend(pilot.CountedBackend):
    """The counted backend; present for symmetry with the sweep."""


def _recipe_sets() -> dict[str, tuple[str, ...]]:
    """Timing and area from the frozen sets; power from worker C's own dated file."""

    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    power = load_recipe_sets(PROJECT / "configs/power_recipe_set.yaml")
    power.require_signed_off()
    return {
        "timing": tuple(sets.worker_sets["timing"]),
        "area": tuple(sets.worker_sets["area"]),
        "power": tuple(power.worker_sets["power"]),
    }


def _make_plan(runtime) -> dict[str, Any]:
    selected = _recipe_sets()
    if any(len(selected[w]) != n for w, n in SELECTED_WORKERS.items()):
        raise pilot.PilotSafetyStop("P0 pilot needs the signed 5 + 5 + 4 recipe workers")
    if list(selected["power"]) != _ranking():
        raise pilot.PilotSafetyStop("power recipe order differs from the ranking file")
    return {
        "schema_version": "friday-p0-pilot-v1",
        "training_only": True,
        "bookkeeping_root": str(ROOT),
        "engine_root": str(ENGINE_ROOT),
        "engine_root_plan_sha256": _hash(SOURCE_ROOT / "plan.json"),
        "training_inputs": pilot._training_inputs(),
        "training_launch_manifest_sha256": _hash(pilot.TRAINING_ROOT / "launch_manifest.json"),
        "current_b2_sources": pilot._start_source_hashes(
            runtime.meter.version, pilot._training_inputs()
        ),
        "schedule": [list(row) for row in rows()],
        "brief_hashes": {BRIEF: _brief(BRIEF).fingerprint()},
        "policy_versions": {key: policy_for(key).version for key in POLICIES},
        "worker_sets": {w: list(ids) for w, ids in selected.items()},
        "worker_c_ranking_sha256": _hash(RANKING_FILE),
        "recipe_versions": {
            rid: runtime.registry.get(rid).version
            for ids in selected.values() for rid in ids
        },
        "meter_version": runtime.meter.version,
        "smoke_sha256": _hash(pilot.SMOKE),
        "source_hashes": pilot.source_hashes(),
        "max_new_real_recipe_runs": CAP,
        "shared_slots": pilot.MAX_SLOTS,
        "load_ceiling_1m": pilot.LOAD_CEILING,
        "available_memory_floor_bytes": pilot.MEMORY_FLOOR,
    }


def _require_plan(runtime) -> dict[str, Any]:
    frozen = _make_plan(runtime)
    path = ROOT / "plan.json"
    expected = _json_bytes(frozen)
    ROOT.mkdir(parents=True, exist_ok=True)
    with _file_lock(ROOT / "plan.lock"):
        if path.is_file():
            if path.read_bytes() != expected:
                raise pilot.PilotSafetyStop("P0 plan differs from frozen source or input")
        else:
            _atomic_bytes(path, expected)
    return frozen


def _status(phase: str, *, error: str | None = None) -> dict[str, Any]:
    done = [pilot.read(p) for p in (ROOT / "results").glob("*.json")] if (ROOT / "results").is_dir() else []
    counter = pilot.read(ROOT / "new_real_runs.json") if (ROOT / "new_real_runs.json").is_file() else {"reservations": []}
    value = {
        "schema_version": "friday-p0-status-v1", "phase": phase,
        "updated_at_utc": pilot.utc(), "planned_cells": len(rows()),
        "completed_cells": len(done),
        "new_real_runs_reserved": len(counter["reservations"]),
        "cache_hits": sum(row["cache_hits"] for row in done),
        "recipe_runs_used": sum(row["runs_used"] for row in done), "error": error,
    }
    _write_json(ROOT / "status.json", value)
    return value


def _targets() -> tuple[set[str], set[int]]:
    """Exact tags of tool jobs whose run IDs this pilot reserved; never other campaigns."""

    counter = ROOT / "new_real_runs.json"
    mine = {r["run_id"] for r in pilot.read(counter)["reservations"]} if counter.is_file() else set()
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
        "schema_version": "friday-p0-cleanup-v1",
        "exact_tags": sorted(tags), "groups": sorted(groups), "started_at_utc": pilot.utc(),
    }
    _write_json(ROOT / "watchdog_cleanup_audit.json", result)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in sorted(groups):
            if any(pilot._tag_of(m.pid) in tags for m in pilot.group_members(group)):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    result["tag_sweeps"] = {t: list(pilot.sweep_tag(t, grace_s=5).alive) for t in sorted(tags)}
    result["survivors"] = {t: [p.pid for p in pilot.tagged(t)] for t in sorted(tags) if pilot.tagged(t)}
    result["finished_at_utc"] = pilot.utc()
    _write_json(ROOT / "watchdog_cleanup_audit.json", result)
    return result


def _bind() -> None:
    global _BOUND
    if _BOUND:
        return
    _brief(BRIEF)
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
    pilot.episode_schedule = lambda: tuple(rows())
    _BOUND = True


def _runtime():
    return make_real_runtime(
        ENGINE_ROOT, cache_enabled=True, max_slots=pilot.MAX_SLOTS,
        load_ceiling_1m=pilot.LOAD_CEILING,
        min_available_memory_bytes=pilot.MEMORY_FLOOR,
        latch_smoke_status=pilot.SMOKE,
    )


def _workers(runtime) -> dict[str, Any]:
    workers = dict(pilot._workers(runtime))
    card = load_worker_card(PROJECT / "configs/workers/power.yaml")
    if card.id != "power" or card.objective.metric.value != "dynamic_power_w":
        raise pilot.PilotSafetyStop("power worker card is not the dynamic-power worker")
    workers["power"] = GenericWorker(card, _recipe_sets()["power"])
    return workers


@dataclass
class _Context:
    runtime: Any
    starts: dict[str, Any]
    workers: dict[str, Any]
    provider: SavedTelemetryProvider
    ledger: VerificationLedger
    backend: PhasedBackend
    engine: pilot.VerifiedEpisodeEngine


def _context() -> _Context:
    runtime = _runtime()
    frozen = _require_plan(runtime)
    starts = pilot._seed_starts(runtime, frozen["training_inputs"])
    workers = _workers(runtime)
    provider = SavedTelemetryProvider(ENGINE_ROOT)
    ledger = VerificationLedger(
        store=runtime.store, runner=runtime.runner,
        validator=runtime.validator, references=frozen["training_inputs"],
    )
    backend = PhasedBackend(
        runtime.backend, ROOT, _hash(ROOT / "plan.json"),
        lambda: _require_plan(runtime), cap=CAP,
    )
    engine = pilot.VerifiedEpisodeEngine(
        StepEngine(
            store=runtime.store, cache=runtime.cache, backend=backend,
            meter_version=runtime.meter.version, cache_enabled=True,
        ), ledger,
    )
    (ROOT / "results").mkdir(exist_ok=True)
    return _Context(runtime, starts, workers, provider, ledger, backend, engine)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with _file_lock(path.with_suffix(path.suffix + ".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


def _run_cell(ctx: _Context, cell: tuple[str, str, str]) -> str:
    design, brief_id, key = cell
    episode = episode_id(design, brief_id, key)
    result_path = ROOT / "results" / f"{episode}.json"
    if result_path.is_file():
        saved = pilot.read(result_path)
        if (
            saved.get("episode_id") != episode or saved.get("design") != design
            or saved.get("brief") != brief_id or saved.get("policy") != key
            or saved.get("final_brief_valid") is not True
        ):
            raise pilot.PilotSafetyStop("saved P0 result identity differs")
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
            policy=policy_for(key), brief=brief, start_state=start,
            episode_id=episode, noise=noise, telemetry_provider=ctx.provider,
            validator=pilot._validator(ctx.ledger, brief),
        )
    except pilot.PilotRunCap as exc:
        _append_jsonl(ROOT / "capped_cells.jsonl", {
            "episode_id": episode, "reason": str(exc), "at_utc": pilot.utc(),
        })
        return "capped"
    finally:
        ctx.backend.active_episode = None
    pilot._audit_episode(episode, ctx.runtime.store, ctx.ledger)
    row = pilot._result_row(result, ctx.runtime.store, ctx.backend, design, brief_id, key)
    row.update({"campaign": "friday-p0-pilot", "policy_version": policy_for(key).version})
    _atomic_bytes(result_path, _json_bytes(row))
    _status("running")
    return "done"


def prepare() -> dict[str, Any]:
    """Freeze the plan, build all three workers and the provider; no tool job."""

    _bind()
    ROOT.mkdir(parents=True, exist_ok=True)
    runtime = _runtime()
    frozen = _require_plan(runtime)
    starts = pilot._seed_starts(runtime, frozen["training_inputs"])
    workers = _workers(runtime)
    SavedTelemetryProvider(ENGINE_ROOT)
    plan_sha = _hash(ROOT / "plan.json")
    PhasedBackend(runtime.backend, ROOT, plan_sha, lambda: _require_plan(runtime), cap=CAP)
    result = {
        "schema_version": "friday-p0-prepared-v1", "prepared_at_utc": pilot.utc(),
        "plan_sha256": plan_sha, "meter_version": runtime.meter.version,
        "start_states": len(starts), "workers": sorted(workers),
        "power_order": list(workers["power"].recipe_ids), "new_real_cap": CAP,
        "cells": len(rows()), "engine_root": str(ENGINE_ROOT), "tool_jobs_launched": 0,
    }
    _write_json(ROOT / "prepared.json", result)
    return result


def _write_shard(shard: int, state: str, counts: dict[str, int], planned: int,
                 error: str | None = None) -> None:
    _write_json(ROOT / f"p0_shard_{shard}.json", {
        "shard": shard, "state": state, "planned": planned, "counts": counts,
        "error": error, "updated_at_utc": pilot.utc(),
    })


def run_shard(shard: int) -> None:
    _bind()
    ctx = _context()
    mine = rows()[shard::SHARD_COUNT]
    counts = {"done": 0, "skipped": 0, "capped": 0}
    _write_shard(shard, "running", counts, len(mine))
    try:
        for cell in mine:
            _require_plan(ctx.runtime)
            counts[_run_cell(ctx, cell)] += 1
            _write_shard(shard, "running", counts, len(mine))
    except BaseException as exc:
        _write_shard(shard, "failed", counts, len(mine), f"{type(exc).__name__}: {exc}")
        raise
    _write_shard(shard, "complete", counts, len(mine))


def _ctl(state: str, **extra: Any) -> None:
    _write_json(CONTROL, {
        "state": state, "controller_pid": os.getpid(), "updated_at_utc": pilot.utc(), **extra,
    })


def _spawn_shard(shard: int) -> subprocess.Popen:
    with (ROOT / f"p0_s{shard}.stdout").open("ab") as out, (ROOT / f"p0_s{shard}.stderr").open("ab") as err:
        return subprocess.Popen(
            [sys.executable, str(SCRIPT), "--shard", str(shard)],
            cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            start_new_session=True, close_fds=True,
        )


def _run_shards() -> dict[str, Any]:
    attempts = {s: 1 for s in range(SHARD_COUNT)}
    live = {s: _spawn_shard(s) for s in attempts}
    final: dict[int, int] = {}
    _ctl("running", pids={str(s): p.pid for s, p in live.items()})
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
                live[shard] = _spawn_shard(shard)
            else:
                final[shard] = code
        _ctl("running", live=sorted(live), finished={str(s): c for s, c in final.items()})
    summary = {"exit_codes": {str(s): c for s, c in final.items()},
               "attempts": {str(s): a for s, a in attempts.items()}}
    _write_json(ROOT / "p0_shards_summary.json", summary)
    return summary


def control() -> None:
    _bind()
    _ctl("starting")
    _run_shards()
    _ctl("reporting")
    for attempt in range(1, 25):
        proc = subprocess.run(
            [sys.executable, str(REPORT_SCRIPT)], cwd=PROJECT, capture_output=True, text=True,
        )
        _append_jsonl(ROOT / "report_attempts.jsonl", {
            "attempt": attempt, "returncode": proc.returncode, "at_utc": pilot.utc(),
            "stderr_tail": proc.stderr[-1500:],
        })
        if proc.returncode == 0 and REPORT_OUTPUT.is_file():
            _ctl("complete")
            return
        _ctl("report_retrying", attempt=attempt)
        time.sleep(300)
    _ctl("report_failed")


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
    deadline = time.monotonic() + 48 * 3600
    while time.monotonic() < deadline:
        if not _alive(pid, b"--control"):
            state = pilot.read(CONTROL) if CONTROL.is_file() else {}
            terminal = state.get("state") in ("complete", "report_failed")
            tags, _ = _targets()
            if not terminal or tags:
                _write_json(ROOT / "p0_watchdog_cleanup.json", _cleanup())
                if not terminal:
                    _ctl("failed", error="controller exited early; watchdog swept exact tags")
            return
        time.sleep(15)
    _write_json(ROOT / "p0_watchdog_cleanup.json", _cleanup())
    _ctl("failed", error="watchdog lifetime expired")


def launch() -> dict[str, Any]:
    _bind()
    prepared = prepare()
    load, available = Runner._resource_snapshot()
    if load > pilot.LOAD_CEILING or available < pilot.MEMORY_FLOOR:
        raise pilot.PilotSafetyStop("load or available memory blocks the P0 launch")
    with _file_lock(ROOT / "p0_launch.lock"):
        if LAUNCH.exists():
            raise FileExistsError("the P0 pilot was already launched")
        manifest: dict[str, Any] = {
            "schema_version": "friday-p0-launch-v1", "started_at_utc": pilot.utc(),
            "plan_sha256": prepared["plan_sha256"], "driver_sha256": _hash(SCRIPT),
            "new_real_cap": CAP, "shards": SHARD_COUNT,
            "load_1m_at_launch": load, "available_memory_bytes_at_launch": available,
        }
        _write_json(LAUNCH, manifest)
        with (ROOT / "p0_control.stdout").open("wb") as out, (ROOT / "p0_control.stderr").open("wb") as err:
            controller = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--control"], cwd=PROJECT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True, close_fds=True,
            )
        manifest["controller_pid"] = controller.pid
        _write_json(LAUNCH, manifest)
        with (ROOT / "p0_watchdog.stdout").open("wb") as out, (ROOT / "p0_watchdog.stderr").open("wb") as err:
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
    shards = {p.stem: pilot.read(p) for p in sorted(ROOT.glob("p0_shard_*.json"))}
    counter = pilot.read(ROOT / "new_real_runs.json") if (ROOT / "new_real_runs.json").is_file() else {"reservations": []}
    results = list((ROOT / "results").glob("*.json")) if (ROOT / "results").is_dir() else []
    return {
        "control": pilot.read(CONTROL) if CONTROL.is_file() else None,
        "launch": pilot.read(LAUNCH) if LAUNCH.is_file() else None,
        "completed_cells": len(results), "planned_cells": len(rows()),
        "new_real_runs_reserved": len(counter["reservations"]),
        "shards": {k: {"state": v["state"], "counts": v["counts"], "planned": v["planned"],
                       "error": v["error"]} for k, v in shards.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "launch", "control", "watch", "status"):
        modes.add_argument(f"--{name}", action="store_true")
    modes.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), sort_keys=True))
    elif args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.control:
        control()
    elif args.watch:
        watch()
    elif args.status:
        print(json.dumps(status(), sort_keys=True))
    else:
        run_shard(args.shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
