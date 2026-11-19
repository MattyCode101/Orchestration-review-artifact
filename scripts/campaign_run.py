"""Run a campaign (training rehearsal now, the test campaign later) through the harness.

  --prepare   write the frozen plan (cells, caps, hashes) for a named run
  --launch    start controller + shards detached, in ONE process group (the controller's)
  --resume    the same after an interruption; finished episodes are skipped, journals resumed
  --stop      SIGTERM the controller's process group (exact group), nothing else
  --cleanup   kill leftover tool jobs of this run by exact job tag
  --status    counts, caps, runs, pin state

Scopes: train (5 training designs, training manifest) or test (5 test designs, test manifest).
A test launch is refused unless --freeze-approved names a freeze file whose hashes all match, and
this tool never launches the test scope without it. Priority order: tier 1, then 2, then 3.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT))

import campaign_harness as h  # noqa: E402
from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file  # noqa: E402

pilot = h.pilot
SCRIPT = Path(__file__).resolve()
DATA = PROJECT / "data"
TRAIN_MANIFEST = DATA / "campaign_prep_2026-11-07/train_cache_manifest.json"
TEST_MANIFEST = DATA / "test_campaign_prep_2026-11-06/cache_manifest.json"


def work_root(name: str) -> Path:
    return DATA / f"campaign_{name}"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


def scope_for(plan: dict[str, Any], work: Path):
    if plan["scope"] == "train":
        scope = h.train_scope(work, TRAIN_MANIFEST, plan["manifest_sha256"])
    else:
        scope = h.test_scope(work, TEST_MANIFEST, plan["manifest_sha256"])
    if plan.get("telemetry") == "live":
        scope.provider_for = h.live_provider_factory(work, plan["caps"], plan_sha(work), int(plan["extraction_jobs"]))
    elif plan["scope"] == "test":
        raise pilot.PilotSafetyStop("the test scope has no stored telemetry: it needs the live provider")
    return scope


TRAIN_DEFAULTS = {"designs": "rosetta_spam-filter,finn_radioml", "briefs": "T0,A0,T,A,P0", "repeats": 2,
                  "shards": 12, "cap_runs": 60, "cap_job_hours": 240.0, "cap_tokens": 5_000_000, "cap_usd": 5.0,
                  "cap_extraction_jobs": 120}


def resolve_arguments(args: argparse.Namespace) -> None:
    """The test scope takes its whole configuration from the approved freeze; a flag cannot change it."""

    given = {k: getattr(args, k) for k in (*TRAIN_DEFAULTS, "approaches", "telemetry") if getattr(args, k) is not None}
    if args.scope == "test":
        if not args.freeze_approved:
            raise pilot.PilotSafetyStop("the test scope needs an approved freeze file; none was given")
        if given.keys() - {"shards"}:
            raise pilot.PilotSafetyStop("the test scope takes designs, briefs, approaches, repeats and caps only from "
                                        "the freeze; refused flags: " + ", ".join(sorted(given.keys() - {"shards"})))
        import test_campaign as tc
        frozen = tc.frozen_settings(Path(args.freeze_approved))
        args.designs, args.briefs, args.repeats = ",".join(tc.DESIGNS), ",".join(tc.BRIEFS), frozen["repeats"]
        args.cap_runs, args.cap_job_hours = frozen["caps"]["new_real_runs"], float(frozen["caps"]["job_hours"])
        args.cap_tokens, args.cap_usd = frozen["caps"]["tokens"], float(frozen["caps"]["usd"])
        args.cap_extraction_jobs = frozen["extraction_jobs"]
        args.telemetry = "live"
        args.shards = args.shards or 12
    else:
        for key, value in TRAIN_DEFAULTS.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        args.telemetry = args.telemetry or "saved"


def build_plan(args: argparse.Namespace, work: Path) -> dict[str, Any]:
    resolve_arguments(args)
    designs = tuple(args.designs.split(","))
    briefs = tuple(args.briefs.split(","))
    manifest = TRAIN_MANIFEST if args.scope == "train" else TEST_MANIFEST
    cells = h.make_cells(args.scope if args.scope == "test" else args.name, designs, briefs, args.repeats)
    if args.approaches:
        keep = set(args.approaches.split(","))
        cells = [c for c in cells if c.approach in keep]
    cfg = h.settings()
    model = yaml.safe_load((PROJECT / cfg["llm_yaml"]).read_text())["model"]
    return {
        "schema_version": "campaign-plan-v1", "name": args.name, "scope": args.scope, "designs": list(designs),
        "briefs": list(briefs), "repeats": args.repeats, "manifest_path": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "caps": {"new_real_runs": args.cap_runs, "job_hours": args.cap_job_hours, "tokens": args.cap_tokens,
                 "usd": args.cap_usd},
        "model": model, "price": cfg["price"], "telemetry": args.telemetry,
        "extraction_jobs": args.cap_extraction_jobs,
        "cells": [{"episode_id": c.episode_id, "design": c.design, "brief": c.brief, "approach": c.approach,
                   "tag": c.tag, "tier": c.tier} for c in cells],
        "episode_list_sha256": h.sha(json.dumps([c.episode_id for c in cells])),
        "config_pins": {k: v for k, v in h.imported_hashes().items() if k in h.CONFIG_PINS},
        "harness_sha256": _sha256_file(Path(h.__file__)), "runner_sha256": _sha256_file(SCRIPT),
        "shards": args.shards,
    }


def load_plan(name: str) -> dict[str, Any]:
    work = work_root(name)
    plan = _read(work / "plan.json")
    if plan["config_pins"] != {k: v for k, v in h.imported_hashes().items() if k in h.CONFIG_PINS}:
        raise pilot.PilotSafetyStop("configuration files differ from the frozen plan")
    if plan["harness_sha256"] != _sha256_file(Path(h.__file__)) or plan["runner_sha256"] != _sha256_file(SCRIPT):
        raise pilot.PilotSafetyStop("harness or runner differs from the frozen plan")
    return plan


def plan_sha(work: Path) -> str:
    return _sha256_file(work / "plan.json")


def cells_of(plan: dict[str, Any], attempt_of) -> list[h.Cell]:
    rows = []
    for c in plan["cells"]:
        n = attempt_of(c["episode_id"])
        rows.append(h.Cell(c["design"], c["brief"], c["approach"], c["tag"], c["tier"],
                           c["episode_id"] + (f"_a{n}" if n > 1 else ""), cache="fresh"))
    return rows


# ------------------------------------------------------------------------------ run bookkeeping
def run_numbers(work: Path) -> list[int]:
    return sorted(int(p.name) for p in (work / "runs").glob("[0-9]*")) if (work / "runs").is_dir() else []


def void_touched(work: Path) -> dict[str, int]:
    """Episodes touched by void runs -> number of void runs that touched them."""

    counts: dict[str, int] = {}
    for n in run_numbers(work):
        if (work / "runs" / str(n) / "VOID.json").is_file():
            touched = work / "runs" / str(n) / "touched.jsonl"
            if touched.is_file():
                for line in touched.read_text().splitlines():
                    base = json.loads(line)["base"]
                    counts[base] = counts.get(base, 0) + 1
    return counts


def current_run(work: Path) -> int:
    runs = run_numbers(work)
    return runs[-1] if runs else 0


# ------------------------------------------------------------------------------ shard
REQUEUE_DELAY_S = 120          # an infrastructure-failed cell is re-queued once, after its tier and this delay
LOG_FILES = ("capped_cells", "error_cells", "invalid_output_cells")


def infra_failures(work: Path) -> dict[str, list[dict[str, Any]]]:
    path = work / "infra_failures.jsonl"
    out: dict[str, list[dict[str, Any]]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                out.setdefault(e["episode_id"], []).append(e)
    return out


def record_infra(work: Path, cell: h.Cell, info: dict[str, Any], run_no: int) -> dict[str, Any]:
    """Count the failure atomically: the first re-queues the cell, the second drops it (never substituted)."""

    with _file_lock(work / "infra_failures.count.lock"):
        attempt = len(infra_failures(work).get(cell.episode_id, [])) + 1
        entry = {"episode_id": cell.episode_id, "design": cell.design, "brief": cell.brief, "approach": cell.approach,
                 "tag": cell.tag, "tier": cell.tier, "kind": info["kind"], "reason": info["reason"],
                 "attempt": attempt, "dropped": attempt >= 2, "run_no": run_no, "at_epoch": time.time(),
                 "at_utc": pilot.utc()}
        _append(work / "infra_failures.jsonl", entry)
    return entry


def terminal_ids(work: Path, run_no: int) -> set[str]:
    """Cells this run has capped, found invalid or errored: no shard tries them again in this run."""

    ids: set[str] = set()
    for name in LOG_FILES:
        path = work / f"{name}.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    if e.get("run_no") == run_no:
                        ids.add(e["episode_id"])
    return ids


def pending_cells(cells: list[h.Cell], finished: set[str], stuck: set[str],
                  fails: dict[str, list[dict[str, Any]]]) -> list[h.Cell]:
    """Cells still to run: not finished, not capped or errored in this run, and failed fewer than twice."""

    return [c for c in cells if c.episode_id not in finished and c.episode_id not in stuck
            and len(fails.get(c.episode_id, [])) < 2]


def eligible_now(pending: list[h.Cell], fails: dict[str, list[dict[str, Any]]], now: float) -> list[h.Cell]:
    """Plan order, except a cell with one infrastructure failure waits for every fresh cell of its tier (and below)
    and for the pause: it is re-queued once, at the end of its tier."""

    fresh_tiers = [c.tier for c in pending if not fails.get(c.episode_id)]
    out = []
    for cell in pending:
        if len(fails.get(cell.episode_id, [])) == 1:
            if fresh_tiers and min(fresh_tiers) <= cell.tier:
                continue
            if now < fails[cell.episode_id][-1]["at_epoch"] + REQUEUE_DELAY_S:
                continue
        out.append(cell)
    return out


def run_shard(name: str, shard: int, run_no: int) -> int:
    work = work_root(name)
    plan = load_plan(name)
    start_pins = _read(work / "runs" / str(run_no) / "pins_start.json")["files"]
    scope = scope_for(plan, work)
    caps = h.CapLedger(work, plan["caps"], plan_sha(work))
    changed: list[str] = []

    def check_pins() -> None:
        bad = h.pin_changes(start_pins)
        if bad:
            changed.extend(bad)
            raise pilot.PilotSafetyStop("imported project file changed during the run: " + ", ".join(bad[:5]))

    ctx = h.build_context(scope, mode="run", caps=caps, check_pins=check_pins)
    result_dir = work / "results"
    result_dir.mkdir(exist_ok=True)
    (work / "claims").mkdir(exist_ok=True)
    skipped: set[str] = set()
    counts = {"done": 0, "capped": 0, "infrastructure_failures": 0, "dropped": 0}

    def status_file(state: str, error: str | None = None) -> None:
        _write(work / "runs" / str(run_no) / f"shard_{shard}.json",
               {"shard": shard, "state": state, "counts": counts, "error": error, "at_utc": pilot.utc()})

    status_file("running")
    try:
        while True:
            attempts = void_touched(work)
            cells = cells_of(plan, lambda base: attempts.get(base, 0) + 1)
            fails = infra_failures(work)
            stuck = terminal_ids(work, run_no) | skipped
            finished = {c.episode_id for c in cells if (result_dir / f"{c.episode_id}.json").is_file()}
            pending = pending_cells(cells, finished, stuck, fails)
            if not pending:
                break
            progressed = False
            for cell in eligible_now(pending, fails, time.time()):
                seen = len(fails.get(cell.episode_id, []))
                lock_path = work / "claims" / f"{cell.episode_id}.lock"
                handle = open(lock_path, "a+")
                try:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    if (result_dir / f"{cell.episode_id}.json").is_file() or \
                            len(infra_failures(work).get(cell.episode_id, [])) != seen:
                        continue                   # another shard finished or failed it since the list was made
                    check_pins()
                    with (work / "runs" / str(run_no) / "touched.jsonl").open("a") as t:
                        base = next(c["episode_id"] for c in plan["cells"] if cell.episode_id.startswith(c["episode_id"]))
                        t.write(json.dumps({"base": base, "episode_id": cell.episode_id, "shard": shard}) + "\n")
                    try:
                        status, info = h.run_cell(ctx, cell, result_dir, run_no=run_no)
                    except pilot.PilotSafetyStop as exc:
                        message = str(exc)
                        if message.startswith(("AMD simulation did not PASS", "more DONT_TOUCH", "missing DONT_TOUCH")):
                            _append(work / "invalid_output_cells.jsonl", {"episode_id": cell.episode_id,
                                    "reason": message[:300], "run_no": run_no})
                            skipped.add(cell.episode_id)
                            progressed = True
                            continue
                        raise
                    except Exception as exc:  # one cell's failure is recorded and reported, never hidden
                        import traceback
                        _append(work / "error_cells.jsonl", {"episode_id": cell.episode_id,
                                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                "trace": traceback.format_exc()[-1500:], "at_utc": pilot.utc(), "run_no": run_no})
                        counts["errors"] = counts.get("errors", 0) + 1
                        skipped.add(cell.episode_id)
                        progressed = True
                        continue
                    if status == "capped":
                        counts["capped"] += 1
                        skipped.add(cell.episode_id)
                        _append(work / "capped_cells.jsonl", {**(info or {}), "at_utc": pilot.utc(), "run_no": run_no})
                    elif status == "infrastructure_failure":
                        entry = record_infra(work, cell, info or {}, run_no)
                        counts["infrastructure_failures"] += 1
                        counts["dropped"] += int(entry["dropped"])
                    else:
                        counts["done"] += 1
                        check_pins()
                    status_file("running")
                    progressed = True
                finally:
                    handle.close()
                if progressed:
                    break
            if not progressed:
                time.sleep(30)                     # cells held by another shard, or a re-queue waiting its turn
    except BaseException as exc:
        status_file("failed", f"{type(exc).__name__}: {exc}")
        raise
    status_file("complete")
    return 0


def _append(path: Path, value: dict[str, Any]) -> None:
    with _file_lock(path.with_suffix(path.suffix + ".lock")):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")


# ------------------------------------------------------------------------------ controller
def control(name: str, run_no: int) -> int:
    work = work_root(name)
    plan = load_plan(name)
    run_dir = work / "runs" / str(run_no)
    pins = h.imported_hashes()
    _write(run_dir / "pins_start.json", {"run_no": run_no, "at_utc": pilot.utc(), "files": pins})
    _write(run_dir / "control.json", {"state": "running", "pid": os.getpid(), "pgid": os.getpgrp(),
                                      "started_at_utc": pilot.utc()})
    shards = int(plan["shards"])
    attempts = {s: 1 for s in range(shards)}

    def spawn(s: int) -> subprocess.Popen:
        with (run_dir / f"shard_{s}.stdout").open("ab") as out, (run_dir / f"shard_{s}.stderr").open("ab") as err:
            return subprocess.Popen([sys.executable, str(SCRIPT), "--name", name, "--shard", str(s), "--run-no", str(run_no)],
                                    cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err, close_fds=True)

    live = {s: spawn(s) for s in range(shards)}
    exit_codes: dict[int, int] = {}
    while live:
        time.sleep(10)
        for s, proc in list(live.items()):
            code = proc.poll()
            if code is None:
                continue
            del live[s]
            if code != 0 and attempts[s] < 3:
                attempts[s] += 1
                time.sleep(20)
                live[s] = spawn(s)
            else:
                exit_codes[s] = code
    end = h.pin_changes(pins)
    state = "complete" if all(c == 0 for c in exit_codes.values()) else "complete_with_failures"
    _write(run_dir / "pins_end.json", {"run_no": run_no, "at_utc": pilot.utc(), "changed": end,
                                       "files": h.imported_hashes()})
    if end:
        _write(run_dir / "VOID.json", {"reason": "an imported project file changed during the run", "changed": end})
        state = "void"
    _write(run_dir / "control.json", {"state": state, "pid": os.getpid(), "pgid": os.getpgrp(),
                                      "exit_codes": {str(k): v for k, v in exit_codes.items()}, "at_utc": pilot.utc()})
    return 0


def launch(name: str, *, resume: bool, freeze_ok: bool, freeze_path: str | None = None) -> dict[str, Any]:
    work = work_root(name)
    plan = load_plan(name)
    if plan["scope"] == "test":
        if not freeze_ok or not freeze_path:
            raise pilot.PilotSafetyStop("the test scope needs an approved freeze check; none was given")
        import test_campaign as tc
        tc.verify_plan_against_freeze(Path(freeze_path), plan)
    if not os.environ.get("GOOGLE_API_KEY"):
        raise pilot.PilotSafetyStop("GOOGLE_API_KEY is not set in this shell")
    previous = current_run(work)
    if previous:
        ctl = _read(work / "runs" / str(previous) / "control.json") if (work / "runs" / str(previous) / "control.json").is_file() else {}
        if ctl.get("state") == "running" and _alive(int(ctl.get("pid", 0))):
            raise pilot.PilotSafetyStop("the previous run is still alive")
        if not resume:
            raise FileExistsError("this run was launched before; use --resume")
    run_no = previous + 1
    run_dir = work / "runs" / str(run_no)
    run_dir.mkdir(parents=True)
    with (run_dir / "controller.stdout").open("wb") as out, (run_dir / "controller.stderr").open("wb") as err:
        proc = subprocess.Popen([sys.executable, str(SCRIPT), "--name", name, "--control", "--run-no", str(run_no)],
                                cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True, close_fds=True)
    return {"run_no": run_no, "controller_pid": proc.pid, "pgid": proc.pid, "started_at_utc": pilot.utc()}


def _alive(pid: int) -> bool:
    info = pilot._stat(pid) if pid else None
    return info is not None and info.state != "Z"


def stop(name: str) -> dict[str, Any]:
    work = work_root(name)
    run_no = current_run(work)
    ctl = _read(work / "runs" / str(run_no) / "control.json")
    pgid = int(ctl["pgid"])
    cmdline = Path(f"/proc/{ctl['pid']}/cmdline").read_bytes().split(b"\0") if _alive(int(ctl["pid"])) else []
    if not any(str(SCRIPT).encode() == a for a in cmdline) or pgid != int(ctl["pid"]):
        raise pilot.PilotSafetyStop("the recorded controller is not alive as the group leader of this run")
    os.killpg(pgid, signal.SIGTERM)
    _write(work / "runs" / str(run_no) / "control.json", {**ctl, "state": "stopped_by_operator", "stopped_at_utc": pilot.utc()})
    return {"stopped_pgid": pgid, "run_no": run_no}


def cleanup(name: str) -> dict[str, Any]:
    """Kill leftover tool jobs of this run by exact job tag (the development pattern)."""

    work = work_root(name)
    plan = load_plan(name)
    scope = scope_for(plan, work)
    mine = {r["run_id"] for r in _read(work / "caps.json")["reservations"]}
    prefix = str(scope.engine_root / "runs").encode() + b"/"
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
        if run_id in mine and tag:
            tags.add(tag)
            groups.add(info.pgrp)
    groups.discard(os.getpgrp())
    result: dict[str, Any] = {"exact_tags": sorted(tags), "groups": sorted(groups), "started_at_utc": pilot.utc()}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in sorted(groups):
            if any(pilot._tag_of(m.pid) in tags for m in pilot.group_members(group)):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    result["survivors"] = {t: [p.pid for p in pilot.tagged(t)] for t in sorted(tags) if pilot.tagged(t)}
    # AMD validator jobs run under <engine>/verification/runs; they carry a job tag too.
    needle = str(scope.engine_root / "verification" / "runs").encode() + b"/verify_"
    verify_tags: set[str] = set()
    for pid in pilot._own_pids():
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        tag = pilot._tag_of(pid) if needle in args else None
        if tag:
            verify_tags.add(tag)
    # worker B extraction jobs run under <work>/telemetry_cache/<xx>/<state>/attempts/<id>
    cache_needle = str(work / "telemetry_cache").encode()
    for pid in pilot._own_pids():
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd").encode()
            args_b = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if cache_needle in cwd or cache_needle in args_b:
            tag = pilot._tag_of(pid)
            if tag:
                verify_tags.add(tag)
    result["verification_tags"] = sorted(verify_tags)
    result["verification_survivors"] = {t: list(pilot.sweep_tag(t, grace_s=5).alive) for t in sorted(verify_tags)}
    _write(work / "cleanup_audit.json", result)
    return result


def status(name: str) -> dict[str, Any]:
    work = work_root(name)
    plan = _read(work / "plan.json")
    results = [_read(p) for p in (work / "results").glob("*.json")] if (work / "results").is_dir() else []
    caps = _read(work / "caps.json") if (work / "caps.json").is_file() else None
    runs = {}
    for n in run_numbers(work):
        ctl = work / "runs" / str(n) / "control.json"
        runs[n] = {"control": _read(ctl) if ctl.is_file() else None,
                   "void": (work / "runs" / str(n) / "VOID.json").is_file()}
    return {"name": name, "scope": plan["scope"], "planned": len(plan["cells"]), "completed": len(results),
            "used": caps["used"] if caps else None, "caps": plan["caps"], "stopped": caps["stopped"] if caps else None,
            "runs": runs, "capped_cells": sum(1 for _ in open(work / "capped_cells.jsonl")) if (work / "capped_cells.jsonl").is_file() else 0,
            "infrastructure_failures": {"episodes": len(infra_failures(work)),
                                        "dropped": sum(1 for v in infra_failures(work).values() if len(v) >= 2)},
            "extraction_states": len(_read(work / "extraction_accounted.json")) if (work / "extraction_accounted.json").is_file() else 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    for flag in ("prepare", "launch", "resume", "stop", "cleanup", "status", "control"):
        modes.add_argument(f"--{flag}", action="store_true")
    modes.add_argument("--shard", type=int)
    parser.add_argument("--run-no", type=int, default=0)
    parser.add_argument("--scope", choices=("train", "test"), default="train")
    parser.add_argument("--designs", default=None)
    parser.add_argument("--briefs", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--approaches", default=None, help="comma list; default every approach (train scope only)")
    parser.add_argument("--shards", type=int, default=None)
    parser.add_argument("--cap-runs", type=int, default=None)
    parser.add_argument("--cap-job-hours", type=float, default=None)
    parser.add_argument("--cap-tokens", type=int, default=None)
    parser.add_argument("--cap-usd", type=float, default=None)
    parser.add_argument("--cap-extraction-jobs", type=int, default=None)
    parser.add_argument("--telemetry", choices=("saved", "live"), default=None,
                        help="worker B telemetry: stored (training only) or the live extraction provider")
    parser.add_argument("--freeze-approved", default=None)
    args = parser.parse_args()
    work = work_root(args.name)
    if args.prepare:
        if (work / "plan.json").exists():
            raise FileExistsError("this run already has a plan")
        work.mkdir(parents=True, exist_ok=True)
        plan = build_plan(args, work)
        _write(work / "plan.json", plan)
        print(json.dumps({"plan_sha256": plan_sha(work), "cells": len(plan["cells"]),
                          "episode_list_sha256": plan["episode_list_sha256"]}))
    elif args.launch or args.resume:
        print(json.dumps(launch(args.name, resume=args.resume, freeze_ok=bool(args.freeze_approved),
                                freeze_path=args.freeze_approved)))
    elif args.control:
        return control(args.name, args.run_no)
    elif args.shard is not None:
        return run_shard(args.name, args.shard, args.run_no)
    elif args.stop:
        print(json.dumps(stop(args.name)))
    elif args.cleanup:
        print(json.dumps(cleanup(args.name)))
    else:
        print(json.dumps(status(args.name), indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
