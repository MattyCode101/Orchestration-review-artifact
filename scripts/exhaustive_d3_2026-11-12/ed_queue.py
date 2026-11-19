"""Detached queue for the exhaustive depth-3 reference (timing and area) on the five TEST designs.

Runs through the pinned runtime exactly like scripts/power_mediation_2026-11-10/pm_runner.py: orch.runtime.make_real_runtime
(StepEngine, cache, meter, tagged Runner on the shared flock slots) and VerificationLedger (AMD validator, reference = B2).
Two scheduling-only changes, both in a subclass of the pinned Runner (no frozen file is edited): the free-slot scan runs from the
top slot downwards (so users with a smaller max_slots keep the low slots), and the per-job RSS sampling, which scans /proc for
every process of the user, is throttled to once per 30 s per job (peak memory still comes from GNU time -v).

Work files (data/exhaustive_d3_2026-11-12/plan/): tasks_phase0.json (AMD checks against B2), tasks_phase1.json (depth 3, timing and
area), tasks_phase2.json (optional, read when phase 1 is complete). Order is the file order: phase 0, then phase 1 design by design.
Journal (append-only): journal.jsonl; a task with a terminal row is never run again. Stop: STOP file or SIGTERM (drain), second
SIGTERM exits at once. Admission is re-read from queue_config.json every loop.

usage: ed_queue.py --run | --smoke | --fake
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import fcntl
import json
import os
import random
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ed_common as ec  # noqa: E402

DEFAULT_CONFIG = {
    "max_active": 60,            # most jobs of this queue in flight (recipe runs and AMD checks)
    "slot_ceiling": 80,          # total busy slots (all users) this queue aims not to exceed: 192 logical CPUs x 0.85 / 2.1 per job
    "reserve_for_others": 16,    # slots always left available to the other users
    "load_guard": 160.0,         # no new job while the 1-minute load average is above this (192 logical CPUs, 96 physical cores)
    "runnable_guard": 185,       # nor while this many threads are runnable right now
    "admit_per_loop": 4,         # ramp: at most this many new jobs per 2-second loop
    "min_free_disk_gb": 1500,
    "status_period_s": 60,
    "history_period_s": 900,
    "max_runner_slots": 96,      # Runner(max_slots=...): the pinned hard maximum (orch/engine/runner.py)
    "retry_limit": 2,
    "rss_sample_period_s": 30.0,
    "eta_report_after_s": 3600,
    "final_report_utc": "2026-11-13T09:00:00+00:00",
}
RETRYABLE = {"timeout", "transient", "meter_error"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if ec.CONFIG.is_file():
        try:
            cfg.update(ec.jread(ec.CONFIG))
        except Exception:
            pass
    return cfg


def task_key(t: dict) -> str:
    if t["kind"].startswith("amd_cell"):                  # amd_cell (phase 0) and amd_cell_d3 (phase 1b) have separate keys
        return f"{t['kind']}|{t['design']}|{t['brief']}"
    return "|".join([t["kind"], t["design"], t["state"], t.get("recipe") or ""])


def read_journals() -> dict[str, dict]:
    """Latest terminal row per task key over all journals (smoke included)."""

    rows: dict[str, dict] = {}
    for path in (ec.SMOKE_JOURNAL, ec.JOURNAL):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue            # a torn last line after a hard kill: ignored, the task simply runs again
            if row.get("status") in ("ok", "failed_terminal", "error_exhausted"):
                rows[row["task"]] = row
    return rows


def proc_loadavg() -> tuple[float, int]:
    parts = Path("/proc/loadavg").read_text().split()
    return float(parts[0]), int(parts[3].split("/")[0])


def busy_slots(slot_dir: Path, limit: int = 96) -> int:
    n = 0
    for i in range(limit):
        p = slot_dir / f"slot{i}"
        if not p.is_file():
            continue
        with p.open("a+") as h:
            try:
                fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                n += 1
            else:
                fcntl.flock(h, fcntl.LOCK_UN)
    return n


def make_runner_class():
    from orch.engine.runner import Runner

    class ThrottledRunner(Runner):
        """The pinned Runner; only the slot scan order and the RSS sampling cadence differ."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.held = 0
            self._held_lock = threading.Lock()
            self._rss: dict[int, tuple[float, int]] = {}
            self.rss_period = DEFAULT_CONFIG["rss_sample_period_s"]

        @contextlib.contextmanager
        def _slot(self):
            self.slot_dir.mkdir(parents=True, exist_ok=True)
            while True:
                for index in range(self.max_slots - 1, -1, -1):
                    handle = (self.slot_dir / f"slot{index}").open("a+")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        continue
                    with self._held_lock:
                        self.held += 1
                    try:
                        yield index
                    finally:
                        with self._held_lock:
                            self.held -= 1
                        fcntl.flock(handle, fcntl.LOCK_UN)
                        handle.close()
                    return
                time.sleep(self.slot_poll_s)

        def _sample_rss(self, pgrp, tag):  # type: ignore[override]
            now = time.monotonic()
            last = self._rss.get(pgrp)
            if last is not None and now - last[0] < self.rss_period:
                return last[1]
            value = Runner._sample_rss(pgrp, tag)
            if len(self._rss) > 2000:
                self._rss.clear()
            self._rss[pgrp] = (now, value)
            return value

    return ThrottledRunner


class Queue:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.cfg = load_config()
        self.t_start = time.time()
        self.t_start_utc = utc()
        self.lock = threading.Lock()
        self.stop_requested = False
        self.hard_stop = False
        self.journal_path = ec.SMOKE_JOURNAL if mode == "smoke" else ec.JOURNAL
        self.completed: collections.deque = collections.deque(maxlen=4000)   # (time, est_wall_s, wall_s)
        self.imported: dict[str, threading.Lock] = {}
        self.imp_lock = threading.Lock()
        self.fake = mode == "fake"
        self.done_rows: dict[str, dict] = {}
        self.limit = 0
        self.runtime = None
        self.ledger = None
        self.runner = None
        self.last_probe = (0.0, 0)
        self.last_status = 0.0
        self.last_history = 0.0
        self.eta_report_done = False
        self.final_report_done = False
        if not self.fake:
            self._build_runtime()

    # ---------------------------------------------------------------- runtime
    def _build_runtime(self) -> None:
        import orch.runtime as rt
        from scripts import thursday_night_pilot as pilot
        from orch.baselines.verification import VerificationLedger

        rt.Runner = make_runner_class()
        self.pilot = pilot
        self.runtime = rt.make_real_runtime(
            ec.ENGINE, cache_enabled=True, max_slots=self.cfg["max_runner_slots"],
            load_ceiling_1m=pilot.LOAD_CEILING, min_available_memory_bytes=pilot.MEMORY_FLOOR,
            latch_smoke_status=pilot.SMOKE,
        )
        self.runner = self.runtime.runner
        self.runner.rss_period = float(self.cfg["rss_sample_period_s"])
        if self.runtime.meter.version != ec.CUR_METER:
            raise SystemExit(f"meter version differs from the stored runs: {self.runtime.meter.version}")
        for d, sid in ec.B2.items():
            src = ec.SRC_ROOTS["test_base"] / "states" / sid / "design.dcp"
            st = self.runtime.store.import_state(src, design=d, parent_id=None, made_by_run_id=None, stage="routed", fake=False)
            if st.id != sid:
                raise SystemExit(f"B2 import differs: {d}")
        self.ledger = VerificationLedger(
            store=self.runtime.store, runner=self.runner, validator=self.runtime.validator, references=dict(ec.B2),
        )

    def import_state(self, design: str, state_id: str, src: str, parent_id, made_by):
        with self.imp_lock:
            lk = self.imported.setdefault(state_id, threading.Lock())
        if str(src).startswith(str(ec.ENGINE)):                # a state this engine already holds (phase 1b): load it, do not copy it
            from orch.contract.models import DesignState
            return DesignState.model_validate_json((self.runtime.store.states_dir / state_id / "state.json").read_bytes())
        with lk:
            st = self.runtime.store.import_state(Path(src), design=design, parent_id=parent_id, made_by_run_id=made_by,
                                                 stage="routed", fake=False)
        if st.id != state_id:
            raise ValueError(f"imported state differs: {state_id}")
        return st

    # ---------------------------------------------------------------- execution
    def row_base(self, t: dict) -> dict:
        return {"task": task_key(t), "phase": t["phase"], "kind": t["kind"], "design": t["design"],
                "state": t.get("state"), "recipe": t.get("recipe"), "est_wall_s": t.get("est_wall_s"), "started_at_utc": utc()}

    def run_recipe(self, t: dict) -> dict:
        from orch.contract.models import MetricName, RunStatus

        row = self.row_base(t)
        if self.fake:
            time.sleep(random.uniform(0.5, 2.5))
            row.update(status="ok", run_id="fake", run_status="ok", cache_hit=False, wall_s=random.uniform(300, 900),
                       output_state="f" * 64, metrics={"wns_ns": 0.0, "packed_luts": 1.0})
            return row
        if t.get("after"):
            from orch.contract.models import DesignState
            sid = self.done_rows[t["after"]]["output_state"]
            state = DesignState.model_validate_json((self.runtime.store.states_dir / sid / "state.json").read_bytes())
            row["input_state"] = sid
        else:
            state = self.import_state(t["design"], t["state"], t["src"], t.get("parent_id"), t.get("made_by_run_id"))
        recipe = self.registry_get(t["recipe"])
        attempts = 0
        while True:
            attempts += 1
            outcome = self.runtime.engine.execute(state, recipe)
            run = outcome.run
            row.update(run_id=run.id, run_status=run.status.value, cache_hit=bool(run.cache_hit), attempt=attempts,
                       wall_s=float(run.wall_s or 0.0), cpu_user_s=run.cpu_user_s, cpu_sys_s=run.cpu_sys_s,
                       peak_memory_bytes=run.peak_memory_bytes, load_1m_start=run.load_1m_start,
                       load_1m_end=run.load_1m_end, busy_slots_start=run.busy_slots_start,
                       original_run_id=getattr(run, "original_run_id", None))
            if run.status == RunStatus.OK and outcome.state is not None and outcome.evidence is not None:
                row["output_state"] = outcome.state.id
                row["metrics"] = {n.value: m.value for n, m in outcome.evidence.metrics.items()}
                row["meter_version"] = outcome.evidence.meter_version
                row["routed"] = outcome.evidence.routed
                row["route_errors"] = outcome.evidence.route_errors
                row["drc_errors"] = outcome.evidence.drc_errors
                row["status"] = "ok"
                return row
            if run.status.value in RETRYABLE and attempts <= self.cfg["retry_limit"]:
                time.sleep(30 * attempts)
                continue
            row["status"] = "failed_terminal"
            row["failure_reason"] = run.failure_reason
            return row

    def registry_get(self, recipe_id: str):
        return self.runtime.registry.get(recipe_id)

    def run_amd_cell(self, t: dict) -> dict:
        row = self.row_base(t)
        row["brief"] = t["brief"]
        checked = []
        winner = None
        for cand in t["candidates"]:
            if self.fake:
                time.sleep(0.5)
                checked.append({"state": cand["state"], "result": "pass", "wall_s": 1.0})
                winner = cand["state"]
                break
            self.import_state(t["design"], cand["state"], cand["src"], cand.get("parent_id"), cand.get("made_by_run_id"))
            rec = self.ledger.verify(t["design"], cand["state"])
            wall = sum(a["wall_s"] for a in rec["attempts"])
            compared = None
            if rec["result"] == "pass":
                try:
                    compared = self.pilot.outputs_compared(rec, self.ledger.root / "runs")
                except Exception as exc:  # recorded, not fatal
                    compared = f"unreadable: {exc}"
            checked.append({"state": cand["state"], "path": cand.get("path"), "result": rec["result"], "wall_s": round(wall, 1),
                            "outputs_compared": compared, "record_fingerprint": rec.get("validator_fingerprint")})
            if rec["result"] == "pass":
                winner = cand["state"]
                break
        row.update(checked=checked, best_pass_vs_b2=winner, status="ok",
                   wall_s=sum(c["wall_s"] for c in checked))
        return row

    def one(self, t: dict) -> dict:
        attempts = 0
        while True:
            attempts += 1
            try:
                row = self.run_amd_cell(t) if t["kind"].startswith("amd_cell") else self.run_recipe(t)
                row["finished_at_utc"] = utc()
                return row
            except Exception as exc:     # infrastructure error: retried, then listed, never raised
                if attempts <= self.cfg["retry_limit"] and not self.stop_requested:
                    time.sleep(60 * attempts)
                    continue
                row = self.row_base(t)
                row.update(status="error_exhausted", error=f"{type(exc).__name__}: {exc}", attempt=attempts, finished_at_utc=utc())
                return row

    def commit(self, row: dict) -> None:
        with self.lock:
            with self.journal_path.open("a", encoding="utf-8") as h:
                h.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                h.flush()
                os.fsync(h.fileno())
            if row.get("status") in ("ok", "failed_terminal", "error_exhausted"):
                self.done_rows[row["task"]] = row
            if row.get("status") in ("ok", "failed_terminal") and not str(row.get("kind")).startswith("amd_cell"):
                self.completed.append((time.time(), float(row.get("est_wall_s") or 0.0), float(row.get("wall_s") or 0.0),
                                       bool(row.get("cache_hit"))))

    # ---------------------------------------------------------------- admission
    def probe(self) -> int:
        now = time.monotonic()
        if now - self.last_probe[0] > 3.0:
            self.last_probe = (now, busy_slots(ec.PROJECT / "data" / "slots"))
        return self.last_probe[1]

    def admit_ok(self, n_active: int) -> tuple[bool, str]:
        c = self.cfg
        if n_active >= c["max_active"]:
            return False, "max_active"
        load1, runnable = proc_loadavg()
        if load1 > c["load_guard"]:
            return False, f"load {load1:.0f}"
        if runnable > c["runnable_guard"]:
            return False, f"runnable {runnable}"
        if self.fake:
            return True, ""
        busy = self.probe()
        held = self.runner.held
        others = max(0, busy - held)
        # my jobs in flight may not hold a slot yet: count them all against the ceiling
        mine = max(n_active, held)
        ceiling = c["slot_ceiling"] - max(0, c["reserve_for_others"] - others)
        if others + mine + 1 > ceiling:
            return False, f"slots others={others} mine={mine} ceiling={ceiling}"
        if shutil.disk_usage(ec.WORK).free < c["min_free_disk_gb"] * 2**30:
            return False, "disk"
        return True, ""

    # ---------------------------------------------------------------- status
    def phase_files(self) -> list[Path]:
        return [ec.PLAN / f"tasks_phase{i}.json" for i in ("0", "1", "1b", "2")]

    def all_tasks(self) -> list[dict]:
        tasks = []
        for p in self.phase_files():
            if p.is_file():
                tasks.extend(ec.jread(p))
        return tasks

    def status_payload(self, tasks: list[dict], active: dict, reason: str) -> dict:
        rows = read_journals()
        per = collections.OrderedDict()
        est_left = 0.0
        for t in tasks:
            k = task_key(t)
            key = f"phase{t['phase']}:{t['design']}"
            d = per.setdefault(key, {"total": 0, "done": 0, "failed": 0, "error": 0, "est_left_s": 0.0})
            d["total"] += 1
            r = rows.get(k)
            if r and r["status"] == "ok":
                d["done"] += 1
            elif r and r["status"] == "failed_terminal":
                d["done"] += 1
                d["failed"] += 1
            elif r:
                d["error"] += 1
                d["est_left_s"] += float(t.get("est_wall_s") or 0.0)
            else:
                d["est_left_s"] += float(t.get("est_wall_s") or 0.0)
        for d in per.values():
            est_left += d["est_left_s"]
        now = time.time()
        win = min(3600.0, max(60.0, now - self.t_start))
        recent = [(est, w) for (ts, est, w, hit) in self.completed if now - ts <= win and not hit]
        thr_est = sum(e for e, _ in recent) / win if recent else 0.0              # estimated job-seconds finished per second
        slow = [w / e for e, w in recent[-40:] if e > 0 and w > 0]
        eta_s = est_left / thr_est if thr_est > 0 else None
        load1, runnable = proc_loadavg()
        payload = {
            "updated_at_utc": utc(), "queue_started_at_utc": self.t_start_utc, "mode": self.mode, "pid": os.getpid(),
            "in_flight": len(active), "runner_slots_held": getattr(self.runner, "held", None),
            "slots_busy_all_users": self.last_probe[1], "load_1m": load1, "runnable_threads": runnable,
            "admission": reason, "config": self.cfg, "by_phase_design": per,
            "done": sum(d["done"] for d in per.values()), "total": sum(d["total"] for d in per.values()),
            "failed_terminal": sum(d["failed"] for d in per.values()), "errors": sum(d["error"] for d in per.values()),
            "throughput_window_s": win, "throughput_est_jobseconds_per_s": round(thr_est, 3),
            "mean_slowdown_vs_stored_last40": round(sum(slow) / len(slow), 3) if slow else None,
            "est_job_hours_left": round(est_left / 3600, 1), "eta_hours_at_measured_throughput": None if eta_s is None else round(eta_s / 3600, 2),
            "eta_utc": None if eta_s is None else (datetime.now(timezone.utc) + timedelta(seconds=eta_s)).isoformat(timespec="minutes"),
        }
        return payload

    def design_etas(self, payload: dict, tasks: list[dict]) -> list[str]:
        """Per-design finishing time at the measured throughput, designs in queue order (phase 1 only)."""

        thr = payload["throughput_est_jobseconds_per_s"]
        lines = []
        if not thr:
            return ["no throughput measured yet"]
        cum = 0.0
        for key, d in payload["by_phase_design"].items():
            if not key.startswith("phase1:"):
                continue
            cum += d["est_left_s"]
            if d["est_left_s"] == 0:
                lines.append(f"{key.split(':')[1]}: done" if not d["error"] else f"{key.split(':')[1]}: INCOMPLETE ({d['error']} errors)")
                continue
            eta = datetime.now(timezone.utc) + timedelta(seconds=cum / thr)
            lines.append(f"{key.split(':')[1]}: {d['done']}/{d['total']} done, ETA {eta.strftime('%Y-%m-%d %H:%M')} UTC")
        return lines

    def append_progress(self, title: str, body: str) -> None:
        text = f"\n\n### {utc().replace('T', ' ').replace('+00:00', '')} UTC\n\n## {title}\n{body}\n"
        data = text.encode("utf-8")
        fd = os.open(str(ec.PROJECT / "notes" / "progress.md"), os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)

    def maybe_status(self, tasks, active, reason) -> None:
        now = time.time()
        if now - self.last_status >= self.cfg["status_period_s"]:
            self.last_status = now
            payload = self.status_payload(tasks, active, reason)
            ec.jwrite(ec.STATUS, payload)
            if now - self.last_history >= self.cfg["history_period_s"]:
                self.last_history = now
                with ec.STATUS_HISTORY.open("a", encoding="utf-8") as h:
                    slim = {k: payload[k] for k in ("updated_at_utc", "in_flight", "slots_busy_all_users", "load_1m", "done", "total",
                                                    "failed_terminal", "errors", "throughput_est_jobseconds_per_s",
                                                    "mean_slowdown_vs_stored_last40", "est_job_hours_left", "eta_utc")}
                    h.write(json.dumps(slim) + "\n")
            if self.mode == "run":
                if not self.eta_report_done and now - self.t_start >= self.cfg["eta_report_after_s"]:
                    self.eta_report_done = True
                    self.eta_report(payload, tasks)
                if not self.final_report_done and datetime.now(timezone.utc) >= datetime.fromisoformat(self.cfg["final_report_utc"]):
                    self.final_report_done = True
                    self.final_report(payload, tasks, "09:00 UK checkpoint (queue keeps running)")

    def eta_report(self, payload: dict, tasks: list[dict]) -> None:
        lines = self.design_etas(payload, tasks)
        deadline = datetime.fromisoformat(self.cfg["final_report_utc"])
        late = []
        thr = payload["throughput_est_jobseconds_per_s"]
        cum = 0.0
        for key, d in payload["by_phase_design"].items():
            if key.startswith("phase1:") and thr:
                cum += d["est_left_s"]
                if d["est_left_s"] and datetime.now(timezone.utc) + timedelta(seconds=cum / thr) > deadline:
                    late.append(key.split(":")[1])
        body = (f"- Queue process {os.getpid()}, started {self.t_start_utc}; first-hour check. Done {payload['done']} of {payload['total']} tasks "
                f"({payload['failed_terminal']} terminal failures, {payload['errors']} infrastructure errors); in flight {payload['in_flight']}; "
                f"slots busy on the server {payload['slots_busy_all_users']}; 1-minute load {payload['load_1m']:.0f}.\n"
                f"- Measured throughput {payload['throughput_est_jobseconds_per_s']} stored-cost job-seconds per second (window {payload['throughput_window_s']:.0f} s); "
                f"mean wall against the stored mean for the same recipe and design: {payload['mean_slowdown_vs_stored_last40']}. "
                f"Estimated job-hours left {payload['est_job_hours_left']}; ETA for everything {payload['eta_utc']} UTC.\n"
                + "".join(f"  - {l}\n" for l in lines)
                + (f"- Designs that will not finish by 09:00 UK at this rate: {', '.join(late)}. Design order is kept; the queue keeps running into Friday unless stopped.\n"
                   if late else "- At this rate every design finishes before 09:00 UK.\n"))
        self.append_progress("Exhaustive depth-3 queue: ETA after the first hour (written by the queue process)", body)

    def final_report(self, payload: dict, tasks: list[dict], why: str) -> None:
        lines = self.design_etas(payload, tasks)
        per = payload["by_phase_design"]
        body = f"- {why}. Done {payload['done']} of {payload['total']} tasks; terminal failures {payload['failed_terminal']}; infrastructure errors {payload['errors']}.\n"
        for key, d in per.items():
            body += f"  - {key}: {d['done']}/{d['total']} done, {d['failed']} failed, {d['error']} errors\n"
        body += "- Remaining designs (ETA at measured throughput):\n" + "".join(f"  - {l}\n" for l in lines)
        if payload["errors"]:
            body += f"- {payload['errors']} tasks ended in an infrastructure error and are not counted as done; ed_ctl.py resume runs them again.\n"
        self.append_progress("Exhaustive depth-3 queue: status (written by the queue process)", body)

    # ---------------------------------------------------------------- main loop
    def on_signal(self, signum, _frame) -> None:
        if self.stop_requested:
            self.hard_stop = True
            os._exit(1)
        self.stop_requested = True

    def run(self, only_keys: set[str] | None = None) -> int:
        signal.signal(signal.SIGTERM, self.on_signal)
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        loaded ={str(p) for p in self.phase_files() if p.is_file()}
        tasks = self.all_tasks()
        if getattr(self, "limit", 0):
            tasks = tasks[: self.limit]
        if only_keys is not None:
            tasks = [t for t in tasks if task_key(t) in only_keys]
        rows = read_journals()
        # an infrastructure error (error_exhausted) is not a result: such a task runs again on resume
        self.done_rows = {k: r for k, r in rows.items() if r.get("status") in ("ok", "failed_terminal")}
        pending = collections.deque(t for t in tasks if task_key(t) not in self.done_rows)
        print(f"{utc()} queue {self.mode}: {len(tasks)} tasks, {len(tasks) - len(pending)} already finished, {len(pending)} pending", flush=True)
        active: dict = {}
        reason = ""
        with ThreadPoolExecutor(max_workers=max(8, int(self.cfg["max_active"]) + 8)) as pool:
            while (pending or active) and not self.hard_stop:
                self.cfg = load_config()
                if self.runner is not None:
                    self.runner.rss_period = float(self.cfg["rss_sample_period_s"])
                if ec.STOPFILE.exists():
                    self.stop_requested = True
                if active:
                    done, _ = wait(tuple(active), timeout=2.0, return_when=FIRST_COMPLETED)
                    for fut in done:
                        t = active.pop(fut)
                        try:
                            row = fut.result()
                        except Exception as exc:                       # should not happen: one() never raises
                            row = self.row_base(t)
                            row.update(status="error_exhausted", error=f"{type(exc).__name__}: {exc}", finished_at_utc=utc())
                        self.commit(row)
                else:
                    time.sleep(2.0)
                if not pending and not self.stop_requested:        # a later phase file written while this queue runs
                    for p in self.phase_files():
                        if p.is_file() and str(p) not in loaded:
                            loaded.add(str(p))
                            more = ec.jread(p)
                            tasks.extend(more)
                            pending.extend(t for t in more if task_key(t) not in self.done_rows)
                            print(f"{utc()} loaded {p.name}: {len(more)} tasks", flush=True)
                admitted = 0
                rotations = 0
                while pending and not self.stop_requested and admitted < self.cfg["admit_per_loop"]:
                    t = pending[0]
                    if t.get("phase") == 2 and any(tt.get("phase") in (0, 1) for tt in active.values()):
                        break                                         # phase 2 only once phase 0 and 1 have nothing left in flight
                    after = t.get("after")
                    if after:
                        prow = self.done_rows.get(after)
                        if prow is None:                           # parent not finished yet: look at the next task
                            pending.rotate(-1)
                            rotations += 1
                            if rotations >= len(pending):
                                break
                            continue
                        if prow.get("status") == "error_exhausted":
                            pending.popleft()                          # parent hit an infrastructure error: no journal row, resume retries both
                            continue
                        if prow.get("status") != "ok" or not prow.get("output_state"):
                            pending.popleft()
                            row = self.row_base(t)
                            row.update(status="failed_terminal", run_status="parent_failed",
                                       failure_reason=f"parent {after} produced no state", finished_at_utc=utc())
                            self.commit(row)
                            continue
                    ok, reason = self.admit_ok(len(active))
                    if not ok:
                        break
                    pending.popleft()
                    active[pool.submit(self.one, t)] = t
                    admitted += 1
                    reason = "admitting"
                if self.stop_requested and not active:
                    break
                self.maybe_status(tasks, active, reason if not self.stop_requested else "stop requested: draining")
        payload = self.status_payload(tasks, active, "finished" if not pending else "stopped")
        ec.jwrite(ec.STATUS, payload)
        if self.mode == "run":
            self.final_report(payload, tasks, "Queue ended" if not pending else "Queue stopped on request")
        print(f"{utc()} queue {self.mode} exiting: pending {len(pending)}", flush=True)
        return 0


def write_pidfile() -> None:
    ec.jwrite(ec.PIDFILE, {"pid": os.getpid(), "pgid": os.getpgrp(), "sid": os.getsid(0), "argv": sys.argv,
                           "started_at_utc": utc(), "engine_root": str(ec.ENGINE)})


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--fake", action="store_true")
    ap.add_argument("--smoke-tasks", type=Path, default=ec.PLAN / "tasks_smoke.json")
    ap.add_argument("--limit", type=int, default=0, help="fake mode only: use the first N tasks")
    ap.add_argument("--tasks-file", type=Path, default=None, help="fake mode only: task list to use instead of the plan files")
    args = ap.parse_args()
    ec.LOGS.mkdir(parents=True, exist_ok=True)
    if args.run:
        if ec.PIDFILE.is_file():
            old = ec.jread(ec.PIDFILE)
            try:
                cmd = Path(f"/proc/{old['pid']}/cmdline").read_bytes().decode(errors="replace")
            except OSError:
                cmd = ""
            if "ed_queue.py" in cmd:
                print(f"a queue is already running: pid {old['pid']}")
                return 1
        write_pidfile()
    if args.fake:
        # mechanics only: no tool, no engine, fake journal in a scratch folder
        ec.JOURNAL = ec.WORK / "fake" / "journal.jsonl"
        ec.SMOKE_JOURNAL = ec.WORK / "fake" / "journal_smoke.jsonl"
        ec.STATUS = ec.WORK / "fake" / "status.json"
        ec.STATUS_HISTORY = ec.WORK / "fake" / "status_history.jsonl"
        (ec.WORK / "fake").mkdir(parents=True, exist_ok=True)
        q = Queue("fake")
        q.journal_path = ec.JOURNAL
        q.limit = args.limit
        if args.tasks_file:
            ftasks = ec.jread(args.tasks_file)
            q.phase_files = lambda: []                   # type: ignore[assignment]
            q.all_tasks = lambda: ftasks                 # type: ignore[assignment]
        return q.run()
    if args.smoke:
        q = Queue("smoke")
        tasks = ec.jread(args.smoke_tasks)
        q.phase_files = lambda: []                       # type: ignore[assignment]
        q.all_tasks = lambda: tasks                      # type: ignore[assignment]
        return q.run()
    attempt = 0
    while True:                                  # an unexpected exception restarts the loop; finished work is skipped, in-flight work re-reads the cache
        q = Queue("run")
        try:
            return q.run()
        except Exception:
            attempt += 1
            import traceback
            print(f"{utc()} queue crashed (attempt {attempt}):", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            if attempt >= 10 or ec.STOPFILE.exists():
                raise
            time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
