"""D6-v2 detached queue (pattern of scripts/exhaustive_d3_2026-11-12/ed_queue.py): one process in its own session and
process group; each episode runs as a child process (scripts/d6v2_2026-11-12/d6_episode.py) inside that group.

Admission, re-read from queue_config.json every loop:
  * episodes in flight: at most 16 while the depth-3 queue is alive, then at most 40. An episode runs its tool jobs
    (recipe runs, telemetry extractions, AMD checks) one at a time, so this also bounds the tool slots it holds;
  * no new episode while the 1-minute load average is above 160 (and every tool job inside an episode waits while
    the load is above 160: the Runner's load ceiling is set to 160 in d6_common.build).
Journal (append-only): journal.jsonl, one row per finished attempt. A task with a 'done' or 'invalid_output' row is
never run again. 'capped', 'infrastructure_failure' and 'error' rows are run again on the next resume; within one run
an infrastructure failure is retried twice and an error once, at the end of the queue. Episodes resume from their
own journals. Stop: a STOP file or SIGTERM to the queue process stops admission and drains; a second SIGTERM exits.
SIGHUP is ignored (the queue does not depend on the shell or ssh session that started it).
usage: d6_queue.py --run | --fake
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_common as dc  # noqa: E402

DEFAULT_CONFIG = {
    "max_active_with_depth3": 16,
    "max_active": 40,
    "load_guard": dc.LOAD_GUARD,
    "admit_per_loop": 2,
    "status_period_s": 60,
    "retry_infra": 2,
    "retry_error": 1,
    "min_free_disk_gb": 300,
    "checkpoint_utc": "2026-11-13T09:00:00+00:00",
}
TERMINAL = ("done", "invalid_output")
STATUS_OF_CODE = {0: "done", 3: "capped", 4: "infrastructure_failure", 5: "invalid_output"}
EPISODE = Path(__file__).resolve().parent / "d6_episode.py"


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if dc.CONFIG.is_file():
        try:
            cfg.update(dc.jread(dc.CONFIG))
        except Exception:  # noqa: BLE001
            pass
    return cfg


def loadavg() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


class Queue:
    def __init__(self, fake: bool = False) -> None:
        self.fake = fake
        self.journal = dc.WORK / "fake" / "journal.jsonl" if fake else dc.JOURNAL
        self.status_path = dc.WORK / "fake" / "status.json" if fake else dc.STATUS
        self.cfg = load_config()
        self.t0 = time.time()
        self.t0_utc = dc.utc()
        self.stop_requested = False
        self.active: dict[str, dict] = {}
        self.finished: collections.deque = collections.deque(maxlen=2000)
        self.attempts: dict[str, int] = collections.defaultdict(int)
        self.last_status = 0.0
        self.checkpoint_done = False
        self.reason = ""

    # ------------------------------------------------------------------ journal
    def rows(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if self.journal.is_file():
            for line in self.journal.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[row["task"]] = row
        return out

    def commit(self, row: dict) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------------ children
    def spawn(self, task: dict) -> None:
        key = task["episode_id"]
        dc.LOGS.mkdir(parents=True, exist_ok=True)
        log = open(dc.LOGS / f"{key}.log", "ab")
        if self.fake:
            cmd = [sys.executable, "-c", "import random, time; time.sleep(random.uniform(1, 3))"]
        else:
            cmd = [str(dc.PROJECT / ".venv/bin/python"), str(EPISODE), "--key", key]
        proc = subprocess.Popen(cmd, cwd=str(dc.PROJECT), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                close_fds=True)                       # same process group as the queue
        self.attempts[key] += 1
        self.active[key] = {"proc": proc, "task": task, "started": time.time(), "started_utc": dc.utc(),
                            "attempt": self.attempts[key], "log": log}

    def reap(self) -> list[tuple[dict, str]]:
        done = []
        for key, item in list(self.active.items()):
            code = item["proc"].poll()
            if code is None:
                continue
            item["log"].close()
            del self.active[key]
            status = STATUS_OF_CODE.get(code, "error")
            row = {"task": key, "status": status, "exit_code": code, "attempt": item["attempt"],
                   "started_at_utc": item["started_utc"], "finished_at_utc": dc.utc(),
                   "wall_s": round(time.time() - item["started"], 1), "approach": item["task"]["approach"],
                   "design": item["task"]["design"], "brief": item["task"]["brief"], "tag": item["task"]["tag"]}
            if not self.fake:
                res = dc.RESULTS / (f"{key}.json" if status == "done" else f"{key}.{status}.json")
                if status == "error" and not res.is_file():
                    res = dc.RESULTS / f"{key}.error.json"
                if res.is_file():
                    try:
                        data = dc.jread(res)
                        row.update({k: data.get(k) for k in ("runs_used", "new_real_runs", "stop_reason", "reason",
                                                             "final_metrics", "kind")})
                    except Exception:  # noqa: BLE001
                        pass
            self.commit(row)
            self.finished.append((time.time(), row["wall_s"], status))
            done.append((item["task"], status))
        return done

    # ------------------------------------------------------------------ admission
    def max_active(self) -> tuple[int, bool]:
        alive = dc.depth3_running()
        return (int(self.cfg["max_active_with_depth3"]) if alive else int(self.cfg["max_active"])), alive

    def admit_ok(self) -> tuple[bool, str]:
        limit, alive = self.max_active()
        if len(self.active) >= limit:
            return False, f"in flight {len(self.active)} = limit {limit} ({'depth-3 queue alive' if alive else 'depth-3 queue ended'})"
        load = loadavg()
        if load > float(self.cfg["load_guard"]):
            return False, f"load {load:.0f} > {self.cfg['load_guard']:.0f}"
        if shutil.disk_usage(dc.WORK).free < float(self.cfg["min_free_disk_gb"]) * 2**30:
            return False, "disk"
        return True, "admitting"

    # ------------------------------------------------------------------ status and notes
    def payload(self, tasks: list[dict], pending: collections.deque) -> dict:
        rows = self.rows()
        by: dict[str, dict] = collections.OrderedDict()
        for t in tasks:
            k = f"{t['group']}|{t['brief']}|{t['approach']}"
            d = by.setdefault(k, {"total": 0, "done": 0, "invalid_output": 0, "capped": 0, "failed": 0})
            d["total"] += 1
            r = rows.get(t["episode_id"])
            if r is None:
                continue
            if r["status"] == "done":
                d["done"] += 1
            elif r["status"] == "invalid_output":
                d["invalid_output"] += 1
            elif r["status"] == "capped":
                d["capped"] += 1
            else:
                d["failed"] += 1
        now = time.time()
        window = min(3600.0, max(60.0, now - self.t0))
        recent = [x for x in self.finished if now - x[0] <= window and x[2] in TERMINAL]
        rate = len(recent) / window
        left = len(pending) + len(self.active)
        caps = dc.jread(dc.WORK / "caps.json") if (dc.WORK / "caps.json").is_file() else {}
        limit, alive = self.max_active()
        return {
            "updated_at_utc": dc.utc(), "queue_started_at_utc": self.t0_utc, "pid": os.getpid(), "fake": self.fake,
            "in_flight": [{"episode_id": k, "elapsed_min": round((now - v["started"]) / 60, 1), "attempt": v["attempt"]}
                          for k, v in self.active.items()],
            "pending": len(pending), "max_active": limit, "depth3_alive": alive, "load_1m": loadavg(),
            "admission": "stop requested: draining" if self.stop_requested else self.reason,
            "by_group_brief_approach": by,
            "done": sum(1 for r in rows.values() if r["status"] in TERMINAL), "tasks": len(tasks),
            "caps": {"used": caps.get("used"), "caps": caps.get("caps"), "stopped": caps.get("stopped")},
            "episodes_per_hour_last_window": round(rate * 3600, 2), "window_s": round(window),
            "eta_utc": None if rate <= 0 else (datetime.now(timezone.utc) + timedelta(seconds=left / rate)).isoformat(timespec="minutes"),
        }

    def progress(self, title: str, body: str) -> None:
        if self.fake:
            return
        text = f"\n\n### {dc.utc().replace('T', ' ').replace('+00:00', '')} UTC\n\n## {title}\n{body}\n"
        fd = os.open(str(dc.PROJECT / "notes" / "progress.md"), os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, text.encode("utf-8"))
        finally:
            os.close(fd)

    def summary_body(self, p: dict) -> str:
        used = (p["caps"] or {}).get("used") or {}
        lines = [f"- Queue pid {p['pid']} (started {p['queue_started_at_utc']}): {p['done']} of {p['tasks']} episodes "
                 f"finished (done or invalid output), {len(p['in_flight'])} in flight, {p['pending']} pending. "
                 f"Caps used: {used.get('new_real_runs')} of 450 new runs, "
                 f"{round(float(used.get('job_seconds') or 0) / 3600, 1)} of 100 job-hours, US${used.get('usd')} of 15. "
                 f"Stopped caps: {(p['caps'] or {}).get('stopped') or 'none'}."]
        for k, d in p["by_group_brief_approach"].items():
            if d["done"] != d["total"]:
                lines.append(f"  - {k}: {d['done']}/{d['total']} done, {d['capped']} capped, {d['failed']} failed, "
                             f"{d['invalid_output']} invalid output")
        lines.append("- Status: from the project root, .venv/bin/python scripts/d6v2_2026-11-12/d6_ctl.py status")
        return "\n".join(lines)

    def maybe_status(self, tasks, pending) -> None:
        now = time.time()
        if now - self.last_status < float(self.cfg["status_period_s"]):
            return
        self.last_status = now
        p = self.payload(tasks, pending)
        dc.jwrite(self.status_path, p)
        if (not self.checkpoint_done and not self.fake
                and datetime.now(timezone.utc) >= datetime.fromisoformat(self.cfg["checkpoint_utc"])):
            self.checkpoint_done = True
            self.progress("D6-v2 evaluation queue: 09:00 UK checkpoint (written by the queue process)", self.summary_body(p))

    # ------------------------------------------------------------------ main loop
    def on_signal(self, signum, _frame) -> None:
        if self.stop_requested:
            os._exit(1)
        self.stop_requested = True

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.on_signal)
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        tasks = dc.tasks()
        rows = self.rows()
        pending = collections.deque(t for t in tasks if rows.get(t["episode_id"], {}).get("status") not in TERMINAL)
        print(f"{dc.utc()} d6 queue: {len(tasks)} tasks, {len(tasks) - len(pending)} finished, {len(pending)} pending",
              flush=True)
        retries = {"infrastructure_failure": int(self.cfg["retry_infra"]), "error": int(self.cfg["retry_error"])}
        while (pending or self.active):
            try:
                self.cfg = load_config()
                if dc.STOPFILE.exists():
                    self.stop_requested = True
                for task, status in self.reap():
                    key = task["episode_id"]
                    if status in retries and self.attempts[key] <= retries[status] and not self.stop_requested:
                        pending.append(task)                      # retried at the end of the queue
                if self.stop_requested and not self.active:
                    break
                admitted = 0
                while pending and not self.stop_requested and admitted < int(self.cfg["admit_per_loop"]):
                    ok, self.reason = self.admit_ok()
                    if not ok:
                        break
                    task = pending.popleft()
                    if task["episode_id"] in self.active:
                        continue
                    self.spawn(task)
                    admitted += 1
                self.maybe_status(tasks, pending)
            except Exception as exc:  # noqa: BLE001 - the loop keeps running; children are independent processes
                print(f"{dc.utc()} loop error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(2.0)
        p = self.payload(tasks, pending)
        dc.jwrite(self.status_path, p)
        self.progress("D6-v2 evaluation queue: " + ("finished" if not pending else "stopped on request")
                      + " (written by the queue process)", self.summary_body(p))
        print(f"{dc.utc()} d6 queue exiting: pending {len(pending)}", flush=True)
        return 0


def write_pidfile(fake: bool) -> None:
    path = dc.WORK / "fake" / "queue.pid.json" if fake else dc.PIDFILE
    dc.jwrite(path, {"pid": os.getpid(), "pgid": os.getpgrp(), "sid": os.getsid(0), "argv": sys.argv,
                     "started_at_utc": dc.utc()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--fake", action="store_true")
    args = ap.parse_args()
    dc.LOGS.mkdir(parents=True, exist_ok=True)
    if args.run and dc.PIDFILE.is_file():
        old = dc.jread(dc.PIDFILE)
        try:
            cmd = Path(f"/proc/{old['pid']}/cmdline").read_bytes().decode(errors="replace")
        except OSError:
            cmd = ""
        if "d6_queue.py" in cmd and old["pid"] != os.getpid():
            print(f"a queue is already running: pid {old['pid']}")
            return 1
    write_pidfile(args.fake)
    attempt = 0
    while True:
        try:
            return Queue(fake=args.fake).run()
        except Exception:  # noqa: BLE001
            attempt += 1
            import traceback

            print(f"{dc.utc()} queue crashed (attempt {attempt}):", flush=True)
            traceback.print_exc()
            if attempt >= 10 or dc.STOPFILE.exists():
                raise
            time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
