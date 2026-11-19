"""Test-run detached queue (pattern of scripts/d6v2_2026-11-12/d6_queue.py): one process in its own session and process group; each
episode runs as a child process (tr_episode.py) inside that group. The task list (plan/tasks.json) is re-read every loop, so rows added
later by tr_plan.py are picked up by a running queue.

Admission, re-read from queue_config.json every loop:
  * at most 30 episodes in flight (an episode runs its tool jobs one at a time, so at most 30 tool jobs);
  * no new episode while the 1-minute load average is above 160, and every tool job waits while the load is above 160 (Runner ceiling,
    tr_common.install_load_rule);
  * stall rule: if admission has been blocked by load for more than 30 minutes, an override starts: at most 8 episodes in flight and at
    most 8 tool jobs past the guard (private tokens), and one counts-only line is appended to notes/progress.md ("say so"). The override
    ends when the load is back under the guard (a second line).
Journal (append-only, STATUS ONLY: no metric ever enters it): journal.jsonl. A task with a terminal row (done, invalid_output,
not_applicable, cache_miss) is never run again. 'capped', 'infrastructure_failure' and 'error' rows run again on the next resume; within
one run an infrastructure failure is retried twice and an error once, at the end of the queue. Episodes resume from their own journals.
Stop: a STOP file or SIGTERM to the queue process stops admission and drains; a second SIGTERM exits. SIGHUP is ignored.
usage: tr_queue.py --run
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
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as tc  # noqa: E402

DEFAULT_CONFIG = {
    "max_active": tc.MAX_TOOL_JOBS,
    "override_max_active": tc.OVERRIDE_JOBS,
    "load_guard": tc.LOAD_GUARD,
    "stall_override_min": tc.STALL_OVERRIDE_MIN,
    "admit_per_loop": 2,
    "status_period_s": 60,
    "retry_infra": 2,
    "retry_error": 1,
    "min_free_disk_gb": 300,
    # two queues sharing one tool-job budget (13 Nov evening): shared_limit = the total of episodes in flight over both queues; while the
    # peer has pending work this queue keeps `reserve_for_peer` of them free; peer_status = the peer's status.json (read only)
    "shared_limit": 0,
    "reserve_for_peer": 0,
    "peer_status": None,
}
LABEL = {"": "baselines", "llm1": "LLM rows, queue 1", "power": "power review, queue 2", "llm3": "v2.2 variants, queue 3",
         "llm4": "power briefs, queue 4", "llm5": "queue 5 (Gemini blocks)", "llm6": "V9 test rows, queue 6",
         "llm7": "ChatGPT block, queue 5"}[tc.QUEUE]
TERMINAL = ("done", "invalid_output", "not_applicable", "cache_miss")
STATUS_OF_CODE = {0: "done", 3: "capped", 4: "infrastructure_failure", 5: "invalid_output", 6: "not_applicable", 7: "cache_miss"}
EPISODE = Path(__file__).resolve().parent / "tr_episode.py"


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if tc.CONFIG.is_file():
        try:
            cfg.update(tc.jread(tc.CONFIG))
        except Exception:  # noqa: BLE001
            pass
    return cfg


def progress(title: str, body: str) -> None:
    """Counts-only note in the shared log (append, O_APPEND). Not written by a dry run."""

    if tc.REPLAY:
        return
    text = f"\n\n###{tc.utc().replace('T', ' ').replace('+00:00', '')} UTC\n\n## {title}\n{body}\n"
    fd = os.open(str(tc.PROJECT / "notes" / "progress.md"), os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, text.encode("utf-8"))
    finally:
        os.close(fd)


class Queue:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.t0, self.t0_utc = time.time(), tc.utc()
        self.stop_requested = False
        self.active: dict[str, dict] = {}
        self.attempts: dict[str, int] = collections.defaultdict(int)
        self.last_status = 0.0
        self.reason = ""
        self.stall_since: float | None = None
        self.override = False
        self.finished: collections.deque = collections.deque(maxlen=2000)

    # ------------------------------------------------------------------ journal
    def rows(self) -> dict[str, dict]:
        return tc.read_journal()

    def commit(self, row: dict) -> None:
        tc.JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with tc.JOURNAL.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------------ children
    def spawn(self, task: dict) -> None:
        key = task["episode_id"]
        tc.LOGS.mkdir(parents=True, exist_ok=True)
        log = open(tc.LOGS / f"{key}.log", "ab")
        cmd = [str(tc.PROJECT / ".venv/bin/python"), str(EPISODE), "--key", key]
        proc = subprocess.Popen(cmd, cwd=str(tc.PROJECT), stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True)
        self.attempts[key] += 1
        self.active[key] = {"proc": proc, "task": task, "started": time.time(), "started_utc": tc.utc(),
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
                   "started_at_utc": item["started_utc"], "finished_at_utc": tc.utc(),
                   "wall_s": round(time.time() - item["started"], 1), "approach": item["task"]["approach"],
                   "design": item["task"]["design"], "brief": item["task"]["brief"], "tag": item["task"]["tag"]}
            if status != "done":
                res = tc.RESULTS / f"{key}.{status}.json"
                if res.is_file():
                    try:
                        data = tc.jread(res)
                        row.update({k: data.get(k) for k in ("reason", "kind")})
                    except Exception:  # noqa: BLE001
                        pass
            self.commit(row)
            self.finished.append((time.time(), row["wall_s"], status))
            done.append((item["task"], status))
        return done

    # ------------------------------------------------------------------ admission and the stall rule
    def max_active(self) -> int:
        return int(self.cfg["override_max_active"] if self.override else self.cfg["max_active"])

    def update_stall(self, pending) -> None:
        load, guard = tc.loadavg(), float(self.cfg["load_guard"])
        wanting = bool(pending) and not self.stop_requested
        if wanting and load > guard:
            if self.stall_since is None:
                self.stall_since = time.time()
            elif not self.override and time.time() - self.stall_since >= float(self.cfg["stall_override_min"]) * 60.0:
                self.override = True
                tc.jwrite(tc.OVERRIDE, {"active": True, "since": tc.utc(), "load_1m": load, "max_jobs": tc.OVERRIDE_JOBS})
                progress(f"Test-run driver ({LABEL}): admission stalled for more than 30 minutes, running up to 8 jobs regardless",
                         f"- 1-minute load {load:.0f} against the guard {guard:.0f} since "
                         f"{time.strftime('%H:%M', time.gmtime(self.stall_since))} UTC. At most {tc.OVERRIDE_JOBS} episodes in flight and "
                         f"{tc.OVERRIDE_JOBS} tool jobs past the guard until the load is back under {guard:.0f}. Counts only.")
        elif load <= guard:
            self.stall_since = None
            if self.override:
                self.override = False
                tc.jwrite(tc.OVERRIDE, {"active": False, "ended": tc.utc(), "load_1m": load})
                progress(f"Test-run driver ({LABEL}): load back under the guard, override ended",
                         f"- 1-minute load {load:.0f} (guard {guard:.0f}). Normal admission (up to {self.cfg['max_active']} in flight).")

    def peer(self) -> tuple[int, int]:
        """(in flight, pending) of the other queue from its status file when that is fresh (under 5 minutes), else (0, 0)."""

        path = self.cfg.get("peer_status")
        if not path:
            return 0, 0
        try:
            st = tc.jread(path)
            age = time.time() - datetime.fromisoformat(st["updated_at_utc"]).timestamp()
            return (len(st["in_flight"]), int(st["pending"])) if age < 300 else (0, 0)
        except Exception:  # noqa: BLE001
            return 0, 0

    def admit_ok(self) -> tuple[bool, str]:
        if len(self.active) >= self.max_active():
            return False, f"in flight {len(self.active)} = limit {self.max_active()}{' (override)' if self.override else ''}"
        shared = int(self.cfg.get("shared_limit") or 0)
        if shared:
            peer_in, peer_pending = self.peer()
            # the reserve is what the peer is still owed: the part of `reserve_for_peer` it has not yet taken up (so two queues that each
            # reserve 8 settle at 12 and 8 or better, instead of both stopping at 12 in total)
            keep = max(0, int(self.cfg.get("reserve_for_peer") or 0) - peer_in) if peer_pending > 0 else 0
            if len(self.active) + peer_in >= shared - keep:
                return False, f"shared limit: {len(self.active)} here + {peer_in} in the other queue, limit {shared}, {keep} kept for it"
        load = tc.loadavg()
        if load > float(self.cfg["load_guard"]) and not self.override:
            return False, f"load {load:.0f} > {self.cfg['load_guard']:.0f}"
        if shutil.disk_usage(tc.ROOT).free < float(self.cfg["min_free_disk_gb"]) * 2**30:
            return False, "disk"
        return True, "admitting" + (" (override)" if self.override else "")

    # ------------------------------------------------------------------ status (counts only)
    def payload(self, tasks: list[dict], pending) -> dict:
        rows = self.rows()
        by: dict[str, dict] = collections.OrderedDict()
        for t in tasks:
            k = f"{t['group']}|{t['brief']}|{t['approach']}"
            d = by.setdefault(k, {"total": 0, "done": 0, "terminal_other": 0, "capped": 0, "failed": 0})
            d["total"] += 1
            r = rows.get(t["episode_id"])
            if r is None:
                continue
            if r["status"] == "done":
                d["done"] += 1
            elif r["status"] in TERMINAL:
                d["terminal_other"] += 1
            elif r["status"] == "capped":
                d["capped"] += 1
            else:
                d["failed"] += 1
        caps = tc.jread(tc.ROOT / "caps.json") if (tc.ROOT / "caps.json").is_file() else {}
        return {
            "updated_at_utc": tc.utc(), "queue_started_at_utc": self.t0_utc, "pid": os.getpid(),
            "in_flight": [{"episode_id": k, "elapsed_min": round((time.time() - v["started"]) / 60, 1), "attempt": v["attempt"]}
                          for k, v in self.active.items()],
            "pending": len(pending), "max_active": self.max_active(), "override": self.override, "load_1m": tc.loadavg(),
            "stall_min": None if self.stall_since is None else round((time.time() - self.stall_since) / 60, 1),
            "admission": "stop requested: draining" if self.stop_requested else self.reason,
            "by_group_brief_approach": by, "done": sum(1 for r in rows.values() if r["status"] in TERMINAL), "tasks": len(tasks),
            "caps": {"used": caps.get("used"), "caps": caps.get("caps"), "stopped": caps.get("stopped")},
        }

    def summary_body(self, p: dict) -> str:
        used = (p["caps"] or {}).get("used") or {}
        lines = [f"- Queue pid {p['pid']} (started {p['queue_started_at_utc']}): {p['done']} of {p['tasks']} episodes in a terminal state, "
                 f"{len(p['in_flight'])} in flight, {p['pending']} pending. Caps used: {used.get('new_real_runs')} of "
                 f"{tc.CAPS['new_real_runs']} new runs, {round(float(used.get('job_seconds') or 0) / 3600, 1)} of {tc.CAPS['job_hours']} "
                 f"job-hours, US${used.get('usd')} of {tc.CAPS['usd']}. Stopped caps: {(p['caps'] or {}).get('stopped') or 'none'}."]
        for k, d in p["by_group_brief_approach"].items():
            if d["done"] != d["total"]:
                lines.append(f"  - {k}: {d['done']}/{d['total']} done, {d['capped']} capped, {d['failed']} failed, "
                             f"{d['terminal_other']} other terminal")
        lines.append("- Counts and completion only (results are sealed). Status: .venv/bin/python scripts/test_r2_2026-11-13/tr_ctl.py status")
        return "\n".join(lines)

    def maybe_status(self, tasks, pending) -> None:
        if time.time() - self.last_status < float(self.cfg["status_period_s"]):
            return
        self.last_status = time.time()
        tc.jwrite(tc.STATUS, self.payload(tasks, pending))

    # ------------------------------------------------------------------ main loop
    def on_signal(self, signum, _frame) -> None:
        if self.stop_requested:
            os._exit(1)
        self.stop_requested = True

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self.on_signal)
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        known: set[str] = set()
        pending: collections.deque = collections.deque()
        retries = {"infrastructure_failure": int(self.cfg["retry_infra"]), "error": int(self.cfg["retry_error"])}
        tasks: list[dict] = []
        print(f"{tc.utc()} queue starting", flush=True)
        while True:
            try:
                self.cfg = load_config()
                tasks = tc.tasks()
                rows = self.rows()
                for t in tasks:                          # rows added since the last loop
                    if t["episode_id"] not in known:
                        known.add(t["episode_id"])
                        if rows.get(t["episode_id"], {}).get("status") not in TERMINAL:
                            pending.append(t)
                if tc.STOPFILE.exists():
                    self.stop_requested = True
                for task, status in self.reap():
                    key = task["episode_id"]
                    if status in retries and self.attempts[key] <= retries[status] and not self.stop_requested:
                        pending.append(task)             # retried at the end of the queue
                if not pending and not self.active:
                    break
                if self.stop_requested and not self.active:
                    break
                self.update_stall(pending)
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
                print(f"{tc.utc()} loop error: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(2.0)
        p = self.payload(tasks, pending)
        tc.jwrite(tc.STATUS, p)
        if self.override:
            tc.jwrite(tc.OVERRIDE, {"active": False, "ended": tc.utc()})
        progress(f"Test-run driver queue ({LABEL}): " + ("finished" if not pending else "stopped on request") + " (written by the queue process)",
                 self.summary_body(p))
        print(f"{tc.utc()} queue exiting: pending {len(pending)}", flush=True)
        return 0


def write_pidfile() -> None:
    tc.jwrite(tc.PIDFILE, {"pid": os.getpid(), "pgid": os.getpgrp(), "sid": os.getsid(0), "argv": sys.argv, "started_at_utc": tc.utc()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true", required=True)
    ap.parse_args()
    tc.LOGS.mkdir(parents=True, exist_ok=True)
    if tc.PIDFILE.is_file():
        old = tc.jread(tc.PIDFILE)
        try:
            cmd = Path(f"/proc/{old['pid']}/cmdline").read_bytes().decode(errors="replace")
        except OSError:
            cmd = ""
        if "tr_queue.py" in cmd and old["pid"] != os.getpid():
            print(f"a queue is already running: pid {old['pid']}")
            return 1
    write_pidfile()
    attempt = 0
    while True:
        try:
            return Queue().run()
        except Exception:  # noqa: BLE001
            attempt += 1
            import traceback

            print(f"{tc.utc()} queue crashed (attempt {attempt}):", flush=True)
            traceback.print_exc()
            if attempt >= 10 or tc.STOPFILE.exists():
                raise
            time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
