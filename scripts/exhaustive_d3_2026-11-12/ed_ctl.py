"""Control commands for the depth-3 queue: status | resume | stop [--now].

status  reads the journal and the plan (exact counts), the status file, the process table and the shared slots.
resume  starts the queue detached (own session and process group) unless one is running; finished work is skipped.
stop    sends SIGTERM to the queue's exact process group (read from queue.pid.json, cmdline verified): the queue stops
        admitting and drains its in-flight jobs. With --now it SIGKILLs that group and then the in-flight tool job groups whose
        command line names this task's engine root. Nothing else is signalled.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ed_common as ec  # noqa: E402
import ed_queue as eq  # noqa: E402

STOPPED_MARK = ec.WORK / "STOPPED_BY_USER"      # written by stop, removed by resume; the auto-resume honours it


def queue_pid() -> dict | None:
    if not ec.PIDFILE.is_file():
        return None
    info = ec.jread(ec.PIDFILE)
    try:
        cmd = Path(f"/proc/{info['pid']}/cmdline").read_bytes().decode(errors="replace")
        if "ed_queue.py" not in cmd:
            return None
        stat = Path(f"/proc/{info['pid']}/stat").read_text()
        if stat[stat.rindex(")") + 2:].split()[0] == "Z":
            return None
    except OSError:
        return None
    return info


def parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


def status() -> int:
    tasks = []
    for i in ("0", "1", "1b", "2"):
        p = ec.PLAN / f"tasks_phase{i}.json"
        if p.is_file():
            tasks.extend(ec.jread(p))
    rows = eq.read_journals()
    per = OrderedDict()
    failures = []
    for t in tasks:
        k = eq.task_key(t)
        key = f"phase{t['phase']} {t['design']}"
        d = per.setdefault(key, {"total": 0, "done": 0, "failed": 0, "error": 0, "est_left_s": 0.0})
        d["total"] += 1
        r = rows.get(k)
        if r and r["status"] == "ok":
            d["done"] += 1
        elif r and r["status"] == "failed_terminal":
            d["done"] += 1
            d["failed"] += 1
            failures.append((key, k, r.get("run_status"), r.get("failure_reason")))
        elif r:
            d["error"] += 1
            failures.append((key, k, "error", r.get("error")))
            d["est_left_s"] += float(t.get("est_wall_s") or 0.0)
        else:
            d["est_left_s"] += float(t.get("est_wall_s") or 0.0)
    now = datetime.now(timezone.utc)
    info = queue_pid()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC   queue process: " +
          (f"RUNNING pid {info['pid']} pgid {info['pgid']} since {info['started_at_utc']}" if info else "NOT RUNNING"))
    st = ec.jread(ec.STATUS) if ec.STATUS.is_file() else {}
    if st:
        age = (now - parse_utc(st["updated_at_utc"])).total_seconds()
        print(f"status file age {age:.0f} s | in flight {st.get('in_flight')} | runner slots held {st.get('runner_slots_held')} | "
              f"slots busy (all users) {st.get('slots_busy_all_users')} | load1 {st.get('load_1m'):.0f} | admission: {st.get('admission')}")
    live_busy = eq.busy_slots(ec.PROJECT / "data" / "slots")
    print(f"live probe now: slots busy {live_busy}, loadavg {eq.proc_loadavg()[0]:.0f}")
    # measured throughput from the journal (stored-cost seconds finished in the last 60 minutes, cache hits excluded)
    window = 3600.0
    est_done = 0.0
    n_win = 0
    wall_ratio = []
    for r in rows.values():
        if str(r.get("kind")).startswith("amd_cell") or r.get("cache_hit") or not r.get("finished_at_utc"):
            continue
        if (now - parse_utc(r["finished_at_utc"])).total_seconds() <= window:
            est_done += float(r.get("est_wall_s") or 0.0)
            n_win += 1
            if r.get("est_wall_s") and r.get("wall_s"):
                wall_ratio.append(float(r["wall_s"]) / float(r["est_wall_s"]))
    first = parse_utc(info["started_at_utc"]) if info else min((parse_utc(r["started_at_utc"]) for r in rows.values()), default=None)
    if first is not None:
        window = min(window, max(60.0, (now - first).total_seconds()))
    thr = est_done / window if window else 0.0
    print(f"measured: {n_win} runs finished in the last {window / 60:.0f} min; stored-cost throughput {thr:.2f} job-s/s "
          f"({thr * 3600 / 3600:.1f} job-hours per hour); mean wall / stored mean {sum(wall_ratio) / len(wall_ratio):.2f}" if wall_ratio else
          f"measured: {n_win} runs finished in the last {window / 60:.0f} min")
    print()
    print(f"{'phase / design':58s} {'done':>5s}/{'total':<5s} {'failed':>6s} {'errors':>6s}   est job-h left   ETA (UTC)")
    cum = 0.0
    for key, d in per.items():
        cum += d["est_left_s"]
        eta = "-" if not d["est_left_s"] else ("unknown" if thr <= 0 else (now + timedelta(seconds=cum / thr)).strftime("%a %H:%M"))
        print(f"{key:58s} {d['done']:5d}/{d['total']:<5d} {d['failed']:6d} {d['error']:6d}   {d['est_left_s'] / 3600:13.1f}   {eta}")
    tot = sum(d["total"] for d in per.values()); done = sum(d["done"] for d in per.values())
    left = sum(d["est_left_s"] for d in per.values())
    print(f"{'TOTAL':58s} {done:5d}/{tot:<5d} {sum(d['failed'] for d in per.values()):6d} {sum(d['error'] for d in per.values()):6d}   {left / 3600:13.1f}   " +
          (f"all done ~{(now + timedelta(seconds=left / thr)).strftime('%a %H:%M')} UTC" if thr > 0 and left else ""))
    if failures:
        print("\nfailures / errors (first 15):")
        for f in failures[:15]:
            print("  ", f)
    return 0


def launch() -> int:
    if queue_pid():
        print("queue already running:", queue_pid()["pid"])
        return 0
    ec.LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = open(ec.LOGS / f"queue_{stamp}.stdout", "ab")
    err = open(ec.LOGS / f"queue_{stamp}.stderr", "ab")
    proc = subprocess.Popen([str(ec.PROJECT / ".venv/bin/python"), str(Path(__file__).resolve().parent / "ed_queue.py"), "--run"],
                            cwd=str(ec.PROJECT), stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True, close_fds=True)
    if ec.STOPFILE.exists():
        print("note: STOP file present; removing it so the resumed queue can start")
        ec.STOPFILE.unlink()
    if STOPPED_MARK.exists():
        STOPPED_MARK.unlink()                        # an explicit resume cancels the "stopped by a person" marker
    time.sleep(8)
    info = queue_pid()
    print("launched pid", proc.pid, "| pidfile", info)
    return 0


def stop(now_flag: bool) -> int:
    info = queue_pid()
    if not info:
        print("no queue process found")
        return 0
    pgid = info["pgid"]
    # exact verification: the group leader is the queue itself
    if os.getpgid(info["pid"]) != pgid or pgid != info["pid"]:
        print("refusing: the recorded pid is not the leader of its recorded process group", info)
        return 2
    sig = signal.SIGKILL if now_flag else signal.SIGTERM
    STOPPED_MARK.write_text(f"stopped by ed_ctl.py stop{' --now' if now_flag else ''} at {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    os.killpg(pgid, sig)
    print(f"sent {sig.name} to process group {pgid}")
    if now_flag:
        time.sleep(2)
        root = str(ec.ENGINE)
        groups = set()
        for p in os.scandir("/proc"):
            if not p.name.isdigit():
                continue
            try:
                if os.stat(f"/proc/{p.name}").st_uid != os.getuid():
                    continue
                cmd = Path(f"/proc/{p.name}/cmdline").read_bytes().decode(errors="replace")
            except OSError:
                continue
            if root in cmd and "ed_ctl.py" not in cmd and "ed_queue.py" not in cmd:
                try:
                    groups.add(os.getpgid(int(p.name)))
                except OSError:
                    pass
        for g in sorted(groups):
            try:
                os.killpg(g, signal.SIGKILL)
                print("killed in-flight tool job group", g)
            except OSError:
                pass
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        return status()
    if cmd in ("resume", "launch"):
        return launch()
    if cmd == "stop":
        return stop("--now" in sys.argv)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
