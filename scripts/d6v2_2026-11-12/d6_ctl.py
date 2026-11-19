"""D6-v2 control (pattern of scripts/exhaustive_d3_2026-11-12/ed_ctl.py): status | resume | stop [--now].

status  the queue process, the status file, done/total per design group, brief and approach, the caps used, the
        episodes in flight and the failures, read from the plan, the journal and the result files (exact counts).
resume  starts the queue detached (new session and process group, stdin closed, output to logs/, SIGHUP ignored)
        unless one is running; finished episodes are skipped and interrupted ones resume from their journals.
        The model key must be in the environment (run it from a login shell); it is checked, never printed.
stop    SIGTERM to the queue process only: it stops starting episodes and drains the ones in flight.
        --now: SIGKILL to the queue's process group (the queue and its episode processes), then to the process groups
        of tool jobs whose command line or working directory lies under data/d6v2_2026-11-12. Nothing else.
"""
from __future__ import annotations

import collections
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d6_common as dc  # noqa: E402

QUEUE = Path(__file__).resolve().parent / "d6_queue.py"
STOPPED_MARK = dc.WORK / "STOPPED_BY_USER"


def queue_pid() -> dict | None:
    if not dc.PIDFILE.is_file():
        return None
    info = dc.jread(dc.PIDFILE)
    try:
        cmd = Path(f"/proc/{info['pid']}/cmdline").read_bytes().decode(errors="replace")
        stat = Path(f"/proc/{info['pid']}/stat").read_text()
    except OSError:
        return None
    if "d6_queue.py" not in cmd or stat[stat.rindex(")") + 2:].split()[0] == "Z":
        return None
    return info


def status() -> int:
    now = datetime.now(timezone.utc)
    info = queue_pid()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC   D6-v2 queue: " +
          (f"RUNNING pid {info['pid']} pgid {info['pgid']} since {info['started_at_utc']}" if info else "NOT RUNNING"))
    tasks = dc.tasks()
    rows = dc.read_journal()
    st = dc.jread(dc.STATUS) if dc.STATUS.is_file() else {}
    if st:
        age = (now - datetime.fromisoformat(st["updated_at_utc"])).total_seconds()
        print(f"status file age {age:.0f} s | in flight {len(st.get('in_flight', []))} (limit {st.get('max_active')}, "
              f"depth-3 queue {'alive' if st.get('depth3_alive') else 'ended'}) | load1 {st.get('load_1m'):.0f} | "
              f"admission: {st.get('admission')} | episodes/h {st.get('episodes_per_hour_last_window')} | "
              f"ETA {st.get('eta_utc')}")
    caps = dc.jread(dc.WORK / "caps.json") if (dc.WORK / "caps.json").is_file() else None
    if caps:
        u = caps["used"]
        print(f"caps: new real runs {u['new_real_runs']}/{caps['caps']['new_real_runs']} | job-hours "
              f"{u['job_seconds'] / 3600:.1f}/{caps['caps']['job_hours']} | US$ {u['usd']:.2f}/{caps['caps']['usd']} | "
              f"tokens {u['tokens']} | stopped: {caps['stopped'] or 'none'}")
    per: dict[str, collections.Counter] = collections.OrderedDict()
    failures = []
    for t in tasks:
        key = f"{t['group']:11s} {t['brief']} {t['approach']}"
        c = per.setdefault(key, collections.Counter())
        c["total"] += 1
        r = rows.get(t["episode_id"])
        if r is None:
            c["pending"] += 1
            continue
        c[r["status"]] += 1
        if r["status"] != "done":
            failures.append((t["episode_id"], r["status"], str(r.get("reason") or "")[:160]))
    print()
    print(f"{'group       brief approach':34s} {'done':>5s} {'total':>5s} {'pending':>8s} {'capped':>7s} "
          f"{'invalid':>8s} {'infra':>6s} {'error':>6s}")
    for key, c in per.items():
        print(f"{key:34s} {c['done']:5d} {c['total']:5d} {c['pending']:8d} {c['capped']:7d} {c['invalid_output']:8d} "
              f"{c['infrastructure_failure']:6d} {c['error']:6d}")
    total = sum(c["total"] for c in per.values())
    done = sum(c["done"] for c in per.values())
    print(f"{'TOTAL':34s} {done:5d} {total:5d}")
    if st.get("in_flight"):
        print("\nin flight:", ", ".join(f"{x['episode_id']} ({x['elapsed_min']} min)" for x in st["in_flight"]))
    if failures:
        print("\nnot done (latest row per episode; capped/failed ones run again on resume):")
        for f in failures[:20]:
            print("  ", f)
    return 0


def launch() -> int:
    if queue_pid():
        print("queue already running:", queue_pid()["pid"])
        return 0
    if not os.environ.get("GOOGLE_API_KEY"):
        print("refused: GOOGLE_API_KEY is not set in this shell (start it from a login shell: bash -lc '...')")
        return 2
    if not (dc.PLAN / "tasks.json").is_file():
        print("refused: no task list (run d6_plan.py --write first)")
        return 2
    dc.LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = open(dc.LOGS / f"queue_{stamp}.stdout", "ab")
    err = open(dc.LOGS / f"queue_{stamp}.stderr", "ab")
    if dc.STOPFILE.exists():
        print("note: STOP file present; removing it so the resumed queue can start")
        dc.STOPFILE.unlink()
    if STOPPED_MARK.exists():
        STOPPED_MARK.unlink()
    proc = subprocess.Popen([str(dc.PROJECT / ".venv/bin/python"), str(QUEUE), "--run"], cwd=str(dc.PROJECT),
                            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True, close_fds=True)
    time.sleep(8)
    print("launched pid", proc.pid, "| pidfile", queue_pid())
    return 0


def _tool_groups() -> set[int]:
    root = str(dc.WORK)
    groups: set[int] = set()
    mine = os.getpgrp()
    for p in os.scandir("/proc"):
        if not p.name.isdigit():
            continue
        try:
            if os.stat(f"/proc/{p.name}").st_uid != os.getuid():
                continue
            cmd = Path(f"/proc/{p.name}/cmdline").read_bytes().decode(errors="replace")
            try:
                cwd = os.readlink(f"/proc/{p.name}/cwd")
            except OSError:
                cwd = ""
        except OSError:
            continue
        if (root in cmd or cwd.startswith(root)) and "d6_ctl.py" not in cmd:
            try:
                g = os.getpgid(int(p.name))
            except OSError:
                continue
            if g != mine:
                groups.add(g)
    return groups


def stop(now_flag: bool) -> int:
    info = queue_pid()
    if not info:
        print("no queue process found")
    else:
        if os.getpgid(info["pid"]) != info["pgid"] or info["pgid"] != info["pid"]:
            print("refusing: the recorded pid is not the leader of its recorded process group", info)
            return 2
        STOPPED_MARK.write_text(f"stopped by d6_ctl.py stop{' --now' if now_flag else ''} at {dc.utc()}\n")
        if now_flag:
            os.killpg(info["pgid"], signal.SIGKILL)
            print("sent SIGKILL to process group", info["pgid"])
        else:
            os.kill(info["pid"], signal.SIGTERM)
            print("sent SIGTERM to the queue process", info["pid"], "(drains the episodes in flight)")
    if now_flag:
        time.sleep(2)
        for g in sorted(_tool_groups()):
            try:
                os.killpg(g, signal.SIGKILL)
                print("killed tool job group", g)
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
