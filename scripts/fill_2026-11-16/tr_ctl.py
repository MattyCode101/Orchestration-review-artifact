"""Test-run driver control (pattern of scripts/d6v2_2026-11-12/d6_ctl.py): status | resume | stop [--now].

status  the queue process, the status file, done/total per group, brief and row, the caps used, the episodes in flight and the failures
        (status words only; results are sealed and are never printed), read from the plan, the journal and the result file names.
resume  starts the queue detached (new session and process group, stdin closed, output to logs/, SIGHUP ignored) unless one is running;
        finished episodes are skipped and interrupted ones resume from their journals. Model keys needed by pending rows (GOOGLE_API_KEY
        for Gemini, LLM_GATEWAY_API_KEY for gateway models) must be in the environment (run from a login shell); they are checked, never printed.
stop    SIGTERM to the queue process only: it stops starting episodes and drains the ones in flight.
        --now: SIGKILL to the queue's process group (the queue and its episode processes), then to the process groups of tool jobs whose
        command line or working directory lies under this run's folder. Nothing else.
TR_DRY=1 selects the dry-run folder (VALIDATION designs, replay mode).
"""
from __future__ import annotations

import collections
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as tc  # noqa: E402

QUEUE = Path(__file__).resolve().parent / "tr_queue.py"
STOPPED_MARK = tc.ROOT / "STOPPED_BY_USER"


def queue_pid() -> dict | None:
    if not tc.PIDFILE.is_file():
        return None
    info = tc.jread(tc.PIDFILE)
    try:
        cmd = Path(f"/proc/{info['pid']}/cmdline").read_bytes().decode(errors="replace")
        stat = Path(f"/proc/{info['pid']}/stat").read_text()
    except OSError:
        return None
    if "tr_queue.py" not in cmd or stat[stat.rindex(")") + 2:].split()[0] == "Z":
        return None
    return info


def status() -> int:
    now = datetime.now(timezone.utc)
    info = queue_pid()
    print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} UTC   test-run driver ({'DRY RUN' if tc.DRY else 'FILL RUNS'}) queue: " +
          (f"RUNNING pid {info['pid']} pgid {info['pgid']} since {info['started_at_utc']}" if info else "NOT RUNNING"))
    tasks = tc.tasks()
    rows = tc.read_journal()
    st = tc.jread(tc.STATUS) if tc.STATUS.is_file() else {}
    if st:
        age = (now - datetime.fromisoformat(st["updated_at_utc"])).total_seconds()
        print(f"status file age {age:.0f} s | in flight {len(st.get('in_flight', []))} (limit {st.get('max_active')}, "
              f"override {'ON' if st.get('override') else 'off'}) | load1 {st.get('load_1m'):.0f} | stalled {st.get('stall_min')} min | "
              f"admission: {st.get('admission')}")
    caps = tc.jread(tc.ROOT / "caps.json") if (tc.ROOT / "caps.json").is_file() else None
    if caps:
        u = caps["used"]
        print(f"caps: new real runs {u['new_real_runs']}/{caps['caps']['new_real_runs']} | job-hours "
              f"{u['job_seconds'] / 3600:.1f}/{caps['caps']['job_hours']} | US$ {u['usd']:.2f}/{caps['caps']['usd']} | "
              f"tokens {u['tokens']} | stopped: {caps['stopped'] or 'none'}")
    per: dict[str, collections.Counter] = collections.OrderedDict()
    failures = []
    for t in tasks:
        key = f"{t['group']:10s} {t['brief']:4s} {t['approach']}"
        c = per.setdefault(key, collections.Counter())
        c["total"] += 1
        r = rows.get(t["episode_id"])
        if r is None:
            c["pending"] += 1
            continue
        c[r["status"]] += 1
        if r["status"] not in ("done", "not_applicable"):
            failures.append((t["episode_id"], r["status"], str(r.get("reason") or "")[:140]))
    print()
    print(f"{'group      brief row':36s} {'done':>5s} {'total':>5s} {'pending':>8s} {'capped':>7s} {'invalid':>8s} {'infra':>6s} "
          f"{'error':>6s} {'n/a':>4s} {'miss':>5s}")
    for key, c in per.items():
        print(f"{key:36s} {c['done']:5d} {c['total']:5d} {c['pending']:8d} {c['capped']:7d} {c['invalid_output']:8d} "
              f"{c['infrastructure_failure']:6d} {c['error']:6d} {c['not_applicable']:4d} {c['cache_miss']:5d}")
    print(f"{'TOTAL':36s} {sum(c['done'] for c in per.values()):5d} {sum(c['total'] for c in per.values()):5d}")
    if st.get("in_flight"):
        print("\nin flight:", ", ".join(f"{x['episode_id']} ({x['elapsed_min']} min)" for x in st["in_flight"][:12]),
              "..." if len(st["in_flight"]) > 12 else "")
    if failures:
        print("\nnot done (latest row per episode; capped/failed ones run again on resume):")
        for f in failures[:20]:
            print("  ", f)
    return 0


def keys_needed() -> list[str]:
    rows = tc.read_journal()
    need = set()
    for t in tc.tasks():
        if rows.get(t["episode_id"], {}).get("status") in ("done", "invalid_output", "not_applicable", "cache_miss"):
            continue
        if t["approach"] in tc.LLM_APPROACHES:
            need.add("LLM_GATEWAY_API_KEY" if str(t.get("model") or "").startswith("gateway:") else "GOOGLE_API_KEY")
    return sorted(need)


def launch() -> int:
    if queue_pid():
        print("queue already running:", queue_pid()["pid"])
        return 0
    missing = [k for k in keys_needed() if not os.environ.get(k)]
    if missing:
        print(f"refused: {', '.join(missing)} not set in this shell (start it from a login shell: bash -lc '...')")
        return 2
    if not (tc.PLAN / "tasks.json").is_file():
        print("refused: no task list (run tr_plan.py first)")
        return 2
    tc.LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = open(tc.LOGS / f"queue_{stamp}.stdout", "ab")
    err = open(tc.LOGS / f"queue_{stamp}.stderr", "ab")
    if tc.STOPFILE.exists():
        print("note: STOP file present; removing it so the resumed queue can start")
        tc.STOPFILE.unlink()
    if STOPPED_MARK.exists():
        STOPPED_MARK.unlink()
    proc = subprocess.Popen([str(tc.PROJECT / ".venv/bin/python"), str(QUEUE), "--run"], cwd=str(tc.PROJECT),
                            stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True, close_fds=True)
    time.sleep(8)
    print("launched pid", proc.pid, "| pidfile", queue_pid())
    return 0


def _tool_groups() -> set[int]:
    root = str(tc.ROOT)
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
        if (root in cmd or cwd.startswith(root)) and "tr_ctl.py" not in cmd:
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
        STOPPED_MARK.write_text(f"stopped by tr_ctl.py stop{' --now' if now_flag else ''} at {tc.utc()}\n")
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
