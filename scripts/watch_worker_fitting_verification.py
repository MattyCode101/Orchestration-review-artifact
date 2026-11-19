"""Watch a detached training-DCP validator batch and sweep only its exact tags."""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from orch.engine.jobtag import _own_pids, _stat, _tag_of, group_members, sweep_tag, tagged
from orch.engine.store import _atomic_bytes, _json_bytes

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "data" / "sweeps" / "worker_fitting_train" / "verification"
RUNS = ROOT / "runs"
DRIVER = PROJECT / "scripts" / "verify_worker_fitting_train.py"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _driver_alive(pid: int, mode: str) -> bool:
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return False
        args = [
            item.decode("utf-8", errors="replace")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\x00")
            if item
        ]
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0]
    except (OSError, IndexError):
        return False
    return state != "Z" and str(DRIVER) in args and f"--run-{mode}" in args


def _targets() -> tuple[set[str], set[int]]:
    """Discover own exact tags rooted in private validator run directories."""
    prefix = str(RUNS).encode("utf-8")
    tags: set[str] = set()
    groups: set[int] = set()
    for pid in _own_pids():
        info = _stat(pid)
        if info is None or info.state == "Z":
            continue
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            cwd = os.readlink(f"/proc/{pid}/cwd").encode("utf-8")
        except OSError:
            continue
        if prefix not in command and not cwd.startswith(prefix):
            continue
        tag = _tag_of(pid)
        if tag and re.fullmatch(r"[0-9a-f]{32}", tag):
            tags.add(tag)
            groups.add(info.pgrp)
    groups.discard(os.getpgrp())
    return tags, groups


def _signal_group(group: int, tags: set[str], sig: signal.Signals) -> bool:
    if not any(_tag_of(member.pid) in tags for member in group_members(group)):
        return False
    try:
        os.killpg(group, sig)
        return True
    except ProcessLookupError:
        return False


def _cleanup(mode: str) -> dict[str, object]:
    tags, groups = _targets()
    audit: dict[str, object] = {
        "schema_version": "worker-fitting-validator-watchdog-v1",
        "mode": mode, "started_at_utc": _utc(),
        "exact_tags": sorted(tags), "process_groups": sorted(groups),
    }
    path = ROOT / f"{mode}_watchdog_cleanup_audit.json"
    _atomic_bytes(path, _json_bytes(audit))
    audit["term_groups"] = [
        group for group in sorted(groups)
        if _signal_group(group, tags, signal.SIGTERM)
    ]
    time.sleep(3)
    audit["kill_groups"] = [
        group for group in sorted(groups)
        if _signal_group(group, tags, signal.SIGKILL)
    ]
    audit["tag_sweeps"] = {
        tag: {"targeted_count": result.targeted_count, "survivors": list(result.alive)}
        for tag in sorted(tags)
        for result in [sweep_tag(tag, grace_s=5)]
    }
    audit["survivors"] = {
        tag: [item.pid for item in tagged(tag)]
        for tag in sorted(tags) if tagged(tag)
    }
    audit["finished_at_utc"] = _utc()
    _atomic_bytes(path, _json_bytes(audit))
    state_path = ROOT / f"{mode}_status.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {}
    if state.get("phase") != "complete":
        state["phase"] = "failed"
        state["watchdog_cleanup_audit"] = str(path)
        state["watchdog_survivors"] = audit["survivors"]
        state["updated_at_utc"] = _utc()
        _atomic_bytes(state_path, _json_bytes(state))
    return audit


def watch(mode: str, pid: int) -> None:
    if not _driver_alive(pid, mode):
        raise RuntimeError("recorded validator driver is not alive at watchdog launch")
    deadline = time.monotonic() + 36 * 3600
    while time.monotonic() < deadline:
        state_path = ROOT / f"{mode}_status.json"
        state = (
            json.loads(state_path.read_text(encoding="utf-8"))
            if state_path.is_file() else {}
        )
        if state.get("phase") == "complete":
            return
        if not _driver_alive(pid, mode):
            _cleanup(mode)
            return
        time.sleep(30)
    raise RuntimeError("validator watchdog lifetime ended while driver remained alive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--mode", choices=("pilot", "backfill"), required=True)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    if args.watch:
        if args.pid is None:
            parser.error("--watch requires --pid")
        watch(args.mode, args.pid)
    else:
        tags, groups = _targets()
        print(json.dumps({
            "mode": args.mode, "active_exact_tags": len(tags),
            "tagged_process_groups": len(groups),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
