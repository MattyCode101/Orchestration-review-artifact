"""Watch the detached worker-fitting driver and clean exact-tag jobs if it dies.

This watchdog never signals by process name. It scopes tags through the
campaign's private run paths, signals their recorded process groups, then
sweeps each exact CONTRACT_RUN_ID tag. Normal completion needs no cleanup.
"""
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
ROOT = PROJECT / "data" / "sweeps" / "worker_fitting_train"
RESUME_MANIFEST = ROOT / "resume_manifest.json"
MANIFEST = RESUME_MANIFEST if RESUME_MANIFEST.is_file() else ROOT / "launch_manifest.json"
STATUS = ROOT / "status.json"
AUDIT = ROOT / (
    "watchdog_verified_cleanup_audit.json"
    if MANIFEST == RESUME_MANIFEST else "watchdog_cleanup_audit.json"
)
DRIVER_SCRIPT = PROJECT / "scripts" / "launch_worker_fitting_train.py"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _driver_alive(pid: int) -> bool:
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
    return state != "Z" and str(DRIVER_SCRIPT) in args and "--run" in args


def _status() -> dict[str, object]:
    if not STATUS.is_file():
        return {}
    return json.loads(STATUS.read_text(encoding="utf-8"))


def _targets() -> tuple[set[str], set[int]]:
    """Discover only same-user tags rooted in this campaign's run directory."""

    prefix = str(ROOT).encode("utf-8")
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


def _signal_known_group(group: int, tags: set[str], sig: signal.Signals) -> bool:
    members = group_members(group)
    if not any(_tag_of(member.pid) in tags for member in members):
        return False
    try:
        os.killpg(group, sig)
        return True
    except ProcessLookupError:
        return False


def _cleanup() -> dict[str, object]:
    tags, groups = _targets()
    audit: dict[str, object] = {
        "schema_version": "worker-fitting-watchdog-v1",
        "reason": "recorded detached driver exited before campaign completion",
        "started_at_utc": _utc(),
        "exact_tags": sorted(tags),
        "process_groups": sorted(groups),
    }
    _atomic_bytes(AUDIT, _json_bytes(audit))
    audit["term_groups"] = [
        group for group in sorted(groups)
        if _signal_known_group(group, tags, signal.SIGTERM)
    ]
    time.sleep(3)
    audit["kill_groups"] = [
        group for group in sorted(groups)
        if _signal_known_group(group, tags, signal.SIGKILL)
    ]
    audit["tag_sweeps"] = {
        tag: {
            "targeted_count": result.targeted_count,
            "survivors": list(result.alive),
        }
        for tag in sorted(tags)
        for result in [sweep_tag(tag, grace_s=5)]
    }
    audit["survivors"] = {
        tag: [item.pid for item in tagged(tag)]
        for tag in sorted(tags) if tagged(tag)
    }
    audit["finished_at_utc"] = _utc()
    _atomic_bytes(AUDIT, _json_bytes(audit))
    state = _status()
    state["phase"] = "failed"
    state["error"] = (
        "detached driver exited before completion; watchdog swept "
        f"{len(tags)} exact CONTRACT_RUN_ID tags, survivors={audit['survivors']}"
    )
    state["updated_at_utc"] = _utc()
    _atomic_bytes(STATUS, _json_bytes(state))
    return audit


def watch() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pid = manifest["driver_pid"]
    if not isinstance(pid, int) or not _driver_alive(pid):
        raise RuntimeError("recorded driver is not alive at watchdog launch")
    deadline = time.monotonic() + 36 * 3600
    while time.monotonic() < deadline:
        state = _status()
        if state.get("phase") == "complete":
            return
        if not _driver_alive(pid):
            _cleanup()
            return
        time.sleep(30)
    raise RuntimeError("watchdog lifetime ended while driver remained alive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pid = manifest["driver_pid"]
    if args.watch:
        watch()
        print(json.dumps({"phase": "watchdog_done", "at_utc": _utc()}))
    else:
        tags, groups = _targets()
        print(json.dumps({
            "driver_alive": isinstance(pid, int) and _driver_alive(pid),
            "campaign_phase": _status().get("phase"),
            "active_exact_tags": len(tags),
            "tagged_process_groups": len(groups),
            "cleanup_audit_exists": AUDIT.is_file(),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

