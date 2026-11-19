"""Exact, same-user Linux process-tag discovery and cleanup.

Adapted from run 3 code/jobtag.py (SHA-256
0f6a3d3e215dd64c5aed3def4bdc2a51014b91023ea5a5cecdc72746e6d153db),
especially lines 21-97. This module never matches process names or command lines.
A child that clears its environment at exec cannot be found by tag; the runner's
process-group cleanup covers such a child while it remains in the group.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path


TAG_KEY = "CONTRACT_RUN_ID"
LEGACY_TAG_KEY = "ORCH_RUN_TAG"
_PREFIX = (TAG_KEY + "=").encode("ascii")


@dataclass(frozen=True)
class ProcessInfo:
    """One live process owned by the current user."""

    pid: int
    pgrp: int
    rss_bytes: int
    state: str


@dataclass(frozen=True)
class SweepResult:
    """PIDs targeted by TERM and KILL, plus still-live tagged PIDs."""

    found: tuple[int, ...]
    hard: tuple[int, ...]
    alive: tuple[int, ...]

    @property
    def targeted_count(self) -> int:
        return len(set(self.found) | set(self.hard))


def _stat(pid: int) -> ProcessInfo | None:
    """Read only the process group, state and resident pages from /proc stat."""

    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = text[text.rindex(")") + 2 :].split()
        state = fields[0]
        pgrp = int(fields[2])
        rss_pages = int(fields[21])
        return ProcessInfo(
            pid=pid,
            pgrp=pgrp,
            rss_bytes=rss_pages * os.sysconf("SC_PAGE_SIZE"),
            state=state,
        )
    except (OSError, ValueError, IndexError):
        return None


def _ours(pid: int) -> bool:
    try:
        return os.stat(f"/proc/{pid}").st_uid == os.getuid()
    except OSError:
        return False


def _tag_of(pid: int) -> str | None:
    """Read only the exact tag value; never expose the rest of the environment."""

    try:
        with open(f"/proc/{pid}/environ", "rb") as handle:
            entries = handle.read().split(b"\x00")
    except OSError:
        return None
    for entry in entries:
        if entry.startswith(_PREFIX):
            return entry[len(_PREFIX) :].decode("ascii", errors="replace")
    return None


def _own_pids() -> list[int]:
    """List numeric /proc entries belonging to this user."""

    result: list[int] = []
    for item in os.scandir("/proc"):
        if not item.name.isdigit():
            continue
        pid = int(item.name)
        if pid != os.getpid() and _ours(pid):
            result.append(pid)
    return result


def tagged(tag: str) -> list[ProcessInfo]:
    """Find this user's live processes carrying exactly this run's tag."""

    if not tag:
        raise ValueError("refusing to inspect an empty job tag")
    result: list[ProcessInfo] = []
    for pid in _own_pids():
        if _tag_of(pid) != tag:
            continue
        info = _stat(pid)
        if info is not None and info.state != "Z":
            result.append(info)
    return result


def group_members(pgrp: int) -> list[ProcessInfo]:
    """Find this user's live members of one known process group."""

    result: list[ProcessInfo] = []
    for pid in _own_pids():
        info = _stat(pid)
        if info is not None and info.pgrp == pgrp and info.state != "Z":
            result.append(info)
    return result


def _signal_exact(pids: tuple[int, ...], tag: str, sig: signal.Signals) -> tuple[int, ...]:
    """Re-check tag and ownership immediately before signalling a PID."""

    targeted: list[int] = []
    for pid in pids:
        if not _ours(pid) or _tag_of(pid) != tag:
            continue
        info = _stat(pid)
        if info is None or info.state == "Z":
            continue
        try:
            os.kill(pid, sig)
            targeted.append(pid)
        except ProcessLookupError:
            pass
    return tuple(targeted)


def sweep_tag(
    tag: str, *, grace_s: float = 30.0, poll_s: float = 0.1
) -> SweepResult:
    """TERM then KILL only same-user processes with the exact run tag."""

    if not tag:
        raise ValueError("refusing to sweep an empty job tag")
    initial = tuple(proc.pid for proc in tagged(tag))
    term = _signal_exact(initial, tag, signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, grace_s)
    remaining = tuple(proc.pid for proc in tagged(tag))
    while remaining and time.monotonic() < deadline:
        time.sleep(max(0.0, min(max(0.01, poll_s), deadline - time.monotonic())))
        remaining = tuple(proc.pid for proc in tagged(tag))
    hard = _signal_exact(remaining, tag, signal.SIGKILL)
    if hard:
        # Let the kernel reap the signalled children before reporting survivors.
        time.sleep(min(0.2, max(0.01, poll_s)))
    alive = tuple(proc.pid for proc in tagged(tag))
    return SweepResult(found=term, hard=hard, alive=alive)

