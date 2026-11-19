"""Slot-limited, tagged Linux process runner for tool and meter jobs.

This ports the safety mechanisms in run 3 code/runner3.py:57-69,129-155
(SHA-256 c8fa2c815267d29b631bd2348c734a3860ea6d02c2751dac8e6475c2cb520394)
and code/jobtag.py:21-97, adding a process-group kill before the exact-tag
sweep as required by final architecture section 8. Real Vivado and LATCH
execution through this runner is UNTESTED until the scheduled smoke.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from orch.contract.models import RunAttempt, RunStatus
from orch.engine.jobtag import LEGACY_TAG_KEY, TAG_KEY, group_members, sweep_tag, tagged
from orch.interfaces import LaunchSpec, ProcessResult


@dataclass(frozen=True)
class _Attempt:
    """Private attempt result before aggregation into ProcessResult."""

    status: RunStatus
    exit_code: int | None
    wall_s: float
    cpu_user_s: float | None
    cpu_sys_s: float | None
    peak_memory_bytes: int | None
    load_1m_start: float
    load_1m_end: float
    busy_slots_start: int
    backpressure_wait_s: float
    backpressure_events: int
    process_group_kills: int
    tag_sweep_kills: int
    tag_survivors: tuple[int, ...]
    stdout_path: Path
    stderr_path: Path
    reason: str | None


class Runner:
    """Run one job per call under a shared OS slot, group and exact run tag."""

    def __init__(
        self,
        slot_dir: Path,
        *,
        max_slots: int = 24,
        slot_poll_s: float = 0.1,
        process_poll_s: float = 0.5,
        group_settle_s: float = 5.0,
        tag_grace_s: float = 30.0,
        transient_signatures: Mapping[str, str] | None = None,
        max_transient_retries: int = 2,
        retry_backoff_s: float = 1.0,
        load_ceiling_1m: float | None = None,
        min_available_memory_bytes: int | None = None,
        capacity_poll_s: float = 10.0,
    ) -> None:
        if not 1 <= max_slots <= 96:
            raise ValueError("max_slots must be between 1 and the shared-server cap 96")
        if load_ceiling_1m is not None and (not math.isfinite(load_ceiling_1m) or load_ceiling_1m <= 0):
            raise ValueError("load ceiling must be positive and finite")
        if min_available_memory_bytes is not None and min_available_memory_bytes <= 0:
            raise ValueError("memory floor must be positive")
        if capacity_poll_s <= 0:
            raise ValueError("capacity poll duration must be positive")
        if not 0 <= max_transient_retries <= 2:
            raise ValueError("at most two transient retries are allowed")
        if any(value < 0 for value in (slot_poll_s, process_poll_s, group_settle_s, tag_grace_s, retry_backoff_s)):
            raise ValueError("poll and grace durations must be nonnegative")
        if not Path("/proc").is_dir():
            raise RuntimeError("the tagged runner requires Linux /proc")
        self.slot_dir = Path(slot_dir)
        self.max_slots = max_slots
        self.slot_poll_s = max(0.01, slot_poll_s)
        self.process_poll_s = max(0.01, process_poll_s)
        self.group_settle_s = group_settle_s
        self.tag_grace_s = tag_grace_s
        self.max_transient_retries = max_transient_retries
        self.retry_backoff_s = retry_backoff_s
        self.load_ceiling_1m = load_ceiling_1m
        self.min_available_memory_bytes = min_available_memory_bytes
        self.capacity_poll_s = capacity_poll_s
        signatures = transient_signatures or {}
        for citation in signatures.values():
            if not citation or ":" not in citation:
                raise ValueError("each transient signature needs a real log file:line citation")
        self._signatures = tuple(
            (re.compile(pattern), citation) for pattern, citation in signatures.items()
        )

    @contextmanager
    def _slot(self) -> Iterator[int]:
        """Hold one flock slot through retries and all process cleanup."""

        self.slot_dir.mkdir(parents=True, exist_ok=True)
        while True:
            for index in range(self.max_slots):
                handle = (self.slot_dir / f"slot{index}").open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                try:
                    yield index
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    handle.close()
                return
            time.sleep(self.slot_poll_s)

    @staticmethod
    def _sample_rss(pgrp: int, tag: str) -> int:
        """Sample total RSS across group members and tagged detached children."""

        processes = {info.pid: info for info in group_members(pgrp)}
        processes.update({info.pid: info for info in tagged(tag)})
        return sum(info.rss_bytes for info in processes.values())

    def _cleanup(self, pgrp: int, tag: str) -> tuple[int, int, tuple[int, ...]]:
        """TERM/KILL the group, wait five seconds by default, then sweep by tag."""

        group_targeted = {info.pid for info in group_members(pgrp)}
        if group_targeted:
            try:
                os.killpg(pgrp, signal.SIGTERM)
            except ProcessLookupError:
                group_targeted.clear()
        # A detached Vivado engine may appear after its parent exits. Keep the
        # run's slot during this settle period, as run 3 did before its sweep.
        time.sleep(self.group_settle_s)
        hard = {info.pid for info in group_members(pgrp)}
        if hard:
            try:
                os.killpg(pgrp, signal.SIGKILL)
                group_targeted.update(hard)
            except ProcessLookupError:
                pass
        sweep = sweep_tag(tag, grace_s=self.tag_grace_s)
        return len(group_targeted), sweep.targeted_count, sweep.alive

    @staticmethod
    def _time_peak(path: Path) -> int | None:
        """Read the observed GNU time -v maximum RSS field, when available."""

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        return int(match.group(1)) * 1024 if match else None

    @staticmethod
    def _time_cpu(path: Path) -> tuple[float | None, float | None]:
        """Read GNU time -v user/system seconds observed in run-3 time.txt."""

        try:
            report = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None, None
        def number(label: str) -> float | None:
            pattern = rf"^\s*{re.escape(label)}:\s*([0-9]+(?:\.[0-9]+)?)\s*$"
            match = re.search(pattern, report, re.MULTILINE)
            return float(match.group(1)) if match else None
        return number("User time (seconds)"), number("System time (seconds)")

    @staticmethod
    def _resource_snapshot() -> tuple[float, int]:
        """Read the one-minute load and MemAvailable immediately before launch."""

        load = os.getloadavg()[0]
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return load, int(line.split()[1]) * 1024
        raise RuntimeError("MemAvailable is absent; refuse tool launch")

    def _busy_slots(self) -> int:
        """Count live shared flock tokens, including this job's acquired slot."""

        busy = 0
        for index in range(96):
            path = self.slot_dir / f"slot{index}"
            if not path.is_file():
                continue
            with path.open("a+") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    busy += 1
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        return busy

    def _wait_for_capacity(self) -> tuple[float, int]:
        """Pause every new tool launch while either overnight threshold fails."""

        if self.load_ceiling_1m is None and self.min_available_memory_bytes is None:
            return 0.0, 0
        began: float | None = None
        while True:
            load, memory = self._resource_snapshot()
            load_blocked = self.load_ceiling_1m is not None and load > self.load_ceiling_1m
            memory_blocked = (
                self.min_available_memory_bytes is not None
                and memory < self.min_available_memory_bytes
            )
            if not (load_blocked or memory_blocked):
                return (time.monotonic() - began if began is not None else 0.0,
                        int(began is not None))
            if began is None:
                began = time.monotonic()
            time.sleep(self.capacity_poll_s)

    def _transient_reason(self, stdout_path: Path, stderr_path: Path) -> str | None:
        """Match only explicitly cited real-log signatures, never guessed ones."""

        if not self._signatures:
            return None
        for path in (stdout_path, stderr_path):
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        for pattern, citation in self._signatures:
                            if pattern.search(line):
                                return f"transient signature from {citation}"
            except OSError:
                continue
        return None

    def _run_attempt(
        self,
        spec: LaunchSpec,
        *,
        run_dir: Path,
        tag: str,
        index: int,
    ) -> _Attempt:
        stdout_path = run_dir / f"stdout.attempt-{index}.txt"
        stderr_path = run_dir / f"stderr.attempt-{index}.txt"
        time_path = run_dir / f"time.attempt-{index}.txt"
        env = os.environ.copy()
        env.update(spec.env_overrides)
        env[TAG_KEY] = tag
        # Keep the former tag for traceability while cleanup uses CONTRACT_RUN_ID.
        env[LEGACY_TAG_KEY] = tag
        command = list(spec.argv)
        if Path("/usr/bin/time").is_file():
            command = ["/usr/bin/time", "-v", "-o", str(time_path), *command]

        started = time.monotonic()
        capacity_wait_s, capacity_events = self._wait_for_capacity()
        load_start, _available_start = self._resource_snapshot()
        busy_start = self._busy_slots()
        timed_out = False
        sampled_peak = 0
        group_kills = 0
        tag_kills = 0
        survivors: tuple[int, ...] = ()
        exit_code: int | None = None
        reason: str | None = None
        process: subprocess.Popen[bytes] | None = None
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=spec.cwd,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except OSError as exc:
                reason = f"launch error: {exc}"
                stderr.write((reason + "\n").encode("utf-8", errors="replace"))
            else:
                # Capacity waits are queue time, not tool execution time.
                deadline = time.monotonic() + spec.timeout_s
                while process.poll() is None:
                    sampled_peak = max(sampled_peak, self._sample_rss(process.pid, tag))
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    time.sleep(min(self.process_poll_s, remaining))
                group_kills, tag_kills, survivors = self._cleanup(process.pid, tag)
                try:
                    exit_code = process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    reason = "process survived group and tag cleanup"
                sampled_peak = max(sampled_peak, self._sample_rss(process.pid, tag))

        wall_s = time.monotonic() - started
        cpu_user_s, cpu_sys_s = self._time_cpu(time_path)
        timed_peak = self._time_peak(time_path)
        peak = max(timed_peak or 0, sampled_peak) or None
        load_end, _available_end = self._resource_snapshot()
        if survivors:
            status = RunStatus.TOOL_ERROR
            reason = ("deadline exceeded; " if timed_out else "") + "tagged processes survived cleanup"
        elif timed_out:
            status = RunStatus.TIMEOUT
            reason = reason or f"deadline exceeded ({spec.timeout_s:g} s)"
        elif exit_code == 0:
            # This means process success only. The adapter still checks its
            # native status file, output checkpoint and independent meter.
            status = RunStatus.OK
        else:
            status = RunStatus.TOOL_ERROR
            reason = reason or f"exit code {exit_code}"
            transient = self._transient_reason(stdout_path, stderr_path)
            if transient is not None:
                status = RunStatus.TRANSIENT
                reason = transient
        return _Attempt(
            status=status,
            exit_code=exit_code,
            wall_s=wall_s,
            cpu_user_s=cpu_user_s,
            cpu_sys_s=cpu_sys_s,
            peak_memory_bytes=peak,
            load_1m_start=load_start,
            load_1m_end=load_end,
            busy_slots_start=busy_start,
            backpressure_wait_s=capacity_wait_s,
            backpressure_events=capacity_events,
            process_group_kills=group_kills,
            tag_sweep_kills=tag_kills,
            tag_survivors=survivors,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            reason=reason,
        )

    @staticmethod
    def _latest_log(source: Path, target: Path) -> None:
        """Expose the last attempt at the familiar run-3 stdout.txt path."""

        temporary = target.with_name(target.name + f".tmp-{uuid.uuid4().hex}")
        os.link(source, temporary)
        os.replace(temporary, target)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        """Publish run metadata atomically after fsync, including failure paths."""

        temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def run(self, spec: LaunchSpec, *, run_id: str, run_dir: Path) -> ProcessResult:
        """Run one job and return process facts; no output validity is inferred."""

        if not run_id or not spec.argv:
            raise ValueError("run_id and argv must be nonempty")
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        cwd = Path(spec.cwd).resolve(strict=True)
        if not cwd.is_relative_to(run_dir):
            raise ValueError("tool cwd must be inside its private run directory")
        tag = uuid.uuid4().hex
        started_at = datetime.now(timezone.utc).isoformat()
        overall_start = time.monotonic()
        attempts: list[RunAttempt] = []
        group_kills = 0
        tag_kills = 0
        peak: int | None = None
        cpu_user_total = 0.0
        cpu_sys_total = 0.0
        cpu_complete = True
        backpressure_wait_total = 0.0
        backpressure_events_total = 0
        last: _Attempt | None = None
        with self._slot() as slot_index:
            for index in range(1, self.max_transient_retries + 2):
                last = self._run_attempt(spec, run_dir=run_dir, tag=tag, index=index)
                attempts.append(
                    RunAttempt(
                        index=index,
                        status=last.status,
                        exit_code=last.exit_code,
                        wall_s=last.wall_s,
                        cpu_user_s=last.cpu_user_s,
                        cpu_sys_s=last.cpu_sys_s,
                        peak_memory_bytes=last.peak_memory_bytes,
                        load_1m_start=last.load_1m_start,
                        load_1m_end=last.load_1m_end,
                        busy_slots_start=last.busy_slots_start,
                        backpressure_wait_s=last.backpressure_wait_s,
                        backpressure_events=last.backpressure_events,
                        stdout_path=str(last.stdout_path),
                        stderr_path=str(last.stderr_path),
                        failure_reason=last.reason,
                    )
                )
                group_kills += last.process_group_kills
                tag_kills += last.tag_sweep_kills
                if last.peak_memory_bytes is not None:
                    peak = max(peak or 0, last.peak_memory_bytes)
                if last.cpu_user_s is None or last.cpu_sys_s is None:
                    cpu_complete = False
                else:
                    cpu_user_total += last.cpu_user_s
                    cpu_sys_total += last.cpu_sys_s
                backpressure_wait_total += last.backpressure_wait_s
                backpressure_events_total += last.backpressure_events
                if last.status != RunStatus.TRANSIENT or index > self.max_transient_retries:
                    break
                time.sleep(self.retry_backoff_s * index)
            assert last is not None
            self._latest_log(last.stdout_path, run_dir / "stdout.txt")
            self._latest_log(last.stderr_path, run_dir / "stderr.txt")
            result = ProcessResult(
                run_id=run_id,
                status=last.status,
                exit_code=last.exit_code,
                wall_s=time.monotonic() - overall_start,
                cpu_user_s=cpu_user_total if cpu_complete else None,
                cpu_sys_s=cpu_sys_total if cpu_complete else None,
                peak_memory_bytes=peak,
                load_1m_start=attempts[0].load_1m_start,
                load_1m_end=last.load_1m_end,
                busy_slots_start=attempts[0].busy_slots_start,
                backpressure_wait_s=backpressure_wait_total,
                backpressure_events=backpressure_events_total,
                process_group_kills=group_kills,
                tag_sweep_kills=tag_kills,
                tag=f"{TAG_KEY}={tag}",
                stdout_path=str(run_dir / "stdout.txt"),
                stderr_path=str(run_dir / "stderr.txt"),
                attempts=tuple(attempts),
            )
            overrides_hash = hashlib.sha256(
                json.dumps(spec.env_overrides, sort_keys=True).encode("utf-8")
            ).hexdigest()
            self._write_json(
                run_dir / "runner.json",
                {
                    "process": result.model_dump(mode="json"),
                    "command": list(spec.argv),
                    "env_overrides_sha256": overrides_hash,
                    "script_sha256": spec.script_sha256,
                    "tool_versions": spec.tool_versions,
                    "java_version": spec.java_version,
                    "slot_index": slot_index,
                    "host": socket.gethostname(),
                    "started_at": started_at,
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "cleanup_survivors": list(last.tag_survivors),
                },
            )
            return result

