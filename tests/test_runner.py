"""Dummy-process tests for slot, deadline, group and exact-tag cleanup."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from orch.contract.models import RunStatus
from orch.engine.jobtag import LEGACY_TAG_KEY, TAG_KEY, sweep_tag, tagged
from orch.engine.runner import Runner
from orch.interfaces import LaunchSpec


def make_runner(tmp_path: Path, **changes: object) -> Runner:
    settings: dict[str, object] = {
        "max_slots": 1,
        "slot_poll_s": 0.01,
        "process_poll_s": 0.01,
        "group_settle_s": 0.03,
        "tag_grace_s": 0.10,
        "retry_backoff_s": 0.01,
    }
    settings.update(changes)
    return Runner(tmp_path / "slots", **settings)


def launch(run_dir: Path, code: str, timeout_s: float = 2.0, *args: str) -> LaunchSpec:
    run_dir.mkdir(parents=True, exist_ok=True)
    return LaunchSpec(
        argv=(sys.executable, "-c", code, *args),
        cwd=str(run_dir),
        timeout_s=timeout_s,
    )


def tag_value(tag_field: str) -> str:
    key, value = tag_field.split("=", 1)
    assert key == TAG_KEY
    return value


def test_normal_process_and_provenance(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    run_dir = tmp_path / "normal"
    result = runner.run(
        launch(run_dir, "print('hello from child')"),
        run_id="normal",
        run_dir=run_dir,
    )
    assert result.status == RunStatus.OK
    assert result.exit_code == 0
    assert len(result.attempts) == 1
    assert Path(result.stdout_path).read_text().strip() == "hello from child"
    assert tagged(tag_value(result.tag)) == []
    metadata = json.loads((run_dir / "runner.json").read_text())
    assert metadata["process"]["status"] == "ok"
    assert metadata["command"][0] == sys.executable
    assert metadata["env_overrides_sha256"]
    assert result.peak_memory_bytes is not None
    assert result.cpu_user_s is not None and result.cpu_user_s >= 0
    assert result.cpu_sys_s is not None and result.cpu_sys_s >= 0
    assert result.load_1m_start is not None and result.load_1m_end is not None
    assert result.busy_slots_start is not None and result.busy_slots_start >= 1
    assert metadata["process"]["cpu_user_s"] == result.cpu_user_s
    assert metadata["process"]["busy_slots_start"] == result.busy_slots_start



def test_contract_run_id_and_legacy_alias_reach_subprocess(tmp_path: Path) -> None:
    """The exact cleanup token survives the shell environment boundary."""

    runner = make_runner(tmp_path)
    run_dir = tmp_path / "contract_tag"
    code = (
        "import os; print(os.environ['CONTRACT_RUN_ID']);"
        "print(os.environ['ORCH_RUN_TAG'])"
    )
    result = runner.run(
        launch(run_dir, code), run_id="contract-tag", run_dir=run_dir
    )
    contract, legacy = Path(result.stdout_path).read_text().splitlines()
    assert TAG_KEY == "CONTRACT_RUN_ID"
    assert LEGACY_TAG_KEY == "ORCH_RUN_TAG"
    assert contract == legacy == tag_value(result.tag)
    assert tagged(contract) == []


def test_timeout_kills_own_process_group(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    run_dir = tmp_path / "timeout"
    result = runner.run(
        launch(run_dir, "import time; time.sleep(30)", timeout_s=0.15),
        run_id="timeout",
        run_dir=run_dir,
    )
    assert result.status == RunStatus.TIMEOUT
    assert result.process_group_kills >= 1
    assert tagged(tag_value(result.tag)) == []


def test_detached_child_is_swept_by_tag(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    run_dir = tmp_path / "detached"
    pid_file = run_dir / "child.pid"
    code = (
        "import pathlib,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
        "start_new_session=True);"
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid));"
        "time.sleep(0.05)"
    )
    result = runner.run(
        launch(run_dir, code, 2.0, str(pid_file)),
        run_id="detached",
        run_dir=run_dir,
    )
    assert result.status == RunStatus.OK
    assert pid_file.exists()
    assert result.tag_sweep_kills >= 1
    assert tagged(tag_value(result.tag)) == []


def test_cleanup_never_touches_another_job_tag(tmp_path: Path) -> None:
    other_tag = "unrelated-" + uuid.uuid4().hex
    env = os.environ.copy()
    env[TAG_KEY] = other_tag
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env=env,
        start_new_session=True,
    )
    try:
        run_dir = tmp_path / "own"
        result = make_runner(tmp_path).run(
            launch(run_dir, "print('own job')"),
            run_id="own",
            run_dir=run_dir,
        )
        assert result.status == RunStatus.OK
        assert other.poll() is None
        assert tagged(other_tag)
    finally:
        sweep_tag(other_tag, grace_s=0.1)
        if other.poll() is None:
            os.killpg(other.pid, signal.SIGKILL)
        other.wait(timeout=2)


def test_slots_serialize_two_concurrent_calls(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    paths = [tmp_path / "slot_a", tmp_path / "slot_b"]

    def run_one(index: int) -> RunStatus:
        spec = launch(paths[index], "import time; time.sleep(0.35)")
        return runner.run(spec, run_id=f"slot-{index}", run_dir=paths[index]).status

    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(run_one, range(2)))
    elapsed = time.monotonic() - began
    assert statuses == [RunStatus.OK, RunStatus.OK]
    assert elapsed >= 0.60


def test_unknown_error_is_not_retried(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    run_dir = tmp_path / "error"
    result = runner.run(
        launch(run_dir, "print('licence-like words'); raise SystemExit(7)"),
        run_id="error",
        run_dir=run_dir,
    )
    assert result.status == RunStatus.TOOL_ERROR
    assert result.exit_code == 7
    assert len(result.attempts) == 1



def _separate_process_slot_job(base: str, index: int, start_event: object, queue: object) -> None:
    """Independent Python process using the same flock slot directory."""

    root = Path(base)
    run_dir = root / f"multiprocess-{index}"
    runner = Runner(
        root / "shared-slots",
        max_slots=1,
        slot_poll_s=0.01,
        process_poll_s=0.01,
        group_settle_s=0.03,
        tag_grace_s=0.10,
    )
    start_event.wait()
    result = runner.run(
        launch(run_dir, "import time; time.sleep(0.35)"),
        run_id=f"multiprocess-{index}",
        run_dir=run_dir,
    )
    queue.put(result.status.value)


def test_slot_lock_works_across_independent_processes(tmp_path: Path) -> None:
    """The server cap must hold across separate campaign processes too."""

    import multiprocessing

    context = multiprocessing.get_context("fork")
    start_event = context.Event()
    queue = context.Queue()
    processes = [
        context.Process(
            target=_separate_process_slot_job,
            args=(str(tmp_path), index, start_event, queue),
        )
        for index in (0, 1)
    ]
    for process in processes:
        process.start()
    began = time.monotonic()
    start_event.set()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0
    elapsed = time.monotonic() - began
    assert sorted(queue.get(timeout=1) for _ in processes) == ["ok", "ok"]
    assert elapsed >= 0.60


def test_slot_limit_refuses_more_than_server_cap(tmp_path: Path) -> None:
    import pytest

    Runner(tmp_path / "slots", max_slots=96)
    with pytest.raises(ValueError, match="cap 96"):
        Runner(tmp_path / "slots", max_slots=97)



def test_cpu_parser_matches_saved_real_gnu_time_report() -> None:
    """Pin CPU and RSS parsing to a completed test-noise Vivado time report."""

    fixture = Path(__file__).parent / "fixtures/runner/fir_transposed_repeat_time.txt"
    assert Runner._time_cpu(fixture) == (813.99, 136.42)
    assert Runner._time_peak(fixture) == 4573124 * 1024


def test_resource_backpressure_waits_before_launch(tmp_path: Path, monkeypatch) -> None:
    """No subprocess starts until both observed load and memory clear."""

    snapshots = iter([
        (181.0, 300_000_000_000),
        (5.0, 199_000_000_000),
        (5.0, 300_000_000_000),
    ])
    monkeypatch.setattr(
        Runner, "_resource_snapshot",
        staticmethod(lambda: next(snapshots, (5.0, 300_000_000_000))),
    )
    runner = make_runner(
        tmp_path, load_ceiling_1m=180.0,
        min_available_memory_bytes=200_000_000_000,
        capacity_poll_s=0.01,
    )
    run_dir = tmp_path / "capacity"
    result = runner.run(
        launch(run_dir, "print('capacity-cleared')"),
        run_id="capacity", run_dir=run_dir,
    )
    assert result.status == RunStatus.OK
    assert result.backpressure_events == 1
    assert result.backpressure_wait_s >= 0.02
    assert result.load_1m_start == 5.0
    assert result.busy_slots_start is not None and result.busy_slots_start >= 1
    assert Path(result.stdout_path).read_text().strip() == "capacity-cleared"


def test_backpressure_wait_does_not_consume_tool_deadline(
    tmp_path: Path, monkeypatch
) -> None:
    """The tool timeout starts when Popen succeeds, after capacity clears."""

    blocked_until = time.monotonic() + 0.15
    monkeypatch.setattr(
        Runner, "_resource_snapshot",
        staticmethod(
            lambda: (
                (181.0, 300_000_000_000)
                if time.monotonic() < blocked_until
                else (5.0, 300_000_000_000)
            )
        ),
    )
    runner = make_runner(
        tmp_path, load_ceiling_1m=180.0,
        min_available_memory_bytes=200_000_000_000,
        capacity_poll_s=0.01,
    )
    run_dir = tmp_path / "capacity_deadline"
    result = runner.run(
        launch(run_dir, "print('finished')", timeout_s=0.10),
        run_id="capacity-deadline", run_dir=run_dir,
    )
    assert result.status == RunStatus.OK
    assert result.backpressure_events == 1
    assert result.backpressure_wait_s >= 0.10
