"""No-tool tests for the sharded Thursday pilot wrapper."""
from __future__ import annotations

import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import pytest

from orch.engine.store import _json_bytes
from scripts import thursday_night_pilot as pilot
from scripts import thursday_night_pilot_shards as shards


def test_shards_partition_all_policy_brief_design_cells() -> None:
    complete = pilot.episode_schedule()
    grouped = [shards.rows_for(index) for index in range(shards.SHARDS)]
    flattened = [row for group in grouped for row in group]
    assert len(flattened) == len(set(flattened)) == 140
    assert set(flattened) == set(complete)
    assert max(map(len, grouped)) - min(map(len, grouped)) <= 1
    with pytest.raises(ValueError, match="shard"):
        shards.rows_for(shards.SHARDS)


def test_shard_manifest_rejects_changed_source_or_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "wrapper.py"
    script.write_text("original\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    launch = tmp_path / "sharded_launch.json"
    payload = {
        "schema_version": "thursday-night-pilot-shards-v1",
        "shards": shards.SHARDS,
        "wrapper_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "pilot_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "schedule_sha256": hashlib.sha256(
            _json_bytes(pilot.episode_schedule())
        ).hexdigest(),
    }
    launch.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(shards, "ROOT", tmp_path)
    monkeypatch.setattr(shards, "SCRIPT", script)
    monkeypatch.setattr(shards, "LAUNCH", launch)
    assert shards._manifest()["shards"] == shards.SHARDS
    script.write_text("changed\n", encoding="utf-8")
    with pytest.raises(pilot.PilotSafetyStop, match="source or schedule"):
        shards._manifest()


def _reserve(root: str, index: int, queue) -> None:
    """Each process owns a backend; only the file lock coordinates them."""
    class Backend:
        def execute(self, *args, **kwargs):
            return True

    counted = pilot.CountedBackend(
        Backend(), Path(root), "f" * 64, lambda: None, cap=3
    )
    counted.active_episode = f"episode_{index}"
    try:
        counted.execute(
            SimpleNamespace(id="a" * 64),
            SimpleNamespace(id="recipe"),
            run_id=f"run_{index}", run_dir=Path(root),
        )
    except pilot.PilotRunCap:
        queue.put("capped")
    else:
        queue.put("reserved")


def test_global_run_cap_survives_concurrent_processes(tmp_path: Path) -> None:
    ctx = get_context("fork")
    queue = ctx.Queue()
    workers = [
        ctx.Process(target=_reserve, args=(str(tmp_path), index, queue))
        for index in range(10)
    ]
    for worker in workers:
        worker.start()
    outcomes = [queue.get(timeout=10) for _ in workers]
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0
    assert outcomes.count("reserved") == 3
    assert outcomes.count("capped") == 7
    saved = json.loads((tmp_path / "new_real_runs.json").read_text())
    assert len(saved["reservations"]) == 3

