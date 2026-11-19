"""Critical storage, locking and crash-resume behaviors."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from orch.contract.models import (
    Claims,
    DesignState,
    Evidence,
    MetricName,
    MetricValue,
    RunOutcome,
    RunRecord,
    RunStatus,
)
from orch.engine.cache import RunCache
from orch.engine.journal import Journal
from orch.engine.store import StateStore


def _fake_outcome(store: StateStore, run_id: str) -> RunOutcome:
    run_dir = store.runs_dir / run_id
    run_dir.mkdir(parents=True)
    output = run_dir / "out.dcp"
    output.write_bytes(f"fake checkpoint {run_id}".encode())
    state = store.import_state(
        output, design="fixture", parent_id="a" * 64,
        made_by_run_id=run_id, fake=True,
    )
    return RunOutcome(
        run=RunRecord(
            id=run_id, recipe_id="A.fixture", recipe_version="v1",
            status=RunStatus.OK, fake=True,
            input_sha256_before="a" * 64, input_sha256_after="a" * 64,
        ),
        state=state,
        evidence=Evidence(
            id=f"e-{run_id}", state_id=state.id,
            metrics={
                MetricName.WNS_NS: MetricValue(
                    value=-0.7, unit="ns", method="fake-meter", basis="routed"
                ),
                MetricName.PACKED_LUTS: MetricValue(
                    value=100, unit="count", method="fake-meter", basis="routed"
                ),
            },
            routed=True, route_errors=0, drc_errors=0,
            constraint_fingerprint="xdc", clock_digest="clock",
            meter_version="fake-v1", fake=True,
        ),
        claims=Claims(id=f"c-{run_id}", run_id=run_id, fake=True),
    )


def _contending_caller(
    root: str, key: str, marker: str, gate: multiprocessing.synchronize.Event,
    ready: multiprocessing.queues.Queue, results: multiprocessing.queues.Queue,
) -> None:
    cache = RunCache(root)
    ready.put(1)
    gate.wait(timeout=5)
    with cache.locked(key):
        hit = cache.get(key)
        if hit is None:
            with open(marker, "a", encoding="ascii") as handle:
                handle.write("computed\n")
                handle.flush()
                os.fsync(handle.fileno())
            time.sleep(0.3)
            store = StateStore(root)
            store.save_outcome(_fake_outcome(store, "shared-run"))
            cache.put(key, "shared-run")
            hit = "shared-run"
    results.put(hit)


def test_external_state_is_copied_private_run_output_is_hardlinked(tmp_path: Path) -> None:
    shared = tmp_path / "shared" / "start.dcp"
    shared.parent.mkdir()
    shared.write_bytes(b"B2 checkpoint bytes")
    store = StateStore(tmp_path / "project" / "data")

    start = store.import_state(shared, design="fixture")
    canonical = store.state_path(start.id)
    assert canonical.read_bytes() == shared.read_bytes()
    assert canonical.stat().st_ino != shared.stat().st_ino
    shared.write_bytes(b"changed source outside store")
    assert canonical.read_bytes() == b"B2 checkpoint bytes"

    output = store.runs_dir / "r1" / "out.dcp"
    output.parent.mkdir()
    output.write_bytes(b"new private checkpoint")
    state = store.import_state(
        output, design="fixture", parent_id=start.id, made_by_run_id="r1"
    )
    assert store.state_path(state.id).stat().st_ino == output.stat().st_ino


def test_same_hash_keeps_first_canonical_provenance_but_each_occurrence_lineage(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "data")
    first_out = store.runs_dir / "r1" / "out.dcp"
    second_out = store.runs_dir / "r2" / "out.dcp"
    first_out.parent.mkdir()
    second_out.parent.mkdir()
    first_out.write_bytes(b"unchanged selected output")
    second_out.write_bytes(b"unchanged selected output")

    first = store.import_state(
        first_out, design="fixture", parent_id="a" * 64, made_by_run_id="r1"
    )
    second = store.import_state(
        second_out, design="fixture", parent_id="b" * 64, made_by_run_id="r2"
    )
    assert first.id == second.id
    assert first.parent_id == "a" * 64
    assert second.parent_id == "b" * 64
    canonical = store.state_path(first.id)
    assert canonical.stat().st_ino == second_out.stat().st_ino
    metadata = json.loads((canonical.parent / "state.json").read_text())
    assert metadata["parent_id"] == "a" * 64
    assert metadata["made_by_run_id"] == "r1"


def test_outcome_roundtrip_and_unknown_state_rejected(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data")
    outcome = _fake_outcome(store, "r1")
    store.save_outcome(outcome)
    assert store.load_outcome("r1") == outcome
    store.save_outcome(outcome)  # Idempotent after a completed write.

    unknown = DesignState(
        id="f" * 64, design="fixture", path=str(store.states_dir / ("f" * 64) / "design.dcp"),
        fake=True,
    )
    bad = RunOutcome(
        run=RunRecord(id="r2", recipe_id="A.fixture", recipe_version="v1",
                      status=RunStatus.NO_OUTPUT, fake=True),
        state=unknown,
    )
    with pytest.raises(FileNotFoundError):
        store.save_outcome(bad)


def test_incomplete_outcome_is_not_loadable_then_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orch.engine.store as store_module

    store = StateStore(tmp_path / "data")
    outcome = _fake_outcome(store, "r1")
    original = store_module._atomic_bytes

    def fail_commit(path: Path, content: bytes) -> None:
        if path.name == "complete.json":
            raise RuntimeError("simulated crash before completion marker")
        original(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "_atomic_bytes", fail_commit)
        with pytest.raises(RuntimeError, match="simulated crash"):
            store.save_outcome(outcome)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        store.load_outcome("r1")
    store.save_outcome(outcome)
    assert store.load_outcome("r1") == outcome


def test_cache_hit_miss_and_retryable_record(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data")
    for run_id in ("r1", "r2"):
        store.save_outcome(_fake_outcome(store, run_id))
    cache = RunCache(store.root)
    key = cache.key("a" * 64, "A.fixture", "v1")
    assert cache.get(key) is None
    with cache.locked(key):
        assert cache.get(key) is None
        cache.put(key, "r1")
    assert cache.get(key) == "r1"
    with cache.locked(key):
        cache.put(key, "r2", retryable=True)
    assert cache.get(key) is None
    assert cache.last_attempt(key) == "r2"


def test_two_processes_compute_one_cache_key_once(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("fork")
    root = str(tmp_path / "data")
    key = RunCache(root).key("a" * 64, "A.fixture", "v1")
    marker = str(tmp_path / "computed.txt")
    gate = ctx.Event()
    ready = ctx.Queue()
    results = ctx.Queue()
    workers = [
        ctx.Process(target=_contending_caller, args=(root, key, marker, gate, ready, results))
        for _ in range(2)
    ]
    for process in workers:
        process.start()
    for _ in workers:
        assert ready.get(timeout=5) == 1
    gate.set()
    for process in workers:
        process.join(timeout=10)
        assert process.exitcode == 0
    assert [results.get(timeout=2) for _ in workers] == ["shared-run", "shared-run"]
    assert Path(marker).read_text().splitlines() == ["computed"]


def test_cache_atomic_replace_keeps_old_entry_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import orch.engine.store as store_module

    store = StateStore(tmp_path / "data")
    for run_id in ("old-run", "new-run"):
        store.save_outcome(_fake_outcome(store, run_id))
    cache = RunCache(store.root)
    key = cache.key("a" * 64, "A.fixture", "v1")
    with cache.locked(key):
        cache.put(key, "old-run")
    original_replace = store_module.os.replace

    def fail_replace(source: str | Path, target: str | Path) -> None:
        if Path(target).name == f"{key}.json":
            raise OSError("simulated crash during cache publish")
        original_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(store_module.os, "replace", fail_replace)
        with cache.locked(key):
            with pytest.raises(OSError, match="simulated crash"):
                cache.put(key, "new-run")
    assert cache.get(key) == "old-run"
    assert list(cache.cache_dir.glob(f".{key}.json.*.tmp")) == []


def test_journal_resumes_after_torn_last_line_and_is_idempotent(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "data")
    journal.append("ep1", {"id": "step-1", "run_id": "r1", "fake": True})
    with journal.path("ep1").open("ab") as handle:
        handle.write(b'{"event":{"id":"torn"')
        handle.flush()
        os.fsync(handle.fileno())

    assert [event["id"] for event in journal.read("ep1")] == ["step-1"]
    journal.append("ep1", {"id": "step-2", "run_id": "r2", "fake": True})
    journal.append("ep1", {"id": "step-2", "run_id": "r2", "fake": True})
    assert [event["id"] for event in journal.read("ep1")] == ["step-1", "step-2"]
    raw = journal.path("ep1").read_bytes()
    assert raw.endswith(b"\n")
    assert len(raw.splitlines()) == 2


def test_journal_detects_corrupt_complete_line(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "data")
    journal.append("ep1", {"id": "step-1"})
    with journal.path("ep1").open("ab") as handle:
        handle.write(b'{"event":{"id":"bad"},"sha256":"wrong"}\n')
    with pytest.raises(ValueError, match="checksum"):
        journal.read("ep1")



def test_cache_refuses_incomplete_run_and_dangling_terminal_pointer(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data")
    cache = RunCache(store.root)
    key = cache.key("a" * 64, "A.fixture", "v1")
    with cache.locked(key):
        with pytest.raises(FileNotFoundError, match="complete before caching"):
            cache.put(key, "r1")
    store.save_outcome(_fake_outcome(store, "r1"))
    with cache.locked(key):
        cache.put(key, "r1")
    assert cache.get(key) == "r1"
    (store.runs_dir / "r1" / "complete.json").unlink()
    with pytest.raises(ValueError, match="dangling"):
        cache.get(key)


def test_state_path_refuses_incomplete_state_metadata(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "data")
    output = store.runs_dir / "r1" / "out.dcp"
    output.parent.mkdir()
    output.write_bytes(b"checkpoint")
    state = store.import_state(output, design="fixture")
    (store.states_dir / state.id / "state.json").unlink()
    with pytest.raises(FileNotFoundError, match="incomplete"):
        store.state_path(state.id)



def test_cache_key_changes_with_state_recipe_and_version(tmp_path: Path) -> None:
    cache = RunCache(tmp_path / "data")
    base = cache.key("a" * 64, "A.fixture", "v1")
    assert cache.key("b" * 64, "A.fixture", "v1") != base
    assert cache.key("a" * 64, "B.fixture", "v1") != base
    assert cache.key("a" * 64, "A.fixture", "v2") != base
    assert cache.key("a" * 64, "A.fixture", "v1") == base

