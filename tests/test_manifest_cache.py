"""A manifest lookup returns the stored outcome, charges its original cost, runs no tool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orch.contract.models import (
    DesignState, Evidence, MetricName, MetricValue, RunOutcome, RunRecord, RunStatus,
)
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.manifest_cache import (
    COST_FIELDS, SCHEMA, CacheManifest, ManifestRunCache, ManifestStateStore,
    ManifestStepEngine, manifest_hash,
)
from orch.engine.store import StateStore, _json_bytes, _sha256_file
from orch.interfaces import RecipeSpec

METER = "unit-meter-v1"
RECIPE = RecipeSpec(
    id="unit_recipe", worker="area", kind="tcl", script="unit/unit_recipe", replaces=False,
    cause="unit", source="unit", source_sha256="unit", timeout_s=10, version="unit-recipe-v1",
)


def _evidence(state: DesignState, meter: str = METER) -> Evidence:
    return Evidence(
        id=hashlib.sha256(f"{state.id}:{meter}".encode()).hexdigest(), state_id=state.id,
        metrics={
            MetricName.WNS_NS: MetricValue(value=-0.5, unit="ns", method="unit", basis="routed"),
            MetricName.PACKED_LUTS: MetricValue(value=90.0, unit="count", method="unit", basis="routed"),
        },
        routed=True, route_errors=0, drc_errors=0, constraint_fingerprint="xdc",
        clock_digest="clk", meter_version=meter, fake=True,
    )


class StoredRunBackend:
    """Produces the stored run in the source root, with a distinctive original cost."""

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def execute(self, state, recipe, *, run_id, run_dir):
        out = run_dir / "out.dcp"
        out.write_bytes(f"{state.id}:{recipe.id}".encode())
        succ = self.store.import_state(out, design="unit", parent_id=state.id,
                                       made_by_run_id=run_id, fake=True)
        run = RunRecord(
            id=run_id, recipe_id=recipe.id, recipe_version=recipe.version, status=RunStatus.OK,
            exit_code=0, wall_s=777.5, cpu_user_s=1234.5, input_sha256_before=state.id,
            input_sha256_after=state.id, fake=True,
        )
        return RunOutcome(run=run, state=succ, evidence=_evidence(succ))

    def measure_start(self, state, *, run_dir):
        return _evidence(state)


class NoToolBackend:
    """Any call is a tool run; the test fails if one happens."""

    calls = 0

    def execute(self, *a, **k):
        NoToolBackend.calls += 1
        raise AssertionError("a tool run was attempted")

    def measure_start(self, *a, **k):
        NoToolBackend.calls += 1
        raise AssertionError("a start measurement was attempted")


def _tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _build(tmp_path: Path):
    src = tmp_path / "source_root"
    store = StateStore(src)
    start_file = tmp_path / "start.dcp"
    start_file.write_bytes(b"unit-start")
    start = store.import_state(start_file, design="unit", fake=True)
    engine = StepEngine(store=store, cache=RunCache(src), backend=StoredRunBackend(store),
                        meter_version=METER, cache_enabled=True)
    stored = engine.execute(start, RECIPE)
    key = RunCache.key(start.id, RECIPE.id, RECIPE.version)
    run_dir = src / "runs" / stored.run.id
    start_evidence_path = src / "measurements" / start.id / "unit" / "evidence.json"
    start_evidence_path.parent.mkdir(parents=True)
    start_evidence_path.write_bytes(_json_bytes(_evidence(start).model_dump(mode="json")))
    body = {
        "schema_version": SCHEMA, "meter_version": METER, "roots": {"src": str(src)},
        "entries": {key: {
            "root": "src", "run_id": stored.run.id,
            "complete_sha256": _sha256_file(run_dir / "complete.json"),
            "recipe_id": RECIPE.id, "recipe_version": RECIPE.version,
            "input_state_id": start.id, "output_state_id": stored.state.id,
            "meter_version": METER, "cost": {f: getattr(stored.run, f) for f in COST_FIELDS},
            "approval": {"kind": "native_current_meter"},
        }},
        "states": {start.id: [str(src / "states" / start.id / "design.dcp")],
                   stored.state.id: [str(src / "states" / stored.state.id / "design.dcp")]},
        "start_evidence": {start.id: {"path": str(start_evidence_path), "meter_version": METER}},
    }
    return src, body, start_file, stored, key


def _campaign(tmp_path: Path, body, start_file, *, meter=METER):
    manifest = CacheManifest(body, expected_hash=manifest_hash(body))
    root = tmp_path / "campaign_root"
    store = ManifestStateStore(root, manifest)
    backend = NoToolBackend()
    engine = ManifestStepEngine(
        manifest=manifest, require_zero_tool=True, store=store,
        cache=ManifestRunCache(root, manifest, meter_version=meter), backend=backend,
        meter_version=meter, cache_enabled=True,
    )
    imported = store.import_state(start_file, design="unit", fake=True)
    # one deterministic location per state: the manifest's first, so the start uses that path
    return engine, store, backend, imported.model_copy(update={"path": str(store.state_path(imported.id))})


def test_manifest_hit_returns_stored_outcome_charges_original_cost_runs_no_tool(tmp_path: Path) -> None:
    src, body, start_file, stored, key = _build(tmp_path)
    before = _tree(src)
    NoToolBackend.calls = 0
    engine, store, backend, start = _campaign(tmp_path, body, start_file)
    start_evidence = engine.measure_start(start)          # from the manifest, no meter
    assert start_evidence.state_id == start.id
    result = engine.execute(start, RECIPE)
    assert NoToolBackend.calls == 0
    assert result.run.cache_hit is True
    assert result.run.original_run_id == stored.run.id
    assert result.run.wall_s == 777.5 and result.run.cpu_user_s == 1234.5   # original cost
    assert result.state.id == stored.state.id and result.evidence.id == stored.evidence.id
    assert Path(result.state.path) == src / "states" / stored.state.id / "design.dcp"
    assert engine.execute(start, RECIPE).run.id != result.run.id            # each call its own run
    assert _tree(src) == before                                             # source root untouched
    assert not (store.root / "states" / stored.state.id).exists()           # nothing copied or linked
    assert store.load_outcome(result.run.id).state.id == stored.state.id    # resumable from own record


def test_changed_source_record_or_manifest_is_refused(tmp_path: Path) -> None:
    src, body, start_file, stored, key = _build(tmp_path)
    with pytest.raises(ValueError):
        CacheManifest(body, expected_hash="0" * 64)
    engine, _store, _backend, start = _campaign(tmp_path, body, start_file)
    marker = src / "runs" / stored.run.id / "complete.json"
    marker.write_bytes(marker.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed since the manifest"):
        engine.execute(start, RECIPE)


def test_other_meter_version_is_a_miss_that_would_start_a_tool(tmp_path: Path) -> None:
    _src, body, start_file, _stored, _key = _build(tmp_path)
    NoToolBackend.calls = 0
    engine, _store, _backend, start = _campaign(tmp_path, body, start_file, meter="other-meter")
    with pytest.raises(AssertionError, match="tool run"):
        engine.execute(start, RECIPE)
    assert NoToolBackend.calls == 1


def test_start_without_stored_evidence_refuses_to_start_the_meter(tmp_path: Path) -> None:
    _src, body, start_file, _stored, _key = _build(tmp_path)
    body = dict(body, start_evidence={})
    NoToolBackend.calls = 0
    engine, _store, _backend, start = _campaign(tmp_path, body, start_file)
    with pytest.raises(RuntimeError, match="none is allowed"):
        engine.measure_start(start)
    assert NoToolBackend.calls == 0
