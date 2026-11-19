"""Software checks for the worker-fitting phase driver and strict cache import."""
from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from scripts import launch_worker_fitting_train as launcher

from orch.baselines.cache_seed import seed_first_steps
from orch.baselines.worker_fitting import (
    AREA_IDS, TIMING_IDS, TRAIN_DESIGNS, WorkerFittingCampaign,
)
from orch.config import load_brief
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.store import StateStore
from orch.fake.backend import FakeBackend
from orch.recipes.registry import load_registry

ROOT = Path(__file__).resolve().parents[1]


def _fake_engine(root: Path) -> tuple[StepEngine, FakeBackend]:
    store = StateStore(root)
    backend = FakeBackend(store=store)
    engine = StepEngine(
        store=store, cache=RunCache(root), backend=backend,
        meter_version=backend.meter_version, cache_enabled=True,
    )
    return engine, backend


def _campaign(engine: StepEngine, start):
    registry = load_registry(ROOT)
    sets = {
        "timing": tuple(registry.get(r) for r in TIMING_IDS[:2]),
        "area": tuple(registry.get(r) for r in (AREA_IDS[0], AREA_IDS[-1])),
    }
    briefs = {
        name: load_brief(ROOT / "configs" / "briefs" / f"{name}.yaml")
        for name in ("timing", "area")
    }
    return WorkerFittingCampaign(
        engine=engine, starts={start.design: start},
        recipe_sets=sets, briefs=briefs, max_workers=4,
        campaign_id="fake_worker_fitting",
        depth_study_design=start.design,
    )


def test_phase_counts_and_resume_use_one_run_per_edge(tmp_path: Path) -> None:
    engine, backend = _fake_engine(tmp_path / "campaign")
    start = backend.create_start("vexriscv_re-place")
    campaign = _campaign(engine, start)
    result = campaign.run()
    assert result["D1"]["planned_sequences"] == 4
    assert result["D2"]["planned_sequences"] == 16
    assert result["D3"]["planned_sequences"] == 16
    assert result["D4"]["planned_sequences"] == 60
    assert result["D4"]["reused_prior_phase"] == 28
    assert len(campaign.index) == 68
    assert all(edge.fake for edge in campaign.index.values())
    assert all(
        set(edge.verdict_status) == {"brief_t", "brief_a"}
        for edge in campaign.index.values() if edge.run_id is not None
    )
    completed_before = sorted(engine.store.root.glob("runs/*/complete.json"))
    resumed = _campaign(engine, start)
    assert resumed.run() == result
    assert sorted(engine.store.root.glob("runs/*/complete.json")) == completed_before


def test_resume_refuses_corrupt_completed_run(tmp_path: Path) -> None:
    engine, backend = _fake_engine(tmp_path / "corrupt")
    start = backend.create_start("vexriscv_re-place")
    campaign = _campaign(engine, start)
    campaign.run()
    first = next(edge for edge in campaign.index.values() if edge.run_id)
    path = engine.store.root / "runs" / first.run_id / "runinfo.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        _campaign(engine, start).run()


def test_seed_imports_only_exact_version_and_start(tmp_path: Path) -> None:
    source_engine, source_backend = _fake_engine(tmp_path / "old")
    old_start = source_backend.create_start("vexriscv_re-place")
    recipe = load_registry(ROOT).get(TIMING_IDS[0])
    source_engine.measure_start(old_start)
    original = source_engine.execute(old_start, recipe)

    destination_engine, _ = _fake_engine(tmp_path / "new")
    new_start = destination_engine.store.import_state(
        Path(old_start.path), design=old_start.design, fake=True,
    )
    report = seed_first_steps(
        engine=destination_engine, source_root=source_engine.store.root,
        starts={new_start.design: new_start}, recipes=(recipe,),
    )
    assert report["seeded_run_count"] == 1
    assert report["seeded_start_designs"] == [new_start.design]
    assert destination_engine.measure_start(new_start).id == source_engine.measure_start(old_start).id
    reused = destination_engine.execute(new_start, recipe)
    assert reused.run.cache_hit is True
    assert reused.run.original_run_id == original.run.id
    assert reused.state is not None
    assert Path(reused.state.path).is_relative_to(destination_engine.store.root)


def test_train_allowlist_and_recipe_sets_are_exact() -> None:
    assert TRAIN_DESIGNS == (
        "vexriscv_re-place", "rosetta_spam-filter", "rosetta_3d-rendering",
        "logicnets_jscl", "finn_radioml",
    )
    assert len(TIMING_IDS) == len(AREA_IDS) == 5
    assert not set(TIMING_IDS) & set(AREA_IDS)



def test_launcher_plan_rejects_any_design_outside_train(
    tmp_path: Path, monkeypatch,
) -> None:
    """The real launch surface has a closed five-checkpoint allowlist."""

    hashes = {}
    for design in TRAIN_DESIGNS:
        content = design.encode("ascii")
        (tmp_path / f"{design}_B2.dcp").write_bytes(content)
        hashes[design] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(launcher, "B2", tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED_SHA", hashes)
    plan = launcher._plan()
    assert [row["design"] for row in plan["train_inputs"]] == list(TRAIN_DESIGNS)
    assert plan["D1_planned_edges"] == 50
    assert plan["D2_planned_edges"] == 500
    assert plan["D3_max_new_edges"] == 200
    assert plan["D4_total_within_worker_edges"] == 1560
    monkeypatch.setattr(launcher, "EXPECTED_SHA", {**hashes, "unselected": "0" * 64})
    with pytest.raises(ValueError, match="train allowlist"):
        launcher._plan()
