"""Approved meter-version aliasing must preserve data and charged cost."""
from __future__ import annotations

from pathlib import Path

import pytest

from orch.contract.models import (
    Claims, Evidence, MetricName, MetricValue, RunOutcome, RunRecord, RunStatus,
)
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.meter_equivalence import (
    CURRENT_METER, OLD_TEST_METER, MeterEquivalence,
    approved_test_equivalence, canonical_evidence_id, promote_evidence,
)
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes
from orch.interfaces import RecipeSpec
from scripts import promote_test_cache_meter as promotion


def evidence(state_id: str, version: str) -> Evidence:
    """Create a canonical, real-labelled fixture with no hardware calls."""
    result = Evidence(
        id="pending", state_id=state_id,
        metrics={
            MetricName.WNS_NS: MetricValue(
                value=-0.5, unit="ns", method="fixture", basis="routed",
            ),
            MetricName.PACKED_LUTS: MetricValue(
                value=120, unit="count", method="fixture", basis="routed",
            ),
        },
        routed=True, route_errors=0, drc_errors=0,
        constraint_fingerprint="fixture-xdc", clock_digest="fixture-clock",
        meter_version=version, fake=False,
    )
    return result.model_copy(update={"id": canonical_evidence_id(result)})


class NoToolBackend:
    """Fail if a promoted cache hit reaches the backend."""

    def measure_start(self, *args, **kwargs):
        raise AssertionError("unexpected start meter call")

    def execute(self, *args, **kwargs):
        raise AssertionError("unexpected recipe or meter call")


def test_only_the_audited_test_meter_pair_is_approved() -> None:
    approved = approved_test_equivalence()
    assert (approved.old_version, approved.new_version) == (
        OLD_TEST_METER, CURRENT_METER,
    )
    with pytest.raises(ValueError, match="approved old meter"):
        promote_evidence(evidence("a" * 64, "unapproved-meter"), approved)


def test_promoted_test_cache_hit_keeps_original_cost_and_uses_no_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise pointer promotion and an ordinary StepEngine cache hit."""
    root = tmp_path / "campaign"
    output = tmp_path / "promotion"
    monkeypatch.setattr(promotion, "ROOT", output)
    store = StateStore(root)
    cache = RunCache(root)
    approval = MeterEquivalence("old-meter", "new-meter", "a" * 64)

    start_file = tmp_path / "start.dcp"
    start_file.write_bytes(b"fixture-start")
    start = store.import_state(start_file, design="training-fixture")
    output_file = tmp_path / "output.dcp"
    output_file.write_bytes(b"fixture-output")
    child = store.import_state(
        output_file, design=start.design, parent_id=start.id,
        made_by_run_id="run_original",
    )
    recipe = RecipeSpec(
        id="fixture.recipe", worker="area", kind="tcl",
        script="fixture.tcl", replaces=True, cause="fixture",
        source="fixture", source_sha256="fixture", timeout_s=5,
        version="fixture-version",
    )
    original = RunOutcome(
        run=RunRecord(
            id="run_original", recipe_id=recipe.id,
            recipe_version=recipe.version, status=RunStatus.OK,
            wall_s=11.5, cpu_user_s=7.25, cpu_sys_s=1.5,
            peak_memory_bytes=123456, load_1m_start=9.0,
            load_1m_end=10.0, busy_slots_start=3,
            input_sha256_before=start.id, input_sha256_after=start.id,
        ),
        state=child,
        evidence=evidence(child.id, approval.old_version),
        claims=Claims(id="run_original:claims", run_id="run_original"),
    )
    store.save_outcome(original)
    key = cache.key(start.id, recipe.id, recipe.version)
    with cache.locked(key):
        cache.put(key, original.run.id)
    old_start = evidence(start.id, approval.old_version)
    old_folder = root / "measurements" / start.id / promotion.version_folder(
        approval.old_version
    )
    _atomic_bytes(
        old_folder / "evidence.json",
        _json_bytes(old_start.model_dump(mode="json")),
    )

    plan = promotion.prepare(root, approval)
    assert len(plan["pointers"]) == len(plan["starts"]) == 1
    _atomic_bytes(output / "plan.json", _json_bytes(plan))
    result = promotion.migrate(root, plan, approval, output)
    assert result["promoted_pointers"] == result["promoted_starts"] == 1
    alias = store.load_outcome(cache.get(key))
    assert alias.run.original_run_id == original.run.id
    assert alias.evidence.meter_version == approval.new_version
    assert store.load_outcome(original.run.id) == original
    assert (output / "cache_backups" / f"{key}.json").is_file()

    engine = StepEngine(
        store=store, cache=cache, backend=NoToolBackend(),
        meter_version=approval.new_version, cache_enabled=True,
    )
    assert engine.measure_start(start).id == promote_evidence(
        old_start, approval
    ).id
    hit = engine.execute(start, recipe)
    assert hit.run.cache_hit is True
    assert hit.run.original_run_id == alias.run.id
    assert hit.run.wall_s == original.run.wall_s
    assert hit.run.cpu_user_s == original.run.cpu_user_s
    assert hit.run.cpu_sys_s == original.run.cpu_sys_s
    assert hit.run.peak_memory_bytes == original.run.peak_memory_bytes
    assert hit.run.load_1m_start == original.run.load_1m_start
    assert hit.run.load_1m_end == original.run.load_1m_end
    assert hit.run.busy_slots_start == original.run.busy_slots_start
