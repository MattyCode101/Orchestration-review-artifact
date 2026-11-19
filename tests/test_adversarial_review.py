"""Independent adversarial checks for the contract, gate and run cache.

These tests use only labelled fake data or in-memory evidence. They never
launch a hardware tool or read an external checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orch.config import load_brief
from orch.contract.models import (
    Evidence,
    Guard,
    MetricName,
    MetricValue,
    RunOutcome,
    VerdictStatus,
)
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.gate import judge
from orch.engine.store import StateStore
from orch.fake.backend import FakeBackend
from orch.recipes.registry import load_registry


PROJECT = Path(__file__).resolve().parents[1]


def _evidence(
    marker: str,
    *,
    wns: float,
    packed_luts: float,
    clock: str = "clock-start",
) -> Evidence:
    """Build physically comparable fake meter evidence for gate probes."""

    return Evidence(
        id=f"review-{marker}",
        state_id=marker * 64,
        metrics={
            MetricName.WNS_NS: MetricValue(
                value=wns, unit="ns", method="review-fake-meter", basis="fake-routed"
            ),
            MetricName.PACKED_LUTS: MetricValue(
                value=packed_luts,
                unit="count",
                method="review-fake-meter",
                basis="fake-routed",
            ),
        },
        routed=True,
        route_errors=0,
        drc_errors=0,
        constraint_fingerprint="xdc-start",
        clock_digest=clock,
        meter_version="review-meter-v1",
        fake=True,
    )


def _fake_engine(tmp_path: Path):
    """Use the production store/cache/engine with the labelled fake backend."""

    store = StateStore(tmp_path)
    cache = RunCache(tmp_path)
    backend = FakeBackend(store=store)
    engine = StepEngine(
        store=store,
        cache=cache,
        backend=backend,
        meter_version=backend.meter_version,
        cache_enabled=True,
    )
    start = backend.create_start("vexriscv_re-place")
    recipe = load_registry(PROJECT).get("A_R_EA")
    return store, cache, backend, engine, start, recipe


def test_gate_uses_start_even_when_parent_loss_is_small() -> None:
    """Two small parent losses must not evade the fixed brief floor."""

    start = _evidence("a", wns=-0.785, packed_luts=10_000)
    parent = _evidence("b", wns=-0.865, packed_luts=9_600)
    candidate = _evidence("c", wns=-0.945, packed_luts=9_400)
    brief = load_brief(PROJECT / "configs/briefs/area.yaml")
    verdict = judge(
        brief,
        start,
        parent,
        candidate,
        noise={MetricName.WNS_NS: 0.2},
    )
    assert verdict.physically_valid
    assert not verdict.brief_valid
    assert verdict.status == VerdictStatus.BRIEF_VIOLATION
    assert verdict.terms["wns_floor"].status == "fail"
    assert verdict.delta_from_parent[MetricName.WNS_NS] == pytest.approx(-0.08)
    assert verdict.delta_from_start[MetricName.WNS_NS] == pytest.approx(-0.16)
    assert verdict.within_noise[MetricName.WNS_NS] is True


def test_guard_failure_does_not_rewrite_brief_validity() -> None:
    """An absolute guard is separate from the start-relative brief."""

    start = _evidence("a", wns=-0.785, packed_luts=10_000)
    candidate = _evidence("c", wns=-0.70, packed_luts=10_080)
    brief = load_brief(PROJECT / "configs/briefs/timing.yaml")
    verdict = judge(
        brief,
        start,
        start,
        candidate,
        guards=(
            Guard(
                id="strict_wns",
                metric=MetricName.WNS_NS,
                comparison=">=",
                threshold=-0.65,
            ),
        ),
    )
    assert verdict.physically_valid and verdict.brief_valid
    assert not verdict.guard_pass
    assert verdict.status == VerdictStatus.GUARD_VIOLATION
    assert verdict.terms["packed_lut_cap"].status == "pass"


def test_gate_refuses_clock_change_even_for_better_objective() -> None:
    """A timing gain cannot promote a state with changed clocks."""

    start = _evidence("a", wns=-0.785, packed_luts=10_000)
    candidate = _evidence(
        "c", wns=-0.60, packed_luts=9_500, clock="different-clock"
    )
    brief = load_brief(PROJECT / "configs/briefs/timing.yaml")
    verdict = judge(brief, start, start, candidate)
    assert verdict.status == VerdictStatus.INVALID
    assert not verdict.physically_valid and not verdict.brief_valid
    assert verdict.validity["clocks"].status == "fail"


def test_cache_recipe_and_meter_versions_force_new_measurement(tmp_path: Path) -> None:
    """Version changes must not silently reuse an old measured outcome."""

    store, cache, backend, engine, start, recipe = _fake_engine(tmp_path)
    first = engine.execute(start, recipe)
    second = engine.execute(start, recipe)
    assert not first.run.cache_hit and second.run.cache_hit
    assert second.run.original_run_id == first.run.id
    changed_recipe = recipe.model_copy(update={"version": "changed-adapter-version"})
    changed = engine.execute(start, changed_recipe)
    assert not changed.run.cache_hit
    assert cache.key(start.id, recipe.id, recipe.version) != cache.key(
        start.id, recipe.id, changed_recipe.version
    )

    changed_backend = FakeBackend(store=store, interaction_fraction=0.25)
    changed_meter_engine = StepEngine(
        store=store,
        cache=cache,
        backend=changed_backend,
        meter_version=changed_backend.meter_version,
        cache_enabled=True,
    )
    remeasured = changed_meter_engine.execute(start, recipe)
    assert not remeasured.run.cache_hit
    assert remeasured.evidence is not None
    assert remeasured.evidence.meter_version == changed_backend.meter_version


def test_cached_run_rejects_recorded_input_mutation(tmp_path: Path) -> None:
    """A committed cache pointer must not bypass fresh-run input integrity."""

    store, cache, _, engine, start, recipe = _fake_engine(tmp_path)
    original = engine.execute(start, recipe)
    forged_run = original.run.model_copy(
        update={"id": "review_forged_input", "input_sha256_after": "f" * 64}
    )
    forged = RunOutcome(
        run=forged_run,
        state=original.state,
        evidence=original.evidence,
    )
    store.save_outcome(forged)
    key = cache.key(start.id, recipe.id, recipe.version)
    with cache.locked(key):
        cache.put(key, forged_run.id)
    with pytest.raises(ValueError, match="input|mutat"):
        engine.execute(start, recipe)


def test_run_outcome_rejects_mixed_fake_origin(tmp_path: Path) -> None:
    """A fake checkpoint/evidence may never be held by an unlabelled real run."""

    _, _, _, engine, start, recipe = _fake_engine(tmp_path)
    original = engine.execute(start, recipe)
    unlabelled_run = original.run.model_copy(update={"fake": False})
    with pytest.raises(ValueError, match="fake|origin|label"):
        RunOutcome(
            run=unlabelled_run,
            state=original.state,
            evidence=original.evidence,
        )


def test_metric_rejects_nonfinite_value() -> None:
    """NaN must not reach ranking, JSON persistence or a noise comparison."""

    with pytest.raises(ValueError):
        MetricValue(
            value=float("nan"),
            unit="ns",
            method="review-fake-meter",
            basis="fake-routed",
        )


def test_cache_hit_checks_input_checkpoint_bytes(tmp_path: Path) -> None:
    """A cache pointer cannot make a corrupted content-addressed input valid."""

    _, _, _, engine, start, recipe = _fake_engine(tmp_path)
    first = engine.execute(start, recipe)
    assert not first.run.cache_hit
    checkpoint = Path(start.path)
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(b"corrupted fake checkpoint")
    with pytest.raises(ValueError, match="hash|content|state"):
        engine.execute(start, recipe)


def test_cached_start_evidence_checks_input_checkpoint_bytes(tmp_path: Path) -> None:
    """Cached baseline evidence must still name the bytes now at its state path."""

    _, _, _, engine, start, _ = _fake_engine(tmp_path)
    first = engine.measure_start(start)
    assert first.state_id == start.id
    checkpoint = Path(start.path)
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(b"corrupted fake checkpoint")
    with pytest.raises(ValueError, match="hash|content|state"):
        engine.measure_start(start)


def test_gate_limits_reject_nonfinite_values() -> None:
    """NaN bounds cannot define a brief term or a mandate guard."""

    from orch.contract.models import BriefTerm

    with pytest.raises(ValueError):
        BriefTerm(
            id="bad_term",
            metric=MetricName.WNS_NS,
            comparison=">=",
            factor=float("nan"),
        )
    with pytest.raises(ValueError):
        Guard(
            id="bad_guard",
            metric=MetricName.WNS_NS,
            comparison=">=",
            threshold=float("nan"),
        )
