"""Software-only checks for the CSV-seeded, fully labelled fake backend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from orch.config import load_brief
from orch.contract.models import MetricName, RunStatus
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.store import StateStore
from orch.fake.backend import FakeBackend, METRIC_METHOD, SEED_SHA256
from orch.recipes.registry import load_registry


PROJECT = Path(__file__).resolve().parents[1]


def _stack(tmp_path: Path):
    store = StateStore(tmp_path / "data")
    backend = FakeBackend(store=store)
    engine = StepEngine(
        store=store,
        cache=RunCache(store.root),
        backend=backend,
        meter_version=backend.meter_version,
        cache_enabled=True,
    )
    registry = load_registry(PROJECT)
    return store, backend, engine, registry


def test_seed_copy_and_all_eight_b2_references(tmp_path: Path) -> None:
    """The fake baseline is traceable even where no direct recipe row exists."""

    copied = PROJECT / "data/fake_seed/combined_results.csv"
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == SEED_SHA256
    _, backend, engine, _ = _stack(tmp_path)
    main = {
        "rosetta_3d-rendering", "corescore_500_mod",
        "rosetta_spam-filter", "vexriscv_re-place",
        "vexriscv_re-place_v2", "finn_radioml",
        "vtr_mcml", "rosetta_digit-recognition",
    }
    assert main.issubset(backend.designs)
    states = [backend.create_start(design) for design in sorted(main)]
    assert len({state.id for state in states}) == len(main)
    assert all(state.fake for state in states)
    for state in states:
        evidence = engine.measure_start(state)
        assert evidence.fake
        assert evidence.routed and evidence.drc_errors == 0
        assert "fake" in evidence.metrics[MetricName.WNS_NS].method
        assert MetricName.TNS_NS not in evidence.metrics
        assert MetricName.NETLIST_LUTS not in evidence.metrics


def test_direct_vex_phys_opt_equals_seed_row_and_gate_can_judge(tmp_path: Path) -> None:
    """A direct B2 row is reproduced exactly, without claiming a real run."""

    _, backend, engine, registry = _stack(tmp_path)
    start_state = backend.create_start("vexriscv_re-place")
    start = engine.measure_start(start_state)
    recipe = registry.get("LATCH.phys_opt")
    assert backend.available_direct(start_state.design, recipe)
    brief = load_brief(PROJECT / "configs/briefs/timing.yaml")
    applied = engine.apply(
        start_state, recipe, brief=brief, start=start, parent=start
    )
    assert start.metric(MetricName.WNS_NS).value == -0.785
    assert start.metric(MetricName.PACKED_LUTS).value == 1895
    assert applied.outcome.run.status == RunStatus.OK
    assert applied.outcome.run.fake
    assert applied.outcome.state.fake
    assert applied.outcome.evidence.fake
    assert applied.outcome.claims.fake
    assert applied.verdict.fake and applied.verdict.brief_valid
    assert applied.outcome.evidence.metric(MetricName.WNS_NS).value == -0.736
    assert applied.outcome.evidence.metric(MetricName.PACKED_LUTS).value == 1895
    assert applied.outcome.claims.values["effect_kind"] == "direct_B2_seed_row"
    assert applied.outcome.claims.values["seed_run_id"] == (
        "a_phys_opt_vexriscv_re-place"
    )
    assert all(
        metric.method == METRIC_METHOD
        for metric in applied.outcome.evidence.metrics.values()
    )
    sidecar = json.loads(
        Path(applied.outcome.evidence.raw_report_path).read_text()
    )
    assert sidecar["fake"] is True
    assert sidecar["hardware_measurement"] is False
    assert sidecar["steps"][-1]["seed_row_line"] > 0
    assert sidecar["steps"][-1]["result_kind"] == "direct_B2_seed_row"


def test_unobserved_pair_is_explicit_synthetic_neutral(tmp_path: Path) -> None:
    """CoreScore has a B2 baseline but no eligible direct B2 recipe row."""

    _, backend, engine, registry = _stack(tmp_path)
    state = backend.create_start("corescore_500_mod")
    start = engine.measure_start(state)
    recipe = registry.get("A_E_DUPL")
    assert not backend.available_direct(state.design, recipe)
    brief = load_brief(PROJECT / "configs/briefs/area.yaml")
    applied = engine.apply(state, recipe, brief=brief, start=start, parent=start)
    assert applied.outcome.evidence.metrics == start.metrics
    assert applied.outcome.state.id != state.id
    assert applied.outcome.claims.values["effect_kind"] == "synthetic_neutral_missing"
    sidecar = json.loads(
        Path(applied.outcome.evidence.raw_report_path).read_text()
    )
    assert sidecar["steps"][-1]["missing_observation_assumption"]
    assert applied.verdict.fake and applied.verdict.brief_valid


def test_depth_two_is_reproducible_synthetic_composition(tmp_path: Path) -> None:
    """The second metric is a cited B2 delta plus a small synthetic interaction."""

    _, backend, engine, registry = _stack(tmp_path)
    state = backend.create_start("vexriscv_re-place")
    start = engine.measure_start(state)
    brief = load_brief(PROJECT / "configs/briefs/area.yaml")
    timing = registry.get("LATCH.phys_opt")
    area = registry.get("A_R_RSA")
    first = engine.apply(state, timing, brief=brief, start=start, parent=start)
    second = engine.apply(
        first.outcome.state, area, brief=brief,
        start=start, parent=first.outcome.evidence,
    )
    assert second.outcome.claims.values["effect_kind"] == "synthetic_composition"
    assert second.outcome.claims.values["component_kind"] == "direct_B2_seed_row"
    sidecar = json.loads(
        Path(second.outcome.evidence.raw_report_path).read_text()
    )
    step = sidecar["steps"][-1]
    assert step["seed_row_line"] > 0
    assert step["result_kind"] == "synthetic_composition"
    expected_wns = round(
        first.outcome.evidence.metric(MetricName.WNS_NS).value
        + step["delta_from_B2"]["wns_ns"]
        + step["interaction_from_previous_step"]["wns_ns"], 6
    )
    assert second.outcome.evidence.metric(MetricName.WNS_NS).value == expected_wns
    assert MetricName.NETLIST_LUTS not in second.outcome.evidence.metrics

    # Replaying the same sequence from the same bytes gives the same fake state.
    again = backend.execute(
        first.outcome.state, area,
        run_id="fake_repeat", run_dir=backend.store.root / "runs/fake_repeat",
    )
    assert again.state.id == second.outcome.state.id
    assert again.evidence.metrics == second.outcome.evidence.metrics

    reverse_first = engine.apply(state, area, brief=brief, start=start, parent=start)
    reverse_second = engine.apply(
        reverse_first.outcome.state, timing, brief=brief,
        start=start, parent=reverse_first.outcome.evidence,
    )
    assert reverse_second.outcome.state.id != second.outcome.state.id


def test_cache_hit_stays_fake_and_charges_original_synthetic_time(tmp_path: Path) -> None:
    """A cache hit is a separate run record and retains visible fake provenance."""

    _, backend, engine, registry = _stack(tmp_path)
    state = backend.create_start("vexriscv_re-place")
    start = engine.measure_start(state)
    recipe = registry.get("LATCH.phys_opt")
    brief = load_brief(PROJECT / "configs/briefs/timing.yaml")
    first = engine.apply(state, recipe, brief=brief, start=start, parent=start)
    second = engine.apply(state, recipe, brief=brief, start=start, parent=start)
    assert not first.outcome.run.cache_hit
    assert second.outcome.run.cache_hit
    assert second.outcome.run.original_run_id == first.outcome.run.id
    assert second.outcome.run.id != first.outcome.run.id
    assert second.outcome.run.wall_s == first.outcome.run.wall_s == 1.0
    assert second.outcome.run.fake
    assert second.outcome.claims.fake
    assert second.outcome.evidence.fake
    assert second.verdict.fake


def test_strict_missing_policy_fails_instead_of_inventing_an_effect(tmp_path: Path) -> None:
    """A caller can make unobserved pairs a hard fixture error."""

    store = StateStore(tmp_path / "data")
    backend = FakeBackend(store=store, missing_policy="error")
    registry = load_registry(PROJECT)
    state = backend.create_start("corescore_500_mod")
    with pytest.raises(KeyError, match="no direct B2 seed row"):
        backend.execute(
            state, registry.get("A_E_DUPL"),
            run_id="strict_missing",
            run_dir=store.root / "runs/strict_missing",
        )
