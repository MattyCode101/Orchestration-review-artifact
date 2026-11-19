"""Training-cache aliases must be audited and produce ordinary no-tool hits."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from orch.contract.models import (
    Claims, Evidence, MetricName, MetricValue, RunOutcome, RunRecord, RunStatus,
)
from orch.engine.apply import StepEngine
from orch.engine.cache import RunCache
from orch.engine.meter_equivalence import MeterEquivalence, canonical_evidence_id
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes, _sha256_file
from orch.interfaces import RecipeSpec
from scripts import promote_training_cache_meter as promotion


class NoToolBackend:
    """Any fallback to a real backend makes this test fail."""

    def measure_start(self, *args, **kwargs):
        raise AssertionError("unexpected measurement job")

    def execute(self, *args, **kwargs):
        raise AssertionError("unexpected recipe or meter job")


def evidence(state_id: str, version: str) -> Evidence:
    result = Evidence(
        id="pending", state_id=state_id,
        metrics={
            MetricName.WNS_NS: MetricValue(value=-0.5, unit="ns", method="fixture", basis="routed"),
            MetricName.PACKED_LUTS: MetricValue(value=120, unit="count", method="fixture", basis="routed"),
        },
        routed=True, route_errors=0, drc_errors=0,
        constraint_fingerprint="fixture-xdc", clock_digest="fixture-clock",
        v_props={name: 0 for name in promotion.PROPERTIES},
        meter_version=version, fake=False,
    )
    return result.model_copy(update={"id": canonical_evidence_id(result)})


def fixture(tmp_path: Path):
    source_root, target_root = tmp_path / "training", tmp_path / "v4"
    source_store, target_store = StateStore(source_root), StateStore(target_root)
    source_cache, target_cache = RunCache(source_root), RunCache(target_root)
    start_file, output_file = tmp_path / "start.dcp", tmp_path / "output.dcp"
    start_file.write_bytes(b"saved training B2")
    output_file.write_bytes(b"saved training output")
    start = source_store.import_state(start_file, design="training-fixture")
    origin_id = "run_" + "a" * 32
    child = source_store.import_state(
        output_file, design=start.design, parent_id=start.id,
        made_by_run_id=origin_id,
    )
    spec = RecipeSpec(
        id="fixture.recipe", worker="area", kind="tcl",
        script="fixture.tcl", replaces=True, cause="fixture",
        source="fixture", source_sha256="fixture", timeout_s=5,
        version="recipe-v1",
    )
    original = RunOutcome(
        run=RunRecord(
            id=origin_id, recipe_id=spec.id, recipe_version=spec.version,
            status=RunStatus.OK, input_sha256_before=start.id,
            input_sha256_after=start.id, wall_s=11.5, cpu_user_s=7.25,
            cpu_sys_s=1.5, peak_memory_bytes=123456,
            load_1m_start=9.0, load_1m_end=10.0, busy_slots_start=3,
        ),
        state=child, evidence=evidence(child.id, "old-meter"),
        claims=Claims(id=f"{origin_id}:claims", run_id=origin_id),
    )
    source_store.save_outcome(original)
    key = source_cache.key(start.id, spec.id, spec.version)
    with source_cache.locked(key):
        source_cache.put(key, origin_id)

    verifier = source_root / "verification"
    fingerprint = "c" * 64
    _atomic_bytes(verifier / "plan.json", _json_bytes({
        "fingerprint": fingerprint, "references": {start.design: start.id},
    }))
    verification_run = verifier / "runs" / "verify_fixture"
    report = verification_run / "validator_work" / "validation_report.json"
    part = {
        "passed": True, "mismatch_count": 0,
        "protocol_mismatch_count": 0,
    }
    raw = {
        "overall_passed": True, "phase1_passed": True,
        "phase2_passed": True, "infrastructure_failure": False,
        "num_vectors": 200, "precheck_vectors": 100,
        "simulation_report": {
            "precheck_report": {**part, "cycles_simulated": 100},
            "full_report": {**part, "cycles_simulated": 200},
        },
    }
    _atomic_bytes(report, _json_bytes(raw))
    _atomic_bytes(verification_run / "validate.log", b"Outputs (1):\n")
    record = {
        "schema_version": "worker-fitting-equivalence-v1",
        "design": start.design, "reference_sha256": start.id,
        "output_sha256": child.id, "validator_fingerprint": fingerprint,
        "result": "pass", "vectors": 200, "precheck_vectors": 100,
        "raw_report_path": str(report),
        "raw_report_sha256": _sha256_file(report),
        "attempts": [{"run_dir": str(verification_run)}],
    }
    _atomic_bytes(
        verifier / "records" / fingerprint / start.design / f"{child.id}.json",
        _json_bytes(record),
    )
    _atomic_bytes(
        verifier / "latest" / start.design / f"{child.id}.json",
        _json_bytes(record),
    )
    _atomic_bytes(target_root / "plan.json", _json_bytes({
        "training_inputs": {start.design: start.id},
    }))
    source = promotion.Source("fixture", source_root, "old-meter", 1, "plan.json")
    approval = MeterEquivalence("old-meter", "new-meter", "d" * 64)
    return (source, approval, source_store, target_store, target_cache,
            start, child, spec, original, key)


def test_training_alias_yields_no_tool_cache_hit_and_preserves_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (source, approval, source_store, target_store, target_cache,
     start, child, spec, original, key) = fixture(tmp_path)
    row = promotion.candidate(
        source, source.root / "cache" / f"{key}.json",
        {"recipe_versions": {spec.id: spec.version},
         "training_inputs": {start.design: start.id}},
        {start.design: evidence(start.id, "new-meter")},
        {"old-meter": approval},
    )
    row["alias_run_id"] = "run_" + "b" * 32
    monkeypatch.setattr(promotion, "DEST", target_store.root)
    assert promotion.copy_one(
        row, source, target_store, target_cache, approval, approval.audit_sha256
    ) == "imported"
    assert promotion.copy_one(
        row, source, target_store, target_cache, approval, approval.audit_sha256
    ) == "already"
    assert source_store.load_outcome(original.run.id) == original
    alias = target_store.load_outcome(row["alias_run_id"])
    assert alias.evidence.meter_version == "new-meter"
    assert alias.run.original_run_id == original.run.id
    assert target_store.state_path(start.id).stat().st_ino == source_store.state_path(start.id).stat().st_ino
    assert target_store.state_path(child.id).stat().st_ino == source_store.state_path(child.id).stat().st_ino
    assert (target_store.runs_dir / row["alias_run_id"] / "training_cache_alias.json").is_file()

    target_start = start.model_copy(update={"path": str(target_store.state_path(start.id))})
    engine = StepEngine(
        store=target_store, cache=target_cache, backend=NoToolBackend(),
        meter_version="new-meter", cache_enabled=True,
    )
    hit = engine.execute(target_start, spec)
    assert hit.run.cache_hit is True
    assert hit.run.original_run_id == alias.run.id
    for field in (
        "wall_s", "cpu_user_s", "cpu_sys_s", "peak_memory_bytes",
        "load_1m_start", "load_1m_end", "busy_slots_start",
    ):
        assert getattr(hit.run, field) == getattr(original.run, field)

    sidecar = target_store.runs_dir / row["alias_run_id"] / "training_cache_alias.json"
    _atomic_bytes(sidecar, _json_bytes({"corrupt": True}))
    with pytest.raises(ValueError, match="provenance"):
        promotion.copy_one(
            row, source, target_store, target_cache, approval, approval.audit_sha256
        )


def test_zero_amd_output_census_is_not_imported(tmp_path: Path) -> None:
    (source, approval, _, _, _, start, _, spec, _, key) = fixture(tmp_path)
    log = source.root / "verification/runs/verify_fixture/validate.log"
    _atomic_bytes(log, b"Outputs (0):\n")
    with pytest.raises(promotion.MissingProof, match="compared no outputs"):
        promotion.candidate(
            source, source.root / "cache" / f"{key}.json",
            {"recipe_versions": {spec.id: spec.version},
             "training_inputs": {start.design: start.id}},
            {start.design: evidence(start.id, "new-meter")},
            {"old-meter": approval},
        )


def test_audit_must_pass_both_old_versions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(promotion, "AUDIT_ROOT", tmp_path)
    _atomic_bytes(tmp_path / "plan.json", _json_bytes({
        "current_meter_version": promotion.CURRENT_METER,
    }))
    audit = {
        "schema_version": "training-meter-equivalence-audit-v1",
        "plan_sha256": _sha256_file(tmp_path / "plan.json"),
        "versions": {
            promotion.OLD_FIT: {"samples": 5, "identical": 5, "different": 0},
            promotion.OLD_DEPTH3_METER: {"samples": 5, "identical": 4, "different": 1},
        },
        "normalized_differences": [],
        "live_exact_tags": [],
        "tagged_runner_jobs": 20,
    }
    _atomic_bytes(tmp_path / "audit.json", _json_bytes(audit))
    with pytest.raises(ValueError, match="failed comparison"):
        promotion.approvals(_sha256_file(tmp_path / "audit.json"))


def test_ambiguous_dcp_outputs_are_excluded_from_aliases() -> None:
    """Source precedence cannot silently choose one divergent checkpoint."""
    different = [
        {"key": "ambiguous", "source": "first", "output_id": "dcp-one"},
        {"key": "ambiguous", "source": "second", "output_id": "dcp-two"},
    ]
    identical = [
        {"key": "safe", "source": "first", "output_id": "same-dcp"},
        {"key": "safe", "source": "second", "output_id": "same-dcp"},
    ]
    selected, conflicts = promotion.select_unambiguous(
        {"ambiguous": different, "safe": identical},
        {"ambiguous": different[0], "safe": identical[0]},
    )
    assert list(selected) == ["safe"]
    assert conflicts == [{
        "key": "ambiguous", "chosen_source": "first",
        "source_output_hashes": {"first": "dcp-one", "second": "dcp-two"},
    }]


def test_same_dcp_duplicate_requires_identical_promoted_evidence() -> None:
    """Meter-version approval does not excuse two different metric records."""
    rows = [
        {"key": "same-key", "source": "first", "output_id": "same-dcp",
         "new_evidence_id": "metrics-one"},
        {"key": "same-key", "source": "second", "output_id": "same-dcp",
         "new_evidence_id": "metrics-one"},
    ]
    checks = promotion.same_output_evidence_checks({"same-key": rows})
    assert checks == [{
        "key": "same-key", "output_id": "same-dcp",
        "current_evidence_id": "metrics-one", "sources": ["first", "second"],
    }]
    rows[1] = {**rows[1], "new_evidence_id": "metrics-two"}
    with pytest.raises(ValueError, match="same-output duplicate Evidence differs"):
        promotion.same_output_evidence_checks({"same-key": rows})
