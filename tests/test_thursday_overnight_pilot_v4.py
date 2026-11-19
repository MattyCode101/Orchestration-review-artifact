"""Offline safety tests for the versioned Thursday pilot restart."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts import thursday_overnight_pilot_v4 as v4


def test_schedule_partitions_every_training_cell_once() -> None:
    rows = [row for shard in range(v4.SHARDS) for row in v4.rows_for(shard)]
    assert len(rows) == 140
    assert len(set(rows)) == 140
    assert v4.CAP == 600
    with pytest.raises(ValueError):
        v4.rows_for(v4.SHARDS)


def test_only_expected_source_hash_changes_are_accepted() -> None:
    old = {"registry": "r0", "telemetry": "t0", "A_fit": "a0"}
    v4._require_expected_source_changes(
        old, {"registry": "r1", "telemetry": "t1", "A_fit": "a0"}
    )
    with pytest.raises(v4.pilot.PilotSafetyStop):
        v4._require_expected_source_changes(
            old, {"registry": "r1", "telemetry": "t1", "A_fit": "a1"}
        )
    with pytest.raises(v4.pilot.PilotSafetyStop):
        v4._require_expected_source_changes(
            old, {"registry": "r0", "telemetry": "t1", "A_fit": "a0"}
        )


def test_carryover_counts_old_launches_without_charging_new_episode() -> None:
    old = [
        {"episode_id": "pilot_one", "run_id": "run_one"},
        {"episode_id": "pilot_two", "run_id": "run_two"},
    ]
    rows = v4._carryover_rows(old)
    assert [row["run_id"] for row in rows] == ["run_one", "run_two"]
    assert [row["original_episode_id"] for row in rows] == [
        "pilot_one", "pilot_two"
    ]
    assert all(row["episode_id"].startswith("carryover:v3:") for row in rows)
    assert old[0]["episode_id"] == "pilot_one"
    with pytest.raises(v4.pilot.PilotSafetyStop):
        v4._carryover_rows([old[0], old[0]])


def test_v4_plan_refuses_changed_worker_recipe_version(monkeypatch) -> None:
    worker_hashes = {
        "A_fit": "a", "B_fit": "b", "timing_card": "t",
        "area_card": "u", "recipe_sets": "s",
    }
    frozen = {
        "data_root": str(v4.ROOT), "max_new_real_recipe_runs": 600,
        "recipe_versions": {"LATCH.phys_opt": "v1"},
        "source_hashes": dict(worker_hashes),
    }
    old = {
        "recipe_versions": {"LATCH.phys_opt": "v1"},
        "source_hashes": dict(worker_hashes),
    }
    monkeypatch.setattr(v4, "ORIGINAL_MAKE_PLAN", lambda runtime: dict(frozen))
    monkeypatch.setattr(v4, "_v3_provenance", lambda: {"episode_reuse": False})
    monkeypatch.setattr(v4, "_read", lambda path: old)
    assert v4._new_plan(None)["schema_version"] == "thursday-overnight-pilot-v4"
    assert v4._new_plan(None)["episode_reuse"] is False
    old["recipe_versions"]["LATCH.phys_opt"] = "v2"
    with pytest.raises(v4.pilot.PilotSafetyStop):
        v4._new_plan(None)


def test_launch_gate_requires_exact_promotion_audit_hash(tmp_path, monkeypatch) -> None:
    root = tmp_path / "v4"
    root.mkdir()
    plan = root / "plan.json"
    plan.write_text("{}\n", encoding="utf-8")
    audit_path = tmp_path / "promotion.json"
    audit = {
        "destination_root": str(root),
        "pilot_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "tool_jobs_launched": 0, "promoted_pointers": 3,
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    monkeypatch.setattr(v4, "ROOT", root)
    monkeypatch.setattr(v4, "PROMOTION_AUDIT", audit_path)
    approved = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert v4._promotion(approved) == audit
    with pytest.raises(v4.pilot.PilotSafetyStop):
        v4._promotion("0" * 64)
    audit["tool_jobs_launched"] = 1
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(v4.pilot.PilotSafetyStop):
        v4._promotion(hashlib.sha256(audit_path.read_bytes()).hexdigest())

