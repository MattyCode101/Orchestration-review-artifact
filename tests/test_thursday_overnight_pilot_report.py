"""Synthetic, no-tool checks for the training-only v4 morning reporter."""
from __future__ import annotations

import hashlib

import pytest

from orch.engine.store import _json_bytes
from scripts import thursday_overnight_pilot_report as report


def test_read_journal_uses_existing_lock_and_creates_nothing(tmp_path) -> None:
    root = tmp_path / "training"
    assert report._journal(root, "pilot_synthetic") == []
    assert not root.exists()
    folder = root / "episodes" / "pilot_synthetic"
    folder.mkdir(parents=True)
    event = {"type": "step", "id": "synthetic"}
    wrapper = {
        "event": event,
        "sha256": hashlib.sha256(_json_bytes(event)).hexdigest(),
    }
    (folder / "journal.jsonl").write_bytes(_json_bytes(wrapper))
    (folder / "journal.lock").touch()
    before = {str(p) for p in root.rglob("*")}
    assert report._journal(root, "pilot_synthetic") == [event]
    assert {str(p) for p in root.rglob("*")} == before


def test_counter_excludes_historical_reservations_from_v4_cell(tmp_path, monkeypatch) -> None:
    root = tmp_path / "v4"
    root.mkdir()
    identity = report.pilot.episode_id(
        "vexriscv_re-place", "T0", "repair_loop"
    )
    carry = [
        {"episode_id": f"carryover:v3:{i}", "run_id": f"old_{i}"}
        for i in range(62)
    ]
    new = {"episode_id": identity, "run_id": "new_1"}
    payload = {
        "schema_version": "thursday-night-new-runs-v1",
        "plan_sha256": "frozen", "cap": 600,
        "reservations": carry + [new],
    }
    (root / "new_real_runs.json").write_bytes(_json_bytes(payload))
    monkeypatch.setattr(report, "ROOT", root)
    record, counts = report._counter("frozen")
    assert len(record["reservations"]) == 63
    assert counts[identity] == 1
    assert all(not key.startswith("carryover:") for key in counts)
    payload["reservations"].append(dict(new))
    (root / "new_real_runs.json").write_bytes(_json_bytes(payload))
    with pytest.raises(ValueError, match="duplicate"):
        report._counter("frozen")


def test_render_has_all_28_cells_and_pending_audits() -> None:
    table = report._table([])
    assert len(table) == 28
    status = {
        "phase": "prepared", "completed_episodes": 0,
        "cap_used": 62, "new_v4_reservations": 0,
    }
    source = {
        "v4_physically_valid_step_occurrences": 0,
        "training_edges": 750, "candidate_counts": {"depth3": 570},
        "v3_partial_step_occurrences": 71,
    }
    audits = {
        "meter": {"status": "pending"},
        "promotion": {"status": "pending"},
        "ceiling": {"status": "pending"},
    }
    text = report.render(table, status, source, audits)
    assert len(text.split()) < 500
    assert text.count("| repair_loop |") == 1
    assert text.count("| random_seed_2 |") == 1
    assert "0/5;-;0/0/0;0/0" in text
    assert "ceiling audit pending" in text
    assert "D1-D3" in text
    assert "?" not in text


def test_present_but_invalid_promotion_audit_fails_closed(tmp_path, monkeypatch) -> None:
    meter = tmp_path / "meter.json"
    promotion = tmp_path / "promotion.json"
    ceiling = tmp_path / "ceiling.json"
    promotion.write_bytes(_json_bytes({
        "destination_root": str(tmp_path), "pilot_plan_sha256": "frozen",
        "tool_jobs_launched": 0, "promoted_pointers": 1,
    }))
    monkeypatch.setattr(report, "METER_AUDIT", meter)
    monkeypatch.setattr(report, "PROMOTION_AUDIT", promotion)
    monkeypatch.setattr(report, "CEILING_SUMMARY", ceiling)
    with pytest.raises(ValueError, match="promotion"):
        report._audit_state("frozen")


def test_ceiling_formatting_keeps_missing_gap_explicit() -> None:
    assert report._gap(None) == "-"
    assert report._gap(0.125) == "0.125"

