"""Counts-only phase monitor tests; no held-out DCP or result is read."""
from __future__ import annotations

import json

import pytest

from scripts import overnight_baseline_milestones as monitor


def state(d1: int = 10, d2: int = 0):
    rows = {name: {"c0_pass": 1, "d1_pass": d1, "d2_pass": d2}
            for name in monitor.ORDER}
    return {
        "phase": "two_step", "by_design": rows, "stopped_designs": [],
        "telemetry_unique_ok": 77, "completed_edges": 7 + 7 * d1 + 7 * d2,
    }


def test_complete_phase_events_use_counts_only() -> None:
    rows = monitor.events(state(d2=15), None)
    keys = {key for key, _ in rows}
    assert {"c0_complete", "d1_complete", "d2_100"} <= keys
    assert "d2_200" not in keys
    text = " ".join(message for _, message in rows)
    assert "wns_ns" not in text and "packed_luts" not in text


def test_frozen_labels_are_logged_even_if_one_design_stopped(tmp_path, monkeypatch) -> None:
    path = tmp_path / "admitted.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(monitor, "ADMISSION", path)
    current = state()
    current["by_design"][monitor.ORDER[1]]["d1_pass"] = 3
    current["stopped_designs"] = [monitor.ORDER[1]]
    flags = {name: name != monitor.ORDER[1] and index < 6
             for index, name in enumerate(monitor.ORDER)}
    result = monitor.events(current, {"admitted_flags": flags})
    assert "room_screen" in {key for key, _ in result}
    label = next(message for key, message in result if key == "room_screen")
    assert f"{monitor.ORDER[1]}=not admitted" in label


def test_scope_error_does_not_write_milestone() -> None:
    current = state()
    current["by_design"].pop(monitor.ORDER[-1])
    with pytest.raises(ValueError, match="seven frozen"):
        monitor.events(current, None)
