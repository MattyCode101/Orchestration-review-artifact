"""Priority-one completion gate for detached optional work."""
from __future__ import annotations

from scripts import overnight_optional_controller as controller


def complete_summary() -> dict:
    return {
        "phase": "complete",
        "successful_edges_by_phase": {"C0": 7, "D1": 70, "D2": 500},
        "starts_amd_source_pass_and_telemetry_ok": 7,
        "failed_designs": [],
        "admitted_test_designs": ["a", "b", "c", "d", "e"],
    }


def test_optional_work_waits_for_final_summary() -> None:
    assert controller.gate({"phase": "one_step_and_c0", "stopped_designs": []}, None) == "wait"
    assert controller.gate({"phase": "complete", "stopped_designs": []}, None) == "wait"


def test_optional_work_requires_every_verified_baseline_edge() -> None:
    summary = complete_summary()
    assert controller.gate({"phase": "complete", "stopped_designs": []}, summary) == "launch"
    summary["successful_edges_by_phase"]["D2"] = 499
    assert controller.gate({"phase": "complete", "stopped_designs": []}, summary) == "stop"


def test_optional_work_stops_after_any_failed_design() -> None:
    assert controller.gate({"phase": "complete_with_failures", "stopped_designs": ["x"]}, None) == "stop"
    assert controller.gate({"phase": "complete", "stopped_designs": ["x"]}, complete_summary()) == "stop"
