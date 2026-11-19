"""No-tool scope and safety gates for the optional reserve campaign."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import overnight_optional_reserve as reserve


def _complete() -> tuple[dict, dict]:
    return (
        {"phase": "complete", "stopped_designs": [], "error": None},
        {"phase": "complete",
         "successful_edges_by_phase": {"C0": 7, "D1": 70, "D2": 500},
         "starts_amd_source_pass_and_telemetry_ok": 7,
         "failed_designs": []},
    )


def test_scope_is_exactly_100_ordered_pairs() -> None:
    assert len(reserve.RECIPES) == 10
    assert len(set(reserve.RECIPES)) == 10
    assert len(reserve.TASKS) == len(set(reserve.TASKS)) == 100
    assert set(reserve.TASKS) == {
        (first, second) for first in reserve.RECIPES for second in reserve.RECIPES
    }
    assert all("C0_reroute" not in pair for pair in reserve.TASKS)
    assert reserve.ROOT.is_relative_to(reserve.PROJECT / "data/sweeps")
    assert reserve.MAX_ACTIVE <= reserve.MAX_SLOTS == 96


def test_reserve_needs_complete_baseline_and_two_fir_admissions() -> None:
    status, summary = _complete()
    admitted = {"admitted_in_order": [
        "rosetta_digit-recognition", "corescore_500_mod",
        "vexriscv_re-place_v2", "fir_symmetric_systolic",
        "fir_systolic_transposed",
    ]}
    assert reserve.eligibility(status, None, None) == "waiting_for_priority_one"
    assert reserve.eligibility(status, summary, admitted) == "eligible"
    assert reserve.eligibility(
        {"phase": "two_step", "stopped_designs": [], "error": None},
        summary, admitted,
    ) == "blocked_by_incomplete_priority_one"
    assert reserve.eligibility(
        {"phase": "complete_with_failures", "stopped_designs": [], "error": None},
        summary, admitted,
    ) == "blocked_by_priority_one"
    assert reserve.eligibility(
        status, {**summary, "successful_edges_by_phase": {"C0": 7, "D1": 70, "D2": 499}},
        admitted,
    ) == "blocked_by_incomplete_priority_one"
    assert reserve.eligibility(
        status, summary, {"admitted_in_order": [
            "rosetta_digit-recognition", "corescore_500_mod",
            "vexriscv_re-place_v2", "fir_symmetric_systolic",
            "rosetta_3d-rendering_v2",
        ]},
    ) == "skipped_reserve_is_admitted_test"
    assert reserve.eligibility(
        status, summary, {"admitted_in_order": [
            "rosetta_digit-recognition", "corescore_500_mod",
            "vexriscv_re-place_v2", "fir_symmetric_systolic",
            "amd_mini-isp_v2",
        ]},
    ) == "skipped_fir_condition_not_met"


def test_source_refuses_before_priority_one_finishes(tmp_path, monkeypatch) -> None:
    plan = b"frozen test plan"
    (tmp_path / "plan.json").write_bytes(plan)
    (tmp_path / "status.json").write_text(
        json.dumps({"phase": "one_step_and_c0", "stopped_designs": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(reserve, "TEST", tmp_path)
    monkeypatch.setattr(reserve, "TEST_PLAN_SHA", hashlib.sha256(plan).hexdigest())
    with pytest.raises(ValueError, match="not eligible"):
        reserve.source()


def test_watchdog_excludes_prefix_sibling(tmp_path, monkeypatch) -> None:
    root = tmp_path / "reserve"
    root.mkdir()
    monkeypatch.setattr(reserve, "ROOT", root)
    monkeypatch.setattr(reserve, "_own_pids", lambda: (101, 102))
    monkeypatch.setattr(
        reserve, "_stat",
        lambda pid: SimpleNamespace(state="S", pgrp=10001 if pid == 101 else 10002),
    )
    monkeypatch.setattr(
        reserve.os, "readlink",
        lambda path: str(root / "run") if path == "/proc/101/cwd"
        else str(tmp_path / "reserve-sibling"),
    )
    monkeypatch.setattr(
        reserve, "_tag_of",
        lambda pid: "a" * 32 if pid == 101 else "b" * 32,
    )
    tags, groups = reserve.targets()
    assert tags == {"a" * 32}
    assert groups == {10001}


def test_cost_requires_cpu_load_and_original_cache_cost() -> None:
    values = dict(
        wall_s=2.0, cpu_user_s=1.0, cpu_sys_s=0.2,
        peak_memory_bytes=512, load_1m_start=12.0,
        load_1m_end=14.0, busy_slots_start=3, cache_hit=False,
        original_run_id=None, backpressure_events=0, backpressure_wait_s=0.0,
    )
    assert reserve.run_cost(SimpleNamespace(**values))["cpu_user_s"] == 1.0
    with pytest.raises(ValueError, match="original run identity"):
        reserve.run_cost(SimpleNamespace(**{**values, "cache_hit": True}))
    with pytest.raises(ValueError, match="cost telemetry"):
        reserve.run_cost(SimpleNamespace(**{**values, "cpu_sys_s": None}))


def test_early_preflight_failure_is_recorded(tmp_path, monkeypatch) -> None:
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(reserve, "STATUS", status_path)

    def broken():
        raise ValueError("frozen source mismatch")

    monkeypatch.setattr(reserve, "prepare", broken)
    with pytest.raises(ValueError, match="frozen source mismatch"):
        reserve.run()
    record = json.loads(status_path.read_text(encoding="utf-8"))
    assert record["phase"] == "failed"
    assert record["stopped"] is True
    assert "frozen source mismatch" in record["error"]


def test_watchdog_cleans_if_driver_exits_before_watch_start(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "launch.json"
    manifest.write_text(json.dumps({"driver_pid": 12345}), encoding="utf-8")
    monkeypatch.setattr(reserve, "LAUNCH", manifest)
    monkeypatch.setattr(reserve, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(reserve, "driver_alive", lambda pid: False)
    calls = []
    monkeypatch.setattr(reserve, "cleanup", lambda: calls.append(True))
    reserve.watch()
    assert calls == [True]
