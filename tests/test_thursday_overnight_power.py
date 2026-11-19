"""Offline safety tests for the two-phase training-only power campaign."""
from __future__ import annotations

import json
import importlib
from pathlib import Path

import pytest

from orch.contract.models import Evidence, MetricName, MetricValue
from scripts import thursday_overnight_power as power


def test_a_tasks_cover_each_distinct_state_once_per_power_recipe() -> None:
    inputs = {
        "b2_hash": "a" * 64,
        "states": [{"hash": "a" * 64}, {"hash": "b" * 64}],
    }
    tasks = power.a_tasks(inputs)
    assert len(tasks) == 8
    assert len({task["key"] for task in tasks}) == 8
    assert {task["input_hash"] for task in tasks} == {"a" * 64, "b" * 64}
    assert {task["recipe_id"] for task in tasks} == set(power.POWER)


def test_b_keeps_four_lineages_when_all_power_outputs_are_identical() -> None:
    start = "a" * 64
    output = "b" * 64
    rows = {
        f"a:{start}:{recipe}": {
            "status": "ok", "recipe_id": recipe, "input_hash": start,
            "output_hash": output, "run_id": f"run_{index}",
            "amd_result": "pass", "amd_outputs_compared": 1,
        }
        for index, recipe in enumerate(power.POWER)
    }
    tasks = power.b_tasks(rows, start)
    assert len(tasks) == 40
    assert len({task["key"] for task in tasks}) == 40
    assert {task["input_hash"] for task in tasks} == {output}
    assert {task["source_run_id"] for task in tasks} == {
        f"run_{index}" for index in range(4)
    }


def test_b_refuses_unchecked_power_parent() -> None:
    start = "a" * 64
    rows = {
        f"a:{start}:{recipe}": {
            "status": "ok", "recipe_id": recipe, "input_hash": start,
            "output_hash": "b" * 64, "run_id": f"run_{index}",
            "amd_result": "pass", "amd_outputs_compared": 1,
        }
        for index, recipe in enumerate(power.POWER)
    }
    rows[f"a:{start}:{power.POWER[0]}"]["amd_outputs_compared"] = 0
    with pytest.raises(ValueError, match="AMD-PASS"):
        power.b_tasks(rows, start)


def test_evidence_requires_zero_placement_properties_and_b2_digests() -> None:
    metrics = {
        name: MetricValue(value=1.0, unit="unit", method="meter")
        for name in MetricName
    }
    evidence = Evidence(
        id="e", state_id="a" * 64, metrics=metrics, routed=True,
        route_errors=0, drc_errors=0, constraint_fingerprint="constraint",
        clock_digest="clock", v_props={key: 0 for key in power.PROPS},
        meter_version="meter",
    )
    plan = {"meter_version": "meter"}
    inputs = {
        "b2_clock_digest": "clock",
        "b2_constraint_fingerprint": "constraint",
    }
    assert len(power.check_evidence(evidence, "a" * 64, plan, inputs)) == len(MetricName)
    changed = evidence.model_copy(update={
        "v_props": {**evidence.v_props, "n_IS_LOC_FIXED": 1}
    })
    with pytest.raises(ValueError, match="placement properties"):
        power.check_evidence(changed, "a" * 64, plan, inputs)


def test_global_barrier_waits_for_all_five_terminal_designs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(power, "TOP", tmp_path)
    for design in power.DESIGNS:
        root = power.root_for("a", design)
        root.mkdir(parents=True)
        (root / "status.json").write_text(
            json.dumps({"phase": "complete"}), encoding="utf-8"
        )
    pending = power.root_for("a", power.DESIGNS[-1]) / "status.json"
    pending.write_text(json.dumps({"phase": "power_one_step"}), encoding="utf-8")
    with pytest.raises(ValueError, match="terminal"):
        power.phase_a_barrier()
    pending.write_text(json.dumps({"phase": "complete"}), encoding="utf-8")
    barrier = power.phase_a_barrier()
    assert set(barrier["statuses"]) == set(power.DESIGNS)
    assert all(value == "complete" for value in barrier["statuses"].values())


def test_report_follower_does_not_finish_between_phase_a_and_b(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    report = importlib.import_module("thursday_overnight_power_report")
    a = {name: {"phase": "complete"} for name in power.DESIGNS}
    b = {name: {"phase": "not_launched"} for name in power.DESIGNS}
    assert report.terminal_status(a, b, False) is False
    assert report.terminal_status(a, b, True) is False
    b = {name: {"phase": "complete"} for name in power.DESIGNS}
    assert report.terminal_status(a, b, True) is True
    a[power.DESIGNS[0]] = {"phase": "stopped"}
    b[power.DESIGNS[0]] = {"phase": "not_launched"}
    assert report.terminal_status(a, b, True) is True


def test_complete_morning_report_stays_under_word_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    report = importlib.import_module("thursday_overnight_power_report")
    marker = tmp_path / "registry.done"
    marker.write_text(
        "updated\nregistry_sha256="
        "25f0a39e4d94c91b2c7b8d450a47788e95c8dab237558c6e45539a2a79b33311\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(report, "REGISTRY_DONE", marker)
    monkeypatch.setattr(report, "controller_issue", lambda: None)
    monkeypatch.setattr(report.campaign, "selected", lambda: {
        "by_design": {name: {} for name in power.DESIGNS}
    })
    monkeypatch.setattr(report, "depth_three", lambda: (570, 570, 0))
    counts_a = {
        "vexriscv_re-place": 36, "rosetta_spam-filter": 36,
        "rosetta_3d-rendering": 40, "logicnets_jscl": 32,
        "finn_radioml": 32,
    }
    monkeypatch.setattr(report.campaign, "phase_status", lambda phase, design: {
        "phase": "complete",
        "amd_pass": counts_a[design] if phase == "a" else 40,
        "failed": 0, "planned": counts_a[design] if phase == "a" else 40,
    })
    monkeypatch.setattr(report, "per_design", lambda design, inputs: {
        "band_pct": 7.3525,
        "best_b2": ("P_ROUTE_HIGHDELAY", 12.3456, -0.1234, 10, True),
        "best_d1": (
            "LATCH.hardblock_relocate > P_ROUTE_HIGHDELAY",
            13.4567, 0.2345, -100, True,
        ),
        "raised_a": 12, "raised_b": 13,
        "power_first_wins": 14, "power_last_wins": 15,
    })
    content = report.render()
    assert "power-first" in content.lower()
    assert content.count("**vexriscv_re-place**") == 1


def test_morning_report_surfaces_failed_controller(tmp_path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    report = importlib.import_module("thursday_overnight_power_report")
    monkeypatch.setattr(report.campaign, "TOP", tmp_path)
    (tmp_path / "controller_status.json").write_text(
        json.dumps({"phase": "failed", "error": "barrier changed"}),
        encoding="utf-8",
    )
    assert report.controller_issue() == "power controller failed: barrier changed"
