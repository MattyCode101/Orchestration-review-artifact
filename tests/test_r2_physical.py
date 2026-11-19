"""Round-2 physical description: field extraction, the meaningful-change rules and the section text."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orch.policies.r2_physical import (
    BY_GROUP, FIELDS, GROUPS, FieldSpec, change_text, facts_from_record, meaningful, render_physical_section,
    start_text,
)

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data/exp_evening_2026-11-04/training_telemetry_v3/states"


def record(**over) -> dict:
    base = {
        "metrics": {"tns_ns": {"value": -70407.172}, "dynamic_power_w": {"value": 2.353}},
        "probe": {
            "timing": {"failing_endpoints": 95853},
            "critical_path_features": {"dominant_path_type": "routing_dominated"},
            "utilization_summary": {
                "CLB LUTs": {"used": 13543, "utilization_pct": 3.44}, "CLB Registers": {"used": 5337, "utilization_pct": 0.68},
                "Block RAM Tile": {"used": 0, "utilization_pct": 0.0}, "DSPs": {"used": 3, "utilization_pct": 0.13},
                "URAM": {"used": 0, "utilization_pct": 0.0},
            },
            "lut_primitive_counts": {"LUT1": 267, "LUT2": 800, "LUT3": 1003, "LUT4": 401, "LUT5": 424, "LUT6": 1565},
            "lut_as_shift_register": 98, "fanout": {"max_fanout": 12288, "num_high_fanout_nets": 20},
        },
        "wpath": {
            "logic_levels": 11, "data_path_delay_ns": 3.689, "net_delay_ns": 2.899,
            "tile_span": {"dx": 6, "dy": 46, "x_max": 10, "x_min": 4, "y_max": 59, "y_min": 13},
        },
    }
    for path, value in over.items():
        keys = path.split("__")
        node = base
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return base


def test_facts_come_from_the_existing_record_fields() -> None:
    f = facts_from_record(record())
    assert f["failing_endpoints"] == 95853 and f["logic_levels"] == 11 and f["tile_span"] == 52
    assert f["route_share_pct"] == pytest.approx(100 * 2.899 / 3.689)
    assert f["path_type"] == "routing_dominated" and f["tns_ns"] == pytest.approx(-70407.172)
    assert f["clb_luts"] == 13543 and f["clb_luts_pct"] == pytest.approx(3.44) and f["dsps"] == 3
    assert f["small_lut_share_pct"] == pytest.approx(100 * (267 + 800 + 1003) / 4460)
    assert f["srl_luts"] == 98 and f["max_fanout"] == 12288 and f["high_fanout_nets"] == 20


def test_missing_pieces_become_none_not_errors() -> None:
    f = facts_from_record({"metrics": {}, "probe": {}, "wpath": {}})
    assert all(f[spec.name] is None for spec in FIELDS if spec.group != "power")
    corner = facts_from_record(record(wpath__tile_span={"dx": None, "dy": None, "x_max": 10, "x_min": 4, "y_max": 59, "y_min": 13}))
    assert corner["tile_span"] == 6 + 46


@pytest.mark.parametrize(("name", "old", "new", "expected"), [
    ("failing_endpoints", 1000, 1099, False), ("failing_endpoints", 1000, 1100, True), ("failing_endpoints", 50, 55, False),
    ("logic_levels", 9, 9, False), ("logic_levels", 9, 10, True),
    ("path_delay_ns", 3.0, 3.2, False), ("path_delay_ns", 3.0, 3.31, True), ("path_delay_ns", 0.2, 0.23, False),
    ("route_share_pct", 70.0, 74.9, False), ("route_share_pct", 70.0, 75.0, True),
    ("path_type", "routing_dominated", "routing_dominated", False), ("path_type", "routing_dominated", "logic_dominated", True),
    ("tile_span", 40, 47, False), ("tile_span", 40, 48, True), ("dsps", 3, 3, False), ("dsps", 3, 4, True),
    ("high_fanout_nets", 20, 23, False), ("high_fanout_nets", 20, 24, True),
    ("clb_luts", 100, 5000, False),                                        # shown for the start only
])
def test_the_declared_meaningful_change_rules(name, old, new, expected) -> None:
    spec = next(f for f in FIELDS if f.name == name)
    assert meaningful(spec, old, new) is expected


def test_the_groups_partition_the_fields_and_power_is_one_number() -> None:
    assert [spec.name for g in GROUPS for spec in BY_GROUP[g]] == [spec.name for spec in FIELDS]
    assert [spec.name for spec in BY_GROUP["power"]] == ["dynamic_power_w"]


def test_the_start_is_described_in_full_and_other_states_by_changes() -> None:
    start = facts_from_record(record())
    start["dynamic_power_w"] = 2.353
    moved = facts_from_record(record(wpath__logic_levels=10, probe__fanout={"max_fanout": 12288, "num_high_fanout_nets": 40}))
    moved["dynamic_power_w"] = 2.353
    same = dict(start)
    text = render_physical_section([("S0", True, start), ("S1", False, moved), ("S2", False, same)], GROUPS, power_band=0.002)
    lines = text.splitlines()
    assert lines[0] == "Physical description of the saved states"
    assert "Fields shown: worst-path timing, utilisation and fan-out, dynamic power." in lines[1]
    assert "S0 (start)" in lines
    assert any(l.startswith("  timing: failing endpoints 95853; total negative slack -70407.2 ns; worst-path logic levels 11") for l in lines)
    assert any("CLB LUTs 13543 (3.4% of the device)" in l for l in lines)
    assert "  power: dynamic power 2.353 W" in lines
    s1 = lines[lines.index("S1") + 1: lines.index("S2")]
    assert s1 == ["  timing: worst-path logic levels 11 -> 10", "  utilisation: high-fan-out nets 20 -> 40"]
    assert lines[lines.index("S2") + 1] == "  no meaningful change from the start in these fields"


def test_each_group_can_be_shown_alone() -> None:
    start = facts_from_record(record()); start["dynamic_power_w"] = 2.353
    moved = facts_from_record(record(wpath__logic_levels=8)); moved["dynamic_power_w"] = 2.9
    for group, expect_in, expect_out in (("timing", "logic levels 11 -> 8", "dynamic power"),
                                          ("utilisation", "no meaningful change", "logic levels"),
                                          ("power", "dynamic power 2.353 W -> 2.900 W (large_loss)", "logic levels")):
        text = render_physical_section([("S0", True, start), ("S1", False, moved)], [group], power_band=0.1)
        assert expect_in in text and expect_out not in text.split("S1", 1)[1], (group, text)
        assert f"Fields shown: {group if group != 'timing' else 'worst-path timing'}" in text or group != "timing"


def test_power_uses_the_recorded_band_and_falls_back_to_five_per_cent() -> None:
    a, b = {"dynamic_power_w": 1.000}, {"dynamic_power_w": 1.040}
    assert change_text(a, b, "power", power_band=0.05) == ""                         # within noise
    assert "(small_loss)" in change_text(a, {"dynamic_power_w": 1.080}, "power", power_band=0.05)
    assert change_text(a, b, "power", power_band=None) == ""
    assert change_text(a, {"dynamic_power_w": 1.06}, "power", power_band=None) != ""


def test_start_text_skips_fields_that_are_missing() -> None:
    assert start_text({"failing_endpoints": 12}, "timing") == "failing endpoints 12"


@pytest.mark.skipif(not V3.is_dir(), reason="stored training telemetry not present")
def test_every_stored_training_record_yields_all_timing_and_utilisation_fields() -> None:
    paths = sorted(V3.glob("*/dcp_telemetry.json"))[:60]
    assert paths
    for path in paths:
        facts = facts_from_record(json.loads(path.read_text(encoding="utf-8")))
        missing = [s.name for s in FIELDS if s.group != "power" and facts.get(s.name) is None]
        assert not missing, (path.parent.name, missing)
