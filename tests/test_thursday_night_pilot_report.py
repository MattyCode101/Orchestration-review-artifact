"""Synthetic checks for train-only pilot comparison arithmetic."""

from __future__ import annotations

import pytest

from orch.contract.models import Evidence, MetricName, MetricValue
from scripts.thursday_night_pilot_report import _aggregate, _same_baseline, render


def _evidence(wns: float, *, meter: str) -> Evidence:
    return Evidence(
        id=f"e-{meter}", state_id="a" * 64,
        metrics={
            MetricName.WNS_NS: MetricValue(value=wns, unit="ns", method="timing"),
            MetricName.PACKED_LUTS: MetricValue(value=100.0, unit="count", method="area"),
        },
        routed=True, route_errors=0, drc_errors=0,
        constraint_fingerprint="xdc", clock_digest="clock",
        v_props={
            "n_DONT_TOUCH": 0, "n_IS_LOC_FIXED": 0, "n_IS_BEL_FIXED": 0,
        },
        meter_version=meter,
    )


def test_cross_version_denominator_requires_identical_b2_values() -> None:
    _same_baseline(_evidence(-0.7, meter="old"), _evidence(-0.7, meter="new"))
    with pytest.raises(ValueError, match="wns_ns"):
        _same_baseline(_evidence(-0.7, meter="old"), _evidence(-0.6, meter="new"))


def test_policy_brief_table_counts_only_completed_verified_rows() -> None:
    rows = [{
        "policy": "repair_loop", "brief": "T0", "share_of_best": 0.5,
        "final_brief_valid": True, "runs_used": 4, "cache_hits": 2,
        "new_real_runs_reserved": 2, "repair_final_phase": "repair",
    }, {
        "policy": "repair_loop", "brief": "T0", "share_of_best": None,
        "final_brief_valid": True, "runs_used": 2, "cache_hits": 2,
        "new_real_runs_reserved": 0, "repair_final_phase": "gentle",
    }]
    table = _aggregate(rows)
    selected = next(row for row in table if (row["policy"], row["brief"]) == ("repair_loop", "T0"))
    assert selected == {
        "policy": "repair_loop", "brief": "T0", "completed": 2, "planned": 5,
        "brief_valid_final_picks": 2, "mean_share_of_best": 0.5,
        "defined_shares": 1, "runs_used": 6, "cache_hits": 4,
        "new_real_runs": 2, "repair_final_picks": 1, "gentle_final_picks": 1,
    }
    assert len(table) == 28
    text = render(table, {
        "completed_episodes": 2, "phase": "running",
        "new_real_runs_reserved": 2,
    }, {
        "training_edges": 750, "candidate_counts": {"depth3": 12, "pilot": 6},
    })
    assert "| repair_loop | 2/5;50.0%;6/4/2;1/1" in text
    assert "| fixed_AB | 0/5;—;0/0/0;0/0" in text
