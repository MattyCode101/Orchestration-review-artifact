"""Synthetic checks for the training-only provisional null ranking."""
from __future__ import annotations

from pathlib import Path

import pytest

from orch.analysis.provisional_null import (
    Observation,
    TRAIN_DESIGNS,
    deduplicate,
    gain_pct,
    null_payload,
    rank_observations,
    require_training_root,
    zero_gain_reason,
)
from orch.contract.models import Evidence


def _evidence(label: str, wns: float, luts: float) -> Evidence:
    return Evidence.model_validate({
        "id": label,
        "state_id": label * 64,
        "routed": True,
        "meter_version": "meter-v1",
        "metrics": {
            "wns_ns": {
                "value": wns, "unit": "ns", "method": "meter",
            },
            "packed_luts": {
                "value": luts, "unit": "count", "method": "meter",
            },
        },
    })


def _row(design: str, recipe: str, index: int, gain: float) -> Observation:
    return Observation(
        design=design, sequence=(recipe, str(index)),
        input_state_id=f"{index:064x}", recipe_id=recipe,
        output_state_id=f"{index + 10000:064x}", gain_pct=gain,
        zero_reason=None, verification="pass",
    )


def test_frozen_meter_targets() -> None:
    parent = _evidence("a", -1.0, 100.0)
    child = _evidence("b", -0.5, 90.0)
    assert gain_pct("timing", parent, child, 2.0) == pytest.approx(20.0)
    assert gain_pct("area", parent, child, 2.0) == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("run_status", "valid", "output", "verification", "expected"),
    [
        ("invalid", False, "b", "pass", "invalid_or_failed"),
        ("ok", True, None, "no_output", "no_output"),
        ("ok", True, "a", "pass", "unchanged_output"),
        ("ok", True, "b", "fail", "amd_nonpass"),
        ("ok", True, "b", "unknown", "amd_nonpass"),
        ("ok", True, "b", "no_record", "amd_nonpass"),
    ],
)
def test_zero_gain_rules(
    run_status: str, valid: bool, output: str | None,
    verification: str, expected: str | None,
) -> None:
    assert zero_gain_reason(
        run_status, valid, "a", output, verification
    ) == expected


def test_training_data_root_guard_rejects_other_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="training allow-list"):
        require_training_root(tmp_path)
    with pytest.raises(ValueError, match="training allow-list"):
        require_training_root("data/sweeps/test_designs")


def test_duplicate_removal_and_conflict_rejection() -> None:
    first = Observation(
        design=TRAIN_DESIGNS[0], sequence=("A",), input_state_id="a" * 64,
        recipe_id="A", output_state_id="b" * 64, gain_pct=4.0,
        zero_reason=None, verification="pass",
    )
    duplicate = Observation(
        design=TRAIN_DESIGNS[0], sequence=("B", "A"),
        input_state_id=first.input_state_id, recipe_id="A",
        output_state_id=first.output_state_id, gain_pct=first.gain_pct,
        zero_reason=None, verification="pass",
    )
    unique, removed = deduplicate([duplicate, first])
    assert unique == (first,)
    assert removed == 1
    conflicting = Observation(
        **{**duplicate.__dict__, "output_state_id": "c" * 64}
    )
    with pytest.raises(ValueError, match="conflicting duplicate"):
        deduplicate([first, conflicting])


def test_design_weighting_and_alphabetical_ties() -> None:
    timing = ("T_E", "T_D", "T_C", "T_B", "T_A")
    area = ("A_E", "A_D", "A_C", "A_B", "A_A")
    rows: list[Observation] = []
    index = 1
    for design_index, design in enumerate(TRAIN_DESIGNS):
        for recipe in timing + area:
            repeats = 10 if recipe == "T_A" and design_index == 0 else 1
            for _ in range(repeats):
                gain = (
                    10.0 if recipe == "T_A" and design_index == 0 else
                    3.0 if recipe == "T_B" else 0.0
                )
                rows.append(_row(design, recipe, index, gain))
                index += 1
    ranked = rank_observations(rows, {"timing": timing, "area": area})
    assert ranked["timing"][0]["recipe_id"] == "T_B"
    assert ranked["timing"][0]["beta_pct"] == pytest.approx(3.0)
    assert ranked["timing"][1]["recipe_id"] == "T_A"
    assert ranked["timing"][1]["beta_pct"] == pytest.approx(2.0)
    assert [row["recipe_id"] for row in ranked["area"]] == sorted(area)


def test_null_payload_carries_explicit_provisional_label() -> None:
    rankings = {
        "timing": [{"recipe_id": "T", "beta_pct": 1.25}],
        "area": [{"recipe_id": "A", "beta_pct": 2.5}],
    }
    metadata = {
        "journal_event_count": 99,
        "journal_snapshot_sha256": "a" * 64,
        "feature_spec_sha256": "b" * 64,
        "observed_at_utc": "2026-11-04T18:00:00Z",
    }
    payload = null_payload("timing", rankings, metadata)
    assert payload["status"] == "PROVISIONAL"
    assert payload["schema_version"].endswith("provisional")
    assert payload["worker"] == "timing"
    assert payload["beta_pct_by_recipe"] == {"T": 1.25}
    assert payload["journal_snapshot_sha256"] == "a" * 64

