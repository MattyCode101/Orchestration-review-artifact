"""Integrity checks for the FINAL training-only E1 landscape."""

from __future__ import annotations

from pathlib import Path

import pytest

from orch.analysis.landscape_e1 import SCENARIOS, TRAIN_ROOT, best_choice, load_training
from orch.analysis.landscape_e1_final import baseline_shares, final_rankings


@pytest.fixture(scope="module")
def final_snapshot():
    if not TRAIN_ROOT.is_dir():
        pytest.skip("training campaign is not present")
    return load_training(TRAIN_ROOT, final=True)


def test_d3_final_snapshot_is_all_amd_pass_and_deeper_than_e1(final_snapshot) -> None:
    assert final_snapshot.journal_edge_count == 750
    assert len(final_snapshot.edges) == 750
    assert all(edge.amd_status == "pass" for edge in final_snapshot.edges.values())
    assert sum(len(edge.sequence) > 2 for edge in final_snapshot.edges.values()) == 200
    two = best_choice(final_snapshot, "vexriscv_re-place", SCENARIOS[5], max_depth=2)
    full = best_choice(final_snapshot, "vexriscv_re-place", SCENARIOS[5])
    assert len(two.sequence) == 2 and len(full.sequence) == 4
    assert full.metrics.packed_luts < two.metrics.packed_luts


def test_final_fixed_baselines_have_180_bounded_share_rows(final_snapshot) -> None:
    rankings, hashes = final_rankings(final_snapshot, TRAIN_ROOT)
    assert len(rankings) == 10
    assert all(len(order) == 5 for order in rankings.values())
    assert len(hashes) == 2 and all(len(digest) == 64 for digest in hashes.values())
    rows = baseline_shares(final_snapshot, rankings)
    assert len(rows) == 180
    assert {row["brief"] for row in rows} == {"T", "A"}
    assert all(
        share is None or 0 <= share <= 1 + 1e-9
        for share in (row["share_of_best"] for row in rows)
    )
    assert sum(row["baseline"].startswith("worker ") for row in rows) == 60

