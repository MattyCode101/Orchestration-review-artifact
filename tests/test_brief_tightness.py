"""Synthetic brief-tightness tests; no Vivado or held-out data."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from orch.analysis.brief_tightness import _state_paths, analyze, evaluate_rule
from orch.analysis.landscape_e1 import Edge, Metrics, Snapshot


def _id(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _metrics(label: str, wns: float, luts: float) -> Metrics:
    return Metrics(
        wns_ns=wns, packed_luts=luts, state_id=_id(label),
        meter_version="test", constraint_fingerprint="xdc", clock_digest="clk",
        wns_method=("ns", "test", "routed"),
        lut_method=("count", "test", "routed"),
    )


def _snapshot(overrides: dict[tuple[str, ...], tuple[str, float, float]]) -> Snapshot:
    start = _metrics("start", -1.0, 100.0)
    rows = {(recipe,): (recipe, -1.0, 100.0) for recipe in ("TA", "TB", "AR", "BR")}
    rows[("TA", "AR")] = ("default-ab", -1.0, 100.0)
    rows[("AR", "TA")] = ("default-ba", -1.0, 100.0)
    rows.update(overrides)
    edges = {}
    for sequence, (label, wns, luts) in sorted(rows.items(), key=lambda row: (len(row[0]), row[0])):
        metrics = _metrics(label, wns, luts)
        parent = start.state_id if len(sequence) == 1 else edges[("toy", sequence[:-1])].output_id
        edges[("toy", sequence)] = Edge(
            design="toy", sequence=sequence, input_id=parent,
            output_id=metrics.state_id, run_id=None, run_status="ok",
            physically_valid=True, metrics=metrics, amd_status="pass",
        )
    return Snapshot(
        starts={"toy": start}, edges=edges,
        recipes={"timing": ("TA", "TB"), "area": ("AR", "BR")},
        noise={}, journal_prefix_sha256="test", journal_edge_count=len(edges),
        source_sha256={},
    )


def _row(snapshot: Snapshot, brief_id: str):
    rankings = {("toy", "timing"): ("TA", "TB"), ("toy", "area"): ("AR", "BR")}
    return next(row for row in analyze(snapshot, rankings) if row.brief.id == brief_id)


def test_brief_b_uses_start_violation_magnitude_and_rejects_regression() -> None:
    snapshot = _snapshot({
        ("TA",): ("timing", -0.5, 100),
        ("AR",): ("area", -1.0, 80),
        ("BR",): ("worse-wns", -1.1, 70),
        ("TA", "AR"): ("joint", -0.5, 80),
    })
    row = _row(snapshot, "B")
    assert row.winner.score == pytest.approx(70.0)
    assert row.winner.candidate.sequence == ("TA", "AR")
    assert row.specialist.worker == "timing"
    assert row.specialist.share == pytest.approx(50 / 70)
    assert row.singles["BR"].share == 0.0


def test_duplicate_hash_keeps_alternative_single_worker_path() -> None:
    snapshot = _snapshot({
        ("TA",): ("first", -0.9, 100),
        ("TA", "AR"): ("shared", -0.5, 100),
        ("TA", "TB"): ("shared", -0.5, 100),
    })
    grouped, _ = _state_paths(snapshot, "toy")
    assert len(grouped[_id("shared")][1]) == 2
    row = _row(snapshot, "T0")
    assert row.winner.candidate.sequence == ("TA", "TB")
    assert not row.winner.needs_both
    assert not row.winner.complexity_needed
    assert row.specialist.share == 1.0


def test_violation_is_checked_on_intermediate_not_final() -> None:
    snapshot = _snapshot({
        ("TA",): ("violating", -0.8, 110),
        ("TA", "AR"): ("recovered", -0.5, 99),
    })
    row = _row(snapshot, "T0")
    assert row.winner.candidate.sequence == ("TA", "AR")
    assert row.winner.needs_both
    assert row.winner.violation_unavoidable
    assert row.winner.candidate.violating_intermediate
    assert row.specialist.share == 0.0
    assert row.a_then_b.share == 1.0


def test_tight_and_default_briefs_apply_distinct_start_relative_limits() -> None:
    snapshot = _snapshot({
        ("TA",): ("plus-luts", -0.5, 100.5),
        ("AR",): ("lower-luts-worse-wns", -1.05, 80),
    })
    assert _row(snapshot, "T").winner.candidate.sequence == ("TA",)
    assert _row(snapshot, "T0").winner.candidate.sequence == ()
    assert _row(snapshot, "A").winner.candidate.sequence == ("AR",)
    assert _row(snapshot, "A0").winner.candidate.sequence == ()


def test_rule_requires_two_complex_designs_and_mean_below_ninety() -> None:
    row = _row(_snapshot({}), "T")
    complex_winner = replace(row.winner, complexity_needed=True)
    low = replace(row.specialist, share=0.5)
    high = replace(row.specialist, share=1.0)
    rows = [replace(row, winner=complex_winner, specialist=low) for _ in range(2)]
    rows += [replace(row, specialist=high) for _ in range(3)]
    assert evaluate_rule("T", rows).keep
    assert not evaluate_rule("T", [replace(item, winner=row.winner) for item in rows]).keep
    assert not evaluate_rule("T", [replace(item, specialist=high) for item in rows]).keep


def test_cli_rejects_non_training_data_root(tmp_path) -> None:
    from orch.analysis.brief_tightness import main

    with pytest.raises(ValueError, match="training allow-list"):
        main(["--data-root", str(tmp_path)])
