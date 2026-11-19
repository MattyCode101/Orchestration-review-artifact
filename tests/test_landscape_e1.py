"""Synthetic tests for training-only E1 landscape semantics and provenance."""
from __future__ import annotations

import hashlib
import json

import pytest

from orch.analysis.landscape_e1 import (
    Edge, Metrics, Scenario, Snapshot, TRAIN_ROOT, _checked_run,
    best_choice, fixed_baselines, noise_flags, require_training_root,
)


DESIGN = "vexriscv_re-place"
A = "LATCH.phys_opt"
A2 = "LATCH.retime"
B = "A_R_EA"


def metric(state: str, wns: float, luts: float) -> Metrics:
    """Make one synthetic meter record with a shared physical basis."""
    return Metrics(
        wns_ns=wns, packed_luts=luts, state_id=state * 64,
        meter_version="meter-test", constraint_fingerprint="xdc-test",
        clock_digest="clock-test",
        wns_method=("ns", "run3_timing_summary", "routed"),
        lut_method=("count", "run3_report_utilization", "routed"),
    )


def edge(path: tuple[str, ...], parent: Metrics, output: Metrics) -> Edge:
    """Make an eligible synthetic edge; no hardware or training data read."""
    return Edge(
        design=DESIGN, sequence=path, input_id=parent.state_id,
        output_id=output.state_id, run_id="run_" + "f" * 32,
        run_status="ok", physically_valid=True, metrics=output,
        amd_status="pass",
    )


def snapshot() -> Snapshot:
    """A crossed path recovers a Brief T violation and beats a timing path."""
    start = metric("a", -0.5, 100)
    first = metric("b", -0.1, 103)
    crossed = metric("c", 0.1, 100)
    timing = metric("d", 0.0, 100)
    edges = {
        (DESIGN, (A,)): edge((A,), start, first),
        (DESIGN, (A, B)): edge((A, B), first, crossed),
        (DESIGN, (A2,)): edge((A2,), start, timing),
    }
    return Snapshot(
        starts={DESIGN: start}, edges=edges,
        recipes={"timing": (A, A2), "area": (B,)},
        noise={DESIGN: {"wns_ns": 0.05, "packed_luts": 1.0}},
        journal_prefix_sha256="0" * 64, journal_edge_count=3,
        source_sha256={},
    )


def test_training_allow_list_rejects_arbitrary_root(tmp_path):
    """The public loader cannot be redirected to test or held-out data."""
    assert require_training_root(TRAIN_ROOT) == TRAIN_ROOT
    with pytest.raises(ValueError, match="training allow-list"):
        require_training_root(tmp_path)


def test_cross_worker_needed_and_violating_intermediate():
    """A physically valid first step may violate T and be repaired by B."""
    data = snapshot()
    choice = best_choice(
        data, DESIGN, Scenario("T", "timing", lut_cap_pct=1.0)
    )
    assert choice.sequence == (A, B)
    assert choice.cross_worker is True
    assert choice.cross_worker_needed is True
    assert choice.intermediate_violates is True


def test_cross_worker_not_needed_when_within_worker_ties():
    """A crossed selected path is not necessary when an equal local path exists."""
    data = snapshot()
    first = data.edges[(DESIGN, (A,))].metrics
    assert first is not None
    equal = metric("e", 0.1, 100)
    data.edges[(DESIGN, (A2, A))] = edge((A2, A), data.edges[(DESIGN, (A2,))].metrics, equal)
    choice = best_choice(
        data, DESIGN, Scenario("T", "timing", lut_cap_pct=1.0)
    )
    assert choice.sequence == (A, B)
    assert choice.cross_worker is True
    assert choice.cross_worker_needed is False


def test_start_relative_grid_and_noise_unknown():
    """The cap and floor are relative to B2; missing noise is not zero noise."""
    data = snapshot()
    start = data.starts[DESIGN]
    first = data.edges[(DESIGN, (A,))].metrics
    assert first is not None
    assert Scenario("tight T", "timing", lut_cap_pct=0.5).valid(first, start) is False
    assert Scenario("loose T", "timing", lut_cap_pct=5).valid(first, start) is True
    assert Scenario("tight A", "area", wns_floor_ns=0.05).valid(first, start) is True
    assert noise_flags(data, DESIGN, metric("x", -0.46, 101)) == (True, True)
    data.noise[DESIGN] = {}
    assert noise_flags(data, DESIGN, first) == (None, None)


def test_known_amd_fail_cannot_win():
    """A known failed output is excluded even in a provisional description."""
    data = snapshot()
    crossed = data.edges[(DESIGN, (A, B))]
    data.edges[(DESIGN, (A, B))] = Edge(
        **{**crossed.__dict__, "amd_status": "fail"}
    )
    choice = best_choice(
        data, DESIGN, Scenario("T", "timing", lut_cap_pct=1.0)
    )
    assert choice.sequence == (A2,)


def test_fixed_baselines_include_every_single_and_both_orders():
    """The two fixed pair orders must be present in the D2 snapshot."""
    data = snapshot()
    start = data.starts[DESIGN]
    area = metric("e", -0.5, 95)
    data.edges[(DESIGN, (B,))] = edge((B,), start, area)
    data.edges[(DESIGN, (B, A))] = edge(
        (B, A), area, metric("f", -0.1, 96)
    )
    baselines = fixed_baselines(data, DESIGN, A, B)
    assert len(baselines) == 5
    assert [name for name, _, _ in baselines[-2:]] == ["A then B", "B then A"]
    assert {path for _, path, _ in baselines[:3]} == {(A,), (A2,), (B,)}


def test_complete_marker_detects_tampered_evidence(tmp_path):
    """No raw run artifact is trusted without its complete.json digest."""
    run_id = "run_" + "f" * 32
    directory = tmp_path / "runs" / run_id
    directory.mkdir(parents=True)
    artifacts = {
        "runinfo.json": b'{"id":"run_' + b"f" * 32 + b'"}\n',
        "evidence.json": b'{"state_id":"x"}\n',
    }
    for name, content in artifacts.items():
        (directory / name).write_bytes(content)
    (directory / "complete.json").write_text(json.dumps({
        "run_id": run_id,
        "files": {name: hashlib.sha256(content).hexdigest()
                  for name, content in artifacts.items()},
    }))
    assert _checked_run(tmp_path, run_id)["evidence.json"]["state_id"] == "x"
    (directory / "evidence.json").write_text('{"state_id":"changed"}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        _checked_run(tmp_path, run_id)



def test_exhaustive_repeat_winner_has_no_repeat_alternative():
    """D2 can measure a repeat that the frozen worker cannot plan."""
    data = snapshot()
    first = data.edges[(DESIGN, (A2,))].metrics
    assert first is not None
    repeated = metric("z", 0.2, 100)
    data.edges[(DESIGN, (A2, A2))] = edge((A2, A2), first, repeated)
    scenario = Scenario("T", "timing", lut_cap_pct=1.0)
    assert best_choice(data, DESIGN, scenario).sequence == (A2, A2)
    assert best_choice(
        data, DESIGN, scenario, allow_repeat=False
    ).sequence == (A, B)

