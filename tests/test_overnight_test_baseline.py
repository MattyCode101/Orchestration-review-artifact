"""No-tool safety and scope checks for the held-out overnight baseline."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import overnight_test_baseline as baseline


def test_frozen_scope_is_seven_test_starts_and_ten_selected_recipes() -> None:
    config = baseline._config()
    sources = baseline._sources()
    assert tuple(sources) == baseline.ORDER
    assert len(baseline.SELECTED) == 10
    assert baseline.C0 not in baseline.SELECTED
    assert config["room_screen"]["max_admitted"] == 5
    assert baseline.ROOT.is_relative_to(baseline.PROJECT / "data")
    assert not baseline.ROOT.is_relative_to(
        baseline.PROJECT / "data/sweeps/worker_fitting_train"
    )


def test_watchdog_root_match_requires_path_component_boundary() -> None:
    root = b"/project/data/test_campaign"
    assert baseline._cwd_under_root(root, root)
    assert baseline._cwd_under_root(root + b"/runs/run_1", root)
    assert not baseline._cwd_under_root(root + b"_v2/runs/run_1", root)
    assert not baseline._cwd_under_root(b"/project/data/other", root)


def test_run_cost_requires_cpu_load_rss_and_original_id_for_cache_hits() -> None:
    full = SimpleNamespace(
        wall_s=10.0, cpu_user_s=5.0, cpu_sys_s=1.0,
        peak_memory_bytes=1024, load_1m_start=9.0, load_1m_end=8.0,
        busy_slots_start=4, backpressure_wait_s=0.0,
        backpressure_events=0, cache_hit=True, original_run_id="run_origin",
    )
    assert baseline._cost(full)["original_run_id"] == "run_origin"
    missing = SimpleNamespace(**{**vars(full), "cpu_sys_s": None})
    with pytest.raises(ValueError, match="lacks wall, CPU"):
        baseline._cost(missing)
    no_origin = SimpleNamespace(**{**vars(full), "original_run_id": None})
    with pytest.raises(ValueError, match="original run ID"):
        baseline._cost(no_origin)


def test_amd_pass_requires_nonzero_compared_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "validation"
    run_dir.mkdir()
    log = run_dir / "validate.log"
    log.write_text("Outputs (0):\nOutputs (3):\n")
    assert baseline._outputs_compared({"attempts": [{"run_dir": str(run_dir)}]}) == 3
    log.write_text("Outputs (0):\n")
    assert baseline._outputs_compared({"attempts": [{"run_dir": str(run_dir)}]}) == 0


def test_readable_aliases_cover_intermediate_and_failed_outputs() -> None:
    campaign = object.__new__(baseline.Campaign)
    d2 = {
        "design": baseline.ORDER[0], "phase": "D2",
        "sequence": [baseline.SELECTED[0], baseline.SELECTED[5]],
        "output_sha256": "a" * 64, "status": "ok",
    }
    assert campaign._edge_alias_path(d2) == (
        baseline.ROOT / "by_design" / baseline.ORDER[0] / "depth2"
        / f"{baseline.SELECTED[0]}__{baseline.SELECTED[5]}.dcp"
    )
    failed = {**d2, "status": "failed"}
    assert "/failed/" in str(campaign._edge_alias_path(failed))


def test_one_design_failure_stops_its_queue_but_other_designs_finish() -> None:
    """Only the already in-flight work may finish after an AMD-style failure."""
    import time

    campaign = object.__new__(baseline.Campaign)
    campaign.stopped = {}
    campaign.edges = {}
    campaign.starts = {design: SimpleNamespace(id="b2") for design in baseline.ORDER}
    campaign._phase = lambda _phase: None
    campaign._input_for = lambda _design, _sequence: (
        SimpleNamespace(id="input"), SimpleNamespace(), SimpleNamespace()
    )

    def run_one(design, sequence, phase, *_args):
        if design == baseline.ORDER[0] and sequence == (baseline.C0,):
            time.sleep(0.005)
            status = "failed"
        else:
            time.sleep(0.03)
            status = "ok"
        return {
            "design": design, "sequence": list(sequence), "phase": phase,
            "status": status, "error": "AMD fail" if status == "failed" else None,
        }

    def commit(row):
        key = row["design"], tuple(row["sequence"])
        campaign.edges[key] = row
        if row["status"] != "ok":
            campaign.stopped[row["design"]] = row["error"]

    campaign._run_one = run_one
    campaign._commit_edge = commit
    tasks = [
        (design, (recipe,))
        for design in baseline.ORDER
        for recipe in (baseline.C0, *baseline.SELECTED)
    ]
    campaign._run_layer(tasks, phase="one_step_and_c0", per_design=2, max_active=14)
    assert 1 <= sum(name == baseline.ORDER[0] for name, _ in campaign.edges) <= 2
    assert all(
        sum(name == design for name, _ in campaign.edges) == 11
        for design in baseline.ORDER[1:]
    )
    assert campaign.stopped == {baseline.ORDER[0]: "AMD fail"}


def test_d2_enumerates_all_100_ordered_pairs_per_admitted_design(
    tmp_path: Path, monkeypatch
) -> None:
    frozen = tmp_path / "admitted.json"
    frozen.write_text("{}")
    monkeypatch.setattr(baseline, "ADMISSION", frozen)
    campaign = object.__new__(baseline.Campaign)
    seen = []
    campaign._run_layer = lambda tasks, **_kwargs: seen.extend(tasks)
    campaign.run_d2(list(baseline.ORDER[:5]))
    assert len(seen) == 500
    assert len(set(seen)) == 500
    assert all(len(sequence) == 2 for _, sequence in seen)
    assert {design for design, _ in seen} == set(baseline.ORDER[:5])
