"""Pure-function checks for the Friday repair-loop variant sweep driver."""
from __future__ import annotations

import json

import pytest

from scripts import friday_sweep as sweep


def test_grid_is_the_sixteen_predeclared_variants_with_v1_first() -> None:
    assert len(sweep.GRID) == 16 and len(set(sweep.GRID)) == 16
    assert sweep.GRID[0] == (1, True, True, "rollback")
    assert sweep.GRID_IDS[0] == "g01" and sweep.GRID_IDS[-1] == "g16"
    versions = {sweep.policy_for_grid(g).version for g in sweep.GRID_IDS}
    assert len(versions) == 16
    assert sweep.policy_for_grid("g01").version == "1.0-frozen"


def test_phase_one_covers_every_design_brief_and_variant_once() -> None:
    rows = sweep.phase1_rows()
    assert len(rows) == 5 * 2 * 16 == len(set(rows))
    assert {brief for _, brief, _ in rows} == {"A0", "T0"}
    assert [row[0] for row in rows[:32]] == [rows[0][0]] * 32  # design-major order


def test_phase_two_work_file_is_validated(tmp_path) -> None:
    good = {
        "selected_grid_id": "g05",
        "rows": [[d, b, "g05"] for d in sweep.DESIGNS for b in ("T", "A")],
    }
    path = tmp_path / "phase2_work.json"
    path.write_text(json.dumps(good))
    assert len(sweep.phase2_rows(path)) == 10
    for bad_rows in (
        [["vexriscv_re-place", "T0", "g05"]],
        [["not_a_design", "T", "g05"]],
        [["vexriscv_re-place", "T", "g99"]],
        [["vexriscv_re-place", "T", "g06"]],
        [["vexriscv_re-place", "T", "g05"]] * 2,
    ):
        path.write_text(json.dumps({"selected_grid_id": "g05", "rows": bad_rows}))
        with pytest.raises(ValueError):
            sweep.phase2_rows(path)


def test_episode_ids_are_unique_per_cell() -> None:
    ids = {sweep.episode_id(*row) for row in sweep.phase1_rows()}
    assert len(ids) == 160


def test_sweep_pins_exclude_files_the_p0_work_will_edit() -> None:
    pinned = set(sweep.SOURCE_FILES.values())
    assert "orch/policies/repair_loop.py" in pinned
    assert "scripts/friday_sweep.py" in pinned
    for later in ("configs/briefs.yaml", "orch/contract/brief_text.py",
                  "configs/recipe_sets.yaml", "orch/policies/training_baselines.py"):
        assert later not in pinned
