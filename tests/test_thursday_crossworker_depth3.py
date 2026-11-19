"""Scope and failure gates for the detached cross-worker training driver."""
from types import SimpleNamespace

import pytest

from scripts.thursday_crossworker_depth3 import (
    RECIPES, complete_cost, paths, tasks,
)


def test_only_five_training_designs_have_output_roots() -> None:
    with pytest.raises(ValueError, match="not a training design"):
        paths("fir_systolic_transposed")


def test_selected_parent_cap_and_dedup() -> None:
    twenty = {"selected_states": [{"hash": f"{i:064x}"} for i in range(20)]}
    assert len(tasks(twenty)) == 20 * len(RECIPES) == 200
    with pytest.raises(ValueError, match="cap exceeded"):
        tasks({"selected_states": twenty["selected_states"] + [{"hash": f"{20:064x}"}]})
    with pytest.raises(ValueError, match="duplicate"):
        tasks({"selected_states": twenty["selected_states"][:2] * 2})


def test_cache_hit_needs_original_id_and_complete_costs() -> None:
    fields = dict(
        wall_s=10.0, cpu_user_s=2.0, cpu_sys_s=1.0, peak_memory_bytes=100,
        load_1m_start=3.0, load_1m_end=4.0, busy_slots_start=2,
        backpressure_wait_s=0.0, backpressure_events=0, cache_hit=True,
        original_run_id=None,
    )
    with pytest.raises(ValueError, match="original run ID"):
        complete_cost(SimpleNamespace(**fields))
    fields["original_run_id"] = "run_original"
    assert complete_cost(SimpleNamespace(**fields))["wall_s"] == 10.0
    fields["cpu_sys_s"] = None
    with pytest.raises(ValueError, match="cpu_sys_s"):
        complete_cost(SimpleNamespace(**fields))
