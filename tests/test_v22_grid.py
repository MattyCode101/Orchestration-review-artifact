"""Template v2.2 grid driver (revision 2): plan, estimate and the generated queue scripts (new tests; nothing is launched).

The driver modules import the pinned D6-v2 driver, which reads D6_WORK at import; it is pointed at a throw-away folder here.
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ["D6_WORK"] = tempfile.mkdtemp(prefix="mm_v22_grid_test_")
os.environ.pop("V22_SMOKE", None)
sys.path[:0] = [str(ROOT / "scripts/v22_2026-11-13"), str(ROOT / "scripts/d6v2_2026-11-12"), str(ROOT / "scripts")]

if not (ROOT / "scripts/d6v2_2026-11-12/d6_common.py").exists():                 # pragma: no cover
    pytest.skip("the D6-v2 driver is not on this machine", allow_module_level=True)

import v22_common as dc  # noqa: E402
import v22_estimate as est  # noqa: E402
import v22_make_driver as mk  # noqa: E402
import v22_plan as plan  # noqa: E402

TASKS = plan.make_tasks()
D6_RESULTS = ROOT / "data/d6v2_2026-11-12/results"
RUN = [f"V{i}" for i in range(2, 9)]


def test_the_plan_has_the_declared_shape() -> None:
    """Decisions of 2026-11-13 21:45: V0 and V1 carried over, V2 to V8 run, T0 and A0, 8 designs, 3 repeats."""

    assert len(TASKS) == 432 and len({t["episode_id"] for t in TASKS}) == 432
    assert Counter(t["tier"] for t in TASKS) == {0: 96, 1: 112, 2: 224}
    assert {t["approach"] for t in TASKS} == {f"V{i}" for i in range(9)}
    assert {t["brief"] for t in TASKS} == {"T0", "A0"}
    assert Counter(t["tag"] for t in TASKS) == {"r1": 144, "r2": 144, "r3": 144}
    assert Counter(t["approach"] for t in TASKS if t["tier"] > 0) == {v: 48 for v in RUN}     # 2 briefs x 8 designs x 3 repeats
    for variant in (*RUN, "V0", "V1"):
        assert Counter((t["brief"], t["tag"]) for t in TASKS if t["approach"] == variant) == {
            (b, tag): 8 for b in ("T0", "A0") for tag in ("r1", "r2", "r3")}


def test_only_training_and_validation_designs_appear() -> None:
    designs = {t["design"] for t in TASKS}
    assert designs == set(dc.TRAINING) | set(dc.DEVELOPMENT) and len(designs) == 8
    assert {t["scope"] for t in TASKS if t["design"] in dc.TRAINING} == {"train"}
    assert {t["scope"] for t in TASKS if t["design"] in dc.DEVELOPMENT} == {"dev"}
    assert "t2_or1200" not in designs and not designs & set(dc.TEST_DESIGNS if hasattr(dc, "TEST_DESIGNS") else ())


def test_tier_one_is_repeat_one_of_v2_to_v8_and_tier_two_is_repeats_two_and_three() -> None:
    tier1 = [t for t in TASKS if t["tier"] == 1]
    assert all(t["tag"] == "r1" for t in tier1) and not any("carryover" in t for t in tier1)
    assert Counter((t["brief"], t["approach"]) for t in tier1) == {(b, v): 8 for b in ("T0", "A0") for v in RUN}
    tier2 = [t for t in TASKS if t["tier"] == 2]
    assert {t["tag"] for t in tier2} == {"r2", "r3"} and not any("carryover" in t for t in tier2)
    assert Counter((t["brief"], t["approach"], t["tag"]) for t in tier2) == {
        (b, v, tag): 8 for b in ("T0", "A0") for v in RUN for tag in ("r2", "r3")}


def test_the_queue_order_puts_tier_one_first_and_the_scored_cells_first_inside_a_tier() -> None:
    to_run = [t for t in TASKS if "carryover" not in t]
    assert [t["tier"] for t in to_run] == sorted(t["tier"] for t in to_run)
    first = to_run[: 9 * len(RUN)]
    assert {(t["design"], t["brief"]) for t in first} == set(plan.SCORED_CELLS) and all(t["scored_cell"] for t in first)
    assert len(plan.SCORED_CELLS) == 9 and len(plan.cells()) == 16 and len(set(plan.cells())) == 16
    assert [t["approach"] for t in to_run[:7]] == RUN                                  # a cell's variants stay together
    assert Counter(t["scored_cell"] for t in TASKS if t["tier"] == 1) == {True: 63, False: 49}


def test_carry_over_is_v0_and_v1_and_names_a_d6_episode() -> None:
    carried = [t for t in TASKS if "carryover" in t]
    assert len(carried) == 96 and {t["approach"] for t in carried} == {"V0", "V1"} and {t["brief"] for t in carried} == {"T0", "A0"}
    assert all(t["tier"] == 0 for t in carried)
    if D6_RESULTS.is_dir():
        missing = [t["carryover"] for t in carried if not (D6_RESULTS / f"{t['carryover']}.json").is_file()]
        assert not missing, missing[:3]


def test_caps_and_the_tool_job_limit_follow_the_decisions_message() -> None:
    assert dc.CAPS == {"new_real_runs": 300, "job_hours": 120, "tokens": 30_000_000, "usd": 5.0}
    assert dc.MAX_TOOL_JOBS == 15 and dc.LOAD_GUARD == 160
    assert dc.BRIEFS_MAIN == ("T0", "A0")                                                  # power plays no part in this grid


def test_estimate_tiers_add_up_inside_the_caps() -> None:
    result = est.estimate(TASKS)
    assert result["tier1"]["episodes_to_run"] == 112 and result["tier2"]["episodes_to_run"] == 224
    assert result["total"]["episodes_to_run"] == 336 and result["carried_over"] == 96
    assert result["total"]["usd"] == pytest.approx(336 * est.USD_PER_EPISODE, abs=0.01)
    assert result["total"]["new_runs"] < dc.CAPS["new_real_runs"] and result["total"]["job_hours"] < dc.CAPS["job_hours"]
    assert result["total"]["usd"] < dc.CAPS["usd"]
    assert sum(result[k]["new_runs"] for k in ("tier1", "tier2")) == pytest.approx(result["total"]["new_runs"], abs=1)


@pytest.mark.parametrize("kind", ["queue", "ctl", "episode"])
def test_generated_driver_scripts_equal_the_files_on_disk(kind: str) -> None:
    path = ROOT / "scripts/v22_2026-11-13" / f"v22_{kind}.py"
    assert path.read_text(encoding="utf-8") == mk.make(kind)


def test_generated_queue_skips_carryover_has_fifteen_slots_and_no_stale_checkpoint() -> None:
    text = (ROOT / "scripts/v22_2026-11-13/v22_queue.py").read_text(encoding="utf-8")
    assert 'not t.get("blocked") and not t.get("carryover")' in text
    assert '"max_active_with_depth3": 15,' in text and '"max_active": 15,' in text
    assert "2026-11-13T09:00:00" not in text and '"checkpoint_utc": None' in text
    assert "d6_" not in text.replace("d6_common", "").replace("D6_", "")


def test_the_pinned_d6_scripts_are_unchanged_by_the_generator() -> None:
    for kind in ("queue", "ctl", "episode"):
        assert (ROOT / "scripts/d6v2_2026-11-12" / f"d6_{kind}.py").is_file()
        assert mk.make(kind) != (ROOT / "scripts/d6v2_2026-11-12" / f"d6_{kind}.py").read_text(encoding="utf-8")
