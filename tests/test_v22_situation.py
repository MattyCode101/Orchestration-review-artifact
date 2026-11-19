"""Template v2.2, switch S' (situation experience): rows, detection, lines and the stored table (new tests)."""
from __future__ import annotations

import json
from pathlib import Path

import llm_support as ls
import pytest
from orch.policies.v22_situation import (
    BUCKETS, bucket_of, classify_base, dedupe, detect, load_rows, render_situation, rows_from_calls_v2,
    rows_from_diagnosis_episode, rows_from_v4_episode,
)
from orch.policies.v22_switches import load_switch_texts
from orch.contract.brief_text import load_frozen_brief

ROOT = Path(__file__).resolve().parents[1]
TEXTS = load_switch_texts(ROOT)["situation"]
TABLE = ROOT / "configs/experience/situations_v22.json"
DESIGNS = ("finn_radioml", "logicnets_jscl", "rosetta_3d-rendering", "rosetta_spam-filter", "vexriscv_re-place",
           "t2_cordic_s", "vtr_mcml_v2", "amd_mini-isp")


def row(design: str, worker: str, preserve: bool, situation: str, *, depth: float | None = None, outcome: str = "selected",
        d_goal: float | None = 2.0, d_limit: float | None = 0.0, restored: bool | None = None, brief: str = "T0",
        base: str = "b", n_runs: int = 2, source: str = "v4") -> dict:
    goal, limit = ("wns_ns", "packed_luts") if brief == "T0" else ("packed_luts", "wns_ns")
    return {"design": design, "group": "TRAINING", "brief": brief, "source": source, "worker": worker, "preserve": preserve,
            "base": base, "n_runs": n_runs, "situation": situation, "depth_b": depth, "outcome": outcome, "d_goal_b": d_goal,
            "d_limit_b": d_limit, "restored": restored, "goal": goal, "limit": limit}


AA, BB, CC, DD = "d_alpha", "d_beta", "d_gamma", "d_delta"


def grid_rows(designs=(AA, BB, CC, DD)) -> list[dict]:
    out = []
    for d in designs:
        for worker in ("timing", "area"):
            for preserve in (True, False):
                out.append(row(d, worker, preserve, "start", d_goal=2.0 if worker == "timing" else 0.0))
                out.append(row(d, worker, preserve, "gain", d_goal=0.2, d_limit=-0.1))
                out.append(row(d, worker, preserve, "break", depth=3.0, restored=(worker == "area" and not preserve),
                               outcome="selected" if worker == "area" else "nothing_selectable",
                               d_goal=-0.5 if worker == "area" else None, d_limit=3.0 if worker == "area" else None))
    return out


def view_open():
    return ls.view(visited=[ls.state(10, -0.30, 1000)])


def view_broken():
    """Start, then S1: WNS up two bands, packed LUTs over the start by three bands (the limit is broken by 3.0 bands)."""

    visited = [ls.state(10, -0.30, 1000), ls.state(11, -0.10, 1030, brief_valid=False)]
    calls = (ls.call(1, "timing", 10, 11, attempts=2, runs=2, reason="push"),)
    return ls.view(visited=visited, calls=calls, steps=(ls.step("timing", 10, 11),), budget_left=4)


def view_gained():
    visited = [ls.state(10, -0.30, 1000), ls.state(11, -0.10, 990)]
    calls = (ls.call(1, "timing", 10, 11, attempts=2, runs=2, reason="push"),)
    return ls.view(visited=visited, calls=calls, steps=(ls.step("timing", 10, 11),), budget_left=4, best=visited[1])


def view_nothing_gained():
    visited = [ls.state(10, -0.30, 1000), ls.state(11, -0.28, 995)]
    calls = (ls.call(1, "timing", 10, 11, attempts=2, runs=2, reason="push"),)
    return ls.view(visited=visited, calls=calls, steps=(ls.step("timing", 10, 11),), budget_left=4)


# ------------------------------------------------------------------------------------------------ classification and rows
def test_buckets_follow_the_declared_depths() -> None:
    assert [b[0] for b in BUCKETS] == ["under half a band", "half a band to two bands", "over two bands"]
    assert bucket_of(0.0) == "under half a band" and bucket_of(0.4999) == "under half a band"
    assert bucket_of(0.5) == "half a band to two bands" and bucket_of(1.99) == "half a band to two bands"
    assert bucket_of(2.0) == "over two bands" and bucket_of(250.0) == "over two bands"


def test_a_base_state_is_start_gain_break_or_other() -> None:
    assert classify_base(0.0, 0.0, True, True) == ("start", None)
    assert classify_base(2.0, 0.5, True, False) == ("gain", None)
    assert classify_base(2.0, -3.0, False, False) == ("break", 3.0)
    assert classify_base(0.6, -3.0, False, False) == ("other", None)          # no goal gain of a band: not a repair situation
    assert classify_base(-2.0, 1.0, True, False) == ("other", None)


def test_dedupe_keeps_the_most_complete_call_and_uses_the_exhaustive_row_only_when_no_episode_covers_it() -> None:
    a = row("d", "timing", True, "start", n_runs=1, source="d6v2", d_goal=1.0)
    b = row("d", "timing", True, "start", n_runs=3, source="v4", d_goal=2.0)
    c = row("d", "timing", True, "start", n_runs=0, source="calls_v2", d_goal=9.0)
    other = row("d", "area", True, "start", n_runs=0, source="calls_v2", d_goal=4.0)
    kept = dedupe([a, b, c, other])
    assert len(kept) == 2
    assert next(r for r in kept if r["worker"] == "timing")["d_goal_b"] == 2.0           # most runs, not the exhaustive row
    assert next(r for r in kept if r["worker"] == "area")["source"] == "calls_v2"        # nothing else covers it
    tie = dedupe([row("d", "timing", True, "start", n_runs=2, source="d6v2", d_goal=1.0),
                  row("d", "timing", True, "start", n_runs=2, source="v4", d_goal=1.0)])
    assert [r["source"] for r in tie] == ["v4"]


def test_diagnosis_episode_rows_use_the_guarded_selection_and_the_restored_flag() -> None:
    steps = [{"out": "x1", "gain_b": 2.0, "margin_b": -3.0, "valid": False}, {"out": "x2", "gain_b": 1.5, "margin_b": 0.5, "valid": True}]
    ep = {"design": "t2_cordic_s", "group": "VALIDATION", "brief": "A0",
          "states": {"s0": {"gain_b": 0.0, "margin_b": 0.0, "valid": True, "call": -1},
                     "s1": {"gain_b": 2.0, "margin_b": -3.0, "valid": False, "call": 0}},
          "calls": [{"worker": "area", "base": "s0", "preserve": False, "steps": steps},
                    {"worker": "timing", "base": "s1", "preserve": True, "steps": steps},
                    {"worker": "timing", "base": "s1", "preserve": False, "steps": [{**steps[0]}]}]}
    rows = rows_from_diagnosis_episode(ep)
    first, kept, nothing = rows
    assert (first["situation"], first["outcome"], first["d_goal_b"]) == ("start", "selected", 1.5)    # area picks by its goal
    assert (kept["situation"], kept["depth_b"], kept["restored"]) == ("break", 3.0, True)
    assert kept["d_limit_b"] == pytest.approx(3.5)                                                     # margin -3.0 -> +0.5
    assert (nothing["outcome"], nothing["restored"]) == ("nothing_selectable", False)
    assert all(r["group"] == "VALIDATION" for r in rows)
    assert rows_from_diagnosis_episode({**ep, "brief": "P0"}) == []


def test_v4_episode_rows_rebuild_states_in_bands() -> None:
    brief = load_frozen_brief("A0")
    bands = {"packed_luts": 10.0, "wns_ns": 0.1}
    ok = {"out_id": "o1", "physically_valid": True, "status": "ACCEPTABLE", "metrics": {"packed_luts": 800.0, "wns_ns": -0.5}}
    bad = {"out_id": "o2", "physically_valid": True, "status": "BRIEF_VIOLATION", "metrics": {"packed_luts": 700.0, "wns_ns": -2.0}}
    ep = {"design": "d1", "brief": "A0", "start_state_id": "s", "start_metrics": {"packed_luts": 1000.0, "wns_ns": -0.5},
          "calls": [{"completed": True, "worker": "area", "base_id": "s", "preserve": False, "steps": [ok, bad]},
                    {"completed": True, "worker": "timing", "base_id": "o2", "preserve": True, "steps": [ok]},
                    {"completed": False, "worker": "area", "base_id": "s", "preserve": True, "steps": [ok]}]}
    rows = rows_from_v4_episode(ep, brief, bands, "TRAINING")
    assert [r["situation"] for r in rows] == ["start", "break"]                       # the unfinished call is left out
    assert rows[0]["d_goal_b"] == pytest.approx(20.0)                                 # 200 fewer LUTs = 20 bands
    assert rows[1]["restored"] is True and rows[1]["depth_b"] == pytest.approx(15.0)  # WNS 1.5 ns under the start = 15 bands


def test_exhaustive_call_rows_come_from_start_and_one_step_bases() -> None:
    brief = {"T0": load_frozen_brief("T0")}
    start = {"wns_ns": -0.3, "packed_luts": 1000.0}
    items = [{"design": "d1", "brief": "T0", "worker": "timing", "scope": "preserve_placement_false", "outcome": "selected",
              "base": "start", "base_state": "s0", "base_values": start, "bands": {"wns_ns": 0.1, "packed_luts": 10.0},
              "result_values": {"wns_ns": -0.1, "packed_luts": 1000.0}},
             {"design": "d1", "brief": "T0", "worker": "area", "scope": "preserve_placement_true", "outcome": "selected",
              "base": "one_step", "base_state": "s1", "base_values": {"wns_ns": -0.1, "packed_luts": 1030.0},
              "bands": {"wns_ns": 0.1, "packed_luts": 10.0}, "result_values": {"wns_ns": -0.1, "packed_luts": 990.0}}]
    rows = rows_from_calls_v2(items, brief, {"d1": "TRAINING"})
    assert [(r["situation"], r["d_goal_b"]) for r in rows] == [("start", pytest.approx(2.0)), ("break", pytest.approx(0.0))]
    assert rows[1]["restored"] is True and rows[1]["depth_b"] == pytest.approx(3.0)


# ------------------------------------------------------------------------------------------------ detection
def test_detection_names_the_situation_of_the_decision() -> None:
    assert detect(view_open()).opening and not detect(view_open()).start_lines
    found = detect(view_broken())
    assert found.repair == (("S1", pytest.approx(3.0), "packed_luts"),) and found.push_from is None and not found.opening
    pushed = detect(view_gained())
    assert pushed.push_from == "S1" and pushed.repair == () and not pushed.start_lines
    quiet = detect(view_nothing_gained())
    assert quiet.start_lines and quiet.repair == () and quiet.push_from is None


def test_a_broken_state_with_no_option_left_for_any_worker_is_not_a_repair_situation() -> None:
    visited = [ls.state(10, -0.30, 1000), ls.state(11, -0.10, 1030, brief_valid=False)]
    spent = tuple(ls.call(i, w, 11, None, preserve=p, attempts=2, runs=1)
                  for i, (w, p) in enumerate((("timing", True), ("timing", False), ("area", True), ("area", False)), start=2))
    assert detect(ls.view(visited=visited, calls=spent, budget_left=2)).repair == ()


# ------------------------------------------------------------------------------------------------ the lines
def test_opening_lines_use_the_experience_words_and_leave_the_evaluated_design_out() -> None:
    text, meta = render_situation(grid_rows(), view_open(), design="z", texts=TEXTS, workers=["timing", "area"])
    assert text.startswith(TEXTS["header"] + "\n" + TEXTS["intro"] + "\n" + TEXTS["opening"] + "\ntiming\n")
    assert "  placement must not change (4 calls from 4 designs): goal wns:" in text
    assert "limit packed_luts:" in text and "nothing selectable rarely" in text
    assert meta["blocks"] == ["opening"] and meta["excluded_design"] == "z" and len(meta["shown"]) == 4
    no_line = ": " + TEXTS["no_evidence"]                     # the intro sentence names the phrase too, so count line endings
    only_c = [r for r in grid_rows() if r["design"] != DD]
    short, meta2 = render_situation(only_c, view_open(), design=CC, texts=TEXTS, workers=["timing", "area"])
    assert short.count(no_line) == 4 and not meta2["shown"]                               # two other designs: not enough
    assert "(2 calls" not in short and "calls from 2 designs" not in short
    full, _ = render_situation(grid_rows(), view_open(), design=DD, texts=TEXTS, workers=["timing", "area"])
    assert "(3 calls from 3 designs)" in full and no_line not in full


def test_a_line_needs_three_designs_and_names_no_design() -> None:
    rows = [r for r in grid_rows((AA, BB, CC)) if not (r["worker"] == "area" and r["preserve"])]
    text, meta = render_situation(rows, view_open(), design="zz", texts=TEXTS, workers=["timing", "area"])
    assert "  placement must not change: no evidence for this situation" in text
    assert "placement must not change (3 calls from 3 designs)" in text                          # the timing line
    assert len(meta["shown"]) == 3 and len(meta["omitted"]) == 1
    for name in (*DESIGNS, AA, BB, CC, DD):
        assert name not in text
    assert "recommend" not in text.lower()


def test_a_worker_nobody_called_gets_no_line_and_other_briefs_get_no_section() -> None:
    text, _ = render_situation(grid_rows(), view_open(), design="z", texts=TEXTS, workers=["timing", "area", "power"])
    assert "\npower\n" not in text
    t0_only = grid_rows()
    text2, meta = render_situation(t0_only, ls.view(visited=[ls.state(10, -0.3, 1000)], brief=ls.brief_a()), design="z",
                                   texts=TEXTS, workers=["timing", "area"])
    assert text2 == "" and meta["blocks"] == []


def test_repair_lines_are_per_worker_per_depth_and_per_placement() -> None:
    text, meta = render_situation(grid_rows(), view_broken(), design="z", texts=TEXTS, workers=["timing", "area"])
    assert meta["blocks"] == ["repair"] and meta["repair_states"] == [["S1", pytest.approx(3.0), "packed_luts"]]
    assert "S1 (packed_luts broken by 3.0 noise bands)" in text
    assert TEXTS["repair_bucket"].format(bucket="over two bands") in text
    assert "under half a band in the base state" not in text                                 # only the depth present
    area = text[text.index("\narea\n"):]
    assert "  placement may change (4 calls from 4 designs): restored the limit and kept a goal gain of at least one band usually." in area
    assert "  placement must not change (4 calls from 4 designs): restored the limit and kept a goal gain of at least one band rarely." in area
    assert TEXTS["opening"] not in text and TEXTS["start_no_gain"] not in text              # the other situations are not shown
    deep = [r for r in grid_rows() if r["situation"] != "break"]
    none, _ = render_situation(deep, view_broken(), design="z", texts=TEXTS, workers=["timing", "area"])
    assert none.count(": " + TEXTS["no_evidence"]) == 4


def test_a_push_situation_shows_calls_from_states_that_had_gained() -> None:
    text, meta = render_situation(grid_rows(), view_gained(), design="z", texts=TEXTS, workers=["timing", "area"])
    assert meta["blocks"] == ["push"] and meta["push_from"] == "S1"
    assert text.count(TEXTS["push"].format(alias="S1")) == 1 and TEXTS["opening"] not in text
    assert "goal wns: within noise" in text


def test_a_decision_with_no_gain_yet_shows_the_start_lines_under_their_own_title() -> None:
    text, meta = render_situation(grid_rows(), view_nothing_gained(), design="z", texts=TEXTS, workers=["timing", "area"])
    assert meta["blocks"] == ["start"] and TEXTS["start_no_gain"] in text and TEXTS["opening"] not in text


# ------------------------------------------------------------------------------------------------ the stored table
@pytest.mark.skipif(not TABLE.exists(), reason="the situation table has not been built on this machine")
def test_the_stored_table_holds_only_training_and_validation_calls_on_t0_and_a0() -> None:
    rows, sha = load_rows(TABLE)
    assert len(sha) == 64 and rows
    assert {r["design"] for r in rows} <= set(DESIGNS) and "t2_or1200" not in {r["design"] for r in rows}
    assert {r["brief"] for r in rows} <= {"T0", "A0"} and {r["situation"] for r in rows} <= {"start", "gain", "break"}
    assert {r["group"] for r in rows} <= {"TRAINING", "VALIDATION"}
    assert all(r["depth_b"] is not None for r in rows if r["situation"] == "break")
    assert all(r["restored"] is None for r in rows if r["situation"] != "break")
    keys = {(r["design"], r["brief"], r["base"], r["worker"], r["preserve"]) for r in rows}
    assert len(keys) == len(rows)                                                             # one row per combination
    payload = json.loads(TABLE.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v22-situation-calls-v2" and "episodes_v4" in payload["source"]


@pytest.mark.skipif(not TABLE.exists(), reason="the situation table has not been built on this machine")
def test_every_line_the_table_supports_rests_on_other_designs_only() -> None:
    rows, _ = load_rows(TABLE)
    for design in DESIGNS:
        text, meta = render_situation(rows, view_open(), design=design, texts=TEXTS, workers=["timing", "area", "power"])
        assert design not in text and meta["excluded_design"] == design
        for entry in meta["shown"]:
            others = {r["design"] for r in rows if r["brief"] == "T0" and r["situation"] == "start"
                      and r["worker"] == entry["worker"] and r["preserve"] == entry["preserve"] and r["design"] != design}
            assert entry["designs"] == len(others) >= 3
