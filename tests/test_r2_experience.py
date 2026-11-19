"""Round-2 experience: leave-one-design-out tables, fuzzy and count text, and the greedy reference rule."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from orch.analysis.profiles import frequency_word as round1_frequency_word
from orch.policies.r2_experience import (
    Call, SCOPES, build_table, frequency_word, greedy_score, greedy_scores, load_calls, pick_greedy,
    render_experience_section, scope_of, strength_word,
)

ROOT = Path(__file__).resolve().parents[1]
CALLS_FILE = ROOT / "configs/experience/calls_ab.json"
KEEP, FREE = SCOPES


def _calls() -> list[Call]:
    """Three designs. The area worker re-places well on d1 and d2 but d3 shows a large loss only there."""

    out: list[Call] = []
    for design, wns_free in (("d1", "small_loss"), ("d2", "small_loss"), ("d3", "large_loss")):
        for _ in range(4):
            out.append(Call(design, "area", FREE, True, {"packed_luts": "large_gain", "wns": wns_free}))
            out.append(Call(design, "area", KEEP, True, {"packed_luts": "within_noise", "wns": "within_noise"}))
            out.append(Call(design, "timing", KEEP, True, {"packed_luts": "within_noise", "wns": "small_gain"}))
            out.append(Call(design, "timing", FREE, True, {"packed_luts": "small_loss", "wns": "small_gain"}))
    out.append(Call("d1", "area", FREE, False, {}))
    return out


def test_frequency_words_equal_the_round_1_words_on_a_grid() -> None:
    for i in range(0, 101):
        share = i / 100.0
        assert frequency_word(share) == round1_frequency_word(share), share


@pytest.mark.parametrize(("calls", "word"), [
    (0, "few observations"), (9, "few observations"), (10, "some observations"),
    (29, "some observations"), (30, "many observations"), (200, "many observations"),
])
def test_the_evidence_word_follows_the_pooled_number_of_calls(calls, word) -> None:
    assert strength_word(calls) == word


def test_leaving_a_design_out_removes_its_calls_before_anything_is_counted() -> None:
    everything = build_table(_calls(), "src")
    lodo = build_table(_calls(), "src", exclude_design="d3")
    assert everything.designs == ("d1", "d2", "d3") and lodo.designs == ("d1", "d2")
    assert everything.get("area", FREE).calls == 13 and lodo.get("area", FREE).calls == 9
    assert everything.get("area", FREE).counts["wns"]["large_loss"] == 4
    assert lodo.get("area", FREE).counts["wns"]["large_loss"] == 0
    assert lodo.excluded == "d3" and everything.excluded is None
    assert everything.sha256() != lodo.sha256()
    assert build_table(_calls(), "src", exclude_design="d3").sha256() == lodo.sha256()


def test_shares_weigh_every_design_the_same_and_counts_pool_them() -> None:
    table = build_table(_calls(), "src")
    free = table.get("area", FREE)
    assert free.shares["wns"]["small_loss"] == pytest.approx(2 / 3)       # d1 and d2 of three designs
    assert free.shares["wns"]["large_loss"] == pytest.approx(1 / 3)
    assert free.counts["wns"] == {"small_loss": 8, "large_loss": 4, "large_gain": 0, "small_gain": 0, "within_noise": 0}
    assert free.invalid_rate == pytest.approx((1 / 5) / 3)                 # d1 has 1 invalid of 5 calls
    assert free.designs == 3 and free.valid_calls == 12


def test_the_fuzzy_form_has_words_and_no_numbers() -> None:
    text = render_experience_section(
        build_table(_calls(), "src"), goal="packed_luts", limits=["wns"], workers=["timing", "area"], form="fuzzy",
    )
    assert not re.search(r"\d", text), text
    assert "re-placing allowed (some observations)" in text and "placement preserved (some observations)" in text
    assert "brief goal packed_luts: large_gain usually" in text
    assert "brief limit wns: small_loss often, large_loss sometimes" in text
    assert text.index("timing") < text.index("area")
    assert text.splitlines()[0].startswith("Past experience with the workers")


def test_the_count_form_shows_pooled_counts() -> None:
    text = render_experience_section(
        build_table(_calls(), "src"), goal="packed_luts", limits=["wns"], workers=["area"], form="count",
    )
    assert "re-placing allowed (13 calls on 3 designs)" in text
    assert "brief goal packed_luts: large_gain 12 (of 12)" in text
    assert "brief limit wns: small_loss 8, large_loss 4 (of 12)" in text
    assert "no usable result 1 of 13" in text and "no change 0 of 13" in text


def test_the_greedy_score_is_goal_value_minus_limit_worsening() -> None:
    table = build_table(_calls(), "src")
    free = table.get("area", FREE)
    expected = (1 - free.invalid_rate) * (2.0 * 1.0 - 1.0)                 # large_gain all, wns worsens in all valid calls
    assert greedy_score(free, "packed_luts", ["wns"]) == pytest.approx(expected)
    scores = greedy_scores(table, "packed_luts", ["wns"], ["timing", "area"])
    assert set(scores) == {(w, p) for w in ("timing", "area") for p in (True, False)}
    assert scores[("area", False)] > scores[("area", True)]


def test_the_greedy_pick_skips_tried_pairs_and_breaks_ties_by_card_order() -> None:
    table = build_table(_calls(), "src")
    best = pick_greedy(table, "packed_luts", ["wns"], ["timing", "area"])
    assert (best.worker, best.preserve_placement) == ("area", False)
    nxt = pick_greedy(table, "packed_luts", ["wns"], ["timing", "area"], tried={("area", False)})
    assert (nxt.worker, nxt.preserve_placement) != ("area", False)
    everything = {(w, p) for w in ("timing", "area") for p in (True, False)}
    assert pick_greedy(table, "packed_luts", ["wns"], ["timing", "area"], tried=everything) is None
    tie = build_table([Call("d", w, s, True, {"packed_luts": "within_noise"}) for w in ("timing", "area") for s in SCOPES], "s")
    assert pick_greedy(tie, "packed_luts", [], ["area", "timing"]).worker == "area"
    assert pick_greedy(tie, "packed_luts", [], ["area", "timing"]).preserve_placement is False


def test_scope_of() -> None:
    assert scope_of(True) == KEEP and scope_of(False) == FREE


@pytest.mark.skipif(not CALLS_FILE.is_file(), reason="call record not built")
def test_the_real_call_record_reproduces_the_round_1_profiles_and_leaves_designs_out() -> None:
    calls, sha = load_calls(CALLS_FILE)
    table = build_table(calls, sha)
    for worker in ("timing", "area"):
        stored = yaml.safe_load((ROOT / f"configs/profiles/{worker}.yaml").read_text(encoding="utf-8"))["settings"]
        for setting, st in stored.items():
            mine = table.get(worker, setting)
            assert (mine.calls, mine.designs) == (st["calls"], st["designs"])
            assert mine.invalid_rate == pytest.approx(st["invalid_rate"], abs=5e-5)
            assert mine.no_change_rate == pytest.approx(st["no_change_rate"], abs=5e-5)
            for metric, cats in st["metrics"].items():
                for category, share in cats.items():
                    assert mine.shares[metric][category] == pytest.approx(share, abs=5e-5)
    for held in sorted({c.design for c in calls}):
        lodo = build_table(calls, sha, exclude_design=held)
        assert held not in lodo.designs
        assert sum(st.calls for st in lodo.stats.values()) == sum(c.design != held for c in calls)


@pytest.mark.skipif(not CALLS_FILE.is_file(), reason="call record not built")
def test_the_greedy_rule_prefers_the_area_worker_free_to_replace_on_the_area_brief() -> None:
    calls, sha = load_calls(CALLS_FILE)
    for held in (None, "vexriscv_re-place", "rosetta_spam-filter", "finn_radioml", "rosetta_3d-rendering", "logicnets_jscl"):
        pick = pick_greedy(build_table(calls, sha, exclude_design=held), "packed_luts", ["wns"], ["timing", "area"])
        assert (pick.worker, pick.preserve_placement) == ("area", False), held
