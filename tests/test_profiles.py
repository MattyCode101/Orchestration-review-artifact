"""Behaviour-profile generator: binning reuse, equal design weights, words, text, hashes."""
from __future__ import annotations

import hashlib
import json
import re

import pytest

import llm_support as ls
from orch.analysis.profiles import (
    Observation, frequency_word, observation_from_metrics, render_recipe_text,
    render_worker_text, summarise, write_profiles,
)
from orch.contract.models import MetricName

BANDS = {MetricName.WNS_NS: 0.1, MetricName.PACKED_LUTS: 10.0}


@pytest.mark.parametrize(("share", "word"), [
    (0.0, "rarely"), (0.149, "rarely"), (0.15, "sometimes"), (0.40, "sometimes"),
    (0.4001, "often"), (0.70, "often"), (0.7001, "usually"), (1.0, "usually"),
])
def test_frequency_words_follow_the_specified_thresholds(share, word) -> None:
    assert frequency_word(share) == word


def _obs(base, result, *, design="d1", setting="preserve_placement_false"):
    return observation_from_metrics(
        design, setting, ls.metrics(*base), None if result is None else ls.metrics(*result),
        start_metrics=ls.metrics(-0.5, 1000), noise=BANDS,
    )


def test_an_observation_bins_every_metric_with_a_band_against_its_base() -> None:
    obs = _obs((-0.5, 1000), (-0.1, 1004))                  # wns +0.4 (large gain); luts +4 (noise)
    assert obs.valid and obs.changes["wns"] == "large_gain" and obs.changes["packed_luts"] == "within_noise"
    assert "power" not in obs.changes                       # no recorded power band, none invented
    assert obs.no_change is False


def test_a_result_inside_noise_on_every_metric_counts_as_no_change() -> None:
    obs = _obs((-0.5, 1000), (-0.49, 1003))
    assert obs.valid and obs.no_change is True


def test_a_missing_result_is_invalid_not_no_change() -> None:
    obs = _obs((-0.5, 1000), None)
    assert obs.valid is False and obs.no_change is False and obs.changes == {}


def _many(design, categories, setting="preserve_placement_false"):
    return [Observation(design, setting, True, {"wns": c, "packed_luts": "within_noise"}, c == "within_noise")
            for c in categories]


def test_each_design_counts_equally_however_many_calls_it_has() -> None:
    observations = _many("big", ["large_gain"] * 10) + _many("small", ["small_gain"])
    stats = summarise(observations)["preserve_placement_false"]
    assert stats.designs == 2 and stats.calls == 11
    assert stats.metrics["wns"]["large_gain"] == pytest.approx(0.5)
    assert stats.metrics["wns"]["small_gain"] == pytest.approx(0.5)


def test_invalid_rate_is_over_all_calls_and_distributions_over_valid_calls() -> None:
    observations = _many("d", ["large_gain", "small_loss"]) + [Observation("d", "preserve_placement_false", False, {}, False)] * 2
    stats = summarise(observations)["preserve_placement_false"]
    assert stats.invalid_rate == pytest.approx(0.5) and stats.no_change_rate == 0.0
    assert stats.metrics["wns"]["large_gain"] == pytest.approx(0.5)


def test_settings_are_summarised_separately() -> None:
    both = _many("d", ["large_gain"], "preserve_placement_true") + _many("d", ["large_loss"], "preserve_placement_false")
    stats = summarise(both)
    assert set(stats) == {"preserve_placement_true", "preserve_placement_false"}
    assert stats["preserve_placement_true"].metrics["wns"] == {"large_gain": 1.0}


def test_worker_text_uses_words_not_numbers_and_orders_by_frequency() -> None:
    observations = (
        _many("a", ["large_gain"] * 6 + ["within_noise"] * 3 + ["small_loss"]) +
        _many("b", ["large_gain"] * 8 + ["within_noise"] * 2) +
        _many("a", ["within_noise"] * 9 + ["small_gain"], "preserve_placement_true") +
        _many("b", ["within_noise"] * 10, "preserve_placement_true")
    )
    text = render_worker_text(summarise(observations), goal="wns")
    assert not re.search(r"\d", text), text
    lines = text.splitlines()
    assert lines[0].lstrip().startswith("preserve_placement=true:") and lines[1].lstrip().startswith("preserve_placement=false:")
    assert "goal within_noise usually" in lines[0]
    assert re.search(r"goal large_gain (often|usually)", lines[1])
    assert lines[1].index("large_gain") < lines[1].index("within_noise")
    assert "no change" in lines[0] and "invalid" in lines[0]
    assert "rarely" in lines[1] and "small_loss" in lines[1]            # a five-percent-plus category is named


def test_recipe_text_states_whether_the_recipe_replaces_the_placement() -> None:
    stats = summarise(_many("a", ["large_gain", "within_noise"], "recipe") + _many("b", ["large_gain"], "recipe"))["recipe"]
    text = render_recipe_text("some_recipe", stats, replaces=True, targets="wns")
    assert "re-places the design" in text and "targets wns" in text
    assert "re-places the design" not in render_recipe_text("other", stats, replaces=False, targets="wns")
    assert "keeps the placement" in render_recipe_text("other", stats, replaces=False, targets="wns")


def test_writing_profiles_hashes_every_file_and_is_deterministic(tmp_path) -> None:
    observations = _many("a", ["large_gain", "within_noise"], "preserve_placement_true") + _many("a", ["small_gain"], "preserve_placement_false")
    stats = summarise(observations)
    worker = {"worker": "timing", "objective": "wns", "stats": stats, "provenance": {"designs": ["a"], "note": "unit"}}
    recipes = {"r1": {"worker": "timing", "replaces": False, "stats": summarise(_many("a", ["large_gain"], "recipe"))["recipe"]}}
    out = tmp_path / "profiles"
    first = write_profiles(out, [worker], recipes, provenance={"meter_version": "m"})
    for name, digest in first["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert set(first["files"]) == {"timing.yaml", "recipes.yaml"}
    assert json.loads((out / "manifest.json").read_text())["files"] == first["files"]
    again = write_profiles(tmp_path / "profiles2", [worker], recipes, provenance={"meter_version": "m"})
    assert again["files"] == first["files"]


def test_the_worker_profile_file_exposes_the_exact_text_for_the_card(tmp_path) -> None:
    import yaml

    stats = summarise(_many("a", ["large_gain"], "preserve_placement_true") + _many("a", ["small_gain"], "preserve_placement_false"))
    write_profiles(tmp_path, [{"worker": "area", "objective": "packed_luts", "stats": stats, "provenance": {}}], {}, provenance={})
    data = yaml.safe_load((tmp_path / "area.yaml").read_text())
    assert data["kind"] == "worker" and data["worker"] == "area"
    assert data["text"].count("\n") >= 1 and data["text_sha256"] == hashlib.sha256(data["text"].encode()).hexdigest()
    assert not re.search(r"\d", data["text"])
