"""Synthetic and saved-result checks for exploratory worker comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orch.analysis.explore_workers import (
    DETAIL, LinearModel, StateRows, _context, _linear_fit, _linear_scores,
    _matrix, switch_rule,
)
from orch.analysis.fit_workers import load_feature_spec

PROJECT = Path(__file__).resolve().parents[1]


def test_switch_requires_ten_percent_and_four_strict_folds() -> None:
    final = [10.0] * 5
    assert switch_rule(final, [8.0, 8.0, 8.0, 8.0, 10.0])["eligible_for_author_review"]
    assert not switch_rule(final, [8.0, 8.0, 8.0, 10.0, 10.0])["eligible_for_author_review"]
    assert not switch_rule(final, [9.5] * 5)["eligible_for_author_review"]
    with pytest.raises(ValueError, match="five"):
        switch_rule(final[:4], final)


def test_all_feature_clips_and_pls_recovers_one_predictive_axis() -> None:
    recipes = ("a", "b")
    states = [
        StateRows(
            design="d0" if i < 2 else "d1", state_id=str(i),
            context="B2", features={"signal": float(i), "constant": 1.0},
            gains={"a": 2.0 * i, "b": 1.0},
        )
        for i in range(4)
    ]
    model = _linear_fit(states, recipes, method="V3", setting=1)
    assert model.columns == ("signal",)
    held = [
        StateRows("held", "middle", "B2", {"signal": 1.5, "constant": 1.0}, {"a": 0, "b": 0}),
        StateRows("held", "high", "B2", {"signal": 100.0, "constant": 1.0}, {"a": 0, "b": 0}),
    ]
    predicted = _linear_scores(model, held)
    assert predicted[0, 0] == pytest.approx(3.0, abs=1e-8)
    assert predicted[1, 0] == pytest.approx(6.0, abs=1e-8)
    assert predicted[0, 1] == pytest.approx(1.0, abs=1e-8)
    ridge = _linear_fit(states, recipes, method="V2", setting=1e-6)
    assert _linear_scores(ridge, held)[0, 0] == pytest.approx(3.0, abs=1e-4)


def test_canonical_b2_context_overrides_noop_lineage_ambiguity() -> None:
    spec = load_feature_spec()
    state_id = "a" * 64
    alternatives = {
        "timing": SimpleNamespace(sequence=("A_R_EA", "LATCH.phys_opt")),
        "area": SimpleNamespace(sequence=("LATCH.retime", "A_R_EA")),
    }
    assert _context(alternatives, spec, state_id, {state_id}) == "B2"
    assert _context(alternatives, spec, state_id, set()) == "ambiguous"


@pytest.mark.skipif(not DETAIL.is_file(), reason="exploratory artifact not generated")
def test_final_one_try_folds_match_published_fit_record() -> None:
    analysis = json.loads(DETAIL.read_text())
    assert analysis["label"] == "EXPLORATORY"
    assert analysis["model_switch_performed"] is False
    for worker, name in (("timing", "A_fit.json"), ("area", "B_fit.json")):
        published = json.loads((PROJECT / "configs/workers" / name).read_text())
        expected = {
            fold["held_design"]: (
                fold["feature_regret"] if published["keep_features"]
                else fold["null_regret"]
            )
            for fold in published["folds"]
        }
        actual = {
            fold["held_design"]: fold["regret_pct_points"]["FINAL"]["1"]
            for fold in analysis["workers"][worker]["folds"]
        }
        assert actual == pytest.approx(expected)

