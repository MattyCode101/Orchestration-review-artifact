"""Compatibility and per-card attempt defaults for mandate settings."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from orch.config import load_worker_card
from orch.contract.models import (
    Brief, Mandate, MetricName, Objective, WorkerCard, WorkerSettings,
)
from orch.contract.views import PolicyView, StateSummary
from orch.policies.random_policy import RandomPolicy
from orch.policies.trivial import TrivialPolicy

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("old_allow", "expected_preserve"), [(True, False), (False, True)])
def test_old_journal_mandate_alias_inverts_and_serializes_new_name(
    old_allow: bool, expected_preserve: bool,
) -> None:
    mandate = Mandate.model_validate({
        "id": "legacy", "worker_id": "timing", "base_state_id": "a" * 64,
        "brief_hash": "historical", "settings": {
            "allow_replace": old_allow, "max_runs": 1,
        },
    })
    assert mandate.settings.preserve_placement is expected_preserve
    saved = mandate.model_dump(mode="json")["settings"]
    assert saved == {"preserve_placement": expected_preserve, "max_runs": 1}


def test_legacy_alias_refuses_conflicting_or_nonboolean_values() -> None:
    with pytest.raises(ValidationError, match="both placement settings"):
        WorkerSettings.model_validate({
            "allow_replace": True, "preserve_placement": True,
        })
    with pytest.raises(ValidationError, match="must be a boolean"):
        WorkerSettings.model_validate({"allow_replace": "false"})


def _view(default_attempts: int) -> PolicyView:
    objective = Objective(metric=MetricName.WNS_NS, direction="max")
    brief = Brief(id="timing-test", objective=objective, budget_runs=5)
    card = WorkerCard(
        id="timing", version="test", objective=objective, recipe_set_id="timing",
        default_attempts=default_attempts,
    )
    start = StateSummary(
        state_id="a" * 64, metrics={MetricName.WNS_NS: -1.0},
        physically_valid=True, brief_valid=True,
    )
    return PolicyView(
        brief=brief, budget_left=5, start=start, current=start,
        best_valid=start, worker_cards=(card,), episode_id="settings-test",
    )


def test_card_attempt_default_is_used_when_policy_or_mandate_omits_override() -> None:
    view = _view(3)
    card = view.worker_cards[0]
    assert WorkerSettings().resolved_max_runs(card) == 3
    assert WorkerSettings(max_runs=1).resolved_max_runs(card) == 1
    assert TrivialPolicy().decide(view).settings.max_runs == 3
    assert TrivialPolicy(max_runs=1).decide(view).settings.max_runs == 1
    random_counts = {
        RandomPolicy(seed=seed).decide(view).settings.max_runs
        for seed in range(40)
    }
    assert random_counts == {1, 2, 3}


def test_signed_worker_cards_declare_two_attempts_and_positive_validation() -> None:
    for worker in ("timing", "area"):
        card = load_worker_card(PROJECT / "configs/workers" / f"{worker}.yaml")
        assert card.default_attempts == 2
        assert card.accepted_settings == ("preserve_placement", "max_runs")
    with pytest.raises(ValidationError):
        _view(0)
    with pytest.raises(ValidationError):
        WorkerSettings(max_runs=0)
