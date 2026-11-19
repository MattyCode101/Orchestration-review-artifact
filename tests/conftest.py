"""Tests that verify the authors' frozen research records are skipped when those records are not part of this release.

Each rule is (test id prefix, path that must exist). With the record in place the tests run unchanged.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRES = [
    ("tests/test_launch_freeze.py", "notes/analysis_plan.md"),
    ("tests/test_test_campaign.py::test_a_changed_pinned_file_or_cap_fails_the_freeze_check", "notes/analysis_plan.md"),
    ("tests/test_bandit_frozen.py::test_frozen_files_are_the_declared_ones", "notes/bandit_test_freeze_plan.md"),
    ("tests/test_landscape_e1_final.py", "data/sweeps/worker_fitting_train/episodes"),
    ("tests/test_fitted_episode.py::test_saved_provider_prefers_saved_utilization_and_checks_state", "data/exp_evening_2026-11-04/training_telemetry_v3/plan.json"),
    ("tests/test_fitted_episode.py::test_post_plan_d3_state_uses_saved_report_and_flags_missing_path", "data/exp_evening_2026-11-04/training_telemetry_v3/plan.json"),
    ("tests/test_orchestrator_ceiling.py::test_cache_oracle_refuses_a_non_training_root", "data/pilots/thursday_overnight_train_v4_2026-11-05"),
    ("tests/test_overnight_test_baseline.py::test_frozen_scope_is_seven_test_starts_and_ten_selected_recipes", "data/exp_evening_2026-11-04/test_noise_family/plan.json"),
    ("tests/test_development_designs.py::test_development_configs_agree_with_the_screen", "../benchmark_expansion_2026-11-10/data/b2/t2_cordic_s/start.json"),
]


def pytest_collection_modifyitems(config, items):
    for item in items:
        for prefix, needed in REQUIRES:
            if item.nodeid.startswith(prefix) and not (ROOT / needed).exists():
                item.add_marker(pytest.mark.skip(reason="needs " + needed + ", a research record that is not part of this release"))
                break
