"""Plan shape and freeze enforcement of the test-campaign launcher (no tool, no launch)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import test_campaign as tc


def test_episode_plan_counts_tiers_and_identity() -> None:
    rows = tc.episodes()
    assert len(rows) == 370 and len({r["episode_id"] for r in rows}) == 370
    assert len(tc.episodes(3)) == 420
    for design in tc.DESIGNS:
        mine = [r for r in rows if r["design"] == design]
        assert len(mine) == 74 and sum(r["brief"] == "P0" for r in mine) == 18
    assert [r["tier"] for r in rows] == sorted(r["tier"] for r in rows)
    approaches = {r["approach"] for r in rows}
    assert {"v2", "v1", "single", "F1-R1", "F1-R0", "E0-R1", "GA-R1", "random_2", "fixed_AB", "fixed_CBA"} <= approaches


def test_tier_sizes_follow_the_priority_rule() -> None:
    rows = tc.episodes()
    tier = {t: sum(r["tier"] == t for r in rows) for t in (1, 2, 3)}
    assert tier == {1: 270, 2: 50, 3: 50}


def test_caps_are_the_decided_ones_and_the_token_cap_is_three_times_training_use() -> None:
    caps = tc.caps()
    assert (caps["new_real_runs"], caps["job_hours"], caps["usd"]) == (800, 240, 50)
    assert caps["tokens"] >= 3 * tc.expected_tokens() and caps["tokens"] < 3 * tc.expected_tokens() + 100_000
    assert tc.caps(3)["tokens"] > caps["tokens"]


def test_a_changed_pinned_file_or_cap_fails_the_freeze_check(tmp_path: Path, monkeypatch) -> None:
    saved = json.loads(tc.FREEZE.read_text(encoding="utf-8"))
    assert tc.check_freeze()["ok"] is True
    saved["files"]["configs/briefs.yaml"] = "0" * 64
    altered = tmp_path / "freeze.json"
    altered.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(tc, "FREEZE", altered)
    result = tc.check_freeze()
    assert result["ok"] is False and result["files_changed"] == ["configs/briefs.yaml"]
    saved["files"]["configs/briefs.yaml"] = tc._sha256_file(tc.PROJECT / "configs/briefs.yaml")
    saved["caps"] = dict(saved["caps"], new_real_runs=801)
    altered.write_text(json.dumps(saved), encoding="utf-8")
    assert tc.check_freeze()["ok"] is False


def test_the_freeze_list_covers_what_the_task_names() -> None:
    files = set(tc.FREEZE_FILES)
    for needed in ("configs/prompts/orchestrator_v2.txt", "configs/policies/llm.yaml", "configs/profiles/manifest.json",
                   "tools/llm_bridge/call.py", "tools/llm_bridge/requirements.lock", "orch/policies/llm.py",
                   "orch/baselines/general_agent.py", "notes/analysis_plan.md", "scripts/campaign_analysis.py",
                   "data/test_campaign_prep_2026-11-06/cache_manifest.json", "scripts/campaign_harness.py"):
        assert needed in files


def test_the_launcher_refuses_the_test_scope_without_an_approved_freeze() -> None:
    assert tc.launcher_refuses_test_scope_without_freeze()
    source = (tc.PROJECT / "scripts/test_campaign.py").read_text(encoding="utf-8")
    assert "--launch" not in source.replace("--launch is", "")
