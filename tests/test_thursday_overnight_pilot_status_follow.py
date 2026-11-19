"""No-tool checks for the bounded Thursday v4 pilot status follower."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts import thursday_overnight_pilot_status_follow as follower


def _empty_table() -> list[dict]:
    return [
        {
            "policy": policy,
            "brief": brief,
            "completed": 0,
            "planned": len(follower.report.DESIGNS),
            "brief_valid_final_picks": 0,
            "mean_share_of_best": None,
            "defined_shares": 0,
            "runs_used": 0,
            "cache_hits": 0,
            "new_real_runs": 0,
            "repair_final_picks": 0,
            "gentle_final_picks": 0,
        }
        for policy in follower.report.POLICIES
        for brief in follower.report.BRIEFS
    ]


def test_exact_table_shape_and_counts() -> None:
    table = _empty_table()
    table[0].update(
        completed=2, brief_valid_final_picks=2, defined_shares=2,
        runs_used=8, cache_hits=6, new_real_runs=2,
        repair_final_picks=1, gentle_final_picks=1,
    )
    totals = follower.checked_table_totals(table, 2)
    assert totals["completed_episodes"] == 2
    assert totals["cache_hits"] == 6
    assert totals["new_real_runs_in_completed_episodes"] == 2
    assert totals["repair_final_picks"] == 1
    table[1]["policy"] = table[0]["policy"]
    table[1]["brief"] = table[0]["brief"]
    with pytest.raises(ValueError, match="28 cells"):
        follower.checked_table_totals(table, 2)


def test_table_rejects_impossible_cache_count() -> None:
    table = _empty_table()
    table[0]["cache_hits"] = 1
    with pytest.raises(ValueError, match="impossible counts"):
        follower.checked_table_totals(table, 0)


def test_absolute_deadline_bounds_polling() -> None:
    assert follower.seconds_until_next(follower.DEADLINE - timedelta(seconds=9)) == 9
    assert follower.seconds_until_next(follower.DEADLINE - timedelta(minutes=20)) == 300
    assert follower.seconds_until_next(follower.DEADLINE) == 0
    with pytest.raises(ValueError, match="aware UTC"):
        follower.seconds_until_next(datetime(2026, 11, 6, 7, 14))


def test_collect_publishes_only_own_training_snapshot(monkeypatch, tmp_path) -> None:
    table = _empty_table()
    table[0].update(
        completed=1, brief_valid_final_picks=1, defined_shares=1,
        runs_used=2, cache_hits=1, new_real_runs=1,
    )
    source = {"training_edges": 750, "candidate_counts": {"depth3": 570}}
    monkeypatch.setattr(follower, "ROOT", tmp_path)
    monkeypatch.setattr(follower, "DIRECTORY", tmp_path / "status_follow")
    monkeypatch.setattr(follower, "SNAPSHOT", tmp_path / "status_follow/snapshot.json")
    monkeypatch.setattr(follower, "_sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(follower, "_phase", lambda: "running")
    monkeypatch.setattr(follower, "_shards", lambda: {})
    monkeypatch.setattr(
        follower, "utc_now",
        lambda: datetime(2026, 11, 5, 22, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        follower.report, "_preflight_sources",
        lambda: {"source_hashes": {"registry": "r", "A_fit": "a", "B_fit": "b"}},
    )
    monkeypatch.setattr(follower.report, "_baseline", lambda: ({}, {}, source))
    monkeypatch.setattr(
        follower.report, "_v4_events",
        lambda starts, best: ({}, {
            "v4_physically_valid_step_occurrences": 0,
            "v4_distinct_physically_valid_states": 0,
        }, None),
    )
    monkeypatch.setattr(
        follower.report, "_counter",
        lambda plan_sha: ({"reservations": [{}] * 65}, {}),
    )
    monkeypatch.setattr(
        follower.report, "_completed",
        lambda starts, best, episodes, ledger, reservations: [{}],
    )
    monkeypatch.setattr(follower.report, "_table", lambda rows: table)
    monkeypatch.setattr(follower.report, "_audit_state", lambda plan_sha: {})
    monkeypatch.setattr(
        follower.subprocess, "Popen",
        lambda *args, **kwargs: pytest.fail("no tool or child process is allowed"),
    )
    result = follower.collect()
    assert result["training_only"] is True
    assert result["exact_counts"]["new_v4_recipe_reservations"] == 3
    assert result["exact_counts"]["completed_episodes"] == 1
    assert result["follower_tool_jobs_launched"] == 0
    published = json.loads(follower.SNAPSHOT.read_text())
    assert len(published["policy_brief"]) == 28
    assert published["policy_brief"][0]["cell_status"] == "partial"
    assert published["shard_counts"]["not_started"] == follower.pilot_v4.SHARDS
    assert published["censored_shards"] == []
    assert published["failed_shards"] == []
    assert not (tmp_path / "notes").exists()

def test_report_error_is_recorded_and_retried_without_stopping_pilot(
    monkeypatch,
) -> None:
    phases = iter(("running", "complete"))
    reports = iter((RuntimeError("temporary report read"), {
        "report_words": 400, "completed_episodes": 1,
    }))
    statuses: list[tuple[str, dict]] = []

    def fake_report() -> dict:
        result = next(reports)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        follower, "utc_now",
        lambda: datetime(2026, 11, 5, 22, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(follower, "_launch_record", lambda: {})
    monkeypatch.setattr(follower, "collect", lambda: {"pilot_phase": next(phases)})
    monkeypatch.setattr(follower.report, "generate", fake_report)
    monkeypatch.setattr(follower, "_sha256_file", lambda path: "b" * 64)
    monkeypatch.setattr(follower.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        follower, "_write_status",
        lambda phase, cycles, error=None, reporter=None: statuses.append(
            (phase, dict(reporter or {}))
        ),
    )

    follower.run()
    assert statuses[0][0] == "running"
    assert statuses[0][1]["current_error"] == "RuntimeError: temporary report read"
    assert statuses[-1][0] == "complete"
    assert statuses[-1][1]["successes"] == 1
    assert statuses[-1][1]["errors"] == 1
    assert statuses[-1][1]["current_error"] is None

def test_deadline_takes_final_snapshot_and_records_report_error(
    monkeypatch,
) -> None:
    collected: list[int] = []
    statuses: list[tuple[str, int, dict]] = []

    def fake_collect() -> dict:
        collected.append(1)
        return {"pilot_phase": "running"}

    def failed_report() -> dict:
        raise RuntimeError("final report audit failed")

    monkeypatch.setattr(follower, "utc_now", lambda: follower.DEADLINE)
    monkeypatch.setattr(follower, "_launch_record", lambda: {})
    monkeypatch.setattr(follower, "collect", fake_collect)
    monkeypatch.setattr(follower.report, "generate", failed_report)
    monkeypatch.setattr(
        follower, "_write_status",
        lambda phase, cycles, error=None, reporter=None: statuses.append(
            (phase, cycles, dict(reporter or {}))
        ),
    )

    follower.run()
    assert len(collected) == 1
    assert statuses[-1][0:2] == ("deadline", 1)
    assert statuses[-1][2]["attempts"] == 1
    assert statuses[-1][2]["errors"] == 1
    assert statuses[-1][2]["current_error"] == "RuntimeError: final report audit failed"