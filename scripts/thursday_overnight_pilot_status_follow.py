"""Bounded, training-only status snapshots for the Thursday v4 pilot.

The helper writes JSON under status_follow and invokes the existing checked
reporter, which writes notes/thursday_overnight_dev.md. It never starts a
recipe, meter, or validator job.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.engine.store import _atomic_bytes, _file_lock, _json_bytes, _sha256_file
from scripts import thursday_overnight_pilot_report as report
from scripts import thursday_overnight_pilot_v4 as pilot_v4

PROJECT = pilot_v4.PROJECT
ROOT = pilot_v4.ROOT
SCRIPT = Path(__file__).resolve()
DIRECTORY = ROOT / "status_follow"
LAUNCH = DIRECTORY / "launch.json"
STATUS = DIRECTORY / "status.json"
SNAPSHOT = DIRECTORY / "snapshot.json"
DEADLINE = datetime(2026, 11, 6, 7, 15, tzinfo=timezone.utc)
INTERVAL_SECONDS = 300
TERMINAL_PHASES = frozenset({"complete", "capped", "failed"})


def utc_now() -> datetime:
    """Return an aware clock reading so the overnight cutoff is absolute."""
    return datetime.now(timezone.utc)


def seconds_until_next(now: datetime) -> float:
    """Bound each poll by both the regular interval and the UTC deadline."""
    if now.tzinfo is None:
        raise ValueError("status follower needs an aware UTC clock")
    return max(0.0, min(float(INTERVAL_SECONDS), (DEADLINE - now).total_seconds()))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _launch_record() -> dict[str, Any]:
    """Require the exact checked v4 launch, promotion, and source hashes."""
    value = _read(LAUNCH)
    if (
        value.get("schema_version") != "thursday-overnight-pilot-status-follow-v1"
        or value.get("helper_sha256") != _sha256_file(SCRIPT)
        or value.get("reporter_sha256") != _sha256_file(Path(report.__file__))
        or value.get("pilot_plan_sha256") != _sha256_file(ROOT / "plan.json")
        or value.get("deadline_utc") != DEADLINE.isoformat()
        or value.get("interval_seconds") != INTERVAL_SECONDS
    ):
        raise ValueError("status follower launch or frozen source changed")
    promotion_sha = value.get("promotion_audit_sha256")
    if not isinstance(promotion_sha, str):
        raise ValueError("status follower lacks the promotion audit hash")
    pilot_v4._promotion(promotion_sha)
    pilot_v4._launch_manifest(promotion_sha)
    return value


def checked_table_totals(table: list[dict[str, Any]], completed_rows: int) -> dict[str, int]:
    """Validate the entire seven-policy, four-brief table and sum exact rows."""
    expected = {(policy, brief) for policy in report.POLICIES for brief in report.BRIEFS}
    pairs = [(row.get("policy"), row.get("brief")) for row in table]
    if len(table) != len(expected) or len(set(pairs)) != len(pairs) or set(pairs) != expected:
        raise ValueError("pilot table does not contain exactly the frozen 28 cells")
    counts = (
        "completed", "planned", "brief_valid_final_picks", "defined_shares",
        "runs_used", "cache_hits", "new_real_runs",
        "repair_final_picks", "gentle_final_picks",
    )
    for row in table:
        if any(type(row.get(key)) is not int or row[key] < 0 for key in counts):
            raise ValueError("pilot table has a missing or negative exact count")
        if (
            row["planned"] != len(report.DESIGNS)
            or row["completed"] > row["planned"]
            or row["brief_valid_final_picks"] > row["completed"]
            or row["defined_shares"] > row["completed"]
            or row["cache_hits"] > row["runs_used"]
            or row["repair_final_picks"] + row["gentle_final_picks"] > row["completed"]
        ):
            raise ValueError("pilot table contains impossible counts")
    totals = {
        "completed_episodes": sum(row["completed"] for row in table),
        "valid_finals": sum(row["brief_valid_final_picks"] for row in table),
        "recipe_runs_used": sum(row["runs_used"] for row in table),
        "cache_hits": sum(row["cache_hits"] for row in table),
        "new_real_runs_in_completed_episodes": sum(row["new_real_runs"] for row in table),
        "repair_final_picks": sum(row["repair_final_picks"] for row in table),
        "gentle_final_picks": sum(row["gentle_final_picks"] for row in table),
    }
    if totals["completed_episodes"] != completed_rows:
        raise ValueError("pilot table and checked completed-result rows differ")
    return totals


def _phase() -> str:
    """Use the watchdog's terminal status when it has been sealed."""
    sealed = ROOT / "v4_sharded_status.json"
    path = sealed if sealed.is_file() else ROOT / "status.json"
    if not path.is_file():
        return "prepared"
    phase = _read(path).get("phase")
    if phase not in {"prepared", "running", "complete", "capped", "failed"}:
        raise ValueError("v4 pilot phase is unknown")
    return phase


def _shards() -> dict[str, dict[str, Any]]:
    """Copy bounded progress fields, without reading process names or outcomes."""
    result: dict[str, dict[str, Any]] = {}
    for shard in range(pilot_v4.SHARDS):
        path = ROOT / f"v4_shard_{shard}.json"
        if not path.is_file():
            continue
        item = _read(path)
        if item.get("shard") != shard:
            raise ValueError("v4 shard status identity differs")
        if item.get("phase") not in {"running", "complete", "capped", "failed"}:
            raise ValueError("v4 shard status phase is unknown")
        result[str(shard)] = {
            key: item.get(key)
            for key in ("phase", "planned", "processed", "error", "updated_at_utc")
        }
    return result


def collect() -> dict[str, Any]:
    """Read the verified training aggregation and publish one atomic JSON snapshot."""
    plan = report._preflight_sources()
    starts, best, source = report._baseline()
    episodes, extra, ledger = report._v4_events(starts, best)
    source.update(extra)
    plan_sha = _sha256_file(ROOT / "plan.json")
    counter, reservations = report._counter(plan_sha)
    rows = report._completed(starts, best, episodes, ledger, reservations)
    table = report._table(rows)
    audits = report._audit_state(plan_sha)
    totals = checked_table_totals(table, len(rows))
    cumulative = len(counter["reservations"])
    if not 62 <= cumulative <= pilot_v4.CAP:
        raise ValueError("v4 cumulative recipe reservations escaped the cap")
    totals.update({
        "planned_episodes": len(report.DESIGNS) * len(report.BRIEFS) * len(report.POLICIES),
        "cumulative_recipe_reservations": cumulative,
        "preserved_v3_reservations": 62,
        "new_v4_recipe_reservations": cumulative - 62,
        "remaining_recipe_reservations": pilot_v4.CAP - cumulative,
        "journaled_episodes": len(episodes),
        "physically_valid_step_occurrences": source["v4_physically_valid_step_occurrences"],
    })
    shards = _shards()
    cells = [
        {
            **row,
            "cell_status": "complete" if row["completed"] == row["planned"] else "partial",
        }
        for row in table
    ]
    shard_counts = {
        phase: sum(item["phase"] == phase for item in shards.values())
        for phase in ("running", "complete", "capped", "failed")
    }
    shard_counts["not_started"] = pilot_v4.SHARDS - len(shards)
    snapshot = {
        "schema_version": "thursday-overnight-pilot-status-snapshot-v1",
        "training_only": True,
        "generated_at_utc": utc_now().isoformat(),
        "deadline_utc": DEADLINE.isoformat(),
        "pilot_plan_sha256": plan_sha,
        "registry_sha256": plan["source_hashes"]["registry"],
        "worker_model_sha256": {
            key: plan["source_hashes"][key] for key in ("A_fit", "B_fit")
        },
        "pilot_phase": _phase(),
        "exact_counts": totals,
        "policy_brief": cells,
        "shards": shards,
        "shard_counts": shard_counts,
        "censored_shards": sorted(
            shard for shard, item in shards.items() if item["phase"] == "capped"
        ),
        "failed_shards": sorted(
            shard for shard, item in shards.items() if item["phase"] == "failed"
        ),
        "source_counts": {
            "training_edges": source["training_edges"],
            "candidate_counts": source["candidate_counts"],
            "v4_distinct_physically_valid_states": source["v4_distinct_physically_valid_states"],
        },
        "audit_status": {
            key: {"status": value["status"], "sha256": value.get("sha256")}
            for key, value in audits.items()
        },
        "follower_tool_jobs_launched": 0,
    }
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(SNAPSHOT, _json_bytes(snapshot))
    return snapshot


def _write_status(
    phase: str, cycles: int, error: str | None = None,
    reporter: dict[str, Any] | None = None,
) -> None:
    """Publish checked progress and any report error without touching the pilot."""
    if phase not in {"running", "complete", "capped", "failed", "deadline"}:
        raise ValueError("status follower phase is unknown")
    payload = {
        "schema_version": "thursday-overnight-pilot-status-follow-v1",
        "phase": phase,
        "cycles": cycles,
        "error": error,
        "reporter": reporter,
        "updated_at_utc": utc_now().isoformat(),
        "pilot_phase": _phase(),
        "snapshot_sha256": _sha256_file(SNAPSHOT) if SNAPSHOT.is_file() else None,
        "follower_tool_jobs_launched": 0,
    }
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(STATUS, _json_bytes(payload))


def _refresh_report(reporter: dict[str, Any]) -> None:
    """Run the existing checked no-tool reporter and retain any failure."""
    reporter["attempts"] += 1
    try:
        result = report.generate()
        reporter["successes"] += 1
        reporter["current_error"] = None
        reporter["last_success_at_utc"] = utc_now().isoformat()
        reporter["last_report_sha256"] = _sha256_file(report.REPORT)
        reporter["last_report_words"] = result["report_words"]
        reporter["last_report_completed_episodes"] = result["completed_episodes"]
    except Exception as exc:
        reporter["errors"] += 1
        reporter["current_error"] = f"{type(exc).__name__}: {exc}"
        reporter["last_error"] = reporter["current_error"]
        reporter["last_error_at_utc"] = utc_now().isoformat()


def run() -> None:
    """Refresh checked snapshots and the report until terminal or 07:15 UTC."""
    cycles = 0
    reporter: dict[str, Any] = {
        "attempts": 0, "successes": 0, "errors": 0,
        "current_error": None, "last_error": None,
        "last_success_at_utc": None, "last_error_at_utc": None,
        "last_report_sha256": None, "last_report_words": None,
        "last_report_completed_episodes": None,
    }
    try:
        while utc_now() < DEADLINE:
            _launch_record()
            snapshot = collect()
            _refresh_report(reporter)
            cycles += 1
            phase = snapshot["pilot_phase"]
            # Retry an auxiliary report failure without signalling any pilot job.
            if phase in TERMINAL_PHASES and reporter["current_error"] is None:
                _write_status(phase, cycles, reporter=reporter)
                return
            _write_status("running", cycles, reporter=reporter)
            delay = seconds_until_next(utc_now())
            if delay <= 0:
                break
            time.sleep(delay)
        # The final checked refresh runs at the cutoff before sealing status.
        # Reporter errors remain auxiliary and are recorded in that status.
        _launch_record()
        collect()
        _refresh_report(reporter)
        cycles += 1
        _write_status("deadline", cycles, reporter=reporter)
    except Exception as exc:
        _write_status(
            "failed", cycles, f"{type(exc).__name__}: {exc}", reporter=reporter
        )
        raise


def launch() -> dict[str, Any]:
    """Start the no-tool follower in its own recorded process group."""
    if utc_now() >= DEADLINE:
        raise ValueError("status follower deadline has elapsed")
    v4_manifest = _read(ROOT / "v4_launch.json")
    promotion_sha = v4_manifest.get("promotion_audit_sha256")
    if not isinstance(promotion_sha, str):
        raise ValueError("v4 pilot has no promotion audit hash")
    pilot_v4._promotion(promotion_sha)
    pilot_v4._launch_manifest(promotion_sha)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with _file_lock(DIRECTORY / "launch.lock"):
        if LAUNCH.exists():
            raise FileExistsError("v4 status follower already launched")
        value = {
            "schema_version": "thursday-overnight-pilot-status-follow-v1",
            "started_at_utc": utc_now().isoformat(),
            "deadline_utc": DEADLINE.isoformat(),
            "interval_seconds": INTERVAL_SECONDS,
            "helper_sha256": _sha256_file(SCRIPT),
            "reporter_sha256": _sha256_file(Path(report.__file__)),
            "pilot_plan_sha256": _sha256_file(ROOT / "plan.json"),
            "promotion_audit_sha256": promotion_sha,
            "follower_tool_jobs_launched": 0,
        }
        _atomic_bytes(LAUNCH, _json_bytes(value))
        with (DIRECTORY / "stdout").open("wb") as out, (
            DIRECTORY / "stderr"
        ).open("wb") as err:
            child = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--run"],
                cwd=PROJECT,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
                close_fds=True,
            )
        value["pid"] = child.pid
        value["pgid"] = child.pid
        _atomic_bytes(LAUNCH, _json_bytes(value))
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true")
    group.add_argument("--launch", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.once:
        snapshot = collect()
        print(json.dumps({
            "pilot_phase": snapshot["pilot_phase"],
            "exact_counts": snapshot["exact_counts"],
            "snapshot": str(SNAPSHOT),
        }, sort_keys=True))
    elif args.launch:
        print(json.dumps(launch(), sort_keys=True))
    elif args.run:
        run()
    else:
        print(json.dumps(_read(STATUS), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())