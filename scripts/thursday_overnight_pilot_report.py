"""Read-only training-policy reporter for the versioned Thursday overnight pilot.

Only designated report/summary files are written. The denominator reuses the
checked D1-D3, five cross-worker campaign queues, and preserved v3 journal logic from
thursday_night_pilot_report; it adds committed, AMD-PASS v4 steps. No tool is
launched and no test-design outcome root is opened.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orch.analysis.report import share_of_best
from orch.baselines.verification import VerificationLedger
from orch.contract.brief_text import load_frozen_brief
from orch.contract.models import Evidence, RunStatus
from orch.engine.gate import judge
from orch.engine.journal import Journal
from orch.engine.store import StateStore, _atomic_bytes, _json_bytes, _sha256_file
from orch.episode import EpisodeResult
from orch.runtime import make_real_runtime
from scripts import thursday_night_pilot as pilot
from scripts import thursday_night_pilot_report as previous
from scripts.thursday_overnight_pilot_v4 import (
    CAP, PROJECT, ROOT, SCRIPT as V4_DRIVER, V3_ROOT,
)

REPORT = PROJECT / "notes/thursday_overnight_dev.md"
SUMMARY = ROOT / "policy_brief_summary.json"
METER_AUDIT = PROJECT / "data/scratch_thursday_night_2026-11-05/training_meter_equivalence/audit.json"
PROMOTION_AUDIT = PROJECT / "data/scratch_thursday_night_2026-11-05/training_cache_promotion/audit.json"
CEILING_SUMMARY = PROJECT / "data/scratch_thursday_night_2026-11-05/orchestrator_ceiling_summary.json"
OLD_VERSIONS = {
    "e11abbe86bd48e7cfd070dd6d9adeda1424517642cd6c1e09c7e29a727947a8f",
    "2ee8ee9efb9416d84afbef9f1e37350c7b2ea0b054637efd8cff02dbed784435",
}
DESIGNS = pilot.DESIGNS
BRIEFS = pilot.BRIEFS
POLICIES = pilot.POLICIES


def _read(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected JSON object: {path}")
    return result


def _hash(path: Path) -> str:
    return _sha256_file(path)


def _preflight_sources() -> dict[str, Any]:
    """Prove the old checked loader has only existing source dirs and locks."""
    if ROOT.resolve() != (PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05").resolve():
        raise ValueError("v4 reporter escaped its named training root")
    for root in (
        V3_ROOT,
        *(PROJECT / "data/sweeps/thursday_crossworker_depth3_2026-11-05" / design
          for design in DESIGNS),
    ):
        for relative in (
            "states", "runs", "cache", "episodes",
            "verification/plan.json", "verification/plan.lock",
        ):
            if not (root / relative).exists():
                raise FileNotFoundError(f"read-only training source missing: {root / relative}")
    for relative in ("states", "runs", "cache", "plan.json", "new_real_runs.json"):
        if not (ROOT / relative).exists():
            raise FileNotFoundError(f"prepared v4 training root missing: {ROOT / relative}")
    plan = _read(ROOT / "plan.json")
    if (plan.get("schema_version") != "thursday-overnight-pilot-v4"
            or plan.get("training_only") is not True
            or plan.get("data_root") != str(ROOT)
            or plan.get("episode_reuse") is not False
            or plan.get("max_new_real_recipe_runs") != CAP
            or set(plan.get("training_inputs", {})) != set(DESIGNS)
            or plan.get("recipe_versions") !=
            _read(V3_ROOT / "plan.json")["recipe_versions"]):
        raise ValueError("v4 plan is not the frozen five-design training pilot")
    source_files = {
        **pilot.SOURCE_FILES,
        "v4_driver": "scripts/thursday_overnight_pilot_v4.py",
    }
    if set(source_files) != set(plan["source_hashes"]):
        raise ValueError("v4 source pin names differ")
    for key, relative in source_files.items():
        if _hash(PROJECT / relative) != plan["source_hashes"][key]:
            raise ValueError(f"v4 source pin changed: {key}")
    if _hash(V4_DRIVER) != plan["source_hashes"]["v4_driver"]:
        raise ValueError("v4 driver source pin differs")
    return plan


def _journal(root: Path, episode_id: str) -> list[dict[str, Any]]:
    """Read a checksummed journal under its existing lock, creating no files."""
    if not episode_id.startswith("pilot_") or "/" in episode_id:
        raise ValueError("episode identity escaped the frozen schedule")
    path = root / "episodes" / episode_id / "journal.jsonl"
    if not path.is_file():
        return []
    lock = path.parent / "journal.lock"
    if not lock.is_file():
        raise ValueError("committed journal lacks its existing lock")
    fd = os.open(lock, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        events, _ = Journal._parse(path.read_bytes())
        return events
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _baseline() -> tuple[dict[str, Evidence], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    """Reuse the checked denominator while keeping all source stores read-only."""
    _preflight_sources()
    starts, best, source = previous._load_sources()
    store = StateStore(ROOT)
    plan = _read(ROOT / "plan.json")
    for design in DESIGNS:
        state_id = plan["training_inputs"][design]
        current = previous._start_evidence(store, state_id, plan["meter_version"])
        if current.state_id != starts[design].state_id:
            raise ValueError("v4 B2 checkpoint differs from checked v3 baseline")
        previous._same_baseline(starts[design], current)
    source["v3_partial_step_occurrences"] = source["candidate_counts"].get("pilot", 0)
    source["v4_plan_sha256"] = _hash(ROOT / "plan.json")
    return starts, best, source


def _offer(
    best: dict[tuple[str, str], dict[str, Any]],
    design: str, evidence: Evidence, start: Evidence,
) -> None:
    """Use the same objective, validity and hash tie rule as the checked loader."""
    pilot.require_clean_properties(evidence, start)
    for brief_id in BRIEFS:
        brief = load_frozen_brief(brief_id)
        verdict = judge(brief, start, start, evidence)
        if not verdict.brief_valid:
            continue
        metric = evidence.metric(brief.objective.metric)
        if metric is None:
            raise ValueError("brief-valid Evidence lacks objective metric")
        key = (design, brief_id)
        previous_best = best.get(key)
        better = previous_best is None or (
            metric.value > previous_best["value"]
            if brief.objective.direction == "max"
            else metric.value < previous_best["value"]
        )
        if better or (
            previous_best is not None
            and metric.value == previous_best["value"]
            and evidence.state_id < previous_best["state_id"]
        ):
            best[key] = {
                "state_id": evidence.state_id,
                "value": metric.value, "source": "v4 pilot",
                "evidence_id": evidence.id,
            }


def _v4_events(
    starts: dict[str, Evidence],
    best: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], VerificationLedger | None]:
    """Add only committed, physically valid, exact-DCP AMD-PASS v4 steps."""
    store = StateStore(ROOT)
    episodes: dict[str, list[dict[str, Any]]] = {}
    for design in DESIGNS:
        for brief_id in BRIEFS:
            for policy_key in POLICIES:
                identity = pilot.episode_id(design, brief_id, policy_key)
                events = _journal(ROOT, identity)
                if events:
                    episodes[identity] = events
    valid_steps = [
        (identity, event)
        for identity, events in episodes.items()
        for event in events
        if event.get("type") == "step"
        and event.get("verdict", {}).get("physically_valid") is True
    ]
    ledger: VerificationLedger | None = None
    if valid_steps:
        for relative in ("verification/plan.json", "verification/plan.lock"):
            if not (ROOT / relative).is_file():
                raise ValueError("v4 AMD source plan is absent; reporter will not create it")
        runtime = make_real_runtime(
            ROOT, cache_enabled=True, max_slots=pilot.MAX_SLOTS,
            load_ceiling_1m=pilot.LOAD_CEILING,
            min_available_memory_bytes=pilot.MEMORY_FLOOR,
            latch_smoke_status=pilot.SMOKE,
        )
        ledger = VerificationLedger(
            store=store, runner=runtime.runner, validator=runtime.validator,
            references=_read(ROOT / "plan.json")["training_inputs"],
        )
    seen: set[tuple[str, str]] = set()
    for identity, event in valid_steps:
        parts = identity.removeprefix("pilot_")
        design = next((d for d in DESIGNS if parts.startswith(d + "_")), None)
        if design is None:
            raise ValueError("v4 step design escaped the schedule")
        outcome = store.load_outcome(event["step"]["run_id"])
        if (outcome.run.status != RunStatus.OK
                or outcome.state is None or outcome.evidence is None
                or outcome.state.design != design
                or outcome.evidence.state_id != outcome.state.id
                or outcome.evidence.meter_version != starts[design].meter_version
                or outcome.run.input_sha256_before != outcome.run.input_sha256_after):
            raise ValueError("v4 journal step differs from its measured run")
        if ledger is None:
            raise ValueError("v4 step lacks an AMD ledger")
        pilot.require_amd_pass(ledger, design, outcome.state.id, allow_new=False)
        _offer(best, design, outcome.evidence, starts[design])
        seen.add((design, outcome.state.id))
    hashes = {
        identity: hashlib.sha256(_json_bytes(events)).hexdigest()
        for identity, events in episodes.items()
    }
    source = {
        "v4_journal_hashes": hashes,
        "v4_physically_valid_step_occurrences": len(valid_steps),
        "v4_distinct_physically_valid_states": len(seen),
    }
    return episodes, source, ledger


def _counter(plan_sha: str) -> tuple[dict[str, Any], Counter[str]]:
    record = _read(ROOT / "new_real_runs.json")
    rows = record.get("reservations")
    if (record.get("schema_version") != "thursday-night-new-runs-v1"
            or record.get("plan_sha256") != plan_sha
            or record.get("cap") != CAP
            or not isinstance(rows, list)
            or not 62 <= len(rows) <= CAP):
        raise ValueError("v4 cumulative recipe counter differs")
    carry = rows[:62]
    if any(row.get("episode_id") != f"carryover:v3:{i}" for i, row in enumerate(carry)):
        raise ValueError("v4 historical reservations were lost or double-counted")
    ids = [row.get("run_id") for row in rows]
    if any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("v4 counter has duplicate or missing run IDs")
    allowed = {
        pilot.episode_id(design, brief_id, policy_key)
        for design in DESIGNS for brief_id in BRIEFS for policy_key in POLICIES
    }
    new = rows[62:]
    if any(row.get("episode_id") not in allowed for row in new):
        raise ValueError("v4 reservation escaped training schedule")
    return record, Counter(row["episode_id"] for row in new)


def _repair_phase(result: EpisodeResult, events: list[dict[str, Any]]) -> str | None:
    if result.final_rank is None or not result.validation_attempts:
        return None
    chosen = result.validation_attempts[result.final_rank].candidate_run_id
    if chosen is None:
        return None
    steps = {
        event["step"]["run_id"]: event["step"]["mandate_id"]
        for event in events if event.get("type") == "step"
    }
    mandates = {
        event["mandate"]["id"]: event["mandate"]
        for event in events if event.get("type") == "mandate"
    }
    if chosen not in steps or steps[chosen] not in mandates:
        raise ValueError("v4 final run lacks a mandate")
    context = json.loads(mandates[steps[chosen]]["context_ref"])
    return context.get("phase") if context.get("policy") == "repair_loop" else None


def _completed(
    starts: dict[str, Evidence], best: dict[tuple[str, str], dict[str, Any]],
    episodes: dict[str, list[dict[str, Any]]],
    ledger: VerificationLedger | None, reservations: Counter[str],
) -> list[dict[str, Any]]:
    store = StateStore(ROOT)
    results = ROOT / "results"
    allowed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for design in DESIGNS:
        for brief_id in BRIEFS:
            for policy_key in POLICIES:
                identity = pilot.episode_id(design, brief_id, policy_key)
                allowed.add(f"{identity}.json")
                path = results / f"{identity}.json"
                if not path.is_file():
                    continue
                saved = _read(path)
                events = episodes.get(identity) or _journal(ROOT, identity)
                finals = [event["result"] for event in events if event.get("type") == "final"]
                if len(finals) != 1:
                    raise ValueError("v4 result lacks exactly one committed journal final")
                result = EpisodeResult.model_validate(finals[0])
                if (result.episode_id != identity or result.brief_id != brief_id
                        or result.final_state is None or result.final_evidence is None
                        or result.final_verdict is None
                        or result.final_verdict.brief_valid is not True
                        or result.validation_status != "pass"
                        or saved.get("episode_id") != identity
                        or saved.get("design") != design
                        or saved.get("brief") != brief_id
                        or saved.get("policy") != policy_key
                        or saved.get("final_state_id") != result.final_state.id
                        or saved.get("final_brief_valid") is not True
                        or saved.get("validation_status") != result.validation_status
                        or saved.get("runs_used") != result.recipe_runs
                        or saved.get("stop_reason") != result.stop_reason
                        or saved.get("journal_path") != result.journal_path):
                    raise ValueError("v4 saved result differs from committed final")
                store.state_path(result.final_state.id)
                pilot.require_clean_properties(result.final_evidence, starts[design])
                if result.final_state.id != starts[design].state_id:
                    if ledger is None:
                        raise ValueError("v4 derived final lacks AMD ledger")
                    pilot.require_amd_pass(
                        ledger, design, result.final_state.id, allow_new=False
                    )
                expected_metrics = {
                    name.value: metric.value
                    for name, metric in result.final_evidence.metrics.items()
                }
                if saved.get("final_metrics") != expected_metrics:
                    raise ValueError("v4 result metrics differ from journal Evidence")
                brief = load_frozen_brief(brief_id)
                if not judge(
                    brief, starts[design], starts[design], result.final_evidence
                ).brief_valid:
                    raise ValueError("v4 final does not meet its frozen brief")
                steps = [event["step"] for event in events if event.get("type") == "step"]
                if len(steps) != result.recipe_runs:
                    raise ValueError("v4 journal recipe-run count differs")
                hits = sum(
                    store.load_outcome(step["run_id"]).run.cache_hit for step in steps
                )
                if (saved.get("cache_hits") != hits
                        or saved.get("new_real_runs_reserved") != reservations[identity]
                        or saved.get("repair_final_phase") != _repair_phase(result, events)):
                    raise ValueError("v4 result accounting differs from checked journal")
                start_value = starts[design].metric(brief.objective.metric).value
                candidate_value = result.final_evidence.metric(brief.objective.metric).value
                best_value = best[(design, brief_id)]["value"]
                share = share_of_best(
                    start_value, best_value, candidate_value,
                    brief.objective.direction,
                )
                if share is not None and share > 1.000000001:
                    raise ValueError("v4 final exceeds supposedly best known training state")
                rows.append({
                    **saved,
                    "start_value": start_value,
                    "candidate_value": candidate_value,
                    "best_value": best_value,
                    "best_state_id": best[(design, brief_id)]["state_id"],
                    "share_of_best": share,
                })
    if results.is_dir():
        extras = {path.name for path in results.glob("*.json")} - allowed
        if extras:
            raise ValueError("v4 result escaped the frozen training schedule")
    return rows


def _audit_state(plan_sha: str) -> dict[str, Any]:
    """Missing optional audits are pending; malformed present audits stop reporting."""
    out: dict[str, Any] = {}
    if METER_AUDIT.is_file():
        meter = _read(METER_AUDIT)
        versions = meter.get("versions", {})
        if (meter.get("schema_version") != "training-meter-equivalence-audit-v1"
                or set(versions) != OLD_VERSIONS
                or any(v.get("samples") != 5 or v.get("identical") != 5 or v.get("different") != 0
                       for v in versions.values())
                or meter.get("different") != 0
                or meter.get("normalized_differences") != []
                or meter.get("live_exact_tags") != []):
            raise ValueError("present training meter audit does not approve both versions")
        out["meter"] = {
            "status": "approved", "sha256": _hash(METER_AUDIT),
            "versions": versions,
        }
    else:
        out["meter"] = {"status": "pending"}
    if PROMOTION_AUDIT.is_file():
        promotion = _read(PROMOTION_AUDIT)
        if (promotion.get("destination_root") != str(ROOT)
                or promotion.get("pilot_plan_sha256") != plan_sha
                or promotion.get("tool_jobs_launched") != 0
                or promotion.get("promoted_pointers", 0) < 1
                or out["meter"]["status"] != "approved"
                or promotion.get("approval_audit_sha256") != out["meter"]["sha256"]):
            raise ValueError("present training promotion audit differs from approval")
        out["promotion"] = {
            "status": "complete", "sha256": _hash(PROMOTION_AUDIT),
            "promoted_pointers": promotion["promoted_pointers"],
        }
    else:
        out["promotion"] = {"status": "pending"}
    if CEILING_SUMMARY.is_file():
        ceiling = _read(CEILING_SUMMARY)
        if (ceiling.get("schema_version") != "orchestrator-ceiling-training-v1"
                or ceiling.get("training_only") is not True
                or out["promotion"]["status"] != "complete"
                or ceiling.get("promotion_audit_sha256") != out["promotion"]["sha256"]):
            raise ValueError("present training ceiling summary differs from approved cache")
        out["ceiling"] = {
            "status": "complete", "sha256": _hash(CEILING_SUMMARY),
            "per_brief": ceiling["per_brief"],
            "beam_width": ceiling.get("beam_width"),
        }
    else:
        out["ceiling"] = {"status": "pending"}
    return out


def _table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reuse the checked per-policy/per-brief aggregation shape."""
    return previous._aggregate(rows)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _gap(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def render(table: list[dict[str, Any]], status: dict[str, Any],
           source: dict[str, Any], audits: dict[str, Any]) -> str:
    """Compact, clearly labelled 7-by-4 training snapshot."""
    completed = status["completed_episodes"]
    rows = [
        "# Thursday overnight development - training pilot",
        "",
        f"[Verified] V4 pilot: {completed}/140 completed episodes; phase {status['phase']}; "
        f"cumulative recipe reservations {status['cap_used']}/600 "
        f"(62 preserved, {status['new_v4_reservations']} v4). "
        f"Committed v4 steps: {source['v4_physically_valid_step_occurrences']}.",
        "",
    ]
    meter = audits["meter"]
    promotion = audits["promotion"]
    if meter["status"] == "approved":
        rows.append(
            f"[Verified] Training meter equivalence: both old versions sampled 5/5 identical "
            f"to current; audit SHA-256 {meter['sha256']}. "
            f"Cache aliases: {promotion.get('promoted_pointers', 0)} promoted"
            if promotion["status"] == "complete"
            else f"[Verified] Training meter equivalence: both old versions sampled 5/5 identical "
                 f"to current; audit SHA-256 {meter['sha256']}. Cache alias audit pending."
        )
    else:
        rows.append("[Verified] Training meter comparison and cache alias approval pending.")
    rows.extend([
        "",
        "[Verified] Cells show valid finals/5; mean share of best-known improvement; "
        "recipe runs/cache hits/new v4 real runs; repair/gentle final picks. "
        "A dash means no defined share or an uncompleted cell.",
        "",
        "| Policy | T0 | A0 | T | A |",
        "|---|---|---|---|---|",
    ])
    cells = {(row["policy"], row["brief"]): row for row in table}
    for policy in POLICIES:
        values = []
        for brief_id in BRIEFS:
            item = cells[(policy, brief_id)]
            values.append(
                f"{item['brief_valid_final_picks']}/5;{_pct(item['mean_share_of_best'])};"
                f"{item['runs_used']}/{item['cache_hits']}/{item['new_real_runs']};"
                f"{item['repair_final_picks']}/{item['gentle_final_picks']}"
            )
        rows.append(f"| {policy} | " + " | ".join(values) + " |")
    rows.extend([
        "",
        f"[Verified] Best-known pool: {source['training_edges']} D1-D3 AMD-PASS edges, "
        f"{source['candidate_counts'].get('depth3', 0)} Thursday cross-worker campaign edges, "
        f"{source['v3_partial_step_occurrences']} preserved v3 step occurrences, "
        f"and {source['v4_physically_valid_step_occurrences']} v4 step occurrences. "
        "Original and current B2 Evidence agree.",
    ])
    ceiling = audits["ceiling"]
    if ceiling["status"] == "complete":
        rows.append("")
        rows.append(
            "[Verified] Ceiling mean share of best known at budgets 4/6/8/10, "
            "then worker/policy gap at budget 6 (objective units):"
        )
        for brief_id in BRIEFS:
            item = ceiling["per_brief"][brief_id]
            by_budget = item["mean_ceiling_share_of_best_pct_by_budget"]
            shares = "/".join(
                "-" if by_budget.get(str(b)) is None else f"{by_budget[str(b)]:.1f}%"
                for b in (4, 6, 8, 10)
            )
            worker_gap = item["mean_worker_limited_gap_budget_6"]
            policy_gap = item["mean_policy_limited_gap_budget_6"]
            rows.append(
                f"- {brief_id}: {shares}; gaps {_gap(worker_gap)}/{_gap(policy_gap)} "
                f"{item['unit']}."
            )
    else:
        rows.extend(["", "[Verified] Orchestrator ceiling audit pending."])
    rows.extend([
        "",
        "[Inferred] Partial policy cells are not a policy ranking. "
        "A cache hit may still require a fresh pilot-ledger AMD validation.",
    ])
    report = "\n".join(rows) + "\n"
    if len(report.split()) >= 500:
        raise ValueError("Thursday overnight report exceeds 500 words")
    return report


def generate() -> dict[str, Any]:
    """Publish only summary artifacts after a checked, training-only snapshot."""
    plan = _preflight_sources()
    starts, best, source = _baseline()
    episodes, extra, ledger = _v4_events(starts, best)
    source.update(extra)
    counter, reservation_counts = _counter(_hash(ROOT / "plan.json"))
    rows = _completed(starts, best, episodes, ledger, reservation_counts)
    table = _table(rows)
    audits = _audit_state(_hash(ROOT / "plan.json"))
    status_path = ROOT / "status.json"
    status = _read(status_path) if status_path.is_file() else {
        "phase": "prepared",
    }
    status = {
        **status,
        "completed_episodes": len(rows),
        "cap_used": len(counter["reservations"]),
        "new_v4_reservations": len(counter["reservations"]) - 62,
        "cache_hits": sum(row["cache_hits"] for row in rows),
        "recipe_runs_used": sum(row["runs_used"] for row in rows),
    }
    if status.get("phase") in ("complete", "capped") and len(rows) != 140 and status["phase"] == "complete":
        raise ValueError("complete v4 pilot lacks all 140 results")
    result = {
        "schema_version": "thursday-overnight-pilot-report-v1",
        "training_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_plan_sha256": _hash(ROOT / "plan.json"),
        "reporter_sha256": _hash(Path(__file__)),
        "pilot_status": status,
        "sources": source,
        "audits": audits,
        "best_known": {
            f"{design}:{brief}": row for (design, brief), row in best.items()
        },
        "policy_brief": table,
        "completed_episode_rows": len(rows),
        "v4_new_real_reservations_in_completed_rows": sum(
            row["new_real_runs_reserved"] for row in rows
        ),
    }
    report = render(table, status, source, audits)
    _atomic_bytes(SUMMARY, _json_bytes(result))
    _atomic_bytes(REPORT, report.encode("utf-8"))
    return {
        "completed_episodes": len(rows),
        "cap_used": status["cap_used"],
        "new_v4_reservations": status["new_v4_reservations"],
        "report_words": len(report.split()),
        "summary": str(SUMMARY), "report": str(REPORT),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

