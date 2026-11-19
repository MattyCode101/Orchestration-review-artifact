"""Audit a finished (or stopped) campaign run: journals, caps, token and cost accounting, AMD validation, pins.

Read-only. Prints a JSON verdict; every check is a count, and a failed check lists its cases.
Usage: campaign_audit.py NAME      (the work folder is data/campaign_NAME)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from orch.engine.journal import Journal  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(name: str) -> dict:
    work = PROJECT / f"data/campaign_{name}"
    plan = read(work / "plan.json")
    rows = [read(p) for p in sorted((work / "results").glob("*.json"))]
    journal = Journal(work / "engine")
    problems: dict[str, list] = {k: [] for k in (
        "no_final_event", "validation_not_pass", "charged_wall_mismatch", "step_output_without_amd_census",
        "llm_record_missing_or_hash_mismatch", "row_not_in_plan", "extraction_count_mismatch",
        "job_seconds_not_runs_plus_extraction", "dropped_cell_has_final_event", "llm_row_without_identity",
        "state_needing_extraction_without_features")}
    planned = {c["episode_id"] for c in plan["cells"]}
    census_root = work / "engine/verification/output_census"
    for r in rows:
        events = journal.read(r["episode_id"])
        base = r["episode_id"].rsplit("_a", 1)[0] if r["episode_id"] not in planned else r["episode_id"]
        if base not in planned:
            problems["row_not_in_plan"].append(r["episode_id"])
        final = [e for e in events if e.get("type") == "final"]
        validation = [e for e in events if e.get("type") == "validation"]
        if not final:
            problems["no_final_event"].append(r["episode_id"])
            continue
        if not validation or validation[-1]["record"].get("result") != "pass" and validation[-1]["record"].get("status") != "pass":
            if r["validation_status"] != "pass":
                problems["validation_not_pass"].append(r["episode_id"])
        charged = final[-1]["result"].get("charged_wall_s")
        if charged is None or abs(charged - r["charged_wall_s"]) > 1e-6:
            problems["charged_wall_mismatch"].append([r["episode_id"], charged, r["charged_wall_s"]])
        for e in events:
            if e.get("type") == "step" and e["verdict"].get("physically_valid"):
                out = e["step"].get("output_state_id")
                if out and not (census_root / r["design"] / f"{out}.json").is_file():
                    problems["step_output_without_amd_census"].append([r["episode_id"], out[:12]])
    caps = read(work / "caps.json")
    reservations = caps["reservations"]
    engine_runs = work / "engine/runs"
    complete = [x for x in reservations if (engine_runs / x["run_id"] / "complete.json").is_file()]
    interrupted = [x for x in reservations if x not in complete]
    calls = [json.loads(line) for line in (work / "llm_calls.jsonl").read_text().splitlines()] if (work / "llm_calls.jsonl").is_file() else []
    charged = [c for c in calls if c["charged"]]
    for c in calls:
        rec = Path(c["record"])
        if not rec.is_file():
            problems["llm_record_missing_or_hash_mismatch"].append([c["key"][:12], "no record"])
            continue
        d = read(rec)
        if sha(d["prompt"]) != c["prompt_sha256"] or sha(d["reply"]["text"]) != c["reply_sha256"]:
            problems["llm_record_missing_or_hash_mismatch"].append([c["key"][:12], "hash"])
    runs = {}
    for p in sorted((work / "runs").glob("[0-9]*")):
        start = read(p / "pins_start.json")["files"] if (p / "pins_start.json").is_file() else None
        end = read(p / "pins_end.json") if (p / "pins_end.json").is_file() else None
        runs[p.name] = {
            "control": read(p / "control.json")["state"] if (p / "control.json").is_file() else None,
            "void": (p / "VOID.json").is_file(), "pins_start_files": len(start) if start else None,
            "pins_end_recorded": end is not None, "pins_changed_during_run": end["changed"] if end else "no end record (interrupted)",
        }
    names = sorted(runs)
    same_across = None
    if len(names) > 1 and all((work / "runs" / n / "pins_start.json").is_file() for n in names):
        a = read(work / "runs" / names[0] / "pins_start.json")["files"]
        b = read(work / "runs" / names[-1] / "pins_start.json")["files"]
        same_across = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
    touched = {}
    for n in names:
        t = work / "runs" / n / "touched.jsonl"
        if t.is_file():
            for line in t.read_text().splitlines():
                touched.setdefault(json.loads(line)["episode_id"], set()).add(n)
    resumed = sorted(e for e, ns in touched.items() if len(ns) > 1)
    resumed_done = [e for e in resumed if (work / "results" / f"{e}.json").is_file()]
    logs = {k: sum(1 for _ in open(work / f"{k}.jsonl")) if (work / f"{k}.jsonl").is_file() else 0
            for k in ("capped_cells", "error_cells", "invalid_output_cells", "llm_unavailable")}
    # worker B live extraction: jobs reserved = extracted + failed; every accounted state has its features and cost record
    accounted = read(work / "extraction_accounted.json") if (work / "extraction_accounted.json").is_file() else {}
    budget = read(work / "extraction_budget.json") if (work / "extraction_budget.json").is_file() else {"reservations": []}
    events = [json.loads(x) for x in (work / "extraction.jsonl").read_text().splitlines()] if (work / "extraction.jsonl").is_file() else []
    extracted = [e for e in events if e["event"] == "extracted"]
    failed = [e for e in events if e["event"] == "refused" and "extraction failed" in e.get("reason", "")]
    extraction = {"jobs_reserved": len(budget["reservations"]), "cap": budget.get("cap"), "extracted": len(extracted),
                  "failed_extractions": len(failed), "accounted_states": len(accounted),
                  "wall_s_accounted": sum(v["wall_s_total"] for v in accounted.values()),
                  "refusals_other": [e for e in events if e["event"] == "refused" and "extraction failed" not in e.get("reason", "")][:5]}
    if plan.get("telemetry") == "live":
        # a reservation without a result is an extraction killed by an operator stop (redone after the resume)
        ext_interrupted = extraction["jobs_reserved"] - extraction["extracted"] - extraction["failed_extractions"]
        extraction["reserved_but_ext_interrupted"] = ext_interrupted
        stopped_runs = sum(1 for p_ in (work / "runs").glob("*/control.json") if read(p_)["state"] == "stopped_by_operator")
        if ext_interrupted < 0 or (ext_interrupted > 0 and stopped_runs == 0)                 or extraction["accounted_states"] != extraction["extracted"]:
            problems["extraction_count_mismatch"].append(extraction)
        run_wall = sum((read(engine_runs / x["run_id"] / "runinfo.json").get("wall_s") or 0.0) for x in complete)
        if abs(caps["used"]["job_seconds"] - (run_wall + extraction["wall_s_accounted"])) > 1.0 + 0.001 * caps["used"]["job_seconds"]:
            problems["job_seconds_not_runs_plus_extraction"].append(
                [caps["used"]["job_seconds"], run_wall, extraction["wall_s_accounted"]])
        for state in accounted:
            if not (work / "telemetry_cache" / state[:2] / state / "features.json").is_file():
                problems["state_needing_extraction_without_features"].append(state[:12])
    # infrastructure failures: a cell that failed twice has no result row and no final event
    infra = {}
    if (work / "infra_failures.jsonl").is_file():
        for line in (work / "infra_failures.jsonl").read_text().splitlines():
            e = json.loads(line)
            infra.setdefault(e["episode_id"], []).append(e)
    have_rows = {r["episode_id"] for r in rows}
    for episode, entries in infra.items():
        if len(entries) >= 2 and episode not in have_rows and any(e.get("type") == "final" for e in journal.read(episode)):
            problems["dropped_cell_has_final_event"].append(episode)
    for r in rows:
        if r["approach"] in ("F1-R1", "F1-R0", "E0-R1", "GA-R1") and not (isinstance(r.get("llm"), dict) and r["llm"].get("template_sha256")):
            problems["llm_row_without_identity"].append(r["episode_id"])
    return {
        "extraction": extraction,
        "infrastructure_failures": {"episodes": len(infra), "failed_once": sum(len(v) == 1 for v in infra.values()),
                                    "dropped": sum(len(v) >= 2 and k not in have_rows for k, v in infra.items()),
                                    "completed_after_requeue": sum(k in have_rows for k in infra)},
        "name": name, "scope": plan["scope"], "planned_cells": len(plan["cells"]), "result_rows": len(rows),
        "caps": plan["caps"], "used": caps["used"], "stopped": caps["stopped"],
        "reservations": len(reservations), "reservations_with_complete_run": len(complete),
        "reservations_interrupted_before_completion": len(interrupted),
        "rows_new_real_runs_sum": sum(r["new_real_runs"] for r in rows),
        "tokens": {"ledger": caps["used"]["tokens"], "charged_calls_sum": sum(c["total_tokens"] for c in charged),
                   "match": caps["used"]["tokens"] == sum(c["total_tokens"] for c in charged),
                   "all_calls": len(calls), "charged_calls": len(charged)},
        "usd": {"ledger": caps["used"]["usd"], "charged_calls_sum": round(sum(c["usd"] for c in charged), 6),
                "match": abs(caps["used"]["usd"] - sum(c["usd"] for c in charged)) < 1e-4},
        "job_seconds": {"ledger": caps["used"]["job_seconds"], "includes": "recipe runs and worker B extraction (not AMD validation)",
                        "sum_of_completed_new_runs": sum(
                            (read(engine_runs / x["run_id"] / "runinfo.json").get("wall_s") or 0.0) for x in complete)},
        "runs": runs, "pins_that_differ_between_first_and_last_run": same_across,
        "episodes_touched_by_more_than_one_run": len(resumed), "of_which_finished": len(resumed_done),
        "cell_logs": logs, "problems": {k: v for k, v in problems.items() if v},
        "problem_counts": {k: len(v) for k, v in problems.items()},
    }


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), indent=1, sort_keys=True))
