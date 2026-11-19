"""Learned Bandit W2R (post-freeze, 17 Nov 2026): collect the five TEST results into what Table V reports. Reads only; writes report/bandit_w2r_cells.json and .csv in the run folder.

Per design: the final pick's gain (ps; WNS final minus start), brief-validity re-derived from the stored metrics (packed LUTs <= start, dynamic power <= start + band),
the AMD validation status of the final pick, runs used (charged in full), new tool runs, charged wall time (reused runs at their recorded time), the calls (worker, placement
scope, attempts, arm) and what the first call did. Headroom designs for W2R: the stored flag of data/paper_tables_2026-11-15/cells.json (3 of 5).
Cost combined with the stored 10 L2R and P0 tasks of the cost csv (tool runs and job-hours as that file computes them: mean runs_used, mean charged wall hours).
usage (project root, venv): python scripts/bandit_w2r_2026-11-17/bw_collect.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts/bandit_2026-11-11"))
import bd_extract as be  # noqa: E402

RUN = PROJECT / "data/test_r2_2026-11-13/bandit_w2r_2026-11-17"
OUT = RUN / "report"
CELLS = PROJECT / "data/paper_tables_2026-11-15/cells.json"
COST = PROJECT / "paper_figures/inputs/Paper_Tables_Table2_cost_per_task.csv"
ORDER = ("rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2", "fir_symmetric_systolic", "fir_systolic_transposed")
EPS = 1e-9


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def main() -> int:
    cells = json.loads(CELLS.read_text(encoding="utf-8"))["cells"]
    rows = []
    for d in ORDER:
        eid = f"tr_{d}_W2R_bandit_pf1"
        path = RUN / "results" / f"{eid}.json"
        if not path.is_file():
            print("missing result:", eid)
            if os.environ.get("BW_PARTIAL") == "1":
                continue                                      # a preview of the finished cells only: nothing is written below
            return 1
        r = json.loads(path.read_text(encoding="utf-8"))
        s, f, noise = r["start_metrics"], r["final_metrics"], r["noise"]
        band = float(noise["dynamic_power_w"])
        gain_ns = f["wns_ns"] - s["wns_ns"]
        limits = {"packed_luts_le_start": f["packed_luts"] <= s["packed_luts"] + EPS,
                  "power_le_start_plus_band": f["dynamic_power_w"] <= s["dynamic_power_w"] + band + EPS}
        rec = be.episode_record(eid, be.read_events(RUN / "test/engine/episodes" / eid / "journal.jsonl"))
        events = be.read_events(RUN / "test/engine/episodes" / eid / "journal.jsonl")
        arms = [json.loads(e["mandate"]["context_ref"])["arm"] for e in events if e.get("type") == "mandate"]
        calls = []
        for c, arm in zip(rec["calls"], arms):
            sel = None
            for st in c["steps"]:
                if st["out_id"] == c["selected_out_id"] and st["metrics"]:
                    sel = st["metrics"]
            calls.append({"i": c["i"], "arm": arm, "worker": c["worker"], "preserve": c["preserve"], "attempts": c["max_runs"], "recipe_runs": c["recipe_runs"],
                          "selected": None if sel is None else {"wns_ps": round((sel["wns_ns"] - s["wns_ns"]) * 1000), "luts": sel["packed_luts"] - s["packed_luts"],
                                                                "power_mw": round((sel["dynamic_power_w"] - s["dynamic_power_w"]) * 1000)}})
        rows.append({
            "design": d, "episode_id": eid, "head": bool(cells[f"{d}|W2R"]["head"]), "band_ns": cells[f"{d}|W2R"]["band"], "ref_ns": cells[f"{d}|W2R"]["ref"],
            "start_wns_ns": s["wns_ns"], "final_wns_ns": f["wns_ns"], "gain_ps": gain_ns * 1000.0,
            "start_luts": s["packed_luts"], "final_luts": f["packed_luts"], "start_power_w": s["dynamic_power_w"], "final_power_w": f["dynamic_power_w"],
            "power_band_w": band, "final_brief_valid_journal": bool(r["final_brief_valid"]), "limits_rederived": limits, "amd_validation": r["validation_status"],
            "final_state": r["final_state_id"], "stop_reason": r["stop_reason"], "runs_used": r["runs_used"], "new_real_runs": r["new_real_runs"],
            "cache_hits": r.get("cache_hits"), "charged_wall_h": r["charged_wall_s"] / 3600.0, "charged_cpu_h": r["charged_cpu_s"] / 3600.0,
            "calls": calls, "workers": r["loop"]["workers"], "loop": r["loop"], "resumed": r["resumed"], "status": r["status"],
            "policy_version": r["policy_version"],
        })
    if os.environ.get("BW_PARTIAL") == "1":
        caps = json.loads((RUN / "caps.json").read_text(encoding="utf-8"))["used"]
        pending = [f"tr_{d}_W2R_bandit_pf1" for d in ORDER if not (RUN / "results" / f"tr_{d}_W2R_bandit_pf1.json").is_file()]
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "bandit_w2r_cells_partial.json").write_text(json.dumps({"summary": {
            "partial": True, "pending": pending, "caps_new_runs_used": caps["new_real_runs"], "caps_job_h_used": caps["job_seconds"] / 3600.0,
            "finished": len(rows)}, "cells": rows}, indent=1, sort_keys=True), encoding="utf-8")
        for x in rows:
            print(x["design"], "head" if x["head"] else "    ", "start", x["start_wns_ns"], "final", x["final_wns_ns"], "gain_ps", round(x["gain_ps"], 1), "valid", x["final_brief_valid_journal"],
                  x["limits_rederived"], "amd", x["amd_validation"], "runs", x["runs_used"], "new", x["new_real_runs"], "wall_h", round(x["charged_wall_h"], 3), "stop", x["stop_reason"])
            for c in x["calls"]:
                print("    call", c["i"], c["arm"], "runs", c["recipe_runs"], "->", c["selected"])
        return 0
    head = [x for x in rows if x["head"]]
    old = {}
    for line in csv.DictReader(COST.open(encoding="utf-8")):
        old[line["key"]] = line
    ob = old["bandit"]
    n_old = int(ob["tasks"])
    runs15 = (n_old * float(ob["tool_runs_per_task"]) + sum(x["runs_used"] for x in rows)) / (n_old + 5)
    jobh15 = (n_old * float(ob["job_hours_per_task"]) + sum(x["charged_wall_h"] for x in rows)) / (n_old + 5)
    new15 = (n_old * float(ob["new_runs_per_task"]) + sum(x["new_real_runs"] for x in rows)) / (n_old + 5)
    summary = {
        "gain_ps_mean_head_designs": mean(x["gain_ps"] for x in head), "head_designs": [x["design"] for x in head],
        "gain_ps_mean_all5": mean(x["gain_ps"] for x in rows),
        "w2r_tasks": {"tool_runs_per_task": mean(x["runs_used"] for x in rows), "job_h_per_task": mean(x["charged_wall_h"] for x in rows),
                      "new_runs_per_task": mean(x["new_real_runs"] for x in rows)},
        "new_runs_total": sum(x["new_real_runs"] for x in rows), "job_h_charged_total": sum(x["charged_wall_h"] for x in rows),
        "cost_15_tasks": {"tasks": n_old + 5, "tool_runs_per_task": runs15, "job_hours_per_task": jobh15, "new_runs_per_task": new15},
        "cost_old_10_tasks": {k: ob[k] for k in ("tasks", "tool_runs_per_task", "job_hours_per_task", "new_runs_per_task")},
        "caps_new_runs_used": json.loads((RUN / "caps.json").read_text(encoding="utf-8"))["used"]["new_real_runs"],
        "caps_job_h_used": json.loads((RUN / "caps.json").read_text(encoding="utf-8"))["used"]["job_seconds"] / 3600.0,
        "all_valid": all(x["final_brief_valid_journal"] and all(x["limits_rederived"].values()) for x in rows),
        "all_amd_pass": all(x["amd_validation"] == "pass" for x in rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bandit_w2r_cells.json").write_text(json.dumps({"summary": summary, "cells": rows}, indent=1, sort_keys=True), encoding="utf-8")
    with (OUT / "bandit_w2r_cells.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "design", "brief", "headroom", "start_wns_ns", "final_wns_ns", "gain_ps", "final_brief_valid", "amd_validation", "runs_used", "new_real_runs",
                    "charged_wall_h", "calls", "workers", "stop_reason", "scope"])
        for x in rows:
            w.writerow(["TEST", x["design"], "W2R", int(x["head"]), x["start_wns_ns"], x["final_wns_ns"], round(x["gain_ps"], 3), int(x["final_brief_valid_journal"]), x["amd_validation"],
                        x["runs_used"], x["new_real_runs"], round(x["charged_wall_h"], 4), x["loop"]["calls"], "+".join(x["workers"]), x["stop_reason"],
                        "post-freeze, prior fitted 17 Nov from TRAINING and VALIDATION only"])
    print(json.dumps(summary, indent=1, sort_keys=True))
    for x in rows:
        print(x["design"], "head" if x["head"] else "    ", "gain_ps", round(x["gain_ps"], 1), "valid", x["final_brief_valid_journal"], x["limits_rederived"], "amd", x["amd_validation"],
              "runs", x["runs_used"], "new", x["new_real_runs"], "wall_h", round(x["charged_wall_h"], 3), "stop", x["stop_reason"])
        for c in x["calls"]:
            print("    call", c["i"], c["arm"], "runs", c["recipe_runs"], "->", c["selected"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
