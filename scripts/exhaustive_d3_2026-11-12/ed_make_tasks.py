"""Build tasks_phase0.json (AMD checks against B2) and tasks_smoke.json from the plan files. Read-only apart from plan/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ed_common as ec  # noqa: E402


def main() -> int:
    summary = ec.jread(ec.PLAN / "summary.json")
    order = summary["design_order_cheapest_first"]
    p1 = ec.jread(ec.PLAN / "tasks_phase1.json")
    cells = ec.jread(ec.PLAN / "phase0_cells.json")
    tasks0 = []
    for c in cells["cells"]:
        L = c["better_candidates_baseline_only"]
        if not L:
            continue
        cands = []
        for x in L:
            info = cells["states"][x["state"]]
            cands.append({"state": x["state"], "path": x["path"], "src": info["src"], "parent_id": info["parent_id"],
                          "made_by_run_id": info["made_by_run_id"]})
        tasks0.append({"phase": 0, "kind": "amd_cell", "design": c["design"], "brief": c["brief"], "no_room_in_frame": c["no_room_in_frame"],
                       "candidates": cands, "est_wall_s": 900.0})
    ec.jwrite(ec.PLAN / "tasks_phase0.json", tasks0, indent=None)
    print("phase 0 cells:", len(tasks0), "first candidates:", sorted({t["candidates"][0]["state"][:10] for t in tasks0}))

    cheapest = order[0]
    new = [t for t in p1 if t["design"] == cheapest and t["recipe"] == "A_E_DUPL_Q1000"]
    new.sort(key=lambda t: t["state"])
    smoke = [dict(t) for t in new[:2]]
    reuse = ec.jread(ec.PLAN / "reuse_phase1.json")
    stored = [r for r in reuse if r["design"] == cheapest and r["exact"]]
    stored.sort(key=lambda r: (r["recipe"] != "A_E_DUPL_Q1000", r["wall_s"], r["run"]))
    r = stored[0]
    src, meta = ec.state_source(r["state"], ec.SRC_ROOTS)
    smoke.append({"phase": 9, "kind": "run", "design": cheapest, "state": r["state"], "recipe": r["recipe"], "src": src,
                  "parent_id": (meta or {}).get("parent_id"), "made_by_run_id": (meta or {}).get("made_by_run_id"),
                  "est_wall_s": r["wall_s"], "replicate_of": {"root": r["root"], "run": r["run"], "orig": r["orig"], "output": r["output"]}})
    ec.jwrite(ec.PLAN / "tasks_smoke.json", smoke, indent=1)
    print("smoke tasks:", [(t["design"], t["recipe"], t["state"][:10], t["phase"]) for t in smoke])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
