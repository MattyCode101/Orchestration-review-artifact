"""Pre-declared selection of repair-loop v2 from the 16-variant sweep (phase 1).

Rules (decision of 2026-11-06), applied exactly and without adding variants:
  * leave one design out: choose the variant with the best mean share over A0
    and T0 on the four other designs, then record its result on the held-out
    design; report the mean held-out result for v1 and for the procedure;
  * select v2 on all five designs by the same rule; ties go to the fewest knob
    changes from v1 (a tie is equality within 1e-9);
  * phase 2 then runs v2 on T and A.
A variant needs every one of its cells (designs x {A0, T0}) to be ranked, so a
capped or failed cell removes that variant from the ranking and is reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts import friday_analysis as fa
from scripts import friday_sweep as sweep

TOL = 1e-9
BRIEFS = sweep.PHASE1_BRIEFS


def knob_changes(grid_id: str) -> int:
    base = sweep.GRID[0]
    return sum(a != b for a, b in zip(sweep.GRID[sweep.GRID_IDS.index(grid_id)], base))


def choose(scores: dict[str, float | None]) -> tuple[str | None, list[str]]:
    valid = {g: v for g, v in scores.items() if v is not None}
    if not valid:
        return None, []
    top = max(valid.values())
    tied = sorted((g for g, v in valid.items() if v >= top - TOL),
                  key=lambda g: (knob_changes(g), g))
    return tied[0], tied


def cell_table(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    starts, best = fa.start_metrics(), fa.best_known()
    table = {}
    for row in rows:
        if row.get("phase") != 1:
            continue
        metric = fa.OBJECTIVE[row["brief"]][0]
        value = row["final_metrics"][metric]
        table[(row["grid_id"], row["design"], row["brief"])] = {
            "share": fa.share(row["brief"], row["design"], value, starts, best),
            "value": value, "final_state_id": row["final_state_id"],
            "runs_used": row["runs_used"], "cache_hits": row["cache_hits"],
            "new_real_runs": row["new_real_runs_reserved"],
            "stop_reason": row["stop_reason"],
        }
    return table


def mean_share(table, grid_id: str, designs) -> float | None:
    values = []
    for design in designs:
        for brief in BRIEFS:
            cell = table.get((grid_id, design, brief))
            if cell is None or cell["share"] is None:
                return None
            values.append(cell["share"])
    return sum(values) / len(values)


def select(table) -> dict:
    designs = fa.DESIGNS
    lodo = []
    for held in designs:
        train = [d for d in designs if d != held]
        scores = {g: mean_share(table, g, train) for g in sweep.GRID_IDS}
        pick, tied = choose(scores)
        held_result = mean_share(table, pick, [held]) if pick else None
        v1_result = mean_share(table, "g01", [held])
        lodo.append({
            "held_out": held, "chosen": pick, "tied": tied,
            "train_mean": scores.get(pick) if pick else None,
            "held_out_result": held_result, "v1_held_out": v1_result,
        })
    both = [row for row in lodo if row["held_out_result"] is not None and row["v1_held_out"] is not None]
    summary = {
        "folds_complete": len(both),
        "v1_mean_held_out": (sum(r["v1_held_out"] for r in both) / len(both)) if both else None,
        "procedure_mean_held_out": (sum(r["held_out_result"] for r in both) / len(both)) if both else None,
    }
    scores_all = {g: mean_share(table, g, designs) for g in sweep.GRID_IDS}
    v2, tied = choose(scores_all)
    return {
        "lodo": lodo, "lodo_summary": summary,
        "all_five_scores": scores_all,
        "v2": None if v2 is None else {
            "grid_id": v2, "knobs": list(sweep.GRID[sweep.GRID_IDS.index(v2)]),
            "mean_share_all_five": scores_all[v2], "tied": tied,
            "knob_changes_from_v1": knob_changes(v2),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="ignore the phase-1 gate and write only under --out")
    parser.add_argument("--out", type=Path, default=sweep.ROOT)
    args = parser.parse_args()
    if not args.dry_run and not (sweep.ROOT / "sweep_phase1_summary.json").is_file():
        print("phase 1 has not finished", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    rows = fa.result_rows(sweep.ROOT)
    table = cell_table(rows)
    expected = {(g, d, b) for g in sweep.GRID_IDS for d in fa.DESIGNS for b in BRIEFS}
    result = select(table)
    result["coverage"] = {
        "expected_cells": len(expected), "present_cells": len(expected & set(table)),
        "missing_cells": sorted("|".join(c) for c in expected - set(table)),
    }
    result["cells"] = {"|".join(k): v for k, v in sorted(table.items())}
    (args.out / "selection.json").write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    if result["v2"] is None:
        print("no variant has a complete set of cells", file=sys.stderr)
        return 3
    if not args.dry_run:
        grid_id = result["v2"]["grid_id"]
        work = {"selected_grid_id": grid_id,
                "rows": [[d, b, grid_id] for d in fa.DESIGNS for b in sweep.PHASE2_BRIEFS]}
        (args.out / "phase2_work.json").write_text(json.dumps(work, indent=1), encoding="utf-8")
    print(json.dumps({"v2": result["v2"], "lodo_summary": result["lodo_summary"],
                      "coverage": {k: v for k, v in result["coverage"].items() if k != "missing_cells"}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
