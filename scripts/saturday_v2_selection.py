"""The authors' Saturday rule for the rule-based v2 (training designs only, read-only).

v2 = the best variant by the pre-declared leave-one-design-out criterion (mean
share of best known over A0 and T0, ties to the fewest knob changes) among the
variants within 5 points of v1 on BOTH control briefs T and A. If none other
qualifies, v2 = v1 (v1 always qualifies against itself).

Reading adopted and flagged: "within 5 points" is one-sided, that is, no more
than 5 points BELOW v1 on the five-design mean share of T and of A, because the
check exists to catch a variant that damages a control brief. The two-sided
reading (|difference| <= 5) is computed too and shown next to it.

The same filter is applied inside each leave-one-design-out fold, using only
that fold's four training designs, before the held-out design is scored.
Nothing here launches a tool job or touches a test design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts import friday_analysis as fa
from scripts import friday_sweep as sweep
from scripts import friday_sweep_select as old_select
from scripts import saturday_sweep_ta as ext

TOL = 1e-9
POINTS = 5.0
V1 = "g01"
CRITERION_BRIEFS = ("A0", "T0")
CONTROL_BRIEFS = ("T", "A")
NOTE = fa.PROJECT / "notes/repair_loop_v2_selection.md"
PREV = fa.PROJECT / "notes/repair_loop_v2_selection_g14_2026-11-06_prev.md"

Shares = dict[tuple[str, str, str], "float | None"]


def mean_share(shares: Shares, grid: str, designs, briefs) -> float | None:
    values = []
    for design in designs:
        for brief in briefs:
            value = shares.get((grid, design, brief))
            if value is None:
                return None
            values.append(value)
    return sum(values) / len(values)


def evaluate(shares: Shares, designs) -> dict[str, dict[str, float | None]]:
    return {
        grid: {
            "criterion": mean_share(shares, grid, designs, CRITERION_BRIEFS),
            "T": mean_share(shares, grid, designs, ("T",)),
            "A": mean_share(shares, grid, designs, ("A",)),
        }
        for grid in sweep.GRID_IDS
    }


def qualifies(ev: dict[str, dict[str, float | None]], grid: str, *, one_sided: bool = True) -> bool:
    mine, base = ev[grid], ev[V1]
    if mine["criterion"] is None:
        return False
    for brief in CONTROL_BRIEFS:
        if mine[brief] is None or base[brief] is None:
            return False
        diff = (mine[brief] - base[brief]) * 100.0
        ok = diff >= -POINTS - TOL if one_sided else abs(diff) <= POINTS + TOL
        if not ok:
            return False
    return True


def choose_v2(shares: Shares, designs, *, one_sided: bool = True):
    ev = evaluate(shares, designs)
    eligible = {g: ev[g]["criterion"] for g in sweep.GRID_IDS if qualifies(ev, g, one_sided=one_sided)}
    pick, tied = old_select.choose(eligible)
    return pick, tied, ev


def select(shares: Shares) -> dict[str, Any]:
    designs = fa.DESIGNS
    folds = []
    for held in designs:
        train = [d for d in designs if d != held]
        pick, tied, _ = choose_v2(shares, train)
        folds.append({
            "held_out": held, "chosen": pick, "tied": tied,
            "held_out_result": mean_share(shares, pick, [held], CRITERION_BRIEFS) if pick else None,
            "v1_held_out": mean_share(shares, V1, [held], CRITERION_BRIEFS),
        })
    both = [f for f in folds if f["held_out_result"] is not None and f["v1_held_out"] is not None]
    pick, tied, ev = choose_v2(shares, designs)
    pick2, tied2, _ = choose_v2(shares, designs, one_sided=False)

    def describe(grid, group):
        if grid is None:
            return None
        return {"grid_id": grid, "knobs": list(sweep.GRID[sweep.GRID_IDS.index(grid)]), "tied": group,
                "knob_changes_from_v1": old_select.knob_changes(grid), **ev[grid]}

    return {
        "lodo": folds,
        "lodo_summary": {
            "folds_complete": len(both),
            "v1_mean_held_out": sum(f["v1_held_out"] for f in both) / len(both) if both else None,
            "procedure_mean_held_out": sum(f["held_out_result"] for f in both) / len(both) if both else None,
        },
        "evaluation": ev,
        "qualifies_one_sided": {g: qualifies(ev, g) for g in sweep.GRID_IDS},
        "qualifies_two_sided": {g: qualifies(ev, g, one_sided=False) for g in sweep.GRID_IDS},
        "v2": describe(pick, tied),
        "v2_two_sided": describe(pick2, tied2),
    }


def gather_shares() -> tuple[Shares, dict[str, int], int]:
    """Merge the three result sources into one share table; refuse a conflicting duplicate."""

    starts, best = fa.start_metrics(), fa.best_known()
    shares: Shares = {}
    counts = {"friday_sweep": 0, "saturday_extension": 0, "pilot_v1": 0}

    def put(grid: str, row: dict[str, Any], origin: str) -> None:
        brief, design = row["brief"], row["design"]
        value = row["final_metrics"][fa.OBJECTIVE[brief][0]]
        share = fa.share(brief, design, value, starts, best)
        key = (grid, design, brief)
        if key in shares and shares[key] != share:
            raise ValueError(f"conflicting results for {key}")
        shares[key] = share
        counts[origin] += 1

    for row in fa.result_rows(ext.FRIDAY_ROOT):
        put(row["grid_id"], row, "friday_sweep")
    for row in fa.result_rows(ext.ROOT):
        put(row["grid_id"], row, "saturday_extension")
    for row in fa.result_rows(fa.V4):
        if row["policy"] == "repair_loop" and row["brief"] in CONTROL_BRIEFS:
            put(V1, row, "pilot_v1")
    counter = ext.ROOT / "new_real_runs.json"
    new_runs = len(fa.read(counter)["reservations"]) if counter.is_file() else 0
    return shares, counts, new_runs


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def pts(value: float | None, base: float | None) -> str:
    return "-" if value is None or base is None else f"{(value - base) * 100:+.1f}"


def render(result: dict[str, Any], counts: dict[str, int], new_runs: int, prev_sha: str) -> str:
    ev, v2 = result["evaluation"], result["v2"]
    ls = result["lodo_summary"]
    knobs = ("repair_attempts", "repair_preserve_placement", "gentle_step", "after_failed_repair")
    lines = [
        "# Repair-loop v2 selection, Saturday revision (training designs)", "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Rule (decision of Saturday): v2 = the best variant by the "
        "leave-one-design-out criterion (mean share over A0 and T0, ties to the fewest knob changes) among the variants "
        "within 5 points of v1 on both control briefs T and A; if none qualifies, v2 = v1. Verified against saved results; "
        "shares use the pilot's fixed best-known denominators.", "",
        "## Summary", "",
    ]
    if v2 is None:
        lines += ["No variant has a complete set of cells, so no v2 can be chosen.", ""]
    else:
        lines += [
            f"* **v2 = {v2['grid_id']}** {dict(zip(knobs, v2['knobs']))}; {v2['knob_changes_from_v1']} knob change(s) from v1; "
            f"mean share over A0 and T0 {pct(v2['criterion'])} (v1 {pct(ev[V1]['criterion'])}); "
            f"T {pct(v2['T'])} (v1 {pct(ev[V1]['T'])}); A {pct(v2['A'])} (v1 {pct(ev[V1]['A'])}).",
        ]
        two = result["v2_two_sided"]
        lines.append(
            "* Two-sided reading (|difference| <= 5 points): "
            + (f"v2 = {two['grid_id']}." if two else "none.")
            + (" Same choice." if two and two["grid_id"] == v2["grid_id"] else " DIFFERENT choice: flag for the authors.")
        )
        lines += [
            f"* The Friday choice g14 does not qualify: T {pts(ev['g14']['T'], ev[V1]['T'])}, "
            f"A {pts(ev['g14']['A'], ev[V1]['A'])} points against v1." if not result["qualifies_one_sided"]["g14"] else
            "* The Friday choice g14 qualifies.",
            f"* Leave-one-design-out with the same filter ({ls['folds_complete']} folds): mean held-out share "
            f"**v1 {pct(ls['v1_mean_held_out'])}** vs **procedure {pct(ls['procedure_mean_held_out'])}**.", "",
        ]
    lines += ["## 1. Every variant on the criterion and on the controls", "",
              "| id | attempts | preserve | gentle | after failed repair | A0+T0 | T | A | T vs v1 | A vs v1 | qualifies (one-sided) | qualifies (two-sided) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for grid, k in zip(sweep.GRID_IDS, sweep.GRID):
        e = ev[grid]
        lines.append(
            f"| {grid}{' (v1)' if grid == V1 else ''} | {k[0]} | {k[1]} | {'on' if k[2] else 'off'} | {k[3]} | "
            f"{pct(e['criterion'])} | {pct(e['T'])} | {pct(e['A'])} | {pts(e['T'], ev[V1]['T'])} | {pts(e['A'], ev[V1]['A'])} | "
            f"{'yes' if result['qualifies_one_sided'][grid] else 'no'} | {'yes' if result['qualifies_two_sided'][grid] else 'no'} |")
    lines += ["", "## 2. Leave one design out, filter applied on the four training designs", "",
              "| held-out design | chosen (tied set) | held-out result | v1 held-out |", "|---|---|---|---|"]
    for f in result["lodo"]:
        lines.append(f"| {f['held_out']} | {f['chosen']} ({', '.join(f['tied'])}) | {pct(f['held_out_result'])} | {pct(f['v1_held_out'])} |")
    lines += [
        "", "## 3. Cost and coverage", "",
        f"* Cells read: Friday sweep {counts['friday_sweep']}, Saturday extension {counts['saturday_extension']}, pilot v1 {counts['pilot_v1']}.",
        f"* New real recipe runs reserved by the Saturday extension: {new_runs} (cap {ext.CAP}).",
        "* Reading adopted (inferred, to confirm): \"within 5 points\" is one-sided: a variant may not sit more than 5 points below v1 on T or on A; a gain never disqualifies.",
        f"* The Friday note is kept below unchanged and as a byte-for-byte copy at `notes/{PREV.name}` (SHA-256 {prev_sha}).", "",
        "---", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ext.ROOT)
    parser.add_argument("--note", type=Path, default=NOTE)
    parser.add_argument("--no-note", action="store_true")
    args = parser.parse_args()
    shares, counts, new_runs = gather_shares()
    result = select(shares)
    result["coverage"] = {"cells": counts, "new_real_runs": new_runs}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "selection_saturday.json").write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    if result["v2"] is None:
        print("no variant has a complete set of cells", file=sys.stderr)
        return 3
    if not args.no_note:
        if not PREV.exists():
            shutil.copy2(args.note, PREV)
        prev_sha = hashlib.sha256(PREV.read_bytes()).hexdigest()
        args.note.write_text(render(result, counts, new_runs, prev_sha) + PREV.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"v2": result["v2"], "v2_two_sided": result["v2_two_sided"],
                      "lodo_summary": result["lodo_summary"], "coverage": result["coverage"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
