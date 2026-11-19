"""Verify the exact ceilings and write notes/orchestrator_ceiling_training.md.

The verifier asserts, for every cell: the ceiling never falls with a larger
budget; it is at least every simulated baseline policy at the same budget; at
budget 6 it is at least every actual pilot final of that design and brief (the
failure the beam showed: fixed_BA reached 65.9% on A0 against a 57.1% ceiling);
and all cells used one cache snapshot. Case-insensitive label checks only.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts import friday_analysis as fa

PROJECT = fa.PROJECT
WORK = PROJECT / "data/scratch_friday_2026-11-06/ceiling_exact"
REPORT = PROJECT / "notes/orchestrator_ceiling_training.md"
PREVIOUS = PROJECT / "notes/orchestrator_ceiling_training_beam128_2026-11-06_prev.md"
BUDGETS = ("4", "6", "8", "10")
# Mean shares of the previous beam lower bound, copied from notes/thursday_overnight_dev.md.
OLD_BEAM = {
    "T0": (63.5, 63.5, 63.5, 66.9), "A0": (56.2, 57.1, 57.1, 57.1),
    "T": (65.4, 65.4, 65.4, 68.7), "A": (74.4, 75.4, 75.4, 75.4),
}
TOL = 1e-9


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def main(briefs: tuple[str, ...] = ("T0", "A0", "T", "A")) -> int:
    starts, pool = fa.start_metrics(), fa.best_known()
    cells, problems, notes = {}, [], []
    for brief in briefs:
        for design in fa.DESIGNS:
            path = WORK / f"cell_{design}_{brief}.json"
            if (WORK / f"cell_{design}_{brief}.FAILED").is_file() or not path.is_file():
                problems.append(f"missing cell {design}/{brief}")
                continue
            cells[(design, brief)] = fa.read(path)
    snapshots = {c["snapshot_keys_sha256"] for c in cells.values()}
    if len(snapshots) > 1:
        problems.append(f"cells used {len(snapshots)} different cache snapshots")
    pilot = {}
    for row in fa.result_rows(fa.V4):
        pilot.setdefault((row["design"], row["brief"]), []).append(row)
    checks = 0
    for (design, brief), cell in sorted(cells.items()):
        imps = [cell["budgets"][b]["improvement"] for b in BUDGETS]
        checks += 1
        if any(b < a - TOL for a, b in zip(imps, imps[1:])):
            problems.append(f"{design}/{brief}: ceiling falls with a larger budget {imps}")
        for b in BUDGETS:
            for name, pol in cell["budgets"][b]["policies"].items():
                checks += 1
                metric, direction = fa.OBJECTIVE[brief]
                gain = (pol["value"] - starts[design][metric]) * (1 if direction == "max" else -1)
                if gain > cell["budgets"][b]["improvement"] + TOL:
                    problems.append(f"{design}/{brief}/B{b}: simulated {name} beats the ceiling")
        if brief in fa.BRIEFS:
            for row in pilot.get((design, brief), []):
                checks += 1
                metric = fa.OBJECTIVE[brief][0]
                gain = fa.improvement(brief, design, row["final_metrics"][metric], starts)
                if gain > cell["budgets"]["6"]["improvement"] + TOL:
                    problems.append(
                        f"{design}/{brief}: pilot {row['policy']} final ({gain:.4g}) exceeds the B6 ceiling "
                        f"({cell['budgets']['6']['improvement']:.4g})")
        if not cell["search"]["complete"]:
            notes.append(f"{design}/{brief}: search stopped on its {cell['search'].get('stop_reason', 'limit')}, "
                         "so this ceiling is only a lower bound (nothing generated was discarded)")
    if REPORT.is_file() and not PREVIOUS.is_file():
        shutil.copy2(REPORT, PREVIOUS)
    summary = fa.read(fa.V4 / "policy_brief_summary.json")["policy_brief"]
    best_policy = {}
    for row in summary:
        v = row.get("mean_share_of_best")
        if v is not None and (row["brief"] not in best_policy or v > best_policy[row["brief"]][1]):
            best_policy[row["brief"]] = (row["policy"], v)
    L = ["# Orchestrator ceiling, exact (training designs)", "",
         f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. Replaces the beam-128 lower bound "
         f"(preserved as `{PREVIOUS.name}`).", "", "## Summary", ""]
    L.append(f"* **Verifier: {'PASSED' if not problems else 'FAILED'}** ({checks} checks, "
             f"{len(problems)} problem(s)). " + ("Every ceiling is at least every simulated baseline and every actual pilot final."
                                                 if not problems else "See section 4."))
    for brief in briefs:
        if brief == "P0":
            continue
        row = []
        for b in BUDGETS:
            vals = [cells[(d, brief)]["budgets"][b]["share_of_pool_best"] for d in fa.DESIGNS if (d, brief) in cells]
            vals = [v for v in vals if v is not None]
            row.append(sum(vals) / len(vals) if vals else None)
        old = OLD_BEAM.get(brief)
        bp = best_policy.get(brief)
        L.append(f"* {brief}: exact ceiling share at budgets 4/6/8/10 = " + " / ".join(pct(v) for v in row)
                 + (f"; previous beam {' / '.join(f'{x:.1f}%' for x in old)}" if old else "")
                 + (f"; best pilot policy {bp[0]} {pct(bp[1])}" if bp else "") + ".")
    L += ["", "Shares are mean over the five designs of (ceiling gain) / (pool-best gain), with the pilot's fixed "
          "best-known denominators; above 100% means the cache holds a state better than that pool.", ""]
    L += ["## 1. Ceiling by design at budget 6, with pilot policies", "",
          "| brief | design | ceiling gain | ceiling share | best pilot final share | repair_loop share | what the ceiling does |",
          "|---|---|---|---|---|---|---|"]
    for brief in briefs:
        if brief == "P0":
            continue
        metric = fa.OBJECTIVE[brief][0]
        for design in fa.DESIGNS:
            cell = cells.get((design, brief))
            if cell is None:
                continue
            c6 = cell["budgets"]["6"]
            shares = {r["policy"]: fa.share(brief, design, r["final_metrics"][metric], starts, pool)
                      for r in pilot.get((design, brief), [])}
            bestp = max((v for v in shares.values() if v is not None), default=None)
            L.append(f"| {brief} | {design} | {c6['improvement']:+.4g} | {pct(c6['share_of_pool_best'])} | "
                     f"{pct(bestp)} | {pct(shares.get('repair_loop'))} | {' -> '.join(c6['path'])[:150]} |")
    L += ["", "## 2. Ceiling by budget and design (share of pool best)", "",
          "| brief | design | B4 | B6 | B8 | B10 |", "|---|---|---|---|---|---|"]
    for (design, brief), cell in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        L.append(f"| {brief} | {design} | " + " | ".join(
            pct(cell['budgets'][b]['share_of_pool_best']) for b in BUDGETS) + " |")
    L += ["", "## 3. Coverage and cost of the search", "",
          "| brief | design | nodes explored | mandates proposed | pruned (uncached) | pruned share | no-recipe | merged identical | exact? | peak GB | seconds |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    tot = {"explored": 0, "proposed": 0, "cache_pruned": 0}
    misses = {}
    for (design, brief), cell in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        s = cell["search"]
        for k in tot:
            tot[k] += s[k]
        for k, v in cell["oracle_miss_reasons"].items():
            misses[k] = misses.get(k, 0) + v
        share = s["cache_pruned"] / s["proposed"] if s["proposed"] else 0
        L.append(f"| {brief} | {design} | {s['explored']} | {s['proposed']} | {s['cache_pruned']} | {share * 100:.1f}% | "
                 f"{s['no_recipe']} | {s['merged_identical']} | {'exact' if s['complete'] else 'LOWER BOUND (' + s.get('stop_reason', 'limit') + ')'} | "
                 f"{s.get('peak_rss_gb', '?')} | {s['elapsed_s']} |")
    snap = next(iter(cells.values()))["snapshot_pointers"] if cells else "?"
    L += ["", f"Totals: {tot['explored']} nodes explored, {tot['proposed']} mandates proposed, "
          f"{tot['cache_pruned']} pruned because a planned recipe had no checked cached outcome "
          f"({(tot['cache_pruned'] / tot['proposed'] * 100) if tot['proposed'] else 0:.1f}%). "
          f"Cache snapshot: {snap} terminal pointers (hash `{sorted(snapshots)[0][:16] if snapshots else '?'}`...). "
          f"Oracle miss reasons: {misses}.", ""]
    L += ["## 4. Verifier", ""]
    L += [f"* PROBLEM: {p}" for p in problems] or ["* No problems found."]
    L += [f"* NOTE: {n}" for n in notes]
    L += ["", "## 5. Method", "",
          "* Mandate space: worker (timing, area; plus power for P0) x any visited physically valid base x "
          "preserve placement yes/no x 1..3 attempts, no guards. Each mandate runs the real worker planner "
          "(FINAL fitted models with saved telemetry), the real selector and the pure gate over cached outcomes.",
          "* Search: breadth-first by charged cost to budget 10, merging only nodes with identical accessible "
          "state; no beam, no width limit, no pruning except uncached mandates. One exploration yields all four budgets.",
          "* Oracle: terminal pointers of the approved v4 training cache frozen at launch, with status, hash, "
          "current-meter, clean-property and exact-DCP AMD-PASS checks. This includes outcomes created by the pilot.",
          "* Training designs only; no tool job launched."]
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {REPORT}; problems={len(problems)}")
    return 0 if not problems else 4


if __name__ == "__main__":
    raise SystemExit(main())
