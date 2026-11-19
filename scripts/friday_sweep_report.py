"""Write notes/repair_loop_v2_selection.md from the sweep results (read-only)."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import json

from orch.engine.journal import Journal
from orch.engine.store import StateStore
from scripts import friday_analysis as fa
from scripts import friday_sweep as sweep

OUTPUT = fa.PROJECT / "notes/repair_loop_v2_selection.md"


def episode_lines(episode_id: str, metric: str) -> list[str]:
    """Compact step list of one episode, from its committed journal (read-only)."""

    store = StateStore(fa.V4)
    events = Journal(fa.V4).read(episode_id)
    mandates = {e["index"]: e["mandate"] for e in events if e["type"] == "mandate"}
    lines = []
    for index in sorted(mandates):
        m = mandates[index]
        ctx = json.loads(m["context_ref"])
        steps = []
        for e in (x for x in events if x["type"] == "step" and x["decision_index"] == index):
            outcome = store.load_outcome(e["step"]["run_id"])
            v = e["verdict"]
            tag = "OK" if v["brief_valid"] else ("VIOL" if v["physically_valid"] else "INV")
            d = v["delta_from_start"]
            steps.append(f"{outcome.run.recipe_id} {tag} (dLUT {d.get('packed_luts', 0):+.0f}, dWNS {d.get('wns_ns', 0):+.3f})")
        lines.append(f"  {index + 1}. {ctx['phase']} {m['worker_id']} x{m['settings']['max_runs']} "
                     f"{'keep' if m['settings']['preserve_placement'] else 'free'} placement: " + "; ".join(steps))
    return lines


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=sweep.ROOT / "selection.json")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sel = fa.read(args.selection)
    starts, best = fa.start_metrics(), fa.best_known()
    sweep_rows = fa.result_rows(sweep.ROOT)
    pilot_rows = {(r["design"], r["brief"]): r for r in fa.result_rows(fa.V4) if r["policy"] == "repair_loop"}
    cells = sel["cells"]
    v2 = sel["v2"]
    L: list[str] = ["# Repair-loop variant sweep and v2 selection (training designs)", ""]
    cov = sel["coverage"]
    L += [
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from `{sweep.ROOT.relative_to(fa.PROJECT)}`. "
        "Verified against saved results; shares use the pilot's fixed best-known denominators.", "",
        "## Summary", "",
    ]
    if v2 is None:
        L += ["No variant has a complete set of cells, so no v2 could be selected.", ""]
    else:
        knobs = dict(zip(("repair_attempts", "repair_preserve_placement", "gentle_step", "after_failed_repair"), v2["knobs"]))
        ls = sel["lodo_summary"]
        L += [
            f"* **v2 = {v2['grid_id']}** {knobs}; mean share over A0 and T0 on all five designs "
            f"{pct(v2['mean_share_all_five'])}; tied set {v2['tied']}; {v2['knob_changes_from_v1']} knob change(s) from v1.",
            f"* Leave-one-design-out ({ls['folds_complete']} complete folds): mean held-out share "
            f"**v1 {pct(ls['v1_mean_held_out'])}** vs **selection procedure {pct(ls['procedure_mean_held_out'])}**.",
            f"* Coverage: {cov['present_cells']}/{cov['expected_cells']} phase-1 cells.", "",
        ]
    # variant table
    L += ["## 1. All sixteen variants (mean share of best known)", "",
          "| id | attempts | preserve | gentle | after failed repair | A0 | T0 | A0+T0 | runs | new real runs |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for gid, knobs in zip(sweep.GRID_IDS, sweep.GRID):
        def m(brief):
            vals = [cells.get(f"{gid}|{d}|{brief}", {}).get("share") for d in fa.DESIGNS]
            return None if any(v is None for v in vals) else sum(vals) / len(vals)
        runs = sum(c["runs_used"] for k, c in cells.items() if k.startswith(gid + "|"))
        new = sum(c["new_real_runs"] for k, c in cells.items() if k.startswith(gid + "|"))
        L.append(f"| {gid}{' (v1)' if gid == 'g01' else ''} | {knobs[0]} | {knobs[1]} | {'on' if knobs[2] else 'off'} | {knobs[3]} | "
                 f"{pct(m('A0'))} | {pct(m('T0'))} | {pct(sel['all_five_scores'].get(gid))} | {runs} | {new} |")
    L += ["", "New real runs are attributed to the cell that first needed them; later cells reuse the cache.", ""]
    # LODO
    L += ["## 2. Leave one design out", "",
          "| held-out design | chosen (tied set) | mean on other four | held-out result | v1 held-out |",
          "|---|---|---|---|---|"]
    for row in sel["lodo"]:
        L.append(f"| {row['held_out']} | {row['chosen']} ({', '.join(row['tied'])}) | {pct(row['train_mean'])} | "
                 f"{pct(row['held_out_result'])} | {pct(row['v1_held_out'])} |")
    L += ["", f"Mean held-out result: v1 **{pct(sel['lodo_summary']['v1_mean_held_out'])}**, "
          f"selection procedure **{pct(sel['lodo_summary']['procedure_mean_held_out'])}**.", ""]
    # v2 on T and A
    L += ["## 3. v2 on the control briefs T and A against v1", ""]
    if v2 is None:
        L += ["Not run.", ""]
    else:
        rows2 = {(r["design"], r["brief"]): r for r in sweep_rows if r.get("phase") == 2}
        L += ["| brief | design | v1 (pilot) | v2 | difference (points) |", "|---|---|---|---|---|"]
        flags = []
        worst = (0.0, None, None)
        for brief in sweep.PHASE2_BRIEFS:
            d1, d2 = [], []
            for design in fa.DESIGNS:
                metric = fa.OBJECTIVE[brief][0]
                s1 = fa.share(brief, design, pilot_rows[(design, brief)]["final_metrics"][metric], starts, best)
                r2 = rows2.get((design, brief))
                s2 = None if r2 is None else fa.share(brief, design, r2["final_metrics"][metric], starts, best)
                diff = None if (s1 is None or s2 is None) else (s2 - s1) * 100
                L.append(f"| {brief} | {design} | {pct(s1)} | {pct(s2)} | {'-' if diff is None else f'{diff:+.1f}'} |")
                if s1 is not None and s2 is not None:
                    d1.append(s1); d2.append(s2)
                    if (s1 - s2) * 100 > worst[0]:
                        worst = ((s1 - s2) * 100, brief, design)
            if d1 and len(d1) == len(d2):
                drop = (sum(d1) - sum(d2)) / len(d1) * 100
                L.append(f"| **{brief} mean** | {len(d1)} of 5 designs | **{pct(sum(d1)/len(d1))}** | **{pct(sum(d2)/len(d2))}** | **{-drop:+.1f}** |")
                if drop > 5:
                    flags.append(f"{brief}: mean drop {drop:.1f} points")
        L += ["", ("**FLAG: v2 drops more than 5 points against v1 on " + "; ".join(flags) + ".**") if flags
              else "No mean drop above 5 points against v1 on T or A (per-design drops are in the table).", ""]
        if worst[1] is not None and worst[0] > 5:
            brief, design = worst[1], worst[2]
            metric = fa.OBJECTIVE[brief][0]
            L += [f"### Evidence: the largest single drop ({design} / {brief}, {worst[0]:.1f} points), from the committed journals", "",
                  "v1 (pilot):"] + episode_lines(f"pilot_{design}_{brief}_repair_loop", metric) + [
                  "", f"v2 ({v2['grid_id']}):"] + episode_lines(f"sweep_{design}_{brief}_{v2['grid_id']}", metric) + [
                  "", "Inferred reading: compare the repair step (the second mandate). If v2 lets the repair re-place and spends "
                  "two attempts, it can use them on re-placing recipes that do not fix the limit, while v1's placement-keeping "
                  "single attempt did; budget then runs out. The four best variants on A0 and T0 all allow re-placing in the repair, "
                  "so the selection rule favoured the knob that the control brief penalises.", ""]
    # reproduction check
    L += ["## 4. Does g01 reproduce the pilot's repair loop?", ""]
    mismatches = []
    for design in fa.DESIGNS:
        for brief in sweep.PHASE1_BRIEFS:
            cell = cells.get(f"g01|{design}|{brief}")
            pil = pilot_rows.get((design, brief))
            if cell is None or pil is None:
                mismatches.append(f"{design}/{brief}: missing")
            elif cell["final_state_id"] != pil["final_state_id"]:
                mismatches.append(f"{design}/{brief}: final state differs")
    L += [("Yes: g01 ends in the pilot's exact final state on all ten A0/T0 cells." if not mismatches
           else "Differences: " + "; ".join(mismatches)), ""]
    L += ["## 5. Cost, coverage and caveats", "",
          f"* Phase-1 cells present {cov['present_cells']}/{cov['expected_cells']}; missing: {cov['missing_cells'] or 'none'}.",
          f"* Total new real recipe runs reserved by this sweep: "
          f"{len(fa.read(sweep.ROOT / 'new_real_runs.json')['reservations'])} (cap {sweep.CAP}).",
          "* Interpretation (inferred, to confirm): `continue` = after a repair that did not yield a brief-valid improvement, "
          "take the next bold step from the repaired state if it improved the objective over the cycle-start best valid state "
          "(physically valid, still violating); `gentle off` removes the gentle step, so a failed bold stops the episode "
          "(the identical bold would be a duplicate).",
          "* Shares above 100% mean the sweep found a state better than the pilot's best-known pool.",
          "* Selection used only training designs; nothing here touches held-out test designs.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
