"""Test-run driver: add rows to the task list (plan/tasks.json), idempotent and append-only.

A row is (design, brief, approach, tag, model). Tiers set the queue order (a cap that binds blocks the least important rows first).
Every addition is recorded in plan/plan_log.jsonl with the hash of the task list after it. A running queue re-reads the list every loop.

  tr_plan.py baselines [--designs ...] [--briefs ...]       the sealed deterministic rows of the task: v1, v2, fixed, grid, random_0..2,
                                                             bandit (not W2R)
  tr_plan.py llm --rows F1-R1 v21 GA-R1 V3 V6 --repeats r1 r2 r3 [--model gemini-3.1-flash-lite | gateway:<exact name>] [--designs ...] [--briefs ...]
                                                             (default briefs: T0 A0 T A for llm1 and llm3, W2R L2R P0 for llm4)
  tr_plan.py dry --design D --brief B --row R [--model M]    one VALIDATION episode for the dry run (TR_DRY=1)
Add --write to append; without it the additions are only printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as tc  # noqa: E402

TIER = {"v1": 1, "v2": 1, "fixed": 1, "grid": 1, "bandit": 1, "random_0": 1, "random_1": 2, "random_2": 2,
        "F1-R1": 1, "v21": 1, "GA-R1": 1, **{v: 1 for v in tc.V22_ROWS}}
LLM = ("F1-R1", "v21", "GA-R1") + tc.V22_ROWS             # V2..V8: template v2.2 variants (Sat 14 Nov)
DESIGN_ORDER = ("vexriscv_re-place_v2", "corescore_500_mod", "rosetta_digit-recognition", "fir_symmetric_systolic", "fir_systolic_transposed")
POWER_DESIGNS = tuple(tc.DEVELOPMENT) + tuple(tc.TRAINING)


def make(design: str, brief: str, approach: str, tag: str, model: str | None = None, budget: int | None = None, tier: int | None = None) -> dict:
    task = {"episode_id": tc.episode_id(design, brief, approach, tag, model, budget), "design": tc.guard_design(design),
            "group": tc.group_of(design), "scope": tc.scope_of(design), "brief": brief, "approach": approach, "tag": tag,
            "tier": TIER[approach] + (0 if tag in ("x", "r1") else 1) if tier is None else int(tier)}
    if budget not in (None, 6):
        task["budget"] = tc.brief_for(design, brief, budget).budget_runs      # validated (3 or 10); the episode runner rebuilds the brief from it
    if approach == "fixed":
        order = tc.jread(tc.FIXED_ORDERS)["chosen"][brief]
        task["fixed_order"] = order
    if approach in LLM:
        task["model"] = model or tc.GEMINI
        task["seed"] = tc.SEEDS.get(tag)
        task["temperature"] = tc.TEMPERATURE_LLM       # 0.3 for every LLM row (decision of Fri 13 Nov 21:45; GA-R1 ran at 0.0 in earlier dry runs)
    return task


def existing() -> list[dict]:
    return tc.tasks()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind", choices=("baselines", "llm", "dry", "power"))
    ap.add_argument("--designs", nargs="*", default=None)
    ap.add_argument("--briefs", nargs="*", default=None)
    ap.add_argument("--rows", nargs="*", default=None)
    ap.add_argument("--repeats", nargs="*", default=["r1"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--budget", type=int, default=None, help="recipe-run budget of the brief: 3 or 10 (default: the frozen 6)")
    ap.add_argument("--tier", type=int, default=None, help="queue tier of every added row (lower runs first); default by row and repeat")
    ap.add_argument("--design")
    ap.add_argument("--brief")
    ap.add_argument("--row")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    new: list[dict] = []
    if args.kind == "baselines":
        for design in args.designs or DESIGN_ORDER:
            for brief in args.briefs or tc.BRIEFS:
                for row in args.rows or ("v1", "v2", "fixed", "grid", "random_0", "random_1", "random_2", "bandit"):
                    if row == "bandit" and brief == "W2R":
                        continue                                  # no prior: not applicable
                    new.append(make(design, brief, row, "x", None, args.budget, args.tier))
    elif args.kind == "llm":
        if tc.QUEUE == "power":
            raise SystemExit("refused: the power-review queue is planned with the 'power' kind")
        # llm1 and llm3: the four two-worker briefs (decision of 13 and 14 Nov); llm4: the three power briefs (decision of 14 Nov)
        default_briefs = ("T0", "A0", "T", "A") if tc.QUEUE in ("llm1", "llm3", "llm6") else tc.THREE_WORKER if tc.QUEUE == "llm4" else tc.BRIEFS
        for design in args.designs or DESIGN_ORDER:
            for brief in args.briefs or default_briefs:
                for row in args.rows or LLM:
                    for rep in args.repeats:
                        new.append(make(design, brief, row, rep, args.model, args.budget, args.tier))
    elif args.kind == "power":
        # Power review (TR_QUEUE=power): W2R, L2R, P0 on the 5 TRAINING and 3 VALIDATION designs; goal-repair v2 and the declared fixed order
        # (one task each), F1-R1 and v2.1 (one repeat, temperature 0.3).
        if tc.QUEUE != "power":
            raise SystemExit("refused: the 'power' kind belongs to TR_QUEUE=power")
        for design in args.designs or POWER_DESIGNS:
            for brief in args.briefs or tc.THREE_WORKER:
                for row in ("v2", "fixed"):
                    new.append(make(design, brief, row, "x"))
                for row in ("F1-R1", "v21"):
                    new.append(make(design, brief, row, "r1", args.model))
    else:
        new.append(make(args.design, args.brief, args.row, args.repeats[0], args.model, args.budget, args.tier))
    have = {t["episode_id"] for t in existing()}
    add = [t for t in new if t["episode_id"] not in have]
    new_sorted = sorted(add, key=lambda t: (t["tier"], t["brief"] in tc.THREE_WORKER, t["design"]))
    counts: dict[str, int] = {}
    for t in add:
        k = f"{t['brief']}:{t['approach']}" + (f"@b{t['budget']}" if t.get("budget") else "")
        counts[k] = counts.get(k, 0) + 1
    print(json.dumps({"requested": len(new), "already_planned": len(new) - len(add), "to_add": len(add),
                      "by_brief_row": counts}, sort_keys=True))
    if args.write and add:
        total = append(new_sorted, args.kind, {"rows": args.rows, "repeats": args.repeats, "model": args.model})
        print("written:", total, "tasks")
    return 0


def append(add: list[dict], kind: str, meta: dict) -> int:
    """Append tasks (already filtered for duplicates) to plan/tasks.json atomically and log the addition."""

    allt = existing() + add
    body = json.dumps(allt, indent=1, sort_keys=True).encode() + b"\n"
    tc.PLAN.mkdir(parents=True, exist_ok=True)
    tmp = tc.PLAN / "tasks.json.tmp"
    tmp.write_bytes(body)
    tmp.replace(tc.PLAN / "tasks.json")
    with (tc.PLAN / "plan_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at_utc": tc.utc(), "kind": kind, "added": len(add), "tasks_total": len(allt),
                            "tasks_sha256": hashlib.sha256(body).hexdigest(), **meta, "caps": tc.CAPS,
                            "plan_note_sha256": tc.plan_sha()}, sort_keys=True) + "\n")
    return len(allt)


if __name__ == "__main__":
    raise SystemExit(main())
