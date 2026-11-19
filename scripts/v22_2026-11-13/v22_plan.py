"""Template v2.2 feature grid, revision 2: the task list (data/v22_2026-11-13/grid/plan/tasks.json).

Decisions of 2026-11-13 21:45 (the authors): T0 and A0 only; power plays no part in choosing a version. TRAINING and VALIDATION
designs only (5 + 3), 3 repeats (seeds 0, 1, 2) at temperature 0.3.

  V0 F1-R1 and V1 v2.1                       carried over from D6-v2 (the same policy, loop, seeds and designs; not run again)
  V2  v2.1 without E          V3  v2.1 without P          V4  v2.1 with R          V5  v2.1 with U
  V6  v2.1 with S' and U      V7  v2.1 with S', U and R    V8  compact H, without P, with S', U and R

  tier 1  repeat 1 of V2 to V8
  tier 2  repeats 2 and 3 of V2 to V8

Queue order inside a tier: the cells (design, brief) whose reference exceeds three bands first, then the others; every
variant of one cell together, so a stop leaves complete cells. A task carries `carryover` when it is a D6-v2 episode.
usage: v22_plan.py [--write] [--smoke DESIGN BRIEF VARIANT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v22_common as dc  # noqa: E402

DESIGN_ORDER = ("vtr_mcml_v2", "amd_mini-isp", "t2_cordic_s",
                "finn_radioml", "vexriscv_re-place", "rosetta_3d-rendering", "rosetta_spam-filter", "logicnets_jscl")
TAGS = ("r1", "r2", "r3")
RUN_VARIANTS = ("V2", "V3", "V4", "V5", "V6", "V7", "V8")
CARRY_VARIANTS = ("V0", "V1")
# Cells whose reference exceeds three noise bands (the diagnosis note, Step 1): the cells the selection rule scores.
SCORED_CELLS = (("vtr_mcml_v2", "A0"), ("t2_cordic_s", "A0"), ("t2_cordic_s", "T0"), ("finn_radioml", "A0"),
                ("vexriscv_re-place", "A0"), ("rosetta_3d-rendering", "A0"), ("rosetta_3d-rendering", "T0"),
                ("rosetta_spam-filter", "A0"), ("logicnets_jscl", "A0"))


def cells() -> list[tuple[str, str]]:
    """(design, brief) in queue order: the scored cells first, then the others in design order."""

    rest = [(d, b) for d in DESIGN_ORDER for b in dc.BRIEFS_MAIN if (d, b) not in SCORED_CELLS]
    return [*SCORED_CELLS, *rest]


def rows() -> list[tuple[str, str, str, str, int]]:
    """(variant, design, brief, tag, tier) in queue order; tier 0 is the carried-over D6-v2 rows."""

    out = []
    for tag in TAGS:
        for design, brief in cells():
            for variant in CARRY_VARIANTS:
                out.append((variant, design, brief, tag, 0))
    for tag in TAGS:
        for design, brief in cells():
            for variant in RUN_VARIANTS:
                out.append((variant, design, brief, tag, 1 if tag == "r1" else 2))
    return out


def make_tasks() -> list[dict]:
    if set(DESIGN_ORDER) != set(dc.TRAINING) | set(dc.DEVELOPMENT):
        raise SystemExit("design order differs from the training and development lists")
    tasks = []
    for variant, design, brief, tag, tier in rows():
        dc.guard_design(design)
        task = {"episode_id": dc.episode_id(design, brief, variant, tag), "design": design, "group": dc.group_of(design),
                "scope": "train" if design in dc.TRAINING else "dev", "brief": brief, "approach": variant, "tag": tag,
                "tier": tier, "sub": str(tier), "seed": dc.SEEDS[tag], "scored_cell": (design, brief) in SCORED_CELLS}
        if variant in dc.CARRYOVER_APPROACH:
            task["carryover"] = f"d6_{design}_{brief}_{dc.CARRYOVER_APPROACH[variant]}_{tag}"
        tasks.append(task)
    if len({t["episode_id"] for t in tasks}) != len(tasks):
        raise SystemExit("duplicate episode id")
    return tasks


def smoke_tasks(design: str, brief: str, variant: str, tags: tuple[str, ...]) -> list[dict]:
    dc.guard_design(design)
    if design not in dc.DEVELOPMENT:
        raise SystemExit("the smoke run uses a validation design")
    return [{"episode_id": dc.episode_id(design, brief, variant, tag), "design": design, "group": dc.group_of(design),
             "scope": "dev", "brief": brief, "approach": variant, "tag": tag, "tier": 1, "sub": "1",
             "seed": dc.SEEDS[tag]} for tag in tags]


def summarise(tasks: list[dict]) -> dict:
    by_tier = {}
    for label, pred in (("carryover", lambda t: t["tier"] == 0), ("tier1", lambda t: t["tier"] == 1),
                        ("tier2", lambda t: t["tier"] == 2), ("all", lambda t: True)):
        sel = [t for t in tasks if pred(t)]
        run = [t for t in sel if "carryover" not in t]
        by_tier[label] = {"planned": len(sel), "carryover": len(sel) - len(run), "to_run": len(run),
                          "to_run_by_brief": dict(Counter(t["brief"] for t in run)),
                          "to_run_by_variant": dict(sorted(Counter(t["approach"] for t in run).items())),
                          "to_run_in_scored_cells": sum(1 for t in run if t.get("scored_cell"))}
    return by_tier


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--smoke", nargs=3, metavar=("DESIGN", "BRIEF", "VARIANT"),
                    help="write a two-episode smoke task list (repeats r1 and r2) for one validation design")
    args = ap.parse_args()
    tasks = smoke_tasks(*args.smoke, ("r1", "r2")) if args.smoke else make_tasks()
    body = json.dumps(tasks, indent=1, sort_keys=True).encode() + b"\n"
    situation_sha = hashlib.sha256((dc.PROJECT / dc.SITUATION_TABLE).read_bytes()).hexdigest() \
        if (dc.PROJECT / dc.SITUATION_TABLE).is_file() else None
    plan = {"schema_version": "v22-plan-v2", "tasks": len(tasks), "tasks_sha256": hashlib.sha256(body).hexdigest(),
            "caps": dc.CAPS, "load_guard": dc.LOAD_GUARD, "max_tool_jobs": dc.MAX_TOOL_JOBS, "temperature": dc.TEMPERATURE,
            "seeds": dc.SEEDS, "price": dc.PRICE, "work": str(dc.WORK), "situation_table_sha256": situation_sha,
            "created_at_utc": dc.utc()}
    print(json.dumps({"plan": plan, "summary": summarise(tasks)}, indent=1, sort_keys=True))
    if args.write:
        if (dc.PLAN / "tasks.json").exists():
            raise SystemExit("the task list exists; it is written once")
        dc.PLAN.mkdir(parents=True, exist_ok=True)
        (dc.PLAN / "tasks.json").write_bytes(body)
        dc.jwrite(dc.PLAN / "plan.json", plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
