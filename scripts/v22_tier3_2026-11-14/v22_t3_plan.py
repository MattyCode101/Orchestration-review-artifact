"""Template v2.2 tier 3: the task list (data/v22_2026-11-14/tier3/plan/tasks.json).

Decisions of 2026-11-14 11:30 (the authors): V9 and V10 (16 cells, 3 repeats each) and a same-day re-run of V0 (F1-R1) and V1 (v2.1)
(16 cells, 3 repeats, new seeds, current renderer path). 4 rows x 48 = 192 episodes (cap 200). TRAINING and VALIDATION designs only.

Queue order: repeat by repeat (t1, t2, t3), inside a repeat the scored cells first (reference over three bands), every variant of a
cell together, so a stop leaves complete repeats and complete cells. Tags t1 to t3 are seeds 3, 4, 5 (the grid used 0, 1, 2).
usage: v22_t3_plan.py [--write]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v22_t3_common as dc  # noqa: E402
import v22_plan as old  # noqa: E402  (the grid's plan module: only cells() and SCORED_CELLS are used)


def make_tasks() -> list[dict]:
    if set(old.DESIGN_ORDER) != set(dc.TRAINING) | set(dc.DEVELOPMENT):
        raise SystemExit("design order differs from the training and development lists")
    tasks = []
    for tag in dc.TAGS_T3:
        for design, brief in old.cells():
            for variant in dc.VARIANTS_T3:
                dc.guard_design(design)
                tier = 1 if tag == "t1" else 2
                tasks.append({"episode_id": dc.episode_id(design, brief, variant, tag), "design": design,
                              "group": dc.group_of(design), "scope": "train" if design in dc.TRAINING else "dev",
                              "brief": brief, "approach": variant, "tag": tag, "tier": tier, "sub": str(tier),
                              "seed": dc.SEEDS[tag], "scored_cell": (design, brief) in old.SCORED_CELLS})
    if len({t["episode_id"] for t in tasks}) != len(tasks):
        raise SystemExit("duplicate episode id")
    if len(tasks) > dc.EPISODE_CAP:
        raise SystemExit(f"{len(tasks)} episodes exceed the cap of {dc.EPISODE_CAP}")
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    tasks = make_tasks()
    body = json.dumps(tasks, indent=1, sort_keys=True).encode() + b"\n"
    sha = lambda rel: hashlib.sha256((dc.PROJECT / rel).read_bytes()).hexdigest()  # noqa: E731
    plan = {"schema_version": "v22-t3-plan-v1", "tasks": len(tasks), "tasks_sha256": hashlib.sha256(body).hexdigest(),
            "episode_cap": dc.EPISODE_CAP, "caps": dc.CAPS, "load_guard": dc.LOAD_GUARD, "max_tool_jobs": dc.MAX_TOOL_JOBS,
            "temperature": dc.TEMPERATURE, "seeds": {t: dc.SEEDS[t] for t in dc.TAGS_T3}, "price": dc.PRICE,
            "work": str(dc.WORK), "situation_table_s_prime_sha256": sha("configs/experience/situations_v22.json"),
            "situation_table_s_prime2_sha256": sha("configs/experience/situations_v22_llm.json"),
            "population_text_sha256": sha("configs/prompts/template_v22/situation_llm_texts.yaml"),
            "created_at_utc": dc.utc()}
    summary = {"episodes": len(tasks), "by_variant": dict(sorted(Counter(t["approach"] for t in tasks).items())),
               "by_tag": dict(sorted(Counter(t["tag"] for t in tasks).items())),
               "by_brief": dict(sorted(Counter(t["brief"] for t in tasks).items())),
               "scored_cell_episodes": sum(1 for t in tasks if t["scored_cell"])}
    print(json.dumps({"plan": plan, "summary": summary}, indent=1, sort_keys=True))
    if args.write:
        if (dc.PLAN / "tasks.json").exists():
            raise SystemExit("the task list exists; it is written once")
        dc.PLAN.mkdir(parents=True, exist_ok=True)
        (dc.PLAN / "tasks.json").write_bytes(body)
        dc.jwrite(dc.PLAN / "plan.json", plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
