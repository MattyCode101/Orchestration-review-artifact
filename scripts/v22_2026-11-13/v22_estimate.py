"""Template v2.2 grid, revision 2: cost and time estimate per tier. PLANNING FIGURES, LOW CONFIDENCE. Not yet run.

    .venv/bin/python scripts/v22_2026-11-13/v22_estimate.py [--json]

Basis (all from the D6-v2 campaign on the same designs, TRAINING and VALIDATION; ledger and journal in data/d6v2_2026-11-12):
  LLM cost    US$ 0.005 per episode, a deliberately high figure: D6-v2 averaged US$ 0.0038 per LLM episode (0.3689 over 96).
              The S' variants add a section to every prompt, so their input is longer; the cap (US$ 5) leaves room
  new runs    per episode that is run: T0 0.21 and A0 0.58 (mean of F1 and v2.1 in D6-v2, 38 new runs over 96 episodes)
  job-hours   0.43 per new run (D6-v2: 38.9 job-hours over 91 runs)
  wall        new runs / 28.5 per hour (D6-v2: 91 runs in 3.19 h), when the load guard (160) admits them, plus 15 minutes; no
              discount is taken for runs shared between repeats, and episodes that need no new run add a few minutes only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v22_plan as plan  # noqa: E402

USD_PER_EPISODE = 0.005
NEW_RUNS_PER_EPISODE = {"T0": 0.21, "A0": 0.58}
JOB_HOURS_PER_RUN = 0.43
RUNS_PER_HOUR = 28.5
TIERS = (("tier1", lambda t: t["tier"] == 1), ("tier2", lambda t: t["tier"] == 2), ("total", lambda t: t["tier"] > 0))


def figures(tasks: list[dict]) -> dict:
    run = [t for t in tasks if "carryover" not in t]
    new_runs = sum(NEW_RUNS_PER_EPISODE[t["brief"]] for t in run)
    return {"episodes_to_run": len(run), "usd": round(len(run) * USD_PER_EPISODE, 2), "new_runs": round(new_runs),
            "job_hours": round(new_runs * JOB_HOURS_PER_RUN, 1), "wall_hours": round(new_runs / RUNS_PER_HOUR + 0.25, 1),
            "by_brief": dict(sorted(Counter(t["brief"] for t in run).items())),
            "by_variant": dict(sorted(Counter(t["approach"] for t in run).items()))}


def estimate(tasks: list[dict]) -> dict:
    out = {}
    for name, pred in TIERS:
        sel = [t for t in tasks if pred(t)]
        out[name] = {"planned": len(sel), **figures(sel)}
    out["carried_over"] = sum(1 for t in tasks if "carryover" in t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    result = estimate(plan.make_tasks())
    if args.json:
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0
    print(f"{'tier':8s} {'episodes':>8s} {'US$':>6s} {'new runs':>8s} {'job-h':>6s} {'wall h':>6s}")
    for name in ("tier1", "tier2", "total"):
        row = result[name]
        print(f"{name:8s} {row['episodes_to_run']:8d} {row['usd']:6.2f} {row['new_runs']:8d} {row['job_hours']:6.1f} "
              f"{row['wall_hours']:6.1f}")
    print("carried over from D6-v2 (V0, V1):", result["carried_over"], "| to run by brief:", result["total"]["by_brief"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
