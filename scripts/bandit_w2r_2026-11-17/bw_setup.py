"""Learned Bandit W2R set-up (Tue 17 Nov 2026): the run folder, the five-task list (plan/tasks.json) and the seeded telemetry cache. Idempotent.
Run with TR_QUEUE=bw (set here); with TR_SMOKE=1 for the replay-only smoke folder (TEST designs, no tool job can start).
usage: python bw_setup.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TR_QUEUE", "bw")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as tc  # noqa: E402

ORDER = ("rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2", "fir_symmetric_systolic", "fir_systolic_transposed")


def main() -> int:
    tc.ROOT.mkdir(parents=True, exist_ok=True)
    tasks = [{"episode_id": tc.episode_id(d, "W2R", "bandit", "pf1"), "design": tc.guard_design(d), "group": tc.group_of(d), "scope": tc.scope_of(d),
              "brief": "W2R", "approach": "bandit", "tag": "pf1", "tier": 1} for d in ORDER]
    path = tc.PLAN / "tasks.json"
    if not path.exists():
        tc.jwrite(path, tasks)
        print("wrote", path)
    elif tc.jread(path) != tasks:
        raise SystemExit("refused: the task list differs from the declared one")
    print(json.dumps({"root": str(tc.ROOT), "smoke": tc.SMOKE, "tasks": [t["episode_id"] for t in tasks], "telemetry_seed": tc.seed_telemetry_cache(),
                      "plan_sha": tc.plan_sha()[:12], "caps": tc.CAPS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
