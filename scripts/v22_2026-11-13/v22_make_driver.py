"""Make the grid's queue, control and episode scripts from the pinned D6-v2 ones (which are read, never edited).

    .venv/bin/python scripts/v22_2026-11-13/v22_make_driver.py [--write]

Each output is the D6-v2 script with its names renamed (d6_* -> v22_*, paths and titles) and a short list of exact-once
patches, all listed below; an output that exists is checked, never overwritten. The patches:
  queue    at most 15 episodes in flight (one tool job each); no 09:00 checkpoint; blocked and carry-over tasks are not run
           and not counted; the cap texts use the grid's caps
  ctl      status counts the same tasks as the queue
  episode  every variant but V0 gets a description source; the policy comes from v22_common.make_policy_v22
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "scripts/d6v2_2026-11-12"
DST = PROJECT / "scripts/v22_2026-11-13"
RENAMES = [("d6_common", "v22_common"), ("d6_episode.py", "v22_episode.py"), ("d6_queue.py", "v22_queue.py"),
           ("d6_ctl.py", "v22_ctl.py"), ("d6_plan.py", "v22_plan.py"), ("D6-v2", "v2.2 grid"),
           ("d6v2_2026-11-12", "v22_2026-11-13"), ("d6v2-result-v1", "v22-result-v1"), ("d6 queue", "v22 queue")]
FILTER = '        tasks = [t for t in tasks if not t.get("blocked") and not t.get("carryover")]\n'
PATCHES = {
    "queue": [
        ('    "max_active_with_depth3": 16,\n    "max_active": 40,\n',
         '    "max_active_with_depth3": 15,\n    "max_active": 15,\n'),          # an episode holds one tool job at a time
        ('    "checkpoint_utc": "2026-11-13T09:00:00+00:00",\n', '    "checkpoint_utc": None,\n'),
        ('        if (not self.checkpoint_done and not self.fake\n', '        if (self.cfg["checkpoint_utc"] and not self.checkpoint_done and not self.fake\n'),
        ('        tasks = dc.tasks()\n        rows = self.rows()\n',
         '        tasks = dc.tasks()\n' + FILTER + '        rows = self.rows()\n'),
        ("of 450 new runs, \"", "of {dc.CAPS['new_real_runs']} new runs, \""),
        ("of 100 job-hours, US${used.get('usd')} of 15. \"", "of {dc.CAPS['job_hours']} job-hours, US${used.get('usd')} of {dc.CAPS['usd']}. \""),
    ],
    "ctl": [
        ("    tasks = dc.tasks()\n    rows = dc.read_journal()\n",
         "    tasks = [t for t in dc.tasks() if not t.get(\"blocked\") and not t.get(\"carryover\")]\n    rows = dc.read_journal()\n"),
    ],
    "episode": [
        ('    if task["approach"] == "v21":\n', '    if task["approach"] != "V0":\n'),
        ("    policy = dc.make_policy(ctx, task, sorted(workers), describe=describe)\n",
         "    policy = dc.make_policy_v22(ctx, task, sorted(workers), describe=describe)\n"),
    ],
}


def make(kind: str) -> str:
    text = (SRC / f"d6_{kind}.py").read_text(encoding="utf-8")
    for old, new in RENAMES:
        text = text.replace(old, new)
    for i, (old, new) in enumerate(PATCHES[kind]):
        if text.count(old) != 1:
            raise SystemExit(f"{kind} patch {i}: the old text occurs {text.count(old)} times")
        text = text.replace(old, new)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    for kind in ("queue", "ctl", "episode"):
        path = DST / f"v22_{kind}.py"
        text = make(kind)
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise SystemExit(f"{path} exists and differs; it is never overwritten")
            print("exists, identical:", path.name)
        elif args.write:
            path.write_text(text, encoding="utf-8")
            print("written:", path.name, len(text), "chars")
        else:
            print("would write:", path.name, len(text), "chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
