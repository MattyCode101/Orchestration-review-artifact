"""Saturday extension of the Friday repair-loop sweep: every variant on briefs T and A.

The authors' Saturday rule for the rule-based v2 asks for the best variant among those
within 5 points of v1 on both control briefs T and A. The Friday sweep ran those
briefs only for g14 (and the pilot only for v1 = g01), so this driver runs the
other fourteen variants on T and A for the five training designs.

It reuses scripts/friday_sweep.py unchanged: same verified engine, the sealed v4
training cache root used in place (additions only), the counted real-run
backend, AMD checks of every new output. The Friday module's campaign constants
are rebound only inside `rebound()`, never at import, so neither the finished
Friday campaign nor its tests can be disturbed. Training designs only.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from scripts import friday_analysis as fa
from scripts import friday_sweep as sweep
from scripts import friday_sweep_select as old_select

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
FRIDAY_ROOT = sweep.ROOT  # captured at import, before any rebinding
ROOT = PROJECT / "data/pilots/saturday_sweep_ta_2026-11-06"
SELECT_SCRIPT = PROJECT / "scripts/saturday_v2_selection.py"
CAP = 150
SHARDS = {1: 12, 2: 1}
BRIEFS = ("T", "A")
SKIP = ("g01", "g14")  # v1 is the pilot's own result; g14 already ran on T and A on Friday


def variant_order(scores: dict[str, float]) -> list[str]:
    """Best criterion first so that, if the cap ever binds, the likeliest picks are done."""

    return sorted(
        (g for g in sweep.GRID_IDS if g not in SKIP),
        key=lambda g: (-scores[g], old_select.knob_changes(g), g),
    )


def rows(scores: dict[str, float] | None = None) -> list[tuple[str, str, str]]:
    if scores is None:
        scores = fa.read(FRIDAY_ROOT / "selection.json")["all_five_scores"]
    return [
        (design, brief, grid)
        for grid in variant_order(scores) for design in sweep.DESIGNS for brief in BRIEFS
    ]


@contextlib.contextmanager
def rebound():
    """Point the Friday sweep module at this campaign, then restore it exactly."""

    names = (
        "ROOT", "LAUNCH", "CONTROL", "SCRIPT", "CAP", "PHASE1_SOFT_LIMIT", "SHARDS",
        "PHASE1_BRIEFS", "SOURCE_FILES", "phase1_rows", "_BOUND",
    )
    saved = {name: getattr(sweep, name) for name in names}
    schedule = rows()
    try:
        sweep.ROOT = ROOT
        sweep.LAUNCH = ROOT / "sweep_launch.json"
        sweep.CONTROL = ROOT / "sweep_control.json"
        sweep.SCRIPT = SCRIPT
        sweep.CAP = CAP
        sweep.PHASE1_SOFT_LIMIT = None
        sweep.SHARDS = dict(SHARDS)
        sweep.PHASE1_BRIEFS = BRIEFS
        sweep.SOURCE_FILES = {**saved["SOURCE_FILES"], "ta_driver": "scripts/saturday_sweep_ta.py"}
        sweep.phase1_rows = lambda: list(schedule)
        sweep._BOUND = False
        yield
    finally:
        for name, value in saved.items():
            setattr(sweep, name, value)


def control() -> None:
    """Run the one phase, then the revised v2 selection (retried every five minutes)."""

    sweep._bind()
    sweep._ctl("starting")
    sweep._run_phase(1)
    sweep._ctl("selecting")
    ok = sweep._run_until_success(SELECT_SCRIPT, ROOT / "selection_saturday.json", "select", 24)
    sweep._ctl("complete" if ok else "select_failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "launch", "control", "watch", "status", "plan"):
        modes.add_argument(f"--{name}", action="store_true")
    modes.add_argument("--shard", type=int)
    parser.add_argument("--phase", type=int, choices=(1,))
    args = parser.parse_args()
    if args.plan:
        schedule = rows()
        print(json.dumps({"cells": len(schedule), "first": schedule[:3], "cap": CAP}))
        return 0
    with rebound():
        if args.prepare:
            print(json.dumps(sweep.prepare(), sort_keys=True))
        elif args.launch:
            print(json.dumps(sweep.launch(), sort_keys=True))
        elif args.control:
            control()
        elif args.watch:
            sweep.watch()
        elif args.status:
            print(json.dumps(sweep.status(), sort_keys=True))
        else:
            if args.phase is None:
                parser.error("--shard needs --phase")
            sweep.run_shard(args.phase, args.shard)
    return 0


if __name__ == "__main__":
    sys.exit(main())
