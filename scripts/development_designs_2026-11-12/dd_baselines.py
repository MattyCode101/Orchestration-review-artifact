"""Step 6: deterministic baselines on the development designs, through the campaign harness (imported, not edited).

Cells per design: briefs T0, A0, W2R, L2R, P0; approaches v2, the best fixed order (training-chosen), random seeds 0, 1, 2.
(The grid search needs no episode: it is read from the stored depth-one outcomes by dd_exhaustive.py.)
Each episode is the harness's own: run_episode with the FINAL workers (timing, area, and power for the three-worker briefs),
live worker B telemetry, every new output AMD-checked before it can be a parent, caps enforced by the shared cap ledger.
W2R and L2R run on the harness's three-worker path (cell brief 'P0') with their own Brief object.

Shards: this process runs the cells whose index modulo --of equals --shard, in tier order (tier 1 = v2 and the fixed order,
tier 2 = random seed 0, tier 3 = seeds 1 and 2), so a cap that binds blocks the least important rows first.
Results: data/.../results/<episode>.json (done) or <episode>.blocked.json (capped / infrastructure / error; re-queued on resume).

usage: dd_baselines.py --shard I --of N [--min-tier 1] [--max-tier 3] [--plan-only] [--cells design:brief:approach ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dd_common as dd  # noqa: E402
import dd_scope as ds  # noqa: E402
import campaign_harness as h  # noqa: E402

RESULTS = dd.WORK / "results"
TRAIN = dd.WORK / "train_choices.json"
DESIGN_ORDER = ("vtr_mcml_v2", "t2_cordic_s", "amd_mini-isp")        # the slowest design first
STOP = dd.WORK / "STOP_BASELINES"


def fixed_for(brief_id: str) -> str:
    """The best fixed order, chosen on the five training designs only (train_choices.json, hashed in the plan inputs)."""

    fixed = dd.jread(TRAIN)["fixed"]
    return fixed["two_worker"][brief_id] if brief_id in dd.TWO_WORKER else fixed["three_worker"][brief_id]["chosen"]


def plan_cells() -> list[dict]:
    cells = []
    for tier, approaches in ((1, ("v2", "FIXED")), (2, ("random_0",)), (3, ("random_1", "random_2"))):
        for brief in dd.BRIEFS:
            for design in DESIGN_ORDER:
                for a in approaches:
                    approach = fixed_for(brief) if a == "FIXED" else a
                    cells.append({"design": design, "brief": brief, "approach": approach, "tag": "x", "tier": tier,
                                  "episode_id": f"dev_{design}_{brief}_{approach}_x"})
    if len({c["episode_id"] for c in cells}) != len(cells):
        raise SystemExit("duplicate cell")
    return cells


def run_one(ctx: h.Context, c: dict) -> tuple[str, dict]:
    design, brief_id = c["design"], c["brief"]
    brief = dd.make_brief(brief_id, dd.power_offset(design))
    hb = "P0" if brief_id in dd.THREE_WORKER else brief_id          # the harness picks workers and policy family from this
    cell = h.Cell(design, hb, c["approach"], c["tag"], c["tier"], c["episode_id"])
    start = ctx.starts[design]
    start_evidence = ctx.engine.measure_start(start)
    workers = h.make_workers(ctx, cell)
    policy = h.make_policy(ctx, cell, sorted(workers))
    fitted = any(isinstance(w, h.FittedWorker) for w in workers.values())
    ctx.backend.active_episode = cell.episode_id
    ctx.provider.active_episode = cell.episode_id
    try:
        result = h.run_episode(
            engine=ctx.engine, registry=ctx.registry, workers=workers, policy=policy, brief=brief, start_state=start,
            episode_id=cell.episode_id, noise=ctx.scope.noise(design, start_evidence),
            telemetry_provider=ctx.provider if fitted else None, validator=h.final_validator(ctx.ledger, brief))
    except (h.CapReached, h.ExtractionCapReached) as exc:
        return "capped", {"episode_id": cell.episode_id, "reason": str(exc)}
    except Exception as exc:
        kind = h.infrastructure_kind(exc)
        info = {"episode_id": cell.episode_id, "kind": kind or "error", "reason": f"{type(exc).__name__}: {str(exc)[:400]}"}
        if kind is None:
            info["trace"] = traceback.format_exc()[-1500:]
        return "blocked", info
    finally:
        ctx.backend.active_episode = None
        ctx.provider.active_episode = None
    row = h.result_row(ctx, cell, result, policy, 0)
    row.update(brief=brief_id, cell_brief_for_workers=hb, tier=c["tier"], design_split="development")
    return "done", row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--max-tier", type=int, default=3)
    ap.add_argument("--min-tier", type=int, default=1)
    ap.add_argument("--ignore-stop", action="store_true", help="do not honour the STOP_BASELINES file (used to hand queued cells to new shards)")
    ap.add_argument("--list-queued", action="store_true", help="print cells with no result and no episode directory, then exit")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--cells", nargs="*", default=None, help="design:brief:approach filters")
    args = ap.parse_args()
    cells = [c for c in plan_cells() if args.min_tier <= c["tier"] <= args.max_tier]
    if args.cells:
        keep = {tuple(x.split(":")) for x in args.cells}
        cells = [c for c in cells if (c["design"], c["brief"], c["approach"]) in keep]
    mine = [c for i, c in enumerate(cells) if i % args.of == args.shard]
    if args.list_queued:
        import os

        started = set(os.listdir(dd.WORK / "engine" / "episodes")) if (dd.WORK / "engine" / "episodes").is_dir() else set()
        for c in cells:
            if not (RESULTS / f"{c['episode_id']}.json").is_file() and c["episode_id"] not in started:
                print(f"{c['design']}:{c['brief']}:{c['approach']}")
        return 0
    if args.plan_only:
        print(json.dumps({"cells": len(cells), "mine": len(mine), "first": [c["episode_id"] for c in mine[:5]]}, indent=1))
        by = {}
        for c in cells:
            by.setdefault((c["tier"], c["approach"]), 0)
            by[(c["tier"], c["approach"])] += 1
        print({f"t{k[0]}:{k[1]}": v for k, v in sorted(by.items())})
        return 0
    RESULTS.mkdir(parents=True, exist_ok=True)
    ctx = ds.build("run", with_provider=True)
    for c in mine:
        if (STOP.exists() and not args.ignore_stop) or ctx.caps.stopped():
            print("stopping:", "STOP file" if STOP.exists() else ctx.caps.stopped(), flush=True)
            break
        done = RESULTS / f"{c['episode_id']}.json"
        if done.is_file():
            continue
        status, row = run_one(ctx, c)
        if status == "done":
            done.write_text(json.dumps(row, indent=1, sort_keys=True, default=str), encoding="utf-8")
            (RESULTS / f"{c['episode_id']}.blocked.json").unlink(missing_ok=True)
            print("done", c["episode_id"], "runs", row["runs_used"], "new", row["new_real_runs"], "final", {k: row["final_metrics"][k] for k in ("wns_ns", "packed_luts", "dynamic_power_w")}, flush=True)
        else:
            (RESULTS / f"{c['episode_id']}.blocked.json").write_text(json.dumps({"status": status, **row, "at_utc": dd.utc()}, indent=1), encoding="utf-8")
            print(status, c["episode_id"], row.get("reason"), flush=True)
            if status == "capped":
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
