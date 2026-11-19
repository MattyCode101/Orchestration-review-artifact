"""v2.2 grid: run (or resume) one planned episode and write its result; started by v22_queue.py, one process per episode.

The episode is the campaign harness's own (scripts/campaign_harness.py, imported): manifest-cached stored outcomes,
real runs only for uncached (state, recipe) pairs, every new output start-relative and AMD-checked before it can be a
parent, the final pick AMD-checked, caps in the shared ledger. The loop is orch.episode_r2.run_episode_r2 with
ROUND2 (empty calls are observations, the stop rule). The journal resumes after any interruption.

Exit codes: 0 done, 3 capped, 4 infrastructure failure (model API or telemetry extraction), 5 invalid output
(a new output failed the AMD check), 1 any other error. The result or the reason is written to data/.../results/.
usage: v22_episode.py --key EPISODE_ID
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v22_common as dc  # noqa: E402

import campaign_harness as h  # noqa: E402
import thursday_night_pilot as pilot  # noqa: E402
from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.engine.journal import Journal  # noqa: E402
from orch.episode_r2 import ROUND2, loop_facts, run_episode_r2  # noqa: E402
from orch.workers.fitted import FittedWorker  # noqa: E402
from orch.workers.live_telemetry import ExtractionCapReached  # noqa: E402


def llm_stats(episode: str) -> dict:
    records = []
    if dc.TELEMETRY.is_file():
        for line in dc.TELEMETRY.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("episode_id") == episode:
                records.append(r)
    tokens = {k: sum(int(r["tokens"].get(k, 0)) for r in records) for k in ("prompt", "completion", "thoughts", "total")}
    return {
        "decision_records": len(records), "model_calls": sum(len(r.get("calls", [])) for r in records),
        "repairs": sum(int(r.get("repair_attempts", 0)) for r in records),
        "fallbacks": sum(bool(r.get("fallback_triggered")) for r in records),
        "noted_asks": sum(bool(r.get("harness_note")) for r in records),
        "truncated_decisions": sum(bool(r.get("truncated_fields")) for r in records),
        "tokens": tokens,
    }


def journal_facts(store_root: Path, episode: str) -> dict:
    events = Journal(store_root).read(episode)
    facts = loop_facts(events)
    mandates = [e["mandate"] for e in events if e.get("type") == "mandate"]
    facts["mandates"] = len(mandates)
    facts["placement_kept"] = sum(bool(m["settings"].get("preserve_placement")) for m in mandates)
    facts["respect_limits"] = sum(bool(m.get("guards")) for m in mandates)
    facts["workers"] = [m["worker_id"] for m in mandates]
    return facts


def run(task: dict) -> tuple[int, dict]:
    design, brief_id = dc.guard_design(task["design"]), task["brief"]
    ctx = dc.build(task["scope"])
    brief = load_frozen_brief(brief_id)
    start = ctx.starts[design]
    start_evidence = ctx.engine.measure_start(start)
    cell = h.Cell(design, brief_id, task["approach"], task["tag"], task["tier"], task["episode_id"])
    workers = h.make_workers(ctx, cell)
    describe = None
    if task["approach"] != "V0":
        live = ctx.provider if task["scope"] == "dev" else dc.live_telemetry(ctx.store, ctx.caps)
        describe = dc.DescribeSource(ctx, live)
    policy = dc.make_policy_v22(ctx, task, sorted(workers), describe=describe)
    fitted = any(isinstance(w, FittedWorker) for w in workers.values())
    if fitted and ctx.provider is None:
        raise pilot.PilotSafetyStop("no telemetry provider for this scope (worker B needs one)")
    ctx.backend.active_episode = task["episode_id"]
    if hasattr(ctx.provider, "active_episode"):
        ctx.provider.active_episode = task["episode_id"]
    if describe is not None and hasattr(describe.live, "active_episode"):
        describe.live.active_episode = task["episode_id"]
    base = {"episode_id": task["episode_id"], "task": task}
    try:
        result = run_episode_r2(
            engine=ctx.engine, registry=ctx.registry, workers=workers, policy=policy, brief=brief, start_state=start,
            episode_id=task["episode_id"], noise=dc.noise_for(design, start_evidence),
            telemetry_provider=ctx.provider if fitted else None, validator=h.final_validator(ctx.ledger, brief),
            options=ROUND2,
        )
    except (h.CapReached, ExtractionCapReached) as exc:
        return 3, {**base, "status": "capped", "reason": str(exc)[:400]}
    except pilot.PilotSafetyStop as exc:
        message = str(exc)
        if message.startswith(("AMD simulation did not PASS", "missing DONT_TOUCH", "more DONT_TOUCH")):
            return 5, {**base, "status": "invalid_output", "reason": message[:400]}
        raise
    except Exception as exc:
        kind = h.infrastructure_kind(exc)
        if kind is None:
            raise
        return 4, {**base, "status": "infrastructure_failure", "kind": kind,
                   "reason": f"{type(exc).__name__}: {str(exc)[:400]}"}
    finally:
        ctx.backend.active_episode = None
    row = h.result_row(ctx, cell, result, policy, 0)
    row.update({
        "schema_version": "v22-result-v1", "status": "done", "group": dc.group_of(design), "task": task,
        "start_metrics": {n.value: m.value for n, m in start_evidence.metrics.items()},
        "noise": {k.value: v for k, v in dc.noise_for(design, start_evidence).items()},
        "loop": journal_facts(ctx.store.root, task["episode_id"]),
        "llm_stats": llm_stats(task["episode_id"]) if task["approach"] in dc.LLM_APPROACHES else None,
        "descriptions_served": dict(describe.served) if describe is not None else None,
        "resumed": bool(result.resumed), "loop_options": ROUND2.as_json(),
    })
    return 0, row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()
    task = next((t for t in dc.tasks() if t["episode_id"] == args.key), None)
    if task is None:
        print(json.dumps({"error": f"no planned task {args.key}"}))
        return 1
    dc.RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        code, row = run(task)
    except Exception as exc:  # noqa: BLE001 - recorded for the queue, which decides on a retry
        code, row = 1, {"episode_id": args.key, "task": task, "status": "error",
                        "reason": f"{type(exc).__name__}: {str(exc)[:600]}", "trace": traceback.format_exc()[-2500:]}
    row["finished_at_utc"] = dc.utc()
    name = f"{args.key}.json" if code == 0 else f"{args.key}.{row['status']}.json"
    dc.jwrite(dc.RESULTS / name, row)
    if code == 0:
        for stale in dc.RESULTS.glob(f"{args.key}.*.json"):
            stale.unlink()
    print(json.dumps({k: row.get(k) for k in ("episode_id", "status", "reason", "runs_used", "new_real_runs",
                                               "stop_reason")}, default=str), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
