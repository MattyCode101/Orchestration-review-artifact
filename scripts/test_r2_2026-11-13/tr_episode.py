"""Test-run driver: run (or resume) one planned episode and write its result; started by tr_queue.py, one process per episode.

The episode is the campaign harness's own (scripts/campaign_harness.py, imported): manifest-cached stored outcomes (every root, including
data/exhaustive_d3_2026-11-12/engine), real runs only for uncached (state, recipe) pairs, every new output start-relative and AMD-checked
before it can be a parent, the final pick AMD-checked, caps in the shared ledger. The loop is orch.episode_r2.run_episode_r2 with ROUND2.
The journal resumes after any interruption.

Exit codes: 0 done, 3 capped, 4 infrastructure failure (model API or telemetry extraction), 5 invalid output (a new output failed the AMD
check), 6 not applicable (the frozen bandit has no prior for W2R), 7 cache miss in a dry run (replay mode: a tool run would be needed),
1 any other error. The result or the reason is written to RESULTS (the sealed folder for a real run).
usage: tr_episode.py --key EPISODE_ID
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tr_common as tc  # noqa: E402
import tr_policies as tp  # noqa: E402

import campaign_harness as h  # noqa: E402
import thursday_night_pilot as pilot  # noqa: E402
from orch.engine.journal import Journal  # noqa: E402
from orch.episode_r2 import ROUND2, loop_facts, run_episode_r2  # noqa: E402
from orch.workers.fitted import FittedWorker  # noqa: E402
from orch.workers.live_telemetry import ExtractionCapReached  # noqa: E402


@dataclass(frozen=True)
class TCell(h.Cell):
    """The campaign cell with three workers for every power brief (the harness's own rule knows P0 only)."""

    @property
    def workers(self) -> str:
        return "abc" if self.brief in tc.THREE_WORKER else "ab"


def llm_stats(episode: str) -> dict:
    records = []
    if tc.TELEMETRY.is_file():
        for line in tc.TELEMETRY.read_text(encoding="utf-8").splitlines():
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


def miss_facts(ctx, task: dict) -> dict:
    """Replay-only runs (TR_DRY, TR_SMOKE): how far the loop got before it needed a tool run, from the episode's own journal (operational counts)."""

    try:
        return journal_facts(ctx.store.root, task["episode_id"])
    except Exception as exc:  # noqa: BLE001 - a diagnostic only
        return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def run_amd(task: dict) -> tuple[int, dict]:
    """One AMD simulation check of a stored state against the start (B2), through the harness ledger (read-through over every root, so a
    state with a record anywhere is not checked again; a new check is charged to the job-hour cap). Task kind 'amd'."""

    design, state = tc.guard_design(task["design"]), task["state"]
    base = {"episode_id": task["episode_id"], "task": task}
    ctx = tc.build(task["scope"])
    known = ctx.ledger.record_location(design, state) is not None
    try:
        record = ctx.ledger.verify(design, state)
    except (h.CapReached, ExtractionCapReached) as exc:
        return 3, {**base, "status": "capped", "reason": str(exc)[:400]}
    wall = sum(float(a.get("wall_s") or 0.0) for a in record.get("attempts", []))
    return 0, {**base, "status": "done", "amd_result": record.get("result"), "was_known": known, "wall_s": wall,
               "state": state, "design": design}


def run(task: dict) -> tuple[int, dict]:
    if task.get("kind") == "amd":
        return run_amd(task)
    design, brief_id = tc.guard_design(task["design"]), task["brief"]
    base = {"episode_id": task["episode_id"], "task": task}
    ctx = tc.build(task["scope"])
    brief = tc.brief_for(design, brief_id, task.get("budget"))        # budget 3 or 10 only for the queue 5 sensitivity rows
    start = ctx.starts[design]
    start_evidence = ctx.engine.measure_start(start)
    cell = TCell(design, brief_id, task["approach"], task["tag"], task["tier"], task["episode_id"])
    workers = h.make_workers(ctx, cell)
    if task["approach"] == "grid":
        workers = tp.direct_workers(workers)
    describe = None
    if task["approach"] == "v21" or task["approach"] in tc.V22_ROWS:
        # (the v2.2 variants get a description source too; a variant without the physical description never calls it)
        # TEST and VALIDATION scopes: the scope's live provider; TRAINING scope: its provider is the stored v4 telemetry, so descriptions
        # of states it does not hold go through the accounted live extractor (as D6-v2 did)
        live = ctx.provider if task["scope"] in ("dev", "test") else tc.live_telemetry(ctx.store, ctx.caps)
        describe = tc.DescribeSource(ctx, live)
    try:
        policy = tp.make_policy(ctx, task, sorted(workers), describe=describe)
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ == "BanditNotApplicable":
            return 6, {**base, "status": "not_applicable", "reason": str(exc)[:300]}
        raise
    fitted = any(isinstance(w, FittedWorker) for w in workers.values())
    if fitted and ctx.provider is None:
        raise pilot.PilotSafetyStop("no telemetry provider for this scope (worker B needs one)")
    ctx.backend.active_episode = task["episode_id"]
    if hasattr(ctx.provider, "active_episode"):
        ctx.provider.active_episode = task["episode_id"]
    if describe is not None and hasattr(describe.live, "active_episode"):
        describe.live.active_episode = task["episode_id"]
    noise = tc.noise_for(design, start_evidence)
    try:
        result = run_episode_r2(
            engine=ctx.engine, registry=ctx.registry, workers=workers, policy=policy, brief=brief, start_state=start,
            episode_id=task["episode_id"], noise=noise, telemetry_provider=ctx.provider if fitted else None,
            validator=h.final_validator(ctx.ledger, brief), options=ROUND2,
        )
    except (h.CapReached, ExtractionCapReached) as exc:
        extra = {"loop_at_miss": miss_facts(ctx, task)} if tc.REPLAY else {}          # replay-only: how far the loop got before it needed an extraction
        return 3, {**base, "status": "capped", "reason": str(exc)[:400], **extra}
    except pilot.PilotSafetyStop as exc:
        message = str(exc)
        if message.startswith(("AMD simulation did not PASS", "missing DONT_TOUCH", "more DONT_TOUCH")):
            return 5, {**base, "status": "invalid_output", "reason": message[:400]}
        if tc.REPLAY and message.startswith("replay would start"):
            return 7, {**base, "status": "cache_miss", "reason": message[:400], "loop_at_miss": miss_facts(ctx, task)}
        raise
    except Exception as exc:
        if tc.REPLAY and type(exc).__name__ == "VerificationFailure" and "missing validator record" in str(exc):
            return 7, {**base, "status": "cache_miss", "reason": f"{type(exc).__name__}: {str(exc)[:300]}", "loop_at_miss": miss_facts(ctx, task)}
        kind = h.infrastructure_kind(exc)
        if kind is None:
            raise
        return 4, {**base, "status": "infrastructure_failure", "kind": kind,
                   "reason": f"{type(exc).__name__}: {str(exc)[:400]}"}
    finally:
        ctx.backend.active_episode = None
    row = h.result_row(ctx, cell, result, policy, 0)
    row.update({
        "schema_version": "tr-result-v1", "status": "done", "group": tc.group_of(design), "task": task,
        "start_metrics": {n.value: m.value for n, m in start_evidence.metrics.items()},
        "noise": {k.value: v for k, v in noise.items()},
        "loop": journal_facts(ctx.store.root, task["episode_id"]),
        "llm_stats": llm_stats(task["episode_id"]) if task["approach"] in tp.LLM_ROWS else None,
        "descriptions_served": dict(describe.served) if describe is not None else None,
        "resumed": bool(result.resumed), "loop_options": ROUND2.as_json(),
    })
    return 0, row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", required=True)
    args = ap.parse_args()
    task = next((t for t in tc.tasks() if t["episode_id"] == args.key), None)
    if task is None:
        print(json.dumps({"error": f"no planned task {args.key}"}))
        return 1
    tc.RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        code, row = run(task)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded for the queue, which decides on a retry
        code, row = 1, {"episode_id": args.key, "task": task, "status": "error",
                        "reason": f"{type(exc).__name__}: {str(exc)[:600]}", "trace": traceback.format_exc()[-2500:]}
    row["finished_at_utc"] = tc.utc()
    name = f"{args.key}.json" if code == 0 else f"{args.key}.{row['status']}.json"
    tc.jwrite(tc.RESULTS / name, row)
    if code == 0:
        for stale in tc.RESULTS.glob(f"{args.key}.*.json"):
            stale.unlink()
    print(json.dumps({"episode_id": args.key, "status": row.get("status"),
                      "reason": row.get("reason") if code != 0 else None}, default=str), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
