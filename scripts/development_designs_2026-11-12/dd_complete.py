"""Step 5a: complete the exhaustive two-step graph of the development designs (new recipe runs only where nothing is stored).

For every distinct depth-1 output of B2 (a state that differs from B2) and every recipe of the 14-recipe pool, the pair
(parent, recipe) is run once through the campaign harness's engine in run mode: a stored outcome replays from the manifest
(no tool, no cap use), anything else is a real run reserved against the caps (600 new runs, 150 job-hours). The 10-recipe
pool of T0 and A0 is a subset. Results are journalled to data/.../complete_runs.jsonl; finished pairs are skipped on resume.
Graceful stop: create data/development_designs_2026-11-12/STOP.

usage: dd_complete.py [--max-active 24] [--designs d1 d2 ...] [--plan-only]
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dd_common as dd  # noqa: E402
import dd_scope as ds  # noqa: E402
import campaign_harness as h  # noqa: E402
from orch.contract.models import MetricName, RunStatus  # noqa: E402
from orch.engine.store import _atomic_bytes, _json_bytes  # noqa: E402

JOURNAL = dd.WORK / "complete_runs.jsonl"
STATUS = dd.WORK / "complete_status.json"
STOP = dd.WORK / "STOP"
ORDER = ("vtr_mcml_v2", "t2_cordic_s", "amd_mini-isp")      # the longest runs first, so they do not form the tail


def build_tasks(ctx, designs) -> tuple[list[dict], dict]:
    g = dd.Graph(ctx.scope.manifest.body)
    amd = dd.amd_records({k: Path(v) for k, v in ctx.scope.manifest.body["roots"].items()}, {d: dd.start_id(d) for d in ds.DESIGNS})
    tasks, info = [], {}
    for d in designs:
        sid = dd.start_id(d)
        states, missing, failed = dd.reachable(g, sid, dd.RECIPES3, dd.RECIPES3)
        parents = [s for s, v in states.items() if v["steps"] == 1]
        bad_parents = [s for s in parents if amd.get((d, s)) != "pass"]
        todo = [(p, r) for (p, r) in missing if p in parents and p not in bad_parents]
        info[d] = {"depth1_parents": len(parents), "parents_without_amd_pass": len(bad_parents), "stored_depth2": len(states) - 1 - len(parents),
                   "missing_pairs": len(todo), "failed_stored_pairs": len(failed)}
        for p, r in todo:
            tasks.append({"design": d, "parent": p, "recipe": r})
    return tasks, info


def task_key(t: dict) -> str:
    return f"{t['design']}|{t['parent']}|{t['recipe']}"


class Driver:
    def __init__(self, ctx, tasks: list[dict], max_active: int) -> None:
        self.ctx, self.tasks, self.max_active = ctx, tasks, max_active
        self.lock = threading.Lock()
        self.stop = False
        self.counts = {"done": 0, "new": 0, "hits": 0, "failed": 0, "capped": 0}
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        ctx.backend.active_episode = "dd_complete"       # a constant label for the cap ledger's reservations
        self.done = self.load_done()

    @staticmethod
    def load_done() -> set[str]:
        done = set()
        if JOURNAL.is_file():
            for line in JOURNAL.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("status") in ("ok", "run_failed"):
                    done.add(row["task"])
        return done

    def one(self, t: dict) -> dict:
        row = {"task": task_key(t), **t, "started_at_utc": dd.utc()}
        try:
            state = ds.load_state(self.ctx, t["parent"])
            out = self.ctx.base_engine.execute(state, self.ctx.registry.get(t["recipe"]))
            run = out.run
            row.update(run_id=run.id, run_status=run.status.value, cache_hit=bool(run.cache_hit), wall_s=float(run.wall_s or 0.0),
                       cpu_s=float((run.cpu_user_s or 0.0) + (run.cpu_sys_s or 0.0)), status="ok" if run.status == RunStatus.OK else "run_failed")
            if run.status == RunStatus.OK and out.state is not None and out.evidence is not None:
                row["output_state"] = out.state.id
                row["metrics"] = {m.value: out.evidence.metric(m).value for m in (MetricName.WNS_NS, MetricName.PACKED_LUTS, MetricName.DYNAMIC_POWER_W)}
        except h.CapReached as exc:
            row.update(status="capped", error=str(exc))
        except Exception as exc:  # recorded, never raised: one bad pair must not stop the queue
            row.update(status="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        row["finished_at_utc"] = dd.utc()
        return row

    def commit(self, row: dict) -> None:
        with self.lock:
            with JOURNAL.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            key = {"ok": "done", "run_failed": "done", "capped": "capped", "failed": "failed"}[row["status"]]
            self.counts[key] += 1
            if row["status"] in ("ok", "run_failed"):
                self.done.add(row["task"])
                self.counts["hits" if row.get("cache_hit") else "new"] += 1

    def status(self, pending: int, active: int, note: str = "") -> None:
        _atomic_bytes(STATUS, _json_bytes({"pending": pending, "active": active, "counts": self.counts, "caps_used": self.ctx.caps.used(),
                                           "stopped": self.ctx.caps.stopped(), "note": note, "updated_at_utc": dd.utc()}))

    def run(self) -> None:
        pending = [t for t in self.tasks if task_key(t) not in self.done]
        self.status(len(pending), 0, "starting")
        active: dict = {}
        with ThreadPoolExecutor(max_workers=self.max_active) as pool:
            note = ""
            while pending or active:
                while pending and len(active) < self.max_active and not (self.stop or STOP.exists()) and not self.ctx.caps.stopped():
                    t = pending.pop(0)
                    active[pool.submit(self.one, t)] = t
                if not active:
                    note = "stopped: " + ("stop file or signal" if (self.stop or STOP.exists()) else json.dumps(self.ctx.caps.stopped()))
                    break
                finished, _ = wait(tuple(active), timeout=60, return_when=FIRST_COMPLETED)
                for f in finished:
                    active.pop(f)
                    self.commit(f.result())
                self.status(len(pending), len(active), note)
        self.status(len(pending), 0, note or "complete")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-active", type=int, default=24)
    ap.add_argument("--designs", nargs="*", default=list(ORDER))
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    ctx = ds.build("run", with_provider=False)
    tasks, info = build_tasks(ctx, args.designs)
    print(json.dumps({"tasks": len(tasks), "per_design": info, "caps_used": ctx.caps.used()}, indent=1), flush=True)
    if args.plan_only:
        return 0
    Driver(ctx, tasks, args.max_active).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
