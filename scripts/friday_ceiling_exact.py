"""Exact orchestrator ceiling over the approved training cache (training only).

Replaces the beam lower bound. For one design and brief it explores every
mandate sequence (worker x visited base x placement x 1..3 attempts) within
the largest budget, merging only nodes with identical accessible state (the
existing conservative signature) and discarding nothing. A mandate is pruned
only if one of its planned recipes has no checked cached outcome (cache
pointer in the frozen snapshot, status/hash checks, current meter, clean
properties and an exact-DCP AMD PASS record); coverage counts those prunes.
One exploration to the largest budget gives the ceilings for every smaller
budget (the explored set is prefix-closed). No tool job is ever launched.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]

from orch.analysis import orchestrator_ceiling as old  # noqa: E402
from orch.analysis.orchestrator_ceiling import (  # noqa: E402
    Simulator, _best_key, _improvement, _policy, _signature, simulate_policy,
)
from orch.baselines.verification import VerificationFailure, VerificationLedger  # noqa: E402
from orch.config import load_noise, load_recipe_sets, load_worker_card  # noqa: E402
from orch.contract.brief_text import load_frozen_brief  # noqa: E402
from orch.contract.models import (  # noqa: E402
    DesignState, Evidence, Mandate, RunOutcome, RunStatus, WorkerSettings,
)
from orch.engine.cache import RunCache  # noqa: E402
from orch.engine.store import StateStore, _sha256_file  # noqa: E402
from orch.recipes.registry import load_registry  # noqa: E402
from orch.runtime import make_real_runtime  # noqa: E402
from orch.workers.base import GenericWorker  # noqa: E402
from orch.workers.fitted import FittedWorker  # noqa: E402
from orch.workers.telemetry import SavedTelemetryProvider  # noqa: E402
from scripts import friday_analysis as fa  # noqa: E402
from scripts import thursday_night_pilot as pilot  # noqa: E402

ENGINE_ROOT = fa.V4
WORK = PROJECT / "data/scratch_friday_2026-11-06/ceiling_exact"
SNAPSHOT = WORK / "cache_snapshot.json"
SCRIPT = Path(__file__).resolve()
BUDGETS = (4, 6, 8, 10)
BRIEFS_DEFAULT = ("T0", "A0", "T", "A")
POLICY_NAMES = ("repair_loop", "single_specialist", "fixed_AB", "fixed_BA")
REPORT = PROJECT / "notes/orchestrator_ceiling_training.md"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# Memory control. Gate verdicts depend only on (brief, start, parent, candidate,
# guards, noise, status), so one shared verdict object serves every node that
# repeats the same attempt; and a node never needs its policy-view tuples.
_real_judge = old.judge
_JUDGE_MEMO: dict[tuple, Any] = {}
_BRIEF_REFS: dict[int, tuple[Any, str]] = {}


def _brief_key(brief) -> str:
    entry = _BRIEF_REFS.get(id(brief))
    if entry is None or entry[0] is not brief:
        entry = (brief, brief.fingerprint())
        _BRIEF_REFS[id(brief)] = entry
    return entry[1]


def _memo_judge(brief, start, parent, candidate, **kwargs):
    guards = kwargs.get("guards", ())
    noise = kwargs.get("noise")
    key = (
        _brief_key(brief), start.id, parent.id, None if candidate is None else candidate.id,
        tuple(g.id for g in guards), tuple(sorted((k.value, v) for k, v in (noise or {}).items())),
        kwargs.get("run_status", "default"),
    )
    value = _JUDGE_MEMO.get(key)
    if value is None:
        value = _real_judge(brief, start, parent, candidate, **kwargs)
        _JUDGE_MEMO[key] = value
    return value


old.judge = _memo_judge


def _rss_gb() -> float:
    with open("/proc/self/statm", encoding="ascii") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def take_snapshot() -> dict[str, Any]:
    """Freeze the set of terminal cache pointers once, so every cell sees the same cache."""

    WORK.mkdir(parents=True, exist_ok=True)
    if SNAPSHOT.is_file():
        return json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    keys = sorted(p.stem for p in (ENGINE_ROOT / "cache").glob("*.json"))
    record = {
        "schema_version": "friday-ceiling-cache-snapshot-v1", "taken_at_utc": utc(),
        "cache_root": str(ENGINE_ROOT / "cache"), "pointers": len(keys),
        "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(), "keys": keys,
    }
    tmp = SNAPSHOT.with_suffix(".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    tmp.replace(SNAPSHOT)
    return record


class LiveCacheOracle:
    """Checked outcomes from the approved v4 training cache, restricted to a snapshot."""

    def __init__(self, registry, meter_version: str, ledger: VerificationLedger,
                 keys: set[str]) -> None:
        self.registry, self.meter_version, self.ledger = registry, meter_version, ledger
        self.keys = keys
        self.store = StateStore(ENGINE_ROOT)
        self.cache = RunCache(ENGINE_ROOT)
        self.memo: dict[tuple[str, str], RunOutcome | None] = {}
        self.miss_reasons: Counter[str] = Counter()
        self._starts: dict[str, Evidence] = {}

    def _start(self, design: str) -> Evidence:
        """The design's training B2 start Evidence (current meter): the reference for the property rule."""

        if design not in self._starts:
            state_id = pilot._training_inputs()[design]
            key = hashlib.sha256(self.meter_version.encode("utf-8")).hexdigest()
            path = ENGINE_ROOT / "measurements" / state_id / key / "evidence.json"
            self._starts[design] = Evidence.model_validate_json(path.read_bytes())
        return self._starts[design]

    def _miss(self, ident, reason: str):
        self.memo[ident] = None
        self.miss_reasons[reason] += 1
        return None

    def get(self, base: DesignState, recipe_id: str) -> RunOutcome | None:
        ident = base.id, recipe_id
        if ident in self.memo:
            return self.memo[ident]
        recipe = self.registry.get(recipe_id)
        key = RunCache.key(base.id, recipe.id, recipe.version)
        if key not in self.keys:
            return self._miss(ident, "not_in_snapshot")
        run_id = self.cache.get(key)
        if run_id is None:
            return self._miss(ident, "no_terminal_pointer")
        outcome = self.store.load_outcome(run_id)
        run = outcome.run
        if (
            run.recipe_id != recipe.id or run.recipe_version != recipe.version
            or run.input_sha256_before != base.id or run.input_sha256_after != base.id
            or run.fake
        ):
            return self._miss(ident, "identity_mismatch")
        if run.status == RunStatus.OK:
            if outcome.state is None or outcome.evidence is None:
                return self._miss(ident, "ok_without_output")
            if outcome.evidence.meter_version != self.meter_version:
                return self._miss(ident, "stale_meter")
            try:
                pilot.require_clean_properties(outcome.evidence, self._start(base.design))
                pilot.require_amd_pass(
                    self.ledger, base.design, outcome.state.id, allow_new=False,
                )
            except (pilot.PilotSafetyStop, VerificationFailure):
                return self._miss(ident, "no_checked_amd_pass_or_unclean")
        elif run.status == RunStatus.TRANSIENT:
            return self._miss(ident, "transient")
        self.memo[ident] = outcome
        return outcome


def build_context() -> dict[str, Any]:
    runtime = make_real_runtime(
        ENGINE_ROOT, cache_enabled=True, max_slots=pilot.MAX_SLOTS,
        load_ceiling_1m=pilot.LOAD_CEILING, min_available_memory_bytes=pilot.MEMORY_FLOOR,
        latch_smoke_status=pilot.SMOKE,
    )
    registry = load_registry(PROJECT)
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    workers: dict[str, Any] = {
        wid: FittedWorker(
            load_worker_card(PROJECT / "configs/workers" / f"{wid}.yaml"),
            sets.worker_sets[wid], registry,
        )
        for wid in ("timing", "area")
    }
    if any(w.model_label != "FINAL" for w in workers.values()):
        raise ValueError("ceiling requires both published FINAL workers")
    power = load_recipe_sets(PROJECT / "configs/power_recipe_set.yaml")
    workers["power"] = GenericWorker(
        load_worker_card(PROJECT / "configs/workers/power.yaml"), power.worker_sets["power"],
    )
    plan = json.loads((ENGINE_ROOT / "plan.json").read_text(encoding="utf-8"))
    ledger = VerificationLedger(
        store=runtime.store, runner=runtime.runner, validator=runtime.validator,
        references=plan["training_inputs"],
    )
    snapshot = take_snapshot()
    oracle = LiveCacheOracle(registry, runtime.meter.version, ledger, set(snapshot["keys"]))
    starts_evidence = fa.start_evidence()
    starts = {}
    for design in fa.DESIGNS:
        sid = plan["training_inputs"][design]
        state = DesignState.model_validate_json(
            (ENGINE_ROOT / "states" / sid / "state.json").read_bytes()
        )
        if state.id != sid or state.design != design or state.parent_id is not None or state.fake:
            raise ValueError(f"B2 identity differs: {design}")
        starts[design] = state, starts_evidence[design]
    return {
        "registry": registry, "workers": workers, "oracle": oracle,
        "provider": SavedTelemetryProvider(ENGINE_ROOT), "starts": starts,
        "snapshot": snapshot, "meter_version": runtime.meter.version,
        "model_hashes": {k: w.model_sha256 for k, w in workers.items() if isinstance(w, FittedWorker)},
    }


def exact_search(sim: Simulator, budgets: tuple[int, ...], *, max_nodes: int,
                 time_limit_s: float, rss_limit_gb: float = 24.0) -> dict[str, Any]:
    """Explore every reachable node up to the largest budget; discard nothing.

    A node/time/memory guard only stops EXPANSION (the result is then flagged as
    a lower bound); it never discards a node it has already generated.
    """

    top = max(budgets)
    started = time.monotonic()
    layers: dict[int, dict[tuple, Any]] = defaultdict(dict)
    first = sim.initial()
    layers[0][_signature(first)] = first
    best_at: dict[int, Any] = {}
    proposed = pruned = no_recipe = merged = explored = 0
    reasons: Counter[str] = Counter()
    per_cost = {}
    complete = True
    stop_reason = "complete"
    for cost in range(top + 1):
        layer = layers.pop(cost, {})
        per_cost[cost] = len(layer)
        for node in layer.values():
            explored += 1
            champion = best_at.get(cost)
            if champion is None or _best_key(sim.brief, node.final(sim.brief)) < _best_key(
                sim.brief, champion.final(sim.brief)
            ):
                best_at[cost] = node
            if cost >= top:
                continue
            if complete:
                if explored > max_nodes:
                    complete, stop_reason = False, "node limit"
                elif time.monotonic() - started > time_limit_s:
                    complete, stop_reason = False, "time limit"
                elif explored % 500 == 0 and _rss_gb() > rss_limit_gb:
                    complete, stop_reason = False, "memory guard"
            if not complete:
                continue
            base_ids = tuple(dict.fromkeys(v.state.id for v in node.visits))
            for base_id in base_ids:
                for worker_id in sorted(sim.workers):
                    for preserve in (False, True):
                        for attempts in (1, 2, 3):
                            if cost + attempts > top:
                                continue
                            mandate = Mandate(
                                id=f"x_{cost}_{len(node.path)}_{worker_id}_{base_id[:10]}_{int(preserve)}_{attempts}",
                                worker_id=worker_id, base_state_id=base_id,
                                settings=WorkerSettings(preserve_placement=preserve, max_runs=attempts),
                                brief_hash=sim.brief.fingerprint(),
                            )
                            proposed += 1
                            successor, reason = sim.apply(node, mandate)
                            if successor is not None:
                                successor = dataclasses.replace(successor, calls=(), steps=())
                            if successor is None:
                                if reason == "uncached":
                                    pruned += 1
                                elif reason == "no_recipe":
                                    no_recipe += 1
                                else:
                                    raise ValueError(f"unexpected mandate result: {reason}")
                                reasons[reason] += 1
                                continue
                            signature = _signature(successor)
                            bucket = layers[successor.used]
                            if signature in bucket:
                                merged += 1
                            else:
                                bucket[signature] = successor
    result_by_budget = {}
    running = None
    for cost in range(top + 1):
        node = best_at.get(cost)
        if node is not None and (
            running is None or _best_key(sim.brief, node.final(sim.brief))
            < _best_key(sim.brief, running.final(sim.brief))
        ):
            running = node
        if cost in budgets:
            result_by_budget[cost] = running
    return {
        "best": result_by_budget, "explored": explored, "proposed": proposed,
        "cache_pruned": pruned, "no_recipe": no_recipe, "merged_identical": merged,
        "nodes_per_cost": per_cost, "complete": complete, "stop_reason": stop_reason,
        "peak_rss_gb": round(_rss_gb(), 1), "elapsed_s": round(time.monotonic() - started, 1),
    }


def run_cell(design: str, brief_id: str, *, max_nodes: int, time_limit_s: float,
             rss_limit_gb: float = 24.0) -> Path:
    ctx = build_context()
    state, evidence = ctx["starts"][design]
    noise = load_noise(PROJECT / "configs/noise.yaml").for_start(design, evidence)
    fixed = load_frozen_brief(brief_id)
    starts, best_pool = fa.start_metrics(), fa.best_known()
    cell_workers = ctx["workers"]
    if brief_id != "P0":
        cell_workers = {k: v for k, v in cell_workers.items() if k != "power"}

    def sim_for(budget: int) -> Simulator:
        return Simulator(
            brief=fixed.model_copy(update={"budget_runs": budget}), start_state=state,
            start_evidence=evidence, workers=cell_workers, registry=ctx["registry"],
            oracle=ctx["oracle"], provider=ctx["provider"], noise=noise,
        )

    search = exact_search(sim_for(max(BUDGETS)), BUDGETS, max_nodes=max_nodes,
                          time_limit_s=time_limit_s, rss_limit_gb=rss_limit_gb)
    metric, direction = fa.OBJECTIVE[brief_id]
    cells = {}
    for budget in BUDGETS:
        node = search["best"][budget]
        brief_b = fixed.model_copy(update={"budget_runs": budget})
        final = node.final(brief_b)
        value = final.evidence.metric(old.MetricName(metric)).value
        policies = {}
        sim_b = sim_for(budget)
        for name in POLICY_NAMES:
            if brief_id == "P0":
                continue
            res = simulate_policy(sim_b, _policy(name), episode_id=f"cx_{design}_{brief_id}_{budget}_{name}")
            pv = res.final.evidence.metric(old.MetricName(metric)).value
            policies[name] = {"value": pv, "used": res.used, "censored": res.censored}
        share = None
        if (design, brief_id) in best_pool:
            share = fa.share(brief_id, design, value, starts, best_pool)
        cells[str(budget)] = {
            "final_state_id": final.state.id, "value": value,
            "improvement": _improvement(brief_b, evidence, final.evidence),
            "share_of_pool_best": share, "used": node.used,
            "path": [t.label() for t in node.path], "policies": policies,
        }
    out = {
        "schema_version": "friday-ceiling-exact-cell-v1", "design": design, "brief": brief_id,
        "generated_at_utc": utc(), "snapshot_keys_sha256": ctx["snapshot"]["keys_sha256"],
        "snapshot_pointers": ctx["snapshot"]["pointers"], "meter_version": ctx["meter_version"],
        "model_hashes": ctx["model_hashes"], "driver_sha256": _sha256_file(SCRIPT),
        "search": {k: v for k, v in search.items() if k != "best"},
        "oracle_miss_reasons": dict(ctx["oracle"].miss_reasons), "budgets": cells,
    }
    path = WORK / f"cell_{design}_{brief_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def run_all(briefs: tuple[str, ...], *, workers: int, max_nodes: int, time_limit_s: float,
            rss_limit_gb: float = 24.0) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    take_snapshot()
    todo = [(d, b) for b in briefs for d in fa.DESIGNS
            if not (WORK / f"cell_{d}_{b}.json").is_file()]
    live: dict[tuple[str, str], subprocess.Popen] = {}
    status = WORK / "run_all_status.json"
    while todo or live:
        while todo and len(live) < workers:
            cell = todo.pop(0)
            log = (WORK / f"cell_{cell[0]}_{cell[1]}.log").open("ab")
            live[cell] = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--cell", cell[0], cell[1],
                 "--max-nodes", str(max_nodes), "--time-limit", str(time_limit_s),
                 "--rss-limit-gb", str(rss_limit_gb)],
                cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, close_fds=True,
            )
        time.sleep(10)
        for cell, proc in list(live.items()):
            if proc.poll() is not None:
                if proc.returncode != 0:
                    (WORK / f"cell_{cell[0]}_{cell[1]}.FAILED").write_text(str(proc.returncode))
                del live[cell]
        status.write_text(json.dumps({"todo": len(todo), "running": sorted("|".join(c) for c in live),
                                      "updated_at_utc": utc()}))
    verify_and_report(briefs)


def verify_and_report(briefs: tuple[str, ...]) -> None:
    sys.path.insert(0, str(PROJECT / "scripts"))
    import friday_ceiling_report as report  # noqa: E402
    report.main(briefs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cell", nargs=2, metavar=("DESIGN", "BRIEF"))
    modes.add_argument("--run-all", action="store_true")
    modes.add_argument("--report", action="store_true")
    parser.add_argument("--briefs", nargs="+", default=list(BRIEFS_DEFAULT))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=4_000_000)
    parser.add_argument("--time-limit", type=float, default=5400.0)
    parser.add_argument("--rss-limit-gb", type=float, default=24.0)
    args = parser.parse_args()
    if args.cell:
        print(run_cell(args.cell[0], args.cell[1], max_nodes=args.max_nodes,
                       time_limit_s=args.time_limit, rss_limit_gb=args.rss_limit_gb))
    elif args.run_all:
        run_all(tuple(args.briefs), workers=args.workers, max_nodes=args.max_nodes,
                time_limit_s=args.time_limit, rss_limit_gb=args.rss_limit_gb)
    else:
        verify_and_report(tuple(args.briefs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
