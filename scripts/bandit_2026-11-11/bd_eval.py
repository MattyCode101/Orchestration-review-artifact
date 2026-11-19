"""Bandit (O1): simulate policies cell by cell on the union graph. No tool is started here.

The simulator is the power study's (orch.analysis.orchestrator_ceiling.Simulator via pm_common): the actual worker
planner and selector, the gate and the policy view, applied to cached outcomes. A mandate with an uncached
(state, recipe) pair censors the episode; all uncached pairs of that mandate are logged so they can be run in one batch.

usage:
  bd_eval.py --design D --tag T --spec main [--spec label=variant,k=v,...] [--baselines] [--briefs T0 A0 ...]
  bd_eval.py --all-designs --tag T ...        one detached process per design

Rows go to data/bandit_2026-11-11/runs/<tag>/cell_<design>_<brief>.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "power_readiness_2026-11-11"))
import bd_common as bc  # noqa: E402
import pr_common as pr  # noqa: E402  (also puts the power study on the path)

pm = pr.pm
from orch.contract.models import MetricName as M  # noqa: E402
from orch.contract.models import RunStatus  # noqa: E402
from orch.engine.cache import RunCache  # noqa: E402
from orch.policies.bandit import BanditPolicy, BanditSettings, Prior  # noqa: E402
from orch.policies.bandit_peek import PeekBandit  # noqa: E402   (choice rule with no in-episode learning; agreement analysis only)
from orch.policies.relevant_workers import FixedOrderPolicy  # noqa: E402
from orch.policies.repair_loop import RepairLoopPolicy  # noqa: E402
from orch.policies.training_baselines import FixedAlternationPolicy  # noqa: E402
from orch.workers.fitted import FittedWorker  # noqa: E402

BD_ROOT = bc.DATA / "engine"
RUNS = bc.DATA / "runs"
ORDERS = ("ABC", "ACB", "BAC", "BCA", "CAB", "CBA")
V2 = dict(repair_attempts=1, repair_preserve_placement=False, gentle_step=True, after_failed_repair="rollback")


def utc() -> str:
    return pr.utc()


def union_roots() -> dict:
    roots = pr.roots()
    if (BD_ROOT / "cache").is_dir():
        roots["new_bd"] = BD_ROOT
    return roots


def build_ctx(twin: bool, lenient: bool) -> dict:
    return pm.build_context(include_new=True, roots=union_roots(), amd_lenient=lenient, twin_alias=twin)


def common_settings() -> dict:
    import yaml
    return dict(yaml.safe_load((bc.PROJECT / "configs/policies/bandit.yaml").read_text(encoding="utf-8"))["settings"])


class ProbeSim(pm.TracingSimulator):
    """Logs every uncached (state, recipe) pair a mandate would need, not only the first."""

    last_misses: list
    last_miss_detail: list

    def apply(self, node, mandate):
        self.last_misses, self.last_miss_detail = [], []
        try:
            base = node.latest(mandate.base_state_id)
        except ValueError:
            return super().apply(node, mandate)
        worker = self.workers[mandate.worker_id]
        features = self._features(base) if isinstance(worker, FittedWorker) else None
        ordered = worker.plan(base.state, features, mandate, node.history.get(base.state.id, ()))
        planned = tuple(r for r in ordered if not (self.registry.get(r).replaces and mandate.settings.preserve_placement))
        if mandate.settings.max_runs is not None:
            planned = planned[:mandate.settings.max_runs]
        self.last_misses = [(base.state.id, r) for r in planned if self.oracle.get(base.state, r) is None]
        for state_id, recipe_id in self.last_misses:
            reason = self.oracle.miss_log.get((state_id, recipe_id), {}).get("reason")
            output = None
            if reason == "no_amd_pass":                 # the run exists and is cached; its output has no AMD PASS record
                recipe = self.registry.get(recipe_id)
                hit = self.oracle.index.key_to_run.get(RunCache.key(state_id, recipe.id, recipe.version))
                if hit is not None:
                    output = self.oracle.store.load_outcome(hit[1]).state.id
            self.last_miss_detail.append({"state": state_id, "recipe": recipe_id, "reason": reason, "output_state": output})
        return super().apply(node, mandate)


class Recorder:
    """Wraps a policy and keeps what it was shown and what it chose."""

    def __init__(self, policy, keep_views: bool = False) -> None:
        self.policy, self.keep_views = policy, keep_views
        self.id, self.version = policy.id, policy.version
        self.log: list[dict] = []
        self.views: list = []

    def decide(self, view):
        mandate = self.policy.decide(view)
        entry = {"call": len(view.calls), "budget_left": view.budget_left, "stop": mandate is None}
        if mandate is not None:
            entry.update(worker=mandate.worker_id, preserve=mandate.settings.preserve_placement,
                         attempts=mandate.settings.max_runs, base_id=mandate.base_state_id,
                         best_id=(view.best_valid or view.start).state_id, current_id=view.current.state_id)
            try:
                entry["phase"] = json.loads(mandate.context_ref).get("phase")
            except (ValueError, TypeError):
                pass
            if isinstance(self.policy, BanditPolicy):
                entry["arm"] = json.loads(mandate.context_ref)["arm"]
        if isinstance(self.policy, BanditPolicy):                    # also when it stops: why did nothing reach the threshold?
            rows = sorted(self.policy.scored(view), key=lambda r: -r["ucb"])[:3]
            entry["top"] = [{k: r[k] for k in ("key", "n", "mean", "bonus", "ucb")} for r in rows]
        self.log.append(entry)
        if self.keep_views:
            self.views.append((view, mandate))
        return mandate


def policies_for(design: str, brief_id: str, specs: list[dict], baselines: bool) -> dict:
    out: dict = {}
    for spec in specs:
        common = dict(common_settings(), **spec["overrides"])
        settings = BanditSettings.for_variant(spec["variant"], **common)
        path = bc.DATA / spec["priors"] / f"{spec['variant']}_{brief_id}_{design}.json"
        out[spec["label"]] = BanditPolicy(prior=Prior.load(path), settings=settings)
    if baselines:
        out["v2"] = RepairLoopPolicy(**V2)
        if brief_id in ("L2R", "P0"):
            for order in ORDERS:
                out[f"fixed_{order}"] = FixedOrderPolicy(order=order)
        else:
            out["fixed_AB"] = FixedAlternationPolicy(first="A")
            out["fixed_BA"] = FixedAlternationPolicy(first="B")
    return out


def row_of(brief, start_ev, res, store, extra: dict | None = None) -> dict:
    final = res.final
    wall, runs = 0.0, 0
    for attempts in res.node.history.values():
        for step in attempts:
            if step.outcome.run.status != RunStatus.TRANSIENT:
                wall += float(step.outcome.run.wall_s or 0.0)
                runs += 1
    visits = {v.state.id: v for v in res.node.visits}              # the latest visit of each state id

    def metrics3(visit):
        return None if visit is None else {m.value: visit.evidence.metric(m).value
                                           for m in (M.WNS_NS, M.PACKED_LUTS, M.DYNAMIC_POWER_W)}

    row = {
        "final_state_id": final.state.id,
        "final_metrics": {m.value: final.evidence.metric(m).value for m in M if final.evidence.metric(m) is not None},
        "improvement": pm._improvement(brief, start_ev, final.evidence),
        "used": res.used, "runs_executed": runs, "wall_s": wall, "censored": bool(res.censored), "reason": res.reason,
        "calls": [{"worker": t.worker, "base_id": t.base_id, "preserve": t.preserve_placement, "attempts": t.attempts,
                   "selected_id": t.selected_id, "recipes": list(t.recipes),
                   "base_metrics": metrics3(visits.get(t.base_id)), "selected_metrics": metrics3(visits.get(t.selected_id)),
                   "selected_brief_valid": (None if visits.get(t.selected_id) is None
                                            else bool(visits[t.selected_id].verdict.brief_valid))}
                  for t in res.calls],
        "final_brief_valid": bool(final.verdict.brief_valid),
    }
    row.update(extra or {})
    return row


def agreement(design: str, brief_id: str, brief, sim, v2_rec: Recorder, spec: dict) -> list[dict]:
    """Ask a bandit variant (this cell's leave-one-design-out prior, no in-episode learning) at every v2 decision."""

    common = dict(common_settings(), **spec["overrides"])
    settings = BanditSettings.for_variant(spec["variant"], **common)
    peek = PeekBandit(prior=Prior.load(bc.DATA / spec["priors"] / f"{spec['variant']}_{brief_id}_{design}.json"),
                      settings=settings)
    out = []
    for i, (view, v2_mandate) in enumerate(v2_rec.views):
        best_id = (view.best_valid or view.start).state_id

        def kind(base_id: str) -> str:
            return "best" if base_id == best_id else ("latest" if base_id == view.current.state_id else "other")

        row = {"design": design, "brief": brief_id, "i": i, "budget_left": view.budget_left}
        if v2_mandate is None:
            row["v2"] = {"stop": True}
        else:
            ctx = json.loads(v2_mandate.context_ref)
            row["v2"] = {"worker": v2_mandate.worker_id, "preserve": v2_mandate.settings.preserve_placement,
                         "attempts": v2_mandate.settings.max_runs, "base": kind(v2_mandate.base_state_id),
                         "phase": ctx.get("phase")}
        mine = peek.decide(view)
        if mine is None:
            row["bandit"] = {"stop": True}
        else:
            ctx = json.loads(mine.context_ref)
            row["bandit"] = {"worker": mine.worker_id, "preserve": mine.settings.preserve_placement,
                             "attempts": mine.settings.max_runs, "base": ctx["arm"].split("|")[3], "arm": ctx["arm"]}
        out.append(row)
    return out


def run_cell(ctx: dict, design: str, brief_id: str, specs: list[dict], baselines: bool, tag: str) -> Path:
    brief = bc.make_brief(brief_id, design)
    keys = ("timing", "area", "power") if brief_id in ("L2R", "P0") else ("timing", "area")
    workers = {k: ctx["workers"][k] for k in keys}
    state, start_ev = ctx["starts"][design]
    noise = bc.design_noise()[design]
    sim = ProbeSim(brief=brief, start_state=state, start_evidence=start_ev, workers=workers, registry=ctx["registry"],
                   oracle=ctx["oracle"], provider=ctx["provider"], noise=noise)
    rows: dict[str, dict] = {}
    decisions: dict[str, list[dict]] = {}
    for name, policy in policies_for(design, brief_id, specs, baselines).items():
        rec = Recorder(policy, keep_views=(name == "v2"))
        res = pm.simulate_policy(sim, rec, episode_id=f"bd_{design}_{brief_id}_{name}")
        extra = {"decisions": rec.log}
        if res.censored:
            extra["pending"] = [{"design": design, **d} for d in sim.last_miss_detail]
        rows[name] = row_of(brief, start_ev, res, ctx["store"], extra)
        if name == "v2":
            for spec in specs:
                if spec["label"] in ("main", "main_tau0"):
                    decisions[spec["label"]] = agreement(design, brief_id, brief, sim, rec, spec)
    out = {
        "schema": "bd-cell-v1", "design": design, "brief": brief_id, "tag": tag, "generated_at_utc": utc(),
        "start": {"state_id": state.id, "metrics": {m.value: start_ev.metric(m).value for m in M if start_ev.metric(m) is not None}},
        "noise": {m.value: v for m, v in noise.items()}, "union_digest": ctx["index"].digest(),
        "union_pointers": len(ctx["index"].key_to_run), "roots": sorted(ctx["index"].roots),
        "specs": specs, "rows": rows, "agreement": decisions,
        "unverified_states_accepted": sorted(s for (_d, s) in ctx["amd"].unverified_accepted),
        "alias_edges_used": len(ctx["oracle"].alias_used),
    }
    folder = RUNS / tag
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"cell_{design}_{brief_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def parse_spec(text: str) -> dict:
    """label=variant,k=v,... or just a variant name. priors=<dir> picks another prior folder."""

    label, _, rest = text.partition("=")
    if not rest:
        label, rest = text, text
    parts = rest.split(",")
    variant, overrides, priors = parts[0], {}, "priors"
    for item in parts[1:]:
        key, _, value = item.partition("=")
        if key == "priors":
            priors = value
        else:
            overrides[key] = float(value)
    return {"label": label, "variant": variant, "overrides": overrides, "priors": priors}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--design")
    mode.add_argument("--all-designs", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--spec", action="append", default=[])
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--briefs", nargs="*", default=list(bc.BRIEFS))
    ap.add_argument("--no-twin", action="store_true")
    ap.add_argument("--strict-amd", action="store_true")
    args = ap.parse_args()
    specs = [parse_spec(s) for s in (args.spec or ["main"])]
    if args.all_designs:
        folder = RUNS / args.tag
        folder.mkdir(parents=True, exist_ok=True)
        procs = {}
        for design in bc.DESIGNS:
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--design", design, "--tag", args.tag, "--briefs", *args.briefs]
            for s in args.spec or ["main"]:
                cmd += ["--spec", s]
            if args.baselines:
                cmd.append("--baselines")
            if args.no_twin:
                cmd.append("--no-twin")
            if args.strict_amd:
                cmd.append("--strict-amd")
            log = (folder / f"design_{design}.log").open("ab")
            procs[design] = subprocess.Popen(cmd, cwd=bc.PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        while procs:
            time.sleep(5)
            for design, proc in list(procs.items()):
                if proc.poll() is not None:
                    if proc.returncode != 0:
                        (folder / f"design_{design}.FAILED").write_text(str(proc.returncode))
                    del procs[design]
        return 0
    t0 = time.monotonic()
    ctx = build_ctx(twin=not args.no_twin, lenient=not args.strict_amd)
    print(f"context built in {time.monotonic() - t0:.0f}s, {len(ctx['index'].key_to_run)} pointers", flush=True)
    for brief_id in args.briefs:
        t1 = time.monotonic()
        path = run_cell(ctx, args.design, brief_id, specs, args.baselines, args.tag)
        print(f"{args.design} {brief_id}: {path.name} in {time.monotonic() - t1:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
