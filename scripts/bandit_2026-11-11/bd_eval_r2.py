"""Bandit round 2 (O1): simulate the round-2 policies cell by cell on the union graph. No tool is started here.

Same simulator, same union graph, same cells as round 1 (bd_eval.py); two differences:
  - the policies are the round-2 labels of configs/policies/bandit_r2.yaml (learned, obs_only, feat_only, r1_stop0);
  - for a label with observe_empty the contract is run as the authors decided: a mandate that finds no recipe is an observation. A
    zero-run call with nothing selected joins the record the policy is shown, the budget is untouched and the episode goes on.
    (orch/episode.py, which is frozen, still ends an episode there; this harness is where the decision is applied.)
A mandate with an uncached (state, recipe) pair censors the episode; all uncached pairs of that mandate are logged so they can
be run in one batch (bd_pending.py, bd_runner.py, unchanged).

usage:
  bd_eval_r2.py --design D --tag T [--labels learned ...] [--baselines] [--briefs T0 A0 ...] [--held-out none]
  bd_eval_r2.py --all-designs --tag T ...        one detached process per design
Rows go to data/bandit_2026-11-11/runs/<tag>/cell_<design>_<brief>.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bd_common as bc  # noqa: E402
import bd_eval as be  # noqa: E402  (also puts the power study and the power-readiness scripts on the path)

pm = be.pm
from orch.analysis.orchestrator_ceiling import PolicyResult  # noqa: E402
from orch.contract.models import MetricName as M  # noqa: E402
from orch.contract.views import CallSummary  # noqa: E402
from orch.policies import bandit_r2 as r2  # noqa: E402

CONFIG = bc.PROJECT / "configs/policies/bandit_r2.yaml"
PRIORS = "priors_r2"
LABELS = ("learned", "obs_only", "feat_only", "r1_stop0")
MAX_EMPTY = 64


def declared() -> dict:
    import yaml
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def label_settings(label: str) -> r2.R2Settings:
    d = declared()
    return r2.R2Settings.for_variant("main", **{**d["settings"], **d["labels"][label]})


def prior_path(settings: r2.R2Settings, brief_id: str, held_out: str, priors_dir: str = PRIORS) -> Path:
    name = f"{'feat' if settings.repair_feature else 'nofeat'}_{'drop' if settings.observe_empty else 'keep'}_{brief_id}_{held_out}.json"
    return bc.DATA / priors_dir / name


def make_policy(label: str, brief_id: str, held_out: str, *, peek: bool = False) -> r2.BanditPolicyR2:
    settings = label_settings(label)
    cls = r2.PeekBanditR2 if peek else r2.BanditPolicyR2
    return cls(prior=r2.Prior2.load(prior_path(settings, brief_id, held_out)), settings=settings)


def policies_for(brief_id: str, labels: list[str], baselines: bool, held_out: str) -> dict:
    out: dict = {label: make_policy(label, brief_id, held_out) for label in labels}
    if baselines:
        out.update(be.policies_for("", brief_id, [], True))
    return out


# ------------------------------------------------------------------------------------ the contract change
def with_empty_call(node, mandate):
    """The record after a mandate that found no recipe: one more call, nothing charged, nothing selected."""

    call = CallSummary(
        mandate_id=mandate.id, worker_id=mandate.worker_id, base_state_id=mandate.base_state_id, settings=mandate.settings,
        selected_state_id=None, selected_verdict_status=None, recipe_runs=0, context_ref=mandate.context_ref,
    )
    return dataclasses.replace(node, calls=(*node.calls, call))


def simulate_observing(sim, policy, *, episode_id: str, observe: bool):
    """pm.simulate_policy, except that with observe an empty call is shown to the policy and the episode goes on.

    Returns (PolicyResult, empties). Without observe this is exactly the round-1 loop and empties is []. A policy that asks
    again for a worker and scope an empty call has removed is a bug in the policy, and raises."""

    if not observe:
        return pm.simulate_policy(sim, policy, episode_id=episode_id), []
    node = sim.initial()
    empties: list[dict] = []
    gone: set[tuple[str, bool, str]] = set()
    for _ in range(sim.brief.budget_runs + 1 + MAX_EMPTY):
        mandate = policy.decide(sim.view(node, episode_id))
        if mandate is None:
            return PolicyResult(node.final(sim.brief), node.path, node.used, False, "stop", node), empties
        successor, reason = sim.apply(node, mandate)
        if successor is None:
            if reason == "no_recipe":
                key = (mandate.worker_id, bool(mandate.settings.preserve_placement), mandate.base_state_id)
                if key in gone:
                    raise ValueError("the policy asked again for a worker and scope that an empty call had removed")
                gone.add(key)
                empties.append({"call": len(node.calls), "worker": key[0], "preserve": key[1], "base_id": key[2],
                                "attempts": mandate.settings.max_runs, "budget_left": sim.brief.budget_runs - node.used})
                node = with_empty_call(node, mandate)
                continue
            return PolicyResult(node.final(sim.brief), node.path, node.used, reason == "uncached", reason, node), empties
        node = successor
    raise ValueError("policy exceeded the maximum number of calls")


class RecorderR2(be.Recorder):
    """The round-1 recorder, plus for every decision the repair flag the policy saw and the kind of base it chose."""

    def decide(self, view):
        mandate = super().decide(view)
        entry = self.log[-1]
        entry["repairable"] = r2.repair_pending(view)
        if mandate is not None:
            entry["base_kind"] = ("best" if mandate.base_state_id == entry["best_id"]
                                  else ("latest" if mandate.base_state_id == entry["current_id"] else "other"))
        return mandate


# ------------------------------------------------------------------------------------ agreement with v2
def agreement(design: str, brief_id: str, v2_rec: be.Recorder, label: str, held_out: str) -> list[dict]:
    """Ask a label's bandit (this cell's prior, no in-episode learning) at every decision view of v2."""

    peek = make_policy(label, brief_id, held_out, peek=True)
    out = []
    for i, (view, v2_mandate) in enumerate(v2_rec.views):
        best_id = (view.best_valid or view.start).state_id

        def kind(base_id: str) -> str:
            return "best" if base_id == best_id else ("latest" if base_id == view.current.state_id else "other")

        row = {"design": design, "brief": brief_id, "i": i, "budget_left": view.budget_left,
               "repairable": r2.repair_pending(view)}
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


# ------------------------------------------------------------------------------------------------ cells
def run_cell(ctx: dict, design: str, brief_id: str, labels: list[str], baselines: bool, tag: str, held_out: str) -> Path:
    brief = bc.make_brief(brief_id, design)
    keys = ("timing", "area", "power") if brief_id in ("L2R", "P0") else ("timing", "area")
    workers = {k: ctx["workers"][k] for k in keys}
    state, start_ev = ctx["starts"][design]
    noise = bc.design_noise()[design]
    sim = be.ProbeSim(brief=brief, start_state=state, start_evidence=start_ev, workers=workers, registry=ctx["registry"],
                      oracle=ctx["oracle"], provider=ctx["provider"], noise=noise)
    rows: dict[str, dict] = {}
    decisions: dict[str, list[dict]] = {}
    for name, policy in policies_for(brief_id, labels, baselines, held_out).items():
        rec = RecorderR2(policy, keep_views=(name == "v2"))
        observe = isinstance(policy, r2.BanditPolicyR2) and policy.settings.observe_empty
        res, empties = simulate_observing(sim, rec, episode_id=f"bd2_{design}_{brief_id}_{name}", observe=observe)
        extra = {"decisions": rec.log, "empty_calls": empties}
        if res.censored:
            extra["pending"] = [{"design": design, **d} for d in sim.last_miss_detail]
        rows[name] = be.row_of(brief, start_ev, res, ctx["store"], extra)
        if name == "v2":
            for label in labels:
                decisions[label] = agreement(design, brief_id, rec, label, held_out)
    out = {
        "schema": "bd-cell-r2-v1", "design": design, "brief": brief_id, "tag": tag, "generated_at_utc": be.utc(),
        "held_out": held_out, "start": {"state_id": state.id, "metrics": {m.value: start_ev.metric(m).value for m in M
                                                                          if start_ev.metric(m) is not None}},
        "noise": {m.value: v for m, v in noise.items()}, "union_digest": ctx["index"].digest(),
        "union_pointers": len(ctx["index"].key_to_run), "roots": sorted(ctx["index"].roots),
        "labels": labels, "rows": rows, "agreement": decisions,
        "unverified_states_accepted": sorted(s for (_d, s) in ctx["amd"].unverified_accepted),
        "alias_edges_used": len(ctx["oracle"].alias_used),
    }
    folder = be.RUNS / tag
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"cell_{design}_{brief_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--design")
    mode.add_argument("--all-designs", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--labels", nargs="*", default=list(LABELS))
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--briefs", nargs="*", default=list(bc.BRIEFS))
    ap.add_argument("--designs", nargs="*", default=list(bc.DESIGNS), help="with --all-designs: the designs to run")
    ap.add_argument("--held-out", default=None, help="prior fold; default: the design itself (leave-one-design-out)")
    ap.add_argument("--no-twin", action="store_true")
    ap.add_argument("--strict-amd", action="store_true")
    args = ap.parse_args()
    if args.all_designs:
        folder = be.RUNS / args.tag
        folder.mkdir(parents=True, exist_ok=True)
        procs = {}
        for design in args.designs:
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--design", design, "--tag", args.tag,
                   "--briefs", *args.briefs, "--labels", *args.labels]
            if args.baselines:
                cmd.append("--baselines")
            if args.held_out:
                cmd += ["--held-out", args.held_out]
            if args.no_twin:
                cmd.append("--no-twin")
            if args.strict_amd:
                cmd.append("--strict-amd")
            log = (folder / f"design_{design}.log").open("ab")
            procs[design] = subprocess.Popen(cmd, cwd=bc.PROJECT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                             start_new_session=True)
        while procs:
            time.sleep(5)
            for design, proc in list(procs.items()):
                if proc.poll() is not None:
                    if proc.returncode != 0:
                        (folder / f"design_{design}.FAILED").write_text(str(proc.returncode))
                    del procs[design]
        return 0
    t0 = time.monotonic()
    ctx = be.build_ctx(twin=not args.no_twin, lenient=not args.strict_amd)
    print(f"context built in {time.monotonic() - t0:.0f}s, {len(ctx['index'].key_to_run)} pointers", flush=True)
    for brief_id in args.briefs:
        t1 = time.monotonic()
        path = run_cell(ctx, args.design, brief_id, args.labels, args.baselines, args.tag, args.held_out or args.design)
        print(f"{args.design} {brief_id}: {path.name} in {time.monotonic() - t1:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
