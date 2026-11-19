"""Bandit (O1): replay the training call record under any brief, and build the leave-one-design-out priors.

Every logged call is re-scored under the target brief from its own measured outcomes: the journals hold the
base and output metrics of each call, which do not depend on the brief that was running. A genuine PolicyView
is rebuilt before and after every call, and the policy's own functions (features, credited_gain,
flat_reward) are applied to it, so training-time and decision-time numbers come from one code path.

Cross-checks built in: for the episode's own brief the replayed brief-validity of every output must equal the
journal's verdict (ACCEPTABLE), and the replayed budget must equal the journalled budget.

usage: bd_prior.py --build      write priors/<variant>_<brief>_<heldout>.json and coverage.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bd_common as bc  # noqa: E402
from orch.contract.models import MetricName, VerdictStatus, WorkerSettings  # noqa: E402
from orch.contract.views import CallSummary, PolicyView, StateSummary  # noqa: E402
from orch.policies import bandit as bd  # noqa: E402

M = MetricName


@dataclass(frozen=True)
class Visit:
    summary: StateSummary
    lineage: int
    run_id: str | None


def _summary(state_id, metrics, valid):
    return StateSummary(state_id=state_id, metrics={M(k): float(v) for k, v in metrics.items()},
                        physically_valid=True, brief_valid=valid)


def _valid_under(brief, start_metrics, metrics) -> bool:
    for term in brief.terms:
        limit = term.factor * start_metrics[term.metric.value] + term.offset
        value = metrics[term.metric.value]
        if not (value <= limit if term.comparison == "<=" else value >= limit):
            return False
    return True


def _rank_key(brief, visit: Visit):
    """The episode's own ranking (orch/episode.py _rank): goal, then the other metric, then lineage, then ids."""

    metrics = visit.summary.metrics
    other = M.PACKED_LUTS if brief.objective.metric == M.WNS_NS else M.WNS_NS
    primary = metrics[brief.objective.metric]
    secondary = metrics.get(other)
    first = -primary if brief.objective.direction == "max" else primary
    second = 0.0 if secondary is None else (secondary if other == M.PACKED_LUTS else -secondary)
    return (first, secondary is None, second, visit.lineage, visit.summary.state_id, visit.run_id or "")


def _view(brief, noise, cards, episode_id, start, visits, current, calls, budget_left) -> PolicyView:
    ranked = sorted((v for v in visits if v.summary.brief_valid), key=lambda v: _rank_key(brief, v))
    return PolicyView(
        brief=brief, budget_left=budget_left, start=start, current=current.summary,
        best_valid=ranked[0].summary if ranked else None, visited=tuple(v.summary for v in visits),
        steps=(), calls=tuple(calls), worker_cards=cards, noise_bands=noise, episode_id=episode_id, fake=False,
    )


def replay(rec: dict, brief, noise, cards, settings: bd.BanditSettings, checks: Counter) -> list[dict]:
    """One sample per call: arm, features and both rewards. Calls outside the action set are skipped, not dropped
    from the episode state."""

    start_metrics = rec["start_metrics"]
    start = _summary(rec["start_state_id"], start_metrics, True)
    visits = [Visit(start, 0, None)]
    current = visits[0]
    calls: list[CallSummary] = []
    budget_left = rec["budget_runs"]
    view = _view(brief, noise, cards, rec["episode_id"], start, visits, current, calls, budget_left)
    samples = []
    prefix = []
    for call in rec["calls"]:
        if call["budget_left_before"] != budget_left:
            checks["budget_mismatch"] += 1
        base_visit = next((v for v in reversed(visits) if v.summary.state_id == call["base_id"]), None)
        if base_visit is None:
            checks["base_not_visited"] += 1
            break
        prefix.append((call["worker"], call["base_id"], call["preserve"], call["max_runs"]))
        best_id = view.best_valid.state_id
        kind = "best" if call["base_id"] == best_id else ("latest" if call["base_id"] == view.current.state_id else "other")
        attempts = call["max_runs"]
        x = bd.features(view, base_visit.summary, call["worker"], attempts) if attempts in bd.ATTEMPTS else None
        phi_before = bd.credited_gain(view, settings.rho)
        # advance the episode by this call
        new = []
        for step in call["steps"]:
            if step["physically_valid"] and step["metrics"] is not None:
                valid = _valid_under(brief, start_metrics, step["metrics"])
                if brief.id == rec["brief"] and valid != (step["status"] == "ACCEPTABLE"):
                    checks["validity_mismatch_vs_journal"] += 1
                if brief.id == rec["brief"]:
                    checks["validity_checked"] += 1
                new.append(Visit(_summary(step["out_id"], step["metrics"], valid), base_visit.lineage + 1, step["run_id"]))
        visits.extend(new)
        chosen = next((v for v in new if v.summary.state_id == call["selected_out_id"]), None) if call["selected_out_id"] else None
        if call["selected_out_id"] is not None and chosen is None:
            checks["selected_not_visited"] += 1
        if chosen is not None:
            current = chosen
        budget_left -= call["recipe_runs"]
        if budget_left < 0:
            checks["budget_negative"] += 1
            break
        summary = CallSummary(
            mandate_id=f"{rec['episode_id']}#{call['i']}", worker_id=call["worker"], base_state_id=call["base_id"],
            settings=WorkerSettings(preserve_placement=call["preserve"], max_runs=call["max_runs"]),
            selected_state_id=chosen.summary.state_id if chosen else None,
            selected_verdict_status=(None if chosen is None else (
                VerdictStatus.ACCEPTABLE if chosen.summary.brief_valid else VerdictStatus.BRIEF_VIOLATION)),
            recipe_runs=call["recipe_runs"], context_ref="",
        )
        calls.append(summary)
        after = _view(brief, noise, cards, rec["episode_id"], start, visits, current, calls, budget_left)
        if x is not None and kind in ("best", "latest"):
            signature = hashlib.sha256(json.dumps([rec["design"], prefix], separators=(",", ":")).encode()).hexdigest()
            samples.append({
                "sig": signature, "design": rec["design"], "episode_id": rec["episode_id"], "source": rec["source"],
                "orig_brief": rec["brief"], "i": call["i"], "base_kind": kind,
                "key": bd.arm_key(call["worker"], call["preserve"], attempts, kind), "x": list(x),
                "r_credited": bd.credited_gain(after, settings.rho) - phi_before,
                "r_flat": bd.flat_reward(after, summary, settings.flat_penalty),
                "runs": call["recipe_runs"], "got_state": chosen is not None,
            })
        else:
            checks["skipped_" + ("attempts" if x is None else "base_other")] += 1
        view = after
    return samples


def all_samples(brief_id: str, records, noises, settings, checks) -> list[dict]:
    out = []
    for rec in records:
        brief = bc.make_brief(brief_id, rec["design"])
        out.extend(replay(rec, brief, noises[rec["design"]], bc.cards_for(brief_id), settings, checks))
    return out


def dedupe(samples: list[dict]) -> tuple[list[dict], int]:
    """A deterministic repeat (same design, same call path) adds no information: keep the first."""

    seen, unique = set(), []
    for s in samples:
        if s["sig"] in seen:
            continue
        seen.add(s["sig"])
        unique.append(s)
    return unique, len(samples) - len(unique)


def fit(samples: list[dict], settings: bd.BanditSettings, meta: dict) -> dict:
    """Ridge statistics per arm plus the pooled residual spread (sigma) used to scale the exploration bonus."""

    reward_key = "r_credited" if settings.reward == "credited" else "r_flat"
    arms: dict[str, dict] = {}
    for s in samples:
        r = s[reward_key]
        arm = arms.setdefault(s["key"], {
            "A": [[settings.ridge if i == j else 0.0 for j in range(bd.DIM)] for i in range(bd.DIM)],
            "b": [0.0] * bd.DIM, "n": 0, "sum_r": 0.0,
        })
        x = s["x"]
        for i in range(bd.DIM):
            for j in range(bd.DIM):
                arm["A"][i][j] += x[i] * x[j]
            arm["b"][i] += r * x[i]
        arm["n"] += 1
        arm["sum_r"] += r
    sse, dof = 0.0, 0
    thetas = {key: bd._solve(arm["A"], arm["b"]) for key, arm in arms.items()}
    for s in samples:
        sse += (s[reward_key] - bd._dot(thetas[s["key"]], s["x"])) ** 2
    n_total = sum(arm["n"] for arm in arms.values())
    dof = max(1, n_total - sum(min(bd.DIM, arm["n"]) for arm in arms.values()))
    sigma = max(0.5, math.sqrt(sse / dof))
    return {
        "schema": bd.SCHEMA, "features": list(bd.FEATURES), "ridge": settings.ridge, "sigma": sigma,
        "settings": json.loads(settings.canonical()), "arms": {
            key: {"A": arm["A"], "b": arm["b"], "n": arm["n"], "mean_reward": arm["sum_r"] / arm["n"]}
            for key, arm in sorted(arms.items())
        }, "meta": meta,
    }


def load_records() -> list[dict]:
    return [json.loads(line) for line in bc.CALLS.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--settings", default=str(bc.PROJECT / "configs/policies/bandit.yaml"))
    ap.add_argument("--rho", type=float, default=None, help="declared sensitivity only: another rho")
    ap.add_argument("--priors-dir", default="priors")
    args = ap.parse_args()
    import yaml
    common = dict(yaml.safe_load(Path(args.settings).read_text(encoding="utf-8"))["settings"])
    if args.rho is not None:
        common["rho"] = args.rho
    settings = bd.BanditSettings(**common)                          # base choice and reward are set per variant below
    priors_dir = bc.DATA / args.priors_dir
    records = load_records()
    record_sha = hashlib.sha256(bc.CALLS.read_bytes()).hexdigest()
    noises = bc.design_noise(records)                            # raises if a journal band differs from the table
    for design in bc.DESIGNS:                                    # the table's power band equals the L2R offset
        if abs(noises[design][M.DYNAMIC_POWER_W] - bc.power_band_w(design)) > 1e-6 * bc.power_band_w(design):
            raise ValueError(f"power band differs for {design}")
    priors_dir.mkdir(parents=True, exist_ok=True)
    coverage: dict = {"record_sha256": record_sha, "settings_sha256": settings.sha256(), "rho": settings.rho, "cells": {}}
    checks_all: dict = {}
    for brief_id in bc.BRIEFS:
        checks: Counter = Counter()
        samples = all_samples(brief_id, records, noises, settings, checks)
        unique, removed = dedupe(samples)
        checks_all[brief_id] = dict(checks, samples=len(samples), duplicates_removed=removed, unique=len(unique))
        for variant in bc.VARIANTS:
            vs = bd.BanditSettings.for_variant(variant, **common)
            pool = [s for s in unique if vs.base_choice == "best_or_latest" or s["base_kind"] == "best"]
            for held_out in (*bc.DESIGNS, "none"):
                use = [s for s in pool if s["design"] != held_out]
                meta = {"brief": brief_id, "variant": variant, "held_out": held_out, "record_sha256": record_sha,
                        "samples": len(use), "designs": sorted({s["design"] for s in use})}
                payload = fit(use, vs, meta)
                path = priors_dir / f"{variant}_{brief_id}_{held_out}.json"
                path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                if held_out == "none":
                    per_arm = {k: v["n"] for k, v in payload["arms"].items()}
                    coverage["cells"][f"{variant}|{brief_id}"] = {
                        "samples": len(use), "sigma": payload["sigma"], "arms": len(per_arm), "per_arm_n": per_arm,
                    }
        if brief_id in ("T0", "P0") and args.priors_dir == "priors":   # one reward table per brief for the report
            (bc.DATA / f"samples_{brief_id}.jsonl").write_text(
                "\n".join(json.dumps(s, sort_keys=True, separators=(",", ":")) for s in unique) + "\n", encoding="utf-8")
    coverage["checks"] = checks_all
    (priors_dir / "coverage.json").write_text(json.dumps(coverage, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"checks": checks_all, "cells": {k: {"samples": v["samples"], "sigma": round(v["sigma"], 3), "arms": v["arms"]}
                                                      for k, v in coverage["cells"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
