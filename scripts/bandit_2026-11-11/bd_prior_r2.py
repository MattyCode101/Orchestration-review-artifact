"""Bandit round 2 (O1): leave-one-design-out priors for the round-2 policy, from the same training call record.

The replay is the round-1 replay (bd_prior.replay), run through the round-2 policy module, so training-time and decision-time
numbers come from one code path: the eleventh feature is computed by the policy's own function on a genuine PolicyView.
Four prior sets, because the two switches change what the prior contains:
  feat | nofeat    with | without the eleventh feature (nofeat: x[10] = 0 and degrees of freedom of the ten round-1 features)
  keep | drop      empty calls kept as zero-reward samples (round 1) | dropped (an empty call is an observation, not a sample)
Files: data/bandit_2026-11-11/priors_r2/<feat|nofeat>_<keep|drop>_<brief>_<held_out>.json, held_out = a training design or
"none" (all five training designs, for designs outside the five). nofeat_keep must equal the round-1 prior files exactly:
--check-regression compares them.

usage: bd_prior_r2.py --build | --check-regression
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bd_common as bc  # noqa: E402
import bd_prior as bp  # noqa: E402
from orch.contract.models import MetricName  # noqa: E402
from orch.policies import bandit_r2 as r2  # noqa: E402

bp.bd = r2                                    # replay() resolves features, credited_gain, flat_reward, arm_key through r2
assert bp.bd is r2

M = MetricName
PRIORS_R2 = bc.DATA / "priors_r2"


def is_empty_sample(sample: dict) -> bool:
    """A logged call that found no recipe: nothing charged, nothing returned."""

    return sample["runs"] == 0 and not sample["got_state"]


def fit_r2(samples: list[dict], settings: r2.R2Settings, meta: dict, feature: bool) -> dict:
    """bp.fit with the eleventh feature switched on or held at zero. With the feature off, every number equals bp.fit's."""

    active = r2.DIM if feature else r2.ROUND1_DIM
    dim = r2.DIM
    arms: dict[str, dict] = {}

    def vector(s: dict) -> list[float]:
        return list(s["x"]) if feature else [*s["x"][: r2.ROUND1_DIM], 0.0]

    for s in samples:
        r = s["r_credited"]
        arm = arms.setdefault(s["key"], {
            "A": [[settings.ridge if i == j else 0.0 for j in range(dim)] for i in range(dim)],
            "b": [0.0] * dim, "n": 0, "sum_r": 0.0,
        })
        x = vector(s)
        for i in range(dim):
            for j in range(dim):
                arm["A"][i][j] += x[i] * x[j]
            arm["b"][i] += r * x[i]
        arm["n"] += 1
        arm["sum_r"] += r
    thetas = {key: r2._solve(arm["A"], arm["b"]) for key, arm in arms.items()}
    sse = 0.0
    for s in samples:
        sse += (s["r_credited"] - r2._dot(thetas[s["key"]], vector(s))) ** 2
    n_total = sum(arm["n"] for arm in arms.values())
    dof = max(1, n_total - sum(min(active, arm["n"]) for arm in arms.values()))
    sigma = max(0.5, math.sqrt(sse / dof))
    return {
        "schema": r2.SCHEMA, "features": list(r2.FEATURES), "ridge": settings.ridge, "sigma": sigma,
        "settings": json.loads(settings.canonical()), "arms": {
            key: {"A": arm["A"], "b": arm["b"], "n": arm["n"], "mean_reward": arm["sum_r"] / arm["n"]}
            for key, arm in sorted(arms.items())
        }, "meta": meta,
    }


def build() -> int:
    import yaml
    declared = yaml.safe_load((bc.PROJECT / "configs/policies/bandit_r2.yaml").read_text(encoding="utf-8"))
    settings = r2.R2Settings.for_variant("main", **declared["settings"])
    records = bp.load_records()
    record_sha = hashlib.sha256(bc.CALLS.read_bytes()).hexdigest()
    noises = bc.design_noise(records)
    for design in bc.DESIGNS:
        if abs(noises[design][M.DYNAMIC_POWER_W] - bc.power_band_w(design)) > 1e-6 * bc.power_band_w(design):
            raise ValueError(f"power band differs for {design}")
    PRIORS_R2.mkdir(parents=True, exist_ok=True)
    coverage: dict = {"record_sha256": record_sha, "settings_sha256": settings.sha256(), "cells": {}}
    checks_all: dict = {}
    for brief_id in bc.BRIEFS:
        checks: Counter = Counter()
        samples = bp.all_samples(brief_id, records, noises, settings, checks)
        unique, removed = bp.dedupe(samples)
        empties = [s for s in unique if is_empty_sample(s)]
        checks_all[brief_id] = dict(checks, samples=len(samples), duplicates_removed=removed, unique=len(unique),
                                    empty_samples=len(empties))
        for drop in (False, True):
            pool = [s for s in unique if not (drop and is_empty_sample(s))]
            for feature in (False, True):
                for held_out in (*bc.DESIGNS, "none"):
                    use = [s for s in pool if s["design"] != held_out]
                    meta = {"brief": brief_id, "feature": feature, "empty_dropped": drop, "held_out": held_out,
                            "record_sha256": record_sha, "samples": len(use), "designs": sorted({s["design"] for s in use})}
                    payload = fit_r2(use, settings, meta, feature)
                    name = f"{'feat' if feature else 'nofeat'}_{'drop' if drop else 'keep'}_{brief_id}_{held_out}.json"
                    (PRIORS_R2 / name).write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                    if held_out == "none":
                        coverage["cells"][f"{'feat' if feature else 'nofeat'}|{'drop' if drop else 'keep'}|{brief_id}"] = {
                            "samples": len(use), "sigma": payload["sigma"], "arms": len(payload["arms"]),
                            "per_arm_n": {k: v["n"] for k, v in payload["arms"].items()},
                        }
    coverage["checks"] = checks_all
    (PRIORS_R2 / "coverage.json").write_text(json.dumps(coverage, indent=1, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256("".join(
        f"{p.name}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in sorted(PRIORS_R2.glob("*_*.json"))).encode()).hexdigest()
    print(json.dumps({"checks": checks_all, "prior_files_digest": digest,
                      "cells_none": {k: {"samples": v["samples"], "sigma": round(v["sigma"], 3), "arms": v["arms"]}
                                     for k, v in coverage["cells"].items() if k.startswith(("feat|drop", "nofeat|keep"))}}, indent=1))
    return 0


def check_regression() -> int:
    """nofeat_keep priors equal the round-1 main priors: same arms, same statistics (first ten rows and columns), same sigma."""

    bad, checked = [], 0
    for brief_id in bc.BRIEFS:
        for held_out in (*bc.DESIGNS, "none"):
            old = json.loads((bc.DATA / "priors" / f"main_{brief_id}_{held_out}.json").read_text())
            new = json.loads((PRIORS_R2 / f"nofeat_keep_{brief_id}_{held_out}.json").read_text())
            checked += 1
            ok = old["sigma"] == new["sigma"] and old["ridge"] == new["ridge"] and sorted(old["arms"]) == sorted(new["arms"])
            if ok:
                for key, a in old["arms"].items():
                    b = new["arms"][key]
                    n = len(a["b"])
                    ok = ok and a["n"] == b["n"] and a["mean_reward"] == b["mean_reward"] and a["b"] == b["b"][:n] and b["b"][n:] == [0.0] \
                        and [row for row in a["A"]] == [row[:n] for row in b["A"][:n]] \
                        and all(v == 0.0 for row in b["A"][:n] for v in row[n:]) and b["A"][n] == [0.0] * n + [1.0]
            if not ok:
                bad.append(f"{brief_id}|{held_out}")
    print(json.dumps({"checked": checked, "differ": bad}))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--check-regression", action="store_true")
    args = ap.parse_args()
    return build() if args.build else check_regression()


if __name__ == "__main__":
    raise SystemExit(main())
