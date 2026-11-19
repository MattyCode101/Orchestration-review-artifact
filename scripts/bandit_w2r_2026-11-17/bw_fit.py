"""Learned Bandit W2R prior (post-freeze, Tue 17 Nov 2026): extract the new W2R call record, check the procedure, fit, count the arms.

Plan, hashed before this script ran: notes/bandit_w2r_postfreeze_plan.md. Nothing frozen is edited: bd_freeze_fit, bd_prior, bd_prior_r2, bd_common and
bd_extract are imported as they are. New files only: data/bandit_2026-11-11/calls/episodes_w2r_postfreeze_v1.jsonl, data/bandit_2026-11-11/priors_postfreeze/.

  extract   write the new W2R record from the power review and post-freeze row c (TRAINING and VALIDATION designs only)
  check     check a: the frozen L2R and P0 priors are reproduced byte for byte from the frozen pool by this script's replay set-up
  fit       fit the W2R prior over frozen pool + new record, write it, write counts of the opening arms next to the frozen priors
usage (project root, project venv): python scripts/bandit_w2r_2026-11-17/bw_fit.py extract|check|fit
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
os.environ["TR_QUEUE"] = "fl"                                 # only to import the fill queue's brief builder (TRAINING and VALIDATION designs only)
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts/bandit_2026-11-11"))
sys.path.insert(0, str(PROJECT / "scripts/fill_2026-11-16"))
import bd_common as bc  # noqa: E402
import bd_extract as be  # noqa: E402
import bd_freeze_fit as bf  # noqa: E402
import bd_prior as bp  # noqa: E402
import bd_prior_r2 as bpr  # noqa: E402
import tr_common as tc  # noqa: E402
from orch.config import load_worker_card  # noqa: E402

DATA = bc.DATA
NEW_RECORD = DATA / "calls" / "episodes_w2r_postfreeze_v1.jsonl"
OUT = DATA / "priors_postfreeze"
PLAN = PROJECT / "notes/bandit_w2r_postfreeze_plan.md"
BASE = PROJECT / "data/test_r2_2026-11-13"
SOURCES = (                                                    # (label, root): TRAINING and VALIDATION folders only
    ("power_review", BASE / "power_review"),
    ("post_freeze_row_c", BASE / "post_freeze_2026-11-16"),
)
SCOPES = ("train", "dev")
WORKERS = ("area", "power", "timing")
GATE_FROZEN = ("T0", "A0", "T", "A", "L2R", "P0")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opening_keys(workers) -> list[str]:
    return [f"{w}|{s}|{a}|best" for w in workers for s in ("free", "keep") for a in (1, 2)]


# ------------------------------------------------------------------------------------------------ extraction
def extract() -> int:
    rows, skipped, inventory = [], Counter(), Counter()
    for label, root in SOURCES:
        for scope in SCOPES:
            for journal in sorted((root / scope / "engine" / "episodes").glob("*/journal.jsonl")):
                record = be.episode_record(journal.parent.name, be.read_events(journal))
                if record is None:
                    skipped["general_agent_or_fake"] += 1
                    continue
                if record["brief"] != "W2R":
                    skipped["other_brief"] += 1
                    continue
                if not record["calls"]:
                    skipped["no_completed_call"] += 1
                    continue
                if record["design"] not in tc.TRAINING + tc.DEVELOPMENT:
                    raise SystemExit(f"refused: {record['design']} is not a TRAINING or VALIDATION design")
                record["source"] = f"{label}:{scope}"
                rows.append(record)
                inventory[(label, scope, record["policy_id"])] += 1
    if NEW_RECORD.exists():
        raise SystemExit(f"refused: {NEW_RECORD.name} exists; not overwritten")
    text = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n"
    NEW_RECORD.write_text(text, encoding="utf-8")
    calls = sum(len(r["calls"]) for r in rows)
    print(json.dumps({"episodes": len(rows), "calls": calls, "empty_calls": sum(c["recipe_runs"] == 0 and c["selected_out_id"] is None for r in rows for c in r["calls"]),
                      "by_source_scope_policy": {"|".join(k): v for k, v in sorted(inventory.items())}, "skipped": dict(skipped),
                      "designs": dict(Counter(r["design"] for r in rows)), "record_sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ set-up shared by check and fit
def load_pool():
    settings = bf.declared_settings()
    train = bp.load_records()
    dev = [json.loads(line) for line in bf.DEV_CALLS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {r["design"] for r in dev} != set(bf.DEV_DESIGNS):
        raise SystemExit("the development record must hold exactly the three development designs")
    if any(r["design"] in bf.DEV_DESIGNS for r in train) or {r["design"] for r in train} != set(bc.DESIGNS):
        raise SystemExit("the training record must hold exactly the five training designs")
    noises = bc.design_noise(train)
    noises.update(bf.dev_bands(dev))
    bf.patch_l2r_power_bands(noises)
    return settings, train, dev, noises


def patch_w2r() -> None:
    """bd_common knows T0 A0 T A L2R P0: W2R is built by the fill queue's brief builder, with three worker cards."""

    original_make, original_cards = bc.make_brief, bc.cards_for

    def make_brief(brief_id: str, design: str):
        return tc.brief_for(design, "W2R") if brief_id == "W2R" else original_make(brief_id, design)

    def cards_for(brief_id: str):
        if brief_id == "W2R":
            return tuple(load_worker_card(PROJECT / "configs/workers" / f"{n}.yaml") for n in ("timing", "area", "power"))
        return original_cards(brief_id)

    bc.make_brief, bc.cards_for = make_brief, cards_for


def build_prior(brief_id, groups, noises, settings, meta_extra, plan_sha):
    unique, rep = bf.unique_samples(brief_id, groups, noises, settings)
    use = [s for s in unique if not bpr.is_empty_sample(s)]
    train_sha, dev_sha = sha(bc.CALLS), sha(bf.DEV_CALLS)
    meta = {"brief": brief_id, "feature": True, "empty_dropped": True, "held_out": "none", "fit": "all8",
            "record_sha256": train_sha, "development_record_sha256": dev_sha, "plan_sha256": plan_sha,
            "samples": len(use), "designs": sorted({s["design"] for s in use})}
    meta.update(meta_extra)
    return bpr.fit_r2(use, settings, meta, True), use, rep


def check() -> int:
    settings, train, dev, noises = load_pool()
    plan_sha = sha(bf.PLAN_NOTE)                               # the freeze plan: the frozen priors carry its hash in their meta
    bad = []
    for brief_id in ("L2R", "P0"):
        payload, _, _ = build_prior(brief_id, [("training", train), ("development", dev)], noises, settings, {}, plan_sha)
        mine = bf.dump(payload)
        old = (bf.FROZEN / f"feat_drop_{brief_id}_all8.json").read_text(encoding="utf-8")
        print(brief_id, "reproduced byte for byte:", mine == old, "| bytes", len(mine), len(old))
        if mine != old:
            bad.append(brief_id)
    print(json.dumps({"check": "a: the frozen L2R and P0 priors from the frozen pool", "differ": bad}))
    return 1 if bad else 0


def new_record_noise_report(new, noises) -> dict:
    """Bands recorded in the new journals against the bands used in the fit (reported only)."""

    keys = {"wns_ns": bc.M.WNS_NS, "packed_luts": bc.M.PACKED_LUTS, "dynamic_power_w": bc.M.DYNAMIC_POWER_W}
    diffs: dict = {}
    for r in new:
        for k, metric in keys.items():
            if k in r["noise"] and abs(float(r["noise"][k]) - noises[r["design"]][metric]) > 1e-6 * max(1.0, abs(noises[r["design"]][metric])):
                diffs[f"{r['design']}|{k}"] = {"journal": r["noise"][k], "fit": noises[r["design"]][metric]}
    return diffs


def arm_counts(payload: dict, keys: list[str]) -> dict:
    return {k: payload["arms"].get(k, {"n": 0})["n"] for k in keys}


def fit() -> int:
    settings, train, dev, noises = load_pool()
    patch_w2r()
    new = [json.loads(line) for line in NEW_RECORD.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(r["design"] not in tc.TRAINING + tc.DEVELOPMENT or r["brief"] != "W2R" for r in new):
        raise SystemExit("the new record must be W2R episodes of TRAINING and VALIDATION designs only")
    plan_sha, new_sha = sha(PLAN), sha(NEW_RECORD)
    extra = {"fit": "all8+w2r_postfreeze", "w2r_record_sha256": new_sha, "post_freeze": True}
    payload, use, rep = build_prior("W2R", [("training", train), ("development", dev), ("w2r_new", new)], noises, settings, extra, plan_sha)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "feat_drop_W2R_all8plus.json"
    if path.exists():
        raise SystemExit(f"refused: {path.name} exists; not overwritten")
    path.write_text(bf.dump(payload), encoding="utf-8")
    mismatches = sum(rep[g]["checks"].get("validity_mismatch_vs_journal", 0) for g in ("training", "development", "w2r_new"))
    checked_new = rep["w2r_new"]["checks"].get("validity_checked", 0)
    # sensitivity (counts only, nothing is written as a prior): the new W2R record alone, and the frozen pool alone replayed under W2R
    only_new, _, rep_new = build_prior("W2R", [("w2r_new", new)], noises, settings, {}, plan_sha)
    frozen_pool, _, rep_pool = build_prior("W2R", [("training", train), ("development", dev)], noises, settings, {}, plan_sha)
    # counts next to the frozen priors
    frozen = {b: json.loads((bf.FROZEN / f"feat_drop_{b}_all8.json").read_text(encoding="utf-8")) for b in GATE_FROZEN}
    open3, open2 = opening_keys(WORKERS), opening_keys(("area", "timing"))
    table = {"W2R_new": arm_counts(payload, open3), "W2R_new_only_w2r_record": arm_counts(only_new, open3), "W2R_frozen_pool_only": arm_counts(frozen_pool, open3)}
    for b, p in frozen.items():
        table[f"frozen_{b}"] = arm_counts(p, open3 if b in ("L2R", "P0") else open2)
    frozen_min = {b: min(table[f"frozen_{b}"].values()) for b in frozen}
    smallest_frozen = min(frozen_min.values())
    smallest_new = min(table["W2R_new"].values())
    gate = {"smallest_opening_arm_frozen_L2R": frozen_min["L2R"], "smallest_opening_arm_all_six_frozen": smallest_frozen,
            "smallest_opening_arm_W2R_new": smallest_new, "gate_vs_all_six_passes": smallest_new >= smallest_frozen,
            "gate_vs_L2R_passes": smallest_new >= frozen_min["L2R"]}
    report = {"plan_sha256": plan_sha, "training_record_sha256": sha(bc.CALLS), "development_record_sha256": sha(bf.DEV_CALLS), "w2r_record_sha256": new_sha,
              "prior_sha256": sha(path), "settings_sha256": settings.sha256(), "samples_in_fit": len(use), "arms": len(payload["arms"]), "sigma": payload["sigma"],
              "episodes": {"training": len(train), "development": len(dev), "w2r_new": len(new)},
              "replay": {g: rep[g] for g in ("training", "development", "w2r_new", "all")},
              "validity_mismatch_vs_journal_total": mismatches, "validity_checked_on_new_w2r_record": checked_new,
              "noise_differences_new_journals_vs_fit": new_record_noise_report(new, noises),
              "per_arm_n_all": {k: v["n"] for k, v in payload["arms"].items()},
              "opening_arm_counts": table, "gate": gate,
              "frozen_L2R": {"samples": frozen["L2R"]["meta"]["samples"], "arms": len(frozen["L2R"]["arms"]), "sigma": frozen["L2R"]["sigma"],
                             "per_arm_n_all": {k: v["n"] for k, v in frozen["L2R"]["arms"].items()}},
              "samples_by_group": dict(Counter("development" if s["design"] in bf.DEV_DESIGNS else "training" for s in use)),
              "samples_new_only": rep_new["all"], "samples_frozen_pool_only": rep_pool["all"]}
    (OUT / "fit_report_W2R.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"prior_sha256": report["prior_sha256"], "samples_in_fit": len(use), "arms": len(payload["arms"]), "sigma": round(payload["sigma"], 4),
                      "validity_mismatch_vs_journal_total": mismatches, "validity_checked_on_new_w2r_record": checked_new,
                      "noise_differences": report["noise_differences_new_journals_vs_fit"], "gate": gate,
                      "samples_new_only": rep_new["all"], "samples_frozen_pool_only": rep_pool["all"]}, indent=1, sort_keys=True))
    print("opening arm counts (rows = prior, columns = arm):")
    print("arm".ljust(22) + "".join(k.replace("W2R_", "W").replace("frozen_", "fz_")[:14].rjust(15) for k in table))
    for k in open3:
        print(k.ljust(22) + "".join(str(table[t].get(k, "-")).rjust(15) for t in table))
    return 0 if mismatches == 0 else 2


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "extract":
        return extract()
    if mode == "check":
        return check()
    if mode == "fit":
        return fit()
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
