"""Bandit freeze (O1): fit the frozen prior of the learned policy on the training and development call records.

Plan, declared and hashed before this script ran: notes/bandit_test_freeze_plan.md. The pipeline is the declared one (bd_prior_r2.py and bd_prior.py, imported and not edited):
every logged call is replayed under the target brief with the round-2 policy's own functions, duplicates (same design, same call path) are kept once, empty-call samples are
dropped, ridge statistics per arm, sigma floor 0.5. What this script adds is only the pooling over the eight designs and the development designs' bands and L2R power offset.

  --check-training-only   refit from the training record alone with the declared pipeline's meta and compare the six feat_drop_<brief>_none.json byte for byte (check a)
  --build                 write data/bandit_2026-11-11/priors_frozen/feat_drop_<brief>_all8.json (six briefs), coverage.json, fit_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bd_common as bc  # noqa: E402
import bd_prior as bp  # noqa: E402
import bd_prior_r2 as bpr  # noqa: E402  (sets bp.bd to the round-2 policy module)
from orch.contract.models import MetricName  # noqa: E402
from orch.policies import bandit_r2 as r2  # noqa: E402

M = MetricName
DEV_CALLS = bc.DATA / "calls" / "episodes_dev_v1.jsonl"
FROZEN = bc.DATA / "priors_frozen"
DEV_DESIGNS = ("t2_cordic_s", "vtr_mcml_v2", "amd_mini-isp")
BRIEFS = bc.BRIEFS                                   # T0 A0 T A L2R P0
PLAN_NOTE = bc.PROJECT / "notes/bandit_test_freeze_plan.md"
NOISE_ROUND2 = bc.PROJECT / "configs/noise_round2.yaml"
NOISE_KEYS = {"wns_ns": M.WNS_NS, "packed_luts": M.PACKED_LUTS, "dynamic_power_w": M.DYNAMIC_POWER_W}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_settings() -> r2.R2Settings:
    import yaml
    declared = yaml.safe_load((bc.PROJECT / "configs/policies/bandit_r2.yaml").read_text(encoding="utf-8"))
    learned = r2.R2Settings.for_variant("main", **{**declared["settings"], **declared["labels"]["learned"]})
    if learned != r2.R2Settings.for_variant("main", **declared["settings"]):     # what bd_prior_r2.build() fits with
        raise SystemExit("the learned label differs from the settings the declared priors were fitted with")
    if not (learned.stop_threshold == 0.0 and learned.repair_feature and learned.observe_empty):
        raise SystemExit("the declared learned policy is stop 0, repair feature on, empty call as an observation")
    return learned


def dev_bands(dev_records: list[dict]) -> dict[str, dict[MetricName, float]]:
    """Bands of the development designs as the journals recorded them; they must equal configs/noise_round2.yaml and every journal of a design must agree."""

    import yaml
    table = yaml.safe_load(NOISE_ROUND2.read_text(encoding="utf-8"))["bands"]
    out: dict[str, dict[MetricName, float]] = {}
    for design in DEV_DESIGNS:
        recorded = {json.dumps(r["noise"], sort_keys=True) for r in dev_records if r["design"] == design}
        if len(recorded) != 1:
            raise SystemExit(f"{design}: the journals record {len(recorded)} different band sets")
        band = {NOISE_KEYS[k]: float(v) for k, v in json.loads(next(iter(recorded))).items() if k in NOISE_KEYS}
        for key, metric in NOISE_KEYS.items():
            row = table[design][key]
            if row["relative"] or abs(float(row["value"]) - band[metric]) > 1e-9:
                raise SystemExit(f"{design}: recorded {key} band {band[metric]} differs from configs/noise_round2.yaml ({row['value']})")
        if abs(round(band[M.DYNAMIC_POWER_W], 3) - band[M.DYNAMIC_POWER_W]) > 1e-12:
            raise SystemExit(f"{design}: the power band is not a whole number of mW")
        out[design] = band
    return out


def patch_l2r_power_bands(noises: dict) -> None:
    """bd_common.make_brief builds L2R from the power study's table, which has the five training designs only: development designs use their recorded power band."""

    original = bc.power_band_w
    dev = {d: noises[d][M.DYNAMIC_POWER_W] for d in DEV_DESIGNS if d in noises}

    def power_band_w(design: str) -> float:
        return dev[design] if design in dev else original(design)

    bc.power_band_w = power_band_w


def unique_samples(brief_id: str, groups: list[tuple[str, list[dict]]], noises: dict, settings: r2.R2Settings) -> tuple[list[dict], dict]:
    """All samples of the groups in order, then the declared dedupe. Returns (unique, per-group check counters and counts)."""

    samples: list[dict] = []
    report: dict = {}
    for name, records in groups:
        checks: Counter = Counter()
        part = bp.all_samples(brief_id, records, noises, settings, checks)
        report[name] = {"episodes": len(records), "samples": len(part), "checks": dict(checks)}
        samples.extend(part)
    unique, removed = bp.dedupe(samples)
    report["all"] = {"samples": len(samples), "duplicates_removed": removed, "unique": len(unique),
                     "empty_samples": sum(bpr.is_empty_sample(s) for s in unique)}
    return unique, report


def dump(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def check_training_only() -> int:
    settings = declared_settings()
    records = bp.load_records()
    record_sha = sha(bc.CALLS)
    noises = bc.design_noise(records)
    bad, checked = [], 0
    for brief_id in BRIEFS:
        unique, _ = unique_samples(brief_id, [("training", records)], noises, settings)
        use = [s for s in unique if not bpr.is_empty_sample(s)]
        meta = {"brief": brief_id, "feature": True, "empty_dropped": True, "held_out": "none", "record_sha256": record_sha,
                "samples": len(use), "designs": sorted({s["design"] for s in use})}
        text = dump(bpr.fit_r2(use, settings, meta, True))
        old = (bpr.PRIORS_R2 / f"feat_drop_{brief_id}_none.json").read_text(encoding="utf-8")
        checked += 1
        if text != old:
            bad.append(brief_id)
    print(json.dumps({"check": "a: training-only refit equals feat_drop_<brief>_none.json byte for byte", "checked": checked, "differ": bad}))
    return 1 if bad else 0


def build() -> int:
    settings = declared_settings()
    train = bp.load_records()
    dev = [json.loads(line) for line in DEV_CALLS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if {r["design"] for r in dev} != set(DEV_DESIGNS):
        raise SystemExit("the development record must hold exactly the three development designs")
    if any(r["design"] in DEV_DESIGNS for r in train) or {r["design"] for r in train} != set(bc.DESIGNS):
        raise SystemExit("the training record must hold exactly the five training designs")
    noises = bc.design_noise(train)
    noises.update(dev_bands(dev))
    patch_l2r_power_bands(noises)
    train_sha, dev_sha, plan_sha = sha(bc.CALLS), sha(DEV_CALLS), sha(PLAN_NOTE)
    old_cov = json.loads((bpr.PRIORS_R2 / "coverage.json").read_text(encoding="utf-8"))["cells"]
    FROZEN.mkdir(parents=True, exist_ok=True)
    report: dict = {"training_record_sha256": train_sha, "development_record_sha256": dev_sha, "plan_sha256": plan_sha,
                    "settings_sha256": settings.sha256(), "designs": sorted(noises), "briefs": {}}
    coverage: dict = {"training_record_sha256": train_sha, "development_record_sha256": dev_sha, "cells": {}}
    mismatches = 0
    for brief_id in BRIEFS:
        unique, rep = unique_samples(brief_id, [("training", train), ("development", dev)], noises, settings)
        use = [s for s in unique if not bpr.is_empty_sample(s)]
        meta = {"brief": brief_id, "feature": True, "empty_dropped": True, "held_out": "none", "fit": "all8",
                "record_sha256": train_sha, "development_record_sha256": dev_sha, "plan_sha256": plan_sha,
                "samples": len(use), "designs": sorted({s["design"] for s in use})}
        payload = bpr.fit_r2(use, settings, meta, True)
        path = FROZEN / f"feat_drop_{brief_id}_all8.json"
        path.write_text(dump(payload), encoding="utf-8")
        by_group = Counter("development" if s["design"] in DEV_DESIGNS else "training" for s in use)
        arm_n_dev = Counter(s["key"] for s in use if s["design"] in DEV_DESIGNS)
        old = old_cov[f"feat|drop|{brief_id}"]
        rep["fit"] = {
            "samples_in_fit": len(use), "by_group": dict(by_group), "arms": len(payload["arms"]), "sigma": payload["sigma"],
            "per_arm_n": {k: v["n"] for k, v in payload["arms"].items()}, "per_arm_n_from_development": dict(arm_n_dev),
            "training_only": {"samples": old["samples"], "arms": old["arms"], "sigma": old["sigma"]},
            "prior_sha256": sha(path),
        }
        for group in ("training", "development"):
            mismatches += rep[group]["checks"].get("validity_mismatch_vs_journal", 0)
        report["briefs"][brief_id] = rep
        coverage["cells"][f"feat|drop|{brief_id}"] = {"samples": len(use), "sigma": payload["sigma"], "arms": len(payload["arms"]),
                                                      "per_arm_n": rep["fit"]["per_arm_n"], "prior_sha256": rep["fit"]["prior_sha256"]}
    digest = hashlib.sha256("".join(f"{p.name}:{sha(p)}\n" for p in sorted(FROZEN.glob("feat_drop_*_all8.json"))).encode()).hexdigest()
    report["prior_files_digest"] = digest
    report["validity_mismatches_total"] = mismatches
    (FROZEN / "coverage.json").write_text(json.dumps(coverage, indent=1, sort_keys=True), encoding="utf-8")
    (FROZEN / "fit_report.json").write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    brief = {b: {"samples": r["fit"]["samples_in_fit"], "dev_samples": r["fit"]["by_group"].get("development", 0), "arms": r["fit"]["arms"],
                 "sigma": round(r["fit"]["sigma"], 4), "sigma_training_only": round(r["fit"]["training_only"]["sigma"], 4),
                 "dev_checks": r["development"]["checks"], "train_checks": r["training"]["checks"]} for b, r in report["briefs"].items()}
    print(json.dumps({"prior_files_digest": digest, "validity_mismatches_total": mismatches, "briefs": brief}, indent=1, sort_keys=True))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-training-only", action="store_true")
    mode.add_argument("--build", action="store_true")
    args = ap.parse_args()
    return check_training_only() if args.check_training_only else build()


if __name__ == "__main__":
    raise SystemExit(main())
