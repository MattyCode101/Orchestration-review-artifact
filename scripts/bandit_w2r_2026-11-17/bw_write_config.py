"""Write configs/policies/bandit_w2r_postfreeze.yaml (new file, refuses to overwrite): the prior path and hash, the W2R brief signature, the record hashes.
The signature must be equal for the W2R brief of all eight fit designs (the power limit's offset is the design's band and is left out of it).
usage (project root, project venv): python scripts/bandit_w2r_2026-11-17/bw_write_config.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
os.environ["TR_QUEUE"] = "fl"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts/fill_2026-11-16"))
import tr_common as tc  # noqa: E402
import yaml  # noqa: E402
from orch.policies import bandit_frozen as bf  # noqa: E402

OUT = PROJECT / "configs/policies/bandit_w2r_postfreeze.yaml"
FIT = PROJECT / "data/bandit_2026-11-11/priors_postfreeze"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if OUT.exists():
        print("refused: exists")
        return 1
    sigs = {d: json.loads(json.dumps(bf.brief_signature(tc.brief_for(d, "W2R")))) for d in tc.TRAINING + tc.DEVELOPMENT}
    if len({json.dumps(s, sort_keys=True) for s in sigs.values()}) != 1:
        raise SystemExit("the W2R brief differs in kind between the fit designs")
    report = json.loads((FIT / "fit_report_W2R.json").read_text(encoding="utf-8"))
    prior = FIT / "feat_drop_W2R_all8plus.json"
    cfg = {
        "schema_version": "bandit-w2r-postfreeze-v1",
        "declared": "decision of Tue 17 Nov 2026: fill the missing Learned Bandit cell (three workers, timing-first W2R, TEST) as a labelled post-freeze row; the prior is fitted "
                    "on 17 Nov from TRAINING and VALIDATION call records only, with the procedure of the frozen L2R and P0 priors; no frozen file is edited.",
        "prior": str(prior.relative_to(PROJECT)), "prior_sha256": sha(prior), "samples": report["samples_in_fit"], "sigma": report["sigma"],
        "signature": next(iter(sigs.values())),
        "plan": "notes/bandit_w2r_postfreeze_plan.md", "plan_sha256": report["plan_sha256"],
        "records": {"training": report["training_record_sha256"], "development": report["development_record_sha256"], "w2r_postfreeze": report["w2r_record_sha256"]},
        "fit_designs": sorted(tc.TRAINING + tc.DEVELOPMENT),
        "frozen_config_sha256": sha(PROJECT / "configs/policies/bandit_test_frozen.yaml"),
        "post_freeze": True,
    }
    OUT.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    print("wrote", OUT.name, sha(OUT), "signature", cfg["signature"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
