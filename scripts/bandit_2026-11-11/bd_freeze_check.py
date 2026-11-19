"""Bandit freeze (O1): recheck the freeze. Read-only; exit code 0 only if everything is equal.

  1. notes/bandit_test_freeze.sha256: every listed file equals its recorded hash (the `sha256sum -c` of the list).
  2. the entry point: orch.policies.bandit_frozen builds the policy of each of the six briefs, refuses W2R, and records a provenance with seed_used false.
  3. frozen elsewhere, unchanged: the 139-file campaign freeze (data/test_campaign_prep_2026-11-06/freeze_final.json: every file's hash, and the file's own hash)
     and the template v2.1 freeze list (notes/template_v2.1_freeze.sha256).

usage: bd_freeze_check.py [--no-external]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bd_common as bc  # noqa: E402

P = bc.PROJECT
LIST = P / "notes/bandit_test_freeze.sha256"
FREEZE139 = P / "data/test_campaign_prep_2026-11-06/freeze_final.json"
FREEZE139_PREFIX = "e36dd1a5"
TEMPLATE_V21 = P / "notes/template_v2.1_freeze.sha256"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_list(path: Path) -> dict:
    bad, n = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        want, rel = line.split("  ", 1)
        n += 1
        target = P / rel
        if not target.is_file() or sha(target) != want:
            bad.append(rel)
    return {"files": n, "differ": bad}


def check_entry_point() -> dict:
    from orch.policies import bandit_frozen as bf
    out: dict = {"briefs": {}, "problems": []}
    for brief_id in ("T0", "A0", "T", "A", "L2R", "P0"):
        try:
            policy = bf.bandit("any_design", brief_id, 3)
            info = policy.provenance()
            out["briefs"][brief_id] = {"version": policy.version, "prior": info["prior_sha256"][:12], "seed_used": info["seed_used"]}
            if info["seed_used"] or len(info["fit_designs"]) != 8:
                out["problems"].append(f"{brief_id}: provenance")
        except Exception as exc:  # noqa: BLE001
            out["problems"].append(f"{brief_id}: {type(exc).__name__}: {str(exc)[:160]}")
    try:
        bf.bandit("any_design", "W2R", 0)
        out["problems"].append("W2R was not refused")
    except bf.BanditNotApplicable:
        out["w2r_refused"] = True
    return out


def check_external() -> dict:
    out: dict = {}
    freeze = json.loads(FREEZE139.read_text(encoding="utf-8"))
    own = sha(FREEZE139)
    bad = [rel for rel, want in freeze["files"].items() if not (P / rel).is_file() or sha(P / rel) != want]
    out["campaign_freeze_139"] = {"files": len(freeze["files"]), "differ": bad, "own_sha256": own, "own_prefix_ok": own.startswith(FREEZE139_PREFIX)}
    out["template_v21_freeze"] = check_list(TEMPLATE_V21)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-external", action="store_true")
    args = ap.parse_args()
    report = {"hash_list": check_list(LIST), "entry_point": check_entry_point()}
    ok = not report["hash_list"]["differ"] and not report["entry_point"]["problems"]
    if not args.no_external:
        report["external"] = check_external()
        ext = report["external"]
        ok = ok and not ext["campaign_freeze_139"]["differ"] and ext["campaign_freeze_139"]["own_prefix_ok"] and not ext["template_v21_freeze"]["differ"]
    report["all_equal"] = ok
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
