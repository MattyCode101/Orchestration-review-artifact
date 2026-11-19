"""Build a read-only cache manifest over stored run roots (generic form).

The test manifest (scripts/test_cache_manifest.py) is unchanged; this builder serves the
training replay and rehearsal, so that they use the same manifest-backed engine as the test
campaign. It reads run identity, cost fields, meter version, approval and file hashes only.
No source root is written. Modes: --train-dry-run, --train-write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from orch.engine.cache import RunCache
from orch.engine.manifest_cache import COST_FIELDS, SCHEMA, manifest_hash
from orch.engine.store import _atomic_bytes, _json_bytes, _sha256_file

PROJECT = Path(__file__).resolve().parents[1]
V4 = PROJECT / "data/pilots/thursday_overnight_train_v4_2026-11-05"
OUT = PROJECT / "data/campaign_prep_2026-11-07"
APPROVED_ALIAS = ("2ee8ee9efb9416d84afbef9f1e37350c7b2ea0b054637efd8cff02dbed784435",)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_generic(roots: dict[str, Path], meter: str, starts: dict[str, str],
                  start_evidence_roots: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    seen: dict[str, list[str]] = {}
    problems: list[str] = []
    skipped = {"retryable": 0, "older_meter_evidence": 0, "dangling": 0, "no_marker_files": 0}
    for rid, root in roots.items():
        for pointer in sorted((root / "cache").glob("*.json")):
            ptr = _json(pointer)
            if ptr["retryable"]:
                skipped["retryable"] += 1
                continue
            key, run_id = ptr["key"], ptr["run_id"]
            run_dir = root / "runs" / run_id
            if not (run_dir / "complete.json").is_file():
                skipped["dangling"] += 1
                continue
            info = _json(run_dir / "runinfo.json")
            if RunCache.key(info["input_sha256_before"], info["recipe_id"], info["recipe_version"]) != key:
                problems.append(f"pointer key differs from run identity {rid} {key[:12]}")
                continue
            ev_path = run_dir / "evidence.json"
            ev_meter = _json(ev_path).get("meter_version") if ev_path.is_file() else None
            if ev_meter is not None and ev_meter != meter:
                skipped["older_meter_evidence"] += 1
                continue
            out_path = run_dir / "output_state.json"
            out_state = _json(out_path) if out_path.is_file() else None
            approval: dict[str, Any] = {"kind": "native_current_meter" if ev_meter else "no_evidence"}
            eq = run_dir / "meter_equivalence.json"
            if eq.is_file():
                record = _json(eq)
                original = root / "runs" / record["old_run_id"] / "runinfo.json"
                original_info = _json(original) if original.is_file() else None
                same_cost = bool(original_info) and all(original_info.get(f) == info.get(f) for f in COST_FIELDS)
                if record.get("old_version") not in APPROVED_ALIAS or record.get("new_version") != meter or not same_cost:
                    problems.append(f"alias lacks the approved pair or original cost {rid} {key[:12]}")
                    continue
                approval = {
                    "kind": "meter_equivalence_alias", "audit_sha256": record["audit_sha256"],
                    "old_version": record["old_version"], "new_version": record["new_version"],
                    "original_run_id": record["old_run_id"], "cost_equals_original": True,
                    "equivalence_file_sha256": _sha256_file(eq),
                }
            seen.setdefault(key, []).append(rid)
            entries[key] = {
                "root": rid, "run_id": run_id, "complete_sha256": _sha256_file(run_dir / "complete.json"),
                "runinfo_sha256": _sha256_file(run_dir / "runinfo.json"),
                "recipe_id": info["recipe_id"], "recipe_version": info["recipe_version"],
                "input_state_id": info["input_sha256_before"],
                "output_state_id": out_state["id"] if out_state else None,
                "output_state_path": out_state["path"] if out_state else None,
                "status": info["status"], "fake": info["fake"], "meter_version": ev_meter,
                "cost": {f: info.get(f) for f in COST_FIELDS},
                "stored_cache_hit_flag": info["cache_hit"], "approval": approval,
            }
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    wanted = {e["input_state_id"] for e in entries.values()} | set(starts.values())
    wanted |= {e["output_state_id"] for e in entries.values() if e["output_state_id"]}
    states: dict[str, list[str]] = {}
    for state_id in sorted(wanted):
        locs = sorted({str(r / "states" / state_id / "design.dcp") for r in roots.values()
                       if (r / "states" / state_id / "design.dcp").is_file()})
        states[state_id] = locs
        if not locs:
            problems.append(f"state bytes not found in any root {state_id[:12]}")
    version_key = hashlib.sha256(meter.encode()).hexdigest()
    start: dict[str, Any] = {}
    for design, b2 in sorted(starts.items()):
        for rid, root in start_evidence_roots.items():
            path = root / "measurements" / b2 / version_key / "evidence.json"
            if path.is_file():
                ev = _json(path)
                if ev.get("meter_version") == meter and ev.get("state_id") == b2:
                    start[b2] = {"root": rid, "path": str(path), "sha256": _sha256_file(path),
                                 "meter_version": meter, "design": design}
                    break
        else:
            problems.append(f"no current-meter start evidence for {design}")
    body = {
        "schema_version": SCHEMA, "meter_version": meter, "roots": {k: str(v) for k, v in roots.items()},
        "entries": dict(sorted(entries.items())), "states": states, "start_evidence": start,
    }
    report = {
        "entries": len(entries), "keys_in_more_than_one_root": len(multi), "skipped": skipped,
        "entries_without_evidence": sum(e["meter_version"] is None for e in entries.values()),
        "alias_entries": sum(e["approval"]["kind"] == "meter_equivalence_alias" for e in entries.values()),
        "states_with_location": sum(bool(v) for v in states.values()), "states": len(states),
        "start_evidence_found": len(start), "problems": problems, "manifest_sha256": manifest_hash(body),
    }
    return body, report


def train_inputs() -> tuple[str, dict[str, str]]:
    plan = _json(V4 / "plan.json")
    return plan["meter_version"], dict(plan["training_inputs"])


def build_train() -> tuple[dict[str, Any], dict[str, Any]]:
    meter, starts = train_inputs()
    return build_generic({"v4": V4}, meter, starts, {"v4": V4})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train-dry-run", action="store_true")
    mode.add_argument("--train-write", action="store_true")
    args = parser.parse_args()
    body, report = build_train()
    if args.train_write:
        OUT.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(OUT / "train_cache_manifest.json", _json_bytes(body))
        _atomic_bytes(OUT / "train_cache_manifest.sha256", (report["manifest_sha256"] + "\n").encode())
        _atomic_bytes(OUT / "train_cache_manifest_report.json", _json_bytes(report))
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
