"""Build the read-only test-cache manifest over the 11 stored roots.

Reads run records (identity, cost, meter version, approval) and file hashes only.
No metric value is parsed or printed, no tool starts, and no source root is
written. Output goes to data/test_campaign_prep_2026-11-06/.

Modes: --dry-run (build in memory, print counts)  --write (also publish files)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from orch.engine.cache import RunCache
from orch.engine.manifest_cache import COST_FIELDS, SCHEMA, manifest_hash
from orch.engine.store import _atomic_bytes, _json_bytes, _sha256_file

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "data/test_campaign_prep_2026-11-06"
BASE = PROJECT / "data/exp_evening_2026-11-04/test_b2/overnight_baseline_2026-11-04"
POWER = PROJECT / "data/sweeps/friday_test_power_2026-11-06"
SELECTION = POWER / "selection.json"
METER = "0dd0cf9424bbb7ab20dc5cf981bc35068c4ba08612ecd0f6c46098e6bf880bbe"
APPROVED_ALIAS = ("2ee8ee9efb9416d84afbef9f1e37350c7b2ea0b054637efd8cff02dbed784435", METER)
DESIGNS = (
    "rosetta_digit-recognition", "corescore_500_mod", "vexriscv_re-place_v2",
    "fir_symmetric_systolic", "fir_systolic_transposed",
)


def roots() -> dict[str, Path]:
    found = {"test_base": BASE}
    for phase in "ab":
        for design in DESIGNS:
            found[f"power_{phase}:{design}"] = POWER / f"phase_{phase}" / design
    return found


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    rts = roots()
    entries: dict[str, dict[str, Any]] = {}
    seen: dict[str, list[str]] = {}
    problems: list[str] = []
    retryable = 0
    for rid, root in rts.items():
        for pointer in sorted((root / "cache").glob("*.json")):
            ptr = _json(pointer)
            if ptr["retryable"]:
                retryable += 1
                continue
            key, run_id = ptr["key"], ptr["run_id"]
            seen.setdefault(key, []).append(rid)
            run_dir = root / "runs" / run_id
            complete = run_dir / "complete.json"
            if not complete.is_file():
                problems.append(f"dangling pointer {rid} {key[:12]}")
                continue
            info = _json(run_dir / "runinfo.json")
            ev_meter = _json(run_dir / "evidence.json").get("meter_version") if (run_dir / "evidence.json").is_file() else None
            out_state = _json(run_dir / "output_state.json") if (run_dir / "output_state.json").is_file() else None
            if RunCache.key(info["input_sha256_before"], info["recipe_id"], info["recipe_version"]) != key:
                problems.append(f"pointer key differs from run identity {rid} {key[:12]}")
                continue
            if info["status"] != "ok" or ev_meter != METER:
                problems.append(f"not an ok current-meter run {rid} {key[:12]}")
                continue
            approval: dict[str, Any] = {"kind": "native_current_meter"}
            eq = run_dir / "meter_equivalence.json"
            if eq.is_file():
                record = _json(eq)
                original = root / "runs" / record["old_run_id"] / "runinfo.json"
                original_info = _json(original) if original.is_file() else None
                same_cost = bool(original_info) and all(original_info.get(f) == info.get(f) for f in COST_FIELDS)
                if (record.get("old_version"), record.get("new_version")) != APPROVED_ALIAS or not same_cost:
                    problems.append(f"alias lacks approved pair or original cost {rid} {key[:12]}")
                    continue
                approval = {
                    "kind": "meter_equivalence_alias",
                    "audit_sha256": record["audit_sha256"],
                    "old_version": record["old_version"],
                    "new_version": record["new_version"],
                    "original_run_id": record["old_run_id"],
                    "cost_equals_original": True,
                    "equivalence_file_sha256": _sha256_file(eq),
                }
            entries[key] = {
                "root": rid, "run_id": run_id,
                "complete_sha256": _sha256_file(complete),
                "runinfo_sha256": _sha256_file(run_dir / "runinfo.json"),
                "recipe_id": info["recipe_id"], "recipe_version": info["recipe_version"],
                "input_state_id": info["input_sha256_before"],
                "output_state_id": out_state["id"] if out_state else None,
                "output_state_path": out_state["path"] if out_state else None,
                "status": info["status"], "fake": info["fake"],
                "meter_version": ev_meter,
                "cost": {f: info.get(f) for f in COST_FIELDS},
                "stored_cache_hit_flag": info["cache_hit"],
                "approval": approval,
            }
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    # State locations: every root holding the bytes, plus each recorded output path.
    wanted = {e["input_state_id"] for e in entries.values()}
    wanted |= {e["output_state_id"] for e in entries.values() if e["output_state_id"]}
    states: dict[str, list[str]] = {}
    for state_id in sorted(wanted):
        locs = {str(root / "states" / state_id / "design.dcp") for root in rts.values()
                if (root / "states" / state_id / "design.dcp").is_file()}
        states[state_id] = sorted(locs)
        if not locs:
            problems.append(f"state bytes not found in any root {state_id[:12]}")
    # Start evidence at the current meter version, per test design B2.
    version_key = hashlib.sha256(METER.encode()).hexdigest()
    start: dict[str, Any] = {}
    selection = _json(SELECTION)["by_design"]
    for design in DESIGNS:
        b2 = selection[design]["b2_hash"]
        for rid, root in rts.items():
            path = root / "measurements" / b2 / version_key / "evidence.json"
            if path.is_file():
                ev = _json(path)
                if ev.get("meter_version") == METER and ev.get("state_id") == b2:
                    start[b2] = {"root": rid, "path": str(path), "sha256": _sha256_file(path),
                                 "meter_version": METER, "design": design}
                    break
        else:
            problems.append(f"no current-meter start evidence for {design}")
    body = {
        "schema_version": SCHEMA, "meter_version": METER,
        "roots": {k: str(v) for k, v in rts.items()},
        "entries": dict(sorted(entries.items())), "states": states, "start_evidence": start,
    }
    # Reconcile the 914 stored outcome rows with the manifest.
    rows: list[tuple[str, str, str]] = []  # (root id, run id, output hash)
    for row in csv.DictReader((BASE / "by_design/index.csv").open(newline="", encoding="utf-8")):
        if row["design"] in DESIGNS and row["kind"] == "recipe":
            rows.append(("test_base", row["run_id"], row["output_sha256"]))
    for phase in "ab":
        for design in DESIGNS:
            idx = POWER / f"phase_{phase}" / design / "by_design/index.csv"
            for row in csv.DictReader(idx.open(newline="", encoding="utf-8")):
                rows.append((f"power_{phase}:{design}", row["run_id"], row["output_hash"]))
    resolved_once = unresolved = mismatch = multi_root_rows = 0
    row_keys: set[str] = set()
    for rid, run_id, out_hash in rows:
        info = _json(rts[rid] / "runs" / run_id / "runinfo.json")
        key = RunCache.key(info["input_sha256_before"], info["recipe_id"], info["recipe_version"])
        row_keys.add(key)
        roots_for_key = seen.get(key, [])
        if key in multi:
            multi_root_rows += 1
        elif len(roots_for_key) == 1 and key in entries:
            resolved_once += 1
            if entries[key]["output_state_id"] != out_hash:
                mismatch += 1
        else:
            unresolved += 1
    report = {
        "outcome_rows": len(rows), "distinct_keys_of_rows": len(row_keys),
        "rows_resolving_exactly_once": resolved_once, "rows_unresolved": unresolved,
        "rows_whose_key_is_in_more_than_one_root": multi_root_rows,
        "rows_whose_manifest_output_differs_from_index": mismatch,
        "entries": len(entries), "entries_by_root_kind": {
            "test_base": sum(e["root"] == "test_base" for e in entries.values()),
            "power_a": sum(e["root"].startswith("power_a") for e in entries.values()),
            "power_b": sum(e["root"].startswith("power_b") for e in entries.values()),
        },
        "alias_entries": sum(e["approval"]["kind"] == "meter_equivalence_alias" for e in entries.values()),
        "retryable_pointers_skipped": retryable,
        "keys_in_more_than_one_root": sorted(multi),
        "states_with_location": sum(bool(v) for v in states.values()),
        "start_evidence_found": len(start), "problems": problems,
        "manifest_sha256": manifest_hash(body),
    }
    return body, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    body, report = build()
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(OUT / "cache_manifest.json", _json_bytes(body))
        _atomic_bytes(OUT / "cache_manifest.sha256", (report["manifest_sha256"] + "\n").encode())
        _atomic_bytes(OUT / "cache_manifest_report.json", _json_bytes(report))
        assert _sha256_file(OUT / "cache_manifest.json") == hashlib.sha256(_json_bytes(body)).hexdigest()
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if not report["problems"] and not report["rows_unresolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
