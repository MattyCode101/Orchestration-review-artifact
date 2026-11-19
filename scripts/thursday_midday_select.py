"""Freeze training-only depth-one/two beam parents for five tight briefs.

This is an offline selector. It reads existing AMD-checked training records and
v3 telemetry, and writes no tool command or Vivado Tcl.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from orch.config import load_recipe_sets
from orch.contract.models import Evidence, MetricName

PROJECT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT / "data/sweeps/worker_fitting_train"
TELE = PROJECT / "data/exp_evening_2026-11-04/training_telemetry_v3"
OUT = PROJECT / "data/sweeps/thursday_crossworker_depth3_2026-11-05"
DESIGNS = (
    "vexriscv_re-place", "rosetta_spam-filter", "rosetta_3d-rendering",
    "logicnets_jscl", "finn_radioml",
)
TIMING = (
    "LATCH.phys_opt", "LATCH.retime", "LATCH.auto_pblock",
    "LATCH.hardblock_relocate", "LATCH.memory_path_opt",
)
AREA = ("A_R_EA", "A_R_RSA_SRLM", "A_R_RSA_AREM", "A_R_RSA_MED", "A_E_DUPL_Q1000")
BRIEFS = ("T", "A", "A0", "T0", "B")
MAX_RUNS_PER_DESIGN = 200


def sha(path: Path) -> str:
    """Hash existing bytes without changing the source."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(raw: str, base: Path) -> Path:
    """Reject a missing source or a path outside its read-only data root."""
    path = Path(raw).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(base):
        raise ValueError(f"source escaped {base}: {raw}")
    return path


def number(value: Any) -> Decimal:
    """Use recorded decimal text for exact guard thresholds."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("non-numeric beam value")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("non-finite beam value")
    return result


def metric(evidence: Evidence, name: MetricName) -> Decimal:
    item = evidence.metric(name)
    if item is None or not math.isfinite(item.value):
        raise ValueError(f"missing {name.value}")
    return number(item.value)


def objective(brief: str, start: dict[str, Decimal], state: dict[str, Decimal]) -> Decimal:
    """Rank valid and repair candidates by the same brief objective."""
    if brief in ("T", "T0"):
        return state["wns"]
    if brief in ("A", "A0"):
        return state["lut"]
    if brief != "B":
        raise ValueError(f"unknown brief {brief}")
    start_period = start["requirement"] - start["wns"]
    candidate_period = state["requirement"] - state["wns"]
    if start_period <= 0 or candidate_period <= 0 or start["lut"] <= 0:
        raise ValueError("invalid Fmax or LUT denominator")
    fmax_gain_pct = Decimal(100) * (start_period / candidate_period - 1)
    lut_cut_pct = Decimal(100) * (1 - state["lut"] / start["lut"])
    return fmax_gain_pct + lut_cut_pct


def brief_valid(brief: str, start: dict[str, Decimal], state: dict[str, Decimal]) -> bool:
    """Apply the authors' exact guards against the B2 start, without noise relaxation."""
    if brief == "T":
        return state["lut"] <= Decimal("1.01") * start["lut"]
    if brief == "A":
        return state["wns"] >= start["wns"] - Decimal("0.1")
    if brief == "A0":
        return state["wns"] >= start["wns"]
    if brief == "T0":
        return state["lut"] <= start["lut"]
    if brief == "B":
        return state["wns"] >= start["wns"] and state["lut"] <= start["lut"]
    raise ValueError(f"unknown brief {brief}")


def ranked(brief: str, start: dict[str, Decimal], candidates: list[dict[str, Any]],
           valid: bool) -> list[dict[str, Any]]:
    """Resolve objective ties by DCP hash so the frozen list is reproducible."""
    subset = [state for state in candidates if brief_valid(brief, start, state) is valid]
    if brief in ("A", "A0"):
        return sorted(subset, key=lambda item: (objective(brief, start, item), item["hash"]))
    return sorted(subset, key=lambda item: (-objective(brief, start, item), item["hash"]))


def compared_outputs(amd: dict[str, Any]) -> int:
    attempts = amd.get("attempts") or []
    if not attempts:
        return 0
    log = Path(attempts[-1]["run_dir"]) / "validate.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    return max((int(value) for value in re.findall(r"Outputs\s*\((\d+)\):", text)), default=0)


def load_telemetry() -> tuple[dict[tuple[str, str], Path], str]:
    path = TELE / "logical_index.csv"
    lookup: dict[tuple[str, str], Path] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["design"] not in DESIGNS:
                raise ValueError("non-training design in telemetry index")
            key = (row["design"], row["state_sha256"])
            report = safe_file(row["telemetry_json"], TELE)
            if key in lookup and lookup[key] != report:
                raise ValueError(f"ambiguous telemetry {key}")
            lookup[key] = report
    return lookup, sha(path)


def source_candidate(row: dict[str, str], baseline: Evidence, start_req: Decimal,
                     telemetry: dict[tuple[str, str], Path]) -> dict[str, Any]:
    """Read one saved edge, rejecting missing independent meter/AMD evidence."""
    design, output = row["design"], row["output_sha256"]
    if (row["run_status"] != "ok" or row["physically_valid"] != "True"
            or row["equivalence_status"] != "pass" or not output):
        raise ValueError(f"invalid saved training edge {design}: {row['sequence']}")
    dcp = safe_file(row["canonical_dcp"], TRAIN)
    evidence_path = TRAIN / "runs" / row["run_id"] / "evidence.json"
    evidence = Evidence.model_validate_json(evidence_path.read_bytes())
    if (evidence.state_id != output or evidence.fake or evidence.routed is not True
            or evidence.route_errors != 0 or evidence.drc_errors != 0
            or evidence.constraint_fingerprint != baseline.constraint_fingerprint
            or evidence.clock_digest != baseline.clock_digest
            or evidence.meter_version != baseline.meter_version):
        raise ValueError(f"invalid saved meter Evidence {design}: {row['sequence']}")
    report_path = telemetry.get((design, output))
    if report_path is None:
        raise ValueError(f"missing v3 telemetry {design}: {output}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    req = number(report["wpath"]["requirement_ns"])
    wns, lut = metric(evidence, MetricName.WNS_NS), metric(evidence, MetricName.PACKED_LUTS)
    if (req != start_req or report["state_sha256"] != output
            or number(report["metrics"]["wns_ns"]["value"]) != wns
            or number(report["metrics"]["packed_luts"]["value"]) != lut
            or report["validity"]["constraint_fingerprint"] != baseline.constraint_fingerprint
            or report["validity"]["clock_digest"] != baseline.clock_digest):
        raise ValueError(f"telemetry/meter mismatch {design}: {output}")
    return {
        "hash": output, "wns": wns, "lut": lut, "requirement": req,
        "source_depth": int(row["depth"]), "source_sequence": row["sequence"],
        "source_run_id": row["run_id"], "dcp": str(dcp),
        "evidence": str(evidence_path), "amd": row["verification_record"],
        "telemetry": str(report_path),
    }


def verify_selected(state: dict[str, Any], b2_hash: str) -> None:
    """Recheck selected checkpoint bytes and its saved AMD PASS report."""
    if sha(Path(state["dcp"])) != state["hash"]:
        raise ValueError(f"selected DCP changed: {state['hash']}")
    amd_path = safe_file(state["amd"], TRAIN / "verification")
    amd = json.loads(amd_path.read_text(encoding="utf-8"))
    raw = safe_file(amd["raw_report_path"], TRAIN / "verification")
    if (amd.get("result") != "pass" or amd.get("output_sha256") != state["hash"]
            or amd.get("reference_sha256") != b2_hash
            or sha(raw) != amd.get("raw_report_sha256")
            or compared_outputs(amd) <= 0):
        raise ValueError(f"selected state lacks retained AMD PASS: {state['hash']}")
    state["dcp_sha256"] = state["hash"]
    state["evidence_sha256"] = sha(Path(state["evidence"]))
    state["amd_sha256"] = sha(amd_path)
    state["telemetry_sha256"] = sha(Path(state["telemetry"]))


def select_design(design: str, rows: list[dict[str, str]],
                  telemetry: dict[tuple[str, str], Path],
                  references: dict[str, str]) -> dict[str, Any]:
    starts = [row for row in rows if row["design"] == design and row["depth"] == "0"]
    if len(starts) != 1 or starts[0]["sequence"] != "B2":
        raise ValueError(f"missing unique B2 start: {design}")
    b2_hash = starts[0]["output_sha256"]
    if references.get(design) != b2_hash:
        raise ValueError(f"AMD reference is not B2: {design}")
    b2_dcp = safe_file(starts[0]["canonical_dcp"], TRAIN)
    if sha(b2_dcp) != b2_hash:
        raise ValueError(f"B2 DCP changed: {design}")
    start_records = list((TRAIN / "measurements" / b2_hash).glob("*/evidence.json"))
    if len(start_records) != 1:
        raise ValueError(f"missing unique B2 meter Evidence: {design}")
    baseline = Evidence.model_validate_json(start_records[0].read_bytes())
    if (baseline.state_id != b2_hash or baseline.fake or baseline.routed is not True
            or baseline.route_errors != 0 or baseline.drc_errors != 0):
        raise ValueError(f"invalid B2 meter Evidence: {design}")
    start_telemetry = telemetry[(design, b2_hash)]
    start_report = json.loads(start_telemetry.read_text(encoding="utf-8"))
    start = {
        "wns": metric(baseline, MetricName.WNS_NS),
        "lut": metric(baseline, MetricName.PACKED_LUTS),
        "requirement": number(start_report["wpath"]["requirement_ns"]),
    }
    if (number(start_report["metrics"]["wns_ns"]["value"]) != start["wns"]
            or number(start_report["metrics"]["packed_luts"]["value"]) != start["lut"]):
        raise ValueError(f"B2 telemetry/meter mismatch: {design}")
    edges = [row for row in rows if row["design"] == design and row["depth"] in ("1", "2")]
    if len(edges) != 110 or sum(row["depth"] == "1" for row in edges) != 10:
        raise ValueError(f"incomplete D1/D2 training set: {design}")
    unique: dict[str, dict[str, Any]] = {}
    for row in edges:
        state = source_candidate(row, baseline, start["requirement"], telemetry)
        if state["hash"] == b2_hash:
            continue
        old = unique.get(state["hash"])
        if old is not None:
            if (old["wns"], old["lut"], old["requirement"]) != (
                    state["wns"], state["lut"], state["requirement"]):
                raise ValueError(f"same DCP has different meter result: {design}")
            if (state["source_depth"], state["source_sequence"], state["source_run_id"]) >= (
                    old["source_depth"], old["source_sequence"], old["source_run_id"]):
                continue
        unique[state["hash"]] = state
    candidates = list(unique.values())
    chosen: dict[str, dict[str, Any]] = {}
    by_brief: dict[str, Any] = {}
    for brief in BRIEFS:
        by_brief[brief] = {}
        for side, is_valid in (("valid", True), ("violating", False)):
            order = ranked(brief, start, candidates, is_valid)
            picked = order[:2]
            by_brief[brief][side] = {
                "eligible": len(order), "chosen": [state["hash"] for state in picked],
            }
            for rank, state in enumerate(picked, 1):
                item = chosen.setdefault(state["hash"], dict(state, memberships=[]))
                item["memberships"].append({
                    "brief": brief, "side": side, "rank": rank,
                    "objective": str(objective(brief, start, state)),
                })
    # Five briefs x (two valid + two violating) gives at most 20 states.
    # Thus the 200-run cap cannot bind under this specified selection rule.
    if len(chosen) * 10 > MAX_RUNS_PER_DESIGN:
        raise ValueError("selection exceeded its mathematical 200-run cap")
    for state in chosen.values():
        verify_selected(state, b2_hash)
    selected = []
    for state in sorted(chosen.values(), key=lambda item: item["hash"]):
        for name in ("wns", "lut", "requirement"):
            state[name] = str(state[name])
        selected.append(state)
    return {
        "start": {
            "hash": b2_hash, "dcp": str(b2_dcp), "evidence": str(start_records[0]),
            "evidence_sha256": sha(start_records[0]),
            "telemetry": str(start_telemetry), "telemetry_sha256": sha(start_telemetry),
            "wns_ns": str(start["wns"]), "packed_luts": str(start["lut"]),
            "requirement_ns": str(start["requirement"]),
            "constraint_fingerprint": baseline.constraint_fingerprint,
            "clock_digest": baseline.clock_digest, "meter_version": baseline.meter_version,
        },
        "candidate_states": len(candidates), "by_brief": by_brief,
        "selected_states": selected, "planned_runs": len(selected) * 10,
    }


def build() -> dict[str, Any]:
    """Return a deterministic training-only plan, except for its freeze time."""
    sets = load_recipe_sets(PROJECT / "configs/recipe_sets.yaml")
    sets.require_signed_off()
    if (tuple(sets.worker_sets.get("timing", ())) != TIMING
            or tuple(sets.worker_sets.get("area", ())) != AREA):
        raise ValueError("selected worker recipes differ from signed recipe_sets.yaml")
    index = TRAIN / "by_design/index.csv"
    rows = list(csv.DictReader(index.open(newline="", encoding="utf-8")))
    if {row["design"] for row in rows} != set(DESIGNS):
        raise ValueError("training index design set changed")
    telemetry, telemetry_sha = load_telemetry()
    references = json.loads((TRAIN / "verification/plan.json").read_text())["references"]
    designs = {design: select_design(design, rows, telemetry, references) for design in DESIGNS}
    return {
        "schema_version": "thursday-crossworker-selection-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_designs": list(DESIGNS),
        "selected_recipe_ids": list(TIMING + AREA),
        "source_index_sha256": sha(index),
        "telemetry_index_sha256": telemetry_sha,
        "recipe_sets_sha256": sha(PROJECT / "configs/recipe_sets.yaml"),
        "briefs": {
            "T": "max WNS; packed LUT <= 1.01 * B2",
            "A": "min packed LUT; WNS >= B2 - 0.1 ns",
            "A0": "min packed LUT; WNS >= B2",
            "T0": "max WNS; packed LUT <= B2",
            "B": "max percent Fmax gain plus percent LUT cut; WNS >= B2 and LUT <= B2",
        },
        "by_design": designs,
        "total_planned_runs": sum(row["planned_runs"] for row in designs.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    plan = build()
    counts = {design: {
        "candidates": row["candidate_states"],
        "selected": len(row["selected_states"]),
        "planned_runs": row["planned_runs"],
        "briefs": {brief: {
            "valid": len(groups["valid"]["chosen"]),
            "violating": len(groups["violating"]["chosen"]),
        } for brief, groups in row["by_brief"].items()},
    } for design, row in plan["by_design"].items()}
    if args.write:
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / "selected_states.json"
        payload = (json.dumps(plan, sort_keys=True, indent=2) + "\n").encode("utf-8")
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(payload).hexdigest()
        checksum = OUT / "selected_states.sha256"
        with checksum.open("x", encoding="ascii") as stream:
            stream.write(f"{digest}  selected_states.json\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps({"selection_sha256": digest, "total_planned_runs": plan["total_planned_runs"],
                          "by_design": counts}, sort_keys=True))
    else:
        print(json.dumps({"dry_run": True, "total_planned_runs": plan["total_planned_runs"],
                          "by_design": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
